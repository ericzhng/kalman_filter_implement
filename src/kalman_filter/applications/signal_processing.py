r"""Module: kalman_filter.applications.signal_processing

Description: Standard Kalman Filter applications in meteorology & signal processing:
    - Linear Forecast Assimilation combining physical forecasting dynamics with observations.
    - Harmonic Oscillator Signal Model for extracting frequency components from noisy streams.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from collections.abc import Sequence

import numpy as np

from kalman_filter.models.linear import LinearModel

logger = logging.getLogger(__name__)


__all__ = ["HarmonicSignalModel", "LinearForecastAssimilationModel"]


class LinearForecastAssimilationModel(LinearModel):
    r"""Linear advection-diffusion atmospheric forecast model with sparse observation network.

    Models discrete 1D scalar transport (e.g. temperature, pollutant) over
    :math:`N` spatial grid cells with periodic boundary conditions:
    .. math::
        \partial_t \phi + c \partial_x \phi = \kappa \partial_{xx} \phi



    Discretized using forward-time centered-space (FTCS) or upwind scheme:
    :math:`x_k = F x_{k-1} + w_k`.
    Sparse observation stations measure a subset of the spatial cells with sensor noise.

    Args:
        num_grid: Number of spatial grid points :math:`N`.
        dt: Integration time step.
        dx: Spatial grid spacing.
        advection_velocity: Flow velocity :math:`c`.
        diffusion_coeff: Physical diffusion coefficient :math:`\kappa \ge 0`.
        obs_indices: Sequence of observed grid indices. If None, every 2nd cell is observed.
        process_std: Standard deviation of model physics error.
        obs_std: Standard deviation of meteorological sensor observations.
    """

    def __init__(
        self,
        num_grid: int = 20,
        dt: float = 0.1,
        dx: float = 1.0,
        advection_velocity: float = 0.5,
        diffusion_coeff: float = 0.05,
        obs_indices: Sequence[int] | None = None,
        process_std: float = 0.05,
        obs_std: float = 0.5,
    ) -> None:
        if num_grid < 3:
            raise ValueError("num_grid must be >= 3")
        if dt <= 0 or dx <= 0:
            raise ValueError("dt and dx must be positive")

        self.num_grid = num_grid
        self.dt = dt
        self.dx = dx

        # Courant and Peclet numbers
        c_num = advection_velocity * dt / dx
        d_num = diffusion_coeff * dt / (dx**2)

        # Assemble transition matrix F with periodic boundary conditions
        F = np.zeros((num_grid, num_grid))
        for i in range(num_grid):
            im1 = (i - 1) % num_grid
            ip1 = (i + 1) % num_grid
            # Central difference advection + diffusion
            F[i, i] = 1.0 - 2.0 * d_num
            F[i, im1] = 0.5 * c_num + d_num
            F[i, ip1] = -0.5 * c_num + d_num

        Q = (process_std**2) * np.eye(num_grid)

        if obs_indices is not None:
            self.obs_idx = np.asarray(obs_indices, dtype=int)
        else:
            self.obs_idx = np.arange(0, num_grid, 2, dtype=int)

        m = len(self.obs_idx)
        H = np.zeros((m, num_grid))
        for row, col in enumerate(self.obs_idx):
            H[row, col] = 1.0

        R = (obs_std**2) * np.eye(m)
        super().__init__(F=F, H=H, Q=Q, R=R)


class HarmonicSignalModel(LinearModel):
    r"""Linear harmonic oscillator state-space model for frequency tracking and denoising.

    Models a sinusoidal carrier signal :math:`s(t) = A \cos(\omega t + \phi)` in quadrature
    state space :math:`x = [s, \dot{s} / \omega]^T`:
    .. math::
        \begin{bmatrix} s_k \\ s^*_k \end{bmatrix} =
        \begin{bmatrix} \cos(\omega \Delta t) & \sin(\omega \Delta t) \\
        -\sin(\omega \Delta t) & \cos(\omega \Delta t) \end{bmatrix}
        \begin{bmatrix} s_{k-1} \\ s^*_{k-1} \end{bmatrix} + w_k

    Observation is the noisy scalar signal :math:`z_k = s_k + v_k`.

    Args:
        freq_hz: Frequency of the harmonic component in Hertz.
        dt: Sample period in seconds.
        q_std: Process noise standard deviation on oscillator amplitude/phase drift.
        r_std: Observation noise standard deviation.
    """

    def __init__(
        self,
        freq_hz: float = 10.0,
        dt: float = 0.01,
        q_std: float = 1e-3,
        r_std: float = 0.2,
    ) -> None:
        if freq_hz <= 0 or dt <= 0:
            raise ValueError("Frequency and sample period must be strictly positive")

        omega = 2.0 * np.pi * freq_hz
        theta = omega * dt
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        F = np.array(
            [
                [cos_t, sin_t],
                [-sin_t, cos_t],
            ]
        )
        H = np.array([[1.0, 0.0]])
        Q = (q_std**2) * np.eye(2)
        R = np.array([[r_std**2]])

        super().__init__(F=F, H=H, Q=Q, R=R)
