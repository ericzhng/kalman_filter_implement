r"""Module: kalman_filter.models.assimilation

Description: Benchmark chaotic models widely used in geophysical data assimilation,
    meteorology, and climate prediction:
    - Lorenz-96 spatial multi-variable model with periodic boundaries and distance matrices.
    - Lorenz-63 three-variable chaotic convective model with Runge-Kutta 4th order propagation.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from collections.abc import Sequence

import numpy as np

from kalman_filter.localization import gaspari_cohn
from kalman_filter.models.nonlinear import NonlinearModel
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["Lorenz63Model", "Lorenz96Model"]


class Lorenz96Model(NonlinearModel):
    r"""The classic Lorenz-96 D-dimensional atmospheric data assimilation benchmark model.

    Governed by the coupled ordinary differential equations on a periodic latitude circle:
    .. math::
        \frac{dx_i}{dt} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F_{\text{force}}

    for :math:`i = 0, \dots, D-1` with periodic boundary conditions:
    :math:`x_{-1} = x_{D-1}, x_{-2} = x_{D-2}, x_D = x_0`.

    Integrated forward with classical Runge-Kutta 4th order (RK4).

    Args:
        dim: Number of spatial state variables :math:`D` (typically 40).
        dt: Discrete integration time step (default 0.05, ~6 hours atmospheric time).
        forcing: External constant forcing parameter :math:`F` (default 8.0, fully chaotic).
        obs_indices: Sequence of observed grid indices (e.g., every 2nd grid point).
            If None, all variables are observed.
        q_std: Process noise standard deviation.
        r_std: Measurement noise standard deviation.
    """

    def __init__(
        self,
        dim: int = 40,
        dt: float = 0.05,
        forcing: float = 8.0,
        obs_indices: Sequence[int] | None = None,
        q_std: float = 1e-2,
        r_std: float = 1.0,
    ) -> None:
        if dim < 4:
            raise ValueError("dim must be >= 4 for Lorenz-96")
        self.dim = dim
        self.dt = dt
        self.forcing = forcing

        if obs_indices is not None:
            self.obs_idx = np.asarray(obs_indices, dtype=int)
        else:
            self.obs_idx = np.arange(dim, dtype=int)

        m = len(self.obs_idx)
        Q = (q_std**2) * np.eye(dim)
        R = (r_std**2) * np.eye(m)

        super().__init__(
            f=self._propagate_rk4,
            h=self._measure_obs,
            Q=Q,
            R=R,
            h_jac=self._h_jacobian,
        )

    def rhs(self, x: FloatArray) -> FloatArray:
        r"""Evaluates the continuous Lorenz-96 vector field :math:`dx/dt`.

        Args:
            x: State vector on the periodic circle, shape ``(dim,)``.

        Returns:
            Time derivative vector, shape ``(dim,)``.
        """
        # Using modular indexing for clean periodic boundaries
        im2 = np.roll(x, 2)
        im1 = np.roll(x, 1)
        ip1 = np.roll(x, -1)
        return (ip1 - im2) * im1 - x + self.forcing


    def _propagate_rk4(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        r"""Advances state by dt using classical Runge-Kutta 4th order integration."""
        dt = self.dt
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * dt * k1)
        k3 = self.rhs(x + 0.5 * dt * k2)
        k4 = self.rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _measure_obs(self, x: FloatArray) -> FloatArray:
        r"""Observes the state at the specified network stations."""
        return x[self.obs_idx]

    def _h_jacobian(self, x: FloatArray) -> FloatArray:
        r"""Constant measurement Jacobian matrix."""
        H = np.zeros((len(self.obs_idx), self.dim))
        for row, col in enumerate(self.obs_idx):
            H[row, col] = 1.0
        return H

    def compute_periodic_distances(self) -> FloatArray:
        r"""Computes the circular shortest-path distance matrix between all state grid points.

        Returns:
            Distance matrix of shape ``(dim, dim)`` with distances in :math:`[0, \text{dim} / 2]`.
        """
        D = self.dim
        idx = np.arange(D)
        diff = np.abs(idx[:, None] - idx[None, :])
        return np.minimum(diff, D - diff).astype(float)

    def compute_localization_matrix(self, cutoff_radius: float) -> FloatArray:
        r"""Generates a Gaspari-Cohn spatial cross-localization matrix :math:`\rho_{xy}`.

        Used by :class:`~kalman_filter.filters.ensemble.EnsembleKalmanFilter` to eliminate
        spurious ensemble cross-correlations over long physical distances.

        Args:
            cutoff_radius: Localization half-width parameter (in grid points).

        Returns:
            Tapering matrix of shape ``(dim, m)`` matching state-to-measurement channels.
        """
        dist_full = self.compute_periodic_distances()
        dist_obs = dist_full[:, self.obs_idx]
        return gaspari_cohn(dist_obs, cutoff_radius)


class Lorenz63Model(NonlinearModel):
    r"""The classic Lorenz-63 3D chaotic convective atmospheric model.

    Equations:
    .. math::
        \dot{x} = \sigma (y - x) \\
        \dot{y} = x (\rho - z) - y \\
        \dot{z} = x y - \beta z

    Integrated via RK4.

    Args:
        dt: Integration time step (default 0.05).
        sigma: Prandtl number (default 10.0).
        rho: Rayleigh number (default 28.0).
        beta: Geometric factor (default 8.0 / 3.0).
        observed_channels: Indices of observed states (default [0, 2] observing x and z).
        q_std: Process noise standard deviation.
        r_std: Measurement noise standard deviation.
    """

    def __init__(
        self,
        dt: float = 0.05,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        observed_channels: Sequence[int] = (0, 2),
        q_std: float = 1e-2,
        r_std: float = 1.0,
    ) -> None:
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.obs_channels = np.asarray(observed_channels, dtype=int)

        dim = 3
        m = len(self.obs_channels)
        Q = (q_std**2) * np.eye(dim)
        R = (r_std**2) * np.eye(m)

        super().__init__(
            f=self._propagate_rk4,
            h=self._measure,
            Q=Q,
            R=R,
            h_jac=self._h_jacobian,
        )

    def rhs(self, state: FloatArray) -> FloatArray:
        r"""Lorenz-63 vector field."""
        x, y, z = state
        return np.array(
            [
                self.sigma * (y - x),
                x * (self.rho - z) - y,
                x * y - self.beta * z,
            ]
        )

    def _propagate_rk4(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        r"""One RK4 step."""
        dt = self.dt
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * dt * k1)
        k3 = self.rhs(x + 0.5 * dt * k2)
        k4 = self.rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _measure(self, x: FloatArray) -> FloatArray:
        r"""Observes selected state channels."""
        return x[self.obs_channels]

    def _h_jacobian(self, x: FloatArray) -> FloatArray:
        r"""Linear measurement Jacobian."""
        H = np.zeros((len(self.obs_channels), 3))
        for row, col in enumerate(self.obs_channels):
            H[row, col] = 1.0
        return H
