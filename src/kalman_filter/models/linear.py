r"""Module: kalman_filter.models.linear

Description: Linear-Gaussian model and a helper that builds common multi-channel models.

Example:
    >>> import numpy as np
    >>> from kalman_filter.models.linear import constant_velocity_model
    >>> m = constant_velocity_model(2, 0.1, 1.0, np.eye(4)[:2], np.eye(2))
    >>> m.state_dim, m.meas_dim
    (4, 2)

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["LinearModel", "constant_velocity_model"]


class LinearModel:
    r"""Linear model ``x_k = F x_{k-1} + B u + w``, ``z_k = H x_k + v``.

    Args:
        F: State transition matrix, shape ``(n, n)``.
        H: Measurement matrix, shape ``(m, n)``; one row per channel.
        Q: Process noise covariance, shape ``(n, n)``.
        R: Measurement noise covariance, shape ``(m, m)``.
        B: Optional control matrix, shape ``(n, p)``.

    Raises:
        ValueError: If matrix shapes are inconsistent.
    """

    def __init__(
        self,
        F: FloatArray,
        H: FloatArray,
        Q: FloatArray,
        R: FloatArray,
        B: FloatArray | None = None,
    ) -> None:
        self.F = np.atleast_2d(np.asarray(F, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.B = None if B is None else np.atleast_2d(np.asarray(B, dtype=float))
        n, m = self.F.shape[0], self.H.shape[0]
        if self.F.shape != (n, n) or self.H.shape[1] != n or self.Q.shape != (n, n):
            raise ValueError("F, H, Q shapes are inconsistent")
        if self.R.shape != (m, m):
            raise ValueError("R must be (m, m) where m is the number of rows of H")

    @property
    def state_dim(self) -> int:
        r"""Number of states ``n``.

        Returns:
            The state dimension.
        """
        return self.F.shape[0]

    @property
    def meas_dim(self) -> int:
        r"""Number of measurement channels ``m``.

        Returns:
            The measurement dimension.
        """
        return self.H.shape[0]

    def propagate(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        r"""Applies ``F x + B u``.

        Args:
            x: State, shape ``(n,)``.
            u: Optional control input, shape ``(p,)``; ignored when ``B`` is ``None``.

        Returns:
            The next state, shape ``(n,)``.
        """
        out = self.F @ x
        if u is not None and self.B is not None:
            out = out + self.B @ u
        return out

    def measure(self, x: FloatArray) -> FloatArray:
        r"""Applies ``H x``.

        Args:
            x: State, shape ``(n,)``.

        Returns:
            The expected measurement, shape ``(m,)``.
        """
        return self.H @ x

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        r"""Computes ``z - z_pred``.

        Args:
            z: Measurement, shape ``(m,)``.
            z_pred: Predicted measurement, shape ``(m,)``.

        Returns:
            The innovation, shape ``(m,)``.
        """
        return z - z_pred

    def state_jacobian(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        r"""Returns the constant matrix ``F``.

        Args:
            x: Linearisation point (unused, the model is linear).
            u: Control input (unused).

        Returns:
            The matrix ``F``, shape ``(n, n)``.
        """
        return self.F

    def measurement_jacobian(self, x: FloatArray) -> FloatArray:
        r"""Returns the constant matrix ``H``.

        Args:
            x: Linearisation point (unused, the model is linear).

        Returns:
            The matrix ``H``, shape ``(m, n)``.
        """
        return self.H


def constant_velocity_model(
    dim: int, dt: float, accel_std: float, H: FloatArray, R: FloatArray
) -> LinearModel:
    r"""Builds a nearly-constant-velocity model with state ``[pos_1..d, vel_1..d]``.

    Process noise is the discrete white-noise-acceleration model.

    Args:
        dim: Spatial dimension ``d`` (state size is ``2 d``).
        dt: Sample period.
        accel_std: Standard deviation of the acceleration noise.
        H: Measurement matrix, shape ``(m, 2 d)``.
        R: Measurement noise covariance, shape ``(m, m)``.

    Returns:
        The assembled ``LinearModel``.
    """
    eye = np.eye(dim)
    F = np.block([[eye, dt * eye], [np.zeros((dim, dim)), eye]])
    G = np.vstack([0.5 * dt**2 * eye, dt * eye])
    return LinearModel(F=F, H=H, Q=accel_std**2 * G @ G.T, R=R)
