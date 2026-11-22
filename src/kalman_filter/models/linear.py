"""Module: kalman_filter.models.linear
Description: Linear-Gaussian model and helpers to build common multi-channel models.
"""

import logging

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


class LinearModel:
    """Linear model ``x_k = F x_{k-1} + B u + w``, ``z_k = H x_k + v``.

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
        """Number of states ``n``."""
        return self.F.shape[0]

    @property
    def meas_dim(self) -> int:
        """Number of measurement channels ``m``."""
        return self.H.shape[0]

    def propagate(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        """Return ``F x + B u``."""
        out = self.F @ x
        if u is not None and self.B is not None:
            out = out + self.B @ u
        return out

    def measure(self, x: FloatArray) -> FloatArray:
        """Return ``H x``."""
        return self.H @ x

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Return ``z - z_pred``."""
        return z - z_pred

    def state_jacobian(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        """Return the constant matrix ``F``."""
        return self.F

    def measurement_jacobian(self, x: FloatArray) -> FloatArray:
        """Return the constant matrix ``H``."""
        return self.H


def constant_velocity_model(
    dim: int, dt: float, accel_std: float, H: FloatArray, R: FloatArray
) -> LinearModel:
    """Build a nearly-constant-velocity model with state ``[pos_1..d, vel_1..d]``.

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
