"""Module: kalman_filter.models.nonlinear
Description: Nonlinear model defined by arbitrary f and h with optional analytic Jacobians.
"""

import logging
from collections.abc import Callable

import numpy as np

from kalman_filter.jacobian import numerical_jacobian
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

TransitionFn = Callable[[FloatArray, FloatArray | None], FloatArray]
MeasurementFn = Callable[[FloatArray], FloatArray]
ResidualFn = Callable[[FloatArray, FloatArray], FloatArray]


class NonlinearModel:
    """Model ``x_k = f(x_{k-1}, u) + w``, ``z_k = h(x_k) + v``.

    Jacobians default to central-difference approximations when not supplied.

    Args:
        f: State transition ``f(x, u)``; ``u`` may be ``None``.
        h: Measurement function ``h(x)``.
        Q: Process noise covariance, shape ``(n, n)``.
        R: Measurement noise covariance, shape ``(m, m)``.
        f_jac: Optional analytic Jacobian of ``f`` w.r.t. ``x``, called as ``(x, u)``.
        h_jac: Optional analytic Jacobian of ``h``, called as ``(x)``.
        residual_fn: Optional innovation function, e.g. to wrap bearing angles.
    """

    def __init__(
        self,
        f: TransitionFn,
        h: MeasurementFn,
        Q: FloatArray,
        R: FloatArray,
        f_jac: Callable[[FloatArray, FloatArray | None], FloatArray] | None = None,
        h_jac: Callable[[FloatArray], FloatArray] | None = None,
        residual_fn: ResidualFn | None = None,
    ) -> None:
        self._f = f
        self._h = h
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self._f_jac = f_jac
        self._h_jac = h_jac
        self._residual_fn = residual_fn

    @property
    def state_dim(self) -> int:
        """Number of states ``n``."""
        return self.Q.shape[0]

    @property
    def meas_dim(self) -> int:
        """Number of measurement channels ``m``."""
        return self.R.shape[0]

    def propagate(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        """Return ``f(x, u)``."""
        return np.asarray(self._f(x, u), dtype=float)

    def measure(self, x: FloatArray) -> FloatArray:
        """Return ``h(x)``."""
        return np.asarray(self._h(x), dtype=float)

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Return the innovation, using ``residual_fn`` if provided."""
        if self._residual_fn is not None:
            return self._residual_fn(z, z_pred)
        return z - z_pred

    def state_jacobian(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        """Return the Jacobian of ``f`` at ``x`` (analytic if given, else numerical)."""
        if self._f_jac is not None:
            return np.asarray(self._f_jac(x, u), dtype=float)
        return numerical_jacobian(lambda s: self.propagate(s, u), x)

    def measurement_jacobian(self, x: FloatArray) -> FloatArray:
        """Return the Jacobian of ``h`` at ``x`` (analytic if given, else numerical)."""
        if self._h_jac is not None:
            return np.asarray(self._h_jac(x), dtype=float)
        return numerical_jacobian(self.measure, x)
