"""Module: kalman_filter.models.base
Description: Protocol every state-space model must satisfy to be used by a filter.
"""

import logging
from typing import Protocol

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


class StateSpaceModel(Protocol):
    """Discrete-time model ``x_k = f(x_{k-1}, u) + w``, ``z_k = h(x_k) + v``."""

    Q: FloatArray
    R: FloatArray

    @property
    def state_dim(self) -> int:
        """Number of states ``n``."""
        ...

    @property
    def meas_dim(self) -> int:
        """Number of measurement channels ``m``."""
        ...

    def propagate(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        """Noise-free state transition ``f(x, u)``."""
        ...

    def measure(self, x: FloatArray) -> FloatArray:
        """Noise-free measurement ``h(x)``."""
        ...

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Innovation ``z - z_pred`` (may wrap angles)."""
        ...

    def state_jacobian(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        """Jacobian of ``f`` with respect to ``x``, shape ``(n, n)``."""
        ...

    def measurement_jacobian(self, x: FloatArray) -> FloatArray:
        """Jacobian of ``h`` with respect to ``x``, shape ``(m, n)``."""
        ...
