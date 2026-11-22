r"""Module: kalman_filter.models.base

Description: Protocol every state-space model must satisfy to be used by a filter.

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from typing import Protocol

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["StateSpaceModel"]


class StateSpaceModel(Protocol):
    r"""Discrete-time model ``x_k = f(x_{k-1}, u) + w``, ``z_k = h(x_k) + v``.

    Attributes:
        Q: Process noise covariance, shape ``(n, n)``.
        R: Measurement noise covariance, shape ``(m, m)``.
    """

    Q: FloatArray
    R: FloatArray

    @property
    def state_dim(self) -> int:
        r"""Number of states ``n``.

        Returns:
            The state dimension.
        """
        ...

    @property
    def meas_dim(self) -> int:
        r"""Number of measurement channels ``m``.

        Returns:
            The measurement dimension.
        """
        ...

    def propagate(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        r"""Applies the noise-free state transition ``f(x, u)``.

        Args:
            x: State, shape ``(n,)``.
            u: Optional control input.

        Returns:
            The next state, shape ``(n,)``.
        """
        ...

    def measure(self, x: FloatArray) -> FloatArray:
        r"""Applies the noise-free measurement function ``h(x)``.

        Args:
            x: State, shape ``(n,)``.

        Returns:
            The expected measurement, shape ``(m,)``.
        """
        ...

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        r"""Computes the innovation ``z - z_pred`` (may wrap angles).

        Args:
            z: Measurement, shape ``(m,)``.
            z_pred: Predicted measurement, shape ``(m,)``.

        Returns:
            The innovation, shape ``(m,)``.
        """
        ...

    def state_jacobian(self, x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        r"""Computes the Jacobian of ``f`` with respect to ``x``.

        Args:
            x: Linearisation point, shape ``(n,)``.
            u: Optional control input.

        Returns:
            The Jacobian, shape ``(n, n)``.
        """
        ...

    def measurement_jacobian(self, x: FloatArray) -> FloatArray:
        r"""Computes the Jacobian of ``h`` with respect to ``x``.

        Args:
            x: Linearisation point, shape ``(n,)``.

        Returns:
            The Jacobian, shape ``(m, n)``.
        """
        ...
