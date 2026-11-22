r"""Module: kalman_filter.filters.extended

Description: Extended Kalman filter that linearises ``f`` and ``h`` around the current mean.

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

from kalman_filter.filters.base import BaseFilter
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["ExtendedKalmanFilter"]


class ExtendedKalmanFilter(BaseFilter):
    r"""First-order EKF for any ``StateSpaceModel`` (including ``LinearModel``).

    The mean is propagated through the true nonlinear ``f``/``h``; covariances use
    the Jacobians evaluated at the current mean.
    """

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        r"""Evaluates the Jacobian of ``f`` at the current mean.

        Args:
            u: Optional control input.

        Returns:
            The Jacobian, shape ``(n, n)``.
        """
        return self.model.state_jacobian(self.x, u)

    def _measurement_matrix(self) -> FloatArray:
        r"""Evaluates the Jacobian of ``h`` at the current mean.

        Returns:
            The Jacobian, shape ``(m, n)``.
        """
        return self.model.measurement_jacobian(self.x)
