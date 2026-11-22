"""Module: kalman_filter.filters.extended
Description: Extended Kalman filter that linearises f and h around the current mean.
"""

import logging

from kalman_filter.filters.base import BaseFilter
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter(BaseFilter):
    """First-order EKF for any ``StateSpaceModel`` (including ``LinearModel``).

    The mean is propagated through the true nonlinear ``f``/``h``; covariances use
    the Jacobians evaluated at the current mean.
    """

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        return self.model.state_jacobian(self.x, u)

    def _measurement_matrix(self) -> FloatArray:
        return self.model.measurement_jacobian(self.x)
