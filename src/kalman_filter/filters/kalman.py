"""Module: kalman_filter.filters.kalman
Description: Linear Kalman filter for LinearModel.
"""

import logging

from kalman_filter.filters.base import BaseFilter
from kalman_filter.models.linear import LinearModel
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


class KalmanFilter(BaseFilter):
    """Optimal estimator for linear-Gaussian models.

    Args:
        model: A ``LinearModel``.
        x0: Initial mean, shape ``(n,)``.
        P0: Initial covariance, shape ``(n, n)``.
    """

    def __init__(self, model: LinearModel, x0: FloatArray, P0: FloatArray) -> None:
        super().__init__(model, x0, P0)
        self._linear = model

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        return self._linear.F

    def _measurement_matrix(self) -> FloatArray:
        return self._linear.H
