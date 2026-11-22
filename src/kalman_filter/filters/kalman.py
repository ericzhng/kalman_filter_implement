r"""Module: kalman_filter.filters.kalman

Description: Linear Kalman filter for :class:`kalman_filter.models.linear.LinearModel`.

Example:
    >>> from kalman_filter import KalmanFilter, LinearModel
    >>> model = LinearModel(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]])
    >>> kf = KalmanFilter(model, [0.0], [[1.0]])
    >>> _ = kf.predict()
    >>> post = kf.update([3.0])
    >>> post.mean.round(4).tolist(), post.cov.round(4).tolist()
    ([2.0], [[0.6667]])

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

from kalman_filter.filters.base import BaseFilter
from kalman_filter.models.linear import LinearModel
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["KalmanFilter"]


class KalmanFilter(BaseFilter):
    r"""Optimal estimator for linear-Gaussian models.

    Args:
        model: A ``LinearModel``.
        x0: Initial mean, shape ``(n,)``.
        P0: Initial covariance, shape ``(n, n)``.
    """

    def __init__(self, model: LinearModel, x0: FloatArray, P0: FloatArray) -> None:
        super().__init__(model, x0, P0)
        self._linear = model

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        r"""Returns the constant transition matrix ``F``.

        Args:
            u: Control input (unused, the model is linear).

        Returns:
            The matrix ``F``, shape ``(n, n)``.
        """
        return self._linear.F

    def _measurement_matrix(self) -> FloatArray:
        r"""Returns the constant measurement matrix ``H``.

        Returns:
            The matrix ``H``, shape ``(m, n)``.
        """
        return self._linear.H
