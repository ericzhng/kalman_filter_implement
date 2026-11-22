r"""Module: kalman_filter

Description: Modular Kalman filter (KF) and extended Kalman filter (EKF) with multi-channel
    support. See docs/theory.md and docs/implementation.md.

Example:
    >>> from kalman_filter import KalmanFilter, LinearModel
    >>> model = LinearModel(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]])
    >>> KalmanFilter(model, [0.0], [[1.0]]).state.mean.shape
    (1,)

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from kalman_filter.filters import BaseFilter, ExtendedKalmanFilter, KalmanFilter
from kalman_filter.jacobian import numerical_jacobian
from kalman_filter.metrics import nees, rmse
from kalman_filter.models import (
    LinearModel,
    NonlinearModel,
    StateSpaceModel,
    constant_velocity_model,
)
from kalman_filter.simulation import simulate
from kalman_filter.types import FilterResult, GaussianState

__all__ = [
    "BaseFilter",
    "ExtendedKalmanFilter",
    "FilterResult",
    "GaussianState",
    "KalmanFilter",
    "LinearModel",
    "NonlinearModel",
    "StateSpaceModel",
    "constant_velocity_model",
    "nees",
    "numerical_jacobian",
    "rmse",
    "simulate",
]
