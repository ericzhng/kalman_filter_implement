"""Module: kalman_filter
Description: Modular Kalman filter (KF) and extended Kalman filter (EKF) with multi-channel support.
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
