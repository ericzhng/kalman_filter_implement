r"""Module: kalman_filter

Description: A professional, modular library for Kalman filtering, data assimilation,
    and multi-sensor fusion. Supports:
    - Linear Kalman Filter (KF)
    - Extended Kalman Filter (EKF)
    - Iterated Extended Kalman Filter (IEKF)
    - Unscented Kalman Filter (UKF)
    - Ensemble Kalman Filter (EnKF) with spatial Gaspari-Cohn covariance localization
    - Rauch-Tung-Striebel (RTS) fixed-interval smoother
    - Multi-rate, asynchronous sensors with missing channel (NaN) handling
    - Target tracking models: Constant Velocity (CV), Constant Acceleration (CA),
      Coordinated Turn (CT)
    - Data assimilation models: Lorenz-96 spatial atmospheric model, Lorenz-63 convective model

    - Consistency diagnostics: RMSE, MAE, NEES, Chi-square consistency testing, Innovation Whiteness

Example:
    >>> import numpy as np
    >>> from kalman_filter import KalmanFilter, LinearModel
    >>> model = LinearModel(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]])
    >>> kf = KalmanFilter(model, [0.0], [[1.0]])
    >>> kf.predict()
    GaussianState(mean=array([0.]), cov=array([[2.]]))
    >>> kf.update([3.0])
    GaussianState(mean=array([2.]), cov=array([[0.66666667]]))

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from kalman_filter.filters import (
    BaseFilter,
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    IteratedExtendedKalmanFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
    gaspari_cohn,
    rts_smooth,
)
from kalman_filter.jacobian import numerical_jacobian
from kalman_filter.metrics import (
    autocorrelation,
    chi2_consistency_test,
    mae,
    nees,
    rmse,
    whiteness_test,
)
from kalman_filter.models import (
    LinearModel,
    Lorenz63Model,
    Lorenz96Model,
    NonlinearModel,
    SensorStream,
    StateSpaceModel,
    bearing_residual,
    constant_acceleration_model,
    constant_velocity_model,
    coordinated_turn_model,
    merge_sensor_streams,
    random_walk_model,
    wrap_angle,
)
from kalman_filter.simulation import simulate
from kalman_filter.types import FilterResult, GaussianState, SmootherResult

__all__ = [
    "BaseFilter",
    "EnsembleKalmanFilter",
    "ExtendedKalmanFilter",
    "FilterResult",
    "GaussianState",
    "IteratedExtendedKalmanFilter",
    "KalmanFilter",
    "LinearModel",
    "Lorenz63Model",
    "Lorenz96Model",
    "NonlinearModel",
    "SensorStream",
    "SmootherResult",
    "StateSpaceModel",
    "UnscentedKalmanFilter",
    "autocorrelation",
    "bearing_residual",
    "chi2_consistency_test",
    "constant_acceleration_model",
    "constant_velocity_model",
    "coordinated_turn_model",
    "gaspari_cohn",
    "mae",
    "merge_sensor_streams",
    "nees",
    "numerical_jacobian",
    "random_walk_model",
    "rmse",
    "rts_smooth",
    "simulate",
    "whiteness_test",
    "wrap_angle",
]
