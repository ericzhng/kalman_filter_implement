r"""Module: kalman_filter.models

Description: State-space models and kinematic/sensor utilities:
    - Base protocol: :class:`StateSpaceModel`
    - Linear models: :class:`LinearModel`, :func:`constant_velocity_model`
    - Kinematic: :func:`constant_acceleration_model`, :func:`coordinated_turn_model`,
      :func:`random_walk_model`
    - Geophysical models: :class:`Lorenz96Model`, :class:`Lorenz63Model`
    - Sensor fusion: :class:`SensorStream`, :func:`bearing_residual`,
      :func:`merge_sensor_streams`, :func:`wrap_angle`

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from kalman_filter.models.assimilation import Lorenz63Model, Lorenz96Model
from kalman_filter.models.base import StateSpaceModel
from kalman_filter.models.kinematic import (
    constant_acceleration_model,
    coordinated_turn_model,
    random_walk_model,
)
from kalman_filter.models.linear import LinearModel, constant_velocity_model
from kalman_filter.models.nonlinear import NonlinearModel
from kalman_filter.models.sensors import (
    SensorStream,
    bearing_residual,
    merge_sensor_streams,
    wrap_angle,
)

__all__ = [
    "LinearModel",
    "Lorenz63Model",
    "Lorenz96Model",
    "NonlinearModel",
    "SensorStream",
    "StateSpaceModel",
    "bearing_residual",
    "constant_acceleration_model",
    "constant_velocity_model",
    "coordinated_turn_model",
    "merge_sensor_streams",
    "random_walk_model",
    "wrap_angle",
]

