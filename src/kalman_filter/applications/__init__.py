r"""Module: kalman_filter.applications

Description: Practical application modules for Standard KF and Extended Kalman Filtering:
    - Target & radar tracking: :class:`RadarTrackingModel`
    - Linear time-series & finance: :class:`LocalLinearTrendModel`, :class:`DynamicHedgeRatioFilter`
    - Meteorology & signal: :class:`LinearForecastAssimilationModel`, :class:`HarmonicSignalModel`
    - Navigation & sensor fusion: :class:`GPSIMUNavigationModel`, :class:`AttitudeHeadingModel`
    - Robotics: :class:`WheeledRobotSlipModel`, :class:`EKFSLAMModel`
    - Aerospace: :class:`SphericalRadarTrackingModel`, :class:`DroneFlightModel`


Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from kalman_filter.applications.aerospace import (
    DroneFlightModel,
    SphericalRadarTrackingModel,
)
from kalman_filter.applications.finance import (
    DynamicHedgeRatioFilter,
    LocalLinearTrendModel,
)
from kalman_filter.applications.navigation import (
    AttitudeHeadingModel,
    GPSIMUNavigationModel,
)
from kalman_filter.applications.radar_tracking import (
    RadarTrackingModel,
    create_3d_radar_tracker,
)
from kalman_filter.applications.robotics import (
    EKFSLAMModel,
    WheeledRobotSlipModel,
)
from kalman_filter.applications.signal_processing import (
    HarmonicSignalModel,
    LinearForecastAssimilationModel,
)

__all__ = [
    "AttitudeHeadingModel",
    "DroneFlightModel",
    "DynamicHedgeRatioFilter",
    "EKFSLAMModel",
    "GPSIMUNavigationModel",
    "HarmonicSignalModel",
    "LinearForecastAssimilationModel",
    "LocalLinearTrendModel",
    "RadarTrackingModel",
    "SphericalRadarTrackingModel",
    "WheeledRobotSlipModel",
    "create_3d_radar_tracker",
]
