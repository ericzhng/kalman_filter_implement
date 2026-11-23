r"""Module: kalman_filter.applications

Description: Practical application modules for Standard Kalman Filtering:
    - Target & radar tracking: :class:`RadarTrackingModel`
    - Linear time-series & finance: :class:`LocalLinearTrendModel`, :class:`DynamicHedgeRatioFilter`
    - Meteorology & signal: :class:`LinearForecastAssimilationModel`, :class:`HarmonicSignalModel`

Author: Hui Zhang

Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from kalman_filter.applications.finance import (
    DynamicHedgeRatioFilter,
    LocalLinearTrendModel,
)
from kalman_filter.applications.radar_tracking import (
    RadarTrackingModel,
    create_3d_radar_tracker,
)
from kalman_filter.applications.signal_processing import (
    HarmonicSignalModel,
    LinearForecastAssimilationModel,
)

__all__ = [
    "DynamicHedgeRatioFilter",
    "HarmonicSignalModel",
    "LinearForecastAssimilationModel",
    "LocalLinearTrendModel",
    "RadarTrackingModel",
    "create_3d_radar_tracker",
]
