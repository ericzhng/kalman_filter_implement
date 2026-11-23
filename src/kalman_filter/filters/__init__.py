r"""Module: kalman_filter.filters

Description: Comprehensive collection of optimal linear and nonlinear Kalman estimators:
    - Linear Kalman Filter (KF)
    - Extended Kalman Filter (EKF)
    - Iterated Extended Kalman Filter (IEKF)
    - Unscented Kalman Filter (UKF)
    - Stochastic Ensemble Kalman Filter (EnKF) with spatial localization
    - Rauch-Tung-Striebel (RTS) backward fixed-interval smoother

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from kalman_filter.filters.base import BaseFilter
from kalman_filter.filters.ensemble import EnsembleKalmanFilter, gaspari_cohn
from kalman_filter.filters.extended import ExtendedKalmanFilter
from kalman_filter.filters.iterated import IteratedExtendedKalmanFilter
from kalman_filter.filters.kalman import KalmanFilter
from kalman_filter.filters.smoother import rts_smooth
from kalman_filter.filters.unscented import UnscentedKalmanFilter

__all__ = [
    "BaseFilter",
    "EnsembleKalmanFilter",
    "ExtendedKalmanFilter",
    "IteratedExtendedKalmanFilter",
    "KalmanFilter",
    "UnscentedKalmanFilter",
    "gaspari_cohn",
    "rts_smooth",
]

