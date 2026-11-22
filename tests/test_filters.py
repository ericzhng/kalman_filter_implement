r"""Module: tests.test_filters

Description: Unit tests for the KF, EKF, Jacobians and missing-channel handling.

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np
import pytest

from kalman_filter import (
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    KalmanFilter,
    LinearModel,
    NonlinearModel,
    constant_velocity_model,
    nees,
    numerical_jacobian,
    simulate,
)

logger = logging.getLogger(__name__)


def _cv_model() -> LinearModel:
    r"""Builds a 2D constant-velocity model with three position channels.

    Returns:
        The linear model.
    """
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0]])
    return constant_velocity_model(2, 0.1, 0.5, H, np.diag([1.0, 1.0, 4.0]))


def test_scalar_kf_matches_hand_computation() -> None:
    r"""Checks a scalar KF step against the hand-computed gain, mean and variance."""
    model = LinearModel(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]])
    kf = KalmanFilter(model, np.array([0.0]), np.array([[1.0]]))
    kf.predict()  # P = 2
    post = kf.update(np.array([3.0]))  # K = 2/3
    assert post.mean[0] == pytest.approx(2.0)
    assert post.cov[0, 0] == pytest.approx(2.0 / 3.0)


def test_ekf_equals_kf_on_linear_model() -> None:
    r"""Checks that the EKF reproduces the KF exactly on a linear model."""
    model = _cv_model()
    _, z = simulate(model, np.array([0.0, 0.0, 1.0, 1.0]), 50, np.random.default_rng(0))
    x0, P0 = np.zeros(4), np.eye(4)
    a = KalmanFilter(model, x0, P0).run(z)
    b = ExtendedKalmanFilter(model, x0, P0).run(z)
    np.testing.assert_allclose(a.filtered_means, b.filtered_means)
    np.testing.assert_allclose(a.filtered_covs, b.filtered_covs)


def test_missing_channels_are_skipped() -> None:
    r"""Checks that NaN channels are skipped and all-NaN updates leave the belief unchanged."""
    model = _cv_model()
    kf = KalmanFilter(model, np.zeros(4), np.eye(4))
    before = kf.state
    after = kf.update(np.full(3, np.nan))
    np.testing.assert_array_equal(before.mean, after.mean)
    partial = kf.update(np.array([1.0, np.nan, np.nan]))
    assert np.isnan(kf.last_innovation[1]) and not np.isnan(kf.last_innovation[0])
    assert partial.cov[0, 0] < before.cov[0, 0]


def test_numerical_jacobian_matches_analytic() -> None:
    r"""Checks the central-difference Jacobian against the analytic one."""
    x = np.array([1.0, 2.0])
    jac = numerical_jacobian(lambda s: np.array([s[0] * s[1], np.sin(s[0])]), x)
    np.testing.assert_allclose(jac, [[2.0, 1.0], [np.cos(1.0), 0.0]], atol=1e-7)


def test_ekf_tracks_nonlinear_range_model() -> None:
    r"""Checks EKF accuracy and consistency on a nonlinear range measurement."""
    model = NonlinearModel(
        f=lambda x, u: np.array([x[0] + 0.1 * x[1], x[1]]),
        h=lambda x: np.array([np.hypot(x[0] + 10.0, 5.0)]),
        Q=np.diag([1e-4, 1e-4]),
        R=np.array([[0.01]]),
    )
    states, z = simulate(model, np.array([0.0, 1.0]), 300, np.random.default_rng(3))
    res = ExtendedKalmanFilter(model, np.array([0.5, 0.5]), np.eye(2)).run(z)
    assert np.abs(res.filtered_means[-1] - states[-1]).max() < 0.5
    assert 0.2 < nees(res.filtered_means, res.filtered_covs, states[1:]).mean() < 10


def test_shape_validation() -> None:
    r"""Checks that a mismatched initial state raises ``ValueError``."""
    with pytest.raises(ValueError):
        KalmanFilter(_cv_model(), np.zeros(3), np.eye(3))


def test_enkf_approaches_kf_on_linear_model() -> None:
    r"""Checks that a large-ensemble EnKF matches the exact KF on a linear model."""
    model = _cv_model()
    states, z = simulate(model, np.array([0.0, 0.0, 1.0, 1.0]), 100, np.random.default_rng(0))
    x0, P0 = np.zeros(4), np.eye(4)
    kf = KalmanFilter(model, x0, P0).run(z)
    enkf = EnsembleKalmanFilter(model, x0, P0, n_members=2000, rng=np.random.default_rng(1)).run(z)
    assert np.abs(enkf.filtered_means - kf.filtered_means).max() < 0.5
    assert np.abs(enkf.filtered_means[-1] - states[-1]).max() < 1.5


def test_enkf_skips_missing_channels_and_validates() -> None:
    r"""Checks NaN handling and argument validation of the EnKF."""
    enkf = EnsembleKalmanFilter(_cv_model(), np.zeros(4), np.eye(4), n_members=20)
    before = enkf.ensemble.copy()
    enkf.update(np.full(3, np.nan))
    np.testing.assert_array_equal(before, enkf.ensemble)
    enkf.update(np.array([1.0, np.nan, np.nan]))
    assert np.isnan(enkf.last_innovation[1]) and not np.isnan(enkf.last_innovation[0])
    with pytest.raises(ValueError):
        EnsembleKalmanFilter(_cv_model(), np.zeros(4), np.eye(4), n_members=1)
