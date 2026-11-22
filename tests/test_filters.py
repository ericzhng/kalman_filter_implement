"""
Module: tests.test_filters
Description: Unit tests for the Kalman filter, EKF, Jacobians and missing-channel handling.
"""

import logging

import numpy as np
import pytest

from kalman_filter import (
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
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0]])
    return constant_velocity_model(2, 0.1, 0.5, H, np.diag([1.0, 1.0, 4.0]))


def test_scalar_kf_matches_hand_computation() -> None:
    model = LinearModel(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]])
    kf = KalmanFilter(model, np.array([0.0]), np.array([[1.0]]))
    kf.predict()  # P = 2
    post = kf.update(np.array([3.0]))  # K = 2/3
    assert post.mean[0] == pytest.approx(2.0)
    assert post.cov[0, 0] == pytest.approx(2.0 / 3.0)


def test_ekf_equals_kf_on_linear_model() -> None:
    model = _cv_model()
    _, z = simulate(model, np.array([0.0, 0.0, 1.0, 1.0]), 50, np.random.default_rng(0))
    x0, P0 = np.zeros(4), np.eye(4)
    a = KalmanFilter(model, x0, P0).run(z)
    b = ExtendedKalmanFilter(model, x0, P0).run(z)
    np.testing.assert_allclose(a.filtered_means, b.filtered_means)
    np.testing.assert_allclose(a.filtered_covs, b.filtered_covs)


def test_missing_channels_are_skipped() -> None:
    model = _cv_model()
    kf = KalmanFilter(model, np.zeros(4), np.eye(4))
    before = kf.state
    after = kf.update(np.full(3, np.nan))
    np.testing.assert_array_equal(before.mean, after.mean)
    partial = kf.update(np.array([1.0, np.nan, np.nan]))
    assert np.isnan(kf.last_innovation[1]) and not np.isnan(kf.last_innovation[0])
    assert partial.cov[0, 0] < before.cov[0, 0]


def test_numerical_jacobian_matches_analytic() -> None:
    x = np.array([1.0, 2.0])
    jac = numerical_jacobian(lambda s: np.array([s[0] * s[1], np.sin(s[0])]), x)
    np.testing.assert_allclose(jac, [[2.0, 1.0], [np.cos(1.0), 0.0]], atol=1e-7)


def test_ekf_tracks_nonlinear_range_model() -> None:
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
    with pytest.raises(ValueError):
        KalmanFilter(_cv_model(), np.zeros(3), np.eye(3))
