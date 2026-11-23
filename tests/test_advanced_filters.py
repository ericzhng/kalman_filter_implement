r"""Module: tests.test_advanced_filters

Description: Unit tests for UKF, IEKF, RTS smoother, and EnKF covariance localization.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from kalman_filter import (
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    IteratedExtendedKalmanFilter,
    KalmanFilter,
    LinearModel,
    NonlinearModel,
    UnscentedKalmanFilter,
    constant_velocity_model,
    gaspari_cohn,
    rts_smooth,
    simulate,
)


def _linear_cv_model() -> LinearModel:
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    return constant_velocity_model(2, 0.1, 0.5, H, np.eye(2))


def test_ukf_matches_kf_on_linear_model() -> None:
    r"""Checks that the UKF reproduces the linear Kalman filter on linear models."""
    model = _linear_cv_model()
    states, z = simulate(model, np.array([0.0, 0.0, 1.0, 1.0]), 60, np.random.default_rng(42))
    x0, P0 = np.zeros(4), 5.0 * np.eye(4)

    kf_res = KalmanFilter(model, x0, P0).run(z)
    ukf_res = UnscentedKalmanFilter(model, x0, P0, alpha=1.0, beta=2.0, kappa=0.0).run(z)

    np.testing.assert_allclose(kf_res.filtered_means, ukf_res.filtered_means, atol=1e-5)
    np.testing.assert_allclose(kf_res.filtered_covs, ukf_res.filtered_covs, atol=1e-5)


def test_ukf_handles_missing_channels() -> None:
    r"""Checks UKF behavior when all channels or partial channels are missing."""
    model = _linear_cv_model()
    ukf = UnscentedKalmanFilter(model, np.zeros(4), np.eye(4))

    # All NaN leaves state unchanged
    before = ukf.state
    after = ukf.update(np.array([np.nan, np.nan]))
    np.testing.assert_array_equal(before.mean, after.mean)

    # Partial observation
    partial = ukf.update(np.array([2.0, np.nan]))
    assert partial.cov[0, 0] < before.cov[0, 0]
    assert np.isnan(ukf.last_innovation[1])
    assert not np.isnan(ukf.last_innovation[0])


def test_iekf_improves_on_severe_nonlinearity() -> None:
    r"""Checks that IEKF converges and reduces error on severe range measurement nonlinearity."""
    # Target close to the sensor: range = sqrt(x^2 + y^2) has severe curvature near (0, 0)
    model = NonlinearModel(
        f=lambda x, u: np.array([x[0] + 0.1 * x[1], x[1]]),
        h=lambda x: np.array([np.hypot(x[0], 0.2)]),
        Q=np.diag([1e-4, 1e-4]),
        R=np.array([[0.01]]),
    )
    states, z = simulate(model, np.array([0.5, -0.1]), 40, np.random.default_rng(7))
    x0 = np.array([0.8, 0.0])
    P0 = np.eye(2)

    ekf_res = ExtendedKalmanFilter(model, x0, P0).run(z)
    iekf_res = IteratedExtendedKalmanFilter(model, x0, P0, max_iter=15, tol=1e-6).run(z)

    ekf_err = np.mean((ekf_res.filtered_means - states[1:]) ** 2)
    iekf_err = np.mean((iekf_res.filtered_means - states[1:]) ** 2)
    assert iekf_err <= ekf_err + 1e-6


def test_rts_smoother_reduces_variance() -> None:
    r"""Checks that RTS smoothing yields lower or equal covariance compared to forward filter."""
    model = _linear_cv_model()
    states, z = simulate(model, np.array([0.0, 0.0, 2.0, 1.0]), 100, np.random.default_rng(10))
    x0, P0 = np.zeros(4), 10.0 * np.eye(4)

    kf_res = KalmanFilter(model, x0, P0).run(z)
    smooth_res = rts_smooth(kf_res, model)

    truth = states[1:]
    kf_rmse = np.sqrt(np.mean((kf_res.filtered_means - truth) ** 2))
    smooth_rmse = np.sqrt(np.mean((smooth_res.smoothed_means - truth) ** 2))

    # Smoothed trajectory should be more accurate
    assert smooth_rmse <= kf_rmse + 1e-6

    # Smoothed variance should be <= filtered variance at every interior step
    for k in range(kf_res.steps - 1):
        for i in range(4):
            assert smooth_res.smoothed_covs[k, i, i] <= kf_res.filtered_covs[k, i, i] + 1e-6


def test_gaspari_cohn_properties() -> None:
    r"""Checks mathematical properties of the Gaspari-Cohn tapering function."""
    cutoff = 5.0
    assert gaspari_cohn(0.0, cutoff) == pytest.approx(1.0)
    assert gaspari_cohn(2.0 * cutoff, cutoff) == pytest.approx(0.0)
    assert gaspari_cohn(3.0 * cutoff, cutoff) == pytest.approx(0.0)

    # Strictly decreasing on [0, 2 * cutoff]
    dists = np.linspace(0.0, 2.0 * cutoff, 20)
    vals = gaspari_cohn(dists, cutoff)
    assert np.all(np.diff(vals) <= 0.0)


def test_enkf_with_localization() -> None:
    r"""Checks that EnKF runs with spatial localization matrix."""
    model = _linear_cv_model()
    states, z = simulate(model, np.array([0.0, 0.0, 1.0, 1.0]), 50, np.random.default_rng(3))
    x0, P0 = np.zeros(4), np.eye(4)

    # Localization matrix: 4 states x 2 measurements
    loc_mat = np.ones((4, 2))
    enkf = EnsembleKalmanFilter(
        model,
        x0,
        P0,
        n_members=50,
        rng=np.random.default_rng(99),
        localization_matrix=loc_mat,
    )
    res = enkf.run(z)
    assert res.filtered_means.shape == (50, 4)
    assert not np.any(np.isnan(res.filtered_means))
