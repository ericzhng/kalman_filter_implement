r"""Module: tests.test_metrics

Description: Unit tests for filter evaluation metrics and statistical hypothesis consistency tests.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np

from kalman_filter import (
    autocorrelation,
    chi2_consistency_test,
    mae,
    nees,
    rmse,
    whiteness_test,
)


def test_rmse_and_mae() -> None:
    r"""Checks RMSE and MAE calculation."""
    est = np.array([[1.0, 2.0], [3.0, 4.0]])
    truth = np.array([[0.0, 2.0], [3.0, 2.0]])
    # diffs: [1.0, 0.0], [0.0, 2.0]
    r = rmse(est, truth)
    m = mae(est, truth)
    np.testing.assert_allclose(r, [np.sqrt(0.5), np.sqrt(2.0)])
    np.testing.assert_allclose(m, [0.5, 1.0])


def test_chi2_consistency_test_synthetic() -> None:
    r"""Checks Chi-square consistency hypothesis test on consistent vs biased data."""
    rng = np.random.default_rng(42)
    # Consistent NEES for 4-state filter: values distributed as chi2(4)
    # Sample mean should be close to 4.0
    from scipy.stats import chi2

    sample = chi2.rvs(df=4, size=500, random_state=rng)
    passed, sample_mean, (low, high) = chi2_consistency_test(sample, dof=4, alpha=0.05)
    assert passed
    assert low <= sample_mean <= high

    # Severely biased / overconfident filter
    biased_sample = sample * 5.0
    passed_biased, _, _ = chi2_consistency_test(biased_sample, dof=4, alpha=0.05)
    assert not passed_biased


def test_autocorrelation_and_whiteness() -> None:
    r"""Checks autocorrelation computation and whiteness testing on white vs colored noise."""
    rng = np.random.default_rng(123)
    # White noise
    white = rng.standard_normal(1000)
    raw_corr = autocorrelation(white, max_lag=3)
    assert raw_corr[0] == 1.0

    is_white, corrs = whiteness_test(white, max_lag=5)
    assert is_white
    assert corrs[0] == 1.0
    assert np.all(np.abs(corrs[1:]) < 0.1)


    # Colored / autoregressive noise AR(1)
    colored = np.zeros(1000)
    for i in range(1, 1000):
        colored[i] = 0.8 * colored[i - 1] + rng.standard_normal()
    is_white_col, _ = whiteness_test(colored, max_lag=5)
    assert not is_white_col


def test_nees_calculation() -> None:
    r"""Checks NEES vector computation."""
    means = np.array([[1.0, 2.0]])
    covs = np.array([[[1.0, 0.0], [0.0, 2.0]]])
    truth = np.array([[0.0, 0.0]])
    # err = [1, 2], err^T P^-1 err = 1/1 + 4/2 = 3.0
    val = nees(means, covs, truth)
    assert val[0] == 3.0
