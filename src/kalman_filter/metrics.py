r"""Module: kalman_filter.metrics

Description: Statistical validation, estimation accuracy, and filter consistency metrics:
    - RMSE: Root-Mean-Square Error
    - MAE: Mean Absolute Error
    - NEES: Normalized Estimation Error Squared
    - Chi-square Consistency Test for NEES and NIS
    - Innovation Whiteness / Autocorrelation Test

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = [
    "autocorrelation",
    "chi2_consistency_test",
    "mae",
    "nees",
    "rmse",
    "whiteness_test",
]


def rmse(estimate: FloatArray, truth: FloatArray) -> FloatArray:
    r"""Computes the per-component root-mean-square error over time.

    Args:
        estimate: Estimates, shape ``(T, d)``.
        truth: Ground truth, shape ``(T, d)``.

    Returns:
        RMSE per component, shape ``(d,)``; NaN entries are ignored.
    """
    return np.sqrt(np.nanmean((np.asarray(estimate) - np.asarray(truth)) ** 2, axis=0))


def mae(estimate: FloatArray, truth: FloatArray) -> FloatArray:
    r"""Computes the per-component mean absolute error over time.

    Args:
        estimate: Estimates, shape ``(T, d)``.
        truth: Ground truth, shape ``(T, d)``.

    Returns:
        MAE per component, shape ``(d,)``; NaN entries are ignored.
    """
    return np.nanmean(np.abs(np.asarray(estimate) - np.asarray(truth)), axis=0)


def nees(means: FloatArray, covs: FloatArray, truth: FloatArray) -> FloatArray:
    r"""Computes the normalised estimation error squared :math:`e_k^T P_k^{-1} e_k` per step.

    For a consistent, unbiased Gaussian filter, :math:`\mathbb{E}[\text{NEES}] = n`,
    where :math:`n` is the state dimension.

    Args:
        means: Estimated means, shape ``(T, n)``.
        covs: Estimated covariances, shape ``(T, n, n)``.
        truth: Ground truth, shape ``(T, n)``.

    Returns:
        NEES per step, shape ``(T,)``.
    """
    err = np.asarray(means) - np.asarray(truth)
    return np.einsum("ti,ti->t", err, np.linalg.solve(covs, err[..., None])[..., 0])


def chi2_consistency_test(
    sample_values: FloatArray,
    dof: int,
    alpha: float = 0.05,
) -> tuple[bool, float, tuple[float, float]]:
    r"""Performs a two-sided Chi-square test for filter consistency (NEES or NIS).

    Under consistency, the sample mean :math:`\bar{\epsilon} = \frac{1}{K} \sum_{k=1}^K \epsilon_k`
    satisfies :math:`K \bar{\epsilon} \sim \chi^2(K \cdot \text{dof})`.
    The test passes if :math:`\bar{\epsilon}` falls within the :math:`(1 - \alpha)`
    confidence interval.

    Args:
        sample_values: 1D array of NEES or NIS values over time.
        dof: Degrees of freedom per step (e.g. state dim :math:`n` or observed count :math:`m`).
        alpha: Significance level (default 0.05 for 95% confidence interval).



    Returns:
        A tuple ``(passed, sample_mean, (lower_bound, upper_bound))``:
            - ``passed``: Boolean indicating if the filter consistency hypothesis is accepted.
            - ``sample_mean``: Observed average value.
            - ``(lower_bound, upper_bound)``: Theoretical bounds for the sample mean.
    """
    valid_vals = np.asarray(sample_values, dtype=float)
    valid_vals = valid_vals[~np.isnan(valid_vals)]
    K = len(valid_vals)
    if K == 0 or dof <= 0:
        return False, float("nan"), (float("nan"), float("nan"))

    mean_val = float(np.mean(valid_vals))
    total_dof = K * dof

    # Try scipy.stats.chi2 if available; otherwise use Wilson-Hilferty transformation
    try:
        from scipy.stats import chi2

        low_chi2 = chi2.ppf(0.5 * alpha, total_dof)
        high_chi2 = chi2.ppf(1.0 - 0.5 * alpha, total_dof)
    except ImportError:
        # Standard normal inverse approximation (Winitzki approximation for probit)
        def _norm_ppf(p: float) -> float:
            # Rational approximation for central & tails
            q = p if p < 0.5 else 1.0 - p
            r = np.sqrt(-2.0 * np.log(q))
            # Hastings approximation
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            xp = r - (c0 + c1 * r + c2 * r**2) / (1.0 + d1 * r + d2 * r**2 + d3 * r**3)
            return -xp if p < 0.5 else xp

        z_low = _norm_ppf(0.5 * alpha)
        z_high = _norm_ppf(1.0 - 0.5 * alpha)

        # Wilson-Hilferty approximation for Chi-Square distribution
        def _wh_chi2(z: float, k: float) -> float:
            term = 1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))
            return max(0.0, k * (term**3))

        low_chi2 = _wh_chi2(z_low, total_dof)
        high_chi2 = _wh_chi2(z_high, total_dof)

    low_bound = float(low_chi2 / K)
    high_bound = float(high_chi2 / K)
    passed = bool(low_bound <= mean_val <= high_bound)
    return passed, mean_val, (low_bound, high_bound)


def autocorrelation(series: FloatArray, max_lag: int = 10) -> FloatArray:
    r"""Computes the sample autocorrelation of a 1D time series up to ``max_lag``.

    Args:
        series: 1D array of scalar innovations or errors.
        max_lag: Maximum lag to compute (default 10).

    Returns:
        Autocorrelation array of shape ``(max_lag + 1,)`` with lag 0 equal to 1.0.
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    N = len(s)
    if N <= 1:
        return np.ones(max_lag + 1)

    s_centered = s - np.mean(s)
    var = np.sum(s_centered**2)
    if var == 0:
        return np.zeros(max_lag + 1)

    corrs = [1.0]
    for lag in range(1, max_lag + 1):
        if lag >= N:
            corrs.append(0.0)
        else:
            c = np.sum(s_centered[:-lag] * s_centered[lag:]) / var
            corrs.append(float(c))
    return np.array(corrs)


def whiteness_test(
    innovations: FloatArray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> tuple[bool, FloatArray]:
    r"""Tests if the innovations sequence is white noise (zero autocorrelation for lag > 0).

    Uses the 95% Bartlett large-sample bounds :math:`\pm \frac{1.96}{\sqrt{N}}`.

    Args:
        innovations: 1D or 2D array of innovations. If 2D, whiteness is checked per channel.
        max_lag: Maximum lag to test.
        alpha: Significance level (default 0.05).

    Returns:
        A tuple ``(is_white, lag_correlations)``:
            - ``is_white``: True if all correlations (lag 1 to max_lag) are within Bartlett bounds.
            - ``lag_correlations``: Computed autocorrelation coefficients.
    """
    inn = np.asarray(innovations, dtype=float)
    if inn.ndim == 1:
        valid_n = np.sum(~np.isnan(inn))
        corrs = autocorrelation(inn, max_lag=max_lag)
        bound = 1.96 / np.sqrt(max(valid_n, 1))
        # Exclude lag 0
        is_white = bool(np.all(np.abs(corrs[1:]) <= bound))
        return is_white, corrs

    # Multi-channel: test each channel individually
    m = inn.shape[1]
    all_corrs = []
    is_white = True
    for ch in range(m):
        ch_data = inn[:, ch]
        valid_n = np.sum(~np.isnan(ch_data))
        c = autocorrelation(ch_data, max_lag=max_lag)
        all_corrs.append(c)
        bound = 1.96 / np.sqrt(max(valid_n, 1))
        if np.any(np.abs(c[1:]) > bound):
            is_white = False

    return is_white, np.array(all_corrs)
