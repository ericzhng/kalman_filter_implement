"""Module: kalman_filter.metrics
Description: Accuracy and consistency metrics (RMSE, NEES) for evaluating filters.
"""

import logging

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


def rmse(estimate: FloatArray, truth: FloatArray) -> FloatArray:
    """Per-component root-mean-square error over time.

    Args:
        estimate: Estimates, shape ``(T, d)``.
        truth: Ground truth, shape ``(T, d)``.

    Returns:
        RMSE per component, shape ``(d,)``; NaN entries are ignored.
    """
    return np.sqrt(np.nanmean((estimate - truth) ** 2, axis=0))


def nees(means: FloatArray, covs: FloatArray, truth: FloatArray) -> FloatArray:
    """Normalised estimation error squared ``e^T P^-1 e`` per step.

    For a consistent filter its average is close to the state dimension.

    Args:
        means: Estimated means, shape ``(T, n)``.
        covs: Estimated covariances, shape ``(T, n, n)``.
        truth: Ground truth, shape ``(T, n)``.

    Returns:
        NEES per step, shape ``(T,)``.
    """
    err = means - truth
    return np.einsum("ti,ti->t", err, np.linalg.solve(covs, err[..., None])[..., 0])
