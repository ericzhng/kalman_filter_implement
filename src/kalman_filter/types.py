"""Module: kalman_filter.types
Description: Shared data containers (Gaussian belief and filter run results).
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianState:
    """A Gaussian belief N(mean, cov) over the state vector.

    Attributes:
        mean: State mean, shape ``(n,)``.
        cov: State covariance, shape ``(n, n)``.
    """

    mean: FloatArray
    cov: FloatArray


@dataclass(frozen=True)
class FilterResult:
    """Time series produced by ``BaseFilter.run``.

    Row ``k`` corresponds to time ``k + 1``; the prior is time 0. Channels that
    were missing (NaN) at a step have NaN innovations.

    Attributes:
        predicted_means: Prior means after the predict step, shape ``(T, n)``.
        predicted_covs: Prior covariances, shape ``(T, n, n)``.
        filtered_means: Posterior means after the update step, shape ``(T, n)``.
        filtered_covs: Posterior covariances, shape ``(T, n, n)``.
        innovations: Measurement residuals, shape ``(T, m)``.
        nis: Normalised innovation squared per step, shape ``(T,)``.
        dof: Number of observed channels per step, shape ``(T,)``.
    """

    predicted_means: FloatArray
    predicted_covs: FloatArray
    filtered_means: FloatArray
    filtered_covs: FloatArray
    innovations: FloatArray
    nis: FloatArray
    dof: NDArray[np.int_]
