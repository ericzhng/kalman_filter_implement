r"""Module: kalman_filter.types

Description: Shared data containers: Gaussian belief, forward filtering results, and
    backward smoothing results.

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["FilterResult", "FloatArray", "GaussianState", "SmootherResult"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianState:
    r"""A Gaussian belief N(mean, cov) over the state vector.

    Attributes:
        mean: State mean vector, shape ``(n,)``.
        cov: State covariance matrix, shape ``(n, n)``.
    """

    mean: FloatArray
    cov: FloatArray

    @property
    def dim(self) -> int:
        r"""Dimension of the state vector.

        Returns:
            State dimension ``n``.
        """
        return self.mean.shape[0]

    @property
    def std(self) -> FloatArray:
        r"""Standard deviations (square root of diagonal variances).

        Returns:
            Standard deviations vector, shape ``(n,)``.
        """
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))


@dataclass(frozen=True)
class FilterResult:
    r"""Holds the time series produced by a filter run.

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

    @property
    def steps(self) -> int:
        r"""Number of filtered time steps ``T``.

        Returns:
            Number of steps.
        """
        return self.filtered_means.shape[0]

    @property
    def state_dim(self) -> int:
        r"""State dimension ``n``.

        Returns:
            Dimension of state vector.
        """
        return self.filtered_means.shape[1]

    @property
    def meas_dim(self) -> int:
        r"""Measurement channel dimension ``m``.

        Returns:
            Total number of measurement channels.
        """
        return self.innovations.shape[1]

    @property
    def filtered_stds(self) -> FloatArray:
        r"""Posterior standard deviations over time, shape ``(T, n)``.

        Returns:
            Square root of posterior diagonal variances.
        """
        variances = np.diagonal(self.filtered_covs, axis1=1, axis2=2)
        return np.sqrt(np.maximum(variances, 0.0))

    def to_dict(self) -> dict[str, Any]:
        r"""Converts the result arrays into a dictionary.

        Returns:
            Dictionary mapping attribute names to numpy arrays.
        """
        return {
            "predicted_means": self.predicted_means,
            "predicted_covs": self.predicted_covs,
            "filtered_means": self.filtered_means,
            "filtered_covs": self.filtered_covs,
            "innovations": self.innovations,
            "nis": self.nis,
            "dof": self.dof,
        }


@dataclass(frozen=True)
class SmootherResult:
    r"""Holds the smoothed trajectory produced by the Rauch-Tung-Striebel (RTS) smoother.

    Attributes:
        smoothed_means: Smoothed posterior means, shape ``(T, n)``.
        smoothed_covs: Smoothed posterior covariances, shape ``(T, n, n)``.
        smoother_gains: Smoother gain matrices ``C_k``, shape ``(T - 1, n, n)``.
    """

    smoothed_means: FloatArray
    smoothed_covs: FloatArray
    smoother_gains: FloatArray

    @property
    def steps(self) -> int:
        r"""Number of smoothed time steps ``T``.

        Returns:
            Number of steps.
        """
        return self.smoothed_means.shape[0]

    @property
    def state_dim(self) -> int:
        r"""State dimension ``n``.

        Returns:
            State dimension.
        """
        return self.smoothed_means.shape[1]

    @property
    def smoothed_stds(self) -> FloatArray:
        r"""Smoothed standard deviations over time, shape ``(T, n)``.

        Returns:
            Square root of diagonal elements of smoothed covariances.
        """
        variances = np.diagonal(self.smoothed_covs, axis1=1, axis2=2)
        return np.sqrt(np.maximum(variances, 0.0))

    def to_dict(self) -> dict[str, Any]:
        r"""Converts the smoother result arrays into a dictionary.

        Returns:
            Dictionary of arrays.
        """
        return {
            "smoothed_means": self.smoothed_means,
            "smoothed_covs": self.smoothed_covs,
            "smoother_gains": self.smoother_gains,
        }

