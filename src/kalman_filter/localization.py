r"""Module: kalman_filter.localization

Description: Gaspari-Cohn spatial correlation tapering and covariance localization functions
    used in atmospheric and oceanographic ensemble data assimilation.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["gaspari_cohn"]


def gaspari_cohn(distances: FloatArray, cutoff: float) -> FloatArray:
    r"""Computes the 5th-order Gaspari-Cohn correlation tapering function.

    Standard in spatial data assimilation (e.g., meteorology, oceanography) to suppress
    spurious correlations at large distances caused by finite ensemble sizes.

    Args:
        distances: Array of non-negative physical distances between state/sensor locations.
        cutoff: Localization half-width parameter :math:`c > 0`. Correlations vanish at :math:`2 c`.

    Returns:
        Tapering correlation coefficients in :math:`[0, 1]`, same shape as ``distances``.

    Raises:
        ValueError: If ``cutoff <= 0``.
    """
    if cutoff <= 0.0:
        raise ValueError("Cutoff radius must be strictly positive")
    d = np.asarray(distances, dtype=float)
    r = np.abs(d) / cutoff
    rho = np.zeros_like(r)

    # Region 1: 0 <= r <= 1
    mask1 = (r >= 0.0) & (r <= 1.0)
    r1 = r[mask1]
    rho[mask1] = (
        1.0
        - (5.0 / 3.0) * r1**2
        + (5.0 / 8.0) * r1**3
        + 0.5 * r1**4
        - 0.25 * r1**5
    )

    # Region 2: 1 < r <= 2
    mask2 = (r > 1.0) & (r <= 2.0)
    r2 = r[mask2]
    rho[mask2] = (
        4.0
        - 5.0 * r2
        + (5.0 / 3.0) * r2**2
        + (5.0 / 8.0) * r2**3
        - 0.5 * r2**4
        + (1.0 / 12.0) * r2**5
        - 2.0 / (3.0 * r2)
    )

    # Region 3: r > 2 remains 0.0
    return np.clip(rho, 0.0, 1.0)
