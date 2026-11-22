r"""Module: kalman_filter.jacobian

Description: Central-difference numerical Jacobian, used when no analytic Jacobian is given.

Example:
    >>> import numpy as np
    >>> from kalman_filter.jacobian import numerical_jacobian
    >>> numerical_jacobian(lambda s: np.array([2 * s[0], s[0] * s[1]]),
    ...                    np.array([1.0, 3.0])).round(4).tolist()
    [[2.0, 0.0], [3.0, 1.0]]

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from collections.abc import Callable

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["numerical_jacobian"]


def numerical_jacobian(
    fun: Callable[[FloatArray], FloatArray], x: FloatArray, eps: float = 1e-6
) -> FloatArray:
    r"""Approximates the Jacobian of ``fun`` at ``x`` by central differences.

    Args:
        fun: Vector function mapping shape ``(n,)`` to shape ``(m,)``.
        x: Evaluation point, shape ``(n,)``.
        eps: Relative step size.

    Returns:
        Jacobian matrix with shape ``(m, n)``.
    """
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(x.size):
        step = eps * max(1.0, abs(x[i]))
        dx = np.zeros_like(x)
        dx[i] = step
        cols.append((np.asarray(fun(x + dx)) - np.asarray(fun(x - dx))) / (2.0 * step))
    return np.stack(cols, axis=1)
