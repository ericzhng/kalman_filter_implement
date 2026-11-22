"""Module: kalman_filter.jacobian
Description: Central-difference numerical Jacobian used when no analytic Jacobian is given.
"""

import logging
from collections.abc import Callable

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


def numerical_jacobian(
    fun: Callable[[FloatArray], FloatArray], x: FloatArray, eps: float = 1e-6
) -> FloatArray:
    """Approximate the Jacobian of ``fun`` at ``x`` by central differences.

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
