"""Module: kalman_filter.simulation
Description: Generate ground-truth trajectories and noisy multi-channel measurements.
"""

import logging

import numpy as np

from kalman_filter.models.base import StateSpaceModel
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)


def simulate(
    model: StateSpaceModel,
    x0: FloatArray,
    steps: int,
    rng: np.random.Generator,
    controls: FloatArray | None = None,
    dropout: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simulate the model forward from ``x0``.

    Args:
        model: Model providing ``propagate``, ``measure``, ``Q`` and ``R``.
        x0: True initial state, shape ``(n,)``.
        steps: Number of steps ``T``.
        rng: Random generator.
        controls: Optional controls of shape ``(T, p)``.
        dropout: Optional per-channel probability (shape ``(m,)``) that a channel
            is missing at a step; missing values are returned as NaN.

    Returns:
        ``(states, measurements)`` with shapes ``(T + 1, n)`` (row 0 is ``x0``)
        and ``(T, m)`` (row ``k`` measures ``states[k + 1]``).
    """
    n, m = model.state_dim, model.meas_dim
    states = np.empty((steps + 1, n))
    meas = np.empty((steps, m))
    states[0] = x0
    for k in range(steps):
        u = None if controls is None else np.asarray(controls[k], dtype=float)
        w = rng.multivariate_normal(np.zeros(n), model.Q, method="eigh")
        states[k + 1] = model.propagate(states[k], u) + w
        v = rng.multivariate_normal(np.zeros(m), model.R, method="eigh")
        meas[k] = model.measure(states[k + 1]) + v
    if dropout is not None:
        missing = rng.random((steps, m)) < np.asarray(dropout)
        meas[missing] = np.nan
    return states, meas
