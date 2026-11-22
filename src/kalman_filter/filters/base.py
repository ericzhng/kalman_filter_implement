"""Module: kalman_filter.filters.base
Description: Shared predict/update/run machinery; subclasses supply the linearisation.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from kalman_filter.models.base import StateSpaceModel
from kalman_filter.types import FilterResult, FloatArray, GaussianState

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """Gaussian filter with a covariance-form predict step and Joseph-form update.

    Missing measurement channels (NaN entries in ``z``) are skipped, which makes
    asynchronous multi-sensor data a first-class use case.

    Args:
        model: State-space model providing ``f``, ``h``, ``Q``, ``R``.
        x0: Initial mean, shape ``(n,)``.
        P0: Initial covariance, shape ``(n, n)``.

    Raises:
        ValueError: If ``x0``/``P0`` do not match the model state dimension.
    """

    def __init__(self, model: StateSpaceModel, x0: FloatArray, P0: FloatArray) -> None:
        self.model = model
        x = np.asarray(x0, dtype=float).copy()
        P = np.atleast_2d(np.asarray(P0, dtype=float)).copy()
        n = model.state_dim
        if x.shape != (n,) or P.shape != (n, n):
            raise ValueError(f"x0 must be ({n},) and P0 must be ({n}, {n})")
        self.x = x
        self.P = P
        self.last_innovation: FloatArray | None = None
        self.last_nis: float = float("nan")

    @property
    def state(self) -> GaussianState:
        """Current belief as an immutable snapshot."""
        return GaussianState(self.x.copy(), self.P.copy())

    @abstractmethod
    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        """Return the (linearised) transition matrix at the current mean."""

    @abstractmethod
    def _measurement_matrix(self) -> FloatArray:
        """Return the (linearised) measurement matrix at the current mean."""

    def predict(self, u: FloatArray | None = None) -> GaussianState:
        """Propagate the belief one step: ``x <- f(x, u)``, ``P <- F P F^T + Q``.

        Args:
            u: Optional control input.

        Returns:
            The predicted belief.
        """
        F = self._transition_matrix(u)
        self.x = self.model.propagate(self.x, u)
        self.P = F @ self.P @ F.T + self.model.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.state

    def update(self, z: FloatArray) -> GaussianState:
        """Fuse a measurement vector, ignoring NaN channels.

        Args:
            z: Measurement, shape ``(m,)``; NaN marks a missing channel.

        Returns:
            The posterior belief (unchanged if every channel is missing).
        """
        z = np.asarray(z, dtype=float)
        observed = np.flatnonzero(~np.isnan(z))
        self.last_innovation = np.full(self.model.meas_dim, np.nan)
        self.last_nis = float("nan")
        if observed.size == 0:
            return self.state

        H = self._measurement_matrix()[observed]
        R = self.model.R[np.ix_(observed, observed)]
        nu = self.model.residual(z, self.model.measure(self.x))[observed]

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        eye = np.eye(self.model.state_dim)
        IKH = eye - K @ H
        self.x = self.x + K @ nu
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

        self.last_innovation[observed] = nu
        self.last_nis = float(nu @ np.linalg.solve(S, nu))
        return self.state

    def step(self, z: FloatArray, u: FloatArray | None = None) -> GaussianState:
        """Run ``predict`` then ``update``."""
        self.predict(u)
        return self.update(z)

    def run(self, measurements: FloatArray, controls: FloatArray | None = None) -> FilterResult:
        """Filter a whole measurement sequence starting from the current belief.

        Args:
            measurements: Array of shape ``(T, m)``; NaN marks missing channels.
            controls: Optional controls of shape ``(T, p)``.

        Returns:
            A ``FilterResult`` with per-step predicted/filtered beliefs and diagnostics.
        """
        z_all = np.asarray(measurements, dtype=float)
        T = z_all.shape[0]
        n, m = self.model.state_dim, self.model.meas_dim
        pm, pP = np.empty((T, n)), np.empty((T, n, n))
        fm, fP = np.empty((T, n)), np.empty((T, n, n))
        innov, nis = np.empty((T, m)), np.empty(T)
        for k in range(T):
            u = None if controls is None else np.asarray(controls[k], dtype=float)
            pred = self.predict(u)
            pm[k], pP[k] = pred.mean, pred.cov
            post = self.update(z_all[k])
            fm[k], fP[k] = post.mean, post.cov
            innov[k] = self.last_innovation
            nis[k] = self.last_nis
        dof = np.sum(~np.isnan(z_all), axis=1).astype(int)
        return FilterResult(pm, pP, fm, fP, innov, nis, dof)
