r"""Module: kalman_filter.filters.ensemble

Description: Stochastic Ensemble Kalman filter (perturbed observations) with support for
    multiplicative covariance inflation and Gaspari-Cohn spatial covariance localization.
    The belief is represented by a Monte-Carlo ensemble propagated through true nonlinear
    dynamics without linearisation or Jacobians.

Example:
    >>> import numpy as np
    >>> from kalman_filter import EnsembleKalmanFilter, LinearModel
    >>> model = LinearModel(F=[[1.0]], H=[[1.0]], Q=[[1.0]], R=[[1.0]])
    >>> enkf = EnsembleKalmanFilter(model, [0.0], [[1.0]], n_members=50)
    >>> enkf.ensemble.shape
    (50, 1)

Author: Hui Zhang
Date: 2026-11-22
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.filters.base import BaseFilter
from kalman_filter.localization import gaspari_cohn
from kalman_filter.models.base import StateSpaceModel
from kalman_filter.types import FloatArray, GaussianState

logger = logging.getLogger(__name__)

__all__ = ["EnsembleKalmanFilter", "gaspari_cohn"]



class EnsembleKalmanFilter(BaseFilter):
    r"""Stochastic EnKF with perturbed observations and spatial covariance localization.

    Missing channels (NaN) are skipped as in the other filters. Only ``propagate``,
    ``measure`` and ``residual`` of the model are used, so Jacobians are never required.

    Args:
        model: State-space model providing ``propagate``, ``measure``, ``Q``, ``R``.
        x0: Initial mean, shape ``(n,)``.
        P0: Initial covariance, shape ``(n, n)``.
        n_members: Ensemble size ``N`` (at least 2).
        rng: Random generator; a fresh default one is created if omitted.
        inflation: Multiplicative covariance inflation (>= 1) applied to the ensemble
            anomalies after each update, to counter sampling-induced underdispersion.
        localization_matrix: Optional cross-covariance tapering matrix of shape ``(n, m)``
            (e.g., computed via :func:`gaspari_cohn`) to eliminate spurious long-range correlations.

    Raises:
        ValueError: If ``n_members < 2``, ``inflation < 1``, or localization matrix shape invalid.
    """


    def __init__(
        self,
        model: StateSpaceModel,
        x0: FloatArray,
        P0: FloatArray,
        n_members: int = 100,
        rng: np.random.Generator | None = None,
        inflation: float = 1.0,
        localization_matrix: FloatArray | None = None,
    ) -> None:
        super().__init__(model, x0, P0)
        if n_members < 2:
            raise ValueError("n_members must be at least 2")
        if inflation < 1.0:
            raise ValueError("inflation must be >= 1")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.inflation = inflation

        if localization_matrix is not None:
            loc = np.asarray(localization_matrix, dtype=float)
            if loc.shape != (model.state_dim, model.meas_dim):
                raise ValueError(
                    f"localization_matrix must be shape ({model.state_dim}, {model.meas_dim})"
                )
            self.localization_matrix: FloatArray | None = loc
        else:
            self.localization_matrix = None

        self.ensemble: FloatArray = self.rng.multivariate_normal(
            self.x, self.P, size=n_members, method="eigh"
        )
        self._sync()

    def _sync(self) -> None:
        r"""Refreshes ``x`` and ``P`` from the ensemble sample statistics."""
        self.x = self.ensemble.mean(axis=0)
        self.P = np.atleast_2d(np.cov(self.ensemble, rowvar=False))

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        r"""Not used: the ensemble replaces linearisation.

        Args:
            u: Optional control input.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("EnsembleKalmanFilter does not linearise the model")

    def _measurement_matrix(self) -> FloatArray:
        r"""Not used: the ensemble replaces linearisation.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("EnsembleKalmanFilter does not linearise the model")

    def predict(self, u: FloatArray | None = None) -> GaussianState:
        r"""Propagates every member through ``f`` and adds process noise ``w ~ N(0, Q)``.

        Args:
            u: Optional control input.

        Returns:
            The predicted ensemble mean and covariance.
        """
        N, n = self.ensemble.shape
        w = self.rng.multivariate_normal(np.zeros(n), self.model.Q, size=N, method="eigh")
        self.ensemble = np.array([self.model.propagate(x, u) for x in self.ensemble]) + w
        self._sync()
        return self.state

    def update(self, z: FloatArray) -> GaussianState:
        r"""Fuses a measurement using perturbed observations, ignoring NaN channels.

        Each member is updated with ``K (z + v_i - h(x_i))``, ``v_i ~ N(0, R)``, where the
        gain ``K = P_xy (P_yy + R)^-1`` is built from ensemble cross-covariances.
        If ``localization_matrix`` is set, ``P_xy`` is tapered by Schur product.

        Args:
            z: Measurement, shape ``(m,)``; NaN marks a missing channel.

        Returns:
            The posterior ensemble mean and covariance (unchanged if all channels missing).
        """
        z = np.asarray(z, dtype=float)
        observed = np.flatnonzero(~np.isnan(z))
        self.last_innovation = np.full(self.model.meas_dim, np.nan)
        self.last_nis = float("nan")
        if observed.size == 0:
            return self.state

        N = self.ensemble.shape[0]
        R = self.model.R[np.ix_(observed, observed)]
        Y_full = np.array([self.model.measure(x) for x in self.ensemble])
        Y = Y_full[:, observed]
        A = self.ensemble - self.ensemble.mean(axis=0)
        B = Y - Y.mean(axis=0)
        Pxy = A.T @ B / (N - 1)

        # Apply spatial localization if provided
        if self.localization_matrix is not None:
            Pxy = Pxy * self.localization_matrix[:, observed]

        S = B.T @ B / (N - 1) + R
        K = np.linalg.solve(S, Pxy.T).T

        v = self.rng.multivariate_normal(np.zeros(observed.size), R, size=N, method="eigh")
        z_pert = np.tile(z, (N, 1))
        z_pert[:, observed] += v
        nu_i = np.array(
            [self.model.residual(zp, yp)[observed] for zp, yp in zip(z_pert, Y_full, strict=True)]
        )
        self.ensemble = self.ensemble + nu_i @ K.T
        if self.inflation > 1.0:
            mean = self.ensemble.mean(axis=0)
            self.ensemble = mean + np.sqrt(self.inflation) * (self.ensemble - mean)
        self._sync()

        nu = self.model.residual(z, Y_full.mean(axis=0))[observed]
        self.last_innovation[observed] = nu
        self.last_nis = float(nu @ np.linalg.solve(S, nu))
        return self.state

