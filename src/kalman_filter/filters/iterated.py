r"""Module: kalman_filter.filters.iterated

Description: Iterated Extended Kalman Filter (IEKF). Solves the nonlinear measurement
    update as a local Gauss-Newton optimization problem, significantly improving accuracy
    and robustness for highly nonlinear observation geometries (e.g., radar, bearing-only tracking).

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.filters.base import BaseFilter
from kalman_filter.models.base import StateSpaceModel
from kalman_filter.types import FloatArray, GaussianState

logger = logging.getLogger(__name__)

__all__ = ["IteratedExtendedKalmanFilter"]


class IteratedExtendedKalmanFilter(BaseFilter):
    r"""Iterated Extended Kalman Filter with Gauss-Newton measurement updates.

    At each update step, the state estimate is refined iteratively:
    .. math::
        \eta_{j+1} = \hat{x} + K_j \left( \nu_j - H_j (\hat{x} - \eta_j) \right)

    where :math:`\eta_0 = \hat{x}` is the predicted state, :math:`H_j` is the measurement
    Jacobian evaluated at :math:`\eta_j`, and :math:`\nu_j = \text{residual}(z, h(\eta_j))`.

    Missing channels (NaN) are skipped automatically.

    Args:
        model: State-space model implementing ``StateSpaceModel``.
        x0: Initial state mean, shape ``(n,)``.
        P0: Initial state covariance, shape ``(n, n)``.
        max_iter: Maximum number of Gauss-Newton iterations per update (default is 10).
        tol: Convergence tolerance on step norm ``||eta_{j+1} - eta_j||`` (default is 1e-5).
    """

    def __init__(
        self,
        model: StateSpaceModel,
        x0: FloatArray,
        P0: FloatArray,
        max_iter: int = 10,
        tol: float = 1e-5,
    ) -> None:
        super().__init__(model, x0, P0)
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        self.max_iter = max_iter
        self.tol = tol

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        r"""Evaluates the Jacobian of ``f`` at the current mean.

        Args:
            u: Optional control input.

        Returns:
            The Jacobian matrix, shape ``(n, n)``.
        """
        return self.model.state_jacobian(self.x, u)

    def _measurement_matrix(self) -> FloatArray:
        r"""Evaluates the Jacobian of ``h`` at the current mean.

        Returns:
            The Jacobian matrix, shape ``(m, n)``.
        """
        return self.model.measurement_jacobian(self.x)

    def update(self, z: FloatArray) -> GaussianState:
        r"""Performs the iterated Gauss-Newton update, skipping NaN channels.

        Args:
            z: Measurement vector, shape ``(m,)``; NaN indicates missing channels.

        Returns:
            The posterior Gaussian belief.
        """
        z = np.asarray(z, dtype=float)
        observed = np.flatnonzero(~np.isnan(z))
        self.last_innovation = np.full(self.model.meas_dim, np.nan)
        self.last_nis = float("nan")

        if observed.size == 0:
            return self.state

        R = self.model.R[np.ix_(observed, observed)]
        x_prior = self.x.copy()
        P_prior = self.P.copy()
        eta = x_prior.copy()

        H_last = None
        K_last = None
        S_last = None
        nu_last = None

        for _ in range(self.max_iter):
            H_all = self.model.measurement_jacobian(eta)
            H = H_all[observed]
            z_pred = self.model.measure(eta)
            nu = self.model.residual(z, z_pred)[observed]

            S = H @ P_prior @ H.T + R
            K = np.linalg.solve(S, H @ P_prior).T

            # Gauss-Newton step
            eta_next = x_prior + K @ (nu - H @ (x_prior - eta))
            diff = np.linalg.norm(eta_next - eta)
            eta = eta_next
            H_last = H
            K_last = K
            S_last = S
            nu_last = nu

            if diff < self.tol:
                break

        # Joseph-form covariance update using final linearization
        eye = np.eye(self.model.state_dim)
        IKH = eye - K_last @ H_last
        self.x = eta
        self.P = IKH @ P_prior @ IKH.T + K_last @ R @ K_last.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_innovation[observed] = nu_last
        self.last_nis = float(nu_last @ np.linalg.solve(S_last, nu_last))
        return self.state
