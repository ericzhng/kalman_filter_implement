r"""Module: kalman_filter.filters.unscented

Description: Unscented Kalman filter (UKF) using the Scaled Unscented Transform.
    Propagates a deterministic set of sigma points through nonlinear dynamics
    and measurement functions, capturing posterior mean and covariance to 3rd-order
    Taylor series accuracy for Gaussian inputs without requiring Jacobians.

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

__all__ = ["UnscentedKalmanFilter"]


class UnscentedKalmanFilter(BaseFilter):
    r"""Unscented Kalman Filter with Scaled Unscented Transform and missing channel handling.

    Uses Merwe scaled sigma points determined by scaling parameters ``alpha``,
    ``beta``, and ``kappa``:
        - ``alpha``: Determines the spread of sigma points around the mean (usually 1e-4 to 1).
        - ``beta``: Incorporates prior knowledge of the distribution (2.0 is optimal for Gaussians).
        - ``kappa``: Secondary scaling parameter, typically ``0`` or ``3 - n``.

    Missing measurement channels (indicated by ``NaN``) are skipped adaptively at each step.

    Args:
        model: State-space model implementing ``StateSpaceModel``.
        x0: Initial state mean, shape ``(n,)``.
        P0: Initial state covariance, shape ``(n, n)``.
        alpha: Spread parameter for sigma points (0 < alpha <= 1). Default is 1e-3.
        beta: Distribution parameter (default is 2.0, optimal for Gaussian).
        kappa: Secondary scaling parameter (default is 0.0).

    Raises:
        ValueError: If dimensions mismatch or parameters are out of valid range.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        x0: FloatArray,
        P0: FloatArray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(model, x0, P0)
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        n = self.model.state_dim
        self.lambda_param = self.alpha**2 * (n + self.kappa) - n
        self.gamma = np.sqrt(n + self.lambda_param)

        # Compute weights for mean and covariance reconstruction
        num_sigmas = 2 * n + 1
        self.wm = np.full(num_sigmas, 1.0 / (2.0 * (n + self.lambda_param)))
        self.wc = np.full(num_sigmas, 1.0 / (2.0 * (n + self.lambda_param)))
        self.wm[0] = self.lambda_param / (n + self.lambda_param)
        self.wc[0] = self.lambda_param / (n + self.lambda_param) + (1.0 - self.alpha**2 + self.beta)

    def _transition_matrix(self, u: FloatArray | None) -> FloatArray:
        r"""Not used: UKF propagates sigma points directly through nonlinear functions.

        Args:
            u: Optional control input.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("UnscentedKalmanFilter does not compute transition matrices")

    def _measurement_matrix(self) -> FloatArray:
        r"""Not used: UKF computes statistical cross-covariances via sigma points.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("UnscentedKalmanFilter does not compute measurement matrices")

    def compute_sigma_points(self, mean: FloatArray, cov: FloatArray) -> FloatArray:
        r"""Generates 2n + 1 Merwe scaled sigma points around ``(mean, cov)``.

        Args:
            mean: State mean vector, shape ``(n,)``.
            cov: State covariance matrix, shape ``(n, n)``.

        Returns:
            Sigma points matrix, shape ``(2n + 1, n)``.
        """
        n = mean.shape[0]
        # Robust matrix square root using Cholesky with eigenvalue fallback
        scaled_cov = (n + self.lambda_param) * 0.5 * (cov + cov.T)
        try:
            sqrt_cov = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(scaled_cov)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals))

        sigmas = np.empty((2 * n + 1, n))
        sigmas[0] = mean
        for i in range(n):
            sigmas[i + 1] = mean + sqrt_cov[:, i]
            sigmas[n + i + 1] = mean - sqrt_cov[:, i]
        return sigmas

    def predict(self, u: FloatArray | None = None) -> GaussianState:
        r"""Propagates sigma points through the nonlinear dynamics ``f(x, u)``.

        Args:
            u: Optional control input vector.

        Returns:
            The predicted Gaussian belief.
        """
        sigmas = self.compute_sigma_points(self.x, self.P)
        sigmas_f = np.array([self.model.propagate(s, u) for s in sigmas])

        # Reconstruct predicted mean
        x_pred = np.sum(self.wm[:, None] * sigmas_f, axis=0)

        # Reconstruct predicted covariance
        diff_x = sigmas_f - x_pred[None, :]
        P_pred = (diff_x.T * self.wc) @ diff_x + self.model.Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x = x_pred
        self.P = P_pred
        return self.state

    def update(self, z: FloatArray) -> GaussianState:
        r"""Updates the belief using observed measurement channels, skipping NaNs.

        Args:
            z: Measurement vector, shape ``(m,)``; NaN denotes an unobserved channel.

        Returns:
            The updated posterior Gaussian belief.
        """
        z = np.asarray(z, dtype=float)
        observed = np.flatnonzero(~np.isnan(z))
        self.last_innovation = np.full(self.model.meas_dim, np.nan)
        self.last_nis = float("nan")

        if observed.size == 0:
            return self.state

        sigmas = self.compute_sigma_points(self.x, self.P)
        z_full_sigmas = np.array([self.model.measure(s) for s in sigmas])
        z_obs_sigmas = z_full_sigmas[:, observed]

        # Predicted measurement mean
        z_pred_obs = np.sum(self.wm[:, None] * z_obs_sigmas, axis=0)

        # Innovations between sigma points and predicted measurement
        diff_z = z_obs_sigmas - z_pred_obs[None, :]
        diff_x = sigmas - self.x[None, :]

        # Innovation covariance S and cross-covariance Pxz
        R_obs = self.model.R[np.ix_(observed, observed)]
        S = (diff_z.T * self.wc) @ diff_z + R_obs
        Pxz = (diff_x.T * self.wc) @ diff_z

        # Kalman gain
        K = np.linalg.solve(S, Pxz.T).T

        # True measurement innovation
        z_pred_full = np.empty(self.model.meas_dim)
        z_pred_full[observed] = z_pred_obs
        nu_obs = self.model.residual(z, z_pred_full)[observed]

        # Posterior state and covariance
        self.x = self.x + K @ nu_obs
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_innovation[observed] = nu_obs
        self.last_nis = float(nu_obs @ np.linalg.solve(S, nu_obs))
        return self.state
