r"""Module: kalman_filter.applications.finance

Description: Standard Kalman Filter applications in linear time-series and quantitative finance:
    - Local Linear Trend model for signal smoothing and trend/drift extraction.
    - Dynamic Linear Regression and Pairs Trading model for online hedge ratio estimation.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.models.linear import LinearModel
from kalman_filter.types import FloatArray, GaussianState

logger = logging.getLogger(__name__)

__all__ = ["DynamicHedgeRatioFilter", "LocalLinearTrendModel"]


class LocalLinearTrendModel(LinearModel):
    r"""Local Linear Trend state-space model for economic and financial time series.

    Decomposes an observed price or economic series :math:`y_t` into a stochastic level
    :math:`\mu_t` and stochastic slope :math:`\beta_t`:
    .. math::
        \mu_t = \mu_{t-1} + \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
    .. math::
        \beta_t = \beta_{t-1} + \zeta_t, \quad \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2)
    .. math::
        y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)


    Args:
        level_var: Process noise variance for level innovation :math:`\sigma_\eta^2`.
        slope_var: Process noise variance for slope/trend innovation :math:`\sigma_\zeta^2`.
        obs_var: Observation noise variance :math:`\sigma_\epsilon^2`.
    """

    def __init__(
        self,
        level_var: float = 1e-4,
        slope_var: float = 1e-5,
        obs_var: float = 1e-2,
    ) -> None:
        if level_var < 0 or slope_var < 0 or obs_var <= 0:
            raise ValueError("Variances must be non-negative, and obs_var strictly positive")

        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([level_var, slope_var])
        R = np.array([[obs_var]])
        super().__init__(F=F, H=H, Q=Q, R=R)


class DynamicHedgeRatioFilter:
    r"""Dynamic Linear Regression for algorithmic pairs trading and rolling beta estimation.

    Tracks the time-varying cointegration relation between asset Y and asset X:
    .. math::
        y_t = \alpha_t + \beta_t x_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, R)

    where the state vector is :math:`\theta_t = [\alpha_t, \beta_t]^T` modeled as a random walk:
    :math:`\theta_t = \theta_{t-1} + w_t, w_t \sim \mathcal{N}(0, Q)`.

    Args:
        delta: Random-walk process noise covariance parameter or 2x2 matrix :math:`Q`.
        obs_noise_var: Measurement error variance :math:`R`.
        initial_alpha: Initial intercept guess.
        initial_beta: Initial slope / hedge ratio guess.
        p0_diag: Diagonal variance of initial uncertainty.
    """

    def __init__(
        self,
        delta: float | FloatArray = 1e-4,
        obs_noise_var: float = 1e-2,
        initial_alpha: float = 0.0,
        initial_beta: float = 1.0,
        p0_diag: float = 1.0,
    ) -> None:
        self.state_mean = np.array([initial_alpha, initial_beta], dtype=float)
        self.state_cov = p0_diag * np.eye(2)

        if np.isscalar(delta):
            self.Q = float(delta) * np.eye(2)
        else:
            self.Q = np.atleast_2d(np.asarray(delta, dtype=float))

        self.R = float(obs_noise_var)

    @property
    def alpha(self) -> float:
        r"""Current estimated intercept.

        Returns:
            Scalar intercept.
        """
        return float(self.state_mean[0])

    @property
    def beta(self) -> float:
        r"""Current estimated hedge ratio.

        Returns:
            Scalar hedge ratio.
        """
        return float(self.state_mean[1])

    def step(self, y: float, x: float) -> tuple[GaussianState, float]:
        r"""Updates belief with a new pair observation :math:`(y_t, x_t)`.

        Args:
            y: Price of dependent asset Y.
            x: Price of independent asset X.

        Returns:
            A tuple ``(state, spread)`` where ``state`` is the updated parameter belief
            and ``spread`` is the regression residual :math:`y - (\alpha + \beta x)`.
        """
        # Time update: random walk
        P_pred = self.state_cov + self.Q
        x_pred = self.state_mean.copy()

        # Measurement update: H_t = [1.0, x]
        H_t = np.array([[1.0, float(x)]])
        y_pred = float((H_t @ x_pred)[0])
        spread = float(y - y_pred)

        S = float((H_t @ P_pred @ H_t.T)[0, 0] + self.R)
        K = (P_pred @ H_t.T) / S


        self.state_mean = x_pred + (K * spread).ravel()
        IKH = np.eye(2) - K @ H_t
        self.state_cov = IKH @ P_pred @ IKH.T + (K * self.R) @ K.T
        self.state_cov = 0.5 * (self.state_cov + self.state_cov.T)

        return GaussianState(self.state_mean.copy(), self.state_cov.copy()), spread

    def run(
        self,
        prices_y: FloatArray,
        prices_x: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        r"""Processes full price time series of pairs Y and X.

        Args:
            prices_y: 1D array of asset Y prices, shape ``(T,)``.
            prices_x: 1D array of asset X prices, shape ``(T,)``.

        Returns:
            A tuple ``(alphas, betas, spreads)``:
                - ``alphas``: Array of time-varying intercepts, shape ``(T,)``.
                - ``betas``: Array of time-varying hedge ratios, shape ``(T,)``.
                - ``spreads``: Array of residuals/spreads, shape ``(T,)``.
        """
        y_arr = np.asarray(prices_y, dtype=float)
        x_arr = np.asarray(prices_x, dtype=float)
        T = len(y_arr)
        if len(x_arr) != T:
            raise ValueError("prices_y and prices_x must have equal length")

        alphas = np.empty(T)
        betas = np.empty(T)
        spreads = np.empty(T)

        for t in range(T):
            _, spread = self.step(y_arr[t], x_arr[t])
            alphas[t] = self.alpha
            betas[t] = self.beta
            spreads[t] = spread

        return alphas, betas, spreads
