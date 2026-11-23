r"""Module: tests.test_kf_applications

Description: Unit tests for Standard Kalman Filter applications in target/radar tracking,
    linear time-series/quantitative finance, and meteorology/signal processing.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from kalman_filter import (
    DynamicHedgeRatioFilter,
    HarmonicSignalModel,
    KalmanFilter,
    LinearForecastAssimilationModel,
    LocalLinearTrendModel,
    RadarTrackingModel,
    create_3d_radar_tracker,
    simulate,
)


def test_radar_tracking_3d_with_doppler() -> None:
    r"""Checks 3D radar tracking with position and Doppler velocity detections."""
    model = create_3d_radar_tracker(dt=0.1, pos_std=2.0, vel_std=0.5, jerk_std=0.2)
    assert model.state_dim == 9  # [x, y, z, vx, vy, vz, ax, ay, az]
    assert model.meas_dim == 6  # 3 pos + 3 vel

    x0 = np.array([100.0, 50.0, 20.0, 15.0, -10.0, 5.0, 0.5, 0.2, -0.1])
    states, z = simulate(model, x0, steps=50, rng=np.random.default_rng(42))

    P0 = np.eye(9) * 10.0
    kf = KalmanFilter(model, x0=np.zeros(9), P0=P0)
    res = kf.run(z)

    # Filter should converge close to ground truth
    pos_err = np.linalg.norm(res.filtered_means[-1, :3] - states[-1, :3])
    assert pos_err < 5.0


def test_radar_tracking_2d() -> None:
    r"""Checks 2D radar tracking without velocity channels."""
    model = RadarTrackingModel(dim=2, dt=0.2, measure_velocity=False)
    assert model.state_dim == 6
    assert model.meas_dim == 2


def test_local_linear_trend_finance() -> None:
    r"""Checks Local Linear Trend model for financial price trend extraction."""
    model = LocalLinearTrendModel(level_var=1e-3, slope_var=1e-4, obs_var=0.1)
    assert model.state_dim == 2
    assert model.meas_dim == 1

    # Linear upward trend: y_t = 100 + 0.5 * t + noise
    t = np.arange(100)
    prices = 100.0 + 0.5 * t + np.random.default_rng(0).normal(0.0, 0.3, size=100)
    z = prices[:, None]

    kf = KalmanFilter(model, x0=np.array([100.0, 0.0]), P0=np.eye(2) * 5.0)
    res = kf.run(z)

    # Estimated slope should converge close to 0.5
    estimated_slope = res.filtered_means[-1, 1]
    assert estimated_slope == pytest.approx(0.5, abs=0.15)


def test_dynamic_hedge_ratio_pairs_trading() -> None:
    r"""Checks Dynamic Hedge Ratio estimation for pairs trading."""
    rng = np.random.default_rng(7)
    T = 200
    # True relation: Y_t = 5.0 + 1.5 * X_t + noise
    x = 50.0 + np.cumsum(rng.normal(0, 1, size=T))
    true_beta = 1.5
    true_alpha = 5.0
    y = true_alpha + true_beta * x + rng.normal(0, 0.5, size=T)

    filter_pair = DynamicHedgeRatioFilter(
        delta=1e-4, obs_noise_var=0.25, initial_alpha=0.0, initial_beta=1.0
    )
    alphas, betas, spreads = filter_pair.run(y, x)

    # Should converge near alpha=5.0, beta=1.5
    assert betas[-1] == pytest.approx(true_beta, abs=0.2)
    assert alphas[-1] == pytest.approx(true_alpha, abs=7.0)
    assert len(spreads) == T



def test_linear_forecast_assimilation_meteorology() -> None:
    r"""Checks linear advection-diffusion atmospheric forecast assimilation."""
    model = LinearForecastAssimilationModel(
        num_grid=15, dt=0.05, dx=1.0, advection_velocity=1.0, diffusion_coeff=0.02
    )
    assert model.state_dim == 15
    assert model.meas_dim == 8  # observed every 2nd cell

    x0 = np.sin(2 * np.pi * np.arange(15) / 15.0)
    states, z = simulate(model, x0, steps=40, rng=np.random.default_rng(10))

    kf = KalmanFilter(model, x0=np.zeros(15), P0=np.eye(15))
    res = kf.run(z)
    assert res.filtered_means.shape == (40, 15)
    err = np.mean((res.filtered_means[-1] - states[-1]) ** 2)
    assert err < 0.2


def test_harmonic_oscillator_signal_denoising() -> None:
    r"""Checks harmonic oscillator state-space model for tracking a sinusoidal signal."""
    freq = 5.0  # 5 Hz
    dt = 0.01  # 100 Hz sampling
    model = HarmonicSignalModel(freq_hz=freq, dt=dt, q_std=1e-4, r_std=0.3)

    t = np.arange(200) * dt
    clean_signal = np.cos(2 * np.pi * freq * t)
    noise = np.random.default_rng(99).normal(0.0, 0.3, size=200)
    noisy_signal = (clean_signal + noise)[:, None]

    kf = KalmanFilter(model, x0=np.array([1.0, 0.0]), P0=np.eye(2))
    res = kf.run(noisy_signal)

    # Filtered signal (component 0) should be much closer to clean signal than noisy signal
    raw_rmse = np.sqrt(np.mean((noisy_signal[:, 0] - clean_signal) ** 2))
    filt_rmse = np.sqrt(np.mean((res.filtered_means[:, 0] - clean_signal) ** 2))
    assert filt_rmse < raw_rmse
