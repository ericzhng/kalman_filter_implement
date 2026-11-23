r"""Module: examples.sensor_fusion_gps_imu

Description: Multi-rate asynchronous sensor fusion combining high-rate odometry and
    low-rate GPS with satellite outages. Demonstrates:
    - Merging asynchronous sensor streams using :func:`merge_sensor_streams` with NaNs.
    - Forward state estimation with :class:`UnscentedKalmanFilter`.
    - Backward trajectory reconstruction during GPS outages using :func:`rts_smooth`.
    - Performance evaluation via RMSE and Chi-square consistency testing.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kalman_filter import (  # noqa: E402
    SensorStream,
    UnscentedKalmanFilter,
    chi2_consistency_test,
    constant_velocity_model,
    merge_sensor_streams,
    rmse,
    rts_smooth,
    simulate,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    r"""Runs the multi-sensor GPS + Odometry fusion and smoothing demo."""
    rng = np.random.default_rng(42)
    dt = 0.1
    t_end = 30.0
    steps = int(t_end / dt)

    # 2D Constant Velocity model: state = [x, y, vx, vy]
    # Channels: GPS (x, y) + Odometer (vx, vy)
    pos_h = np.hstack([np.eye(2), np.zeros((2, 2))])
    vel_h = np.hstack([np.zeros((2, 2)), np.eye(2)])
    H = np.vstack([pos_h, vel_h])
    R = np.diag([2.0**2, 2.0**2, 0.3**2, 0.3**2])

    model = constant_velocity_model(dim=2, dt=dt, accel_std=0.8, H=H, R=R)
    x0 = np.array([0.0, 0.0, 8.0, 4.0])

    # True trajectory
    truth_states, _ = simulate(model, x0, steps, rng)
    truth = truth_states[1:]
    times = np.arange(1, steps + 1) * dt

    # Create asynchronous sensor streams:
    # 1. GPS at 2 Hz (every 0.5s) with a tunnel outage between t = 10s and t = 18s
    gps_times = []
    gps_vals = []
    for t in np.arange(0.5, t_end + 1e-6, 0.5):
        if not (10.0 <= t <= 18.0):  # Tunnel outage
            idx = int(round(t / dt)) - 1
            if 0 <= idx < steps:
                gps_noise = rng.normal(0.0, 2.0, size=2)
                gps_times.append(t)
                gps_vals.append(truth[idx, :2] + gps_noise)

    gps_stream = SensorStream(
        name="GPS",
        timestamps=np.array(gps_times),
        values=np.array(gps_vals),
        noise_std=np.array([2.0, 2.0]),
    )

    # 2. Wheel Odometry at 10 Hz (every 0.1s, continuous)
    odom_noise = rng.normal(0.0, 0.3, size=(steps, 2))
    odom_stream = SensorStream(
        name="Odometry",
        timestamps=times,
        values=truth[:, 2:] + odom_noise,
        noise_std=np.array([0.3, 0.3]),
    )

    # Synchronize streams onto uniform grid with NaNs during outages
    t_grid, z_fused, _ = merge_sensor_streams([gps_stream, odom_stream], dt=dt, t_end=t_end)

    # Forward filtering with UKF
    P0 = np.diag([5.0, 5.0, 2.0, 2.0])
    ukf = UnscentedKalmanFilter(model, x0=np.zeros(4), P0=P0)
    ukf_result = ukf.run(z_fused)

    # Backward fixed-interval smoothing with RTS
    smooth_result = rts_smooth(ukf_result, model)

    # Diagnostics
    filt_rmse = rmse(ukf_result.filtered_means[:, :2], truth[:, :2])
    smooth_rmse = rmse(smooth_result.smoothed_means[:, :2], truth[:, :2])
    passed_chi2, mean_nis, (low, high) = chi2_consistency_test(ukf_result.nis, dof=2)

    logger.info("UKF Filter Position RMSE: %s m", np.round(filt_rmse, 3))
    logger.info("RTS Smoothed Position RMSE: %s m", np.round(smooth_rmse, 3))
    logger.info(
        "NIS consistency test passed: %s (mean: %.2f, 95%% CI: [%.2f, %.2f])",
        passed_chi2,
        mean_nis,
        low,
        high,
    )

    # Visualization
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: 2D Trajectory
    axes[0].plot(truth[:, 0], truth[:, 1], "k-", lw=1.5, label="Ground Truth")
    axes[0].plot(
        gps_stream.values[:, 0],
        gps_stream.values[:, 1],
        "b.",
        ms=4,
        alpha=0.6,
        label="GPS Fixes",
    )

    axes[0].plot(
        ukf_result.filtered_means[:, 0],
        ukf_result.filtered_means[:, 1],
        "r--",
        label="UKF Forward",
    )
    axes[0].plot(
        smooth_result.smoothed_means[:, 0],
        smooth_result.smoothed_means[:, 1],
        "g-",
        lw=2,
        label="RTS Smoothed",
    )
    axes[0].axvspan(
        truth[int(10.0 / dt), 0],
        truth[int(18.0 / dt), 0],
        color="gray",
        alpha=0.15,
        label="GPS Outage",
    )
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Trajectory with GPS Outage")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Position Error & Uncertainty over time
    err_filt = np.hypot(
        ukf_result.filtered_means[:, 0] - truth[:, 0],
        ukf_result.filtered_means[:, 1] - truth[:, 1],
    )
    err_smooth = np.hypot(
        smooth_result.smoothed_means[:, 0] - truth[:, 0],
        smooth_result.smoothed_means[:, 1] - truth[:, 1],
    )
    axes[1].plot(times, err_filt, "r--", label="UKF Position Error")
    axes[1].plot(times, err_smooth, "g-", lw=2, label="RTS Smoothed Error")
    axes[1].axvspan(10.0, 18.0, color="gray", alpha=0.2, label="GPS Tunnel Outage")
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Position Error [m]",
        title="Position Error Tracking (Outage Recovery)",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensor_fusion_gps_imu.png", dpi=120)
    logger.info("Saved %s", OUTPUT_DIR / "sensor_fusion_gps_imu.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
