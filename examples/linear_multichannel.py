r"""Module: examples.linear_multichannel

Description: Linear KF, EKF and EnKF fusing three heterogeneous sensor channels (with dropouts)
    for a 2D target. Sensor A: position, 10% dropout; B: noisy position; C: velocity, 40% dropout.
    See docs/examples.md.

Author: Hui Zhang
Date: 2026-11-21
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kalman_filter import (  # noqa: E402
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    KalmanFilter,
    constant_velocity_model,
    nees,
    rmse,
    simulate,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    r"""Simulates, filters and plots the multi-channel linear tracking problem."""
    rng = np.random.default_rng(0)
    dt, steps = 0.1, 400

    # State [x, y, vx, vy]; 6 measurement channels = A(x,y) + B(x,y) + C(vx,vy).
    pos = np.hstack([np.eye(2), np.zeros((2, 2))])
    vel = np.hstack([np.zeros((2, 2)), np.eye(2)])
    H = np.vstack([pos, pos, vel])
    R = np.diag([1.0**2, 1.0**2, 3.0**2, 3.0**2, 0.5**2, 0.5**2])
    model = constant_velocity_model(dim=2, dt=dt, accel_std=1.0, H=H, R=R)

    x0 = np.array([0.0, 0.0, 5.0, 2.0])
    dropout = np.array([0.1, 0.1, 0.0, 0.0, 0.4, 0.4])
    states, z = simulate(model, x0, steps, rng, dropout=dropout)

    P0 = np.diag([10.0, 10.0, 10.0, 10.0])
    result = KalmanFilter(model, x0=np.zeros(4), P0=P0).run(z)
    ekf = ExtendedKalmanFilter(model, np.zeros(4), P0).run(z)
    enkf = EnsembleKalmanFilter(
        model, np.zeros(4), P0, n_members=200, rng=np.random.default_rng(10)
    ).run(z)

    truth = states[1:]
    raw_rmse = rmse(z[:, 0:2], truth[:, 0:2])  # sensor A taken at face value
    for name, res in (("KF", result), ("EKF", ekf), ("EnKF", enkf)):
        logger.info(
            "%-4s RMSE [x, y, vx, vy]: %s  mean NEES (expect ~4): %.2f",
            name,
            np.round(rmse(res.filtered_means, truth), 3),
            nees(res.filtered_means, res.filtered_covs, truth).mean(),
        )
    logger.info("Raw sensor-A position RMSE [x, y]: %s", np.round(raw_rmse, 3))

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(truth[:, 0], truth[:, 1], "k-", label="truth")
    axes[0].plot(z[:, 0], z[:, 1], ".", ms=3, alpha=0.4, label="sensor A")
    axes[0].plot(result.filtered_means[:, 0], result.filtered_means[:, 1], "r-", label="KF")
    axes[0].plot(ekf.filtered_means[:, 0], ekf.filtered_means[:, 1], "b--", label="EKF")
    axes[0].plot(enkf.filtered_means[:, 0], enkf.filtered_means[:, 1], "g:", label="EnKF")
    axes[0].set(xlabel="x", ylabel="y", title="Trajectory")
    axes[0].legend()
    t = np.arange(1, steps + 1) * dt
    axes[1].plot(t, truth[:, 2], "k-", label="true vx")
    axes[1].plot(t, result.filtered_means[:, 2], "r-", label="KF vx")
    axes[1].plot(t, ekf.filtered_means[:, 2], "b--", label="EKF vx")
    axes[1].plot(t, enkf.filtered_means[:, 2], "g:", label="EnKF vx")
    sigma = np.sqrt(result.filtered_covs[:, 2, 2])
    axes[1].fill_between(
        t,
        result.filtered_means[:, 2] - 2 * sigma,
        result.filtered_means[:, 2] + 2 * sigma,
        color="r",
        alpha=0.2,
        label="2 sigma",
    )
    axes[1].set(xlabel="time [s]", ylabel="vx", title="Velocity estimate (unmeasured by A/B)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "linear_multichannel.png", dpi=120)
    logger.info("Saved %s", OUTPUT_DIR / "linear_multichannel.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
