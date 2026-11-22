r"""Module: examples.nonlinear_multichannel

Description: EKF and EnKF tracking a coordinated-turn target with two range-bearing stations (4
    channels). State is ``[x, y, vx, vy, omega]``; bearing residuals are wrapped.
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
    NonlinearModel,
    nees,
    rmse,
    simulate,
)
from kalman_filter.types import FloatArray  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DT = 0.5
STATIONS = np.array([[-60.0, -20.0], [100.0, -20.0]])


def turn_transition(x: FloatArray, u: FloatArray | None) -> FloatArray:
    r"""Applies a coordinated-turn transition with turn rate ``omega`` in ``x[4]``.

    Args:
        x: State ``[x, y, vx, vy, omega]``, shape ``(5,)``.
        u: Control input (unused).

    Returns:
        The next state, shape ``(5,)``.
    """
    px, py, vx, vy, w = x
    wt = w * DT
    s = DT * np.sinc(wt / np.pi)  # sin(w dt) / w, safe at w = 0
    c = 0.5 * w * DT**2 * np.sinc(wt / (2 * np.pi)) ** 2  # (1 - cos(w dt)) / w
    cw, sw = np.cos(wt), np.sin(wt)
    return np.array(
        [px + s * vx - c * vy, py + c * vx + s * vy, cw * vx - sw * vy, sw * vx + cw * vy, w]
    )


def range_bearing(x: FloatArray) -> FloatArray:
    r"""Computes stacked ``[range_i, bearing_i]`` from every station.

    Args:
        x: State ``[x, y, vx, vy, omega]``, shape ``(5,)``.

    Returns:
        The noise-free measurement, shape ``(4,)``.
    """
    d = x[None, :2] - STATIONS
    return np.column_stack([np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])]).ravel()


def range_bearing_jacobian(x: FloatArray) -> FloatArray:
    r"""Computes the analytic Jacobian of :func:`range_bearing`.

    Args:
        x: State ``[x, y, vx, vy, omega]``, shape ``(5,)``.

    Returns:
        The Jacobian, shape ``(4, 5)``.
    """
    jac = np.zeros((2 * len(STATIONS), 5))
    for i, (dx, dy) in enumerate(x[None, :2] - STATIONS):
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        jac[2 * i, :2] = [dx / r, dy / r]
        jac[2 * i + 1, :2] = [-dy / r2, dx / r2]
    return jac


def wrap_bearing(z: FloatArray, z_pred: FloatArray) -> FloatArray:
    r"""Computes the innovation with bearing channels (odd indices) wrapped to ``[-pi, pi)``.

    Args:
        z: Measurement, shape ``(4,)``.
        z_pred: Predicted measurement, shape ``(4,)``.

    Returns:
        The wrapped innovation, shape ``(4,)``.
    """
    nu = z - z_pred
    nu[1::2] = (nu[1::2] + np.pi) % (2 * np.pi) - np.pi
    return nu


def main() -> None:
    r"""Simulates, filters with an EKF and plots the nonlinear tracking problem."""
    rng = np.random.default_rng(1)
    steps = 200

    G = np.zeros((5, 3))
    G[0, 0] = G[1, 1] = 0.5 * DT**2
    G[2, 0] = G[3, 1] = G[4, 2] = DT
    Q = G @ np.diag([0.05**2, 0.05**2, 0.002**2]) @ G.T
    R = np.diag([1.0**2, 0.01**2, 1.0**2, 0.01**2])
    model = NonlinearModel(
        f=turn_transition,
        h=range_bearing,
        Q=Q,
        R=R,
        h_jac=range_bearing_jacobian,
        residual_fn=wrap_bearing,
    )

    x0 = np.array([20.0, 10.0, 5.0, 0.0, 0.1])
    states, z = simulate(model, x0, steps, rng, dropout=np.array([0.1, 0.1, 0.1, 0.1]))

    x_init = np.array([25.0, 5.0, 4.0, 1.0, 0.0])
    P0 = np.diag([25.0, 25.0, 4.0, 4.0, 0.05**2])
    # KalmanFilter needs a LinearModel, so it cannot run on this nonlinear problem.
    result = ExtendedKalmanFilter(model, x_init, P0).run(z)
    enkf = EnsembleKalmanFilter(
        model, x_init, P0, n_members=200, rng=np.random.default_rng(2), inflation=1.02
    ).run(z)

    truth = states[1:]
    for name, res in (("EKF", result), ("EnKF", enkf)):
        logger.info(
            "%-4s RMSE [x, y, vx, vy, w]: %s  final omega %.3f (true 0.100)  "
            "mean NEES (expect ~5): %.2f",
            name,
            np.round(rmse(res.filtered_means, truth), 3),
            res.filtered_means[-1, 4],
            nees(res.filtered_means, res.filtered_covs, truth).mean(),
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(truth[:, 0], truth[:, 1], "k-", label="truth")
    axes[0].plot(result.filtered_means[:, 0], result.filtered_means[:, 1], "r-", label="EKF")
    axes[0].plot(enkf.filtered_means[:, 0], enkf.filtered_means[:, 1], "g--", label="EnKF")
    axes[0].plot(*STATIONS.T, "b^", label="stations")
    axes[0].set(xlabel="x", ylabel="y", title="Trajectory", aspect="equal")
    axes[0].legend()
    t = np.arange(1, steps + 1) * DT
    axes[1].plot(t, np.full(steps, 0.1), "k-", label="true omega")
    axes[1].plot(t, result.filtered_means[:, 4], "r-", label="EKF omega")
    axes[1].plot(t, enkf.filtered_means[:, 4], "g--", label="EnKF omega")
    sigma = np.sqrt(result.filtered_covs[:, 4, 4])
    axes[1].fill_between(
        t,
        result.filtered_means[:, 4] - 2 * sigma,
        result.filtered_means[:, 4] + 2 * sigma,
        color="r",
        alpha=0.2,
        label="2 sigma",
    )
    axes[1].set(xlabel="time [s]", ylabel="rad/s", title="Turn-rate estimate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nonlinear_multichannel.png", dpi=120)
    logger.info("Saved %s", OUTPUT_DIR / "nonlinear_multichannel.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
