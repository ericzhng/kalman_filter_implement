r"""Module: examples.lorenz63_benchmark

Description: Classic Lorenz-63 chaotic-system benchmark for data assimilation. A Monte-Carlo
    study over independent runs compares the EKF with the EnKF (several ensemble sizes) on
    RMSE (against the raw-measurement baseline) and NEES consistency. Observed channels are
    ``x`` and ``z`` (10% dropout), so ``y`` must be inferred through the dynamics.
    ``KalmanFilter`` needs a linear model and is not applicable. See docs/examples.md.

Author: Hui Zhang
Date: 2026-11-22
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
    simulate,
)
from kalman_filter.types import FloatArray  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.05
N_RUNS = 20
STEPS = 300
BURN_IN = 50
OBS_STD = 1.0
ENSEMBLE_SIZES = (10, 20, 50, 200)


def lorenz_rhs(x: FloatArray) -> FloatArray:
    r"""Computes the Lorenz-63 vector field.

    Args:
        x: State ``[x, y, z]``, shape ``(3,)``.

    Returns:
        The time derivative, shape ``(3,)``.
    """
    return np.array([SIGMA * (x[1] - x[0]), x[0] * (RHO - x[2]) - x[1], x[0] * x[1] - BETA * x[2]])


def lorenz_step(x: FloatArray, u: FloatArray | None) -> FloatArray:
    r"""Advances the state by ``DT`` with one classical Runge-Kutta (RK4) step.

    Args:
        x: State ``[x, y, z]``, shape ``(3,)``.
        u: Control input (unused).

    Returns:
        The next state, shape ``(3,)``.
    """
    k1 = lorenz_rhs(x)
    k2 = lorenz_rhs(x + 0.5 * DT * k1)
    k3 = lorenz_rhs(x + 0.5 * DT * k2)
    k4 = lorenz_rhs(x + DT * k3)
    return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def observe_xz(x: FloatArray) -> FloatArray:
    r"""Observes the ``x`` and ``z`` components only.

    Args:
        x: State ``[x, y, z]``, shape ``(3,)``.

    Returns:
        The noise-free measurement ``[x, z]``, shape ``(2,)``.
    """
    return x[[0, 2]]


def run_once(model: NonlinearModel, seed: int) -> tuple[FloatArray, dict[str, object]]:
    r"""Simulates one trajectory and filters it with every filter.

    Args:
        model: The Lorenz-63 model.
        seed: Seed for this Monte-Carlo run.

    Returns:
        ``(truth, results)`` with ``truth`` of shape ``(STEPS, 3)`` and ``results`` mapping a
        filter name to its ``FilterResult``.
    """
    rng = np.random.default_rng(seed)
    spin_up = np.array([1.0, 1.0, 1.0])
    for _ in range(500):  # move onto the attractor
        spin_up = lorenz_step(spin_up, None)
    states, z = simulate(model, spin_up, STEPS, rng, dropout=np.array([0.1, 0.1]))
    x_init = states[0] + rng.normal(0.0, 2.0, 3)  # wrong initial guess
    P0 = 4.0 * np.eye(3)
    results: dict[str, object] = {"EKF": ExtendedKalmanFilter(model, x_init, P0).run(z)}
    for n in ENSEMBLE_SIZES:
        enkf = EnsembleKalmanFilter(
            model, x_init, P0, n_members=n, rng=np.random.default_rng(seed + 1000), inflation=1.05
        )
        results[f"EnKF N={n}"] = enkf.run(z)
    results["_z"] = z
    return states[1:], results


def main() -> None:
    r"""Runs the Monte-Carlo benchmark, logs a summary table and plots one trajectory."""
    model = NonlinearModel(
        f=lorenz_step,
        h=observe_xz,
        Q=np.diag([0.05**2] * 3),
        R=np.diag([OBS_STD**2] * 2),
    )
    sq_err: dict[str, list[FloatArray]] = {}
    nees_vals: dict[str, list[float]] = {}
    first: tuple[FloatArray, dict[str, object]] | None = None
    for seed in range(N_RUNS):
        truth, results = run_once(model, seed)
        first = first or (truth, results)
        for name, res in results.items():
            if name == "_z":
                continue
            err = res.filtered_means[BURN_IN:] - truth[BURN_IN:]
            sq_err.setdefault(name, []).append((err**2).mean(axis=0))
            nees_vals.setdefault(name, []).append(
                float(
                    nees(
                        res.filtered_means[BURN_IN:], res.filtered_covs[BURN_IN:], truth[BURN_IN:]
                    ).mean()
                )
            )

    logger.info(
        "Lorenz-63, %d runs x %d steps (burn-in %d), obs std %.1f on x, z",
        N_RUNS,
        STEPS,
        BURN_IN,
        OBS_STD,
    )
    logger.info(
        "%-12s %8s %8s %8s %8s %12s",
        "filter",
        "rmse_x",
        "rmse_y",
        "rmse_z",
        "rmse_all",
        "NEES(exp 3)",
    )
    summary = {}
    for name, errs in sq_err.items():
        comp = np.sqrt(np.mean(errs, axis=0))
        summary[name] = (float(np.sqrt(np.mean(errs))), float(np.mean(nees_vals[name])))
        logger.info(
            "%-12s %8.3f %8.3f %8.3f %8.3f %12.2f",
            name,
            *comp,
            *summary[name],
        )

    assert first is not None
    truth, results = first
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.arange(1, STEPS + 1) * DT
    axes[0].plot(t, truth[:, 1], "k-", label="true y (unobserved)")
    axes[0].plot(t, results["EKF"].filtered_means[:, 1], "r--", label="EKF")
    axes[0].plot(t, results["EnKF N=50"].filtered_means[:, 1], "g-", label="EnKF N=50")
    axes[0].set(xlabel="time", ylabel="y", title="Unobserved component (run 0)")
    axes[0].legend()
    names = list(summary)
    axes[1].bar(names, [summary[n][0] for n in names], color="g")
    axes[1].axhline(OBS_STD, color="k", ls="--", label="measurement noise std")
    axes[1].set(ylabel="RMSE (all components)", title=f"Monte Carlo ({N_RUNS} runs)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lorenz63_benchmark.png", dpi=120)
    logger.info("Saved %s", OUTPUT_DIR / "lorenz63_benchmark.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
