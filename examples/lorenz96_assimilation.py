r"""Module: examples.lorenz96_assimilation

Description: Atmospheric data assimilation benchmark on the 40-variable Lorenz-96 system.
    Demonstrates:
    - Highly nonlinear, chaotic spatial multi-scale dynamics.
    - Filter divergence in standard EnKF when ensemble size is small (N << D).
    - Drastic stabilization and error reduction using Gaspari-Cohn spatial covariance localization.
    - Evaluation of assimilation accuracy across all latitude stations.

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
    EnsembleKalmanFilter,
    Lorenz96Model,
    rmse,
    simulate,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    r"""Runs the 40-variable Lorenz-96 assimilation experiment with and without localization."""
    rng = np.random.default_rng(101)
    dim = 40
    dt = 0.05
    steps = 150
    forcing = 8.0

    # Observing every 2nd grid point (20 observation channels)
    obs_indices = list(range(0, dim, 2))
    model = Lorenz96Model(
        dim=dim,
        dt=dt,
        forcing=forcing,
        obs_indices=obs_indices,
        q_std=0.05,
        r_std=1.0,
    )

    # Spin up true trajectory onto the chaotic attractor
    x_spinup = np.full(dim, forcing)
    x_spinup[19] += 0.01  # Perturb middle station
    for _ in range(400):
        x_spinup = model.propagate(x_spinup)

    # Generate ground truth and noisy network measurements
    states, z = simulate(model, x0=x_spinup, steps=steps, rng=rng)
    truth = states[1:]

    # Small ensemble size: N = 20 << 40 (severe rank deficiency without localization)
    n_members = 20
    P0 = np.eye(dim)
    x0 = x_spinup + rng.normal(0.0, 1.0, size=dim)

    # 1. Standard EnKF (without localization)
    logger.info("Running Standard EnKF (N=%d, D=%d)...", n_members, dim)
    enkf_standard = EnsembleKalmanFilter(
        model,
        x0=x0,
        P0=P0,
        n_members=n_members,
        rng=np.random.default_rng(202),
        inflation=1.05,
    )
    res_standard = enkf_standard.run(z)

    # 2. Localized EnKF (using Gaspari-Cohn spatial tapering with cutoff = 4 grid points)
    logger.info("Running Localized EnKF with Gaspari-Cohn tapering...")
    loc_matrix = model.compute_localization_matrix(cutoff_radius=4.0)
    enkf_localized = EnsembleKalmanFilter(
        model,
        x0=x0,
        P0=P0,
        n_members=n_members,
        rng=np.random.default_rng(202),
        inflation=1.05,
        localization_matrix=loc_matrix,
    )
    res_localized = enkf_localized.run(z)

    # Evaluation
    burn_in = 30
    eval_truth = truth[burn_in:]
    eval_std = res_standard.filtered_means[burn_in:]
    eval_loc = res_localized.filtered_means[burn_in:]

    rmse_std = np.mean(rmse(eval_std, eval_truth))
    rmse_loc = np.mean(rmse(eval_loc, eval_truth))

    logger.info("Standard EnKF mean RMSE: %.3f", rmse_std)
    logger.info("Localized EnKF mean RMSE: %.3f", rmse_loc)

    # Visualization
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Spatio-temporal state trajectory (Hovmoller diagram of truth)
    im0 = axes[0].imshow(
        truth.T,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[0, steps * dt, 0, dim],
    )
    axes[0].set(
        xlabel="Atmospheric Time [dimensionless]",
        ylabel="Grid Station Index (0 to 39)",
        title="Lorenz-96 True Atmospheric State Evolution",
    )
    fig.colorbar(im0, ax=axes[0], label="State Value")

    # Time series of RMSE comparison
    time_series = np.arange(1, steps + 1) * dt
    err_std_t = np.sqrt(np.mean((res_standard.filtered_means - truth) ** 2, axis=1))
    err_loc_t = np.sqrt(np.mean((res_localized.filtered_means - truth) ** 2, axis=1))

    axes[1].plot(
        time_series, err_std_t, "r--", lw=1.5, label=f"Standard EnKF (RMSE: {rmse_std:.2f})"
    )

    axes[1].plot(
        time_series,
        err_loc_t,
        "g-",
        lw=2.0,
        label=f"Localized EnKF (Gaspari-Cohn, RMSE: {rmse_loc:.2f})",
    )
    axes[1].axvline(burn_in * dt, color="gray", linestyle=":", label="Burn-in Period")
    axes[1].set(
        xlabel="Time [dimensionless]",
        ylabel="Spatial RMSE",
        title="Localization Prevents Filter Divergence (N=20, D=40)",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lorenz96_assimilation.png", dpi=120)
    logger.info("Saved %s", OUTPUT_DIR / "lorenz96_assimilation.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
