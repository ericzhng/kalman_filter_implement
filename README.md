# kalman_filter

A powerful, modular Python package for Kalman filtering, multi-sensor fusion, and geophysical data assimilation, featuring first-class support for multi-channel measurements with missing data (NaN dropouts).

[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen)](tests/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-black)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

- [docs/theory.md](docs/theory.md) – mathematical formulation, estimator derivations, and consistency diagnostics
- [docs/implementation.md](docs/implementation.md) – architecture, protocols, extensible designs, and conventions
- [docs/examples.md](docs/examples.md) – sensor fusion, target tracking, chaotic data assimilation, and benchmarking

---

## Key Capabilities

- **Optimal Linear & Nonlinear Estimators**:
  - `KalmanFilter` (KF): Standard minimum-variance optimal filter for linear-Gaussian systems.
  - `ExtendedKalmanFilter` (EKF): First-order Taylor linearization with analytic or central-difference Jacobians.
  - `IteratedExtendedKalmanFilter` (IEKF): Local Gauss-Newton optimization for severe measurement non-linearities.
  - `UnscentedKalmanFilter` (UKF): Scaled Unscented Transform capturing 3rd-order Gaussian statistics without Jacobians.
  - `EnsembleKalmanFilter` (EnKF): Stochastic Monte-Carlo ensemble filter with multiplicative covariance inflation and **Gaspari-Cohn spatial covariance localization**.
  - `rts_smooth`: Rauch-Tung-Striebel optimal fixed-interval backward smoother for both linear and nonlinear models.

- **Multi-Sensor Fusion & Dropouts**:
  - Seamless handling of asynchronous, multi-rate sensors.
  - Channels marked as `NaN` are automatically skipped in the update step without matrix restructuring.
  - `merge_sensor_streams` aggregates asynchronous timestamped `SensorStream` objects onto a synchronized grid with NaN padding.
  - Modular innovation residual handling with `wrap_angle` for heading/bearing channels.

- **Pre-Built Application Models**:
  - **Target Tracking & Kinematics**:
    - `constant_velocity_model`: Nearly Constant Velocity (CV) with discrete acceleration noise.
    - `constant_acceleration_model`: Constant Acceleration (CA) motion with continuous white-noise jerk.
    - `coordinated_turn_model`: 2D maneuvering target with unknown turn rate and range/bearing stations.
    - `random_walk_model`: Online parameter estimation and sensor bias drift.
  - **Geophysical Data Assimilation**:
    - `Lorenz96Model`: 40-variable atmospheric benchmark on periodic latitude circle with Gaspari-Cohn localization matrices.
    - `Lorenz63Model`: 3D chaotic convective model with Runge-Kutta 4 (RK4) integration.

- **Statistical Consistency & Diagnostics**:
  - `rmse` and `mae`: Accuracy evaluation over state trajectories.
  - `nees`: Normalized Estimation Error Squared for filter consistency.
  - `chi2_consistency_test`: Formal two-sided Chi-square hypothesis testing for NEES and NIS.
  - `whiteness_test` & `autocorrelation`: Verify that measurement innovations form an uncorrelated white noise sequence.

---

## Installation

Install in development / editable mode:

```bash
pip install -e ".[examples,dev]"
```

Run test suite and linter:

```bash
pytest
ruff check .
```

---

## Quick Start Examples

### 1. Multi-Sensor Tracking with Dropouts

```python
import numpy as np
from kalman_filter import KalmanFilter, constant_velocity_model, simulate

# 2D tracking with position sensors: A (clean), B (noisy)
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]])
R = np.diag([1.0, 1.0, 4.0, 4.0])
model = constant_velocity_model(dim=2, dt=0.1, accel_std=0.5, H=H, R=R)

# Simulate 200 steps with 10% dropout on sensor A
states, z = simulate(model, np.array([0, 0, 5.0, 2.0]), steps=200, rng=np.random.default_rng(0),
                     dropout=np.array([0.1, 0.1, 0.0, 0.0]))

# Run Kalman Filter
result = KalmanFilter(model, x0=np.zeros(4), P0=10.0 * np.eye(4)).run(z)
print("Filtered state at end:", result.filtered_means[-1])
```

### 2. Unscented Kalman Filter & RTS Smoothing

```python
from kalman_filter import UnscentedKalmanFilter, rts_smooth

# Propagate forward with UKF
ukf = UnscentedKalmanFilter(model, x0=np.zeros(4), P0=10.0 * np.eye(4))
ukf_result = ukf.run(z)

# Backward fixed-interval smoothing
smoothed = rts_smooth(ukf_result, model)
print("Smoothed final state:", smoothed.smoothed_means[-1])
```

### 3. High-Dimensional Data Assimilation (Lorenz-96 with Localization)

```python
from kalman_filter import EnsembleKalmanFilter, Lorenz96Model, simulate

# 40-variable chaotic atmospheric system
lorenz = Lorenz96Model(dim=40, dt=0.05, forcing=8.0, obs_indices=list(range(0, 40, 2)))
states, z = simulate(lorenz, x0=np.full(40, 8.0), steps=100, rng=np.random.default_rng(42))

# Compute Gaspari-Cohn spatial tapering matrix to eliminate spurious correlations
loc_matrix = lorenz.compute_localization_matrix(cutoff_radius=4.0)

# Run Localized EnKF with small ensemble N = 20 << 40
enkf = EnsembleKalmanFilter(lorenz, x0=np.full(40, 8.0), P0=np.eye(40), n_members=20,
                            localization_matrix=loc_matrix)
enkf_result = enkf.run(z)
```

---

## Runnable Example Scripts

- `examples/sensor_fusion_gps_imu.py`: GPS + Wheel Odometry fusion with tunnel outages, UKF tracking, and RTS smoothing.
- `examples/lorenz96_assimilation.py`: 40-variable Lorenz-96 spatial atmospheric data assimilation demonstrating how Gaspari-Cohn localization prevents filter divergence.
- `examples/linear_multichannel.py`: Multi-sensor fusion with asynchronous dropouts (KF vs EKF vs EnKF).
- `examples/nonlinear_multichannel.py`: Coordinated-turn vehicle tracking with range-bearing stations and bearing residual wrapping.
- `examples/lorenz63_benchmark.py`: Monte-Carlo benchmark comparing EKF vs EnKF on the 3D Lorenz attractor.
