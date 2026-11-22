# Examples

Run from the repository root (plots are saved to `examples/output/`):

```
python examples/linear_multichannel.py
python examples/nonlinear_multichannel.py
python examples/lorenz63_benchmark.py   # ~40 s
```

Requires `matplotlib` (`pip install -e ".[examples]"`).

## Linear multi-channel: sensor fusion with dropouts

[`examples/linear_multichannel.py`](../examples/linear_multichannel.py)

A 2D target moves with near-constant velocity, state `[x, y, vx, vy]`. Three sensors give
six channels:

| Sensor | Channels | Noise std | Dropout |
|---|---|---|---|
| A | x, y | 1.0 | 10 % |
| B | x, y | 3.0 | 0 % |
| C | vx, vy | 0.5 | 40 % |

`H` stacks the sensor rows and `R` is diagonal. Missing readings are `NaN` and are
skipped automatically. Typical result (seed 0): filter position RMSE ≈ 0.22–0.28
versus ≈ 0.95 for raw sensor A, and mean NEES ≈ 4.1 (expected 4 for a consistent filter).

## Nonlinear multi-channel: EKF with range-bearing stations

[`examples/nonlinear_multichannel.py`](../examples/nonlinear_multichannel.py)

A target performs a coordinated turn with **unknown turn rate**; state
`[x, y, vx, vy, ω]`. Two fixed stations each report range and bearing (4 channels,
10 % dropout):

```
range_i   = ‖p − s_i‖
bearing_i = atan2(p_y − s_{i,y}, p_x − s_{i,x})
```

Both the transition and measurement functions are nonlinear. The example shows:

- an analytic measurement Jacobian, and a numerical dynamics Jacobian (the default);
- wrapped bearing innovations via `residual_fn`;
- estimating an unmeasured parameter (`ω`) by augmenting the state.

Typical result (seed 1): position RMSE ≈ 0.3–0.4, `ω` converging towards the true 0.1,
mean NEES ≈ 5 (expected 5).

## Lorenz-63 benchmark: Monte-Carlo performance check

[`examples/lorenz63_benchmark.py`](../examples/lorenz63_benchmark.py)

The classic chaotic data-assimilation test (σ=10, ρ=28, β=8/3, RK4 step 0.05). Only `x`
and `z` are measured (std 1, 10 % dropout); `y` must be inferred through the dynamics.
Each of 20 independent runs (300 steps, 50 burn-in, wrong initial guess) is filtered by the
EKF and by the EnKF with N = 10, 20, 50, 200. `KalmanFilter` does not apply (nonlinear).

| filter | RMSE x / y / z | RMSE all | NEES (expect 3) |
|---|---|---|---|
| EKF | 0.377 / 0.619 / 0.484 | 0.503 | 9.14 |
| EnKF N=10 | 0.385 / 0.645 / 0.479 | 0.514 | 9.37 |
| EnKF N=20 | 0.322 / 0.537 / 0.391 | 0.426 | 3.54 |
| EnKF N=50 | 0.322 / 0.538 / 0.383 | 0.424 | 2.79 |
| EnKF N=200 | 0.319 / 0.533 / 0.375 | 0.419 | 2.43 |

Pass criteria: every filter beats the raw measurement noise (RMSE < 1), EnKF error
shrinks and then plateaus as N grows, and the EnKF is better calibrated than the EKF
(whose linearisation makes it over-confident on this chaotic system).

## Filters used

- **Linear example:** `KalmanFilter`, `ExtendedKalmanFilter` and `EnsembleKalmanFilter`
  (200 members). KF and EKF are identical on a linear model; the EnKF agrees up to
  sampling noise (RMSE within ~0.01, NEES ≈ 4.1–4.2).
- **Nonlinear example:** `ExtendedKalmanFilter` and `EnsembleKalmanFilter` (200 members,
  inflation 1.02). `KalmanFilter` requires a `LinearModel`, so it cannot run here.
  Position RMSE ≈ 0.31–0.37 for both; NEES 4.98 (EKF) and 4.55 (EnKF).
