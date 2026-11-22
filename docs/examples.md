# Examples

Run from the repository root (plots are saved to `examples/output/`):

```
python examples/linear_multichannel.py
python examples/nonlinear_multichannel.py
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
