# Implementation

## Layout

```
src/kalman_filter/
├── types.py          GaussianState, FilterResult (data containers)
├── jacobian.py       numerical_jacobian (central differences)
├── models/
│   ├── base.py       StateSpaceModel protocol
│   ├── linear.py     LinearModel, constant_velocity_model
│   └── nonlinear.py  NonlinearModel (f, h, optional Jacobians / residual)
├── filters/
│   ├── base.py       BaseFilter: predict / update / step / run
│   ├── kalman.py     KalmanFilter  (uses constant F, H)
│   ├── extended.py   ExtendedKalmanFilter (Jacobians at the current mean)
│   └── ensemble.py   EnsembleKalmanFilter (perturbed-observation EnKF, Jacobian-free)
├── simulation.py     simulate(): ground truth + noisy, possibly missing, measurements
└── metrics.py        rmse, nees
examples/             runnable linear and nonlinear multi-channel demos
tests/                pytest suite
```

## Design

**Models and filters are separate.** A model describes the world
(`propagate`, `measure`, `Q`, `R`, Jacobians, `residual`); a filter only consumes the
`StateSpaceModel` protocol. Any object with those members works, so you can add your
own model class without touching the filters.

**One update code path.** `BaseFilter` implements predict and the Joseph-form update.
Subclasses implement just two hooks, `_transition_matrix` and `_measurement_matrix`:

| Filter | `F` | `H` |
|---|---|---|
| `KalmanFilter` | constant `model.F` | constant `model.H` |
| `ExtendedKalmanFilter` | `model.state_jacobian(x, u)` | `model.measurement_jacobian(x)` |

Because `LinearModel` also implements the Jacobian methods, the EKF reduces exactly to the
KF on linear models (verified in `tests/test_filters.py`).

**Multi-channel data.** `z` has one entry per channel. A `NaN` entry means that channel
is missing at that step; `update` keeps only the observed rows of `H`, `R` and `z`.
`filter.run` returns `innovations` with NaN in those slots.

**Nonlinear models.** `NonlinearModel(f, h, Q, R, f_jac=None, h_jac=None, residual_fn=None)`.
Omitted Jacobians fall back to `numerical_jacobian`; supply analytic ones for speed
and accuracy. `f` has the signature `f(x, u)` where `u` may be `None`.

## Conventions

- Time 0 is the prior `(x0, P0)`. Row `k` of the measurements and of every
  `FilterResult` array corresponds to time `k + 1`. `simulate` returns
  `states` with `T + 1` rows (row 0 is `x0`), so compare `result.filtered_means`
  against `states[1:]`.
- Covariances are symmetrised after every step.
- Process noise is additive: `x_k = f(x_{k-1}, u) + w`.

## Usage

```python
import numpy as np
from kalman_filter import KalmanFilter, constant_velocity_model, simulate

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])  # measure x, y
model = constant_velocity_model(dim=2, dt=0.1, accel_std=1.0, H=H, R=np.eye(2))
states, z = simulate(model, np.array([0, 0, 5.0, 2.0]), steps=200, rng=np.random.default_rng(0))
result = KalmanFilter(model, np.zeros(4), 10 * np.eye(4)).run(z)
result.filtered_means  # (200, 4)
```

Step-by-step use (e.g. streaming data): `kf.predict(u)`, then `kf.update(z)`, or
`kf.step(z, u)`; read the belief from `kf.state`.

For nonlinear problems:

```python
from kalman_filter import ExtendedKalmanFilter, NonlinearModel

model = NonlinearModel(f=my_f, h=my_h, Q=Q, R=R, h_jac=my_h_jac, residual_fn=wrap)
ekf = ExtendedKalmanFilter(model, x0, P0)
```

## Extending

- **New filter** (e.g. iterated EKF): subclass `BaseFilter`, override the two hooks or
  `update`.
- **New model**: implement the `StateSpaceModel` protocol.
- **Smoothing, UKF**: add modules under `filters/`, export them in
  `filters/__init__.py` and `kalman_filter/__init__.py`.

## Development

Use the `env_api` conda environment:

```
C:\Users\ZhangHui\.conda\envs\env_api\python.exe -m pip install -e .
C:\Users\ZhangHui\.conda\envs\env_api\python.exe -m pytest
C:\Users\ZhangHui\.conda\envs\env_api\python.exe -m ruff check .
```
