# kalman_filter

A small, modular Python library for the Kalman filter (KF) and extended Kalman filter
(EKF) with first-class support for multi-channel measurements, including missing channels.

- [docs/theory.md](docs/theory.md) – equations and background
- [docs/implementation.md](docs/implementation.md) – architecture, conventions, usage
- [docs/examples.md](docs/examples.md) – linear and nonlinear multi-channel examples

## Install (env_api)

```
C:\Users\ZhangHui\.conda\envs\env_api\python.exe -m pip install -e ".[examples,dev]"
```

## Quick start

```python
import numpy as np
from kalman_filter import KalmanFilter, constant_velocity_model, simulate

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
model = constant_velocity_model(dim=2, dt=0.1, accel_std=1.0, H=H, R=np.eye(2))
states, z = simulate(model, np.array([0, 0, 5.0, 2.0]), 200, np.random.default_rng(0))
result = KalmanFilter(model, np.zeros(4), 10 * np.eye(4)).run(z)
```

Use `NaN` in a measurement row to mark a channel as missing at that step.
