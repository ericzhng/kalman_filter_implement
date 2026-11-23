# Implementation

## Layout

```
src/kalman_filter/
├── types.py            GaussianState, FilterResult, SmootherResult (data containers)
├── localization.py     gaspari_cohn correlation tapering function
├── jacobian.py         numerical_jacobian (central differences)
├── models/
│   ├── base.py         StateSpaceModel protocol
│   ├── linear.py       LinearModel, constant_velocity_model
│   ├── kinematic.py    constant_acceleration_model, coordinated_turn_model, random_walk_model
│   ├── assimilation.py Lorenz96Model, Lorenz63Model (geophysical data assimilation)
│   ├── sensors.py      SensorStream, merge_sensor_streams, wrap_angle, bearing_residual
│   └── nonlinear.py    NonlinearModel (f, h, optional Jacobians, residual wrapping)
├── filters/
│   ├── base.py         BaseFilter: predict / update / step / run
│   ├── kalman.py       KalmanFilter (linear-Gaussian optimal filter)
│   ├── extended.py     ExtendedKalmanFilter (1st-order Jacobian linearization)
│   ├── iterated.py     IteratedExtendedKalmanFilter (local Gauss-Newton MAP update)
│   ├── unscented.py    UnscentedKalmanFilter (Scaled Unscented Transform)
│   ├── ensemble.py     EnsembleKalmanFilter (Monte-Carlo EnKF with spatial localization)
│   └── smoother.py     rts_smooth (Rauch-Tung-Striebel fixed-interval backward smoother)
├── simulation.py       simulate(): ground truth + noisy measurements with channel dropouts
└── metrics.py          rmse, mae, nees, chi2_consistency_test, autocorrelation, whiteness_test
examples/               runnable multi-sensor, tracking, and data assimilation benchmarks
tests/                  comprehensive pytest test suite
```

## Design Architecture

### 1. Separation of Models and Filters
A state-space model defines the physical problem:
- `propagate(x, u)`: state transition :math:`f(x, u)`.
- `measure(x)`: observation mapping :math:`h(x)`.
- `Q`: process noise covariance matrix :math:`(n \times n)`.
- `R`: measurement noise covariance matrix :math:`(m \times m)`.
- `residual(z, z_pred)`: innovation calculation with optional angular wrapping.
- Optional Jacobians: `state_jacobian(x, u)` and `measurement_jacobian(x)`.

Filters consume only the `StateSpaceModel` protocol. Any custom simulation, differential equation solver, or physical apparatus implementing this protocol operates seamlessly with all filters.

### 2. Multi-Channel Sensor Fusion & Missing Data (NaNs)
Multi-sensor systems frequently operate with asynchronous clocks, varying sample rates, and transmission dropouts.
In `kalman_filter`:
- Measurements are vectors of shape `(m,)` or sequences of shape `(T, m)`.
- A channel that failed to report at time step `k` is indicated by `NaN`.
- In `BaseFilter.update()`, `UnscentedKalmanFilter.update()`, and `EnsembleKalmanFilter.update()`:
  - Active indices `observed = np.flatnonzero(~np.isnan(z))` are isolated.
  - Submatrices :math:`H_{\text{obs}}`, :math:`R_{\text{obs}}`, and :math:`\nu_{\text{obs}}` are extracted.
  - The Kalman gain and belief updates proceed cleanly without matrix resizing or error.
  - If all channels are missing at step `k`, the filter propagates the prior without updating.

### 3. Estimator Taxonomy

| Filter | Class | Propagation | Update Mechanism | Computational Cost |
|---|---|---|---|---|
| Linear KF | `KalmanFilter` | :math:`F x + B u` | Exact linear Kalman gain | :math:`O(n^3 + m^3)` |
| Extended KF | `ExtendedKalmanFilter` | :math:`f(x, u)` | 1st-order Jacobian linearization | :math:`O(n^3 + m^3)` + Jacobians |
| Iterated EKF | `IteratedExtendedKalmanFilter` | :math:`f(x, u)` | Gauss-Newton iteration on :math:`h(x)` | :math:`O(K \cdot m^3)` |
| Unscented KF | `UnscentedKalmanFilter` | :math:`2n+1` sigma points | Statistical covariance matching | :math:`O(n^3 + m^3)` |
| Ensemble KF | `EnsembleKalmanFilter` | :math:`N` ensemble members | Sample covariance + Gaspari-Cohn | :math:`O(N \cdot n + m^3)` |
| RTS Smoother| `rts_smooth` | Backward recursion | Backward gain :math:`C_k` | :math:`O(T \cdot n^3)` |

### 4. Covariance Localization for Data Assimilation
In large spatial models (e.g., weather or oceanography where :math:`n \ge 40` to :math:`10^6`), finite ensemble sizes (:math:`N \ll n`) introduce spurious cross-correlations at long physical distances.
`kalman_filter.localization.gaspari_cohn` provides the standard 5th-order Gaspari-Cohn (1999) compact polynomial tapering function:
.. math::
    \rho_{xy}(r) = \begin{cases}
    1 - \frac{5}{3} r^2 + \frac{5}{8} r^3 + \frac{1}{2} r^4 - \frac{1}{4} r^5, & 0 \le r \le 1 \\
    4 - 5r + \frac{5}{3} r^2 + \frac{5}{8} r^3 - \frac{1}{2} r^4 + \frac{1}{12} r^5 - \frac{2}{3r}, & 1 < r \le 2 \\
    0, & r > 2
    \end{cases}

Applying this via Schur product :math:`P_{xy} \leftarrow P_{xy} \odot L_{xy}` eliminates long-distance noise and prevents filter divergence.

## Testing & Quality Assurance

Run the test suite:
```bash
pytest
```

Run code formatting and linting:
```bash
ruff check .
```
All code strictly follows Google Python docstring conventions, strict static typing, and PEP-8 line limits.
