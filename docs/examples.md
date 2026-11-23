# Examples

Run examples from the repository root (plots are saved to `examples/output/`):

```bash
python examples/sensor_fusion_gps_imu.py
python examples/lorenz96_assimilation.py
python examples/linear_multichannel.py
python examples/nonlinear_multichannel.py
python examples/lorenz63_benchmark.py
```

Requires `matplotlib` (`pip install -e ".[examples]"`).

---

## 1. Asynchronous Multi-Sensor Fusion & GPS Outages

[`examples/sensor_fusion_gps_imu.py`](../examples/sensor_fusion_gps_imu.py)

Simulates real-world ground vehicle localization fusing high-rate wheel odometry (10 Hz) with low-rate GPS (2 Hz). During an 8-second tunnel passage, GPS satellite signals are completely lost.

Key techniques demonstrated:
- **Stream Merging**: `merge_sensor_streams` aligns asynchronous timestamps onto a unified grid, inserting `NaN` when sensors have not triggered.
- **UKF Tracking**: The Unscented Kalman Filter propagates odometry measurements and updates position whenever GPS fixes arrive.
- **RTS Smoothing**: Backward Rauch-Tung-Striebel smoother reconstructs vehicle trajectory during the tunnel blackout, reducing position error and tightening uncertainty bounds.
- **Consistency Verification**: `chi2_consistency_test` assesses whether Normalized Innovation Squared (NIS) values fall within the theoretical 95% confidence interval.

---

## 2. Atmospheric Data Assimilation: Lorenz-96 Spatial Model

[`examples/lorenz96_assimilation.py`](../examples/lorenz96_assimilation.py)

The canonical benchmark for numerical weather prediction (Lorenz 1996). 40 coupled atmospheric variables evolve on a periodic latitude circle under constant forcing ($F = 8.0$), exhibiting strong spatio-temporal chaos.

Only 20 of the 40 variables are observed with noisy sensors. An ensemble of size $N = 20 \ll 40$ is used to assimilate measurements:
- **Standard EnKF**: Spurious sample cross-correlations at long physical distances cause severe filter divergence (mean RMSE $\approx 2.88$).
- **Localized EnKF**: The Gaspari-Cohn spatial tapering matrix (`model.compute_localization_matrix(cutoff_radius=4.0)`) cuts off spurious correlations, stabilizing the filter and reducing mean RMSE to $\approx 0.53$ (an 81% error reduction).

---

## 3. Linear Multi-Channel Fusion with Dropouts

[`examples/linear_multichannel.py`](../examples/linear_multichannel.py)

A 2D target moves with near-constant velocity, state `[x, y, vx, vy]`. Three sensors provide six measurement channels:

| Sensor | Measured Channels | Noise std | Dropout Probability |
|---|---|---|---|
| A | $x, y$ | 1.0 | 10% |
| B | $x, y$ | 3.0 | 0% |
| C | $v_x, v_y$ | 0.5 | 40% |

Demonstrates that `KalmanFilter`, `ExtendedKalmanFilter`, and `EnsembleKalmanFilter` agree on linear models while skipping missing channels automatically. Position RMSE drops from $\approx 0.95$ (raw sensor A) to $\approx 0.22 - 0.28$ (fused state).

---

## 4. Nonlinear Multi-Channel: Coordinated Turn Tracking

[`examples/nonlinear_multichannel.py`](../examples/nonlinear_multichannel.py)

A maneuvering target performs a coordinated turn with **unknown, varying turn rate** $\omega$; state $[x, y, v_x, v_y, \omega]$. Two fixed radar stations report range and bearing (4 channels, 10% dropout).

Demonstrates:
- Nonlinear kinematics and observation geometry.
- Angular residual wrapping in $[-\pi, \pi)$ via `wrap_angle`.
- Estimating hidden physical parameters ($\omega$) by state augmentation.

---

## 5. Lorenz-63 Chaotic Convection Benchmark

[`examples/lorenz63_benchmark.py`](../examples/lorenz63_benchmark.py)

The classic 3D chaotic attractor ($\sigma=10, \rho=28, \beta=8/3$). States $x$ and $z$ are observed with 10% dropouts; $y$ is unobserved. A 20-run Monte Carlo study compares the EKF against EnKF with varying ensemble sizes ($N = 10, 20, 50, 200$), proving that EnKF yields superior calibration on chaotic nonlinear dynamics.
