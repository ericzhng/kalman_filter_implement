# Theory

## 1. Problem setting

A hidden state `x_k ∈ R^n` evolves over time and is observed through noisy sensors. Each
sensor contributes one or more **channels**; stacking all channels gives `z_k ∈ R^m`.

```
x_k = f(x_{k-1}, u_k) + w_k,   w_k ~ N(0, Q)     (process model)
z_k = h(x_k)          + v_k,   v_k ~ N(0, R)     (measurement model)
```

The filter maintains a Gaussian belief `x_k | z_{1:k} ~ N(x̂_k, P_k)`.

## 2. Linear Kalman filter

For linear models, `f(x, u) = F x + B u` and `h(x) = H x`, and the Gaussian belief is
propagated **exactly**; the filter is the minimum-variance estimator.

**Predict**

```
x̂⁻ = F x̂ + B u
P⁻  = F P Fᵀ + Q
```

**Update**

```
ν = z − H x̂⁻                 innovation
S = H P⁻ Hᵀ + R              innovation covariance
K = P⁻ Hᵀ S⁻¹                Kalman gain
x̂ = x̂⁻ + K ν
P  = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ      Joseph form
```

The Joseph form is algebraically equal to `(I − K H) P⁻` but keeps `P` symmetric and
positive semi-definite under rounding errors, so it is used throughout this library.

## 3. Multiple channels

Multi-channel fusion needs no new theory: stack the sensors.

```
z = [z_A; z_B; z_C],   H = [H_A; H_B; H_C],   R = blockdiag(R_A, R_B, R_C)
```

If sensor noises are independent, `R` is block diagonal and the update is equivalent to
processing sensors one after another. **Missing channels** (dropouts, asynchronous
sensors) are handled by keeping only the observed rows: with index set `o`,
use `H_o`, `R_oo`, `z_o`. If `o` is empty the update is skipped. See
[implementation.md](implementation.md) for how NaN encodes this.

## 4. Extended Kalman filter (EKF)

For nonlinear `f` and `h`, Gaussians no longer stay Gaussian. The EKF approximates by
linearising around the current estimate with Jacobians

```
F_k = ∂f/∂x |_{x̂_{k-1}, u_k}        H_k = ∂h/∂x |_{x̂⁻_k}
```

**Predict**: `x̂⁻ = f(x̂, u)`, `P⁻ = F P Fᵀ + Q`.

**Update**: as in the linear case but with `ν = z − h(x̂⁻)` and `H = H_k`.

The *means* go through the true nonlinear functions; only the *covariances* use the
linearisation.

### Angles and other wrapped residuals

If a channel is an angle (e.g. bearing), the innovation must be wrapped:
`ν_θ = wrap(z_θ − h_θ(x̂⁻))` to `[−π, π)`. Otherwise a measurement near the ±π cut
produces a huge false innovation. The model accepts a `residual_fn` for this.

### Limitations

- Accuracy degrades when the nonlinearity is strong relative to the uncertainty
  (large `P`) because the first-order linearisation is poor.
- Bad initial guesses can make the filter diverge or become over-confident; tune `Q`
  and `P0` and check consistency (below).
- For stronger nonlinearities consider the ensemble Kalman filter (below), or the
  unscented or particle filters (not included).

## 4b. Ensemble Kalman filter (EnKF)

The belief is an ensemble `{x_i}`, `i = 1..N`. Predict: `x_i ← f(x_i, u) + w_i`,
`w_i ~ N(0, Q)`. Update (perturbed observations), with anomalies `A = X − x̄`,
`B = Y − ȳ`, `Y_i = h(x_i)`:

```
P_xy = AᵀB / (N−1),   S = BᵀB / (N−1) + R,   K = P_xy S⁻¹
x_i ← x_i + K (z + v_i − h(x_i)),   v_i ~ N(0, R)
```

No Jacobians are needed and the covariance is the ensemble sample covariance. As
`N → ∞` it matches the KF for linear-Gaussian models. Small `N` gives sampling error
and under-dispersion (over-confidence); use more members or multiplicative inflation.

## 5. Checking a filter: consistency

If the model is correct, then

- **NEES** `eᵀ P⁻¹ e` (with true error `e = x − x̂`) averages `n`;
- **NIS** `νᵀ S⁻¹ ν` averages the number of observed channels.

Persistently larger values mean the filter is over-confident (`Q`/`R` too small or
linearisation error); persistently smaller means it is too conservative.
