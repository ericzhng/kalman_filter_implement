# Theory

## 1. Problem setting

A hidden state $x_k \in \mathbb{R}^n$ evolves over time and is observed through noisy sensors. Each
sensor contributes one or more **channels**; stacking all channels gives $z_k \in \mathbb{R}^m$.

```
x_k = f(x_{k-1}, u_k) + w_k,   w_k ~ N(0, Q)     (process model)
z_k = h(x_k)          + v_k,   v_k ~ N(0, R)     (measurement model)
```

The filter maintains a Gaussian belief $x_k \mid z_{1:k} \sim \mathcal{N}(\hat{x}_k, P_k)$.

---

## 2. Linear Kalman filter

For linear models, $f(x, u) = F x + B u$ and $h(x) = H x$, and the Gaussian belief is
propagated **exactly**; the filter is the minimum-variance unbiased estimator.

**Predict**

```
x̂⁻ = F x̂ + B u
P⁻  = F P Fᵀ + Q
```

**Update**

```
ν = z − H x̂⁻                                    (innovation)
S = H P⁻ Hᵀ + R                                 (innovation covariance)
K = P⁻ Hᵀ S⁻¹                                   (Kalman gain)
x̂ = x̂⁻ + K ν
P  = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ          (Joseph form)
```

The Joseph form is algebraically equal to $(I - K H) P^-$ but guarantees symmetry and
positive semi-definiteness under floating-point arithmetic.

---

## 3. Multiple channels and dropouts

Multi-sensor fusion stacks the sensor rows:

```
z = [z_A; z_B; z_C],   H = [H_A; H_B; H_C],   R = blockdiag(R_A, R_B, R_C)
```

If sensor errors are independent, $R$ is block diagonal. **Missing channels** (dropouts, asynchronous
sensors) are marked with `NaN` and handled by isolating observed indices $o$:
use $H_o, R_{oo}, z_o$. If $o$ is empty, the update step is skipped and the prior belief is retained.

---

## 4. Nonlinear Estimators

### 4a. Extended Kalman Filter (EKF)

Linearises $f$ and $h$ around the current estimate using first-order Taylor expansions:

```
F_k = ∂f/∂x |_{x̂_{k-1}, u_k},        H_k = ∂h/∂x |_{x̂⁻_k}
```

- **Predict**: $\hat{x}^- = f(\hat{x}, u)$, $P^- = F P F^T + Q$.
- **Update**: $\nu = z - h(\hat{x}^-)$, $S = H P^- H^T + R$, $K = P^- H^T S^{-1}$.

### 4b. Iterated Extended Kalman Filter (IEKF)

When $h(x)$ has strong curvature or severe observation geometry (e.g. radar close to target),
standard EKF linearization error degrades accuracy. IEKF formulates the update as a local
Gauss-Newton MAP optimization:

```
η_{j+1} = x̂⁻ + K_j (z − h(η_j) − H_j (x̂⁻ − η_j))
```

iterating until $\|\eta_{j+1} - \eta_j\| < \text{tol}$ or maximum iterations are reached.

### 4c. Unscented Kalman Filter (UKF)

Uses the Scaled Unscented Transform (Julier & Uhlmann; Wan & van der Merwe) to propagate
$2n + 1$ deterministic sigma points $\mathcal{X}_i$ through the exact nonlinear functions:

```
λ = α²(n + κ) − n,   γ = √(n + λ)
X_0 = x̂,   X_i = x̂ + (γ √P)_i,   X_{n+i} = x̂ − (γ √P)_i
```

Weights for mean $W_i^{(m)}$ and covariance $W_i^{(c)}$ capture mean and covariance to 3rd-order
accuracy for Gaussian inputs without requiring any Jacobian evaluations.

### 4d. Stochastic Ensemble Kalman Filter (EnKF)

Represents belief by a Monte-Carlo ensemble $\{x_i\}_{i=1}^N$.
- **Predict**: each member is integrated forward with random process noise $w_i \sim \mathcal{N}(0, Q)$.
- **Update**: perturbed observations $z_i = z + v_i, v_i \sim \mathcal{N}(0, R)$, updating members
  with gain $K = P_{xy} (P_{yy} + R)^{-1}$.

### 4e. Spatial Covariance Localization

In high-dimensional systems ($n \gg N$), sampling error produces spurious long-range correlations.
The 5th-order Gaspari-Cohn (1999) tapering function modulates the cross-covariance:

$$\tilde{P}_{xy} = P_{xy} \odot \rho_{xy}$$

where $\rho_{xy}(r)$ smoothly transitions from $1$ at distance $0$ to exactly $0$ at distance $2 c$,
preventing filter divergence in spatial data assimilation.

---

## 5. Rauch-Tung-Striebel (RTS) Smoothing

Given forward filtering estimates $\{\hat{x}_{k \mid k}, P_{k \mid k}\}$ and predictions
$\{\hat{x}_{k+1 \mid k}, P_{k+1 \mid k}\}$, the RTS backward recursion runs from $T-2$ down to $0$:

```
C_k = P_{k|k} F_kᵀ P_{k+1|k}⁻¹
x̂_{k|T}^s = x̂_{k|k} + C_k (x̂_{k+1|T}^s − x̂_{k+1|k})
P_{k|T}^s = P_{k|k} + C_k (P_{k+1|T}^s − P_{k+1|k}) C_kᵀ
```

Because it incorporates both past and future observations, $P_{k \mid T}^s \le P_{k \mid k}$ across
all interior steps, yielding minimum-variance trajectory reconstructions.

---

## 6. Statistical Consistency Diagnostics

A Kalman filter is consistent if its state errors and innovations match its reported uncertainty:

1. **NEES (Normalized Estimation Error Squared)**:
   $$\epsilon_k^x = (x_k - \hat{x}_k)^T P_k^{-1} (x_k - \hat{x}_k)$$
   For a consistent filter, $\mathbb{E}[\epsilon_k^x] = n$ (state dimension).

2. **NIS (Normalized Innovation Squared)**:
   $$\epsilon_k^z = \nu_k^T S_k^{-1} \nu_k$$
   $\mathbb{E}[\epsilon_k^z] = m_k$ (number of observed channels at step $k$).

3. **Chi-Square Consistency Hypothesis Test**:
   Over $K$ time steps, the average test statistic satisfies $K \bar{\epsilon} \sim \chi^2(K \cdot \text{dof})$.
   `chi2_consistency_test` computes the theoretical two-sided 95% confidence interval and checks if the filter
   is correctly calibrated (neither overconfident nor conservative).

4. **Whiteness Test**:
   Optimal Kalman innovations must be uncorrelated white noise across time. `whiteness_test` checks
   sample autocorrelations against the 95% Bartlett bounds $\pm 1.96 / \sqrt{N}$.
