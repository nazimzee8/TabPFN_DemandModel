---
name: nonlinear-synthetic-data
description: >
  Design, generate, audit, and validate task-level nonlinear synthetic regression
  and classification datasets for DeepSet meta-learning. Use when working with
  src/generate_nonlinear_dgp.py, nonlinear DGP task-family routing, smooth
  additive, interaction, polynomial, RBF, piecewise, periodic, heteroskedastic,
  discontinuous, or market nonlinear response functions, nonlinear classification
  logits and probabilities, mixed-categorical nonlinear features, Parquet schemas,
  suite manifests, seed audits, coverage audits, or DeepSet training-data contracts.
---

# Nonlinear Synthetic Data

## 1. Purpose and Philosophy

This skill defines nonlinear synthetic regression and classification datasets
that extend the existing linear suite as parallel nonlinear task families. The
progression principle is:

```text
Linear suite              -> proves linear statistical competence (DONE)
Generic nonlinear suite   -> proves nonlinear function learning (THIS SKILL)
Market nonlinear suite    -> tests transfer to structured economic demand (FUTURE, deferred)
Full market DGP           -> trains the final TabPFN-style market prior (FUTURE, deferred)
```

Core rules:

- Keep regression and classification schemas separate.
- Do not mutate or reinterpret existing linear DGPs.
- Preserve deterministic generation, manifests, seed audits, coverage audits,
  and schema versioning from the linear infrastructure.
- Preserve context/query semantics and task-level train/validation/test splits.
- Each Parquet file is one independent meta-learning task.
- Outer train/validation/test splits are at the task-file level.
- Use the first 80 percent of task rows as context and the remaining 20 percent
  as query unless a versioned schema explicitly changes this policy.
- Generate latent targets from `X_clean`; expose `X_observed` when feature
  measurement noise is enabled.
- Make generation deterministic under a fixed command, git revision, and
  base seed.
- Audit realized coverage because profile weights are probabilities rather
  than quotas.

Use `src/generate_nonlinear_dgp.py` as the source of truth for nonlinear
regression. Read its tests and downstream schema consumers before changing
generation behavior.

---

## 2. Task Family Routing

Four nonlinear task families:

```text
task_family = "synthetic_nonlinear_regression"
task_family = "synthetic_nonlinear_classification"
task_family = "synthetic_nonlinear_regression_mixed_categorical"
task_family = "synthetic_nonlinear_classification_mixed_categorical"
```

Routing rules:

- All four families are generic nonlinear.
- Market-specific families (e.g., `synthetic_market_nonlinear_demand`) are
  deferred until the market mental model is defined and validated.
- Do not combine regression and classification schemas.
- Use `mu_clean_*` and target noise only for regression.
- Use `logits_*`, `probs_*`, label noise, and classification teachers only
  for classification.
- Preserve `X_train`, `y_train`, `X_test`, `y_test` as shared episode-level
  concepts.

---

## 3. Core Nonlinear DGP Abstraction

Shared feature generation:

```text
X_clean ~ feature_distribution(regime)
X_observed = X_clean + optional_feature_measurement_noise
```

### Regression canonical form

```text
mu_i = f_theta(x_i_clean)
y_i = mu_i + sigma_i * epsilon_i
```

- `mu_clean` is the canonical noiseless target name (NOT `betaX`).
- `betaX` remains reserved for linear regression only.
- Note: the existing v2 DGP (`src/generate_nonlinear_dgp.py`) uses `betaX`
  for backward compatibility; this SKILL documents the migration path to
  `mu_clean`.

### Classification canonical form

```text
logits_ik = f_theta,k(X_clean) + b_k
logits_i0 = 0   (reference class)
probs_i = softmax(logits_i / temperature)
y_clean_i ~ Categorical(probs_i)
y_i = apply_symmetric_label_noise(y_clean_i)
```

- Class imbalance through intercept calibration, not resampling.
- SMOTE is explicitly rejected. Class imbalance is an intentional DGP regime;
  `inverse_frequency_class_weight` provides the training-side correction.
  SMOTE would corrupt class priors, covariance structure, and teacher
  coefficients. Do not add `imbalanced-learn` or any SMOTE variant.

---

## 4. Feature Distribution Families

Full table of 14 feature distributions:

| Distribution | Sampling | Purpose |
|---|---|---|
| `iid_gaussian` | X ~ N(0, I) | Baseline nonlinear recovery |
| `correlated_ar` | X ~ N(0, Sigma_AR), Sigma_ij = rho^abs(i-j) | Multicollinearity |
| `block_correlated` | Block covariance with within-block rho | Grouped structure |
| `equicorrelated` | Sigma_ij = rho for i != j | Global collinearity stress |
| `student_t_features` | Standardized Student-t covariates | Heavy-tailed marginals |
| `laplace_features` | Laplace(0, 1/sqrt(2)) | Sharp-peaked features |
| `uniform_bounded` | Uniform(-sqrt(3), sqrt(3)) | Bounded features (prices, rates) |
| `skewed_lognormal` | Standardized lognormal | Skewed positive economic variables |
| `mixture_gaussian` | sum_c pi_c N(mu_c, Sigma_c) | Hidden segments, clusters |
| `low_rank_factor` | X = ZL^T + noise, Z ~ N(0, I_r) | Latent factor structure |
| `manifold_features` | X = nonlinear_transform(Z) + noise | Low-dim nonlinear manifold |
| `noise_features` | Append independent irrelevant columns | Irrelevant-column resistance |
| `feature_noise` | X_observed = X_clean + delta | Errors-in-variables |
| `mixed_categorical` | Numeric + category IDs (entity-token discipline) | Nonlinear num/cat interaction |

Reuse existing covariance, feature noise, noise feature, and mixed-categorical
utilities from `src/dgp_helpers.py` and `src/generate_nonlinear_dgp.py`.

---

## 5. Nonlinear Regression DGP Families

The nonlinear regression DGP families are organized in three tiers.

### Tier 1 -- Core parametric nonlinear models (y = f(X, theta) + eps)

These are the primary nonlinear function families. Each defines a parametric
nonlinear regression model where the teacher function `f(X, theta)` maps
features to the noiseless target `mu_clean`. The parameters `theta` are
sampled per-task.

| Family | Name | Equation |
|---|---|---|
| A | `poly` | `mu = sum_{d=1}^D (X^d) @ w_d + sum_{(i,j)} c_ij x_i x_j` |
| B | `sparse_interact` | `mu = x_S @ beta + sum_{(j,k) in I} gamma_jk x_j x_k` |
| C | `smooth_additive` | `mu = sum_j [a_j sin(w_j x_j) + b_j tanh(c_j x_j) + d_j x_j^2]` |
| D | `rbf_local` | `mu = sum_m alpha_m exp(-\|\|x_S - c_m\|\|^2 / (2 l_m^2))` |
| E | `piecewise_relu` | `mu = x @ beta + sum_m alpha_m max(0, x_j - tau_m)` |
| F | `discontinuous_threshold` | `mu = x @ beta + sum_m alpha_m 1{x_j > tau_m}` |
| G | `exp_growth` | `mu = exp(rate * clip(X @ w_unit, -5, 5)) - 1` |
| H | `power` | `mu = prod_{j in active} \|x_j + shift\|^{s_j * beta_j}` |
| I | `sigmoid` | `mu = L / (1 + exp(-k * (X @ w_unit - x0)))` |
| J | `saturation` | `mu = V_max * \|X @ w_unit\| / (K_m + \|X @ w_unit\|)` |
| K | `gompertz` | `mu = a * exp(-b * exp(-c * X @ w_unit))` |
| L | `logarithmic` | `mu = sum_j a_j * log(\|x_j\| + 1)` |

### Tier 2 -- Structural nonlinear regimes (combine core families with structural complexity)

| Regime | Name | Structure |
|---|---|---|
| M | `periodic` | `mu = x @ beta + sum_j alpha_j sin(w_j x_j + phi_j)` |
| N | `heteroskedastic` | `mu = f(x); sigma(x) = sigma_0 exp(0.5 g^T x); y = mu + sigma(x) * eps` |
| O | `high_dim_sparse_nonlinear` | `mu = f_theta(x_S)` with many irrelevant/noise features |
| P | `low_rank_compositional` | `z = xL; mu = h(z)` -- latent factor nonlinear |
| Q | `mixed_linear` | `mu = alpha * (x @ beta) + (1 - alpha) * tanh(x @ w)` |

### Tier 3 -- Domain-specific nonlinear families

| Regime | Name | Structure |
|---|---|---|
| R | `mixed_categorical_nonlinear` | `mu = f_num(x_num) + sum_j cat_effects[j][cat_j] + sum_j h_j(x_num, cat_j)` |

Note: Market-specific families (`demand_mono`, `market_nonlinear`) are deferred
until the market mental model is defined. They are not included in this plan.

### Detailed DGP equations for Tier 1 families

**A. Polynomial regression (`poly`):**

Generalizes the existing `poly_quad` (degree-2 only) to support degree D in
{2, 3, 4, 5}.

```text
mu = sum_{d=1}^{D} (X^d) @ w_d + sum_{(i,j)} c_ij x_i x_j
```

- `D` is sampled per-task from `{2, 3, 4, 5}` during training; fixed per-cell
  during evaluation.
- Coefficients `w_d` are scaled by `1/sqrt(d)` to prevent higher-order term
  domination.
- Cross-terms limited to `min(p, 5)` random pairs to control complexity.
- **Univariate case (p=1):** reduces to `mu = sum_{d=1}^D a_d x^d` -- standard
  polynomial curve.
- Purpose: tests curvature recovery, higher-order polynomial approximation.

**B. Sparse interaction regression (`sparse_interact`):**

```text
mu = x_S @ beta + sum_{(j,k) in I} gamma_jk x_j x_k
```

- Sparse random graph of pairwise interactions; `|I| = max(1, k)` pairs.
- Useful bridge toward cross-price interaction structure.
- Purpose: tests whether DeepSet can infer nonlinear interactions from context.

**C. Smooth additive nonlinear regression (`smooth_additive`):**

```text
mu = sum_j [a_j sin(w_j x_j) + b_j tanh(c_j x_j) + d_j x_j^2]
```

- GAM-like smooth nonlinear main effects.
- Purpose: in-distribution foundational nonlinear regime.

**D. RBF local-bump regression (`rbf_local`):**

```text
mu = sum_m alpha_m exp(-||x_S - c_m||^2 / (2 l_m^2))
```

- `M` centers, `l_m` bandwidths; M in `{2, 4, 8, 16}` -- sweepable grid.
- Purpose: kernel-like local structure, hidden segments.

**E. Piecewise ReLU / spline regression (`piecewise_relu`):**

```text
mu = x @ beta + sum_m alpha_m max(0, x_j - tau_m)
```

- Kinks and saturation at thresholds; M in `{1, 2, 4, 8}` -- sweepable grid.
- Purpose: nonlinear price sensitivity, inventory thresholds.

**F. Discontinuous threshold regression (`discontinuous_threshold`):**

```text
mu = x @ beta + sum_m alpha_m 1{x_j > tau_m}
```

- Abrupt regime changes; harder than smooth families.
- Purpose: stress/OOD regime for step-function decision rules.

**G. Exponential growth/decay regression (`exp_growth`):**

```text
mu = exp(rate * clip(X @ w_unit, -5, 5)) - 1
```

- `w_unit` = unit-norm projection, `rate ~ U(0.3, 1.5)`.
- Projection clipped to `[-5, 5]` to prevent overflow.
- Purpose: tests explosive and saturating exponential patterns; growth/decay
  curves.
- Relates to parametric NLS models: `y = a * exp(b * x) + c`.
- Double exponential variant: `mu = a1 * exp(b1 * x_S1) + a2 * exp(b2 * x_S2)`.

**H. Power / Cobb-Douglas regression (`power`):**

```text
mu = prod_{j in active} |x_j + shift|^{s_j * beta_j}
```

- Exponents `beta_j ~ U(0.2, 1.5)`, signs `s_j in {-1, +1}`.
- Features shifted to positive: `|x_j + 2|` to avoid zero/negative base.
- Purpose: Cobb-Douglas production functions, multiplicative power
  relationships.
- Relates to parametric NLS: `y = A * prod_j(x_j^{beta_j})`.
- Inverse power variant: `mu = a / |X @ w_unit + shift|^b + c`.

**I. Logistic/sigmoid saturation regression (`sigmoid`):**

```text
mu = L / (1 + exp(-k * (X @ w_unit - x0)))
```

- `L ~ U(1, 5)` carrying capacity, `k ~ U(0.5, 3.0)` steepness,
  `x0 ~ U(-1, 1)` midpoint.
- Purpose: smooth S-curve saturation, adoption curves, dose-response.
- Relates to parametric NLS: `y = L / (1 + exp(-k * (x - x0)))`.

**J. Michaelis-Menten / saturation regression (`saturation`):**

```text
mu = V_max * |X @ w_unit| / (K_m + |X @ w_unit|)
```

- `V_max ~ U(2, 10)`, `K_m ~ U(0.5, 3.0)`.
- Non-negative substrate via `|X @ w_unit|`.
- Purpose: enzymatic kinetics, diminishing returns, saturation effects.
- Multi-substrate variant: `mu = V * z1 / (K1 + z1) * z2 / (K2 + z2)`.

**K. Gompertz double-exponential regression (`gompertz`):**

```text
mu = a * exp(-b * exp(-c * X @ w_unit))
```

- `a ~ U(2, 8)` asymptote, `b ~ U(1, 4)` displacement, `c ~ U(0.3, 2.0)`
  growth rate.
- Asymmetric sigmoidal growth -- faster initial deceleration than logistic.
- Purpose: population growth, market penetration, tumor growth curves.

**L. Logarithmic regression (`logarithmic`):**

```text
mu = sum_j a_j * log(|x_j| + 1)
```

- `a_j ~ N(0, 1)` coefficients.
- Purpose: sublinear growth, diminishing marginal effects (like Weber-Fechner
  law).
- Multivariate variant: `mu = a * log(X @ w + 1) + b`.

### Parametric nonlinear model taxonomy

The DGP families above correspond to the following parametric nonlinear model
classes:

| Parametric class | DGP family | Fitting algorithm (for baselines) |
|---|---|---|
| Exponential: `y = a * exp(bx) + c` | `exp_growth` (G) | Levenberg-Marquardt via `scipy.optimize.curve_fit` |
| Power: `y = a * x^b` | `power` (H) | Gauss-Newton via `scipy.optimize.least_squares` |
| Logistic: `y = L / (1 + exp(-k(x-x0)))` | `sigmoid` (I) | Levenberg-Marquardt via `scipy.optimize.curve_fit` |
| Saturation: `y = Vx / (K + x)` | `saturation` (J) | Gauss-Newton via `scipy.optimize.least_squares` |
| Gompertz: `y = a * exp(-b * exp(-cx))` | `gompertz` (K) | Levenberg-Marquardt via `scipy.optimize.curve_fit` |
| Logarithmic: `y = a * log(x) + b` | `logarithmic` (L) | OLS on log-transformed features |
| Polynomial: `y = sum a_d x^d` | `poly` (A) | OLS on polynomial feature expansion |
| Trigonometric: `y = a * sin(wx + phi)` | `smooth_additive` (C), `periodic` (M) | Gradient descent |
| Piecewise linear: `y = a * max(0, x-t) + bx` | `piecewise_relu` (E) | Gradient descent |

Note: The fitting algorithms (Gauss-Newton, Gradient descent, Levenberg-Marquardt)
are NOT DGP families -- they are optimization methods relevant to baselines. The
DGP generates the data; the model/baseline must learn the relationship from
context without knowing the functional form.

### Univariate regression

All families above support the univariate case (p_signal=1, p_noise=0):

- `p_signal=1` is included in the parameter grid.
- Univariate polynomial: `mu = a_D x^D + ... + a_1 x + a_0`.
- Univariate exponential: `mu = exp(rate * x) - 1`.
- Univariate sigmoid: `mu = L / (1 + exp(-k * (x - x0)))`.
- Tests: basic nonlinear recovery from single-variable context.
- Important for interpretability and visualization of learned functions.

---

## 6. Nonlinear Classification DGP Families

All classification families share the canonical form:

```text
logits_ik = f_k(x_i_clean) + b_k
logits_i0 = 0   (reference class)
probs_i = softmax(logits_i / T)
y_clean_i ~ Categorical(probs_i)
y_i = apply_symmetric_label_noise(y_clean_i)
```

| Family | Name | Description |
|---|---|---|
| A | `nonlinear_binary_logistic` | Binary classification with nonlinear logit function; `logit = f(x)` uses smooth additive or polynomial nonlinearity |
| B | `sparse_interaction_logistic` | Binary/multiclass with sparse pairwise interaction logits; `logit_k = x_S @ w_k + sum_{(j,l)} gamma_jl x_j x_l` |
| C | `radial_decision_boundary` | Decision boundary defined by radial distance; `logit_k = alpha_k * exp(-\|\|x - c_k\|\|^2 / (2 l_k^2))` |
| D | `piecewise_margin` | Piecewise linear logit functions with kink thresholds; `logit_k = x @ w_k + sum_m alpha_km max(0, x_j - tau_m)` |
| E | `multiclass_smooth_softmax` | K-class (K in {3, 5, 10}) smooth nonlinear softmax; each class logit is a smooth additive function of X |
| F | `multiclass_sparse_highdim` | K-class with high-dimensional sparse features; each class uses a different sparse active support |
| G | `correlated_nonlinear_softmax` | K-class with correlated features (AR1, block, equicorr) and nonlinear logits |
| H | `low_margin_overlap` | Nonlinear logits with small between-class separation; tests calibration under high overlap |
| I | `label_noise_nonlinear` | Nonlinear logits with symmetric label noise sweep (0.02 to 0.20) |
| J | `class_imbalance_nonlinear` | Nonlinear logits with intercept-calibrated class imbalance (mild/moderate/severe); no SMOTE |
| K | `mixed_categorical_nonlinear` | Nonlinear logits combining numeric features and categorical entity embeddings |

### Classification DGP equations

**A. Nonlinear binary logistic:**
```text
logit_1 = sum_j [a_j sin(w_j x_j) + b_j tanh(c_j x_j)]
logit_0 = 0
probs = softmax([logit_0, logit_1] / T)
```

**B. Sparse interaction logistic:**
```text
logit_k = x_S @ w_k + sum_{(j,l) in I_k} gamma_jl x_j x_l + b_k
logit_0 = 0
```

**C. Radial decision-boundary:**
```text
logit_k = alpha_k * exp(-||x - c_k||^2 / (2 l_k^2))
logit_0 = 0
```

**D. Piecewise margin:**
```text
logit_k = x @ w_k + sum_m alpha_km max(0, x_j - tau_m) + b_k
logit_0 = 0
```

**E. Multiclass smooth softmax (K in {3, 5, 10}):**
```text
logit_k = sum_j [a_jk sin(w_jk x_j) + b_jk tanh(c_jk x_j) + d_jk x_j^2]
logit_0 = 0
```

**F. Multiclass sparse high-dimensional:**
```text
logit_k = x_{S_k} @ w_k + b_k
logit_0 = 0
```
Each class uses a different sparse active support `S_k`.

**G. Correlated nonlinear softmax:**
```text
logit_k = f_k(X_clean)   where X_clean ~ N(0, Sigma_corr)
logit_0 = 0
```

**H. Low-margin overlap:**
```text
logit_k = epsilon * f_k(x) + b_k     (epsilon small -> low margin)
logit_0 = 0
```

**I. Label-noise nonlinear:**
```text
logit_k = f_k(x) + b_k
logit_0 = 0
y_clean ~ Categorical(softmax(logits / T))
y = apply_symmetric_label_noise(y_clean, rate)
```
Label noise rate swept over `{0.02, 0.05, 0.10, 0.20}`.

**J. Class-imbalance nonlinear:**
```text
logit_k = f_k(x) + b_k     (b_k calibrated for target class prior)
logit_0 = 0
```
Imbalance through intercept calibration. SMOTE explicitly rejected.

**K. Mixed categorical nonlinear:**
```text
logit_k = f_k(x_num) + sum_j cat_class_effects[j][cat_j, k] + h_k(x_num, cat_j)
logit_0 = 0
```

---

## 7. Nonlinear Profile -- `nonlinear_stat_aware`

### Regime weights for core parametric families (Tier 1, sum to ~0.70)

| Regime | Weight | DGP family | Capability tested |
|---|---:|---|---|
| A_poly | 0.07 | `poly` (degree 2-5) | Polynomial curvature, higher-order approximation |
| B_sparse_interact | 0.07 | `sparse_interact` | Pairwise interaction recovery |
| C_smooth_additive | 0.07 | `smooth_additive` | GAM-like smooth nonlinear effects |
| D_rbf_local | 0.06 | `rbf_local` | Local kernel-like structure |
| E_piecewise_relu | 0.06 | `piecewise_relu` | Kinks, saturation, thresholds |
| F_discontinuous | 0.04 | `discontinuous_threshold` | Abrupt regime changes (stress) |
| G_exp_growth | 0.06 | `exp_growth` | Exponential growth/decay patterns |
| H_power | 0.06 | `power` | Cobb-Douglas, multiplicative power |
| I_sigmoid | 0.06 | `sigmoid` | S-curve saturation, logistic growth |
| J_saturation | 0.05 | `saturation` | Michaelis-Menten, diminishing returns |
| K_gompertz | 0.05 | `gompertz` | Asymmetric sigmoidal growth |
| L_logarithmic | 0.05 | `logarithmic` | Sublinear growth, diminishing marginal |

### Regime weights for structural families (Tier 2, sum to ~0.20)

| Regime | Weight | DGP family | Capability tested |
|---|---:|---|---|
| M_periodic | 0.04 | `periodic` | Cyclic nonlinear effects |
| N_heteroskedastic | 0.04 | `heteroskedastic` | Input-dependent noise |
| O_high_dim_sparse | 0.05 | `high_dim_sparse_nonlinear` | Feature selection + nonlinear |
| P_low_rank | 0.04 | `low_rank_compositional` | Latent factor nonlinear |
| Q_mixed_linear | 0.03 | `mixed_linear` | Linear + nonlinear blend |

### Regime weights for domain families (Tier 3, sum to ~0.05)

| Regime | Weight | DGP family | Capability tested |
|---|---:|---|---|
| R_mixed_cat | 0.05 | `mixed_categorical_nonlinear` | Numeric-categorical interaction |

Rules:

- Market families (`demand_mono`, `market_nonlinear`) deferred until market
  mental model defined.
- Univariate tasks (p_signal=1) are distributed across all regimes via the
  parameter grid.
- Remaining 0.05 weight redistributed to Tier 1 families proportionally.

---

## 8. Suite Families

Table of 15 suite families (market_nonlinear deferred):

| Family | Purpose |
|---|---|
| `primary` | Standard nonlinear mixture across regimes |
| `feature_noise` | Sweep feature_noise_level |
| `target_noise` | Regression residual noise sweep |
| `label_noise` | Classification label-noise sweep |
| `training_size` | Context size sweep; TabPFN anchor |
| `sparsity` | Active nonlinear support size sweep |
| `interaction_order` | None / pairwise / limited third-order sweep |
| `correlation` | rho sweep |
| `dimensionality` | p_signal, p_noise, p_total sweep |
| `nonlinearity_strength` | Weak / medium / strong amplitude sweep |
| `ood` | Withheld function families or feature marginals |
| `eval_only_unseen` | Regimes not seen during training |
| `hidden_holdout` | Separate seed namespace |
| `stress` | Balanced difficult regimes |

Note: `market_nonlinear` is deferred until the market mental model is defined.

---

## 9. Parameter Grids

### Shared grids

```text
n_grid:                   32, 64, 128, 256, 512, 1024
p_signal_grid:            1, 4, 8, 16, 32, 64
p_noise_grid:             0, 8, 24, 56, 120
active_s_grid:            1, 2, 4, 8, 16, 32
rho_grid:                 -0.6, 0.0, 0.3, 0.6, 0.9
feature_noise_grid:       0.0, 0.05, 0.10, 0.25
nonlinearity_strength:    weak, medium, strong
interaction_order_grid:   0, 1, 2, 3
```

Note: `p_signal=1` enables univariate regression tasks across all teacher
families. When `p_signal=1`, correlated feature regimes (AR1, block, equicorr)
degenerate to IID since correlation structure requires p >= 2. `active_s=1`
is also included for single-active-feature sparse tasks.

### Regression-specific

```text
target_noise_grid:          0.25, 0.5, 1.0, 2.0
target_noise_type_grid:     gaussian, student_t, laplace, cauchy, heteroscedastic, contaminated, gross_outlier
heteroskedasticity_grid:    0.0, 0.25, 0.5, 1.0
rbf_centers_grid:           2, 4, 8, 16
periodic_frequency_grid:    low, medium, high
threshold_count_grid:       1, 2, 4, 8
polynomial_degree_grid:     2, 3, 4, 5
```

Note on noise types for robust regression evaluation:

- `gaussian` -- standard baseline noise.
- `student_t` -- heavy-tailed, finite variance (df=3).
- `laplace` -- sharp-peaked, heavier tails than Gaussian.
- `cauchy` -- infinite variance, clipped to [-20, 20].
- `heteroscedastic` -- `sigma(x) = sigma_0 * |mu_clean|`, input-dependent.
- `contaminated` -- 10% of observations replaced by 5x noise scale outliers.
- `gross_outlier` -- 5% of observations replaced by 10x signal-scale outliers.

### Classification-specific

```text
num_classes_grid:          2, 3, 5, 10
temperature_grid:          0.5, 1.0, 2.0, 5.0
label_noise_grid:          0.0, 0.02, 0.05, 0.10, 0.20
class_imbalance_grid:      balanced, mild, moderate, severe
margin_grid:               low, medium, high
logit_scale_grid:          0.5, 1.0, 2.0, 5.0
intercept_scale_grid:      0.0, 0.5, 1.0
```

---

## 10. Parquet Schemas

### Nonlinear regression Parquet -- required columns

```text
X_train, X_test, y_train, y_test, mu_clean_train, mu_clean_test,
schema_version, task_family, task_objective, profile, prior_regime,
nonlinear_family, n_train, n_test, p_signal, p_noise, p_total,
active_s, sparsity_ratio, covariance_type, rho, feature_noise_level,
target_noise_scale, target_noise_type, heteroskedasticity_strength,
nonlinearity_strength, interaction_order, teacher_available,
polynomial_degree, parametric_model_class
```

Note: `parametric_model_class` records the parametric nonlinear model type
(e.g., "exponential", "power", "sigmoid", "saturation", "gompertz",
"polynomial", "logarithmic", "additive", "interaction", "piecewise", "rbf",
"periodic", "threshold", "heteroskedastic", "mixed", "market").

`polynomial_degree` is populated for the `poly` family; NULL for other families.

### Nonlinear regression -- optional diagnostics

```text
active_support, interaction_support, rbf_centers, rbf_bandwidths,
thresholds, function_parameters, difficulty_score, difficulty_tier,
sample_complexity_bucket, condition_number, matrix_rank, effective_rank,
snr, p_over_n, n_over_p, is_training_allowed, is_eval_only, is_ood,
is_hidden_holdout, task_fingerprint, dataset_seed,
sigmoid_steepness, sigmoid_midpoint, sigmoid_carrying_capacity,
power_exponents, exp_rate, gompertz_asymptote, saturation_vmax,
saturation_km, polynomial_coefficients, teacher_param_seed,
cal_mean, cal_std, normalization_constant
```

### Nonlinear classification Parquet -- required columns

```text
X_train, X_test, y_train, y_test, y_clean_train, y_clean_test,
label_noise_mask_train, label_noise_mask_test, logits_train, logits_test,
probs_train, dgp_teacher_probs_test, schema_version, task_family,
task_objective, profile, prior_regime, nonlinear_family, n_train, n_test,
p_signal, p_noise, p_total, num_classes, realized_num_classes,
class_imbalance_type, margin_level, realized_margin_level, class_prior,
realized_class_prior, train_class_counts, test_class_counts, temperature,
label_noise_rate, realized_label_noise_rate, feature_noise_level,
nonlinearity_strength, interaction_order, teacher_available
```

### Nonlinear classification -- optional diagnostics

```text
active_support, class_active_support, interaction_support,
logit_function_parameters, mean_margin, median_margin, min_margin,
class_entropy, minority_class_fraction, majority_class_fraction,
bayes_error_proxy, condition_number, matrix_rank, effective_rank,
p_over_n, n_over_p, is_training_allowed, is_eval_only, is_ood,
is_hidden_holdout, task_fingerprint, task_seed
```

---

## 11. Validation Gates

### Nonlinear regression validation

- Recompute `mu_clean = f_theta(X_clean)` from stored parameters.
- Assert stored `mu_clean` equals recomputed within tolerance.
- Assert noise-feature effects are zero.
- Assert `finite(X)`, `finite(mu_clean)`, `finite(y)`.
- Assert target SNR within requested bucket.
- Assert `p_signal + p_noise = p_total`.
- Assert train/query split preserved.
- Assert `n_train + n_test == n`.
- Assert `active_s <= p_signal <= p_total`.

### Nonlinear classification validation

- Recompute `logits = f_theta,k(X_clean) + b_k` from stored parameters.
- Assert `probs = softmax(logits / temperature)`.
- Assert probability rows sum to 1.
- Assert `label_noise_mask` count = `round(label_noise_rate * n)` exactly.
- Assert class imbalance through intercept calibration (not resampling).
- Assert reference-class constraints (`logits_0 = 0`, `b_0 = 0`).
- Assert all K classes present unless stress mode.
- Assert noise-column coefficients are all zero.
- Assert margin bucket matches requested `margin_level`.

### Mixed-categorical nonlinear validation

- `X_cat` int64, IDs in valid range.
- PAD/MISSING/UNKNOWN semantics preserved (special tokens 0, 1, 2 reserved).
- Cardinalities match.
- Reference-class categorical effects = zero.
- Entity embedding IDs in range
  `[ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + C)`.

---

## 12. Existing v2 DGP Implementation and Family Mapping

The current `src/generate_nonlinear_dgp.py` v2 implementation provides:

- 6 target families: `poly_quad`, `sin_low`, `hinge`, `sparse_interact`,
  `mixed_linear`, `demand_mono`.
- 7 feature regimes: `iid_dense`, `iid_sparse`, `ar1`, `block`, `equicorr`,
  `noise_feats`, `feat_noise`.
- Dual-RNG contract: `teacher_rng` for DGP params/calibration, `sample_rng`
  for eval rows/noise.
- Learnability gate: Ridge R^2 >= 1e-4.
- Suite composition: Component A (300 core), B (80 robustness), C (40 OOD).
- Training Parquet: `write_parquet()` -- X_train, y_train, X_test,
  betaX_test (v1-compat).
- Eval Parquet: `write_parquet_eval()` -- full metadata including 13 v2
  columns.
- Note: Uses `betaX` field name for backward compat; future migration to
  `mu_clean`.

### Mapping from existing v2 families to extended family registry

| v2 family | Maps to | Change |
|---|---|---|
| `poly_quad` | `poly` (A) | GENERALIZE: add `degree` param, support D in {2,3,4,5} |
| `sin_low` | `smooth_additive` (C) | SUBSUME: sin_low is a special case; smooth_additive adds tanh + quadratic terms |
| `hinge` | `piecewise_relu` (E) | RENAME: hinge is a ReLU threshold family |
| `sparse_interact` | `sparse_interact` (B) | KEEP: unchanged |
| `mixed_linear` | `mixed_linear` (Q, Tier 2) | KEEP: moves to Tier 2 structural regime |
| `demand_mono` | DEFERRED | Market family, deferred until market mental model defined |

New families with NO existing counterpart: `rbf_local` (D),
`discontinuous_threshold` (F), `exp_growth` (G), `power` (H), `sigmoid` (I),
`saturation` (J), `gompertz` (K), `logarithmic` (L), `periodic` (M),
`heteroskedastic` (N), `high_dim_sparse_nonlinear` (O),
`low_rank_compositional` (P), `mixed_categorical_nonlinear` (R).

### Suite sizing with expanded families

The current 420-cell suite (6 families x 7 regimes) expands to handle more
families:

- Option A: Keep ~420 total, reduce per-cell count (12 Tier 1 families x
  7 regimes = 84 cells, ~4 datasets/cell = 336 core).
- Option B: Expand to ~630 total (504 core + 84 robustness + 42 OOD) to
  maintain per-family coverage density.
- Recommended: Option B for comprehensive coverage.

---

## 13. Market Nonlinear Extension (Deferred)

Market-specific nonlinear families (`demand_mono`, `market_nonlinear`,
`synthetic_market_nonlinear_demand`) are deferred until the market mental
model is defined and provided. This includes:

- Market regression (demand equations with own-price/cross-price
  elasticities).
- Market classification (utility/choice models with product graphs).
- Market graph structure (sparse product interaction networks).

These families will be added as a separate SKILL.md update after:

1. The generic nonlinear suite is implemented and validated.
2. The market mental model equations are provided.
3. The DeepSet model demonstrates generic nonlinear competence.

---

## 14. Multivariate Regression (Future Extension)

**Status: DEFERRED -- requires architectural changes.**

Multivariate regression where `y` is a vector `(y_1, ..., y_K)` does not fit
the current DeepSet architecture, which predicts a single scalar output per
query row. Integration would require changes to:

- `model.py`: Add `n_outputs` field; prediction head outputs `(m, K)` not
  `(m,)`.
- `train.py`: Loss computation for multi-output targets.
- Evaluation pipeline: Per-output and joint metrics.
- Parquet schema: `y_train` as 2-D matrix not 1-D list.

**Workaround (available now):** Treat K-output tasks as K separate
single-output tasks sharing the same `X`. Generate K Parquet files from the
same feature matrix with different `y` vectors. Evaluate K times and correlate
results. This preserves existing infrastructure but does NOT capture
cross-output dependencies.

**DGP equation (when implemented):**

```text
Y = f(X, Theta) + Noise    where Y: (n, K), X: (n, p), Noise: (n, K)
```

Multivariate linear: `Y = X @ Beta + Eps`, `Beta: (p, K)`.
Multivariate nonlinear: `Y[:, k] = teacher_k(X, params_k) + eps_k` for
`k = 1..K`.

---

## 15. Stepwise and Robust Regression Integration

### Stepwise regression

**Stepwise regression** is a fitting methodology, NOT a data generating
process. It integrates as:

1. **Evaluation baseline:** `SelectKBest(f_regression, k=auto)` +
   `Ridge(alpha=1.0)` as a fast F-statistic proxy for forward stepwise
   selection.
2. **DGP regime property:** The existing `iid_sparse` and `noise_feats`
   feature regimes already generate data with sparse active feature sets
   where stepwise methods are most relevant. No new DGP regime needed.
3. **Analysis axis:** "Does DeepSet's implicit feature selection match or
   exceed explicit stepwise selection?" tracked via `beats_stepwise_ridge`
   comparison metric.

### Robust regression

**Robust regression** integrates as both DGP noise regimes and evaluation
baselines:

1. **DGP noise types (extending `_generate_noise`):**
   - `cauchy`: `eps ~ Cauchy(0, 1)`, clipped to `[-20, 20]` -- infinite
     variance.
   - `gross_outlier`: 5% of observations replaced by `10 * std(mu_clean)`
     scale outliers.
   - These join existing: `gaussian`, `student_t`, `heteroscedastic`,
     `contaminated`.
2. **Evaluation baselines:** `HuberRegressor`, `TheilSenRegressor`,
   `RANSACRegressor`.
3. **Analysis axis:** "How robust is DeepSet to heavy-tailed noise and
   outlier contamination compared to purpose-built robust estimators?"
   tracked via per-noise-type breakdowns.

---

## 16. Seed Derivation

Per-family SeedSequence derivation follows the same pattern as linear
regression:

```python
base = hidden_holdout_base_seed if family == "hidden_holdout" else base_seed
magic = _NONLINEAR_REG_HIDDEN_SEED_MAGIC if family == "hidden_holdout" else _NONLINEAR_REG_SEED_MAGIC
seed = SeedSequence([base, _FAMILY_MAGIC[family], family_idx, magic])
```

`_FAMILY_MAGIC[family]` is the first 4 bytes of
`SHA-256(family.encode())`, interpreted as a little-endian integer.

Hidden holdout seed magic constants are distinct from linear regression
magic constants to ensure no seed overlap across task families:

- `_NONLINEAR_REG_SEED_MAGIC`: distinct from `_REG_SEED_MAGIC`.
- `_NONLINEAR_REG_HIDDEN_SEED_MAGIC`: distinct from
  `_REG_HIDDEN_SEED_MAGIC`.

Classification nonlinear seeds use separate magic constants:

- `_NONLINEAR_CLS_SEED_MAGIC`: distinct from all regression magic constants.
- `_NONLINEAR_CLS_HIDDEN_SEED_MAGIC`: distinct from all other magic
  constants.

Seeds are unique across families and individually reproducible without
re-running the full suite.

---

## 17. Suite Manifest Schema

The manifest JSON at `{suite_id}/synthetic_nonlinear_manifest.json` records:

```text
schema_version
task_family
profile
base_seed
suite_id
hidden_holdout_suite_id
hidden_holdout_base_seed
allocation_mode
enabled_suite_families
realized_suite_family_counts
n_datasets_by_regime_group
grid_metadata (all grid values)
generation_controls:
  allocation_mode
  curriculum_policy
  difficulty_mix
  memory_guard_bytes
  min_regime_count
seed_audit:
  all_dataset_seeds_unique
  hidden_normal_seed_overlap
coverage_audit:
  per-axis counts and missing-coverage flags
memory_audit:
  counts by memory_class
difficulty_audit:
  counts by difficulty_tier
train_eval_alignment_audit:
  seed overlap
  fingerprint overlap
  eval-only regime leakage checks
source_git_revision
per-dataset entries:
  dataset_id
  suite_id
  suite_family
  dataset_seed
  task_fingerprint
  prior_regime
  nonlinear_family
  parametric_model_class
  n_total
  p_signal
  p_noise
  p_total
  active_s
  sparsity_ratio
  covariance_type
  rho
  target_noise_scale
  target_noise_type
  feature_noise_level
  nonlinearity_strength
  interaction_order
  polynomial_degree
  payload_bytes
  stage_path
  split_seeds
  is_training_allowed
  is_eval_only
  is_ood
  is_hidden_holdout
  difficulty_score
  difficulty_tier
  memory_class
```

---

## 18. Training-Format Data Generation and Staging

### Overview

`src/generate_nonlinear_dgp.py` exposes two generation entry points:

- **`main()`** — evaluation-format parquet (420 datasets, 6 families × 7 regimes). Used to populate
  `@EVALUATION_DATASET_STAGE`. Unchanged.
- **`main_nonlinear_training()`** — training-format parquet with 80/10/10 splits written to
  `train/`, `val/`, `test/` subdirectories. Used to populate the `@META_NONLINEAR_*` training stages.
  Invoked with `--task_family <family>` selecting one of the four nonlinear families.

### Task families and their training stages

| `--task_family` | Training stage | Training index table |
|---|---|---|
| `synthetic_nonlinear_regression` | `@META_NONLINEAR_REGRESSION_DATASET_STAGE` | `META_NONLINEAR_REGRESSION_DATASET_INDEX` |
| `synthetic_nonlinear_classification` | `@META_NONLINEAR_CLASSIFICATION_DATASET_STAGE` | `META_NONLINEAR_CLASSIFICATION_DATASET_INDEX` |
| `synthetic_nonlinear_regression_mixed_categorical` | `@META_NONLINEAR_REGRESSION_DATASET_STAGE/mixed` | `META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX` |
| `synthetic_nonlinear_classification_mixed_categorical` | `@META_NONLINEAR_CLASSIFICATION_DATASET_STAGE/mixed` | `META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX` |

All four stages are defined in `sql/synthetic_nonlinear_pipeline.sql`. All four training index
tables are `TRANSIENT TABLE`s in the same file with `DATA_RETENTION_TIME_IN_DAYS = 0`.

### Index builders

Index builders read from the corresponding `@META_NONLINEAR_*` stage, extract per-dataset metadata
from the parquet, and write to the training index table using the TRUNCATE + `save_as_table`
pattern (plain families) or DELETE + `INSERT … PARSE_JSON` pattern (mixed-categorical families
that store `categorical_cardinalities` as a VARIANT column).

| Index builder entrypoint | Family |
|---|---|
| `src/build_meta_nonlinear_classification_dataset_index.py` | `synthetic_nonlinear_classification` |
| `src/build_meta_nonlinear_mixed_regression_dataset_index.py` | `synthetic_nonlinear_regression_mixed_categorical` |
| `src/build_meta_nonlinear_mixed_classification_dataset_index.py` | `synthetic_nonlinear_classification_mixed_categorical` |

SQL build procedures (0-arg and `(EXPECTED_TOTAL INTEGER)` overloads for each) are defined in
`sql/synthetic_nonlinear_pipeline.sql`. The run-time handlers are in `scripts/run_training_job.py`.

### Seed magic constants

To prevent seed collision across the four families, `main_nonlinear_training()` applies a
per-family offset (seed magic constant) defined at the top of `generate_nonlinear_dgp.py`:

| Family | Seed magic constant |
|---|---|
| `synthetic_nonlinear_regression` | `_NONLINEAR_REG_SEED_MAGIC` |
| `synthetic_nonlinear_classification` | `_NONLINEAR_CLS_SEED_MAGIC` |
| `synthetic_nonlinear_regression_mixed_categorical` | `_NONLINEAR_MIXED_REG_SEED_MAGIC` |
| `synthetic_nonlinear_classification_mixed_categorical` | `_NONLINEAR_MIXED_CLS_SEED_MAGIC` |

### Mixed-categorical training fields

Training-format parquets for mixed families must contain:
- `X_train`, `X_test` (numeric features), `X_cat_train`, `X_cat_test` (integer-encoded categoricals)
- `categorical_cardinalities` (list of ints, one per categorical feature)
- `p_num`, `p_cat` (integer counts)
- `task_family`, `training_data_family`, `task_objective`, `schema_version`

These fields are required by the corresponding `build_meta_nonlinear_mixed_*` builder's
`_read_metadata` function, which inserts them via `PARSE_JSON` into the VARIANT column.

---

## 20. Source Ownership

Keep responsibilities conceptually separated:

- `src/generate_nonlinear_dgp.py`: nonlinear teacher functions, feature
  regimes, calibration, Parquet writing, suite composition.
- `src/dgp_helpers.py`: shared covariance, coefficient generation,
  mixed-categorical helpers, serialization helpers, validation utilities.
- `src/nonlinear_dgp_helpers.py` (if created): nonlinear-specific DGP
  helpers (e.g., parametric teacher parameter sampling, normalization,
  learnability gates).
- `scripts/generate_nonlinear.py`: CLI and eval suite generation.
- `src/generate_dgp.py`: task-family routing for new nonlinear families.

Do not duplicate shared feature distribution, covariance, or
mixed-categorical logic across files.

---

## 21. Change Workflow

When changing any nonlinear task family:

1. State the statistical capability being introduced.
2. Identify shared behavior versus task-family-specific behavior.
3. Preserve existing field semantics and version incompatible schemas.
4. Add deterministic per-task validation.
5. Generate a small suite and audit realized coverage.
6. Smoke-test representative files through the actual task-family loader.
7. Record the command, git revision, seed, profile, grids, schema version,
   and checksums in the suite manifest.
