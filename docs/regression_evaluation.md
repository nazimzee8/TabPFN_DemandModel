# Synthetic Regression Evaluation Suite

## Methodology and Reproducibility Reference

---

## 1. Overview

This document describes a controlled generalization and stability benchmark designed to
evaluate a trained DeepSet model on unseen synthetic linear regression tasks before
applying the model to richer market-demand data. The benchmark measures three
distinct dimensions of model behavior:

1. **Noise robustness** — how well a model separates signal from irrelevant features as the
   proportion of uninformative predictors increases.
2. **Training set size sensitivity** — how quickly a model's signal-recovery improves as
   the number of available training examples grows from very small (25) to the model's
   pretraining regime (4,832).
3. **Target noise sensitivity** — how gracefully a model degrades as the standard deviation
   of the additive observation noise grows from zero to the standard regime.

All metrics are computed against the **noiseless signal** `betaX = Xβ`, not against noisy
observations `y_observed`. Using the noiseless signal as the evaluation target eliminates
irreducible noise variance and enables fair comparison of signal-recovery ability across
models and noise conditions. The primary ranking metric throughout is mean squared error
against `betaX` (`mse_betaX`).

---

## 2. Data Generation

### 2a. Data Generating Process (DGP)

Each dataset is drawn from a linear model:

```
y = Xβ + ε · σ
```

where `X` is the feature matrix, `β` is a coefficient vector, `ε` is a noise draw, `σ` is
a noise scale multiplier, and `betaX = Xβ` is the noiseless signal stored as the
evaluation target. Four DGP regimes are used:

| Regime | Feature distribution | Coefficient structure | Noise distribution |
|--------|---------------------|-----------------------|-------------------|
| **A** | Standard normal, iid | Standard normal β | Standard normal ε |
| **B** | Standard normal, iid | Sparse: 70% of coefficients zeroed; nonzero drawn from N(0, 2) | Standard normal ε |
| **C** | Standard normal, iid | Standard normal β | Heavy-tailed t(3) ε |
| **D** | AR(1) correlated (ρ = 0.6) | Standard normal β | Standard normal ε |

In regime D the feature rows are generated sequentially:
`X[t] = 0.6 · X[t−1] + √0.64 · z[t]` where `z[t] ~ N(0, I)`, inducing temporal
autocorrelation within each row.

All datasets across all suites are generated from a single root RNG seeded at
`np.random.default_rng(seed=20260512)`, with suites drawn in order (primary →
feature noise → training size → target noise), making the full corpus deterministically
reproducible.

### 2b. Parameter Sampling

Dataset dimensions `(n, p)` are sampled differently per suite family to control for the
confounds each suite is designed to isolate.

**Primary suite** — Rejection sample until `p ≥ 1`, `n ≥ 5`, and `n ≥ 5p`:

```python
n ~ max(5, Poisson(200))
p ~ max(1, Poisson(10))
# accept when n ≥ 5p
```

**Feature noise suite** — Signal dimensionality sampled first; row count then enforced to
be at least five times the signal dimension and at least 50:

```python
p_signal ~ Poisson(10)  # accept when ≥ 1
n = max(Poisson(200), 5 · p_signal, 50)
```

Noise features are appended after the signal block and are independent of `β` and
`betaX`.

**Training size suite** — A fixed `n_total = 6,203` rows per dataset (4,832 train anchor +
1,371 holdout). Feature dimension sampled from `Poisson(10)`, accepted when
`n_total ≥ max(5, p)`.

**Target noise suite** — Same rejection scheme as the primary suite.

### 2c. Four Evaluation Suites

| Suite | Datasets | Split seeds | Conditions | Stored NPZs |
|-------|----------|-------------|------------|-------------|
| Primary | 200 | 5 (seeds 0–4) | — | 200 |
| Feature noise | 80 | 3 (seeds 0–2) | 6 noise counts: 0, 10, 25, 50, 75, 100 appended features | 480 |
| Training size | 40 | 3 (seeds 0–2) | 8 n_train values: 25, 50, 100, 200, 500, 1,000, 2,000, 4,832 | 40 |
| Target noise | 40 | 3 (seeds 0–2) | 5 σ scales: 0.0, 0.1, 0.25, 0.5, 1.0 | 200 |

Datasets are distributed evenly across the four regimes: the primary suite contains 50
datasets per regime (A/B/C/D); similarly for feature noise (20 base datasets per regime),
training size (10 per regime), and target noise (10 per regime).

**Feature noise suite** — Noise is *pre-baked* into the NPZ. A separate file is stored for
each (base dataset, noise count) pair; the noise features are iid N(0, 1) columns appended
after the signal columns. The `betaX` signal is derived exclusively from the signal columns
and is unaffected by appended noise. Six stored noise counts — {0, 10, 25, 50, 75, 100}
extra columns — span the range from no corruption to a heavily noise-dominated feature
space.

**Training size suite** — A single NPZ is stored per dataset with `n_total = 6,203` rows.
The training set size is varied at evaluation time by slicing the permuted row order.
The eight grid values {25, 50, 100, 200, 500, 1,000, 2,000, 4,832} include 4,832 as the
"TabPFN anchor" — the training size matching the DeepSet model's pretraining regime. The
holdout is always the fixed last 1,371 rows of the permutation.

**Target noise suite** — Similar to the primary suite but each base dataset is regenerated
at five noise scale values. The `betaX` signal stored in each NPZ is computed from the
*original* `β` with `σ = 1.0`; the scale only multiplies the additive noise term `ε`, so
the signal target is consistent across scales.

### 2d. NPZ Schema

Each dataset is stored as a compressed NumPy archive (`np.savez_compressed`,
`allow_pickle=False`). All metadata fields are stored as 1-element typed arrays to avoid
object pickling.

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `X` | (n_total, p_total) | float64 | Feature matrix (signal columns first, noise columns last) |
| `y` | (n_total,) | float64 | Noisy observations (y = betaX + σ · ε) |
| `betaX` | (n_total,) | float64 | Noiseless signal — primary evaluation target |
| `suite_family` | (1,) | str | `primary` / `feature_noise` / `training_size` / `target_noise` |
| `prior_regime` | (1,) | str | DGP regime: A / B / C / D |
| `n_total` | (1,) | int64 | Total rows in the file |
| `p_signal` | (1,) | int64 | Number of signal features (first p_signal columns) |
| `p_noise` | (1,) | int64 | Number of appended pure-noise features |
| `p_total` | (1,) | int64 | p_signal + p_noise (total columns in X) |
| `target_noise_scale` | (1,) | float64 | σ multiplier applied to ε (1.0 = standard regime) |
| `training_size_anchor` | (1,) | bool | True for all training-size suite datasets (n_train = 4,832 anchor present) |
| `feature_noise_level` | (1,) | int64 | Count of appended pure-noise columns; takes values from {0, 10, 25, 50, 75, 100} |

### 2e. Train/Holdout Splitting

Splitting is performed at evaluation time, not baked into the NPZ. For a given
`split_seed`, a deterministic permutation of row indices is generated:

```python
perm = np.random.default_rng(split_seed).permutation(n_total)
train_idx   = perm[:n_train]
holdout_idx = perm[n_total - n_holdout:]
```

This ensures that training and holdout sets never overlap regardless of the `n_train`
value chosen. For the primary, feature noise, and target noise suites the split is
approximately 80/20 (`n_holdout = n_total - round(n_total × 0.8)`). For the training
size suite the holdout is always exactly 1,371 rows, with `n_train` drawn from the
eight-point grid at evaluation time.

---

## 3. Preprocessing

A `StandardScaler` is fitted exclusively on `X_train` and the resulting transform is
applied identically to `X_holdout`. No statistics derived from holdout data are used at
any point in the pipeline. After scaling, any residual NaN or infinite values (which do
not occur in this purely synthetic corpus but are guarded against for robustness) are
replaced with 0.

For the DeepSet model, a feature cap of 128 columns is enforced. When `p_total > 128`,
univariate `f_regression` scores are computed on `(X_train_scaled, y_train)` and the top
128 columns are selected. The same column mask derived from training data is applied to
`X_holdout_scaled`. This selection is identified in the output as
`feature_selector = "train_f_regression"`.

Synthetic data contains no categorical variables; no encoding is required.

---

## 4. Models Evaluated

### 4a. DeepSet (Primary Model)

The DeepSet model is a permutation-invariant neural network pretrained on synthetic
tabular regression tasks. The checkpoint is loaded from an internal model stage
(`@MODEL_STAGE/checkpoints/best.pt`) and verified with permutation invariance tests before
any inference is performed. A failed permutation test is treated as a fatal error.

Inference uses a **bounded-context ensemble**:

- **Context windows**: 5 independent subsamples of 200 training rows each
- **Monte Carlo passes per window**: 8 (`MC_K = 8`)
- **Test batch size**: 128 rows per forward pass
- **Feature cap**: 128 columns (train-only univariate selection when `p_total > 128`)

The final prediction for each holdout row is the mean across all 5 × 8 = 40 forward
passes. GPU memory is checked before inference; datasets that would exceed available VRAM
are marked as skipped rather than causing an OOM crash.

### 4b. Baseline Methods

Ten baseline methods are evaluated on every dataset:

| Method | Description |
|--------|-------------|
| `FixedRidgeLambda1` | Ridge regression with α = 1.0, no hyperparameter tuning — serves as the primary ratio denominator |
| `LinearRegression` | Ordinary least squares |
| `Ridge` | Ridge regression with cross-validated α |
| `RandomForest` | Random forest ensemble |
| `XGBoost` | Gradient boosting (XGBoost) |
| `LightGBM` | Gradient boosting (LightGBM) |
| `CatBoost` | Gradient boosting (CatBoost 1.2.10) |
| `KNN` | k-nearest neighbours regression |
| `SVR` | Support vector regression |
| `MLP` | Multi-layer perceptron |

CatBoost is installed at runtime via pip (`catboost==1.2.10`). All other baseline
dependencies are available in the benchmark runtime environment.

### 4c. AutoGluon

AutoGluon is run with a 300-second time budget and `best_quality` presets:

```python
TabularPredictor(problem_type='regression').fit(
    time_limit=300,
    presets='best_quality',
)
```

This trains and ensembles multiple internal learners automatically. AutoGluon artifacts
written to `/tmp` are deleted after each dataset fit to prevent disk exhaustion across the
30-shard pool. AutoGluon is installed at runtime (`autogluon.tabular==1.3.0`).

---

## 5. Evaluation Metrics

All primary metrics are computed against `betaX_holdout` — the noiseless signal on the
holdout rows. Secondary metrics are additionally computed against the noisy observations
`y_holdout` for reference.

**Primary metrics (vs. noiseless signal `betaX`):**

| Metric | Description |
|--------|-------------|
| `mse_betaX` | Mean squared error — primary ranking metric |
| `rmse_betaX` | Root MSE |
| `mae_betaX` | Mean absolute error |
| `r2_betaX` | Coefficient of determination |

**Secondary metrics (vs. noisy observations `y_observed`):**

| Metric | Description |
|--------|-------------|
| `mse_y_observed` | MSE against noisy target |
| `rmse_y_observed` | Root MSE against noisy target |
| `mae_y_observed` | MAE against noisy target |
| `r2_y_observed` | R² against noisy target |

Evaluating against `betaX` rather than `y_observed` removes the irreducible noise floor
from the error signal, so that differences in `mse_betaX` reflect differences in the
model's ability to recover the true linear signal rather than differences in realized noise.

---

## 6. Aggregation and Ranking

### 6a. Evaluation Unit

Each unique combination of `(dataset_id, split_seed, suite_condition)` constitutes an
independent evaluation unit. The suite condition is the noise count for the feature noise
suite, `n_train` for the training size suite, `target_noise_scale` for the target noise
suite, and absent for the primary suite.

### 6b. Within-Unit Ranking

Within each evaluation unit, all methods with a finite `mse_betaX` are ranked using dense
rank (lower MSE → rank 1). R² is ranked in descending order (higher R² → rank 1).
Methods with a skipped or failed status receive NaN ranks.

### 6c. Reference Ratios

Three reference ratios are computed per evaluation unit:

| Ratio | Formula | Interpretation |
|-------|---------|----------------|
| `ratio_mse_to_fixed_ridge` | method_mse / FixedRidgeLambda1_mse | < 1.0 means method beats Ridge(α=1) |
| `ratio_mse_to_autogluon` | method_mse / AutoGluon_mse | < 1.0 means method beats AutoGluon |
| `ratio_mse_to_best_tree` | method_mse / min(RF, XGB, LGB, CatBoost)_mse | < 1.0 means method beats best tree |

`FixedRidgeLambda1` is used as the primary denominator because it provides a stable, tuning-free
linear reference that is always computable regardless of the other methods' success.

### 6d. Stability Analysis (Feature Noise Suite)

The zero-noise condition (`feature_noise_level = 0`) serves as the baseline for each
(method, prior_regime) group. Stability metrics relative to this baseline:

- `mse_degradation_vs_noise0` — absolute MSE increase as noise features are added
- `relative_mse_degradation_pct` — percentage MSE increase relative to the zero-noise baseline
- `r2_drop_vs_noise0` — R² decrease from the zero-noise baseline
- `rank_change_vs_noise0` — rank change from the zero-noise baseline

### 6e. Sample Efficiency Analysis (Training Size Suite)

The smallest training set (`n_train = 25`) serves as the baseline. Sample efficiency
metrics relative to this baseline:

- `mse_improvement_vs_25` — absolute MSE reduction as more training data is added
- `relative_mse_improvement_pct` — percentage MSE reduction relative to the 25-row baseline
- `is_tabpfn_anchor` — flag marking the `n_train = 4,832` condition (matches the DeepSet
  model's pretraining regime)

---

## 7. Compute Environment

Evaluation jobs were submitted as Snowpark ML container services on Snowflake. Each job
runs as a single-instance container (`target_instances = 1`); no distributed collective
communication is used.

| Pool | Hardware | Purpose | Shards |
|------|----------|---------|--------|
| `DEEPSET_GPU_POOL` | NVIDIA A10G GPU instances | DeepSet inference | 10 |
| `DEEPSET_CPU_POOL` | Standard CPU instances | Baselines + aggregation | 3 (baselines), 1 (aggregation) |
| `AUTOGLUON_CPU_POOL` | Standard CPU instances | AutoGluon | 30 |

Dataset index rows are distributed to shards by modulo assignment
(`row_index % num_shards == shard_index`), so each shard receives a balanced and
non-overlapping subset of the dataset corpus. AutoGluon shards process one dataset at a
time, and temporary artifacts are cleaned after each fit to manage disk pressure.

External PyPI access (`TABPFN_PYPI_EAI`) is used to install `catboost==1.2.10` on baseline
nodes and `autogluon.tabular==1.3.0` on AutoGluon nodes at job startup.

---

## 8. Output Files

All output files are written to `@EVALUATION_RESULTS_STAGE`.

| File | Granularity | Description |
|------|-------------|-------------|
| `synthetic_regression_model_comparison.csv` | One row per (method, dataset, split_seed, condition) | Full canonical results with ranks and ratios |
| `synthetic_regression_model_comparison_summary.csv` | One row per method | Aggregated stats: mean/median MSE, win rates, ratio medians, beat-rates |
| `synthetic_regression_summary_by_regime.csv` | (suite_family, prior_regime, method) | Per-regime breakdown of MSE, R², and rank |
| `synthetic_regression_summary_by_feature_noise.csv` | (feature_noise_level, prior_regime, method) | Stability metrics including MSE degradation vs. zero-noise baseline |
| `synthetic_regression_summary_by_training_size.csv` | (n_train, prior_regime, method) | Sample efficiency metrics including MSE improvement vs. 25-row baseline |
| `synthetic_regression_chart_data_noise_features.csv` | (noise_level, regime, method) | Chart-ready aggregated data for noise stability plots |
| `synthetic_regression_chart_data_training_size.csv` | (n_train, regime, method) | Chart-ready aggregated data for sample efficiency plots |
| `synthetic_regression_chart_data_model_rank.csv` | (suite_family, condition, regime, method) | Chart-ready rank comparison across conditions |

Part files produced by each shard (named
`{method}_shard{i}_of_{n}_detailed.csv`) are written to a subdirectory
`@EVALUATION_RESULTS_STAGE/regression` and are consumed by the
aggregation step.

---

## 9. Adapting to a Different Model

The evaluation infrastructure is designed so that the primary model can be substituted
without modifying the data preparation, preprocessing, metrics, or aggregation logic.

**Steps to substitute a model:**

1. Implement a prediction function with the signature:
   ```python
   def predict_my_model(
       X_train: np.ndarray,
       y_train: np.ndarray,
       X_holdout: np.ndarray,
   ) -> np.ndarray:
       ...
   ```

2. Add a new mode function (e.g., `run_mymodel_synthetic_regression()`) in
   `evaluate_synthetic_regression.py`, modelled on
   `run_deepset_synthetic_regression()`. The dataset loading, split construction,
   preprocessing, metric computation, and output writing are all reusable without
   modification:
   - `preprocess_train_only(X_train, X_holdout)` — leakage-free StandardScaler
   - `build_split_for_seed(data, split_seed, n_train_override)` — deterministic splits
   - `compute_regression_metrics(y_pred, betaX_holdout, y_observed)` — all eight metrics
   - `assign_synthetic_regression_shard(rows, shard_index, num_shards)` — modulo sharding
   - `write_part_csv_to_stage(session, rows, method, shard_index, num_shards)` — stage upload

3. Remove DeepSet-specific components: checkpoint loading and verification,
   permutation invariance tests, bounded-context ensembling, GPU memory guards, and the
   128-feature cap with `f_regression` selection.

4. Register the new mode in the `main()` dispatch and set
   `SYNTHETIC_REGRESSION_MODE` to the new mode name in the submission environment.

5. The aggregation step (`SYNTHETIC_REGRESSION_MODE = "aggregate"`) is fully
   model-agnostic. It reads any `*_detailed.csv` part files found under
   `@EVALUATION_RESULTS_STAGE/regression` and produces the same
   summary CSVs regardless of which models contributed rows.

A reader with a standard Python ML stack (scikit-learn, numpy, pandas) can run the
baselines and aggregation locally by replacing the Snowpark session calls with local file
I/O and substituting the stage paths with local directory paths.
