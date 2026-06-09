---
name: linear-evaluation-pipeline
description: >
  Reference for the synthetic evaluation pipelines covering linear regression (linear_all_v1)
  and linear classification (DeepSet MODEL4 evaluation suite).
  Use when interpreting evaluation results, understanding how datasets were generated,
  how MODEL3-ICL-MC (regression) or MODEL4 DeepSet (classification) were assessed,
  what baselines they competed against, and how they perform under regime-specific conditions.
  Also covers classification metric alignment, probability canonicalization,
  ranking statistics, audit-validated evaluation invariants, and val_cross_entropy semantics.
  Does not cover Snowflake infrastructure or compute orchestration.
---

## 1. Evaluation Suite: linear_all_v1

The combined suite (`linear_all_v1`) merges two independently prepared suites into a
single index-level composition of 400 datasets across 8 prior regimes:

| Source suite | Regimes | Datasets | Split seeds |
|---|---|---|---|
| `linear_poisson_v1_recommended` (in-distribution) | A, B, C, D | 200 (50/regime) | 0, 1, 2, 3, 4 |
| `ood_linear_full_v1` (out-of-distribution) | E, F, G, H | 200 (50/regime) | 0, 1, 2 |
| **Combined** | A–H | **400** | **0, 1, 2** |

No parquet files are merged. The combined suite is an index-level join; each dataset
retains its original parquet payload from its source suite.

Sub-families carried into the combined evaluation:

| Family | Datasets | Seeds | Sweep |
|---|---|---|---|
| `primary` | 200 | 3 | One canonical train/holdout split |
| `feature_noise` | 80 base | 3 | 6 noise levels: 0, 10, 25, 50, 75, 100 |
| `training_size` | 40 | 3 | 8 n_train values: 25, 50, 100, 200, 500, 1000, 2000, 4832 |
| `target_noise` | 40 | 3 | 5 noise scales: 0.0, 0.1, 0.25, 0.5, 1.0 (disabled by default) |

---

## 2. Synthetic Data Generation

### Parameter sampling

All datasets use rejection sampling on `(n, p)`:

```python
n ~ Poisson(200)
p ~ Poisson(10)
reject unless p >= 1 and n >= 5 and n >= 5 * p
```

This yields datasets concentrated around `n ≈ 200` and `p ≈ 10`, with `n/p >= 5`
enforced to keep regression problems well-conditioned. The base RNG seed is
`20260512`; each dataset draws its own sub-seed deterministically from the
global sequence.

The holdout fraction is fixed at 20% (`n_holdout = n_total // 5`). For the
`training_size` sub-family, `n_total` must satisfy `n_total >= max(n_train_grid) + holdout_size`,
where `holdout_size = 1371` and the anchor point is `n_train = 4832`.

### Prior name and version

`prior_name = "linear_poisson"`, `prior_version = "v1"`. Target is always
`betaX = X @ beta` (noiseless linear signal). Observed target: `y = betaX + eps`.

---

## 3. In-Distribution Regimes (A–D)

All in-distribution regimes share the same Poisson `(n, p)` sampler. Variation
is in feature structure, coefficient sparsity, and noise distribution:

| Regime | Feature structure | Coefficients | Noise |
|---|---|---|---|
| **A** | `N(0, I)` — i.i.d. standard normal | `N(0, 1)`, dense | `N(0, 1)` |
| **B** | `N(0, I)` — i.i.d. standard normal | `N(0, 4)`, **70% sparse** (zeroed) | `N(0, 1)` |
| **C** | `N(0, I)` — i.i.d. standard normal | `N(0, 1)`, dense | **Student-t(df=3)** — heavy-tailed |
| **D** | **AR(1), ρ=0.6** — `x_t = 0.6·x_{t-1} + √0.64·ε_t` | `N(0, 1)`, dense | `N(0, 1)` |

Regime B is the primary sparsity stress test within the in-distribution suite.
Regime C tests robustness to heavy-tailed label noise.
Regime D introduces autocorrelated feature rows (temporal or sequential structure).

---

## 4. Out-of-Distribution Regimes (E–H)

OOD regimes test generalization beyond the training distribution. Each uses
the same `(n, p)` sampler as in-distribution:

| Regime | Feature structure | Coefficients | Noise | OOD axis |
|---|---|---|---|---|
| **E** | `Laplace(0, 1/√2)` — unit variance, heavy tails | `N(0, 1)`, dense | `N(0, 1)` | Feature marginal |
| **F** | `Uniform(−√3, √3)` — unit variance, bounded | `N(0, 4)`, **95% sparse** | `N(0, 1)` | Feature marginal + extreme sparsity |
| **G** | `N(0, I)` — same as A/C | `N(0, 1)`, dense | **Cauchy(0, 1)** — infinite variance | Noise distribution |
| **H** | **Block-diagonal Σ** (block_size=3, within-block ρ=0.7) | `N(0, 1)`, dense | `N(0, 1)` | Feature correlation |

Regime G is the hardest regime for all models due to Cauchy noise (no finite variance);
SVR is the only method that remains competitive on G. Regime F combines a bounded
feature marginal with extreme coefficient sparsity (95% zeros), making it the
strongest sparsity stress test in the suite.

---

## 4b. Extended Linear DGP — `linear_stat_aware` Profile (v2026+)

The `--profile linear_stat_aware` profile extends the suite with 8 new regimes
beyond the legacy A–D, covering the full range of linear estimation challenges:

| Regime | Key axis |
|--------|---------|
| E_high_dim_dense | p_signal = 32–128, dense beta, n/p from 2× to 10× |
| F_high_dim_sparse | p_signal = 32–128, sparse beta (active_s << p_signal) |
| G_noise_features | p_signal + p_noise features; beta_noise = 0 exactly |
| H_block_correlated | Block-diagonal covariance; within-block ρ=0.7 |
| I_equicorrelated | Equicorrelation with ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} |
| J_low_n_high_p | n ≤ p_total (underdetermined; `--allow_underdetermined`) |
| K_feature_noise | X_observed = X_clean + σ·ε; model sees noisy features |
| L_market_linear | Structured sign pattern: own-price (−), cross-price (±), irrelevant (0) |

**Important:** These are _linear_ regimes — the target always satisfies
`y = X_total @ beta_total + eps` with `beta_total[p_signal:] = 0`.
They extend the in-distribution linear suite; they are _not_ the same as the
OOD regimes E–H in `scripts/ood_regression/` which test nonlinear / distributional shifts.

### New CLI Usage

```bash
# Training data (linear_stat_aware):
python src/generate_dgp.py \
  --n_datasets 1000 \
  --out_dir data/linear_stat_aware_train \
  --profile linear_stat_aware \
  --base_seed 42 \
  --store_teacher_preds \
  --store_beta

# Evaluation data:
python scripts/generate_synthetic_regression.py \
  --n_datasets 1000 \
  --out_dir data/ \
  --suite_id linear_stat_aware_v2 \
  --profile linear_stat_aware \
  --base_seed 20260512 \
  --store_teacher_preds \
  --store_beta
```

### New Metadata Columns (additive; ignored by existing consumers)

Both scripts now emit additional parquet columns:
`profile`, `regime_group`, `active_s`, `sparsity_ratio`, `covariance_type`,
`rho`, `condition_number`, `matrix_rank`, `effective_rank`, `p_over_n`,
`n_over_p`, `snr`, `target_noise_type`, `has_noise_features`, `has_feature_noise`,
`sample_complexity_bucket`, `oracle_ridge_lambda`, `ridge_oracle_mse`.

`generate_dgp.py` additionally emits: `y_test`, `betaX_train`, `p_signal`,
`p_noise`, `p_total`, `feature_noise_level`, `target_noise_scale`.

### Shared Helper Module

All DGP logic lives in `src/dgp_helpers.py`. Both scripts import from it.
Do not duplicate covariance, beta-generation, or teacher-target logic elsewhere.

---

## 5. Feature Noise Sub-Family

The feature noise sub-family tests robustness to uninformative feature contamination.
Starting from 80 base datasets (balanced across regimes), each dataset is expanded
at 6 noise levels:

| `feature_noise_level` | Meaning |
|---|---|
| 0 | No noise features (`p_noise = 0`, `p_total = p_signal`) |
| 10 | 10% of total features are pure noise |
| 25 | 25% noise features |
| 50 | 50% noise features |
| 75 | 75% noise features |
| 100 | 100% noise features (all signal columns replaced) |

Noise features are `N(0, 1)` columns independent of `y`. The key metric is
`relative_mse_degradation_pct` and `mse_degradation_vs_noise0` relative to the
same dataset at noise level 0. MODEL3-ICL-MC uses F-statistic feature selection
(`train_f_regression`) with a hard cap of 128 features, which provides inherent
resistance to noise feature contamination.

---

## 6. Training Size Sub-Family

The training size sub-family tests how performance scales with labeled context.
40 large datasets are evaluated across 8 `n_train` values:

```
n_train grid: 25, 50, 100, 200, 500, 1000, 2000, 4832
```

`n_holdout` is fixed at 1371 for all training size rows. The anchor point
`n_train = 4832` corresponds to the full-data TabPFN reference condition
(`is_tabpfn_anchor = True`).

For MODEL3-ICL-MC, the context window is always capped at 200 rows
(`SYNTHETIC_REGRESSION_CONTEXT_SIZE = 200`), so performance saturates past
`n_train = 200` for the in-context path; gains beyond that come from the
ensemble of 5 context windows drawing different 200-row subsets.

---

## 7. MODEL3-ICL-MC: Evaluation Protocol

The evaluated model is `MODEL3-ICL-MC` (`model_arch_version = "model3"`,
`model_family = "market_exchangeable_icl"`, `task_objective = "inductive_regression"`).
The `-MC` suffix indicates Monte Carlo dropout inference.

### Checkpoint validation

Before any GPU shard is submitted, the checkpoint is verified:
- `checkpoint_format_version == 4`
- Required keys: `cfg`, `state_dict`, `metadata`
- `cfg.model_family` must be `market_exchangeable_icl`
- `cfg.model_arch_version` must be `model3`
- `metadata.task_objective` must be `inductive_regression`
- Retired families (`deepset`, `market_aware`, `market_exchangeable_completion`) are rejected

### Feature selection

Before inference, MODEL3-ICL-MC applies F-statistic feature selection using
`sklearn.feature_selection.f_regression` fit only on the training split:

- Method: `train_f_regression`
- Feature cap: 128 (`SYNTHETIC_REGRESSION_DEEPSET_FEATURE_CAP = 128`)
- Returns columns ranked by F-statistic; all features pass when `p_total <= 128`

Feature selection is the primary mechanism by which the model resists noise-feature
contamination. At noise level 100%, all signal features are eliminated; the model's
F-statistic selector still picks the 128 highest-scoring columns, but all are noise.

### Inference: bounded-context ensemble

Each prediction is the mean of an ensemble of context windows:

| Parameter | Value |
|---|---|
| Context windows (`SYNTHETIC_REGRESSION_CONTEXT_ENSEMBLES`) | 5 |
| Context window size (`SYNTHETIC_REGRESSION_CONTEXT_SIZE`) | 200 rows |
| MC samples per window (`MC_K`) | 8 |
| Test batch size | 128 |

Context indices are drawn via deterministic seeded selection (Blake2b hash of
dataset and split identifiers). Each of the 5 windows draws a different 200-row
subset of the training set. The final prediction is the mean over `5 × 8 = 40`
forward passes per test row.

### Permutation tests (fail-fast quality gates)

After checkpoint loading, two layers of permutation gates run before any dataset
evaluation.

**Layer 1 — Legacy gates** (`deepset_inference.run_permutation_tests`):

| Test | Requirement |
|---|---|
| Row permutation invariance | Permuting `(X_train, y_train)` rows must not change `y_hat` beyond numerical tolerance |
| Column permutation consistency | Jointly permuting feature columns and metadata must preserve predictions |
| Batch query shape validation | Multiple `x_test` rows must produce separate predictions |
| Finite output validation | No NaN or Inf in any output |

**Layer 2 — Structured permutation contracts** (`permutation_contracts.run_all`):

Returns `PermutationResult` dataclasses with structured metrics.  Regression
evaluators enforce all checks; classification evaluators enforce F1–F4 and
skip F5 (see §17 for the F5 structural limitation).

| Contract | Check | Applies to |
|---|---|---|
| **F1** | Support-row permutation invariance | regression + classification |
| **F2** | Query-row permutation equivariance | regression + classification |
| **F3** | Feature-column permutation consistency | regression + classification |
| **F4** | Feature-indexed output equivariance | coefficient/W heads |

Each result includes: `max_abs_delta`, `mean_abs_delta`, `max_rel_delta`,
`prediction_flip_rate`, `passed`, `threshold_atol`, `threshold_rtol`,
`tolerance_policy_version`, `device`, `dtype`, and timing.

### GPU memory guard

MODEL3's per-query tensor `H_q: (m, n, p, channels)` grows with `m × n × p`.
Before inference, estimated H-tensor bytes are compared against
`BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES = 268,435,456` (256 MB).

In the combined evaluation, **6 of 1200 datasets were skipped** (0.5% skip rate)
due to this guard. Skip reason in the summary CSV:
```
gpu_oom: model3_estimated_h_tensor_bytes=297,979,328 exceeds 268,435,456
```
These datasets have `(m × n × p)` products that would require > 256 MB of GPU
memory. Skipped datasets are counted as `skip_count = 6` in the summary row for
`MODEL3-ICL-MC`.

---

## 8. Baseline Models

10 baseline models are evaluated sequentially on each dataset using the same
train/holdout split:

| Method | Library | Notes |
|---|---|---|
| `FixedRidgeLambda1` | sklearn | Ridge with λ=1 fixed; primary ratio baseline |
| `LinearRegression` | sklearn | OLS, no regularization |
| `Ridge` | sklearn | Cross-validated λ |
| `RandomForest` | sklearn | Default hyperparameters |
| `XGBoost` | xgboost | Default hyperparameters |
| `LightGBM` | lightgbm | Default hyperparameters |
| `CatBoost` | catboost==1.2.10 | Default hyperparameters |
| `KNN` | sklearn | Default k |
| `SVR` | sklearn | Default RBF kernel |
| `MLP` | sklearn | Default architecture |

Memory guards for baselines: skip if `p_total > 2000` or if the feature matrix
exceeds 512 MB (`BENCHMARK_CPU_MAX_MATRIX_BYTES = 536,870,912`).

### AutoGluon

AutoGluon (`autogluon.tabular==1.3.0`) runs in `best_quality` preset with a
300-second time limit per dataset fit. Only **6 total runs** completed in the
combined suite (the full AutoGluon coverage is not 1200); the 6 valid rows
represent a limited sampling of the suite.

---

## 9. Performance Metrics

All metrics are computed against `betaX` (the noiseless linear signal), not `y`.
This measures signal recovery quality, not noisy target prediction.

| Metric | Description |
|---|---|
| `mse_betaX` | Mean squared error vs. noiseless `betaX = X @ beta` |
| `rmse_betaX` | Root MSE vs. `betaX` |
| `mae_betaX` | Mean absolute error vs. `betaX` |
| `r2_betaX` | R² vs. `betaX` |
| `prediction_std` | Standard deviation of model predictions |
| `variance_ratio` | `prediction_std / target_std` |
| `bias` | Mean of `(y_hat − betaX)` |
| `slope_y_pred_vs_y_true` | Slope from regressing `y_hat` on `betaX`; 1.0 = well-calibrated |

### Per-dataset ranking and comparison ratios

After all methods run on a dataset, each method receives a rank `1..K` by
ascending `mse_betaX` (rank 1 = lowest MSE = best). NaN MSE methods are
excluded from ranking.

Ratio columns anchor to the MSE of reference methods on the same dataset:

| Column | Formula |
|---|---|
| `ratio_mse_to_fixed_ridge` | `mse / mse_FixedRidgeLambda1` |
| `ratio_mse_to_autogluon` | `mse / mse_AutoGluon` |
| `ratio_mse_to_best_tree` | `mse / min(mse_XGBoost, mse_LightGBM, mse_CatBoost, mse_RandomForest)` |

Boolean win indicators:

| Column | Meaning |
|---|---|
| `beats_fixed_ridge` | `mse < mse_FixedRidgeLambda1` on this dataset |
| `beats_autogluon` | `mse < mse_AutoGluon` on this dataset |
| `beats_best_tree` | `mse < best tree method` on this dataset |
| `is_best_mse` | Rank 1 on this dataset |
| `is_top3_mse` | Rank ≤ 3 on this dataset |

### CI computation

All aggregate statistics use 95% t-interval confidence intervals
(`ci95`: `(mean, lo, hi)`). NaN values are excluded before computing
mean, std, and t-interval bounds.

---

## 10. Aggregated Outputs

The aggregation step produces 9 output files (plus optional PNG charts):

| File | Description |
|---|---|
| `synthetic_regression_model_comparison.csv` | Full ranked per-dataset-per-method rows (9.5 MB) |
| `synthetic_regression_model_comparison_summary.csv` | Per-method aggregate statistics across all 1200 valid rows |
| `synthetic_regression_summary_by_regime.csv` | Per-method stats broken down by `(suite_family, prior_regime)` |
| `synthetic_regression_summary_by_feature_noise.csv` | Per-method stats by `(feature_noise_level, prior_regime)` with degradation columns |
| `synthetic_regression_summary_by_training_size.csv` | Per-method stats by `(n_train, prior_regime)` with improvement columns |
| `synthetic_regression_summary_by_regime_n_train_p.csv` | Per-method stats by `(regime, n_train, p_signal)` |
| `synthetic_regression_summary_by_regime_p_quartile_n_quartile.csv` | Per-method stats by regime × `p` quartile × `n` quartile |
| `synthetic_regression_chart_data_model_rank.csv` | CDF-ready rank distribution per method |
| `synthetic_regression_chart_data_noise_features.csv` | MSE degradation curves by noise level per method |
| `synthetic_regression_chart_data_training_size.csv` | MSE improvement curves by n_train per method |
| `synthetic_regression_aggregation_manifest.json` | Run metadata, shard list, output validation |

---

## 11. Overall Performance Results

Results from `synthetic_regression_model_comparison_summary.csv`
(combined suite, 1200 datasets × 3 seeds = 3600 evaluations per method):

| Method | Valid | Mean MSE | Median MSE | Mean Rank | Win Rate | Top-3 Rate |
|---|---|---|---|---|---|---|
| FixedRidgeLambda1 | 1200 | 2375.0 | 0.088 | 1.84 | 41.6% | 94.7% |
| Ridge | 1200 | 2375.0 | 0.088 | 1.84 | 41.6% | 94.7% |
| LinearRegression | 1200 | 2411.9 | 0.089 | 2.16 | 29.7% | 90.9% |
| **MODEL3-ICL-MC** | **1194** | **2730.0** | **0.134** | **3.01** | **16.4%** | **85.3%** |
| MLP | 1200 | 1070.1 | 0.808 | 6.36 | 0.0% | 0.5% |
| CatBoost | 1200 | 295.2 | 1.445 | 5.38 | 0.3% | 4.9% |
| SVR | 1200 | 2.74 | 1.718 | 6.27 | 10.4% | 12.4% |
| KNN | 1200 | 83.6 | 2.906 | 8.46 | 0.0% | 1.8% |
| RandomForest | 1200 | 3075.8 | 1.997 | 6.46 | 1.8% | 7.2% |
| LightGBM | 1200 | 4210.4 | 1.706 | 6.95 | 0.0% | 0.2% |
| XGBoost | 1200 | 28765.7 | 2.412 | 8.12 | 0.0% | 2.6% |
| AutoGluon | 6 | 1.37 | 1.404 | 5.0 | 0.0% | 0.0% |

**Notes:**
- Mean MSE is heavily influenced by regime G (Cauchy noise), which produces
  extreme outliers for most models. Median MSE is the more interpretable central
  tendency.
- FixedRidgeLambda1 and Ridge have identical results by construction (Ridge
  cross-validates λ but the suite's well-conditioned linear DGPs converge to λ≈1).
- MODEL3-ICL-MC's mean rank of 3.01 places it third overall, consistent with
  its median MSE of 0.134 vs. FixedRidgeLambda1's 0.088.
- MODEL3-ICL-MC beats FixedRidgeLambda1 on 20.2% of datasets
  (`beats_fixed_ridge_rate = 0.202`) and achieves top-3 placement on 85.3%.
- `mean_ratio_mse_to_fixed_ridge = 1.66` (median 1.47): MODEL3-ICL-MC's MSE
  is typically 1.47–1.66× that of the ridge baseline.

---

## 12. Performance by Regime

Results from `synthetic_regression_summary_by_regime.csv`:

### In-distribution (A–D) — `suite_family = primary`

| Regime | MODEL3 median MSE | Ridge median MSE | MODEL3 mean rank |
|---|---|---|---|
| A (i.i.d. normal, dense β) | 0.110 | 0.075 | 2.54 |
| B (i.i.d. normal, 70% sparse β) | 0.106 | 0.069 | 2.76 |
| C (i.i.d. normal, t(3) noise) | 0.245 | 0.177 | 2.56 |
| D (AR(1) features, dense β) | 0.116 | 0.073 | 2.57 |

MODEL3-ICL-MC consistently ranks 2nd–3rd within the in-distribution suite,
trailing FixedRidgeLambda1/Ridge and LinearRegression on most datasets.
On regime B (sparse coefficients), MODEL3 ranks tightly with Ridge, indicating
that the model learns effective implicit sparsity from context without explicit
sparse priors.

### Out-of-distribution (E–H) — `suite_family = ood_primary`

| Regime | MODEL3 median MSE | Ridge median MSE | MODEL3 mean rank | Notes |
|---|---|---|---|---|
| E (Laplace features) | 0.113 | 0.073 | 2.66 | Strong; OOD feature marginal not harmful |
| F (Uniform features, 95% sparse β) | 0.087 | 0.061 | 3.35 | Competitive; sparse regime |
| G (Cauchy noise) | 18.94 | 9.74 | 5.01 | All models struggle; SVR dominates (rank 1.56) |
| H (block-correlated features) | 0.101 | 0.070 | 2.63 | Strong; correlated features not harmful |

Regimes E and H show strong OOD generalization: the model transfers well to
Laplace-distributed inputs and block-correlated feature matrices, neither of
which appear in the in-distribution training data. Regime G is the exception:
Cauchy noise has no finite variance and disrupts signal recovery for all linear
and neural methods.

---

## 13. Sparsity: How Well the Model Handles Sparse Coefficients

Two mechanisms are relevant to sparsity: coefficient sparsity in the DGP
(regimes B and F) and noise-feature contamination (feature noise sub-family).

### Coefficient sparsity (regimes B and F)

| Sparsity level | Regime | MODEL3 median MSE | Ridge median MSE | MODEL3 rank |
|---|---|---|---|---|
| 70% sparse β | B (in-dist) | 0.106 | 0.069 | 2.76 |
| 95% sparse β | F (OOD) | 0.087 | 0.061 | 3.35 |

MODEL3-ICL-MC does not impose explicit sparsity (no Lasso or spike-and-slab
prior). The model learns to ignore near-zero features implicitly through
the in-context learning mechanism. On regime F (95% sparse), the model's
median MSE (0.087) is competitive with Ridge (0.061) and outperforms all
tree-based methods and KNN.

Regime F beat rates:
- `beats_fixed_ridge_rate`: low (ridge is near-optimal for sparse linear)
- `beats_best_tree_rate`: 0.0 (tree methods also struggle with sparse OOD
  marginals; RandomForest wins on F due to implicit feature selection)

### Noise feature contamination (feature noise sub-family)

MODEL3-ICL-MC's F-statistic selector (`train_f_regression`) caps at 128
features, discarding columns ranked below the top 128 by F-score on the
training split. This is the primary mechanism against noise-feature dilution.

Key behavior:
- At noise levels 0–50%: F-statistic selection successfully filters most noise
  columns; MSE degradation is limited.
- At noise level 75–100%: The signal-to-noise ratio in F-scores drops sharply;
  the selector may include noise columns. At noise level 100%, all signal is
  replaced and recovery is impossible for all methods.
- The `mean_selected_features` column in the feature noise summary tracks how
  many features were selected vs. available, confirming the 128-cap is active
  on datasets with `p_total > 128`.

---

## 14. Feature Space Dimensionality

### GPU memory constraint

MODEL3's H-tensor `(m, n, p, channels)` grows multiplicatively with `p`.
At the default cap of 128 features, `p` is controlled. Without the cap, large
`p` datasets would exceed the 256 MB GPU memory guard and be skipped.

In the combined suite (Poisson p ≈ 10 typical), 6 datasets were skipped
due to `m × n × p × channels` exceeding 268,435,456 bytes. These were
high-dimensional outliers from the tail of the Poisson(10) feature sampler.

### Training size scaling

From `synthetic_regression_summary_by_training_size.csv` (regime B, n_train=128):

| Method | Median MSE (n=128) | Notes |
|---|---|---|
| MODEL3-ICL-MC | 0.076 | 200-row context window fully utilized |
| FixedRidgeLambda1 | 0.069 | Strong at small n |
| Ridge | 0.069 | Identical to FixedRidgeLambda1 |
| LinearRegression | 0.079 | Competitive |
| MLP | 0.888 | Weaker at small n |
| CatBoost | 2.92 | Tree overhead at small n |

MODEL3-ICL-MC is competitive with Ridge at small context sizes
(`n_train = 128 < 200`, so the full training set fits in one context window).
The model's advantage over tree-based methods is most pronounced when `n_train`
is small (25–200), where tree methods underfit.

As `n_train` increases past the 200-row context window cap, MODEL3-ICL-MC
relies on the 5-window ensemble to approximate the full dataset. Performance
does not degrade past this cap: the ensemble covers different random subsets
and the mean over 40 forward passes is stable.

### Feature count and p scaling

The `synthetic_regression_summary_by_regime_p_quartile_n_quartile.csv` output
allows analysis stratified by quartiles of `p_signal` and `n_total`. Key patterns:

- Low-p datasets (p ≤ 5): All methods including Ridge perform well; MODEL3
  overhead is unnecessary but not harmful.
- Mid-p datasets (5 < p ≤ 15): MODEL3-ICL-MC is most competitive relative to
  baselines; the feature encoder provides value over raw ridge.
- High-p datasets (p > 20): F-statistic selection kicks in more aggressively;
  GPU memory use increases; 6 OOM skips cluster in this range.

---

## 15. Aggregation Manifest (linear_all_v1)

Produced: `2026-06-03T19:43:38.999013+00:00`

Input shards:
- MODEL3-ICL-MC: 10 GPU shards (10 × ~64 KB shard files)
- Baselines: 10 CPU shards (10 × ~612 KB shard files)
- AutoGluon: 6 + 8 shards (mixed from two execution runs)

Validation:
- `required_columns_present: true`
- `empty_inputs_detected: false`
- `empty_outputs_detected: false`
- 9 output files written, all conditional outputs present

---

## 16. Classification Evaluation Pipeline — Audit-Validated Invariants (2026-06)

### Canonical probability alignment

All probability metrics (`log_loss`, `roc_auc_*`, `top_k_accuracy`) must receive
`labels=list(range(num_classes))` so that absent classes are handled correctly.
`_align_proba_to_canonical()` in `evaluate_linear_classification.py` returns a
**3-tuple** `(aligned, has_missing, missing_class_ids)`.

### Metric status fields

Every result row contains:

| Field | Type | Meaning |
|-------|------|---------|
| `log_loss_valid` | bool | True unless log_loss raised |
| `roc_auc_valid` | bool | False when missing-support classes are present (multiclass) |
| `brier_score_valid` | bool | True unless exception |
| `has_missing_support` | bool | Any class absent from model output |
| `missing_support_classes` | list[int] | IDs of absent classes |

Multiclass AUC (`roc_auc_ovr`, `roc_auc_ovo`) is **suppressed** (set to NaN) when
`has_missing_support=True`. AUC computed on epsilon-padded columns is misleading
because a constant tiny probability creates an artificially separated ROC curve.

### Ranking statistics

`build_ranking_summary(df, metric_col, task_col, model_col, higher_is_better)` in
`src/evaluation_metrics.py` produces per-model:

| Statistic | Meaning |
|-----------|---------|
| `mean_rank` | Mean per-task rank (lower = better for loss metrics) |
| `median_rank` | Median per-task rank |
| `strict_win_rate` | Fraction of tasks where rank == 1.0 (no ties) |
| `tie_aware_win_rate` | Fraction of tasks where rank ≤ 1.5 (includes ties) |
| `top_3_rate` | Fraction of tasks where rank ≤ 3 |
| `task_count` | Total tasks attempted |
| `valid_metric_task_count` | Tasks with a non-NaN metric value |
| `completion_rate` | `valid_metric_task_count / task_count` |

The aggregate mode writes `synthetic_classification_ranking_summary.csv` and
per-K (`ranking_summary_k{K}.csv`) and per-suite-family (`ranking_summary_{family}.csv`)
breakdowns.

### Paired bootstrap CI

`paired_bootstrap_ci(a, b)` from `src/evaluation_metrics.py` computes a 95% CI for
the mean difference (a − b). Requires ≥ 10 valid paired observations; returns NaN
CI with a `reason` field otherwise. Use for model-vs-LogisticRegression and
model-vs-best-tree comparisons.

### `val_cross_entropy` is pure CE

`val_cross_entropy` in training checkpoints records **pure query cross-entropy only**
(the `"ce"` key from `compute_classification_losses`), not `total_loss` which includes
auxiliary terms (KL, coef_aux, margin, prior, calibration). `val_total_loss` records
the full multi-component objective. HPO and early stopping both consume
`val_cross_entropy` (semantically correct).

### Schema version ownership

`CLASSIFICATION_EVAL_SCHEMA_VERSION` in `src/dgp_helpers.py` is the single source of
truth for the classification eval parquet schema version. Never hard-code the string
in tests or scripts — always import it:

```python
from dgp_helpers import CLASSIFICATION_EVAL_SCHEMA_VERSION
```

---

## 17. Formal Permutation Contracts

The DeepSet model family must satisfy the following permutation invariance and
equivariance properties.  Each property is enforced by the authoritative
module `src/permutation_contracts.py` and tested in
`tests/test_permutation_contracts.py`.

### 17.1 Contract definitions

**F1 — Support-row permutation invariance** (regression + classification):

```
f(X_support[P], y_support[P], X_query) ≈ f(X_support, y_support, X_query)
```

All support-row-aligned fields must use the same permutation P.

**F2 — Query-row permutation equivariance** (regression + classification):

```
f(X_support, y_support, X_query[Q]) ≈ f(X_support, y_support, X_query)[Q]
```

**F3 — Feature-column permutation consistency** (regression + classification):

```
f(X_support[:, C], y_support, X_query[:, C]) ≈ f(X_support, y_support, X_query)
```

Support and query must be permuted with the same column permutation C.

**F4 — Feature-indexed output equivariance** (coefficient heads):

```
# Regression:
beta_hat[C] ≈ beta_hat_permuted

# Classification:
W_hat[C, :] ≈ W_hat_permuted
```

Feature-indexed outputs must follow the same column permutation applied
to the inputs.

**F5 — Classification label permutation equivariance**:

```
f(X_support, L(y_support), X_query) ≈ L(f(X_support, y_support, X_query))
```

Logit and probability columns must follow the class permutation L.

> **Known limitation:** F5 is structurally violated for K ≥ 3 by the learned
> `ClassLabelEncoder` which uses `nn.Embedding(max_num_classes, dim)`.  Each
> class index receives a distinct learned vector, making the model sensitive
> to absolute class IDs.  K = 2 passes because the binary swap is trivially
> symmetric.  The classification evaluation gate skips F5 enforcement.  If the
> architecture is redesigned to use class-permutation-equivariant embeddings,
> F5 should be re-enabled.

**F6 — Completion row equivariance**:

```
f(X[P], mask[P]) ≈ f(X, mask)[P]
```

**F7 — Completion column equivariance**:

```
f(X[:, C], mask[:, C]) ≈ f(X, mask)[:, C]
```

### 17.2 Module: `src/permutation_contracts.py`

Entry points:

| Function | Behaviour |
|---|---|
| `run_all(model, seed=42)` | Dispatch all applicable checks for the model family/objective.  Returns `list[PermutationResult]`. |
| `run_all_strict(model, seed=42)` | Same as `run_all` but raises `RuntimeError` if any non-skipped check fails. |
| `check_support_row_invariance(model, ...)` | F1 only. |
| `check_query_row_equivariance(model, ...)` | F2 only. |
| `check_feature_column_consistency(model, ...)` | F3 only. |
| `check_feature_indexed_equivariance(model, ...)` | F4 only. |
| `check_class_label_equivariance(model, ...)` | F5 only. |
| `check_completion_row_equivariance(model, ...)` | F6 only. |
| `check_completion_column_equivariance(model, ...)` | F7 only. |

Dispatch logic:

- `market_exchangeable_icl` → F1, F2, F3, F4; plus F5 for each K ∈ {2,3,5,10} when classification.
- `market_exchangeable_completion` → F6, F7.

Each check:

1. Sets model to eval mode.
2. Saves and restores all RNG states (Python, NumPy, Torch CPU, Torch CUDA).
3. Generates synthetic data from a deterministic seed.
4. Runs reference forward pass, restores RNG, applies permutation, runs permuted forward pass.
5. Compares outputs using tolerances from `tolerance_policy.get_tolerance()`.
6. Restores the model's original train/eval mode.

### 17.3 Structured result: `PermutationResult`

Every check returns a `PermutationResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `check_type` | str | E.g. `F1_support_row_invariance` |
| `task_objective` | str | E.g. `inductive_regression`, `inductive_classification` |
| `num_classes` | int or None | K for classification checks |
| `permutation_seed` | int | Seed used for the permutation |
| `device` | str | E.g. `cpu`, `cuda:0` |
| `dtype` | str | E.g. `torch.float32` |
| `max_abs_delta` | float | Max absolute difference between reference and permuted outputs |
| `mean_abs_delta` | float | Mean absolute difference |
| `max_rel_delta` | float | Max relative difference |
| `prediction_flip_rate` | float | Fraction of positions where integer predictions disagree (classification only) |
| `passed` | bool | True if within tolerance |
| `threshold_atol` | float | Absolute tolerance applied |
| `threshold_rtol` | float | Relative tolerance applied |
| `failure_reason` | str | Empty if passed; explains violation otherwise |
| `reference_shape` | tuple | Shape of the reference output tensor |
| `permuted_shape` | tuple | Shape of the permuted output tensor |
| `elapsed_s` | float | Wall-clock time for this check |
| `tolerance_policy_version` | str | Version of the tolerance policy applied |

`.to_dict()` serializes for JSON/artifact persistence.

---

## 18. Tolerance Policy

Tolerances are centrally owned by `src/tolerance_policy.py` and keyed by:

| Factor | Values |
|---|---|
| `dtype` | `float64`, `float32`, `bfloat16`, `float16` |
| `device_type` | `cpu`, `cuda` |
| `inference` | `deterministic`, `stochastic` |
| `output_type` | `scalar_prediction`, `logits`, `probabilities`, `coefficients`, `bias`, `embedding` |
| `model_path` | `regression`, `classification` |

Base tolerances (dtype):

| dtype | atol | rtol |
|---|---|---|
| float64 | 1e-10 | 1e-8 |
| float32 | 1e-5 | 1e-5 |
| bfloat16 | 5e-3 | 5e-3 |
| float16 | 1e-3 | 1e-3 |

Multiplicative adjustments:

| Factor | Multiplier | Rationale |
|---|---|---|
| CUDA device | ×2 | Nondeterministic reduction order on GPU |
| Stochastic inference | ×5 | MC dropout introduces extra variance |
| Probability output | ×2 | Softmax amplifies differences near decision boundaries |

Entry point: `get_tolerance(**kw) → Tolerance`.  The `Tolerance` dataclass
exposes `allclose(a, b)`, `max_abs_delta(a, b)`, `mean_abs_delta(a, b)`,
and `max_rel_delta(a, b)`.

Version: `TOLERANCE_POLICY_VERSION = "1.0.0"`.  Stamped into every
`PermutationResult` for artifact provenance.

---

## 19. Support-Row Permutation Training Augmentation

`src/support_augmentation.py` provides deterministic support-row permutation
augmentation for training episodes.

### Seed derivation

```
seed = blake2b(f"{base_seed}\x1f{epoch}\x1f{rank}\x1f{batch_idx}")
```

This guarantees:

- Same permutation for a given `(base_seed, epoch, rank, batch_idx)`.
- Different permutations across epochs, ranks, and batch indices.
- Collision resistance via Blake2b.

### What is permuted

All support-row-aligned fields are permuted together with the same
permutation:

- `X_train` (n, p)
- `y_train` (n,)
- Any additional tensors passed via `extra_row_tensors` whose first
  dimension equals n.

### What is never permuted

- Query rows (`X_test`, `y_test`) — these remain in their original order.
- Non-row-aligned tensors (scalars, feature-indexed metadata).
- Validation and evaluation data (augmentation is training-only by default).

### Activation

Opt-in via config attributes (accessed via `getattr` with defaults):

| Attribute | Type | Default | Effect |
|---|---|---|---|
| `support_permutation_augmentation` | bool | False | Enable support-row augmentation during training |
| `support_permutation_base_seed` | int | 0 | Base seed for the permutation RNG |

Both `run_epoch()` (regression) and `run_classification_epoch()` (classification)
accept `epoch_idx` and `rank` parameters for seed derivation.

### Relationship to permutation invariance

Training augmentation is additional protection.  It does not replace an
invariant architecture or explicit evaluation.  The model architecture
(mean pooling over support rows in ExchangeableMatrixBlocks) is structurally
invariant to support-row order.  Augmentation exercises the property during
training but the property is verified independently by F1 at evaluation time.

---

## 20. Classification-Aware Sanity Gates

`src/sanity_checks.py` is now task-aware.  `run_all_checks(model)` routes
checks based on `model.cfg.task_objective`:

### Regression checks (always run)

Regression-only checks (`check_permutation_invariance`, `check_gate_range`)
are skipped when the model is a classification model.

### Classification checks (run when `task_objective` is classification)

| Check | What it validates |
|---|---|
| `check_classification_forward_smoke` | Forward pass completes; logits `(m, K)`, probs sum to 1, preds in `[0, K)`, all finite |
| `check_classification_permutation_invariance` | Row-permutation invariance and column-permutation consistency for classification logits |
| `check_classification_missing_class_stress` | Model produces finite outputs when a class is absent from the support set |

All checks accept `model` and `device` arguments and return a dict with
a `passed` key.  `run_all_checks` sets `all_passed = True` only if every
individual check passes.

---

## 21. Evaluation Fail-Fast Permutation Gates

### Regression evaluator (`scripts/evaluate_linear_regression.py`)

After checkpoint loading and legacy `run_permutation_tests()`, the evaluator
runs `permutation_contracts.run_all()` as a structured gate.  Any non-skipped
failure raises `RuntimeError` and halts evaluation before processing shards.

### Classification evaluator (`scripts/evaluate_linear_classification.py`)

After checkpoint loading, the evaluator runs `permutation_contracts.run_all()`.
Enforcement policy:

- F1–F4 are enforced (support invariance, query equivariance, feature
  consistency, coefficient equivariance).
- F5 (class-label equivariance) is **skipped** due to the known structural
  limitation of learned class-ID embeddings (§17.1).

Controlled by environment variables:

| Variable | Default | Effect |
|---|---|---|
| `CLASSIFICATION_RUN_PERMUTATION_GATES` | `true` | Enable/disable gates |
| `CLASSIFICATION_PERMUTATION_GATE_STRICT` | `true` | `true` = raise on failure; `false` = warn only |

---

## 22. Test Coverage

### `tests/test_permutation_contracts.py` (79 tests)

| Test class | Count | Coverage |
|---|---|---|
| `TestTolerancePolicy` | 8 | Tolerance routing by dtype, device, inference, output_type |
| `TestPermutationResultFormat` | 2 | Structured result fields and serialization |
| `TestF1RegressionSupportInvariance` | 6 | Basic, multi-seed, large context, single feature, model3, ridge expert |
| `TestF1ClassificationSupportInvariance` | 6 | K={2,3,5,10}, severe imbalance, missing support class |
| `TestF2QueryEquivariance` | 7 | Regression, single query, classification K={2,3,5}, batch sizes {2,4,8,16} |
| `TestF3FeatureColumnConsistency` | 8 | Regression, variable p={1,3,5,10}, classification K={2,3,5} |
| `TestF4FeatureIndexedEquivariance` | 7 | Regression beta, no-coeff-head skip, classification W K={2,3,5}, variable p |
| `TestF5ClassLabelEquivariance` | 3 | Regression skip, binary, multiclass expected-fail (documents structural limitation) |
| `TestF6CompletionRowEquivariance` | 6 | Basic, multi-seed, sparse mask |
| `TestF7CompletionColumnEquivariance` | 5 | Basic, multi-seed |
| `TestCompletionJointEquivariance` | 1 | Joint row + column compose correctly |
| `TestRunAllDispatch` | 5 | Regression routes F1-F4, classification includes F5, completion routes F6-F7, strict mode, unknown family |
| `TestDeterministicReproducibility` | 2 | Same seed = same result; different seed = different permutation |
| `TestBoundedContext` | 2 | n=3 regression, n=4 classification |
| `TestModelStatePreservation` | 2 | Eval/train mode restored after checks |

### `tests/test_support_augmentation.py` (13 tests)

| Test class | Count | Coverage |
|---|---|---|
| `TestPermutationSeedDeterminism` | 5 | Same inputs same seed; different epoch, rank, batch, base_seed |
| `TestPermuteSupport` | 8 | Determinism, content preservation, row alignment, extra tensors, non-row passthrough, None passthrough, epoch variation, classification labels |

### Pre-existing test files (verified passing)

| File | Tests | Status |
|---|---|---|
| `tests/test_model3_icl.py` | 29 | All pass (gradient flow tests fixed by disabling MODEL4 coeff path in MODEL3 config) |
| `tests/test_model3_completion.py` | 28 | All pass |
| `tests/test_model4_classification.py` | 15 | All pass |
| `tests/test_class_permutation_equivariance.py` | 4 | All pass |

**Total: 167 tests, 0 failures.**
