---
name: linear-synthetic-data
description: Design, generate, audit, and validate task-level linear synthetic regression or classification datasets for DeepSet meta-learning. Use when working with src/generate_dgp.py, src/dgp_helpers.py, linear DGP task-family routing, Gaussian dense or sparse coefficients, correlated features, high-dimensional tasks, irrelevant features, feature noise, regression target noise, classification logits and probabilities, class imbalance, margin, label noise, Parquet schemas, suite manifests, or DeepSet training-data contracts.
---

# Linear Synthetic Data

Use this skill to preserve a common synthetic-data philosophy while keeping
linear regression and linear classification as explicit, separate task
families.

## Route by Task Family

Select one path before designing, generating, or auditing data:

```text
task_family = "linear_regression"      -> implemented; training and eval pipelines active
task_family = "linear_classification"  -> implemented; eval pipeline active
```

- Read [references/linear-regression.md](references/linear-regression.md) for
  the regression DGP, A-L regimes, Parquet contract, commands, and
  validation workflow.
- Read
  [references/linear-classification.md](references/linear-classification.md)
  for the classification DGP, A-L regimes, schema, target semantics,
  teachers, validation, and suite families.

Do not combine the two schemas. In particular:

- Use `betaX_*`, target noise, and OLS/ridge teachers only for regression.
- Use `logits_*`, `probs_*`, label noise, and classification teachers only for
  classification.
- Preserve `X_train`, `y_train`, `X_test`, `y_test` as shared episode-level
  concepts.

## Preserve Shared Principles

Apply these rules to both paths:

1. Make each Parquet file one independent meta-learning task.
2. Keep outer train/validation/test splits at the task-file level.
3. Use the first 80 percent of task rows as context and the remaining 20
   percent as query unless a versioned schema explicitly changes this policy.
4. Generate latent targets from `X_clean`; expose `X_observed` when feature
   measurement noise is enabled.
5. Keep the latent function linear in clean features.
6. Use regimes to isolate statistical capabilities and failure modes.
7. Record sufficient task metadata and a mandatory suite-level manifest.
8. Make generation deterministic under a fixed command, git revision, and
   base seed.
9. Audit realized coverage because profile weights are probabilities rather
   than quotas.
10. Fit preprocessing on context rows only and preserve task boundaries.

## Maintain Source Ownership

Keep responsibilities conceptually separated:

- `src/generate_dgp.py`: own CLI and task-family routing, task sampling, outer
  and inner splits, validation orchestration, Parquet dispatch, and manifest
  writing.
- `src/dgp_helpers.py`: own shared feature distributions, covariance,
  coefficient generation, regimes, noise mechanisms, diagnostics, teachers,
  validation helpers, and serialization helpers.
- Task-family-specific helpers: make regression and classification profiles,
  targets, teachers, schemas, and validation explicit without duplicating
  shared feature logic.

Do not replace or reinterpret the regression path when adding classification.

---

## Linear Regression Generator

**Script:** `scripts/generate_synthetic_regression.py`
**DGP helpers:** `src/dgp_helpers.py`
**Eval writer:** `write_parquet_eval()`
**Training writer:** `write_parquet_dgp()`
**Output root:** `data/synthetic_regression_prepared/{suite_id}/`

### Profiles and Regimes

Four regression profiles are implemented:

| Profile | Description |
|---------|-------------|
| `legacy` | Original A-D mixture with Poisson n/p sampling |
| `linear_stat_aware` | Broad A-L mixture, recommended default |
| `linear_stress` | Equal weight on difficult statistical regimes |
| `market_linear` | Emphasises L_market_linear with structured signs |

The `linear_stat_aware` A-L regime catalog (12 regimes, weights sum to 1.0
excluding J when underdetermined is disabled):

| Regime | Weight | Capability |
|--------|-------:|-----------|
| `A_iid_dense` | 0.15 | Baseline dense recovery |
| `B_iid_sparse` | 0.15 | Sparse signal recovery |
| `C_heavy_tail_noise` | 0.10 | Student-t residual robustness |
| `D_correlated_ar` | 0.10 | AR(1) multicollinearity |
| `E_high_dim_dense` | 0.05 | Dense larger feature spaces |
| `F_high_dim_sparse` | 0.05 | Sparse larger feature spaces |
| `G_noise_features` | 0.10 | Irrelevant-column resistance |
| `H_block_correlated` | 0.05 | Grouped multicollinearity |
| `I_equicorrelated` | 0.05 | Global correlation stress |
| `J_low_n_high_p` | 0.05 | Underdetermined sparse inference (opt-in) |
| `K_feature_noise` | 0.05 | Errors-in-variables robustness |
| `L_market_linear` | 0.10 | Structured coefficient signs |

### Suite Families

The generator produces up to 11 suite families, controlled by
`--include_<family>` / `--no-include_<family>` flags. For the
`linear_stat_aware` profile all families are enabled by default:

| Family | Variation axis | Regimes used |
|--------|----------------|--------------|
| `primary` | Regime, n/p mixture | Regular profile regimes |
| `feature_noise` | Sweep `feature_noise_grid` | Regular profile regimes |
| `target_noise` | Sweep `target_noise_grid` | Regular profile regimes |
| `training_size` | Sweep `n_grid` | Regular profile regimes |
| `sparsity` | Sweep `active_s_grid` | Regular profile regimes |
| `correlation` | Sweep `rho_grid` | Regular profile regimes |
| `dimensionality` | Sweep `p_signal_grid` | Regular profile regimes |
| `ood` | Fixed | `REGRESSION_EVAL_ONLY_REGIMES` |
| `eval_only_unseen` | Fixed | `REGRESSION_EVAL_ONLY_REGIMES` |
| `hidden_holdout` | Fixed; separate seed namespace | `REGRESSION_EVAL_ONLY_REGIMES` |
| `stress` | Fixed; balanced | `linear_stress` profile regimes |

`hidden_holdout` uses `--hidden_holdout_base_seed` (default `20260607`) and a
separate seed magic constant (`_REG_HIDDEN_SEED_MAGIC = 0x48DDE771`) so its
seeds never overlap with normal families. The hidden suite ID is stored
separately via `--hidden_holdout_suite_id` and must differ from `--suite_id`.

Families `ood`, `eval_only_unseen`, `hidden_holdout`, and `stress` are
flagged `is_ood = True` in every dataset record.

### Parameter Grids (default)

```text
n_grid:               32, 64, 128, 256, 512, 1024
p_signal_grid:        4, 8, 16, 32, 64
p_noise_grid:         0, 8, 24, 56, 120
active_s_grid:        2, 4, 8, 16, 32
rho_grid:             -0.6, 0.0, 0.3, 0.6, 0.9
target_noise_grid:    0.25, 0.5, 1.0, 2.0
feature_noise_grid:   0.0, 0.05, 0.10, 0.25
```

### Allocation Modes

`--allocation_mode` controls how `n_datasets` tasks are distributed across
regimes. Valid choices (from `ALLOCATION_MODES`): `balanced`, `weighted`,
`quota`, and variants. Default is `balanced`.

### Parquet Contract (`write_parquet_eval`)

Evaluation-suite format (one row per task):

**Locked columns (always written, exact types):**

| Column | Type | Notes |
|--------|------|-------|
| `X` | `list<list<float64>>` | Full observed feature matrix |
| `y` | `list<float64>` | Noisy response vector |
| `betaX` | `list<float64>` | Noiseless linear signal |
| `suite_family` | `utf8` | One of the 11 family names above |
| `prior_regime` | `utf8` | Regime name |
| `n_total` | `int64` | Total rows |
| `p_signal` | `int64` | Signal columns |
| `p_noise` | `int64` | Appended noise columns |
| `p_total` | `int64` | `p_signal + p_noise` |
| `target_noise_scale` | `float64` | Residual noise scale |
| `training_size_anchor` | `bool` | `True` only at the `n_train=4832` anchor point |
| `feature_noise_level` | `float64` | Measurement noise (REG-S1: stored as float64) |

**Always-present additive columns:** `feature_noise_level_float`, `profile`,
`regime_group`, `active_s`, `sparsity_ratio`, `covariance_type`, `rho`,
`target_noise_type`, `has_noise_features`, `has_feature_noise`,
`sample_complexity_bucket`, `condition_number`, `matrix_rank`, `effective_rank`,
`snr`, `p_over_n`, `n_over_p`, `oracle_ridge_lambda`, `ridge_oracle_mse`,
`ols_or_min_norm_mse`, `teacher_available`.

**Extra metadata columns** (written via `extra_metadata`):
`global_idx`, `dataset_seed`, `n_train_default`, `n_holdout_default`,
`is_training_allowed`, `is_eval_only`, `is_ood`, `is_hidden_holdout`,
`task_fingerprint`, `difficulty_tier`, `difficulty_score`, `memory_class`,
and components from `compute_difficulty_metadata`.

**Conditional columns:** `beta`, `support_mask`, `beta_signal`, `beta_noise`
(with `--store_beta`); `ridge_pred_lambda_*_test` (with `--store_teacher_preds`);
`Sxx`, `Sxy` (with `--store_linear_moments`).

### Teacher Targets

`compute_linear_teacher_targets(X_train, y_train, X_test, y_test)` fits OLS
(min-norm) and ridge regression at lambdas `[0.0, 0.01, 0.1, 1.0, 10.0]`.

> **ORACLE WARNING:** `oracle_ridge_lambda` is selected by minimising MSE on
> `y_test` (query labels). This is a DGP diagnostic baseline. It **must not**
> enter model training, HPO, or checkpoint selection.

### Seed Derivation

Each dataset seed is derived independently from the family index:

```python
base = hidden_holdout_base_seed if family == "hidden_holdout" else base_seed
magic = _REG_HIDDEN_SEED_MAGIC if family == "hidden_holdout" else _REG_SEED_MAGIC
seed = SeedSequence([base, _FAMILY_MAGIC[family], family_idx, magic])
```

`_FAMILY_MAGIC[family]` is the first 4 bytes of `SHA-256(family.encode())`,
interpreted as a little-endian integer. This ensures seeds are unique across
families and individually reproducible without re-running the full suite.

### Suite Manifest

The JSON manifest at `{suite_id}/synthetic_regression_manifest.json` records:

- `schema_version` (`REGRESSION_EVAL_SCHEMA_VERSION`; `"linear_regression_eval_v1"` with `--legacy_schema`)
- `task_family: "linear_regression"`, `profile`, `base_seed`, `suite_id`
- `hidden_holdout_suite_id`, `hidden_holdout_base_seed`
- `allocation_mode`, `enabled_suite_families`, `realized_suite_family_counts`
- `n_datasets_by_regime_group`
- `grid_metadata` with all grid values
- `generation_controls`: `allocation_mode`, `curriculum_policy`, `difficulty_mix`,
  `memory_guard_bytes`, `min_regime_count`
- `seed_audit`: `all_dataset_seeds_unique`, `hidden_normal_seed_overlap`
- `coverage_audit`: per-axis counts and missing-coverage flags
- `memory_audit`: counts by `memory_class`
- `difficulty_audit`: counts by `difficulty_tier`
- `train_eval_alignment_audit`: seed overlap, fingerprint overlap, eval-only
  regime leakage checks (populated when `--training_manifest` is provided)
- `alignment_audit`: populated when `--emit_alignment_report` is provided
- Per-dataset entries with `dataset_id`, `suite_id`, `suite_family`,
  `dataset_seed`, `task_fingerprint`, `prior_regime`, `n_total`, `p_signal`,
  `p_noise`, `p_total`, `active_s`, `sparsity_ratio`, `covariance_type`, `rho`,
  `target_noise_scale`, `target_noise_type`, `feature_noise_level`,
  `distribution_family`, `training_size_anchor`, `payload_bytes`,
  `stage_path`, `split_seeds`, `is_training_allowed`, `is_eval_only`,
  `is_ood`, `is_hidden_holdout`, plus all difficulty and memory fields

### Regression Generator Changes (Audit)

| ID | Change | Location |
|----|--------|----------|
| REG-S1 | `feature_noise_level` stored as `float64` instead of `int64`; fractional values (0.05, 0.10, 0.25) are now preserved without truncation | `dgp_helpers.py` `write_parquet_eval` and `write_parquet_dgp` |
| REG-S1b | `feature_noise_level` loader cast changed from `int()` to `float()` | `evaluate_synthetic_regression.py` |
| REG-C1 | `best_ridge_lambda` renamed to `oracle_ridge_lambda` throughout; explicit oracle-diagnostic warning added to `compute_linear_teacher_targets` docstring | `dgp_helpers.py` |
| REG-R1 | `dataset_seed` now derived via a family-scoped `SeedSequence` instead of sampling from the main generation RNG; datasets are individually reproducible | `generate_synthetic_regression.py` |
| REG-M1 | Manifest now includes `"schema_version"` at top-level; legacy schema available via `--legacy_schema` | `generate_synthetic_regression.py` |

---

## Linear Classification Generator

**Script:** `scripts/generate_synthetic_classification.py`
**DGP helpers:** `src/dgp_helpers.py`
**Training writer:** `write_classification_parquet()` — wide format, one row per task
**Eval writer:** `write_parquet_classification_eval()` — per-row format, one row per sample
**Output root:** `data/synthetic_classification_prepared/{suite_id}/`

### Profiles and Regimes

Four classification profiles are implemented:

| Profile | Description |
|---------|-------------|
| `linear_classification_stat_aware` | Broad A-L mixture, recommended default |
| `linear_classification_stress` | Emphasises overlap, high-dimension, and noise |
| `market_classification` | Emphasises L_market_sign and feature-noise regimes |
| `classification_legacy_debug` | A-D only, for debugging |

The `linear_classification_stat_aware` A-L classification regime catalog:

| Regime | Weight | Capability |
|--------|-------:|-----------|
| `A_iid_dense_logistic` | 0.15 | Baseline binary dense classification |
| `B_iid_sparse_logistic` | 0.15 | Sparse discriminative recovery |
| `C_label_noise_margin` | 0.10 | Overlap and label-noise robustness |
| `D_correlated_ar_logistic` | 0.10 | Classification under multicollinearity |
| `E_high_dim_dense_softmax` | 0.05 | Dense high-dimensional multiclass |
| `F_high_dim_sparse_softmax` | 0.05 | Sparse high-dimensional multiclass |
| `G_noise_features_classification` | 0.10 | Irrelevant-column resistance |
| `H_block_correlated_classification` | 0.05 | Grouped discriminative structure |
| `I_equicorrelated_classification` | 0.05 | Near-collinearity stress |
| `J_low_n_high_p_classification` | 0.05 | Underdetermined classification (opt-in) |
| `K_feature_noise_classification` | 0.05 | Errors-in-variables classification |
| `L_market_sign_classification` | 0.10 | Structured coefficient signs |

### Parameter Grids (default)

**Shared with regression:**
```text
n_grid:               100, 200, 500, 1000, 2000, 5000
p_signal_grid:        2, 5, 10, 20, 50, 100
p_noise_grid:         0, 5, 10, 25, 50
active_s_grid:        2, 4, 8, 16, 32
rho_grid:             0.0, 0.3, 0.6, 0.9
feature_noise_grid:   0.0, 0.05, 0.10, 0.25
```

`--feature_noise_amplitude_grid` overrides `--feature_noise_grid` when set.

**Classification-specific:**
```text
num_classes_grid:          2, 3, 5, 10
temperature_grid:          0.5, 1.0, 2.0, 5.0
label_noise_grid:          0.0, 0.02, 0.05, 0.10, 0.20
class_imbalance_grid:      balanced, mild, moderate, severe
margin_grid:               low, medium, high
coefficient_scale_grid:    0.5, 1.0, 2.0, 5.0
intercept_scale_grid:      0.0, 0.5, 1.0
```

**Class count prior:** K=2 (0.50), K=3 (0.25), K=5 (0.15), K=10 (0.10).

**Imbalance strengths:** balanced (0.00), mild (0.15), moderate (0.35),
severe (0.60). Calibrate intercepts against clean logits to approximate the
target prior; do not resample labels. **SMOTE is explicitly rejected** — class
imbalance is an intentional DGP regime; `inverse_frequency_class_weight` provides
the training-side correction. SMOTE would corrupt class priors, covariance structure,
and teacher coefficients; it also fails for rare classes in stress regimes (count <
k_neighbors+1). Do not add `imbalanced-learn` or any SMOTE variant.

**Margin buckets:** low < 0.75, 0.75 ≤ medium ≤ 2.00, high > 2.00
(temperature-normalised median logit gap).

### DGP Formula

Binary:
```text
z = X_clean @ w_true + b_true
binary_logits = [0, z]              # reference-class parameterisation
probs = softmax(binary_logits / T)
y_clean ~ Categorical(probs)
y = apply_symmetric_label_noise(y_clean)
X = X_clean + optional_feature_noise
```

Multiclass (K classes):
```text
logits = X_clean @ W_true + b_true  # W_true: (p, K), col 0 = zeros (reference)
probs = softmax(logits / T)
y_clean ~ Categorical(probs)
y = apply_symmetric_label_noise(y_clean)
X = X_clean + optional_feature_noise
```

### Label Semantics

- `y_clean`: DGP-optimal label before symmetric noise (`Categorical(probs)`).
- `y`: observed label after `apply_symmetric_label_noise(y_clean)`. Used for training and evaluation.
- Teacher coefficients (`W_true`, `b_true`), `logits`, `probs`, and `class_prior` are always
  derived from the **clean** DGP. Do not overwrite them with values computed from `y`.
- The `label_noise_mask` records which samples were flipped; count equals
  `round(label_noise_rate * n)` exactly.
- `y_clean_train` / `y_clean_test` are stored in training Parquet files for auxiliary loss
  computation. `y_train` / `y_test` are the observed (possibly noisy) labels consumed by the model.

### Coefficient Patterns

| Pattern | Regimes |
|---------|---------|
| `dense` | A, E, and general |
| `sparse` | B, F, and general |
| `decaying` | general |
| `group_sparse` | H block-correlated |
| `market_sign` | L market-sign |

`W_true` is always `(p_total, K)` for all `K`, including binary. Column 0 is
the reference class (all zeros). `b_true` is always `(K,)` with `b_true[0] = 0`.
A deprecated `w_true = W_true[:, 1]` alias is kept for backward compatibility.

### Suite Families

The generator produces up to 11 suite families:

| Family | Variation |
|--------|-----------|
| `primary` | Regime, class count, and imbalance mixture |
| `feature_noise` | Sweep over `feature_noise_grid` (amplitude), per-level subdirectory |
| `label_noise` | Sweep over `label_noise_grid` (rates: 0.0, 0.02, 0.05, 0.10, 0.20) |
| `training_size` | Sweep over `n_train` values (25, 50, 100, 200, 500, 1000, 2000, 4832); fixed `n_holdout = 1371` |
| `class_imbalance` | Full grid of K × imbalance level cells |
| `margin` | Sweep over `margin_grid` |
| `num_classes` | Sweep over `num_classes_grid` |
| `ood` | Six fixed OOD scenarios (see below) |
| `eval_only_unseen` | `CLASSIFICATION_EVAL_ONLY_REGIMES`; balanced allocation |
| `hidden_holdout` | `CLASSIFICATION_EVAL_ONLY_REGIMES`; separate seed namespace |
| `stress` | `linear_classification_stress` profile; balanced allocation |

`hidden_holdout` uses `--hidden_holdout_base_seed` (default `20260607`) and
`_HIDDEN_EVAL_SEED_MAGIC = 0xC1A55EED`. Its seeds never overlap with normal
families. The `training_size` anchor point (`n_train = 4832`) is flagged
`is_tabpfn_anchor = True`.

#### OOD Scenarios

The `ood` family generates datasets for six fixed scenarios:

| Scenario name | Regime | K | Imbalance | Margin | Label noise | Notes |
|---------------|--------|---|-----------|--------|-------------|-------|
| `heavy_tailed` | `C_label_noise_margin` | 2 | balanced | low | 0.10 | temperature overridden to [4.0, 5.0] |
| `bounded` | `A_iid_dense_logistic` | 2 | balanced | medium | 0.0 | |
| `equicorrelated` | `I_equicorrelated_classification` | 3 | mild | medium | 0.0 | |
| `high_dim_sparse` | `F_high_dim_sparse_softmax` | 5 | balanced | high | 0.0 | |
| `severe_imbalance` | `G_noise_features_classification` | 2 | severe | medium | 0.0 | |
| `high_noise` | `K_feature_noise_classification` | 2 | balanced | medium | 0.20 | feature_noise_grid overridden to [50.0, 75.0, 100.0] |

Each scenario generates `n_datasets_per_sweep` tasks in a subdirectory named
after the scenario.

### Seed Derivation

Each eval task seed is derived independently from:

```python
magic = _HIDDEN_EVAL_SEED_MAGIC if suite_family == "hidden_holdout" else _EVAL_SEED_MAGIC
base_seed = hidden_holdout_base_seed if suite_family == "hidden_holdout" else base_seed
seed = SeedSequence([base_seed, global_idx, magic]).generate_state(1)[0]
```

`_EVAL_SEED_MAGIC = 0xECA1C1A5`, `_HIDDEN_EVAL_SEED_MAGIC = 0xC1A55EED`.
Seeds are recorded in the manifest per-dataset entry as `task_seed`.

### Parquet Formats

**Training format (`write_classification_parquet`)** — one row per task, wide
nested-array layout matching the regression training contract:

Required columns: `X_train`, `y_train`, `X_test`, `y_test`, `y_clean_train`,
`y_clean_test`, `label_noise_mask_train`, `label_noise_mask_test`,
`logits_train`, `logits_test`, `probs_train`, `dgp_teacher_probs_test`,
`n`, `p`, `n_train`, `n_test`, `p_signal`, `p_noise`, `p_total`,
`num_classes`, `realized_num_classes`, `schema_version`, `task_family`,
`task_objective`, `default_target_mode`, `prior_regime`,
`classification_regime`, `coefficient_regime`, `class_imbalance_type`,
`margin_level`, `realized_margin_level`, `class_prior`, `realized_class_prior`,
`train_class_counts`, `test_class_counts`.

Scalars: `active_s`, `sparsity_ratio`, `class_sparsity_ratio`,
`covariance_type`, `rho`, `block_size`, `temperature`, `label_noise_rate`,
`realized_label_noise_rate`, `coefficient_scale`, `intercept_scale`,
`feature_noise_level` (float64), `feature_noise_level_float`,
`teacher_available`, `teacher_type`, `teacher_failure_reason`.

Diagnostics: `condition_number`, `effective_rank`, `mean_margin`,
`median_margin`, `min_margin`, `class_entropy`, `minority_class_fraction`,
`majority_class_fraction`, `bayes_error_proxy`, `matrix_rank`,
`min_class_count`, `max_class_count`.

Conditional (with `--store_class_params`): `W_true` `(p, K)`, `b_true` `(K,)`,
`w_true` (deprecated alias for `W_true[:, 1]`), `active_support`,
`class_active_support`.

Schema metadata key: `coeff_schema_version = "canonical_v1"`.

**Eval format (`write_parquet_classification_eval`)** — one row per sample
(context + query concatenated), per-row layout:

Required columns: `feature_vector`, `label`, `split` (`"context"` or
`"query"`), `schema_version`, `suite_id`, `suite_family`, `dataset_idx`,
`global_idx`, `profile`, `regime`, `task_objective`, `task_family`, `n_train`,
`n_test`, `n_features`, `num_classes`, `imbalance_type`, `margin_level`,
`label_noise_rate`, `feature_noise_level`, `task_seed`, `is_tabpfn_anchor`.

Conditional: `class_weight_matrix`, `class_bias_vector` (with
`--store_class_params`); `teacher_prob_{k}`, `teacher_converged`,
`teacher_n_iter`, `teacher_C` (with `--store_class_teacher_preds`).

### Validation

`validate_classification_dataset(ds)` enforces:

- Array shapes for `X`, `X_clean`, `y`, `y_clean`, `label_noise_mask`,
  `logits`, `probs`, `W_true`, `b_true`, `active_support`, `class_active_support`
- `logits = X_clean @ W_true + b_true` identity (atol 1e-9)
- `probs = softmax(logits / temperature)` identity (atol 1e-10)
- Reference class invariants: `W_true[:, 0] == 0`, `b_true[0] == 0`,
  `logits[:, 0] == 0`
- `label_noise_mask` count equals `round(label_noise_rate * n)` exactly
- All K classes present in `y_clean`
- Noise-column coefficients are all zero
- Margin bucket matches requested `margin_level`
- Prior calibration error ≤ 0.03
- `J_low_n_high_p_classification` enforces `n < p`
- `K_feature_noise_classification` enforces `feature_noise_level > 0`

`validate_classification_eval_dataset(table)` validates the per-row eval
Parquet table against the `linear_classification_eval_v1` schema.

**CLS-S1 read-after-write validation:** after `write_parquet_classification_eval`
writes each file, `_assert_required_eval_fields()` immediately re-reads and
checks all required per-row columns. Errors are caught at generation time, not
hours later at index time. Required fields: `feature_vector`, `label`, `split`,
`task_family`, `task_objective`, `num_classes`, `suite_id`, `suite_family`,
`global_idx`, `n_train`, `n_test`.

### Loader Contract (`load_classification_parquet`)

`load_classification_parquet(path, *, min_support_per_class=1, global_idx=None)` in
`src/train.py` enforces the following invariants before returning a task dict:

**F8 — Label map derived from `y_train` only:**
The canonical label map is built from `torch.unique(y_train, sorted=True)`.
`y_test` does not influence which class indices are considered canonical.
Query labels not in the training label map are mapped to `num_classes - 1`
(OOD bucket) and reported under `"unseen_query_classes"` in the returned dict.
`num_classes` is always preserved from the Parquet spec, not reduced to the
count of unique training classes.

**F3 — Minimum support count policy:**
After label remapping, every class index `0..K-1` must have at least
`min_support_per_class` training examples (default 1). Classes with 0 support
raise `ValueError` whose message includes the `global_idx` for debugging.
Stress-family evaluation tasks should pass `min_support_per_class=0`.
The returned dict always includes `"support_class_counts"` (dict mapping class →
count) and `"missing_support_classes"` (list of 0-support class IDs).

### Classification Generator Changes (Audit)

| ID | Change | Location |
|----|--------|----------|
| CLS-D1 | `probs_test` column renamed to `dgp_teacher_probs_test` in `write_classification_parquet`; clarifies this is the DGP teacher output (softmax at temperature T), not a Bayes-optimal predictor. Metric `bayes_log_loss` renamed to `dgp_teacher_log_loss` in evaluator. Column `dgp_teacher_probs_test` now required in `_REQUIRED_PARQUET_FIELDS`. | `dgp_helpers.py`, `evaluate_synthetic_classification.py`, `prepare_synthetic_classification.py` |
| CLS-S1 | Read-after-write parquet validation added to eval generator: after `write_parquet_classification_eval` writes each file, `_assert_required_eval_fields()` immediately re-reads and checks all required per-row columns. Errors are caught at generation time, not hours later at index time. Required fields: `feature_vector`, `label`, `split`, `task_family`, `task_objective`, `num_classes`, `suite_id`, `suite_family`, `global_idx`, `n_train`, `n_test`. | `generate_synthetic_classification.py` |
| CLS-M1 | `task_seed` added to per-dataset entries in the classification manifest JSON. Each dataset's seed is now surfaced in the manifest without requiring individual parquet reads. | `generate_synthetic_classification.py` `_make_dataset_record` and all suite-family generation functions |
| CLS-O1 | Checkpoint quality gate env vars corrected from `SYNREG_RUN_CHECKPOINT_GATES` / `SYNREG_CHECKPOINT_GATE_STRICT` to `SYNCLS_RUN_CHECKPOINT_GATES` / `SYNCLS_CHECKPOINT_GATE_STRICT` in the orchestrator. A deprecation shim in the evaluator still reads `SYNREG_*` as a fallback for one release cycle. | `run_synthetic_classification_evaluation.py`, `evaluate_synthetic_classification.py` |
| CLS-S2 | `split_seeds` silent default of `[0, 1, 2]` in prepare script replaced with: if `task_seed` present, derive deterministically from it; otherwise store `NULL` and emit `logging.warning`. Prevents fabricated defaults from being stored as genuine generation parameters. | `prepare_synthetic_classification.py` |
| CLS-F1 | Per-split K coverage validation added. `validate_per_split_k_coverage(datasets, required_k, strict_coverage=False)` checks that all values in `num_classes_grid` appear in the generated suite. Result stored under `k_coverage_audit` in the manifest JSON. `build_meta_classification_dataset_index.py` emits a `UserWarning` (not failure) when coverage is incomplete. | `src/dgp_helpers.py`, `scripts/generate_synthetic_classification.py`, `src/build_meta_classification_dataset_index.py` |
| CLS-F2 | Class-label permutation augmentation added. `permute_class_labels(y_train, y_test, teacher_dict, rng, num_classes)` in `classification.py` applies a consistent random class permutation to labels, teacher probs, teacher logits, and `W_true`/`b_true`. Enabled per-batch in `run_classification_epoch` via `cfg.class_permutation_augmentation=True` (off by default). | `src/classification.py`, `src/train.py` |
| CLS-F3 | Minimum support count policy added to loader. See Loader Contract section above. | `src/train.py` |
| CLS-F4 | `log_loss` and `top_k_accuracy_score` now receive `labels=list(range(num_classes))` so absent classes are handled correctly. Per-metric status fields (`log_loss_valid`, `roc_auc_valid`, `brier_score_valid`, `has_missing_support`, `missing_support_classes`) added to every result row. Multiclass AUC (`roc_auc_ovr`, `roc_auc_ovo`) suppressed (NaN) when `has_missing_support=True`. `_align_proba_to_canonical()` now returns a 3-tuple `(aligned, has_missing, missing_class_ids)`. | `scripts/evaluate_synthetic_classification.py` |
| CLS-F5 | Paired robustness task builder added. `build_paired_robustness_tasks(base_params, sweep_param, sweep_values, base_seed)` produces task descriptors that share the same latent seed across sweep values, enabling controlled experiments over `feature_noise_level`, `label_noise_rate`, `p_noise`, `p_signal`, and `n_train`. Tasks carry a shared `pair_id` UUID and `is_reference` flag. | `src/dgp_helpers.py` |
| CLS-F6 | `val_cross_entropy` in checkpoints now records **pure query cross-entropy only** (the `"ce"` key from `compute_classification_losses`), separated from `val_total_loss` which records the full multi-component objective (CE + KL + coef_aux + margin + prior + calibration). HPO and early stopping use `val_cross_entropy` (semantically correct). | `src/train.py` |
| CLS-F7 | `build_ranking_summary(df, metric_col, task_col, model_col, higher_is_better)` added to `evaluation_metrics.py`. Returns per-model `mean_rank`, `median_rank`, `strict_win_rate`, `tie_aware_win_rate`, `top_3_rate`, `task_count`, `valid_metric_task_count`, `completion_rate`. Aggregate mode writes `synthetic_classification_ranking_summary.csv` and per-K/per-suite-family breakdowns. | `src/evaluation_metrics.py`, `scripts/evaluate_synthetic_classification.py` |
| CLS-F8 | Query-label leakage fixed in loader. Canonical label map derived from `y_train` only. See Loader Contract section above. | `src/train.py` |
| CLS-F9/F11 | Schema version constant `CLASSIFICATION_EVAL_SCHEMA_VERSION` in `dgp_helpers.py` is the single source of truth. Never hard-code the version string in tests or scripts — always import it: `from dgp_helpers import CLASSIFICATION_EVAL_SCHEMA_VERSION`. Test updated from a hard-coded string to the imported constant. | `src/dgp_helpers.py`, `tests/test_classification_eval_manifest.py` |
| CLS-F10 | Evidence chain fields added to checkpoints: `checkpoint_task_objective`, `checkpoint_max_k`, `checkpoint_best_epoch`, `checkpoint_training_family`, `checkpoint_val_ce_by_k`. Manifest gains `completion_gates` (`binary_datasets_present`, `multiclass_datasets_present`, `all_suite_families_present`), `k_coverage_audit`, and `source_git_revision`. | `src/train.py`, `scripts/generate_synthetic_classification.py` |

---

## DeepSet Classification Evaluation Protocol

**Script:** `scripts/evaluate_synthetic_classification.py`
**Orchestrator:** `scripts/run_synthetic_classification_evaluation.py`
**Mode env var:** `SYNTHETIC_CLASSIFICATION_MODE`

The evaluation pipeline has four modes dispatched by
`SYNTHETIC_CLASSIFICATION_MODE`: `deepset`, `baselines`, `autogluon`,
`aggregate`.

### Pipeline Phases

| Phase | Handler | Compute pool | Description |
|-------|---------|--------------|-------------|
| 1 | `run_synthetic_classification_linear_prep` | `DEEPSET_CPU_POOL` | Index parquets into `LINEAR_CLASSIFICATION_DATASET_INDEX` via `prepare_synthetic_classification.py` |
| 2 | `run_synthetic_classification_linear_deepset_evaluation` | `DEEPSET_GPU_POOL` | MODEL4 GPU inference, sharded |
| 3 | `run_synthetic_classification_linear_baseline_evaluation` | `DEEPSET_CPU_POOL` | sklearn/boosting baselines, sharded |
| 4 | `run_synthetic_classification_linear_autogluon_evaluation` | `AUTOGLUON_CPU_POOL` | AutoGluon single-node, sharded |
| 5 | `run_synthetic_classification_linear_aggregation` | `DEEPSET_CPU_POOL` | Gather shards, produce CSVs and manifest |

Path guard: `SYNCLS_RESULTS_STAGE` must contain `/classification/linear/`.

### MODEL4 DeepSet Inference (`deepset` mode)

**Checkpoint:** loaded from `SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH`
(default `@MODEL_STAGE/checkpoints/best_classification.pt`).

**Checkpoint validation:** `validate_classification_checkpoint()` from
`sanity_checks.py` is called before any inference. Failure raises
`RuntimeError` and aborts the shard.

**Model name in results:** `MODEL-ICL-MC`
**`prediction_source`:** `model4_checkpoint`
**`metric_schema_version`:** `2.0` (real MODEL4 forward pass; not DGP teacher probabilities)

#### Feature Selection

`select_deepset_classification_features_train_only()` from `deepset_inference.py`
applies feature selection fitted only on the training split:

- Score function: `f_classif` (classification F-statistic)
- Feature cap: resolved dynamically via `resolve_deepset_feature_cap(model)`
  from the loaded model's configuration; not hardcoded
- Returns `X_train_selected`, `X_test_selected`, and `selector_metadata`

#### Inference: Bounded-Context Ensemble

| Parameter | Default | Env var |
|-----------|---------|---------|
| Context ensembles | 8 | `SYNCLS_N_ENSEMBLES` |
| Test batch size | 256 | `SYNCLS_TEST_BATCH_SIZE` |
| Context size | per `cfg.max_n_train` or `SYNTHETIC_CLASSIFICATION_CONTEXT_SIZE` | `SYNTHETIC_CLASSIFICATION_CONTEXT_SIZE` |

For each ensemble index, `select_deepset_context_indices()` draws a
deterministic context window of `ctx_size` rows from the training set
(seeded with `context_index=ens_idx`). The model receives `(X_ctx, y_ctx,
X_test_batch)` and returns `out["logits"]`. Probabilities are computed via
manual softmax and accumulated; the final output is `all_probs / n_ensembles`.

The model is called with `task_objective="inductive_classification"` and
`num_classes=K` on every forward pass.

#### GPU Memory Guard

Before inference, `_guard_inference_sizes()` estimates memory as:

```python
estimated = 2.0 * 4 * 128 * min(p_selected, 256) * (min(context, 1024) + min(query, 256))
```

If `estimated > usable_bytes` (where `usable = GPU_free * safety_factor`,
default safety_factor=0.8), query batch size is halved first, then context
size, down to minimums (`SYNCLS_MIN_QUERY_BATCH_SIZE=1`,
`SYNCLS_MIN_CONTEXT_SIZE=32`). If memory is still insufficient after
downshifting, the dataset is skipped with `skip_reason` in the result row.

GPU memory is read from `torch.cuda.mem_get_info()[0]` unless
`SYNTHETIC_EVAL_GPU_MEMORY_BYTES` is set explicitly.

### Baseline Models (`baselines` mode)

Baselines are evaluated in up to two `feature_mode` variants controlled by
`SYNTHETIC_EVAL_BASELINE_FEATURE_MODES` (default `selected`; can add `raw`):

| Model | Library | Notes |
|-------|---------|-------|
| `DummyClassifier_most_frequent` | sklearn | Majority-class predictor |
| `DummyClassifier_stratified` | sklearn | Stratified random predictor |
| `LogisticRegression` | sklearn | `max_iter=1000` |
| `RidgeClassifier` | sklearn | Default |
| `LinearSVC` | sklearn | `max_iter=2000` |
| `RandomForestClassifier` | sklearn | `n_estimators=100` |
| `XGBClassifier` | xgboost | `eval_metric="logloss"` |
| `LGBMClassifier` | lightgbm | `verbosity=-1` |
| `CatBoostClassifier` | catboost==1.2.10 | `verbose=0` |
| `KNeighborsClassifier` | sklearn | Default k |
| `MLPClassifier` | sklearn | `max_iter=500` |

Feature selection for baselines (`selected` mode): `SelectKBest(f_classif,
k=min(p, max(1, int(sqrt(p * n / n)))))`. In `raw` mode no selection is
applied. Model names are suffixed with `_raw` or `_train_f_classif` unless
using a legacy suite (suffix omitted for backward compatibility).

For models that return `predict_proba`, `_align_proba_to_canonical()` aligns
the probability matrix to canonical class IDs `0..K-1`, epsilon-padding
absent classes. `has_missing_support = True` suppresses multiclass AUC
to avoid misleading metrics from epsilon-padded columns.

### AutoGluon (`autogluon` mode)

- Preset: `high_quality` (default; configurable via `SYNCLS_AUTOGLUON_PRESETS`)
- Time limit: 300 s per dataset (default; configurable via `SYNCLS_AUTOGLUON_TIME_LIMIT`)
- `problem_type`: `"binary"` for K=2, `"multiclass"` for K>2
- Single-node mode only; Ray distributed classification has not been validated
  (`autogluon_cluster_shards=0` is enforced)
- pip: `autogluon.tabular==1.3.0`

### Classification Metrics

All metrics are computed by `_compute_classification_metrics()`:

| Metric | Type | Notes |
|--------|------|-------|
| `accuracy` | label | |
| `balanced_accuracy` | label | |
| `macro_f1` | label | `zero_division=0` |
| `weighted_f1` | label | `zero_division=0` |
| `macro_precision` | label | `zero_division=0` |
| `macro_recall` | label | `zero_division=0` |
| `mcc` | label | Matthews correlation coefficient |
| `cohen_kappa` | label | |
| `log_loss` | probability | `labels=canonical_labels` passed to handle absent classes |
| `roc_auc_ovr` | probability | Macro OVR; suppressed (`NaN`) when `has_missing_support` |
| `roc_auc_ovo` | probability | Macro OVO; suppressed (`NaN`) when `has_missing_support` |
| `brier_score` | probability | Multiclass: mean over `sum((p - onehot)^2)` per sample |
| `average_precision` | probability | Macro; binary uses positive-class column |
| `expected_calibration_error` | probability | 10-bin ECE, max-confidence calibration |
| `top_2_accuracy` | probability | `NaN` when `num_classes <= 2` |
| `top_3_accuracy` | probability | `NaN` when `num_classes <= 3` |

The DeepSet mode also records `dgp_teacher_log_loss` (log-loss of stored DGP
teacher probabilities against `y_test`) as a Bayes-reference diagnostic only.

### Comparative / Rank Metrics (Aggregation)

After combining all shards, `_add_rank_metrics()` computes per-task rankings
and comparisons:

| Column | Description |
|--------|-------------|
| `rank_by_log_loss` | Ascending rank by log-loss per task |
| `rank_by_accuracy` | Descending rank by accuracy per task |
| `rank_by_macro_f1` | Descending rank by macro F1 per task |
| `is_best_log_loss` / `is_best_accuracy` | Rank 1 on this task |
| `is_top3_log_loss` / `is_top3_accuracy` | Rank ≤ 3 on this task |
| `beats_logistic_regression` | Accuracy > LogisticRegression on this task |
| `beats_ridge_classifier` | Accuracy > RidgeClassifier on this task |
| `beats_autogluon` | Accuracy > AutoGluon on this task |
| `beats_best_tree` | Accuracy > best tree model (RF/XGB/LGB/CB) on this task |
| `ratio_log_loss_to_logistic_regression` | `log_loss / log_loss_LogisticRegression` |
| `ratio_log_loss_to_autogluon` | `log_loss / log_loss_AutoGluon` |
| `ratio_error_rate_to_logistic_regression` | `(1-acc) / (1-acc_LR)` |
| `accuracy_delta_vs_logistic_regression` | `accuracy - accuracy_LR` |
| `macro_f1_delta_vs_logistic_regression` | `macro_f1 - macro_f1_LR` |

All beat metrics are `NaN` (not `False`) when the reference model has no
valid result on that task.

### Aggregation Outputs

The `aggregate` mode produces the following files, uploaded to `SYNCLS_RESULTS_STAGE`:

| File | Description |
|------|-------------|
| `synthetic_classification_model_comparison.csv` | Full per-dataset-per-method rows with all metrics and rank columns |
| `synthetic_classification_model_comparison_summary.csv` | Per-method mean metrics across all development datasets (excludes `hidden_holdout`) |
| `synthetic_classification_summary_by_regime.csv` | Per-method means by `classification_regime` |
| `synthetic_classification_summary_by_suite_family.csv` | Per-method means by `suite_family` |
| `synthetic_classification_summary_by_feature_noise.csv` | Per-method means by `feature_noise_level` |
| `synthetic_classification_summary_by_label_noise.csv` | Per-method means by `label_noise_rate` |
| `synthetic_classification_summary_by_training_size.csv` | Per-method means by `n_train_default` |
| `synthetic_classification_summary_by_class_imbalance.csv` | Per-method means by `class_imbalance_type` |
| `synthetic_classification_summary_by_margin.csv` | Per-method means by `margin_level` |
| `synthetic_classification_summary_by_num_classes.csv` | Per-method means by `num_classes` |
| `synthetic_classification_summary_by_sample_complexity.csv` | Per-method means by `sample_complexity_bucket` |
| `synthetic_classification_summary_by_p_quartile_n_quartile.csv` | Per-method means by p/n quartile cells |
| `synthetic_classification_chart_data_model_rank.csv` | Per-dataset rank columns for CDF charts |
| `synthetic_classification_chart_data_noise_features.csv` | Mean metrics by `feature_noise_level` |
| `synthetic_classification_chart_data_label_noise.csv` | Mean metrics by `label_noise_rate` |
| `synthetic_classification_chart_data_training_size.csv` | Mean metrics by `n_train_default` |
| `synthetic_classification_chart_data_calibration.csv` | Mean ECE by model |
| `synthetic_classification_aggregation_manifest.json` | Run metadata, shard counts, output validation |

**Additional conditional outputs** (not in `_REQUIRED_OUTPUT_FILES`):

| File | Description |
|------|-------------|
| `synthetic_classification_hidden_summary.csv` | Summary over `hidden_holdout` datasets only |
| `synthetic_classification_eval_only_summary.csv` | Summary over `eval_only_unseen` datasets only |
| `synthetic_classification_id_ood_hidden_summary.csv` | Summary by `distribution_group` (id/ood/hidden) |
| `synthetic_classification_distribution_degradation.csv` | Per-model accuracy/F1/log-loss degradation from id→ood |
| `synthetic_classification_method_completeness_audit.csv` | Per-model task completion rate |
| `synthetic_classification_bootstrap_confidence_intervals.csv` | 2000-replicate task-level bootstrap CIs per metric |
| `synthetic_classification_ranking_summary.csv` | Full-suite ranking summary (requires `evaluation_metrics.build_ranking_summary`) |
| `synthetic_classification_ranking_summary_k{K}.csv` | Per-K ranking breakdown |
| `synthetic_classification_ranking_summary_{family}.csv` | Per-suite-family ranking breakdown |

The aggregation manifest records: `suite_id`, `task_family`,
`task_objective`, `schema_version`, `metric_schema_version`,
`evaluation_run_id`, `input_shard_list`, expected/actual shard topology,
`checkpoint_path`, `feature_selector`, `baseline_feature_modes`,
`bootstrap_replicates`, `bootstrap_seed`, `method_completeness_audit`,
`completion_gates` (`binary_datasets_present`, `multiclass_datasets_present`,
`all_suite_families_present`), `validation_status`.

---

## Cross-Pipeline Changes (Aggregation)

These changes apply to the aggregation layer in
`scripts/evaluate_synthetic_regression.py`:

| ID | Change |
|----|--------|
| AGG-C1 | Per-shard column validation before `pd.concat`: shards missing any column in `_SHARD_REQUIRED_COLUMNS` raise `ValueError` immediately rather than silently NaN-filling the combined frame. |
| AGG-C2 | Metric range assertions after valid-row classification: negative `mse_betaX` values in valid rows raise `RuntimeError`. |
| AGG-C3 | Method completeness warning before ranking: logs missing methods per task group so incomplete coverage is visible rather than silently biasing average rank. |

---

## Mixed-Categorical Linear Families

Two additional training-data families extend the pure numeric pipelines with
categorical features:

```text
training_data_family = "synthetic_linear_regression_mixed_categorical"
training_data_family = "synthetic_linear_classification_mixed_categorical"
```

### DGP Formulas

**Regression:**
```
y = X_num_clean @ beta_num
  + sum_j cat_effects[j][X_cat_clean[:, j] - FIRST_REAL_ID]
  + epsilon
```

**Classification:**
```
logits = X_num_clean @ W_num + b
       + sum_j cat_class_effects[j][X_cat_clean[:, j] - FIRST_REAL_ID, :]
probs = softmax(logits / temperature)
y ~ Categorical(probs)
```

Reference class constraint: `W_num[:, 0] = 0`, `b[0] = 0`,
`cat_class_effects[j][:, 0] = 0`.

### Balanced Label Generation

Categorical classification training uses balanced labels by default
(`label_noise_rate=0.0`). All K classes must be present in both context and
query splits after the 80/20 split.

### Categorical Parameter Grids

| Parameter | Default Values |
|-----------|---------------|
| `p_cat_signal` | 1, 2, 4, 8 |
| `p_cat_noise` | 0, 2, 4, 8 |
| `cardinality` | 2, 3, 5, 10, 20, 50 |
| `cat_effect_scale` | 0.25, 0.5, 1.0, 2.0 |
| `cat_missing_rate` | 0.0, 0.01, 0.05, 0.10 |
| `cat_imbalance_type` | balanced, mild, moderate, severe |

### Entity Embedding Token IDs

Category IDs are stored as `int64` in the range
`[ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + C)`.
Special tokens 0 (PAD), 1 (MISSING), 2 (UNKNOWN) are reserved.

### Parquet Schema Additions

Mixed-categorical Parquet files include these additional columns beyond the
pure numeric schema:

- `X_cat_train`, `X_cat_test`: `list(list(int64))`
- `cat_missing_mask_train`, `cat_missing_mask_test`: `list(list(bool))`
- `cat_unknown_mask_test`: `list(list(bool))`
- `categorical_cardinalities`: `list(int64)`
- `p_cat`: `int64`
- `cat_effects` (regression) or `cat_class_effects` (classification)
- `cat_support_mask`: `list(bool)`

Schema versions: `MIXED_REG_DGP_SCHEMA_VERSION`, `MIXED_CLS_DGP_SCHEMA_VERSION`.

### Balance Validation Gates

Classification datasets are validated for:
- All K classes present in y
- Reference class columns are zero
- Schema version matches
- Entity embedding IDs in valid range (no PAD/UNKNOWN in X_cat)

---

## Change Workflow

When changing either task family:

1. State the statistical capability being introduced.
2. Identify shared behavior versus task-family-specific behavior.
3. Preserve existing field semantics and version incompatible schemas.
4. Add deterministic per-task validation.
5. Generate a small suite and audit realized coverage.
6. Smoke-test representative files through the actual task-family loader.
7. Record the command, git revision, seed, profile, grids, schema version, and
   checksums in the suite manifest.

