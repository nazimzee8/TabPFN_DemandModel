---
name: nonlinear-evaluation-pipeline
description: >
  Reference for the synthetic nonlinear evaluation pipelines covering nonlinear
  regression, nonlinear classification, mixed-categorical nonlinear, and market
  nonlinear demand. Use when interpreting evaluation results, understanding how
  nonlinear datasets were generated, how the DeepSet model was assessed against
  nonlinear baselines, what baselines it competed against, and how it performs
  under regime-specific nonlinear conditions. Also covers nonlinear-specific
  metrics, permutation contracts, calibration metrics, and ranking statistics.
---

# Nonlinear Evaluation Pipeline

## 1. Existing Nonlinear Evaluation Suite (v2)

The current nonlinear regression evaluation suite provides:

- 420 datasets, 6 families x 7 regimes.
- Index table: `NONLINEAR_REGRESSION_DATASET_INDEX`.
- Orchestrator: `scripts/run_nonlinear_regression_evaluation.py`.
- Prep script: `scripts/prepare_nonlinear_regression.py`.
- Pipeline reuses regression evaluation infrastructure via index table
  override.

The nonlinear classification evaluation suite:
- Index table: `NONLINEAR_CLASSIFICATION_DATASET_INDEX`.
- Orchestrator: `scripts/run_nonlinear_classification_evaluation.py`.
- Prep script: `scripts/prepare_nonlinear_classification.py`.

The 6 target families: `poly_quad`, `sin_low`, `hinge`, `sparse_interact`,
`mixed_linear`, `demand_mono`.

The 7 feature regimes: `iid_dense`, `iid_sparse`, `ar1`, `block`, `equicorr`,
`noise_feats`, `feat_noise`.

---

## 2. Nonlinear Regression Metrics

### Primary metrics (against mu_clean, NOT noisy y)

```text
mse_mu_clean          Mean squared error vs. noiseless mu_clean
rmse_mu_clean         Root MSE vs. mu_clean
mae_mu_clean          Mean absolute error vs. mu_clean
r2_mu_clean           R-squared vs. mu_clean
bias_vs_mu_clean      Mean of (y_hat - mu_clean)
variance_ratio        prediction_std / target_std
calibration_slope     Slope from regressing y_hat on mu_clean; 1.0 = well-calibrated
prediction_std        Standard deviation of model predictions
```

### Secondary metrics (against observed y)

```text
mse_y_observed        Mean squared error vs. noisy y
rmse_y_observed       Root MSE vs. y
mae_y_observed        Mean absolute error vs. y
r2_y_observed         R-squared vs. y
```

### Ranking / comparison

```text
rank_by_mse_mu_clean              Ascending rank by MSE vs. mu_clean per task
is_best_mse_mu_clean              Rank 1 on this task
is_top3_mse_mu_clean              Rank <= 3 on this task
beats_fixed_ridge                 MSE < MSE_FixedRidgeLambda1 on this task
beats_best_tree                   MSE < best tree method on this task
beats_kernel_method               MSE < best kernel method on this task
beats_autogluon                   MSE < AutoGluon on this task
ratio_mse_to_fixed_ridge          mse / mse_FixedRidgeLambda1
ratio_mse_to_best_tree            mse / min(mse_XGBoost, mse_LightGBM, mse_CatBoost, mse_RandomForest)
ratio_mse_to_kernel_method        mse / min(mse_SVR, mse_KernelRidge_RBF)
ratio_mse_to_autogluon            mse / mse_AutoGluon
```

All beat metrics are `NaN` (not `False`) when the reference model has no
valid result on that task.

---

## 3. Nonlinear Classification Metrics

Reuse existing classification metric stack:

```text
accuracy                     Label metric
balanced_accuracy            Label metric
macro_f1                     Label metric; zero_division=0
weighted_f1                  Label metric; zero_division=0
macro_precision              Label metric; zero_division=0
macro_recall                 Label metric; zero_division=0
mcc                          Matthews correlation coefficient
cohen_kappa                  Label metric
log_loss                     Probability metric; labels=canonical_labels
roc_auc_ovr                  Macro OVR; suppressed (NaN) when has_missing_support
roc_auc_ovo                  Macro OVO; suppressed (NaN) when has_missing_support
brier_score                  Multiclass: mean over sum((p - onehot)^2) per sample
average_precision            Macro; binary uses positive-class column
expected_calibration_error   10-bin ECE, max-confidence calibration
top_2_accuracy               NaN when num_classes <= 2
top_3_accuracy               NaN when num_classes <= 3
dgp_teacher_log_loss         Log-loss of DGP teacher probs against y_test; diagnostic only
```

### Ranking / comparison

```text
rank_by_log_loss                          Ascending rank by log-loss per task
rank_by_accuracy                          Descending rank by accuracy per task
rank_by_macro_f1                          Descending rank by macro F1 per task
is_best_log_loss / is_best_accuracy       Rank 1 on this task
is_top3_log_loss / is_top3_accuracy       Rank <= 3 on this task
beats_logistic_regression                 Accuracy > LogisticRegression on this task
beats_ridge_classifier                    Accuracy > RidgeClassifier on this task
beats_best_tree                           Accuracy > best tree model (RF/XGB/LGB/CB) on this task
beats_autogluon                           Accuracy > AutoGluon on this task
ratio_log_loss_to_logistic_regression     log_loss / log_loss_LogisticRegression
ratio_log_loss_to_autogluon               log_loss / log_loss_AutoGluon
accuracy_delta_vs_logistic_regression     accuracy - accuracy_LR
accuracy_delta_vs_autogluon               accuracy - accuracy_AutoGluon
```

---

## 4. Regression Baselines

### Existing baselines (always run)

| Method | Library | Notes |
|---|---|---|
| FixedRidgeLambda1 | sklearn | Ridge with lambda=1 fixed; primary ratio baseline |
| LinearRegression | sklearn | OLS, no regularization |
| Ridge | sklearn | Cross-validated lambda |
| RandomForest | sklearn | Default hyperparameters |
| XGBoost | xgboost | Default hyperparameters |
| LightGBM | lightgbm | Default hyperparameters |
| CatBoost | catboost==1.2.10 | Default hyperparameters |
| KNN | sklearn | Default k |
| SVR | sklearn | RBF kernel |
| MLP | sklearn | Default architecture |
| AutoGluon | autogluon | `high_quality` preset, 300s time limit |

### Polynomial baselines (nonlinear-specific)

| Method | Library | Notes |
|---|---|---|
| PolynomialRidge_D2 | sklearn | `PolynomialFeatures(degree=2)` + `StandardScaler` + `Ridge(alpha=1.0)` |
| PolynomialRidge_D3 | sklearn | `PolynomialFeatures(degree=3)` + `StandardScaler` + `Ridge(alpha=1.0)` |
| PolynomialRidge_D4 | sklearn | `PolynomialFeatures(degree=4)` + `StandardScaler` + `Ridge(alpha=1.0)`; skip if `p_total > 20` (feature explosion) |

### Robust regression baselines

| Method | Library | Notes |
|---|---|---|
| HuberRegression | sklearn | `HuberRegressor(epsilon=1.35)` -- primary robust baseline |
| TheilSen | sklearn | `TheilSenRegressor(max_subpopulation=1000)` -- robust slope estimation; run only on contaminated/cauchy/gross_outlier noise subsets |
| RANSAC | sklearn | `RANSACRegressor(estimator=Ridge(alpha=1.0))` -- outlier-robust; run only on contaminated/cauchy/gross_outlier noise subsets |

### Stepwise regression baseline

| Method | Library | Notes |
|---|---|---|
| StepwiseForward_Ridge | sklearn | `SelectKBest(f_regression, k=auto)` + `Ridge(alpha=1.0)` -- fast F-statistic proxy for stepwise; run only on datasets with `p_total <= 32` to control compute |

### Kernel / nonlinear-specific baselines (optional)

| Method | Library | Notes |
|---|---|---|
| KernelRidge_RBF | sklearn | Optional, nonlinear-specific |
| GaussianProcessRegressor | sklearn | Optional, small-n only: `n <= 200` |

### Parametric NLS baselines (second-phase, not initially required)

| Method | Library | Notes |
|---|---|---|
| NLS_Exponential | scipy | `curve_fit` with LM for `y = a * exp(b * x) + c`; oracle-form baseline |
| NLS_Sigmoid | scipy | `curve_fit` with LM for `y = L / (1 + exp(-k*(x-x0)))`; oracle-form baseline |

Note: NLS baselines assume the correct functional form (oracle advantage).
They are diagnostic comparators -- useful for measuring how close the DeepSet
model gets to the theoretically optimal fit when the functional form is known.
Add only when evaluating whether DeepSet matches oracle-form parametric fitters.

---

## 5. Classification Baselines

Full existing baseline set:

| Method | Library | Notes |
|---|---|---|
| DummyClassifier_most_frequent | sklearn | Majority-class predictor |
| DummyClassifier_stratified | sklearn | Stratified random predictor |
| LogisticRegression | sklearn | `max_iter=1000` |
| RidgeClassifier | sklearn | Default |
| LinearSVC | sklearn | `max_iter=2000` |
| RandomForestClassifier | sklearn | `n_estimators=100` |
| XGBClassifier | xgboost | `eval_metric="logloss"` |
| LGBMClassifier | lightgbm | `verbosity=-1` |
| CatBoostClassifier | catboost==1.2.10 | `verbose=0` |
| KNeighborsClassifier | sklearn | Default k |
| MLPClassifier | sklearn | `max_iter=500` |
| AutoGluon | autogluon | `high_quality` preset, 300s time limit |

Tree/boosting/MLP/KNN are essential because linear classifiers are
insufficient comparators for nonlinear tasks.

---

## 6. DeepSet Inference Protocol

Nonlinear-specific inference uses the same bounded-context ensemble as linear:

| Parameter | Default | Env var |
|---|---|---|
| Context windows | 5 | `SYNTHETIC_NONLINEAR_CONTEXT_ENSEMBLES` |
| Context window size | 200 rows | `SYNTHETIC_NONLINEAR_CONTEXT_SIZE` |
| MC samples per window | 8 | `MC_K` |
| Test batch size | 128 (regression) / 256 (classification) | `SYNTHETIC_NONLINEAR_TEST_BATCH_SIZE` |

### Feature selection

- Method: F-statistic on numeric columns (`f_regression` for regression,
  `f_classif` for classification).
- Feature cap: resolved dynamically via `resolve_deepset_feature_cap(model)`
  from the loaded model's configuration.
- Categorical features: pass through unchanged (no feature selection on
  categoricals).

### GPU memory guard

Before inference, estimated memory is compared against usable GPU memory.
If estimated exceeds usable (with safety_factor=0.8), query batch size is
halved first, then context size, down to minimums. If memory is still
insufficient after downshifting, the dataset is skipped with `skip_reason`
in the result row.

### Model semantics

- Model called with same `task_objective` semantics as linear evaluation.
- Regression: `task_objective="inductive_regression"`.
- Classification: `task_objective="inductive_classification"`,
  `num_classes=K`.
- The same model architecture processes both linear and nonlinear tasks.

---

## 7. Permutation Contracts

Existing F1-F7 contracts remain unchanged. DeepSet must still satisfy:

- **F1:** Support-row permutation invariance.
- **F2:** Query-row permutation equivariance.
- **F3:** Feature-column permutation consistency.
- **F4:** Feature-indexed output equivariance.
- **F5:** Classification label permutation equivariance (known limitation
  for K>=3 due to learned `ClassLabelEncoder` using `nn.Embedding`).
- **F6, F7:** Completion equivariance.

Nonlinear data does NOT change the model's invariance properties -- the same
model architecture processes both linear and nonlinear tasks. Permutation
contracts are model properties, not data properties.

Enforcement policy:

- Regression evaluators enforce F1-F4.
- Classification evaluators enforce F1-F4 and skip F5.
- Controlled by `CLASSIFICATION_RUN_PERMUTATION_GATES` (default `true`) and
  `CLASSIFICATION_PERMUTATION_GATE_STRICT` (default `true`).

---

## 8. Pipeline Phases

### Regression pipeline

| Phase | Handler | Pool | Description |
|---|---|---|---|
| 1 | Prep | CPU | Index nonlinear parquets into dataset index |
| 2 | DeepSet | GPU | MODEL4 inference, sharded |
| 3 | Baselines | CPU | sklearn/boosting baselines, sharded |
| 4 | AutoGluon | CPU | AutoGluon single-node |
| 5 | Aggregate | CPU | Merge shards, CSVs + manifest |

### Classification pipeline

| Phase | Handler | Pool | Description |
|---|---|---|---|
| 1 | Prep | CPU | Index nonlinear classification parquets into dataset index |
| 2 | DeepSet | GPU | MODEL4 inference, sharded |
| 3 | Baselines | CPU | sklearn/boosting baselines, sharded |
| 4 | AutoGluon | CPU | AutoGluon single-node |
| 5 | Aggregate | CPU | Merge shards, CSVs + manifest |

Both pipelines share the same 5-phase structure as the linear evaluation
pipeline.

---

## 9. Aggregation Outputs

### Nonlinear regression aggregation CSVs

```text
synthetic_nonlinear_regression_model_comparison.csv
synthetic_nonlinear_regression_summary.csv
synthetic_nonlinear_regression_summary_by_regime.csv
synthetic_nonlinear_regression_summary_by_nonlinear_family.csv
synthetic_nonlinear_regression_summary_by_feature_regime.csv
synthetic_nonlinear_regression_summary_by_feature_noise.csv
synthetic_nonlinear_regression_summary_by_target_noise.csv
synthetic_nonlinear_regression_summary_by_training_size.csv
synthetic_nonlinear_regression_summary_by_nonlinearity_strength.csv
synthetic_nonlinear_regression_summary_by_interaction_order.csv
synthetic_nonlinear_regression_summary_by_suite_component.csv
synthetic_nonlinear_regression_chart_data_model_rank.csv
synthetic_nonlinear_regression_aggregation_manifest.json
```

### Nonlinear classification aggregation CSVs

```text
synthetic_nonlinear_classification_model_comparison.csv
synthetic_nonlinear_classification_summary.csv
synthetic_nonlinear_classification_summary_by_regime.csv
synthetic_nonlinear_classification_summary_by_nonlinear_family.csv
synthetic_nonlinear_classification_summary_by_num_classes.csv
synthetic_nonlinear_classification_summary_by_label_noise.csv
synthetic_nonlinear_classification_summary_by_class_imbalance.csv
synthetic_nonlinear_classification_summary_by_margin.csv
synthetic_nonlinear_classification_summary_by_nonlinearity_strength.csv
synthetic_nonlinear_classification_ranking_summary.csv
synthetic_nonlinear_classification_aggregation_manifest.json
```

---

## 10. Mixed-Categorical Nonlinear Evaluation

### Eval index tables (F5 — now implemented)

Two dedicated eval index tables separate the mixed-categorical nonlinear datasets from the
standard nonlinear ones, mirroring the pattern used for linear mixed-categorical:

| Suite | Eval index table | SQL file |
|---|---|---|
| Nonlinear mixed regression | `NONLINEAR_MIXED_REGRESSION_DATASET_INDEX` | `sql/synthetic_nonlinear_pipeline.sql` |
| Nonlinear mixed classification | `NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX` | `sql/synthetic_nonlinear_classification_pipeline.sql` |

Both tables include mixed-specific columns: `p_num`, `p_cat`, `categorical_cardinalities` (VARIANT),
`cat_effect_scale`, `max_cardinality`, `missing_rate`, and all standard nonlinear index columns.

### Orchestrator wiring (F6 — now implemented)

The nonlinear orchestrators inject both `*_IS_MIXED_CATEGORICAL=true` and the mixed eval index
table name via dedicated module-level env dicts:

- **`run_nonlinear_regression_evaluation.py`**: `_NONLINEAR_MIXED_INDEX_ENV`, suite params from
  `_nonlinear_regression_suite_params(is_mixed_categorical=True)`.
  `suite_id = "nonlinear_mixed_regression"`, parts prefix = `@EVALUATION_RESULTS_STAGE/nonlinear/regression/mixed/nonlinear_mixed_regression`.
- **`run_nonlinear_classification_evaluation.py`**: `_NONLINEAR_MIXED_CLS_INDEX_ENV`, suite params
  from `_nonlinear_classification_suite_params(is_mixed_categorical=True)`.
  `suite_id = "nonlinear_mixed_classification"`, results stage = `@EVALUATION_RESULTS_STAGE/nonlinear/classification/mixed/nonlinear_mixed_classification`.

### SQL procedures for mixed-categorical phases

**Regression** (in `sql/synthetic_nonlinear_pipeline.sql`):
- `run_synthetic_nonlinear_mixed_regression_prep(BENCH_RT)`
- `run_synthetic_nonlinear_mixed_regression_deepset_evaluation(BENCH_RT)`
- `run_synthetic_nonlinear_mixed_regression_baseline_evaluation(BENCH_RT [, SHARDS [, CONCURRENT_NODES]])`
- `run_synthetic_nonlinear_mixed_regression_autogluon_spcs_evaluation(AG_IMAGE, SHARDS, WPS, CC [, ...])`
- `run_synthetic_nonlinear_mixed_regression_aggregation(BENCH_RT [, DS_SHARDS, BL_SHARDS, AG_SHARDS])`

**Classification** (in `sql/synthetic_nonlinear_classification_pipeline.sql`):
- `run_nonlinear_mixed_classification_prep(BENCH_RT)`
- `run_nonlinear_mixed_classification_deepset_evaluation(BENCH_RT)`
- `run_nonlinear_mixed_classification_baseline_evaluation(BENCH_RT [, SHARDS])`
- `run_nonlinear_mixed_classification_autogluon_evaluation(BENCH_RT, AG_RT [, SHARDS, CPUS, TL, PRESETS])`
- `run_nonlinear_mixed_classification_aggregation(BENCH_RT [, DS_SHARDS, BL_SHARDS, AG_SHARDS])`

### Prep script index table parameterization

`prepare_nonlinear_regression.py` reads the output index table name from
`SYNREG_INDEX_TABLE` (fallback: `NONLINEAR_REGRESSION_DATASET_INDEX`).
`prepare_nonlinear_classification.py` reads from `SYNCLS_INDEX_TABLE`
(fallback: `NONLINEAR_CLASSIFICATION_DATASET_INDEX`).
The orchestrators inject the mixed table name via the `*_INDEX_ENV` dicts so the same prep
script writes to the correct index without any code fork.

### Categorical encoding for baselines

Categorical encoding for baselines follows the same scheme as linear mixed-categorical:

| Encoding | Models |
|---|---|
| `one_hot` | Linear/kernel models (LogisticRegression, Ridge, SVR, KernelRidge) |
| `target_encode` | Ridge variant |
| `ordinal` | Tree models (RandomForest, XGBoost, LightGBM) |
| `native_cat` | CatBoost, LightGBM (native categorical support) |

Feature selection: numeric-only; categorical columns always pass through.

Mixed-categorical Parquet files include additional columns beyond the pure
numeric schema:

- `X_cat_train`, `X_cat_test`: `list(list(int64))`
- `cat_missing_mask_train`, `cat_missing_mask_test`: `list(list(bool))`
- `cat_unknown_mask_test`: `list(list(bool))`
- `categorical_cardinalities`: `list(int64)`
- `p_cat`: `int64`
- `cat_effects` (regression) or `cat_class_effects` (classification)
- `cat_support_mask`: `list(bool)`

Entity embedding token IDs are stored as `int64` in the range
`[ENTITY_EMBED_FIRST_REAL_ID, ENTITY_EMBED_FIRST_REAL_ID + C)`.
Special tokens 0 (PAD), 1 (MISSING), 2 (UNKNOWN) are reserved.

---

## 11. Market Nonlinear Evaluation (Deferred)

Market-specific evaluation metrics are deferred until the market mental
model is defined and provided. Deferred metrics include:

- Own-price elasticity recovery.
- Cross-price coefficient sign accuracy.
- Substitute/complement identification rate.
- Market graph edge detection precision.
- Demand curve shape fidelity.

These will be added as a separate SKILL.md update after:

1. The generic nonlinear suite is implemented and validated.
2. The market mental model equations are provided.
3. The DeepSet model demonstrates generic nonlinear competence.

---

## 12. Acceptance Criteria

All 16 acceptance criteria:

1. Existing linear regression generation still works.
2. Existing linear classification generation still works.
3. Existing mixed-categorical families still work.
4. New nonlinear regression generation produces valid Parquet.
5. New nonlinear classification generation produces valid Parquet.
6. Correct train/query splits (80/20 inner, 80/10/10 outer).
7. Regression stores and validates `mu_clean`.
8. Classification stores and validates logits, probs, y_clean, y,
   label-noise masks.
9. Manifests contain suite-level and per-task metadata.
10. Seeds deterministic and non-overlapping across hidden holdout.
11. Coverage audits report realized distribution.
12. DeepSet training can load nonlinear tasks.
13. Evaluation reports by regime, family, noise, dimensionality, sparsity,
    training size, nonlinearity strength.
14. Regression metrics primarily against `mu_clean`.
15. Classification metrics preserve canonical label/probability semantics.
16. Market nonlinear families deferred until market mental model is defined.
