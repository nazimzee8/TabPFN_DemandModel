# Linear Classification Design Recommendation

## Contents

1. Executive recommendation
2. Source ownership
3. Task-level DeepSet semantics
4. Linear classification DGP
5. Regime catalog
6. Profiles and weights
7. Parameter grids
8. Coefficients and class structure
9. Feature distributions
10. Parquet contract
11. Training targets
12. Teachers
13. Validation
14. CLI design
15. Suite manifest
16. Interpretation guardrails
17. Initial suite
18. Migration and acceptance

## 1. Executive Recommendation

Preserve the current regression path and add classification as a parallel task
family inside `src/generate_dgp.py`:

```text
task_family = "linear_regression"      -> existing regression path
task_family = "linear_classification"  -> new classification path
```

Route at the CLI and orchestration layer. Reuse shared feature-generation and
diagnostic utilities, but use classification-native targets, metadata,
teachers, schemas, and validators.

Generate linear logits from clean features, map them to probabilities, sample
labels, and store both observed labels and clean latent targets. Sigmoid and
softmax do not make the task nonlinear; the decision scores remain linear in
`X_clean`.

Treat this document as a target architecture. The current repository's model,
training metadata, feature selector, checkpoint validation, and Snowflake
routing remain regression-specific.

## 2. Source Ownership

Keep these boundaries:

```text
src/generate_dgp.py
  - CLI and task-family routing
  - task sampling
  - outer train/validation/test split
  - inner context/query split
  - validation orchestration
  - Parquet writer dispatch
  - suite manifest writing

src/dgp_helpers.py
  - shared feature and covariance generation
  - shared coefficient primitives
  - task-family profile and regime definitions
  - classification parameter generation
  - logits, probabilities, labels, and noise
  - diagnostics and teachers
  - task-family validation and serialization
```

Reuse covariance, sparsity, irrelevant-feature, feature-noise, rank,
effective-rank, and condition-number utilities. Keep classification profiles,
regimes, logits, label sampling, label noise, imbalance calibration, teachers,
and validation explicit.

Do not alter existing regression field semantics or silently route regression
profiles through classification helpers.

## 3. Task-Level DeepSet Semantics

Make one file one classification episode:

```text
context: X_train, y_train
query:   X_test, y_test
```

Use the first 80 percent of rows as context and the remaining 20 percent as
query. Preserve the existing deterministic row-order split unless a future
schema version changes it.

The model should:

1. Encode context feature-label rows.
2. Aggregate context rows as an order-invariant set.
3. Condition class predictions on each query feature row.
4. Produce query logits or probabilities.
5. Compare predictions with observed labels or clean probabilities according
   to the configured training target.

Never mix task files into one context set. Fit preprocessing only on context
rows, then apply it to query rows.

## 4. Linear Classification DGP

Generate binary tasks as:

```text
X_clean ~ feature distribution
w_true ~ coefficient prior
b_true ~ intercept prior
z = X_clean @ w_true + b_true
binary_logits = [0, z]
probs = softmax(binary_logits / temperature)
y_clean ~ Categorical(probs)
y_observed = apply_symmetric_label_noise(y_clean)
X_observed = X_clean + optional_feature_noise
```

Use the zero score for class 0 as the canonical binary reference. This makes
`logits` and `probs` consistently shaped `(n, 2)` while retaining
`w_true: (p_total,)` and scalar `b_true`.

Generate multiclass tasks as:

```text
X_clean ~ feature distribution
W_true ~ coefficient prior
b_true ~ intercept prior
logits = X_clean @ W_true + b_true
probs = softmax(logits / temperature)
y_clean ~ Categorical(probs)
y_observed = apply_symmetric_label_noise(y_clean)
X_observed = X_clean + optional_feature_noise
```

Use a reference-class parameterization for identifiability: class 0 has zero
coefficients and zero intercept; sample columns 1 through `K-1`. Serialize
`W_true` with shape `(p_total, K)` and `b_true` with shape `(K,)`.

Interpret fields consistently:

- `logits_*`: clean linear scores before temperature scaling.
- `probs_*`: clean probabilities after temperature scaling.
- `y_clean_*`: labels sampled from clean probabilities.
- `y_*`: labels exposed after optional corruption.
- `label_noise_mask_*`: rows whose observed label differs from `y_clean`.
- `feature_noise`: measurement error in observed features.
- `temperature`: overlap control; higher values flatten probabilities.
- `margin`: separation induced by clean logits.

Apply symmetric label noise by selecting exactly
`round(label_noise_rate * n)` full-task rows without replacement and replacing
each selected label uniformly with a different class. Split rows only after
feature and label generation so masks align with context and query arrays.

## 5. Regime Catalog

Use this A-L classification catalog:

| Regime | Description | Capability |
| --- | --- | --- |
| `A_iid_dense_logistic` | Independent Gaussian features, dense binary coefficients, mostly balanced labels | Baseline linear classification |
| `B_iid_sparse_logistic` | Independent Gaussian features and sparse binary coefficients | Sparse discriminative recovery |
| `C_label_noise_margin` | Low margins, high temperature, or explicit label flips | Overlap and label-noise robustness |
| `D_correlated_ar_logistic` | AR(1) features with dense or sparse binary coefficients | Classification under multicollinearity |
| `E_high_dim_dense_softmax` | Larger feature spaces and dense multiclass coefficients | Dense high-dimensional multiclass recovery |
| `F_high_dim_sparse_softmax` | Larger feature spaces and sparse per-class coefficients | Sparse high-dimensional multiclass recovery |
| `G_noise_features_classification` | Signal columns plus zero-effect irrelevant columns | Distractor resistance |
| `H_block_correlated_classification` | Correlated blocks with signal concentrated in selected blocks | Grouped discriminative structure |
| `I_equicorrelated_classification` | Global pairwise feature correlation | Near-collinearity stress |
| `J_low_n_high_p_classification` | More features than observations with sparse coefficients | Underdetermined classification |
| `K_feature_noise_classification` | Labels from clean features while the model sees noisy features | Errors-in-variables classification |
| `L_market_sign_classification` | Structured positive and negative signs with optional irrelevant attributes | Domain-structured classification |

Disable `J_low_n_high_p_classification` unless generation, teachers, loaders,
and training explicitly support underdetermined classification.

## 6. Profiles and Weights

Define these classification profiles:

```text
linear_classification_stat_aware
linear_classification_stress
market_classification
classification_legacy_debug
```

Use `linear_classification_stat_aware` by default:

| Regime | Weight |
| --- | ---: |
| `A_iid_dense_logistic` | 0.15 |
| `B_iid_sparse_logistic` | 0.15 |
| `C_label_noise_margin` | 0.10 |
| `D_correlated_ar_logistic` | 0.10 |
| `E_high_dim_dense_softmax` | 0.05 |
| `F_high_dim_sparse_softmax` | 0.05 |
| `G_noise_features_classification` | 0.10 |
| `H_block_correlated_classification` | 0.05 |
| `I_equicorrelated_classification` | 0.05 |
| `J_low_n_high_p_classification` | 0.05 |
| `K_feature_noise_classification` | 0.05 |
| `L_market_sign_classification` | 0.10 |

Treat weights as probabilities. Remove the J regime and renormalize when
`--allow_underdetermined` is absent. Audit realized counts in the manifest.

## 7. Parameter Grids

Reuse shared grids:

```text
n_grid:                 32, 64, 128, 256, 512, 1024
p_signal_grid:          4, 8, 16, 32, 64
p_noise_grid:           0, 8, 24, 56, 120
active_s_grid:          2, 4, 8, 16, 32
rho_grid:               0.0, 0.3, 0.6, 0.9
feature_noise_grid:     0.0, 0.05, 0.10, 0.25
```

Add classification grids:

```text
num_classes_grid:       2, 3, 5, 10
temperature_grid:       0.5, 1.0, 2.0, 4.0
label_noise_grid:       0.0, 0.02, 0.05, 0.10
class_imbalance_grid:   balanced, mild, moderate, severe
margin_grid:            low, medium, high
coefficient_scale_grid: 0.5, 1.0, 2.0
intercept_scale_grid:   0.0, 0.5, 1.0, 2.0
```

Sample class counts with probabilities:

```text
K=2:  0.50
K=3:  0.25
K=5:  0.15
K=10: 0.10
```

Sample imbalance levels with probabilities:

```text
balanced: 0.50
mild:     0.25
moderate: 0.20
severe:   0.05
```

Generate a target class prior by choosing a majority class and interpolating
between uniform and one-hot priors:

```text
class_prior = (1 - imbalance_strength) / K + imbalance_strength * one_hot(majority)

balanced strength: 0.00
mild strength:     0.15
moderate strength: 0.35
severe strength:   0.60
```

Calibrate intercepts against the generated clean logits so mean clean
probabilities approximate the target prior. Do not resample labels to create
imbalance.

Define the raw margin as the largest clean logit minus the second-largest clean
logit. For binary tasks this equals `abs(z)`. Bucket the median
temperature-normalized margin:

```text
low:    < 0.75
medium: 0.75 through 2.00
high:   > 2.00
```

Use coefficient scale, temperature, and bounded rejection sampling to meet the
requested margin bucket. Record requested and realized values.

## 8. Coefficients and Class Structure

Support:

```text
dense
sparse
decaying
group_sparse
market_sign
```

For binary tasks, store `w_true: (p_total,)` and scalar `b_true`. For
multiclass tasks, store `W_true: (p_total, K)` and `b_true: (K,)`.

Apply these rules:

- Set all noise-feature rows in `W_true`, or entries in `w_true`, to zero.
- Record `active_support` for binary and shared-support tasks.
- Record `class_active_support: (p_total, K)` for per-class sparse tasks.
- Exclude the reference class from active-support counts.
- Define `active_s` per non-reference class for per-class sparse regimes.
- Record both aggregate and per-class sparsity when supports differ.
- Scale coefficients before intercept calibration.
- Generate market-sign effects through coefficient signs, not label rewriting.

## 9. Feature Distributions

Reuse:

```text
iid Gaussian
AR(1)-correlated Gaussian
block-correlated Gaussian
equicorrelated Gaussian
signal plus irrelevant independent noise columns
latent clean plus measurement-noisy observed features
```

Record `covariance_type`, `rho`, `block_size`, `p_signal`, `p_noise`,
`p_total`, `feature_noise_level`, and `feature_noise_level_float`.

In the feature-noise regime, compute logits and labels from `X_clean`, then
store `X_observed`. Keep irrelevant columns and measurement error distinct.

## 10. Parquet Contract

Use one Parquet row per task, matching the existing nested-array storage model.
Use float64 for features, parameters, logits, probabilities, and continuous
metadata; int64 for labels, dimensions, counts, and class IDs; bool for masks.

Require observations:

```text
X_train, y_train
X_test, y_test
```

Require clean audit and latent targets:

```text
y_clean_train, y_clean_test
label_noise_mask_train, label_noise_mask_test
logits_train, logits_test
probs_train, probs_test
```

Serialize `logits_*` and `probs_*` as `(n_split, K)` for every task, including
binary tasks.

Require parameters:

```text
w_true or W_true
b_true
active_support
class_active_support, when per-class supports differ
```

Require dimensions:

```text
n, p, n_train, n_test
p_signal, p_noise, p_total
num_classes
```

Require generation metadata:

```text
schema_version = linear_classification_v1
task_family = linear_classification
task_objective = inductive_classification
default_target_mode = observed_label
prior_regime
classification_regime
coefficient_regime
active_s
sparsity_ratio
class_sparsity_ratio
covariance_type
rho
block_size
temperature
label_noise_rate
realized_label_noise_rate
class_imbalance_type
class_prior
realized_class_prior
realized_num_classes
margin_level
coefficient_scale
intercept_scale
feature_noise_level
feature_noise_level_float
```

Require diagnostics:

```text
condition_number
matrix_rank
effective_rank
mean_margin
median_margin
min_margin
class_entropy
min_class_count
max_class_count
train_class_counts
test_class_counts
minority_class_fraction
majority_class_fraction
bayes_error_proxy
teacher_available
teacher_type
teacher_failure_reason
```

Define `bayes_error_proxy = mean(1 - max(probs, axis=1))`. Compute margin
diagnostics from clean pre-temperature logits and also record normalized margin
statistics when needed for bucket validation.

Do not write `betaX_train`, `betaX_test`, `target_noise_scale`, or
regression-teacher fields into the classification schema.

## 11. Training Targets

Support two explicit target modes:

```text
observed_label:
  target = y_test
  loss = cross_entropy(pred_logits, y_test)

clean_probability:
  target = probs_test
  loss = KL(probs_test || pred_probs)
```

Store observed labels and clean probabilities for every canonical task. Use
observed-label mode as the default. Reserve clean probabilities and logits for
calibration, teacher alignment, function-recovery objectives, and auxiliary
losses.

Keep the selected mode in training configuration and checkpoint metadata. Do
not infer target semantics from field presence.

## 12. Teachers

Do not reuse OLS or ridge regression teachers. Support optional:

```text
binary logistic regression
multinomial logistic regression
L2-regularized logistic regression
one-vs-rest logistic regression
shrinkage LDA
linear SVM as an optional margin teacher
```

Use L2-regularized binary or multinomial logistic regression as the default
teacher. Keep teacher outputs optional:

```text
teacher_logits_train, teacher_logits_test
teacher_probs_train, teacher_probs_test
teacher_W, teacher_b
teacher_regularization
teacher_available
teacher_type
teacher_failure_reason
```

Teacher failure must not invalidate a task unless generation explicitly
requires teacher outputs. Catch convergence, singularity, and missing-class
failures and record a stable reason code.

## 13. Validation

Validate every task:

1. Require all core arrays and metadata.
2. Check feature, label, latent-target, parameter, support, and mask shapes.
3. Require integer labels in `[0, K-1]`.
4. Require finite probabilities in `[0, 1]` whose rows sum to one.
5. Recompute probabilities from logits and temperature.
6. Require at least two realized classes in the full task and context split.
7. Verify class counts and priors against clean and observed labels.
8. Verify the corruption mask and realized label-noise rate exactly.
9. Verify `p_total = p_signal + p_noise`.
10. Require irrelevant-feature coefficients to be exactly zero.
11. Verify sparse support, `active_s`, and sparsity metadata.
12. Guard against accidentally sparse dense regimes.
13. For feature-noise tasks, verify latent targets derive from `X_clean` while
    stored features equal `X_observed`.
14. Reject non-finite arrays, parameters, teachers, or diagnostics.
15. Verify requested margin and imbalance buckets within documented tolerance.

Use bounded task resampling when class-coverage, margin, or imbalance
constraints fail. Fail generation with the seed and parameter set after the
retry limit; do not silently relax the requested regime.

Audit every suite:

1. Tabulate realized counts by split, regime, `K`, `n`, `p_total`, sparsity,
   correlation, temperature, label noise, feature noise, margin, and imbalance.
2. Verify outer split counts and task-file isolation.
3. Inspect condition number, effective rank, `p/n`, class imbalance, and margin
   tails.
4. Confirm hard tasks exist without dominating the suite.
5. Regenerate a small suite with the same command and seed.
6. Smoke-test representative files through the classification loader.
7. Verify the manifest, schema version, and output checksum.

## 14. CLI Design

Extend routing conceptually:

```powershell
python src/generate_dgp.py `
  --out_dir data `
  --n_datasets 1000 `
  --task_family linear_regression `
  --profile linear_stat_aware `
  --base_seed 42
```

```powershell
python src/generate_dgp.py `
  --out_dir data\linear_classification `
  --n_datasets 1000 `
  --task_family linear_classification `
  --profile linear_classification_stat_aware `
  --base_seed 42
```

Add classification options:

```text
--num_classes_grid
--temperature_grid
--label_noise_grid
--class_imbalance_grid
--margin_grid
--coefficient_scale_grid
--intercept_scale_grid
--store_class_params / --no-store_class_params
--store_class_teacher_preds
--require_class_teachers
--allow_underdetermined
```

Default omitted `--task_family` to `linear_regression` during migration so
existing commands retain their semantics. Default `--store_class_params` to
enabled for canonical classification suites. Reject a profile that does not
belong to the selected task family.

Require a classification output root distinct from any existing regression
suite. Never mix task families in one `train/val/test` directory tree.

## 15. Suite Manifest

Write mandatory `manifest.json` at the suite root:

```text
suite_id
task_family
task_objective
profile
base_seed
n_datasets
outer_split_counts
generation_command
git_revision
schema_version
grid_values
profile_weights
allow_underdetermined
store_class_params
store_class_teacher_preds
require_class_teachers
realized_regime_counts
realized_K_counts
realized_imbalance_counts
realized_temperature_counts
realized_margin_counts
realized_label_noise_counts
output_checksum
created_at
```

Compute `output_checksum` as SHA-256 over the sorted relative Parquet paths and
their individual SHA-256 digests. Generate `suite_id` from task family,
profile, schema version, base seed, and a content-derived suffix.

## 16. Interpretation Guardrails

- Do not call linear classification nonlinear.
- Treat sigmoid and softmax as probability links over linear logits.
- Do not use `betaX_*` for classification.
- Do not use OLS or ridge regression teachers for classification.
- Do not confuse label noise with regression target noise.
- Do not confuse measurement noise with irrelevant columns.
- Do not generate feature-noise labels from `X_observed`.
- Do not create imbalance by post-hoc label resampling.
- Do not assume all tasks are overdetermined.
- Do not enable low-n/high-p classification by default.
- Do not equate configured profile weights with realized counts.
- Do not mix task rows or task files across split boundaries.
- Do not fit preprocessing on query rows.
- Do not overwrite a suite without recording provenance and coverage.
- Do not claim current training compatibility until downstream classification
  contracts pass end-to-end tests.

## 17. Initial Suite

Use this first canonical suite:

```text
n_datasets:                    1000
task_family:                   linear_classification
task_objective:                inductive_classification
profile:                       linear_classification_stat_aware
outer task split:              80/10/10
within-task split:             80% context, 20% query
class mix:                     50% K=2, 25% K=3, 15% K=5, 10% K=10
imbalance mix:                 50% balanced, 25% mild,
                               20% moderate, 5% severe
underdetermined tasks:         disabled
store parameters:              enabled
store clean labels/masks:      enabled
store logits/probabilities:    enabled
store teacher predictions:     optional
default target mode:           observed_label
```

Prioritize learnability and stability. Audit the first suite before increasing
severe imbalance, high-dimensional extremes, label noise, or underdetermined
coverage.

## 18. Migration and Acceptance

Migrate in this order when implementation is authorized:

1. Add task-family constants, profile validation, and backward-compatible CLI
   routing without changing regression output.
2. Add classification DGP helpers, task validation, writer, schema version, and
   manifest.
3. Add local loader support and round-trip schema tests.
4. Add a separate classification stage and index contract rather than
   overloading the regression index without a migration.
5. Add classification context encoding, output head, losses, metrics, training
   metadata, and checkpoint validation.
6. Add end-to-end tests from deterministic generation through indexing,
   training, checkpoint loading, and inference.
7. Generate and audit a small pilot before the 1,000-task canonical suite.

Accept the design only when:

- omitted task-family routing produces regression output identical in semantics
  to the pre-migration path;
- classification files satisfy `linear_classification_v1`;
- fixed-seed pilot generation is deterministic;
- profile and split audits match the manifest;
- representative binary and multiclass tasks load and train end to end;
- regression tests remain unchanged and passing.

The generator should preserve the existing linear regression path and add a parallel linear classification path that generates linear logits, sampled labels, stored clean probabilities, classification-native metadata, and an A-L regime suite mirroring the regression generator while adding class count, class imbalance, margin, temperature, and label-noise controls.

