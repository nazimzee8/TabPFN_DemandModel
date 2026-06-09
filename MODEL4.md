# MODEL4 — Linear-Statistic-Aware DeepSet ICL

**Normative architecture specification. Supersedes MODEL3.md for new training runs.**
MODEL3.md is retained as historical reference.

---

## Architecture Version

```
model_arch_version = "model4"
```

---

## Model Family / Design Pattern

Unchanged from MODEL3:
- `model_family = "market_exchangeable_icl"`
- `model_design_pattern = "inductive_forecasting"`

---

## Inherited MODEL3 Components (Backbone — Unchanged)

| Component | Role |
|---|---|
| `ExchangeableMatrixBlock` | Exchangeable-matrix encoder; permutation-equivariant over rows and columns |
| `ColumnEncoder` / `CellEncoder` | Input projection; encodes feature columns and (X, y) cells |
| `SetPool` | Sample-axis and feature-axis pooling (mean / PNA / attention) |
| `RidgeExpert` | Closed-form ridge solver; **retained as OLS/Ridge teacher utility** for auxiliary losses and for loading MODEL3 checkpoints; no longer the primary prediction path by default |

---

## New MODEL4 Components

| Component | Role |
|---|---|
| `LinearStatisticExtractor` | No parameters. Computes sufficient linear statistics `(feat_stats (p,6), global_stats (7,))` from normalized `(X_norm, y_norm)`. Always runs in float32 (protected from AMP via inner `autocast(enabled=False)`). |
| `LinearStatisticEncoder` | Shared MLP over feature rows (`feat_proj`) + global MLP (`global_proj`). Encodes `(p,6) + (7,)` → `(linear_stat_dim,)` embedding `z_linear`. Handles any `p` with the same weights. |
| `FusionModule` | Gated fusion: `gate = σ(MLP([h_glob, z_linear]))`, output = `gate*h_glob + (1-gate)*z_proj`. Combines DeepSet neural embedding with linear-stat embedding. |
| `CoefficientHead` | Per-feature shared MLP. Input: `[feat_stats_j (6,), h_global (neural_out_dim,)]` → scalar `beta_hat_j`. Output: `beta_hat_norm (p,)`. Handles any `p`. |
| `LambdaHead` | Linear → ReLU → Linear → Softplus. Output: positive scalar `lambda_hat` (dataset-specific regularization estimate). |

---

## ModelConfig Fields (MODEL4 Defaults)

| Field | Default | Notes |
|---|---|---|
| `model_arch_version` | `"model4"` | Was `"model3"` |
| `use_ridge_expert` | `False` | Was `True`; MODEL3 legacy path; set `True` to restore MODEL3 behavior |
| `use_linear_stats` | `True` | Enables `LinearStatisticExtractor` + encoder + fusion |
| `use_coefficient_head` | `True` | Enables `CoefficientHead` as primary prediction path |
| `use_lambda_head` | `True` | Enables `LambdaHead` for dataset-specific λ estimate |
| `use_residual_head` | `False` | When `True`, adds `neural_norm` as a residual: `pred = y_coeff + neural` |
| `linear_stat_dim` | `128` | Embedding dim for linear statistics |
| `max_moment_p` | `128` | Reserved for future p-capping in extractor |
| `use_sxx_triangle` | `False` | Reserved for future upper-triangle Sxx feature |
| `coeff_head_hidden_dim` | `64` | `CoefficientHead` MLP hidden dim |
| `lambda_head_hidden_dim` | `32` | `LambdaHead` MLP hidden dim |
| `beta_aux_loss_weight` | `0.10` | Tier 1: scale-invariant coefficient MSE |
| `pred_aux_loss_weight` | `0.05` | Tier 2: normalized prediction MSE |
| `cos_aux_loss_weight` | `0.02` | Tier 3: cosine direction alignment |
| `lambda_aux_loss_weight` | `0.01` | Lambda soft prior (`p/n` heuristic) |
| `teacher_ols_threshold` | `5.0` | Use OLS teacher when `n/p >= threshold` |
| `linear_encoder_hidden_dim` | `256` | `LinearStatisticEncoder` MLP hidden dim |
| `fusion_gate_hidden_dim` | `64` | `FusionModule` gate MLP hidden dim |
| `d_phi` | `128` | Unchanged from MODEL3 |
| `pool` | `"pna"` | Unchanged from MODEL3 |
| `n_sab_feat` | `1` | Unchanged from MODEL3 |
| `norm_feat` | `True` | Unchanged from MODEL3 |
| `norm_target` | `True` | Unchanged from MODEL3 |
| `dropout` | `0.1` | Unchanged from MODEL3 |

---

## Forward Pass Shape Trace

```
Input:
  X_train: (n, p)    y_train: (n,)    x_test: (m, p)    beta: (p,) or None

Normalization:
  X_norm:   (n, p)   y_norm:  (n,)    xq_norm: (m, p)
  col_mean: (p,)     col_std: (p,)    y_mean: scalar    y_std: scalar

LinearStatisticExtractor  [float32, autocast disabled]:
  feat_stats_f32:   (p, 6)           — sxy, diag_sxx, x_mean, x_std, corr, norm_sxy
  global_stats_f32: (7,)             — y_mean, y_std, y_var, p/n, n/p, cond, eff_rank
  → cast to input dtype:
  feat_stats:   (p, 6)
  global_stats: (7,)

LinearStatisticEncoder:
  feat_proj(feat_stats) → (p, 128).mean(0) → feat_pooled: (128,)
  global_proj([feat_pooled, global_stats]) → z_linear: (128,)

DeepSet backbone (unchanged):
  ExchangeableMatrixBlock: H: (m, n, p, d_phi) → ...
  SetPool (sample axis):  h_feat: (m, p, d_phi)
  SetPool (feature axis): h_glob: (m, neural_out_dim=128)

FusionModule:
  z_lin_exp: z_linear.expand(m, -1) → (m, 128)
  gate = σ(MLP([h_glob, z_lin_exp])) → (m, 128)
  h_fused = gate*h_glob + (1-gate)*z_lin_exp → (m, 128)

CoefficientHead:
  h_global_mean = h_fused.mean(0) → (128,)
  combined = [feat_stats, h_global_mean.expand(p,-1)] → (p, 134)
  beta_hat_norm = mlp(combined).squeeze(-1) → (p,)
  y_coeff_norm = xq_norm @ beta_hat_norm → (m,)

LambdaHead:
  lambda_hat = mlp(h_fused.mean(0)) → scalar (positive)

OLS teacher [no_grad, float32]:
  n/p >= 5.0 → lstsq(X_norm, y_norm) → beta_ols_teacher: (p,)
  y_ols_norm_teacher = xq_norm @ beta_ols_teacher → (m,)

Prediction:
  neural_norm = pred_head(h_fused) → (m,)
  pred_norm = y_coeff_norm                           (use_coefficient_head=True)
            [+ neural_norm if use_residual_head]

Denormalization:
  y_hat = pred_norm * y_std + y_mean → (m,)

Output: y_hat (m,)
```

---

## Teacher Loss Formulas

OLS is used as the teacher signal (unbiased, avoids lambda-selection bias; training data guarantees `n/p >= 5.0`). Ground-truth `beta_norm` takes priority when available.

```
beta_teacher = beta_norm  (if available — new parquet with --store_beta)
             else beta_ols_teacher

# Tier 1: Scale-invariant coefficient MSE
L_beta = mean((beta_hat - beta_teacher)²) / (mean(beta_teacher²) + 1e-6)

# Tier 2: Normalized prediction MSE (NMSE)
L_pred = mean((y_coeff - y_ols)²) / (Var(y_ols) + 1e-6)

# Tier 3: Cosine direction alignment
L_cos  = 1 - dot(beta_hat, beta_teacher) / (||beta_hat|| * ||beta_teacher|| + 1e-8)

# Lambda soft prior
L_lambda = MSE(log(lambda_hat), log(p/n))   [log-space, clamped at -10]

# Total loss
L = L_primary
  + 0.10 * L_beta
  + 0.05 * L_pred
  + 0.02 * L_cos
  + 0.01 * L_lambda
```

---

## Acceptance Criteria

Inherits all MODEL3 acceptance criteria, plus:

7. **Linear-stat permutation invariance:** Row-shuffle `X_train` → `z_linear` unchanged (max_abs_delta ≤ 1e-4). Column-permute `X_train` + `x_test` simultaneously → `y_coeff_norm` consistent.
8. **CoefficientHead variable-p:** Same weights, forward with `p=3`, `p=7`, `p=50`. `beta_hat_norm.shape == (p,)` in all cases. No exceptions.
9. **AMP dtype safety:** Forward under `autocast(bfloat16)`. `feat_stats_f32.dtype == float32` inside extractor. `y_hat` is finite in original dtype.
10. **Categorical stat extractor context-only:** Alter query y → categorical stats unchanged. Extractor never sees query labels.
11. **Mixed forward variable p_cat:** Same weights, different `p_cat` values (e.g. 3, 7, 20). `cat_effect.shape == (m,)` in all cases. No exceptions.
12. **Pure numeric path isolation:** `X_cat_train=None` produces identical output to a non-categorical model (`use_categorical_features=False`).

---

## Checkpoint Metadata Requirements

Pure-numeric regression (format version 4):

```json
{
  "checkpoint_format_version": 4,
  "cfg": { "model_arch_version": "model4", ... },
  "metadata": {
    "model_arch_version": "model4",
    "model_family": "market_exchangeable_icl",
    "model_design_pattern": "inductive_forecasting",
    "task_objective": "inductive_regression"
  }
}
```

Mixed-categorical regression (format version 6):

```json
{
  "checkpoint_format_version": 6,
  "cfg": { "model_arch_version": "model4", "use_categorical_features": true, ... },
  "metadata": {
    "model_arch_version": "model4",
    "task_objective": "inductive_regression",
    "feature_contract_version": "mixed_categorical_linear_v1",
    "uses_entity_embeddings": true,
    "use_categorical_features": true
  }
}
```

Mixed-categorical classification (format version 7):

```json
{
  "checkpoint_format_version": 7,
  "cfg": { "model_arch_version": "model4", "task_objective": "inductive_classification", "use_categorical_features": true, ... },
  "metadata": {
    "model_arch_version": "model4",
    "task_objective": "inductive_classification",
    "feature_contract_version": "mixed_categorical_linear_v1",
    "uses_entity_embeddings": true,
    "use_categorical_features": true
  }
}
```

Evaluation accepts `{"model3", "model4"}` in `model_arch_version` for historical comparison.

---

## Linear Classification Path

MODEL4 regression remains the default and keeps checkpoint format 4. Setting
`task_objective="inductive_classification"` enables a separate config-gated path:

- `ClassLabelEncoder` embeds integer support labels and uses a learned mask embedding for queries.
- `ClassificationStatisticExtractor` computes float32 class-conditional feature/global statistics.
- `ClassificationStatisticEncoder` and `ClassFusionModule` add permutation-invariant class context.
- `ClassCoefficientHead` and `ClassBiasHead` emit `W_hat_norm (p, K)` and `b_hat (K,)`.
- Predictions are multiclass-compatible logits `(m, K)`; classification never normalizes class IDs or denormalizes logits.
- Primary optimization is cross-entropy. Logistic teacher KL/coefficient losses are optional and failure-safe.

Explicit routing is mandatory. `TRAINING_DATA_FAMILY=synthetic_linear_classification`
maps to `task_objective="inductive_classification"`; regression families map to
`task_objective="inductive_regression"`.

Classification checkpoints use format 5 and include:

```json
{
  "checkpoint_format_version": 5,
  "cfg": {
    "model_arch_version": "model4",
    "task_objective": "inductive_classification"
  },
  "metadata": {
    "classification_path_version": "class_linear_v1",
    "supports_variable_p": true,
    "supports_variable_k": true
  }
}
```

Existing MODEL4 regression checkpoints remain valid and load strictly with no
classification modules instantiated.

---

## Mixed-Categorical Path

Two additional `TRAINING_DATA_FAMILY` values extend MODEL4 with categorical
feature handling:

- `synthetic_linear_regression_mixed_categorical` → regression + entity embeddings
- `synthetic_linear_classification_mixed_categorical` → classification + entity embeddings

**Config gate:** `use_categorical_features=True` instantiates all categorical
modules (`CategoricalTokenEncoder`, `CategoricalStatisticExtractor`,
`CategoricalStatisticEncoder`, and the task-appropriate effect head). When
`False` (the default), the pure-numeric path is preserved exactly—no
categorical parameters are created, no forward-pass branches are entered.

**Entity embedding design:** A fixed 53-token vocabulary covers all categories:

| Token ID | Meaning |
|---|---|
| 0 | `PAD` — padding for variable `p_cat` |
| 1 | `MISSING` — null / missing category |
| 2 | `UNKNOWN` — query category not seen in context |
| 3–52 | Real category IDs (max cardinality 50) |

Cardinalities are bucketed into 6 bins (`≤2, ≤5, ≤10, ≤20, ≤50, >50`) for a
learned cardinality embedding.

---

## Mixed-Categorical Components

| Component | Role |
|---|---|
| `CategoricalTokenEncoder` | Entity embedding → feature-identity positional embed → cardinality bucket embed → Linear projection to `d_phi`. Produces `(n, p_cat, d_phi)`. |
| `CategoricalStatisticExtractor` | No parameters. Context-only per-feature/per-category statistics (count, frequency, mean_y, shrunk effect, missing rate). Always float32. Never sees query labels. |
| `CategoricalStatisticEncoder` | Masked mean-pooling over `(p_cat, max_card)` stat tensor → global MLP → `z_cat_stat (cat_stat_dim,)`. Handles variable `p_cat`/cardinality with same weights. |
| `RegressionCategoryEffectHead` | Per-(feature, query-token) MLP: `[token_embed, z_cat_stat, h_global]` → scalar effect. Summed over `p_cat` → `(m,)`. |
| `ClassificationCategoryEffectHead` | Same pattern but outputs K-dim logit shifts. Reference class (col 0) zeroed. Summed over `p_cat` → `(m, K)`. |

---

## Mixed-Categorical ModelConfig Fields

| Field | Default | Notes |
|---|---|---|
| `use_categorical_features` | `False` | Gate: enables all categorical modules |
| `cat_max_vocab_size` | `53` | 50 real + 3 special tokens (PAD, MISSING, UNKNOWN) |
| `cat_embed_dim` | `32` | Entity embedding dim before projection |
| `cat_feat_id_embed_dim` | `16` | Feature-identity positional embedding dim |
| `cat_cardinality_embed_dim` | `8` | Cardinality bucket embedding dim |
| `cat_stat_dim` | `64` | Output dim of `CategoricalStatisticEncoder` |
| `cat_stat_hidden_dim` | `128` | MLP hidden dim inside `CategoricalStatisticEncoder` |
| `cat_head_hidden_dim` | `64` | MLP hidden dim inside effect heads |
| `cat_effect_aux_loss_weight` | `0.05` | Regression category-effect auxiliary MSE loss |
| `cat_class_effect_aux_loss_weight` | `0.05` | Classification category-effect auxiliary loss |
| `cat_max_p_cat` | `64` | Max number of categorical features (feature-ID embed table size) |

---

## Mixed-Categorical Forward Pass Shape Trace

Regression:

```
Optional additional inputs (when use_categorical_features=True):
  X_cat_train: (n, p_cat)    X_cat_test: (m, p_cat)    cardinalities: (p_cat,)

CategoricalTokenEncoder:
  ctx_cat_tokens  = encode(X_cat_train, cardinalities) → (n, p_cat, d_phi)
  query_cat_tokens = encode(X_cat_test, cardinalities)  → (m, p_cat, d_phi)

CategoricalStatisticExtractor [float32, no_grad, context-only]:
  feat_cat_stats:   (p_cat, max_card, 5)  — count, freq, mean_y, shrunk_effect, missing_rate
  global_cat_stats: (5,)                  — n_ctx, p_cat, max_card, avg_missing, max_missing

CategoricalStatisticEncoder:
  masked_pool(feat_cat_stats) + global_cat_stats → z_cat_stat: (cat_stat_dim,)

RegressionCategoryEffectHead:
  [query_cat_tokens, z_cat_stat.expand, h_global_mean.expand] → (m, p_cat, 1) → sum → cat_effect: (m,)

Fusion with numeric prediction:
  pred_norm = y_coeff_norm + cat_effect  →  (m,)
```

Classification: same pattern but `ClassificationCategoryEffectHead` outputs
`(m, K)` logit shifts added to `logits` before softmax. Reference class
(column 0) is zeroed. Uses non-normalized `y_train` and passes `num_classes=K`.

---

## Mixed-Categorical Checkpoint Metadata

Mixed-categorical checkpoints use format versions 6 (regression) and 7
(classification). Version selection logic:

| `TRAINING_DATA_FAMILY` | Format version |
|---|---|
| `synthetic_linear_regression_mixed_categorical` | 6 |
| `synthetic_linear_classification_mixed_categorical` | 7 |
| `synthetic_linear_classification` | 5 |
| `synthetic_linear_regression` (default) | 4 |

Mixed-categorical checkpoints include additional metadata fields:

```json
{
  "checkpoint_format_version": 6,
  "cfg": {
    "model_arch_version": "model4",
    "task_objective": "inductive_regression",
    "use_categorical_features": true
  },
  "metadata": {
    "feature_contract_version": "mixed_categorical_linear_v1",
    "uses_entity_embeddings": true,
    "uses_context_only_categorical_statistics": true,
    "use_categorical_features": true
  }
}
```

Classification variant (format version 7):

```json
{
  "checkpoint_format_version": 7,
  "cfg": {
    "model_arch_version": "model4",
    "task_objective": "inductive_classification",
    "use_categorical_features": true
  },
  "metadata": {
    "feature_contract_version": "mixed_categorical_linear_v1",
    "uses_entity_embeddings": true,
    "uses_context_only_categorical_statistics": true,
    "use_categorical_features": true
  }
}
```

---

## Six-Step Ablation Ladder

| Step | Config | Key overrides from MODEL4 defaults |
|---|---|---|
| 1 `baseline` | Pure MODEL3 behavior | `use_linear_stats=False`, `use_coefficient_head=False`, `use_lambda_head=False`, `use_ridge_expert=True` |
| 2 `stats_fusion` | Linear stats + fusion, no coeff head | `use_coefficient_head=False`, `use_lambda_head=False` |
| 3 `coeff_head_no_aux` | Coefficient head, no auxiliary losses | `beta_aux_loss_weight=0`, `pred_aux_loss_weight=0`, `cos_aux_loss_weight=0` |
| 4 `coeff_tier1_only` | + Tier 1 (scale-invariant coeff MSE) | `pred_aux_loss_weight=0`, `cos_aux_loss_weight=0` |
| 5 `coeff_tier1_tier2` | + Tier 2 (NMSE) | `cos_aux_loss_weight=0` |
| 6 `full` | All tiers + lambda head (MODEL4 defaults) | — |

Primary metric: median `error_ratio_vs_ridge` on regime A (linear iid). Target: ratio approaching 1.0.

---

## README Invocation Reference

```bash
# MODEL4 default (all new features on — no overrides needed)
BEST_CONFIG='{}' python -m src.train

# Ablation: MODEL3 behavior (explicitly opt back)
BEST_CONFIG='{"use_linear_stats":false,"use_coefficient_head":false,"use_lambda_head":false,"use_ridge_expert":true}' python -m src.train

# Ablation: coefficient head only, no teacher losses
BEST_CONFIG='{"use_lambda_head":false,"beta_aux_loss_weight":0,"pred_aux_loss_weight":0,"cos_aux_loss_weight":0}' python -m src.train

# HPO with MODEL4 linear_stats sweep (primary MODEL4 sweep mode)
HPO_SWEEP_MODE=linear_stats python -m src.hpo
```
