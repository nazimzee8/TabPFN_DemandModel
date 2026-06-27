# MODEL5-LBACNP: DeepSet Architecture and Single-Sweep HPO

**Status:** Implemented — fully wired in Phases A–F.
**Predecessor:** MODEL4 (Linear-Statistic-Aware DeepSet ICL, `model_arch_version="model4"`)
**Target file:** `src/model/model.py` (architecture) + `src/model/hpo.py` / `src/model/train.py` (training)

**Implementation anchors:**
- Config: `src/model/model.py:143` — `use_lbacnp`, `lbacnp_latent_dim`, `lbacnp_n_blocks`, etc.
- Modules: `model.py:396` `NumericPLEEncoder`, `:523` `LatentBottleneckACNP`, `:564` `RegressionDistributionHead`
- Forward paths: `model.py:1873` `forward_regression_lbacnp`, `:1995` `forward_classification_lbacnp`
- HPO search space: `src/model/hpo.py:797` — `lbacnp_model` branch, per-family flag routing from `spec.is_nonlinear`
- Arch version propagation: `scripts/jobs/run_hpo_job.py:133` — auto-sets `model5_lbacnp` for `lbacnp_model` mode
- Checkpoint naming: `scripts/jobs/run_model_training_job.py:534-538` — 4 names unchanged from MODEL4

---

## Repository Reconciliation

> **Read this section before any implementation.** The master-plan prose was written before a
> full code audit. The table below lists every place the spec's approximations differ from the
> actual repository. Each correction is also applied in-line throughout the document.

| Spec says | Repository reality (verified) | Evidence |
|---|---|---|
| `scripts/jobs/hpo.py`, `scripts/jobs/train.py` | **`src/model/hpo.py`**, **`src/model/train.py`** — these are the MLJob entrypoints, PUT to `@MODEL_STAGE/scripts/`. `scripts/jobs/` holds only the `run_*.py` SPCS submission wrappers. | `hpo.py:56`, `train.py:151` |
| "Add `model_arch_version` field" | **Already exists and extended.** `model.py:91`: `model_arch_version: str = "model4"`, whitelist at `model.py:184-188` now includes `{"model3", "model4", "model5_lbacnp"}`. | `model.py:91, :184-188` |
| `MODEL_ARCH_VERSION = "model4"` hardcoded (one place) | Hardcoded in **three** places: `src/model/hpo.py:56`, `src/model/train.py:151`, `scripts/jobs/run_training_job.py:809`. All three become `os.environ.get("MODEL_ARCH_VERSION", "model4")`. | — |
| Checkpoint format versions 8–15 (one per family) | Loader (`evaluate.py:433-464`) is **version-agnostic** — it reconstructs the model from the embedded `cfg` dict. Current scheme: flat 4/5/6/7. **Use single `checkpoint_format_version=8` + rich `metadata` block** (user-confirmed). | `evaluate.py:433-464`; `constants.py:123-124` |
| `synthetic_nonlinear_regression` (numeric nonlinear) | Registered as **`synthetic_regression_nonlinear`** (`task_routing.py:20`, `:69-78`). Word order is reversed. The *mixed* variant is `synthetic_nonlinear_regression_mixed_categorical` (`constants.py:131`). No `synthetic_nonlinear_regression` key exists — using it raises `ValueError`. | `task_routing.py:20, :69-78`; `constants.py:131` |
| Regression GaussianNLL / distribution head already present | **No NLL path exists.** `src/model/train.py:502,590` regression loss is MSE + optional Huber + aux losses only. The distribution head (`RegressionDistributionHead`) and `GaussianNLL` are genuinely new. | `train.py:502, :590, :607-642` |
| `HPO_DATA_SPEC.is_nonlinear` drives HPO branching | `src/model/hpo.py` branches on `HPO_SWEEP_MODE.startswith("nonlinear_model")` (lines `:102-103`, `:1437`). The new `lbacnp_model` mode will **not** match that prefix. LBACNP HPO must use `spec.is_nonlinear` (from `get_training_data_spec()`) for nonlinear detection. | `hpo.py:102-103, :1437` |
| `TrainingDataSpec.is_mixed` flag | **No such field.** `task_routing.py:31-48` has only `is_classification` and `is_nonlinear`. Mixedness is derived as `"mixed" in spec.family`. | `task_routing.py:31-48` |
| Sweep-mode set lives in one place | Set duplicated in **three** places: `hpo.py:71-76`, `run_hpo_job.py:36-41`, and alias dicts at `hpo.py:64-70`, `run_hpo_job.py:29-35`, `run_model_training_job.py:58-64`. All six locations need `lbacnp_model` + aliases. | — |
| Checkpoint name keyed on sweep mode | Name selected at `run_model_training_job.py:515-520` from `(is_classification, is_nonlinear_config)` only. A 5th LBACNP-specific name requires a new branch there. | `run_model_training_job.py:515-520` |
| 80/20 context/query split is inside the model | Split happens **at data-generation time** (`dgp_helpers.py:2353-2355`). The model receives pre-split `X_train`/`y_train`/`X_test` tensors. | `dgp_helpers.py:2353-2355` |

**Existing MODEL4 primitives to reuse** (all in `src/model/model.py`):

| Primitive | Line | Used by MODEL5 |
|---|---|---|
| `build_mlp` | `:19` | MLPs in all new heads |
| `MAB` (Multi-head Attention Block) | `:287` | `LatentBottleneckBlock` cross-attention |
| `SAB` (Set Attention Block) | `:310` | Latent self-attention |
| `CategoricalTokenEncoder` | `:1753` | Categorical feature tokens (unchanged) |
| `CategoricalStatisticExtractor` | `:1810` | Categorical stats (unchanged) |
| `CategoricalStatisticEncoder` | `:1932` | Categorical encoding (unchanged) |
| `LinearStatisticExtractor` | `:517` | Linear task primary path (unchanged) |
| `LinearStatisticEncoder` | `:579` | Linear task primary path (unchanged) |
| `FusionModule` | `:611` | Linear task primary path (unchanged) |
| `CoefficientHead` | `:654` | Linear regression primary path (unchanged) |
| `LambdaHead` | `:685` | Linear regression primary path (unchanged) |
| `ClassLabelEncoder` | `:709` | Classification primary path (unchanged) |
| `ClassificationStatisticExtractor` | `:736` | Classification primary path (unchanged) |
| `ClassificationStatisticEncoder` | `:838` | Classification primary path (unchanged) |
| `ClassFusionModule` | `:870` | Classification primary path (unchanged) |
| `ClassCoefficientHead` | `:874` | Classification primary path (unchanged) |
| `ClassBiasHead` | `:909` | Classification primary path (unchanged) |

---

## 0. Executive Summary

Implement a new non-breaking architecture path:

```text
MODEL5-LBACNP = MODEL4 linear-statistical competence
              + deterministic Latent Bottlenecked Attentive Conditional Neural Process path
              + context-only PLE numeric encoding
              + existing entity-embedding categorical path
              + single conditional HPO sweep
```

MODEL5 must **not replace MODEL4**. MODEL4 remains the trusted linear-statistic baseline for linear regression/classification. MODEL5 adds a deterministic latent bottleneck attention path that performs query-conditioned prediction.

For linear tasks, MODEL5 should preserve MODEL4's OLS/Ridge/logistic-style structure and use LBACNP as a residual correction. For nonlinear tasks, MODEL5 should use LBACNP as the primary prediction path.

The HPO strategy should change from multiple sequential sweeps into a **single conditional sweep**:

```text
HPO_SWEEP_MODE = "lbacnp_model"
```

This single sweep tunes only the critical LBACNP, PLE, residual-gating, optimizer, and task-specific loss parameters while keeping large architecture dimensions fixed.

---

## 1. Current System Constraints to Preserve

Before implementing MODEL5, verify these constraints from the repository:

1. MODEL4 is the current architecture version for new training runs.
   *(Verified: `model_arch_version: str = "model4"` at `src/model/model.py:91`)*
2. MODEL4 uses:
   - `LinearStatisticExtractor` (`model.py:517`)
   - `LinearStatisticEncoder` (`model.py:579`)
   - `FusionModule` (`model.py:611`)
   - `CoefficientHead` (`model.py:654`)
   - `LambdaHead` (`model.py:685`)
   - classification-specific class-statistic and coefficient/bias heads (`model.py:709–909`)
   - mixed-categorical entity embedding modules (`model.py:1753–1932`)
3. Existing checkpoint formats must remain loadable:
   - format 4: linear regression
   - format 5: linear classification
   - format 6: mixed-categorical regression
   - format 7: mixed-categorical classification
   *(Verified: `train.py:1533-1544`, `constants.py:123-124`. The loader at `evaluate.py:433-464` is
   version-agnostic and reconstructs from the embedded `cfg` dict.)*
4. Existing training-data family routing must remain intact.
   *(Verified: `src/model/task_routing.py:62-142` — all 8 family specs.)*
5. Existing 80/20 context/query semantics must remain intact.
   *(Verified: split is applied at data-generation time in `dgp_helpers.py:2353-2355`. The model
   receives pre-split tensors; no change required to the model interface.)*
6. Existing SQL procedures in `/sql` must continue to route all training/evaluation families correctly.
7. Existing Snowflake MLJob orchestration must still submit pretrain, HPO, and final training jobs.
8. `use_lbacnp=False` must preserve existing MODEL4 behavior.

---

## 2. Core Architecture Change

### 2.1 Current MODEL4 pattern

Current MODEL4 roughly follows this structure:

```text
(X_context, y_context, X_query)
    -> feature/cell encoding
    -> exchangeable matrix blocks
    -> sample/feature pooling
    -> global or fused representation
    -> coefficient/classification/prediction head
```

This is strong for global task-level inference but weak for the desired query-specific behavior:

```text
For each query sample, infer the most relevant local task representation from the context set.
```

### 2.2 Desired MODEL5 pattern

MODEL5 should follow:

```text
Raw features
    -> numeric PLE / categorical entity embeddings
    -> context/query row encoders
    -> latent bottleneck context compression
    -> query-to-latent cross-attention
    -> task-specific prediction head
```

Core equations:

```text
LEMB_i = SelfAttention(CrossAttention(LEMB_{i-1}, CONTEXT))
QEMB_i = CrossAttention(QEMB_{i-1}, LEMB_i)
Output = PredictionHead(QEMB_k)
```

Where:

```text
k = number of LBACNP attention blocks
```

This is deterministic. The latent vectors are learned attention slots, not stochastic latent variables.

---

## 3. Feature Encoding Order

The correct order is:

```text
1. Fit context-only normalization / PLE bin boundaries
2. Encode numeric scalar features with PLE
3. Encode categorical features with entity embeddings
4. Merge numeric and categorical feature tokens
5. Build context row embeddings using X_context and y_context
6. Build query row embeddings using X_query only
7. Apply latent bottleneck attention
8. Decode query embeddings into predictions
```

### 3.1 Numeric scalar features: PLE

Add:

```python
class NumericPLEEncoder(nn.Module):
    ...
```

Input:

```text
X_context: (n, p)
X_query:   (m, p)
```

Output:

```text
E_context_num: (n, p, d_phi)
E_query_num:   (m, p, d_phi)
```

Recommended defaults:

```text
use_numeric_ple = True for nonlinear tasks
use_numeric_ple = HPO choice(False, True) for linear tasks
ple_num_bins = 32
ple_projection_dim = d_phi = 128
ple_boundary_mode = "context_quantile"
ple_target_aware = False
```

Important design choice:

```text
Do not make ple_num_bins equal to d_phi by default.
```

Use:

```text
PLE(x_ij) in R^T -> Linear(T, d_phi)
```

Default:

```text
T = 32
```

HPO candidates:

```text
ple_num_bins ∈ {16, 32, 64}
```

Optional later candidate:

```text
ple_num_bins = 128
```

Do not use query-aware bins. Do not use query labels. Target-aware PLE must remain disabled until context-only supervised binning is implemented and tested.

### 3.2 Categorical features: entity embeddings

Do not replace existing entity embeddings with one-hot encoding.

Use existing categorical design (reuses `CategoricalTokenEncoder` at `model.py:1753`):

```text
category id embedding
+ feature identity embedding
+ cardinality bucket embedding
-> projection to d_phi
```

For native categorical features:

```text
X_cat -> CategoricalTokenEncoder -> categorical feature tokens
```

For already one-hot-encoded features:

```text
Treat one-hot columns as binary numeric features through NumericPLEEncoder for v1.
```

---

## 4. Latent Bottleneck Design

Defaults:

```text
d_phi = 128
D_L = d_phi = 128
L_max = 64
L_eff = min(64, floor(0.25 * n_context))
lbacnp_n_blocks = 4
lbacnp_heads = 4
```

Implementation (reuses `MAB` at `model.py:287` and `SAB` at `model.py:310`):

```python
class LatentBottleneckBlock(nn.Module):
    def forward(self, latent, context, query):
        latent = self.context_cross_attn(latent, context)  # MAB(latent, context)
        latent = self.latent_self_attn(latent)              # SAB(latent)
        query = self.query_cross_attn(query, latent)        # MAB(query, latent)
        return latent, query
```

Use learned latent seed bank:

```text
latent_seed_bank: (L_max, D_L)
```

At runtime:

```text
L_eff = min(lbacnp_latent_max, floor(lbacnp_latent_frac * n_context))
latent = latent_seed_bank[:L_eff]
```

Complexity target:

```text
Context -> latent cross-attention: O(n * L_eff)
Latent self-attention:           O(L_eff^2)
Query -> latent cross-attention:  O(m * L_eff)
```

Avoid direct full query-context attention:

```text
Do not build O(m * n) query-context attention in MODEL5 v1.
```

---

## 5. Task-Specific Behavior

### 5.1 Linear Regression

Preserve MODEL4 coefficient path as primary:

```text
y_linear = x_query @ beta_hat
```

Add LBACNP residual correction:

```text
y_hat = y_linear + gate(query, context) * delta_lbacnp(query, context)
```

Initialize residual gate close to zero:

```text
linear_lbacnp_residual_gate_init ∈ {-4.0, -3.0, -2.0}
recommended default = -3.0
```

Preserve MODEL4 auxiliary losses:

```text
beta_aux_loss_weight
pred_aux_loss_weight
cos_aux_loss_weight
lambda_aux_loss_weight
```

Add residual penalty:

```text
lbacnp_residual_penalty_weight ∈ {0.0, 1e-4, 1e-3}
```

Regression output:

```text
mu, log_var = RegressionDistributionHead(QEMB_k)
```

For linear regression, the default final prediction may still optimize MSE against the query target. NLL can be added as an auxiliary objective if distributional calibration is desired.

### 5.2 Linear Classification

Preserve MODEL4 class coefficient/bias path as primary:

```text
logits_linear = X_query @ W_hat + b_hat
```

Add LBACNP residual logits:

```text
logits = logits_linear + gate(query, context) * delta_logits_lbacnp
```

Preserve (`model.py:709-909`):

```text
ClassLabelEncoder
ClassificationStatisticExtractor
ClassificationStatisticEncoder
ClassFusionModule
ClassCoefficientHead
ClassBiasHead
```

Primary objective:

```text
cross_entropy
```

Optional objectives:

```text
class_logit_kl_loss_weight
class_coef_aux_loss_weight
class_margin_aux_loss_weight
class_prior_aux_loss_weight
class_label_smoothing
class_imbalance_reweighting
```

### 5.3 Nonlinear Regression

LBACNP should be primary:

```text
mu, log_var = LBACNPRegressionHead(QEMB_k)
```

Default nonlinear config:

```text
use_lbacnp = True
use_numeric_ple = True
use_linear_stats = False or auxiliary_only
use_coefficient_head = False
use_lambda_head = False
regression_distribution_head = True
```

Training objective:

```text
mse_loss_weight * MSE(mu, y_query)
+ nll_loss_weight * GaussianNLL(mu, log_var, y_query)
```

If `mu_clean` is available in nonlinear training data, optionally add:

```text
mu_clean auxiliary MSE
```

But do not require it for v1 if loaders currently expose only `y_test`.

### 5.4 Nonlinear Classification

LBACNP should be primary:

```text
logits = LBACNPClassificationHead(QEMB_k)
```

Default nonlinear classification config:

```text
use_lbacnp = True
use_numeric_ple = True
use_classification_path = True
use_classification_stats = False or auxiliary_only
use_class_coefficient_head = False
use_class_bias_head = False
```

Training objective:

```text
cross_entropy
+ optional DGP teacher KL if teacher probabilities/logits are available
+ optional calibration loss
```

---

## 6. Required ModelConfig Additions

> **Note:** `model_arch_version` **already exists** in `ModelConfig` at `src/model/model.py:91`
> (default `"model4"`). The whitelist at `model.py:184-188` now allows `{"model3", "model4", "model5_lbacnp"}`
> — extended as part of MODEL5-LBACNP implementation.

Add to `ModelConfig` (`src/model/model.py`):

```python
# Architecture selector (existing field — extend whitelist to include "model5_lbacnp")
# model_arch_version: str = "model4"  # already at model.py:91

# LBACNP controls
use_lbacnp: bool = False
lbacnp_latent_dim: int = 128
lbacnp_latent_max: int = 64
lbacnp_latent_frac: float = 0.25
lbacnp_n_blocks: int = 4
lbacnp_heads: int = 4
lbacnp_dropout: float = 0.1
lbacnp_output_mode: str = "last"  # "last" only for v1

# Numeric PLE controls
use_numeric_ple: bool = False
ple_num_bins: int = 32
ple_boundary_mode: str = "context_quantile"
ple_target_aware: bool = False
ple_projection_dim: int = 128

# Linear residual controls
linear_lbacnp_residual: bool = True
linear_lbacnp_residual_gate_init: float = -3.0
lbacnp_residual_penalty_weight: float = 1e-4

# Regression distribution controls
regression_distribution_head: bool = True
nll_loss_weight: float = 0.0
mse_loss_weight: float = 1.0
min_log_var: float = -8.0
max_log_var: float = 4.0
```

Add to `__post_init__` (`src/model/model.py`, after the existing validation block):

```python
if self.use_lbacnp:
    if self.lbacnp_latent_dim != self.d_phi:
        raise ValueError("MODEL5 v1 requires lbacnp_latent_dim == d_phi")
    if self.lbacnp_latent_dim % self.lbacnp_heads != 0:
        raise ValueError("lbacnp_latent_dim must be divisible by lbacnp_heads")
    if not (0.0 < self.lbacnp_latent_frac <= 1.0):
        raise ValueError("lbacnp_latent_frac must be in (0, 1]")
    if self.lbacnp_latent_max < 1:
        raise ValueError("lbacnp_latent_max must be positive")
    if self.lbacnp_n_blocks < 1:
        raise ValueError("lbacnp_n_blocks must be positive")

if self.use_numeric_ple:
    if self.ple_num_bins < 2:
        raise ValueError("ple_num_bins must be >= 2")
    if self.ple_boundary_mode not in {"context_quantile", "fixed", "context_supervised"}:
        raise ValueError("Invalid ple_boundary_mode")
    if self.ple_target_aware:
        raise ValueError("ple_target_aware must remain False until context-only supervised PLE is implemented")
```

Also extend the `model_arch_version` whitelist at `model.py:184-188`:

```python
# Before (model.py:184-188):
if self.model_arch_version not in {"model3", "model4"}:
    raise ValueError(...)

# After:
if self.model_arch_version not in {"model3", "model4", "model5_lbacnp"}:
    raise ValueError(...)
```

---

## 7. Required New Modules

Add these modules to `src/model/model.py` or a clean submodule imported by `model.py`:

```python
class NumericPLEEncoder(nn.Module)      # context-quantile PLE + Linear(T, d_phi); reuses build_mlp(:19)
class ContextPairEncoder(nn.Module)     # (x_ctx, y_ctx) -> context row embedding; reuses MAB(:287)
class QueryFeatureEncoder(nn.Module)    # x_query -> query row embedding; reuses MAB(:287)
class LatentBottleneckBlock(nn.Module)  # cross-attn(latent←context) + SAB(latent) + cross-attn(query←latent); reuses MAB(:287), SAB(:310)
class LatentBottleneckACNP(nn.Module)   # stack of LatentBottleneckBlocks with learned seed bank
class RegressionDistributionHead(nn.Module)      # QEMB -> (mu, log_var); reuses build_mlp(:19)
class ClassificationLBACNPHead(nn.Module)        # QEMB -> logits; reuses build_mlp(:19)
class ResidualGate(nn.Module)           # sigmoid gate initialized near zero; reuses build_mlp(:19)
```

Reuse existing primitives where possible:

```python
MAB          # model.py:287 — multi-head attention block
SAB          # model.py:310 — set attention block
build_mlp    # model.py:19  — shared MLP builder
CategoricalTokenEncoder       # model.py:1753 — entity embeddings (unchanged)
CategoricalStatisticExtractor # model.py:1810 — context-only cat stats (unchanged)
CategoricalStatisticEncoder   # model.py:1932 — cat encoding (unchanged)
```

Do not duplicate working MODEL4 logic unnecessarily.

---

## 8. Forward Pass Integration

Do not delete current MODEL4 forward paths (`forward_regression` at `model.py:1315`, `forward_classification` at `model.py:1562`).

Add routing alongside existing paths in `DeepSetICLModel.forward` (`model.py:1271`):

```python
if cfg.use_lbacnp:
    return self.forward_regression_lbacnp(...)
else:
    return existing MODEL4 regression path  # forward_regression unchanged
```

and:

```python
if cfg.use_lbacnp:
    return self.forward_classification_lbacnp(...)
else:
    return existing MODEL4 classification path  # forward_classification unchanged
```

For linear tasks:

```text
MODEL4 coefficient/stat path remains primary.
LBACNP is residual.
```

For nonlinear tasks:

```text
LBACNP is primary.
MODEL4 linear-stat path is disabled or auxiliary only.
```

---

## 9. HPO Strategy: Single Conditional Sweep

### 9.1 New canonical HPO mode

Add:

```text
HPO_SWEEP_MODE = "lbacnp_model"
```

Add aliases (in `src/model/hpo.py:64-70` **and** `scripts/jobs/run_hpo_job.py:29-35`):

```python
_HPO_SWEEP_MODE_ALIASES = {
    "ridge_residual": "linear_model",
    "architecture": "linear_model_architecture",
    "linear_stats": "linear_model",
    "nonlinear_meta": "nonlinear_model",
    "nonlinear_architecture": "nonlinear_model_architecture",
    "attentive_cnp": "lbacnp_model",    # new
    "model5": "lbacnp_model",           # new
}
```

Allowed modes (in `src/model/hpo.py:71-76` **and** `scripts/jobs/run_hpo_job.py:36-41`
**and** `scripts/jobs/run_model_training_job.py:58-64`):

```python
_ALLOWED_HPO_SWEEP_MODES = {
    "linear_model",
    "linear_model_architecture",
    "nonlinear_model",
    "nonlinear_model_architecture",
    "lbacnp_model",    # new
}
```

Update both `src/model/hpo.py` and `scripts/jobs/run_hpo_job.py` and `scripts/jobs/run_model_training_job.py`.

### 9.2 Keep architecture fixed in v1

Do not sweep:

```text
d_phi
d_rho
pool
n_sab_feat
lbacnp_latent_dim
PLE boundary mode
target-aware PLE
lbacnp_n_blocks > 4
lbacnp_latent_max > 64
```

Fixed defaults:

```text
d_phi = 128
d_rho = 256
pool = "pna"
n_sab_feat = 1
lbacnp_latent_dim = 128
```

This preserves a single sweep and avoids the need for a separate architecture HPO.

### 9.3 Recommended v1 search space

Add to `build_hpo_search_space(tune, baseline_config=None)` in `src/model/hpo.py:717`:

> **Implementation note:** `is_nonlinear` must be sourced from `HPO_DATA_SPEC.is_nonlinear`
> (i.e. `get_training_data_spec(HPO_TRAINING_DATA_FAMILY).is_nonlinear`) — **not** from
> `HPO_SWEEP_MODE.startswith("nonlinear_model")`, which will not match `"lbacnp_model"`.
> See `hpo.py:102-103, :1437` for the existing prefix checks that need a companion
> `is_nonlinear` variable in the `lbacnp_model` branch.

```python
if HPO_SWEEP_MODE == "lbacnp_model":
    is_classification = HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
    is_nonlinear = HPO_DATA_SPEC.is_nonlinear   # from get_training_data_spec(), NOT startswith check
    is_mixed = "mixed" in HPO_TRAINING_DATA_FAMILY

    return {
        # Optimizer
        "lr": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.0, 0.20),
        "lambda_l1": tune.choice([0.0, 1e-6, 1e-5]),

        # Fixed model identity
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_sab_feat": 1,
        "model_family": MODEL_FAMILY,
        "model_arch_version": "model5_lbacnp",
        "model_design_pattern": MODEL_DESIGN_PATTERN,
        "task_objective": HPO_TASK_OBJECTIVE,
        "hpo_sweep_mode": "lbacnp_model",

        # LBACNP
        "use_lbacnp": True,
        "lbacnp_latent_dim": 128,
        "lbacnp_latent_max": tune.choice([32, 64]),
        "lbacnp_latent_frac": tune.choice([0.125, 0.25, 0.5]),
        "lbacnp_n_blocks": tune.choice([2, 4]),
        "lbacnp_heads": tune.choice([4, 8]),
        "lbacnp_dropout": tune.uniform(0.0, 0.15),
        "lbacnp_output_mode": "last",

        # PLE
        "use_numeric_ple": tune.choice([False, True]) if not is_nonlinear else True,
        "ple_num_bins": tune.choice([16, 32, 64]),
        "ple_boundary_mode": "context_quantile",
        "ple_target_aware": False,
        "ple_projection_dim": 128,

        # Linear-stat path
        "use_linear_stats": not is_nonlinear,
        "use_coefficient_head": (not is_nonlinear and not is_classification),
        "use_lambda_head": (not is_nonlinear and not is_classification),

        # Linear residual behavior
        "linear_lbacnp_residual": not is_nonlinear,
        "linear_lbacnp_residual_gate_init": tune.choice([-4.0, -3.0, -2.0]),
        "lbacnp_residual_penalty_weight": tune.choice([0.0, 1e-4, 1e-3]),

        # Regression distribution head
        "regression_distribution_head": not is_classification,
        "nll_loss_weight": tune.choice([0.0, 0.25, 0.5, 1.0]) if not is_classification else 0.0,
        "mse_loss_weight": 1.0 if not is_classification else 0.0,

        # Classification path
        "use_classification_path": is_classification,
        "use_class_label_embeddings": is_classification,
        "use_classification_stats": is_classification and not is_nonlinear,
        "use_class_stat_fusion": is_classification and not is_nonlinear,
        "use_class_coefficient_head": is_classification and not is_nonlinear,
        "use_class_bias_head": is_classification and not is_nonlinear,
        "use_class_residual_head": False,
        "max_num_classes": 10,
        "class_ce_loss_weight": 1.0,
        "class_logit_kl_loss_weight": tune.choice([0.0, 0.025, 0.05]) if is_classification else 0.0,
        "class_label_smoothing": tune.choice([0.0, 0.05]) if is_classification else 0.0,
        "class_imbalance_reweighting": tune.choice([False, True]) if is_classification else False,

        # Categorical path
        "use_categorical_features": is_mixed,
        "cat_embed_dim": tune.choice([16, 32]) if is_mixed else 32,
        "cat_stat_dim": tune.choice([32, 64]) if is_mixed else 64,
        "cat_head_hidden_dim": tune.choice([32, 64]) if is_mixed else 64,
    }
```

### 9.4 Trial budget

Keep initial LBACNP HPO compatible with the current HPO capacity:

```text
NUM_TRIALS = 20
TRIAL_MAX_EPOCHS = 30
HPO_SPLIT_LIMITS = {"train": 200, "val": 40}
```

Later expansion can increase to 40-60 trials only after memory and stability gates pass.

---

## 10. Required HPO/Training File Changes

> **Path correction:** The entrypoints are `src/model/hpo.py` and `src/model/train.py`,
> **not** `scripts/jobs/hpo.py` / `scripts/jobs/train.py`. The `scripts/jobs/` directory
> contains only the Snowflake stored-procedure submission wrappers (`run_*.py`).

### 10.1 `src/model/hpo.py`

Current state: `MODEL_ARCH_VERSION = "model4"` hardcoded at `hpo.py:56`.

Implement:

1. Replace hardcoded `MODEL_ARCH_VERSION = "model4"` at `hpo.py:56` with:
   ```python
   MODEL_ARCH_VERSION = os.environ.get("MODEL_ARCH_VERSION", "model4")
   ```
2. Add `"lbacnp_model"` to `_CANONICAL_HPO_SWEEP_MODES` at `hpo.py:71-76`.
3. Add aliases `"attentive_cnp"` and `"model5"` to `_HPO_SWEEP_MODE_ALIASES` at `hpo.py:64-70`.
4. Add `"lbacnp_model"` branch in `build_hpo_search_space` at `hpo.py:717` (see §9.3).
5. Parse LBACNP fields inside `ray_trainable` (`hpo.py:1039`, within `_build_ray_trainable`).
6. Pass LBACNP and PLE fields into `ModelConfig` at `hpo.py:1158-1233`.
7. Ensure `best_config.json` includes all LBACNP keys.
8. Ensure `_meta` block (written at `hpo.py:1697-1735`) records:
   ```text
   training_data_family
   model_arch_version
   hpo_sweep_mode
   HPO metric name/value
   pretrain checkpoint path if used
   ```

### 10.2 `scripts/jobs/run_hpo_job.py`

Current state: allowed modes at `run_hpo_job.py:36-41`; aliases at `:29-35`.

Implement:

1. Add `"lbacnp_model"` to `_ALLOWED_HPO_SWEEP_MODES` at `run_hpo_job.py:36-41`.
2. Add aliases `"attentive_cnp"` and `"model5"` to `_HPO_SWEEP_MODE_ALIASES` at `:29-35`.
3. Pass `MODEL_ARCH_VERSION` into the HPO MLJob env-vars in `_run_hpo_impl` at `run_hpo_job.py:128-139`:
   ```python
   "MODEL_ARCH_VERSION": "model5_lbacnp" if canonical_mode == "lbacnp_model" else DEFAULT_MODEL_ARCH_VERSION,
   ```
4. Default `MODEL_ARCH_VERSION` to `"model5_lbacnp"` only when `HPO_SWEEP_MODE=lbacnp_model`; otherwise keep `"model4"`:
   ```python
   DEFAULT_MODEL_ARCH_VERSION = os.getenv(
       "MODEL_ARCH_VERSION",
       "model5_lbacnp" if DEFAULT_HPO_SWEEP_MODE == "lbacnp_model" else "model4",
   )
   ```

### 10.3 `src/model/train.py`

Current state: `MODEL_ARCH_VERSION = "model4"` hardcoded at `train.py:151`. Regression loss is MSE + Huber + aux only (`train.py:502, :590`). No GaussianNLL path.

Implement:

1. Replace hardcoded `MODEL_ARCH_VERSION = "model4"` at `train.py:151` with:
   ```python
   MODEL_ARCH_VERSION = os.environ.get("MODEL_ARCH_VERSION", "model4")
   ```
2. Parse all LBACNP and PLE fields from `BEST_CONFIG` (`train.py:1184-1386`).
3. Pass them into `ModelConfig` at `train.py:1392-1465`.
4. Add regression loss handling for distribution outputs (`train.py:run_epoch`, `train.py:502`):
   - if model returns `dict` with `mu`/`log_var`, compute MSE (`mse_loss_weight`) and optional NLL (`nll_loss_weight`)
   - preserve old tensor-output path for MODEL4 (`use_lbacnp=False` falls through unchanged)
5. Add classification loss handling for LBACNP logits:
   - existing `compute_classification_losses` (`train.py:818-826`, imported `train.py:50`) should work if output dict includes `logits`, `probs`, and `pred`
6. Preserve mixed-categorical kwargs.
7. Preserve DDP behavior (`train.py:1479-1480`).
8. Preserve checkpoint upload behavior (`train.py:1607-1624`).

### 10.4 `scripts/jobs/run_model_training_job.py`

Current state: `MODEL_ARCH_VERSION` is NOT passed (documented at `run_model_training_job.py:641`); checkpoint name derived from `(is_classification, is_nonlinear_config)` at `:515-520`; alias dict at `:58-64`.

Implement:

1. Add `"lbacnp_model"` to the allowed-modes set at `run_model_training_job.py:58-64`.
2. Detect `model_arch_version` and `use_lbacnp` from `best_config.json` (parsed at `:426-428`).
3. Set `MODEL_ARCH_VERSION` env var for final training (add to `env_vars` dict at `:512-535`).
4. For LBACNP, set `PRETRAIN_LOAD_POLICY = "load_compatible_backbone"` unless explicitly overridden.
5. Checkpoint output names (`run_model_training_job.py:515-520`) should remain family-aware:
   ```text
   linear regression          -> best_regression.pt
   linear classification      -> best_classification.pt
   nonlinear regression       -> best_nonlinear.pt
   nonlinear classification   -> best_nonlinear_cls.pt
   ```
   If desired, add MODEL5-specific names only after evaluation scripts support them:
   ```text
   best_model5_regression.pt
   best_model5_classification.pt
   ```
   Do not change checkpoint names prematurely if downstream SQL/evaluation expects current names.
   A 5th LBACNP-specific name requires a **new branch** at `run_model_training_job.py:515-520`.
6. Note: `is_nonlinear_config` at `:437-439` is currently `use_latent_ridge_expert OR hpo_sweep_mode.startswith("nonlinear_")`. With `lbacnp_model` this predicate will be False. Add: `or best_config.get("use_lbacnp") and spec.is_nonlinear`.

### 10.5 `scripts/jobs/run_pretrain_job.py`

For LBACNP:

1. Do not require pretraining for every latent configuration.
2. Use one compatible backbone pretrain.
3. Load compatible tensors only into LBACNP trials.
4. Cold-start new LBACNP/PLE/distribution-head modules.

Avoid a combinatorial pretrain matrix.

---

## 11. Pretrain Policy for MODEL5

Do not create one checkpoint per LBACNP configuration.

Use:

```text
PRETRAIN_LOAD_POLICY = "load_compatible_backbone"
```

Policy:

```text
Load MODEL4-compatible backbone tensors by name and shape.
Skip new LBACNP/PLE/distribution-head tensors.
Cold-start skipped modules.
```

Rationale:

```text
MODEL5 has too many architecture knobs to require gate-style pretraining per candidate.
A single compatible backbone warm start preserves pipeline simplicity and enables one HPO sweep.
```

---

## 12. Checkpoint Metadata Requirements

> **Versioning decision:** The checkpoint loader (`evaluate.py:433-464`) is version-agnostic —
> it reconstructs the model entirely from the embedded `cfg` dict. The current scheme uses a
> flat integer (4/5/6/7) and the load side never branches on the version. Therefore MODEL5 uses
> a **single `checkpoint_format_version=8`** for all families, with family-specific details
> encoded in `metadata`. The 8–15 per-family split (as in the original spec) is not necessary
> and would add 8 constants to maintain with no loader benefit.

MODEL5 checkpoints should include:

```json
{
  "checkpoint_format_version": 8,
  "cfg": {
    "model_arch_version": "model5_lbacnp",
    "use_lbacnp": true,
    "use_numeric_ple": true
  },
  "metadata": {
    "model_arch_version": "model5_lbacnp",
    "model_family": "market_exchangeable_icl",
    "model_design_pattern": "inductive_forecasting",
    "task_objective": "inductive_regression or inductive_classification",
    "training_data_family": "...",
    "hpo_sweep_mode": "lbacnp_model",
    "uses_lbacnp": true,
    "uses_numeric_ple": true,
    "ple_boundary_mode": "context_quantile",
    "uses_context_only_ple_boundaries": true,
    "uses_stochastic_latent_variables": false,
    "lbacnp_latent_dim": 128,
    "lbacnp_latent_max": "32 or 64",
    "lbacnp_latent_frac": "0.125 or 0.25 or 0.5",
    "lbacnp_n_blocks": "2 or 4",
    "lbacnp_heads": "4 or 8"
  }
}
```

---

## 13. Acceptance Tests

### 13.1 Non-breaking tests

```text
use_lbacnp=False produces identical or tolerance-equivalent outputs to MODEL4.
Existing MODEL4 checkpoints load.
Existing MODEL4 linear regression training still works.
Existing MODEL4 linear classification training still works.
Existing mixed-categorical training still works.
```

### 13.2 Shape tests

Test:

```text
n_context ∈ {32, 64, 128, 512, 1024}
m_query ∈ {1, 8, 64, 256}
p ∈ {1, 4, 16, 64, 128}
p_cat ∈ {0, 1, 4, 16}
K ∈ {2, 3, 5, 10}
```

Expected shapes:

```text
context rows:          (n, d_phi)
query rows:            (m, d_phi)
latent embeddings:     (L_eff, d_phi)
query embeddings:      (m, d_phi)
regression mu:         (m,)
regression log_var:    (m,)
classification logits: (m, K)
```

### 13.3 No-leakage tests

```text
PLE boundaries fit on X_context only.
Changing X_query distribution does not change context PLE boundaries.
Changing y_query does not change PLE boundaries.
Categorical statistics use context only.
Query classification labels are never passed into the model.
```

### 13.4 Permutation tests

```text
Context row shuffle -> same predictions within tolerance.
Numeric feature column permutation with matched query permutation -> same predictions within tolerance.
Categorical feature order behavior explicitly validated with feature-id embeddings.
```

### 13.5 Memory tests

Verify:

```text
MODEL5 does not construct full O(m * n) query-context attention.
Memory scales with L_eff.
Worst-case HPO shape does not OOM on target GPU pool.
```

### 13.6 Linear preservation tests

Regression:

```text
No degradation versus MODEL4/Ridge/OLS on clean linear regimes.
Residual gate remains small on clean linear tasks.
Coefficient cosine similarity remains strong.
```

Classification:

```text
No degradation versus MODEL4/LogisticRegression/RidgeClassifier on clean linear regimes.
Residual logit path does not dominate clean linear classification.
```

### 13.7 Nonlinear improvement tests

Expected improvements on:

```text
piecewise_relu
discontinuous_threshold
rbf_local
smooth_additive
sparse_interact
periodic
heteroskedastic
low_rank_compositional
mixed_categorical_nonlinear
```

---

## 14. SQL Pipeline Verification: 8 SWE Agents

> **Implementation-phase work.** These agents are not dispatched by this design document.
> They execute during the implementation phase, after MODEL5 code changes are complete.

Claude Opus should deploy 8 SWE agents. Each agent owns exactly one pipeline family and verifies the corresponding SQL in `/sql` against MODEL5 changes.

Each agent must produce:

```text
1. SQL files inspected
2. Procedures found
3. Stages found
4. Index tables found
5. Expected training_data_family strings
6. HPO invocation compatibility
7. Pretrain invocation compatibility
8. Final training invocation compatibility
9. Checkpoint output compatibility
10. Required code changes
11. Risks/blockers
12. Patch plan
```

### Agent 1 — Linear Regression Numeric Pipeline

**SQL file:** `sql/linear_regression_numeric_pipeline.sql`

Verify:

```text
training_data_family = synthetic_linear_regression
stage = @META_DATASET_STAGE / @META_DATASET_STAGE/linear/regression/numeric/{split}/
index = META_REGRESSION_DATASET_INDEX
checkpoint = best_regression.pt
HPO_SWEEP_MODE = lbacnp_model supported
MODEL_ARCH_VERSION propagated
run_hpo_pipeline HANDLER = run_hpo_job.run_hpo_pipeline_model_sweep (confirmed)
```

Must ensure MODEL5 linear regression keeps MODEL4 coefficient path primary and LBACNP residual only.

### Agent 2 — Linear Classification Numeric Pipeline

**SQL file:** `sql/linear_classification_numeric_pipeline.sql`

Verify:

```text
training_data_family = synthetic_linear_classification
stage = @META_DATASET_STAGE / @META_DATASET_STAGE/linear/classification/numeric/{split}/
index = META_CLASSIFICATION_DATASET_INDEX
checkpoint = best_classification.pt
num_classes metadata available
HPO_SWEEP_MODE = lbacnp_model supported
MODEL_ARCH_VERSION propagated
```

Must ensure classification routes to `task_objective="inductive_classification"` and uses LBACNP residual logits only.

### Agent 3 — Linear Regression Mixed-Categorical Pipeline

**SQL file:** `sql/linear_regression_mixed_pipeline.sql`

Verify:

```text
training_data_family = synthetic_linear_regression_mixed_categorical
stage path includes mixed/
index = META_MIXED_REGRESSION_DATASET_INDEX
checkpoint = best_regression.pt
use_categorical_features=True
use_lbacnp=True supported
```

Must ensure numeric PLE and categorical entity embeddings both execute before LBACNP.

### Agent 4 — Linear Classification Mixed-Categorical Pipeline

**SQL file:** `sql/linear_classification_mixed_pipeline.sql`

Verify:

```text
training_data_family = synthetic_linear_classification_mixed_categorical
stage path includes mixed/
index = META_MIXED_CATEGORICAL_DATASET_INDEX
checkpoint = best_classification.pt
use_categorical_features=True
use_lbacnp=True supported
```

Must ensure class-label embeddings, categorical embeddings, and LBACNP routing are compatible.

### Agent 5 — Nonlinear Regression Numeric Pipeline

**SQL file:** `sql/nonlinear_regression_numeric_pipeline.sql`

> **ℹ Family-name note:** The canonical family string for numeric nonlinear regression is
> **`synthetic_nonlinear_regression`** (`task_routing.py:21`). The string
> `synthetic_regression_nonlinear` is a registered back-compat alias (`task_routing.py:122`) and
> also resolves correctly via `get_training_data_spec()`. New code should use the canonical form
> `synthetic_nonlinear_regression`.

Verify:

```text
training_data_family = synthetic_nonlinear_regression   ← canonical; alias synthetic_regression_nonlinear also works
stage = @META_DATASET_STAGE / @META_DATASET_STAGE/nonlinear/regression/numeric/{split}/
index = META_NONLINEAR_REGRESSION_DATASET_INDEX
checkpoint = best_regression.pt   ← nonlinear regression shares this name with linear regression
use_numeric_ple=True
use_lbacnp=True
use_coefficient_head=False by default
```

Must also verify the extra `run_pretrain_pipeline_nonlinear` proc (`:150-164`) is compatible with `PRETRAIN_LOAD_POLICY=load_compatible_backbone`.

### Agent 6 — Nonlinear Classification Numeric Pipeline

**SQL file:** `sql/nonlinear_classification_numeric_pipeline.sql`

Verify:

```text
training_data_family = synthetic_nonlinear_classification
stage = @META_DATASET_STAGE / @META_DATASET_STAGE/nonlinear/classification/numeric/{split}/
index = META_NONLINEAR_CLASSIFICATION_DATASET_INDEX
checkpoint = best_nonlinear_cls.pt
num_classes metadata available
use_numeric_ple=True
use_lbacnp=True
classification logits from LBACNP primary path
```

Must ensure HPO metric is classification-appropriate (`val_cross_entropy`).

### Agent 7 — Nonlinear Regression Mixed-Categorical Pipeline

**SQL file:** `sql/nonlinear_regression_mixed_pipeline.sql`

Verify:

```text
training_data_family = synthetic_nonlinear_regression_mixed_categorical
stage includes mixed/
index = META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX
checkpoint = best_regression.pt   ← nonlinear mixed regression shares this name
use_categorical_features=True
use_numeric_ple=True
use_lbacnp=True
```

Must ensure both numeric nonlinear PLE and categorical token embeddings are fed into context/query encoders.

### Agent 8 — Nonlinear Classification Mixed-Categorical Pipeline

**SQL file:** `sql/nonlinear_classification_mixed_pipeline.sql`

Verify:

```text
training_data_family = synthetic_nonlinear_classification_mixed_categorical
stage includes mixed/
index = META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX
checkpoint = best_nonlinear_cls.pt
use_categorical_features=True
use_numeric_ple=True
use_lbacnp=True
```

Must ensure no query-label leakage, categorical statistics are context-only, and class logits are LBACNP-primary.

---

## 15. SQL Verification Checklist for Every Agent

Each SWE agent must inspect `/sql` and answer:

```text
 1. Which stored procedures launch generation, index build, pretrain, HPO, final training, and evaluation?
 2. Which env vars are passed into each MLJob?
 3. Is MODEL_ARCH_VERSION passed anywhere? If not, add it.
 4. Is HPO_SWEEP_MODE accepted by the SQL procedure and Python handler?
 5. Does the procedure allow HPO_SWEEP_MODE='lbacnp_model'?
 6. Does the procedure pass TRAINING_DATA_FAMILY consistently?
 7. Does the procedure pass MODEL_FAMILY and MODEL_DESIGN_PATTERN consistently?
 8. Does the index table include required columns for the family?
 9. Does the stage path use numeric/ or mixed/ correctly?
10. Does the checkpoint output name match downstream evaluation expectations?
11. Does final training read @MODEL_STAGE/hpo/best_config.json?
12. Does final training pass MODEL_ARCH_VERSION from best_config or env?
13. Is there any hardcoded model4 assumption in SQL or handler code?
14. Is there any hardcoded nonlinear family name mismatch?
15. Is the expected total count env var family-specific and correct?
```

---

## 16. Implementation Order

Claude should implement in this order:

```text
 1. Add ModelConfig fields and validation (model.py: new fields + __post_init__ rules + whitelist).
 2. Add NumericPLEEncoder and test context-only boundaries.
 3. Add LBACNP modules using existing MAB/SAB (LatentBottleneckBlock, LatentBottleneckACNP).
 4. Add regression and classification LBACNP heads (RegressionDistributionHead, ClassificationLBACNPHead, ResidualGate).
 5. Add non-breaking MODEL5 forward routing (if cfg.use_lbacnp: branches alongside existing paths).
 6. Add train.py BEST_CONFIG parsing for MODEL5 fields; add GaussianNLL distribution loss path.
 7. Add hpo.py lbacnp_model search space and Ray trainable parsing.
 8. Add run_hpo_job.py support for lbacnp_model and MODEL_ARCH_VERSION.
 9. Add run_model_training_job.py support for MODEL_ARCH_VERSION and compatible backbone loading.
10. Add checkpoint metadata for MODEL5 (format_version=8 + metadata block).
11. Add tests (§13 acceptance test suite).
12. Deploy 8 SQL agents to validate /sql pipelines (§14).
13. Patch SQL procedures/env vars as needed (per agent reports).
14. Run smoke HPO on one small family (see §17).
15. Run full HPO on linear regression.
16. Run full HPO on nonlinear regression.
17. Expand to classification and mixed-categorical families.
```

---

## 17. Smoke Test Plan

Before full HPO, run a smoke test with:

```text
NUM_TRIALS = 2
TRIAL_MAX_EPOCHS = 2
HPO_SPLIT_LIMITS = {"train": 4, "val": 2}
HPO_SWEEP_MODE = "lbacnp_model"
MODEL_ARCH_VERSION = "model5_lbacnp"
```

Smoke test must verify:

```text
HPO job launches.
Ray workers instantiate ModelConfig.
MODEL5 model builds.
Forward pass succeeds.
Loss computes.
Metric reports.
best_config.json uploads to @MODEL_STAGE/hpo/.
Final training can read best_config.json.
Checkpoint uploads to @MODEL_STAGE/checkpoints/.
```

---

## 18. Production HPO Invocation Pattern

Recommended SQL call shape (matches the existing `run_hpo_pipeline` 4-arg proc confirmed in all 8 pipeline files):

```sql
CALL run_hpo_pipeline(
    'market_exchangeable_icl',
    '<TRAINING_DATA_FAMILY>',
    'inductive_forecasting',
    'lbacnp_model'
);
```

For nonlinear families with pretrain warm-start:

```sql
CALL run_hpo_pipeline(
    'market_exchangeable_icl',
    '<NONLINEAR_TRAINING_DATA_FAMILY>',
    'inductive_forecasting',
    'lbacnp_model',
    '',
    '@MODEL_STAGE/checkpoints/<compatible_pretrain>.pt'
);
```

But prefer compatible backbone loading (`PRETRAIN_LOAD_POLICY=load_compatible_backbone`) over exact-match loading.

---

## 19. Final Guardrails

Do not implement MODEL5 by deleting MODEL4.

Do not use stochastic latent variables in v1.

Do not fit PLE bins on query data.

Do not use query labels in PLE, categorical statistics, class label maps, or attention encoders.

Do not require separate pretrain checkpoints for every LBACNP HPO candidate.

Do not run a broad architecture sweep in v1.

Do not change checkpoint names until downstream SQL/evaluation scripts are verified.

Do not let LBACNP residual dominate clean linear tasks.

Do not mutate linear DGP contracts to support nonlinear data.

Do not combine regression and classification schemas.

---

## 20. Definition of Done

MODEL5 is complete when:

```text
 1. MODEL4 compatibility is preserved.
 2. MODEL5 builds and trains for numeric linear regression.
 3. MODEL5 builds and trains for numeric linear classification.
 4. MODEL5 builds and trains for numeric nonlinear regression.
 5. MODEL5 builds and trains for numeric nonlinear classification.
 6. MODEL5 supports mixed-categorical linear and nonlinear paths.
 7. HPO_SWEEP_MODE='lbacnp_model' runs as a single sweep.
 8. best_config.json contains all MODEL5 fields.
 9. final training consumes the single best_config.json.
10. all 8 SQL pipeline agents report compatibility or provide patches.
11. no leakage tests pass.
12. shape/permutation/memory tests pass.
13. linear regression/classification do not regress versus MODEL4 baselines.
14. nonlinear regression/classification improve versus MODEL4 on nonlinear regimes.
```
