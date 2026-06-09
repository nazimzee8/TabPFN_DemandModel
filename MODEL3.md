# MODEL3 - Architecture Specification

This document is the normative specification for the future `MODEL3` model family.
It is intentionally specification-only: it does not describe code that is already
implemented, and it must not be used as permission to mutate retired MODEL2 classes
in place.

`MODEL2.md` remains the production reference for the current MODEL2 baseline.
MODEL3 is a new model-family specification that inherits MODEL2's core
principle:

> Preserve structured evidence before irreversible pooling.

MODEL3 extends that principle from query-conditioned synthetic regression into a broader
exchangeable tensor architecture family. The family has two valid design patterns:

| Design pattern | Purpose | Default |
|---|---|---|
| `inductive_forecasting` | Synthetic regression and query-conditioned forecasting from `(X_train, y_train)` to `x_test` | Yes |
| `transductive_completion` | Sparse market interaction matrix/tensor completion and missing-cell prediction | No |

The patterns share embedding and exchangeable-block primitives, but they must remain
separate in objective, checkpoint metadata, HPO configuration, runtime selection, and
validation.

---

## 1. Model-Family Boundary

MODEL3 is not a mutation of retired MODEL2 code. It is a future architecture family
that can coexist with MODEL2 behind explicit runtime selectors.

Required defaults:

```text
MODEL_ARCH_VERSION="model2"
MODEL_DESIGN_PATTERN="inductive_forecasting"
MODEL_FAMILY="market_aware"
```

Future MODEL3 synthetic regression:

```text
MODEL_ARCH_VERSION="model3"
MODEL_DESIGN_PATTERN="inductive_forecasting"
MODEL_FAMILY="market_exchangeable_icl"
```

Future MODEL3 market completion:

```text
MODEL_ARCH_VERSION="model3"
MODEL_DESIGN_PATTERN="transductive_completion"
MODEL_FAMILY="market_exchangeable_completion"
```

Runtime selection must reject incompatible combinations. In particular,
`transductive_completion` must not silently run under a synthetic regression objective.

---

## 2. Shared MODEL3 Primitives

### ExchangeableMatrixBlock

The central primitive is an exchangeable block over a structured tensor:

```text
H: (batch, rows, cols, channels)
```

For each cell `(r, c)`, the update branch may consume:

| Component | Shape concept | Meaning |
|---|---|---|
| Self state | `H[:, r, c, :]` | The current cell embedding |
| Row aggregate | reduction over `cols` | Evidence shared by cells in the same row |
| Column aggregate | reduction over `rows` | Evidence shared by cells in the same column |
| Global aggregate | reduction over `rows, cols` | Dataset-level or tensor-level evidence |
| Metadata | broadcast or cell-local | Side information, timestamps, hierarchy, market IDs |
| Mask channel | cell-local | Whether a value is observed, missing, target, or held out |

The block must be permutation equivariant over the axes declared exchangeable. If an axis
is ordered, hierarchical, grouped, or otherwise structured, that structure must be encoded
through positional features, hierarchy features, metadata, masks, or restricted attention
before applying pure permutation-equivariant treatment.

### Masked Reductions

Sparse data must use masked reductions. Missing entries are not zeros.

All row, column, and global reductions over sparse tensors must exclude missing entries
unless the mask semantics explicitly require including them as missing-cell tokens. Any
implementation must distinguish:

| State | Meaning |
|---|---|
| Observed | Value is present and may contribute to reconstruction/conditioning |
| Missing unknown | Value is absent and must not be treated as numeric zero |
| Target | Value is requested for prediction or held out for validation |
| Structural zero | True zero value, distinct from missingness |

### ColumnEncoder and CellEncoder

MODEL3 must encode raw values before exchangeable updates:

| Encoder | Responsibility |
|---|---|
| `ColumnEncoder` | Distribution-aware feature/column embeddings, normalization statistics, type information, optional metadata |
| `CellEncoder` | Cell/channel embeddings from observed values, query values, labels, masks, and interaction channels |

Exchangeable blocks operate on encoded tensors, not raw scalar values alone.

### ISAB Placement

Induced Set Attention Blocks (ISAB) are optional. They may be used only:

1. after structured exchangeable updates have preserved row/column evidence; or
2. inside clean equivariant residual branches where the declared axis semantics remain
   valid.

ISAB must not be used as an early lossy compression that destroys query, feature, row, or
mask identity before the model has represented sufficient evidence.

### Residual Policy

Residual blocks must use a clean identity shortcut:

```text
x_{l+1} = x_l + f(x_l)
```

The shortcut is exact identity. Any normalization, masking, equivariant transform,
attention, or MLP belongs inside `f(x_l)`. The transform branch must preserve the same
equivariance contract as the block itself.

---

## 3. Design Pattern: Inductive Forecasting

`inductive_forecasting` is the default MODEL3 pattern. It is the correct pattern for
synthetic regression and for query-conditioned market forecasting where predictions are
requested for explicit query rows.

Required selector:

```text
MODEL_DESIGN_PATTERN="inductive_forecasting"
```

### Inputs and Output

```text
X_train: (n, p)
y_train: (n,)
x_test:  (m, p)

y_hat:   (m,)
```

The model consumes a labeled context and one or more query rows. It returns one scalar
prediction per query row.

### Architecture Flow

```text
raw context/query values
  -> distribution-aware column embeddings
  -> cell/channel embeddings
  -> per-query tensor H_q: (m, n, p, channels)
  -> Hartford-style exchangeable blocks over (sample, feature)
  -> row/query/feature embeddings
  -> optional ISAB
  -> query-conditioned prediction head
  -> optional ridge/prior expert residual path
  -> y_hat: (m,)
```

The per-query tensor `H_q` is the key structure. Each query keeps its own conditioned view
of the context, so query information cannot collapse before prediction.

### Required Semantics

The model must preserve:

| Axis | Semantics |
|---|---|
| `m` query rows | Separate predictions; no query collapse |
| `n` context samples | Permutation-invariant conditioning |
| `p` features | Feature permutation consistency when metadata and ordering are permuted consistently |
| channels | Encoded values, labels, query interactions, masks, and metadata |

Prediction must remain query-conditioned. A valid inductive MODEL3 model cannot pool the
query axis into one context vector and then reuse the same scalar for all `x_test`.

### Ridge or Prior Expert Residual Path

An optional ridge/prior expert remains appropriate for this pattern because the current
synthetic regression data-generating process is mostly linear. The neural path can learn
nonlinear residual structure, while the expert path supplies a stable signal-recovery
baseline.

The expert path must be a residual or gated residual component. It must not replace the
MODEL3 query-conditioned tensor flow.

---

## 4. Design Pattern: Transductive Completion

`transductive_completion` is a separate MODEL3 pattern for sparse market interaction
surfaces and missing-cell completion. It is not the default synthetic regression objective.

Required selector:

```text
MODEL_DESIGN_PATTERN="transductive_completion"
```

### Inputs and Output

Inputs may include:

| Input | Meaning |
|---|---|
| Observed interaction matrix/tensor | Sparse market, product, store, time, or interaction values |
| Observed-entry mask | Explicit observed/missing/target indicators |
| Optional row features | Metadata or side features for row entities |
| Optional column features | Metadata or side features for column entities |
| Optional tensor metadata | Time, hierarchy, geography, segment, source, reliability |
| Target cells | Cells requested for prediction or validation |

Outputs may be:

| Output | Meaning |
|---|---|
| Completed cells | Predictions for target missing cells |
| Reconstructed entries | Reconstruction of observed or held-out cells |
| Target-cell predictions | Task-specific predictions over selected cells |

### Architecture Flow

```text
sparse tensor
  -> mask-aware cell/channel embeddings
  -> side-feature broadcasting
  -> mask-aware Hartford blocks
  -> latent factors/cell embeddings
  -> decoder/reconstruction head
  -> optional forecasting head only when explicitly configured
```

The forecasting head is optional and must be explicitly configured. Completion training
must not accidentally become inductive query forecasting, and inductive training must not
accidentally become matrix reconstruction.

### Required Semantics

The model must preserve:

| Axis | Semantics |
|---|---|
| Rows | Exchangeable unless row metadata or hierarchy says otherwise |
| Columns | Exchangeable unless column metadata or hierarchy says otherwise |
| Tensor modes | Separately declared as exchangeable, ordered, grouped, or hierarchical |
| Mask | Missingness is evidence and must be represented separately from numeric value |
| Targets | Held-out or requested cells must be tracked distinctly from conditioning cells |

This pattern is useful for market mental-model data because markets often contain sparse
interaction surfaces, meaningful missingness, unordered row/column axes, and side-channel
metadata. Ordered or hierarchical axes need explicit positional, hierarchical, mask, or
metadata treatment before pure permutation equivariance is valid.

---

## 5. Synthetic Regression Guidance

Synthetic regression should use `inductive_forecasting`.

The task is:

```text
predict beta X_test from (X_train, y_train) and x_test
```

This is signal recovery and in-context prediction, not matrix completion. The model must
preserve query conditioning and return:

```text
y_hat: (m,)
```

Synthetic regression validates inductive in-context learning and signal recovery. It does
not, by itself, validate full sparse market interaction completion.

Do not use `transductive_completion` as the default synthetic regression objective.

Do not justify synthetic regression performance using feature-axis AR(1) behavior unless
the generator actually creates that feature correlation. If the generator samples
independent features or a different covariance structure, the architecture discussion and
evaluation claims must match the actual generator.

Ridge and prior residual paths remain useful for the current synthetic regression setup
because the dominant data-generating signal is mostly linear. The neural architecture
still needs query-sensitive exchangeable processing so that it can recover coefficients,
interactions, noise patterns, and nonlinear residuals without query collapse.

---

## 6. Market Mental-Model Guidance

Market completion should use `transductive_completion` when the primary object is a sparse
interaction matrix or tensor.

Examples include:

| Surface | Rows | Columns or modes | Prediction target |
|---|---|---|---|
| Product substitution | Products | Products or baskets | Missing interaction strength |
| Store-product demand | Stores | Products, time | Missing demand cell |
| Price response | Products | Price regimes, segments | Unobserved response |
| Market graph tensor | Entities | Entities, relation type, time | Missing relation value |

This pattern is appropriate when missingness itself carries information and when the goal
is reconstructing or completing a partially observed surface.

For ordered or hierarchical market axes, MODEL3 must encode the order or hierarchy rather
than pretending the axis is freely exchangeable. Valid treatments include positional
features, calendar features, parent-child hierarchy embeddings, group masks, restricted
attention, or metadata-conditioned reductions.

---

## 7. Runtime and Pipeline Specification

Future orchestration must propagate these selectors through pretraining, HPO, final
training, fine-tuning, evaluation, and inference:

```text
MODEL_ARCH_VERSION
MODEL_DESIGN_PATTERN
MODEL_FAMILY
TRAINING_DATA_FAMILY
```

Required runtime values:

| Scenario | `MODEL_ARCH_VERSION` | `MODEL_DESIGN_PATTERN` | Model family |
|---|---|---|---|
| Current production default | `"model2"` | `"inductive_forecasting"` | `"market_aware"` |
| Future MODEL3 synthetic regression | `"model3"` | `"inductive_forecasting"` | `"market_exchangeable_icl"` |
| Future MODEL3 market completion | `"model3"` | `"transductive_completion"` | `"market_exchangeable_completion"` |

Defaults must preserve existing MODEL2 behavior. The presence of
`MODEL_DESIGN_PATTERN="inductive_forecasting"` as a default selector must not cause MODEL3
code paths to run unless `MODEL_ARCH_VERSION="model3"` is also selected.

---

## 8. Checkpoint Metadata

MODEL3 checkpoints must include enough metadata to prevent architecture/objective
confusion.

Required metadata fields:

```python
{
    "model_arch_version": "model3",
    "model_design_pattern": "inductive_forecasting",  # or "transductive_completion"
    "model_family": "market_exchangeable_icl",         # or "market_exchangeable_completion"
    "training_data_family": "linear_poisson_v1",       # must match training prior
    "task_objective": "inductive_regression",          # or "transductive_completion"
    "checkpoint_format_version": 4,
}
```

The exact checkpoint format version may change during implementation, but MODEL3 must not
reuse ambiguous MODEL2 metadata. Loading must validate that:

1. `model_arch_version` matches the instantiated architecture.
2. `model_design_pattern` matches the expected runtime objective.
3. `model_family` maps to the selected model class.
4. `training_data_family` is compatible with the requested inference path.
5. `task_objective` is not silently converted between inductive prediction and
   transductive reconstruction.

---

## 9. HPO Rules

Existing MODEL2 HPO behavior must be preserved.

MODEL3 HPO must be introduced only after the relevant sanity checks pass:

| Pattern | HPO prerequisite |
|---|---|
| `inductive_forecasting` | Query-sensitivity and synthetic regression sanity checks pass |
| `transductive_completion` | Masked reconstruction and missing-cell completion sanity checks pass |

The HPO spaces must be separate. Inductive validation MSE and transductive reconstruction
loss must not be combined into one opaque score.

Valid HPO objective examples:

| Pattern | Objective |
|---|---|
| `inductive_forecasting` | Query-conditioned validation MSE, synthetic linear recovery, OOD regression metrics |
| `transductive_completion` | Masked held-out reconstruction loss, target-cell completion error, mask leakage tests |

Hyperparameters may overlap, but their search spaces and acceptance gates should be
declared separately because the tasks have different failure modes.

---

## 10. SQL Procedure Specification

This section is spec-only. It describes future procedure signatures, not current SQL
behavior.

Future `run_training_job.sql` procedure signatures should accept architecture and pattern
selectors:

```sql
run_pretrain_pipeline(MODEL_ARCH_VERSION, MODEL_DESIGN_PATTERN)
run_hpo_pipeline(MODEL_ARCH_VERSION, MODEL_DESIGN_PATTERN)
run_model_training(MODEL_ARCH_VERSION, MODEL_DESIGN_PATTERN)
run_training_pipeline(MODEL_ARCH_VERSION, MODEL_DESIGN_PATTERN)
```

Default SQL calls must remain equivalent to MODEL2 plus the default design-pattern value:

```text
MODEL_ARCH_VERSION="model2"
MODEL_DESIGN_PATTERN="inductive_forecasting"
```

Future SQL orchestration must also propagate:

```text
MODEL_FAMILY
TRAINING_DATA_FAMILY
```

Procedure implementations must validate incompatible combinations before starting compute
jobs. A completion objective should not start with an inductive model family, and an
inductive synthetic regression job should not start with a completion model family.

---

## 11. Acceptance Criteria

### Inductive Forecasting

An `inductive_forecasting` MODEL3 implementation is acceptable only if it passes:

| Criterion | Requirement |
|---|---|
| Row permutation invariance | Permuting `X_train, y_train` rows does not change `y_hat` except for numerical tolerance |
| Feature permutation consistency | Jointly permuting feature columns and feature metadata permutes internal feature handling consistently and preserves predictions |
| Query sensitivity | Different valid `x_test` rows can produce different predictions |
| No query collapse | Batched queries must not collapse to a shared scalar or near-identical output except when the data implies it |
| Synthetic linear recovery | Recovers mostly linear synthetic data at expected accuracy relative to ridge/baseline checks |
| Sparse/noise robustness | Handles masked, noisy, or partially missing context features according to declared mask semantics |
| Memory guard | Attention over `(m, n, p)` must have explicit memory limits, chunking, or safe defaults |
| Checkpoint metadata validation | Rejects checkpoints with mismatched architecture, family, pattern, data family, or objective |

### Transductive Completion

A `transductive_completion` MODEL3 implementation is acceptable only if it passes:

| Criterion | Requirement |
|---|---|
| Row equivariance | Row permutations produce correspondingly permuted outputs |
| Column equivariance | Column permutations produce correspondingly permuted outputs |
| Joint equivariance | Joint row/column permutations preserve the declared tensor semantics |
| Sparse mask equivariance | Permuting values and masks together preserves output consistency |
| Observed-cell reconstruction | Reconstructs observed or held-out observed cells under masked validation |
| Missing-cell completion | Predicts target missing cells without treating missing values as zeros |
| Mask leakage tests | Cannot infer targets from validation masks or target indicators that leak labels |
| Side-channel metadata tests | Uses row/column/tensor metadata consistently and rejects malformed metadata alignment |
| Separate objective validation | Completion metrics are reported separately from inductive forecasting metrics |

---

## 12. Non-Goals

MODEL3 does not require immediate changes to:

```text
src/model.py
src/train.py
src/hpo.py
pretrain scripts
sql/run_training_job.sql
tests
runtime orchestration
```

Those changes belong to a future implementation task. This file defines the target
architecture and the constraints that future implementation must satisfy.

---

## 13. Final Architecture Judgment

MODEL3 has two valid design patterns. Inductive Forecasting is the default and correct
path for synthetic regression and query-conditioned market forecasting. Transductive
Completion is a separate market-oriented mode for sparse nonlinear interaction surfaces.
The modes share embedding and exchangeable-block primitives, but remain separate in
objective, checkpoint metadata, HPO configuration, runtime selection, and validation.
