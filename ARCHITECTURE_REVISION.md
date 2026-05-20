# Architecture Revision: MODEL2 vs Hartford et al. (2018)

## Scope

This document validates `MODEL2`, the current `retired MODEL2 model`, against Hartford et al. (2018), *Deep Models of Interactions Across Sets*.

This is an architectural audit only. It does not prescribe immediate implementation work in this revision.

Primary references:

- Hartford et al. (2018), *Deep Models of Interactions Across Sets*: https://proceedings.mlr.press/v80/hartford18a.html
- Hartford et al. PDF: https://proceedings.mlr.press/v80/hartford18a/hartford18a.pdf
- Local MODEL2 reference: `MODEL2.md`
- Local implementation: `src/model.py`

## Executive Judgment

Your proposed direction is technically sound, but it should be applied with a sharper boundary.

Hartford-style exchangeable matrix/tensor layers are the right next architectural family for market interaction data where the problem is naturally an interaction array:

- asset x feature,
- customer x product,
- entity x factor,
- instrument x time,
- regime x signal,
- or higher-order market tensors.

However, the current synthetic regression pipeline is not purely a Hartford matrix-completion problem. It is primarily an in-context regression problem:

```text
given context (X_train, y_train) and query x_test -> predict y_test
```

So the recommended path is not to replace `retired MODEL2 model` wholesale with a factorized exchangeable autoencoder. The better path is:

1. Keep `retired MODEL2 model` as the current production model.
2. Add a Hartford-aligned model family for structured interaction tensors.
3. Use exchangeable matrix/tensor blocks before any irreversible pooling.
4. Keep inductive query prediction as the default objective for synthetic regression.
5. Add a separate transductive/factorized autoencoder mode only when the task is true matrix/tensor completion.

In short: adopt Hartford's exchangeable layer as a core block, but do not inherit Hartford's full matrix-completion objective unless the dataset and task actually require transductive interpolation.

## Current MODEL2 Architecture

`retired MODEL2 model` currently does the following:

1. Builds per-query, per-feature, per-sample tokens:

   ```text
   tokens: (m, p, n, 6)
   ```

2. Applies `phi_sample` to each `(query, feature)` sample set.
3. Aggregates sample evidence within each feature.
4. Preserves feature identity as:

   ```text
   ev: (m, p, d_feat)
   ```

5. Applies feature-level SAB or linear feature equivariance over `p`.
6. Produces predictions through beta-like, residual, gate, and optional ridge-expert heads.

This is a real improvement over the legacy feature-pool-first DeepSet. It delays feature pooling and allows cross-feature reasoning before final summarization.

But it is still a nested set model, not a Hartford exchangeable matrix/tensor model.

## Hartford et al. Requirements

Hartford et al. model interactions across two or more exchangeable sets. The canonical representation is a matrix or tensor whose meaning does not change under row/column or axis permutations.

For a 2D matrix, a permutation-equivariant layer should satisfy:

```text
f(P_row X P_col) = P_row f(X) P_col
```

The exchangeable matrix layer achieves this with tied parameters so each output cell can depend on:

- itself,
- its row aggregate,
- its column aggregate,
- the global matrix aggregate,
- a shared bias.

For channelized data:

```text
X: (batch, rows, cols, channels_in)
Y: (batch, rows, cols, channels_out)
```

The same parameter-sharing scheme is applied across channels. Hartford et al. also extend the idea to higher-dimensional tensors.

## Specification-by-Specification Audit

### 1. Permutation Equivariance Across Two Sets

**User specification**

The model should impose permutation equivariance across two sets, i.e. a 2D matrix. For the market mental model, this may extend to structured tensors.

**Current status**

Partially satisfied.

`retired MODEL2 model` is sample-row invariant and feature-permutation consistent under current tests. But it does not maintain a full `(row, column)` cell state through multiple exchangeable layers.

Current MODEL2:

```text
(query, feature, sample) -> sample pooling per feature -> feature SAB -> prediction
```

Hartford target:

```text
(row, column, channel) -> exchangeable matrix/tensor layers -> cell/row/column/query readout
```

**Gap**

The model does not explicitly enforce Hartford-style matrix equivariance:

```text
f(P_rows X P_cols) = P_rows f(X) P_cols
```

It enforces useful set symmetries, but not the full row/column exchangeable matrix symmetry.

**Engineering judgment**

This gap matters for the market mental model more than for the current synthetic regression benchmark. Synthetic regression can still be served by MODEL2, but market interaction tensors should move toward Hartford-style exchangeable layers.

**Recommended solution**

Introduce a new model family:

```text
model_family = "market_exchangeable"
```

This model should operate on a channelized interaction tensor before pooling.

### 2. Exchangeable Matrix Layer: Self, Row, Column, Global

**User specification**

Use an exchangeable matrix layer where each cell learns from itself, its row, its column, and the whole matrix using shared weights.

**Current status**

Not satisfied.

MODEL2 has no explicit self/row/column/global cell update. Its closest operation is:

- sample pooling within each feature,
- then feature SAB across features.

**Gap**

After sample pooling, the model no longer has cell-level access to the full `(sample, feature)` matrix. That prevents Hartford-style row/column/global updates.

**Engineering judgment**

This is the most important architectural gap. If we want Hartford compliance, this block is non-negotiable.

**Recommended solution**

Add an `ExchangeableMatrixBlock` conceptually equivalent to:

```text
Y[i, j] =
  phi(
    W_self   X[i, j]
  + W_row    mean_j X[i, j]
  + W_col    mean_i X[i, j]
  + W_global mean_{i,j} X[i, j]
  + b
  )
```

For sparse data, replace means with masked means over observed cells.

### 3. Parameter Sharing and Depth

**User specification**

Parameter sharing corresponds to summing or averaging across rows/columns. Stacking layers with the same equivariance property preserves permutation equivariance.

**Current status**

Partially satisfied.

MODEL2 uses shared MLPs and feature SAB, but not Hartford's matrix parameter-tying scheme.

**Gap**

MODEL2 can be made deep, but its depth is not a stack of exchangeable matrix/tensor layers. Therefore it does not inherit Hartford's equivariance proof.

**Engineering judgment**

Depth should be added only after the equivariant primitive is correct. Deeper attention blocks without the right sharing pattern may improve accuracy but will not provide the Hartford guarantee.

**Recommended solution**

Stack exchangeable blocks:

```text
H0 -> ExchangeableMatrixBlock -> Norm/Gate -> ExchangeableMatrixBlock -> ... -> Readout
```

Later paper reviews should decide whether normalization should be SetNorm-style rather than LayerNorm.

### 4. Multiple Input/Output Channels

**User specification**

The model must support multiple input-output channels. Row/column features should be broadcast over the matrix and treated as channels.

**Current status**

Partially satisfied.

MODEL2 has a fixed six-channel token:

```text
y,
X_ij,
xq_j,
X_ij * xq_j,
X_ij * y,
|X_ij - xq_j|
```

This is useful but not a general multi-channel interaction schema.

**Gap**

There is no explicit split between:

- cell channels,
- row-side features,
- column-side features,
- query-side features,
- mask channels,
- regime/market-prior channels.

**Engineering judgment**

Hardcoded regression token channels are acceptable for MODEL2, but too narrow for a market mental model. A market model needs a formal channel schema.

**Recommended solution**

Define a channelized input contract:

```text
cell_channels: X_cell[b, i, j, c]
row_features:  R[b, i, r]
col_features:  C[b, j, k]
query_features: Q[b, q, ...]
mask:          M[b, i, j]
```

Broadcast:

```text
R -> R[b, i, j, r]
C -> C[b, i, j, k]
```

Then concatenate:

```text
H0[b, i, j, :] = concat(cell, row, col, query, mask)
```

### 5. Sparse Matrices

**User specification**

For sparse matrices, apply the same parameter-sharing scheme while limiting the model to observed entries.

**Current status**

Not satisfied as a first-class architecture.

The current synthetic regression tensors are dense. MODEL2 does not have a mask-aware exchangeable matrix layer.

**Gap**

If missing entries are represented as zeros, the model may treat missingness as a real value. That would violate the intended sparse exchangeable semantics.

**Engineering judgment**

Mask support should be introduced at the same time as exchangeable matrix blocks. Retrofitting masks later is likely to create subtle bugs in row/column/global reductions.

**Recommended solution**

All exchangeable reductions should accept:

```text
mask: (batch, rows, cols)
```

and compute:

- self term only for observed entries,
- row masked mean,
- column masked mean,
- global masked mean.

The mask should also be included as an input channel.

### 6. Inductive vs Transductive Analysis

**User specification**

Select the right approach for synthetic data and market mental model. Hartford et al. used a factorized exchangeable autoencoder for matrix interpolation.

**Current status**

MODEL2 is inductive. It predicts new query outputs from context. It is not a factorized exchangeable autoencoder.

**Gap**

The current architecture does not distinguish:

- inductive in-context regression,
- transductive matrix completion,
- extrapolation to new rows/columns after observations are provided.

**Engineering judgment**

For the current synthetic regression evaluation, the inductive approach is the right default. A factorized exchangeable autoencoder should not replace it unless the task becomes missing-cell matrix/tensor completion.

For the market mental model, both may be needed:

- inductive forecast/readout for prediction,
- transductive completion for sparse interaction surfaces.

**Recommended solution**

Use two modes:

#### A. Inductive Exchangeable Predictor

Best for synthetic regression and market forecasting.

```text
channel encoder
-> exchangeable matrix/tensor blocks
-> query-conditioned readout
-> optional ridge/prior expert
```

#### B. Factorized Exchangeable Autoencoder

Best for true matrix/tensor interpolation.

```text
exchangeable encoder
-> row factors + column factors
-> decoder
-> reconstruct observed and missing cells
```

Do not conflate these objectives.

## Tabular ICL / Embedding-Then-ICL Revision

### Paper Context and Engineering Judgment

The requested specification is directionally aligned with modern tabular foundation models, but it blends two related ideas:

1. TabPFN / TabPFNv2-style in-context prediction, where labeled context rows and query rows are consumed in a single forward pass.
2. TabICL-style scalable embedding-then-ICL, where column-aware feature embeddings are built before row-wise ICL over fixed-dimensional row embeddings.

The architectural proposal is sound, but I would frame it as a TabICL-inspired upgrade to MODEL2 rather than as a literal TabPFNv2 clone. Public descriptions of TabICL emphasize a two-stage column-then-row architecture that builds fixed-dimensional row embeddings before the final ICL transformer. That is closer to the proposed "distribution-aware column embeddings -> row-wise interaction -> row embedding -> set/ICL model" design than MODEL2's current nested set evidence path.

Primary references for this section:

- TabPFN / TabPFNv2 foundation model context: https://www.nature.com/articles/s41586-024-08328-6
- TabICL, *A Tabular Foundation Model for In-Context Learning on Large Data*: https://proceedings.mlr.press/v267/qu25d.html

MODEL2 remains structurally sound under `MODEL2.md`. The gap is not that MODEL2 is wrong; it is that MODEL2 does not yet implement the proposed distribution-aware tabular encoder stage.

### Requested Specification

The target architecture is:

```text
z_i = phi(x_i; X_context)
```

where `phi` is no longer a row-local MLP. Instead, `phi` is a context-conditioned tabular encoder:

1. Build distribution-aware column embeddings for every feature column.
2. Build cell embeddings where each cell depends on:
   - scalar value `x_ij`,
   - empirical distribution of column `j` across the context set.
3. Apply a row-wise interaction block over embedded feature tokens.
4. Produce fixed-dimensional row embeddings:

   ```text
   z_i in R^{d_phi}
   ```

5. Feed the set of row embeddings `{z_i}` into the final DeepSet / set / ICL model.

### Current MODEL2 Status

MODEL2 currently builds handcrafted six-channel cell tokens:

```text
y,
X_ij,
xq_j,
X_ij * xq_j,
X_ij * y,
|X_ij - xq_j|
```

Then it applies:

```text
phi_sample -> optional sample SAB -> sample_pool per feature -> feature SAB -> prediction heads
```

This preserves feature identity and allows cross-feature reasoning before final prediction. That is consistent with `MODEL2.md`.

However, it does not explicitly learn distribution-aware column embeddings before row encoding. It also does not emit a clean row embedding `z_i` for every row and then pass `{z_i}` into a final set/ICL model.

### Gap 1: Missing Distribution-Aware Column Embeddings

**Specification**

Each feature column should have an embedding informed by its empirical distribution across the context set.

**Current status**

Not satisfied as a first-class component.

MODEL2 normalizes features per context and uses scalar cell values inside handcrafted tokens. This gives the model some distribution awareness through normalization, but there is no learned column-distribution encoder.

**Architectural gap**

There is no explicit object like:

```text
c_j = ColumnEncoder({x_ij}_{i=1..n}, mask_j, optional column metadata)
```

and no explicit cell embedding:

```text
e_ij = CellEncoder(x_ij, c_j)
```

**Engineering judgment**

Per-context normalization is not enough. If we want TabPFN/TabICL-like behavior, the model should learn how to interpret a value relative to the empirical distribution of its column, including scale, skew, tails, discreteness, missingness, and possibly categorical/cardinality structure.

**Recommended solution**

Add a distribution-aware column encoder before row encoding:

```text
ColumnEncoder:
  input:  X_context[:, j], mask[:, j], optional column metadata
  output: c_j in R^{d_col}
```

The encoder can combine:

- robust summary statistics,
- quantile/bin embeddings,
- learned distribution sketches,
- missingness rate,
- optional semantic/market column metadata.

Then build cell embeddings:

```text
e_ij = CellEncoder(
  value=x_ij,
  column_embedding=c_j,
  normalized_value=normalize(x_ij | X_context[:, j]),
  missing_indicator=m_ij
)
```

### Gap 2: No Explicit Cell Embedding Stage

**Specification**

Each cell embedding should depend on both scalar `x_ij` and the empirical distribution of column `j`.

**Current status**

Partially satisfied through handcrafted token interactions, not through a general learned cell embedding stage.

MODEL2's token features are regression-useful, but they are not a reusable cell embedding API. They also mix label/query interactions directly into the token schema.

**Architectural gap**

The model lacks a separable embedding layer:

```text
raw table -> distribution-aware cell tokens -> row interaction -> row embeddings
```

**Engineering judgment**

This separation is worth doing. It makes the architecture cleaner, easier to test, and closer to tabular foundation-model practice. It also makes it easier to extend from synthetic regression to market-prior data without repeatedly redesigning the token schema.

**Recommended solution**

Introduce a `DistributionAwareCellEncoder` conceptually:

```text
E[b, i, j, :] = f_cell(
  scalar_value=x_ij,
  normalized_value=x_norm_ij,
  column_embedding=c_j,
  row/query role_embedding,
  observed_mask=m_ij
)
```

For regression ICL, labels should be handled as context targets or target tokens, not hardwired into every feature cell unless a specific ablation justifies it.

### Gap 3: Missing Row-Wise Interaction Block Over Embedded Feature Tokens

**Specification**

Apply a row-wise interaction block over embedded feature tokens to understand what each feature value means relative to its column distribution and other feature values.

**Current status**

Partially satisfied.

MODEL2 applies feature SAB after sample evidence has already been pooled per feature. It does cross-feature reasoning, but not over per-row embedded feature tokens before producing a row embedding.

**Architectural gap**

MODEL2 does not compute:

```text
E_i = [e_i1, e_i2, ..., e_ip]
z_i = RowEncoder(E_i)
```

for each row `i`.

**Engineering judgment**

This is the central gap for embedding-then-ICL. MODEL2 is query-conditioned and feature-evidence oriented; the requested architecture is row-embedding oriented. Both are valid, but they are not the same.

**Recommended solution**

Add a row-wise feature-token interaction block:

```text
RowEncoder:
  input:  E_i in R^{p x d_cell}
  output: z_i in R^{d_phi}
```

The row encoder can be:

- feature-token SAB,
- Set Transformer block,
- gated pooling after feature attention,
- exchangeable matrix-compatible row readout if combined with the Hartford path.

The output dimension should remain:

```text
d_phi = row embedding dimension
```

This preserves the existing MODEL2 convention that `d_phi` controls the representation consumed by later set reasoning.

### Gap 4: Final Set Model Does Not Consume `{z_i}` as the Primary Interface

**Specification**

The final DeepSet / set model should consume row embeddings `{z_i}`.

**Current status**

Not satisfied in the requested form.

MODEL2 directly computes query-conditioned feature evidence and prediction heads. It does not expose a set of context row embeddings plus query row embeddings as the primary interface.

**Architectural gap**

There is no clean boundary:

```text
TabularEncoder(X_context, x_query) -> {z_context_i}, {z_query_q}
SetICLModel({z_context_i, y_i}, z_query_q) -> y_hat_q
```

**Engineering judgment**

This boundary would make the architecture more foundation-model-like and more reusable. It is also better aligned with ICL, because the final model can reason over labeled examples in representation space.

**Recommended solution**

Split the future model into two stages:

```text
Stage 1: Context-conditioned tabular encoder
  phi(x_i; X_context) -> z_i

Stage 2: ICL set model
  g({(z_i, y_i)}_{i=1..n}, z_query) -> y_hat
```

This should be considered a new model family or major MODEL3 variant, not a small patch to MODEL2.

### Gap 5: Query Rows Need the Same Distribution-Aware Encoding

**Specification**

Feature queries should be interpreted relative to the same column distributions.

**Current status**

Partially satisfied.

MODEL2 normalizes `x_test` using context feature statistics and includes query values in the six-channel feature token. That is directionally correct.

**Architectural gap**

There is no explicit query row encoder:

```text
z_q = phi(x_q; X_context)
```

using the same column embeddings as context rows.

**Engineering judgment**

This should be required. Context rows and query rows must share the same encoder and column distribution reference, otherwise the ICL model can learn inconsistent representations.

**Recommended solution**

Use the same `ColumnEncoder` outputs `c_j` for:

- context cell embeddings `e_ij`,
- query cell embeddings `e_qj`.

Then:

```text
z_context_i = RowEncoder({e_ij}_j)
z_query_q   = RowEncoder({e_qj}_j)
```

### Gap 6: Need Clear Role and Label Encoding for ICL

**Specification**

Leverage in-context learning like TabPFN/TabPFNv2.

**Current status**

Partially satisfied.

MODEL2 uses context labels inside token construction and predicts query targets in one forward pass. But it does not clearly separate row embeddings, label embeddings, role embeddings, and query embeddings.

**Architectural gap**

There is no explicit ICL sequence/set object:

```text
context tokens: (z_i, y_i, role=context)
query tokens:   (z_q, role=query)
```

**Engineering judgment**

This matters for a foundation-model-like architecture. The model should know whether a row is labeled context or unlabeled query through an explicit role/label pathway, not only through handcrafted interactions.

**Recommended solution**

Add:

- label encoder for `y_i`,
- role embeddings for context/query rows,
- optional task/regime embeddings,
- masked target token for query rows.

Then run a permutation-invariant/equivariant ICL block over rows.

### Relationship to Hartford Section

The embedding-then-ICL proposal and the Hartford exchangeable-layer proposal are compatible, but they solve different parts of the architecture.

Recommended integration:

```text
ColumnEncoder / CellEncoder
  -> optional ExchangeableMatrixBlocks over (row, column)
  -> RowEncoder to produce z_i
  -> Set/Transformer ICL model over {(z_i, y_i)}
```

This preserves the MODEL2 principle of avoiding early feature collapse, while adding:

- distribution-aware cell interpretation,
- row embeddings,
- a cleaner ICL boundary,
- and optional Hartford-style exchangeable matrix/tensor structure.

### Recommended Model Family Boundary

Do not mutate current `retired MODEL2 model` into this architecture incrementally. The changes are large enough to justify a new family:

```text
model_family = "market_tabicl"
```

or, if combined with Hartford exchangeable layers:

```text
model_family = "market_exchangeable_icl"
```

MODEL2 should remain the production baseline until the new encoder passes:

- row permutation invariance,
- feature permutation consistency,
- query encoding consistency,
- column distribution perturbation tests,
- missingness tests,
- synthetic regression recovery,
- checkpoint compatibility,
- Snowflake memory guardrails.

### Tabular ICL Gap Summary

| Requirement | MODEL2 Status | My Judgment | Recommended Solution |
|---|---:|---|---|
| TabPFN-style ICL | Partial | Directionally aligned, not cleanly separated | Add explicit row-token ICL stage |
| Distribution-aware column embeddings | Missing | Important for tabular foundation behavior | Add `ColumnEncoder` |
| Cell embedding depends on value and column distribution | Partial | Current handcrafted tokens are too narrow | Add `DistributionAwareCellEncoder` |
| Row-wise feature-token interaction | Partial | Needed before final set model | Add `RowEncoder` over feature tokens |
| `z_i = phi(x_i; X_context)` | Missing as explicit API | Central target interface | Split encoder from ICL model |
| Final set model consumes `{z_i}` | Missing | Needed for clean ICL abstraction | Add `SetICLModel` over row embeddings |
| Shared context/query encoder | Partial | Must be explicit | Encode context/query with same column embeddings |
| Label/role encoding | Partial | Needed for robust ICL | Add label and context/query role encoders |

## Dataset-Wise In-Context Learning Revision

### Core Requirement

The architecture should perform dataset-wise in-context learning. The model should reason over the entire labeled context dataset:

```text
D_context = {(x_i, y_i)} for i = 1..n_train
```

plus all test/query rows:

```text
X_test = {x_q} for q = 1..n_test
```

The goal is to infer relationships between feature values and target values from the context in a single forward pass, rather than requiring gradient-based retraining for every new dataset.

This requirement is consistent with the DeepSet principle: the training context is a set, and the model output should not depend on the arbitrary order of context rows.

### Consistency With MODEL2

MODEL2 is already directionally aligned with dataset-wise ICL:

- `retired MODEL2 model.forward(X_train, y_train, x_test)` receives the full context set and query rows.
- It normalizes features and targets using context statistics.
- It builds query-conditioned evidence from the context.
- It is row-permutation invariant over context rows.
- It supports batched query rows.
- It does not require per-dataset gradient updates at evaluation time.

Therefore, this section does not invalidate MODEL2. It makes the ICL contract more explicit and identifies what a future model should strengthen.

### Current MODEL2 Interpretation

MODEL2 currently implements a dataset-wise contextual predictor:

```text
f_theta(X_train, y_train, X_test) -> y_hat_test
```

For each query row, MODEL2 constructs feature/sample evidence using:

```text
(y_i, X_ij, x_qj, X_ij * x_qj, X_ij * y_i, |X_ij - x_qj|)
```

Then it aggregates sample evidence per feature, applies feature interaction, and predicts the query target.

This is a valid DeepSet-style ICL mechanism. The context set affects predictions through the forward pass, not through gradient updates.

### Gap 1: Dataset-Level Context Is Implicit, Not a First-Class Object

**Current status**

Partially satisfied.

MODEL2 receives the full context tensors, but the architecture does not expose a formal dataset-context representation such as:

```text
C_D = DatasetContextEncoder({(z_i, y_i)}_{i=1..n_train})
```

Instead, dataset context is used through query-conditioned feature evidence.

**Architectural gap**

There is no explicit global dataset embedding or dataset state that summarizes:

- feature-target relationships,
- target distribution,
- noise level,
- sparsity,
- feature relevance,
- regime structure,
- train/test distribution shift.

**Engineering judgment**

MODEL2 is sufficient for current synthetic regression evaluation, but a foundation-model-like tabular ICL architecture should expose a clearer dataset-context abstraction.

**Recommended solution**

Introduce an explicit dataset context encoder in a future model family:

```text
Z_context = {z_i = phi(x_i; X_context)}
C_D = DatasetICLEncoder({(z_i, y_i)})
```

Then predict:

```text
y_hat_q = QueryReadout(z_q, C_D)
```

This preserves DeepSet invariance while making dataset-level reasoning testable.

### Gap 2: Query Rows Are Processed Mostly Independently

**Current status**

Partially satisfied.

MODEL2 supports batched `x_test`, but each query is primarily evaluated against the context. It does not explicitly reason over the test set as a set:

```text
X_test = {x_q}_{q=1..n_test}
```

**Architectural gap**

There is no explicit query-set interaction block that lets test/query rows inform each other under a permutation-equivariant query-set contract.

**Engineering judgment**

For standard supervised prediction, independent query readout is acceptable. For market mental model workflows, query-set interaction may help when the test rows form a coherent scenario, time slice, portfolio, or stress surface.

**Recommended solution**

Add an optional query-set ICL block:

```text
Z_query = {z_q}
Z_query' = QuerySetBlock(Z_query, C_D)
```

This block must be permutation equivariant over query rows. It should be optional, because some evaluation protocols assume predictions are conditionally independent given the context.

### Gap 3: Dataset-Wise ICL Needs Explicit Label/Role Tokens

**Current status**

Partially satisfied.

MODEL2 uses `y_train` directly in feature evidence. This works, but it does not explicitly represent labeled context rows and unlabeled query rows as different token roles.

**Architectural gap**

There is no formal sequence/set like:

```text
context token: (z_i, y_i, role=context)
query token:   (z_q, mask_y, role=query)
```

**Engineering judgment**

For robust ICL, role and label encoding should be explicit. This reduces reliance on handcrafted feature interactions and better matches foundation-model style prompting.

**Recommended solution**

Use:

- shared row encoder for context/query rows,
- label encoder for `y_i`,
- missing-label token for query rows,
- role embedding for context vs query,
- row-set ICL block over all context and query row tokens.

Conceptually:

```text
T_context_i = concat(z_i, label_embed(y_i), role_context)
T_query_q   = concat(z_q, label_mask_token, role_query)

T_all = T_context union T_query
T_out = RowSetICLBlock(T_all)
y_hat_q = Head(T_out_query_q)
```

The block must preserve permutation invariance/equivariance over context rows and query rows.

### Gap 4: Training Objective Should Match Dataset-Wise ICL

**Current status**

Partially satisfied.

The current synthetic regression procedure evaluates in-context prediction, but the architecture document should explicitly state the intended objective:

```text
learn theta such that f_theta(D_context, X_test) predicts y_test
```

**Architectural gap**

If training is framed only as row-wise regression, the model may not be pressured to learn dataset-level adaptation. The task should sample entire datasets/episodes, not isolated rows.

**Engineering judgment**

This is critical. Dataset-wise ICL is learned from episodic training. The unit of training should be a dataset episode with a context/query split.

**Recommended solution**

Preserve the episodic meta-dataset training structure:

```text
episode = (D_context, D_query)
loss = sum_q L(f_theta(D_context, x_q), y_q)
```

For future `market_tabicl` or `market_exchangeable_icl`, the architecture and dataloader should continue using full dataset episodes rather than row-wise IID minibatches.

### Dataset-Wise ICL Contract

A future ICL-aligned model should implement the following contract:

```text
Inputs:
  X_context: (n_train, p)
  y_context: (n_train,)
  X_query:   (n_test, p)

Output:
  y_query_hat: (n_test,)
```

Required invariances:

1. Context row permutation invariance:

   ```text
   f({(x_i, y_i)}, X_query) = f({(x_perm(i), y_perm(i))}, X_query)
   ```

2. Query row permutation equivariance:

   ```text
   f(D_context, P_query X_query) = P_query f(D_context, X_query)
   ```

3. Feature permutation consistency, when feature identities are not semantically fixed:

   ```text
   f(X_context P_feature, y, X_query P_feature) = f(X_context, y, X_query)
   ```

4. No gradient update at evaluation time:

   ```text
   theta fixed; adaptation occurs through D_context only
   ```

### Dataset-Wise ICL Gap Summary

| Requirement | MODEL2 Status | My Judgment | Recommended Solution |
|---|---:|---|---|
| Full `D_context` consumed in forward pass | Satisfied | MODEL2 already does this | Preserve current contract |
| No per-dataset gradient retraining at eval | Satisfied | Core production behavior | Preserve checkpoint/eval path |
| Explicit dataset context state | Missing | Useful for stronger ICL | Add `DatasetICLEncoder` |
| Test rows as a query set | Partial | Optional, task-dependent | Add optional `QuerySetBlock` |
| Label/role tokens | Partial | Important for foundation-style ICL | Add label and role encoders |
| Episodic dataset-level training objective | Partial/documentation gap | Critical for ICL | Train on context/query episodes |
| Context permutation invariance | Satisfied by tests | Keep as hard guardrail | Continue permutation tests |
| Query permutation equivariance | Partial | Needed for set-valued prediction | Add query-set tests |

## SetNorm and Clean-Path Equivariant Residual Revision

### Paper Context and Engineering Judgment

This section addresses Zhang et al., *Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets*.

The key architectural lesson is that deeper set models need residual paths and normalization schemes that respect set structure. Standard LayerNorm, especially when inserted directly into every set block, can remove useful scale/distribution information. For tabular ICL and market priors, that information may be predictive:

- feature scale,
- target scale,
- distribution sharpness,
- regime noise,
- row/column dispersion,
- missingness intensity.

The proposed clean-path residual design is sound and should be adopted as a future depth/stability guardrail. But it must be applied without contradicting MODEL2:

- MODEL2's current pooling order remains structurally sound.
- `n_sab_sample_per_feature=0` remains the production memory guard.
- Clean residual paths should wrap equivariant set/row/column blocks, not introduce non-equivariant shortcuts.
- SetNorm should be evaluated as an alternative normalization path, not blindly substituted everywhere.

### Requested Specification

A clean-path equivariant residual block has:

```text
x_{l+1} = x_l + f(x_l)
```

where `f` is permutation equivariant:

```text
f(pi x) = pi f(x)
```

The shortcut branch must carry `x_l` unchanged. It must not apply:

- normalization,
- pooling,
- attention,
- activation,
- linear projection,
- gating,
- dropout.

All transformations belong only on the `f(x_l)` branch.

### Current MODEL2 Status

MODEL2 uses residual connections inside the existing attention blocks inherited from `MAB`/`SAB` style components, but it does not expose a project-level clean-path residual contract for every set/equivariant block.

MODEL2 also uses normalization mainly through feature/target preprocessing, not a formal SetNorm layer for set hidden states.

This is not a defect in MODEL2. MODEL2 is intentionally compact and memory-guarded. The gap appears when we want to deepen MODEL2 or future `market_tabicl` / `market_exchangeable_icl` models.

### Gap 1: No Explicit Clean-Path Residual Contract

**Specification**

Every deep set/equivariant embedding block should preserve a clean identity path:

```text
x_{l+1} = x_l + f(x_l)
```

with `x_l` passed through unchanged.

**Current status**

Partially satisfied at the internal attention-block level, but not enforced architecturally across all row/column/cell embedding layers.

**Architectural gap**

Future components such as:

- `ColumnEncoder`,
- `DistributionAwareCellEncoder`,
- `RowEncoder`,
- `DatasetICLEncoder`,
- `QuerySetBlock`,
- `ExchangeableMatrixBlock`,
- `ExchangeableTensorBlock`,

do not yet have a documented rule requiring clean-path equivariant residuals.

**Engineering judgment**

This rule should become mandatory for any future deep set or exchangeable stack. It is especially important for distribution-aware embeddings because normalizing or projecting the shortcut can erase the exact distribution cues the model needs for ICL.

**Recommended solution**

Define a reusable clean residual wrapper concept:

```text
CleanEquivariantResidualBlock:
  input:  x
  output: x + f(x)
```

Constraints:

- `f` must be permutation equivariant for the relevant axis or axes.
- The shortcut branch must be exactly identity.
- If dimensions differ, do not silently project the shortcut. Instead, require `f(x)` to return the same shape as `x`, or use an explicit architecture boundary before entering the residual stack.
- Dropout, activation, attention, pooling, normalization, and gating are allowed only inside `f`.

### Gap 2: LayerNorm May Destroy Useful Set/Distribution Information

**Specification**

LayerNorm can hurt set-model performance by normalizing away information useful for prediction. SetNorm should be considered for deeper set models.

**Current status**

Partially addressed through context feature normalization, but not through hidden-state SetNorm.

MODEL2 standardizes features and targets per context. That is not the same as hidden-state normalization inside a deep set architecture.

**Architectural gap**

There is no hidden-state normalization policy for future deep set / tabular ICL blocks. If future blocks use default Transformer LayerNorm everywhere, they may remove useful dataset-level scale and dispersion signals.

**Engineering judgment**

For tabular ICL, normalization should be explicit and conservative. We should not automatically import Transformer LayerNorm conventions into set models. Dataset statistics are often signal, not nuisance.

**Recommended solution**

For future deep set blocks, evaluate:

```text
normalization = "none" | "set_norm" | "layer_norm"
```

Default recommendation:

- keep MODEL2 as-is,
- use no hidden-state norm or SetNorm in early experiments,
- avoid LayerNorm on the clean residual path,
- if using LayerNorm, keep it inside `f(x_l)` only and validate against SetNorm.

SetNorm should preserve set-level structure by normalizing in a way compatible with permutation symmetry, rather than independently flattening away useful per-set statistics.

### Gap 3: Column-Wise and Row-Wise Embeddings Need Clean Residual Paths

**Specification**

The model should preserve column-wise and row-wise embeddings for ICL while enabling clean-path residuals where the input passes through each embedding layer unchanged on the shortcut.

**Current status**

Future-facing requirement. MODEL2 currently preserves feature identity until feature SAB, but it does not have the proposed explicit column/cell/row embedding stack.

**Architectural gap**

The proposed `ColumnEncoder -> CellEncoder -> RowEncoder -> DatasetICLEncoder` path needs a residual policy at each level:

```text
column embeddings: c_j
cell embeddings:   e_ij
row embeddings:    z_i
dataset tokens:    t_i
```

Without a clean-path rule, a future model could accidentally normalize/project away information at each level.

**Engineering judgment**

The clean-path principle is valuable precisely because the earlier sections propose richer embeddings. Once embeddings become distribution-aware, we should preserve them through depth rather than repeatedly re-normalizing them.

**Recommended solution**

Apply clean residual blocks by representation type:

```text
c_j^{l+1}   = c_j^l   + f_col(C^l)_j
e_ij^{l+1}  = e_ij^l  + f_cell(E^l)_{ij}
z_i^{l+1}   = z_i^l   + f_row(Z^l)_i
t_i^{l+1}   = t_i^l   + f_icl(T^l)_i
```

Where:

- `f_col` is column-set equivariant,
- `f_cell` is row/column exchangeable or tensor-equivariant,
- `f_row` is feature-token or row-set equivariant,
- `f_icl` is context/query row-set equivariant.

### Gap 4: Dimension Changes Conflict With Clean Identity Paths

**Specification**

The shortcut path should pass the input through as-is.

**Current status**

MODEL2 uses fixed dimensions within its main blocks. Future encoder stacks may be tempted to change dimensions inside residual blocks.

**Architectural gap**

A residual shortcut cannot be a clean path if it requires projection:

```text
shortcut = W x_l
```

That violates the "as-is" requirement.

**Engineering judgment**

Dimension changes should happen at explicit stage boundaries, not inside clean residual stacks.

**Recommended solution**

Use stage boundaries:

```text
raw scalar/categorical value -> initial embedding projection -> fixed-width clean residual stack
```

Then:

```text
fixed-width stack -> explicit pooling/readout/projection
```

Within each stack, require:

```text
input_dim == output_dim
```

### Gap 5: Need Tests That Prove Clean-Path Equivariance

**Current status**

MODEL2 has permutation tests for current row/feature behavior. It does not have future clean-path tests because those blocks do not exist yet.

**Architectural gap**

Future architecture changes could accidentally:

- normalize the shortcut,
- project the shortcut,
- pool on the shortcut,
- break equivariance,
- destroy query permutation equivariance.

**Recommended solution**

For every future block, require tests:

1. Clean identity branch test:

   ```text
   block(x, zero_transform=True) == x
   ```

2. Equivariance test:

   ```text
   block(pi x) == pi block(x)
   ```

3. Shortcut purity inspection:

   verify shortcut branch contains no norm/projection/dropout/attention.

4. Gradient flow test:

   verify gradients flow through both the residual sum and transform branch.

5. SetNorm ablation:

   compare `none`, `set_norm`, and `layer_norm` on synthetic regression sanity tasks.

### Recommended Normalization and Residual Policy

For MODEL2:

- Do not change current production architecture solely for SetNorm.
- Do not enable deeper sample SAB over `n` without respecting MODEL2 memory guardrails.

For future `market_tabicl`, `market_exchangeable`, or `market_exchangeable_icl`:

```text
Initial embedding projection:
  allowed to change dimension

Clean residual stack:
  x_{l+1} = x_l + f(x_l)
  no shortcut transformation
  f is equivariant
  optional SetNorm inside f only

Stage boundary:
  pooling/readout/projection allowed
```

### SetNorm / Clean Residual Gap Summary

| Requirement | MODEL2 Status | My Judgment | Recommended Solution |
|---|---:|---|---|
| Clean identity residual path | Partial | Needed for deeper future models | Add clean residual block contract |
| Equivariant transform branch | Partial | Mandatory for set/tensor stacks | Require `f(pi x)=pi f(x)` tests |
| No norm/projection on shortcut | Not formalized | Should be hard guardrail | Shortcut must be identity |
| Hidden-state SetNorm | Missing | Evaluate before LayerNorm | Add `normalization` policy |
| Preserve row/column embeddings | Future-facing | Critical for TabICL path | Clean residuals over `c_j`, `e_ij`, `z_i`, `t_i` |
| Dimension changes | Not formalized | Keep outside residual blocks | Use explicit stage boundaries |
| Tests for clean-path behavior | Missing | Required before implementation | Add shortcut/equivariance/gradient tests |

## Induced Set Attention Block Revision for MODEL3

### Verification Against MODEL2

MODEL2 does **not** currently use Induced Set Attention Blocks.

The implementation in `src/model.py` defines:

```text
MAB(Q, K) = MultiheadAttention(Q, K, K) + residual/norm/FFN
SAB(X)    = MAB(X, X)
```

`retired MODEL2 model` then uses:

```text
sab_sample  # optional, over samples n inside each (query, feature)
sab_feat    # over feature tokens p
```

This is consistent with `MODEL2.md`, which explicitly warns that enabling sample-level SAB over `n` can create a large attention matrix:

```text
(m * p, n, n)
```

and therefore keeps:

```text
n_sab_sample_per_feature = 0
```

as the production memory guard.

### Engineering Judgment

ISAB is a good MODEL3 candidate, but it should not be retrofitted into MODEL2 without a separate validation cycle.

The reason is not architectural incompatibility. ISAB is compatible with DeepSet/Set Transformer equivariance when implemented correctly. The issue is production risk:

- MODEL2 is already validated and checkpointed around SAB/MAB behavior.
- MODEL2's memory guard relies on not enabling sample SAB over `n`.
- Replacing SAB with ISAB changes representation dynamics and checkpoint compatibility.

Therefore, ISAB belongs in the future `market_tabicl`, `market_exchangeable`, or `market_exchangeable_icl` family.

### ISAB Specification

In Set Transformer terminology, ISAB uses a fixed number of learnable inducing points to reduce attention cost.

Instead of full self-attention:

```text
SAB(X) = MAB(X, X)
```

with quadratic cost in set size, ISAB uses inducing points:

```text
H = MAB(I, X)
Y = MAB(X, H)
```

where:

```text
I: learnable inducing points, shape (m_induce, d)
```

This gives attention cost roughly:

```text
O(n * m_induce)
```

instead of:

```text
O(n^2)
```

for a set of size `n`, assuming `m_induce << n`.

### Gap 1: MODEL2 Uses SAB/MAB, Not ISAB

**Current status**

MODEL2 uses SAB for feature-level interaction and optional sample-level interaction. It does not define or instantiate ISAB.

**Architectural gap**

For future dataset-wise ICL or exchangeable tensor models, full SAB can become expensive over:

- context rows `n_train`,
- query rows `n_test`,
- feature tokens `p`,
- row/column cells in exchangeable matrix blocks,
- higher-order market tensor axes.

**Engineering judgment**

SAB over features `p` is acceptable when `p` is capped and small enough. SAB over rows or cells is the risk. ISAB is most valuable for large context/query row sets and large exchangeable cell sets.

**Recommended solution**

Add ISAB as an optional attention primitive in MODEL3:

```text
attention_block = "sab" | "isab" | "linear_equivariant"
n_inducing_points = k
```

Use ISAB for large set axes:

```text
RowSetICLBlock
QuerySetBlock
DatasetICLEncoder
large feature-token RowEncoder
```

Do not replace MODEL2's existing `sab_feat` by default.

### Gap 2: Sample-Level Attention Is Still Memory-Risky

**Current status**

MODEL2 avoids sample-level SAB by default:

```text
n_sab_sample_per_feature = 0
```

This is consistent with `MODEL2.md`.

**Architectural gap**

Future dataset-wise ICL wants deeper interaction over context rows. A naive SAB over all rows would reintroduce O(n²) memory pressure.

**Engineering judgment**

For dataset-wise ICL, ISAB is preferable to full SAB on the row axis. It provides a middle ground between:

- no row attention,
- full quadratic row attention.

**Recommended solution**

For MODEL3, use:

```text
RowSetICLBlock = ISAB stack over row tokens
```

with a bounded number of inducing points:

```text
n_inducing_points = 16, 32, or 64
```

The exact value should be tuned under Snowflake GPU memory constraints.

### Gap 3: ISAB Must Preserve Clean-Path Equivariant Residual Rules

**Current status**

The SetNorm section defines clean-path residual constraints for future deeper models.

**Architectural gap**

ISAB must be inserted without violating:

```text
x_{l+1} = x_l + f(x_l)
```

where the shortcut is exact identity and `f` is permutation equivariant.

**Engineering judgment**

ISAB can be the equivariant transform branch `f`. It should not be applied to the shortcut branch.

**Recommended solution**

Use ISAB inside the transform branch only:

```text
CleanEquivariantResidualBlock:
  y = x + ISAB_or_FFN_or_ExchangeableTransform(x)
```

If normalization is needed, use SetNorm or another set-compatible normalization inside `f(x)`, not on the shortcut.

### Gap 4: Inducing Points Need Axis-Specific Semantics

**Current status**

MODEL2 has no inducing points.

**Architectural gap**

Future architectures may apply ISAB over different axes:

- feature tokens,
- row/context tokens,
- query tokens,
- exchangeable matrix cells,
- tensor-axis tokens.

Using one shared inducing-point bank everywhere may blur semantics.

**Engineering judgment**

Inducing points should be axis/block-specific. A row-set ISAB and a feature-token ISAB should not necessarily share the same inducing points.

**Recommended solution**

Define separate inducing-point modules:

```text
row_inducing_points
feature_inducing_points
cell_inducing_points
query_inducing_points
```

Only share them if an ablation proves sharing is beneficial.

### Gap 5: Need ISAB Equivariance and Memory Tests

**Current status**

MODEL2 has permutation tests for current SAB/MAB behavior, but no ISAB tests because ISAB does not exist.

**Recommended solution**

MODEL3 ISAB tests should include:

1. Permutation equivariance:

   ```text
   ISAB(PX) = P ISAB(X)
   ```

2. Clean residual identity:

   ```text
   block(x, zero_transform=True) = x
   ```

3. Memory scaling:

   verify attention memory scales with `n * m_induce`, not `n^2`.

4. Device correctness:

   inducing points must live on the same CUDA device as the input.

5. Batch/query shape stability:

   outputs retain `(batch, set, d)` shape.

### ISAB Recommendation for MODEL3

For future `market_tabicl`:

```text
ColumnEncoder / CellEncoder
  -> RowEncoder over feature tokens
  -> DatasetICLEncoder using ISAB over context row tokens
  -> optional QuerySetBlock using ISAB over query tokens
```

For future `market_exchangeable_icl`:

```text
ExchangeableMatrixBlock for row/column/global cell updates
  -> ISAB over row tokens or cell summaries when set size is large
  -> SetICL readout
```

Do not claim that ISAB replaces Hartford exchangeable matrix layers. They solve different problems:

- Hartford exchangeable layers impose row/column matrix parameter sharing.
- ISAB reduces attention cost for large set attention.

They can be combined, but one is not a substitute for the other.

### Correct Integration Order: Hartford First, Then ISAB

The correct MODEL3 direction is to use Hartford-style exchangeable matrix/tensor layers before ISAB, but with one precise ordering clarification:

Hartford layers should come **after initial column/cell embedding** and **before irreversible pooling or row/query readout**.

They should not operate directly on raw scalar cells unless the implementation intentionally uses a minimal raw-cell channel schema. The preferred order is:

```text
raw context/query values
  -> distribution-aware column embeddings
  -> cell/channel embeddings
  -> Hartford exchangeable matrix/tensor blocks
  -> row/feature/query embeddings
  -> ISAB over embedding sets for scalable dataset-wise ICL
  -> prediction head
```

This ordering is consistent with DeepSet architecture and MODEL2 because:

- context samples remain an unordered set,
- feature tokens remain exchangeable unless feature identity is explicitly semantic,
- no irreversible sample or feature pooling happens before structured interaction,
- `retired MODEL2 model` remains the current production baseline,
- MODEL3 extends MODEL2's "do not collapse feature identity early" principle.

#### Per-Query Hartford Tensor

For each query row `x_q`, MODEL2 currently builds query-conditioned feature/sample evidence similar to:

```text
[(y_1, X_1j, xq_j, ...),
 (y_2, X_2j, xq_j, ...),
 ...
 (y_n, X_nj, xq_j, ...)]
```

for each feature `j`.

MODEL3 should generalize this into a per-query interaction tensor:

```text
H_q: (n_train, p, c)
```

where:

- axis 1 = context samples,
- axis 2 = features,
- axis 3 = channels,
- `xq_j` is broadcast across the sample axis for feature `j`,
- `y_i` is broadcast across the feature axis for sample `i`,
- column distribution embeddings are broadcast into feature channels,
- optional row/context metadata is broadcast into sample channels,
- missingness or observed-entry masks are included as channels.

Then apply Hartford exchangeable matrix blocks over the `(sample, feature)` axes:

```text
H_q' = ExchangeableMatrixBlock(H_q)
H_q'' = ExchangeableMatrixBlock(H_q')
```

This lets every cell learn from:

- itself,
- its sample row,
- its feature column,
- the full query-conditioned context matrix.

Only after this stage should MODEL3 derive:

```text
sample/query-conditioned row embeddings
feature tokens
query embeddings
dataset context embeddings
```

#### Then Apply ISAB Over Embedding Sets

After Hartford blocks preserve and update the structured `(sample, feature, channel)` representation, ISAB should be used for scalable set-level ICL over the derived embeddings.

Examples:

```text
Z_context = {z_i}_i
Z_query   = {z_q}_q
F_query   = {f_{qj}}_j
```

Then:

```text
DatasetICLEncoder = ISAB(Z_context)
QuerySetBlock     = ISAB(Z_query)
FeatureSetBlock   = ISAB(F_query)   # optional when p is large
```

This is the correct division of labor:

- Hartford layers preserve structured sample-feature matrix equivariance.
- ISAB makes large set-level ICL computationally scalable.

#### Important Constraint

Do not use ISAB as a replacement for the Hartford exchangeable matrix layer.

If the model only applies ISAB to row embeddings after pooling away the sample-feature matrix, it loses the Hartford property:

```text
f(P_sample H P_feature) = P_sample f(H) P_feature
```

Therefore, for MODEL3:

```text
Hartford exchangeable blocks must occur before pooling from (sample, feature, channel)
to row/query embeddings.
```

### ISAB Gap Summary

| Requirement | MODEL2 Status | My Judgment | Recommended Solution |
|---|---:|---|---|
| Uses ISAB | No | Correct for current MODEL2; not a defect | Add ISAB only in MODEL3 |
| Reduces O(n²) row attention | Missing | Needed for large dataset-wise ICL | Use row-set ISAB |
| Preserves DeepSet equivariance | Future-facing | Compatible if implemented correctly | Test `ISAB(PX)=PISAB(X)` |
| Maintains clean shortcut | Missing | Must follow SetNorm section | Put ISAB only in `f(x)` branch |
| Inducing point semantics | Missing | Should be axis-specific | Separate inducing-point banks |
| Snowflake memory safety | Future-facing | Required before production | Add memory-scaling tests |

## Recommended Architecture Direction

### Keep MODEL2 as Current Production Baseline

`retired MODEL2 model` should remain the current production architecture until a Hartford-aligned model is implemented and validated.

It is already integrated with:

- pretrain,
- HPO,
- final training,
- synthetic regression evaluation,
- checkpoint gates.

### Add MODEL3 as the Hartford-Aligned Architecture

Recommended new family:

```text
model_family = "market_exchangeable"
```

If the primary target is embedding-then-ICL rather than exchangeable matrix completion, the recommended family name is:

```text
model_family = "market_tabicl"
```

If both Hartford exchangeable blocks and embedding-then-ICL are combined:

```text
model_family = "market_exchangeable_icl"
```

Target flow:

```text
Inputs:
  X_train: (n, p)
  y_train: (n)
  x_test:  (m, p)
  optional row features
  optional column features
  optional market/regime features
  optional observed mask

Build:
  H0: (m, n, p, channels)

Apply:
  H1 = ExchangeableMatrixBlock(H0, mask)
  H2 = ExchangeableMatrixBlock(H1, mask)
  ...

Readout:
  query-conditioned beta/residual/prior head

Output:
  y_hat: (m,)
```

For higher-order market tensors:

```text
H0: (batch, axis_1, axis_2, ..., axis_D, channels)
```

Use exchangeable tensor blocks following Hartford's tensor extension.

## What I Would Not Do

1. I would not replace MODEL2 with a factorized exchangeable autoencoder for the current synthetic regression evaluator.
2. I would not add arbitrary positional embeddings to row/column axes unless the axis is genuinely ordered and no longer exchangeable.
3. I would not treat missing values as zeros without an explicit mask channel.
4. I would not claim Hartford compliance from SAB alone. Attention over a set can be permutation equivariant, but it is not the same as Hartford's row/column/global exchangeable matrix layer.
5. I would not merge inductive and transductive objectives into one opaque head. They should be separate modes or separate model families.

## Gap Summary

| Requirement | MODEL2 Status | My Judgment | Recommended Solution |
|---|---:|---|---|
| Row/column matrix equivariance | Partial | Needed for market interaction tensors | Add `ExchangeableMatrixBlock` |
| Self/row/column/global updates | Missing | Highest-priority Hartford gap | Implement tied matrix reductions |
| Deep equivariant stack | Partial | Add only after primitive is correct | Stack exchangeable blocks |
| Multi-channel schema | Partial | Needed for market data | Formal channel contract |
| Row/column side features | Missing | Needed for real market priors | Broadcast as channels |
| Sparse observed-entry handling | Missing | Must be first-class | Masked reductions + mask channel |
| Higher-order tensor support | Missing | Needed for structured market tensors | `ExchangeableTensorBlock` |
| Factorized EAE | Missing | Only needed for transductive completion | Separate model mode/family |
| Current synthetic regression fit | Good enough | Keep MODEL2 for now | Upgrade after MODEL3 validation |

## Acceptance Criteria for a Hartford-Aligned Future Model

A future `market_exchangeable` model should pass:

1. Row permutation equivariance:

   ```text
   f(P_rows X) = P_rows f(X)
   ```

2. Column permutation equivariance:

   ```text
   f(X P_cols) = f(X) P_cols
   ```

3. Joint row/column equivariance:

   ```text
   f(P_rows X P_cols) = P_rows f(X) P_cols
   ```

4. Stacked-layer equivariance.
5. Sparse-mask equivariance.
6. Multi-channel row/column/cell equivariance.
7. Inductive query-readout correctness.
8. Separate transductive reconstruction correctness if using a factorized exchangeable autoencoder.

## Final Recommendation

MODEL2 is a strong intermediate architecture. It fixes early feature collapse and supports the current synthetic regression pipeline.

But relative to Hartford et al., it is not yet the right final architecture for a market mental model built on structured interactions across exchangeable axes.

The next principled step is not to keep stretching `retired MODEL2 model`; it is to design `MODEL3 = MarketExchangeableModel` around exchangeable matrix/tensor blocks, mask-aware reductions, and a clear inductive-vs-transductive objective split.
