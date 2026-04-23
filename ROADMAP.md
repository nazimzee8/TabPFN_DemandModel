# Market-Specific TabPFN Implementation Roadmap

## Objective

Build a TabPFN-style, prior-data fitted Transformer for market elasticity modeling that estimates the Posterior Predictive Distribution (PPD) under a custom Bayesian Neural Network (BNN) prior. The model is trained on synthetic market tasks in a single forward pass and uses structured attention masking to enforce known independence constraints between product prices and product demands. The roadmap also treats common-demand shocks and endogenous pricing effects as separate latent drivers so shared movement across products is not misclassified as substitution or complementarity.

The target domain is numerical, multi-label classification with no missing values. The output labels discretize elasticity outcomes into three intervals:

- Class 0: `(-inf, -0.1]`
- Class 1: `(-0.1, 0.5]`
- Class 2: `(0.5, inf)`

## Operating Assumptions

- Reproduction target: TabPFN-inspired reproduction with a different prior, not a strict byte-for-byte reimplementation of the original codebase.
- Hardware: single-GPU workstation (9800X3D + 9070 XT test environment).
- Roles: planning-only subagents. They define design, interfaces, acceptance criteria, and handoff contracts.
- Structural constraint: zero cross-price elasticity implies a hard independence prior that should be expressed first in synthetic task generation and then in model attention structure.
- Exogenous common-demand shocks: shared upward demand, quantity, or joint demand-quantity movements that affect multiple products at once should be modeled separately and weighted near zero when inferring cross-product substitution structure.
- Endogeneity control: latent factors that jointly affect price and demand for the same good must be represented explicitly so anticipatory pricing does not masquerade as causal price-demand coupling.
- Sparsity handling: masked pathways corresponding to zero-relationship edges should remain inactive during training and become pruning candidates after validation.

## System Goal

Approximate the PPD:

`p(y_unobserved | x_observed, D_context)`

by meta-training a Transformer on synthetic tasks sampled from a market-aware BNN prior:

`theta ~ p_market(theta | E)`

where `E` is the elasticity structure matrix encoding zero and non-zero dependencies. Additional latent variables capture exogenous common shocks and endogenous price-setting behavior, with inference structured to suppress their contribution to cross-product edge discovery unless strongly identified by instruments. The Transformer learns to map context examples and query features directly to predictive class probabilities in one forward pass.

## Subagent Hierarchy

### Lead AI Architect

Owns the overall execution order, interfaces, technical constraints, and decision gates. Coordinates all planning subagents and resolves design conflicts.

Direct reports:

- Prior Data Architect
- ML Engineer
- Architecture Engineer
- Market Logic Validator

### Prior Data Architect

Owns the synthetic market prior and task generator. Defines how elasticity structure is encoded into synthetic data so that tasks sampled for pretraining reflect valid market mechanics.

Primary outputs:

- Elasticity matrix specification
- Synthetic task family definitions
- BNN prior parameterization for market tasks
- Data sampling contracts for train/validation/meta-test task generation

### ML Engineer

Owns the training plan for prior-data fitted learning. Defines how the Transformer consumes synthetic tasks, approximates the PPD in one pass, and maps predictive distributions into the three label intervals.

Primary outputs:

- Meta-training loop design
- Loss design for multi-label classification
- Label discretization policy
- Hardware-aware training and batching strategy

### Architecture Engineer

Owns the model-structure plan. Defines how standard TabPFN-style attention is modified to consume a product-elasticity mask and how masked edges become pruning candidates.

Primary outputs:

- Tokenization plan for tabular examples
- `SparseAttention` design
- Mask propagation rules across layers and heads
- Post-training pruning criteria for zero-edge pathways

### Market Logic Validator

Owns the evaluation and failure detection plan. Validates that the learned model respects market structure, predicts useful elasticity classes, and does not leak signal across forbidden product pairs.

Primary outputs:

- Offline metrics suite
- Elasticity leakage tests
- Ablation plan for mask/pruning decisions
- Promotion gates before real-data inference

## Phase 1: Knowledge Encoding

### Goal

Define the market structure explicitly before model design. The elasticity matrix is the contract that aligns data generation, attention masking, and validation.

### Deliverables

- A square elasticity adjacency matrix `E` where:
  - `E[i, j] = 0` means product `j` price does not affect product `i` demand.
  - `E[i, j] != 0` means a dependency is permitted and should be represented in the prior.
- A directionality policy:
  - Demand response is directed from price features to demand targets.
  - Cross-price effects are not assumed symmetric unless explicitly encoded.
- A market segmentation schema:
  - Each synthetic task samples a market context with potentially different local elasticity magnitudes while preserving hard-zero edges.
- A confounder-control schema:
  - Separate latent variables distinguish true cross-price effects from exogenous market-wide demand shocks and endogenous price-demand co-movement.
  - GMM-style latent regime components with multiple instruments are used to isolate shared factors before they can be interpreted as substitution structure.

### Design Requirements

- Hard zeros are structural, not statistical accidents. They must be enforced in generation.
- Exogenous factors that only increase demand, quantity, or both across multiple products must be represented as shared latent shocks and assigned near-zero edge weight in the inferred cross-product graph.
- Endogenous factors that jointly shift price and demand for the same good must be encoded as separate latent causes and instrumented where possible.
- Non-zero edges should be sampled from realistic regimes:
  - substitutes
  - complements
  - weakly related goods
  - dominant anchor products
- The matrix should support product groups so masks can be generated automatically for larger catalogs.

### Exit Criteria

- The team can generate a task specification from `E` without ambiguity.
- Zero-edge rules are machine-checkable and flow into both training and evaluation.
- Confounder channels are explicitly separated from cross-product elasticity channels in the prior specification.

## Phase 2: The Masking Layer

### Goal

Implement a `SparseAttention` wrapper around TabPFN-style attention so the model cannot route information through forbidden product-demand edges.

### Core Design

- Extend attention with a product-elasticity mask derived from `E`.
- The mask is applied as an additive attention bias or a hard boolean mask before softmax.
- Forbidden edges receive `-inf` logits (or equivalent masked value) so attention probability is zero.

### Masking Semantics

- Price tokens may attend only to target-demand query positions allowed by `E`.
- Context examples can still provide shared example-level information, but product-level cross-talk must respect the structural mask.
- The mask must be stable across all heads by default.
- Head-specific relaxation is optional and should be treated as a later ablation, not the base plan.

### Engineering Decisions

- Keep masking external to core parameter tensors so the structural prior remains inspectable.
- Treat the mask as part of the batch/task metadata, enabling market-specific inference.
- Log mask utilization statistics per layer to detect accidental bypass paths.

### Exit Criteria

- There is no valid computational path for a forbidden price-demand dependency within masked attention.
- The mask format is aligned with tokenization and task batching.

## Phase 3: Synthetic Pre-Training

### Goal

Generate synthetic supervised tasks from a market-aware BNN prior so the Transformer learns the implied PPD over labels conditioned on context data.

### Prior Family

- Sample latent market parameters `theta` from a structured BNN prior conditioned on `E`.
- Constrain parameter draws so zero-edge elasticities remain exactly zero.
- Sample exogenous shared-shock variables from a Gaussian Mixture Model with multiple instrument-linked components so market-wide demand increases can be separated from product-to-product effects.
- Sample endogenous latent variables that jointly influence price and demand, representing cases such as anticipatory price increases before high demand.
- Down-weight the contribution of these exogenous and endogenous latent factors when inferring cross-product edges unless the instrument evidence identifies a direct relationship.
- Allow non-zero elasticities to vary by market regime, noise level, and competition pattern.

### Synthetic Task Components

- Inputs:
  - product prices
  - optional market covariates (numerical only)
  - contextual pricing patterns across examples
  - instrument variables used to separate common shocks and endogenous price-setting effects
- Hidden generator:
  - latent demand function induced by the sampled BNN
  - GMM-derived shared shock component for market-wide demand or quantity increases
  - endogenous latent driver coupling same-product price and demand when stores react to expected demand
- Outputs:
  - continuous latent elasticity response
  - discretized labels using:
    - Class 0: `(-inf, -0.1]`
    - Class 1: `(-0.1, 0.5]`
    - Class 2: `(0.5, inf)`

### Task Diversity Requirements

- Vary number of products per synthetic market.
- Vary sparsity patterns while preserving realistic product clusters.
- Vary noise distributions and heteroskedasticity.
- Vary exogenous shared-shock mixtures so broad demand increases do not collapse into false substitute signals.
- Vary endogenous pricing regimes where merchants react to expected demand before it materializes.
- Include markets with edge-case regimes near discretization boundaries to prevent brittle classification.

### Exit Criteria

- Synthetic tasks span enough variation for the model to generalize structure, not memorize narrow regimes.
- Every generated dataset is accompanied by its authoritative elasticity mask.
- The synthetic generator can separately audit direct elasticity, exogenous shared shocks, and endogenous drivers.

## Phase 4: Single-Pass PPD Training

### Goal

Train the Transformer to approximate the PPD over unobserved labels using in-context learning on synthetic tasks, without per-dataset gradient-based retraining at inference time.

### Training Formulation

- Each training item is a task:
  - context rows with labels
  - query rows without labels
  - an associated elasticity mask
- The model processes context and query jointly in one forward pass.
- The output is a class distribution per query target approximating:
  - `p(y | x, D_context, E)`

### Recommended Loss Stack

- Primary loss:
  - multi-label cross-entropy over the three class intervals
- Secondary calibration checks:
  - expected calibration error
  - Brier score per label
- Optional auxiliary loss:
  - leakage penalty if any forbidden dependency is inferred through masked pathways
  - confounder suppression penalty if exogenous shared shocks or endogenous price-demand coupling materially increase inferred cross-product edge confidence without instrument support

### Hardware Plan

- Optimize sequence length aggressively because TabPFN-style attention scales quadratically.
- Use task batching with a cap on:
  - number of context examples
  - number of query rows
  - number of product tokens
- Start with conservative depth/width to fit stable training on a single GPU, then scale only after profiling.

### Exit Criteria

- The model produces stable class probabilities in one pass over synthetic tasks.
- Performance improves over unmasked and naive baselines on synthetic meta-test sets.
- The learned predictor distinguishes direct cross-price effects from shared exogenous and endogenous drivers in held-out synthetic interventions.

## Phase 5: Structured Sparsity Elimination via Pruning

### Goal

Convert the hard structural knowledge implied by the mask into durable model simplifications by pruning pathways that correspond to forbidden edges.

### Clarified Interpretation

This phase does not remove sparsity constraints conceptually. It removes unused computational pathways after the masking regime proves they carry no valid signal.

### Pruning Strategy

- Identify attention routes and downstream parameter blocks exclusively associated with masked zero-edge relations.
- Verify that these routes remain inactive across representative tasks.
- Prune only when:
  - the edge is structurally forbidden by `E`
  - the pathway is never needed for any valid market configuration in scope

### Guardrails

- Never prune shared components that carry legal within-product or context-level signal.
- Use ablations to compare:
  - masked-only model
  - masked + pruned model
- Treat pruning as an optimization and simplification step, not as the primary mechanism of enforcing independence.

### Open Design Questions Before Execution

These should be resolved before implementing pruning:

1. Is the mask static across all deployed markets, or must the same model support multiple market-specific masks at inference time?
2. Are products mapped to fixed token identities, or do product sets vary dynamically between tasks?
3. Should pruning target only attention score paths, or also projection matrices and feed-forward channels tied to forbidden edges?
4. What degree of parameter sharing is required across markets with different catalog sizes?
5. Is post-pruning fine-tuning allowed, or must the one-pass behavior remain frozen immediately after pruning?

### Exit Criteria

- The pruned model preserves leakage guarantees and predictive quality within tolerance.
- Model complexity is reduced without breaking the mask semantics for supported deployments.

## Phase 6: Inference and Deployment

### Goal

Use the pretrained model for small-tabular in-context inference and then generalize to larger unseen real-world market datasets.

### Inference Pattern

- Build a context set from observed market examples.
- Attach the relevant elasticity mask.
- Run one forward pass to obtain class distributions for unseen targets.
- Convert class probabilities into actionable elasticity classifications or ranked uncertainty-aware decisions.

### Generalization Strategy

- Train on many small synthetic tasks to teach transferable inference procedures.
- Apply those learned procedures to larger real-world datasets by:
  - selecting informative context subsets
  - chunking larger markets into structurally consistent subproblems
  - preserving the correct market mask for each chunk

### Deployment Concerns

- Real data must be normalized into the same numerical schema as synthetic tasks.
- Real inference payloads should carry the same instrument features used to distinguish exogenous common shocks and endogenous price-demand coupling.
- Drift monitoring should compare real inferred relationships against expected market logic.
- Drift checks should specifically flag sudden global demand lifts or coordinated price changes that could be confounders rather than true substitution signals.
- Mask generation must be auditable, since the model independence assumptions are business-critical.

### Exit Criteria

- The system can run stable one-pass inference on held-out real-market datasets.
- Validation confirms that known zero-edge relationships remain non-influential.
- Monitoring can distinguish structural violations from exogenous or endogenous confounder shifts.

## Subagent Execution Order

1. Prior Data Architect defines `E`, the BNN prior family, and synthetic task schema.
2. Architecture Engineer maps `E` into tokenization and `SparseAttention` mask semantics.
3. ML Engineer defines the single-pass training recipe using synthetic tasks and masked attention.
4. Market Logic Validator designs evaluation, leakage detection, and pruning acceptance gates.
5. Lead AI Architect reconciles interfaces, tradeoffs, and unresolved decisions before implementation begins.

## Handoff Contracts

### Prior Data Architect -> Architecture Engineer

- Canonical elasticity matrix format
- Per-task mask metadata schema
- Product identity and market-regime encoding rules

### Architecture Engineer -> ML Engineer

- Token layout
- Attention-mask API
- Sequence-length and memory constraints

### ML Engineer -> Market Logic Validator

- Output probability schema
- Label discretization logic
- Baseline and ablation definitions

### Market Logic Validator -> Lead AI Architect

- Pass/fail criteria
- Leakage thresholds
- Pruning go/no-go decision framework

## Immediate Build Priorities

1. Freeze the canonical schema for `E`, task metadata, label bins, instrument variables, and latent confounder tags.
2. Prototype synthetic task generation with exact zero-edge enforcement plus explicit exogenous and endogenous confounder simulation.
3. Implement masked attention at the attention-logit level.
4. Add synthetic-only leakage and confounder-isolation tests before any pruning work.
5. Finalize pruning only after the mask semantics and confounder controls are empirically validated.
