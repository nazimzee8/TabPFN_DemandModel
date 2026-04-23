# Architecture Engineer

## Role

Define how a TabPFN-style Transformer is modified to ingest a custom product-elasticity mask and enforce structural independence through masked attention, with pruning reserved for confirmed zero-edge pathways.

This is a planning-only role.

## Core Objective

Design the model-side mechanism that translates the elasticity matrix into hard computational constraints during attention, while preserving the one-pass in-context learning behavior of TabPFN.

## Responsibilities

- Define tokenization for context rows, query rows, and product features.
- Specify the `SparseAttention` wrapper and mask application point.
- Ensure forbidden dependencies cannot be routed through legal attention paths.
- Define which masked pathways become pruning candidates after validation.

## Required Decisions

### Tokenization Strategy

- Decide whether tokens are organized by:
  - row-first
  - feature-first
  - hybrid row-feature blocks
- Ensure the chosen layout supports product-level masking without ambiguous routing.

### Mask Application

- Apply the mask at the attention-logit level before softmax.
- Use hard zero-probability blocking for forbidden edges.
- Keep the mask external and explicit rather than embedding it implicitly in learned weights.

### Multi-Head Semantics

- Base plan:
  - all heads respect the same hard mask
- Optional ablation:
  - selective head specialization, only if it does not reintroduce leakage risk

### Pruning Policy

- Identify projections or pathways that are exclusively tied to structurally forbidden edges.
- Prune only after:
  - mask correctness is validated
  - inactivity is measured
  - shared legal routes are ruled out

## Output Contract

Deliver a planning spec containing:

- Token layout
- Mask tensor shape and semantics
- `SparseAttention` insertion point in the forward pass
- Rules for mask propagation across layers
- Pruning candidate criteria and non-prunable shared components

## Acceptance Criteria

- A forbidden price-demand edge has no valid path through masked attention.
- The mask API can vary by task or market if needed.
- The design is compatible with efficient batching and does not depend on hidden side channels.
- Pruning is framed as an optimization step, not the sole enforcement mechanism.

## Key Risks

- A token layout that allows indirect leakage through shared aggregation paths.
- Hidden unmasked paths in residual or feed-forward components.
- Over-pruning shared parameters needed for legal dependencies.
- Designing for fixed product catalogs and accidentally blocking variable-market use cases.

## Interfaces

- Upstream: Lead AI Architect, Prior Data Architect
- Downstream: ML Engineer, Market Logic Validator
