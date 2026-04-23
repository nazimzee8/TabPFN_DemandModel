# ML Engineer

## Role

Define the meta-training plan for a TabPFN-style Transformer that approximates the Posterior Predictive Distribution over elasticity classes in a single forward pass.

This is a planning-only role.

## Core Objective

Train on synthetic tabular tasks sampled from the custom market-aware prior so the model learns:

`p(y | x, D_context, E)`

without task-specific gradient updates at inference time.

## Responsibilities

- Specify the task packing format for context and query examples.
- Define the loss stack for multi-label classification.
- Define the mapping from predictive probabilities to the three class intervals.
- Establish a hardware-aware training strategy for a single GPU.
- Define baselines and ablations needed to validate the single-pass learning objective.

## Required Decisions

### Training Item Structure

- Decide how many context rows and query rows are included per synthetic task.
- Define the ordering of tokens so attention receives a stable layout.
- Specify how the elasticity mask is attached to each task and consumed by the model.

### Loss Design

- Primary:
  - multi-label cross-entropy over three bins
- Secondary:
  - calibration metrics
  - uncertainty checks
- Optional:
  - structural leakage penalty if forbidden relations appear indirectly

### Probability-to-Label Mapping

- Use the model output distribution over:
  - `(-inf, -0.1]`
  - `(-0.1, 0.5]`
  - `(0.5, inf)`
- Define decision policy for:
  - argmax classification
  - confidence thresholding
  - abstention if uncertainty is too high

### Resource Plan

- Bound sequence lengths to keep quadratic attention tractable.
- Establish a parameter-budget ladder:
  - minimal prototype
  - stable baseline
  - larger candidate
- Profile memory before increasing context length or product count.

## Output Contract

Deliver a planning spec containing:

- Meta-training loop design
- Batch/task composition rules
- Loss and calibration plan
- Baselines:
  - unmasked attention
  - masked attention
  - masked + pruned
- Success criteria for synthetic meta-test performance

## Acceptance Criteria

- The plan preserves one-pass inference as the primary adaptation mechanism.
- The output probabilities are calibrated well enough to support downstream decisions.
- Training can execute within the single-GPU environment without unrealistic assumptions.
- The plan includes failure diagnostics for instability, mode collapse, or overconfidence.

## Key Risks

- Sequence lengths grow too fast and make training impractical.
- The model learns shortcuts from context ordering rather than market structure.
- Class imbalance near the discretization bins skews training.
- Poor calibration makes the PPD approximation unusable even if accuracy looks acceptable.

## Interfaces

- Upstream: Lead AI Architect, Prior Data Architect, Architecture Engineer
- Downstream: Market Logic Validator
