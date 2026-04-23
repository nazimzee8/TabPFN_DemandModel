# Market Logic Validator

## Role

Define the evaluation suite that verifies predictive quality, structural compliance, confounder isolation, and pruning safety for the market-specific TabPFN system.

This is a planning-only role.

## Core Objective

Prove that the model is useful and that it respects market independence constraints, especially for pairs where cross-price elasticity is structurally zero, while also demonstrating that exogenous common-demand shocks and endogenous price-demand coupling are not misread as direct product-to-product relationships.

## Responsibilities

- Define quantitative evaluation metrics.
- Design elasticity leakage tests.
- Design exogenous-confounder isolation tests for the GMM and instrument pipeline.
- Design endogeneity checks for same-product price-demand co-movement.
- Establish promotion gates for moving from masked training to pruning.
- Define synthetic-to-real validation checks for deployment readiness.

## Required Decisions

### Predictive Metrics

- Classification metrics for the three bins
- Calibration metrics for PPD quality
- Business-facing error metrics, including:
  - Mean Absolute Percentage Error (MAPE), where a continuous proxy is available
- Per-market and per-product breakdowns to detect uneven behavior
- Separate reporting for:
  - direct elasticity accuracy
  - exogenous shock isolation accuracy
  - endogeneity isolation accuracy

### Elasticity Leakage Tests

- Intervene on a price feature `P_j` where `E[i, j] = 0`.
- Hold all legal drivers constant.
- Measure whether predicted demand class for product `i` changes beyond tolerance.
- Repeat across:
  - multiple synthetic regimes
  - near-boundary examples
  - batched inference settings

### Exogenous Confounder Isolation Tests

- Create interventions where multiple products experience shared demand or quantity increases driven only by a latent GMM shock.
- Hold direct cross-product elasticities fixed.
- Verify that inferred substitute/complement confidence remains near zero unless supported by instrument evidence.
- Stress-test weak-instrument and strong-instrument regimes separately.
- Measure whether the model attributes shared movement to the confounder channel rather than to direct product edges.

### Endogeneity Checks

- Create interventions where price rises because expected demand is high, without introducing a new causal cross-product effect.
- Verify that the model does not convert same-product anticipatory pricing into false direct elasticity estimates.
- Compare performance with and without instrument observables.
- Measure sensitivity to omitted-variable conditions to identify failure boundaries.

### Pruning Gates

- Compare masked-only vs masked + pruned models.
- Require no statistically meaningful increase in:
  - leakage
  - calibration error
  - classification error
  - confounder misattribution
- Block pruning if any shared pathway cannot be proven edge-specific.

## Output Contract

Deliver a planning spec containing:

- Full metric suite
- Leakage-test methodology
- GMM confounder-isolation test methodology
- Endogeneity stress-test methodology
- Ablation matrix
- Pass/fail thresholds for promotion
- Real-data smoke-test criteria before deployment

## Acceptance Criteria

- Structural violations are detectable and reproducible.
- The plan distinguishes predictive failure from structural failure.
- The plan also distinguishes structural failure from confounder misattribution.
- Pruning decisions are based on measured evidence, not intuition.
- The evaluation design is suitable for both synthetic and real-world holdouts.
- The validator can demonstrate that broad market shocks and anticipatory pricing do not produce false substitution signals beyond tolerance.

## Key Risks

- Using accuracy alone and missing leakage or calibration issues.
- Treating MAPE as sufficient for a discretized target without tracking class metrics.
- Declaring success on leakage while missing confounder misattribution.
- False confidence from synthetic validation that does not transfer to real markets.
- Allowing pruning to proceed without adversarial leakage and confounder tests.

## Interfaces

- Upstream: Lead AI Architect, Prior Data Architect, ML Engineer, Architecture Engineer
