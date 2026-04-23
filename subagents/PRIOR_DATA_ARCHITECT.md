# Prior Data Architect

## Role

Define the synthetic market-data prior used to pretrain the TabPFN-style model. This role encodes market structure into generated tasks so that the learned Transformer approximates the correct Posterior Predictive Distribution under a market-aware BNN prior.

This is a planning-only role.

## Core Objective

Produce a task-generation specification where hard-zero cross-price elasticities are guaranteed by construction, while exogenous common-demand shocks and endogenous price-demand coupling are represented as separate latent causes rather than mistaken for product substitution or complementarity.

- If `E[i, j] = 0`, product `j` price has no causal or statistical effect on product `i` demand in the generator.
- If `E[i, j] != 0`, the generator may express substitute, complement, or weakly related behavior depending on the sampled market regime.
- If multiple products move together because of a shared market shock, that effect must flow through a dedicated latent confounder channel rather than through direct cross-product edges.
- If price and demand move together for the same good because of anticipatory pricing, that effect must be generated through an endogenous driver rather than through a false elasticity edge.

## Responsibilities

- Define the canonical elasticity matrix `E`.
- Specify product graph semantics and directionality.
- Design the synthetic task family used for meta-training, validation, and meta-test.
- Define the custom BNN prior over latent demand functions.
- Define the Gaussian Mixture Model layer used to isolate exogenous common-demand shocks with multiple instruments.
- Define the endogenous latent-variable process that jointly shifts same-product price and demand.
- Specify label generation and discretization into:
  - `(-inf, -0.1]`
  - `(-0.1, 0.5]`
  - `(0.5, inf)`

## Required Decisions

### Elasticity Matrix Schema

- Choose the authoritative format:
  - dense matrix
  - sparse adjacency list
  - grouped block mask
- Define whether self-elasticity and cross-elasticity are stored together or separately.
- Define how market-specific overrides are layered on top of the base structure.
- Define how structural edges are separated from confounder channels in the metadata schema.

### Market Regime Sampling

- Specify how tasks vary across:
  - number of products
  - price scales
  - competitive intensity
  - noise and volatility
  - segment-specific elasticity ranges
- Ensure structural zeros remain fixed across all sampled regimes.
- Vary exogenous shared-shock regimes so broad demand lifts can affect many products without creating direct substitute edges.
- Vary endogenous pricing regimes where merchants react to expected demand before it is observed.

### GMM and Instrument Design

- Specify the Gaussian Mixture Model used for exogenous shared-demand or quantity shocks.
- Define the mixture components, their priors, and how they map to market regimes.
- Define the instrument variables used to identify these shared shocks.
- Specify how instrument strength is varied so the downstream model sees both easy and difficult identification cases.
- Require that these shared-shock components contribute near-zero weight to inferred cross-product edge confidence unless instruments support a direct relationship.

### Endogeneity Process

- Parameterize latent variables that jointly influence price and demand for the same good.
- Model cases such as anticipatory price increases before high demand.
- Define how endogenous factors differ from true self-elasticity and from cross-product elasticity.
- Specify what observables or instruments are available to help identify the endogenous process.

### BNN Prior

- Parameterize a BNN whose connectivity or effective coefficients are conditioned on `E`.
- Define priors over:
  - weights
  - bias terms
  - observation noise
  - latent market factors
  - GMM shared-shock components
  - endogenous same-product confounders
- Decide how much nonlinearity to include without making the induced task family too unrealistic or too broad.

## Output Contract

Deliver a planning spec containing:

- Formal definition of `E`
- Market sampling process
- GMM shared-shock specification and instrument schema
- Endogeneity-generation specification
- BNN prior parameter sampling process
- Synthetic dataset schema:
  - `X_context`
  - `y_context`
  - `X_query`
  - `y_query`
  - `mask`
  - `instruments`
  - `latent_confounder_tags`
- Acceptance rules proving zero-edge enforcement and confounder separation

## Acceptance Criteria

- A downstream engineer can generate tasks without making assumptions.
- Two independent implementations would produce equivalent zero-edge semantics.
- Generated tasks include edge cases near the class thresholds.
- The plan explicitly distinguishes structural zeros from small noisy coefficients.
- The plan separately audits direct elasticity, exogenous shared shocks, and endogenous drivers.
- Instrument-linked GMM factors cannot be confused with direct cross-product edges in the task definition.

## Key Risks

- Confusing near-zero with exactly zero and leaking false dependencies into training.
- Collapsing shared market shocks into substitute signals because the GMM layer is under-specified.
- Encoding weak instruments that fail to identify exogenous or endogenous confounders.
- Overly narrow synthetic priors that fail to cover real market regimes.
- Overly broad priors that destroy learnable structure and weaken the TabPFN effect.

## Interfaces

- Upstream: Lead AI Architect
- Downstream: Architecture Engineer, ML Engineer, Market Logic Validator
