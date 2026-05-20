# retired MODEL1 model — Architecture Reference

> **Reproducibility-grade reference.** All content is derived directly from
> `src/model.py`, `src/train.py`, `src/hpo.py`, `src/evaluate.py`,
> `src/generate_dgp.py`, and `models/best_config.json`.
> No assumptions beyond what the source code states.

---

## Table of Contents

1. [Overview](#1-overview)
2. [ModelConfig — Architecture Hyperparameters](#2-modelconfig--architecture-hyperparameters)
3. [Model Size — Layer-by-Layer Parameter Count](#3-model-size--layer-by-layer-parameter-count)
4. [Permutation Equivariance and Invariance](#4-permutation-equivariance-and-invariance)
5. [Addressing the Fundamental Weakness of DeepSets](#5-addressing-the-fundamental-weakness-of-deepsets)
6. [Set Transformer Self-Attention Borrowings](#6-set-transformer-self-attention-borrowings)
7. [Training Data Requirements for Stability](#7-training-data-requirements-for-stability)
8. [Sparsity and Noise Elimination Strategies](#8-sparsity-and-noise-elimination-strategies)
9. [Posterior Predictive Distribution Inference](#9-posterior-predictive-distribution-inference)
10. [Full Forward Pass Mathematics](#10-full-forward-pass-mathematics)
11. [Training Configuration](#11-training-configuration)
12. [Checkpoint Format v2](#12-checkpoint-format-v2)

---

## 1. Overview

**retired MODEL1 model** is a permutation-equivariant neural network for *in-context regression*
on tabular demand data. It takes a full training context `(X_train, y_train)` and one or
more test points `x_test` as input and produces predictions in a single forward pass —
no gradient update at inference time.

### In-Context Learning Framing

| Concept | Meaning |
|---------|---------|
| Context | `(X_train, y_train)` — the training set, treated as a set of `n` labeled examples |
| Query | `x_test` — one or more test feature vectors |
| Prediction | `ŷ = f(x_test ; X_train, y_train)` — output conditioned on the full context |

The model never stores or updates parameters at test time. The context is consumed
purely through the forward pass.

### Relation to TabPFN Paradigm

Following Hollmann et al. (2022), retired MODEL1 model is **meta-trained** over a prior
distribution of synthetic datasets. By minimizing MSE across this meta-distribution,
the network implicitly approximates `E[y* | x*, D_train]` under the synthetic prior —
the mean of the Bayesian Posterior Predictive Distribution (PPD) — without any
explicit probabilistic inference at test time.

---

## 2. ModelConfig — Architecture Hyperparameters

```python
@dataclasses.dataclass
class ModelConfig:
    d_phi:       int   = 128
    d_rho:       int   = 256
    pool:        str   = "pna"
    n_heads:     int   = 4
    n_sab_feat:  int   = 1
    n_sab_samp:  int   = 1
    norm_feat:   bool  = True
    norm_target: bool  = True
    dropout:     float = 0.1
```

### Field Reference

| Field | Default | Constraint | Meaning |
|-------|---------|------------|---------|
| `d_phi` | 128 | `d_phi % n_heads == 0` when attention is used | Output dimension of the φ (phi) MLP; also the feature-level embedding width |
| `d_rho` | 256 | `d_rho % n_heads == 0` when attention is used | Output dimension of the ρ (rho) MLP; sample-level embedding width |
| `pool` | `"pna"` | One of `{"sum","mean","max","pna","learned","attn","multipool"}` | Aggregation mode used at both feature and sample pooling steps |
| `n_heads` | 4 | Must divide both `d_phi` and `d_rho` when SAB or attention pooling is used | Number of heads in all MultiheadAttention modules |
| `n_sab_feat` | 1 | ≥ 0 | Number of SAB layers applied over the feature dimension; 0 falls back to learned linear equivariance |
| `n_sab_samp` | 1 | ≥ 0 | Number of SAB layers applied over the sample dimension; 0 falls back to learned linear equivariance |
| `norm_feat` | `True` | — | If true, standardise `X_train` columns (and apply the same shift/scale to `x_test`) per-context before the forward pass |
| `norm_target` | `True` | — | If true, standardise `y_train` per-context before the forward pass and denormalize the output |
| `dropout` | 0.1 | — | Dropout probability applied in every MLP, FFN, and MHA submodule |

### POOL_SCALE — Output Multipliers

```python
POOL_SCALE = {
    "sum": 1, "mean": 1, "max": 1, "learned": 1, "attn": 1,
    "pna": 4,        # concatenates [sum, mean, max, std]
    "multipool": 5,  # concatenates pna output + attn output
}
```

A pool mode of `"pna"` with `d_phi=128` produces a `4·128 = 512`-dimensional vector
from feature pooling (the `rho_in` argument to the ρ MLP).

### HPO-Optimised Configuration

From `models/best_config.json` (Ray Tune, 20 trials, `val_mse` objective):

| Hyperparameter | HPO Value |
|----------------|-----------|
| `lr` | 0.00836010 |
| `weight_decay` | 0.00058464 |
| `d_phi` | 128 |
| `d_rho` | 256 |
| `dropout` | 0.14800 |
| `pool` | `"pna"` |

`d_phi`, `d_rho`, and `pool` were fixed during HPO (`FIXED_D_PHI=128`,
`FIXED_D_RHO=256`, `FIXED_POOL="pna"`); only `lr`, `weight_decay`, and `dropout`
were searched over:

```python
search_space = {
    "lr":           tune.loguniform(1e-4, 1e-2),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
    "dropout":      tune.uniform(0.0, 0.3),
}
```

---

## 3. Model Size — Layer-by-Layer Parameter Count

Configuration: `d_phi=128`, `d_rho=256`, `n_heads=4`, `n_sab_feat=1`, `n_sab_samp=1`,
`pool="pna"`.

### φ MLP — `build_mlp(3, 128, 128, dropout)`

```
Linear(3 → 128):     3·128 + 128  =      512
Linear(128 → 128): 128·128 + 128  =   16,512
                                  ──────────
Subtotal                               17,024
```

`build_mlp` signature: `Linear(in, hidden) → ReLU → Dropout → Linear(hidden, out)`.
Dropout and ReLU carry no parameters.

### sab\_feat — SAB × 1, d = 128, h = 4

**MHA(128, 4)**

| Tensor | Shape | Params |
|--------|-------|--------|
| `in_proj_weight` | (384, 128) | 49,152 |
| `in_proj_bias` | (384,) | 384 |
| `out_proj.weight` | (128, 128) | 16,384 |
| `out_proj.bias` | (128,) | 128 |
| **MHA subtotal** | | **66,048** |

**norm1 + norm2** (each: weight + bias, shape 128):
`2 · (128 + 128) = 512`

**FFN** `Linear(128 → 256) → ReLU → Dropout → Linear(256 → 128)`:

```
Linear(128 → 256): 128·256 + 256 =  33,024
Linear(256 → 128): 256·128 + 128 =  32,896
                                  ─────────
FFN subtotal                          65,920
```

```
sab_feat subtotal: 66,048 + 512 + 65,920 = 132,480
```

### feat\_pool (PNA) — 0 learnable parameters

PNA is a static aggregation: `[sum, mean, max, std]` concatenated over the
feature dimension. No parameters.

### ρ MLP — `build_mlp(512, 256, 256, dropout)`

Input `512 = POOL_SCALE["pna"] · d_phi = 4 · 128`.

```
Linear(512 → 256): 512·256 + 256 = 131,328
Linear(256 → 256): 256·256 + 256 =  65,792
                                  ─────────
Subtotal                             197,120
```

### sab\_samp — SAB × 1, d = 256, h = 4

**MHA(256, 4)**

| Tensor | Shape | Params |
|--------|-------|--------|
| `in_proj_weight` | (768, 256) | 196,608 |
| `in_proj_bias` | (768,) | 768 |
| `out_proj.weight` | (256, 256) | 65,536 |
| `out_proj.bias` | (256,) | 256 |
| **MHA subtotal** | | **263,168** |

**norm1 + norm2**: `2 · (256 + 256) = 1,024`

**FFN** `Linear(256 → 512) → ReLU → Dropout → Linear(512 → 256)`:

```
Linear(256 → 512): 256·512 + 512 = 131,584
Linear(512 → 256): 512·256 + 256 = 131,328
                                  ─────────
FFN subtotal                         262,912
```

```
sab_samp subtotal: 263,168 + 1,024 + 262,912 = 527,104
```

### samp\_pool (PNA) — 0 learnable parameters

Same static aggregation as feat_pool; no parameters.

### ψ MLP — `build_mlp(1024, 1, d_rho // 2, dropout)` = `build_mlp(1024, 1, 128, dropout)`

Input `1024 = POOL_SCALE["pna"] · d_rho = 4 · 256`. Hidden dim `d_rho // 2 = 128`.

```
Linear(1024 → 128): 1024·128 + 128 = 131,200
Linear(128 → 1):     128·1  +   1  =     129
                                    ────────
Subtotal                             131,329
```

### Grand Total

| Module | Parameters |
|--------|-----------|
| φ MLP | 17,024 |
| sab\_feat | 132,480 |
| feat\_pool (PNA) | 0 |
| ρ MLP | 197,120 |
| sab\_samp | 527,104 |
| samp\_pool (PNA) | 0 |
| ψ MLP | 131,329 |
| **Total** | **1,005,057** |

**~1M parameters** total under the default HPO-tuned configuration.

---

## 4. Permutation Equivariance and Invariance

### Formal Definitions

Let `π` denote a permutation operator acting on a set `S`.

- **Permutation equivariance**: `f(π·S) = π·f(S)` — reordering inputs reorders
  outputs identically. Each output element corresponds to the same input element,
  regardless of position.
- **Permutation invariance**: `f(π·S) = f(S)` — reordering inputs leaves the
  scalar (or fixed-size) output unchanged.

### Feature-Level Equivariance (over *p* features)

The model must treat the column ordering of `X` as arbitrary.

**SAB path** (`n_sab_feat ≥ 1`):

```
h ← SAB_feat(h)    where h has shape (m·n, p, d_phi)
SAB(X) = MAB(X, X)
```

Self-attention computes scores as `softmax(QKᵀ / √d_k) V`. Permuting the feature
axis permutes Q, K, V, and the output simultaneously, so the output ordering
tracks the input ordering exactly. Hence `SAB(π·X) = π·SAB(X)`.

**Linear fallback** (`n_sab_feat == 0`):

```python
mean_j = h.mean(dim=2, keepdim=True)          # (m, n, 1, d_phi)
h      = lambda_feat * h + gamma_feat * mean_j # (m, n, p, d_phi)
```

Adding a global mean (which is order-invariant) back to each element, scaled by
learned scalars `λ_feat` and `γ_feat`, preserves equivariance: permuting the
feature axis permutes both `h` and `mean_j` identically, so the output permutes
the same way.

### Sample-Level Equivariance (over *n* training samples)

The model must treat the row ordering of `(X_train, y_train)` as arbitrary.

**SAB path** (`n_sab_samp ≥ 1`):

```
r ← SAB_samp(r)    where r has shape (m, n, d_rho)
```

Same argument as feature-level: `SAB(π·R) = π·SAB(R)`.

**Linear fallback** (`n_sab_samp == 0`):

```python
mean_j = r.mean(dim=1, keepdim=True)           # (m, 1, d_rho)
r      = lambda_samp * r + gamma_samp * mean_j  # (m, n, d_rho)
```

### Pooling → Invariance

After equivariant processing, `SetPool` collapses the set dimension to a single
fixed-size vector. All seven pool modes are permutation-invariant:

| Mode | Operation | Invariant because |
|------|-----------|-------------------|
| `sum` | `Σᵢ xᵢ` | Commutative |
| `mean` | `(1/n) Σᵢ xᵢ` | Commutative |
| `max` | `maxᵢ xᵢ` | Order-agnostic |
| `pna` | `[sum, mean, max, std]` | Each component is invariant |
| `learned` | `Σᵢ softmax(score(xᵢ)) · xᵢ` | Scores and values both permute; sum is invariant |
| `attn` | `MAB(seed, X)` with a fixed learned seed | Cross-attention query is fixed; permuting K,V is invariant |
| `multipool` | `[pna, attn]` | Both components invariant |

After `SetPool`, the representation `s` is entirely order-agnostic.
The ψ MLP then maps `s` to a scalar prediction.

---

## 5. Addressing the Fundamental Weakness of DeepSets

### The Canonical DeepSet Limitation

Zaheer et al. (2017) define:

```
f(S) = ρ( Σᵢ φ(xᵢ) )
```

Each element `xᵢ` is encoded **independently** by φ before aggregation. No
element can condition on any other element during encoding. All inter-element
information must be captured by ρ acting on a single aggregate vector — a severe
information bottleneck.

Concretely, for a training context of `n` samples:
- φ cannot tell whether sample `i` is an outlier relative to the other `n−1` samples.
- φ cannot identify which training samples are closest to the query `x_test`.
- φ cannot detect that two features are correlated, because each feature is processed
  in the same independent embedding regardless of the others.

### retired MODEL1 model's Repair

retired MODEL1 model adds equivariant interaction layers at **two levels** before each
pooling step, using the SAB from the Set Transformer (Lee et al. 2019).

**Level 1 — Feature-level SAB** (before feature pooling, Step 2):

```
h ∈ ℝ^{m·n × p × d_phi}
h ← SAB_feat(h)
```

Each feature embedding `h[·,·,j,:]` can attend to all other feature embeddings
`h[·,·,j',:]` for `j' ≠ j`. After Step 2, `h[·,·,j,:]` encodes not just the
raw `(y_train[i], X_train[i,j], x_test[q,j])` tuple but also how feature `j`
relates to all other features in the same (sample, query) context. This is
critical for Regime D (correlated design matrices), where AR(1) structure creates
strong pairwise feature dependencies that a diagonal encoder cannot capture.

**Level 2 — Sample-level SAB** (before sample pooling, Step 5):

```
r ∈ ℝ^{m × n × d_rho}
r ← SAB_samp(r)
```

Each sample representation `r[q,i,:]` can attend to all other sample
representations `r[q,i',:]` for `i' ≠ i`. The model can learn to up-weight
training samples that are informative for query `q` (e.g., samples near `x_test[q]`
in feature space) and down-weight uninformative or noisy samples — dynamically,
without any explicit distance computation.

The final aggregate is therefore a **context-aware, interaction-rich summary**
rather than a linear combination of independent element embeddings.

---

## 6. Set Transformer Self-Attention Borrowings

The following modules are direct adoptions from Lee et al. (2019),
"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks."

### MAB (Multihead Attention Block)

```
H        = LayerNorm(Q + Dropout(MHA(Q, K, K)))
MAB(Q,K) = LayerNorm(H + FFN(H))

FFN(x)   = Linear(d → 2d) → ReLU → Dropout → Linear(2d → d)
```

Implementation in `model.py`:

```python
class MAB(nn.Module):
    def forward(self, Q, K):
        h, _ = self.mha(Q, K, K)           # MHA(Q, K, V=K)
        Q    = self.norm1(Q + self.drop(h))
        return self.norm2(Q + self.ffn(Q))
```

`batch_first=True` throughout. Residual connections and post-norm pattern follow
the Set Transformer paper exactly.

### SAB (Self-Attention Block)

```
SAB(X) = MAB(X, X)
```

Query and key/value are the same tensor. Permutation equivariant:
`SAB(π·X) = π·SAB(X)`. Used at both the feature level (`sab_feat`) and the
sample level (`sab_samp`).

### AttentionPool — PMA with k = 1 Seed

```
PMA_1(X) = MAB(S, X)    where S ∈ ℝ^{1 × d} is a learned seed parameter
```

Implementation:

```python
class AttentionPool(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))   # 1 learned seed

    def forward(self, x):                           # x: (batch, set, d)
        q      = self.seed.expand(x.size(0), -1, -1)  # (batch, 1, d)
        out, _ = self.mha(q, x, x)                    # (batch, 1, d)
        return self.norm(out).squeeze(1)               # (batch, d)
```

The seed is broadcast across the batch, then uses cross-attention to query the
set `x`, reducing it to a single vector. Used when `pool ∈ {"attn", "multipool"}`.

### LearnedPool — Softmax-Weighted Sum

A lighter alternative to full PMA, without the MAB overhead:

```
w = softmax( score_net(X) )    score_net: d → d//2 → 1 (Tanh activation)
pooled = Σᵢ wᵢ · xᵢ
```

Implementation:

```python
class LearnedPool(nn.Module):
    def forward(self, x):                          # x: (batch, set, d)
        w = torch.softmax(self.score(x), dim=1)    # (batch, set, 1)
        return (w * x).sum(dim=1)                  # (batch, d)
```

Used when `pool == "learned"`.

---

## 7. Training Data Requirements for Stability

### Per-Dataset Constraints (Rejection Sampling)

From `generate_dgp.py:sample_params()`:

```python
def sample_params(rng):
    while True:
        p = rng.poisson(10)
        n = rng.poisson(200)
        if p >= 1 and n >= 5 and n >= 5 * p:
            return n, p
```

| Constraint | Rationale |
|------------|-----------|
| `p ≥ 1` | At least one feature |
| `n ≥ 5` | At least 5 samples for meaningful context |
| `n ≥ 5·p` | Sample-to-feature ratio ≥ 5; ensures the linear system is substantially overdetermined and OLS has a stable solution for comparison |
| `p ~ Poisson(10)` | Expected ≈ 10 features; median ≈ 10 |
| `n ~ Poisson(200)` | Expected ≈ 200 samples |

### Meta-Dataset Split

Global random seed: `np.random.default_rng(seed=42)`.

| Split | Count | Fraction | Directory |
|-------|-------|----------|-----------|
| Train | 800 | 80% | `data/train/` |
| Validation | 100 | 10% | `data/val/` |
| Test | 100 | 10% | `data/test/` |

Within each dataset, 80% of rows form the training context and 20% form the
test set used for loss computation.

### Feature Space at Benchmark Inference

- **Hard cap**: `BENCHMARK_DEEPSET_FEATURE_CAP` defaults to `cfg.d_phi = 128`.
  Datasets with more than 128 features have features ranked and truncated before
  context construction.
- **Feature selector**: `BENCHMARK_DEEPSET_FEATURE_SELECTOR = "train_f_regression"` —
  scikit-learn's `f_regression` F-statistic ranking applied to the training fold.

### Regime Coverage

Regimes are sampled uniformly at generation time (`regimes[rng.integers(0, 4)]`).

| Regime | X distribution | β distribution | ε distribution | Primary stress |
|--------|---------------|----------------|----------------|----------------|
| A | N(0, I) | N(0, 1) — all nonzero | N(0, 1) | Baseline linear regression |
| B | N(0, I) | N(0, 4), 70% zeroed | N(0, 1) | Sparse coefficients |
| C | N(0, I) | N(0, 1) — all nonzero | t(df=3) | Heavy-tailed noise |
| D | AR(1, ρ=0.6) | N(0, 1) — all nonzero | N(0, 1) | Correlated design matrix |

**Regime D AR(1) construction** (`generate_dgp.py`):

```python
X[:, 0] = rng.standard_normal(n)
for k in range(1, p):
    X[:, k] = 0.6 * X[:, k-1] + sqrt(0.64) * rng.standard_normal(n)
```

Marginal variance: `Var(X[:,k]) = 0.36·Var(X[:,k-1]) + 0.64 = 1` (stationary).
Adjacent-column correlation: `Corr(X[:,k], X[:,k-1]) = 0.6`.

**Training target**: the *noiseless* linear part `βᵀx` (`betaX_test`), not the
noisy observation `y`. The model learns to denoise by predicting the signal.

---

## 8. Sparsity and Noise Elimination Strategies

### Regime B — Sparse Coefficients (70% Zero)

**Mechanism 1 — Per-context feature normalization:**
All columns of `X_train` are standardized to zero mean and unit variance regardless
of whether `β_j = 0`. A zero-coefficient feature still has nonzero column variance
driven by random noise in `X`; the normalization does not reveal the sparsity, but
it prevents scale differences from dominating the embedding.

**Mechanism 2 — Distinguishable phi patterns:**
The input tuple `(y_train[i], X_train[i,j], x_test[q,j])` for a zero-coefficient
feature `j` shows near-zero covariance between `y_train[i]` and `X_train[i,j]`
across all `n` samples. The φ network sees systematically flat `(y, x_j, x_test_j)`
triples that are distinguishable from nonzero-β features after training on sufficient
Regime B examples.

**Mechanism 3 — Feature-level SAB attention routing:**
With `n_sab_feat=1`, each feature embedding can attend to every other feature. After
training on Regime B data, attention heads can learn to suppress zero-β features by
routing their weight to correlated, nonzero-β features. Concretely, the attention
scores for feature `j` when `β_j ≈ 0` converge toward zero because downstream loss
penalises any prediction variance attributable to inactive features.

**Mechanism 4 — Optional L1 penalty** (`LAMBDA_L1 > 0`, default 0):

```python
if l1_lambda > 0.0 and training:
    loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())
```

Penalises large weight magnitudes globally at training time, encouraging the network
to use a sparse subset of its capacity for low-information inputs.

### Regime C — Heavy-Tailed Noise (t, df=3)

**Mechanism 1 — Per-context target normalization:**
`y_train` is divided by its empirical standard deviation per context. Even when a
single t(df=3) sample is very large, the normalization bounds the effective
prediction scale, preventing gradient explosions from outlier targets.

**Mechanism 2 — Optional Huber loss** (`USE_HUBER=True`, default `False`; `HUBER_DELTA=1.0`):

```
L_Huber(r) = r²/2                for |r| ≤ δ
           = δ(|r| − δ/2)        for |r| > δ
```

Gradient contribution from residuals larger than `δ=1.0` is linear rather than
quadratic, reducing the influence of outlier training points on parameter updates.

**Mechanism 3 — MC dropout averaging:**
With `N_MC_DROPOUT=32` passes at inference (dropout active), the mean of 32 draws
is more robust to heavy-tailed single predictions than any individual pass. The
averaging smooths out variance induced by individual dropout masks.

### Regime D — Correlated Features (AR(1), ρ = 0.6)

**Feature-level SAB with h=4 heads** explicitly models pairwise attention weights
between all feature embeddings. The AR(1) structure produces a tridiagonal covariance
structure; SAB attention heads can specialize to adjacent-feature groupings, encoding
the local correlation structure into each feature's embedding before pooling.

---

## 9. Posterior Predictive Distribution Inference

### Bayesian Framing

The ideal prediction integrates over all parameter values consistent with the
training data:

```
p(y* | x*, D_train) = ∫ p(y* | x*, θ) p(θ | D_train) dθ
```

Computing this integral exactly requires specifying a model family and is
generally intractable. retired MODEL1 model approximates it in two ways.

### Meta-Training as Prior Specification

The model is trained on 800 synthetic datasets drawn from a known prior over
`(X, β, ε)` distributions (4 regimes, uniformly sampled). By minimizing

```
L = E_{(X,β,ε) ~ prior} [ E_{x*,y*} [ (f(x*; D_train) − β·x*)² ] ]
```

the network learns the mapping `(x*, D_train) ↦ E[β·x* | x*, D_train]`
under that prior — i.e., the PPD mean — without any explicit probabilistic
inference at test time. This is the TabPFN paradigm (Hollmann et al. 2022).

### MC Dropout Approximation

Dropout as a Bayesian approximation (Gal & Ghahramani 2016): keeping dropout
active at inference and sampling `K` masks approximates drawing `K` samples from
a variational posterior over the network weights.

```
p(y* | x*, D_train) ≈ (1/K) Σᵢ₌₁ᴷ δ(y* − f_{θᵢ}(x*, D_train))
```

where `θᵢ` is the effective parameter vector under dropout mask `i`.

| Quantity | Formula |
|----------|---------|
| Point estimate | `ŷ_MC = (1/K) Σᵢ f_{θᵢ}(x*, D_train)` |
| Epistemic uncertainty | `σ²_MC = (1/(K−1)) Σᵢ (f_{θᵢ} − ŷ_MC)²` |

**Operational setting** (`evaluate.py`): `N_MC_DROPOUT = K = 32`. The model is kept
in `.train()` mode during inference so that `nn.Dropout` remains active.

### Context Ensemble

For datasets with `n > BENCHMARK_DEEPSET_CONTEXT_SIZE = 200` training rows, five
non-overlapping context windows of 200 rows each are drawn:

```
ŷ_final = (1/5) Σⱼ₌₁⁵ ŷ_context_j
```

This marginalizes over uncertainty in which 200 training samples are most
informative for the query — a data-subsampling approximation to the full posterior
over training sets.

Constants from `evaluate.py`:

```python
N_MC_DROPOUT                        = 32
BENCHMARK_DEEPSET_CONTEXT_SIZE      = 200   # env override available
BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES = 5     # env override available
```

### Combined PPD

```
Total forward passes = CONTEXT_ENSEMBLES × N_MC_DROPOUT = 5 × 32 = 160
```

The ensemble mean is the PPD point estimate. Variance decomposes into:
- **Epistemic** (within-context): variance over the 32 MC dropout masks
- **Data** (across-context): variance over the 5 context windows

---

## 10. Full Forward Pass Mathematics

Dimensions: `m` test points, `n` training samples, `p` features.

### Normalization (pre-Step 1)

**Feature normalization** (`cfg.norm_feat == True`):

```
μ_f = (1/n) Σᵢ X_train[i,:]         ∈ ℝᵖ
σ_f = std(X_train, dim=0).clamp(1e-8) ∈ ℝᵖ

X_train ← (X_train − μ_f) / σ_f      (n, p)
x_test  ← (x_test  − μ_f) / σ_f      (m, p)
```

The same shift and scale are applied to the test points, preventing distribution
shift between context and query.

**Target normalization** (`cfg.norm_target == True`):

```
μ_y = mean(y_train)
σ_y = std(y_train, unbiased=False).clamp(1e-8)
y_train ← (y_train − μ_y) / σ_y      (n,)
```

### Step 1 — Input Construction and φ

Each test point `q`, training sample `i`, and feature `j` form a 3-tuple:

```
I[q, i, j, :] = [ y_train[i],  X_train[i,j],  x_test[q,j] ]  ∈ ℝ³
h = φ(I)   ∈ ℝ^{m × n × p × d_phi}
```

The φ MLP is applied elementwise over the `(m, n, p)` batch dimensions. Every
test-query–training-sample–feature triple is independently embedded.

Implementation:

```python
y_e  = y_train.view(1, n, 1).expand(m, n, p)
X_e  = X_train.view(1, n, p).expand(m, n, p)
xt_e = x_test.view(m, 1, p).expand(m, n, p)
inp  = torch.stack([y_e, X_e, xt_e], dim=3)   # (m, n, p, 3)
h    = self.phi(inp)                            # (m, n, p, d_phi)
```

### Step 2 — Feature Equivariance (SAB over *j*)

```
h_flat = h.reshape(m·n, p, d_phi)
h_flat = SAB_feat(h_flat)              # (m·n, p, d_phi)
h      = h_flat.reshape(m, n, p, d_phi)
```

Each feature embedding attends to all other feature embeddings within the same
(test-query, training-sample) pair. This makes the feature dimension interact
before aggregation.

*Linear fallback* (`n_sab_feat == 0`):

```python
mean_j = h.mean(dim=2, keepdim=True)
h      = lambda_feat * h + gamma_feat * mean_j
```

### Step 3 — Feature Pooling (PNA over *j*)

```
h_flat = h.reshape(m·n, p, d_phi)
z      = SetPool_feat(h_flat)           # (m·n, 4·d_phi)  for pna
z      = z.reshape(m, n, 4·d_phi)
```

Under PNA:

```
z[q,i] = [ Σⱼ h[q,i,j,:],   (1/p)Σⱼ h[q,i,j,:],
           maxⱼ h[q,i,j,:],  std_j h[q,i,j,:] ]  ∈ ℝ^{4·d_phi}
```

### Step 4 — ρ per Sample

```
r = ρ(z)   ∈ ℝ^{m × n × d_rho}
```

The ρ MLP maps each sample's feature-pooled embedding to the sample-level
representation space.

### Step 5 — Sample Equivariance (SAB over *i*)

```
r = SAB_samp(r)   ∈ ℝ^{m × n × d_rho}
```

Each sample representation attends to all other sample representations for the
same test query `q`. The model can learn to up-weight informative training samples
dynamically.

*Linear fallback* (`n_sab_samp == 0`):

```python
mean_i = r.mean(dim=1, keepdim=True)
r      = lambda_samp * r + gamma_samp * mean_i
```

### Step 6 — Sample Pooling (PNA over *i*)

```
s = SetPool_samp(r)   ∈ ℝ^{m × 4·d_rho}   for pna
```

Under PNA:

```
s[q] = [ Σᵢ r[q,i,:],   (1/n)Σᵢ r[q,i,:],
         maxᵢ r[q,i,:],  std_i r[q,i,:] ]  ∈ ℝ^{4·d_rho}
```

`s` is permutation-invariant in the sample dimension: the training context has
been fully aggregated into a fixed-size vector for each test query.

### Step 7 — ψ (Prediction Head)

```
ŷ_raw = ψ(s).squeeze(-1)   ∈ ℝ^m
```

The ψ MLP maps the context summary to a scalar prediction per test point.

### Step 8 — Denormalization

```
ŷ = ŷ_raw · σ_y + μ_y    (if cfg.norm_target)
```

The prediction is rescaled back to the original target space using the
per-context statistics computed before Step 1.

### Summary: Tensor Flow

```
Input:   X_train (n,p), y_train (n,), x_test (m,p)
            ↓  normalize
Step 1:  h  (m, n, p, 3) → φ  → (m, n, p, d_phi)
Step 2:       (m·n, p, d_phi) → SAB_feat → (m, n, p, d_phi)
Step 3:       (m·n, p, d_phi) → PNA_feat → (m, n, 4·d_phi)
Step 4:       (m, n, 4·d_phi) → ρ        → (m, n, d_rho)
Step 5:       (m, n, d_rho)   → SAB_samp → (m, n, d_rho)
Step 6:       (m, n, d_rho)   → PNA_samp → (m, 4·d_rho)
Step 7:       (m, 4·d_rho)    → ψ        → (m,)
            ↓  denormalize
Output:  ŷ  (m,)                         scalar if m=1
```

---

## 11. Training Configuration

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam | `train.py` |
| Learning rate | 0.00836 | `models/best_config.json` (HPO) |
| Weight decay | 0.000585 | `models/best_config.json` (HPO) |
| Dropout | 0.148 | `models/best_config.json` (HPO) |
| Max epochs | 200 | `MAX_EPOCHS = 200` in `train.py` |
| Early stopping patience | 10 epochs | `PATIENCE = 10` in `train.py` |
| Early stopping monitor | Validation MSE | |
| Batch size | 1 meta-dataset | One Parquet file per step |
| AMP dtype | bfloat16 | `torch.bfloat16`, GPU only |
| Primary loss | MSE on `betaX_test` | Noiseless linear target |
| Huber loss | Optional (`USE_HUBER`, default `False`) | δ = 1.0 |
| L1 penalty | Optional (`LAMBDA_L1`, default 0.0) | |

### Distributed Training Topology

```
DDP world size: 10 nodes × 4 A10G GPUs = 40 workers total
Backend: NCCL
DataLoader workers: 4 per GPU, prefetch_factor=2, pin_memory=True
Data sharding: rank-modulo sharding (no sampler padding)
Validation all-reduce: weighted sum-count reduction over all ranks
```

Checkpoint is saved only on `rank == 0` when validation MSE improves. Early
stopping signal is broadcast from rank 0 to all ranks via `dist.broadcast`.

### Compilation

```python
model = torch.compile(model, mode="reduce-overhead")
model = DistributedDataParallel(model, device_ids=[local_rank])
```

`torch.compile` is applied before DDP wrapping. The compiled module is unwrapped
at checkpoint save time (`ckpt._orig_mod`).

---

## 12. Checkpoint Format v2

```python
{
    "checkpoint_format_version": 2,
    "cfg": dataclasses.asdict(model.cfg),   # all 9 ModelConfig fields as plain dict
    "state_dict": model.state_dict(),
    "metadata": {
        "source": "train.py",
        "checkpoint_name": str,             # basename of output path
        "pytorch_version": str,             # torch.__version__
    }
}
```

### Loading Strategy (`evaluate.py:load_checkpoint_compat()`)

Three-tier fallback in order of preference:

1. `torch.load(..., weights_only=True)` — preferred for v2; safest
2. `safe_globals([ModelConfig]) + weights_only=True` — for legacy checkpoints
   where `ModelConfig` was pickled directly rather than as a plain dict
3. `weights_only=False` — only when environment variable
   `ALLOW_UNSAFE_TORCH_LOAD=true` is set explicitly

### Architecture Mismatch Detection

At warm-start (pretrain → fine-tune), the checkpoint loader compares 8 of the
9 `ModelConfig` fields between the saved and current configs:

```python
fields = ("d_phi", "d_rho", "pool", "n_heads",
          "n_sab_feat", "n_sab_samp", "norm_feat", "norm_target")
```

A mismatch on any structural field (`d_phi`, `d_rho`, `pool`, `n_heads`,
`n_sab_feat`, `n_sab_samp`) causes the checkpoint to be silently skipped and
training to restart from random initialisation. Mismatches on `norm_feat` or
`norm_target` alone are also skipped since they affect the forward pass contract.

---

## References

- Zaheer, M. et al. (2017). *Deep Sets*. NeurIPS 2017.
- Lee, J. et al. (2019). *Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks*. ICML 2019.
- Hollmann, N. et al. (2022). *TabPFN: A Transformer That Solves Small
  Tabular Classification Problems in a Second*. ICLR 2023.
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning*. ICML 2016.
