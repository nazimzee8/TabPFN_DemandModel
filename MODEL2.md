# retired MODEL2 model — Reference

This document is the self-contained reference for `retired MODEL2 model`.
`MODEL.md` describes the legacy `retired MODEL1 model` and must not be modified.

---

## §1 Motivation

### The query-collapse problem in retired MODEL1 model

`retired MODEL1 model` builds tokens `(y_i, X_ij, xq_j)` per `(sample i, feature j)` pair, applies
`phi` to each token, then **pools over features first** (per sample), and only afterward
aggregates over samples. This means that feature identity is destroyed before the model has
seen evidence across training samples.

The consequence: the model cannot compute anything analogous to the sufficient statistics of
linear regression:

```
Σ_i X_ij · y_i    (empirical feature j contribution to labels)
Σ_i X_ij · X_ik   (feature covariance structure)
```

Because `x_test[j]` enters the token but is then immediately pooled away over features,
all query-differentiating information is destroyed before sample-level aggregation.
Different `x_test` inputs produce nearly identical predictions — **query collapse**.

### The evidence-before-pooling principle

The fix reverses pooling order:

1. **Aggregate samples** within each `(query q, feature j)` slot first.
2. **Then** let features interact.

This preserves feature identity through the critical evidence-aggregation step, allowing
the model to compute per-feature sufficient statistics before cross-feature interactions.

---

## §2 Architecture Tensor Flow

```
Inputs:  X_train (n, p),  y_train (n,),  x_test (p,) or (m, p)
single = x_test.ndim == 1  →  unsqueeze to (1, p); squeeze output at the end

─── Step 1: Normalize ───────────────────────────────────────────────────────
X_norm:       (n, p)   — column-standardized using X_train statistics
x_test_norm:  (m, p)   — same statistics applied to test
y_norm:       (n,)     — standardized; y_mean, y_std stored for denormalization

─── Step 2: Build 6-feature tokens ──────────────────────────────────────────
y_e    = y_norm.view(1,1,n).expand(m,p,n)             # (m, p, n)
Xi_e   = X_norm.T.unsqueeze(0).expand(m,p,n)          # (m, p, n)
xq_e   = x_test_norm.unsqueeze(2).expand(m,p,n)       # (m, p, n)
tokens = stack([y_e, Xi_e, xq_e,
                Xi_e*xq_e, Xi_e*y_e,
                |Xi_e-xq_e|], dim=3)                  # (m, p, n, 6)
flat   = tokens.view(m*p, n, 6)                        # (m*p, n, 6)

─── Step 3: Sample evidence per feature ─────────────────────────────────────
h      = phi_sample(flat)                              # (m*p, n, d_sample)
[optional: h = sab_sample(h)  if n_sab_sample_per_feature > 0]
ev     = sample_pool_layer(h)                          # (m*p, d_feat)
ev     = ev.view(m, p, d_feat)                         # (m, p, d_feat)
                                                       # ↑ KEY: feature identity preserved

─── Step 4: Cross-feature interaction ───────────────────────────────────────
if n_sab_feat > 0:   ctx = sab_feat(ev)               # (m, p, d_feat)
else:                ctx = λ*ev + γ*ev.mean(1,True)   # (m, p, d_feat)

─── Step 5: Neural prediction ───────────────────────────────────────────────
beta_like = beta_head(ctx).squeeze(-1)                 # (m, p)
lin_pred  = (beta_like * x_test_norm).sum(dim=1)       # (m,)
summary   = feat_summary_pool(ctx)                     # (m, 4*d_feat)   [pna]
resid     = residual_head(summary).squeeze(-1)         # (m,)
neural    = lin_pred + residual_scale * resid           # (m,)

─── Step 6: Ridge expert (optional) ─────────────────────────────────────────
if use_ridge_expert:
    ridge = RidgeExpert.predict(X_norm, y_norm, x_test_norm, ridge_lambda)  # (m,)

─── Step 7: Gate + combine ──────────────────────────────────────────────────
ctx_mean  = ctx.mean(dim=1)                            # (m, d_feat)
gate      = sigmoid(gate_head(ctx_mean).squeeze(-1))   # (m,)  ∈ (0,1)
if use_ridge_expert:
    pred_norm = ridge + gate * neural
else:
    pred_norm = neural

─── Step 8: Denormalize ─────────────────────────────────────────────────────
y_hat = pred_norm * y_std + y_mean   (if norm_target)
return y_hat.squeeze(0) if single else y_hat           # scalar or (m,)
```

---

## §3 Token Construction

Each training sample `i`, feature `j`, and query `q` produces a 6-dimensional token:

| Slot | Expression | Intuition |
|------|------------|-----------|
| 0 | `y_i` | Label signal |
| 1 | `X_ij` | Raw feature value for sample i, feature j |
| 2 | `xq_j` | Query's value for feature j (broadcast across samples) |
| 3 | `X_ij · xq_j` | Feature similarity: how much does sample i's feature j align with the query? |
| 4 | `X_ij · y_i` | Empirical feature contribution: feature j × label for sample i |
| 5 | `|X_ij − xq_j|` | Feature-space distance: how far is sample i from the query on feature j? |

Tokens 3–5 encode query-relative statistics that survive the per-sample pool at step 3,
preventing query collapse.

---

## §4 RidgeExpert

`RidgeExpert` is a **stateless, parameter-free** class (not `nn.Module`).
It provides closed-form ridge regression as an explicit inductive bias.

**Primal form** (`n ≥ p`):

```
β = (XᵀX + λI_p)⁻¹ Xᵀy
ŷ = x_test @ β
```

**Dual form** (`n < p`):

```
α = (XXᵀ + λI_n)⁻¹ y
β = Xᵀα
ŷ = x_test @ β
```

Both forms are mathematically equivalent (Woodbury identity) and are numerically solved
via `torch.linalg.solve`. The primal/dual switch is automatic based on the shape of
`X_norm`.

**Constraint:** `ridge_lambda > 0` is required for positive definiteness.

**Gate mechanism:** The ridge prediction is a stable baseline. The neural residual is
added through a learned gate: `pred = ridge + gate * neural`. The gate is initialized
near 0.5 (via a fresh network) and learned to weight the neural correction.

---

## §5 Memory Guard

**`n_sab_sample_per_feature = 0` is the default and must not be changed in production
without chunking.**

When `n_sab_sample_per_feature > 0`, the SAB operates on a `(m*p, n, n)` attention
matrix:

```
m=128, p=128, n=200 → attention matrix shape: (16384, 200, 200)
                     = 655M float32 elements = 2.6 GB
```

This will OOM on A10G GPUs (24 GB VRAM) when combined with other model activations.

**Safe alternative:** `sample_pool="attn"` uses single-seed cross-attention (O(n) memory)
and already encodes `xq_j` in the token, producing query-differentiated evidence without
O(n²) cost.

Do not set `n_sab_sample_per_feature > 0` until chunking over the `m*p` batch dimension
is implemented.

**Validation:** `ModelConfig.__post_init__` raises `ValueError` if
`n_sab_sample_per_feature > 0` and `d_sample % n_heads != 0`.

---

## §6 Checkpoint v3 Format

`retired MODEL2 model` checkpoints use `checkpoint_format_version=3`.
`retired MODEL1 model` checkpoints keep version 2.

**Full checkpoint dict schema (v3):**

```python
{
    "checkpoint_format_version": 3,
    "cfg": dataclasses.asdict(model.cfg),   # plain dict, NOT ModelConfig object
    "state_dict": model.state_dict(),
    "metadata": {
        "source": "train.py",
        "checkpoint_name": "best.pt",
        "pytorch_version": torch.__version__,
        "model_family": "market_aware",
    },
}
```

**Never save a `market_aware` model with version 2.** `_instantiate_model(cfg)` in
`evaluate.py` reads `cfg.model_family` and routes accordingly; if the version were 2
but `cfg` contains `model_family="market_aware"`, the routing still works correctly,
but the version mismatch would be misleading for debugging.

---

## §7 New ModelConfig Fields

Eight new fields are added to `ModelConfig` (after `dropout`):

| Field | Type | Default | Constraint / Notes |
|-------|------|---------|-------------------|
| `model_family` | `str` | `"deepset"` | `"deepset"` or `"market_aware"` |
| `d_sample` | `int` | `64` | phi_sample output dim; must be divisible by `n_heads` when `n_sab_sample_per_feature > 0` |
| `n_sab_sample_per_feature` | `int` | `0` | SAB layers over n within each (q,j) slot; keep 0 (memory guard) |
| `sample_pool` | `str` | `"attn"` | Pooling over n: any key in `POOL_SCALE` |
| `use_ridge_expert` | `bool` | `False` | Enable RidgeExpert + gate |
| `ridge_lambda` | `float` | `1.0` | Ridge regularisation λ; must be > 0 |
| `residual_scale_init` | `float` | `0.1` | Init value of learned `residual_scale` scalar parameter |
| `gate_hidden_dim` | `int` | `64` | Hidden dim of `gate_head` MLP |

All new fields have safe defaults that do not change `retired MODEL1 model` behavior (backward
compatible for all existing checkpoints with `model_family="deepset"`).

---

## §8 Model Routing

`model.py` exposes `_instantiate_model(cfg: ModelConfig) -> nn.Module`:

```python
def _instantiate_model(cfg: ModelConfig) -> nn.Module:
    family = getattr(cfg, "model_family", "deepset")
    if family == "deepset":
        return retired MODEL1 model(cfg=cfg)
    if family == "market_aware":
        return retired MODEL2 model(cfg=cfg)
    raise ValueError(f"Unknown model_family: {family!r}")
```

**All three consumer files use this function exclusively:**

| File | Change |
|------|--------|
| `evaluate.py` | `load_model()` calls `_instantiate_model(cfg)` instead of `retired MODEL1 model(cfg=cfg)` |
| `train.py` | `train_fn()` reads `model_family` from `hyper_params` + `MODEL_FAMILY` env var; calls `_instantiate_model(cfg)` |
| `hpo.py` | Ray worker reads `model_family` from `config`; calls `_instantiate_model(cfg)` |

**Never hardcode `retired MODEL1 model(cfg=cfg)` in these files.**

---

## §9 Extensibility

The `(m, p, d_feat)` tensor at step 4 (after `sample_pool_layer`) is the natural hook
for future market extensions:

| Extension | Mechanism |
|-----------|-----------|
| Cross-price effects | Cross-feature attention in `sab_feat` (already present when `n_sab_feat > 0`) |
| Sparse substitution | Masked feature SAB — mask non-substitute feature pairs |
| Low-rank market structure | Low-rank factorization of the feature-feature attention weights |
| Treatment effects | Additional token slot encoding treatment indicator (extends 6 → 7 tokens) |
| Seasonality | Positional encoding over the feature axis |
| Market-demand priors | Additional expert (parallel to `RidgeExpert`) mixed via gate extension |

The gate mechanism (`pred = ridge + gate * neural`) is already an **expert-mixing interface**.
Future experts plug in at step 7 alongside `RidgeExpert`. No architectural choices in the
current design foreclose these extensions.

---

## §10 HPO Deferral

The HPO search space (`hpo.py`) is **unchanged**. `model_family` is a **fixed config-time
parameter**, not a search axis.

Do not broaden HPO over model width, pooling, SAB depth, or optimizer settings until the
corrected architecture passes query sanity checks (see `src/sanity_checks.py`). The first
priority is to eliminate query collapse and early feature compression. HPO expansion comes
after the model demonstrates query-sensitive behavior on controlled synthetic contexts.

To train a `market_aware` model, set the environment variable before launching training:

```bash
export MODEL_FAMILY=market_aware
python src/train.py
```

Or pass `model_family` in `hyper_params` from the orchestration layer.
