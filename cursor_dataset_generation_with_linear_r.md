# Dataset generation with linear relations
_Exported on 4/12/2026 at 11:11:02 PDT from Cursor (3.0.16)_

---

**User**

i want generate a dataset of datasets, each dataset shoud be follow a linear relation between y and X, step 1 draw n and p prbably from poisson and enforce n>> p. then draw beta and X and eps and y, in the final dataset each row should be D, xtest, and betaX_test

---

**Cursor**

Let me quickly check what's already in the workspace, then I'll put together a plan.

**Dataset Generation Details**

> **How many datasets (rows in the meta-dataset) should be generated?**
> - ~100
> - ~1,000 ✔
> - ~10,000
> - Configurable parameter (I'll decide later)

> **What output format do you want for the final meta-dataset?**
> - Python pickle (.pkl) — preserves variable-size arrays natively ✔
> - HDF5 (.h5) — efficient for large numerical data
> - Parquet — columnar, good for tabular workflows
> - NumPy .npz archive

---

**User**

@/Users/amdeep/.cursor/projects/Users-amdeep-Downloads-DGP/terminals/2.txt:8-9 

---

**Cursor**

I'll create the dataset generation script now. Since each dataset has different dimensions (varying n and p), pickle is the natural format.

Now let me run it in your `bond310` environment:

It ran successfully. Let me also add a quick sanity-check script so you can inspect the output:

Everything works. Here's a summary of what was generated:

**`meta_dataset.pkl`** — a list of 1,000 dictionaries, each containing:

| Key | Description | Shape |
|---|---|---|
| `D` | Training data dict with `X_train` and `y_train` | `(n_train, p)` and `(n_train,)` |
| `X_test` | Test design matrix | `(n_test, p)` |
| `betaX_test` | True signal on test data (`X_test @ beta`) | `(n_test,)` |
| `meta` | Bookkeeping: `n`, `p`, `n_train`, `n_test` | dict |

**Data generating process per dataset:**

1. **Dimensions** — `p ~ Poisson(10)`, `n ~ Poisson(200)`, rejection-sampled so `n >= 5p` (enforcing n >> p)
2. **Coefficients** — `beta ~ N(0, 1)` of length `p`
3. **Features** — `X ~ N(0, 1)` of size `(n, p)`, split 80/20 into train/test
4. **Noise** — `eps ~ N(0, 1)` on the training portion
5. **Response** — `y_train = X_train @ beta + eps`
6. **Ground truth** — `betaX_test = X_test @ beta` (noise-free)

The Poisson rates, noise level, test fraction, and other knobs are all configurable at the top of `generate_dgp.py`.

---

**User**

show me using asciss example of how the meta dataset looks like. no need to read the pickly just a random example would do

---

**Cursor**

Here's a concrete example of what one row of the meta-dataset looks like, with small made-up numbers:

```python
# Say for dataset i we drew: n=10, p=3, n_test=2, n_train=8

row_i = {
    "D": {
        "X_train": np.array([       # (8 x 3)
            [ 0.12, -1.04,  0.53],
            [-0.87,  0.22,  1.15],
            [ 1.43, -0.61, -0.30],
            [ 0.05,  0.98,  0.77],
            [-1.21,  0.45, -0.63],
            [ 0.66, -0.33,  0.12],
            [-0.44,  1.10,  0.89],
            [ 0.91, -0.78, -1.02],
        ]),
        "y_train": np.array([       # (8,)  = X_train @ beta + eps
            -0.73, 1.92, 0.48, 1.61, -1.05, 0.34, 2.11, -0.89
        ]),
    },

    "X_test": np.array([            # (2 x 3)
        [ 0.37,  0.85, -0.42],
        [-1.10,  0.14,  0.63],
    ]),

    "betaX_test": np.array([        # (2,)  = X_test @ beta  (no noise)
        0.71, -0.28
    ]),

    "meta": {"n": 10, "p": 3, "n_train": 8, "n_test": 2},
}
```

And the full `meta_dataset.pkl` is just a list of 1,000 of these:

```python
meta_dataset = [row_0, row_1, row_2, ..., row_999]
```

So to use it you'd do something like:

```python
import pickle

with open("meta_dataset.pkl", "rb") as f:
    datasets = pickle.load(f)

# grab dataset 42
d = datasets[42]
X_train = d["D"]["X_train"]   # (n_train, p)
y_train = d["D"]["y_train"]   # (n_train,)
X_test  = d["X_test"]         # (n_test, p)
truth   = d["betaX_test"]     # (n_test,) — the noiseless X_test @ beta

# fit your estimator on (X_train, y_train), predict on X_test, compare to truth
```

Each row has **different shapes** because `n` and `p` vary across datasets — that's why pickle is the natural storage format.

---

**User**

i want to use deep set here is specification i want to use, i want to train a model over this metadaset. 
here is the specificaiton -- \psi(\sum_j (\rho_j(\sum_i(\phi_i(y_i,x_i,x_test)))
i am assuming one test, we can change it slightly to incorporate more

---

**Cursor**

I have one critical question about the architecture before proposing a plan.

**DeepSets Architecture Clarification**

> **In your formula psi(sum_j rho(sum_i phi(y_i, x_i, x_test))), what does j index? This determines whether the model is singly or doubly permutation-invariant.**
> - j indexes features (columns of X) -- so phi takes 3 scalars: (y_i, x_ij, x_test_j), making the model invariant to both sample AND feature permutations
> - j indexes test points -- but you said assume 1 test for now, so the sum_j collapses
> - j indexes multiple 'heads' or channels (like multi-head attention)
> - Something else (I'll explain)
>
> Answer: j indexes sample points, i indexes features

> **Which deep learning framework?**
> - PyTorch ✔
> - JAX/Flax
> - TensorFlow/Keras

Good, so to confirm: **j = training samples, i = features**, and PyTorch. That means the architecture is:

$$\hat{y}_{\text{test}} = \psi\left(\sum_{j=1}^{n} \rho\left(\sum_{i=1}^{p} \phi(y_j, x_{ji}, x_{\text{test},i})\right)\right)$$

This is doubly permutation-invariant (over both samples and features), and since `phi` always takes 3 scalars, it naturally handles variable `n` and `p`. Let me read the current script and then propose the full plan.

---

**User**

change the design to bring out x_test so the double permutation structure gives some output which become the param for a function which takes x_Test

---

**Cursor**

Good insight -- separating the encoder (training data) from the predictor (test point) is cleaner and more modular. Let me update the plan.

The key design question is: the double DeepSet over `D` produces a **fixed-size** vector `theta`, but `x_test` has **variable dimension** `p`. So the predictor also needs a set aggregation over test features. Let me lay this out.

Now let me update the plan with the revised architecture:

---

**User**

can you draw the structure first in ascii

---

**Cursor**

Here's the updated architecture with `x_test` pulled out of the double DeepSet:

```
                        ENCODER (double permutation-invariant)
                        ======================================
                        Processes only training data D = {(x_j, y_j)}

    For each sample j, for each feature i:

        (y_j, x_ji)          ← 2 scalars
            │
            ▼
        ┌───────┐
        │  phi  │             MLP: R^2 → R^{d_phi}
        └───┬───┘
            │
            ▼
    ────── sum over i=1..p ──────   (feature permutation invariance)
            │
            ▼                       one vector per sample j
        ┌───────┐
        │  rho  │             MLP: R^{d_phi} → R^{d_rho}
        └───┬───┘
            │
            ▼
    ────── sum over j=1..n ──────   (sample permutation invariance)
            │
            ▼
        theta ∈ R^{d_rho}          ← fixed-size "params" summarizing D
            │
            │
            │
            │     PREDICTOR
            │     =========
            │     Consumes theta + x_test
            │
            │         x_test_i     ← scalar (one feature of test point)
            │             │
            ▼             ▼
        ┌─────────────────────┐
        │   gamma(theta, ·)   │   MLP: R^{d_rho + 1} → R^{d_gamma}
        └──────────┬──────────┘
                   │
                   ▼
    ────── sum over i=1..p ──────   (feature permutation invariance)
                   │
                   ▼
              ┌─────────┐
              │   psi   │         MLP: R^{d_gamma} → R^1
              └────┬────┘
                   │
                   ▼
              y_hat (scalar)        ← prediction for this test point
```

The key change vs. the previous design:

- **Before**: `phi` took 3 scalars `(y_j, x_ji, x_test_i)` -- test point was entangled inside the encoder.
- **Now**: The encoder sees only `(y_j, x_ji)` and produces a fixed-size `theta`. Then `gamma` combines `theta` with each scalar feature of `x_test` separately, summed over features, and `psi` maps to the prediction.

This gives a clean **encoder/predictor split**:
- `theta` is reusable across all test points from the same dataset (compute once, predict many)
- The predictor handles variable `p` via its own sum over test features
- Triple permutation invariance: over training samples `j`, training features `i`, and test features `i`

Want me to turn this into a formal plan, or would you like to adjust anything about the structure first?

---

**User**

i think the previous one was fine, how would this remember the ordering info   gamma(theta, ·)   

---

**Cursor**

You're exactly right. In the new design, `gamma(theta, x_test_3)` and `gamma(theta, x_test_7)` would be processed identically if their scalar values happened to match -- there's no way to know that feature 3 in the test point should correspond to feature 3 in the training data. The `theta` vector has already "forgotten" per-feature identity by summing over `i`.

The original design doesn't have this problem because `phi(y_j, x_ji, x_test_i)` always sees the **same index `i`** from both training and test. The feature alignment is preserved by construction -- the i-th training feature is always paired with the i-th test feature before anything gets summed away.

So let's stick with the original:

```
    (y_j, x_ji, x_test_i)     ← 3 scalars, same feature index i
            │
        ┌───────┐
        │  phi  │              MLP: R^3 → R^{d_phi}
        └───┬───┘
            │
    ── sum over i=1..p ──      (feature permutation invariance)
            │
        ┌───────┐
        │  rho  │              MLP: R^{d_phi} → R^{d_rho}
        └───┬───┘
            │
    ── sum over j=1..n ──      (sample permutation invariance)
            │
        ┌───────┐
        │  psi  │              MLP: R^{d_rho} → R^1
        └───┬───┘
            │
        y_hat (scalar)
```

Shall I update the plan and build it?

---

**User**

yes, build it

---

**Cursor**

Now let me run a quick smoke test to make sure everything loads and trains:

Everything runs. The model is learning (MSE drops from 10.1 to 2.7 in 3 epochs), though it still has a way to go compared to OLS (0.07). Let me kick off a proper training run:

---

---

## Architecture Review — DeepSet OOD Generalization (2026-04-23)

Diagnostic review against four properties: permutation invariance, permutation equivariance,
variable-dimension inputs, and OOD generalization.

---

### Diagnostic Findings

**Finding 1 — Permutation Invariance: SATISFIED**

The confirmed architecture `psi( sum_j  rho( sum_i  phi(y_j, x_ji, x_test_i) ) )`:
- `sum over i` → feature permutation invariance
- `sum over j` → sample permutation invariance
- Feature correspondence is preserved: index `i` is shared across `x_ji` and `x_test_i`
  before being summed away (see lines 281–303 above).

Gap: index alignment must be enforced at preprocessing for real market data.
Sum (not mean) pooling causes magnitude to scale with `n` and `p` — addressed in Change 2.

---

**Finding 2 — Permutation Equivariance: NOT PRESENT (gap)**

The architecture collapses to a scalar after two sum-poolings. No intermediate layer
preserves per-element representations that could be permuted coherently. A purely
invariant encoder treats each `(j, i)` pair as independent of all others; an equivariant
intermediate layer lets sample `j` condition on the aggregate behavior of all other
samples before being summed — analogous to attention.

This is critical: whether product A's price is high *relative to other products in the
context* is a relevant signal that pure sum-pooling cannot capture without equivariant
cross-sample interaction. Change 1 below addresses this.

---

**Finding 3 — Variable Feature Count (p): SATISFIED**

`p ~ Poisson(10)` per dataset. `phi` operates on scalars → sum over `i` handles any `p`.
No fixed input dimension assumed. Gap: feature scale normalization not specified;
addressed in Change 2 (mean-pooling).

---

**Finding 4 — Variable Set Size (n): SATISFIED**

`n ~ Poisson(200)` per dataset. Sum over `j` handles any `n`. Gap: sum-pooling magnitude
scales with `n` — addressed in Change 2.

---

**Finding 5 — OOD Generalization: PARTIALLY SATISFIED**

1,000 varied datasets form a meta-dataset. The in-context learning objective trains the
model to generalize the *procedure* not specific parameters. Gaps: no held-out meta-test
split defined, fixed DGP prior, no regularization plan, sum-pooling scale bias.

---

### Change 1 — Add Equivariant Layers

At both aggregation axes, insert one equivariant layer using the canonical
permutation-equivariant operation from Zaheer et al. 2017 (Deep Sets, Theorem 3):
`h' = λ·h + γ·mean(H)`.

Updated architecture:

```
INNER LOOP  (over features i, for each sample j)
──────────────────────────────────────────────
  h_ji  = phi(y_j, x_ji, x_test_i)      MLP: R^3 → R^d_phi

  EQUIVARIANT FEATURE LAYER (per sample j, across all i):
  h_ji' = λ · h_ji  +  γ · mean_i(h_ji)

  g_j   = mean_i  h_ji'                 aggregate → invariant over features

OUTER LOOP  (over samples j)
──────────────────────────────────────────────
  r_j   = rho(g_j)                      MLP: R^d_phi → R^d_rho  (per-sample)

  EQUIVARIANT SAMPLE LAYER (across all j):
  r_j'  = λ2 · r_j  +  γ2 · mean_j(r_j)

  z     = mean_j  r_j'                  aggregate → invariant over samples

INVARIANT OUTPUT
──────────────────────────────────────────────
  y_hat = psi(z)                        MLP: R^d_rho → R^1
```

PyTorch implementation:

```python
# Equivariant feature layer (after phi, before mean_i)
# h shape: (batch, n, p, d_phi)
mean_i = h.mean(dim=2, keepdim=True)          # (batch, n, 1, d_phi)
h_equiv = lambda_ * h + gamma_ * mean_i       # equivariant over i-axis

# Equivariant sample layer (after rho, before mean_j)
# r shape: (batch, n, d_rho)
mean_j = r.mean(dim=1, keepdim=True)          # (batch, 1, d_rho)
r_equiv = lambda2 * r + gamma2 * mean_j       # equivariant over j-axis
```

Four scalar learnable parameters total (`lambda_`, `gamma_`, `lambda2`, `gamma2`).
No new MLPs required. Multiple equivariant layers may be stacked per axis for greater
depth (see Deep Sets++, arXiv:2206.11925 for Set Norm + residual design).

---

### Change 2 — Switch to Mean-Pooling

Replace `sum` with `mean` at both aggregation steps to decouple representation
magnitude from set size. This ensures datasets with `n=30` and `n=300` produce
activations of similar magnitude.

---

### Change 3 — Meta-Level Train / Val / Test Split

Split 1,000 datasets: **800 meta-train / 100 meta-val / 100 meta-test**.
Store as three separate files. Meta-test is never touched until final evaluation.

```python
import random, pickle

with open("meta_dataset.pkl", "rb") as f:
    all_datasets = pickle.load(f)

random.shuffle(all_datasets)
splits = {
    "train": all_datasets[:800],
    "val":   all_datasets[800:900],
    "test":  all_datasets[900:],
}
for name, data in splits.items():
    with open(f"meta_{name}.pkl", "wb") as f:
        pickle.dump(data, f)
```

---

### Change 4 — Diversify the DGP Prior

Add three regime variants (sampled uniformly) and record `prior_regime` in each
dataset's `meta` dict:

| Regime | beta | eps | X |
|--------|------|-----|---|
| A (baseline) | `N(0,1)` | `N(0,1)` | iid `N(0,1)` |
| B (sparse beta) | 70% zeros, `N(0,2)` nonzero | `N(0,1)` | iid `N(0,1)` |
| C (heavy tail) | `N(0,1)` | `t(3)` | iid `N(0,1)` |
| D (correlated X) | `N(0,1)` | `N(0,1)` | AR(1) `ρ=0.6` |

```python
import numpy as np

def sample_regime():
    return np.random.choice(["A", "B", "C", "D"])

def draw_beta(p, regime):
    if regime == "B":
        beta = np.random.normal(0, 2, p)
        mask = np.random.rand(p) < 0.7
        beta[mask] = 0.0
        return beta
    return np.random.normal(0, 1, p)

def draw_eps(n, regime):
    if regime == "C":
        return np.random.standard_t(df=3, size=n)
    return np.random.normal(0, 1, n)

def draw_X(n, p, regime):
    if regime == "D":
        X = np.zeros((n, p))
        X[:, 0] = np.random.normal(0, 1, n)
        for k in range(1, p):
            X[:, k] = 0.6 * X[:, k-1] + np.sqrt(1 - 0.36) * np.random.normal(0, 1, n)
        return X
    return np.random.normal(0, 1, (n, p))
```

---

### Change 5 — Add Regularization

- Dropout `p=0.1` after each MLP layer in `phi`, `rho`, `psi`
- Adam with weight decay `1e-4`
- Early stopping on meta-val MSE (patience = 10 epochs); restore best checkpoint

```python
import torch, copy

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_mse = float("inf")
patience_count = 0
best_state = None

for epoch in range(max_epochs):
    train_one_epoch(model, train_loader, optimizer)
    val_mse = evaluate(model, val_loader)

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_state = copy.deepcopy(model.state_dict())
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= 10:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
```

---

### Verification Checklist

1. **Equivariance check**: permute training rows → same `y_hat`; permute features with
   matched `x_test` → same `y_hat`; permute only training rows but not test → same `y_hat`.
2. **Mean-pooling scale check**: datasets with `n=30` and `n=300` produce activations of
   similar magnitude (compare `r_equiv.norm()` across both).
3. **Meta-split check**: three files present, counts 800 / 100 / 100.
4. **OOD report**: stratified MSE table broken down by `prior_regime`, `p` quartile,
   `n` quartile.
5. **Regime ordering**: Regime A should have lowest meta-test MSE; Regimes C and D higher
   but not catastrophic.

---

## Model Specification — PPD, Permutation Invariance and Equivariance (2026-04-23)

---

### A0 — Theoretical Grounding: PPD Connection

**TabPFN Posterior Predictive Distribution**

TabPFN computes:

```
p(y | x, D) = ∫ p(y | x, θ) · p(D | θ) · p(θ) dθ
```

TabPFN approximates this integral using a Transformer trained on synthetic datasets
drawn from a prior over θ. Attention lets each test point adaptively weight training
samples: the test query attends to training keys, producing a test-point-specific
context vector.

**Does the DeepSet approximate the same PPD?**

Yes. In the linear regression setting:
- θ = β (regression coefficients)
- p(y | x, θ) = N(x'β, σ²)
- p(D | θ) = ∏_j N(y_j | x_j'β, σ²)
- p(θ) = N(0, I)

The PPD is the Bayesian posterior predictive — computed exactly in closed form as:

```
ŷ_bayes = x_test' · (X'X + I)^{-1} · X'y
```

The DeepSet learns to approximate this from (D, x_test) → ŷ by training on datasets
sampled from p(D|β)·p(β). It never sees θ=β explicitly; it must infer it from context.

**Why the DeepSet can approximate the PPD:**

`phi(y_j, x_ji, x_test_i)` entangles each training sample j with the test point at
feature level i before pooling. This makes the aggregation test-conditioned. Each
`g_j = mean_i(phi(y_j, x_ji, x_test_i))` is a test-dependent summary of sample j.
`mean_j(rho(g_j))` aggregates these into a global summary z. `psi(z)` maps z → ŷ.

**Why this is sufficient for exponential families:**

For exponential families (including Gaussian linear regression), the sufficient
statistics `Σ_j φ(x_j, y_j)` are all that is needed to compute the PPD exactly.
The DeepSet's mean-pooling of phi outputs is learning to compute these sufficient
statistics. For Gaussian regression this is theoretically sufficient — the DeepSet
can in principle learn the exact Bayesian solution.

**Comparison table:**

| Property | TabPFN (Transformer) | DeepSet |
|---|---|---|
| Cross-sample weighting | Adaptive (attention) conditioned on x_test | Uniform (mean/max) |
| Test-point conditioning | Via cross-attention query | Via phi input: x_test_i |
| Expressiveness | Higher (attention heads) | Lower but exact for exponential families |
| Permutation invariance | By construction (attention) | By construction (pooling) |
| Theoretical grounding | PPD via attention approximation | PPD via sufficient statistics |

**Conclusion:**
- No objective or data generation change is needed.
- MSE on betaX_test IS the Bayes-optimal target under squared loss.
- The equivariant layers add cross-sample interaction (analogous to one step of
  self-attention), narrowing the expressiveness gap with Transformers.
- The architecture is theoretically sound for PPD approximation.

---

### A1 — Permutation-Invariant Model

**Formal definition and full data-flow:**

```
h_ji  = phi(y_j, x_ji, x_test_i)         MLP: R^3 → R^d_phi
h_ji' = λ_1·h_ji + γ_1·pool_i(h_j·)     EQUIV FEATURE LAYER
g_j   = mean_i(h_ji')                     invariant over features → R^d_phi
r_j   = rho(g_j)                          MLP: R^d_phi → R^d_rho
r_j'  = λ_2·r_j + γ_2·pool_j(r·)        EQUIV SAMPLE LAYER
z     = mean_j(r_j')                       invariant over samples → R^d_rho
ŷ     = psi(z)                             MLP: R^d_rho → R
```

**Why invariance holds:**
- Permuting j permutes r_j' → mean_j unchanged → invariant over samples.
- Permuting i permutes h_ji' → mean_i unchanged (given x_test_i permuted in lock-step)
  → invariant over features.

---

### A2 — Permutation-Equivariant Layer

**Matrix form (Zaheer et al. 2017, Theorem 3):**

```
f_Θ(X) = σ(ΘX)   where   Θ = λI + γ(11^T)
```

All diagonal entries are tied (λ), all off-diagonal entries are tied (γ).
This is the unique form of a linear layer that is permutation-equivariant.

**Elementwise form:**

```
f(x_j) = λ·x_j + γ·pool(X)    for each j
```

**Two pool variants** (both equivariant because pool is permutation-invariant):

```
Variant 1 (mean):  pool(X) = mean_j(x_j)   →  Θ = λI + γ/n·11^T
Variant 2 (max):   pool(X) = max_j(x_j)    →  f(x) = σ(λx + γ·maxpool(X)·1)
```

**Equivariance proof sketch (mean variant):**

```
f(π(X))_j = λ·x_{π(j)} + γ·pool(π(X))
           = λ·x_{π(j)} + γ·pool(X)       ← pool is permutation-invariant
           = π(f(X))_j  ✓
```

The same proof holds for max-pool since `max_j(x_{π(j)}) = max_j(x_j)`.

---

### A3 — Validation Tests

**Test 1 — Sample permutation invariance (end-to-end):**

Permute rows of (X_train, y_train) by random π.
Assert `model(X_train[π], y_train[π], x_test) ≈ model(X_train, y_train, x_test)`.

**Test 2 — Feature permutation invariance (end-to-end):**

Permute columns of X_train and entries of x_test by same π.
Assert `model(X_train[:,π], y_train, x_test[π]) ≈ model(X_train, y_train, x_test)`.

**Test 3 — Equivariant sample layer isolation (mean variant):**

```python
r = randn(n, d)
r_out = equiv_j(r)
r_out_perm = equiv_j(r[π])
assert r_out_perm ≈ r_out[π]
```

**Test 4 — Equivariant feature layer isolation (mean variant):**

```python
h = randn(n, p, d)
h_out = equiv_i(h)
h_out_perm = equiv_i(h[:, π, :])
assert h_out_perm ≈ h_out[:, π, :]
```

**Test 5 — Max-pool variant:**

Repeat Tests 3 and 4 with `pool='max'` in the equivariant layer implementation.

**Test 6 — Θ matrix form equivalence (mean variant, no activation):**

```python
# Construct Θ = λI + γ/n·11^T explicitly
theta = lambda_ * torch.eye(n) + (gamma_ / n) * torch.ones(n, n)
# Compare: equiv_j(r) ≈ (theta @ r)
assert equiv_j(r) ≈ theta @ r
```

Confirms the implementation matches the Zaheer et al. theoretical form.

**Test 7 — Equivariant + pool restores invariance:**

```python
# Permute r, apply equiv_j, take mean(dim=0)
z_perm = equiv_j(r[π]).mean(dim=0)
z_orig = equiv_j(r).mean(dim=0)
assert z_perm ≈ z_orig
```

---

### A4 — Model Output File (best.pt)

`best.pt` is saved via `torch.save(model.state_dict(), "best.pt")` whenever validation
MSE improves. It contains the state dict of the full DeepSet: phi, rho, psi MLPs plus
the four equivariant scalars (λ_1, γ_1, λ_2, γ_2).

**State dict keys (illustrative):**

```
phi.layers.0.weight   phi.layers.0.bias
phi.layers.2.weight   phi.layers.2.bias
lambda_feat           gamma_feat
rho.layers.0.weight   rho.layers.0.bias
rho.layers.2.weight   rho.layers.2.bias
lambda_samp           gamma_samp
psi.layers.0.weight   psi.layers.0.bias
psi.layers.2.weight   psi.layers.2.bias
```

**Reload snippet:**

```python
import torch
from model import DeepSetModel   # same architecture definition

model = DeepSetModel(d_phi=64, d_rho=64)
model.load_state_dict(torch.load("best.pt", map_location="cpu"))
model.eval()

# Inference on a new dataset
y_hat = model(X_train_new, y_train_new, x_test_new)
```

**Stored path in Snowflake:**

```
@MODEL_STAGE/checkpoints/best.pt
```

Uploaded from the SPCS container via:

```python
session.file.put("best.pt", "@MODEL_STAGE/checkpoints/", overwrite=True)
```

