# CUDA Graphs — Implementation Notes

## Status: Experimental, OFF by default

The CUDA-graph static-shape forward path is implemented but **not enabled** in any shipped job.
Set `USE_CUDA_GRAPHS=true` only for experiments.

## How it works

Tasks are padded to fixed static shapes so `torch.compile(mode="reduce-overhead")` can
capture CUDA graphs internally:

| Constant | Env override | Default | Meaning |
|----------|-------------|---------|---------|
| `CUDA_GRAPH_N_PAD` | `CUDA_GRAPH_N_PAD` | 256 | Max support-set rows |
| `CUDA_GRAPH_P_PAD` | `CUDA_GRAPH_P_PAD` | 32 | Max feature count |
| `CUDA_GRAPH_M_PAD` | `CUDA_GRAPH_M_PAD` | 64 | Query chunk size |

`n_valid`, `p_valid`, `m_valid` are passed as 0-dimensional tensors so Dynamo never
specializes on the sizes. The masked forward in `model.py: forward_regression` applies
boolean masks to exclude padding positions.

Float32 linalg operations (`svdvals`, `lstsq`, `solve`) run inside `@_no_compile` helpers
that graph-break to eager, preventing host synchronization within the captured region.

## Gating

The CUDA graph branch in `train.py` is active only when ALL of:
- `USE_CUDA_GRAPHS=true`
- `not _is_lbacnp` (model5_lbacnp is excluded — current shipped architecture)
- `not cat_kwargs` (mixed-categorical suites excluded)

In practice this means the gate is never triggered by shipped jobs.

## Pad-budget overflow behavior

Tasks with `n > N_PAD` or `p > P_PAD` emit a `RuntimeWarning` and fall back to the
standard eager/chunked path. Training is never aborted.

A one-time startup message is printed to stdout when `USE_CUDA_GRAPHS=true`:

```
[CUDA-GRAPH] Enabled: N_PAD=256, P_PAD=32, M_PAD=64. Tasks exceeding pad budgets fall back to eager.
```

Per-task fallback warnings have the form:

```
[CUDA-GRAPH] n=257 > N_PAD=256; falling back to eager path.
```

## Parity tests

CPU-only parity tests: `tests/test_cuda_graph_parity.py`

Three properties verified:
1. **Numerical parity** — masked-padded forward matches the unpadded (legacy) forward to
   ≤1e-5 absolute tolerance on primary predictions and model4 auxiliary-head debug tensors.
2. **Static shapes** — every padded call presents identical `(N_PAD, M_PAD, P_PAD)` tensor
   shapes regardless of the true `(n, p, m)` in the task.
3. **Fail-loud** — the padded path raises `ValueError` when invariants are violated (SetPool
   active, 1-D x_test, `p > P_PAD`). Note: the `p > P_PAD` and `n > N_PAD` size guards in
   `train.py` now emit `RuntimeWarning` and fall back gracefully; the model-level guards
   tested here are separate.

Covers model3 and model4 (model5_lbacnp excluded by gate, untested).

## Why off in production

Shipped jobs use gradient checkpointing, query chunking, and `expandable_segments` —
strategies sufficient for A10G (22 GiB) without shape restrictions.
