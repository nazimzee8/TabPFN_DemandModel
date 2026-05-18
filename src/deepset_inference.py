"""Shared DeepSet inference, feature selection, and invariance-test helpers."""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import torch
from sklearn.feature_selection import f_regression


def _env_flag(name, default="false"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


BENCHMARK_DEEPSET_CONTEXT_SIZE = int(os.environ.get("BENCHMARK_DEEPSET_CONTEXT_SIZE", "200"))
BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES = int(os.environ.get("BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES", "5"))
BENCHMARK_DEEPSET_TEST_BATCH_SIZE = int(os.environ.get("BENCHMARK_DEEPSET_TEST_BATCH_SIZE", "128"))
BENCHMARK_DEEPSET_FEATURE_SELECTOR = os.environ.get(
    "BENCHMARK_DEEPSET_FEATURE_SELECTOR", "train_f_regression"
)
BENCHMARK_DEEPSET_FEATURE_CAP = os.environ.get("BENCHMARK_DEEPSET_FEATURE_CAP")
BENCHMARK_REQUIRE_CUDA = _env_flag("BENCHMARK_REQUIRE_CUDA")
BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES = int(
    os.environ.get("BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES", "268435456")
)
BENCHMARK_DEEPSET_GPU_MEMORY_SAFETY_FACTOR = float(
    os.environ.get("BENCHMARK_DEEPSET_GPU_MEMORY_SAFETY_FACTOR", "4.0")
)
BENCHMARK_DEEPSET_MAX_GPU_MEMORY_FRACTION = float(
    os.environ.get("BENCHMARK_DEEPSET_MAX_GPU_MEMORY_FRACTION", "0.80")
)


def apply_feat_equiv(model, h):
    """h: (n, p, d) -> (n, p, d). Dispatches to SAB or linear equivariance."""
    if model.cfg.n_sab_feat > 0:
        return model.sab_feat(h)
    mean_i = h.mean(dim=1, keepdim=True)
    return model.lambda_feat * h + model.gamma_feat * mean_i


def apply_samp_equiv(model, r):
    """r: (n, d) -> (n, d). Dispatches to SAB or linear equivariance."""
    if model.cfg.n_sab_samp > 0:
        return model.sab_samp(r.unsqueeze(0)).squeeze(0)
    mean_j = r.mean(dim=0, keepdim=True)
    return model.lambda_samp * r + model.gamma_samp * mean_j


def run_permutation_tests(model):
    torch.manual_seed(0)
    n, p = 20, 5
    cfg = model.cfg
    device = next(model.parameters(), torch.empty(0)).device
    model_family = getattr(cfg, "model_family", "deepset")

    X_train = torch.randn(n, p, device=device)
    y_train = torch.randn(n, device=device)
    x_test  = torch.randn(p, device=device)
    was_training = model.training
    model.eval()
    results = {}

    try:
        with torch.no_grad():
            pi = torch.randperm(n, device=device)
            results["Test 1 (row permutation invariance)"] = torch.allclose(
                model(X_train, y_train, x_test),
                model(X_train[pi], y_train[pi], x_test),
                atol=1e-5,
            )

            pi_col = torch.randperm(p, device=device)
            results["Test 2 (column permutation invariance)"] = torch.allclose(
                model(X_train, y_train, x_test),
                model(X_train[:, pi_col], y_train, x_test[pi_col]),
                atol=1e-5,
            )

            if model_family == "market_aware":
                # Test 3: Feature equivariance via apply_feat_equiv (valid for MarketAware:
                # has sab_feat when n_sab_feat > 0, else lambda_feat/gamma_feat)
                d_feat = getattr(model, "_d_feat", getattr(cfg, "d_sample", 64))
                h = torch.randn(1, p, d_feat, device=device)
                pi_feat = torch.randperm(p, device=device)
                results["Test 3 (feature equiv equivariance)"] = torch.allclose(
                    apply_feat_equiv(model, h[:, pi_feat, :]),
                    apply_feat_equiv(model, h)[:, pi_feat, :],
                    atol=1e-4,
                )

                # Test 4: Finite output (no NaN/Inf)
                out = model(X_train, y_train, x_test)
                results["Test 4 (finite output)"] = bool(
                    torch.isfinite(out).all().item()
                )

                # Test 5: Batch query output shape
                x_batch = torch.randn(4, p, device=device)
                out_batch = model(X_train, y_train, x_batch)
                results["Test 5 (batch query shape)"] = out_batch.shape == (4,)

            elif model_family == "deepset":
                D_PHI = cfg.d_phi
                D_RHO = cfg.d_rho

                r, pi = torch.randn(n, D_RHO, device=device), torch.randperm(n, device=device)
                results["Test 3 (sample equiv equivariance)"] = torch.allclose(
                    apply_samp_equiv(model, r[pi]),
                    apply_samp_equiv(model, r)[pi],
                    atol=1e-5,
                )

                h, pi_feat = torch.randn(n, p, D_PHI, device=device), torch.randperm(p, device=device)
                results["Test 4 (feature equiv equivariance)"] = torch.allclose(
                    apply_feat_equiv(model, h[:, pi_feat, :]),
                    apply_feat_equiv(model, h)[:, pi_feat, :],
                    atol=1e-5,
                )

                r, pi = torch.randn(n, D_RHO, device=device), torch.randperm(n, device=device)
                results["Test 5 (sample invariance after pool)"] = torch.allclose(
                    apply_samp_equiv(model, r).mean(dim=0),
                    apply_samp_equiv(model, r[pi]).mean(dim=0),
                    atol=1e-5,
                )

                if cfg.n_sab_samp == 0:
                    r = torch.randn(n, D_RHO, device=device)
                    lam = model.lambda_samp.item()
                    gam = model.gamma_samp.item()
                    theta = lam * torch.eye(n, device=device) + (gam / n) * torch.ones(n, n, device=device)
                    results["Test 6 (Theta matrix form)"] = torch.allclose(
                        apply_samp_equiv(model, r),
                        theta @ r,
                        atol=1e-5,
                    )

                    r, pi = torch.randn(n, D_RHO, device=device), torch.randperm(n, device=device)
                    results["Test 7 (mean after permuted equiv)"] = torch.allclose(
                        apply_samp_equiv(model, r[pi]).mean(dim=0),
                        apply_samp_equiv(model, r).mean(dim=0),
                        atol=1e-5,
                    )
                else:
                    r, pi = torch.randn(n, D_RHO, device=device), torch.randperm(n, device=device)
                    rb = r.unsqueeze(0)
                    results["Test 6 (SAB sample equivariance)"] = torch.allclose(
                        model.sab_samp(rb[:, pi, :]).squeeze(0),
                        model.sab_samp(rb).squeeze(0)[pi],
                        atol=1e-4,
                    )

                    h, pi_feat = torch.randn(n, p, D_PHI, device=device), torch.randperm(p, device=device)
                    results["Test 7 (SAB feature equivariance)"] = torch.allclose(
                        model.sab_feat(h[:, pi_feat, :]),
                        model.sab_feat(h)[:, pi_feat, :],
                        atol=1e-4,
                    )
            elif model_family == "market_exchangeable_icl":
                # MODEL3 inductive: row permutation invariance + column permutation consistency
                # + finite output + batch query shape
                out_ref = model(X_train, y_train, x_test)
                results["Test 1 (row permutation invariance)"] = torch.allclose(
                    out_ref,
                    model(X_train[pi], y_train[pi], x_test),
                    atol=1e-5,
                )
                pi_col = torch.randperm(p, device=device)
                results["Test 2 (column permutation invariance)"] = torch.allclose(
                    out_ref,
                    model(X_train[:, pi_col], y_train, x_test[pi_col]),
                    atol=1e-5,
                )
                results["Test 3 (finite output)"] = bool(torch.isfinite(out_ref).all().item())
                x_batch = torch.randn(4, p, device=device)
                out_batch = model(X_train, y_train, x_batch)
                results["Test 4 (batch query shape)"] = out_batch.shape == (4,)

            elif model_family == "market_exchangeable_completion":
                # MODEL3 transductive: row and column equivariance + finite output
                X_mat  = torch.randn(n, p, device=device)
                mask   = torch.rand(n, p, device=device) > 0.3
                out_c  = model(X_mat, mask)
                results["Test 1 (finite output)"] = bool(torch.isfinite(out_c).all().item())
                results["Test 2 (output shape)"]  = out_c.shape == (n, p)

                pi_row = torch.randperm(n, device=device)
                out_perm_row = model(X_mat[pi_row], mask[pi_row])
                results["Test 3 (row equivariance)"] = torch.allclose(
                    out_perm_row, out_c[pi_row], atol=1e-5
                )
                pi_col2 = torch.randperm(p, device=device)
                out_perm_col = model(X_mat[:, pi_col2], mask[:, pi_col2])
                results["Test 4 (column equivariance)"] = torch.allclose(
                    out_perm_col, out_c[:, pi_col2], atol=1e-5
                )

            else:
                raise ValueError(
                    f"Unknown model_family: {model_family!r}. "
                    "Cannot run permutation tests."
                )
    finally:
        model.train(was_training)

    print("\nPermutation Invariance Tests:")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    return all_pass


def _stable_u64(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def select_deepset_context_indices(
    n_train,
    context_size,
    seed,
    context_index,
    dataset_identity="benchmark",
    context_ensembles=BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES,
):
    n_train = int(n_train)
    context_size = int(context_size)
    context_index = int(context_index)
    context_ensembles = int(context_ensembles)
    if n_train <= 0:
        raise ValueError("DeepSet context selection requires at least one training row.")
    if context_size <= 0:
        raise ValueError("BENCHMARK_DEEPSET_CONTEXT_SIZE must be positive.")
    if context_ensembles <= 0:
        raise ValueError("BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES must be positive.")
    if context_index < 0 or context_index >= context_ensembles:
        raise ValueError("context_index must be within the configured ensemble count.")
    if n_train < context_ensembles:
        raise ValueError(
            "DeepSet bounded-context ensemble requires at least one training row "
            "per non-overlapping context."
        )

    effective_context_size = min(context_size, 200)
    max_rows = context_ensembles * effective_context_size
    rng_seed = _stable_u64(dataset_identity, int(seed), "deepset_context_permutation")
    rng = np.random.default_rng(rng_seed)
    permuted = rng.permutation(n_train)
    if n_train >= max_rows:
        permuted = permuted[:max_rows]

    windows = np.array_split(permuted, context_ensembles)
    window = windows[context_index]
    if len(window) > effective_context_size:
        window = window[:effective_context_size]
    return window.astype(np.int64)


def resolve_deepset_feature_cap(model, feature_cap=BENCHMARK_DEEPSET_FEATURE_CAP, default_cap=128):
    """Resolve the DeepSet feature cap, defaulting to checkpoint cfg.d_phi."""
    if feature_cap not in (None, ""):
        cap = int(feature_cap)
    else:
        cfg = getattr(model, "cfg", None)
        cap = int(getattr(cfg, "d_phi", default_cap))
    if cap <= 0:
        raise ValueError("BENCHMARK_DEEPSET_FEATURE_CAP must be positive.")
    return cap


def select_deepset_features_train_only(
    X_train_p,
    y_train,
    X_test_p,
    feature_cap,
    feature_selector=BENCHMARK_DEEPSET_FEATURE_SELECTOR,
    *,
    f_regression_func=None,
):
    if feature_selector != "train_f_regression":
        raise ValueError(
            "Unsupported BENCHMARK_DEEPSET_FEATURE_SELECTOR "
            f"{feature_selector!r}; expected 'train_f_regression'."
        )

    processed_features = int(X_train_p.shape[1])
    feature_cap = int(feature_cap)
    selected_features = min(processed_features, feature_cap)
    metadata = {
        "processed_features": processed_features,
        "selected_features": selected_features,
        "feature_selector": feature_selector,
        "feature_cap": feature_cap,
    }

    if processed_features <= feature_cap:
        return X_train_p, X_test_p, metadata

    f_regression_impl = f_regression_func or f_regression
    f_stats, _ = f_regression_impl(X_train_p, y_train)
    scores = np.asarray(f_stats, dtype=np.float64)
    scores[~np.isfinite(scores)] = -np.inf
    column_indices = np.arange(processed_features)
    selected_by_rank = np.lexsort((column_indices, -scores))[:feature_cap]
    selected_columns = np.sort(selected_by_rank)
    return X_train_p[:, selected_columns], X_test_p[:, selected_columns], metadata


def deepset_inference_device(require_cuda=None):
    """Select and log the torch device used only by DeepSet inference."""
    require_cuda = BENCHMARK_REQUIRE_CUDA if require_cuda is None else require_cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(
            f"DeepSet benchmark inference device: cuda ({name}, {total_gb:.1f} GiB)",
            flush=True,
        )
        return device

    if require_cuda:
        raise RuntimeError(
            "BENCHMARK_REQUIRE_CUDA=true but CUDA is unavailable for DeepSet benchmark inference."
        )
    print("DeepSet benchmark inference device: cpu (CUDA unavailable)", flush=True)
    return torch.device("cpu")


def estimate_deepset_gpu_inference_bytes(
    n_train_rows,
    n_test_rows,
    n_features,
    context_size=None,
    test_batch_size=None,
    safety_factor=None,
):
    """Estimate peak explicit GPU tensors for one streamed DeepSet context (MODEL2)."""
    context_size = BENCHMARK_DEEPSET_CONTEXT_SIZE if context_size is None else context_size
    test_batch_size = BENCHMARK_DEEPSET_TEST_BATCH_SIZE if test_batch_size is None else test_batch_size
    safety_factor = (
        BENCHMARK_DEEPSET_GPU_MEMORY_SAFETY_FACTOR
        if safety_factor is None
        else safety_factor
    )
    effective_context_rows = min(int(n_train_rows), int(context_size), 200)
    effective_test_batch_rows = min(int(n_test_rows), int(test_batch_size))
    n_features = int(n_features)
    float32_bytes = 4
    tensor_elements = (
        effective_context_rows * n_features
        + effective_context_rows
        + effective_test_batch_rows * n_features
        + (2 * effective_test_batch_rows)
    )
    return int(math.ceil(tensor_elements * float32_bytes * float(safety_factor)))


def estimate_model3_icl_gpu_inference_bytes(
    n_train_rows,
    n_test_rows,
    n_features,
    d_phi,
    context_size=None,
    test_batch_size=None,
    safety_factor=None,
):
    """Estimate peak GPU memory for MODEL3 ICL inference.

    MODEL3 ICL builds H: (m, n, p, d_phi) where:
      m = test_batch_size (query batch)
      n = effective_context_rows
      p = n_features
      d_phi = model embedding dimension

    This is typically 10–100x larger than the MODEL2 estimate for the same
    (n_train_rows, n_test_rows, n_features) problem.
    """
    context_size = BENCHMARK_DEEPSET_CONTEXT_SIZE if context_size is None else context_size
    test_batch_size = BENCHMARK_DEEPSET_TEST_BATCH_SIZE if test_batch_size is None else test_batch_size
    safety_factor = (
        BENCHMARK_DEEPSET_GPU_MEMORY_SAFETY_FACTOR
        if safety_factor is None
        else safety_factor
    )
    m = min(int(n_test_rows), int(test_batch_size))   # query batch
    n = min(int(n_train_rows), int(context_size))      # context rows
    p = int(n_features)                                # features
    d = int(d_phi)
    float32_bytes = 4
    # Primary cost: H tensor (m, n, p, d_phi) replicated for each query
    h_tensor_elements = m * n * p * d
    # Secondary: input tensors (X_train, y_train, X_test)
    input_elements = n * p + n + m * p
    tensor_elements = h_tensor_elements + input_elements
    return int(math.ceil(tensor_elements * float32_bytes * float(safety_factor)))


def _cuda_free_bytes(device):
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, _ = torch.cuda.mem_get_info()
    return int(free_bytes)


def deepset_gpu_memory_skip_reason(
    X_train_np,
    X_test_np,
    device,
    max_inference_bytes=None,
    max_memory_fraction=None,
    model=None,
):
    """Check if inference should be skipped due to estimated GPU memory usage.

    For MODEL3 ICL models (model_family='market_exchangeable_icl'), the H tensor
    (m × n × p × d_phi) is computed and the estimator accounts for this.
    Returns (skip_reason_str_or_None, estimate_bytes_or_None).
    """
    if device is None or getattr(device, "type", str(device)) != "cuda":
        return None, None

    max_inference_bytes = (
        BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES
        if max_inference_bytes is None
        else max_inference_bytes
    )
    max_memory_fraction = (
        BENCHMARK_DEEPSET_MAX_GPU_MEMORY_FRACTION
        if max_memory_fraction is None
        else max_memory_fraction
    )

    # MODEL3 ICL: use the larger H-tensor estimate
    model_family = None
    if model is not None:
        cfg = getattr(model, "cfg", None)
        if cfg is not None:
            model_family = getattr(cfg, "model_family", None)

    if model_family == "market_exchangeable_icl":
        cfg = model.cfg
        d_phi = int(getattr(cfg, "d_phi", 64))
        estimate = estimate_model3_icl_gpu_inference_bytes(
            n_train_rows=X_train_np.shape[0],
            n_test_rows=X_test_np.shape[0],
            n_features=X_train_np.shape[1],
            d_phi=d_phi,
        )
        if int(max_inference_bytes) > 0 and estimate > int(max_inference_bytes):
            return (
                f"gpu_oom:model3_estimated_h_tensor_bytes={estimate} exceeds "
                f"BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES={int(max_inference_bytes)}"
            ), estimate
        free_budget = int(_cuda_free_bytes(device) * float(max_memory_fraction))
        if estimate > free_budget:
            return (
                f"gpu_oom:model3_estimated_h_tensor_bytes={estimate} exceeds "
                f"available CUDA memory budget {free_budget}"
            ), estimate
        return None, estimate

    # MODEL2 (deepset, market_aware): original estimator
    estimate = estimate_deepset_gpu_inference_bytes(
        n_train_rows=X_train_np.shape[0],
        n_test_rows=X_test_np.shape[0],
        n_features=X_train_np.shape[1],
    )
    if int(max_inference_bytes) > 0 and estimate > int(max_inference_bytes):
        return (
            f"estimated_gpu_inference_bytes={estimate} exceeds "
            "BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES="
            f"{int(max_inference_bytes)}"
        ), estimate

    free_budget = int(_cuda_free_bytes(device) * float(max_memory_fraction))
    if estimate > free_budget:
        return (
            f"estimated_gpu_inference_bytes={estimate} exceeds available CUDA "
            "memory budget "
            f"{free_budget} (BENCHMARK_DEEPSET_MAX_GPU_MEMORY_FRACTION="
            f"{float(max_memory_fraction)})"
        ), estimate
    return None, estimate


def predict_deepset_mc_streamed(
    model,
    X_train_np,
    y_train_np,
    X_test_np,
    K=32,
    test_batch_size=128,
    device=None,
):
    """
    MC dropout mean for one bounded context, streamed over test rows.
    Chunking is memory-only: output order and length match X_test_np exactly.
    """
    if K <= 0:
        raise ValueError("MC dropout K must be positive.")
    if test_batch_size <= 0:
        raise ValueError("BENCHMARK_DEEPSET_TEST_BATCH_SIZE must be positive.")

    device = device or deepset_inference_device()
    model.to(device)
    Xtr = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)
    n_test = int(X_test_np.shape[0])
    out = np.empty(n_test, dtype=np.float64)

    model.train()
    try:
        with torch.no_grad():
            for start in range(0, n_test, test_batch_size):
                end = min(start + test_batch_size, n_test)
                Xte = torch.as_tensor(X_test_np[start:end], dtype=torch.float32, device=device)
                pred_sum = None
                for _ in range(K):
                    pred = model(Xtr, ytr, Xte).detach()
                    pred_sum = pred if pred_sum is None else pred_sum + pred
                out[start:end] = (pred_sum / float(K)).cpu().numpy()
    finally:
        model.eval()
    return out


def predict_deepset_bounded_context_ensemble(
    model,
    X_train_np,
    y_train_np,
    X_test_np,
    seed,
    dataset_identity="benchmark",
    K=32,
    context_size=BENCHMARK_DEEPSET_CONTEXT_SIZE,
    context_ensembles=BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES,
    test_batch_size=BENCHMARK_DEEPSET_TEST_BATCH_SIZE,
    device=None,
    context_selector=select_deepset_context_indices,
    predictor=predict_deepset_mc_streamed,
):
    if context_ensembles <= 0:
        raise ValueError("BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES must be positive.")

    device = device or deepset_inference_device()
    context_preds = []
    for context_index in range(int(context_ensembles)):
        idx = context_selector(
            X_train_np.shape[0],
            context_size,
            seed,
            context_index,
            dataset_identity=dataset_identity,
            context_ensembles=context_ensembles,
        )
        if len(idx) > context_size:
            raise AssertionError("DeepSet context exceeded configured context size.")
        preds = predictor(
            model,
            X_train_np[idx],
            y_train_np[idx],
            X_test_np,
            K=K,
            test_batch_size=test_batch_size,
            device=device,
        )
        if preds.shape[0] != X_test_np.shape[0]:
            raise AssertionError("DeepSet prediction length did not match test split length.")
        context_preds.append(preds)

    return np.mean(np.stack(context_preds, axis=0), axis=0)
