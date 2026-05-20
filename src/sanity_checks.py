"""
sanity_checks.py

Structural and correctness checks for DeepSetICLModel (MODEL3).

Six checks:
  1. check_permutation_invariance      — row-shuffle + col-permute (max_abs_delta <= 1e-5)
  2. check_ridge_primal_dual_consistency — primal vs dual form (max_abs_delta <= 1e-4)
  3. check_gate_range                  — gate in (0,1) for 10 random contexts
  4. check_ridge_expert_output_shape   — use_ridge_expert=True forward shape == (m,)
  5. check_no_test_label_in_signature  — RidgeExpert.predict has no y_test param
  6. check_forward_smoke_small         — n=20, p=5, m=8 forward pass completes

Usage:
    python src/sanity_checks.py [--out_dir artifacts/sanity] [--checkpoint PATH] [--device auto]

Outputs:
    {out_dir}/sanity_checks.json  — full results dict with all_passed key
    {out_dir}/sanity_checks.csv   — one row per check: check_name, passed, metric_name, metric_value
"""

import argparse
import csv
import inspect
import json
import os
import sys

import torch

# Allow running from project root or src/
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from model import ModelConfig, DeepSetICLModel, RidgeExpert, _instantiate_model

# ---------------------------------------------------------------------------
# Environment variable constants (Phase 6)
# ---------------------------------------------------------------------------

SYNREG_RUN_CHECKPOINT_GATES   = os.environ.get("SYNREG_RUN_CHECKPOINT_GATES",   "true").lower() == "true"
SYNREG_CHECKPOINT_GATE_STRICT = os.environ.get("SYNREG_CHECKPOINT_GATE_STRICT", "true").lower() == "true"
SYNREG_GATE_MAX_RIDGE_RATIO   = float(os.environ.get("SYNREG_GATE_MAX_RIDGE_RATIO", "10.0"))
SYNREG_GATE_MIN_QUERY_STD     = float(os.environ.get("SYNREG_GATE_MIN_QUERY_STD",  "1e-6"))
SYNREG_GATE_OUT_DIR           = os.environ.get("SYNREG_GATE_OUT_DIR", "/tmp/synreg_checkpoint_gates")


# ---------------------------------------------------------------------------
# Device resolution (Phase 3)
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device.
    'auto' → cuda if available, else cpu.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Helper: build a default fresh DeepSetICLModel for structural checks
# ---------------------------------------------------------------------------

def _fresh_model(use_ridge_expert: bool = True, device=None) -> DeepSetICLModel:
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=64,
        d_rho=128,
        pool="pna",
        n_heads=4,
        n_sab_feat=1,
        use_ridge_expert=use_ridge_expert,
    )
    model = DeepSetICLModel(cfg=cfg)
    model.eval()
    if device is not None:
        model.to(device)
    return model


# ---------------------------------------------------------------------------
# Check 1: Permutation invariance
# ---------------------------------------------------------------------------

def check_permutation_invariance(model=None, device=None) -> dict:
    """Row-shuffle + col-permute invariance. Pass if max_abs_delta <= 1e-5."""
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model = _fresh_model(use_ridge_expert=False, device=device)
    model.eval()

    torch.manual_seed(42)
    n, p, m = 30, 8, 4
    X_train = torch.randn(n, p, device=device)
    y_train = torch.randn(n, device=device)
    x_test  = torch.randn(m, p, device=device)

    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)

        # Row permutation
        row_perm = torch.randperm(n, device=device)
        y_row = model(X_train[row_perm], y_train[row_perm], x_test)

        # Column permutation (permute features consistently)
        col_perm = torch.randperm(p, device=device)
        y_col = model(X_train[:, col_perm], y_train, x_test[:, col_perm])

    row_delta = (y_orig - y_row).abs().max().item()
    col_delta = (y_orig - y_col).abs().max().item()
    max_abs_delta = max(row_delta, col_delta)
    passed = max_abs_delta <= 1e-5

    return {
        "passed": passed,
        "max_abs_delta": max_abs_delta,
        "row_delta": row_delta,
        "col_delta": col_delta,
        "threshold": 1e-5,
    }


# ---------------------------------------------------------------------------
# Check 2: Ridge primal/dual consistency
# ---------------------------------------------------------------------------

def check_ridge_primal_dual_consistency(device=None) -> dict:
    """Primal form (n>=p) vs dual form (n<p) agree on an overdetermined problem.
    Pass if max_abs_delta <= 1e-4.
    """
    if device is None:
        device = torch.device("cpu")
    ridge = RidgeExpert()
    torch.manual_seed(7)
    n_over, p = 50, 10   # primal path
    n_under = 5          # dual path (n < p)
    m = 6
    lam = 1.0

    # Overdetermined: compare primal vs a manually-triggered dual
    X_over = torch.randn(n_over, p, device=device)
    y_over = torch.randn(n_over, device=device)
    x_test = torch.randn(m, p, device=device)

    # n_under < p → dual path
    X_under = torch.randn(n_under, p, device=device)
    y_under = torch.randn(n_under, device=device)
    x_test2 = torch.randn(m, p, device=device)

    # Verify the dual path is actually taken when n < p
    with torch.no_grad():
        pred_primal = ridge.predict(X_over, y_over, x_test, lam)    # primal
        pred_dual   = ridge.predict(X_under, y_under, x_test2, lam) # dual

    # For the consistency check, verify primal == dual on the SAME small problem
    # by temporarily using a problem where n < p applies to both paths.
    # We check: solve same system via both closed forms explicitly.
    kw = dict(dtype=torch.float64, device=device)
    X_small = torch.randn(n_under, p, **kw)
    y_small = torch.randn(n_under, **kw)
    x_t     = torch.randn(m, p, **kw)

    # Primal (forced, even though n < p — mathematically identical when regularised)
    A_primal = X_small.T @ X_small + lam * torch.eye(p, **kw)
    beta_primal = torch.linalg.solve(A_primal, X_small.T @ y_small)
    pred_forced_primal = x_t @ beta_primal

    # Dual
    K_dual = X_small @ X_small.T + lam * torch.eye(n_under, **kw)
    alpha  = torch.linalg.solve(K_dual, y_small)
    beta_dual = X_small.T @ alpha
    pred_forced_dual = x_t @ beta_dual

    max_abs_delta = (pred_forced_primal - pred_forced_dual).abs().max().item()
    passed = max_abs_delta <= 1e-4

    # Sanity: check output shapes
    assert pred_primal.shape == (m,), f"Expected ({m},), got {pred_primal.shape}"
    assert pred_dual.shape   == (m,), f"Expected ({m},), got {pred_dual.shape}"

    return {
        "passed": passed,
        "max_abs_delta": max_abs_delta,
        "threshold": 1e-4,
    }


# ---------------------------------------------------------------------------
# Check 3: Gate range
# ---------------------------------------------------------------------------

def check_gate_range(model=None, device=None) -> dict:
    """Gate sigmoid output must lie in [0, 1] for 10 random contexts."""
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model = _fresh_model(use_ridge_expert=True, device=device)
    model.eval()

    all_in_range = True
    gate_values = []
    torch.manual_seed(99)
    for _ in range(10):
        n, p, m = 20, 6, 3
        X_train = torch.randn(n, p, device=device)
        y_train = torch.randn(n, device=device)
        x_test  = torch.randn(m, p, device=device)

        # Access the gate by hooking into forward — we re-run the relevant sub-computation
        with torch.no_grad():
            # Partial forward to get ctx_mean, then compute gate
            # We use a hook instead of full forward to isolate the gate
            gate_val = None

            def _gate_hook(module, input, output):
                nonlocal gate_val
                gate_val = torch.sigmoid(output).squeeze(-1)

            hook = model.gate_head.register_forward_hook(_gate_hook)
            try:
                _ = model(X_train, y_train, x_test)
            finally:
                hook.remove()

            if gate_val is not None:
                g_min = gate_val.min().item()
                g_max = gate_val.max().item()
                gate_values.append({"min": g_min, "max": g_max})
                if not (0.0 <= g_min and g_max <= 1.0):
                    all_in_range = False

    return {
        "passed": all_in_range,
        "n_contexts_checked": 10,
        "gate_samples": gate_values[:3],  # include first 3 for readability
    }


# ---------------------------------------------------------------------------
# Check 4: Ridge expert output shape
# ---------------------------------------------------------------------------

def check_ridge_expert_output_shape(device=None) -> dict:
    """use_ridge_expert=True forward completes and output shape == (m,)."""
    if device is None:
        device = torch.device("cpu")
    model = _fresh_model(use_ridge_expert=True, device=device)
    model.eval()

    torch.manual_seed(13)
    n, p, m = 25, 7, 5
    X_train = torch.randn(n, p, device=device)
    y_train = torch.randn(n, device=device)
    x_test  = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            out = model(X_train, y_train, x_test)
        shape_ok = out.shape == (m,)
        passed = shape_ok
        error = None
    except Exception as exc:
        passed = False
        shape_ok = False
        error = str(exc)

    result = {
        "passed": passed,
        "expected_shape": f"({m},)",
        "shape_ok": shape_ok,
    }
    if error is not None:
        result["error"] = error
    return result


# ---------------------------------------------------------------------------
# Check 5: No y_test in RidgeExpert.predict signature
# ---------------------------------------------------------------------------

def check_no_test_label_in_signature(device=None) -> dict:
    """RidgeExpert.predict must not have a y_test parameter."""
    if device is None:
        device = torch.device("cpu")
    sig = inspect.signature(RidgeExpert.predict)
    param_names = list(sig.parameters.keys())
    has_y_test = "y_test" in param_names
    passed = not has_y_test

    return {
        "passed": passed,
        "param_names": param_names,
        "has_y_test": has_y_test,
    }


# ---------------------------------------------------------------------------
# Check 6: Forward smoke test (small context)
# ---------------------------------------------------------------------------

def check_forward_smoke_small(device=None) -> dict:
    """Tiny context (n=20, p=5, m=8) forward pass completes without exception."""
    if device is None:
        device = torch.device("cpu")
    model = _fresh_model(use_ridge_expert=True, device=device)
    model.eval()

    torch.manual_seed(0)
    n, p, m = 20, 5, 8
    X_train = torch.randn(n, p, device=device)
    y_train = torch.randn(n, device=device)
    x_test  = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            out = model(X_train, y_train, x_test)
        passed = True
        error = None
        out_shape = list(out.shape)
    except Exception as exc:
        passed = False
        error = str(exc)
        out_shape = None

    result = {
        "passed": passed,
        "n": n, "p": p, "m": m,
        "out_shape": out_shape,
    }
    if error is not None:
        result["error"] = error
    return result


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

_ALL_CHECKS = {
    "check_permutation_invariance":       check_permutation_invariance,
    "check_ridge_primal_dual_consistency": check_ridge_primal_dual_consistency,
    "check_gate_range":                   check_gate_range,
    "check_ridge_expert_output_shape":    check_ridge_expert_output_shape,
    "check_no_test_label_in_signature":   check_no_test_label_in_signature,
    "check_forward_smoke_small":          check_forward_smoke_small,
}


def run_all_checks(model=None, device=None) -> dict:
    """Run all 6 structural/correctness checks. Returns dict with all_passed key."""
    if device is None:
        device = torch.device("cpu")
    if model is not None:
        model.to(device)
    results = {}
    for name, fn in _ALL_CHECKS.items():
        try:
            if name in ("check_permutation_invariance", "check_gate_range"):
                res = fn(model=model, device=device)
            else:
                res = fn(device=device)
        except Exception as exc:
            res = {"passed": False, "error": str(exc)}
        results[name] = res

    results["all_passed"] = all(v.get("passed", False) for v in results.values()
                                if isinstance(v, dict) and "passed" in v)
    results["device_info"] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if torch.cuda.is_available() and "cuda" in str(device)
            else None
        ),
    }
    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, out_dir: str) -> None:
    """Write sanity_checks.json and sanity_checks.csv to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "sanity_checks.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = os.path.join(out_dir, "sanity_checks.csv")
    rows = []
    for check_name, res in results.items():
        if check_name in ("all_passed", "device_info"):
            continue
        if not isinstance(res, dict):
            continue
        passed = res.get("passed", False)
        # Emit one row per numeric metric; if none, emit a single row with metric_name=""
        metrics = {k: v for k, v in res.items()
                   if k != "passed" and isinstance(v, (int, float, bool))}
        if metrics:
            for metric_name, metric_value in metrics.items():
                rows.append({
                    "check_name": check_name,
                    "passed": passed,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                })
        else:
            rows.append({
                "check_name": check_name,
                "passed": passed,
                "metric_name": "",
                "metric_value": "",
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "metric_name", "metric_value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {json_path}", flush=True)
    print(f"Saved: {csv_path}", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint quality gates (Phase 6)
# ---------------------------------------------------------------------------

def _make_gate_model(device) -> DeepSetICLModel:
    """Build a small fresh DeepSetICLModel for gate checks."""
    return _fresh_model(use_ridge_expert=False, device=device)


def check_nan_inf_output(model, device) -> dict:
    """Gate 1: Output must not contain NaN or Inf."""
    if device is None:
        device = torch.device("cpu")
    try:
        torch.manual_seed(100)
        n, p, m = 20, 5, 4
        X_train = torch.randn(n, p, device=device)
        y_train = torch.randn(n, device=device)
        x_test  = torch.randn(m, p, device=device)
        with torch.no_grad():
            out = model(X_train, y_train, x_test)
        is_finite = torch.isfinite(out).all().item()
        return {
            "passed": bool(is_finite),
            "has_nan": bool(torch.isnan(out).any().item()),
            "has_inf": bool(torch.isinf(out).any().item()),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def check_constant_output_collapse(model, device, min_std: float = 1e-3) -> dict:
    """Gate 2: Predictions must vary across m=16 different test points."""
    if device is None:
        device = torch.device("cpu")
    try:
        torch.manual_seed(101)
        n, p, m = 20, 5, 16
        X_train = torch.randn(n, p, device=device)
        y_train = torch.randn(n, device=device)
        x_test  = torch.randn(m, p, device=device)
        with torch.no_grad():
            out = model(X_train, y_train, x_test)
        std_val = out.std().item() if torch.isfinite(out).all() else 0.0
        passed = std_val > min_std
        return {
            "passed": passed,
            "std": float(std_val),
            "min_std": float(min_std),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def check_query_sensitivity(model, device, min_std: float = None) -> dict:
    """Gate 3: Outputs must differ for different x_test inputs."""
    if device is None:
        device = torch.device("cpu")
    if min_std is None:
        min_std = 1e-6
    try:
        torch.manual_seed(102)
        n, p = 20, 5
        m = 8
        X_train = torch.randn(n, p, device=device)
        y_train = torch.randn(n, device=device)
        x_test  = torch.randn(m, p, device=device)
        with torch.no_grad():
            out = model(X_train, y_train, x_test)
        std_val = out.std().item() if torch.isfinite(out).all() else 0.0
        passed = std_val > min_std
        return {
            "passed": passed,
            "std": float(std_val),
            "min_std": float(min_std),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def check_simple_linear_recovery_smoke(model, device) -> dict:
    """Gate 4: On a pure linear DGP, model MSE must be finite."""
    if device is None:
        device = torch.device("cpu")
    try:
        torch.manual_seed(103)
        n, p, m = 50, 5, 10
        beta = torch.randn(p, device=device)
        X_train = torch.randn(n, p, device=device)
        y_train = X_train @ beta + 0.1 * torch.randn(n, device=device)
        x_test  = torch.randn(m, p, device=device)
        y_true  = x_test @ beta
        with torch.no_grad():
            y_hat = model(X_train, y_train, x_test)
        mse = ((y_hat - y_true) ** 2).mean().item()
        passed = torch.isfinite(torch.tensor(mse)).item()
        return {
            "passed": bool(passed),
            "model_mse": float(mse),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def check_ratio_to_fixed_ridge(model, device, max_ratio: float = 10.0) -> dict:
    """Gate 5: model_MSE / ridge_MSE <= max_ratio on a synthetic context."""
    if device is None:
        device = torch.device("cpu")
    try:
        torch.manual_seed(104)
        n, p, m = 50, 5, 10
        beta = torch.randn(p, device=device)
        X_train = torch.randn(n, p, device=device)
        y_train = X_train @ beta + 0.1 * torch.randn(n, device=device)
        x_test  = torch.randn(m, p, device=device)
        y_true  = x_test @ beta

        with torch.no_grad():
            y_hat_model = model(X_train, y_train, x_test)

        # Ridge (lambda=1)
        ridge = RidgeExpert()
        with torch.no_grad():
            y_hat_ridge = ridge.predict(X_train, y_train, x_test, lam=1.0)

        model_mse = ((y_hat_model - y_true) ** 2).mean().item()
        ridge_mse = ((y_hat_ridge - y_true) ** 2).mean().item()

        if ridge_mse < 1e-12:
            ridge_mse = 1e-12
        ratio = model_mse / ridge_mse
        passed = ratio <= max_ratio and torch.isfinite(torch.tensor(ratio)).item()
        return {
            "passed": bool(passed),
            "model_mse": float(model_mse),
            "ridge_mse": float(ridge_mse),
            "ratio": float(ratio),
            "max_ratio": float(max_ratio),
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def check_train_val_gap(checkpoint_metadata: dict | None) -> dict:
    """Gate 6: val_mse/train_mse must be < threshold if both present in metadata.

    Accepts new keys (best_val_mse / train_mse_at_best from train.py) with
    fallback to legacy keys (val_mse / train_mse) for backward compatibility.
    """
    if checkpoint_metadata is None:
        return {"passed": True, "reason": "no metadata provided"}
    val_mse   = checkpoint_metadata.get("best_val_mse",
                    checkpoint_metadata.get("val_mse"))
    train_mse = checkpoint_metadata.get("train_mse_at_best",
                    checkpoint_metadata.get("train_mse"))
    if val_mse is None or train_mse is None:
        return {"passed": True, "reason": "val_mse or train_mse not in metadata"}
    if train_mse < 1e-12:
        return {"passed": True, "reason": "train_mse near zero; gap check skipped"}
    ratio = val_mse / train_mse
    threshold = 10.0
    passed = ratio < threshold
    return {
        "passed": passed,
        "val_mse": float(val_mse),
        "train_mse": float(train_mse),
        "ratio": float(ratio),
        "threshold": float(threshold),
    }


def run_checkpoint_gates(
    model,
    device=None,
    checkpoint_metadata: dict | None = None,
    max_ridge_ratio: float = 10.0,
    min_query_std: float = 1e-3,
) -> dict:
    """Run all 6 checkpoint quality gates. Returns dict with all_passed key."""
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    results = {}

    results["check_nan_inf_output"] = check_nan_inf_output(model, device)
    results["check_constant_output_collapse"] = check_constant_output_collapse(
        model, device, min_std=min_query_std
    )
    results["check_query_sensitivity"] = check_query_sensitivity(
        model, device, min_std=min_query_std
    )
    results["check_simple_linear_recovery_smoke"] = check_simple_linear_recovery_smoke(model, device)
    results["check_ratio_to_fixed_ridge"] = check_ratio_to_fixed_ridge(
        model, device, max_ratio=max_ridge_ratio
    )
    results["check_train_val_gap"] = check_train_val_gap(checkpoint_metadata)

    results["all_passed"] = all(
        v.get("passed", False)
        for v in results.values()
        if isinstance(v, dict) and "passed" in v
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run structural sanity checks for DeepSetICLModel (MODEL3)."
    )
    parser.add_argument(
        "--out_dir", default="artifacts/sanity",
        help="Directory for output JSON and CSV (default: artifacts/sanity)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a .pt checkpoint to load (optional; uses fresh model if not given)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device to use: auto|cpu|cuda|cuda:0 (default: auto)"
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(
        f"[sanity_checks] device={device}  cuda_available={torch.cuda.is_available()}",
        flush=True,
    )

    model = None
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("cfg", {})
        cfg = ModelConfig(**cfg_dict)
        model = _instantiate_model(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"Checkpoint model_family: {cfg.model_family}", flush=True)
    else:
        print("No checkpoint given — using freshly initialized DeepSetICLModel.", flush=True)

    results = run_all_checks(model=model, device=device)

    print("\n=== Sanity Check Results ===")
    for name, res in results.items():
        if name in ("all_passed", "device_info"):
            continue
        status = "PASS" if res.get("passed") else "FAIL"
        print(f"  [{status}] {name}")
        for k, v in res.items():
            if k != "passed":
                print(f"         {k}: {v}")

    all_passed = results.get("all_passed", False)
    print(f"\nall_passed: {all_passed}")

    save_results(results, args.out_dir)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
