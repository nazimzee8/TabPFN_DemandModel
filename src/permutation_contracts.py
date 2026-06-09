"""
permutation_contracts.py

Authoritative, task-aware permutation invariance and equivariance checks
for DeepSetICLModel (regression and classification) and
DeepSetCompletionModel (transductive completion).

Formal contracts
----------------
F1  Support-row permutation invariance (regression + classification)
F2  Query-row permutation equivariance (regression + classification)
F3  Feature-column permutation consistency (regression + classification)
F4  Feature-indexed output equivariance (beta for regression, W for classification)
F5  Classification label permutation equivariance
F6  Completion row equivariance
F7  Completion column equivariance

Each check returns a ``PermutationResult`` dataclass.  The ``run_all``
function dispatches the appropriate subset based on the model's
``cfg.model_family`` and ``cfg.task_objective``.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import time
from typing import Any

import numpy as np
import torch

from tolerance_policy import Tolerance, get_tolerance, TOLERANCE_POLICY_VERSION


# ─── result dataclass ──────────────────────────────────────────────────

@dataclasses.dataclass
class PermutationResult:
    check_type: str
    task_objective: str
    num_classes: int | None
    permutation_seed: int
    device: str
    dtype: str
    max_abs_delta: float
    mean_abs_delta: float
    max_rel_delta: float
    prediction_flip_rate: float        # classification only
    passed: bool
    threshold_atol: float
    threshold_rtol: float
    failure_reason: str
    reference_shape: tuple[int, ...]
    permuted_shape: tuple[int, ...]
    elapsed_s: float = 0.0
    tolerance_policy_version: str = TOLERANCE_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["reference_shape"] = list(d["reference_shape"])
        d["permuted_shape"] = list(d["permuted_shape"])
        return d


# ─── helpers ───────────────────────────────────────────────────────────

def _save_rng_state(device: torch.device):
    state = {
        "python_hash_seed": None,
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: dict, device: torch.device):
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _flip_rate(ref: torch.Tensor, perm: torch.Tensor) -> float:
    """Fraction of positions where integer predictions disagree."""
    if ref.numel() == 0:
        return 0.0
    return (ref != perm).float().mean().item()


def _resolve_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters(), torch.empty(0)).device


def _tol(
    dtype: torch.dtype,
    device: torch.device,
    output_type: str,
    model_path: str,
    inference: str = "deterministic",
) -> Tolerance:
    return get_tolerance(
        dtype=dtype,
        device_type=device.type,
        inference=inference,
        output_type=output_type,
        model_path=model_path,
    )


# ─── F1: support-row permutation invariance ────────────────────────────

def check_support_row_invariance(
    model,
    *,
    n: int = 20,
    p: int = 5,
    m: int = 4,
    seed: int = 42,
    num_classes: int | None = None,
    inference: str = "deterministic",
) -> PermutationResult:
    """f(X[P], y[P], Xq) ≈ f(X, y, Xq)."""
    device = _resolve_device(model)
    task_obj = getattr(model.cfg, "task_objective", "inductive_regression")
    is_cls = task_obj in {"inductive_classification", "linear_classification"}
    model_path = "classification" if is_cls else "regression"
    K = num_classes or (min(3, int(getattr(model.cfg, "max_num_classes", 3))) if is_cls else None)

    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X_train = torch.randn(n, p, device=device)
    if is_cls:
        y_train = (torch.arange(n, device=device) % K).long()
        x_test = torch.randn(m, p, device=device)
    else:
        y_train = torch.randn(n, device=device)
        x_test = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            if is_cls:
                ref = model(X_train, y_train, x_test,
                            task_objective=task_obj, num_classes=K)["logits"]
            else:
                ref = model(X_train, y_train, x_test)
                if ref.ndim == 0:
                    ref = ref.unsqueeze(0)

            _restore_rng_state(rng_state, device)
            pi = torch.randperm(n, device=device)
            if is_cls:
                perm_out = model(X_train[pi], y_train[pi], x_test,
                                 task_objective=task_obj, num_classes=K)["logits"]
            else:
                perm_out = model(X_train[pi], y_train[pi], x_test)
                if perm_out.ndim == 0:
                    perm_out = perm_out.unsqueeze(0)

        tol = _tol(ref.dtype, device, "logits" if is_cls else "scalar_prediction",
                    model_path, inference)
        mad = tol.max_abs_delta(ref, perm_out)
        mead = tol.mean_abs_delta(ref, perm_out)
        mrd = tol.max_rel_delta(ref, perm_out)
        passed = tol.allclose(ref, perm_out)
        flip = 0.0
        if is_cls:
            flip = _flip_rate(ref.argmax(-1), perm_out.argmax(-1))

        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F1_support_row_invariance",
        task_objective=task_obj,
        num_classes=K,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=flip,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(ref.shape),
        permuted_shape=tuple(perm_out.shape),
        elapsed_s=time.monotonic() - t0,
    )


# ─── F2: query-row permutation equivariance ────────────────────────────

def check_query_row_equivariance(
    model,
    *,
    n: int = 20,
    p: int = 5,
    m: int = 8,
    seed: int = 43,
    num_classes: int | None = None,
    inference: str = "deterministic",
) -> PermutationResult:
    """f(X, y, Xq[Q]) ≈ f(X, y, Xq)[Q]."""
    device = _resolve_device(model)
    task_obj = getattr(model.cfg, "task_objective", "inductive_regression")
    is_cls = task_obj in {"inductive_classification", "linear_classification"}
    model_path = "classification" if is_cls else "regression"
    K = num_classes or (min(3, int(getattr(model.cfg, "max_num_classes", 3))) if is_cls else None)

    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X_train = torch.randn(n, p, device=device)
    if is_cls:
        y_train = (torch.arange(n, device=device) % K).long()
    else:
        y_train = torch.randn(n, device=device)
    x_test = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            if is_cls:
                ref = model(X_train, y_train, x_test,
                            task_objective=task_obj, num_classes=K)["logits"]
            else:
                ref = model(X_train, y_train, x_test)
                if ref.ndim == 0:
                    ref = ref.unsqueeze(0)

            _restore_rng_state(rng_state, device)
            Q = torch.randperm(m, device=device)
            if is_cls:
                perm_out = model(X_train, y_train, x_test[Q],
                                 task_objective=task_obj, num_classes=K)["logits"]
            else:
                perm_out = model(X_train, y_train, x_test[Q])
                if perm_out.ndim == 0:
                    perm_out = perm_out.unsqueeze(0)

        expected = ref[Q]
        tol = _tol(ref.dtype, device, "logits" if is_cls else "scalar_prediction",
                    model_path, inference)
        mad = tol.max_abs_delta(expected, perm_out)
        mead = tol.mean_abs_delta(expected, perm_out)
        mrd = tol.max_rel_delta(expected, perm_out)
        passed = tol.allclose(expected, perm_out)
        flip = 0.0
        if is_cls:
            flip = _flip_rate(expected.argmax(-1), perm_out.argmax(-1))

        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F2_query_row_equivariance",
        task_objective=task_obj,
        num_classes=K,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=flip,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(ref.shape),
        permuted_shape=tuple(perm_out.shape),
        elapsed_s=time.monotonic() - t0,
    )


# ─── F3: feature-column permutation consistency ────────────────────────

def check_feature_column_consistency(
    model,
    *,
    n: int = 20,
    p: int = 5,
    m: int = 4,
    seed: int = 44,
    num_classes: int | None = None,
    inference: str = "deterministic",
) -> PermutationResult:
    """f(X[:, C], y, Xq[:, C]) ≈ f(X, y, Xq)."""
    device = _resolve_device(model)
    task_obj = getattr(model.cfg, "task_objective", "inductive_regression")
    is_cls = task_obj in {"inductive_classification", "linear_classification"}
    model_path = "classification" if is_cls else "regression"
    K = num_classes or (min(3, int(getattr(model.cfg, "max_num_classes", 3))) if is_cls else None)

    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X_train = torch.randn(n, p, device=device)
    if is_cls:
        y_train = (torch.arange(n, device=device) % K).long()
    else:
        y_train = torch.randn(n, device=device)
    x_test = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            if is_cls:
                ref = model(X_train, y_train, x_test,
                            task_objective=task_obj, num_classes=K)["logits"]
            else:
                ref = model(X_train, y_train, x_test)
                if ref.ndim == 0:
                    ref = ref.unsqueeze(0)

            _restore_rng_state(rng_state, device)
            C = torch.randperm(p, device=device)
            if is_cls:
                perm_out = model(X_train[:, C], y_train, x_test[:, C],
                                 task_objective=task_obj, num_classes=K)["logits"]
            else:
                perm_out = model(X_train[:, C], y_train, x_test[:, C])
                if perm_out.ndim == 0:
                    perm_out = perm_out.unsqueeze(0)

        tol = _tol(ref.dtype, device, "logits" if is_cls else "scalar_prediction",
                    model_path, inference)
        mad = tol.max_abs_delta(ref, perm_out)
        mead = tol.mean_abs_delta(ref, perm_out)
        mrd = tol.max_rel_delta(ref, perm_out)
        passed = tol.allclose(ref, perm_out)
        flip = 0.0
        if is_cls:
            flip = _flip_rate(ref.argmax(-1), perm_out.argmax(-1))

        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F3_feature_column_consistency",
        task_objective=task_obj,
        num_classes=K,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=flip,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(ref.shape),
        permuted_shape=tuple(perm_out.shape),
        elapsed_s=time.monotonic() - t0,
    )


# ─── F4: feature-indexed output equivariance ───────────────────────────

def check_feature_indexed_equivariance(
    model,
    *,
    n: int = 20,
    p: int = 5,
    m: int = 4,
    seed: int = 45,
    num_classes: int | None = None,
    inference: str = "deterministic",
) -> PermutationResult:
    """Regression: beta_hat[C] ≈ beta_hat_permuted.
    Classification: W_hat[C, :] ≈ W_hat_permuted.
    """
    device = _resolve_device(model)
    task_obj = getattr(model.cfg, "task_objective", "inductive_regression")
    is_cls = task_obj in {"inductive_classification", "linear_classification"}
    model_path = "classification" if is_cls else "regression"
    K = num_classes or (min(3, int(getattr(model.cfg, "max_num_classes", 3))) if is_cls else None)

    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X_train = torch.randn(n, p, device=device)
    if is_cls:
        y_train = (torch.arange(n, device=device) % K).long()
    else:
        y_train = torch.randn(n, device=device)
    x_test = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            if is_cls:
                ref_out = model(X_train, y_train, x_test,
                                task_objective=task_obj, num_classes=K)
                ref_coeff = ref_out["W_hat_norm"]  # (p, K)
            else:
                _, debug_ref = model(X_train, y_train, x_test, return_debug=True)
                ref_coeff = debug_ref.get("beta_hat_norm")
                if ref_coeff is None:
                    model.train(was_training)
                    return PermutationResult(
                        check_type="F4_feature_indexed_equivariance",
                        task_objective=task_obj,
                        num_classes=K,
                        permutation_seed=seed,
                        device=str(device),
                        dtype="n/a",
                        max_abs_delta=0.0,
                        mean_abs_delta=0.0,
                        max_rel_delta=0.0,
                        prediction_flip_rate=0.0,
                        passed=True,
                        threshold_atol=0.0,
                        threshold_rtol=0.0,
                        failure_reason="skipped: no coefficient head",
                        reference_shape=(),
                        permuted_shape=(),
                        elapsed_s=time.monotonic() - t0,
                    )

            _restore_rng_state(rng_state, device)
            C = torch.randperm(p, device=device)
            if is_cls:
                perm_out = model(X_train[:, C], y_train, x_test[:, C],
                                 task_objective=task_obj, num_classes=K)
                perm_coeff = perm_out["W_hat_norm"]  # (p, K)
            else:
                _, debug_perm = model(X_train[:, C], y_train, x_test[:, C],
                                      return_debug=True)
                perm_coeff = debug_perm.get("beta_hat_norm")

        # expected: ref_coeff[C] for regression (p,),  ref_coeff[C, :] for classification (p, K)
        if is_cls:
            expected = ref_coeff[C, :]
        else:
            expected = ref_coeff[C]

        tol = _tol(ref_coeff.dtype, device, "coefficients", model_path, inference)
        mad = tol.max_abs_delta(expected, perm_coeff)
        mead = tol.mean_abs_delta(expected, perm_coeff)
        mrd = tol.max_rel_delta(expected, perm_coeff)
        passed = tol.allclose(expected, perm_coeff)
        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F4_feature_indexed_equivariance",
        task_objective=task_obj,
        num_classes=K,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref_coeff.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=0.0,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(expected.shape),
        permuted_shape=tuple(perm_coeff.shape),
        elapsed_s=time.monotonic() - t0,
    )


# ─── F5: classification label permutation equivariance ─────────────────

def check_class_label_equivariance(
    model,
    *,
    n: int = 20,
    p: int = 5,
    m: int = 4,
    seed: int = 46,
    num_classes: int | None = None,
    inference: str = "deterministic",
) -> PermutationResult:
    """f(X, L(y), Xq)[:, L] ≈ f(X, y, Xq)  for class permutation L."""
    device = _resolve_device(model)
    task_obj = getattr(model.cfg, "task_objective", "inductive_regression")
    is_cls = task_obj in {"inductive_classification", "linear_classification"}
    if not is_cls:
        return PermutationResult(
            check_type="F5_class_label_equivariance",
            task_objective=task_obj,
            num_classes=None,
            permutation_seed=seed,
            device=str(device),
            dtype="n/a",
            max_abs_delta=0.0,
            mean_abs_delta=0.0,
            max_rel_delta=0.0,
            prediction_flip_rate=0.0,
            passed=True,
            threshold_atol=0.0,
            threshold_rtol=0.0,
            failure_reason="skipped: not a classification model",
            reference_shape=(),
            permuted_shape=(),
        )

    K = num_classes or min(3, int(getattr(model.cfg, "max_num_classes", 3)))
    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X_train = torch.randn(n, p, device=device)
    y_train = (torch.arange(n, device=device) % K).long()
    x_test = torch.randn(m, p, device=device)

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            ref_out = model(X_train, y_train, x_test,
                            task_objective=task_obj, num_classes=K)
            ref_logits = ref_out["logits"]   # (m, K)

            _restore_rng_state(rng_state, device)
            # Build a random class permutation L: L[old_class] = new_class
            rng_np = np.random.default_rng(seed)
            L = torch.tensor(rng_np.permutation(K), dtype=torch.long, device=device)
            y_perm = L[y_train]

            perm_out = model(X_train, y_perm, x_test,
                             task_objective=task_obj, num_classes=K)
            perm_logits = perm_out["logits"]  # (m, K)

        # Expected: ref_logits columns permuted by L
        # ref_logits[:, c] should become the column at position L[c] in perm_logits
        # So perm_logits[:, L[c]] ≈ ref_logits[:, c]
        # Equivalently: perm_logits[:, L] ≈ ref_logits
        expected = perm_logits[:, L]

        tol = _tol(ref_logits.dtype, device, "logits", "classification", inference)
        mad = tol.max_abs_delta(ref_logits, expected)
        mead = tol.mean_abs_delta(ref_logits, expected)
        mrd = tol.max_rel_delta(ref_logits, expected)
        passed = tol.allclose(ref_logits, expected)

        # prediction flip rate: compare argmax through inverse perm
        inv_L = torch.zeros(K, dtype=torch.long, device=device)
        inv_L[L] = torch.arange(K, device=device)
        ref_pred = ref_logits.argmax(-1)
        perm_pred_mapped = inv_L[perm_logits.argmax(-1)]
        flip = _flip_rate(ref_pred, perm_pred_mapped)

        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F5_class_label_equivariance",
        task_objective=task_obj,
        num_classes=K,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref_logits.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=flip,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(ref_logits.shape),
        permuted_shape=tuple(perm_logits.shape),
        elapsed_s=time.monotonic() - t0,
    )


# ─── F6/F7: completion row/column equivariance ────────────────────────

def check_completion_row_equivariance(
    model,
    *,
    rows: int = 10,
    cols: int = 6,
    seed: int = 47,
    inference: str = "deterministic",
) -> PermutationResult:
    """f(X[P], mask[P]) ≈ f(X, mask)[P]."""
    device = _resolve_device(model)
    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X = torch.randn(rows, cols, device=device)
    mask = torch.rand(rows, cols, device=device) > 0.3

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            ref = model(X, mask)

            _restore_rng_state(rng_state, device)
            pi = torch.randperm(rows, device=device)
            perm_out = model(X[pi], mask[pi])

        expected = ref[pi]
        tol = _tol(ref.dtype, device, "scalar_prediction", "regression", inference)
        mad = tol.max_abs_delta(expected, perm_out)
        mead = tol.mean_abs_delta(expected, perm_out)
        mrd = tol.max_rel_delta(expected, perm_out)
        passed = tol.allclose(expected, perm_out)
        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F6_completion_row_equivariance",
        task_objective="transductive_completion",
        num_classes=None,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=0.0,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(ref.shape),
        permuted_shape=tuple(perm_out.shape),
        elapsed_s=time.monotonic() - t0,
    )


def check_completion_column_equivariance(
    model,
    *,
    rows: int = 10,
    cols: int = 6,
    seed: int = 48,
    inference: str = "deterministic",
) -> PermutationResult:
    """f(X[:, C], mask[:, C]) ≈ f(X, mask)[:, C]."""
    device = _resolve_device(model)
    was_training = model.training
    model.eval()
    t0 = time.monotonic()

    torch.manual_seed(seed)
    X = torch.randn(rows, cols, device=device)
    mask = torch.rand(rows, cols, device=device) > 0.3

    try:
        with torch.no_grad():
            rng_state = _save_rng_state(device)
            ref = model(X, mask)

            _restore_rng_state(rng_state, device)
            C = torch.randperm(cols, device=device)
            perm_out = model(X[:, C], mask[:, C])

        expected = ref[:, C]
        tol = _tol(ref.dtype, device, "scalar_prediction", "regression", inference)
        mad = tol.max_abs_delta(expected, perm_out)
        mead = tol.mean_abs_delta(expected, perm_out)
        mrd = tol.max_rel_delta(expected, perm_out)
        passed = tol.allclose(expected, perm_out)
        reason = "" if passed else f"max_abs_delta={mad:.2e} > atol={tol.atol:.2e}"
    finally:
        model.train(was_training)

    return PermutationResult(
        check_type="F7_completion_column_equivariance",
        task_objective="transductive_completion",
        num_classes=None,
        permutation_seed=seed,
        device=str(device),
        dtype=str(ref.dtype),
        max_abs_delta=mad,
        mean_abs_delta=mead,
        max_rel_delta=mrd,
        prediction_flip_rate=0.0,
        passed=passed,
        threshold_atol=tol.atol,
        threshold_rtol=tol.rtol,
        failure_reason=reason,
        reference_shape=tuple(ref.shape),
        permuted_shape=tuple(perm_out.shape),
        elapsed_s=time.monotonic() - t0,
    )


# ─── dispatch: run_all ─────────────────────────────────────────────────

def run_all(model, *, seed: int = 42) -> list[PermutationResult]:
    """Run the full set of applicable permutation checks for `model`.

    Dispatches based on ``model.cfg.model_family`` and
    ``model.cfg.task_objective``.  Returns a list of ``PermutationResult``.
    """
    cfg = model.cfg
    family = getattr(cfg, "model_family", "market_exchangeable_icl")
    task_obj = getattr(cfg, "task_objective", "inductive_regression")
    is_cls = task_obj in {"inductive_classification", "linear_classification"}

    results: list[PermutationResult] = []

    if family == "market_exchangeable_completion":
        results.append(check_completion_row_equivariance(model, seed=seed))
        results.append(check_completion_column_equivariance(model, seed=seed + 1))
        return results

    if family != "market_exchangeable_icl":
        raise ValueError(
            f"Unknown model_family: {family!r}. "
            "Only 'market_exchangeable_icl' and 'market_exchangeable_completion' "
            "are supported."
        )

    # ICL model — regression or classification
    results.append(check_support_row_invariance(model, seed=seed))
    results.append(check_query_row_equivariance(model, seed=seed + 1))
    results.append(check_feature_column_consistency(model, seed=seed + 2))
    results.append(check_feature_indexed_equivariance(model, seed=seed + 3))

    if is_cls:
        # Run class-label equivariance for multiple K values
        max_K = int(getattr(cfg, "max_num_classes", 10))
        for K in [2, 3, 5, 10]:
            if K <= max_K:
                results.append(check_class_label_equivariance(
                    model, seed=seed + 4 + K, num_classes=K,
                ))

    return results


def run_all_strict(model, *, seed: int = 42) -> list[PermutationResult]:
    """Run all checks and raise ``RuntimeError`` if any fail."""
    results = run_all(model, seed=seed)
    failures = [r for r in results if not r.passed and "skipped" not in r.failure_reason]
    if failures:
        msgs = [f"  {r.check_type}: {r.failure_reason}" for r in failures]
        raise RuntimeError(
            f"Permutation contract violations ({len(failures)}):\n" + "\n".join(msgs)
        )
    return results
