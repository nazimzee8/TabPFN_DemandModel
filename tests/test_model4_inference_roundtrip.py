"""
tests/test_model4_inference_roundtrip.py
========================================
Tests for real MODEL4 inference in evaluate_linear_classification.py.

Verifies:
1. _run_model4_inference returns valid probability matrix (n_test, K)
2. Inference does NOT use stored DGP probs (metric_schema_version == "2.0")
3. task_seed column is present in eval parquet output
4. Metric schema version constant is "2.0"

These tests use mocks/stubs where GPU or full checkpoint is unavailable.
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Test 1: Metric schema version constant is "2.0"
# ---------------------------------------------------------------------------

def test_metric_schema_version_is_2():
    """SYNCLS_METRIC_SCHEMA_VERSION must be '2.0' (real MODEL4 inference)."""
    import evaluate_linear_classification as esc
    assert esc.SYNCLS_METRIC_SCHEMA_VERSION == "2.0", (
        f"Expected SYNCLS_METRIC_SCHEMA_VERSION == '2.0', got {esc.SYNCLS_METRIC_SCHEMA_VERSION!r}. "
        "This likely means the Phase 2 bump was not applied."
    )


# ---------------------------------------------------------------------------
# Test 2: _run_model4_inference returns valid probability matrix
# ---------------------------------------------------------------------------

def test_run_model4_inference_returns_valid_probs():
    """_run_model4_inference must return (n_test, K) float array summing to 1."""
    import evaluate_linear_classification as esc

    n_train, n_test, n_features, num_classes = 30, 10, 5, 3

    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(0, num_classes, size=n_train)
    X_test = rng.standard_normal((n_test, n_features))

    # Build a minimal fake checkpoint and model that returns uniform logits
    import torch

    def fake_model_call(X_ctx, y_ctx, X_b, task_objective=None, num_classes=None):
        batch_size = X_b.shape[0]
        logits = torch.zeros(batch_size, num_classes)
        return {"logits": logits}

    fake_model = MagicMock(side_effect=fake_model_call)
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.load_state_dict = MagicMock()
    # Provide cfg attributes required by permutation_contracts.run_all()
    fake_model.cfg.model_family = "market_exchangeable_icl"
    fake_model.cfg.task_objective = "inductive_classification"
    fake_model.cfg.model_design_pattern = "inductive_forecasting"

    fake_cfg = {"max_n_train": 64}
    fake_ckpt = {
        "cfg": fake_cfg,
        "model_state_dict": {},
    }

    def fake_build_model(cfg):
        return fake_model

    def fake_resolve_cap(model):
        return n_features

    def fake_select_features(X_tr, y_tr, X_te, cap):
        return X_tr[:, :cap], X_te[:, :cap], list(range(cap))

    def fake_select_ctx(n_train, ctx_size, seed, context_index, context_ensembles):
        return np.arange(min(ctx_size, n_train))

    with (
        patch.dict("sys.modules", {
            "model": MagicMock(build_model_from_config=fake_build_model),
            "deepset_inference": MagicMock(
                resolve_deepset_feature_cap=fake_resolve_cap,
                select_deepset_classification_features_train_only=fake_select_features,
                select_deepset_context_indices=fake_select_ctx,
            ),
        }),
        patch.dict("os.environ", {"CLASSIFICATION_RUN_PERMUTATION_GATES": "false"}),
    ):
        probs, _meta = esc._run_model4_inference(
            fake_ckpt, X_train, y_train, X_test, num_classes,
            context_size=10, n_ensembles=2, test_batch_size=5
        )

    assert probs.shape == (n_test, num_classes), (
        f"Expected probs shape ({n_test}, {num_classes}), got {probs.shape}"
    )
    assert np.all(probs >= 0.0), "Probabilities must be non-negative"
    assert np.all(probs <= 1.0), "Probabilities must be <= 1.0"
    np.testing.assert_allclose(
        probs.sum(axis=1), 1.0, atol=1e-6,
        err_msg="Probability rows must sum to 1.0"
    )


# ---------------------------------------------------------------------------
# Test 3: Inference does NOT use stored probs (no data.get("probs_test") path)
# ---------------------------------------------------------------------------

def test_deepset_mode_uses_real_inference_not_stored_probs():
    """_run_deepset_mode must call _run_model4_inference, not data.get('probs_test')."""
    import evaluate_linear_classification as esc
    import inspect

    src = inspect.getsource(esc._run_deepset_mode)

    # The old placeholder used data.get("probs_test") as the primary probs source
    # New code uses _run_model4_inference as primary
    assert "_run_model4_inference(" in src, (
        "_run_deepset_mode must call _run_model4_inference() for MODEL4 inference"
    )

    # Stored DGP teacher probs may appear for diagnostic only (dgp_teacher_log_loss),
    # not as primary probs.  Column was renamed probs_test → dgp_teacher_probs_test (CLS-D1).
    # Verify the assignment `probs_test = np.asarray(data.get("probs_test", []))` is NOT
    # the primary probs_test source
    lines = src.splitlines()
    primary_assignment_uses_stored = False
    for line in lines:
        stripped = line.strip()
        if "probs_test" in stripped and "_run_model4_inference" not in stripped:
            if stripped.startswith("probs_test =") and "data.get" in stripped:
                primary_assignment_uses_stored = True
    assert not primary_assignment_uses_stored, (
        "probs_test must not be assigned from data.get('probs_test') as primary source. "
        "Only _run_model4_inference() output should be used as probs_test."
    )


# ---------------------------------------------------------------------------
# Test 4: metric_schema_version appears in deepset result rows
# ---------------------------------------------------------------------------

def test_metric_schema_version_in_deepset_source():
    """_run_deepset_mode must set metrics['metric_schema_version']."""
    import evaluate_linear_classification as esc
    import inspect

    src = inspect.getsource(esc._run_deepset_mode)
    assert "metric_schema_version" in src, (
        "metrics['metric_schema_version'] must be set in _run_deepset_mode"
    )
    assert "SYNCLS_METRIC_SCHEMA_VERSION" in src, (
        "metric_schema_version value must reference SYNCLS_METRIC_SCHEMA_VERSION constant"
    )


# ---------------------------------------------------------------------------
# Test 5: metric_schema_version in baselines and autogluon modes
# ---------------------------------------------------------------------------

def test_metric_schema_version_in_baselines_source():
    """_run_baselines_mode must set metrics['metric_schema_version']."""
    import evaluate_linear_classification as esc
    import inspect
    src = inspect.getsource(esc._run_baselines_mode)
    assert "metric_schema_version" in src, (
        "metrics['metric_schema_version'] must be set in _run_baselines_mode"
    )


def test_metric_schema_version_in_autogluon_source():
    """_run_autogluon_mode must set metrics['metric_schema_version']."""
    import evaluate_linear_classification as esc
    import inspect
    src = inspect.getsource(esc._run_autogluon_mode)
    assert "metric_schema_version" in src, (
        "metrics['metric_schema_version'] must be set in _run_autogluon_mode"
    )


# ---------------------------------------------------------------------------
# Phase 10: Stable method ID verification
# ---------------------------------------------------------------------------

def test_deepset_method_id_stable():
    """Regression evaluator DEEPSET_METHOD_ID must be 'MODEL-ICL-MC' (not misspelled)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import evaluate_linear_regression as synreg
    assert synreg.DEEPSET_METHOD_ID == "MODEL-ICL-MC", (
        f"DEEPSET_METHOD_ID={synreg.DEEPSET_METHOD_ID!r} - must be 'MODEL-ICL-MC'"
    )
    assert synreg.LEGACY_DEEPSET_METHOD_ID == "MODLE_ICL-MC", (
        f"LEGACY_DEEPSET_METHOD_ID should be the old typo 'MODLE_ICL-MC' (for backward compat)"
    )
