"""
tests/test_checkpoint_validation.py

Tests for Phase 5: validate_checkpoint_payload and load_best_deepset_checkpoint.
"""
from __future__ import annotations

import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Patch heavy imports before importing evaluate_synthetic_regression
_snowflake_mock = MagicMock()
sys.modules.setdefault("snowflake", _snowflake_mock)
sys.modules.setdefault("snowflake.snowpark", _snowflake_mock.snowpark)
sys.modules.setdefault("snowflake.snowpark.Session", _snowflake_mock.snowpark.Session)
sys.modules.setdefault("matplotlib", MagicMock())
sys.modules.setdefault("matplotlib.pyplot", MagicMock())
_pyarrow_mock = MagicMock()
_pyarrow_mock.__version__ = "0.0.0"
sys.modules.setdefault("pyarrow", _pyarrow_mock)
sys.modules.setdefault("pyarrow.parquet", MagicMock())
# pandas is installed in the test environment — no mock needed.
sys.modules.setdefault("autogluon_models", MagicMock())
sys.modules.setdefault("baseline_models", MagicMock())
# deepset_inference is NOT mocked — it only uses numpy/torch/sklearn (no heavy deps).

import evaluate_synthetic_regression as esr
from evaluate_synthetic_regression import validate_checkpoint_payload, SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH


def _make_valid_payload(
    model_family="market_aware",
    task_type="regression",
    fmt=3,
    include_metadata=True,
    meta_family=None,
):
    """Build a valid checkpoint payload dict."""
    meta = {}
    if include_metadata:
        meta["model_family"] = meta_family if meta_family is not None else model_family
        if task_type is not None:
            meta["task_type"] = task_type
    return {
        "checkpoint_format_version": fmt,
        "cfg": {"model_family": model_family, "d_phi": 128, "d_rho": 256, "pool": "pna",
                "n_heads": 4, "n_sab_feat": 1, "n_sab_samp": 1,
                "norm_feat": True, "norm_target": True, "dropout": 0.1},
        "state_dict": {"some_weight": torch.zeros(2)},
        "metadata": meta if include_metadata else None,
    }


# ---------------------------------------------------------------------------
# validate_checkpoint_payload tests
# ---------------------------------------------------------------------------

def test_valid_market_aware_payload_passes():
    """Full valid market_aware payload with task_type=regression, fmt=3 → no exception."""
    payload = _make_valid_payload(model_family="market_aware", task_type="regression", fmt=3)
    validate_checkpoint_payload(payload, "/tmp/test.pt")  # no exception


def test_valid_deepset_payload_passes():
    """Full valid deepset payload, fmt=2 → no exception."""
    payload = _make_valid_payload(model_family="deepset", task_type="regression", fmt=2)
    validate_checkpoint_payload(payload, "/tmp/test.pt")  # no exception


def test_missing_cfg_fails():
    """Payload without 'cfg' key → RuntimeError."""
    payload = {"state_dict": {"w": torch.zeros(2)}}
    with pytest.raises(RuntimeError, match="missing required 'cfg'"):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


def test_missing_state_dict_fails():
    """Payload without 'state_dict' key → RuntimeError."""
    payload = {"cfg": {"model_family": "deepset"}}
    with pytest.raises(RuntimeError, match="missing required 'state_dict'"):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


def test_non_regression_task_type_fails():
    """task_type='classification' → RuntimeError."""
    payload = _make_valid_payload(task_type="classification")
    with pytest.raises(RuntimeError, match="task_type="):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


def test_missing_task_type_warns():
    """Payload with no metadata.task_type → UserWarning (not exception)."""
    payload = _make_valid_payload(task_type=None)  # task_type missing from meta
    with pytest.warns(UserWarning, match="missing metadata.task_type"):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


def test_model_family_mismatch_fails():
    """cfg.model_family='deepset' but metadata.model_family='market_aware' → RuntimeError."""
    payload = _make_valid_payload(model_family="deepset", meta_family="market_aware")
    with pytest.raises(RuntimeError, match="disagrees with"):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


def test_market_aware_v2_warns():
    """market_aware model with fmt=2 → UserWarning about checkpoint_format_version."""
    payload = _make_valid_payload(model_family="market_aware", task_type="regression", fmt=2)
    with pytest.warns(UserWarning, match="checkpoint_format_version"):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


def test_no_metadata_warns():
    """Payload with no metadata → UserWarning."""
    payload = _make_valid_payload(include_metadata=False)
    payload["metadata"] = None
    with pytest.warns(UserWarning, match="no 'metadata'"):
        validate_checkpoint_payload(payload, "/tmp/test.pt")


# ---------------------------------------------------------------------------
# SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH constant test
# ---------------------------------------------------------------------------

def test_synreg_deepset_checkpoint_stage_path_constant_exists():
    """SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH constant exists."""
    assert hasattr(esr, "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH")


def test_synreg_deepset_checkpoint_stage_path_default():
    """Default SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH falls back to CHECKPOINT_STAGE_PATH."""
    # If env var not set, should match CHECKPOINT_STAGE_PATH
    import importlib
    import os
    env_val = os.environ.get("SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH")
    if env_val is None:
        assert esr.SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH == esr.CHECKPOINT_STAGE_PATH


def test_load_best_uses_synreg_env_var(monkeypatch):
    """load_best_deepset_checkpoint uses SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH."""
    import importlib

    custom_path = "@stage/custom.pt"
    monkeypatch.setenv("SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH", custom_path)
    importlib.reload(esr)

    # Verify the constant was updated
    assert esr.SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH == custom_path

    importlib.reload(esr)


# ---------------------------------------------------------------------------
# Fix 5: MODEL3 checkpoint validation hardening
# ---------------------------------------------------------------------------

def _make_model3_payload(
    model_family="market_exchangeable_icl",
    fmt=4,
    cfg_arch="model3",
    meta_arch="model3",
    cfg_pattern="inductive_forecasting",
    meta_pattern="inductive_forecasting",
    task_type="regression",
    task_objective="inductive_regression",
    meta_family=None,
):
    """Build a valid MODEL3 checkpoint payload."""
    meta = {
        "model_family": meta_family if meta_family is not None else model_family,
        "model_arch_version": meta_arch,
        "model3_design_pattern": meta_pattern,
        "task_type": task_type,
        "task_objective": task_objective,
    }
    return {
        "checkpoint_format_version": fmt,
        "cfg": {
            "model_family": model_family,
            "model_arch_version": cfg_arch,
            "model3_design_pattern": cfg_pattern,
            "d_phi": 64,
            "d_rho": 128,
            "pool": "pna",
            "n_heads": 4,
            "n_sab_feat": 2,
            "n_sab_samp": 1,
            "norm_feat": True,
            "norm_target": True,
            "dropout": 0.0,
        },
        "state_dict": {"weight": torch.zeros(2)},
        "metadata": meta,
    }


def test_model3_icl_valid_payload_passes():
    """Valid MODEL3 ICL payload passes validation."""
    payload = _make_model3_payload()
    validate_checkpoint_payload(payload, "/fake/model3.pt")  # no exception


def test_completion_family_rejected_in_synreg_eval():
    """market_exchangeable_completion must be rejected — it's transductive."""
    payload = _make_model3_payload(
        model_family="market_exchangeable_completion",
        meta_pattern="transductive_completion",
        cfg_pattern="transductive_completion",
        task_objective="transductive_completion",
    )
    with pytest.raises(RuntimeError, match="(?i)transductive"):
        validate_checkpoint_payload(payload, "/fake/completion.pt")


def test_model3_wrong_format_version_fails():
    """MODEL3 family with fmt=3 (not 4) → RuntimeError."""
    payload = _make_model3_payload(fmt=3)
    with pytest.raises(RuntimeError, match="checkpoint_format_version"):
        validate_checkpoint_payload(payload, "/fake/model3_fmt3.pt")


def test_model3_meta_arch_version_wrong_fails():
    """MODEL3 family but metadata.model_arch_version='model2' → RuntimeError."""
    payload = _make_model3_payload(meta_arch="model2")
    with pytest.raises(RuntimeError, match="model_arch_version"):
        validate_checkpoint_payload(payload, "/fake/bad_arch.pt")


def test_model3_cfg_arch_version_wrong_fails():
    """MODEL3 family but cfg.model_arch_version='model2' → RuntimeError."""
    payload = _make_model3_payload(cfg_arch="model2")
    with pytest.raises(RuntimeError, match="model_arch_version"):
        validate_checkpoint_payload(payload, "/fake/bad_cfg_arch.pt")


def test_model3_design_pattern_mismatch_fails():
    """metadata.model3_design_pattern disagrees with cfg.model3_design_pattern → RuntimeError."""
    payload = _make_model3_payload(
        cfg_pattern="inductive_forecasting",
        meta_pattern="transductive_completion",  # mismatch
    )
    with pytest.raises(RuntimeError, match="model3_design_pattern"):
        validate_checkpoint_payload(payload, "/fake/pattern_mismatch.pt")


def test_model3_wrong_task_objective_fails():
    """task_objective='transductive_completion' for MODEL3 ICL → RuntimeError."""
    payload = _make_model3_payload(task_objective="transductive_completion")
    with pytest.raises(RuntimeError, match="task_objective"):
        validate_checkpoint_payload(payload, "/fake/wrong_objective.pt")
