"""
tests/test_checkpoint_gates.py

Tests for Phase 6: checkpoint quality gates.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sanity_checks import (
    run_checkpoint_gates,
    check_nan_inf_output,
    check_constant_output_collapse,
    check_query_sensitivity,
    check_simple_linear_recovery_smoke,
    check_ratio_to_fixed_ridge,
    check_train_val_gap,
)
from model import ModelConfig, DeepSetICLModel


def _make_healthy_model():
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=64, d_rho=128, pool="pna",
        n_heads=4, n_sab_feat=1,
        use_ridge_expert=False,
        ridge_lambda=1.0,
        gate_hidden_dim=32,
        norm_feat=True,
        norm_target=True,
        dropout=0.0,
    )
    model = DeepSetICLModel(cfg=cfg)
    model.eval()
    return model


class ConstantModel(nn.Module):
    """Model that always returns a fixed constant for collapse gate testing."""
    def __init__(self, const=5.0):
        super().__init__()
        self.const = const

    def forward(self, X_train, y_train, x_test):
        m = x_test.shape[0] if x_test.ndim == 2 else 1
        return torch.full((m,), self.const, device=x_test.device)


class NanModel(nn.Module):
    """Model that always returns NaN."""
    def forward(self, X_train, y_train, x_test):
        m = x_test.shape[0] if x_test.ndim == 2 else 1
        return torch.full((m,), float("nan"), device=x_test.device)


class ZeroModel(nn.Module):
    """Model that always returns zero — will fail ridge ratio gate."""
    def forward(self, X_train, y_train, x_test):
        m = x_test.shape[0] if x_test.ndim == 2 else 1
        return torch.zeros(m, device=x_test.device)


# ---------------------------------------------------------------------------
# Individual gate tests
# ---------------------------------------------------------------------------

def test_healthy_model_passes_nan_gate():
    """Healthy model: check_nan_inf_output passes."""
    model = _make_healthy_model()
    result = check_nan_inf_output(model, torch.device("cpu"))
    assert result["passed"] is True


def test_nan_output_model_fails_gate_1():
    """NaN model: check_nan_inf_output fails."""
    model = NanModel()
    result = check_nan_inf_output(model, torch.device("cpu"))
    assert result["passed"] is False
    assert result.get("has_nan") is True


def test_constant_output_model_fails_gate_2():
    """Constant model: check_constant_output_collapse fails."""
    model = ConstantModel()
    result = check_constant_output_collapse(model, torch.device("cpu"), min_std=1e-3)
    assert result["passed"] is False


def test_healthy_model_passes_gate_2():
    """Healthy model passes collapse gate with loose threshold."""
    model = _make_healthy_model()
    result = check_constant_output_collapse(model, torch.device("cpu"), min_std=1e-6)
    assert result["passed"] is True


def test_severe_ridge_underperformance_fails_gate_5():
    """Zero model: model_MSE/ridge_MSE >> 10 → gate 5 fails."""
    model = ZeroModel()
    result = check_ratio_to_fixed_ridge(model, torch.device("cpu"), max_ratio=10.0)
    assert result["passed"] is False


def test_train_val_gap_gate_passes_when_no_metadata():
    """check_train_val_gap(None) passes trivially."""
    result = check_train_val_gap(None)
    assert result["passed"] is True


def test_train_val_gap_gate_passes_when_fields_missing():
    """Metadata without val_mse/train_mse → passes."""
    result = check_train_val_gap({"model_family": "market_aware"})
    assert result["passed"] is True


def test_train_val_gap_gate_fails_on_severe_overfit():
    """Extreme val/train ratio (> 10) → gate 6 fails."""
    result = check_train_val_gap({"val_mse": 100.0, "train_mse": 1.0})
    assert result["passed"] is False


def test_train_val_gap_gate_passes_on_good_ratio():
    """Normal val/train ratio → gate 6 passes."""
    result = check_train_val_gap({"val_mse": 1.5, "train_mse": 1.0})
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# run_checkpoint_gates orchestrator tests
# ---------------------------------------------------------------------------

def test_healthy_model_passes_all_gates():
    """DeepSetICLModel with random weights passes all gates (loose thresholds).

    A freshly-initialized model has random weights and cannot be expected to
    rival ridge regression on a linear DGP.  The ratio gate is therefore set
    very loosely (1e6) so the test validates structural health (no NaN/Inf,
    non-constant output, finite MSE) without requiring a trained model.
    """
    model = _make_healthy_model()
    results = run_checkpoint_gates(
        model=model,
        device=torch.device("cpu"),
        max_ridge_ratio=1e6,
        min_query_std=1e-6,
    )
    assert results.get("all_passed") is True, f"Gates failed: {results}"


def test_nan_model_fails_run_checkpoint_gates():
    """NaN model fails run_checkpoint_gates."""
    model = NanModel()
    results = run_checkpoint_gates(model=model, device=torch.device("cpu"))
    assert results.get("all_passed") is False


def test_gate_results_are_json_serializable():
    """run_checkpoint_gates result can be JSON serialized."""
    model = _make_healthy_model()
    results = run_checkpoint_gates(
        model=model,
        device=torch.device("cpu"),
        max_ridge_ratio=1e6,
        min_query_std=1e-6,
    )
    # Should not raise
    json.dumps(results)


def test_strict_false_continues():
    """Caller with strict=False on failed gates does not raise (behavior test)."""
    model = NanModel()
    results = run_checkpoint_gates(model=model, device=torch.device("cpu"))
    # Simulate strict=False behavior: just check and continue
    failed = [k for k, v in results.items()
              if isinstance(v, dict) and not v.get("passed", True)]
    # No exception raised; we just have failures in the dict
    assert len(failed) > 0
