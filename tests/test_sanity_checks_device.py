"""
tests/test_sanity_checks_device.py

Tests for Phase 3: device-aware sanity checks.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import sanity_checks
from sanity_checks import _resolve_device, run_all_checks


def test_resolve_device_auto_cpu():
    """When CUDA is unavailable, _resolve_device('auto') returns cpu."""
    with patch("torch.cuda.is_available", return_value=False):
        d = _resolve_device("auto")
    assert d == torch.device("cpu")


def test_resolve_device_explicit_cpu():
    """_resolve_device('cpu') always returns cpu."""
    d = _resolve_device("cpu")
    assert d == torch.device("cpu")


def test_resolve_device_explicit_cuda():
    """_resolve_device('cuda') returns cuda device."""
    d = _resolve_device("cuda")
    assert d.type == "cuda"


def test_cpu_sanity_checks_pass():
    """run_all_checks(device=torch.device('cpu')) passes all checks."""
    results = run_all_checks(device=torch.device("cpu"))
    assert results.get("all_passed") is True, f"Checks failed: {results}"


def test_device_info_in_results():
    """run_all_checks() result includes device_info with device key."""
    results = run_all_checks(device=torch.device("cpu"))
    assert "device_info" in results
    assert results["device_info"]["device"] == "cpu"


def test_device_info_has_cuda_available():
    """device_info includes cuda_available key."""
    results = run_all_checks(device=torch.device("cpu"))
    assert "cuda_available" in results["device_info"]


def test_device_info_no_cuda_device_name_when_cpu():
    """When using CPU device, cuda_device_name is None."""
    results = run_all_checks(device=torch.device("cpu"))
    assert results["device_info"]["cuda_device_name"] is None


def test_cuda_sanity_checks_pass_or_skip():
    """If CUDA is available, run checks on CUDA; otherwise skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    results = run_all_checks(device=torch.device("cuda"))
    assert results.get("all_passed") is True, f"CUDA checks failed: {results}"


def test_run_all_checks_with_explicit_model_on_cpu():
    """Passing a model explicitly to run_all_checks works on CPU."""
    from model import ModelConfig, DeepSetICLModel
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=64, d_rho=128, pool="pna",
        n_heads=4, n_sab_feat=1,
        use_ridge_expert=True, ridge_lambda=1.0,
        gate_hidden_dim=32,
        norm_feat=True, norm_target=True, dropout=0.0,
    )
    model = DeepSetICLModel(cfg=cfg)
    results = run_all_checks(model=model, device=torch.device("cpu"))
    assert results.get("all_passed") is True
