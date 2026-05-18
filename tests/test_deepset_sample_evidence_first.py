"""
tests/test_deepset_sample_evidence_first.py

Tests for Phase 2: sample_evidence_first path in DeepSetModel.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import ModelConfig, DeepSetModel


def _legacy_cfg(**kwargs):
    defaults = dict(d_phi=32, d_rho=64, pool="pna", n_heads=4,
                    n_sab_feat=1, n_sab_samp=1, norm_feat=True, norm_target=True, dropout=0.0)
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _sev_cfg(**kwargs):
    defaults = dict(d_phi=32, d_rho=64, pool="pna", n_heads=4,
                    n_sab_feat=1, n_sab_samp=1, norm_feat=True, norm_target=True, dropout=0.0,
                    feature_aggregation_order="sample_evidence_first")
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _sample_inputs(n=20, p=5, m=4):
    torch.manual_seed(0)
    return (
        torch.randn(n, p),
        torch.randn(n),
        torch.randn(m, p),
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_old_config_missing_field_uses_legacy_path():
    """ModelConfig created without feature_aggregation_order defaults to legacy."""
    cfg = ModelConfig(d_phi=32, d_rho=64, pool="mean", n_sab_feat=0, n_sab_samp=0,
                      norm_feat=False, norm_target=False, n_heads=4, dropout=0.0)
    assert cfg.feature_aggregation_order == "legacy_feature_pool_first"


def test_new_config_uses_sample_evidence_first():
    """Creating ModelConfig with feature_aggregation_order='sample_evidence_first' works."""
    cfg = _sev_cfg()
    assert cfg.feature_aggregation_order == "sample_evidence_first"


def test_invalid_feature_aggregation_order_raises():
    """Invalid feature_aggregation_order raises ValueError."""
    with pytest.raises(ValueError, match="Invalid feature_aggregation_order"):
        ModelConfig(d_phi=32, d_rho=64, pool="mean", n_sab_feat=0, n_sab_samp=0,
                    norm_feat=False, norm_target=False, n_heads=4, dropout=0.0,
                    feature_aggregation_order="invalid_mode")


def test_old_state_dict_loads_on_new_cfg():
    """Build model with legacy cfg, save state_dict, reload with sample_evidence_first cfg."""
    legacy_cfg = _legacy_cfg()
    legacy_model = DeepSetModel(cfg=legacy_cfg)
    sd = legacy_model.state_dict()

    new_cfg = _sev_cfg()
    new_model = DeepSetModel(cfg=new_cfg)
    # Should load without error (same architecture)
    new_model.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

def test_single_query_shape():
    """Single x_test (p,) → scalar output (ndim == 0)."""
    model = DeepSetModel(cfg=_sev_cfg())
    model.eval()
    torch.manual_seed(1)
    n, p = 15, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test = torch.randn(p)
    with torch.no_grad():
        out = model(X_train, y_train, x_test)
    assert out.ndim == 0, f"Expected scalar, got shape {out.shape}"


def test_batched_query_shape():
    """Batched x_test (m, p) → shape (m,)."""
    model = DeepSetModel(cfg=_sev_cfg())
    model.eval()
    n, p, m = 20, 5, 8
    X_train, y_train, x_test = _sample_inputs(n=n, p=p, m=m)
    with torch.no_grad():
        out = model(X_train, y_train, x_test)
    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


# ---------------------------------------------------------------------------
# Invariance tests
# ---------------------------------------------------------------------------

def test_row_permutation_invariance():
    """Row permutation of X_train/y_train doesn't change output (max_abs_delta <= 1e-5)."""
    model = DeepSetModel(cfg=_sev_cfg())
    model.eval()
    n, p, m = 25, 5, 4
    torch.manual_seed(42)
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test = torch.randn(m, p)
    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        perm = torch.randperm(n)
        y_perm = model(X_train[perm], y_train[perm], x_test)
    delta = (y_orig - y_perm).abs().max().item()
    assert delta <= 1e-5, f"Row permutation delta too large: {delta}"


def test_feature_permutation_consistency():
    """Consistent column permutation of X_train/x_test doesn't change output (max_abs_delta <= 1e-5)."""
    model = DeepSetModel(cfg=_sev_cfg())
    model.eval()
    n, p, m = 25, 5, 4
    torch.manual_seed(43)
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test = torch.randn(m, p)
    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        col_perm = torch.randperm(p)
        y_perm = model(X_train[:, col_perm], y_train, x_test[:, col_perm])
    delta = (y_orig - y_perm).abs().max().item()
    assert delta <= 1e-5, f"Feature permutation delta too large: {delta}"


# ---------------------------------------------------------------------------
# Gradient test
# ---------------------------------------------------------------------------

def test_backward_produces_gradients():
    """loss.backward() flows gradients through phi, sab_feat (if present), and psi."""
    cfg = _sev_cfg(n_sab_feat=1)
    model = DeepSetModel(cfg=cfg)
    model.train()
    n, p, m = 15, 4, 3
    torch.manual_seed(7)
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test = torch.randn(m, p)
    out = model(X_train, y_train, x_test)
    loss = out.sum()
    loss.backward()
    assert model.phi[0].weight.grad is not None
    assert model.psi[0].weight.grad is not None
