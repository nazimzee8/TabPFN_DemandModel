"""
tests/test_market_aware_gradients.py

Gradient flow tests for MarketAwareDeepSetModel.
All tests use model.train() mode with dropout=0.0 for determinism.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import ModelConfig, MarketAwareDeepSetModel


def _make_cfg(**kwargs):
    defaults = dict(
        model_family="market_aware",
        d_sample=32,
        n_sab_sample_per_feature=0,
        sample_pool="attn",
        use_ridge_expert=False,
        ridge_lambda=1.0,
        residual_scale_init=0.1,
        gate_hidden_dim=32,
        n_sab_feat=1,
        n_heads=4,
        norm_feat=True,
        norm_target=True,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _make_model(**kwargs) -> MarketAwareDeepSetModel:
    cfg = _make_cfg(**kwargs)
    model = MarketAwareDeepSetModel(cfg=cfg)
    model.train()
    return model


def _sample_inputs(n=20, p=5, m=4, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(n, p),
        torch.randn(n),
        torch.randn(m, p),
    )


def _forward_loss(model, n=20, p=5, m=4):
    X_train, y_train, x_test = _sample_inputs(n=n, p=p, m=m)
    out = model(X_train, y_train, x_test)
    return out.sum()


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

def test_gradient_flows_through_phi_sample():
    """loss.backward() → phi_sample[0].weight.grad is not None."""
    model = _make_model()
    loss = _forward_loss(model)
    loss.backward()
    assert model.phi_sample[0].weight.grad is not None


def test_gradient_flows_through_sab_feat():
    """loss.backward() → sab_feat[0].mab.ffn[0].weight.grad is not None."""
    model = _make_model(n_sab_feat=1)
    loss = _forward_loss(model)
    loss.backward()
    assert model.sab_feat[0].mab.ffn[0].weight.grad is not None


def test_gradient_flows_through_beta_head():
    """loss.backward() → beta_head[0].weight.grad is not None."""
    model = _make_model()
    loss = _forward_loss(model)
    loss.backward()
    assert model.beta_head[0].weight.grad is not None


def test_gradient_flows_through_residual_head():
    """loss.backward() → residual_head[0].weight.grad is not None."""
    model = _make_model()
    loss = _forward_loss(model)
    loss.backward()
    assert model.residual_head[0].weight.grad is not None


def test_gradient_flows_through_gate_head():
    """loss.backward() → gate_head[0].weight.grad is not None (requires use_ridge_expert=True so gate enters the output path)."""
    model = _make_model(use_ridge_expert=True)
    loss = _forward_loss(model)
    loss.backward()
    assert model.gate_head[0].weight.grad is not None


def test_parameter_changes_after_optimizer_step():
    """After optimizer.step(), parameters are updated."""
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    # Save a snapshot
    before = model.phi_sample[0].weight.data.clone()
    loss = _forward_loss(model)
    loss.backward()
    opt.step()
    after = model.phi_sample[0].weight.data
    assert not torch.allclose(before, after), "Parameters did not change after optimizer step"


def test_row_permutation_invariance_train_mode():
    """In train mode (dropout=0), row permutation doesn't change output."""
    model = _make_model()
    n, p, m = 20, 5, 4
    torch.manual_seed(10)
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test = torch.randn(m, p)
    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        perm = torch.randperm(n)
        y_perm = model(X_train[perm], y_train[perm], x_test)
    delta = (y_orig - y_perm).abs().max().item()
    assert delta <= 1e-5, f"Row permutation delta too large: {delta}"


def test_feature_permutation_consistency_train_mode():
    """In train mode (dropout=0), feature permutation doesn't change output."""
    model = _make_model()
    n, p, m = 20, 5, 4
    torch.manual_seed(11)
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test = torch.randn(m, p)
    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        col_perm = torch.randperm(p)
        y_perm = model(X_train[:, col_perm], y_train, x_test[:, col_perm])
    delta = (y_orig - y_perm).abs().max().item()
    assert delta <= 1e-5, f"Feature permutation delta too large: {delta}"


def test_batch_query_output_shape():
    """(m, p) input → (m,) output."""
    model = _make_model()
    n, p, m = 20, 5, 8
    X_train, y_train, x_test = _sample_inputs(n=n, p=p, m=m)
    with torch.no_grad():
        out = model(X_train, y_train, x_test)
    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


def test_use_ridge_expert_true_gradients():
    """With use_ridge_expert=True, loss.backward() completes and neural path gets gradients."""
    model = _make_model(use_ridge_expert=True)
    loss = _forward_loss(model)
    loss.backward()
    # Gate head should still get gradients
    assert model.gate_head[0].weight.grad is not None
    # beta head should still get gradients (neural path)
    assert model.beta_head[0].weight.grad is not None


def test_no_nan_in_output():
    """Forward pass on random inputs produces finite output."""
    model = _make_model()
    loss = _forward_loss(model)
    assert torch.isfinite(loss), f"Forward pass produced non-finite output: {loss.item()}"
