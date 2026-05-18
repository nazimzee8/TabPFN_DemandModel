"""
tests/test_market_aware_deepset_shapes.py

Forward shape tests and model routing for MarketAwareDeepSetModel.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import (  # noqa: E402
    DeepSetModel,
    MarketAwareDeepSetModel,
    ModelConfig,
    _instantiate_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_forward_shape_single_query():
    """x_test shape (p,) → scalar output."""
    model = _make_model()
    torch.manual_seed(0)
    n, p = 20, 6
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(p)   # single query

    with torch.no_grad():
        out = model(X_train, y_train, x_test)

    assert out.ndim == 0, f"Expected scalar (ndim=0), got shape {out.shape}"


def test_forward_shape_batch_query():
    """x_test shape (m, p) → shape (m,)."""
    model = _make_model()
    torch.manual_seed(1)
    n, p, m = 20, 6, 8
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        out = model(X_train, y_train, x_test)

    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


def test_forward_works_without_ridge():
    """Forward pass works when use_ridge_expert=False."""
    model = _make_model(use_ridge_expert=False)
    torch.manual_seed(2)
    n, p, m = 15, 5, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        out = model(X_train, y_train, x_test)

    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


def test_forward_works_with_ridge():
    """Forward pass works when use_ridge_expert=True."""
    model = _make_model(use_ridge_expert=True)
    torch.manual_seed(3)
    n, p, m = 15, 5, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        out = model(X_train, y_train, x_test)

    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------

def test_modelconfig_rejects_bad_model_family():
    """ModelConfig with model_family='bad' raises ValueError."""
    with pytest.raises(ValueError, match="model_family"):
        ModelConfig(model_family="bad")


# ---------------------------------------------------------------------------
# _instantiate_model routing
# ---------------------------------------------------------------------------

def test_instantiate_model_routes_deepset():
    """_instantiate_model returns DeepSetModel when model_family='deepset'."""
    cfg = ModelConfig(model_family="deepset")
    model = _instantiate_model(cfg)
    assert isinstance(model, DeepSetModel)


def test_instantiate_model_routes_market_aware():
    """_instantiate_model returns MarketAwareDeepSetModel when model_family='market_aware'."""
    cfg = _make_cfg()
    model = _instantiate_model(cfg)
    assert isinstance(model, MarketAwareDeepSetModel)


def test_instantiate_model_unknown_family_raises():
    """_instantiate_model raises ValueError for unknown model_family."""
    cfg = ModelConfig.__new__(ModelConfig)
    # Bypass __post_init__ by setting attribute directly
    object.__setattr__(cfg, "model_family", "unknown_family")
    with pytest.raises(ValueError, match="Unknown model_family"):
        _instantiate_model(cfg)
