"""
tests/test_market_aware_deepset_permutation.py

Row permutation invariance and column permutation consistency tests
for MarketAwareDeepSetModel.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import MarketAwareDeepSetModel, ModelConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(use_ridge_expert: bool = False) -> MarketAwareDeepSetModel:
    cfg = ModelConfig(
        model_family="market_aware",
        d_sample=32,
        n_sab_sample_per_feature=0,
        sample_pool="attn",
        use_ridge_expert=use_ridge_expert,
        ridge_lambda=1.0,
        residual_scale_init=0.1,
        gate_hidden_dim=32,
        n_sab_feat=1,
        n_heads=4,
        norm_feat=True,
        norm_target=True,
        dropout=0.0,
    )
    model = MarketAwareDeepSetModel(cfg=cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Row permutation invariance
# ---------------------------------------------------------------------------

def test_row_permutation_invariance():
    """Shuffling training rows must not change predictions (max_abs_delta <= 1e-5)."""
    model = _make_model(use_ridge_expert=False)
    torch.manual_seed(42)

    n, p, m = 30, 8, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        row_perm = torch.randperm(n)
        y_perm = model(X_train[row_perm], y_train[row_perm], x_test)

    max_abs_delta = (y_orig - y_perm).abs().max().item()
    assert max_abs_delta <= 1e-5, (
        f"Row permutation changed output by {max_abs_delta:.2e} (threshold 1e-5)"
    )


# ---------------------------------------------------------------------------
# Column permutation consistency
# ---------------------------------------------------------------------------

def test_column_permutation_consistency():
    """Permuting features consistently in X_train and x_test must not change predictions.

    max_abs_delta <= 1e-5.
    """
    model = _make_model(use_ridge_expert=False)
    torch.manual_seed(99)

    n, p, m = 30, 8, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        col_perm = torch.randperm(p)
        y_col = model(X_train[:, col_perm], y_train, x_test[:, col_perm])

    max_abs_delta = (y_orig - y_col).abs().max().item()
    assert max_abs_delta <= 1e-5, (
        f"Column permutation changed output by {max_abs_delta:.2e} (threshold 1e-5)"
    )


# ---------------------------------------------------------------------------
# Row permutation invariance with ridge expert
# ---------------------------------------------------------------------------

def test_row_perm_invariance_with_ridge_expert():
    """Row permutation invariance holds when use_ridge_expert=True (max_abs_delta <= 1e-5)."""
    model = _make_model(use_ridge_expert=True)
    torch.manual_seed(7)

    n, p, m = 30, 8, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)
        row_perm = torch.randperm(n)
        y_perm = model(X_train[row_perm], y_train[row_perm], x_test)

    max_abs_delta = (y_orig - y_perm).abs().max().item()
    assert max_abs_delta <= 1e-5, (
        f"Row permutation (with ridge expert) changed output by {max_abs_delta:.2e} (threshold 1e-5)"
    )
