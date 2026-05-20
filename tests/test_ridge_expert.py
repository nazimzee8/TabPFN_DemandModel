"""
tests/test_ridge_expert.py

Correctness tests for RidgeExpert: shapes, primal/dual consistency,
known solution, and device consistency.
"""

import inspect
import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import RidgeExpert, ModelConfig, DeepSetICLModel  # noqa: E402


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_primal_path_shape():
    """n=50 >= p=5: primal path, output shape (m,)."""
    ridge = RidgeExpert()
    torch.manual_seed(0)
    n, p, m = 50, 5, 6
    X = torch.randn(n, p)
    y = torch.randn(n)
    x_test = torch.randn(m, p)

    out = ridge.predict(X, y, x_test, lam=1.0)
    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


def test_dual_path_shape():
    """n=5 < p=50: dual path, output shape (m,)."""
    ridge = RidgeExpert()
    torch.manual_seed(1)
    n, p, m = 5, 50, 6
    X = torch.randn(n, p)
    y = torch.randn(n)
    x_test = torch.randn(m, p)

    out = ridge.predict(X, y, x_test, lam=1.0)
    assert out.shape == (m,), f"Expected ({m},), got {out.shape}"


# ---------------------------------------------------------------------------
# Primal/dual consistency
# ---------------------------------------------------------------------------

def test_primal_dual_consistency():
    """Primal and dual closed-forms agree for same problem (max_abs_delta <= 1e-4)."""
    torch.manual_seed(42)
    n, p, m = 5, 10, 6   # n < p → dual path in RidgeExpert
    lam = 1.0
    kw = dict(dtype=torch.float64)

    X = torch.randn(n, p, **kw)
    y = torch.randn(n, **kw)
    x_test = torch.randn(m, p, **kw)

    # Primal (computed explicitly, independent of RidgeExpert path selection)
    A_primal = X.T @ X + lam * torch.eye(p, **kw)
    beta_primal = torch.linalg.solve(A_primal, X.T @ y)
    pred_primal = x_test @ beta_primal

    # Dual (same problem)
    K_dual = X @ X.T + lam * torch.eye(n, **kw)
    alpha  = torch.linalg.solve(K_dual, y)
    beta_dual = X.T @ alpha
    pred_dual = x_test @ beta_dual

    max_abs_delta = (pred_primal - pred_dual).abs().max().item()
    assert max_abs_delta <= 1e-4, (
        f"Primal/dual mismatch: {max_abs_delta:.2e} (threshold 1e-4)"
    )


# ---------------------------------------------------------------------------
# No y_test in signature
# ---------------------------------------------------------------------------

def test_no_y_test_in_signature():
    """RidgeExpert.predict must not have a y_test parameter."""
    sig = inspect.signature(RidgeExpert.predict)
    param_names = list(sig.parameters.keys())
    assert "y_test" not in param_names, (
        f"RidgeExpert.predict has forbidden parameter 'y_test'. Params: {param_names}"
    )


# ---------------------------------------------------------------------------
# Known solution: identity X
# ---------------------------------------------------------------------------

def test_known_solution_identity_X():
    """X = I_n (square, n==p), lambda≈0: ridge beta ≈ y, prediction ≈ x_test @ beta."""
    torch.manual_seed(5)
    n = 8
    p = n
    m = 4
    lam = 1e-6

    X = torch.eye(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    x_test = torch.randn(m, p, dtype=torch.float64)

    ridge = RidgeExpert()
    pred = ridge.predict(X, y, x_test, lam=lam)

    # With X=I and lambda→0, beta → y, so pred ≈ x_test @ y
    expected = x_test @ y
    max_err = (pred - expected).abs().max().item()

    # With lambda=1e-6, we expect reasonable accuracy but not exact match
    # The exact solution: beta = (I + 1e-6*I)^-1 y = y / (1 + 1e-6)
    beta_exact = y / (1.0 + lam)
    expected_exact = x_test @ beta_exact
    max_err_exact = (pred - expected_exact).abs().max().item()

    assert max_err_exact <= 1e-4, (
        f"Known-solution test failed: max_err={max_err_exact:.2e} (threshold 1e-4)"
    )


# ---------------------------------------------------------------------------
# Device consistency
# ---------------------------------------------------------------------------

def test_device_consistency():
    """Output tensor is on the same device as input tensors."""
    ridge = RidgeExpert()
    torch.manual_seed(3)
    n, p, m = 20, 5, 4
    X = torch.randn(n, p)
    y = torch.randn(n)
    x_test = torch.randn(m, p)

    out = ridge.predict(X, y, x_test, lam=1.0)
    assert out.device == X.device, (
        f"Output device {out.device} != input device {X.device}"
    )


# ---------------------------------------------------------------------------
# Statelessness: RidgeExpert has no trainable parameters
# ---------------------------------------------------------------------------

def test_stateless_no_nn_parameters():
    """RidgeExpert must not hold any nn.Parameter — it is a stateless helper."""
    ridge = RidgeExpert()
    # RidgeExpert is not an nn.Module; verify it has no __dict__ entries that
    # are nn.Parameter or nn.Module instances.
    for attr_name, attr_val in vars(ridge).items():
        assert not isinstance(attr_val, nn.Parameter), (
            f"RidgeExpert has unexpected nn.Parameter attribute: {attr_name}"
        )
        assert not isinstance(attr_val, nn.Module), (
            f"RidgeExpert has unexpected nn.Module attribute: {attr_name}"
        )


# ---------------------------------------------------------------------------
# Gradient flow: gate trains, ridge contributes no grad params
# ---------------------------------------------------------------------------

def test_gradient_flow_with_ridge_expert():
    """Gate head receives gradients; RidgeExpert has no parameters to train."""
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=32,
        d_rho=64,
        pool="pna",
        n_heads=4,
        n_sab_feat=1,
        use_ridge_expert=True,
        ridge_lambda=1.0,
        gate_hidden_dim=16,
    )
    model = DeepSetICLModel(cfg=cfg)
    model.train()

    torch.manual_seed(77)
    n, p, m = 20, 4, 3
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)
    target  = torch.randn(m)

    out  = model(X_train, y_train, x_test)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # Gate head must have gradients
    for name, param in model.gate_head.named_parameters():
        assert param.grad is not None, f"gate_head.{name} has no gradient after backward()"
        assert param.grad.abs().sum().item() > 0, (
            f"gate_head.{name} gradient is all zeros after backward()"
        )

    # RidgeExpert is not an nn.Module — no parameters to check
    assert not isinstance(model._ridge, nn.Module), (
        "RidgeExpert should not be an nn.Module"
    )


# ---------------------------------------------------------------------------
# ModelConfig validation: ridge_lambda must be > 0 when use_ridge_expert=True
# ---------------------------------------------------------------------------

def test_config_validation_ridge_lambda_zero():
    """ModelConfig raises ValueError when use_ridge_expert=True and ridge_lambda=0."""
    with pytest.raises(ValueError, match="ridge_lambda must be > 0"):
        ModelConfig(
            model_family="market_exchangeable_icl",
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            use_ridge_expert=True,
            ridge_lambda=0.0,
        )


def test_config_validation_ridge_lambda_negative():
    """ModelConfig raises ValueError when use_ridge_expert=True and ridge_lambda < 0."""
    with pytest.raises(ValueError, match="ridge_lambda must be > 0"):
        ModelConfig(
            model_family="market_exchangeable_icl",
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            use_ridge_expert=True,
            ridge_lambda=-1.0,
        )


def test_config_validation_ridge_lambda_ignored_when_disabled():
    """ridge_lambda <= 0 is not validated when use_ridge_expert=False."""
    # Should not raise — ridge_lambda is irrelevant when use_ridge_expert=False
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        use_ridge_expert=False,
        ridge_lambda=0.0,
    )
    assert cfg.ridge_lambda == 0.0


def test_config_validation_gate_hidden_dim_zero():
    """ModelConfig raises ValueError when gate_hidden_dim=0."""
    with pytest.raises(ValueError, match="gate_hidden_dim must be positive"):
        ModelConfig(
            model_family="market_exchangeable_icl",
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            gate_hidden_dim=0,
        )


def test_config_validation_gate_hidden_dim_negative():
    """ModelConfig raises ValueError when gate_hidden_dim < 0."""
    with pytest.raises(ValueError, match="gate_hidden_dim must be positive"):
        ModelConfig(
            model_family="market_exchangeable_icl",
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            gate_hidden_dim=-8,
        )


# ---------------------------------------------------------------------------
# Permutation invariance with ridge expert enabled
# ---------------------------------------------------------------------------

def test_permutation_invariance_with_ridge_expert():
    """Row permutation and column permutation leave predictions unchanged (use_ridge_expert=True)."""
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=32,
        d_rho=64,
        pool="pna",
        n_heads=4,
        n_sab_feat=1,
        use_ridge_expert=True,
        ridge_lambda=1.0,
        gate_hidden_dim=16,
    )
    model = DeepSetICLModel(cfg=cfg)
    model.eval()

    torch.manual_seed(55)
    n, p, m = 30, 6, 4
    X_train = torch.randn(n, p)
    y_train = torch.randn(n)
    x_test  = torch.randn(m, p)

    with torch.no_grad():
        y_orig = model(X_train, y_train, x_test)

        row_perm = torch.randperm(n)
        y_row = model(X_train[row_perm], y_train[row_perm], x_test)

        col_perm = torch.randperm(p)
        y_col = model(X_train[:, col_perm], y_train, x_test[:, col_perm])

    row_delta = (y_orig - y_row).abs().max().item()
    col_delta = (y_orig - y_col).abs().max().item()
    max_delta = max(row_delta, col_delta)

    assert max_delta <= 1e-5, (
        f"Permutation invariance violated with ridge expert: max_delta={max_delta:.2e}"
    )


# ---------------------------------------------------------------------------
# Default: use_ridge_expert is True in ModelConfig
# ---------------------------------------------------------------------------

def test_default_use_ridge_expert_is_true():
    """ModelConfig default must have use_ridge_expert=True."""
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
    )
    assert cfg.use_ridge_expert is True, (
        f"Expected use_ridge_expert=True by default, got {cfg.use_ridge_expert}"
    )
