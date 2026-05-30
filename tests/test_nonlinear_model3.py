from __future__ import annotations

import dataclasses
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import DeepSetICLModel, LatentRidgeExpert, ModelConfig  # noqa: E402


def _cfg(**overrides):
    params = {
        "model_family": "market_exchangeable_icl",
        "model_arch_version": "model3",
        "model_design_pattern": "inductive_forecasting",
        "d_phi": 32,
        "d_rho": 64,
        "n_heads": 4,
        "n_sab_feat": 1,
        "dropout": 0.0,
        "use_ridge_expert": False,
        "use_latent_ridge_expert": True,
        "latent_ridge_dim": 32,
        "query_context_heads": 4,
    }
    params.update(overrides)
    return ModelConfig(**params)


def _data(n=24, p=6, m=5):
    torch.manual_seed(123)
    X = torch.randn(n, p)
    y = torch.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    xq = torch.randn(m, p)
    return X, y, xq


def test_latent_ridge_shape_primal_and_gradients():
    ridge = LatentRidgeExpert()
    Z = torch.randn(20, 5, requires_grad=True)
    y = torch.randn(20)
    Zq = torch.randn(4, 5, requires_grad=True)
    pred = ridge.predict(Z, y, Zq, lam=1.0, use_bias=True)
    assert pred.shape == (4,)
    pred.sum().backward()
    assert Z.grad is not None and torch.isfinite(Z.grad).all()
    assert Zq.grad is not None and torch.isfinite(Zq.grad).all()


def test_latent_ridge_shape_dual_no_bias():
    ridge = LatentRidgeExpert()
    Z = torch.randn(4, 16, requires_grad=True)
    y = torch.randn(4)
    Zq = torch.randn(3, 16, requires_grad=True)
    pred = ridge.predict(Z, y, Zq, lam=0.5, use_bias=False)
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()


def test_latent_ridge_shape_dual_with_bias():
    ridge = LatentRidgeExpert()
    Z = torch.randn(4, 16, requires_grad=True)
    y = torch.randn(4)
    Zq = torch.randn(3, 16, requires_grad=True)
    pred = ridge.predict(Z, y, Zq, lam=0.5, use_bias=True)
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()


def test_latent_ridge_cuda_autocast_safety():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ridge = LatentRidgeExpert()
    Z = torch.randn(8, 4, device="cuda")
    y = torch.randn(8, device="cuda")
    Zq = torch.randn(2, 4, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = ridge.predict(Z, y, Zq, lam=1.0)
    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"use_query_context_attention": True},
        {"icl_pool_mode": "pna"},
        {"icl_pool_mode": "multipool"},
        {"feature_pool_mode": "attn"},
        {"ridge_mixture_mode": "convex"},
    ],
)
def test_nonlinear_forward_variants(overrides):
    model = DeepSetICLModel(_cfg(**overrides)).eval()
    X, y, xq = _data()
    with torch.no_grad():
        out, debug = model(X, y, xq, return_debug=True)
    assert out.shape == (xq.shape[0],)
    assert torch.isfinite(out).all()
    for key in ("latent_ridge_prediction", "neural_prediction", "gate", "final_normalized_prediction"):
        assert key in debug
        assert debug[key].shape == (xq.shape[0],)


def test_nonlinear_row_permutation_invariance():
    model = DeepSetICLModel(_cfg()).eval()
    X, y, xq = _data()
    pi = torch.randperm(X.shape[0])
    with torch.no_grad():
        out1 = model(X, y, xq)
        out2 = model(X[pi], y[pi], xq)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_nonlinear_column_permutation_consistency():
    model = DeepSetICLModel(_cfg()).eval()
    X, y, xq = _data()
    pi = torch.randperm(X.shape[1])
    with torch.no_grad():
        out1 = model(X, y, xq)
        out2 = model(X[:, pi], y, xq[:, pi])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_nonlinear_batch_query_consistency():
    model = DeepSetICLModel(_cfg()).eval()
    X, y, xq = _data(m=4)
    with torch.no_grad():
        batch = model(X, y, xq)
        singles = torch.stack([model(X, y, xq[i]) for i in range(xq.shape[0])])
    assert torch.allclose(batch, singles, atol=1e-5)


def test_old_config_dicts_still_load_and_new_config_round_trips():
    old = {
        "model_family": "market_exchangeable_icl",
        "model_arch_version": "model3",
        "model_design_pattern": "inductive_forecasting",
        "d_phi": 32,
        "d_rho": 64,
        "n_heads": 4,
        "n_sab_feat": 1,
    }
    assert ModelConfig(**old).use_latent_ridge_expert is False
    cfg = _cfg(ridge_mixture_mode="convex", feature_pool_mode="attn")
    assert ModelConfig(**dataclasses.asdict(cfg)) == cfg


def test_invalid_nonlinear_attention_divisibility_raises():
    with pytest.raises(ValueError, match="latent_ridge_dim"):
        _cfg(latent_ridge_dim=30, use_query_context_attention=True, query_context_heads=4)


@pytest.mark.parametrize(
    "target_fn",
    [
        lambda X: torch.sin(X[:, 0]),
        lambda X: X[:, 0] * X[:, 1],
        lambda X: torch.where(X[:, 0] > 0, X[:, 1] + 1.0, -X[:, 1]),
    ],
)
def test_tiny_nonlinear_training_reduces_loss(target_fn):
    torch.manual_seed(321)
    X = torch.randn(28, 4)
    y = target_fn(X)
    xq = torch.randn(12, 4)
    target = target_fn(xq)
    model = DeepSetICLModel(_cfg(d_phi=16, d_rho=64, latent_ridge_dim=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(X, y, xq), target).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X, y, xq), target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(X, y, xq), target).item()
        pred_std = model(X, y, xq).std().item()
    assert final < initial
    assert pred_std > 1e-4
