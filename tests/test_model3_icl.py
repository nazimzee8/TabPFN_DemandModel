"""
tests/test_model3_icl.py

Tests for MarketExchangeableICLModel — MODEL3 inductive forecasting.

Covers:
  - ModelConfig validation (selector combination checks)
  - Output shape: single and batched queries
  - Row permutation invariance
  - Column permutation consistency
  - Query sensitivity (different queries -> different outputs)
  - No query collapse (batched outputs must not be near-constant scalars)
  - Gradient flow through all main paths
  - Checkpoint v4 metadata format
  - Factory routing (_instantiate_model)
  - MODEL2 default unaffected by MODEL3 additions
"""
from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import (
    ModelConfig,
    MarketExchangeableICLModel,
    _instantiate_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _icl_cfg(n_blocks=2, d_model=64, dropout=0.0, use_ridge=False):
    return ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model3_design_pattern="inductive_forecasting",
        d_phi=d_model,
        n_sab_feat=n_blocks,
        n_heads=4,
        dropout=dropout,
        norm_feat=True,
        norm_target=True,
        use_ridge_expert=use_ridge,
        gate_hidden_dim=32,
    )


def _fresh_icl(n_blocks=2, d_model=64, use_ridge=False):
    cfg = _icl_cfg(n_blocks=n_blocks, d_model=d_model, use_ridge=use_ridge)
    m = MarketExchangeableICLModel(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------

class TestModelConfigValidation:
    def test_valid_icl_config(self):
        cfg = _icl_cfg()
        assert cfg.model_family == "market_exchangeable_icl"
        assert cfg.model_arch_version == "model3"
        assert cfg.model3_design_pattern == "inductive_forecasting"

    def test_icl_family_requires_model3_arch(self):
        with pytest.raises(ValueError, match="model_arch_version"):
            ModelConfig(
                model_family="market_exchangeable_icl",
                model_arch_version="model2",
                model3_design_pattern="inductive_forecasting",
            )

    def test_icl_pattern_must_match_family(self):
        with pytest.raises(ValueError, match="inductive_forecasting"):
            ModelConfig(
                model_family="market_exchangeable_completion",
                model_arch_version="model3",
                model3_design_pattern="inductive_forecasting",
            )

    def test_invalid_arch_version_raises(self):
        with pytest.raises(ValueError, match="model_arch_version"):
            ModelConfig(
                model_family="market_aware",
                model_arch_version="model99",
            )

    def test_invalid_design_pattern_raises(self):
        with pytest.raises(ValueError, match="model3_design_pattern"):
            ModelConfig(
                model_family="market_aware",
                model3_design_pattern="bad_pattern",
            )

    def test_model2_default_unaffected(self):
        """MODEL2 default config still works with the new fields."""
        cfg = ModelConfig(model_family="market_aware", d_phi=128, n_sab_feat=1)
        assert cfg.model_arch_version == "model2"
        assert cfg.model3_design_pattern == "inductive_forecasting"
        m = _instantiate_model(cfg)
        from model import MarketAwareDeepSetModel
        assert isinstance(m, MarketAwareDeepSetModel)

    def test_deepset_default_unaffected(self):
        cfg = ModelConfig(model_family="deepset", d_phi=64, n_sab_feat=0)
        assert cfg.model_arch_version == "model2"
        m = _instantiate_model(cfg)
        from model import DeepSetModel
        assert isinstance(m, DeepSetModel)


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestFactoryRouting:
    def test_instantiate_icl(self):
        cfg = _icl_cfg()
        m = _instantiate_model(cfg)
        assert isinstance(m, MarketExchangeableICLModel)

    def test_wrong_family_for_icl_class_raises(self):
        cfg = ModelConfig(model_family="market_aware", d_phi=64)
        with pytest.raises(ValueError, match="market_exchangeable_icl"):
            MarketExchangeableICLModel(cfg)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_single_query_returns_scalar(self):
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt = torch.randn(5)
        with torch.no_grad():
            out = m(X, y, xt)
        assert out.ndim == 0, f"Expected scalar, got shape {out.shape}"

    def test_batched_query_returns_vector(self):
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt = torch.randn(8, 5)
        with torch.no_grad():
            out = m(X, y, xt)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_single_feature_single_sample(self):
        m = _fresh_icl()
        X = torch.randn(5, 1)
        y = torch.randn(5)
        xt = torch.randn(1)
        with torch.no_grad():
            out = m(X, y, xt)
        assert out.ndim == 0

    def test_large_context(self):
        m = _fresh_icl(d_model=32)
        X = torch.randn(100, 10)
        y = torch.randn(100)
        xt = torch.randn(4, 10)
        with torch.no_grad():
            out = m(X, y, xt)
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Row permutation invariance
# ---------------------------------------------------------------------------

class TestRowPermutationInvariance:
    def test_permuting_context_rows_unchanged(self):
        torch.manual_seed(42)
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt = torch.randn(5)
        pi = torch.randperm(20)
        with torch.no_grad():
            out1 = m(X, y, xt)
            out2 = m(X[pi], y[pi], xt)
        assert torch.allclose(out1, out2, atol=1e-5), (
            f"Row permutation invariance failed: delta={abs(out1 - out2).item():.2e}"
        )

    def test_row_perm_batched_queries(self):
        torch.manual_seed(7)
        m = _fresh_icl()
        X = torch.randn(15, 4)
        y = torch.randn(15)
        xt = torch.randn(6, 4)
        pi = torch.randperm(15)
        with torch.no_grad():
            out1 = m(X, y, xt)
            out2 = m(X[pi], y[pi], xt)
        assert torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# Column permutation consistency
# ---------------------------------------------------------------------------

class TestColumnPermutationConsistency:
    def test_permuting_features_consistently(self):
        torch.manual_seed(99)
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt = torch.randn(5)
        pi_col = torch.randperm(5)
        with torch.no_grad():
            out1 = m(X, y, xt)
            out2 = m(X[:, pi_col], y, xt[pi_col])
        assert torch.allclose(out1, out2, atol=1e-5), (
            f"Column permutation consistency failed: delta={abs(out1 - out2).item():.2e}"
        )


# ---------------------------------------------------------------------------
# Query sensitivity & no collapse
# ---------------------------------------------------------------------------

class TestQuerySensitivity:
    def test_different_queries_produce_different_outputs(self):
        torch.manual_seed(0)
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt1 = torch.randn(5)
        xt2 = torch.randn(5)
        with torch.no_grad():
            o1 = m(X, y, xt1)
            o2 = m(X, y, xt2)
        # Outputs should differ for different queries (unless very unlucky random seed)
        assert not torch.allclose(o1, o2, atol=1e-4), "Different queries produced identical outputs"

    def test_batched_queries_not_constant(self):
        torch.manual_seed(1)
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt = torch.randn(16, 5)
        with torch.no_grad():
            out = m(X, y, xt)
        std = out.std().item()
        assert std > 1e-4, f"Batched outputs collapsed to near-constant: std={std:.2e}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_through_cell_encoder(self):
        m = _fresh_icl()
        m.train()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        out = m(X, y, xt)
        out.backward()
        assert m.cell_encoder.net[0].weight.grad is not None

    def test_gradients_flow_through_col_encoder(self):
        m = _fresh_icl()
        m.train()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        out = m(X, y, xt)
        out.backward()
        assert m.col_encoder.net[0].weight.grad is not None

    def test_gradients_flow_through_blocks(self):
        m = _fresh_icl(n_blocks=2)
        m.train()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        out = m(X, y, xt)
        out.backward()
        assert m.blocks[0].row_update[0].weight.grad is not None
        assert m.blocks[1].col_update[0].weight.grad is not None

    def test_gradients_flow_through_pred_head(self):
        m = _fresh_icl()
        m.train()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        out = m(X, y, xt)
        out.backward()
        assert m.pred_head.weight.grad is not None

    def test_parameter_changes_after_sgd_step(self):
        m = _fresh_icl()
        m.train()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        p_before = m.pred_head.weight.data.clone()
        optimizer = torch.optim.SGD(m.parameters(), lr=1.0)
        out = m(X, y, xt)
        out.backward()
        optimizer.step()
        assert not torch.allclose(m.pred_head.weight.data, p_before)

    def test_ridge_expert_gradients(self):
        m = _fresh_icl(use_ridge=True)
        m.train()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        out = m(X, y, xt)
        out.backward()
        assert m.gate_head[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Checkpoint v4 metadata
# ---------------------------------------------------------------------------

class TestCheckpointV4:
    def test_checkpoint_format_version_4(self):
        cfg = _icl_cfg()
        m = MarketExchangeableICLModel(cfg)
        m.eval()
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "model3.pt")
        torch.save({
            "checkpoint_format_version": 4,
            "cfg": dataclasses.asdict(cfg),
            "state_dict": m.state_dict(),
            "metadata": {
                "model_arch_version": "model3",
                "model3_design_pattern": "inductive_forecasting",
                "model_family": "market_exchangeable_icl",
                "training_data_family": "synthetic_regression_combined",
                "task_objective": "inductive_regression",
                "checkpoint_format_version": 4,
            },
        }, path)
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["checkpoint_format_version"] == 4
        assert ckpt["metadata"]["model_arch_version"] == "model3"
        assert ckpt["metadata"]["model3_design_pattern"] == "inductive_forecasting"
        assert ckpt["metadata"]["model_family"] == "market_exchangeable_icl"
        assert ckpt["metadata"]["task_objective"] == "inductive_regression"

    def test_reload_from_checkpoint(self):
        cfg = _icl_cfg()
        m = MarketExchangeableICLModel(cfg)
        m.eval()
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "model3_reload.pt")
        torch.save({
            "checkpoint_format_version": 4,
            "cfg": dataclasses.asdict(cfg),
            "state_dict": m.state_dict(),
            "metadata": {},
        }, path)
        ckpt = torch.load(path, weights_only=False)
        cfg2 = ModelConfig(**ckpt["cfg"])
        m2 = _instantiate_model(cfg2)
        m2.load_state_dict(ckpt["state_dict"])
        m2.eval()
        X = torch.randn(10, 4)
        y = torch.randn(10)
        xt = torch.randn(4)
        with torch.no_grad():
            o1 = m(X, y, xt)
            o2 = m2(X, y, xt)
        assert torch.allclose(o1, o2, atol=1e-6)

    def test_cfg_serialization_roundtrip(self):
        cfg = _icl_cfg()
        d = dataclasses.asdict(cfg)
        assert d["model_arch_version"] == "model3"
        assert d["model3_design_pattern"] == "inductive_forecasting"
        cfg2 = ModelConfig(**d)
        assert cfg2.model_family == cfg.model_family
        assert cfg2.model_arch_version == cfg.model_arch_version


# ---------------------------------------------------------------------------
# Finite outputs
# ---------------------------------------------------------------------------

class TestFiniteOutputs:
    def test_no_nan_or_inf(self):
        torch.manual_seed(42)
        m = _fresh_icl()
        X = torch.randn(20, 5)
        y = torch.randn(20)
        xt = torch.randn(4, 5)
        with torch.no_grad():
            out = m(X, y, xt)
        assert torch.isfinite(out).all(), f"Non-finite outputs: {out}"

    def test_no_nan_with_dropout(self):
        torch.manual_seed(0)
        cfg = _icl_cfg(dropout=0.2)
        m = MarketExchangeableICLModel(cfg)
        m.train()
        X = torch.randn(15, 3)
        y = torch.randn(15)
        xt = torch.randn(3, 3)
        out = m(X, y, xt)
        assert torch.isfinite(out).all()
