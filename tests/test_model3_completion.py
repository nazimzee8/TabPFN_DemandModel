"""
tests/test_model3_completion.py

Tests for MarketExchangeableCompletionModel — MODEL3 transductive completion.

Covers:
  - ModelConfig validation for completion selectors
  - Output shape (rows x cols)
  - Row equivariance
  - Column equivariance
  - Joint row+col equivariance
  - Finite outputs (no NaN/Inf)
  - Gradient flow
  - Masked aggregation (unobserved cells excluded)
  - Checkpoint v4 metadata
  - Factory routing
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
    MarketExchangeableCompletionModel,
    _instantiate_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _comp_cfg(n_blocks=2, d_model=64, dropout=0.0):
    return ModelConfig(
        model_family="market_exchangeable_completion",
        model_arch_version="model3",
        model3_design_pattern="transductive_completion",
        d_phi=d_model,
        n_sab_feat=n_blocks,
        n_heads=4,
        dropout=dropout,
    )


def _fresh_comp(n_blocks=2, d_model=64):
    cfg = _comp_cfg(n_blocks=n_blocks, d_model=d_model)
    m = MarketExchangeableCompletionModel(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------

class TestCompletionModelConfig:
    def test_valid_completion_config(self):
        cfg = _comp_cfg()
        assert cfg.model_family == "market_exchangeable_completion"
        assert cfg.model_arch_version == "model3"
        assert cfg.model3_design_pattern == "transductive_completion"

    def test_completion_family_requires_model3(self):
        with pytest.raises(ValueError, match="model_arch_version"):
            ModelConfig(
                model_family="market_exchangeable_completion",
                model_arch_version="model2",
                model3_design_pattern="transductive_completion",
            )

    def test_completion_requires_transductive_pattern(self):
        with pytest.raises(ValueError, match="transductive_completion"):
            ModelConfig(
                model_family="market_exchangeable_icl",
                model_arch_version="model3",
                model3_design_pattern="transductive_completion",
            )

    def test_wrong_family_for_completion_class_raises(self):
        cfg = ModelConfig(model_family="market_aware", d_phi=64)
        with pytest.raises(ValueError, match="market_exchangeable_completion"):
            MarketExchangeableCompletionModel(cfg)


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestFactoryRouting:
    def test_instantiate_completion(self):
        cfg = _comp_cfg()
        m = _instantiate_model(cfg)
        assert isinstance(m, MarketExchangeableCompletionModel)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_output_matches_input_shape(self):
        m = _fresh_comp()
        X = torch.randn(10, 8)
        mask = torch.rand(10, 8) > 0.3
        with torch.no_grad():
            out = m(X, mask)
        assert out.shape == (10, 8), f"Expected (10, 8), got {out.shape}"

    def test_square_matrix(self):
        m = _fresh_comp(d_model=32)
        X = torch.randn(5, 5)
        mask = torch.ones(5, 5, dtype=torch.bool)
        with torch.no_grad():
            out = m(X, mask)
        assert out.shape == (5, 5)

    def test_tall_matrix(self):
        m = _fresh_comp(d_model=32)
        X = torch.randn(20, 4)
        mask = torch.rand(20, 4) > 0.4
        with torch.no_grad():
            out = m(X, mask)
        assert out.shape == (20, 4)

    def test_wide_matrix(self):
        m = _fresh_comp(d_model=32)
        X = torch.randn(3, 15)
        mask = torch.rand(3, 15) > 0.4
        with torch.no_grad():
            out = m(X, mask)
        assert out.shape == (3, 15)


# ---------------------------------------------------------------------------
# Row equivariance
# ---------------------------------------------------------------------------

class TestRowEquivariance:
    def test_row_permutation_equivariance(self):
        torch.manual_seed(42)
        m = _fresh_comp()
        X    = torch.randn(10, 6)
        mask = torch.rand(10, 6) > 0.3
        pi   = torch.randperm(10)
        with torch.no_grad():
            out_orig = m(X, mask)
            out_perm = m(X[pi], mask[pi])
        assert torch.allclose(out_perm, out_orig[pi], atol=1e-5), (
            f"Row equivariance failed: max_delta={( out_perm - out_orig[pi]).abs().max().item():.2e}"
        )

    def test_row_perm_with_sparse_mask(self):
        torch.manual_seed(7)
        m = _fresh_comp(d_model=32)
        X    = torch.randn(8, 5)
        mask = torch.rand(8, 5) > 0.6  # ~40% observed
        pi   = torch.randperm(8)
        with torch.no_grad():
            out_orig = m(X, mask)
            out_perm = m(X[pi], mask[pi])
        assert torch.allclose(out_perm, out_orig[pi], atol=1e-5)


# ---------------------------------------------------------------------------
# Column equivariance
# ---------------------------------------------------------------------------

class TestColumnEquivariance:
    def test_column_permutation_equivariance(self):
        torch.manual_seed(0)
        m = _fresh_comp()
        X      = torch.randn(10, 6)
        mask   = torch.rand(10, 6) > 0.3
        pi_col = torch.randperm(6)
        with torch.no_grad():
            out_orig = m(X, mask)
            out_perm = m(X[:, pi_col], mask[:, pi_col])
        assert torch.allclose(out_perm, out_orig[:, pi_col], atol=1e-5), (
            f"Column equivariance failed: max_delta={(out_perm - out_orig[:, pi_col]).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Joint row+col equivariance
# ---------------------------------------------------------------------------

class TestJointEquivariance:
    def test_joint_row_col_permutation(self):
        torch.manual_seed(13)
        m = _fresh_comp()
        X    = torch.randn(8, 6)
        mask = torch.rand(8, 6) > 0.3
        pi_r = torch.randperm(8)
        pi_c = torch.randperm(6)
        with torch.no_grad():
            out_orig = m(X, mask)
            out_perm = m(X[pi_r][:, pi_c], mask[pi_r][:, pi_c])
        expected = out_orig[pi_r][:, pi_c]
        assert torch.allclose(out_perm, expected, atol=1e-5), (
            f"Joint equivariance failed: max_delta={(out_perm - expected).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Finite outputs
# ---------------------------------------------------------------------------

class TestFiniteOutputs:
    def test_no_nan_or_inf_full_mask(self):
        m = _fresh_comp()
        X    = torch.randn(8, 6)
        mask = torch.ones(8, 6, dtype=torch.bool)
        with torch.no_grad():
            out = m(X, mask)
        assert torch.isfinite(out).all(), f"Non-finite outputs: {out}"

    def test_no_nan_or_inf_sparse_mask(self):
        torch.manual_seed(5)
        m = _fresh_comp()
        X    = torch.randn(8, 6)
        mask = torch.rand(8, 6) > 0.5
        with torch.no_grad():
            out = m(X, mask)
        assert torch.isfinite(out).all()

    def test_no_nan_all_missing_mask(self):
        """All-false mask (no observed cells) — model should not produce NaN."""
        m = _fresh_comp()
        X    = torch.randn(6, 5)
        mask = torch.zeros(6, 5, dtype=torch.bool)
        with torch.no_grad():
            out = m(X, mask)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Masked aggregation semantics
# ---------------------------------------------------------------------------

class TestMaskedSemantics:
    def test_unobserved_cells_excluded_from_aggregation(self):
        """
        Masking out a cell should change the output — if unobserved cells
        were treated as zeros rather than being excluded, outputs would differ
        when the value under the mask changes while mask=False.
        """
        torch.manual_seed(99)
        m = _fresh_comp()
        X1 = torch.randn(6, 5)
        X2 = X1.clone()
        X2[0, 0] = 999.0  # change a masked-out cell
        mask = torch.rand(6, 5) > 0.4
        mask[0, 0] = False  # ensure (0,0) is masked

        with torch.no_grad():
            out1 = m(X1, mask)
            out2 = m(X2, mask)

        # Since (0,0) is masked out, its value should NOT affect outputs
        assert torch.allclose(out1, out2, atol=1e-5), (
            "Unobserved cell value affected output — masking may not be applied."
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_through_cell_encoder(self):
        m = _fresh_comp()
        m.train()
        X    = torch.randn(6, 5)
        mask = torch.rand(6, 5) > 0.3
        out  = m(X, mask).sum()
        out.backward()
        assert m.cell_encoder.net[0].weight.grad is not None

    def test_gradients_through_blocks(self):
        m = _fresh_comp(n_blocks=2)
        m.train()
        X    = torch.randn(6, 5)
        mask = torch.rand(6, 5) > 0.3
        out  = m(X, mask).sum()
        out.backward()
        assert m.blocks[0].row_update[0].weight.grad is not None
        assert m.blocks[1].col_update[0].weight.grad is not None

    def test_gradients_through_decoder_head(self):
        m = _fresh_comp()
        m.train()
        X    = torch.randn(6, 5)
        mask = torch.rand(6, 5) > 0.3
        out  = m(X, mask).sum()
        out.backward()
        assert m.decoder_head.weight.grad is not None

    def test_parameter_changes_after_optimizer_step(self):
        m = _fresh_comp()
        m.train()
        X    = torch.randn(6, 5)
        mask = torch.rand(6, 5) > 0.3
        p_before = m.decoder_head.weight.data.clone()
        optimizer = torch.optim.SGD(m.parameters(), lr=1.0)
        out = m(X, mask).sum()
        out.backward()
        optimizer.step()
        assert not torch.allclose(m.decoder_head.weight.data, p_before)


# ---------------------------------------------------------------------------
# Checkpoint v4 metadata
# ---------------------------------------------------------------------------

class TestCheckpointV4:
    def test_checkpoint_format_version_4(self):
        cfg = _comp_cfg()
        m = MarketExchangeableCompletionModel(cfg)
        m.eval()
        tmp  = tempfile.mkdtemp()
        path = os.path.join(tmp, "model3_completion.pt")
        torch.save({
            "checkpoint_format_version": 4,
            "cfg": dataclasses.asdict(cfg),
            "state_dict": m.state_dict(),
            "metadata": {
                "model_arch_version":    "model3",
                "model3_design_pattern": "transductive_completion",
                "model_family":          "market_exchangeable_completion",
                "training_data_family":  "market_mental_model",
                "task_objective":        "transductive_completion",
                "checkpoint_format_version": 4,
            },
        }, path)
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["checkpoint_format_version"] == 4
        assert ckpt["metadata"]["model_arch_version"] == "model3"
        assert ckpt["metadata"]["model3_design_pattern"] == "transductive_completion"
        assert ckpt["metadata"]["task_objective"] == "transductive_completion"

    def test_reload_from_checkpoint(self):
        cfg = _comp_cfg()
        m = MarketExchangeableCompletionModel(cfg)
        m.eval()
        tmp  = tempfile.mkdtemp()
        path = os.path.join(tmp, "comp_reload.pt")
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
        X    = torch.randn(8, 5)
        mask = torch.rand(8, 5) > 0.3
        with torch.no_grad():
            o1 = m(X, mask)
            o2 = m2(X, mask)
        assert torch.allclose(o1, o2, atol=1e-6)
