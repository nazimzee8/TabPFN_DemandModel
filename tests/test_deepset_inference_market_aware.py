"""
tests/test_deepset_inference_market_aware.py

Tests that run_permutation_tests() works correctly for both
MarketAwareDeepSetModel and DeepSetModel (architecture-aware dispatch).
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

from deepset_inference import run_permutation_tests
from model import DeepSetModel, MarketAwareDeepSetModel, ModelConfig


def _make_market_aware_model():
    cfg = ModelConfig(
        model_family="market_aware",
        d_sample=32,
        n_sab_sample_per_feature=0,
        sample_pool="attn",
        use_ridge_expert=False,
        n_sab_feat=1,
        n_heads=4,
        norm_feat=True,
        norm_target=True,
        dropout=0.0,
    )
    return MarketAwareDeepSetModel(cfg=cfg)


def _make_deepset_model():
    cfg = ModelConfig(d_phi=16, d_rho=16, n_sab_feat=0, n_sab_samp=0)
    return DeepSetModel(cfg=cfg)


class TestRunPermutationTestsMarketAware:
    def test_passes_for_fresh_market_aware_model(self):
        """run_permutation_tests returns True for a fresh MarketAwareDeepSetModel."""
        model = _make_market_aware_model()
        result = run_permutation_tests(model)
        assert result is True, "Expected all permutation tests to pass for MarketAware"

    def test_passes_for_deepset_model(self):
        """run_permutation_tests still works for DeepSetModel (regression guard)."""
        model = _make_deepset_model()
        result = run_permutation_tests(model)
        assert result is True, "Expected all permutation tests to pass for DeepSet"

    def test_raises_for_unknown_model_family(self):
        """run_permutation_tests raises ValueError for unknown model_family."""
        model = _make_deepset_model()
        # Monkey-patch cfg.model_family to an unknown value via dataclasses
        import dataclasses
        model.cfg = dataclasses.replace(model.cfg, model_family="deepset")
        object.__setattr__(model.cfg, "model_family", "unknown_family")
        with pytest.raises(ValueError, match="Unknown model_family"):
            run_permutation_tests(model)

    def test_restores_training_mode(self):
        """run_permutation_tests restores training mode when model was in train mode."""
        model = _make_market_aware_model()
        model.train()
        assert model.training is True
        run_permutation_tests(model)
        assert model.training is True, "Training mode should be restored after tests"

    def test_restores_eval_mode(self):
        """run_permutation_tests restores eval mode when model was in eval mode."""
        model = _make_market_aware_model()
        model.eval()
        run_permutation_tests(model)
        assert model.training is False, "Eval mode should be restored after tests"

    def test_market_aware_with_linear_equivariance(self):
        """MarketAware with n_sab_feat=0 (linear equivariance) also passes."""
        cfg = ModelConfig(
            model_family="market_aware",
            d_sample=32,
            n_sab_sample_per_feature=0,
            sample_pool="attn",
            use_ridge_expert=False,
            n_sab_feat=0,
            n_heads=4,
            norm_feat=True,
            norm_target=True,
            dropout=0.0,
        )
        model = MarketAwareDeepSetModel(cfg=cfg)
        result = run_permutation_tests(model)
        assert result is True
