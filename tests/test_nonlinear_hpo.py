from __future__ import annotations

import importlib
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import hpo  # noqa: E402
from model import ModelConfig  # noqa: E402


def test_hpo_accepts_nonlinear_modes(monkeypatch):
    for mode in ("nonlinear_meta", "nonlinear_architecture"):
        monkeypatch.setenv("HPO_SWEEP_MODE", mode)
        importlib.reload(hpo)
        assert hpo.HPO_SWEEP_MODE == mode
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_nonlinear_meta_search_space_has_required_fields(monkeypatch):
    pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "nonlinear_meta")
    importlib.reload(hpo)
    space = hpo.build_hpo_search_space(tune)
    required = {
        "use_latent_ridge_expert",
        "latent_ridge_dim",
        "latent_ridge_lambda",
        "latent_ridge_jitter",
        "latent_ridge_use_bias",
        "use_query_context_attention",
        "query_context_heads",
        "icl_pool_mode",
        "feature_pool_mode",
        "ridge_mixture_mode",
        "nonlinear_head_hidden_mult",
    }
    assert required <= set(space)
    assert space["use_latent_ridge_expert"] is True
    assert space["use_ridge_expert"] is False
    assert space["d_phi"] == hpo.FIXED_D_PHI
    assert space["n_sab_feat"] == hpo.FIXED_N_SAB_FEAT
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_nonlinear_architecture_requires_baseline(monkeypatch):
    pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "nonlinear_architecture")
    importlib.reload(hpo)
    with pytest.raises(ValueError, match="HPO_BASELINE_CONFIG_STAGE_PATH"):
        hpo.build_hpo_search_space(tune, baseline_config=None)
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_nonlinear_architecture_search_space_uses_baseline(monkeypatch):
    pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "nonlinear_architecture")
    importlib.reload(hpo)
    baseline = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "use_ridge_expert": False,
        "ridge_lambda": 1.0,
        "gate_hidden_dim": 64,
        "latent_ridge_lambda": 0.25,
        "latent_ridge_jitter": 1e-6,
        "latent_ridge_use_bias": True,
        "query_context_heads": 4,
        "ridge_mixture_mode": "convex",
        "nonlinear_head_hidden_mult": 2,
        "use_huber": False,
        "huber_delta": 1.0,
        "lambda_l1": 0.0,
    }
    space = hpo.build_hpo_search_space(tune, baseline_config=baseline)
    assert space["lr"] == pytest.approx(baseline["lr"])
    assert space["latent_ridge_lambda"] == pytest.approx(0.25)
    assert "latent_ridge_dim" in space and not isinstance(space["latent_ridge_dim"], int)
    assert "d_phi" in space and not isinstance(space["d_phi"], int)
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_checkpoint_mismatch_catches_nonlinear_architecture_fields():
    saved = ModelConfig(
        d_phi=64,
        d_rho=64,
        n_heads=4,
        n_sab_feat=1,
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        use_ridge_expert=False,
        use_latent_ridge_expert=True,
        latent_ridge_dim=32,
    )
    current = ModelConfig(**{**saved.__dict__, "latent_ridge_dim": 64})
    mismatches = hpo.checkpoint_architecture_mismatches(saved, current)
    assert "latent_ridge_dim" in mismatches


def test_nonlinear_lambda_not_an_architecture_mismatch():
    saved = ModelConfig(
        d_phi=64,
        d_rho=64,
        n_heads=4,
        n_sab_feat=1,
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        use_ridge_expert=False,
        use_latent_ridge_expert=True,
        latent_ridge_lambda=0.1,
    )
    current = ModelConfig(**{**saved.__dict__, "latent_ridge_lambda": 10.0})
    assert hpo.checkpoint_architecture_mismatches(saved, current) == {}
