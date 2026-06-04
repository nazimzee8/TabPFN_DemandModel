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


def test_hpo_accepts_nonlinear_meta_mode(monkeypatch):
    monkeypatch.setenv("HPO_SWEEP_MODE", "nonlinear_meta")
    importlib.reload(hpo)
    assert hpo.HPO_SWEEP_MODE == "nonlinear_meta"
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


def test_nonlinear_architecture_mode_is_rejected(monkeypatch):
    """nonlinear_architecture mode has been removed; reloading hpo with it must raise ValueError."""
    monkeypatch.setenv("HPO_SWEEP_MODE", "nonlinear_architecture")
    with pytest.raises(ValueError, match="HPO_SWEEP_MODE"):
        importlib.reload(hpo)
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_nonlinear_meta_search_space_tunes_latent_ridge_dim(monkeypatch):
    """nonlinear_meta now tunes latent_ridge_dim in {32, 64, 128} without requiring a baseline."""
    pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "nonlinear_meta")
    importlib.reload(hpo)
    space = hpo.build_hpo_search_space(tune)
    assert "latent_ridge_dim" in space
    assert not isinstance(space["latent_ridge_dim"], int), "latent_ridge_dim should be a tune.choice"
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


def test_merge_preserves_baseline_pretrain_checkpoint_when_arch_empty():
    """baseline has checkpoint path, arch has empty → merged gets baseline path."""
    baseline = {"hpo_sweep_mode": "nonlinear_meta", "_meta": {
        "pretrain_checkpoint_stage_path": "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt",
        "pretrain_checkpoint_map": {"nonlinear": "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt"},
        "best_val_mse": 0.5,
    }}
    arch = {"hpo_sweep_mode": "nonlinear_architecture", "d_phi": 128, "_meta": {
        "pretrain_checkpoint_stage_path": "",
        "best_val_mse": 0.4,
    }}
    merged = hpo._merge_sweep_configs(baseline, arch)
    assert merged["_meta"]["pretrain_checkpoint_stage_path"] == (
        "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt"
    )


def test_merge_arch_pretrain_checkpoint_takes_precedence():
    """Both sweeps have paths → arch path wins."""
    baseline = {"hpo_sweep_mode": "nonlinear_meta", "_meta": {
        "pretrain_checkpoint_stage_path": "@MODEL_STAGE/checkpoints/pretrain_base.pt"}}
    arch = {"hpo_sweep_mode": "nonlinear_architecture", "_meta": {
        "pretrain_checkpoint_stage_path": "@MODEL_STAGE/checkpoints/pretrain_arch.pt"}}
    merged = hpo._merge_sweep_configs(baseline, arch)
    assert merged["_meta"]["pretrain_checkpoint_stage_path"] == (
        "@MODEL_STAGE/checkpoints/pretrain_arch.pt"
    )


def test_merge_preserves_checkpoint_map():
    """Merged _meta.pretrain_checkpoint_map is union of both maps."""
    baseline = {"hpo_sweep_mode": "nonlinear_meta", "_meta": {
        "pretrain_checkpoint_map": {"nonlinear": "@MODEL_STAGE/checkpoints/a.pt", "64": "@MODEL_STAGE/checkpoints/b.pt"}}}
    arch = {"hpo_sweep_mode": "nonlinear_architecture", "_meta": {
        "pretrain_checkpoint_map": {"nonlinear": "@MODEL_STAGE/checkpoints/c.pt"}}}
    merged = hpo._merge_sweep_configs(baseline, arch)
    cmap = merged["_meta"]["pretrain_checkpoint_map"]
    assert cmap["nonlinear"] == "@MODEL_STAGE/checkpoints/c.pt"  # arch wins
    assert cmap["64"] == "@MODEL_STAGE/checkpoints/b.pt"          # baseline preserved


def test_merge_propagates_training_data_family():
    """training_data_family from baseline _meta survives into merged _meta."""
    baseline = {"hpo_sweep_mode": "nonlinear_meta", "_meta": {
        "training_data_family": "synthetic_regression_nonlinear"}}
    arch = {"hpo_sweep_mode": "nonlinear_architecture", "_meta": {}}
    merged = hpo._merge_sweep_configs(baseline, arch)
    assert merged["_meta"]["training_data_family"] == "synthetic_regression_nonlinear"
