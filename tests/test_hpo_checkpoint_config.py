import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import hpo  # noqa: E402
from model import ModelConfig  # noqa: E402


def test_checkpoint_architecture_mismatches_accepts_dict_cfg():
    saved_cfg = {
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_heads": 4,
        "n_sab_feat": 1,
        "n_sab_samp": 1,
        "norm_feat": True,
        "norm_target": True,
        "dropout": 0.1,
    }
    current_cfg = ModelConfig(**saved_cfg)

    assert hpo.checkpoint_architecture_mismatches(saved_cfg, current_cfg) == {}


def test_checkpoint_architecture_mismatches_ignores_dropout():
    saved_cfg = {
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_heads": 4,
        "n_sab_feat": 1,
        "n_sab_samp": 1,
        "norm_feat": True,
        "norm_target": True,
        "dropout": 0.1,
    }
    current_cfg = ModelConfig(**{**saved_cfg, "dropout": 0.3})

    assert hpo.checkpoint_architecture_mismatches(saved_cfg, current_cfg) == {}


def test_checkpoint_architecture_mismatches_rejects_missing_cfg():
    with pytest.raises(ValueError, match="missing required cfg"):
        hpo.checkpoint_architecture_mismatches(None, ModelConfig())


# ---------------------------------------------------------------------------
# HPO_MODEL_FAMILY and best_config propagation (Phase 3)
# ---------------------------------------------------------------------------

def test_hpo_model_family_constant_exists():
    """HPO_MODEL_FAMILY constant exists on the hpo module."""
    assert hasattr(hpo, "HPO_MODEL_FAMILY")


def test_hpo_model_family_defaults_to_market_aware(monkeypatch):
    """HPO_MODEL_FAMILY defaults to 'market_aware' when env var not set."""
    import importlib
    monkeypatch.delenv("DEEPSET_MODEL_FAMILY", raising=False)
    importlib.reload(hpo)
    assert hpo.HPO_MODEL_FAMILY == "market_aware"
    importlib.reload(hpo)


def test_hpo_model_family_respects_env_var(monkeypatch):
    """HPO_MODEL_FAMILY reads from DEEPSET_MODEL_FAMILY env var."""
    import importlib
    monkeypatch.setenv("DEEPSET_MODEL_FAMILY", "deepset")
    importlib.reload(hpo)
    assert hpo.HPO_MODEL_FAMILY == "deepset"
    importlib.reload(hpo)


def test_checkpoint_architecture_mismatches_detects_family_change():
    """checkpoint_architecture_mismatches detects model_family mismatch."""
    saved = ModelConfig(model_family="deepset", d_phi=128, d_rho=256, n_sab_feat=1,
                        n_sab_samp=1, norm_feat=True, norm_target=True)
    current = ModelConfig(model_family="market_aware", d_phi=128, d_rho=256,
                          n_sab_feat=1, n_sab_samp=1, norm_feat=True, norm_target=True)
    mismatches = hpo.checkpoint_architecture_mismatches(saved, current)
    assert "model_family" in mismatches
