"""Tests for mixed-categorical checkpoint format versioning (Step 9 — 4 tests)."""

import sys
import tempfile
from pathlib import Path

import torch
import pytest

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from constants import (
    CHECKPOINT_VERSION_MIXED_REGRESSION,
    CHECKPOINT_VERSION_MIXED_CLASSIFICATION,
)
from model import ModelConfig, DeepSetICLModel


def _save_checkpoint(model, format_version, filepath, **extra_meta):
    """Save a checkpoint in the format used by train.py."""
    import dataclasses as _dc
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": _dc.asdict(model.cfg),
        "metadata": {
            "checkpoint_format_version": format_version,
            **extra_meta,
        },
    }
    torch.save(ckpt, filepath)


# Test 23: Mixed regression checkpoint format version = 6
def test_mixed_regression_checkpoint_format_version_6():
    cfg = ModelConfig(
        d_phi=64, d_rho=128, n_heads=4, n_sab_feat=1,
        use_categorical_features=True,
        task_objective="inductive_regression",
    )
    model = DeepSetICLModel(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "best_regression.pt")
        _save_checkpoint(
            model, CHECKPOINT_VERSION_MIXED_REGRESSION, filepath,
            training_data_family="synthetic_linear_regression_mixed_categorical",
            uses_entity_embeddings=True,
        )
        ckpt = torch.load(filepath, weights_only=False)
        assert ckpt["metadata"]["checkpoint_format_version"] == 6
        assert ckpt["metadata"]["uses_entity_embeddings"] is True
        # Verify model config includes categorical fields
        cfg_dict = ckpt["model_config"]
        assert cfg_dict["use_categorical_features"] is True
        assert cfg_dict["cat_max_vocab_size"] == 53


# Test 24: Mixed classification checkpoint format version = 7
def test_mixed_classification_checkpoint_format_version_7():
    cfg = ModelConfig(
        d_phi=64, d_rho=128, n_heads=4, n_sab_feat=1,
        use_categorical_features=True,
        use_classification_path=True,
        task_objective="inductive_classification",
        max_num_classes=5,
    )
    model = DeepSetICLModel(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "best_classification.pt")
        _save_checkpoint(
            model, CHECKPOINT_VERSION_MIXED_CLASSIFICATION, filepath,
            training_data_family="synthetic_linear_classification_mixed_categorical",
            uses_entity_embeddings=True,
        )
        ckpt = torch.load(filepath, weights_only=False)
        assert ckpt["metadata"]["checkpoint_format_version"] == 7
        assert ckpt["metadata"]["uses_entity_embeddings"] is True


# Test 25: Pure numeric regression checkpoint (v4) still loads
def test_pure_numeric_regression_checkpoint_still_loads():
    cfg = ModelConfig(
        d_phi=64, d_rho=128, n_heads=4, n_sab_feat=1,
        use_categorical_features=False,
        task_objective="inductive_regression",
    )
    model = DeepSetICLModel(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "best_regression_v4.pt")
        _save_checkpoint(model, 4, filepath,
                         training_data_family="synthetic_linear_regression")
        ckpt = torch.load(filepath, weights_only=False)
        # Reconstruct config and model from checkpoint
        loaded_cfg = ModelConfig(**ckpt["model_config"])
        loaded_model = DeepSetICLModel(loaded_cfg)
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        assert ckpt["metadata"]["checkpoint_format_version"] == 4
        assert loaded_cfg.use_categorical_features is False
        # Verify no categorical modules
        assert loaded_model.cat_token_encoder is None


# Test 26: Pure numeric classification checkpoint (v5) still loads
def test_pure_numeric_classification_checkpoint_still_loads():
    cfg = ModelConfig(
        d_phi=64, d_rho=128, n_heads=4, n_sab_feat=1,
        use_categorical_features=False,
        use_classification_path=True,
        task_objective="inductive_classification",
        max_num_classes=5,
    )
    model = DeepSetICLModel(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / "best_classification_v5.pt")
        _save_checkpoint(model, 5, filepath,
                         training_data_family="synthetic_linear_classification")
        ckpt = torch.load(filepath, weights_only=False)
        loaded_cfg = ModelConfig(**ckpt["model_config"])
        loaded_model = DeepSetICLModel(loaded_cfg)
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        assert ckpt["metadata"]["checkpoint_format_version"] == 5
        assert loaded_cfg.use_categorical_features is False
        assert loaded_model.cat_token_encoder is None
