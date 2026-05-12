"""
Tests for scripts/migrate_checkpoint.py — legacy → v2 checkpoint migration.
"""

import dataclasses
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import ModelConfig, DeepSetModel  # noqa: E402
from evaluate import load_checkpoint_compat  # noqa: E402


def _load_migrate():
    spec = importlib.util.spec_from_file_location(
        "migrate_checkpoint",
        os.path.join(SCRIPTS_DIR, "migrate_checkpoint.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model():
    cfg = ModelConfig(d_phi=16, d_rho=32, pool="mean", n_sab_feat=0, n_sab_samp=0)
    model = DeepSetModel(cfg=cfg)
    return cfg, model


def test_migrate_legacy_checkpoint_with_pickled_modelconfig():
    """Create a legacy checkpoint with a pickled ModelConfig, migrate it, verify v2 output."""
    tmp_dir = tempfile.mkdtemp()
    try:
        cfg, model = _tiny_model()
        legacy_path = os.path.join(tmp_dir, "legacy.pt")
        # Save cfg as a ModelConfig object (legacy format — will fail weights_only=True).
        torch.save({"cfg": cfg, "state_dict": model.state_dict()}, legacy_path)

        # Confirm the legacy file fails weights_only=True (or safe_globals — either is fine).
        with pytest.raises(Exception):
            torch.load(legacy_path, weights_only=True)

        # Migrate.
        migrate = _load_migrate()
        migrated_path = os.path.join(tmp_dir, "migrated.pt")
        migrate.migrate_checkpoint(legacy_path, migrated_path)

        # Must load cleanly with weights_only=True.
        ckpt = torch.load(migrated_path, weights_only=True)
        assert ckpt["checkpoint_format_version"] == 2
        assert isinstance(ckpt["cfg"], dict), "cfg must be a plain dict"
        assert not isinstance(ckpt["cfg"], ModelConfig)

        # State dict keys and tensor shapes preserved.
        for k, v in model.state_dict().items():
            assert k in ckpt["state_dict"], f"Missing key {k!r} in migrated state_dict"
            assert ckpt["state_dict"][k].shape == v.shape, (
                f"Shape mismatch for {k!r}: {ckpt['state_dict'][k].shape} vs {v.shape}"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_migrate_is_idempotent_on_v2_checkpoint():
    """Migrating a v2 checkpoint produces an identical v2 checkpoint."""
    tmp_dir = tempfile.mkdtemp()
    try:
        cfg, model = _tiny_model()
        v2_path = os.path.join(tmp_dir, "v2.pt")
        torch.save(
            {
                "checkpoint_format_version": 2,
                "cfg": dataclasses.asdict(cfg),
                "state_dict": model.state_dict(),
                "metadata": {"source": "test"},
            },
            v2_path,
        )

        migrate = _load_migrate()
        migrated_path = os.path.join(tmp_dir, "v2_migrated.pt")
        migrate.migrate_checkpoint(v2_path, migrated_path)

        ckpt = torch.load(migrated_path, weights_only=True)
        assert ckpt["checkpoint_format_version"] == 2
        assert isinstance(ckpt["cfg"], dict)
        # All original cfg fields preserved.
        for k, v in dataclasses.asdict(cfg).items():
            assert ckpt["cfg"][k] == v, f"cfg field {k!r} changed after idempotent migration"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_migrate_bare_state_dict():
    """Migrating a bare state-dict checkpoint (no 'state_dict' key) writes a v2 checkpoint."""
    tmp_dir = tempfile.mkdtemp()
    try:
        _, model = _tiny_model()
        bare_path = os.path.join(tmp_dir, "bare.pt")
        torch.save(model.state_dict(), bare_path)

        migrate = _load_migrate()
        migrated_path = os.path.join(tmp_dir, "bare_migrated.pt")
        migrate.migrate_checkpoint(bare_path, migrated_path)

        ckpt = torch.load(migrated_path, weights_only=True)
        assert ckpt["checkpoint_format_version"] == 2
        assert isinstance(ckpt["cfg"], dict)
        # State dict tensors preserved.
        for k, v in model.state_dict().items():
            assert k in ckpt["state_dict"]
            assert ckpt["state_dict"][k].shape == v.shape
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_migrate_backup_created():
    """migrate_local() creates a .bak file before overwriting."""
    tmp_dir = tempfile.mkdtemp()
    try:
        cfg, model = _tiny_model()
        checkpoint_path = os.path.join(tmp_dir, "best.pt")
        torch.save({"cfg": cfg, "state_dict": model.state_dict()}, checkpoint_path)

        migrate = _load_migrate()
        migrate.migrate_local(checkpoint_path)

        backup_path = checkpoint_path + ".bak"
        assert os.path.exists(backup_path), ".bak file must be created by migrate_local()"
        # Backup must be the original (loads with weights_only=False; contains ModelConfig).
        bak = torch.load(backup_path, weights_only=False)
        assert isinstance(bak.get("cfg"), ModelConfig), (
            "Backup must preserve the original pickled ModelConfig"
        )
        # Migrated file must now be v2.
        ckpt = torch.load(checkpoint_path, weights_only=True)
        assert ckpt["checkpoint_format_version"] == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_migrate_metadata_with_primitive_values_passes():
    """migrate_checkpoint() accepts metadata containing only primitives."""
    tmp_dir = tempfile.mkdtemp()
    try:
        _, model = _tiny_model()
        path = os.path.join(tmp_dir, "with_meta.pt")
        torch.save(
            {
                "checkpoint_format_version": 2,
                "cfg": {},
                "state_dict": model.state_dict(),
                "metadata": {
                    "source": "train.py",
                    "epoch": 42,
                    "val_mse": 0.012,
                    "tags": ["v2", "final"],
                    "nested": {"a": 1, "b": None},
                },
            },
            path,
        )
        migrate = _load_migrate()
        out = os.path.join(tmp_dir, "out.pt")
        migrate.migrate_checkpoint(path, out)
        ckpt = torch.load(out, weights_only=True)
        assert ckpt["checkpoint_format_version"] == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_metadata_with_nonprimitive_raises():
    """_validate_metadata() rejects metadata containing custom objects."""
    migrate = _load_migrate()

    class _Custom:
        pass

    with pytest.raises(ValueError, match="Non-primitive"):
        migrate._validate_metadata({"bad": _Custom()})

    # Nested non-primitive is also rejected.
    with pytest.raises(ValueError, match="Non-primitive"):
        migrate._validate_metadata({"nested": {"deep": _Custom()}})


def test_evaluate_load_model_uses_migrated_checkpoint(monkeypatch):
    """evaluate.load_model() can instantiate a model from a migrated v2 checkpoint."""
    import evaluate

    tmp_dir = tempfile.mkdtemp()
    try:
        cfg, model = _tiny_model()
        legacy_path = os.path.join(tmp_dir, "legacy.pt")
        torch.save({"cfg": cfg, "state_dict": model.state_dict()}, legacy_path)

        migrate = _load_migrate()
        migrated_path = os.path.join(tmp_dir, "migrated.pt")
        migrate.migrate_checkpoint(legacy_path, migrated_path)

        # Patch evaluate so it uses the local path without a Snowflake download.
        monkeypatch.setattr(evaluate, "CHECKPOINT_STAGE", "@FAKE_STAGE/checkpoints")
        monkeypatch.setattr(os.path, "exists", lambda p: True)

        loaded_model = evaluate.load_model(migrated_path)
        assert isinstance(loaded_model, DeepSetModel)
        # Architecture must match the original config.
        assert loaded_model.cfg.d_phi == cfg.d_phi
        assert loaded_model.cfg.d_rho == cfg.d_rho
        assert loaded_model.cfg.pool == cfg.pool
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
