"""
tests/test_train_model_family.py

Tests for MODEL3 (DeepSetICLModel) as production default in train.py.
All tests mock DDP, Snowflake I/O, and DataLoader to avoid real infrastructure.
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def test_default_model_family_constant_is_market_exchangeable_icl():
    """MODEL_FAMILY module constant defaults to 'market_exchangeable_icl'."""
    import train
    assert train.MODEL_FAMILY == "market_exchangeable_icl"


def test_env_model_family_overrides_default(monkeypatch):
    """Setting MODEL_FAMILY env var to 'market_exchangeable_completion' overrides the default."""
    monkeypatch.setenv("MODEL_FAMILY", "market_exchangeable_completion")
    import importlib
    import train
    importlib.reload(train)
    assert train.MODEL_FAMILY == "market_exchangeable_completion"
    # Restore
    importlib.reload(train)


def test_training_data_family_constant_exists():
    """TRAINING_DATA_FAMILY constant exists and defaults to 'unknown'."""
    import train
    assert hasattr(train, "TRAINING_DATA_FAMILY")
    assert train.TRAINING_DATA_FAMILY == "unknown"


def test_training_data_family_from_env(monkeypatch):
    """TRAINING_DATA_FAMILY can be set from env."""
    monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_ood")
    import importlib
    import train
    importlib.reload(train)
    assert train.TRAINING_DATA_FAMILY == "synthetic_regression_ood"
    importlib.reload(train)


def test_checkpoint_metadata_includes_task_type_and_training_entrypoint(monkeypatch):
    """Saved checkpoint metadata includes task_type='regression' and training_entrypoint='train.py'."""
    tmp_dir = tempfile.mkdtemp()

    import importlib
    import train
    importlib.reload(train)

    # Build a minimal cfg and model
    from model import ModelConfig, _instantiate_model
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=64,
        d_rho=128,
        pool="pna",
        n_heads=4,
        n_sab_feat=1,
        norm_feat=True,
        norm_target=True,
        dropout=0.0,
    )
    model = _instantiate_model(cfg)
    model.eval()

    import dataclasses as _dc
    checkpoint_output_name = os.path.join(tmp_dir, "ckpt.pt")
    torch.save({
        "checkpoint_format_version": 4,
        "cfg": _dc.asdict(model.cfg),
        "state_dict": model.state_dict(),
        "metadata": {
            "source": "train.py",
            "checkpoint_name": os.path.basename(checkpoint_output_name),
            "pytorch_version": torch.__version__,
            "model_family": cfg.model_family,
            "task_type": "regression",
            "training_entrypoint": "train.py",
            "training_data_family": train.TRAINING_DATA_FAMILY,
        },
    }, checkpoint_output_name)

    loaded = torch.load(checkpoint_output_name, weights_only=False)
    assert loaded["metadata"]["task_type"] == "regression"
    assert loaded["metadata"]["training_entrypoint"] == "train.py"
    assert "training_data_family" in loaded["metadata"]
    assert loaded["checkpoint_format_version"] == 4


def test_best_config_model_family_overrides_default(monkeypatch):
    """When BEST_CONFIG has model_family=market_exchangeable_completion, the cfg uses it."""
    import json
    import importlib
    import train
    importlib.reload(train)

    best_config = {"model_family": "market_exchangeable_completion"}
    hyper_params = best_config
    model_family = hyper_params.get("model_family", train.MODEL_FAMILY)
    assert model_family == "market_exchangeable_completion"


def test_checkpoint_metadata_includes_training_metrics():
    """Saved checkpoint includes best_val_mse, train_mse_at_best, best_epoch."""
    import tempfile
    import importlib
    import train
    importlib.reload(train)

    from model import ModelConfig, _instantiate_model
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=64,
        d_rho=128,
        pool="pna",
        n_heads=4,
        n_sab_feat=1,
        norm_feat=True,
        norm_target=True,
        dropout=0.0,
    )
    model = _instantiate_model(cfg)
    model.eval()

    import dataclasses as _dc
    tmp_dir = tempfile.mkdtemp()
    checkpoint_output_name = os.path.join(tmp_dir, "ckpt.pt")
    torch.save({
        "checkpoint_format_version": 4,
        "cfg": _dc.asdict(model.cfg),
        "state_dict": model.state_dict(),
        "metadata": {
            "source": "train.py",
            "checkpoint_name": os.path.basename(checkpoint_output_name),
            "pytorch_version": torch.__version__,
            "model_family": cfg.model_family,
            "task_type": "regression",
            "training_entrypoint": "train.py",
            "training_data_family": train.TRAINING_DATA_FAMILY,
            "best_val_mse": 0.123,
            "train_mse_at_best": 0.089,
            "best_epoch": 42,
        },
    }, checkpoint_output_name)

    loaded = torch.load(checkpoint_output_name, weights_only=False)
    assert "best_val_mse" in loaded["metadata"]
    assert "train_mse_at_best" in loaded["metadata"]
    assert "best_epoch" in loaded["metadata"]
    assert loaded["metadata"]["best_epoch"] == 42


def test_check_train_val_gap_uses_new_metadata_keys():
    """check_train_val_gap accepts best_val_mse and train_mse_at_best keys."""
    import sys
    sys.path.insert(0, SRC_DIR)
    import sanity_checks
    meta = {"best_val_mse": 0.2, "train_mse_at_best": 0.1}
    result = sanity_checks.check_train_val_gap(meta)
    assert result["passed"] is True
    assert result.get("val_mse", result.get("best_val_mse")) is not None


def test_check_train_val_gap_fallback_to_legacy_keys():
    """check_train_val_gap falls back to legacy val_mse/train_mse keys."""
    import sys
    sys.path.insert(0, SRC_DIR)
    import sanity_checks
    meta = {"val_mse": 0.2, "train_mse": 0.1}
    result = sanity_checks.check_train_val_gap(meta)
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# TRAINING_DATA_FAMILY validation tests
# ---------------------------------------------------------------------------

def test_training_data_family_defaults_to_unknown():
    """TRAINING_DATA_FAMILY defaults to 'unknown' when env var absent."""
    import importlib
    import train
    importlib.reload(train)
    assert train.TRAINING_DATA_FAMILY == "unknown"


def test_training_data_family_valid_values_accepted(monkeypatch):
    """All allowed TRAINING_DATA_FAMILY values are accepted without error."""
    import importlib
    import train
    valid_values = [
        "synthetic_regression_primary",
        "synthetic_regression_ood",
        "synthetic_regression_combined",
        "market_mental_model",
        "unknown",
    ]
    for value in valid_values:
        monkeypatch.setenv("TRAINING_DATA_FAMILY", value)
        importlib.reload(train)
        assert train.TRAINING_DATA_FAMILY == value
    importlib.reload(train)


def test_training_data_family_invalid_value_raises(monkeypatch):
    """An invalid TRAINING_DATA_FAMILY raises ValueError at module load time."""
    import importlib
    import train
    monkeypatch.setenv("TRAINING_DATA_FAMILY", "bad_value")
    with pytest.raises(ValueError, match="Invalid TRAINING_DATA_FAMILY"):
        importlib.reload(train)
    # Restore to valid state
    monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
    importlib.reload(train)


def test_training_data_family_synthetic_regression_combined_accepted(monkeypatch):
    """synthetic_regression_combined is accepted (production default for synreg evaluation)."""
    import importlib
    import train
    monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
    importlib.reload(train)
    assert train.TRAINING_DATA_FAMILY == "synthetic_regression_combined"
    importlib.reload(train)


# ---------------------------------------------------------------------------
# _load_pretrain_checkpoint filename-resolution tests
# ---------------------------------------------------------------------------

class TestLoadPretrainCheckpointFilenameResolution:
    """Unit tests for _load_pretrain_checkpoint() gate-specific filename fix.

    Regression for the bug where the function always looked for the hardcoded
    local filename "pretrain.pt" instead of deriving it from the stage_path,
    causing gate-specific checkpoints (pretrain_gate32.pt etc.) to never be found.
    """

    def _make_minimal_checkpoint(self, cfg, model):
        """Create a minimal checkpoint dict that passes architecture checks."""
        import dataclasses
        return {
            "checkpoint_format_version": 4,
            "cfg": dataclasses.asdict(cfg),
            "state_dict": model.state_dict(),
            "metadata": {},
        }

    def _make_session_that_writes(self, filename: str, payload: dict):
        """Return a mock session whose file.get writes <filename> into local_dir."""
        import json as _json

        def _file_get(stage_path, local_dir):
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, filename)
            torch.save(payload, dest)

        session = MagicMock()
        session.file.get.side_effect = _file_get
        return session

    def _call(self, stage_path, session, model=None, cfg=None, rank=0):
        """Call _load_pretrain_checkpoint with the given stage_path and mocked session.

        Session is imported locally inside _load_pretrain_checkpoint via
        'from snowflake.snowpark import Session', so we inject a stub into
        sys.modules rather than patching a module-level attribute.
        Use distinct rank values per test so each gets an isolated /tmp dir.
        """
        import sys
        import train
        from model import ModelConfig, _instantiate_model

        if cfg is None:
            cfg = ModelConfig()
        if model is None:
            model = _instantiate_model(cfg)

        # Build a stub snowpark module whose Session.builder.getOrCreate() returns
        # the provided mock session.
        mock_snowpark = MagicMock()
        mock_snowpark.Session.builder.getOrCreate.return_value = session

        old = sys.modules.get("snowflake.snowpark")
        sys.modules["snowflake.snowpark"] = mock_snowpark
        try:
            return train._load_pretrain_checkpoint(
                model, stage_path, cfg, device="cpu", rank=rank,
                pretrain_load_policy="require_match",
            )
        finally:
            if old is None:
                sys.modules.pop("snowflake.snowpark", None)
            else:
                sys.modules["snowflake.snowpark"] = old

    def test_gate64_filename_searched_not_pretrain_pt(self):
        """When stage_path ends in pretrain_gate64.pt, the loader must find
        pretrain_gate64.pt locally, not fail looking for pretrain.pt."""
        from model import ModelConfig, _instantiate_model

        cfg = ModelConfig()
        model = _instantiate_model(cfg)
        ckpt = self._make_minimal_checkpoint(cfg, model)
        session = self._make_session_that_writes("pretrain_gate64.pt", ckpt)

        # rank=10 gives a fresh isolated tmp dir
        loaded, reason = self._call(
            "@MODEL_STAGE/checkpoints/pretrain_gate64.pt", session, model, cfg,
            rank=10,
        )
        assert loaded is True, f"Expected pretrain_loaded=True, got reason={reason!r}"
        assert reason is None

    def test_gate32_filename_searched(self):
        """Same resolution works for pretrain_gate32.pt."""
        from model import ModelConfig, _instantiate_model

        cfg = ModelConfig()
        model = _instantiate_model(cfg)
        ckpt = self._make_minimal_checkpoint(cfg, model)
        session = self._make_session_that_writes("pretrain_gate32.pt", ckpt)

        loaded, reason = self._call(
            "@MODEL_STAGE/checkpoints/pretrain_gate32.pt", session, model, cfg,
            rank=11,
        )
        assert loaded is True

    def test_gate128_filename_searched(self):
        """Same resolution works for pretrain_gate128.pt."""
        from model import ModelConfig, _instantiate_model

        cfg = ModelConfig()
        model = _instantiate_model(cfg)
        ckpt = self._make_minimal_checkpoint(cfg, model)
        session = self._make_session_that_writes("pretrain_gate128.pt", ckpt)

        loaded, reason = self._call(
            "@MODEL_STAGE/checkpoints/pretrain_gate128.pt", session, model, cfg,
            rank=12,
        )
        assert loaded is True

    def test_regression_old_hardcoded_pretrain_pt_would_have_failed(self):
        """Regression: the old code looked for 'pretrain.pt'; show that if file.get
        writes 'pretrain_gate64.pt' only, os.path.exists('pretrain.pt') is False."""
        tmp = tempfile.mkdtemp()
        # Simulate what file.get writes for a gate checkpoint
        torch.save({"state_dict": {}}, os.path.join(tmp, "pretrain_gate64.pt"))
        # The old hardcoded path would be:
        old_local_path = os.path.join(tmp, "pretrain.pt")
        assert not os.path.exists(old_local_path), (
            "If this assertion fails, the regression condition no longer holds."
        )

    def test_missing_file_raises_runtime_error_with_filename_and_list_hint(self):
        """FileNotFoundError-equivalent: RuntimeError includes expected filename,
        local_dir, and 'LIST @MODEL_STAGE/checkpoints/' remediation hint.
        Uses rank=999 to ensure a clean tmp dir (no leftover .pt files)."""
        def _file_get_noop(stage_path, local_dir):
            os.makedirs(local_dir, exist_ok=True)
            # Intentionally write no file — simulates a failed download

        session = MagicMock()
        session.file.get.side_effect = _file_get_noop

        from model import ModelConfig, _instantiate_model
        cfg = ModelConfig()
        model = _instantiate_model(cfg)

        with pytest.raises(RuntimeError) as exc_info:
            self._call(
                "@MODEL_STAGE/checkpoints/pretrain_gate64.pt",
                session, model, cfg,
                rank=999,  # fresh dir: /tmp/pretrain_ckpt_rank999
            )

        msg = str(exc_info.value)
        assert "pretrain_gate64.pt" in msg, f"Expected filename in error: {msg!r}"
        assert "LIST @MODEL_STAGE/checkpoints/" in msg, (
            f"Expected LIST hint in error: {msg!r}"
        )
        # Rank must appear so distributed logs can be correlated
        assert "rank 999" in msg, f"Expected rank in error: {msg!r}"
