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

from model import ModelConfig, DeepSetICLModel  # noqa: E402


def _load_migrate():
    spec = importlib.util.spec_from_file_location(
        "migrate_checkpoint",
        os.path.join(SCRIPTS_DIR, "migrate_checkpoint.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model():
    cfg = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=16, d_rho=32, pool="mean", n_sab_feat=1,
    )
    model = DeepSetICLModel(cfg=cfg)
    return cfg, model


# ---------------------------------------------------------------------------
# Retired-family rejection tests (normalize_checkpoint_cfg)
# ---------------------------------------------------------------------------

def test_normalize_cfg_rejects_deepset_family():
    """normalize_checkpoint_cfg raises RuntimeError for retired model_family='deepset'."""
    sys.path.insert(0, SRC_DIR)

    # Patch heavy imports for evaluate_synthetic_regression
    import unittest.mock as mock
    _snowflake_mock = mock.MagicMock()
    for mod in ("snowflake", "snowflake.snowpark", "snowflake.snowpark.Session",
                "matplotlib", "matplotlib.pyplot", "pyarrow", "pyarrow.parquet",
                "autogluon_models", "baseline_models"):
        if mod not in sys.modules:
            sys.modules[mod] = mock.MagicMock()

    from evaluate_synthetic_regression import normalize_checkpoint_cfg

    payload = {"cfg": {"model_family": "deepset", "d_phi": 64, "d_rho": 128,
                       "pool": "pna", "n_heads": 4, "n_sab_feat": 1,
                       "norm_feat": True, "norm_target": True, "dropout": 0.0}}
    with pytest.raises(RuntimeError, match="retired"):
        normalize_checkpoint_cfg(payload)


def test_normalize_cfg_rejects_market_aware_family():
    """normalize_checkpoint_cfg raises RuntimeError for retired model_family='market_aware'."""
    sys.path.insert(0, SRC_DIR)

    import unittest.mock as mock
    for mod in ("snowflake", "snowflake.snowpark", "snowflake.snowpark.Session",
                "matplotlib", "matplotlib.pyplot", "pyarrow", "pyarrow.parquet",
                "autogluon_models", "baseline_models"):
        if mod not in sys.modules:
            sys.modules[mod] = mock.MagicMock()

    from evaluate_synthetic_regression import normalize_checkpoint_cfg

    payload = {"cfg": {"model_family": "market_aware", "d_phi": 64, "d_rho": 128,
                       "pool": "pna", "n_heads": 4, "n_sab_feat": 1,
                       "norm_feat": True, "norm_target": True, "dropout": 0.0}}
    with pytest.raises(RuntimeError, match="retired"):
        normalize_checkpoint_cfg(payload)


def test_normalize_cfg_accepts_model3_icl_family():
    """normalize_checkpoint_cfg returns ModelConfig for market_exchangeable_icl."""
    sys.path.insert(0, SRC_DIR)

    import unittest.mock as mock
    for mod in ("snowflake", "snowflake.snowpark", "snowflake.snowpark.Session",
                "matplotlib", "matplotlib.pyplot", "pyarrow", "pyarrow.parquet",
                "autogluon_models", "baseline_models"):
        if mod not in sys.modules:
            sys.modules[mod] = mock.MagicMock()

    from evaluate_synthetic_regression import normalize_checkpoint_cfg

    payload = {"cfg": {
        "model_family": "market_exchangeable_icl",
        "model_arch_version": "model3",
        "model_design_pattern": "inductive_forecasting",
        "d_phi": 64, "d_rho": 128, "pool": "pna",
        "n_heads": 4, "n_sab_feat": 1,
        "norm_feat": True, "norm_target": True, "dropout": 0.0,
    }}
    cfg = normalize_checkpoint_cfg(payload)
    assert cfg.model_family == "market_exchangeable_icl"


def test_normalize_cfg_strips_legacy_fields():
    """normalize_checkpoint_cfg strips legacy fields before creating ModelConfig."""
    sys.path.insert(0, SRC_DIR)

    import unittest.mock as mock
    for mod in ("snowflake", "snowflake.snowpark", "snowflake.snowpark.Session",
                "matplotlib", "matplotlib.pyplot", "pyarrow", "pyarrow.parquet",
                "autogluon_models", "baseline_models"):
        if mod not in sys.modules:
            sys.modules[mod] = mock.MagicMock()

    from evaluate_synthetic_regression import normalize_checkpoint_cfg

    payload = {"cfg": {
        "model_family": "market_exchangeable_icl",
        "model_arch_version": "model3",
        "model_design_pattern": "inductive_forecasting",
        "d_phi": 64, "d_rho": 128, "pool": "pna",
        "n_heads": 4, "n_sab_feat": 1,
        "norm_feat": True, "norm_target": True, "dropout": 0.0,
        # legacy fields that should be stripped
        "n_sab_samp": 1,
        "feature_aggregation_order": "legacy_feature_pool_first",
        "d_sample": 64,
        "n_sab_sample_per_feature": 0,
        "sample_pool": "attn",
        "residual_scale_init": 0.1,
    }}
    # Should not raise despite legacy fields
    cfg = normalize_checkpoint_cfg(payload)
    assert cfg.model_family == "market_exchangeable_icl"


def test_migrate_metadata_with_primitive_values_passes():
    """migrate_checkpoint() accepts metadata containing only primitives."""
    tmp_dir = tempfile.mkdtemp()
    try:
        _, model = _tiny_model()
        path = os.path.join(tmp_dir, "with_meta.pt")
        torch.save(
            {
                "checkpoint_format_version": 4,
                "cfg": {},
                "state_dict": model.state_dict(),
                "metadata": {
                    "source": "train.py",
                    "epoch": 42,
                    "val_mse": 0.012,
                    "tags": ["v4", "final"],
                    "nested": {"a": 1, "b": None},
                },
            },
            path,
        )
        migrate = _load_migrate()
        out = os.path.join(tmp_dir, "out.pt")
        migrate.migrate_checkpoint(path, out)
        ckpt = torch.load(out, weights_only=True)
        assert ckpt["checkpoint_format_version"] in (2, 4)
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
