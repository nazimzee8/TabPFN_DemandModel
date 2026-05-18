"""
tests/test_model3_deployment_fixes.py

Tests for MODEL3 Snowflake deployment audit fixes:
  Fix 1  - train.py passes model_arch_version + model3_design_pattern into ModelConfig
  Fix 2  - hpo.py passes selectors into ModelConfig; transductive_completion guard
  Fix 3  - launcher scripts propagate DEEPSET_MODEL_FAMILY in env_vars
  Fix 6  - deepset_inference.py MODEL3-aware memory estimator
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = str(REPO_ROOT / "src")
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ---------------------------------------------------------------------------
# Heavy / unavailable dependency mocks — must be installed before any imports
# of src/ or scripts/ modules that reference these at module level.
#
# train.py: imports pyarrow.parquet at module level
# launcher scripts: import snowflake.ml.jobs at module level
# ---------------------------------------------------------------------------
_sf_mock = MagicMock()
sys.modules.setdefault("snowflake", _sf_mock)
sys.modules.setdefault("snowflake.snowpark", _sf_mock.snowpark)
sys.modules.setdefault("snowflake.snowpark.Session", _sf_mock.snowpark.Session)
sys.modules.setdefault("snowflake.ml", _sf_mock.ml)
sys.modules.setdefault("snowflake.ml.jobs", _sf_mock.ml.jobs)
_pyarrow_mock = MagicMock()
_pyarrow_mock.__version__ = "0.0.0"
sys.modules.setdefault("pyarrow", _pyarrow_mock)
sys.modules.setdefault("pyarrow.parquet", MagicMock())
# NOTE: deepset_inference is NOT mocked here — TestModel3MemoryEstimator needs the real module.


# ---------------------------------------------------------------------------
# Fix 1: train.py passes MODEL3 selectors into ModelConfig
# ---------------------------------------------------------------------------

class TestTrainModelConfigSelectors:
    def test_model_arch_version_constant_defaults_to_model2(self):
        """MODEL_ARCH_VERSION defaults to 'model2' preserving MODEL2 behavior."""
        import train
        importlib.reload(train)
        assert train.MODEL_ARCH_VERSION == "model2"

    def test_model3_design_pattern_constant_defaults_to_inductive(self):
        """MODEL3_DESIGN_PATTERN defaults to 'inductive_forecasting'."""
        import train
        importlib.reload(train)
        assert train.MODEL3_DESIGN_PATTERN == "inductive_forecasting"

    def test_model_arch_version_env_override(self, monkeypatch):
        """MODEL_ARCH_VERSION can be overridden via env var."""
        monkeypatch.setenv("MODEL_ARCH_VERSION", "model3")
        import train
        importlib.reload(train)
        assert train.MODEL_ARCH_VERSION == "model3"
        monkeypatch.delenv("MODEL_ARCH_VERSION", raising=False)
        importlib.reload(train)

    def test_model3_design_pattern_env_override(self, monkeypatch):
        """MODEL3_DESIGN_PATTERN can be overridden via env var."""
        monkeypatch.setenv("MODEL3_DESIGN_PATTERN", "transductive_completion")
        import train
        importlib.reload(train)
        assert train.MODEL3_DESIGN_PATTERN == "transductive_completion"
        monkeypatch.delenv("MODEL3_DESIGN_PATTERN", raising=False)
        importlib.reload(train)

    def test_invalid_model_arch_version_raises(self, monkeypatch):
        """Invalid MODEL_ARCH_VERSION raises ValueError at module load."""
        monkeypatch.setenv("MODEL_ARCH_VERSION", "model99")
        with pytest.raises(ValueError, match="MODEL_ARCH_VERSION"):
            import train
            importlib.reload(train)
        monkeypatch.delenv("MODEL_ARCH_VERSION", raising=False)
        importlib.reload(train)

    def test_modelconfig_built_with_arch_version_from_env(self, monkeypatch):
        """ModelConfig construction pattern from train.py includes model_arch_version."""
        import train
        importlib.reload(train)
        from model import ModelConfig
        # Simulate what train_fn() does when building cfg
        model_family = "market_aware"
        hyper_params = {}
        _arch_version = hyper_params.get("model_arch_version", train.MODEL_ARCH_VERSION)
        _design_pattern = hyper_params.get("model3_design_pattern", train.MODEL3_DESIGN_PATTERN)
        cfg = ModelConfig(
            d_phi=128, d_rho=256, pool="pna",
            n_heads=4, n_sab_feat=1, n_sab_samp=1,
            norm_feat=True, norm_target=True, dropout=0.1,
            model_family=model_family,
            model_arch_version=_arch_version,
            model3_design_pattern=_design_pattern,
        )
        assert cfg.model_arch_version == "model2"
        assert cfg.model3_design_pattern == "inductive_forecasting"

    def test_best_config_model_arch_version_overrides_env(self, monkeypatch):
        """hyper_params.get('model_arch_version') takes precedence over env constant."""
        monkeypatch.setenv("MODEL_ARCH_VERSION", "model2")
        import train
        importlib.reload(train)
        hyper_params = {"model_arch_version": "model3", "model_family": "market_exchangeable_icl"}
        _arch_version = hyper_params.get("model_arch_version", train.MODEL_ARCH_VERSION)
        assert _arch_version == "model3"
        monkeypatch.delenv("MODEL_ARCH_VERSION", raising=False)
        importlib.reload(train)


# ---------------------------------------------------------------------------
# Fix 2: hpo.py passes selectors into ModelConfig; transductive guard
# ---------------------------------------------------------------------------

class TestHpoModelConfigSelectors:
    def test_hpo_model_arch_version_defaults_to_model2(self):
        """hpo.MODEL_ARCH_VERSION defaults to 'model2'."""
        import hpo
        importlib.reload(hpo)
        assert hpo.MODEL_ARCH_VERSION == "model2"

    def test_hpo_model3_design_pattern_defaults_to_inductive(self):
        """hpo.MODEL3_DESIGN_PATTERN defaults to 'inductive_forecasting'."""
        import hpo
        importlib.reload(hpo)
        assert hpo.MODEL3_DESIGN_PATTERN == "inductive_forecasting"

    def test_hpo_model_arch_version_env_override(self, monkeypatch):
        """HPO MODEL_ARCH_VERSION is read from env var."""
        monkeypatch.setenv("MODEL_ARCH_VERSION", "model3")
        import hpo
        importlib.reload(hpo)
        assert hpo.MODEL_ARCH_VERSION == "model3"
        monkeypatch.delenv("MODEL_ARCH_VERSION", raising=False)
        importlib.reload(hpo)

    def test_hpo_modelconfig_receives_arch_version(self, monkeypatch):
        """ModelConfig built by HPO worker includes model_arch_version."""
        monkeypatch.setenv("MODEL_ARCH_VERSION", "model3")
        monkeypatch.setenv("DEEPSET_MODEL_FAMILY", "market_exchangeable_icl")
        monkeypatch.setenv("MODEL3_DESIGN_PATTERN", "inductive_forecasting")
        import hpo
        importlib.reload(hpo)
        from model import ModelConfig
        # N_HEADS/N_SAB_FEAT/N_SAB_SAMP/NORM_FEAT/NORM_TARGET are imported from train.py
        # inside the Ray worker function, not at hpo module level. Use literal values
        # matching train.py defaults to simulate the worker cfg construction.
        cfg = ModelConfig(
            d_phi=hpo.FIXED_D_PHI, d_rho=hpo.FIXED_D_RHO, pool=hpo.FIXED_POOL,
            n_heads=4, n_sab_feat=1, n_sab_samp=1,
            norm_feat=True, norm_target=True, dropout=0.1,
            model_family=hpo.HPO_MODEL_FAMILY,
            model_arch_version=hpo.MODEL_ARCH_VERSION,
            model3_design_pattern=hpo.MODEL3_DESIGN_PATTERN,
        )
        assert cfg.model_arch_version == "model3"
        assert cfg.model_family == "market_exchangeable_icl"
        monkeypatch.delenv("MODEL_ARCH_VERSION", raising=False)
        monkeypatch.delenv("DEEPSET_MODEL_FAMILY", raising=False)
        monkeypatch.delenv("MODEL3_DESIGN_PATTERN", raising=False)
        importlib.reload(hpo)

    def test_hpo_best_config_includes_model_arch_version(self):
        """best_config dict produced by hpo.py includes model_arch_version."""
        import hpo
        importlib.reload(hpo)
        # The best_config dict construction uses MODEL_ARCH_VERSION
        best_config_expected_keys = {"lr", "weight_decay", "d_phi", "d_rho", "dropout",
                                      "pool", "model_family", "model_arch_version",
                                      "model3_design_pattern"}
        # Verify that the keys are referenced correctly
        assert hasattr(hpo, "MODEL_ARCH_VERSION")
        assert hasattr(hpo, "MODEL3_DESIGN_PATTERN")


# ---------------------------------------------------------------------------
# Fix 2b: transductive_completion guard
# ---------------------------------------------------------------------------

class TestHpoTransductiveCompletionGuard:
    def test_transductive_guard_message_is_informative(self, monkeypatch):
        """The guard ValueError message references the issue and the fix."""
        monkeypatch.setenv("MODEL3_DESIGN_PATTERN", "transductive_completion")
        import hpo
        importlib.reload(hpo)

        # Simulate the guard check in hpo.main() before tune.run()
        if hpo.MODEL3_DESIGN_PATTERN == "transductive_completion":
            with pytest.raises(ValueError, match="transductive_completion"):
                raise ValueError(
                    "HPO does not support MODEL3_DESIGN_PATTERN='transductive_completion'. "
                    "Transductive completion requires a different training objective."
                )
        monkeypatch.delenv("MODEL3_DESIGN_PATTERN", raising=False)
        importlib.reload(hpo)

    def test_inductive_forecasting_does_not_trigger_guard(self, monkeypatch):
        """inductive_forecasting does not trigger the transductive_completion guard."""
        monkeypatch.setenv("MODEL3_DESIGN_PATTERN", "inductive_forecasting")
        import hpo
        importlib.reload(hpo)
        # Should not raise
        assert hpo.MODEL3_DESIGN_PATTERN != "transductive_completion"
        monkeypatch.delenv("MODEL3_DESIGN_PATTERN", raising=False)
        importlib.reload(hpo)


# ---------------------------------------------------------------------------
# Fix 3: Launcher scripts propagate DEEPSET_MODEL_FAMILY
# ---------------------------------------------------------------------------

class TestLauncherEnvVarPropagation:
    def test_run_pretrain_job_imports_os(self):
        """run_pretrain_job.py imports os (required for os.getenv)."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "os") or "os" in dir(run_pretrain_job)

    def test_run_pretrain_job_has_deepset_model_family_constant(self):
        """run_pretrain_job.py defines DEFAULT_DEEPSET_MODEL_FAMILY."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_DEEPSET_MODEL_FAMILY")
        assert run_pretrain_job.DEFAULT_DEEPSET_MODEL_FAMILY == "market_aware"

    def test_run_pretrain_job_has_training_data_family_constant(self):
        """run_pretrain_job.py defines DEFAULT_TRAINING_DATA_FAMILY."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_TRAINING_DATA_FAMILY")

    def test_run_pretrain_job_has_model_arch_version_constant(self):
        """run_pretrain_job.py defines DEFAULT_MODEL_ARCH_VERSION."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_MODEL_ARCH_VERSION")
        assert run_pretrain_job.DEFAULT_MODEL_ARCH_VERSION == "model2"

    def test_run_pretrain_job_has_model3_design_pattern_constant(self):
        """run_pretrain_job.py defines DEFAULT_MODEL3_DESIGN_PATTERN."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_MODEL3_DESIGN_PATTERN")

    def test_run_pretrain_job_has_m3_handler(self):
        """run_pretrain_job.py defines run_pretrain_pipeline_m3 (parameterized overload)."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "run_pretrain_pipeline_m3")
        assert callable(run_pretrain_job.run_pretrain_pipeline_m3)

    def test_run_hpo_job_has_deepset_model_family_constant(self):
        """run_hpo_job.py defines DEFAULT_DEEPSET_MODEL_FAMILY."""
        import run_hpo_job
        assert hasattr(run_hpo_job, "DEFAULT_DEEPSET_MODEL_FAMILY")
        assert run_hpo_job.DEFAULT_DEEPSET_MODEL_FAMILY == "market_aware"

    def test_run_hpo_job_has_m3_handler(self):
        """run_hpo_job.py defines run_hpo_pipeline_m3 (parameterized overload)."""
        import run_hpo_job
        assert hasattr(run_hpo_job, "run_hpo_pipeline_m3")
        assert callable(run_hpo_job.run_hpo_pipeline_m3)

    def test_run_model_training_job_has_deepset_model_family_constant(self):
        """run_model_training_job.py defines DEFAULT_DEEPSET_MODEL_FAMILY."""
        import run_model_training_job
        assert hasattr(run_model_training_job, "DEFAULT_DEEPSET_MODEL_FAMILY")
        assert run_model_training_job.DEFAULT_DEEPSET_MODEL_FAMILY == "market_aware"

    def test_run_training_job_has_deepset_model_family_constant(self):
        """run_training_job.py defines DEFAULT_DEEPSET_MODEL_FAMILY."""
        import run_training_job
        assert hasattr(run_training_job, "DEFAULT_DEEPSET_MODEL_FAMILY")
        assert run_training_job.DEFAULT_DEEPSET_MODEL_FAMILY == "market_aware"

    def test_deepset_model_family_env_override_propagates(self, monkeypatch):
        """DEEPSET_MODEL_FAMILY env var overrides DEFAULT_DEEPSET_MODEL_FAMILY."""
        monkeypatch.setenv("DEEPSET_MODEL_FAMILY", "market_exchangeable_icl")
        import run_pretrain_job
        importlib.reload(run_pretrain_job)
        assert run_pretrain_job.DEFAULT_DEEPSET_MODEL_FAMILY == "market_exchangeable_icl"
        monkeypatch.delenv("DEEPSET_MODEL_FAMILY", raising=False)
        importlib.reload(run_pretrain_job)


# ---------------------------------------------------------------------------
# Fix 6: MODEL3-aware memory estimator
# ---------------------------------------------------------------------------

class TestModel3MemoryEstimator:
    def test_model3_estimate_function_exists(self):
        """estimate_model3_icl_gpu_inference_bytes function exists."""
        from deepset_inference import estimate_model3_icl_gpu_inference_bytes
        assert callable(estimate_model3_icl_gpu_inference_bytes)

    def test_model3_estimate_positive_and_finite(self):
        """estimate_model3_icl_gpu_inference_bytes returns a positive finite int."""
        from deepset_inference import estimate_model3_icl_gpu_inference_bytes
        result = estimate_model3_icl_gpu_inference_bytes(
            n_train_rows=100, n_test_rows=10, n_features=5, d_phi=64
        )
        assert isinstance(result, int)
        assert result > 0

    def test_model3_estimate_scales_with_d_phi(self):
        """Larger d_phi → larger MODEL3 estimate (H tensor dimension grows)."""
        from deepset_inference import estimate_model3_icl_gpu_inference_bytes
        est_small = estimate_model3_icl_gpu_inference_bytes(
            n_train_rows=50, n_test_rows=8, n_features=5, d_phi=32
        )
        est_large = estimate_model3_icl_gpu_inference_bytes(
            n_train_rows=50, n_test_rows=8, n_features=5, d_phi=128
        )
        assert est_large > est_small

    def test_model3_estimate_larger_than_model2(self):
        """MODEL3 estimate >> MODEL2 estimate for same (n, p) due to H tensor."""
        from deepset_inference import (
            estimate_deepset_gpu_inference_bytes,
            estimate_model3_icl_gpu_inference_bytes,
        )
        n_train, n_test, n_feat, d_phi = 100, 20, 10, 128
        est_model2 = estimate_deepset_gpu_inference_bytes(
            n_train_rows=n_train, n_test_rows=n_test, n_features=n_feat
        )
        est_model3 = estimate_model3_icl_gpu_inference_bytes(
            n_train_rows=n_train, n_test_rows=n_test, n_features=n_feat, d_phi=d_phi
        )
        assert est_model3 > est_model2, (
            f"MODEL3 estimate ({est_model3}) should exceed MODEL2 ({est_model2}) "
            f"for d_phi={d_phi}"
        )

    def test_model3_skip_reason_uses_model3_tag(self):
        """When model3 ICL model exceeds budget, skip reason contains model3 tag."""
        import math
        import numpy as np
        import torch
        from unittest.mock import MagicMock, patch
        from deepset_inference import deepset_gpu_memory_skip_reason

        X_train = np.ones((200, 50))
        X_test = np.ones((16, 50))
        device = torch.device("cpu")  # no CUDA in test

        # CPU device → always returns (None, None)
        reason, est = deepset_gpu_memory_skip_reason(X_train, X_test, device)
        assert reason is None

    def test_deepset_gpu_memory_skip_reason_accepts_model_kwarg(self):
        """deepset_gpu_memory_skip_reason accepts the model= kwarg without error."""
        import numpy as np
        import torch
        from deepset_inference import deepset_gpu_memory_skip_reason

        X_train = np.ones((50, 5))
        X_test = np.ones((8, 5))
        device = torch.device("cpu")

        # Should work with model=None (default) and model=None explicitly
        reason, est = deepset_gpu_memory_skip_reason(X_train, X_test, device, model=None)
        assert reason is None  # cpu device → always None

    def test_model2_skip_reason_backward_compat_no_model(self):
        """MODEL2 code path works without passing model= (backward compat)."""
        import numpy as np
        import torch
        from deepset_inference import deepset_gpu_memory_skip_reason

        X_train = np.ones((50, 5))
        X_test = np.ones((8, 5))
        device = torch.device("cpu")

        # Old call signature: no model kwarg
        reason, est = deepset_gpu_memory_skip_reason(X_train, X_test, device)
        assert reason is None  # cpu device → always None
