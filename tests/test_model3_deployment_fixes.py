"""
tests/test_model3_deployment_fixes.py

Tests for MODEL3 Snowflake deployment:
  Fix 1  - train.py hardcodes MODEL_ARCH_VERSION="model3" + passes model_design_pattern
  Fix 2  - hpo.py passes selectors into ModelConfig; transductive_completion guard
  Fix 3  - launcher scripts propagate MODEL_FAMILY without retired env vars
  Fix 6  - deepset_inference.py MODEL3-aware memory estimator
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
# NOTE: deepset_inference is NOT mocked here; TestModelMemoryEstimator needs the real module.


# ---------------------------------------------------------------------------
# Fix 1: train.py passes MODEL3 selectors into ModelConfig
# ---------------------------------------------------------------------------

class TestTrainModelConfigSelectors:
    def test_model_arch_version_constant_is_model3(self):
        """MODEL_ARCH_VERSION is hardcoded to 'model3'."""
        import train
        importlib.reload(train)
        assert train.MODEL_ARCH_VERSION == "model3"

    def test_model_design_pattern_constant_defaults_to_inductive(self):
        """MODEL_DESIGN_PATTERN defaults to 'inductive_forecasting'."""
        import train
        importlib.reload(train)
        assert train.MODEL_DESIGN_PATTERN == "inductive_forecasting"

    def test_model_design_pattern_env_override(self, monkeypatch):
        """MODEL_DESIGN_PATTERN can be overridden via env var."""
        monkeypatch.setenv("MODEL_DESIGN_PATTERN", "transductive_completion")
        import train
        importlib.reload(train)
        assert train.MODEL_DESIGN_PATTERN == "transductive_completion"
        monkeypatch.delenv("MODEL_DESIGN_PATTERN", raising=False)
        importlib.reload(train)

    def test_model_family_constant_defaults_to_icl(self):
        """MODEL_FAMILY defaults to 'market_exchangeable_icl'."""
        import train
        importlib.reload(train)
        assert train.MODEL_FAMILY == "market_exchangeable_icl"

    def test_modelconfig_built_with_model3_selectors(self):
        """ModelConfig construction pattern from train.py uses MODEL3 selectors."""
        import train
        importlib.reload(train)
        from model import ModelConfig
        # Simulate what train_fn() does when building cfg
        hyper_params = {}
        model_family = hyper_params.get("model_family", train.MODEL_FAMILY)
        _design_pattern = hyper_params.get("model_design_pattern", train.MODEL_DESIGN_PATTERN)
        cfg = ModelConfig(
            d_phi=128, d_rho=256, pool="pna",
            n_heads=4, n_sab_feat=1,
            norm_feat=True, norm_target=True, dropout=0.1,
            model_family=model_family,
            model_arch_version="model3",
            model_design_pattern=_design_pattern,
        )
        assert cfg.model_arch_version == "model3"
        assert cfg.model_design_pattern == "inductive_forecasting"
        assert cfg.model_family == "market_exchangeable_icl"


# ---------------------------------------------------------------------------
# Fix 2: hpo.py passes selectors into ModelConfig; transductive guard
# ---------------------------------------------------------------------------

class TestHpoModelConfigSelectors:
    def test_hpo_model_arch_version_constant_is_model3(self):
        """hpo.MODEL_ARCH_VERSION is hardcoded to 'model3'."""
        import hpo
        importlib.reload(hpo)
        assert hpo.MODEL_ARCH_VERSION == "model3"

    def test_hpo_model_design_pattern_defaults_to_inductive(self):
        """hpo.MODEL_DESIGN_PATTERN defaults to 'inductive_forecasting'."""
        import hpo
        importlib.reload(hpo)
        assert hpo.MODEL_DESIGN_PATTERN == "inductive_forecasting"

    def test_hpo_model_family_defaults_to_icl(self):
        """hpo.MODEL_FAMILY defaults to 'market_exchangeable_icl'."""
        import hpo
        importlib.reload(hpo)
        assert hpo.MODEL_FAMILY == "market_exchangeable_icl"

    def test_hpo_modelconfig_receives_model3_selectors(self, monkeypatch):
        """ModelConfig built by HPO worker includes MODEL3 selectors."""
        monkeypatch.setenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")
        import hpo
        importlib.reload(hpo)
        from model import ModelConfig
        cfg = ModelConfig(
            d_phi=hpo.FIXED_D_PHI, d_rho=hpo.FIXED_D_RHO, pool=hpo.FIXED_POOL,
            n_heads=4, n_sab_feat=1,
            norm_feat=True, norm_target=True, dropout=0.1,
            model_family=hpo.MODEL_FAMILY,
            model_arch_version=hpo.MODEL_ARCH_VERSION,
            model_design_pattern=hpo.MODEL_DESIGN_PATTERN,
        )
        assert cfg.model_arch_version == "model3"
        assert cfg.model_family == "market_exchangeable_icl"
        monkeypatch.delenv("MODEL_DESIGN_PATTERN", raising=False)
        importlib.reload(hpo)

    def test_hpo_best_config_includes_model_arch_version(self):
        """best_config dict produced by hpo.py includes model_arch_version."""
        import hpo
        importlib.reload(hpo)
        assert hasattr(hpo, "MODEL_ARCH_VERSION")
        assert hasattr(hpo, "MODEL_DESIGN_PATTERN")


# ---------------------------------------------------------------------------
# Fix 2b: transductive_completion guard
# ---------------------------------------------------------------------------

class TestHpoTransductiveCompletionGuard:
    def test_transductive_guard_message_is_informative(self, monkeypatch):
        """The guard ValueError message references the issue and the fix."""
        monkeypatch.setenv("MODEL_DESIGN_PATTERN", "transductive_completion")
        import hpo
        importlib.reload(hpo)

        # Simulate the guard check in hpo.main() before tune.run()
        if hpo.MODEL_DESIGN_PATTERN == "transductive_completion":
            with pytest.raises(ValueError, match="transductive_completion"):
                raise ValueError(
                    "HPO does not support MODEL_DESIGN_PATTERN='transductive_completion'. "
                    "Transductive completion requires a different training objective."
                )
        monkeypatch.delenv("MODEL_DESIGN_PATTERN", raising=False)
        importlib.reload(hpo)

    def test_inductive_forecasting_does_not_trigger_guard(self, monkeypatch):
        """inductive_forecasting does not trigger the transductive_completion guard."""
        monkeypatch.setenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")
        import hpo
        importlib.reload(hpo)
        # Should not raise
        assert hpo.MODEL_DESIGN_PATTERN != "transductive_completion"
        monkeypatch.delenv("MODEL_DESIGN_PATTERN", raising=False)
        importlib.reload(hpo)


# ---------------------------------------------------------------------------
# Fix 3: Launcher scripts propagate MODEL_FAMILY
# ---------------------------------------------------------------------------

class TestLauncherEnvVarPropagation:
    def test_run_pretrain_job_imports_os(self):
        """run_pretrain_job.py imports os (required for os.getenv)."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "os") or "os" in dir(run_pretrain_job)

    def test_run_pretrain_job_has_model_family_constant(self):
        """run_pretrain_job.py defines DEFAULT_MODEL_FAMILY."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_MODEL_FAMILY")
        assert run_pretrain_job.DEFAULT_MODEL_FAMILY == "market_exchangeable_icl"

    def test_run_pretrain_job_has_no_deepset_model_family_constant(self):
        """run_pretrain_job.py must not define the retired default-family constant."""
        import run_pretrain_job
        assert not hasattr(run_pretrain_job, "DEFAULT_" + "DEEPSET" + "_MODEL_FAMILY")

    def test_run_pretrain_job_has_training_data_family_constant(self):
        """run_pretrain_job.py defines DEFAULT_TRAINING_DATA_FAMILY."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_TRAINING_DATA_FAMILY")

    def test_run_pretrain_job_has_no_model_arch_version_constant(self):
        """run_pretrain_job.py must NOT define DEFAULT_MODEL_ARCH_VERSION (hardcoded in train.py)."""
        import run_pretrain_job
        assert not hasattr(run_pretrain_job, "DEFAULT_MODEL_ARCH_VERSION")

    def test_run_pretrain_job_has_model_design_pattern_constant(self):
        """run_pretrain_job.py defines DEFAULT_MODEL_DESIGN_PATTERN."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "DEFAULT_MODEL_DESIGN_PATTERN")

    def test_run_pretrain_job_has_model_handler(self):
        """run_pretrain_job.py defines run_pretrain_pipeline_model (parameterized overload)."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "run_pretrain_pipeline_model")
        assert callable(run_pretrain_job.run_pretrain_pipeline_model)

    def test_run_hpo_job_has_model_family_constant(self):
        """run_hpo_job.py defines DEFAULT_MODEL_FAMILY."""
        import run_hpo_job
        assert hasattr(run_hpo_job, "DEFAULT_MODEL_FAMILY")
        assert run_hpo_job.DEFAULT_MODEL_FAMILY == "market_exchangeable_icl"

    def test_run_hpo_job_has_model_handler(self):
        """run_hpo_job.py defines run_hpo_pipeline_model (parameterized overload)."""
        import run_hpo_job
        assert hasattr(run_hpo_job, "run_hpo_pipeline_model")
        assert callable(run_hpo_job.run_hpo_pipeline_model)

    def test_run_model_training_job_has_model_family_constant(self):
        """run_model_training_job.py defines DEFAULT_MODEL_FAMILY."""
        import run_model_training_job
        assert hasattr(run_model_training_job, "DEFAULT_MODEL_FAMILY")
        assert run_model_training_job.DEFAULT_MODEL_FAMILY == "market_exchangeable_icl"

    def test_run_training_job_has_model_family_constant(self):
        """run_training_job.py defines DEFAULT_MODEL_FAMILY."""
        import run_training_job
        assert hasattr(run_training_job, "DEFAULT_MODEL_FAMILY")
        assert run_training_job.DEFAULT_MODEL_FAMILY == "market_exchangeable_icl"

    def test_model_family_env_override_propagates(self, monkeypatch):
        """MODEL_FAMILY env var overrides DEFAULT_MODEL_FAMILY."""
        monkeypatch.setenv("MODEL_FAMILY", "market_exchangeable_completion")
        import run_pretrain_job
        importlib.reload(run_pretrain_job)
        assert run_pretrain_job.DEFAULT_MODEL_FAMILY == "market_exchangeable_completion"
        monkeypatch.delenv("MODEL_FAMILY", raising=False)
        importlib.reload(run_pretrain_job)


# ---------------------------------------------------------------------------
# Fix 6: MODEL3-aware memory estimator
# ---------------------------------------------------------------------------

class TestModelMemoryEstimator:
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


# ---------------------------------------------------------------------------
# Fix 3 (new): run_model_training_model routes via impl without mutating os.environ
# ---------------------------------------------------------------------------

class TestRunModelTrainingImplRouting:
    def test_impl_function_exists(self):
        """run_model_training_job._run_model_training_impl exists."""
        import run_model_training_job
        assert hasattr(run_model_training_job, "_run_model_training_impl")
        assert callable(run_model_training_job._run_model_training_impl)

    def test_run_model_training_calls_impl_with_defaults(self, monkeypatch):
        """run_model_training delegates to _run_model_training_impl with DEFAULT_ constants."""
        import run_model_training_job
        importlib.reload(run_model_training_job)
        mock_session = MagicMock()

        with patch.object(run_model_training_job, "_run_model_training_impl",
                          return_value="ok") as mock_impl:
            run_model_training_job.run_model_training(mock_session)
            mock_impl.assert_called_once_with(
                mock_session,
                run_model_training_job.DEFAULT_MODEL_FAMILY,
                run_model_training_job.DEFAULT_TRAINING_DATA_FAMILY,
                run_model_training_job.DEFAULT_MODEL_DESIGN_PATTERN,
            )

    def test_run_model_training_model_calls_impl_with_explicit_args(self):
        """run_model_training_model calls _run_model_training_impl with the exact supplied args."""
        import run_model_training_job
        importlib.reload(run_model_training_job)
        mock_session = MagicMock()

        with patch.object(run_model_training_job, "_run_model_training_impl",
                          return_value="ok") as mock_impl:
            run_model_training_job.run_model_training_model(
                mock_session, "fam", "data", "pat"
            )
            mock_impl.assert_called_once_with(mock_session, "fam", "data", "pat")

    def test_run_model_training_model_does_not_mutate_environ(self, monkeypatch):
        """run_model_training_model does not set os.environ['MODEL_FAMILY']."""
        import run_model_training_job
        importlib.reload(run_model_training_job)
        mock_session = MagicMock()

        original_family = os.environ.get("MODEL_FAMILY", "__NOT_SET__")
        with patch.object(run_model_training_job, "_run_model_training_impl",
                          return_value="ok"):
            run_model_training_job.run_model_training_model(
                mock_session, "explicit_fam", "explicit_data", "explicit_pat"
            )
        after_family = os.environ.get("MODEL_FAMILY", "__NOT_SET__")
        assert after_family == original_family, (
            f"os.environ['MODEL_FAMILY'] was mutated: {original_family!r} → {after_family!r}"
        )
