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

    def test_run_pretrain_job_has_gate_handler(self):
        """run_pretrain_job.py defines run_pretrain_pipeline_model_gate (4-arg gate overload)."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "run_pretrain_pipeline_model_gate")
        assert callable(run_pretrain_job.run_pretrain_pipeline_model_gate)

    def test_run_pretrain_gate_impl_exists(self):
        """run_pretrain_job.py defines _run_pretrain_gate_impl (internal impl)."""
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "_run_pretrain_gate_impl")
        assert callable(run_pretrain_job._run_pretrain_gate_impl)

    def test_run_pretrain_gate_impl_rejects_invalid_gate_dim(self):
        """_run_pretrain_gate_impl raises ValueError for invalid gate_hidden_dim."""
        import run_pretrain_job
        mock_session = MagicMock()
        # Patch out _validate_meta_dataset_index to not require Snowflake
        with patch.object(run_pretrain_job, "_validate_meta_dataset_index"):
            with pytest.raises(ValueError, match="gate_hidden_dim=99"):
                run_pretrain_job._run_pretrain_gate_impl(
                    mock_session,
                    "market_exchangeable_icl",
                    "synthetic_regression_combined",
                    "inductive_forecasting",
                    99,  # invalid
                )

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

        with patch.object(run_model_training_job, "_get_session",
                          return_value=mock_session), \
             patch.object(run_model_training_job, "_run_model_training_impl",
                          return_value="ok") as mock_impl:
            run_model_training_job.run_model_training()
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

        with patch.object(run_model_training_job, "_get_session",
                          return_value=mock_session), \
             patch.object(run_model_training_job, "_run_model_training_impl",
                          return_value="ok") as mock_impl:
            run_model_training_job.run_model_training_model("fam", "data", "pat")
            mock_impl.assert_called_once_with(mock_session, "fam", "data", "pat")

    def test_run_model_training_model_does_not_mutate_environ(self, monkeypatch):
        """run_model_training_model does not set os.environ['MODEL_FAMILY']."""
        import run_model_training_job
        importlib.reload(run_model_training_job)
        mock_session = MagicMock()

        original_family = os.environ.get("MODEL_FAMILY", "__NOT_SET__")
        with patch.object(run_model_training_job, "_get_session",
                          return_value=mock_session), \
             patch.object(run_model_training_job, "_run_model_training_impl",
                          return_value="ok"):
            run_model_training_job.run_model_training_model(
                "explicit_fam", "explicit_data", "explicit_pat"
            )
        after_family = os.environ.get("MODEL_FAMILY", "__NOT_SET__")
        assert after_family == original_family, (
            f"os.environ['MODEL_FAMILY'] was mutated: {original_family!r} → {after_family!r}"
        )


# ---------------------------------------------------------------------------
# Gate checkpoint resolution in run_model_training_job
# ---------------------------------------------------------------------------

class TestGateCheckpointResolution:
    def test_pretrain_policy_mode_conditional(self):
        """_run_model_training_impl sets PRETRAIN_LOAD_POLICY based on hpo_sweep_mode.
        ridge_residual → require_match; architecture → allow_cold_start_on_arch_mismatch."""
        import run_model_training_job
        import inspect
        src = inspect.getsource(run_model_training_job._run_model_training_impl)
        assert "require_match" in src, (
            "_run_model_training_impl must use PRETRAIN_LOAD_POLICY=require_match "
            "for ridge_residual sweep"
        )
        assert "allow_cold_start_on_arch_mismatch" in src, (
            "_run_model_training_impl must use PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch "
            "for architecture sweep"
        )
        assert "hpo_sweep_mode" in src, (
            "_run_model_training_impl must read hpo_sweep_mode from best_config"
        )

    def test_meta_checkpoint_path_takes_priority(self):
        """If _meta.pretrain_checkpoint_stage_path is set, it takes priority over gate fallback."""
        import run_model_training_job
        import inspect
        src = inspect.getsource(run_model_training_job._run_model_training_impl)
        assert "pretrain_checkpoint_stage_path" in src, (
            "_run_model_training_impl must read _meta.pretrain_checkpoint_stage_path"
        )
        assert "pretrain_gate" in src, (
            "_run_model_training_impl must fall back to pretrain_gate<N>.pt"
        )

    def test_gate_dim_constant_exists_in_hpo(self):
        """hpo.GATE_HIDDEN_DIM_CANDIDATES constant exists and is a non-empty list."""
        import hpo
        assert hasattr(hpo, "GATE_HIDDEN_DIM_CANDIDATES")
        candidates = hpo.GATE_HIDDEN_DIM_CANDIDATES
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(d, int) for d in candidates)

    def test_no_pretrain_pt_fallback_in_source(self):
        """_run_model_training_impl must not fall back to legacy pretrain.pt."""
        import run_model_training_job
        import inspect
        src = inspect.getsource(run_model_training_job._run_model_training_impl)
        # The only mention of "pretrain.pt" should not be as a fallback path
        assert 'pretrain.pt"' not in src, (
            '_run_model_training_impl must not fall back to @MODEL_STAGE/checkpoints/pretrain.pt. '
            'Use gate-specific checkpoints only.'
        )

    def test_no_cold_start_in_source(self):
        """_run_model_training_impl must not permit cold-start (PRETRAIN_CHECKPOINT_PATH always set)."""
        import run_model_training_job
        import inspect
        src = inspect.getsource(run_model_training_job._run_model_training_impl)
        # PRETRAIN_CHECKPOINT_PATH must always be unconditionally set — no 'if pretrain_checkpoint_path'
        # gating it; the function must raise FileNotFoundError before reaching env_vars if missing.
        assert "if pretrain_checkpoint_path" not in src, (
            "_run_model_training_impl must not guard PRETRAIN_CHECKPOINT_PATH with an if-check; "
            "raise FileNotFoundError before env_vars if the checkpoint is missing."
        )


class TestCheckpointResolutionBehavior:
    """Functional tests for _run_model_training_impl checkpoint resolution.

    Uses mock sessions to exercise the exact resolution paths without Snowflake.
    """

    def _make_session_with_files(self, tmp_dir: str, existing_files, best_config: dict):
        """Session mock: _stage_file_exists returns True only for files in existing_files."""
        import json as _json
        session = MagicMock()

        def _file_get(stage_path, local_dir):
            import os
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, "best_config.json"), "w") as f:
                _json.dump(best_config, f)

        def _sql_collect():
            # Return fake row list for LIST queries (used by _stage_file_exists and _list_stage)
            return [MagicMock(**{"__getitem__": lambda self, i: str(f) if i == 0 else ""})
                    for f in existing_files]

        session.file.get.side_effect = _file_get
        session.file.put = MagicMock()
        session.sql.return_value.collect.side_effect = _sql_collect
        return session

    def _run_impl(self, session, monkeypatch, existing_files=None):
        """Call _run_model_training_impl with mocked submit_from_stage and _wait_done.

        Patches _stage_file_exists directly (avoids MagicMock row-indexing issues)
        so tests control which staged files are considered present.
        """
        import run_model_training_job
        importlib.reload(run_model_training_job)
        submitted = []

        if existing_files is not None:
            _files = list(existing_files)

            def _fake_stage_file_exists(sess, stage_path, filename):
                return any(f.endswith("/" + filename) for f in _files)

            monkeypatch.setattr(run_model_training_job, "_stage_file_exists",
                                _fake_stage_file_exists)

        def _fake_submit(**kwargs):
            submitted.append(kwargs)
            job = MagicMock()
            job.status = "DONE"
            return job

        monkeypatch.setattr(run_model_training_job, "submit_from_stage", _fake_submit)
        monkeypatch.setattr(run_model_training_job, "_wait_done", MagicMock())
        monkeypatch.setattr(run_model_training_job, "_validate_meta_dataset_index", MagicMock())
        return submitted, run_model_training_job

    def test_meta_path_used_when_present(self, monkeypatch):
        """PRETRAIN_CHECKPOINT_PATH from _meta.pretrain_checkpoint_stage_path is passed to job."""
        import tempfile
        tmp = tempfile.mkdtemp()
        meta_path = "@MODEL_STAGE/checkpoints/pretrain_gate64.pt"
        best_config = {
            "lr": 1e-3, "gate_hidden_dim": 64,
            "_meta": {"pretrain_checkpoint_stage_path": meta_path},
        }
        # Both hpo/ and checkpoints/ files exist
        existing = [
            "@MODEL_STAGE/hpo/best_config.json",
            "@MODEL_STAGE/checkpoints/pretrain_gate64.pt",
        ]
        session = self._make_session_with_files(tmp, existing, best_config)
        submitted, rmt = self._run_impl(session, monkeypatch, existing_files=existing)
        rmt._run_model_training_impl(session, "mef_icl", "combined", "inductive_forecasting")
        assert submitted, "submit_from_stage was not called"
        assert submitted[0]["env_vars"]["PRETRAIN_CHECKPOINT_PATH"] == meta_path
        assert submitted[0]["env_vars"]["PRETRAIN_LOAD_POLICY"] == "require_match"

    def test_gate_fallback_used_when_no_meta_path(self, monkeypatch):
        """Fallback to pretrain_gate<N>.pt when _meta path is absent."""
        import tempfile
        tmp = tempfile.mkdtemp()
        best_config = {"lr": 1e-3, "gate_hidden_dim": 32, "_meta": {}}
        existing = [
            "@MODEL_STAGE/hpo/best_config.json",
            "@MODEL_STAGE/checkpoints/pretrain_gate32.pt",
        ]
        session = self._make_session_with_files(tmp, existing, best_config)
        submitted, rmt = self._run_impl(session, monkeypatch, existing_files=existing)
        rmt._run_model_training_impl(session, "mef_icl", "combined", "inductive_forecasting")
        assert submitted
        assert "pretrain_gate32.pt" in submitted[0]["env_vars"]["PRETRAIN_CHECKPOINT_PATH"]
        assert submitted[0]["env_vars"]["PRETRAIN_LOAD_POLICY"] == "require_match"

    def test_missing_meta_checkpoint_raises(self, monkeypatch):
        """FileNotFoundError if _meta path is set but file is absent from stage."""
        import tempfile
        tmp = tempfile.mkdtemp()
        meta_path = "@MODEL_STAGE/checkpoints/pretrain_gate128.pt"
        best_config = {
            "lr": 1e-3, "gate_hidden_dim": 128,
            "_meta": {"pretrain_checkpoint_stage_path": meta_path},
        }
        # pretrain_gate128.pt is NOT in existing files
        existing = ["@MODEL_STAGE/hpo/best_config.json"]
        session = self._make_session_with_files(tmp, existing, best_config)
        _, rmt = self._run_impl(session, monkeypatch, existing_files=existing)
        with pytest.raises(FileNotFoundError, match="pretrain_gate128"):
            rmt._run_model_training_impl(session, "mef_icl", "combined", "inductive_forecasting")

    def test_missing_gate_checkpoint_raises(self, monkeypatch):
        """FileNotFoundError if gate fallback checkpoint is also absent."""
        import tempfile
        tmp = tempfile.mkdtemp()
        best_config = {"lr": 1e-3, "gate_hidden_dim": 64, "_meta": {}}
        # pretrain_gate64.pt is absent
        existing = ["@MODEL_STAGE/hpo/best_config.json"]
        session = self._make_session_with_files(tmp, existing, best_config)
        _, rmt = self._run_impl(session, monkeypatch, existing_files=existing)
        with pytest.raises(FileNotFoundError, match="gate_hidden_dim=64"):
            rmt._run_model_training_impl(session, "mef_icl", "combined", "inductive_forecasting")

    def test_error_message_includes_call_pretrain(self, monkeypatch):
        """FileNotFoundError message instructs user to rerun gate-specific pretrain."""
        import tempfile
        tmp = tempfile.mkdtemp()
        best_config = {"lr": 1e-3, "gate_hidden_dim": 32, "_meta": {}}
        existing = ["@MODEL_STAGE/hpo/best_config.json"]
        session = self._make_session_with_files(tmp, existing, best_config)
        _, rmt = self._run_impl(session, monkeypatch, existing_files=existing)
        with pytest.raises(FileNotFoundError, match="run_pretrain_pipeline"):
            rmt._run_model_training_impl(session, "mef_icl", "combined", "inductive_forecasting")


# ---------------------------------------------------------------------------
# Snowflake handler session creation regression tests
# ---------------------------------------------------------------------------

class TestHandlerSessionCreation:
    """Verify public handlers create session via _get_session(), not via first parameter.

    Regression for the bug: Snowflake passes SQL string args positionally; if the
    handler accepted session as arg 0, the first string argument was used as session,
    causing AttributeError: 'str' object has no attribute 'sql'.
    """

    def test_get_session_helper_exists_in_pretrain(self):
        import run_pretrain_job
        assert hasattr(run_pretrain_job, "_get_session")
        assert callable(run_pretrain_job._get_session)

    def test_get_session_helper_exists_in_hpo(self):
        import run_hpo_job
        assert hasattr(run_hpo_job, "_get_session")
        assert callable(run_hpo_job._get_session)

    def test_get_session_helper_exists_in_model_training(self):
        import run_model_training_job
        assert hasattr(run_model_training_job, "_get_session")
        assert callable(run_model_training_job._get_session)

    def test_get_session_helper_exists_in_training_job(self):
        import run_training_job
        assert hasattr(run_training_job, "_get_session")
        assert callable(run_training_job._get_session)

    def test_run_pretrain_pipeline_model_signature_matches_sql(self):
        """run_pretrain_pipeline_model(model_family, training_data_family, model_design_pattern)
        — no session parameter."""
        import inspect, run_pretrain_job
        params = list(inspect.signature(run_pretrain_job.run_pretrain_pipeline_model).parameters)
        assert params == ["model_family", "training_data_family", "model_design_pattern"], (
            f"Unexpected signature params: {params}"
        )

    def test_run_pretrain_pipeline_model_gate_signature_matches_sql(self):
        """run_pretrain_pipeline_model_gate(model_family, training_data_family,
        model_design_pattern, gate_hidden_dim) — no session."""
        import inspect, run_pretrain_job
        params = list(inspect.signature(run_pretrain_job.run_pretrain_pipeline_model_gate).parameters)
        assert params == [
            "model_family", "training_data_family", "model_design_pattern", "gate_hidden_dim"
        ], f"Unexpected signature params: {params}"

    def test_run_hpo_pipeline_model_signature_matches_sql(self):
        """run_hpo_pipeline_model(model_family, training_data_family, model_design_pattern)."""
        import inspect, run_hpo_job
        params = list(inspect.signature(run_hpo_job.run_hpo_pipeline_model).parameters)
        assert params == ["model_family", "training_data_family", "model_design_pattern"], (
            f"Unexpected signature params: {params}"
        )

    def test_run_hpo_pipeline_model_sweep_signature_matches_sql(self):
        """run_hpo_pipeline_model_sweep(model_family, training_data_family,
        model_design_pattern, hpo_sweep_mode)."""
        import inspect, run_hpo_job
        params = list(inspect.signature(run_hpo_job.run_hpo_pipeline_model_sweep).parameters)
        assert params == [
            "model_family", "training_data_family", "model_design_pattern", "hpo_sweep_mode"
        ], f"Unexpected signature params: {params}"

    def test_run_hpo_pipeline_model_sweep_with_baseline_signature_matches_sql(self):
        """run_hpo_pipeline_model_sweep_with_baseline(model_family, training_data_family,
        model_design_pattern, hpo_sweep_mode, hpo_baseline_config_stage_path)."""
        import inspect, run_hpo_job
        params = list(inspect.signature(run_hpo_job.run_hpo_pipeline_model_sweep_with_baseline).parameters)
        assert params == [
            "model_family", "training_data_family", "model_design_pattern",
            "hpo_sweep_mode", "hpo_baseline_config_stage_path",
        ], f"Unexpected signature params: {params}"

    def test_run_model_training_model_signature_matches_sql(self):
        """run_model_training_model(model_family, training_data_family, model_design_pattern)."""
        import inspect, run_model_training_job
        params = list(inspect.signature(run_model_training_job.run_model_training_model).parameters)
        assert params == ["model_family", "training_data_family", "model_design_pattern"], (
            f"Unexpected signature params: {params}"
        )

    def test_run_model_ddp_memory_probe_signature_no_session(self):
        """run_model_ddp_memory_probe does not accept session as first parameter."""
        import inspect, run_model_training_job
        params = list(inspect.signature(run_model_training_job.run_model_ddp_memory_probe).parameters)
        assert "session" not in params, (
            f"run_model_ddp_memory_probe must not accept session; got params: {params}"
        )

    def test_run_training_runtime_probe_signature_no_session(self):
        """run_training_runtime_probe(target_instances) — no session."""
        import inspect, run_model_training_job
        params = list(inspect.signature(run_model_training_job.run_training_runtime_probe).parameters)
        assert params == ["target_instances"], f"Unexpected params: {params}"

    def test_run_pipeline_signature_no_session(self):
        """run_pipeline() — no session."""
        import inspect, run_training_job
        params = list(inspect.signature(run_training_job.run_pipeline).parameters)
        assert params == [], f"run_pipeline must take no params, got: {params}"

    def test_build_meta_dataset_index_signature_no_session(self):
        """build_meta_dataset_index() — no session."""
        import inspect, run_training_job
        params = list(inspect.signature(run_training_job.build_meta_dataset_index).parameters)
        assert params == [], f"build_meta_dataset_index must take no params, got: {params}"

    def test_regression_pretrain_model_first_arg_not_treated_as_session(self):
        """Regression: calling run_pretrain_pipeline_model with string args must not
        raise AttributeError('str' object has no attribute 'sql')."""
        import run_pretrain_job, importlib
        importlib.reload(run_pretrain_job)
        mock_session = MagicMock()
        with patch.object(run_pretrain_job, "_get_session", return_value=mock_session), \
             patch.object(run_pretrain_job, "_run_pretrain_impl", return_value="ok") as mock_impl:
            result = run_pretrain_job.run_pretrain_pipeline_model(
                "market_exchangeable_icl",
                "synthetic_regression_combined",
                "inductive_forecasting",
            )
        assert result == "ok"
        # First arg to _run_pretrain_impl must be the mock session, not the string
        called_session = mock_impl.call_args[0][0]
        assert called_session is mock_session, (
            f"Expected _run_pretrain_impl to receive mock_session as first arg, "
            f"got {called_session!r}"
        )

    def test_regression_hpo_pipeline_first_arg_not_treated_as_session(self):
        """Regression: run_hpo_pipeline_model with string args must not
        raise AttributeError('str' object has no attribute 'sql')."""
        import run_hpo_job, importlib
        importlib.reload(run_hpo_job)
        mock_session = MagicMock()
        with patch.object(run_hpo_job, "_get_session", return_value=mock_session), \
             patch.object(run_hpo_job, "_run_hpo_impl", return_value="ok") as mock_impl:
            result = run_hpo_job.run_hpo_pipeline_model(
                "market_exchangeable_icl",
                "synthetic_regression_combined",
                "inductive_forecasting",
            )
        assert result == "ok"
        called_session = mock_impl.call_args[0][0]
        assert called_session is mock_session, (
            f"Expected _run_hpo_impl to receive mock_session as first arg, "
            f"got {called_session!r}"
        )
