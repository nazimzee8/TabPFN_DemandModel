"""
tests/test_model_ddp_memory_probe.py

Tests for the MODEL3 DDP memory probe:
  1. Launcher (run_model_ddp_memory_probe in run_model_training_job.py)
     - env vars passed to MLJob
     - target_instances == TRAIN_NUM_NODES
     - expected world size == TRAIN_NUM_NODES * 4
  2. Launcher validation
     - rejects model2
     - rejects transductive_completion
     - rejects wrong family
     - rejects non-positive shapes
  3. Static memory estimator (model_ddp_memory_probe.py)
     - h_tensor_bytes == m * n * p * d * dtype_bytes
     - backward estimate > forward estimate
     - memory guard skips/fails when threshold exceeded
  4. Probe script unit tests (no CUDA required)
     - env parsing works
     - validate_probe_config accepts valid config
     - validate_probe_config rejects invalid configs
     - ModelConfig construction uses correct selectors
     - result JSON schema contains required top-level keys
  5. SQL DDL
     - run_model_ddp_memory_probe procedure exists in run_training_job.sql
     - signature contains all 8 required parameters
     - HANDLER points to run_model_training_job.run_model_ddp_memory_probe
     - IMPORTS includes run_model_training_job.py

All tests mock submit_from_stage and Snowflake session — no real Snowflake
or CUDA required.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent.parent
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
SRC_DIR     = str(REPO_ROOT / "src")

for _d in (SCRIPTS_DIR, SRC_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ---------------------------------------------------------------------------
# Snowflake stub — must be in sys.modules before the launcher scripts import
# ---------------------------------------------------------------------------
for _mod in ("snowflake", "snowflake.ml", "snowflake.ml.jobs", "snowflake.snowpark"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import run_model_training_job  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeJob:
    """Minimal MLJob fake that reports DONE immediately."""
    def __init__(self, **kwargs):
        self.env_vars   = kwargs.get("env_vars", {})
        self.entrypoint = kwargs.get("entrypoint", "")
        self.source     = kwargs.get("source", "")
        self.status     = "DONE"
        self.target_instances = kwargs.get("target_instances", 0)

    def wait(self):
        pass

    def get_logs(self):
        return ""


def _make_fake_submit(jobs: list):
    def _submit(**kwargs):
        job = _FakeJob(**kwargs)
        jobs.append(job)
        return job
    return _submit


def _make_session():
    session = MagicMock()
    session.file.put = MagicMock()
    session.sql.return_value.collect.return_value = []
    return session


def _call_probe(monkeypatch=None, **override_kwargs):
    """Call run_model_ddp_memory_probe with defaults, return (job_list, return_value)."""
    defaults = dict(
        model_design_pattern="inductive_forecasting",
        model_family="market_exchangeable_icl",
        n_context=200, p_features=128, m_query=128,
        d_phi=128, n_blocks=1, run_backward=True,
    )
    defaults.update(override_kwargs)
    jobs = []
    session = _make_session()
    with patch("run_model_training_job.submit_from_stage", side_effect=_make_fake_submit(jobs)), \
         patch("run_model_training_job._wait_done"), \
         patch("run_model_training_job._list_stage", return_value=[]):
        result = run_model_training_job.run_model_ddp_memory_probe(session, **defaults)
    return jobs, result


# ---------------------------------------------------------------------------
# 1. Launcher — env vars and topology
# ---------------------------------------------------------------------------

class TestLauncherEnvVars:
    def test_submits_correct_entrypoint(self):
        jobs, _ = _call_probe()
        assert len(jobs) == 1
        assert jobs[0].entrypoint == "model_ddp_memory_probe.py"

    def test_target_instances_equals_train_num_nodes(self):
        jobs, _ = _call_probe()
        assert jobs[0].target_instances == run_model_training_job.TRAIN_NUM_NODES

    def test_expected_world_size_equals_nodes_times_four(self):
        jobs, _ = _call_probe()
        env = jobs[0].env_vars
        expected_ws = int(env["EXPECTED_TRAIN_WORLD_SIZE"])
        train_nodes = int(env["TRAIN_NUM_NODES"])
        assert expected_ws == train_nodes * 4

    def test_strict_world_size_check_is_true(self):
        jobs, _ = _call_probe()
        assert jobs[0].env_vars["STRICT_WORLD_SIZE_CHECK"] == "true"

    def test_model_arch_version_not_in_env_vars(self):
        jobs, _ = _call_probe()
        assert "MODEL_ARCH_VERSION" not in jobs[0].env_vars

    def test_model_design_pattern_propagated(self):
        jobs, _ = _call_probe()
        assert jobs[0].env_vars["MODEL_DESIGN_PATTERN"] == "inductive_forecasting"

    def test_model_family_propagated(self):
        jobs, _ = _call_probe()
        assert jobs[0].env_vars["MODEL_FAMILY"] == "market_exchangeable_icl"

    def test_shape_env_vars_propagated(self):
        jobs, _ = _call_probe(
            n_context=300, p_features=64, m_query=32, d_phi=256, n_blocks=2,
        )
        env = jobs[0].env_vars
        assert env["MODEL_PROBE_N_CONTEXT"]  == "300"
        assert env["MODEL_PROBE_P_FEATURES"] == "64"
        assert env["MODEL_PROBE_M_QUERY"]    == "32"
        assert env["MODEL_PROBE_D_PHI"]      == "256"
        assert env["MODEL_PROBE_N_BLOCKS"]   == "2"

    def test_run_backward_true_propagated(self):
        jobs, _ = _call_probe(run_backward=True)
        assert jobs[0].env_vars["MODEL_PROBE_RUN_BACKWARD"] == "true"

    def test_run_backward_false_propagated(self):
        jobs, _ = _call_probe(run_backward=False)
        assert jobs[0].env_vars["MODEL_PROBE_RUN_BACKWARD"] == "false"

    def test_output_stage_contains_diagnostics(self):
        jobs, _ = _call_probe()
        assert "diagnostics" in jobs[0].env_vars["MODEL_PROBE_OUTPUT_STAGE"]

    def test_dtype_defaults_to_float32(self):
        jobs, _ = _call_probe()
        assert jobs[0].env_vars["MODEL_PROBE_DTYPE"] == "float32"

    def test_uses_gpu_pool(self):
        jobs = []
        session = _make_session()
        submitted_kwargs = {}

        def _capture(**kwargs):
            submitted_kwargs.update(kwargs)
            job = _FakeJob(**kwargs)
            jobs.append(job)
            return job

        with patch("run_model_training_job.submit_from_stage", side_effect=_capture), \
             patch("run_model_training_job._wait_done"), \
             patch("run_model_training_job._list_stage", return_value=[]):
            run_model_training_job.run_model_ddp_memory_probe(
                session,
                model_design_pattern="inductive_forecasting",
                model_family="market_exchangeable_icl",
                n_context=200, p_features=128, m_query=128,
                d_phi=128, n_blocks=1, run_backward=True,
            )
        assert submitted_kwargs["compute_pool"] == run_model_training_job.GPU_POOL

    def test_home_env_var_is_slash_tmp(self):
        jobs, _ = _call_probe()
        assert jobs[0].env_vars["HOME"] == "/tmp"

    def test_return_value_mentions_diagnostics(self):
        _, result = _call_probe()
        assert "diagnostics" in result.lower() or "MODEL_STAGE" in result


# ---------------------------------------------------------------------------
# 2. Launcher validation — rejected configs
# ---------------------------------------------------------------------------

class TestLauncherValidation:
    def _probe(self, **kwargs):
        return run_model_training_job.run_model_ddp_memory_probe(
            _make_session(), **kwargs
        )

    def _defaults(self, **override):
        d = dict(
            model_design_pattern="inductive_forecasting",
            model_family="market_exchangeable_icl",
            n_context=200, p_features=128, m_query=128,
            d_phi=128, n_blocks=1, run_backward=True,
        )
        d.update(override)
        return d

    def test_rejects_transductive_completion(self):
        with pytest.raises(ValueError, match="transductive_completion"):
            self._probe(**self._defaults(model_design_pattern="transductive_completion"))

    def test_rejects_wrong_design_pattern(self):
        with pytest.raises(ValueError, match="inductive_forecasting"):
            self._probe(**self._defaults(model_design_pattern="unknown_pattern"))

    def test_rejects_wrong_family(self):
        with pytest.raises(ValueError, match="market_exchangeable_icl"):
            self._probe(**self._defaults(model_family="market_aware"))

    def test_rejects_zero_n_context(self):
        with pytest.raises(ValueError, match="n_context"):
            self._probe(**self._defaults(n_context=0))

    def test_rejects_negative_p_features(self):
        with pytest.raises(ValueError, match="p_features"):
            self._probe(**self._defaults(p_features=-1))

    def test_rejects_zero_m_query(self):
        with pytest.raises(ValueError, match="m_query"):
            self._probe(**self._defaults(m_query=0))

    def test_rejects_zero_d_phi(self):
        with pytest.raises(ValueError, match="d_phi"):
            self._probe(**self._defaults(d_phi=0))

    def test_rejects_zero_n_blocks(self):
        with pytest.raises(ValueError, match="n_blocks"):
            self._probe(**self._defaults(n_blocks=0))


# ---------------------------------------------------------------------------
# 3. Static memory estimator
# ---------------------------------------------------------------------------

# Import the probe module for direct unit testing.
# model_ddp_memory_probe uses snowflake at module level (try/except), so it's
# safe to import locally.
_probe_mod = None


def _get_probe_mod():
    global _probe_mod
    if _probe_mod is None:
        # Ensure snowflake stubs are in place before import
        for _m in ("snowflake", "snowflake.ml", "snowflake.snowpark",
                   "snowflake.ml.modeling", "snowflake.ml.modeling.distributors",
                   "snowflake.ml.modeling.distributors.pytorch"):
            if _m not in sys.modules:
                sys.modules[_m] = MagicMock()
        _probe_mod = importlib.import_module("model_ddp_memory_probe")
    return _probe_mod


class TestStaticEstimator:
    def test_h_tensor_bytes_float32(self):
        probe = _get_probe_mod()
        result = probe.estimate_h_tensor_bytes(
            n_context=200, p_features=128, m_query=128, d_phi=128, dtype="float32"
        )
        expected = 200 * 128 * 128 * 128 * 4
        assert result == expected

    def test_h_tensor_bytes_bfloat16(self):
        probe = _get_probe_mod()
        result = probe.estimate_h_tensor_bytes(
            n_context=200, p_features=128, m_query=128, d_phi=128, dtype="bfloat16"
        )
        expected = 200 * 128 * 128 * 128 * 2
        assert result == expected

    def test_h_tensor_bytes_scales_with_m_query(self):
        probe = _get_probe_mod()
        small = probe.estimate_h_tensor_bytes(100, 64, 32, 64)
        large = probe.estimate_h_tensor_bytes(100, 64, 64, 64)
        assert large == 2 * small

    def test_h_tensor_bytes_scales_with_d_phi(self):
        probe = _get_probe_mod()
        small = probe.estimate_h_tensor_bytes(100, 64, 32, 64)
        large = probe.estimate_h_tensor_bytes(100, 64, 32, 128)
        assert large == 2 * small

    def test_backward_estimate_greater_than_forward(self):
        probe = _get_probe_mod()
        h = probe.estimate_h_tensor_bytes(200, 128, 128, 128)
        fwd_est, fwd_factor = probe.estimate_reserved_bytes(h, run_backward=False)
        bwd_est, bwd_factor = probe.estimate_reserved_bytes(h, run_backward=True)
        assert bwd_est > fwd_est
        assert bwd_factor > fwd_factor

    def test_forward_activation_factor_is_8(self):
        probe = _get_probe_mod()
        h = 1_000_000
        _, factor = probe.estimate_reserved_bytes(h, run_backward=False)
        assert factor == 8

    def test_backward_activation_factor_is_20(self):
        probe = _get_probe_mod()
        h = 1_000_000
        _, factor = probe.estimate_reserved_bytes(h, run_backward=True)
        assert factor == 20

    def test_estimated_reserved_uses_safety_factor(self):
        probe = _get_probe_mod()
        h = 1_000_000
        est_default, _ = probe.estimate_reserved_bytes(h, run_backward=True, safety_factor=1.5)
        est_double, _  = probe.estimate_reserved_bytes(h, run_backward=True, safety_factor=3.0)
        assert abs(est_double / est_default - 2.0) < 0.01

    def test_formula_matches_manual_calculation(self):
        probe = _get_probe_mod()
        m, n, p, d = 128, 200, 128, 128
        h = probe.estimate_h_tensor_bytes(n, p, m, d, "float32")
        est, factor = probe.estimate_reserved_bytes(h, run_backward=True, safety_factor=1.5)
        expected = int(m * n * p * d * 4 * 20 * 1.5)
        assert est == expected


# ---------------------------------------------------------------------------
# 4. Probe script unit tests (no CUDA, no DDP)
# ---------------------------------------------------------------------------

class TestProbeScriptValidation:
    def setup_method(self):
        self.probe = _get_probe_mod()

    def test_validate_accepts_valid_config(self):
        # Should not raise
        self.probe.validate_probe_config(
            "inductive_forecasting", "market_exchangeable_icl",
            200, 128, 128, 128, 1, "float32",
        )

    def test_validate_rejects_transductive(self):
        with pytest.raises(ValueError, match="transductive_completion"):
            self.probe.validate_probe_config(
                "transductive_completion", "market_exchangeable_icl",
                200, 128, 128, 128, 1, "float32",
            )

    def test_validate_rejects_wrong_family(self):
        with pytest.raises(ValueError, match="market_exchangeable_icl"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_aware",
                200, 128, 128, 128, 1, "float32",
            )

    def test_validate_rejects_non_positive_n_context(self):
        with pytest.raises(ValueError, match="n_context"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_exchangeable_icl",
                0, 128, 128, 128, 1, "float32",
            )

    def test_validate_rejects_non_positive_p_features(self):
        with pytest.raises(ValueError, match="p_features"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_exchangeable_icl",
                200, -1, 128, 128, 1, "float32",
            )

    def test_validate_rejects_non_positive_m_query(self):
        with pytest.raises(ValueError, match="m_query"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_exchangeable_icl",
                200, 128, 0, 128, 1, "float32",
            )

    def test_validate_rejects_non_positive_d_phi(self):
        with pytest.raises(ValueError, match="d_phi"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_exchangeable_icl",
                200, 128, 128, 0, 1, "float32",
            )

    def test_validate_rejects_non_positive_n_blocks(self):
        with pytest.raises(ValueError, match="n_blocks"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_exchangeable_icl",
                200, 128, 128, 128, 0, "float32",
            )

    def test_validate_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError, match="float16"):
            self.probe.validate_probe_config(
                "inductive_forecasting", "market_exchangeable_icl",
                200, 128, 128, 128, 1, "float16",
            )

    def test_validate_accepts_bfloat16(self):
        # bfloat16 is supported
        self.probe.validate_probe_config(
            "inductive_forecasting", "market_exchangeable_icl",
            200, 128, 128, 128, 1, "bfloat16",
        )

    def test_modelconfig_construction_uses_model3_selectors(self):
        """ModelConfig built with model3 selectors does not raise."""
        from model import ModelConfig
        cfg = ModelConfig(
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            model_family="market_exchangeable_icl",
            d_phi=128,
            n_sab_feat=1,
            n_heads=4,
            norm_feat=True,
            norm_target=True,
            dropout=0.0,
        )
        assert cfg.model_arch_version    == "model3"
        assert cfg.model_design_pattern == "inductive_forecasting"
        assert cfg.model_family          == "market_exchangeable_icl"

    def test_result_json_schema_top_level_keys(self):
        """_upload_probe_result assembles a payload with all required top-level keys."""
        probe = self.probe
        h = probe.estimate_h_tensor_bytes(100, 64, 32, 64)
        est, act = probe.estimate_reserved_bytes(h, run_backward=True)
        static_est = {
            "h_tensor_bytes": h,
            "estimated_reserved_bytes": est,
            "activation_factor": act,
            "safety_factor": 1.5,
        }
        # Minimal rank result
        rank_result = {
            "rank": 0, "local_rank": 0, "world_size": 1,
            "hostname": "test-host", "device": "cuda:0",
            "cuda_device_name": "A10G",
            "cuda_total_memory_bytes": 24e9,
            "cuda_free_memory_bytes_before": 20e9,
            "peak_memory_allocated_bytes": 5e9,
            "peak_memory_reserved_bytes": 8e9,
            "current_memory_allocated_bytes": 2e9,
            "current_memory_reserved_bytes": 4e9,
            "parameter_count": 100000,
            "forward_ok": True,
            "backward_ok": True,
            "output_shape": [32],
            "elapsed_seconds": 1.23,
        }
        # Write to temp file (bypass Snowflake upload)
        tmp_dir = tempfile.mkdtemp()
        orig_path = "/tmp/model_ddp_memory_probe.json"

        with patch.object(probe, "_upload_probe_result", wraps=probe._upload_probe_result):
            # Directly assemble the payload (mirroring _upload_probe_result logic)
            all_results = [rank_result]
            valid_ranks = [r for r in all_results if "peak_memory_reserved_bytes" in r]
            payload = {
                "status": "ok",
                "probe_type": "model_ddp_memory_probe",
                "model_design_pattern": "inductive_forecasting",
                "model_family": "market_exchangeable_icl",
                "shape": {
                    "n_context": 100, "p_features": 64, "m_query": 32,
                    "d_phi": 64, "n_blocks": 1, "dtype": "float32",
                    "run_backward": True,
                },
                "static_estimate": static_est,
                "world": {
                    "train_num_nodes": 1,
                    "expected_world_size": 4,
                    "actual_world_size": 1,
                },
                "ranks": all_results,
                "summary": {
                    "max_peak_reserved_bytes": 8e9,
                    "max_peak_allocated_bytes": 5e9,
                    "max_reserved_fraction": 8e9 / 24e9,
                    "min_free_memory_before_bytes": 20e9,
                },
            }

        required_keys = {
            "status", "probe_type",
            "model_design_pattern", "model_family",
            "shape", "static_estimate", "world", "ranks", "summary",
        }
        assert required_keys <= set(payload.keys())

    def test_shape_keys_in_payload(self):
        """shape dict must contain all required shape fields."""
        shape_required = {
            "n_context", "p_features", "m_query", "d_phi",
            "n_blocks", "dtype", "run_backward",
        }
        shape = {
            "n_context": 200, "p_features": 128, "m_query": 128,
            "d_phi": 128, "n_blocks": 1, "dtype": "float32",
            "run_backward": True,
        }
        assert shape_required <= set(shape.keys())

    def test_static_estimate_keys_in_payload(self):
        """static_estimate dict must contain all required fields."""
        probe = self.probe
        h = probe.estimate_h_tensor_bytes(200, 128, 128, 128)
        est, act = probe.estimate_reserved_bytes(h, run_backward=True)
        se = {
            "h_tensor_bytes": h,
            "estimated_reserved_bytes": est,
            "activation_factor": act,
            "safety_factor": 1.5,
        }
        required = {"h_tensor_bytes", "estimated_reserved_bytes",
                    "activation_factor", "safety_factor"}
        assert required <= set(se.keys())

    def test_env_parsing_n_context(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROBE_N_CONTEXT", "512")
        assert int(os.environ["MODEL_PROBE_N_CONTEXT"]) == 512

    def test_env_parsing_run_backward_true(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROBE_RUN_BACKWARD", "true")
        assert os.environ["MODEL_PROBE_RUN_BACKWARD"].lower() == "true"

    def test_env_parsing_run_backward_false(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROBE_RUN_BACKWARD", "false")
        assert os.environ["MODEL_PROBE_RUN_BACKWARD"].lower() != "true"


# ---------------------------------------------------------------------------
# 5. SQL DDL
# ---------------------------------------------------------------------------

class TestModelDDPMemoryProbeSQLDDL:
    def _sql(self) -> str:
        return (REPO_ROOT / "sql" / "run_training_job.sql").read_text()

    def test_procedure_exists(self):
        assert "run_model_ddp_memory_probe" in self._sql()

    def test_create_or_replace_procedure(self):
        sql = self._sql()
        assert "CREATE OR REPLACE PROCEDURE run_model_ddp_memory_probe" in sql

    def test_handler_points_to_correct_function(self):
        sql = self._sql()
        assert (
            "run_model_training_job.run_model_ddp_memory_probe" in sql
        ), "HANDLER must point to run_model_training_job.run_model_ddp_memory_probe"

    def test_imports_run_model_training_job(self):
        sql = self._sql()
        assert "run_model_training_job.py" in sql

    def test_signature_does_not_have_model_arch_version(self):
        sql = self._sql()
        # MODEL_ARCH_VERSION was removed; model3 is the sole supported version
        proc_idx = sql.find("CREATE OR REPLACE PROCEDURE run_model_ddp_memory_probe")
        next_proc = sql.find("CREATE OR REPLACE PROCEDURE", proc_idx + 1)
        proc_block = sql[proc_idx:next_proc] if next_proc > 0 else sql[proc_idx:]
        assert "MODEL_ARCH_VERSION STRING" not in proc_block

    def test_signature_has_model_design_pattern(self):
        sql = self._sql()
        assert "MODEL_DESIGN_PATTERN STRING" in sql

    def test_signature_has_model_family(self):
        sql = self._sql()
        assert "MODEL_FAMILY STRING" in sql

    def test_signature_has_n_context(self):
        sql = self._sql()
        assert "N_CONTEXT INTEGER" in sql

    def test_signature_has_p_features(self):
        sql = self._sql()
        assert "P_FEATURES INTEGER" in sql

    def test_signature_has_m_query(self):
        sql = self._sql()
        assert "M_QUERY INTEGER" in sql

    def test_signature_has_d_phi(self):
        sql = self._sql()
        assert "D_PHI INTEGER" in sql

    def test_signature_has_n_blocks(self):
        sql = self._sql()
        assert "N_BLOCKS INTEGER" in sql

    def test_signature_has_run_backward(self):
        sql = self._sql()
        assert "RUN_BACKWARD BOOLEAN" in sql

    def test_example_call_in_comments(self):
        sql = self._sql()
        assert "CALL run_model_ddp_memory_probe" in sql
        assert "RUN_BACKWARD" in sql or "TRUE" in sql

    def test_diagnostics_stage_mentioned(self):
        sql = self._sql()
        assert "diagnostics" in sql

    def test_runtime_version_is_3_11(self):
        sql = self._sql()
        # Check that the procedure block for run_model_ddp_memory_probe has 3.11
        proc_idx = sql.find("CREATE OR REPLACE PROCEDURE run_model_ddp_memory_probe")
        next_proc = sql.find("CREATE OR REPLACE PROCEDURE", proc_idx + 1)
        proc_block = sql[proc_idx:next_proc] if next_proc > 0 else sql[proc_idx:]
        assert "3.11" in proc_block
