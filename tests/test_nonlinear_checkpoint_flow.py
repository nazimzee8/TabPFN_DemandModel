"""Tests for nonlinear checkpoint flow: HPO propagation and cold-start guard.

Covers Fix 3 (HPO checkpoint propagation), Fix 4 (cold-start blocking), and Fix 8
(autogluon_ray import-time testability).
"""

import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


def _add_scripts_to_path():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


# ---------------------------------------------------------------------------
# Shared fake classes
# ---------------------------------------------------------------------------

class _FakeJob:
    def __init__(self):
        self.status_value = "DONE"

    def status(self):
        return self.status_value

    def wait(self, *args, **kwargs):
        return self.status_value


class _FakeSession:
    def sql(self, query):
        return self

    def collect(self):
        return []

    class file:
        @staticmethod
        def get(src, dst):
            pass

        @staticmethod
        def put(src, dst, **kw):
            pass


# ---------------------------------------------------------------------------
# Fix 3 — HPO checkpoint propagation
# ---------------------------------------------------------------------------

class TestHPOCheckpointPropagation:
    """HPO_PRETRAIN_CHECKPOINT_STAGE_PATH must be passed through to the MLJob env vars."""

    def _load_run_hpo_job(self, monkeypatch):
        _add_scripts_to_path()
        # Provide stub for snowflake.ml.jobs so the module can be imported
        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=_FakeJob())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)

        if "run_hpo_job" in sys.modules:
            del sys.modules["run_hpo_job"]
        import run_hpo_job
        return run_hpo_job

    def test_hpo_impl_propagates_pretrain_checkpoint(self, monkeypatch):
        """HPO_PRETRAIN_CHECKPOINT_STAGE_PATH appears in env_vars when provided."""
        mod = self._load_run_hpo_job(monkeypatch)

        captured_env: list[dict] = []

        def _fake_submit(source, entrypoint, **kwargs):
            captured_env.append(kwargs.get("env_vars", {}))
            return _FakeJob()

        monkeypatch.setattr(mod, "submit_from_stage", _fake_submit)
        monkeypatch.setattr(mod, "_wait_done", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_list_stage", lambda *a, **kw: [])

        session = _FakeSession()
        mod._run_hpo_impl(
            session,
            model_family="mf",
            training_data_family="td",
            model_design_pattern="mdp",
            hpo_sweep_mode="nonlinear_meta",
            hpo_baseline_config_stage_path="",
            hpo_pretrain_checkpoint_stage_path="@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt",
        )

        assert captured_env, "No env_vars were captured"
        assert "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH" in captured_env[0], (
            "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH was not injected into env_vars"
        )
        assert captured_env[0]["HPO_PRETRAIN_CHECKPOINT_STAGE_PATH"] == (
            "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt"
        )

    def test_hpo_impl_omits_pretrain_checkpoint_when_empty(self, monkeypatch):
        """Empty hpo_pretrain_checkpoint_stage_path must NOT inject env var."""
        mod = self._load_run_hpo_job(monkeypatch)

        captured_env: list[dict] = []

        def _fake_submit(source, entrypoint, **kwargs):
            captured_env.append(kwargs.get("env_vars", {}))
            return _FakeJob()

        monkeypatch.setattr(mod, "submit_from_stage", _fake_submit)
        monkeypatch.setattr(mod, "_wait_done", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_list_stage", lambda *a, **kw: [])

        session = _FakeSession()
        mod._run_hpo_impl(
            session,
            model_family="mf",
            training_data_family="td",
            model_design_pattern="mdp",
            hpo_sweep_mode="ridge_residual",
            hpo_baseline_config_stage_path="",
            hpo_pretrain_checkpoint_stage_path="",
        )

        assert captured_env, "No env_vars were captured"
        assert "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH" not in captured_env[0], (
            "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH must NOT be set when argument is empty"
        )

    def test_6arg_entrypoint_exists(self, monkeypatch):
        """run_hpo_pipeline_model_sweep_with_baseline_and_pretrain must be defined."""
        mod = self._load_run_hpo_job(monkeypatch)
        assert hasattr(mod, "run_hpo_pipeline_model_sweep_with_baseline_and_pretrain"), (
            "run_hpo_pipeline_model_sweep_with_baseline_and_pretrain not defined in run_hpo_job"
        )


# ---------------------------------------------------------------------------
# Fix 4 — Nonlinear cold-start guard
# ---------------------------------------------------------------------------

class TestNonlinearColdStartGuard:
    """run_model_training_job must block nonlinear cold-start without override flag."""

    def _load_run_model_training_job(self, monkeypatch):
        _add_scripts_to_path()

        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=_FakeJob())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)

        if "run_model_training_job" in sys.modules:
            del sys.modules["run_model_training_job"]
        import run_model_training_job
        return run_model_training_job

    def _make_nonlinear_best_config(self, *, with_pretrain_path: bool):
        """Return a namespace that mimics the parsed best_config with/without pretrain path."""
        config = types.SimpleNamespace()
        config.use_latent_ridge_expert = True
        config._meta = types.SimpleNamespace(
            pretrain_checkpoint_stage_path=(
                "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt"
                if with_pretrain_path else ""
            )
        )
        return config

    def test_nonlinear_final_training_fails_without_checkpoint(self, monkeypatch):
        """RuntimeError raised for nonlinear config with no _meta path, no override env var."""
        mod = self._load_run_model_training_job(monkeypatch)
        monkeypatch.delenv("ALLOW_NONLINEAR_COLD_START", raising=False)

        # Simulate the branch: is_nonlinear_config=True, no pretrain checkpoint
        config = self._make_nonlinear_best_config(with_pretrain_path=False)

        # Call the internal guard logic directly (it's in _select_pretrain_load_policy or similar)
        # The code lives around lines 238-243 in the original file.
        # We'll invoke it through the module-level function after patching dependencies.
        with pytest.raises(RuntimeError, match="ALLOW_NONLINEAR_COLD_START"):
            mod._apply_nonlinear_cold_start_guard()

    def test_nonlinear_cold_start_allowed_with_override(self, monkeypatch):
        """ALLOW_NONLINEAR_COLD_START=true lets the guard pass without raising."""
        mod = self._load_run_model_training_job(monkeypatch)
        monkeypatch.setenv("ALLOW_NONLINEAR_COLD_START", "true")

        # Must not raise
        mod._apply_nonlinear_cold_start_guard()


# ---------------------------------------------------------------------------
# Fix 8 — autogluon_ray import-time testability
# ---------------------------------------------------------------------------

class TestAutogluonRayImportability:
    """autogluon_ray.py must be importable without any env vars set."""

    def _clear_module(self):
        if "autogluon_ray" in sys.modules:
            del sys.modules["autogluon_ray"]

    def _make_stubs(self):
        """Create minimal stubs for all modules imported by autogluon_ray at module level."""
        stubs = {}

        # ray stub: @ray.remote decorator must work at module-level definition
        ray_stub = types.ModuleType("ray")
        # @ray.remote without args — must return the decorated function unchanged
        ray_stub.remote = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)
        stubs["ray"] = ray_stub

        # numpy stub (just needs to be importable)
        stubs["numpy"] = types.ModuleType("numpy")

        # evaluate_synthetic_regression stub — all names imported at module level
        esr_stub = types.ModuleType("evaluate_synthetic_regression")
        for attr in (
            "create_snowpark_session",
            "load_synthetic_regression_index",
            "assign_synthetic_regression_shard",
            "expand_synreg_work_items",
            "build_compact_synreg_work_item",
            "compact_work_items_serialized_bytes",
            "build_split_for_seed",
            "preprocess_train_only",
            "compute_regression_metrics",
            "write_part_csv_to_stage",
            "memory_guard_matrix",
            "_base_autogluon_row",
            "_empty_row",
            "_skipped_row",
            "_failed_row",
            "_check_tmp_free_bytes",
        ):
            setattr(esr_stub, attr, lambda *a, **kw: None)
        stubs["evaluate_synthetic_regression"] = esr_stub

        # autogluon_models stub
        ag_stub = types.ModuleType("autogluon_models")
        for attr in ("get_tabular_predictor_class", "make_unique_autogluon_model_dir", "predict_autogluon_timed"):
            setattr(ag_stub, attr, lambda *a, **kw: None)
        stubs["autogluon_models"] = ag_stub

        return stubs

    def test_autogluon_ray_helpers_importable_without_env(self, monkeypatch):
        """import autogluon_ray succeeds with no env vars set (no SYNTHETIC_REGRESSION_SUITE_ID etc.)."""
        _add_scripts_to_path()
        monkeypatch.delenv("SYNTHETIC_REGRESSION_SUITE_ID", raising=False)
        monkeypatch.delenv("SYNREG_RESULTS_STAGE", raising=False)

        self._clear_module()
        stubs = self._make_stubs()
        prev = {k: sys.modules.get(k) for k in stubs}
        for k, v in stubs.items():
            sys.modules[k] = v

        try:
            import autogluon_ray
        finally:
            self._clear_module()
            for k, v in prev.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        # Import succeeded without raising RuntimeError for missing env vars

    def test_autogluon_ray_sanitize_item_works_without_env(self, monkeypatch):
        """_sanitize_item_for_debug callable after import-only, without any env vars."""
        _add_scripts_to_path()
        monkeypatch.delenv("SYNTHETIC_REGRESSION_SUITE_ID", raising=False)
        monkeypatch.delenv("SYNREG_RESULTS_STAGE", raising=False)

        self._clear_module()
        stubs = self._make_stubs()
        prev = {k: sys.modules.get(k) for k in stubs}
        for k, v in stubs.items():
            sys.modules[k] = v

        try:
            import autogluon_ray

            item = {
                "suite_id": "test",
                "stage_path": "train/task_0001.parquet",
                "dataset_access": {
                    "mode": "driver_presigned_url",
                    "presigned_url": "https://example.com/secret_url",
                    "stage_path": "train/task_0001.parquet",
                },
            }
            result = autogluon_ray._sanitize_item_for_debug(item)
            assert "presigned_url" not in result.get("dataset_access", {}), (
                "_sanitize_item_for_debug must redact presigned_url"
            )
            assert result["suite_id"] == "test"
        finally:
            self._clear_module()
            for k, v in prev.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_autogluon_ray_module_has_resolve_and_validate_globals(self):
        """autogluon_ray.py must define _resolve_runtime_globals and _validate_runtime_globals."""
        script = ROOT / "scripts" / "autogluon_ray.py"
        text = script.read_text()
        assert "def _resolve_runtime_globals(" in text, (
            "_resolve_runtime_globals() not found in autogluon_ray.py"
        )
        assert "def _validate_runtime_globals(" in text, (
            "_validate_runtime_globals() not found in autogluon_ray.py"
        )

    def test_autogluon_ray_guarded_by_main_block(self):
        """autogluon_ray.py must guard runtime calls under if __name__ == '__main__'."""
        script = ROOT / "scripts" / "autogluon_ray.py"
        text = script.read_text()
        assert '__name__ == "__main__"' in text or "__name__ == '__main__'" in text, (
            "autogluon_ray.py must have if __name__ == '__main__': guard"
        )
        assert "_resolve_runtime_globals()" in text
        assert "_validate_runtime_globals()" in text

    def test_autogluon_ray_uses_options_for_num_cpus(self):
        """autogluon_ray.py must use .options(num_cpus=TASK_CPUS).remote(item) not @ray.remote(num_cpus=...)."""
        script = ROOT / "scripts" / "autogluon_ray.py"
        text = script.read_text()
        assert "@ray.remote(num_cpus=" not in text, (
            "autogluon_ray.py must not use @ray.remote(num_cpus=TASK_CPUS) at definition time"
        )
        assert ".options(num_cpus=TASK_CPUS).remote(item)" in text, (
            "autogluon_ray.py must use .options(num_cpus=TASK_CPUS).remote(item) at call sites"
        )


# ---------------------------------------------------------------------------
# Fix 2 — Family-aware index validation
# ---------------------------------------------------------------------------

class TestFamilyAwareIndexValidation:
    """_validate_training_index routes to the correct index based on training_data_family."""

    def _load_run_model_training_job(self, monkeypatch):
        _add_scripts_to_path()
        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=_FakeJob())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)
        if "run_model_training_job" in sys.modules:
            del sys.modules["run_model_training_job"]
        import run_model_training_job
        return run_model_training_job

    def test_nonlinear_family_queries_nonlinear_index(self, monkeypatch):
        """TRAINING_DATA_FAMILY=synthetic_regression_nonlinear → _validate_nonlinear_training_index called."""
        mod = self._load_run_model_training_job(monkeypatch)

        nonlinear_called = []
        linear_called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_training_index",
                            lambda session: nonlinear_called.append(True))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index",
                            lambda session: linear_called.append(True))

        mod._validate_training_index(_FakeSession(), "synthetic_regression_nonlinear")
        assert nonlinear_called, "_validate_nonlinear_training_index was not called for nonlinear family"
        assert not linear_called, "_validate_meta_dataset_index must not be called for nonlinear family"

    def test_linear_family_does_not_query_nonlinear_index(self, monkeypatch):
        """TRAINING_DATA_FAMILY=synthetic_regression_combined → _validate_meta_dataset_index called."""
        mod = self._load_run_model_training_job(monkeypatch)

        nonlinear_called = []
        linear_called = []
        monkeypatch.setattr(mod, "_validate_nonlinear_training_index",
                            lambda session: nonlinear_called.append(True))
        monkeypatch.setattr(mod, "_validate_meta_dataset_index",
                            lambda session: linear_called.append(True))

        mod._validate_training_index(_FakeSession(), "synthetic_regression_combined")
        assert linear_called, "_validate_meta_dataset_index was not called for linear family"
        assert not nonlinear_called, "_validate_nonlinear_training_index must not be called for linear family"


# ---------------------------------------------------------------------------
# Fix 3 — Family consistency guard
# ---------------------------------------------------------------------------

class TestFamilyConsistencyGuard:
    """_run_model_training_impl raises ValueError on training_data_family mismatch with _meta."""

    def _load_run_model_training_job(self, monkeypatch):
        _add_scripts_to_path()
        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=_FakeJob())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)
        if "run_model_training_job" in sys.modules:
            del sys.modules["run_model_training_job"]
        import run_model_training_job
        return run_model_training_job

    def _patch_impl_dependencies(self, mod, monkeypatch, best_config: dict):
        """Patch all dependencies before the family-check so the guard is reachable."""
        import json, os
        monkeypatch.setattr(mod, "_stage_file_exists", lambda *a, **kw: True)
        monkeypatch.setattr(mod, "_validate_training_index", lambda *a, **kw: None)
        # Write best_config.json where the function will look for it
        config_path = os.path.join(mod.LOCAL_TMP_DIR, "best_config.json")
        with open(config_path, "w") as f:
            json.dump(best_config, f)

    def test_nonlinear_best_config_with_linear_final_family_raises(self, monkeypatch):
        """_meta.training_data_family=nonlinear + linear requested → ValueError."""
        mod = self._load_run_model_training_job(monkeypatch)
        best_config = {
            "gate_hidden_dim": 64,
            "hpo_sweep_mode": "nonlinear_meta",
            "use_latent_ridge_expert": True,
            "_meta": {
                "training_data_family": "synthetic_regression_nonlinear",
                "pretrain_checkpoint_stage_path": "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt",
            },
        }
        self._patch_impl_dependencies(mod, monkeypatch, best_config)

        with pytest.raises(ValueError, match="Family mismatch"):
            mod._run_model_training_impl(
                _FakeSession(),
                model_family="market_exchangeable_icl",
                training_data_family="synthetic_regression_combined",
                model_design_pattern="inductive_forecasting",
            )

    def test_matching_families_pass(self, monkeypatch):
        """Same family in _meta and requested → no ValueError from family guard."""
        mod = self._load_run_model_training_job(monkeypatch)
        best_config = {
            "gate_hidden_dim": 64,
            "hpo_sweep_mode": "nonlinear_meta",
            "use_latent_ridge_expert": True,
            "_meta": {
                "training_data_family": "synthetic_regression_nonlinear",
                "pretrain_checkpoint_stage_path": "@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt",
            },
        }
        self._patch_impl_dependencies(mod, monkeypatch, best_config)
        # Also patch _stage_file_exists to return False for checkpoint so we stop before job submit
        monkeypatch.setattr(mod, "_stage_file_exists", lambda *a, **kw: False)

        with pytest.raises((FileNotFoundError, RuntimeError)):
            # Raises on missing best_config stage file or checkpoint, NOT ValueError from family guard
            mod._run_model_training_impl(
                _FakeSession(),
                model_family="market_exchangeable_icl",
                training_data_family="synthetic_regression_nonlinear",
                model_design_pattern="inductive_forecasting",
            )

    def test_absent_meta_family_is_backward_compatible(self, monkeypatch):
        """No training_data_family key in _meta → passes (old best_config files)."""
        mod = self._load_run_model_training_job(monkeypatch)
        best_config = {
            "gate_hidden_dim": 64,
            "hpo_sweep_mode": "ridge_residual",
            "use_latent_ridge_expert": False,
            "_meta": {
                # no training_data_family key — old format
                "pretrain_checkpoint_stage_path": "",
            },
        }
        self._patch_impl_dependencies(mod, monkeypatch, best_config)
        # Patch _stage_file_exists: True for best_config.json presence check, False for checkpoint
        call_count = [0]
        def _fake_exists(*a, **kw):
            call_count[0] += 1
            return call_count[0] == 1  # True first call (best_config exists), False thereafter
        monkeypatch.setattr(mod, "_stage_file_exists", _fake_exists)

        # Should not raise ValueError about family mismatch; may raise FileNotFoundError for checkpoint
        try:
            mod._run_model_training_impl(
                _FakeSession(),
                model_family="market_exchangeable_icl",
                training_data_family="synthetic_regression_combined",
                model_design_pattern="inductive_forecasting",
            )
        except ValueError as exc:
            assert "Family mismatch" not in str(exc), (
                f"Old best_config without training_data_family should not trigger family guard: {exc}"
            )
        except (FileNotFoundError, RuntimeError):
            pass  # Expected — guard passed, failed later on missing checkpoint


# ---------------------------------------------------------------------------
# Fix 4 (second audit) — Nonlinear pretrain entrypoint enforcement
# ---------------------------------------------------------------------------

class TestNonlinearPretainEntrypoint:
    """run_pretrain_pipeline_nonlinear_model enforces nonlinear-only family."""

    def _load_run_pretrain_job(self, monkeypatch):
        _add_scripts_to_path()
        sfml = types.ModuleType("snowflake")
        sfml.ml = types.ModuleType("snowflake.ml")
        sfml.ml.jobs = types.ModuleType("snowflake.ml.jobs")
        sfml.ml.jobs.submit_from_stage = MagicMock(return_value=_FakeJob())
        sys.modules.setdefault("snowflake", sfml)
        sys.modules.setdefault("snowflake.ml", sfml.ml)
        sys.modules.setdefault("snowflake.ml.jobs", sfml.ml.jobs)
        if "run_pretrain_job" in sys.modules:
            del sys.modules["run_pretrain_job"]
        import run_pretrain_job
        return run_pretrain_job

    def test_3arg_rejects_non_nonlinear_family(self, monkeypatch):
        """ValueError raised when training_data_family != synthetic_regression_nonlinear."""
        mod = self._load_run_pretrain_job(monkeypatch)
        with pytest.raises(ValueError, match="run_pretrain_pipeline_nonlinear only supports"):
            mod.run_pretrain_pipeline_nonlinear_model(
                "market_exchangeable_icl",
                "synthetic_regression_combined",  # wrong family
                "inductive_forecasting",
            )

    def test_3arg_accepts_nonlinear_family(self, monkeypatch):
        """Correct family → ValueError not raised; submission proceeds (or fails on session)."""
        mod = self._load_run_pretrain_job(monkeypatch)
        # Patch _get_session and _run_pretrain_nonlinear_impl to avoid real I/O
        called_with = []
        monkeypatch.setattr(mod, "_get_session", lambda: _FakeSession())
        monkeypatch.setattr(
            mod, "_run_pretrain_nonlinear_impl",
            lambda session, mf, tdf, mdp: called_with.append(tdf) or "ok",
        )
        result = mod.run_pretrain_pipeline_nonlinear_model(
            "market_exchangeable_icl",
            "synthetic_regression_nonlinear",
            "inductive_forecasting",
        )
        assert result == "ok"
        assert called_with == ["synthetic_regression_nonlinear"]

    def test_0arg_uses_nonlinear_family_regardless_of_env(self, monkeypatch):
        """TRAINING_DATA_FAMILY=synthetic_regression_combined env var is ignored by 0-arg entrypoint."""
        mod = self._load_run_pretrain_job(monkeypatch)
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        called_with = []
        monkeypatch.setattr(mod, "_get_session", lambda: _FakeSession())
        monkeypatch.setattr(
            mod, "_run_pretrain_nonlinear_impl",
            lambda session, mf, tdf, mdp: called_with.append(tdf) or "ok",
        )
        mod.run_pretrain_pipeline_nonlinear()
        assert called_with == ["synthetic_regression_nonlinear"], (
            f"0-arg entrypoint must hardcode NONLINEAR_TRAINING_DATA_FAMILY, got {called_with}"
        )
