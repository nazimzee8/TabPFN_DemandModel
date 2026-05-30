"""
tests/test_training_data_family_submission.py

Tests that TRAINING_DATA_FAMILY is correctly propagated through submission scripts:
  - scripts/run_training_job.py  (pretrain + HPO + final train)
  - scripts/run_model_training_job.py (final train only)
  - scripts/run_hpo_job.py (HPO only)

All tests mock submit_from_stage and Snowflake session to avoid real infrastructure.
The submission scripts import snowflake.ml.jobs at module level, so we pre-stub those
modules in sys.modules before loading the scripts.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Pre-stub snowflake.ml so the module-level "from snowflake.ml.jobs import
# submit_from_stage" in the submission scripts succeeds locally.
for _mod in (
    "snowflake",
    "snowflake.ml",
    "snowflake.ml.jobs",
    "snowflake.snowpark",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Now safe to import the scripts (their top-level Snowflake import resolves
# to the MagicMock above; submit_from_stage will be patched per-test).
import run_training_job      # noqa: E402
import run_model_training_job  # noqa: E402
import run_hpo_job           # noqa: E402
import run_pretrain_job      # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — fake job / session
# ---------------------------------------------------------------------------

class _FakeJob:
    """Minimal MLJob fake that immediately reports DONE."""
    def __init__(self, **kwargs):
        self.env_vars = kwargs.get("env_vars", {})
        self.entrypoint = kwargs.get("entrypoint", "")
        self.status = "DONE"

    def wait(self):
        pass

    def get_logs(self):
        return ""


def _make_fake_submit(jobs: list):
    """Return a submit_from_stage replacement that records submitted jobs."""
    def _submit(**kwargs):
        job = _FakeJob(**kwargs)
        jobs.append(job)
        return job
    return _submit


def _make_session_for_model_training(tmp_dir: str, best_config: dict | None = None):
    """
    Minimal session mock for run_model_training_job.run_model_training().

    file.get writes best_config.json to tmp_dir before the script opens it.
    sql(...).collect() returns [] for all LIST queries.
    """
    cfg = best_config or {
        "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "model_family": "market_exchangeable_icl"
    }
    session = MagicMock()

    def _file_get(stage_path, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        dest = os.path.join(local_dir, "best_config.json")
        with open(dest, "w") as f:
            json.dump(cfg, f)

    session.file.get.side_effect = _file_get
    session.file.put = MagicMock()
    session.sql.return_value.collect.side_effect = lambda: []
    return session


def _make_session_for_pipeline(tmp_dir: str, best_config: dict | None = None):
    """Minimal session mock for run_training_job.run_pipeline()."""
    cfg = best_config or {
        "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "model_family": "market_exchangeable_icl"
    }
    session = MagicMock()

    def _file_get(stage_path, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        dest = os.path.join(local_dir, "best_config.json")
        with open(dest, "w") as f:
            json.dump(cfg, f)

    session.file.get.side_effect = _file_get
    session.file.put = MagicMock()
    session.sql.return_value.collect.side_effect = lambda: []
    return session


# ---------------------------------------------------------------------------
# Tests: run_training_job.py — DEFAULT_TRAINING_DATA_FAMILY constant
# ---------------------------------------------------------------------------

class TestRunTrainingJobConstant:
    def test_default_is_combined(self, monkeypatch):
        """DEFAULT_TRAINING_DATA_FAMILY defaults to synthetic_regression_combined."""
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_training_job)
        assert run_training_job.DEFAULT_TRAINING_DATA_FAMILY == "synthetic_regression_combined"
        importlib.reload(run_training_job)

    def test_env_var_override(self, monkeypatch):
        """TRAINING_DATA_FAMILY env var overrides the default."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_primary")
        importlib.reload(run_training_job)
        assert run_training_job.DEFAULT_TRAINING_DATA_FAMILY == "synthetic_regression_primary"
        importlib.reload(run_training_job)

    def test_nonlinear_family_env_var_override(self, monkeypatch):
        """TRAINING_DATA_FAMILY accepts the nonlinear lineage selector."""
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")
        importlib.reload(run_training_job)
        assert run_training_job.DEFAULT_TRAINING_DATA_FAMILY == "synthetic_regression_nonlinear"
        importlib.reload(run_training_job)

    def test_default_is_not_unknown(self, monkeypatch):
        """Production default must not be 'unknown'."""
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_training_job)
        assert run_training_job.DEFAULT_TRAINING_DATA_FAMILY != "unknown"
        importlib.reload(run_training_job)


# ---------------------------------------------------------------------------
# Tests: run_training_job.py — run_pipeline() env_var propagation
# ---------------------------------------------------------------------------

class TestRunPipelinePropagation:
    def _collect_jobs(self, monkeypatch, training_data_family="synthetic_regression_combined"):
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_training_job)
        run_training_job.DEFAULT_TRAINING_DATA_FAMILY = training_data_family

        submitted: list[_FakeJob] = []
        tmp = tempfile.mkdtemp()
        session = _make_session_for_pipeline(tmp)

        with patch("run_training_job.submit_from_stage",
                   side_effect=_make_fake_submit(submitted)), \
             patch("run_training_job._wait_done"), \
             patch("run_training_job._validate_meta_dataset_index"), \
             patch("run_training_job._stage_file_exists", return_value=True), \
             patch("run_training_job._list_stage", return_value=[]), \
             patch("run_training_job._get_session", return_value=session):
            run_training_job.run_pipeline()

        importlib.reload(run_training_job)
        return submitted

    def test_pretrain_job_receives_family(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch)
        # Pipeline now submits a single pretrain job (pretrain.pt, two-sweep strategy)
        pretrain_jobs = [
            j for j in jobs
            if j.entrypoint == "train.py"
            and j.env_vars.get("CHECKPOINT_OUTPUT_NAME", "") == "pretrain.pt"
        ]
        assert pretrain_jobs, (
            "Expected a single pretrain job with CHECKPOINT_OUTPUT_NAME=pretrain.pt"
        )
        for pretrain in pretrain_jobs:
            assert pretrain.env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_combined"

    def test_hpo_job_receives_family(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch)
        hpo = next(j for j in jobs if j.entrypoint == "hpo.py")
        assert hpo.env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_combined"

    def test_final_train_job_receives_family(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch)
        # Final training: train.py with a BEST_CONFIG key (not pretrain)
        final = next(
            j for j in jobs
            if j.entrypoint == "train.py"
            and "BEST_CONFIG" in j.env_vars
        )
        assert final.env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_combined"

    def test_pretrain_and_final_train_use_same_family(self, monkeypatch):
        """Pretrain and final training must share the same TRAINING_DATA_FAMILY."""
        jobs = self._collect_jobs(monkeypatch)
        train_jobs = [j for j in jobs if j.entrypoint == "train.py"]
        families = {j.env_vars["TRAINING_DATA_FAMILY"] for j in train_jobs}
        assert len(families) == 1

    def test_override_primary_propagates_to_all(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch, "synthetic_regression_primary")
        for j in [j for j in jobs if j.entrypoint in ("train.py", "hpo.py")]:
            assert j.env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_primary"

    def test_override_ood_propagates_to_all(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch, "synthetic_regression_ood")
        for j in [j for j in jobs if j.entrypoint in ("train.py", "hpo.py")]:
            assert j.env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_ood"

    def test_override_market_mental_model_propagates_to_all(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch, "market_mental_model")
        for j in [j for j in jobs if j.entrypoint in ("train.py", "hpo.py")]:
            assert j.env_vars["TRAINING_DATA_FAMILY"] == "market_mental_model"


# ---------------------------------------------------------------------------
# Tests: run_pretrain_job.py — purpose-specific training-family env defaults
# ---------------------------------------------------------------------------

class TestRunPretrainJobFamilyDefaults:
    def test_pretrain_specific_env_overrides_shared_training_family(self, monkeypatch):
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        monkeypatch.setenv("PRETRAIN_TRAINING_DATA_FAMILY", "market_mental_model")
        importlib.reload(run_pretrain_job)

        assert run_pretrain_job.DEFAULT_TRAINING_DATA_FAMILY == "market_mental_model"
        importlib.reload(run_pretrain_job)

    def test_pretrain_falls_back_to_shared_training_family(self, monkeypatch):
        monkeypatch.delenv("PRETRAIN_TRAINING_DATA_FAMILY", raising=False)
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")
        importlib.reload(run_pretrain_job)

        assert (
            run_pretrain_job.DEFAULT_TRAINING_DATA_FAMILY
            == "synthetic_regression_nonlinear"
        )
        importlib.reload(run_pretrain_job)

    def test_nonlinear_pretrain_has_nonlinear_default_without_shared_env(self, monkeypatch):
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        monkeypatch.delenv("PRETRAIN_TRAINING_DATA_FAMILY", raising=False)
        monkeypatch.delenv("NONLINEAR_PRETRAIN_TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_pretrain_job)

        assert (
            run_pretrain_job.DEFAULT_NONLINEAR_TRAINING_DATA_FAMILY
            == "synthetic_regression_nonlinear"
        )
        importlib.reload(run_pretrain_job)

    def test_nonlinear_pretrain_specific_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        monkeypatch.setenv(
            "NONLINEAR_PRETRAIN_TRAINING_DATA_FAMILY",
            "market_mental_model",
        )
        importlib.reload(run_pretrain_job)

        assert (
            run_pretrain_job.DEFAULT_NONLINEAR_TRAINING_DATA_FAMILY
            == "market_mental_model"
        )
        importlib.reload(run_pretrain_job)


# ---------------------------------------------------------------------------
# Tests: run_model_training_job.py — run_model_training() propagation
# ---------------------------------------------------------------------------

class TestRunModelTrainingPropagation:
    def _collect_jobs(self, monkeypatch, training_data_family="synthetic_regression_combined"):
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_model_training_job)
        run_model_training_job.DEFAULT_TRAINING_DATA_FAMILY = training_data_family

        submitted: list[_FakeJob] = []
        tmp = tempfile.mkdtemp()
        session = _make_session_for_model_training(tmp)

        with patch("run_model_training_job.submit_from_stage",
                   side_effect=_make_fake_submit(submitted)), \
             patch("run_model_training_job._wait_done"), \
             patch("run_model_training_job._validate_meta_dataset_index"), \
             patch("run_model_training_job._stage_file_exists", return_value=True), \
             patch("run_model_training_job._list_stage", return_value=[]), \
             patch("run_model_training_job._get_session", return_value=session):
            run_model_training_job.run_model_training()

        importlib.reload(run_model_training_job)
        return submitted

    def test_final_train_receives_combined_by_default(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch)
        assert len(jobs) == 1
        assert jobs[0].env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_combined"

    def test_override_primary_accepted(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch, "synthetic_regression_primary")
        assert jobs[0].env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_primary"

    def test_override_ood_accepted(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch, "synthetic_regression_ood")
        assert jobs[0].env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_ood"

    def test_override_market_mental_model_accepted(self, monkeypatch):
        jobs = self._collect_jobs(monkeypatch, "market_mental_model")
        assert jobs[0].env_vars["TRAINING_DATA_FAMILY"] == "market_mental_model"

    def test_default_is_not_unknown(self, monkeypatch):
        """Production default must not be 'unknown'."""
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_model_training_job)
        assert run_model_training_job.DEFAULT_TRAINING_DATA_FAMILY != "unknown"
        importlib.reload(run_model_training_job)

    def test_submission_log_includes_training_data_family(self, monkeypatch):
        """TRAINING_DATA_FAMILY must appear in env_vars passed to submit_from_stage."""
        jobs = self._collect_jobs(monkeypatch)
        assert "TRAINING_DATA_FAMILY" in jobs[0].env_vars

    def test_model_family_env_var_present(self, monkeypatch):
        """MODEL_FAMILY must be in env_vars."""
        jobs = self._collect_jobs(monkeypatch)
        assert "MODEL_FAMILY" in jobs[0].env_vars

    def test_deepset_model_family_env_var_absent(self, monkeypatch):
        """Retired model-family env var must not be in env_vars."""
        jobs = self._collect_jobs(monkeypatch)
        assert "DEEPSET" + "_MODEL_FAMILY" not in jobs[0].env_vars

    def test_model_arch_version_env_var_absent(self, monkeypatch):
        """MODEL_ARCH_VERSION must NOT be in env_vars (hardcoded in train.py)."""
        jobs = self._collect_jobs(monkeypatch)
        assert "MODEL_ARCH_VERSION" not in jobs[0].env_vars


# ---------------------------------------------------------------------------
# Tests: run_hpo_job.py — DEFAULT_TRAINING_DATA_FAMILY constant + propagation
# ---------------------------------------------------------------------------

class TestRunHpoJobPropagation:
    def test_default_is_combined(self, monkeypatch):
        monkeypatch.delenv("HPO_TRAINING_DATA_FAMILY", raising=False)
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_hpo_job)
        assert run_hpo_job.DEFAULT_TRAINING_DATA_FAMILY == "synthetic_regression_combined"
        importlib.reload(run_hpo_job)

    def test_hpo_specific_env_overrides_shared_training_family(self, monkeypatch):
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_combined")
        monkeypatch.setenv("HPO_TRAINING_DATA_FAMILY", "market_mental_model")
        importlib.reload(run_hpo_job)

        assert run_hpo_job.DEFAULT_TRAINING_DATA_FAMILY == "market_mental_model"
        importlib.reload(run_hpo_job)

    def test_hpo_falls_back_to_shared_training_family(self, monkeypatch):
        monkeypatch.delenv("HPO_TRAINING_DATA_FAMILY", raising=False)
        monkeypatch.setenv("TRAINING_DATA_FAMILY", "synthetic_regression_nonlinear")
        importlib.reload(run_hpo_job)

        assert run_hpo_job.DEFAULT_TRAINING_DATA_FAMILY == "synthetic_regression_nonlinear"
        importlib.reload(run_hpo_job)

    def test_hpo_job_receives_training_data_family(self, monkeypatch):
        monkeypatch.delenv("HPO_TRAINING_DATA_FAMILY", raising=False)
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_hpo_job)

        submitted: list[_FakeJob] = []
        session = MagicMock()
        session.sql.return_value.collect.side_effect = lambda: []

        with patch("run_hpo_job.submit_from_stage",
                   side_effect=_make_fake_submit(submitted)), \
             patch("run_hpo_job._wait_done"), \
             patch("run_hpo_job._list_stage", return_value=[]), \
             patch("run_hpo_job._get_session", return_value=session):
            run_hpo_job.run_hpo_pipeline()

        assert len(submitted) == 1
        assert submitted[0].env_vars["TRAINING_DATA_FAMILY"] == "synthetic_regression_combined"
        importlib.reload(run_hpo_job)

    def test_hpo_job_has_model_family_not_deepset_model_family(self, monkeypatch):
        """HPO job uses MODEL_FAMILY env var; MODEL_ARCH_VERSION is absent."""
        monkeypatch.delenv("TRAINING_DATA_FAMILY", raising=False)
        importlib.reload(run_hpo_job)

        submitted: list[_FakeJob] = []
        session = MagicMock()
        session.sql.return_value.collect.side_effect = lambda: []

        with patch("run_hpo_job.submit_from_stage",
                   side_effect=_make_fake_submit(submitted)), \
             patch("run_hpo_job._wait_done"), \
             patch("run_hpo_job._list_stage", return_value=[]), \
             patch("run_hpo_job._get_session", return_value=session):
            run_hpo_job.run_hpo_pipeline()

        job = submitted[0]
        assert "MODEL_FAMILY" in job.env_vars, "MODEL_FAMILY missing from HPO job env_vars"
        assert "DEEPSET" + "_MODEL_FAMILY" not in job.env_vars, "Retired model-family env var in HPO job env_vars"
        assert "MODEL_ARCH_VERSION" not in job.env_vars, "MODEL_ARCH_VERSION in HPO job env_vars (must be hardcoded)"
        importlib.reload(run_hpo_job)
