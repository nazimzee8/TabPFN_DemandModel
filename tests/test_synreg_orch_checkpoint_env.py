"""
tests/test_synreg_orch_checkpoint_env.py

Tests for Phase 8: orchestration env propagation.
Verifies that SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH and gate env vars
are propagated to all DeepSet shard jobs.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

ROOT = __import__("pathlib").Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_synthetic_regression_evaluation as orch


# ---------------------------------------------------------------------------
# Fake job / session infrastructure (mirrors existing orchestration tests)
# ---------------------------------------------------------------------------

class _FakeJob:
    def __init__(self, label: str, **kwargs):
        self.label = label
        self.compute_pool = kwargs.get("compute_pool")
        self.env_vars = kwargs.get("env_vars", {})
        self.pip_requirements = kwargs.get("pip_requirements")
        self.external_access_integrations = kwargs.get("external_access_integrations")
        self.target_instances = kwargs.get("target_instances", 1)
        self.runtime_environment = kwargs.get("runtime_environment")
        self.entrypoint = kwargs.get("entrypoint")
        self.status = "DONE"

    def wait(self):
        pass

    @property
    def logs(self):
        return lambda node_id=0: ""


class _FakeSession:
    pass


class JobCollector:
    def __init__(self):
        self.submitted: list[_FakeJob] = []

    def submit(self, label: str, **kwargs) -> _FakeJob:
        job = _FakeJob(label, **kwargs)
        self.submitted.append(job)
        return job


@pytest.fixture()
def collector():
    return JobCollector()


@pytest.fixture()
def fake_session():
    return _FakeSession()


@pytest.fixture()
def runtime_args():
    return ("2.5.0-py311", "2.5.0-py311", "2.5.0-py311")


def _patch_submit(collector: JobCollector):
    """Patch _submit_synreg and pool checks to record submissions."""
    from contextlib import ExitStack

    def _mock_submit_synreg(session, label, compute_pool, env_vars, runtime_environment,
                             entrypoint="evaluate_linear_regression.py",
                             target_instances=1, pip_requirements=None,
                             external_access_integrations=None):
        return collector.submit(
            label=label,
            compute_pool=compute_pool,
            env_vars=env_vars,
            runtime_environment=runtime_environment,
            entrypoint=entrypoint,
            target_instances=target_instances,
            pip_requirements=pip_requirements,
            external_access_integrations=external_access_integrations,
        )

    @contextmanager
    def _ctx():
        with ExitStack() as stack:
            stack.enter_context(patch.object(orch, "_submit_synreg", side_effect=_mock_submit_synreg))
            stack.enter_context(patch.object(orch, "_ensure_compute_pool_usable", return_value=None))
            stack.enter_context(patch.object(orch, "_stage_file_exists", return_value=True))
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# Main DeepSet shard tests
# ---------------------------------------------------------------------------

def test_main_deepset_shards_include_checkpoint_stage_path(collector, fake_session, runtime_args):
    """All 10 main deepset shard jobs have SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH in env_vars."""
    with _patch_submit(collector):
        orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    assert len(deepset_jobs) == orch.SYNREG_GPU_SHARDS, f"Expected {orch.SYNREG_GPU_SHARDS} shards, got {len(deepset_jobs)}"
    for job in deepset_jobs:
        assert "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH" in job.env_vars, \
            f"Job {job.label} missing SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH"


def test_main_deepset_shards_include_gate_env_vars(collector, fake_session, runtime_args):
    """All 10 main deepset shard jobs have gate env vars set to 'true'."""
    with _patch_submit(collector):
        orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    for job in deepset_jobs:
        assert job.env_vars.get("SYNREG_RUN_CHECKPOINT_GATES") == "true", \
            f"Job {job.label} missing SYNREG_RUN_CHECKPOINT_GATES=true"
        assert job.env_vars.get("SYNREG_CHECKPOINT_GATE_STRICT") == "true", \
            f"Job {job.label} missing SYNREG_CHECKPOINT_GATE_STRICT=true"


def test_deepset_shards_still_no_pip_or_eai(collector, fake_session, runtime_args):
    """Phase 8 additions don't break no-pip/no-EAI contract for deepset shards."""
    with _patch_submit(collector):
        orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    for job in deepset_jobs:
        assert job.pip_requirements is None, f"Job {job.label} has unexpected pip_requirements"
        assert job.external_access_integrations is None, f"Job {job.label} has unexpected EAI"


# ---------------------------------------------------------------------------
# OOD pilot shard tests
# ---------------------------------------------------------------------------

def test_ood_pilot_deepset_includes_gate_env(collector, fake_session):
    """OOD pilot deepset shard jobs include gate env vars."""
    bench_rt = "2.5.0-py311"
    with _patch_submit(collector):
        # Also patch _wait_done to avoid waiting
        with patch.object(orch, "_wait_done", return_value=None):
            with patch.object(orch, "_submit_ood_prep", return_value=_FakeJob("ood_prep")):
                orch.run_synthetic_regression_ood_deepset_pilot(fake_session, bench_rt)

    ood_jobs = [j for j in collector.submitted if "ood_deepset_shard" in j.label]
    assert len(ood_jobs) > 0, "No OOD pilot deepset shard jobs found"
    for job in ood_jobs:
        assert "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH" in job.env_vars, \
            f"OOD pilot job {job.label} missing SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH"
        assert job.env_vars.get("SYNREG_RUN_CHECKPOINT_GATES") == "true"
        assert job.env_vars.get("SYNREG_CHECKPOINT_GATE_STRICT") == "true"


# ---------------------------------------------------------------------------
# OOD full shard tests
# ---------------------------------------------------------------------------

def test_ood_full_deepset_includes_gate_env(collector, fake_session):
    """OOD full deepset shard jobs include gate env vars."""
    bench_rt = "2.5.0-py311"
    ag_rt = "2.5.0-py311"
    with _patch_submit(collector):
        orch.run_synthetic_regression_ood_full_deepset_evaluation(fake_session, bench_rt, ag_rt)

    ood_jobs = [j for j in collector.submitted if "ood_full_deepset_shard" in j.label]
    assert len(ood_jobs) > 0, "No OOD full deepset shard jobs found"
    for job in ood_jobs:
        assert "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH" in job.env_vars
        assert job.env_vars.get("SYNREG_RUN_CHECKPOINT_GATES") == "true"
        assert job.env_vars.get("SYNREG_CHECKPOINT_GATE_STRICT") == "true"


# ---------------------------------------------------------------------------
# Baseline shard isolation test
# ---------------------------------------------------------------------------

def test_baseline_shards_do_not_include_gate_env(collector, fake_session, runtime_args):
    """Baseline shards do NOT get deepset-specific gate/checkpoint env vars."""
    with _patch_submit(collector):
        orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)

    baseline_jobs = [j for j in collector.submitted if "baseline" in j.label]
    assert len(baseline_jobs) > 0, "No baseline jobs found"
    for job in baseline_jobs:
        assert "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH" not in job.env_vars, \
            f"Baseline job {job.label} unexpectedly has SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH"
        assert "SYNREG_RUN_CHECKPOINT_GATES" not in job.env_vars, \
            f"Baseline job {job.label} unexpectedly has SYNREG_RUN_CHECKPOINT_GATES"


# ---------------------------------------------------------------------------
# Env var propagation test
# ---------------------------------------------------------------------------

def test_checkpoint_stage_path_propagates_from_orchestrator_env(collector, fake_session, runtime_args, monkeypatch):
    """Setting SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH in env → shard jobs receive it."""
    custom_path = "@stage/custom.pt"
    monkeypatch.setenv("SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH", custom_path)
    import importlib
    importlib.reload(orch)

    with _patch_submit(collector):
        orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    assert len(deepset_jobs) > 0
    for job in deepset_jobs:
        assert job.env_vars.get("SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH") == custom_path, \
            f"Expected {custom_path}, got {job.env_vars.get('SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH')}"

    importlib.reload(orch)


# ---------------------------------------------------------------------------
# Retired env var absence tests
# ---------------------------------------------------------------------------

def test_deepset_model_family_env_absent_from_shards(collector, fake_session, runtime_args):
    """Evaluation shard jobs must not carry the retired model-family env var."""
    with _patch_submit(collector):
        orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    assert len(deepset_jobs) > 0
    for job in deepset_jobs:
        assert "DEEPSET" + "_MODEL_FAMILY" not in job.env_vars, \
            f"Job {job.label} unexpectedly contains retired model-family env var"


def test_model_arch_version_env_absent_from_shards(collector, fake_session, runtime_args):
    """Evaluation shard jobs must NOT carry MODEL_ARCH_VERSION (hardcoded in train.py)."""
    with _patch_submit(collector):
        orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    assert len(deepset_jobs) > 0
    for job in deepset_jobs:
        assert "MODEL_ARCH_VERSION" not in job.env_vars, \
            f"Job {job.label} unexpectedly contains MODEL_ARCH_VERSION"


# ---------------------------------------------------------------------------
# Checkpoint preflight tests
# ---------------------------------------------------------------------------

def _patch_submit_with_ckpt(collector: JobCollector, checkpoint_exists: bool):
    """Extend _patch_submit to also mock _stage_file_exists."""
    from contextlib import ExitStack

    def _mock_submit_synreg(session, label, compute_pool, env_vars, runtime_environment,
                             entrypoint="evaluate_linear_regression.py",
                             target_instances=1, pip_requirements=None,
                             external_access_integrations=None):
        return collector.submit(
            label=label,
            compute_pool=compute_pool,
            env_vars=env_vars,
            runtime_environment=runtime_environment,
            entrypoint=entrypoint,
            target_instances=target_instances,
            pip_requirements=pip_requirements,
            external_access_integrations=external_access_integrations,
        )

    @contextmanager
    def _ctx():
        with ExitStack() as stack:
            stack.enter_context(patch.object(orch, "_submit_synreg", side_effect=_mock_submit_synreg))
            stack.enter_context(patch.object(orch, "_ensure_compute_pool_usable", return_value=None))
            stack.enter_context(patch.object(orch, "_stage_file_exists", return_value=checkpoint_exists))
            yield

    return _ctx()


def test_main_deepset_raises_when_checkpoint_missing(collector, fake_session, runtime_args):
    """run_synthetic_regression_deepset_evaluation raises RuntimeError when checkpoint absent."""
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        with _patch_submit_with_ckpt(collector, checkpoint_exists=False):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
    assert len(collector.submitted) == 0, "No GPU jobs should be submitted when preflight fails"


def test_main_deepset_proceeds_when_checkpoint_present(collector, fake_session, runtime_args):
    """run_synthetic_regression_deepset_evaluation submits GPU shards when checkpoint present."""
    with _patch_submit_with_ckpt(collector, checkpoint_exists=True):
        with patch.object(orch, "_wait_job_group"):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
    deepset_jobs = [j for j in collector.submitted if "deepset_shard" in j.label]
    assert len(deepset_jobs) == orch.SYNREG_GPU_SHARDS


def test_ood_pilot_raises_when_checkpoint_missing(collector, fake_session):
    """run_synthetic_regression_ood_deepset_pilot raises RuntimeError when checkpoint absent."""
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        with _patch_submit_with_ckpt(collector, checkpoint_exists=False):
            orch.run_synthetic_regression_ood_deepset_pilot(fake_session, "2.5.0-py311")


def test_ood_full_deepset_raises_when_checkpoint_missing(collector, fake_session):
    """run_synthetic_regression_ood_full_deepset_evaluation raises RuntimeError when checkpoint absent."""
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        with _patch_submit_with_ckpt(collector, checkpoint_exists=False):
            orch.run_synthetic_regression_ood_full_deepset_evaluation(fake_session)


def test_combined_deepset_raises_when_checkpoint_missing(collector, fake_session):
    """run_synthetic_regression_combined_deepset_evaluation raises RuntimeError when checkpoint absent."""
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        with _patch_submit_with_ckpt(collector, checkpoint_exists=False):
            orch.run_synthetic_regression_combined_deepset_evaluation(fake_session)


def test_checkpoint_preflight_error_includes_compute_pool(fake_session, runtime_args):
    """Preflight RuntimeError message includes the GPU compute pool name."""
    collector = JobCollector()
    with pytest.raises(RuntimeError) as exc_info:
        with _patch_submit_with_ckpt(collector, checkpoint_exists=False):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
    assert orch.DEEPSET_GPU_POOL in str(exc_info.value)


def test_checkpoint_preflight_error_includes_list_hint(fake_session, runtime_args):
    """Preflight RuntimeError message includes a LIST remediation hint."""
    collector = JobCollector()
    with pytest.raises(RuntimeError) as exc_info:
        with _patch_submit_with_ckpt(collector, checkpoint_exists=False):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
    assert "LIST" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _AG_PROBE_IMPORTS and AutoGluon memory guard tests
# ---------------------------------------------------------------------------

def test_ag_probe_imports_includes_ray():
    """_AG_PROBE_IMPORTS must include 'ray' for the distributed AutoGluon runtime probe."""
    assert "ray" in orch._AG_PROBE_IMPORTS, (
        "_AG_PROBE_IMPORTS is missing 'ray'. Distributed AutoGluon (ray_work_items mode) "
        "requires ray to be available in the AG runtime environment."
    )


def test_autogluon_cluster_shards_default_is_6():
    """SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT must be 6 (distributed mode: 6 × 4 workers)."""
    assert orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT == 6


def test_autogluon_workers_per_shard_default_is_4():
    """SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT must be 4 (6 × 4 = 24 total CPUs)."""
    assert orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT == 4
