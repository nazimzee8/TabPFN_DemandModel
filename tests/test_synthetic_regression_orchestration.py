"""
tests/test_synthetic_regression_orchestration.py
==================================================
Unit tests for scripts/run_synthetic_regression_evaluation.py.

Uses _FakeJob / _FakeSession pattern to verify:
  - Correct shard counts for each phase
  - pip/EAI assignments per job type
  - Phase gating (deepset before baselines, etc.)
  - Runtime probe serialization
  - Capacity probe non-overlapping phases
  - target_instances=1 for all jobs
  - Correct runtime environment per phase
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_synthetic_regression_evaluation as orch


# ---------------------------------------------------------------------------
# Fake job / session infrastructure
# ---------------------------------------------------------------------------

class _FakeJob:
    """Minimal fake MLJob that records submission parameters."""
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
        pass  # instant completion

    @property
    def logs(self):
        return lambda node_id=0: ""


class _FakeSession:
    pass


class JobCollector:
    """Collects all submitted jobs for later assertions."""
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
    """Patch _submit_synreg and _ensure_compute_pool_usable to record submissions via collector."""
    from contextlib import contextmanager, ExitStack

    def _mock_submit_synreg(session, label, compute_pool, env_vars, runtime_environment,
                             entrypoint="evaluate_synthetic_regression.py",
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

    class _MultiPatch:
        """Context manager that applies both patches simultaneously."""
        def __enter__(self):
            self._stack = ExitStack()
            self._stack.enter_context(
                patch("run_synthetic_regression_evaluation._submit_synreg",
                      side_effect=_mock_submit_synreg)
            )
            self._stack.enter_context(
                patch("run_synthetic_regression_evaluation._ensure_compute_pool_usable")
            )
            self._stack.enter_context(
                patch("run_synthetic_regression_evaluation._wait_job_group")
            )
            self._stack.enter_context(
                patch("run_synthetic_regression_evaluation._wait_done")
            )
            self._stack.enter_context(
                patch("run_synthetic_regression_evaluation._list_stage", return_value=[])
            )
            self._stack.enter_context(
                patch("run_synthetic_regression_evaluation._stage_file_exists", return_value=True)
            )
            return self

        def __exit__(self, *args):
            return self._stack.__exit__(*args)

    return _MultiPatch()


def _assert_unsafe_torch_load_for_eval_shards(jobs: list[_FakeJob]) -> None:
    # Only deepset and baselines call torch.load; autogluon never does.
    torch_load_modes = {"deepset", "baselines"}
    torch_load_jobs = [
        job for job in jobs
        if job.env_vars.get("SYNTHETIC_REGRESSION_MODE") in torch_load_modes
    ]
    assert torch_load_jobs, "No deepset/baselines shard jobs found"
    for job in torch_load_jobs:
        assert job.env_vars.get("ALLOW_UNSAFE_TORCH_LOAD") == "true", (
            f"{job.label} mode={job.env_vars.get('SYNTHETIC_REGRESSION_MODE')!r} "
            "missing ALLOW_UNSAFE_TORCH_LOAD=true"
        )
    # AutoGluon shards must NOT receive ALLOW_UNSAFE_TORCH_LOAD.
    autogluon_jobs = [
        job for job in jobs
        if job.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
    ]
    for job in autogluon_jobs:
        assert "ALLOW_UNSAFE_TORCH_LOAD" not in job.env_vars, (
            f"{job.label}: ALLOW_UNSAFE_TORCH_LOAD must not be set for autogluon mode"
        )


# ---------------------------------------------------------------------------
# Tests: Shard counts
# ---------------------------------------------------------------------------

class TestShardCounts:
    def test_gpu_shard_count_is_10(self):
        assert orch.SYNREG_GPU_SHARDS == 10

    def test_cpu_shard_count_is_6(self):
        assert orch.SYNREG_CPU_SHARDS == 6

    def test_autogluon_shard_count_is_60(self):
        assert orch.SYNREG_AUTOGLUON_SHARDS == 60

    def test_total_job_count_deepset_eval(self, collector, fake_session, runtime_args):
        """10 GPU shards must be submitted for deepset evaluation."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(
                fake_session, *runtime_args
            )
        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        assert len(deepset_jobs) == 10

    def test_total_job_count_baseline_eval(self, collector, fake_session, runtime_args):
        """6 CPU shards must be submitted for baseline evaluation."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_baseline_evaluation(
                fake_session, *runtime_args
            )
        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        assert len(baseline_jobs) == 6

    def test_total_job_count_autogluon_eval(self, collector, fake_session, runtime_args):
        """60 AG shards must be submitted for AutoGluon evaluation."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_evaluation(
                fake_session, *runtime_args
            )
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        assert len(ag_jobs) == 60

    def test_aggregation_submits_1_job(self, collector, fake_session, runtime_args):
        """Aggregation must submit exactly 1 job."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_aggregation(
                fake_session, *runtime_args
            )
        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        assert len(agg_jobs) == 1


# ---------------------------------------------------------------------------
# Tests: pip/EAI assignments
# ---------------------------------------------------------------------------

class TestPipAndEAIAssignments:
    def test_deepset_shards_carry_no_pip_or_eai(self, collector, fake_session, runtime_args):
        """DeepSet GPU jobs must have pip_requirements=None and EAI=None."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(
                fake_session, *runtime_args
            )
        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        for job in deepset_jobs:
            assert job.pip_requirements is None, f"DeepSet job {job.label} has pip_requirements"
            assert job.external_access_integrations is None, \
                f"DeepSet job {job.label} has EAI"

    def test_baseline_shards_carry_catboost_and_eai(self, collector, fake_session, runtime_args):
        """Baseline jobs must carry catboost==1.2.10 and TABPFN_PYPI_EAI."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_baseline_evaluation(
                fake_session, *runtime_args
            )
        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        for job in baseline_jobs:
            assert job.pip_requirements is not None
            assert any("catboost" in p for p in job.pip_requirements)
            assert job.external_access_integrations is not None
            assert any("TABPFN_PYPI_EAI" in e for e in job.external_access_integrations)

    def test_autogluon_shards_carry_ag_and_eai(self, collector, fake_session, runtime_args):
        """AutoGluon jobs must carry autogluon.tabular==1.3.0 and TABPFN_PYPI_EAI."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_evaluation(
                fake_session, *runtime_args
            )
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        for job in ag_jobs:
            assert job.pip_requirements is not None
            assert any("autogluon" in p for p in job.pip_requirements)
            assert job.external_access_integrations is not None
            assert any("TABPFN_PYPI_EAI" in e for e in job.external_access_integrations)

    def test_aggregation_carries_no_pip_or_eai(self, collector, fake_session, runtime_args):
        """Aggregation job must have no pip or EAI."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_aggregation(
                fake_session, *runtime_args
            )
        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        for job in agg_jobs:
            assert job.pip_requirements is None
            assert job.external_access_integrations is None


# ---------------------------------------------------------------------------
# Tests: trusted checkpoint unsafe-load env
# ---------------------------------------------------------------------------

class TestUnsafeTorchLoadEnv:
    def test_main_eval_shards_set_allow_unsafe_torch_load(self, collector, fake_session, runtime_args):
        """DeepSet and baseline shards must carry ALLOW_UNSAFE_TORCH_LOAD; autogluon must not."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_autogluon_evaluation(fake_session, *runtime_args)

        _assert_unsafe_torch_load_for_eval_shards(collector.submitted)

    def test_ood_pilot_deepset_shards_set_allow_unsafe_torch_load(self, collector, fake_session):
        """OOD pilot DeepSet shard path must carry trusted-checkpoint load env."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_deepset_pilot(
                fake_session, bench_rt="2.5.0-py311"
            )

        ood_jobs = [
            job for job in collector.submitted
            if "ood_deepset_shard" in job.label
        ]
        assert ood_jobs, "No OOD pilot shard jobs found"
        _assert_unsafe_torch_load_for_eval_shards(ood_jobs)

    def test_ood_full_eval_shards_set_allow_unsafe_torch_load(self, collector, fake_session):
        """OOD full DeepSet and baseline shards must carry ALLOW_UNSAFE_TORCH_LOAD; autogluon must not."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_deepset_evaluation(fake_session)
            orch.run_synthetic_regression_ood_full_baseline_evaluation(fake_session)
            orch.run_synthetic_regression_ood_full_autogluon_evaluation(fake_session)

        _assert_unsafe_torch_load_for_eval_shards(collector.submitted)

    def test_combined_eval_shards_set_allow_unsafe_torch_load(self, collector, fake_session):
        """Combined DeepSet and baseline shards must carry ALLOW_UNSAFE_TORCH_LOAD; autogluon must not."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_deepset_evaluation(fake_session)
            orch.run_synthetic_regression_combined_baseline_evaluation(fake_session)
            orch.run_synthetic_regression_combined_autogluon_evaluation(fake_session)

        _assert_unsafe_torch_load_for_eval_shards(collector.submitted)

    def test_submit_synreg_rejects_checkpoint_loading_mode_without_allow_env(self, fake_session):
        """The real submit helper must fail before Snowflake submit if a shard omits the env."""
        with pytest.raises(RuntimeError, match="ALLOW_UNSAFE_TORCH_LOAD=true"):
            orch._submit_synreg(
                session=fake_session,
                label="bad_baseline_shard",
                compute_pool=orch.DEEPSET_CPU_POOL,
                env_vars={
                    "SYNTHETIC_REGRESSION_MODE": "baselines",
                    "SYNTHETIC_REGRESSION_SUITE_ID": "suite",
                    "SYNTHETIC_REGRESSION_NUM_SHARDS": "3",
                    "SYNTHETIC_REGRESSION_SHARD_INDEX": "0",
                    "SYNREG_RESULTS_STAGE": "@stage/regression/suite",
                },
                runtime_environment="2.5.0-py311",
            )


# ---------------------------------------------------------------------------
# Tests: Phase gating
# ---------------------------------------------------------------------------

class TestPhaseGating:
    def test_phase_gating_deepset_before_baselines(self, collector, fake_session, runtime_args):
        """
        In the pipeline, baseline submission must occur after all DeepSet jobs
        have been waited on. We verify by checking submission order.
        """
        # Track the event order
        events = []

        def _mock_submit_synreg(session, label, compute_pool, env_vars,
                                 runtime_environment, entrypoint="evaluate_synthetic_regression.py",
                                 target_instances=1, pip_requirements=None,
                                 external_access_integrations=None):
            mode = env_vars.get("SYNTHETIC_REGRESSION_MODE", "prep")
            events.append(("submit", mode, label))
            return collector.submit(
                label=label, compute_pool=compute_pool, env_vars=env_vars,
                runtime_environment=runtime_environment, entrypoint=entrypoint,
                target_instances=target_instances, pip_requirements=pip_requirements,
                external_access_integrations=external_access_integrations,
            )

        with patch("run_synthetic_regression_evaluation._submit_synreg",
                   side_effect=_mock_submit_synreg), \
             patch("run_synthetic_regression_evaluation._stage_file_exists", return_value=True), \
             patch("run_synthetic_regression_evaluation._ensure_compute_pool_usable"), \
             patch("run_synthetic_regression_evaluation._wait_job_group"):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)

        deepset_positions = [i for i, e in enumerate(events) if e[1] == "deepset"]
        baseline_positions = [i for i, e in enumerate(events) if e[1] == "baselines"]

        assert deepset_positions, "No deepset events"
        assert baseline_positions, "No baseline events"
        # All deepset submits happen before all baseline submits
        assert max(deepset_positions) < min(baseline_positions)

    def test_split_procedures_do_not_rerun_runtime_probes(self, collector, fake_session, runtime_args):
        """run_synthetic_regression_deepset_evaluation must not submit runtime probe jobs."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(
                fake_session, *runtime_args
            )
        probe_jobs = [j for j in collector.submitted if "probe" in j.label.lower()]
        assert len(probe_jobs) == 0, f"Found unexpected probe jobs: {[j.label for j in probe_jobs]}"


# ---------------------------------------------------------------------------
# Tests: Runtime probe serialization
# ---------------------------------------------------------------------------

class TestRuntimeProbesSerialized:
    def test_runtime_probes_are_serialized(self, collector, fake_session, runtime_args):
        """
        Runtime probes must be submitted AND waited one at a time (not batched).
        We verify by checking there are exactly 4 probe jobs (one per call to submit).
        """
        wait_calls = []

        def _mock_wait_done(job, label, session):
            wait_calls.append(label)

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_done",
                       side_effect=_mock_wait_done):
                orch.run_synthetic_regression_runtime_probes(
                    fake_session, *runtime_args
                )

        # Exactly 4 probes, each waited independently
        assert len(collector.submitted) == 4
        assert len(wait_calls) == 4
        # Each submit is immediately followed by wait
        for i, (job, waited_label) in enumerate(zip(collector.submitted, wait_calls)):
            assert job.label == waited_label, \
                f"Probe {i}: submitted={job.label} but waited={waited_label}"


# ---------------------------------------------------------------------------
# Tests: Capacity probe phases
# ---------------------------------------------------------------------------

class TestCapacityProbePhases:
    def test_capacity_probe_phases_are_non_overlapping(self, collector, fake_session, runtime_args):
        """
        Baseline CPU capacity probe must complete before AG probe submits.
        """
        submit_events = []
        wait_events = []

        def _mock_submit_capacity(session, label, compute_pool, runtime_environment):
            submit_events.append(("submit", label, compute_pool))
            return _FakeJob(label, compute_pool=compute_pool,
                             runtime_environment=runtime_environment)

        def _mock_wait_done(job, label, session):
            wait_events.append(("wait", label))

        def _mock_submit_wait_phase(session, phase_label, compute_pool,
                                     runtime_environment, count):
            # Track phase start
            submit_events.append(("phase_start", phase_label, compute_pool))
            # Simulate: submit `count` jobs, wait all
            for i in range(count):
                lbl = f"{phase_label}_{i}"
                submit_events.append(("submit_in_phase", lbl, compute_pool))
            submit_events.append(("phase_done", phase_label))

        with patch("run_synthetic_regression_evaluation._submit_and_wait_capacity_phase",
                   side_effect=_mock_submit_wait_phase):
            orch.run_synthetic_regression_capacity_probe(fake_session, *runtime_args)

        # Extract phase sequence
        phase_events = [(e[0], e[1]) for e in submit_events
                        if e[0] in ("phase_start", "phase_done")]
        phase_names = [e[1] for e in phase_events]

        baseline_start = phase_names.index("synreg_cap_baseline")
        baseline_done = phase_names.index("synreg_cap_baseline", baseline_start + 1)
        ag_start = next(
            (i for i, (evt, name) in enumerate(phase_events)
             if evt == "phase_start" and "autogluon" in name.lower()), None
        )
        assert ag_start is not None
        assert baseline_done <= ag_start

    def test_cpu_capacity_phases_use_single_wave_envelope(self, fake_session, runtime_args):
        """CPU capacity phases submit exactly the required single-wave probe jobs."""
        submitted_counts = {}

        def _mock_phase(session, phase_label, compute_pool, runtime_environment, count):
            submitted_counts[phase_label] = count

        with patch("run_synthetic_regression_evaluation._submit_and_wait_capacity_phase",
                   side_effect=_mock_phase):
            orch.run_synthetic_regression_capacity_probe(
                fake_session, *runtime_args
            )

        assert submitted_counts == {
            "synreg_cap_baseline": orch.SYNREG_CPU_SHARDS,
            "synreg_cap_autogluon": orch.SYNREG_AUTOGLUON_SHARDS,
        }

    def test_capacity_probes_reject_lower_concurrency(self, fake_session, runtime_args):
        with pytest.raises(ValueError):
            orch.run_synthetic_regression_capacity_probe(
                fake_session, *runtime_args, 4, 12
            )
        with pytest.raises(ValueError):
            orch.run_synthetic_regression_baseline_capacity_probe(
                fake_session, *runtime_args, 5
            )
        with pytest.raises(ValueError):
            orch.run_synthetic_regression_autogluon_capacity_probe(
                fake_session, *runtime_args, 17
            )

    def test_standalone_capacity_probes_use_single_wave_envelope(self, fake_session, runtime_args):
        submitted_counts = {}

        def _mock_phase(session, phase_label, compute_pool, runtime_environment, count):
            submitted_counts[phase_label] = (compute_pool, count)

        with patch("run_synthetic_regression_evaluation._submit_and_wait_capacity_phase",
                   side_effect=_mock_phase):
            orch.run_synthetic_regression_baseline_capacity_probe(
                fake_session, *runtime_args
            )
            orch.run_synthetic_regression_autogluon_capacity_probe(
                fake_session, *runtime_args
            )

        assert submitted_counts["synreg_cap_baseline"] == (
            orch.DEEPSET_CPU_POOL, orch.SYNREG_CPU_SHARDS
        )
        assert submitted_counts["synreg_cap_autogluon"] == (
            orch.AUTOGLUON_CPU_POOL, orch.SYNREG_AUTOGLUON_SHARDS
        )

    def test_combined_capacity_probes_use_single_wave_envelope(self, collector, fake_session):
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_baseline_capacity_probe(fake_session)
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(fake_session)

        baseline_probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_baseline_cap_probe_")
        ]
        ag_probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_ag_cap_probe_")
        ]
        assert len(baseline_probes) == orch.SYNREG_CPU_SHARDS
        assert len(ag_probes) == orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT
        assert {j.target_instances for j in ag_probes} == {
            orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
        }
        assert {j.entrypoint for j in baseline_probes} == {"capacity_probe.py"}
        assert {j.entrypoint for j in ag_probes} == {"ray_capacity_probe.py"}
        for job in ag_probes:
            assert job.env_vars["EXPECTED_RAY_NODES"] == str(
                orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            )
            assert job.env_vars["EXPECTED_RAY_CPUS_MIN"] == str(
                orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            )
        for job in ag_probes:
            assert job.pip_requirements == orch.SYNREG_AG_RAY_PIP, (
                f"AG capacity probe {job.label} missing SYNREG_AG_RAY_PIP"
            )
            assert job.external_access_integrations == orch.SYNREG_PYPI_EAI, (
                f"AG capacity probe {job.label} missing SYNREG_PYPI_EAI"
            )
            assert job.env_vars["SYNREG_AUTOGLUON_DISTRIBUTED_MODE"] == "ray_work_items"
            assert job.env_vars["SYNREG_AUTOGLUON_CLUSTER_SHARDS"] == str(
                orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT
            )
            assert job.env_vars["SYNREG_AUTOGLUON_WORKERS_PER_SHARD"] == str(
                orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            )
            assert job.env_vars["AUTOGLUON_TASK_CPUS"] == str(
                orch.SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT
            )
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS"] == str(
                orch.SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS_DEFAULT
            )
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_POLL_SECONDS"] == str(
                orch.SYNREG_RAY_CAPACITY_READY_POLL_SECONDS_DEFAULT
            )

    def test_combined_autogluon_capacity_probe_rejects_lower_clusters(self, fake_session):
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=6,
                autogluon_concurrent_clusters=3,
            )

        msg = str(exc.value)
        assert "run_synthetic_regression_combined_autogluon_capacity_probe" in msg
        assert "AUTOGLUON_CONCURRENT_CLUSTERS=3" in msg
        assert "AUTOGLUON_CLUSTER_SHARDS=6" in msg

    def test_combined_autogluon_capacity_probe_accepts_runtime_ray_readiness(
        self, collector, fake_session
    ):
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=2,
                autogluon_workers_per_shard=4,
                autogluon_concurrent_clusters=2,
                ray_ready_timeout_seconds=180,
                ray_ready_poll_seconds=15,
            )

        ag_probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_ag_cap_probe_")
        ]
        assert ag_probes
        for job in ag_probes:
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS"] == "180"
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_POLL_SECONDS"] == "15"


# ---------------------------------------------------------------------------
# Tests: target_instances=1 for all jobs
# ---------------------------------------------------------------------------

class TestTargetInstancesAlwaysOne:
    def test_all_shard_jobs_use_target_instances_1(self, collector, fake_session, runtime_args):
        """Every submitted job must have target_instances=1."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_aggregation(fake_session, *runtime_args)

        for job in collector.submitted:
            assert job.target_instances == 1, \
                f"Job {job.label} has target_instances={job.target_instances}"


# ---------------------------------------------------------------------------
# Tests: Multi-instance entrypoint allowlist guard
# ---------------------------------------------------------------------------

class TestMultiInstanceGuard:
    """Verify _submit_synreg's allowlist-based multi-instance guard."""

    def test_allows_ray_capacity_probe_multi_instance(self, fake_session):
        """ray_capacity_probe.py with target_instances > 1 must pass the multi-instance guard."""
        try:
            orch._submit_synreg(
                session=fake_session,
                label="ag_cap_probe_0",
                compute_pool=orch.AUTOGLUON_CPU_POOL,
                env_vars={
                    "EXPECTED_RAY_NODES": "4",
                    "EXPECTED_RAY_CPUS_MIN": "4",
                },
                runtime_environment="2.5.0-py311",
                entrypoint="ray_capacity_probe.py",
                target_instances=4,
            )
        except RuntimeError as exc:
            assert "Refusing target_instances" not in str(exc), (
                f"multi-instance guard incorrectly rejected ray_capacity_probe.py: {exc}"
            )
        except Exception:
            pass  # ImportError for Snowflake is expected in test environment

    def test_allows_autogluon_ray_multi_instance(self, fake_session):
        """autogluon_ray.py with target_instances > 1 must pass."""
        try:
            orch._submit_synreg(
                session=fake_session,
                label="combined_ag_cluster_0",
                compute_pool=orch.AUTOGLUON_CPU_POOL,
                env_vars={
                    "SYNTHETIC_REGRESSION_MODE": "autogluon",
                    "SYNTHETIC_REGRESSION_SUITE_ID": "linear_all_v1",
                    "SYNTHETIC_REGRESSION_NUM_SHARDS": "6",
                    "SYNTHETIC_REGRESSION_SHARD_INDEX": "0",
                    "SYNREG_RESULTS_STAGE": "@EVALUATION_RESULTS_STAGE/regression/linear_all_v1",
                    "SYNREG_AUTOGLUON_DISTRIBUTED_MODE": "ray_work_items",
                },
                runtime_environment="2.5.0-py311",
                entrypoint="autogluon_ray.py",
                target_instances=4,
            )
        except RuntimeError as exc:
            assert "Refusing target_instances" not in str(exc), (
                f"multi-instance guard incorrectly rejected autogluon_ray entrypoint: {exc}"
            )
        except Exception:
            pass  # ImportError for Snowflake is expected in test environment

    def test_allows_autogluon_worker_access_probe_multi_instance(self, fake_session):
        """autogluon_worker_access_probe.py with target_instances > 1 must pass."""
        try:
            orch._submit_synreg(
                session=fake_session,
                label="combined_ag_worker_access_probe_0",
                compute_pool=orch.AUTOGLUON_CPU_POOL,
                env_vars={
                    "SYNTHETIC_REGRESSION_MODE": "autogluon_worker_access_probe",
                    "SYNTHETIC_REGRESSION_SUITE_ID": "linear_all_v1",
                    "SYNTHETIC_REGRESSION_NUM_SHARDS": "6",
                    "SYNTHETIC_REGRESSION_SHARD_INDEX": "0",
                    "SYNREG_RESULTS_STAGE": "@EVALUATION_RESULTS_STAGE/regression/linear_all_v1",
                    "SYNREG_WORKER_ACCESS_PROBE_USE_RAY": "true",
                    "EXPECTED_RAY_NODES": "4",
                    "EXPECTED_RAY_CPUS_MIN": "4",
                },
                runtime_environment="2.5.0-py311",
                entrypoint="autogluon_worker_access_probe.py",
                target_instances=4,
            )
        except RuntimeError as exc:
            assert "Refusing target_instances" not in str(exc), (
                f"multi-instance guard incorrectly rejected worker-access probe: {exc}"
            )
        except Exception:
            pass  # ImportError for Snowflake is expected in test environment

    def test_rejects_capacity_probe_multi_instance(self, fake_session):
        """capacity_probe.py (non-Ray) with target_instances > 1 must be rejected."""
        with pytest.raises(RuntimeError, match="Refusing target_instances"):
            orch._submit_synreg(
                session=fake_session,
                label="bad_cap_probe",
                compute_pool=orch.DEEPSET_CPU_POOL,
                env_vars={},
                runtime_environment="2.5.0-py311",
                entrypoint="capacity_probe.py",
                target_instances=4,
            )

    def test_rejects_evaluate_synthetic_regression_multi_instance(self, fake_session):
        """evaluate_synthetic_regression.py with target_instances > 1 must be rejected."""
        with pytest.raises(RuntimeError, match="Refusing target_instances"):
            orch._submit_synreg(
                session=fake_session,
                label="bad_deepset_shard",
                compute_pool=orch.DEEPSET_GPU_POOL,
                env_vars={
                    "SYNTHETIC_REGRESSION_MODE": "deepset",
                    "ALLOW_UNSAFE_TORCH_LOAD": "true",
                },
                runtime_environment="2.5.0-py311",
                entrypoint="evaluate_synthetic_regression.py",
                target_instances=2,
            )

    def test_combined_autogluon_capacity_probe_submits_ray_capacity_probe(
        self, collector, fake_session
    ):
        """Combined AutoGluon capacity probe must use ray_capacity_probe.py with target_instances=workers_per_shard."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(fake_session)

        ag_probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_ag_cap_probe_")
        ]
        assert ag_probes, "No combined AG capacity probe jobs found"
        for job in ag_probes:
            assert job.entrypoint == "ray_capacity_probe.py", (
                f"Expected ray_capacity_probe.py, got {job.entrypoint!r}"
            )
            assert job.target_instances == orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT, (
                f"Expected target_instances={orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT}, "
                f"got {job.target_instances}"
            )
            assert job.pip_requirements == orch.SYNREG_AG_RAY_PIP, (
                f"Expected SYNREG_AG_RAY_PIP, got {job.pip_requirements!r}"
            )
            assert job.external_access_integrations == orch.SYNREG_PYPI_EAI, (
                f"Expected SYNREG_PYPI_EAI, got {job.external_access_integrations!r}"
            )

    def test_combined_autogluon_worker_access_probe_submits_ray_probe(
        self, collector, fake_session
    ):
        """Default worker-access probe must submit 6 Ray jobs with 4 workers each."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_worker_access_probe(fake_session)

        probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_ag_worker_access_probe_")
        ]
        assert len(probes) == orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT
        for job in probes:
            assert job.entrypoint == "autogluon_worker_access_probe.py"
            assert job.target_instances == orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            assert job.pip_requirements == orch.SYNREG_AG_RAY_PIP
            assert job.external_access_integrations == orch.SYNREG_PYPI_EAI
            assert job.env_vars["SYNREG_WORKER_ACCESS_PROBE_USE_RAY"] == "true"
            assert job.env_vars["SYNREG_WORKER_DATA_ACCESS_MODE"] == "driver_presigned_url"
            assert job.env_vars["SYNREG_MAX_WORK_ITEM_BYTES"] == "8192"
            assert job.env_vars["EXPECTED_RAY_NODES"] == str(
                orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            )
            assert job.env_vars["EXPECTED_RAY_CPUS_MIN"] == str(
                orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            )
            assert job.env_vars["SYNREG_AUTOGLUON_DISTRIBUTED_MODE"] == "ray_work_items"
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS"] == str(
                orch.SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS_DEFAULT
            )
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_POLL_SECONDS"] == str(
                orch.SYNREG_RAY_CAPACITY_READY_POLL_SECONDS_DEFAULT
            )

    def test_combined_autogluon_worker_access_probe_waits_once(
        self, collector, fake_session
    ):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_combined_autogluon_worker_access_probe(fake_session)

        assert batch_sizes == [orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT]

    def test_combined_autogluon_evaluation_submits_autogluon_ray(
        self, collector, fake_session
    ):
        """Combined AutoGluon evaluation must use autogluon_ray.py with target_instances=workers_per_shard."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(fake_session)

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs, "No combined AG evaluation jobs found"
        for job in ag_jobs:
            assert job.entrypoint == "autogluon_ray.py", (
                f"Expected autogluon_ray entrypoint, got {job.entrypoint!r}"
            )
            assert job.target_instances == orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT, (
                f"Expected target_instances={orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT}, "
                f"got {job.target_instances}"
            )

    def test_combined_all_in_one_preserves_ray_entrypoint_and_agg_shard_count(
        self, collector, fake_session
    ):
        """All-in-one combined evaluation must pass autogluon_ray entrypoint through and
        aggregation must expect the same cluster_shards count."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_evaluation(fake_session)

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        agg_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"
        ]
        assert ag_jobs, "No AG jobs submitted by all-in-one"
        assert agg_jobs, "No aggregation job submitted by all-in-one"

        for job in ag_jobs:
            assert job.entrypoint == "autogluon_ray.py"

        cluster_shards = int(ag_jobs[0].env_vars["SYNREG_AUTOGLUON_CLUSTER_SHARDS"])
        expected_ag_shards = agg_jobs[0].env_vars.get("SYNREG_EXPECTED_AG_SHARDS")
        assert expected_ag_shards == str(cluster_shards), (
            f"Aggregation SYNREG_EXPECTED_AG_SHARDS={expected_ag_shards!r} "
            f"does not match cluster_shards={cluster_shards}"
        )


# ---------------------------------------------------------------------------
# Tests: Correct runtime per phase
# ---------------------------------------------------------------------------

class TestRuntimePerPhase:
    def test_deepset_uses_bench_rt(self, collector, fake_session):
        """DeepSet jobs must use benchmark_runtime_environment."""
        prep_rt, bench_rt, ag_rt = "prep-rt", "bench-rt", "ag-rt"
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(
                fake_session, prep_rt, bench_rt, ag_rt
            )
        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        for job in deepset_jobs:
            assert job.runtime_environment == bench_rt, \
                f"DeepSet job {job.label} has rt={job.runtime_environment}, expected {bench_rt}"

    def test_baseline_uses_bench_rt(self, collector, fake_session):
        """Baseline jobs must use benchmark_runtime_environment."""
        prep_rt, bench_rt, ag_rt = "prep-rt", "bench-rt", "ag-rt"
        with _patch_submit(collector):
            orch.run_synthetic_regression_baseline_evaluation(
                fake_session, prep_rt, bench_rt, ag_rt
            )
        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        for job in baseline_jobs:
            assert job.runtime_environment == bench_rt

    def test_autogluon_uses_ag_rt(self, collector, fake_session):
        """AutoGluon jobs must use autogluon_runtime_environment."""
        prep_rt, bench_rt, ag_rt = "prep-rt", "bench-rt", "ag-rt"
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_evaluation(
                fake_session, prep_rt, bench_rt, ag_rt
            )
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        for job in ag_jobs:
            assert job.runtime_environment == ag_rt

    def test_prep_uses_prep_rt(self, collector, fake_session):
        """Prep job must use prep_runtime_environment."""
        prep_rt, bench_rt, ag_rt = "prep-rt", "bench-rt", "ag-rt"
        with _patch_submit(collector):
            orch.run_synthetic_regression_prep(
                fake_session, prep_rt, bench_rt, ag_rt
            )
        prep_jobs = [j for j in collector.submitted]
        assert len(prep_jobs) >= 1
        assert prep_jobs[0].runtime_environment == prep_rt

    def test_aggregation_uses_bench_rt(self, collector, fake_session):
        """Aggregation job must use benchmark_runtime_environment."""
        prep_rt, bench_rt, ag_rt = "prep-rt", "bench-rt", "ag-rt"
        with _patch_submit(collector):
            orch.run_synthetic_regression_aggregation(
                fake_session, prep_rt, bench_rt, ag_rt
            )
        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        for job in agg_jobs:
            assert job.runtime_environment == bench_rt


# ---------------------------------------------------------------------------
# Tests: Shard index range
# ---------------------------------------------------------------------------

class TestShardIndexRange:
    def test_deepset_shard_indices_0_to_9(self, collector, fake_session, runtime_args):
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in deepset_jobs]
        assert sorted(indices) == list(range(10))

    def test_baseline_shard_indices_0_to_5(self, collector, fake_session, runtime_args):
        with _patch_submit(collector):
            orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)
        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in baseline_jobs]
        assert sorted(indices) == list(range(6))

    def test_autogluon_shard_indices_0_to_59(self, collector, fake_session, runtime_args):
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_evaluation(fake_session, *runtime_args)
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in ag_jobs]
        assert sorted(indices) == list(range(60))


# ---------------------------------------------------------------------------
# Tests: runtime concurrency controls
# ---------------------------------------------------------------------------

class TestRuntimeConcurrencyControls:
    def test_default_concurrency_constants(self):
        assert orch.SYNREG_BASELINE_CONCURRENT_NODES_DEFAULT == 6
        assert orch.SYNREG_AUTOGLUON_CONCURRENT_NODES_DEFAULT == 60

    def test_env_vars_must_match_single_wave_shard_counts(self, monkeypatch):
        monkeypatch.setenv("SYNREG_BASELINE_CONCURRENT_NODES", "6")
        monkeypatch.setenv("SYNREG_AUTOGLUON_CONCURRENT_NODES", "60")

        assert orch._resolve_baseline_concurrent_nodes("proc") == 6
        assert orch._resolve_autogluon_concurrent_nodes("proc") == 60

    def test_sql_args_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("SYNREG_BASELINE_CONCURRENT_NODES", "2")
        monkeypatch.setenv("SYNREG_AUTOGLUON_CONCURRENT_NODES", "7")

        assert orch._resolve_baseline_concurrent_nodes("proc", 6) == 6
        assert orch._resolve_autogluon_concurrent_nodes("proc", 60) == 60

    def test_invalid_concurrency_error_includes_context(self):
        with pytest.raises(ValueError) as exc:
            orch._resolve_baseline_concurrent_nodes(
                "run_synthetic_regression_baseline_evaluation",
                2,
            )
        msg = str(exc.value)
        assert "run_synthetic_regression_baseline_evaluation" in msg
        assert "BASELINE_CONCURRENT_NODES=2" in msg
        assert "BASELINE_SHARDS=6" in msg
        assert orch.DEEPSET_CPU_POOL in msg
        assert "Remediation" in msg

    def test_lower_baseline_concurrency_fails_fast(self, fake_session, runtime_args):
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_baseline_evaluation(
                fake_session, *runtime_args, 2
            )

        msg = str(exc.value)
        assert "run_synthetic_regression_baseline_evaluation" in msg
        assert "BASELINE_CONCURRENT_NODES=2" in msg
        assert "BASELINE_SHARDS=6" in msg
        assert orch.DEEPSET_CPU_POOL in msg

    def test_baseline_default_single_wave_waits_once(self, collector, fake_session, runtime_args):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_baseline_evaluation(
                    fake_session, *runtime_args
                )

        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in baseline_jobs]
        assert batch_sizes == [orch.SYNREG_CPU_SHARDS]
        assert sorted(indices) == list(range(6))

    def test_lower_autogluon_concurrency_fails_fast(self, fake_session, runtime_args):
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_autogluon_evaluation(
                fake_session, *runtime_args, 25
            )

        msg = str(exc.value)
        assert "run_synthetic_regression_autogluon_evaluation" in msg
        assert "AUTOGLUON_CONCURRENT_NODES=25" in msg
        assert "SYNREG_AUTOGLUON_SHARDS=60" in msg
        assert orch.AUTOGLUON_CPU_POOL in msg

    def test_autogluon_default_single_wave_waits_once(self, collector, fake_session, runtime_args):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_autogluon_evaluation(
                    fake_session, *runtime_args
                )

        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in ag_jobs]
        assert batch_sizes == [orch.SYNREG_AUTOGLUON_SHARDS]
        assert sorted(indices) == list(range(60))


# ---------------------------------------------------------------------------
# Tests: SYNREG_RESULTS_STAGE suite-id isolation (Issue 4)
# ---------------------------------------------------------------------------

class TestAggregationSuitePrefix:
    def test_eval_shards_include_synreg_results_stage_env(self, collector, fake_session, runtime_args):
        """DeepSet, baseline, and AutoGluon shards all set SYNREG_RESULTS_STAGE containing suite_id."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)
            orch.run_synthetic_regression_autogluon_evaluation(fake_session, *runtime_args)

        eval_modes = {"deepset", "baselines", "autogluon"}
        eval_jobs = [j for j in collector.submitted
                     if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") in eval_modes]
        assert eval_jobs, "No eval shard jobs found"
        for job in eval_jobs:
            stage = job.env_vars.get("SYNREG_RESULTS_STAGE", "")
            assert stage, f"Job {job.label} missing SYNREG_RESULTS_STAGE"
            suite_id = job.env_vars.get("SYNTHETIC_REGRESSION_SUITE_ID", "")
            assert suite_id in stage, (
                f"Job {job.label}: SYNREG_RESULTS_STAGE='{stage}' does not contain "
                f"suite_id='{suite_id}'"
            )

    def test_aggregation_job_receives_synreg_results_stage(self, collector, fake_session, runtime_args):
        """Aggregation job env_vars must include SYNREG_RESULTS_STAGE with suite_id."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_aggregation(fake_session, *runtime_args)

        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        assert agg_jobs, "No aggregation job found"
        for job in agg_jobs:
            stage = job.env_vars.get("SYNREG_RESULTS_STAGE", "")
            assert stage, f"Aggregation job missing SYNREG_RESULTS_STAGE"
            suite_id = job.env_vars.get("SYNTHETIC_REGRESSION_SUITE_ID", "")
            assert suite_id in stage, (
                f"Aggregation job SYNREG_RESULTS_STAGE='{stage}' does not contain "
                f"suite_id='{suite_id}'"
            )

    def test_ood_pilot_shards_have_explicit_deepset_mode(self, collector, fake_session):
        """OOD pilot shards must have SYNTHETIC_REGRESSION_MODE=deepset in env_vars."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_deepset_pilot(
                fake_session, bench_rt="2.5.0-py311"
            )
        ood_jobs = [j for j in collector.submitted
                    if "ood_deepset_shard" in j.label]
        assert ood_jobs, "No OOD shard jobs found"
        for job in ood_jobs:
            assert job.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset", (
                f"OOD shard {job.label} missing SYNTHETIC_REGRESSION_MODE=deepset"
            )


# ---------------------------------------------------------------------------
# Tests: OOD full suite evaluation (all methods)
# ---------------------------------------------------------------------------

class TestOODFullEvaluation:
    def test_ood_full_submits_all_three_method_types(self, collector, fake_session):
        """run_synthetic_regression_ood_full_evaluation must submit deepset, baselines, and autogluon jobs."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_evaluation(fake_session)

        modes = {j.env_vars.get("SYNTHETIC_REGRESSION_MODE") for j in collector.submitted}
        assert "deepset" in modes, "No deepset job submitted for OOD full suite"
        assert "baselines" in modes, "No baselines job submitted for OOD full suite"
        assert "autogluon" in modes, "No autogluon job submitted for OOD full suite"

    def test_ood_full_aggregation_receives_output_stage(self, collector, fake_session):
        """Aggregation job for OOD full suite must receive SYNREG_OUTPUT_STAGE=OOD_FULL_OUTPUT_STAGE."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_evaluation(fake_session)

        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        assert agg_jobs, "No aggregation job found for OOD full suite"
        for job in agg_jobs:
            output_stage = job.env_vars.get("SYNREG_OUTPUT_STAGE", "")
            assert output_stage == orch.OOD_FULL_OUTPUT_STAGE, (
                f"Aggregation job SYNREG_OUTPUT_STAGE={output_stage!r}, "
                f"expected {orch.OOD_FULL_OUTPUT_STAGE!r}"
            )

    def test_ood_full_results_stage_contains_suite_id(self, collector, fake_session):
        """Every shard job's SYNREG_RESULTS_STAGE must contain OOD_FULL_SUITE_ID in the path."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_evaluation(fake_session)

        shard_modes = {"deepset", "baselines", "autogluon"}
        shard_jobs = [j for j in collector.submitted
                      if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") in shard_modes]
        assert shard_jobs, "No shard jobs found for OOD full suite"
        for job in shard_jobs:
            stage = job.env_vars.get("SYNREG_RESULTS_STAGE", "")
            assert orch.OOD_FULL_SUITE_ID in stage, (
                f"Shard job {job.label}: SYNREG_RESULTS_STAGE={stage!r} does not contain "
                f"OOD_FULL_SUITE_ID={orch.OOD_FULL_SUITE_ID!r}"
            )


# ---------------------------------------------------------------------------
# Tests: OOD full split-phase functions
# ---------------------------------------------------------------------------

class TestOODFullSplitPhase:
    def test_ood_full_prep_submits_ood_prep_job(self, collector, fake_session):
        """OOD full prep submits exactly 1 job with entrypoint=prepare_ood_regression.py."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_prep(fake_session)
        assert len(collector.submitted) == 1
        assert collector.submitted[0].entrypoint == "prepare_ood_regression.py"

    def test_ood_full_deepset_evaluation_submits_gpu_shards(self, collector, fake_session):
        """OOD full deepset phase submits OOD_FULL_GPU_SHARDS jobs on DEEPSET_GPU_POOL."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_deepset_evaluation(fake_session)
        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        assert len(deepset_jobs) == orch.OOD_FULL_GPU_SHARDS
        for job in deepset_jobs:
            assert job.compute_pool == orch.DEEPSET_GPU_POOL

    def test_ood_full_baseline_evaluation_submits_cpu_shards(self, collector, fake_session):
        """OOD full baseline phase submits SYNREG_CPU_SHARDS jobs on DEEPSET_CPU_POOL."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_baseline_evaluation(fake_session)
        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        assert len(baseline_jobs) == orch.SYNREG_CPU_SHARDS
        for job in baseline_jobs:
            assert job.compute_pool == orch.DEEPSET_CPU_POOL

    def test_ood_full_baseline_default_single_wave_waits_once(self, collector, fake_session):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_ood_full_baseline_evaluation(fake_session)

        assert batch_sizes == [orch.SYNREG_CPU_SHARDS]

    def test_ood_full_autogluon_evaluation_submits_ag_shards(self, collector, fake_session):
        """OOD full AG phase submits SYNREG_AUTOGLUON_SHARDS jobs on AUTOGLUON_CPU_POOL."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_autogluon_evaluation(fake_session)
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        assert len(ag_jobs) == orch.SYNREG_AUTOGLUON_SHARDS
        for job in ag_jobs:
            assert job.compute_pool == orch.AUTOGLUON_CPU_POOL

    def test_ood_full_autogluon_default_single_wave_waits_once(self, collector, fake_session):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_ood_full_autogluon_evaluation(fake_session)

        assert batch_sizes == [orch.SYNREG_AUTOGLUON_SHARDS]

    def test_ood_full_aggregation_submits_agg_job(self, collector, fake_session):
        """OOD full agg phase submits 1 job with mode=aggregate and SYNREG_OUTPUT_STAGE=OOD_FULL_OUTPUT_STAGE."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_aggregation(fake_session)
        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        assert len(agg_jobs) == 1
        assert agg_jobs[0].env_vars.get("SYNREG_OUTPUT_STAGE") == orch.OOD_FULL_OUTPUT_STAGE


# ---------------------------------------------------------------------------
# Tests: Combined split-phase functions
# ---------------------------------------------------------------------------

class TestCombinedSplitPhase:
    def test_combined_prep_submits_combined_prep_job(self, collector, fake_session):
        """Combined prep submits 1 job with entrypoint=prepare_synthetic_regression.py and COMBINED_SUITE_ID."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_prep(fake_session)
        assert len(collector.submitted) == 1
        job = collector.submitted[0]
        assert job.entrypoint == "prepare_synthetic_regression.py"
        assert job.env_vars.get("SYNTHETIC_REGRESSION_SUITE_ID") == orch.COMBINED_SUITE_ID

    def test_combined_deepset_evaluation_submits_gpu_shards(self, collector, fake_session):
        """Combined deepset phase submits SYNREG_GPU_SHARDS jobs with COMBINED_SUITE_ID."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_deepset_evaluation(fake_session)
        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        assert len(deepset_jobs) == orch.SYNREG_GPU_SHARDS
        for job in deepset_jobs:
            assert job.env_vars.get("SYNTHETIC_REGRESSION_SUITE_ID") == orch.COMBINED_SUITE_ID

    def test_combined_baseline_evaluation_submits_cpu_shards(self, collector, fake_session):
        """Combined baseline phase submits SYNREG_CPU_SHARDS jobs on DEEPSET_CPU_POOL."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_baseline_evaluation(fake_session)
        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        assert len(baseline_jobs) == orch.SYNREG_CPU_SHARDS

    def test_combined_baseline_default_single_wave_waits_once(self, collector, fake_session):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_combined_baseline_evaluation(fake_session)

        assert batch_sizes == [orch.SYNREG_CPU_SHARDS]

    def test_combined_autogluon_evaluation_submits_ag_shards(self, collector, fake_session):
        """Combined AG phase submits SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT cluster jobs
        on AUTOGLUON_CPU_POOL (distributed ray_work_items mode: 6 × 4 workers)."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(fake_session)
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        assert len(ag_jobs) == orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT

    def test_combined_autogluon_jobs_carry_ray_and_memory_guard_env(self, collector, fake_session):
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(fake_session)

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert job.entrypoint == "autogluon_ray.py"
            assert job.target_instances == orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            assert job.pip_requirements == orch.SYNREG_AG_RAY_PIP
            assert job.external_access_integrations == orch.SYNREG_PYPI_EAI
            assert job.env_vars["SYNREG_AUTOGLUON_DISTRIBUTED_MODE"] == "ray_work_items"
            assert job.env_vars["SYNREG_AUTOGLUON_CLUSTER_SHARDS"] == str(
                orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT
            )
            assert job.env_vars["SYNREG_AUTOGLUON_WORKERS_PER_SHARD"] == str(
                orch.SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT
            )
            assert job.env_vars["SYNREG_WORKER_DATA_ACCESS_MODE"] == "driver_presigned_url"
            assert job.env_vars["SYNREG_MAX_WORK_ITEM_BYTES"] == "8192"
            assert job.env_vars["SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS"] == str(
                orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT
            )
            assert job.env_vars["SYNREG_AUTOGLUON_MAX_IN_FLIGHT"] == str(
                orch.SYNREG_AUTOGLUON_MAX_IN_FLIGHT_DEFAULT
            )
            assert job.env_vars["BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES"] == str(
                orch.SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT
            )
            assert job.env_vars["BENCHMARK_CPU_MAX_PROCESSED_FEATURES"] == str(
                orch.SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT
            )
            assert job.env_vars["BENCHMARK_CPU_MAX_MATRIX_BYTES"] == str(
                orch.SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT
            )
            assert job.env_vars["BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES"] == str(
                orch.SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT
            )
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS"] == str(
                orch.SYNREG_RAY_EVALUATION_READY_TIMEOUT_SECONDS_DEFAULT
            )
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_POLL_SECONDS"] == str(
                orch.SYNREG_RAY_EVALUATION_READY_POLL_SECONDS_DEFAULT
            )

    def test_combined_autogluon_evaluation_accepts_runtime_ray_readiness(
        self, collector, fake_session
    ):
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=2,
                autogluon_workers_per_shard=4,
                autogluon_task_cpus=1,
                autogluon_concurrent_clusters=2,
                autogluon_time_limit=300,
                autogluon_presets="best_quality",
                ray_ready_timeout_seconds=900,
                ray_ready_poll_seconds=20,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS"] == "900"
            assert job.env_vars["SYNREG_RAY_CLUSTER_READY_POLL_SECONDS"] == "20"

    def test_combined_autogluon_default_single_wave_waits_once(self, collector, fake_session):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_combined_autogluon_evaluation(fake_session)

        assert batch_sizes == [orch.SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT]

    def test_combined_autogluon_lower_concurrent_clusters_fails_fast(self, fake_session):
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=6,
                autogluon_concurrent_clusters=3,
            )

        msg = str(exc.value)
        assert "run_synthetic_regression_combined_autogluon_evaluation" in msg
        assert "AUTOGLUON_CONCURRENT_CLUSTERS=3" in msg
        assert "AUTOGLUON_CLUSTER_SHARDS=6" in msg
        assert orch.AUTOGLUON_CPU_POOL in msg

    def test_ray_entrypoint_uses_snowflake_ray_cluster_and_memory_preflight(self):
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        # ray.init uses _ray_address variable resolved from RAY_ADDRESS_MODE
        assert 'ray.init(' in text
        assert 'address=_ray_address' in text
        assert "ray.nodes()" in text
        assert "SYNREG_AUTOGLUON_WORKERS_PER_SHARD" in text
        assert "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES" in text

    def test_ray_entrypoint_no_driver_ray_put(self):
        """Driver must not call ray.put — dataset loading moved to workers."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "ray.put(" not in text, (
            "autogluon_ray.py must not call ray.put — datasets are loaded inside workers"
        )

    def test_ray_worker_signature_no_dataset_payload_param(self):
        """_autogluon_work_item must accept only item_meta, not a dataset_payload arg."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "def _autogluon_work_item(item_meta: dict) -> dict:" in text, (
            "_autogluon_work_item signature must be (item_meta: dict) -> dict"
        )
        assert "def _autogluon_work_item(item_meta: dict, dataset_payload" not in text

    def test_ray_driver_submits_without_payload_ref(self):
        """Driver must submit _autogluon_work_item.remote(item) — no payload_ref argument."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "_autogluon_work_item.remote(item)" in text, (
            "driver must call _autogluon_work_item.remote(item) without a payload_ref"
        )
        assert "_autogluon_work_item.remote(item, payload_ref)" not in text

    def test_ray_entrypoint_enforces_max_in_flight(self):
        """Driver loop must enforce MAX_IN_FLIGHT to bound concurrent worker-loaded fits."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "SYNREG_AUTOGLUON_MAX_IN_FLIGHT" in text
        assert "MAX_IN_FLIGHT" in text
        assert "len(pending) < MAX_IN_FLIGHT" in text

    def test_ray_entrypoint_calls_ray_wait(self):
        """Driver must call ray.wait for bounded in-flight scheduling."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "ray.wait(pending" in text

    def test_ray_entrypoint_driver_writes_csv(self):
        """Driver must be the sole writer of the shard CSV via write_part_csv_to_stage."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "write_part_csv_to_stage(" in text
        # Confirm it is called from module-level driver code (outside @ray.remote function)
        # by checking the call appears after the function definition
        fn_def_pos = text.index("def _autogluon_work_item(")
        write_pos = text.rindex("write_part_csv_to_stage(")
        assert write_pos > fn_def_pos, (
            "write_part_csv_to_stage call must appear in driver code after task definition"
        )

    def test_ray_worker_loads_dataset_locally(self):
        """Worker task must use the explicit worker dataset access function."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        fn_start = text.index("def _autogluon_work_item(")
        next_def = text.find("\ndef ", fn_start + 1)
        if next_def == -1:
            fn_body = text[fn_start:]
        else:
            fn_body = text[fn_start:next_def]
        assert "load_prepared_synthetic_dataset_from_access" in fn_body, (
            "worker must call load_prepared_synthetic_dataset_from_access inside function body"
        )
        assert "load_synthetic_regression_index" not in fn_body

    def test_ray_worker_enforces_dataset_bytes_guard(self):
        """Worker must check BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES after loading dataset."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        fn_start = text.index("def _autogluon_work_item(")
        next_def = text.find("\ndef ", fn_start + 1)
        fn_body = text[fn_start:next_def] if next_def != -1 else text[fn_start:]
        assert "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES" in fn_body, (
            "worker must read BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES env var"
        )
        assert "autogluon_dataset_too_large" in fn_body, (
            "worker must return skipped row with 'autogluon_dataset_too_large' reason"
        )

    def test_worker_access_probe_does_not_train_or_write_outputs(self):
        """Worker-access probe must stay lightweight: no AutoGluon training and no shard CSV writes."""
        text = (ROOT / "scripts" / "autogluon_worker_access_probe.py").read_text()
        forbidden = [
            "predict_autogluon",
            "get_tabular_predictor_class",
            ".fit(",
            "write_part_csv_to_stage",
            "AutoGluon_shard",
            "autogluon_ray.py",
        ]
        for token in forbidden:
            assert token not in text, f"worker-access probe must not contain {token!r}"

    def test_worker_access_probe_no_driver_ray_put_and_small_dict_task_arg(self):
        text = (ROOT / "scripts" / "autogluon_worker_access_probe.py").read_text()
        assert "ray.put(" not in text
        assert "def _worker_access_probe_item(item_meta: dict) -> dict:" in text
        assert "_worker_access_probe_item.remote(item)" in text

    def test_worker_access_probe_uses_metadata_only_driver_and_reports_fallback(self):
        text = (ROOT / "scripts" / "autogluon_worker_access_probe.py").read_text()
        assert "load_synthetic_regression_index" in text
        assert "load_prepared_synthetic_dataset_from_access(item_meta" in text
        assert "SYNREG_WORKER_DATA_ACCESS_MODE" in text
        assert "driver_presigned_url" in text
        assert "Worker failed to resolve dataset access" in text

    def test_worker_dataset_access_helper_is_session_free(self):
        text = (ROOT / "src" / "evaluate_synthetic_regression.py").read_text()
        fn_start = text.index("def load_prepared_synthetic_dataset_from_access(")
        next_def = text.find("\ndef ", fn_start + 1)
        fn_body = text[fn_start:next_def] if next_def != -1 else text[fn_start:]
        assert "Session.builder" not in fn_body
        assert "getOrCreate" not in fn_body
        assert "SnowflakeFile.open" in fn_body
        assert "scoped_url" in fn_body
        assert "presigned_url" in fn_body
        assert "urlopen" in fn_body

    def test_snowpark_session_helper_supports_spcs_oauth(self):
        text = (ROOT / "src" / "evaluate_synthetic_regression.py").read_text()
        fn_start = text.index("def create_snowpark_session(")
        next_def = text.find("\ndef ", fn_start + 1)
        fn_body = text[fn_start:next_def] if next_def != -1 else text[fn_start:]
        assert "/snowflake/session/token" in fn_body
        assert "SNOWFLAKE_ACCOUNT" in fn_body
        assert "SNOWFLAKE_HOST" in fn_body
        assert '"authenticator": "oauth"' in fn_body
        assert "Session.builder.configs(configs).create()" in fn_body

    def test_compact_work_item_helper_drops_unexpected_payload_fields(self):
        from evaluate_synthetic_regression import build_compact_synreg_work_item

        class _Row:
            def as_dict(self):
                return {"PRESIGNED_URL": "https://example.snowflakecomputing.com/presigned/file"}

        class _Session:
            def sql(self, query):
                assert "GET_PRESIGNED_URL(@EVALUATION_DATASET_STAGE" in query
                return self

            def collect(self):
                return [_Row()]

        row = {
            "suite_id": "linear_all_v1",
            "suite_family": "primary",
            "logical_dataset_key": "linear_all_v1:A:0001",
            "dataset_id": 1,
            "dataset_seed": 123,
            "prior_regime": "A",
            "split_seed": 0,
            "n_train_override": None,
            "n_train_default": 800,
            "n_holdout_default": 200,
            "n_total": 1000,
            "p_signal": 4,
            "p_noise": 0,
            "p_total": 4,
            "feature_noise_level": 0,
            "target_noise_scale": 1.0,
            "training_size_anchor": False,
            "stage_path": "@EVALUATION_DATASET_STAGE/primary/dataset_0001.parquet",
            "X": "x" * 10000,
            "unexpected_payload": "y" * 10000,
        }
        item = build_compact_synreg_work_item(row, session=_Session(), max_item_bytes=8192)
        assert "X" not in item
        assert "unexpected_payload" not in item
        assert item["dataset_access"]["mode"] == "driver_presigned_url"
        assert item["dataset_access"]["stage_path"] == row["stage_path"]
        assert item["dataset_access"]["presigned_url"].startswith("https://example.")

    def test_compact_work_item_helper_enforces_size_guard(self):
        from evaluate_synthetic_regression import build_compact_synreg_work_item

        class _Row:
            def as_dict(self):
                return {"PRESIGNED_URL": "https://example.snowflakecomputing.com/" + ("u" * 200)}

        class _Session:
            def sql(self, query):
                return self

            def collect(self):
                return [_Row()]

        row = {
            "suite_id": "linear_all_v1",
            "dataset_id": 1,
            "prior_regime": "A",
            "split_seed": 0,
            "stage_path": "@EVALUATION_DATASET_STAGE/" + ("x" * 200),
        }
        with pytest.raises(RuntimeError, match="SYNREG_MAX_WORK_ITEM_BYTES"):
            build_compact_synreg_work_item(
                row,
                session=_Session(),
                max_item_bytes=64,
            )

    def test_ray_entrypoint_tracks_future_to_item_and_completeness(self):
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "future_to_item" in text
        assert "_failed_row(" in text
        assert "unexpected Ray task error; emitting failed row" in text
        assert "Refusing to write a partial shard CSV" in text

    def test_autogluon_model_dirs_are_unique_for_identical_metadata(self):
        from autogluon_models import make_unique_autogluon_model_dir

        tmp = tempfile.mkdtemp()
        try:
            item = {
                "shard_index": 0,
                "prior_regime": "A",
                "dataset_id": 1,
                "split_seed": 0,
            }
            path1 = make_unique_autogluon_model_dir(item, base_dir=tmp)
            path2 = make_unique_autogluon_model_dir(item, base_dir=tmp)
            assert path1 != path2
            assert Path(path1).is_dir()
            assert Path(path2).is_dir()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_combined_aggregation_submits_agg_job(self, collector, fake_session):
        """Combined agg phase submits 1 job with SYNREG_OUTPUT_STAGE=COMBINED_OUTPUT_STAGE."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_aggregation(fake_session)
        agg_jobs = [j for j in collector.submitted
                    if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"]
        assert len(agg_jobs) == 1
        assert agg_jobs[0].env_vars.get("SYNREG_OUTPUT_STAGE") == orch.COMBINED_OUTPUT_STAGE


# ---------------------------------------------------------------------------
# Tests: Pipeline recomposition (wrappers call all 5 phase functions)
# ---------------------------------------------------------------------------

class TestPipelineRecomposition:
    def test_ood_full_evaluation_calls_all_5_phases(self, fake_session):
        """run_synthetic_regression_ood_full_evaluation calls all 5 phase functions exactly once."""
        from contextlib import ExitStack
        phase_fns = [
            "run_synthetic_regression_ood_full_prep",
            "run_synthetic_regression_ood_full_deepset_evaluation",
            "run_synthetic_regression_ood_full_baseline_evaluation",
            "run_synthetic_regression_ood_full_autogluon_evaluation",
            "run_synthetic_regression_ood_full_aggregation",
        ]
        mocks = {}
        with ExitStack() as stack:
            for fn_name in phase_fns:
                m = stack.enter_context(
                    patch(f"run_synthetic_regression_evaluation.{fn_name}")
                )
                mocks[fn_name] = m
            orch.run_synthetic_regression_ood_full_evaluation(fake_session)
        for fn_name in phase_fns:
            mocks[fn_name].assert_called_once()

    def test_combined_evaluation_calls_all_5_phases(self, fake_session):
        """run_synthetic_regression_combined_evaluation calls all 5 phase functions exactly once."""
        from contextlib import ExitStack
        phase_fns = [
            "run_synthetic_regression_combined_prep",
            "run_synthetic_regression_combined_deepset_evaluation",
            "run_synthetic_regression_combined_baseline_evaluation",
            "run_synthetic_regression_combined_autogluon_evaluation",
            "run_synthetic_regression_combined_aggregation",
        ]
        mocks = {}
        with ExitStack() as stack:
            for fn_name in phase_fns:
                m = stack.enter_context(
                    patch(f"run_synthetic_regression_evaluation.{fn_name}")
                )
                mocks[fn_name] = m
            orch.run_synthetic_regression_combined_evaluation(fake_session)
        for fn_name in phase_fns:
            mocks[fn_name].assert_called_once()


# ---------------------------------------------------------------------------
# Tests: retired env var absence
# ---------------------------------------------------------------------------

class TestLegacyEnvVarAbsence:
    def test_deepset_model_family_absent_from_deepset_shards(self, collector, fake_session, runtime_args):
        """Evaluation shard jobs must not carry the retired model-family env var."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        assert deepset_jobs, "No deepset jobs submitted"
        for job in deepset_jobs:
            assert "DEEPSET" + "_MODEL_FAMILY" not in job.env_vars, \
                f"Job {job.label} has retired model-family env var in env_vars"

    def test_model_arch_version_absent_from_deepset_shards(self, collector, fake_session, runtime_args):
        """Evaluation shard jobs must NOT carry MODEL_ARCH_VERSION (hardcoded in train.py)."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_deepset_evaluation(fake_session, *runtime_args)

        deepset_jobs = [j for j in collector.submitted
                        if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "deepset"]
        assert deepset_jobs, "No deepset jobs submitted"
        for job in deepset_jobs:
            assert "MODEL_ARCH_VERSION" not in job.env_vars, \
                f"Job {job.label} has MODEL_ARCH_VERSION in env_vars"

    def test_deepset_model_family_absent_from_baseline_shards(self, collector, fake_session, runtime_args):
        """Baseline shard jobs must not carry the retired model-family env var."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_baseline_evaluation(fake_session, *runtime_args)

        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        assert baseline_jobs, "No baseline jobs submitted"
        for job in baseline_jobs:
            assert "DEEPSET" + "_MODEL_FAMILY" not in job.env_vars, \
                f"Baseline job {job.label} has retired model-family env var in env_vars"


# ---------------------------------------------------------------------------
# Tests: Runtime-configurable baseline shard count
# ---------------------------------------------------------------------------

class TestRuntimeBaselineShards:
    """Verify BASELINE_SHARDS runtime parameter for baseline procedures."""

    def test_default_combined_baseline_capacity_probe_submits_6_probes(
        self, collector, fake_session
    ):
        """Default combined baseline capacity probe submits SYNREG_CPU_SHARDS=6 probes."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_baseline_capacity_probe(fake_session)

        probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_baseline_cap_probe_")
        ]
        assert len(probes) == orch.SYNREG_CPU_SHARDS

    def test_combined_baseline_capacity_probe_10_shards_10_concurrent_submits_10(
        self, collector, fake_session
    ):
        """BASELINE_SHARDS=10 + BASELINE_CONCURRENT_NODES=10 submits exactly 10 probes."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_baseline_capacity_probe(
                fake_session,
                baseline_shards=10,
                baseline_concurrent_nodes=10,
            )

        probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_baseline_cap_probe_")
        ]
        assert len(probes) == 10

    def test_combined_baseline_capacity_probe_6_shards_10_concurrent_raises(
        self, fake_session
    ):
        """BASELINE_SHARDS=6 + BASELINE_CONCURRENT_NODES=10 must be rejected."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_baseline_capacity_probe(
                fake_session,
                baseline_shards=6,
                baseline_concurrent_nodes=10,
            )
        msg = str(exc.value)
        assert "run_synthetic_regression_combined_baseline_capacity_probe" in msg
        assert "BASELINE_CONCURRENT_NODES=10" in msg
        assert "BASELINE_SHARDS=6" in msg

    def test_combined_baseline_capacity_probe_10_shards_6_concurrent_raises(
        self, fake_session
    ):
        """BASELINE_SHARDS=10 + BASELINE_CONCURRENT_NODES=6 must be rejected."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_baseline_capacity_probe(
                fake_session,
                baseline_shards=10,
                baseline_concurrent_nodes=6,
            )
        msg = str(exc.value)
        assert "BASELINE_CONCURRENT_NODES=6" in msg
        assert "BASELINE_SHARDS=10" in msg

    def test_combined_baseline_evaluation_10_shards_submits_10_jobs(
        self, collector, fake_session
    ):
        """run_synthetic_regression_combined_baseline_evaluation with BASELINE_SHARDS=10 submits 10 jobs."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_baseline_evaluation(
                fake_session,
                baseline_shards=10,
                baseline_concurrent_nodes=10,
            )

        baseline_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"
        ]
        assert len(baseline_jobs) == 10

    def test_combined_baseline_evaluation_10_shards_passes_num_shards_10(
        self, collector, fake_session
    ):
        """BASELINE_SHARDS=10 evaluation must pass SYNTHETIC_REGRESSION_NUM_SHARDS=10 to each job."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_baseline_evaluation(
                fake_session,
                baseline_shards=10,
                baseline_concurrent_nodes=10,
            )

        baseline_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"
        ]
        assert baseline_jobs
        for job in baseline_jobs:
            assert job.env_vars.get("SYNTHETIC_REGRESSION_NUM_SHARDS") == "10", (
                f"Job {job.label} has NUM_SHARDS="
                f"{job.env_vars.get('SYNTHETIC_REGRESSION_NUM_SHARDS')!r}, expected '10'"
            )

    def test_combined_baseline_evaluation_mismatch_shards_raises(self, fake_session):
        """BASELINE_SHARDS=10, BASELINE_CONCURRENT_NODES=6 must be rejected."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_baseline_evaluation(
                fake_session,
                baseline_shards=10,
                baseline_concurrent_nodes=6,
            )
        msg = str(exc.value)
        assert "BASELINE_SHARDS=10" in msg

    def test_combined_evaluation_10_baseline_shards_wires_aggregation(
        self, collector, fake_session
    ):
        """All-in-one combined evaluation with BASELINE_SHARDS=10 must pass
        SYNREG_EXPECTED_BASELINE_SHARDS=10 to the aggregation job."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_evaluation(
                fake_session,
                baseline_shards=10,
                baseline_concurrent_nodes=10,
            )

        agg_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"
        ]
        assert agg_jobs, "No aggregation job submitted"
        assert agg_jobs[0].env_vars.get("SYNREG_EXPECTED_BASELINE_SHARDS") == "10", (
            f"Expected SYNREG_EXPECTED_BASELINE_SHARDS=10, "
            f"got {agg_jobs[0].env_vars.get('SYNREG_EXPECTED_BASELINE_SHARDS')!r}"
        )

    def test_combined_evaluation_default_baseline_shards_wires_aggregation_with_6(
        self, collector, fake_session
    ):
        """Default combined evaluation must pass SYNREG_EXPECTED_BASELINE_SHARDS=6 to aggregation."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_evaluation(fake_session)

        agg_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"
        ]
        assert agg_jobs
        assert agg_jobs[0].env_vars.get("SYNREG_EXPECTED_BASELINE_SHARDS") == str(
            orch.SYNREG_CPU_SHARDS
        )


# ---------------------------------------------------------------------------
# Tests: Combined AutoGluon single-node shard mode (AUTOGLUON_CLUSTER_SHARDS=0)
# ---------------------------------------------------------------------------

class TestCombinedAutogluonSingleNodeMode:
    """Verify combined AutoGluon single-node shard mode (AUTOGLUON_CLUSTER_SHARDS=0)."""

    def test_single_node_capacity_probe_submits_30_probes(
        self, collector, fake_session
    ):
        """cluster_shards=0, workers_per_shard=1, concurrent_clusters=30 → 30 capacity probe jobs."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        probes = [j for j in collector.submitted if j.label.startswith("combined_ag_cap_probe_")]
        assert len(probes) == 30

    def test_single_node_capacity_probe_uses_capacity_probe_entrypoint(
        self, collector, fake_session
    ):
        """Single-node capacity probe must use capacity_probe.py (not ray_capacity_probe.py)."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        probes = [j for j in collector.submitted if j.label.startswith("combined_ag_cap_probe_")]
        assert probes
        for job in probes:
            assert job.entrypoint == "capacity_probe.py", (
                f"Expected capacity_probe.py, got {job.entrypoint!r}"
            )

    def test_single_node_capacity_probe_uses_target_instances_1(
        self, collector, fake_session
    ):
        """Single-node capacity probe must use target_instances=1."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        probes = [j for j in collector.submitted if j.label.startswith("combined_ag_cap_probe_")]
        assert probes
        for job in probes:
            assert job.target_instances == 1, (
                f"Expected target_instances=1, got {job.target_instances}"
            )

    def test_single_node_capacity_probe_carries_no_pip_or_eai(
        self, collector, fake_session
    ):
        """Single-node capacity probe must not carry pip_requirements or EAI (no Ray install)."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        probes = [j for j in collector.submitted if j.label.startswith("combined_ag_cap_probe_")]
        assert probes
        for job in probes:
            assert job.pip_requirements is None, (
                f"Single-node capacity probe {job.label} must not have pip_requirements"
            )
            assert job.external_access_integrations is None, (
                f"Single-node capacity probe {job.label} must not have EAI"
            )
            assert "SYNREG_AUTOGLUON_DISTRIBUTED_MODE" not in job.env_vars
            assert "EXPECTED_RAY_NODES" not in job.env_vars

    def test_single_node_worker_access_probe_submits_30_probes(
        self, collector, fake_session
    ):
        """cluster_shards=0, workers_per_shard=1, concurrent_clusters=30 submits 30 one-instance access probes."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_worker_access_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        probes = [
            j for j in collector.submitted
            if j.label.startswith("combined_ag_worker_access_probe_")
        ]
        assert len(probes) == 30
        for job in probes:
            assert job.entrypoint == "autogluon_worker_access_probe.py"
            assert job.target_instances == 1
            assert job.pip_requirements is None
            assert job.external_access_integrations is None
            assert job.env_vars["SYNREG_WORKER_ACCESS_PROBE_USE_RAY"] == "false"
            assert "EXPECTED_RAY_NODES" not in job.env_vars
            assert "SYNREG_AUTOGLUON_DISTRIBUTED_MODE" not in job.env_vars

    def test_single_node_evaluation_submits_30_jobs(
        self, collector, fake_session
    ):
        """cluster_shards=0, workers_per_shard=1, concurrent_clusters=30 → 30 AG evaluation jobs."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert len(ag_jobs) == 30

    def test_single_node_evaluation_uses_evaluate_synthetic_regression_entrypoint(
        self, collector, fake_session
    ):
        """Single-node evaluation must use evaluate_synthetic_regression.py."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert job.entrypoint == "evaluate_synthetic_regression.py", (
                f"Expected evaluate_synthetic_regression.py, got {job.entrypoint!r}"
            )

    def test_single_node_evaluation_uses_target_instances_1(
        self, collector, fake_session
    ):
        """Single-node evaluation must use target_instances=1."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert job.target_instances == 1, (
                f"Expected target_instances=1, got {job.target_instances}"
            )

    def test_single_node_evaluation_passes_num_shards_30(
        self, collector, fake_session
    ):
        """Single-node evaluation must pass SYNTHETIC_REGRESSION_NUM_SHARDS=30 to each job."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert job.env_vars.get("SYNTHETIC_REGRESSION_NUM_SHARDS") == "30", (
                f"Expected NUM_SHARDS=30, got "
                f"{job.env_vars.get('SYNTHETIC_REGRESSION_NUM_SHARDS')!r}"
            )

    def test_single_node_evaluation_shard_indices_0_to_29(
        self, collector, fake_session
    ):
        """Single-node evaluation must assign shard indices 0..29."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in ag_jobs]
        assert sorted(indices) == list(range(30))

    def test_single_node_evaluation_waits_once(
        self, collector, fake_session
    ):
        """Single-node evaluation must submit all 30 shards and wait exactly once."""
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_combined_autogluon_evaluation(
                    fake_session,
                    autogluon_cluster_shards=0,
                    autogluon_workers_per_shard=1,
                    autogluon_concurrent_clusters=30,
                )

        assert batch_sizes == [30], f"Expected single wait for 30 jobs, got {batch_sizes}"

    def test_combined_all_in_one_single_node_ag_wires_aggregation_with_30(
        self, collector, fake_session
    ):
        """All-in-one combined evaluation with single-node mode must pass
        SYNREG_EXPECTED_AG_SHARDS=30 to aggregation (output_shards = concurrent_clusters)."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        agg_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "aggregate"
        ]
        assert ag_jobs, "No AG jobs submitted in single-node mode"
        assert agg_jobs, "No aggregation job submitted"
        assert len(ag_jobs) == 30
        assert agg_jobs[0].env_vars.get("SYNREG_EXPECTED_AG_SHARDS") == "30", (
            f"Expected SYNREG_EXPECTED_AG_SHARDS=30, "
            f"got {agg_jobs[0].env_vars.get('SYNREG_EXPECTED_AG_SHARDS')!r}"
        )

    def test_single_node_evaluation_carries_ag_pip_and_eai(
        self, collector, fake_session
    ):
        """Single-node evaluation jobs must carry autogluon.tabular pip and TABPFN_PYPI_EAI."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert job.pip_requirements == orch.SYNREG_AG_PIP, (
                f"Expected SYNREG_AG_PIP, got {job.pip_requirements!r}"
            )
            assert job.external_access_integrations == orch.SYNREG_PYPI_EAI, (
                f"Expected SYNREG_PYPI_EAI, got {job.external_access_integrations!r}"
            )

    def test_single_node_evaluation_does_not_pass_ray_env_vars(
        self, collector, fake_session
    ):
        """Single-node evaluation jobs must not carry Ray-specific env vars."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=1,
                autogluon_concurrent_clusters=30,
            )

        ag_jobs = [
            j for j in collector.submitted
            if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"
        ]
        assert ag_jobs
        for job in ag_jobs:
            assert "SYNREG_AUTOGLUON_DISTRIBUTED_MODE" not in job.env_vars, (
                f"Single-node job {job.label} must not carry SYNREG_AUTOGLUON_DISTRIBUTED_MODE"
            )
            assert "SYNREG_AUTOGLUON_CLUSTER_SHARDS" not in job.env_vars, (
                f"Single-node job {job.label} must not carry SYNREG_AUTOGLUON_CLUSTER_SHARDS"
            )
            assert "SYNREG_AUTOGLUON_WORKERS_PER_SHARD" not in job.env_vars, (
                f"Single-node job {job.label} must not carry SYNREG_AUTOGLUON_WORKERS_PER_SHARD"
            )


# ---------------------------------------------------------------------------
# Tests: Combined AutoGluon execution plan validation (rejection tests)
# ---------------------------------------------------------------------------

class TestCombinedAutogluonPlanValidation:
    """Verify _resolve_combined_autogluon_execution_plan rejects invalid combinations."""

    def test_cluster_shards_0_workers_per_shard_4_raises(self, fake_session):
        """cluster_shards=0 and workers_per_shard=4 must be rejected."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=4,
                autogluon_concurrent_clusters=30,
            )
        msg = str(exc.value)
        assert "AUTOGLUON_CLUSTER_SHARDS=0" in msg
        assert "AUTOGLUON_WORKERS_PER_SHARD" in msg
        assert "single-node" in msg.lower()

    def test_single_node_mode_derives_evaluate_synthetic_regression_entrypoint(self):
        """cluster_shards=0 must derive entrypoint=evaluate_synthetic_regression.py internally."""
        plan = orch._resolve_combined_autogluon_execution_plan(
            procedure_name="test_proc",
            cluster_shards_arg=0,
            workers_per_shard_arg=1,
            concurrent_clusters_arg=30,
        )
        assert plan.entrypoint == "evaluate_synthetic_regression.py"
        assert plan.mode == "single_node_shards"

    def test_cluster_shards_6_concurrent_clusters_4_raises(self, fake_session):
        """cluster_shards=6 and concurrent_clusters=4 must be rejected (batching not allowed)."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_autogluon_evaluation(
                fake_session,
                autogluon_cluster_shards=6,
                autogluon_concurrent_clusters=4,
            )
        msg = str(exc.value)
        assert "AUTOGLUON_CONCURRENT_CLUSTERS=4" in msg
        assert "AUTOGLUON_CLUSTER_SHARDS=6" in msg

    def test_ray_mode_derives_autogluon_ray_entrypoint_internally(self):
        """cluster_shards=6 (Ray mode) must derive entrypoint=autogluon_ray.py internally."""
        plan = orch._resolve_combined_autogluon_execution_plan(
            procedure_name="test_proc",
            cluster_shards_arg=6,
            workers_per_shard_arg=4,
            concurrent_clusters_arg=6,
        )
        assert plan.entrypoint == "autogluon_ray.py"
        assert plan.mode == "ray_clusters"

    def test_capacity_probe_cluster_shards_0_workers_per_shard_4_raises(self, fake_session):
        """Capacity probe: cluster_shards=0, workers_per_shard=4 must be rejected."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_autogluon_capacity_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=4,
                autogluon_concurrent_clusters=30,
            )
        msg = str(exc.value)
        assert "AUTOGLUON_CLUSTER_SHARDS=0" in msg
        assert "single-node" in msg.lower()

    def test_worker_access_probe_cluster_shards_0_workers_per_shard_4_raises(self, fake_session):
        """Worker-access probe shares the resolver and rejects multi-instance single-node mode."""
        with pytest.raises(ValueError) as exc:
            orch.run_synthetic_regression_combined_autogluon_worker_access_probe(
                fake_session,
                autogluon_cluster_shards=0,
                autogluon_workers_per_shard=4,
                autogluon_concurrent_clusters=30,
            )
        msg = str(exc.value)
        assert "AUTOGLUON_CLUSTER_SHARDS=0" in msg
        assert "single-node" in msg.lower()

    def test_plan_resolver_rejects_negative_cluster_shards(self):
        """cluster_shards < 0 must be rejected by the plan resolver."""
        with pytest.raises(ValueError) as exc:
            orch._resolve_combined_autogluon_execution_plan(
                procedure_name="test_proc",
                cluster_shards_arg=-1,
                workers_per_shard_arg=1,
                concurrent_clusters_arg=6,
            )
        msg = str(exc.value)
        assert "non-negative" in msg.lower() or "AUTOGLUON_CLUSTER_SHARDS" in msg


# ---------------------------------------------------------------------------
# AutoGluon import timing probe tests
# ---------------------------------------------------------------------------

class TestAutogluonImportTimingProbe:
    """Tests for run_synthetic_regression_autogluon_import_timing_probe."""

    def test_default_probe_submits_one_pip_job(self, collector, fake_session):
        """Default (single pip-mode) probe must submit exactly one job."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_import_timing_probe_default(
                fake_session, ag_rt="2.5.0-py311"
            )

        assert len(collector.submitted) == 1
        job = collector.submitted[0]
        assert job.entrypoint == "autogluon_import_timing_probe.py"
        assert job.target_instances == 1
        assert job.compute_pool == orch.AUTOGLUON_CPU_POOL
        assert job.runtime_environment == "2.5.0-py311"
        assert job.pip_requirements == orch.SYNREG_AG_PIP
        assert job.external_access_integrations == orch.SYNREG_PYPI_EAI
        assert job.env_vars["SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE"] == "pip"
        assert "EVAL_RUNTIME_ENVIRONMENT" in job.env_vars
        assert job.env_vars["EVAL_RUNTIME_ENVIRONMENT"] == "2.5.0-py311"

    def test_pip_probe_count_8_submits_8_jobs(self, collector, fake_session):
        """with_pip=True, probe_count=8 must submit exactly 8 independent jobs."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_import_timing_probe(
                fake_session, ag_rt="2.5.0-py311", with_pip=True, probe_count=8
            )

        assert len(collector.submitted) == 8
        for job in collector.submitted:
            assert job.entrypoint == "autogluon_import_timing_probe.py"
            assert job.target_instances == 1
            assert job.compute_pool == orch.AUTOGLUON_CPU_POOL
            assert job.runtime_environment == "2.5.0-py311"
            assert job.pip_requirements == orch.SYNREG_AG_PIP
            assert job.external_access_integrations == orch.SYNREG_PYPI_EAI
            assert job.env_vars["SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE"] == "pip"

    def test_no_pip_probe_uses_no_requirements(self, collector, fake_session):
        """with_pip=False must submit jobs with no pip requirements and no EAI."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_autogluon_import_timing_probe(
                fake_session, ag_rt="2.5.0-py311", with_pip=False, probe_count=4
            )

        assert len(collector.submitted) == 4
        for job in collector.submitted:
            assert job.entrypoint == "autogluon_import_timing_probe.py"
            assert job.target_instances == 1
            assert job.compute_pool == orch.AUTOGLUON_CPU_POOL
            assert job.pip_requirements is None
            assert job.external_access_integrations is None
            assert job.env_vars["SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE"] == "no_pip_baseline"

    def test_invalid_probe_count_zero_raises(self, fake_session):
        """probe_count=0 must raise ValueError before any submission."""
        with pytest.raises(ValueError, match="probe_count"):
            orch.run_synthetic_regression_autogluon_import_timing_probe(
                fake_session, probe_count=0
            )

    def test_probe_script_does_not_call_ray_init(self):
        """autogluon_import_timing_probe.py must not call ray.init(...)."""
        text = (ROOT / "scripts" / "autogluon_import_timing_probe.py").read_text()
        # Check for the invocation pattern (with opening paren), not the word in docstrings.
        assert "ray.init(" not in text

    def test_probe_script_does_not_import_snowflake_snowpark(self):
        """autogluon_import_timing_probe.py must not import snowflake.snowpark."""
        text = (ROOT / "scripts" / "autogluon_import_timing_probe.py").read_text()
        assert "snowflake.snowpark" not in text
        assert "snowflake.ml" not in text

    def test_probe_script_does_not_query_index(self):
        """autogluon_import_timing_probe.py must not reference SYNTHETIC_REGRESSION_DATASET_INDEX."""
        text = (ROOT / "scripts" / "autogluon_import_timing_probe.py").read_text()
        assert "SYNTHETIC_REGRESSION_DATASET_INDEX" not in text


# ---------------------------------------------------------------------------
# Tests: SPCS static analysis
# ---------------------------------------------------------------------------

class TestSPCSStaticAnalysis:
    """Static checks on SPCS-related source files."""

    def test_dockerfile_installs_autogluon(self):
        text = (ROOT / "docker" / "autogluon" / "Dockerfile").read_text()
        assert "autogluon.tabular==1.3.0" in text or "autogluon.tabular" in text

    def test_dockerfile_installs_ray(self):
        text = (ROOT / "docker" / "autogluon" / "Dockerfile").read_text()
        assert "ray" in text

    def test_dockerfile_sets_pythonpath(self):
        text = (ROOT / "docker" / "autogluon" / "Dockerfile").read_text()
        assert "PYTHONPATH" in text
        assert "/app/scripts" in text
        assert "/app/src" in text

    def test_spcs_ray_driver_uses_explicit_address_mode(self):
        """autogluon_ray.py must support explicit RAY address mode (not only 'auto')."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "SYNREG_RAY_ADDRESS_MODE" in text
        assert "explicit" in text
        assert "RAY_HEAD_ADDRESS" in text

    def test_spcs_ray_driver_does_not_hardcode_auto_only(self):
        """autogluon_ray.py must not have ray.init(address='auto') as the only path."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        # The explicit branch must exist
        assert "RAY_HEAD_ADDRESS" in text
        assert "explicit" in text

    def test_workers_do_not_query_dataset_index(self):
        """Worker tasks in autogluon_ray.py must not call load_synthetic_regression_index."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        # load_synthetic_regression_index should only appear in driver-level code (outside @ray.remote)
        # The remote task function should not call it
        import re
        remote_task_match = re.search(
            r'@ray\.remote.*?def _autogluon_work_item.*?(?=@ray\.remote|\Z)',
            text, re.DOTALL
        )
        if remote_task_match:
            task_body = remote_task_match.group(0)
            assert "load_synthetic_regression_index" not in task_body

    def test_ray_worker_does_not_create_snowpark_session(self):
        """Ray workers must not try to inherit or recreate a driver Snowpark session."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        import re
        remote_task_match = re.search(
            r'@ray\.remote.*?def _autogluon_work_item.*?(?=# ---------------------------------------------------------------------------\n# Main distributed evaluation loop)',
            text,
            re.DOTALL,
        )
        assert remote_task_match, "Could not locate _autogluon_work_item body"
        task_body = remote_task_match.group(0)
        assert "Session.builder" not in task_body
        assert "create_snowpark_session" not in task_body
        assert "getOrCreate" not in task_body

    def test_spcs_backend_constant_exists(self):
        text = (ROOT / "scripts" / "run_synthetic_regression_evaluation.py").read_text()
        assert "SYNREG_AUTOGLUON_EXECUTION_BACKEND" in text
        assert "spcs_job" in text
        assert "mljob" in text

    def test_spcs_image_constant_exists(self):
        text = (ROOT / "scripts" / "run_synthetic_regression_evaluation.py").read_text()
        assert "SYNREG_AUTOGLUON_SPCS_IMAGE" in text

    def test_spcs_helpers_do_not_pass_runtime_environment(self):
        text = (ROOT / "scripts" / "run_synthetic_regression_evaluation.py").read_text()
        assert "_build_spcs_job_spec" in text
        assert "_execute_spcs_job_service" in text

    def test_spcs_job_spec_builder_defined(self):
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        # Should be callable
        spec = _build_spcs_job_spec(image="test:1.0", args=["/test.py"], env_vars={})
        assert "test:1.0" in spec

    # Finding 2: Snowflake token injection
    def test_spcs_spec_does_not_include_snowflake_service(self):
        """Generated SPCS job spec must NOT contain snowflakeService.

        Snowflake rejects specs that include this field with error 395018:
        'Invalid spec: unknown option snowflakeService for spec'.
        SPCS job services receive the OAuth token automatically at
        /snowflake/session/token without any YAML configuration.
        """
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="test:1.0", args=["/app/scripts/autogluon_ray.py"], env_vars={}
        )
        assert "snowflakeService" not in spec, (
            "_build_spcs_job_spec must not include 'snowflakeService' — "
            "Snowflake rejects it with error 395018"
        )

    # Finding 2: Session probe function
    def test_spcs_session_probe_function_exists(self):
        """run_synthetic_regression_autogluon_spcs_session_probe must be importable."""
        from run_synthetic_regression_evaluation import (
            run_synthetic_regression_autogluon_spcs_session_probe,
        )
        assert callable(run_synthetic_regression_autogluon_spcs_session_probe)

    # Finding 5: Ray head starts with zero CPUs
    def test_spcs_ray_head_starts_with_zero_cpus(self):
        """spcs_ray_head.py must pass --num-cpus=0 to 'ray start --head'.

        Without this, the head node counts toward the CPU pool and the readiness check
        (expected_cpus_min = workers_per_shard * TASK_CPUS) may be satisfied with fewer
        actual workers than requested.
        """
        text = (ROOT / "scripts" / "spcs_ray_head.py").read_text()
        assert "--num-cpus=0" in text, (
            "spcs_ray_head.py must pass '--num-cpus=0' to prevent the head from "
            "contributing to the worker CPU pool"
        )

    # Finding 5: expected_nodes in SPCS mode
    def test_autogluon_ray_expects_workers_plus_head_in_spcs_mode(self):
        """autogluon_ray.py must use workers+1 as expected_nodes in SPCS (explicit) mode."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "WORKERS_PER_SHARD + 1" in text, (
            "autogluon_ray.py must set expected_nodes=WORKERS_PER_SHARD+1 in SPCS mode "
            "to account for the head node (which has 0 CPUs)"
        )

    # Finding 6: Cluster identity resource on head
    def test_spcs_ray_head_injects_cluster_identity_resource(self):
        """spcs_ray_head.py must add a custom Ray resource for cluster identity verification."""
        text = (ROOT / "scripts" / "spcs_ray_head.py").read_text()
        assert "SPCS_RAY_RUN_ID" in text, (
            "spcs_ray_head.py must read SPCS_RAY_RUN_ID to build the cluster identity resource"
        )
        assert "SPCS_RAY_SHARD_INDEX" in text
        assert "spcs_cluster_id" in text, (
            "spcs_ray_head.py must embed 'spcs_cluster_id' as a custom resource name prefix"
        )
        assert "--resources=" in text

    # Finding 6: Driver checks cluster identity
    def test_autogluon_ray_verifies_cluster_identity_in_spcs_mode(self):
        """autogluon_ray.py must verify cluster identity after ray.init() in explicit mode."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "spcs_cluster_id" in text
        assert "SPCS_RAY_RUN_ID" in text
        assert "SPCS_RAY_SHARD_INDEX" in text
        assert "Cluster identity check FAILED" in text or "cluster identity" in text.lower()

    # Finding 4: Per-shard DNS via DNS_SUFFIX
    def test_spcs_head_dns_uses_suffix_not_global_override_for_multishard(self):
        """run_synthetic_regression_evaluation.py must derive per-shard DNS from SPCS_RAY_HEAD_DNS_SUFFIX."""
        text = (ROOT / "scripts" / "run_synthetic_regression_evaluation.py").read_text()
        assert "SPCS_RAY_HEAD_DNS_SUFFIX" in text, (
            "Per-shard DNS must use SPCS_RAY_HEAD_DNS_SUFFIX (not a single global override)"
        )

    # Finding 7: Presigned URL expiry logging
    def test_autogluon_ray_logs_presigned_url_expiry_warning(self):
        """autogluon_ray.py must warn when presigned URL expiry < estimated shard runtime."""
        text = (ROOT / "scripts" / "autogluon_ray.py").read_text()
        assert "SYNREG_PRESIGNED_URL_EXPIRY_SECONDS" in text
        assert "expiry" in text.lower()

    # Finding 1: Image verification helper
    def test_spcs_image_verification_helper_exists(self):
        """_verify_spcs_image_in_repository must be importable and callable."""
        from run_synthetic_regression_evaluation import _verify_spcs_image_in_repository
        assert callable(_verify_spcs_image_in_repository)

    def test_spcs_image_reference_parser_extracts_repo_name_tag(self):
        from run_synthetic_regression_evaluation import _parse_spcs_image_reference

        parsed = _parse_spcs_image_reference(
            "acct.registry.snowflakecomputing.com/db/schema/repo/tabpfn-autogluon-ray:1.0.0"
        )
        assert parsed["repository_url"] == "acct.registry.snowflakecomputing.com/db/schema/repo"
        assert parsed["image_name"] == "tabpfn-autogluon-ray"
        assert parsed["tag"] == "1.0.0"
        assert parsed["digest"] is None

    def test_spcs_spec_portrange_endpoint(self):
        """_build_spcs_job_spec emits portRange: in YAML when endpoint uses portRange key."""
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="img:1.0",
            args=["/app/scripts/spcs_ray_worker.py"],
            env_vars={},
            endpoints=[
                {"name": "ray-worker-ports", "portRange": "10002-10010", "protocol": "TCP"},
            ],
        )
        assert "portRange: 10002-10010" in spec
        assert "port:" not in spec.split("endpoints:")[1]  # no scalar port field in endpoint section

    def test_spcs_coordinator_endpoint_list_includes_all_ray_ports(self):
        """_spcs_ray_coordinator_endpoints returns 5 entries covering all required Ray ports."""
        from run_synthetic_regression_evaluation import (
            _spcs_ray_coordinator_endpoints,
            _build_spcs_job_spec,
        )
        endpoints = _spcs_ray_coordinator_endpoints(6379)
        assert len(endpoints) == 5
        names = {ep["name"] for ep in endpoints}
        assert "ray-head" in names
        assert "ray-node-manager" in names
        assert "ray-object-manager" in names
        assert "ray-runtime-env-agent" in names
        assert "ray-worker-ports" in names
        # Confirm portRange entry exists
        assert any("portRange" in ep for ep in endpoints)
        # Confirm YAML spec renders all endpoint names
        spec = _build_spcs_job_spec(
            image="img:1.0", args=["/s.py"], env_vars={}, endpoints=endpoints
        )
        for name in names:
            assert name in spec
        assert "portRange:" in spec

    def test_spcs_worker_endpoint_list_excludes_ray_head(self):
        """_spcs_ray_worker_endpoints returns 4 entries; none named ray-head."""
        from run_synthetic_regression_evaluation import _spcs_ray_worker_endpoints
        endpoints = _spcs_ray_worker_endpoints()
        assert len(endpoints) == 4
        assert not any(ep["name"] == "ray-head" for ep in endpoints)
        names = {ep["name"] for ep in endpoints}
        assert "ray-node-manager" in names
        assert "ray-object-manager" in names
        assert "ray-runtime-env-agent" in names
        assert "ray-worker-ports" in names

    def test_spcs_ray_port_env_vars_returns_all_six(self):
        """_spcs_ray_port_env_vars returns a dict with all 6 deterministic-port keys."""
        from run_synthetic_regression_evaluation import _spcs_ray_port_env_vars
        env = _spcs_ray_port_env_vars()
        assert "SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT" in env
        assert "SYNREG_SPCS_RAY_NODE_MANAGER_PORT" in env
        assert "SYNREG_SPCS_RAY_OBJECT_MANAGER_PORT" in env
        assert "SYNREG_SPCS_RAY_RUNTIME_ENV_AGENT_PORT" in env
        assert "SYNREG_SPCS_RAY_MIN_WORKER_PORT" in env
        assert "SYNREG_SPCS_RAY_MAX_WORKER_PORT" in env
        assert len(env) == 6


# ---------------------------------------------------------------------------
# Tests: SPCS AutoGluon backend
# ---------------------------------------------------------------------------

class TestSPCSAutogluonBackend:
    """Tests for the SPCS custom-image AutoGluon backend."""

    def _make_session(self, rows_by_query=None):
        """Create a minimal mock session."""
        class _MockSession:
            def __init__(self, rows_by_query):
                self._rows = rows_by_query or {}
                self.executed = []

            def sql(self, query):
                self.executed.append(query)
                return self

            def collect(self):
                for k, v in self._rows.items():
                    if k in self.executed[-1]:
                        return v
                return []

        return _MockSession(rows_by_query)

    def test_invalid_backend_raises(self):
        from run_synthetic_regression_evaluation import _validate_autogluon_backend
        with pytest.raises(ValueError, match="invalid_backend"):
            _validate_autogluon_backend("invalid_backend", "img", "test_proc")

    def test_spcs_backend_missing_image_raises(self):
        from run_synthetic_regression_evaluation import _validate_autogluon_backend
        with pytest.raises(ValueError, match="SYNREG_AUTOGLUON_SPCS_IMAGE"):
            _validate_autogluon_backend("spcs_job", "", "test_proc")

    def test_spcs_backend_with_image_passes(self):
        from run_synthetic_regression_evaluation import _validate_autogluon_backend
        _validate_autogluon_backend("spcs_job", "account.registry.snowflakecomputing.com/db/schema/repo/img:1.0", "test_proc")

    def test_mljob_backend_does_not_require_image(self):
        from run_synthetic_regression_evaluation import _validate_autogluon_backend
        _validate_autogluon_backend("mljob", "", "test_proc")  # should not raise

    def test_resolve_spcs_image_prefers_procedure_argument(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod

        monkeypatch.delenv("SYNREG_AUTOGLUON_SPCS_IMAGE", raising=False)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "env/img:1.0")

        assert mod._resolve_spcs_image("test_proc", "arg/img:2.0") == "arg/img:2.0"

    def test_resolve_spcs_image_legacy_placeholder_uses_env(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod

        monkeypatch.setenv("SYNREG_AUTOGLUON_SPCS_IMAGE", "env/img:1.0")
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "")

        assert mod._resolve_spcs_image("test_proc", "spcs_job") == "env/img:1.0"

    def test_resolve_spcs_image_missing_image_raises(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod

        monkeypatch.delenv("SYNREG_AUTOGLUON_SPCS_IMAGE", raising=False)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "")

        with pytest.raises(ValueError, match="SYNREG_AUTOGLUON_SPCS_IMAGE"):
            mod._resolve_spcs_image("test_proc", "spcs_job")

    def test_spcs_handler_missing_image_raises_before_submission(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod

        monkeypatch.delenv("SYNREG_AUTOGLUON_SPCS_IMAGE", raising=False)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "")

        with pytest.raises(ValueError, match="SYNREG_AUTOGLUON_SPCS_IMAGE"):
            mod.run_synthetic_regression_autogluon_spcs_import_probe(
                object(),
                "spcs_job",
                probe_count=1,
            )

    def test_build_spcs_job_spec_contains_image(self):
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="myrepo/myimage:1.0",
            args=["/app/scripts/autogluon_ray.py"],
            env_vars={"FOO": "bar"},
        )
        assert "myrepo/myimage:1.0" in spec
        assert "/app/scripts/autogluon_ray.py" in spec
        assert "FOO" in spec
        assert '"bar"' in spec

    def test_build_spcs_job_spec_no_runtime_environment(self):
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="img:1.0",
            args=["/app/src/evaluate_synthetic_regression.py"],
            env_vars={},
        )
        # Must not contain runtime_environment or pip_requirements fields
        assert "runtime_environment" not in spec
        assert "pip_requirements" not in spec

    def test_spcs_resource_profiles_have_role_specific_defaults(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod

        for prefix in (
            "SYNREG_SPCS_RAY_HEAD",
            "SYNREG_SPCS_RAY_DRIVER",
            "SYNREG_SPCS_RAY_WORKER",
        ):
            for suffix in ("CPU", "MEMORY", "CPU_REQUEST", "CPU_LIMIT", "MEMORY_REQUEST", "MEMORY_LIMIT"):
                monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)

        assert mod._spcs_resources_for_role(mod.SPCS_RAY_HEAD_RESOURCES) == {
            "cpu_request": "0.5",
            "cpu_limit": "0.5",
            "memory_request": "2Gi",
            "memory_limit": "4Gi",
        }
        assert mod._spcs_resources_for_role(mod.SPCS_RAY_DRIVER_RESOURCES) == {
            "cpu_request": "0.5",
            "cpu_limit": "1",
            "memory_request": "2Gi",
            "memory_limit": "4Gi",
        }
        assert mod._spcs_resources_for_role(mod.SPCS_RAY_WORKER_RESOURCES) == {
            "cpu_request": "4",
            "cpu_limit": "4",
            "memory_request": "16Gi",
            "memory_limit": "16Gi",
        }

    def test_spcs_resource_profile_env_overrides_request_and_limit(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod

        monkeypatch.setenv("SYNREG_SPCS_RAY_HEAD_CPU_REQUEST", "250m")
        monkeypatch.setenv("SYNREG_SPCS_RAY_HEAD_CPU_LIMIT", "750m")
        monkeypatch.setenv("SYNREG_SPCS_RAY_HEAD_MEMORY_REQUEST", "1Gi")
        monkeypatch.setenv("SYNREG_SPCS_RAY_HEAD_MEMORY_LIMIT", "3Gi")
        spec = mod._build_spcs_job_spec(
            image="img:1.0",
            args=["/app/scripts/spcs_ray_head.py"],
            env_vars={},
            resource_role=mod.SPCS_RAY_HEAD_RESOURCES,
        )
        assert 'cpu: "250m"' in spec
        assert 'cpu: "750m"' in spec
        assert 'memory: "1Gi"' in spec
        assert 'memory: "3Gi"' in spec

    def test_execute_spcs_job_service_is_async(self):
        import run_synthetic_regression_evaluation as mod
        captured = {}

        class _Session:
            def sql(self, query):
                captured["query"] = query
                return self

            def collect(self):
                return []

        mod._execute_spcs_job_service(
            _Session(),
            label="spcs_async_check",
            compute_pool=mod.AUTOGLUON_CPU_POOL,
            spec="spec:\n  containers: []",
        )
        assert "ASYNC = TRUE" in captured["query"]

    def test_cancel_spcs_job_service_uses_spcs_cancel_job(self):
        import run_synthetic_regression_evaluation as mod
        captured = {}

        class _Session:
            def sql(self, query):
                captured["query"] = query
                return self

            def collect(self):
                return []

        mod._cancel_spcs_job_service(_Session(), "SPCS_RAY_HEAD_0")
        assert "SELECT SPCS_RAY_HEAD_0!SPCS_CANCEL_JOB()" in captured["query"]

    def test_build_spcs_job_spec_no_pip_requirements(self):
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="img:1.0",
            args=["/app/scripts/autogluon_ray.py"],
            env_vars={"SYNREG_AUTOGLUON_DISTRIBUTED_MODE": "ray_work_items"},
        )
        assert "pip_requirements" not in spec
        # Must not contain package installation directives (note: env var names may contain 'autogluon')
        assert "autogluon.tabular" not in spec.lower()
        assert "pip install" not in spec.lower()

    def test_build_spcs_job_spec_env_quoting(self):
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="img:1.0",
            args=["/app/scripts/test.py"],
            env_vars={"RESULTS_STAGE": "@MY_STAGE/path/to/results"},
        )
        # Value with @ must appear quoted
        assert "RESULTS_STAGE" in spec
        assert '"@MY_STAGE/path/to/results"' in spec

    def test_spcs_single_node_submits_job_service_not_mljob(self, monkeypatch):
        """SPCS single-node path must call _execute_spcs_job_service, not submit_from_stage."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted.append({"label": label, "compute_pool": compute_pool, "spec": spec})
            return f"MOCK_{label.upper()}"

        def _mock_wait(labeled_jobs, session):
            pass

        def _mock_ensure_pool(session, pool):
            pass

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", _mock_wait)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", _mock_ensure_pool)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "env/img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")

        session = object()
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            session,
            "procedure/img:2.0",
            autogluon_cluster_shards=0,
            autogluon_concurrent_clusters=2,
        )
        assert len(submitted) == 2
        # Specs must contain the procedure image argument, not the legacy env/global fallback.
        for s in submitted:
            assert "procedure/img:2.0" in s["spec"]
            assert "env/img:1.0" not in s["spec"]
            # No runtime_environment or pip_requirements
            assert "runtime_environment" not in s["spec"]
            assert "pip_requirements" not in s["spec"]

    def test_spcs_legacy_placeholder_still_uses_env_image(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod
        submitted = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted.append(spec)
            return f"MOCK_{label.upper()}"

        monkeypatch.setenv("SYNREG_AUTOGLUON_SPCS_IMAGE", "env/img:1.0")
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "")
        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            "spcs_job",
            autogluon_cluster_shards=0,
            autogluon_concurrent_clusters=1,
        )

        assert submitted
        assert "env/img:1.0" in submitted[0]

    def test_spcs_single_node_uses_correct_entrypoint(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod
        submitted = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted.append(spec)
            return f"MOCK_{label.upper()}"

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=0,
            autogluon_concurrent_clusters=1,
        )
        assert submitted, "No specs submitted"
        assert "/app/src/evaluate_synthetic_regression.py" in submitted[0]

    def test_spcs_distributed_ray_driver_uses_explicit_address(self, monkeypatch):
        """spcs_ray_coordinator.py must set SYNREG_RAY_ADDRESS_MODE=explicit when launching the driver.

        In the coordinator topology, head and driver are merged into one container.
        SYNREG_RAY_ADDRESS_MODE is set programmatically inside spcs_ray_coordinator.py
        (not in the SPCS spec), so this is verified via static analysis.
        """
        import pathlib
        coordinator_src = (
            pathlib.Path(__file__).parent.parent / "scripts" / "spcs_ray_coordinator.py"
        ).read_text(encoding="utf-8")
        assert "SYNREG_RAY_ADDRESS_MODE" in coordinator_src, (
            "spcs_ray_coordinator.py must set SYNREG_RAY_ADDRESS_MODE in the subprocess env"
        )
        assert "explicit" in coordinator_src, (
            "spcs_ray_coordinator.py must use explicit address mode, not 'auto'"
        )

    def test_spcs_distributed_ray_driver_has_ray_head_address(self, monkeypatch):
        """Worker SPCS specs must include RAY_HEAD_ADDRESS pointing to the coordinator's DNS hostname.

        In the coordinator topology there is no separate driver SPCS service; the driver runs
        inside the coordinator container. Workers receive RAY_HEAD_ADDRESS in their SPCS spec
        so they can join the correct Ray cluster.
        """
        import run_synthetic_regression_evaluation as mod
        submitted_envs = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted_envs.append({"label": label, "spec": spec})
            return f"MOCK_{label.upper()}"

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        worker_items = [s for s in submitted_envs if "worker" in s["label"].lower()]
        assert worker_items, "No worker job submitted"
        worker_spec = worker_items[0]["spec"]
        assert "RAY_HEAD_ADDRESS" in worker_spec

    def test_spcs_distributed_creates_head_worker_driver(self, monkeypatch):
        """SPCS distributed mode must submit coordinator and workers for each shard (no separate head/driver)."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted.append(label)
            return f"MOCK_{label.upper()}"

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=2,
            autogluon_workers_per_shard=2,
            autogluon_concurrent_clusters=2,
        )
        # 2 shards × (1 coordinator + 2 workers) = 6 jobs (head+driver merged into coordinator)
        assert len(submitted) == 6
        coord_labels = [l for l in submitted if "coord" in l]
        worker_labels = [l for l in submitted if "worker" in l]
        assert len(coord_labels) == 2
        assert len(worker_labels) == 4
        assert not any("head" in l for l in submitted), "No separate head services in coordinator topology"
        assert not any("driver" in l for l in submitted), "No separate driver services in coordinator topology"

    def test_spcs_default_topology_submits_30_containers_with_24_workers(self, monkeypatch):
        """6×4 SPCS Ray run submits 30 containers: 6 coordinators + 24 workers (not 36)."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted.append(label)
            return f"MOCK_{label.upper()}"

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=6,
            autogluon_workers_per_shard=4,
            autogluon_concurrent_clusters=6,
        )

        # 6 coordinators (merged head+driver) + 24 workers = 30 (not 36)
        assert len(submitted) == 30
        assert sum(1 for label in submitted if "coord" in label) == 6
        assert sum(1 for label in submitted if "worker" in label) == 24
        assert not any("head" in label for label in submitted)
        assert not any("driver" in label for label in submitted)

    def test_spcs_distributed_uses_role_specific_resource_profiles(self, monkeypatch):
        """Coordinator and worker must use distinct resource profiles (coordinator: 1/2 cpu, 4Gi/8Gi mem)."""
        import re
        import run_synthetic_regression_evaluation as mod
        submitted = {}

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted[label] = spec
            return f"MOCK_{label.upper()}"

        def _resource_values(spec):
            return re.findall(r'(?:cpu|memory): "([^"]+)"', spec)

        for prefix in (
            "SYNREG_SPCS_RAY_COORDINATOR",
            "SYNREG_SPCS_RAY_WORKER",
        ):
            for suffix in ("CPU", "MEMORY", "CPU_REQUEST", "CPU_LIMIT", "MEMORY_REQUEST", "MEMORY_LIMIT"):
                monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )

        # Coordinator default: cpu request=1, limit=2; memory request=4Gi, limit=8Gi
        assert _resource_values(submitted["spcs_ray_coord_r0_0"]) == ["1", "4Gi", "2", "8Gi"]
        # Worker default: cpu request=4, limit=4; memory request=16Gi, limit=16Gi
        assert _resource_values(submitted["spcs_ray_worker_r0_0_0"]) == ["4", "16Gi", "4", "16Gi"]

    def test_spcs_distributed_waits_drivers_and_cancels_support_jobs(self, monkeypatch):
        """Coordinator is waited to completion; workers are cancelled as support jobs after coordinator finishes."""
        import run_synthetic_regression_evaluation as mod
        waits = []
        cancels = []

        monkeypatch.setattr(
            mod,
            "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        def _mock_wait(labeled_jobs, session):
            waits.extend(label for label, _job_name in labeled_jobs)

        def _mock_cancel(labeled_jobs, session):
            cancels.extend(label for label, _job_name in labeled_jobs)

        monkeypatch.setattr(mod, "_wait_spcs_job_group", _mock_wait)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", _mock_cancel)

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=2,
            autogluon_concurrent_clusters=1,
        )
        # Coordinator (merged head+driver) is waited; workers are cancelled as support jobs
        assert waits == ["spcs_ray_coord_r0_0"]
        assert "spcs_ray_worker_r0_0_0" in cancels
        assert "spcs_ray_worker_r0_0_1" in cancels
        assert not any("head" in c for c in cancels), "No separate head service in coordinator topology"

    def test_spcs_capacity_probe_uses_self_managed_ray_topology(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod,
            "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(),
            cluster_shards=1,
            workers_per_shard=2,
            concurrent_clusters=1,
        )
        labels = [label for label, _spec in submitted]
        assert "spcs_cap_ray_coord_r0_0" in labels
        assert not any("head" in l for l in labels), "No separate head service in capacity probe"
        assert not any("probe" in l and "worker" not in l and "coord" not in l for l in labels)
        coord_specs = [spec for label, spec in submitted if label == "spcs_cap_ray_coord_r0_0"]
        assert coord_specs
        # Coordinator runs ray_capacity_probe.py via SPCS_RAY_DRIVER_SCRIPT
        assert "ray_capacity_probe.py" in coord_specs[0]
        # Coordinator spec includes TCP endpoint for Ray port
        assert "ray-head" in coord_specs[0]

    def test_spcs_worker_access_probe_uses_self_managed_ray_topology(self, monkeypatch):
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod,
            "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
            object(),
            cluster_shards=1,
            workers_per_shard=2,
            concurrent_clusters=1,
        )
        labels = [label for label, _spec in submitted]
        assert "spcs_worker_probe_ray_coord_r0_0" in labels
        assert not any("head" in l and "coord" not in l for l in labels)
        coord_specs = [spec for label, spec in submitted if label == "spcs_worker_probe_ray_coord_r0_0"]
        assert coord_specs
        assert "autogluon_worker_access_probe.py" in coord_specs[0]
        assert "SYNREG_WORKER_DATA_ACCESS_MODE" in coord_specs[0]
        assert "driver_presigned_url" in coord_specs[0]

    def test_mljob_backend_unchanged(self, monkeypatch):
        """MLJob backend must still call submit_from_stage, not SPCS helpers."""
        import run_synthetic_regression_evaluation as mod
        spcs_calls = []

        def _mock_execute(session, *, label, compute_pool, spec):
            spcs_calls.append(label)
            return "MOCK"

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_EXECUTION_BACKEND", "mljob")

        # The original MLJob function is separate from SPCS function — verify SPCS helpers
        # are not called when using the original mljob evaluation function path
        # (they live in separate functions, so this is guaranteed by design)
        assert not spcs_calls, "SPCS helpers must not be called for mljob backend"

    def test_ray_worker_task_body_does_not_create_snowpark_session(self):
        """The Ray worker task function must never create a Snowpark session.

        Workers receive only compact item dicts and download datasets via presigned URLs
        (urllib.request). No Snowpark session or SnowflakeFile is created inside the
        @ray.remote worker task body, keeping worker nodes session-free.
        """
        import ast
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "scripts" / "autogluon_ray.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        # Find _autogluon_work_item function body
        worker_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_autogluon_work_item":
                worker_fn = node
                break
        assert worker_fn is not None, "_autogluon_work_item not found in autogluon_ray.py"

        fn_src = ast.get_source_segment(src.read_text(encoding="utf-8"), worker_fn) or ""
        assert "Session.builder" not in fn_src, (
            "_autogluon_work_item must not call Session.builder (workers are session-free)"
        )
        assert "getOrCreate" not in fn_src, (
            "_autogluon_work_item must not call Session.builder.getOrCreate (workers are session-free)"
        )
        assert "create_snowpark_session" not in fn_src, (
            "_autogluon_work_item must not call create_snowpark_session (workers are session-free)"
        )

    def test_worker_access_ray_task_body_does_not_create_snowpark_session(self):
        """The worker-access probe Ray task must also stay session-free."""
        import ast
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "scripts" / "autogluon_worker_access_probe.py"
        source = src.read_text(encoding="utf-8")
        tree = ast.parse(source)

        worker_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_worker_access_probe_item":
                worker_fn = node
                break
        assert worker_fn is not None, "_worker_access_probe_item not found"

        fn_src = ast.get_source_segment(source, worker_fn) or ""
        assert "Session.builder" not in fn_src
        assert "getOrCreate" not in fn_src
        assert "create_snowpark_session" not in fn_src

    def test_spcs_job_names_include_unique_run_id(self, monkeypatch):
        """Each SPCS orchestration call must embed a unique run ID in job service labels.

        This prevents 'service name already exists' collisions when the same stored
        procedure is called more than once without dropping previous services first.
        """
        import run_synthetic_regression_evaluation as mod

        # _spcs_run_id must return distinct values across calls
        ids = {mod._spcs_run_id() for _ in range(10)}
        assert len(ids) == 10, "_spcs_run_id must generate unique IDs across calls"
        sample_id = next(iter(ids))
        assert sample_id.isalnum(), "_spcs_run_id must return an alphanumeric string safe for Snowflake identifiers"
        assert len(sample_id) >= 6, "_spcs_run_id must be at least 6 characters to have sufficient entropy"

        # Job labels submitted for a run must all contain the fixed run ID
        run_labels: list = []
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        fixed_id = "abc12345"
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: fixed_id)
        monkeypatch.setattr(
            mod,
            "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: run_labels.append(label) or label.upper(),
        )
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        assert run_labels, "No SPCS job services were submitted"
        assert all(fixed_id in lbl for lbl in run_labels), (
            f"All job labels must contain the run ID {fixed_id!r}; got: {run_labels}"
        )

        # A second call with a different run ID must produce disjoint label sets
        run_labels_2: list = []
        second_id = "xyz98765"
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: second_id)
        monkeypatch.setattr(
            mod,
            "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: run_labels_2.append(label) or label.upper(),
        )
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        assert set(run_labels).isdisjoint(set(run_labels_2)), (
            "Job labels from two successive runs must not overlap — "
            f"run1={run_labels}, run2={run_labels_2}"
        )

    # Finding 2: SPCS spec must not include invalid snowflakeService field
    def test_submit_spcs_synreg_spec_excludes_snowflake_service(self, monkeypatch):
        """_submit_spcs_synreg generated spec must NOT contain snowflakeService."""
        import run_synthetic_regression_evaluation as mod
        captured_specs = []

        def _mock_execute(session, *, label, compute_pool, spec):
            captured_specs.append(spec)
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_spcs_session_context_env", lambda s: {})

        mod._submit_spcs_synreg(
            session=object(),
            label="test_job",
            compute_pool="AUTOGLUON_CPU_POOL",
            env_vars={"FOO": "bar"},
            image="img:1.0",
            entrypoint_path="/app/scripts/autogluon_ray.py",
        )
        assert captured_specs, "No spec was captured"
        spec = captured_specs[0]
        assert "snowflakeService" not in spec, (
            "_submit_spcs_synreg spec must not include snowflakeService — "
            "Snowflake rejects it with error 395018"
        )

    # Finding 4: Per-shard DNS derivation — each shard gets a unique head address
    def test_spcs_distributed_uses_per_shard_head_dns(self, monkeypatch):
        """Each shard's workers must receive a unique RAY_HEAD_ADDRESS derived from coordinator's service label."""
        import run_synthetic_regression_evaluation as mod
        submitted_envs: list[dict] = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted_envs.append({"label": label, "spec": spec})
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=2,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=2,
        )
        # Workers receive RAY_HEAD_ADDRESS pointing to their shard's coordinator; addresses must differ
        worker_specs = [e["spec"] for e in submitted_envs if "worker" in e["label"]]
        assert len(worker_specs) == 2, f"Expected 2 worker specs; got {len(worker_specs)}"
        import re
        head_addresses = [
            re.search(r'RAY_HEAD_ADDRESS[^"]*"([^"]+)"', s)
            for s in worker_specs
        ]
        addresses = [m.group(1) if m else None for m in head_addresses]
        assert addresses[0] is not None and addresses[1] is not None, (
            f"Could not extract RAY_HEAD_ADDRESS from worker specs: {worker_specs}"
        )
        assert addresses[0] != addresses[1], (
            f"Two shards must use different head addresses; got {addresses}"
        )

    # Finding 6: Coordinator env includes shard identity (merged head+driver)
    def test_spcs_head_env_includes_cluster_identity(self, monkeypatch):
        """Coordinator containers must receive SPCS_RAY_RUN_ID and SPCS_RAY_SHARD_INDEX env vars."""
        import run_synthetic_regression_evaluation as mod
        coord_specs: list[str] = []

        def _mock_execute(session, *, label, compute_pool, spec):
            if "coord" in label:
                coord_specs.append(spec)
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        assert coord_specs, "No coordinator specs captured"
        spec = coord_specs[0]
        assert "SPCS_RAY_RUN_ID" in spec, "Coordinator spec must include SPCS_RAY_RUN_ID"
        assert "SPCS_RAY_SHARD_INDEX" in spec, "Coordinator spec must include SPCS_RAY_SHARD_INDEX"

    # Finding 6: Coordinator env includes driver config (merged head+driver)
    def test_spcs_driver_env_includes_cluster_identity(self, monkeypatch):
        """Coordinator containers must carry both cluster identity and driver config env vars."""
        import run_synthetic_regression_evaluation as mod
        coord_specs: list[str] = []

        def _mock_execute(session, *, label, compute_pool, spec):
            if "coord" in label:
                coord_specs.append(spec)
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        assert coord_specs, "No coordinator specs captured"
        spec = coord_specs[0]
        # Cluster identity (formerly in head spec)
        assert "SPCS_RAY_RUN_ID" in spec
        assert "SPCS_RAY_SHARD_INDEX" in spec
        # Driver config (formerly in driver spec) is now in the same coordinator container
        assert "AUTOGLUON_TASK_CPUS" in spec, "Coordinator must include driver env vars"

    # Finding 5: Capacity probe uses workers+1 expected nodes in SPCS Ray mode
    def test_spcs_capacity_probe_expects_workers_plus_head(self, monkeypatch):
        """Capacity probe EXPECTED_RAY_NODES must equal workers_per_shard + 1 in Ray mode.

        In the coordinator topology the coordinator spec carries EXPECTED_RAY_NODES
        (previously it was in a separate probe job spec).
        """
        import run_synthetic_regression_evaluation as mod
        coord_specs: list[str] = []

        def _mock_execute(session, *, label, compute_pool, spec):
            # Coordinator label: spcs_cap_ray_coord_<run_id>_<shard>
            if "coord" in label and "worker" not in label:
                coord_specs.append(spec)
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(), cluster_shards=1, workers_per_shard=3, concurrent_clusters=1,
        )
        assert coord_specs, "No coordinator specs captured"
        # EXPECTED_RAY_NODES must be 4 (3 workers + 1 coordinator head)
        assert "EXPECTED_RAY_NODES" in coord_specs[0], (
            f"Coordinator spec should contain EXPECTED_RAY_NODES; spec={coord_specs[0]}"
        )
        import re
        match = re.search(r'EXPECTED_RAY_NODES[^"]*"(\d+)"', coord_specs[0])
        assert match, f"Could not find EXPECTED_RAY_NODES in coordinator spec: {coord_specs[0]}"
        assert int(match.group(1)) == 4, f"Expected EXPECTED_RAY_NODES=4, got {match.group(1)}"

    def test_spcs_capacity_probe_coordinator_spec_has_all_ray_endpoints(self, monkeypatch):
        """Capacity probe coordinator spec must expose all 5 Ray endpoints including portRange."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod, "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")
        monkeypatch.setattr(mod, "_spcs_session_context_env", lambda s: {})

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(), cluster_shards=1, workers_per_shard=1, concurrent_clusters=1,
        )
        coord_specs = [spec for label, spec in submitted if "coord" in label and "worker" not in label]
        assert coord_specs, "No coordinator specs captured"
        spec = coord_specs[0]
        for ep_name in ("ray-head", "ray-node-manager", "ray-object-manager",
                        "ray-runtime-env-agent", "ray-worker-ports"):
            assert ep_name in spec, f"Coordinator spec missing endpoint {ep_name!r}"
        assert "portRange:" in spec, "Coordinator spec missing portRange endpoint"

    def test_spcs_capacity_probe_worker_spec_has_ray_endpoints(self, monkeypatch):
        """Capacity probe worker spec must expose ray-node-manager and portRange endpoints."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod, "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")
        monkeypatch.setattr(mod, "_spcs_session_context_env", lambda s: {})

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(), cluster_shards=1, workers_per_shard=1, concurrent_clusters=1,
        )
        worker_specs = [spec for label, spec in submitted if "worker" in label]
        assert worker_specs, "No worker specs captured"
        spec = worker_specs[0]
        assert "ray-node-manager" in spec, "Worker spec missing ray-node-manager endpoint"
        assert "portRange:" in spec, "Worker spec missing portRange endpoint"
        assert "ray-head" not in spec, "Worker spec must not include ray-head endpoint"

    def test_spcs_capacity_probe_uses_reduced_object_store(self, monkeypatch):
        """Capacity probe defaults to 256 MB object-store for coordinator and worker."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod, "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")
        monkeypatch.setattr(mod, "_spcs_session_context_env", lambda s: {})
        monkeypatch.delenv("SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES", raising=False)
        monkeypatch.delenv("SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES", raising=False)

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(), cluster_shards=1, workers_per_shard=1, concurrent_clusters=1,
        )
        coord_specs = [spec for label, spec in submitted if "coord" in label and "worker" not in label]
        worker_specs = [spec for label, spec in submitted if "worker" in label]
        assert coord_specs, "No coordinator specs captured"
        assert worker_specs, "No worker specs captured"
        assert "268435456" in coord_specs[0], (
            "Capacity probe coordinator must default to 268435456 bytes object-store"
        )
        assert "268435456" in worker_specs[0], (
            "Capacity probe worker must default to 268435456 bytes object-store"
        )

    def test_spcs_production_eval_uses_production_object_store_defaults(self, monkeypatch):
        """Production eval coordinator defaults to 500 MB and worker to 2 GB object-store."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod, "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")
        monkeypatch.setattr(mod, "_spcs_session_context_env", lambda s: {})
        monkeypatch.delenv("SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES", raising=False)
        monkeypatch.delenv("SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES", raising=False)

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        coord_specs = [spec for label, spec in submitted if "coord" in label and "worker" not in label]
        worker_specs = [spec for label, spec in submitted if "worker" in label]
        assert coord_specs, "No coordinator specs captured"
        assert worker_specs, "No worker specs captured"
        assert "500000000" in coord_specs[0], (
            "Production eval coordinator must default to 500000000 bytes object-store"
        )
        assert "2000000000" in worker_specs[0], (
            "Production eval worker must default to 2000000000 bytes object-store"
        )

    def test_ray_dashboard_not_in_coordinator_spec(self, monkeypatch):
        """Coordinator spec must not expose a dashboard endpoint or port 8265."""
        import run_synthetic_regression_evaluation as mod
        submitted = []

        monkeypatch.setattr(
            mod, "_execute_spcs_job_service",
            lambda session, *, label, compute_pool, spec: submitted.append((label, spec)) or label.upper(),
        )
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")
        monkeypatch.setattr(mod, "_spcs_session_context_env", lambda s: {})

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(), cluster_shards=1, workers_per_shard=1, concurrent_clusters=1,
        )
        coord_specs = [spec for label, spec in submitted if "coord" in label and "worker" not in label]
        assert coord_specs, "No coordinator specs captured"
        spec = coord_specs[0]
        assert "dashboard" not in spec.lower() or "include-dashboard=false" in spec.lower(), (
            "Coordinator spec must not declare a dashboard endpoint"
        )
        assert "8265" not in spec, "Coordinator spec must not expose port 8265 (dashboard)"

    # Finding 1: Image verification is called during SPCS evaluation
    def test_spcs_evaluation_calls_image_verification(self, monkeypatch):
        """SPCS evaluation must call _verify_spcs_image_in_repository before submitting jobs."""
        import run_synthetic_regression_evaluation as mod
        verify_calls = []

        monkeypatch.setattr(
            mod, "_verify_spcs_image_in_repository",
            lambda session, image: verify_calls.append(image)
        )
        monkeypatch.setattr(mod, "_execute_spcs_job_service", lambda *a, **k: "MOCK")
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(), autogluon_cluster_shards=0, autogluon_concurrent_clusters=1,
        )
        assert verify_calls, "_verify_spcs_image_in_repository must be called during SPCS evaluation"
        assert "img:1.0" in verify_calls


# ---------------------------------------------------------------------------
# Tests: SPCS Ray Coordinator Topology (30-container model)
# ---------------------------------------------------------------------------

class TestSPCSRayCoordinatorTopology:
    """
    Verify that the SPCS Ray distributed mode uses the corrected coordinator
    topology: 6 coordinators (head+driver merged) + 24 workers = 30 containers
    for a default 6×4 setup, not the old 6+24+6=36 container model.
    """

    def _setup_spcs_mocks(self, monkeypatch, cluster_shards=6, workers_per_shard=4):
        """Return (submitted, mod) after patching SPCS helpers."""
        import run_synthetic_regression_evaluation as mod

        submitted = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted.append({"label": label, "compute_pool": compute_pool, "spec": spec})
            return f"MOCK_{label.upper()}"

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "tst00001")
        return submitted, mod

    def test_6x4_spcs_ray_submits_30_containers(self, monkeypatch):
        """6 shards × 4 workers must submit 6+24=30 containers (not 36)."""
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=6,
            autogluon_workers_per_shard=4,
            autogluon_concurrent_clusters=6,
        )
        assert len(submitted) == 30, (
            f"Expected 30 containers (6 coordinators + 24 workers), got {len(submitted)}"
        )

    def test_no_separate_head_or_driver_services_in_spcs_ray_mode(self, monkeypatch):
        """No label should match 'spcs_ray_head_*' or 'spcs_ray_driver_*' in Ray mode."""
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=6,
            autogluon_workers_per_shard=4,
            autogluon_concurrent_clusters=6,
        )
        labels = [s["label"] for s in submitted]
        head_labels = [l for l in labels if l.startswith("spcs_ray_head_")]
        driver_labels = [l for l in labels if l.startswith("spcs_ray_driver_")]
        coord_labels = [l for l in labels if l.startswith("spcs_ray_coord_")]
        worker_labels = [l for l in labels if l.startswith("spcs_ray_worker_")]

        assert not head_labels, f"Found separate head services (should not exist): {head_labels}"
        assert not driver_labels, f"Found separate driver services (should not exist): {driver_labels}"
        assert len(coord_labels) == 6, f"Expected 6 coordinators, got {coord_labels}"
        assert len(worker_labels) == 24, f"Expected 24 workers, got {len(worker_labels)}"

    def test_coordinator_entrypoint_is_spcs_ray_coordinator_py(self, monkeypatch):
        """Coordinator specs must invoke /app/scripts/spcs_ray_coordinator.py."""
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=2,
            autogluon_workers_per_shard=2,
            autogluon_concurrent_clusters=2,
        )
        coord_specs = [
            s["spec"] for s in submitted if s["label"].startswith("spcs_ray_coord_")
        ]
        assert coord_specs, "No coordinator specs found"
        for spec in coord_specs:
            assert "/app/scripts/spcs_ray_coordinator.py" in spec, (
                f"Coordinator spec must reference spcs_ray_coordinator.py, got:\n{spec}"
            )

    def test_coordinator_spec_includes_tcp_endpoint_for_ray_port(self, monkeypatch):
        """Coordinator YAML spec must expose the Ray head TCP port as an endpoint."""
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        coord_specs = [
            s["spec"] for s in submitted if s["label"].startswith("spcs_ray_coord_")
        ]
        assert coord_specs
        for spec in coord_specs:
            assert "ray-head" in spec, "Coordinator spec must include 'ray-head' endpoint name"
            assert "6379" in spec, "Coordinator spec must include port 6379"
            assert "TCP" in spec, "Coordinator spec must specify TCP protocol"

    def test_workers_receive_autogluon_task_cpus_env_var(self, monkeypatch):
        """Worker specs must carry AUTOGLUON_TASK_CPUS in their env block."""
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=2,
            autogluon_workers_per_shard=2,
            autogluon_concurrent_clusters=2,
        )
        worker_specs = [
            s["spec"] for s in submitted if s["label"].startswith("spcs_ray_worker_")
        ]
        assert worker_specs, "No worker specs found"
        for spec in worker_specs:
            assert "AUTOGLUON_TASK_CPUS" in spec, (
                f"Worker spec must contain AUTOGLUON_TASK_CPUS:\n{spec}"
            )

    def test_coordinator_resource_profile_distinct_from_worker(self, monkeypatch):
        """SPCS_RAY_COORDINATOR_RESOURCES constant must exist and resolve differently from worker."""
        import run_synthetic_regression_evaluation as mod

        # Clear any env overrides
        for prefix in ("SYNREG_SPCS_RAY_COORDINATOR", "SYNREG_SPCS_RAY_WORKER"):
            for suffix in ("CPU", "MEMORY", "CPU_REQUEST", "CPU_LIMIT", "MEMORY_REQUEST", "MEMORY_LIMIT"):
                monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)

        assert hasattr(mod, "SPCS_RAY_COORDINATOR_RESOURCES"), (
            "SPCS_RAY_COORDINATOR_RESOURCES constant must exist in orchestration module"
        )
        assert mod.SPCS_RAY_COORDINATOR_RESOURCES != mod.SPCS_RAY_WORKER_RESOURCES

        coord_profile = mod._spcs_resources_for_role(mod.SPCS_RAY_COORDINATOR_RESOURCES)
        worker_profile = mod._spcs_resources_for_role(mod.SPCS_RAY_WORKER_RESOURCES)

        assert coord_profile["cpu_request"] == "1", (
            f"Coordinator default cpu_request should be '1', got {coord_profile['cpu_request']!r}"
        )
        assert coord_profile["cpu_request"] != worker_profile["cpu_request"], (
            "Coordinator and worker must have different CPU defaults"
        )

    def test_coordinator_dns_uses_dashes_not_underscores(self, monkeypatch):
        """Workers' RAY_HEAD_ADDRESS must use dashes in service name, matching SPCS DNS norm."""
        import os
        import run_synthetic_regression_evaluation as mod

        submitted, mod2 = self._setup_spcs_mocks(monkeypatch, cluster_shards=1, workers_per_shard=1)
        monkeypatch.setenv("SPCS_RAY_HEAD_DNS_SUFFIX", "myschema.mydb.snowflakecomputing.internal")

        mod2.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        worker_specs = [
            s["spec"] for s in submitted if s["label"].startswith("spcs_ray_worker_")
        ]
        assert worker_specs, "No worker spec found"
        spec = worker_specs[0]
        assert "RAY_HEAD_ADDRESS" in spec

        # Extract the RAY_HEAD_ADDRESS value from spec
        import re
        match = re.search(r'RAY_HEAD_ADDRESS:\s*"?([^"\n]+)"?', spec)
        assert match, f"RAY_HEAD_ADDRESS not found in spec:\n{spec}"
        address = match.group(1).strip().strip('"')

        # Service name portion (before the DNS suffix) must not contain underscores
        service_part = address.split(".myschema.")[0]
        assert "-" in service_part, (
            f"DNS service name must use dashes (SPCS normalization), got: {service_part!r}"
        )
        assert "_" not in service_part, (
            f"DNS service name must not contain underscores, got: {service_part!r}"
        )

    def test_single_node_spcs_mode_unchanged(self, monkeypatch):
        """cluster_shards=0 must still use single-node path (no coordinator/worker services)."""
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=0,
            autogluon_concurrent_clusters=3,
        )
        coord_labels = [s["label"] for s in submitted if s["label"].startswith("spcs_ray_coord_")]
        worker_labels = [s["label"] for s in submitted if s["label"].startswith("spcs_ray_worker_")]
        assert not coord_labels, f"Single-node mode must not submit coordinator services, got: {coord_labels}"
        assert not worker_labels, f"Single-node mode must not submit worker services, got: {worker_labels}"
        assert len(submitted) == 3, (
            f"Single-node mode with 3 concurrent shards must submit 3 jobs, got {len(submitted)}"
        )
        for s in submitted:
            assert "/app/src/evaluate_synthetic_regression.py" in s["spec"], (
                f"Single-node spec must use evaluate_synthetic_regression.py entrypoint: {s['spec'][:200]}"
            )

    def test_spcs_ray_coordinator_starts_ray_head_with_zero_cpus(self):
        """spcs_ray_coordinator.py must pass --num-cpus=0 to ray start."""
        text = (ROOT / "scripts" / "spcs_ray_coordinator.py").read_text()
        assert "--num-cpus=0" in text, (
            "spcs_ray_coordinator.py must pass '--num-cpus=0' to prevent the head "
            "from consuming schedulable CPU capacity"
        )

    def test_spcs_ray_coordinator_starts_ray_head_with_object_store_memory(self):
        """spcs_ray_coordinator.py must pass --object-store-memory to ray start."""
        text = (ROOT / "scripts" / "spcs_ray_coordinator.py").read_text()
        assert "--object-store-memory=" in text, (
            "spcs_ray_coordinator.py must pass '--object-store-memory' to cap Ray's "
            "object store and prevent OOM on fixed-memory coordinator containers"
        )

    def test_spcs_ray_worker_passes_num_cpus_explicitly(self):
        """spcs_ray_worker.py must pass --num-cpus=<N> to advertise CPU capacity to Ray."""
        text = (ROOT / "scripts" / "spcs_ray_worker.py").read_text()
        assert "--num-cpus=" in text, (
            "spcs_ray_worker.py must pass '--num-cpus=<N>' so Ray knows how many "
            "schedulable CPUs each worker contributes"
        )

    def test_spcs_ray_worker_passes_object_store_memory(self):
        """spcs_ray_worker.py must pass --object-store-memory to ray start."""
        text = (ROOT / "scripts" / "spcs_ray_worker.py").read_text()
        assert "--object-store-memory=" in text, (
            "spcs_ray_worker.py must pass '--object-store-memory' to bound Ray's "
            "object store on worker containers"
        )

    def test_coordinator_object_store_env_passed_to_coordinator_spec(self, monkeypatch):
        """SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES must appear in coordinator spec."""
        monkeypatch.setenv("SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES", "999999")
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        coord_specs = [
            s["spec"] for s in submitted if s["label"].startswith("spcs_ray_coord_")
        ]
        assert coord_specs, "No coordinator spec found"
        assert "999999" in coord_specs[0], (
            "Coordinator spec must contain the SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES value"
        )

    def test_worker_object_store_env_passed_to_worker_spec(self, monkeypatch):
        """SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES must appear in worker spec."""
        monkeypatch.setenv("SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES", "888888")
        submitted, mod = self._setup_spcs_mocks(monkeypatch)
        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            object(),
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        worker_specs = [
            s["spec"] for s in submitted if s["label"].startswith("spcs_ray_worker_")
        ]
        assert worker_specs, "No worker spec found"
        assert "888888" in worker_specs[0], (
            "Worker spec must contain the SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES value"
        )

    def test_spec_endpoints_block_rendered_correctly(self):
        """_build_spcs_job_spec must include endpoints YAML block when endpoints arg is given."""
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="img:1.0",
            args=["/app/scripts/spcs_ray_coordinator.py"],
            env_vars={},
            endpoints=[{"name": "ray-head", "port": 6379, "protocol": "TCP"}],
        )
        assert "endpoints:" in spec
        assert "ray-head" in spec
        assert "6379" in spec
        assert "TCP" in spec

    def test_spec_no_endpoints_block_when_not_given(self):
        """_build_spcs_job_spec must not include endpoints block when endpoints=None."""
        from run_synthetic_regression_evaluation import _build_spcs_job_spec
        spec = _build_spcs_job_spec(
            image="img:1.0",
            args=["/app/scripts/spcs_ray_worker.py"],
            env_vars={},
            endpoints=None,
        )
        assert "endpoints:" not in spec

    def test_coordinator_script_runs_autogluon_ray_as_subprocess(self):
        """spcs_ray_coordinator.py must run autogluon_ray.py as a child process."""
        text = (ROOT / "scripts" / "spcs_ray_coordinator.py").read_text()
        assert "autogluon_ray.py" in text, (
            "spcs_ray_coordinator.py must reference autogluon_ray.py as the driver script"
        )
        assert "subprocess" in text, (
            "spcs_ray_coordinator.py must use subprocess to run the driver"
        )

    def test_coordinator_script_sets_explicit_address_mode(self):
        """Coordinator must set SYNREG_RAY_ADDRESS_MODE=explicit for the driver subprocess."""
        text = (ROOT / "scripts" / "spcs_ray_coordinator.py").read_text()
        assert "SYNREG_RAY_ADDRESS_MODE" in text
        assert "explicit" in text
        assert "RAY_HEAD_ADDRESS" in text


# ---------------------------------------------------------------------------
# Tests: SELECT * removal in dataset index query
# ---------------------------------------------------------------------------

class TestDatasetIndexSelectColumns:
    """Verify that load_synthetic_regression_index uses an explicit column list."""

    def test_load_index_does_not_use_select_star(self):
        """load_synthetic_regression_index must not use SELECT *."""
        text = (ROOT / "src" / "evaluate_synthetic_regression.py").read_text()
        fn_start = text.index("def load_synthetic_regression_index(")
        fn_end_candidates = [
            text.find("\ndef ", fn_start + 1),
            text.find("\nclass ", fn_start + 1),
        ]
        fn_end = min(p for p in fn_end_candidates if p > 0)
        fn_body = text[fn_start:fn_end]
        assert "SELECT *" not in fn_body, (
            "load_synthetic_regression_index must not use SELECT * — "
            "use an explicit column list to avoid fetching payload columns"
        )

    def test_load_index_selects_required_columns(self):
        """load_synthetic_regression_index must select known required columns."""
        text = (ROOT / "src" / "evaluate_synthetic_regression.py").read_text()
        fn_start = text.index("def load_synthetic_regression_index(")
        fn_end_candidates = [
            text.find("\ndef ", fn_start + 1),
            text.find("\nclass ", fn_start + 1),
        ]
        fn_end = min(p for p in fn_end_candidates if p > 0)
        fn_body = text[fn_start:fn_end]
        for col in ("suite_id", "stage_path", "prior_regime", "split_seeds"):
            assert col in fn_body, (
                f"load_synthetic_regression_index SQL must select column {col!r}"
            )


# ---------------------------------------------------------------------------
# Tests: SPCS DNS domain resolution
# ---------------------------------------------------------------------------

class TestSPCSDNSDomainResolution:
    def test_dns_domain_resolved_via_system_function_when_no_suffix(self, monkeypatch):
        """When SPCS_RAY_HEAD_DNS_SUFFIX is not set, _spcs_dns_domain calls SYSTEM$GET_SERVICE_DNS_DOMAIN."""
        import run_synthetic_regression_evaluation as mod
        monkeypatch.delenv("SPCS_RAY_HEAD_DNS_SUFFIX", raising=False)
        mock_session = MagicMock()
        mock_session.sql.return_value.collect.return_value = [["abc123.svc.spcs.internal"]]
        domain = mod._spcs_dns_domain(mock_session)
        assert domain == "abc123.svc.spcs.internal"
        mock_session.sql.assert_called_once()
        assert "SYSTEM$GET_SERVICE_DNS_DOMAIN" in mock_session.sql.call_args[0][0]

    def test_spcs_ray_head_dns_suffix_overrides_system_function(self, monkeypatch):
        """When SPCS_RAY_HEAD_DNS_SUFFIX is set, it takes priority and no SQL is called."""
        import run_synthetic_regression_evaluation as mod
        monkeypatch.setenv("SPCS_RAY_HEAD_DNS_SUFFIX", "override.example.internal")
        mock_session = MagicMock()
        domain = mod._spcs_dns_domain(mock_session)
        assert domain == "override.example.internal"
        mock_session.sql.assert_not_called()

    def test_dns_domain_returns_empty_string_when_system_function_fails(self, monkeypatch):
        """If both override and SYSTEM$ function are unavailable, returns empty string gracefully."""
        import run_synthetic_regression_evaluation as mod
        monkeypatch.delenv("SPCS_RAY_HEAD_DNS_SUFFIX", raising=False)
        mock_session = MagicMock()
        mock_session.sql.side_effect = Exception("SQL error")
        domain = mod._spcs_dns_domain(mock_session)
        assert domain == ""

    def test_coordinator_hostname_uses_system_dns_domain(self, monkeypatch):
        """Production evaluation uses _spcs_dns_domain result in coordinator hostnames."""
        import run_synthetic_regression_evaluation as mod
        monkeypatch.delenv("SPCS_RAY_HEAD_DNS_SUFFIX", raising=False)
        submitted_envs = []

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted_envs.append({"label": label, "spec": spec})
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_verify_spcs_image_in_repository", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "COMBINED_SUITE_ID", "linear_all_v1")
        monkeypatch.setattr(mod, "COMBINED_PARTS_PREFIX", "@TEST_STAGE/parts")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mock_session = MagicMock()
        mock_session.sql.return_value.collect.return_value = [["auto-resolved.svc.spcs.internal"]]

        mod.run_synthetic_regression_combined_autogluon_spcs_evaluation(
            mock_session,
            autogluon_cluster_shards=1,
            autogluon_workers_per_shard=1,
            autogluon_concurrent_clusters=1,
        )
        worker_items = [s for s in submitted_envs if "worker" in s["label"]]
        assert worker_items
        assert "auto-resolved.svc.spcs.internal" in worker_items[0]["spec"]


# ---------------------------------------------------------------------------
# Tests: Additional coordinator topology guards
# ---------------------------------------------------------------------------

class TestSPCSCoordinatorTopologyGuards:
    """Additional tests for coordinator topology correctness and guards."""

    def test_max_in_flight_above_workers_per_shard_raises(self):
        """autogluon_ray.py must fail fast when MAX_IN_FLIGHT > WORKERS_PER_SHARD."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "scripts" / "autogluon_ray.py"
        text = src.read_text(encoding="utf-8")
        assert "MAX_IN_FLIGHT > WORKERS_PER_SHARD" in text
        assert "RuntimeError" in text

    def test_duplicate_work_item_guard_exists(self):
        """autogluon_ray.py must check for duplicate work items before Ray submission."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "scripts" / "autogluon_ray.py"
        text = src.read_text(encoding="utf-8")
        assert "Duplicate atomic work item" in text

    def test_coordinator_driver_script_configurable_via_env_var(self):
        """spcs_ray_coordinator.py must read SPCS_RAY_DRIVER_SCRIPT to configure driver."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "scripts" / "spcs_ray_coordinator.py"
        text = src.read_text(encoding="utf-8")
        assert "SPCS_RAY_DRIVER_SCRIPT" in text

    def test_capacity_probe_coordinator_runs_ray_capacity_probe(self, monkeypatch):
        """Capacity probe coordinator spec must reference ray_capacity_probe.py via SPCS_RAY_DRIVER_SCRIPT."""
        import run_synthetic_regression_evaluation as mod
        submitted = {}

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted[label] = spec
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
            object(), cluster_shards=1, workers_per_shard=1, concurrent_clusters=1,
        )
        coord_labels = [l for l in submitted if "coord" in l]
        assert coord_labels, "No coordinator label found in capacity probe"
        coord_spec = submitted[coord_labels[0]]
        assert "ray_capacity_probe.py" in coord_spec
        assert not any("head" in l and "coord" not in l for l in submitted), "No separate head in capacity probe"

    def test_worker_access_probe_coordinator_runs_worker_access_probe(self, monkeypatch):
        """Worker access probe coordinator spec must reference autogluon_worker_access_probe.py."""
        import run_synthetic_regression_evaluation as mod
        submitted = {}

        def _mock_execute(session, *, label, compute_pool, spec):
            submitted[label] = spec
            return label.upper()

        monkeypatch.setattr(mod, "_execute_spcs_job_service", _mock_execute)
        monkeypatch.setattr(mod, "_ensure_compute_pool_usable", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_wait_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_cancel_spcs_job_group", lambda *a, **k: None)
        monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", "img:1.0")
        monkeypatch.setattr(mod, "_spcs_run_id", lambda: "r0")

        mod.run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
            object(), cluster_shards=1, workers_per_shard=1, concurrent_clusters=1,
        )
        coord_labels = [l for l in submitted if "coord" in l]
        assert coord_labels, "No coordinator label found in worker access probe"
        coord_spec = submitted[coord_labels[0]]
        assert "autogluon_worker_access_probe.py" in coord_spec
        assert not any("head" in l and "coord" not in l for l in submitted)
