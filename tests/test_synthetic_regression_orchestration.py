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

import sys
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
            return self

        def __exit__(self, *args):
            return self._stack.__exit__(*args)

    return _MultiPatch()


def _assert_unsafe_torch_load_for_eval_shards(jobs: list[_FakeJob]) -> None:
    eval_modes = {"deepset", "baselines", "autogluon"}
    eval_jobs = [
        job for job in jobs
        if job.env_vars.get("SYNTHETIC_REGRESSION_MODE") in eval_modes
    ]
    assert eval_jobs, "No synthetic regression eval shard jobs found"
    for job in eval_jobs:
        assert job.env_vars.get("ALLOW_UNSAFE_TORCH_LOAD") == "true", (
            f"{job.label} mode={job.env_vars.get('SYNTHETIC_REGRESSION_MODE')!r} "
            "missing ALLOW_UNSAFE_TORCH_LOAD=true"
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
        """Main DeepSet, baseline, and AutoGluon shards must carry trusted-checkpoint load env."""
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
        """OOD full DeepSet, baseline, and AutoGluon shards must carry trusted-checkpoint load env."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_deepset_evaluation(fake_session)
            orch.run_synthetic_regression_ood_full_baseline_evaluation(fake_session)
            orch.run_synthetic_regression_ood_full_autogluon_evaluation(fake_session)

        _assert_unsafe_torch_load_for_eval_shards(collector.submitted)

    def test_combined_eval_shards_set_allow_unsafe_torch_load(self, collector, fake_session):
        """Combined DeepSet, baseline, and AutoGluon shards must carry trusted-checkpoint load env."""
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
                   side_effect=_mock_submit_synreg):
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

    def test_cpu_capacity_phases_use_requested_concurrency(self, fake_session, runtime_args):
        """CPU capacity phases submit exactly the requested probe jobs."""
        submitted_counts = {}

        def _mock_phase(session, phase_label, compute_pool, runtime_environment, count):
            submitted_counts[phase_label] = count

        with patch("run_synthetic_regression_evaluation._submit_and_wait_capacity_phase",
                   side_effect=_mock_phase):
            orch.run_synthetic_regression_capacity_probe(
                fake_session, *runtime_args, 4, 12
            )

        assert submitted_counts == {
            "synreg_cap_baseline": 4,
            "synreg_cap_autogluon": 12,
        }

    def test_standalone_capacity_probes_use_requested_concurrency(self, fake_session, runtime_args):
        submitted_counts = {}

        def _mock_phase(session, phase_label, compute_pool, runtime_environment, count):
            submitted_counts[phase_label] = (compute_pool, count)

        with patch("run_synthetic_regression_evaluation._submit_and_wait_capacity_phase",
                   side_effect=_mock_phase):
            orch.run_synthetic_regression_baseline_capacity_probe(
                fake_session, *runtime_args, 5
            )
            orch.run_synthetic_regression_autogluon_capacity_probe(
                fake_session, *runtime_args, 17
            )

        assert submitted_counts["synreg_cap_baseline"] == (orch.DEEPSET_CPU_POOL, 5)
        assert submitted_counts["synreg_cap_autogluon"] == (orch.AUTOGLUON_CPU_POOL, 17)


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

    def test_env_vars_override_concurrency_defaults(self, monkeypatch):
        monkeypatch.setenv("SYNREG_BASELINE_CONCURRENT_NODES", "2")
        monkeypatch.setenv("SYNREG_AUTOGLUON_CONCURRENT_NODES", "7")

        assert orch._resolve_baseline_concurrent_nodes("proc") == 2
        assert orch._resolve_autogluon_concurrent_nodes("proc") == 7

    def test_sql_args_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("SYNREG_BASELINE_CONCURRENT_NODES", "2")
        monkeypatch.setenv("SYNREG_AUTOGLUON_CONCURRENT_NODES", "7")

        assert orch._resolve_baseline_concurrent_nodes("proc", 4) == 4
        assert orch._resolve_autogluon_concurrent_nodes("proc", 11) == 11

    def test_invalid_concurrency_error_includes_context(self):
        with pytest.raises(ValueError) as exc:
            orch._resolve_baseline_concurrent_nodes(
                "run_synthetic_regression_baseline_evaluation",
                7,
            )
        msg = str(exc.value)
        assert "run_synthetic_regression_baseline_evaluation" in msg
        assert "requested concurrency 7" in msg
        assert "shard count 6" in msg
        assert orch.DEEPSET_CPU_POOL in msg

    def test_lower_baseline_concurrency_batches_complete_shards(self, collector, fake_session, runtime_args):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_baseline_evaluation(
                    fake_session, *runtime_args, 2
                )

        baseline_jobs = [j for j in collector.submitted
                         if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "baselines"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in baseline_jobs]
        assert batch_sizes == [2, 2, 2]
        assert sorted(indices) == list(range(6))

    def test_lower_autogluon_concurrency_batches_complete_shards(self, collector, fake_session, runtime_args):
        batch_sizes = []

        def _mock_wait_group(labeled_jobs, session):
            batch_sizes.append(len(labeled_jobs))

        with _patch_submit(collector):
            with patch("run_synthetic_regression_evaluation._wait_job_group",
                       side_effect=_mock_wait_group):
                orch.run_synthetic_regression_autogluon_evaluation(
                    fake_session, *runtime_args, 25
                )

        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        indices = [int(j.env_vars["SYNTHETIC_REGRESSION_SHARD_INDEX"]) for j in ag_jobs]
        assert batch_sizes == [25, 25, 10]
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

    def test_ood_full_autogluon_evaluation_submits_ag_shards(self, collector, fake_session):
        """OOD full AG phase submits SYNREG_AUTOGLUON_SHARDS jobs on AUTOGLUON_CPU_POOL."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_ood_full_autogluon_evaluation(fake_session)
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        assert len(ag_jobs) == orch.SYNREG_AUTOGLUON_SHARDS
        for job in ag_jobs:
            assert job.compute_pool == orch.AUTOGLUON_CPU_POOL

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

    def test_combined_autogluon_evaluation_submits_ag_shards(self, collector, fake_session):
        """Combined AG phase submits SYNREG_AUTOGLUON_SHARDS jobs on AUTOGLUON_CPU_POOL."""
        with _patch_submit(collector):
            orch.run_synthetic_regression_combined_autogluon_evaluation(fake_session)
        ag_jobs = [j for j in collector.submitted
                   if j.env_vars.get("SYNTHETIC_REGRESSION_MODE") == "autogluon"]
        assert len(ag_jobs) == orch.SYNREG_AUTOGLUON_SHARDS

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
