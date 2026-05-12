import importlib.util
import os
import re
import sys
import types


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_evaluation_test.py")
SQL_DDL_PATHS = [
    os.path.join(REPO_ROOT, "sql", "run_training_job.sql"),
    os.path.join(REPO_ROOT, "preload.sql"),
    os.path.join(REPO_ROOT, "docs", "Snowflake_Training.md"),
]


def _prepare_benchmark_datasets_ddl(path):
    with open(path, encoding="utf-8") as handle:
        sql_text = handle.read()
    match = re.search(
        r"CREATE OR REPLACE PROCEDURE prepare_benchmark_datasets\(\).*?"
        r"HANDLER = 'prepare_benchmark_datasets\.prepare_datasets';",
        sql_text,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing prepare_benchmark_datasets DDL in {path}"
    return match.group(0)


def test_prepare_benchmark_datasets_ddl_has_openml_and_eai():
    for path in SQL_DDL_PATHS:
        ddl = _prepare_benchmark_datasets_ddl(path)
        assert (
            "ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository"
        ) in ddl
        assert (
            "PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python', "
            "'openml==0.15.1')"
        ) in ddl
        assert "EXTERNAL_ACCESS_INTEGRATIONS = (BENCHMARK_EXTERNAL_ACCESS)" in ddl
        assert "HANDLER = 'prepare_benchmark_datasets.prepare_datasets';" in ddl


class _FakeJob:
    def __init__(self, status="DONE", wait_exc=None, on_wait=None):
        self.status = status
        self._wait_exc = wait_exc
        self._on_wait = on_wait

    def wait(self):
        if self._on_wait is not None:
            self._on_wait()
        if self._wait_exc is not None:
            raise self._wait_exc
        return None

    def get_logs(self):
        return "fake logs"


class _FakeSql:
    def __init__(self, rows):
        self._rows = rows

    def collect(self):
        return self._rows


class _FakeSession:
    def __init__(self, manifest_exists=False, compute_pool_states=None):
        self.manifest_exists = manifest_exists
        self.compute_pool_states = compute_pool_states or {
            "DEEPSET_GPU_POOL": "ACTIVE",
            "DEEPSET_CPU_POOL": "ACTIVE",
            "AUTOGLUON_CPU_POOL": "ACTIVE",
        }
        self.sql_queries = []

    def sql(self, query):
        self.sql_queries.append(query)
        if query.startswith("SHOW COMPUTE POOLS LIKE"):
            pool_name = query.split("'")[1]
            state = self.compute_pool_states.get(pool_name)
            if state is None:
                return _FakeSql([])
            return _FakeSql([{"name": pool_name, "state": state}])
        if query.startswith("ALTER COMPUTE POOL"):
            pool_name = query.split()[3]
            self.compute_pool_states[pool_name] = "ACTIVE"
            return _FakeSql([])
        if "scripts" in query:
            return _FakeSql([
                ("@MODEL_STAGE/scripts/runtime_probe.py",),
                ("@MODEL_STAGE/scripts/capacity_probe.py",),
            ])
        if "checkpoints" in query:
            return _FakeSql([("@MODEL_STAGE/checkpoints/best.pt",)])
        if "benchmark_prepared" in query and self.manifest_exists:
            return _FakeSql(
                [
                    (
                        "@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json",
                    )
                ]
            )
        return _FakeSql([])


def _load_runner(monkeypatch, submit_calls, fail_entrypoint=None,
                 event_log=None, all_events=None):
    def fake_submit_from_stage(**kwargs):
        submit_calls.append(kwargs)
        ep = kwargs.get("entrypoint", "")
        env = kwargs.get("env_vars", {})
        on_wait_fns = []

        # Legacy: 2-tuple tracking for runtime probes only
        if event_log is not None and ep == "runtime_probe.py":
            label = env["RUNTIME_PROBE_LABEL"]
            event_log.append(("submit", label))
            def _probe_wait(lbl=label):
                event_log.append(("wait", lbl))
            on_wait_fns.append(_probe_wait)

        # New: 3-tuple tracking for all jobs
        if all_events is not None:
            all_events.append(("submit", ep, dict(env)))
            def _all_wait(ep_=ep, env_=dict(env)):
                all_events.append(("wait", ep_, env_))
            on_wait_fns.append(_all_wait)

        combined_on_wait = (lambda fns=on_wait_fns: [f() for f in fns]) if on_wait_fns else None
        if kwargs.get("entrypoint") == fail_entrypoint:
            return _FakeJob(status="FAILED", on_wait=combined_on_wait)
        return _FakeJob(on_wait=combined_on_wait)

    snowflake = types.ModuleType("snowflake")
    snowflake_ml = types.ModuleType("snowflake.ml")
    snowflake_jobs = types.ModuleType("snowflake.ml.jobs")
    snowflake_jobs.submit_from_stage = fake_submit_from_stage
    monkeypatch.setitem(sys.modules, "snowflake", snowflake)
    monkeypatch.setitem(sys.modules, "snowflake.ml", snowflake_ml)
    monkeypatch.setitem(sys.modules, "snowflake.ml.jobs", snowflake_jobs)

    spec = importlib.util.spec_from_file_location("run_evaluation_test_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_runtime_envs(monkeypatch):
    monkeypatch.setenv("PREP_RUNTIME_ENVIRONMENT", "prep-runtime")
    monkeypatch.setenv("BENCHMARK_RUNTIME_ENVIRONMENT", "benchmark-runtime")
    monkeypatch.setenv("AUTOGLUON_RUNTIME_ENVIRONMENT", "autogluon-runtime")


def test_evaluation_orchestrator_submits_combined_baseline_shards(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)

    runner.run_evaluation_pipeline(_FakeSession(manifest_exists=False))

    prep_jobs = [c for c in submit_calls if c["entrypoint"] == "prepare_benchmark_datasets.py"]
    probe_jobs = [c for c in submit_calls if c["entrypoint"] == "runtime_probe.py"]
    eval_jobs = [c for c in submit_calls if c["entrypoint"] == "evaluate.py"]
    synthetic_jobs = [
        c for c in eval_jobs
        if c["env_vars"].get("EVAL_MODE") == "synthetic"
    ]
    aggregate_jobs = [
        c for c in eval_jobs
        if c["env_vars"].get("EVAL_MODE") == "aggregate"
    ]
    baseline_jobs = [
        c for c in eval_jobs
        if c["env_vars"].get("BENCHMARK_METHODS") == ",".join(runner.BASELINE_METHODS)
    ]
    deepset_jobs = [
        c for c in eval_jobs
        if c["env_vars"].get("BENCHMARK_METHOD") == "DeepSetModel-MC"
    ]
    autogluon_jobs = [
        c for c in eval_jobs
        if c["env_vars"].get("BENCHMARK_METHOD") == runner.AUTOGLUON_METHOD
    ]

    assert len(probe_jobs) == 5
    assert [c["compute_pool"] for c in probe_jobs] == [
        runner.GPU_POOL,
        runner.CPU_POOL,
        runner.CPU_POOL,
        runner.CPU_POOL,
        runner.AUTOGLUON_CPU_POOL,
    ]
    assert [c["runtime_environment"] for c in probe_jobs] == [
        "benchmark-runtime",
        "benchmark-runtime",
        "benchmark-runtime",
        "prep-runtime",
        "autogluon-runtime",
    ]
    assert [c["env_vars"]["REQUIRED_IMPORTS"] for c in probe_jobs] == [
        runner.BENCHMARK_REQUIRED_IMPORTS,
        runner.BENCHMARK_REQUIRED_IMPORTS,
        runner.BASELINE_REQUIRED_IMPORTS,
        runner.PREP_REQUIRED_IMPORTS,
        runner.AUTOGLUON_REQUIRED_IMPORTS,
    ]
    assert "sklearn" in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "xgboost" not in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "lightgbm" not in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "catboost" not in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "xgboost" in runner.BASELINE_REQUIRED_IMPORTS.split(",")
    assert "lightgbm" in runner.BASELINE_REQUIRED_IMPORTS.split(",")
    assert "catboost" in runner.BASELINE_REQUIRED_IMPORTS.split(",")
    assert [c["env_vars"]["REQUIRE_CUDA"] for c in probe_jobs] == [
        "true",
        "false",
        "false",
        "false",
        "false",
    ]

    assert len(prep_jobs) == 1
    assert prep_jobs[0]["external_access_integrations"] == [
        runner.BENCHMARK_EXTERNAL_ACCESS_EAI,
        runner.PYPI_EAI,
    ]
    assert prep_jobs[0]["pip_requirements"] == list(runner.PREP_EXTRA_PIP_REQUIREMENTS)
    assert prep_jobs[0]["runtime_environment"] == "prep-runtime"
    assert prep_jobs[0]["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "prep-runtime"

    assert len(synthetic_jobs) == 1
    assert synthetic_jobs[0]["compute_pool"] == runner.GPU_POOL
    assert synthetic_jobs[0]["target_instances"] == 1
    assert synthetic_jobs[0]["runtime_environment"] == "benchmark-runtime"
    assert synthetic_jobs[0]["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "benchmark-runtime"
    assert synthetic_jobs[0]["env_vars"]["ALLOW_UNSAFE_TORCH_LOAD"] == runner.ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS

    assert len(baseline_jobs) == 3
    assert all(c["env_vars"]["BENCHMARK_METHODS"] == ",".join(runner.BASELINE_METHODS) for c in baseline_jobs)
    assert all("BENCHMARK_METHOD" not in c["env_vars"] for c in baseline_jobs)
    assert all(c["compute_pool"] == runner.CPU_POOL for c in baseline_jobs)
    assert all(c["target_instances"] == 1 for c in baseline_jobs)
    assert all(c["runtime_environment"] == "benchmark-runtime" for c in baseline_jobs)
    assert all(c["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "benchmark-runtime" for c in baseline_jobs)

    assert len(deepset_jobs) == 10
    assert all(c["runtime_environment"] == "benchmark-runtime" for c in deepset_jobs)
    assert all(c["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "benchmark-runtime" for c in deepset_jobs)
    assert all(c["env_vars"]["MC_K"] == "8" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_CONTEXT_SIZE"] == "200" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES"] == "5" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_TEST_BATCH_SIZE"] == "128" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_FEATURE_SELECTOR"] == "train_f_regression" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_REQUIRE_CUDA"] == "true" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES"] == "268435456" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_GPU_MEMORY_SAFETY_FACTOR"] == "4.0" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_MAX_GPU_MEMORY_FRACTION"] == "0.80" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_EMPTY_CACHE"] == "true" for c in deepset_jobs)
    assert len(autogluon_jobs) == 30
    assert all(c["runtime_environment"] == "autogluon-runtime" for c in autogluon_jobs)
    assert all(c["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "autogluon-runtime" for c in autogluon_jobs)
    assert len(aggregate_jobs) == 1
    assert aggregate_jobs[0]["runtime_environment"] == "benchmark-runtime"
    assert aggregate_jobs[0]["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "benchmark-runtime"
    assert all(
        c["env_vars"]["ALLOW_UNSAFE_TORCH_LOAD"]
        == runner.ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS
        for c in eval_jobs
    )
    assert runner.ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS == "true", (
        "ALLOW_UNSAFE_TORCH_LOAD is currently 'true' as a temporary escape hatch for the "
        "legacy best.pt checkpoint. Revert to 'false' after migrating best.pt to "
        "checkpoint_format_version=2 via scripts/migrate_checkpoint.py."
    )

    assert all("runtime_environment" in c for c in submit_calls)

    assert runner.PREP_EXTRA_PIP_REQUIREMENTS == [f"openml=={runner.OPENML_VERSION}"]

    # Probes 2, 3, and 4 carry pip_requirements; probes 0 and 1 do not.
    assert probe_jobs[2].get("pip_requirements") == list(runner.BASELINE_EXTRA_PIP_REQUIREMENTS)
    assert probe_jobs[3].get("pip_requirements") == list(runner.PREP_EXTRA_PIP_REQUIREMENTS)
    assert probe_jobs[4].get("pip_requirements") == list(runner.AUTOGLUON_EXTRA_PIP_REQUIREMENTS)
    assert all("pip_requirements" not in c for c in probe_jobs[:2])

    # Baseline evaluation shard jobs must carry pip_requirements.
    assert all(
        c.get("pip_requirements") == list(runner.BASELINE_EXTRA_PIP_REQUIREMENTS)
        for c in baseline_jobs
    )

    # AutoGluon shard jobs carry pip_requirements and EAI.
    assert all(
        c.get("pip_requirements") == list(runner.AUTOGLUON_EXTRA_PIP_REQUIREMENTS)
        for c in autogluon_jobs
    )
    assert all(
        c.get("external_access_integrations") == [runner.PYPI_EAI]
        for c in autogluon_jobs
    )

    # Other eval jobs must NOT carry pip_requirements.
    assert all(
        "pip_requirements" not in c
        for c in deepset_jobs + synthetic_jobs + aggregate_jobs
    )

    # Probes 2, 3, and 4 carry EAI; probes 0 and 1 do not.
    assert probe_jobs[2].get("external_access_integrations") == [runner.PYPI_EAI]
    assert probe_jobs[3].get("external_access_integrations") == [runner.PYPI_EAI]
    assert probe_jobs[4].get("external_access_integrations") == [runner.PYPI_EAI]
    assert all("external_access_integrations" not in c for c in probe_jobs[:2])
    assert all(
        c.get("external_access_integrations") == [runner.PYPI_EAI]
        for c in baseline_jobs
    )
    assert all("external_access_integrations" not in c for c in deepset_jobs)


def test_evaluation_orchestrator_validates_manifest_even_when_manifest_exists(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)

    runner.run_evaluation_pipeline(_FakeSession(manifest_exists=True))

    probe_jobs = [c for c in submit_calls if c["entrypoint"] == "runtime_probe.py"]
    assert len(probe_jobs) == 5
    assert [c["runtime_environment"] for c in probe_jobs] == [
        "benchmark-runtime",
        "benchmark-runtime",
        "benchmark-runtime",
        "prep-runtime",
        "autogluon-runtime",
    ]
    assert [
        c for c in submit_calls
        if c["entrypoint"] == "prepare_benchmark_datasets.py"
    ]
    assert [
        c for c in submit_calls
        if c["entrypoint"] == "evaluate.py"
        and c["env_vars"].get("EVAL_MODE") == "synthetic"
        and c["target_instances"] == 1
        and c["compute_pool"] == runner.GPU_POOL
    ]

    # Probes 2, 3, and 4 carry pip_requirements; probes 0 and 1 do not.
    assert probe_jobs[2].get("pip_requirements") == list(runner.BASELINE_EXTRA_PIP_REQUIREMENTS)
    assert probe_jobs[3].get("pip_requirements") == list(runner.PREP_EXTRA_PIP_REQUIREMENTS)
    assert probe_jobs[4].get("pip_requirements") == list(runner.AUTOGLUON_EXTRA_PIP_REQUIREMENTS)
    assert all("pip_requirements" not in c for c in probe_jobs[:2])
    prep_jobs = [
        c for c in submit_calls
        if c["entrypoint"] == "prepare_benchmark_datasets.py"
    ]
    assert prep_jobs[0].get("pip_requirements") == list(runner.PREP_EXTRA_PIP_REQUIREMENTS)
    assert prep_jobs[0].get("external_access_integrations") == [
        runner.BENCHMARK_EXTERNAL_ACCESS_EAI,
        runner.PYPI_EAI,
    ]
    baseline_eval_jobs = [
        c for c in submit_calls
        if c.get("entrypoint") == "evaluate.py"
        and c["env_vars"].get("BENCHMARK_METHODS") == ",".join(runner.BASELINE_METHODS)
    ]
    assert all(
        c.get("pip_requirements") == list(runner.BASELINE_EXTRA_PIP_REQUIREMENTS)
        for c in baseline_eval_jobs
    )
    autogluon_eval_jobs = [
        c for c in submit_calls
        if c.get("entrypoint") == "evaluate.py"
        and c["env_vars"].get("BENCHMARK_METHOD") == runner.AUTOGLUON_METHOD
    ]
    assert all(
        c.get("pip_requirements") == list(runner.AUTOGLUON_EXTRA_PIP_REQUIREMENTS)
        for c in autogluon_eval_jobs
    )
    assert all(
        c.get("external_access_integrations") == [runner.PYPI_EAI]
        for c in autogluon_eval_jobs
    )
    other_eval_jobs = [
        c for c in submit_calls
        if c.get("entrypoint") == "evaluate.py"
        and c["env_vars"].get("BENCHMARK_METHODS") != ",".join(runner.BASELINE_METHODS)
        and c["env_vars"].get("BENCHMARK_METHOD") != runner.AUTOGLUON_METHOD
    ]
    assert all("pip_requirements" not in c for c in other_eval_jobs)
    assert probe_jobs[2].get("external_access_integrations") == [runner.PYPI_EAI]
    assert probe_jobs[3].get("external_access_integrations") == [runner.PYPI_EAI]
    assert probe_jobs[4].get("external_access_integrations") == [runner.PYPI_EAI]
    assert all("external_access_integrations" not in c for c in probe_jobs[:2])
    assert all(
        c.get("external_access_integrations") == [runner.PYPI_EAI]
        for c in baseline_eval_jobs
    )
    assert all("external_access_integrations" not in c for c in other_eval_jobs)


def test_evaluation_orchestrator_requires_runtime_env(monkeypatch):
    monkeypatch.delenv("PREP_RUNTIME_ENVIRONMENT", raising=False)
    monkeypatch.delenv("BENCHMARK_RUNTIME_ENVIRONMENT", raising=False)
    monkeypatch.delenv("AUTOGLUON_RUNTIME_ENVIRONMENT", raising=False)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)

    try:
        runner.run_evaluation_pipeline(_FakeSession(manifest_exists=True))
    except RuntimeError as exc:
        assert "PREP_RUNTIME_ENVIRONMENT is required" in str(exc)
    else:
        raise AssertionError("missing runtime environment should fail fast")

    assert submit_calls == []


def test_evaluation_orchestrator_probe_failure_stops_before_shards(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls, fail_entrypoint="runtime_probe.py")

    try:
        runner.run_evaluation_pipeline(_FakeSession(manifest_exists=False))
    except RuntimeError as exc:
        assert "Runtime preflight probe" in str(exc)
    else:
        raise AssertionError("runtime probe failure should fail before benchmark submission")

    assert [c["entrypoint"] for c in submit_calls] == ["runtime_probe.py"]


def test_evaluation_orchestrator_runtime_probes_are_serial(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    event_log = []
    runner = _load_runner(monkeypatch, submit_calls, event_log=event_log)

    runner.run_evaluation_runtime_probes(_FakeSession(manifest_exists=False))

    expected_labels = [
        "benchmark GPU runtime",
        "benchmark aggregate CPU runtime",
        "CPU baseline runtime with CatBoost pip dependency",
        "prep CPU runtime",
        "AutoGluon CPU runtime",
    ]
    assert [c["entrypoint"] for c in submit_calls] == ["runtime_probe.py"] * 5
    assert event_log == [
        event
        for label in expected_labels
        for event in (("submit", label), ("wait", label))
    ]


def test_evaluation_orchestrator_missing_runtime_probe_fails_before_submit(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)

    class MissingProbeSession(_FakeSession):
        def sql(self, query):
            if "scripts" in query:
                return _FakeSql([])
            return super().sql(query)

    try:
        runner.run_evaluation_pipeline(MissingProbeSession())
    except FileNotFoundError as exc:
        assert "runtime_probe.py is required" in str(exc)
    else:
        raise AssertionError("missing runtime_probe.py should fail before submit")

    assert submit_calls == []


def test_evaluation_orchestrator_resumes_suspended_compute_pool(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    session = _FakeSession(compute_pool_states={
        "DEEPSET_GPU_POOL": "ACTIVE",
        "DEEPSET_CPU_POOL": "SUSPENDED",
        "AUTOGLUON_CPU_POOL": "ACTIVE",
    })
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    runner.run_evaluation_pipeline(session)

    assert "ALTER COMPUTE POOL DEEPSET_CPU_POOL RESUME" in session.sql_queries
    assert submit_calls


def test_evaluation_orchestrator_missing_compute_pool_fails_before_submit(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    session = _FakeSession(compute_pool_states={
        "DEEPSET_GPU_POOL": "ACTIVE",
        "DEEPSET_CPU_POOL": None,
        "AUTOGLUON_CPU_POOL": "ACTIVE",
    })

    try:
        runner.run_evaluation_pipeline(session)
    except RuntimeError as exc:
        assert "DEEPSET_CPU_POOL does not exist" in str(exc)
    else:
        raise AssertionError("missing compute pool should fail before submit")

    assert submit_calls == []


def test_evaluation_orchestrator_failed_compute_pool_fails_before_submit(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    session = _FakeSession(compute_pool_states={
        "DEEPSET_GPU_POOL": "ACTIVE",
        "DEEPSET_CPU_POOL": "FAILED",
        "AUTOGLUON_CPU_POOL": "ACTIVE",
    })

    try:
        runner.run_evaluation_pipeline(session)
    except RuntimeError as exc:
        assert "DEEPSET_CPU_POOL is unusable" in str(exc)
    else:
        raise AssertionError("failed compute pool should fail before submit")

    assert submit_calls == []


def _event_phase(ep, env):
    if ep == "evaluate.py":
        if env.get("BENCHMARK_METHOD") == "DeepSetModel-MC": return "deepset"
        if env.get("BENCHMARK_METHODS"):                      return "baseline"
        if env.get("BENCHMARK_METHOD") == "AutoGluon":        return "autogluon"
        if env.get("EVAL_MODE") == "aggregate":               return "aggregate"
        if env.get("EVAL_MODE") == "synthetic":               return "synthetic"
    return ep


def test_evaluation_pipeline_phases_are_gated(monkeypatch):
    """DeepSet must fully finish before CPU baselines start; baselines before AutoGluon."""
    _set_runtime_envs(monkeypatch)
    all_events, submit_calls = [], []
    runner = _load_runner(monkeypatch, submit_calls, all_events=all_events)
    runner.run_evaluation_pipeline(_FakeSession())

    def phase_indices(phase_name, kind):
        return [i for i, (k, ep, env) in enumerate(all_events)
                if k == kind and _event_phase(ep, env) == phase_name]

    assert max(phase_indices("deepset", "wait")) < min(phase_indices("baseline", "submit"))
    assert max(phase_indices("baseline", "wait")) < min(phase_indices("autogluon", "submit"))
    assert max(phase_indices("autogluon", "wait")) < min(phase_indices("aggregate", "submit"))


def test_evaluation_pipeline_autogluon_batched(monkeypatch):
    """AutoGluon must be submitted in batches of at most AUTOGLUON_MAX_CONCURRENT_SHARDS."""
    _set_runtime_envs(monkeypatch)
    all_events, submit_calls = [], []
    runner = _load_runner(monkeypatch, submit_calls, all_events=all_events)
    runner.run_evaluation_pipeline(_FakeSession())

    ag_events = [(k, ep, env) for k, ep, env in all_events
                 if _event_phase(ep, env) == "autogluon"]
    assert len(ag_events) == 60  # 30 submits + 30 waits
    batch_size = runner.AUTOGLUON_MAX_CONCURRENT_SHARDS  # 30
    # Event pattern: [batch_size submits, batch_size waits] repeated
    for batch_start in range(0, len(ag_events), batch_size * 2):
        chunk = ag_events[batch_start : batch_start + batch_size * 2]
        submits_in_chunk = [e for e in chunk if e[0] == "submit"]
        waits_in_chunk   = [e for e in chunk if e[0] == "wait"]
        assert len(submits_in_chunk) == batch_size
        assert len(waits_in_chunk) == batch_size
        submit_pos = [i for i, e in enumerate(chunk) if e[0] == "submit"]
        wait_pos   = [i for i, e in enumerate(chunk) if e[0] == "wait"]
        assert max(submit_pos) < min(wait_pos), "All submits in batch must precede waits"


def test_evaluation_capacity_probe_phases_do_not_overlap(monkeypatch):
    """Capacity probe phases must be non-overlapping: GPU done before CPU, CPU done before AG."""
    _set_runtime_envs(monkeypatch)
    all_events, submit_calls = [], []
    runner = _load_runner(monkeypatch, submit_calls, all_events=all_events)
    runner.run_evaluation_capacity_probe(_FakeSession())

    cp_events = [(i, k, env) for i, (k, ep, env) in enumerate(all_events)
                 if ep == "capacity_probe.py"]
    expected_jobs = (
        runner.GPU_BENCHMARK_SHARDS
        + runner.CPU_BASELINE_BENCHMARK_SHARDS
        + runner.AUTOGLUON_MAX_CONCURRENT_SHARDS
    )
    assert len(cp_events) == expected_jobs * 2

    def cp_phase_indices(phase_substr, kind):
        return [i for i, k, env in cp_events
                if k == kind and phase_substr in env.get("CAPACITY_PROBE_LABEL", "")]

    gpu_waits     = cp_phase_indices("GPU",      "wait")
    cpu_submits   = cp_phase_indices("CPU",      "submit")
    cpu_waits     = cp_phase_indices("CPU",      "wait")
    ag_submits    = cp_phase_indices("AutoGluon","submit")

    assert max(gpu_waits)   < min(cpu_submits), "GPU phase must finish before CPU phase starts"
    assert max(cpu_waits)   < min(ag_submits),  "CPU phase must finish before AG phase starts"


def test_run_evaluation_prep_submits_only_prep(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    runner.run_evaluation_prep(_FakeSession())

    ep_list = [c["entrypoint"] for c in submit_calls]
    assert ep_list == ["prepare_benchmark_datasets.py"]


def test_run_deepset_evaluation_submits_synthetic_and_shards(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    runner.run_deepset_evaluation(_FakeSession())

    eval_jobs = [c for c in submit_calls if c["entrypoint"] == "evaluate.py"]
    synthetic = [c for c in eval_jobs if c["env_vars"].get("EVAL_MODE") == "synthetic"]
    deepset   = [c for c in eval_jobs if c["env_vars"].get("BENCHMARK_METHOD") == "DeepSetModel-MC"]
    other     = [c for c in eval_jobs if c not in synthetic and c not in deepset]

    assert len(synthetic) == 1
    assert len(deepset) == runner.GPU_BENCHMARK_SHARDS  # 10
    assert len(other) == 0
    # No prep, baseline, autogluon, or aggregate
    assert not any(c["entrypoint"] == "prepare_benchmark_datasets.py" for c in submit_calls)
    assert not any(c["env_vars"].get("BENCHMARK_METHODS") for c in eval_jobs)
    assert not any(c["env_vars"].get("BENCHMARK_METHOD") == runner.AUTOGLUON_METHOD for c in eval_jobs)
    assert not any(c["env_vars"].get("EVAL_MODE") == "aggregate" for c in eval_jobs)


def test_run_baseline_evaluation_submits_only_baseline_shards(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    runner.run_baseline_evaluation(_FakeSession())

    eval_jobs = [c for c in submit_calls if c["entrypoint"] == "evaluate.py"]
    assert len(eval_jobs) == runner.CPU_BASELINE_BENCHMARK_SHARDS  # 3
    baseline_methods = ",".join(runner.BASELINE_METHODS)
    assert all(c["env_vars"].get("BENCHMARK_METHODS") == baseline_methods for c in eval_jobs)
    assert all(c["compute_pool"] == runner.CPU_POOL for c in eval_jobs)
    assert all(c.get("pip_requirements") == list(runner.BASELINE_EXTRA_PIP_REQUIREMENTS) for c in eval_jobs)
    assert all(c.get("external_access_integrations") == [runner.PYPI_EAI] for c in eval_jobs)
    assert not any(c["entrypoint"] == "prepare_benchmark_datasets.py" for c in submit_calls)


def test_run_autogluon_evaluation_batched(monkeypatch):
    _set_runtime_envs(monkeypatch)
    all_events, submit_calls = [], []
    runner = _load_runner(monkeypatch, submit_calls, all_events=all_events)
    runner.run_autogluon_evaluation(_FakeSession())

    eval_jobs = [c for c in submit_calls if c["entrypoint"] == "evaluate.py"]
    assert len(eval_jobs) == runner.AUTOGLUON_BENCHMARK_SHARDS  # 30
    assert all(c["env_vars"].get("BENCHMARK_METHOD") == runner.AUTOGLUON_METHOD for c in eval_jobs)
    # Verify batching via events
    ag_events = [(k, ep, env) for k, ep, env in all_events if _event_phase(ep, env) == "autogluon"]
    assert len(ag_events) == 60  # 30 submits + 30 waits
    batch_size = runner.AUTOGLUON_MAX_CONCURRENT_SHARDS
    for batch_start in range(0, len(ag_events), batch_size * 2):
        chunk = ag_events[batch_start : batch_start + batch_size * 2]
        submit_pos = [i for i, e in enumerate(chunk) if e[0] == "submit"]
        wait_pos   = [i for i, e in enumerate(chunk) if e[0] == "wait"]
        assert max(submit_pos) < min(wait_pos)


def test_run_evaluation_aggregation_submits_only_aggregate(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    runner.run_evaluation_aggregation(_FakeSession())

    eval_jobs = [c for c in submit_calls if c["entrypoint"] == "evaluate.py"]
    assert len(eval_jobs) == 1
    assert eval_jobs[0]["env_vars"].get("EVAL_MODE") == "aggregate"
    assert not any(c["entrypoint"] == "prepare_benchmark_datasets.py" for c in submit_calls)


def test_evaluation_capacity_probe_job_properties(monkeypatch):
    """Capacity probe jobs must use capacity_probe.py, no pip requirements, no EAIs."""
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)
    runner.run_evaluation_capacity_probe(_FakeSession())

    cp_jobs = [c for c in submit_calls if c["entrypoint"] == "capacity_probe.py"]
    assert len(cp_jobs) == 43  # 10 + 3 + 30
    assert all("pip_requirements" not in c for c in cp_jobs)
    assert all("external_access_integrations" not in c for c in cp_jobs)
    # Phase sizes
    gpu_jobs = [c for c in cp_jobs if c["compute_pool"] == runner.GPU_POOL]
    cpu_jobs = [c for c in cp_jobs if c["compute_pool"] == runner.CPU_POOL]
    ag_jobs  = [c for c in cp_jobs if c["compute_pool"] == runner.AUTOGLUON_CPU_POOL]
    assert len(gpu_jobs) == runner.GPU_BENCHMARK_SHARDS           # 10
    assert len(cpu_jobs) == runner.CPU_BASELINE_BENCHMARK_SHARDS  # 3
    assert len(ag_jobs)  == runner.AUTOGLUON_MAX_CONCURRENT_SHARDS # 30
