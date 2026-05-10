import importlib.util
import os
import sys
import types


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_evaluation_test.py")


class _FakeJob:
    def __init__(self, status="DONE", wait_exc=None):
        self.status = status
        self._wait_exc = wait_exc

    def wait(self):
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
            return _FakeSql([("@MODEL_STAGE/scripts/runtime_probe.py",)])
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


def _load_runner(monkeypatch, submit_calls, fail_entrypoint=None):
    def fake_submit_from_stage(**kwargs):
        submit_calls.append(kwargs)
        if kwargs.get("entrypoint") == fail_entrypoint:
            return _FakeJob(status="FAILED")
        return _FakeJob()

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

    assert len(probe_jobs) == 4
    assert [c["compute_pool"] for c in probe_jobs] == [
        runner.GPU_POOL,
        runner.CPU_POOL,
        runner.CPU_POOL,
        runner.AUTOGLUON_CPU_POOL,
    ]
    assert [c["runtime_environment"] for c in probe_jobs] == [
        "benchmark-runtime",
        "benchmark-runtime",
        "prep-runtime",
        "autogluon-runtime",
    ]
    assert [c["env_vars"]["REQUIRED_IMPORTS"] for c in probe_jobs] == [
        runner.BENCHMARK_REQUIRED_IMPORTS,
        runner.BENCHMARK_REQUIRED_IMPORTS,
        runner.PREP_REQUIRED_IMPORTS,
        runner.AUTOGLUON_REQUIRED_IMPORTS,
    ]
    assert "sklearn" in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "xgboost" not in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "lightgbm" not in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "catboost" not in runner.AUTOGLUON_REQUIRED_IMPORTS.split(",")
    assert "xgboost" in runner.BENCHMARK_REQUIRED_IMPORTS.split(",")
    assert "lightgbm" in runner.BENCHMARK_REQUIRED_IMPORTS.split(",")
    assert "catboost" in runner.BENCHMARK_REQUIRED_IMPORTS.split(",")
    assert [c["env_vars"]["REQUIRE_CUDA"] for c in probe_jobs] == [
        "true",
        "false",
        "false",
        "false",
    ]

    assert len(prep_jobs) == 1
    assert prep_jobs[0]["external_access_integrations"] == ["BENCHMARK_EXTERNAL_ACCESS"]
    assert prep_jobs[0]["runtime_environment"] == "prep-runtime"
    assert prep_jobs[0]["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "prep-runtime"

    assert len(synthetic_jobs) == 1
    assert synthetic_jobs[0]["compute_pool"] == runner.GPU_POOL
    assert synthetic_jobs[0]["target_instances"] == 1
    assert synthetic_jobs[0]["runtime_environment"] == "benchmark-runtime"
    assert synthetic_jobs[0]["env_vars"]["EVAL_RUNTIME_ENVIRONMENT"] == "benchmark-runtime"

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

    assert all("runtime_environment" in c for c in submit_calls)
    assert all("pip_requirements" not in c for c in submit_calls)
    assert all(
        "external_access_integrations" not in c
        for c in baseline_jobs + deepset_jobs + autogluon_jobs
    )


def test_evaluation_orchestrator_validates_manifest_even_when_manifest_exists(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)

    runner.run_evaluation_pipeline(_FakeSession(manifest_exists=True))

    probe_jobs = [c for c in submit_calls if c["entrypoint"] == "runtime_probe.py"]
    assert len(probe_jobs) == 4
    assert [c["runtime_environment"] for c in probe_jobs] == [
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
    assert all("pip_requirements" not in c for c in submit_calls)


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

    assert [c["entrypoint"] for c in submit_calls] == ["runtime_probe.py"] * 4


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
