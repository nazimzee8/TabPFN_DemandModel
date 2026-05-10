import importlib.util
import os
import sys
import types


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_evaluation_test.py")


class _FakeJob:
    status = "DONE"

    def wait(self):
        return None


class _FakeSql:
    def __init__(self, rows):
        self._rows = rows

    def collect(self):
        return self._rows


class _FakeSession:
    def __init__(self, manifest_exists=False):
        self.manifest_exists = manifest_exists

    def sql(self, query):
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


def _load_runner(monkeypatch, submit_calls):
    def fake_submit_from_stage(**kwargs):
        submit_calls.append(kwargs)
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

    assert len(prep_jobs) == 1
    assert prep_jobs[0]["external_access_integrations"] == ["BENCHMARK_EXTERNAL_ACCESS"]
    assert prep_jobs[0]["runtime_environment"] == "prep-runtime"

    assert len(synthetic_jobs) == 1
    assert synthetic_jobs[0]["compute_pool"] == runner.GPU_POOL
    assert synthetic_jobs[0]["target_instances"] == 1
    assert synthetic_jobs[0]["runtime_environment"] == "benchmark-runtime"

    assert len(baseline_jobs) == 3
    assert all(c["env_vars"]["BENCHMARK_METHODS"] == ",".join(runner.BASELINE_METHODS) for c in baseline_jobs)
    assert all("BENCHMARK_METHOD" not in c["env_vars"] for c in baseline_jobs)
    assert all(c["compute_pool"] == runner.CPU_POOL for c in baseline_jobs)
    assert all(c["target_instances"] == 1 for c in baseline_jobs)
    assert all(c["runtime_environment"] == "benchmark-runtime" for c in baseline_jobs)

    assert len(deepset_jobs) == 10
    assert all(c["runtime_environment"] == "benchmark-runtime" for c in deepset_jobs)
    assert all(c["env_vars"]["MC_K"] == "8" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_CONTEXT_SIZE"] == "200" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES"] == "5" for c in deepset_jobs)
    assert all(c["env_vars"]["BENCHMARK_DEEPSET_TEST_BATCH_SIZE"] == "128" for c in deepset_jobs)
    assert len(autogluon_jobs) == 30
    assert all(c["runtime_environment"] == "autogluon-runtime" for c in autogluon_jobs)
    assert len(aggregate_jobs) == 1
    assert aggregate_jobs[0]["runtime_environment"] == "benchmark-runtime"

    assert all("runtime_environment" in c for c in submit_calls)
    assert all("pip_requirements" not in c for c in submit_calls)
    assert all(
        "external_access_integrations" not in c
        for c in baseline_jobs + deepset_jobs + autogluon_jobs
    )


def test_evaluation_orchestrator_skips_prep_when_manifest_exists(monkeypatch):
    _set_runtime_envs(monkeypatch)
    submit_calls = []
    runner = _load_runner(monkeypatch, submit_calls)

    runner.run_evaluation_pipeline(_FakeSession(manifest_exists=True))

    assert not [
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
        assert "BENCHMARK_RUNTIME_ENVIRONMENT is required" in str(exc)
    else:
        raise AssertionError("missing runtime environment should fail fast")

    assert submit_calls == []
