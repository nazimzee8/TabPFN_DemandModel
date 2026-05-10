"""
Orchestrator for the DeepSet evaluation pipeline.

Handler for the run_evaluation_pipeline() Snowpark stored procedure.
The Snowpark session is injected automatically by the stored procedure framework.

Parallelism model:
  - Synthetic eval: single-process, 1 GPU node.
  - DeepSet benchmark: GPU_BENCHMARK_SHARDS independent single-node GPU dataset shards.
  - CPU baselines: CPU_BASELINE_BENCHMARK_SHARDS combined single-node CPU dataset shards.
  - AutoGluon: AUTOGLUON_BENCHMARK_SHARDS independent single-node CPU dataset shards.
  - Aggregate: single-process, 1 CPU node.

submit_from_stage(target_instances=N) does NOT inject RANK, WORLD_SIZE, MASTER_ADDR,
or MASTER_PORT. CPU benchmarks use BENCHMARK_NUM_SHARDS + BENCHMARK_SHARD_INDEX
instead of dist.init_process_group().
"""

import os

from snowflake.ml.jobs import submit_from_stage

GPU_POOL           = "DEEPSET_GPU_POOL"
CPU_POOL           = "DEEPSET_CPU_POOL"
AUTOGLUON_CPU_POOL = "AUTOGLUON_CPU_POOL"
MODEL_STAGE        = "@MODEL_STAGE"
SCRIPTS_STAGE      = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
EVAL_RESULTS_STAGE = "@EVALUATION_RESULTS_STAGE"

BENCHMARK_PREPARED_STAGE      = "@META_DATASET_STAGE/benchmark_prepared/"
BENCHMARK_MANIFEST_STAGE_PATH = f"{BENCHMARK_PREPARED_STAGE}benchmark_manifest.json"

# Shard counts â€” match compute pool MAX_NODES capacities.
GPU_BENCHMARK_SHARDS          = 10   # DeepSet benchmark: 10 GPU shard jobs (DEEPSET_GPU_POOL MAX_NODES=10)
CPU_BASELINE_BENCHMARK_SHARDS = 3    # Combined baselines: 3 CPU shard jobs (DEEPSET_CPU_POOL MAX_NODES=3)
AUTOGLUON_BENCHMARK_SHARDS    = 30   # AutoGluon: 30 CPU shard jobs (AUTOGLUON_CPU_POOL MAX_NODES=30)

# Benchmark shards own datasets, not individual (seed, dataset) pairs. Each
# owned dataset is loaded, evaluated across all configured seeds, then released.

BASELINE_METHODS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "KNN",
    "LinearRegression",
    "Ridge",
    "SVR",
    "MLP",
]
AUTOGLUON_METHOD = "AutoGluon"


def _required_runtime_environment(env_var_name):
    runtime_environment = os.environ.get(env_var_name)
    if not runtime_environment:
        raise RuntimeError(
            f"{env_var_name} is required for evaluation MLJobs. Configure a "
            "Snowflake runtime image with the job dependencies preinstalled; "
            "run_evaluation_test.py does not submit pip_requirements."
        )
    return runtime_environment


def _wait_done(job, label, session):
    try:
        job.wait()
    except Exception as exc:
        if "300002" in str(exc) or "000603" in str(exc):
            raise RuntimeError(
                f"{label} job terminated with Snowflake internal error 300002 "
                "(service status unavailable â€” container likely crashed before reaching "
                "a terminal state). Check @MODEL_STAGE/hpo/hpo_failure.json for the "
                "Python traceback. If that file is absent, inspect container logs in "
                "Snowsight for OOM or pre-Python crash details."
            ) from exc
        raise
    if job.status == "DONE":
        print(f"{label} complete.")
        return

    try:
        logs = job.get_logs()
    except Exception as exc:
        logs = (
            f"(job.get_logs() failed: {exc}. Use Snowflake service/job log "
            "retrieval from Snowsight or the MLJob object for details.)"
        )
    print(f"{label} container logs:\n", logs)
    raise RuntimeError(f"{label} failed with status {job.status!r}\n--- logs ---\n{logs}")


def _submit_eval(session, label, compute_pool, env_vars, target_instances=1):
    print(f"Submitting {label} ...")
    selected_methods = {
        method.strip()
        for method in env_vars.get("BENCHMARK_METHODS", "").split(",")
        if method.strip()
    }
    runtime_env_var = (
        "AUTOGLUON_RUNTIME_ENVIRONMENT"
        if env_vars.get("BENCHMARK_METHOD") == AUTOGLUON_METHOD or AUTOGLUON_METHOD in selected_methods
        else "BENCHMARK_RUNTIME_ENVIRONMENT"
    )
    job_kwargs = {
        "source": SCRIPTS_STAGE,
        "entrypoint": "evaluate.py",
        "compute_pool": compute_pool,
        "stage_name": MLJOB_PAYLOAD_STAGE,
        "target_instances": target_instances,
        "runtime_environment": _required_runtime_environment(runtime_env_var),
        "env_vars": {
            "MODEL_PATH": "best.pt",
            "DATA_DIR": "/tmp/data",
            "RESULTS_DIR": "results/",
            "EVAL_RESULTS_STAGE": EVAL_RESULTS_STAGE,
            "HOME": "/tmp",
            **env_vars,
        },
        "session": session,
    }
    return submit_from_stage(**job_kwargs)


def _list_stage(session, stage_path):
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _stage_file_exists(session, stage_path, filename):
    rows = session.sql(f"LIST {stage_path}").collect()
    return any(str(row[0]).rstrip("/").endswith(f"/{filename}") for row in rows)


def _stage_manifest_exists(session):
    """Return True if benchmark_manifest.json exists on the prepared benchmark stage."""
    return _stage_file_exists(session, BENCHMARK_PREPARED_STAGE, "benchmark_manifest.json")


def _submit_dataset_prep(session):
    """Submit the benchmark dataset preparation job (single CPU node)."""
    print("Submitting benchmark dataset preparation job ...")
    return submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="prepare_benchmark_datasets.py",
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        runtime_environment=_required_runtime_environment("PREP_RUNTIME_ENVIRONMENT"),
        external_access_integrations=["BENCHMARK_EXTERNAL_ACCESS"],
        env_vars={
            "BENCHMARK_PREPARED_STAGE": BENCHMARK_PREPARED_STAGE,
            "EVAL_RESULTS_STAGE": EVAL_RESULTS_STAGE,
            "HOME": "/tmp",
        },
        session=session,
    )


def run_evaluation_pipeline(session) -> str:
    # Phase 0: Preflight â€” verify best.pt exists before submitting anything.
    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "best.pt"):
        raise FileNotFoundError(f"{MODEL_STAGE}/checkpoints/best.pt is required before evaluation.")

    benchmark_manifest_exists = _stage_manifest_exists(session)

    # Phase 1 + 2: Submit concurrently when prep is needed. Synthetic eval does
    # not depend on benchmark prep and always runs as a single GPU-node job.
    synthetic_job = _submit_eval(
        session, "synthetic evaluation job", GPU_POOL,
        {"EVAL_MODE": "synthetic", "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1"},
        target_instances=1,
    )
    prep_job = None if benchmark_manifest_exists else _submit_dataset_prep(session)

    # Wait for synthetic and, only when submitted, prep before launching shards.
    _wait_done(synthetic_job, "Synthetic evaluation", session)
    if prep_job is None:
        print(f"Benchmark dataset preparation skipped; found {BENCHMARK_MANIFEST_STAGE_PATH}.")
    else:
        _wait_done(prep_job, "Benchmark dataset preparation", session)

    # Manifest env vars passed to every benchmark shard.
    manifest_env = {
        "BENCHMARK_PREPARED_STAGE": BENCHMARK_PREPARED_STAGE,
        "BENCHMARK_MANIFEST_PATH":  BENCHMARK_MANIFEST_STAGE_PATH,
        "BENCHMARK_SHARD_STRATEGY": "balanced",
    }

    # Phase 3: DeepSet benchmark â€” GPU_BENCHMARK_SHARDS independent single-node GPU shard jobs.
    deepset_shard_jobs = []
    for shard_idx in range(GPU_BENCHMARK_SHARDS):
        label = f"DeepSetModel-MC benchmark shard {shard_idx + 1}/{GPU_BENCHMARK_SHARDS}"
        job = _submit_eval(session, label, GPU_POOL, {
            "EVAL_MODE": "benchmark", "BENCHMARK_METHOD": "DeepSetModel-MC",
            "MC_K": "8",
            "BENCHMARK_DEEPSET_CONTEXT_SIZE": "200",
            "BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES": "5",
            "BENCHMARK_DEEPSET_TEST_BATCH_SIZE": "128",
            "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1",
            "BENCHMARK_NUM_SHARDS": str(GPU_BENCHMARK_SHARDS),
            "BENCHMARK_SHARD_INDEX": str(shard_idx),
            **manifest_env,
        }, target_instances=1)
        deepset_shard_jobs.append((label, job))

    # Phase 4: CPU baselines - CPU_BASELINE_BENCHMARK_SHARDS combined baseline shard jobs.
    # Each shard runs all baseline methods on its assigned datasets.
    baseline_shard_jobs = []
    baseline_methods = ",".join(BASELINE_METHODS)
    for shard_idx in range(CPU_BASELINE_BENCHMARK_SHARDS):
        label = f"CPU baselines benchmark shard {shard_idx + 1}/{CPU_BASELINE_BENCHMARK_SHARDS}"
        job = _submit_eval(session, label, CPU_POOL, {
            "EVAL_MODE": "benchmark", "BENCHMARK_METHODS": baseline_methods,
            "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1",
            "BENCHMARK_NUM_SHARDS": str(CPU_BASELINE_BENCHMARK_SHARDS),
            "BENCHMARK_SHARD_INDEX": str(shard_idx),
            **manifest_env,
        }, target_instances=1)
        baseline_shard_jobs.append((label, job))

    # Phase 5: AutoGluon â€” AUTOGLUON_BENCHMARK_SHARDS independent single-node CPU shard jobs.
    autogluon_shard_jobs = []
    for shard_idx in range(AUTOGLUON_BENCHMARK_SHARDS):
        label = f"AutoGluon benchmark shard {shard_idx + 1}/{AUTOGLUON_BENCHMARK_SHARDS}"
        job = _submit_eval(session, label, AUTOGLUON_CPU_POOL, {
            "EVAL_MODE": "benchmark", "BENCHMARK_METHOD": AUTOGLUON_METHOD,
            "AUTOGLUON_TIME_LIMIT": "300",
            "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1",
            "BENCHMARK_NUM_SHARDS": str(AUTOGLUON_BENCHMARK_SHARDS),
            "BENCHMARK_SHARD_INDEX": str(shard_idx),
            **manifest_env,
        }, target_instances=1)
        autogluon_shard_jobs.append((label, job))

    # Wait for all shards.
    for label, job in baseline_shard_jobs:
        _wait_done(job, label, session)
    for label, job in deepset_shard_jobs:
        _wait_done(job, label, session)
    for label, job in autogluon_shard_jobs:
        _wait_done(job, label, session)

    # Phase 6: Aggregate.
    aggregate_job = _submit_eval(session, "benchmark aggregate job", CPU_POOL,
                                 {"EVAL_MODE": "aggregate"})
    _wait_done(aggregate_job, "Benchmark aggregate", session)

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    eval_contents = _list_stage(session, f"{EVAL_RESULTS_STAGE}/")
    return (
        "Evaluation pipeline complete.\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
        + "\n\nEVALUATION_RESULTS_STAGE:\n"
        + "\n".join(f"  {p}" for p in eval_contents)
    )
