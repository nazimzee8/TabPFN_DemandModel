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
import time

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

# Shard counts â€" match compute pool MAX_NODES capacities.
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

BENCHMARK_REQUIRED_IMPORTS = (
    "torch,pyarrow,pandas,scipy,sklearn,xgboost,lightgbm,catboost,"
    "snowflake.snowpark,snowflake.ml.jobs"
)
PREP_REQUIRED_IMPORTS = "openml,numpy,snowflake.snowpark"
AUTOGLUON_REQUIRED_IMPORTS = (
    "autogluon.tabular,numpy,pandas,sklearn,scipy,pyarrow,torch,"
    "snowflake.snowpark"
)

COMPUTE_POOL_USABLE_STATES = {"ACTIVE", "IDLE"}
COMPUTE_POOL_RESUMABLE_STATES = {"SUSPENDED"}
COMPUTE_POOL_FAILED_STATES = {
    "FAILED",
    "ERROR",
    "DELETING",
    "STOPPING",
    "UNKNOWN",
}
COMPUTE_POOL_POLL_SECONDS = int(os.environ.get("EVAL_COMPUTE_POOL_POLL_SECONDS", "10"))
COMPUTE_POOL_MAX_POLLS = int(os.environ.get("EVAL_COMPUTE_POOL_MAX_POLLS", "60"))


def _resolve_runtime_environments():
    runtimes = {
        "PREP_RUNTIME_ENVIRONMENT": _required_runtime_environment("PREP_RUNTIME_ENVIRONMENT"),
        "BENCHMARK_RUNTIME_ENVIRONMENT": _required_runtime_environment("BENCHMARK_RUNTIME_ENVIRONMENT"),
        "AUTOGLUON_RUNTIME_ENVIRONMENT": _required_runtime_environment("AUTOGLUON_RUNTIME_ENVIRONMENT"),
    }
    print("Evaluation runtime environments:", flush=True)
    for name, runtime in runtimes.items():
        print(f"  {name}={runtime}", flush=True)
    return runtimes


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
                f"{label} MLJob terminated with Snowflake internal error 300002 / "
                "service status unavailable. The container likely crashed before "
                "reaching a clean terminal state. Inspect the current evaluation "
                "job's container logs in Snowsight. If the failure occurs after "
                "Python starts, check stdout for evaluate.py or runtime_probe.py "
                "diagnostics."
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


def _runtime_for_eval(env_vars, runtimes):
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
    return runtimes[runtime_env_var]


def _submit_eval(session, label, compute_pool, env_vars, runtimes, target_instances=1):
    print(f"Submitting {label} ...")
    runtime_environment = _runtime_for_eval(env_vars, runtimes)
    job_kwargs = {
        "source": SCRIPTS_STAGE,
        "entrypoint": "evaluate.py",
        "compute_pool": compute_pool,
        "stage_name": MLJOB_PAYLOAD_STAGE,
        "target_instances": target_instances,
        "runtime_environment": runtime_environment,
        "env_vars": {
            "MODEL_PATH": "best.pt",
            "DATA_DIR": "/tmp/data",
            "RESULTS_DIR": "results/",
            "EVAL_RESULTS_STAGE": EVAL_RESULTS_STAGE,
            "HOME": "/tmp",
            "EVAL_RUNTIME_ENVIRONMENT": runtime_environment,
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


def _row_as_dict(row):
    if hasattr(row, "as_dict"):
        try:
            return {str(k).lower(): v for k, v in row.as_dict().items()}
        except TypeError:
            pass
    if isinstance(row, dict):
        return {str(k).lower(): v for k, v in row.items()}
    return {}


def _row_get(row, field_names, fallback_indices=()):
    row_dict = _row_as_dict(row)
    for field in field_names:
        if field.lower() in row_dict:
            return row_dict[field.lower()]
    for index in fallback_indices:
        try:
            return row[index]
        except Exception:
            continue
    return None


def _show_compute_pool(session, compute_pool):
    rows = session.sql(f"SHOW COMPUTE POOLS LIKE '{compute_pool}'").collect()
    for row in rows:
        name = _row_get(row, ("name", "compute_pool_name"), fallback_indices=(0,))
        if str(name).upper() == compute_pool.upper():
            state = _row_get(row, ("state", "status"), fallback_indices=(1, 2))
            return str(state).upper() if state is not None else ""
    return None


def _ensure_compute_pool_usable(session, compute_pool):
    state = _show_compute_pool(session, compute_pool)
    if state is None:
        raise RuntimeError(f"Compute pool {compute_pool} does not exist.")
    if state in COMPUTE_POOL_FAILED_STATES:
        raise RuntimeError(f"Compute pool {compute_pool} is unusable: state={state}.")
    if state in COMPUTE_POOL_USABLE_STATES:
        print(f"Compute pool {compute_pool} is {state}.", flush=True)
        return
    if state not in COMPUTE_POOL_RESUMABLE_STATES:
        raise RuntimeError(f"Compute pool {compute_pool} is not usable: state={state}.")

    print(f"Compute pool {compute_pool} is SUSPENDED; resuming ...", flush=True)
    session.sql(f"ALTER COMPUTE POOL {compute_pool} RESUME").collect()
    for _ in range(COMPUTE_POOL_MAX_POLLS):
        state = _show_compute_pool(session, compute_pool)
        if state in COMPUTE_POOL_USABLE_STATES:
            print(f"Compute pool {compute_pool} is {state}.", flush=True)
            return
        if state in COMPUTE_POOL_FAILED_STATES:
            raise RuntimeError(f"Compute pool {compute_pool} failed while resuming: state={state}.")
        time.sleep(COMPUTE_POOL_POLL_SECONDS)
    raise RuntimeError(
        f"Compute pool {compute_pool} did not become usable after "
        f"{COMPUTE_POOL_MAX_POLLS * COMPUTE_POOL_POLL_SECONDS} seconds; last state={state}."
    )


def _preflight_compute_pools(session):
    for compute_pool in (GPU_POOL, CPU_POOL, AUTOGLUON_CPU_POOL):
        _ensure_compute_pool_usable(session, compute_pool)


def _submit_runtime_probe(
    session,
    label,
    compute_pool,
    runtime_environment,
    required_imports,
    require_cuda=False,
):
    print(f"Submitting runtime preflight probe for {label} ...")
    return submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="runtime_probe.py",
        compute_pool=compute_pool,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        runtime_environment=runtime_environment,
        env_vars={
            "HOME": "/tmp",
            "EVAL_RUNTIME_ENVIRONMENT": runtime_environment,
            "RUNTIME_PROBE_LABEL": label,
            "REQUIRED_IMPORTS": required_imports,
            "REQUIRE_CUDA": "true" if require_cuda else "false",
        },
        session=session,
    )


def _preflight_runtime_environments(session, runtimes):
    probe_specs = [
        ("benchmark GPU runtime", GPU_POOL, runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"], BENCHMARK_REQUIRED_IMPORTS, True),
        ("benchmark CPU runtime", CPU_POOL, runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"], BENCHMARK_REQUIRED_IMPORTS, False),
        ("prep CPU runtime", CPU_POOL, runtimes["PREP_RUNTIME_ENVIRONMENT"], PREP_REQUIRED_IMPORTS, False),
        ("AutoGluon CPU runtime", AUTOGLUON_CPU_POOL, runtimes["AUTOGLUON_RUNTIME_ENVIRONMENT"], AUTOGLUON_REQUIRED_IMPORTS, False),
    ]

    probes = [
        (
            label,
            _submit_runtime_probe(
                session,
                label,
                compute_pool,
                runtime_environment,
                required_imports,
                require_cuda=require_cuda,
            ),
        )
        for label, compute_pool, runtime_environment, required_imports, require_cuda in probe_specs
    ]
    for label, job in probes:
        _wait_done(job, f"Runtime preflight probe ({label})", session)


def _submit_dataset_prep(session, runtimes):
    """Submit the benchmark dataset preparation job (single CPU node)."""
    runtime_environment = runtimes["PREP_RUNTIME_ENVIRONMENT"]
    print("Submitting benchmark dataset preparation job ...")
    return submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="prepare_benchmark_datasets.py",
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        runtime_environment=runtime_environment,
        external_access_integrations=["BENCHMARK_EXTERNAL_ACCESS"],
        env_vars={
            "BENCHMARK_PREPARED_STAGE": BENCHMARK_PREPARED_STAGE,
            "EVAL_RESULTS_STAGE": EVAL_RESULTS_STAGE,
            "HOME": "/tmp",
            "EVAL_RUNTIME_ENVIRONMENT": runtime_environment,
        },
        session=session,
    )


def run_evaluation_pipeline(session) -> str:
    # Phase 0: Preflight - verify required staged artifacts before submitting anything.
    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "best.pt"):
        raise FileNotFoundError(f"{MODEL_STAGE}/checkpoints/best.pt is required before evaluation.")
    if not _stage_file_exists(session, SCRIPTS_STAGE, "runtime_probe.py"):
        raise FileNotFoundError(
            f"{SCRIPTS_STAGE}runtime_probe.py is required before evaluation runtime preflight."
        )

    runtimes = _resolve_runtime_environments()
    _preflight_compute_pools(session)
    _preflight_runtime_environments(session, runtimes)

    # Phase 1 + 2: Submit concurrently. Synthetic eval does not depend on
    # benchmark prep. Prep is always submitted as a lightweight manifest/index
    # validation step and exits early when the staged manifest is already valid.
    synthetic_job = _submit_eval(
        session, "synthetic evaluation job", GPU_POOL,
        {"EVAL_MODE": "synthetic", "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1"},
        runtimes,
        target_instances=1,
    )
    prep_job = _submit_dataset_prep(session, runtimes)

    # Wait for synthetic and prep validation before launching shards.
    _wait_done(synthetic_job, "Synthetic evaluation", session)
    _wait_done(prep_job, "Benchmark dataset preparation", session)

    # Manifest env vars passed to every benchmark shard.
    manifest_env = {
        "BENCHMARK_PREPARED_STAGE": BENCHMARK_PREPARED_STAGE,
        "BENCHMARK_MANIFEST_PATH":  BENCHMARK_MANIFEST_STAGE_PATH,
        "BENCHMARK_SHARD_STRATEGY": "balanced",
    }

    # Phase 3: DeepSet benchmark â€" GPU_BENCHMARK_SHARDS independent single-node GPU shard jobs.
    deepset_shard_jobs = []
    for shard_idx in range(GPU_BENCHMARK_SHARDS):
        label = f"DeepSetModel-MC benchmark shard {shard_idx + 1}/{GPU_BENCHMARK_SHARDS}"
        job = _submit_eval(session, label, GPU_POOL, {
            "EVAL_MODE": "benchmark", "BENCHMARK_METHOD": "DeepSetModel-MC",
            "MC_K": "8",
            "BENCHMARK_DEEPSET_CONTEXT_SIZE": "200",
            "BENCHMARK_DEEPSET_CONTEXT_ENSEMBLES": "5",
            "BENCHMARK_DEEPSET_TEST_BATCH_SIZE": "128",
            "BENCHMARK_DEEPSET_FEATURE_SELECTOR": "train_f_regression",
            "BENCHMARK_REQUIRE_CUDA": "true",
            "BENCHMARK_DEEPSET_MAX_GPU_INFERENCE_BYTES": "268435456",
            "BENCHMARK_DEEPSET_GPU_MEMORY_SAFETY_FACTOR": "4.0",
            "BENCHMARK_DEEPSET_MAX_GPU_MEMORY_FRACTION": "0.80",
            "BENCHMARK_DEEPSET_EMPTY_CACHE": "true",
            "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1",
            "BENCHMARK_NUM_SHARDS": str(GPU_BENCHMARK_SHARDS),
            "BENCHMARK_SHARD_INDEX": str(shard_idx),
            **manifest_env,
        }, runtimes, target_instances=1)
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
        }, runtimes, target_instances=1)
        baseline_shard_jobs.append((label, job))

    # Phase 5: AutoGluon â€" AUTOGLUON_BENCHMARK_SHARDS independent single-node CPU shard jobs.
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
        }, runtimes, target_instances=1)
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
                                 {"EVAL_MODE": "aggregate"}, runtimes)
    _wait_done(aggregate_job, "Benchmark aggregate", session)

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    eval_contents = _list_stage(session, f"{EVAL_RESULTS_STAGE}/")
    return (
        "Evaluation pipeline complete.\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
        + "\n\nEVALUATION_RESULTS_STAGE:\n"
        + "\n".join(f"  {p}" for p in eval_contents)
    )
