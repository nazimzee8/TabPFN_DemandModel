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
AUTOGLUON_MAX_CONCURRENT_SHARDS = 30  # cap AutoGluon concurrency; total stays 30

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
DEEPSET_METHOD   = "MODEL-ICL-MC"

BENCHMARK_REQUIRED_IMPORTS = (
    "torch,pyarrow,pandas,scipy,sklearn,"
    "snowflake.snowpark,snowflake.ml.jobs"
)
BASELINE_REQUIRED_IMPORTS = (
    "torch,pyarrow,pandas,scipy,sklearn,xgboost,lightgbm,catboost,"
    "snowflake.snowpark,snowflake.ml.jobs"
)
CATBOOST_VERSION = "1.2.10"
BASELINE_EXTRA_PIP_REQUIREMENTS = [f"catboost=={CATBOOST_VERSION}"]
PYPI_EAI = "TABPFN_PYPI_EAI"
BENCHMARK_EXTERNAL_ACCESS_EAI = "BENCHMARK_EXTERNAL_ACCESS"
OPENML_VERSION = "0.15.1"
PREP_EXTRA_PIP_REQUIREMENTS = [f"openml=={OPENML_VERSION}"]
AUTOGLUON_VERSION = "1.3.0"
AUTOGLUON_EXTRA_PIP_REQUIREMENTS = [f"autogluon.tabular=={AUTOGLUON_VERSION}"]
ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS = "true"
PREP_REQUIRED_IMPORTS = "openml,numpy,snowflake.snowpark"
AUTOGLUON_REQUIRED_IMPORTS = (
    "autogluon.tabular,numpy,pandas,sklearn,scipy,pyarrow,torch,"
    "snowflake.snowpark"
)

COMPUTE_POOL_USABLE_STATES = {"ACTIVE", "IDLE"}
COMPUTE_POOL_FAILED_STATES = {
    "FAILED",
    "ERROR",
    "DELETING",
    "STOPPING",
    "UNKNOWN",
}


def _resolve_runtime_environments(
    prep_runtime_environment=None,
    benchmark_runtime_environment=None,
    autogluon_runtime_environment=None,
):
    runtimes = {
        "PREP_RUNTIME_ENVIRONMENT": _required_runtime_environment(
            "PREP_RUNTIME_ENVIRONMENT", prep_runtime_environment
        ),
        "BENCHMARK_RUNTIME_ENVIRONMENT": _required_runtime_environment(
            "BENCHMARK_RUNTIME_ENVIRONMENT", benchmark_runtime_environment
        ),
        "AUTOGLUON_RUNTIME_ENVIRONMENT": _required_runtime_environment(
            "AUTOGLUON_RUNTIME_ENVIRONMENT", autogluon_runtime_environment
        ),
    }
    print("Evaluation runtime environments:", flush=True)
    for name, runtime in runtimes.items():
        print(f"  {name}={runtime}", flush=True)
    return runtimes


def _required_runtime_environment(env_var_name, explicit_value=None):
    runtime_environment = explicit_value or os.environ.get(env_var_name)
    if not runtime_environment:
        raise RuntimeError(
            f"{env_var_name} is required for evaluation MLJobs. Pass it as an "
            "argument to CALL run_evaluation_pipeline(...)."
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


def _selected_methods_from_env(env_vars):
    selected_methods = {
        method.strip()
        for method in env_vars.get("BENCHMARK_METHODS", "").split(",")
        if method.strip()
    }
    single_method = env_vars.get("BENCHMARK_METHOD", "").strip()
    if single_method:
        selected_methods.add(single_method)
    return selected_methods


def _pip_requirements_for_eval(env_vars):
    """Return pip_requirements list only when the job includes CatBoost."""
    if "CatBoost" in _selected_methods_from_env(env_vars):
        return list(BASELINE_EXTRA_PIP_REQUIREMENTS)
    return None


def _is_node_quota_error(exc):
    msg = str(exc)
    return "395034" in msg or (
        "Requested number of nodes" in msg and "exceeds the node limit" in msg
    )


def _wait_job_group(labeled_jobs, session):
    """Wait for every (label, job) pair; propagates first failure."""
    for label, job in labeled_jobs:
        _wait_done(job, label, session)


def _batched(items, n):
    """Yield successive n-item chunks from items list."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _submit_capacity_probe(session, label, compute_pool, runtime_environment):
    print(f"Submitting capacity probe: {label} ...", flush=True)
    return submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="capacity_probe.py",
        compute_pool=compute_pool,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        runtime_environment=runtime_environment,
        env_vars={
            "HOME": "/tmp",
            "CAPACITY_PROBE_LABEL": label,
            "EVAL_RUNTIME_ENVIRONMENT": runtime_environment,
        },
        session=session,
    )


def _submit_and_wait_capacity_phase(session, phase_label, compute_pool,
                                     runtime_environment, count):
    """Submit `count` capacity probe jobs on `compute_pool`, wait for all."""
    jobs = []
    for i in range(count):
        label = f"{phase_label} {i + 1}/{count}"
        try:
            job = _submit_capacity_probe(session, label, compute_pool, runtime_environment)
        except Exception as exc:
            if _is_node_quota_error(exc):
                raise RuntimeError(
                    f"Capacity probe phase '{phase_label}' failed: Snowflake node quota "
                    f"exceeded while submitting job {i + 1}/{count} on {compute_pool} "
                    f"(requested concurrency={count}). "
                    "Remediation: SHOW COMPUTE POOLS; suspend idle pools; wait for active "
                    "jobs to finish; or request higher Snowflake account node quota."
                ) from exc
            raise
        jobs.append((label, job))
    _wait_job_group(jobs, session)


_UNSET = object()


def _submit_eval(session, label, compute_pool, env_vars, runtimes,
                 target_instances=1, pip_requirements=_UNSET,
                 external_access_integrations=None):
    print(f"Submitting {label} ...")
    runtime_environment = _runtime_for_eval(env_vars, runtimes)
    if pip_requirements is _UNSET:
        pip_requirements = _pip_requirements_for_eval(env_vars)

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
            # ALLOW_UNSAFE_TORCH_LOAD is currently "true" as a temporary escape hatch for the
            # legacy best.pt checkpoint (pre-v2 pickle format). Revert to "false" only after
            # migrating best.pt to checkpoint_format_version=2 via scripts/migrate_checkpoint.py.
            "ALLOW_UNSAFE_TORCH_LOAD": ALLOW_UNSAFE_TORCH_LOAD_FOR_LEGACY_CHECKPOINTS,
            **env_vars,
        },
        "session": session,
    }

    if pip_requirements:
        print(
            f"  Adding pip_requirements for {label}: {pip_requirements}",
            flush=True,
        )
        job_kwargs["pip_requirements"] = pip_requirements

    if external_access_integrations:
        job_kwargs["external_access_integrations"] = external_access_integrations

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


def _truthy_snowflake_value(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _show_compute_pool_info(session, compute_pool):
    rows = session.sql(f"SHOW COMPUTE POOLS LIKE '{compute_pool}'").collect()
    for row in rows:
        name = _row_get(row, ("name", "compute_pool_name"), fallback_indices=(0,))
        if str(name).upper() == compute_pool.upper():
            return {
                "name": str(name),
                "state": str(_row_get(row, ("state", "status"), fallback_indices=(1, 2)) or "").upper(),
                "auto_resume": _row_get(row, ("auto_resume",), fallback_indices=()),
                "auto_suspend_secs": _row_get(row, ("auto_suspend_secs",), fallback_indices=()),
                "active_nodes": _row_get(row, ("active_nodes",), fallback_indices=()),
                "idle_nodes": _row_get(row, ("idle_nodes",), fallback_indices=()),
                "target_nodes": _row_get(row, ("target_nodes",), fallback_indices=()),
            }
    return None


def _ensure_compute_pool_usable(session, compute_pool):
    info = _show_compute_pool_info(session, compute_pool)
    if info is None:
        raise RuntimeError(f"Compute pool {compute_pool} does not exist.")

    state = info["state"]
    auto_resume = _truthy_snowflake_value(info.get("auto_resume"))

    if state in COMPUTE_POOL_FAILED_STATES:
        raise RuntimeError(f"Compute pool {compute_pool} is unusable: state={state}.")

    if state in COMPUTE_POOL_USABLE_STATES:
        print(
            f"Compute pool {compute_pool} is {state}. "
            f"auto_resume={info.get('auto_resume')}, "
            f"auto_suspend_secs={info.get('auto_suspend_secs')}, "
            f"active_nodes={info.get('active_nodes')}, "
            f"idle_nodes={info.get('idle_nodes')}, "
            f"target_nodes={info.get('target_nodes')}",
            flush=True,
        )
        return

    if state == "SUSPENDED":
        if auto_resume:
            print(
                f"Compute pool {compute_pool} is SUSPENDED with AUTO_RESUME=TRUE; "
                "allowing submit_from_stage() to trigger resume. "
                f"auto_suspend_secs={info.get('auto_suspend_secs')}",
                flush=True,
            )
            return
        print(
            f"Compute pool {compute_pool} is SUSPENDED; issuing RESUME.",
            flush=True,
        )
        session.sql(f"ALTER COMPUTE POOL {compute_pool} RESUME").collect()
        time.sleep(10)
        return

    raise RuntimeError(f"Compute pool {compute_pool} is not usable: state={state}.")


def _preflight_compute_pool_list(session, compute_pools):
    """Validate only the specified compute pools."""
    for compute_pool in compute_pools:
        _ensure_compute_pool_usable(session, compute_pool)


def _preflight_compute_pools(session):
    _preflight_compute_pool_list(session, (GPU_POOL, CPU_POOL, AUTOGLUON_CPU_POOL))


def _submit_runtime_probe(
    session,
    label,
    compute_pool,
    runtime_environment,
    required_imports,
    require_cuda=False,
    pip_requirements=None,
    external_access_integrations=None,
):
    print(f"Submitting runtime preflight probe for {label} ...")

    job_kwargs = {
        "source": SCRIPTS_STAGE,
        "entrypoint": "runtime_probe.py",
        "compute_pool": compute_pool,
        "stage_name": MLJOB_PAYLOAD_STAGE,
        "target_instances": 1,
        "runtime_environment": runtime_environment,
        "env_vars": {
            "HOME": "/tmp",
            "EVAL_RUNTIME_ENVIRONMENT": runtime_environment,
            "RUNTIME_PROBE_LABEL": label,
            "REQUIRED_IMPORTS": required_imports,
            "REQUIRE_CUDA": "true" if require_cuda else "false",
        },
        "session": session,
    }

    if pip_requirements:
        print(
            f"  Adding pip_requirements for runtime probe {label}: {pip_requirements}",
            flush=True,
        )
        job_kwargs["pip_requirements"] = pip_requirements

    if external_access_integrations:
        job_kwargs["external_access_integrations"] = external_access_integrations

    return submit_from_stage(**job_kwargs)


def _preflight_runtime_environments(session, runtimes):
    baseline_pip_requirements = list(BASELINE_EXTRA_PIP_REQUIREMENTS)

    probe_specs = [
        (
            "benchmark GPU runtime",
            GPU_POOL,
            runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"],
            BENCHMARK_REQUIRED_IMPORTS,
            True,
            None,
            None,
        ),
        (
            "benchmark aggregate CPU runtime",
            CPU_POOL,
            runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"],
            BENCHMARK_REQUIRED_IMPORTS,
            False,
            None,
            None,
        ),
        (
            "CPU baseline runtime with CatBoost pip dependency",
            CPU_POOL,
            runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"],
            BASELINE_REQUIRED_IMPORTS,
            False,
            baseline_pip_requirements,
            [PYPI_EAI],
        ),
        (
            "prep CPU runtime",
            CPU_POOL,
            runtimes["PREP_RUNTIME_ENVIRONMENT"],
            PREP_REQUIRED_IMPORTS,
            False,
            list(PREP_EXTRA_PIP_REQUIREMENTS),
            [PYPI_EAI],
        ),
        (
            "AutoGluon CPU runtime",
            AUTOGLUON_CPU_POOL,
            runtimes["AUTOGLUON_RUNTIME_ENVIRONMENT"],
            AUTOGLUON_REQUIRED_IMPORTS,
            False,
            list(AUTOGLUON_EXTRA_PIP_REQUIREMENTS),
            [PYPI_EAI],
        ),
    ]

    for (
        label,
        compute_pool,
        runtime_environment,
        required_imports,
        require_cuda,
        pip_requirements,
        external_access_integrations,
    ) in probe_specs:
        job = _submit_runtime_probe(
            session,
            label,
            compute_pool,
            runtime_environment,
            required_imports,
            require_cuda=require_cuda,
            pip_requirements=pip_requirements,
            external_access_integrations=external_access_integrations,
        )
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
        pip_requirements=list(PREP_EXTRA_PIP_REQUIREMENTS),
        external_access_integrations=[BENCHMARK_EXTERNAL_ACCESS_EAI, PYPI_EAI],
        env_vars={
            "BENCHMARK_PREPARED_STAGE": BENCHMARK_PREPARED_STAGE,
            "EVAL_RESULTS_STAGE": EVAL_RESULTS_STAGE,
            "HOME": "/tmp",
            "EVAL_RUNTIME_ENVIRONMENT": runtime_environment,
        },
        session=session,
    )


def _benchmark_manifest_env():
    return {
        "BENCHMARK_PREPARED_STAGE": BENCHMARK_PREPARED_STAGE,
        "BENCHMARK_MANIFEST_PATH":  BENCHMARK_MANIFEST_STAGE_PATH,
        "BENCHMARK_SHARD_STRATEGY": "balanced",
    }


def _run_prep_phase(session, runtimes):
    """Submit and wait for benchmark dataset preparation job."""
    prep_job = _submit_dataset_prep(session, runtimes)
    _wait_done(prep_job, "Benchmark dataset preparation", session)


def _run_deepset_phase(session, runtimes):
    """Submit and wait for synthetic eval, then all 10 DeepSet GPU benchmark shards."""
    synthetic_job = _submit_eval(
        session, "synthetic evaluation job", GPU_POOL,
        {"EVAL_MODE": "synthetic", "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1"},
        runtimes, target_instances=1,
    )
    _wait_done(synthetic_job, "Synthetic evaluation", session)

    manifest_env = _benchmark_manifest_env()
    deepset_jobs = []
    for shard_idx in range(GPU_BENCHMARK_SHARDS):
        label = f"{DEEPSET_METHOD} benchmark shard {shard_idx + 1}/{GPU_BENCHMARK_SHARDS}"
        job = _submit_eval(session, label, GPU_POOL, {
            "EVAL_MODE": "benchmark", "BENCHMARK_METHOD": DEEPSET_METHOD,
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
        deepset_jobs.append((label, job))
    _wait_job_group(deepset_jobs, session)


def _run_baseline_phase(session, runtimes):
    """Submit and wait for all 3 CPU baseline benchmark shards."""
    manifest_env = _benchmark_manifest_env()
    baseline_methods = ",".join(BASELINE_METHODS)
    baseline_jobs = []
    for shard_idx in range(CPU_BASELINE_BENCHMARK_SHARDS):
        label = f"CPU baselines benchmark shard {shard_idx + 1}/{CPU_BASELINE_BENCHMARK_SHARDS}"
        job = _submit_eval(session, label, CPU_POOL, {
            "EVAL_MODE": "benchmark", "BENCHMARK_METHODS": baseline_methods,
            "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1",
            "BENCHMARK_NUM_SHARDS": str(CPU_BASELINE_BENCHMARK_SHARDS),
            "BENCHMARK_SHARD_INDEX": str(shard_idx),
            **manifest_env,
        }, runtimes, target_instances=1,
           pip_requirements=list(BASELINE_EXTRA_PIP_REQUIREMENTS),
           external_access_integrations=[PYPI_EAI])
        baseline_jobs.append((label, job))
    _wait_job_group(baseline_jobs, session)


def _run_autogluon_phase(session, runtimes):
    """Submit 30 AutoGluon shards in batches of AUTOGLUON_MAX_CONCURRENT_SHARDS."""
    manifest_env = _benchmark_manifest_env()
    autogluon_shards = [
        (f"AutoGluon benchmark shard {i + 1}/{AUTOGLUON_BENCHMARK_SHARDS}", i)
        for i in range(AUTOGLUON_BENCHMARK_SHARDS)
    ]
    for batch in _batched(autogluon_shards, AUTOGLUON_MAX_CONCURRENT_SHARDS):
        batch_jobs = []
        for label, shard_idx in batch:
            job = _submit_eval(session, label, AUTOGLUON_CPU_POOL, {
                "EVAL_MODE": "benchmark", "BENCHMARK_METHOD": AUTOGLUON_METHOD,
                "AUTOGLUON_TIME_LIMIT": "300",
                "EVAL_NUM_NODES": "1", "EVAL_WORKERS_PER_NODE": "1",
                "BENCHMARK_NUM_SHARDS": str(AUTOGLUON_BENCHMARK_SHARDS),
                "BENCHMARK_SHARD_INDEX": str(shard_idx),
                **manifest_env,
            }, runtimes, target_instances=1,
               pip_requirements=list(AUTOGLUON_EXTRA_PIP_REQUIREMENTS),
               external_access_integrations=[PYPI_EAI])
            batch_jobs.append((label, job))
        _wait_job_group(batch_jobs, session)


def _run_aggregate_phase(session, runtimes):
    """Submit and wait for benchmark aggregation job."""
    aggregate_job = _submit_eval(
        session, "benchmark aggregate job", CPU_POOL,
        {"EVAL_MODE": "aggregate"}, runtimes,
    )
    _wait_done(aggregate_job, "Benchmark aggregate", session)


def run_evaluation_pipeline(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    # Phase 0: Preflight - verify required staged artifacts before submitting anything.
    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "best.pt"):
        raise FileNotFoundError(f"{MODEL_STAGE}/checkpoints/best.pt is required before evaluation.")
    if not _stage_file_exists(session, SCRIPTS_STAGE, "runtime_probe.py"):
        raise FileNotFoundError(
            f"{SCRIPTS_STAGE}runtime_probe.py is required before evaluation runtime preflight."
        )

    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pools(session)
    _preflight_runtime_environments(session, runtimes)

    # Phases 1-6: delegate to phase helpers.
    # Note: prep (phase 1) runs before deepset (phases 2-3) sequentially.
    # The prior concurrent synthetic+prep submission is replaced by this
    # sequential ordering; split procedures are the recommended path for
    # tight quota.
    _run_prep_phase(session, runtimes)
    _run_deepset_phase(session, runtimes)
    _run_baseline_phase(session, runtimes)
    _run_autogluon_phase(session, runtimes)
    _run_aggregate_phase(session, runtimes)

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    eval_contents = _list_stage(session, f"{EVAL_RESULTS_STAGE}/")
    return (
        "Evaluation pipeline complete.\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
        + "\n\nEVALUATION_RESULTS_STAGE:\n"
        + "\n".join(f"  {p}" for p in eval_contents)
    )


def run_evaluation_runtime_probes(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Probe-only stored procedure handler. Runs all preflight checks without
    submitting evaluation jobs. Useful for validating runtime environments
    before a full evaluation run."""
    if not _stage_file_exists(session, SCRIPTS_STAGE, "runtime_probe.py"):
        raise FileNotFoundError(
            f"{SCRIPTS_STAGE}runtime_probe.py is required before evaluation runtime preflight."
        )

    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )

    _preflight_compute_pools(session)
    _preflight_runtime_environments(session, runtimes)

    return "Evaluation runtime probes completed successfully."


def run_evaluation_capacity_probe(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Lightweight quota/capacity check. Submits capacity_probe.py in 3 non-overlapping
    phases matching the fixed evaluation pipeline envelope. Does not load models or data."""
    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pools(session)

    _submit_and_wait_capacity_phase(
        session, "GPU capacity probe",
        GPU_POOL, runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"],
        GPU_BENCHMARK_SHARDS,           # 10
    )
    _submit_and_wait_capacity_phase(
        session, "CPU baseline capacity probe",
        CPU_POOL, runtimes["BENCHMARK_RUNTIME_ENVIRONMENT"],
        CPU_BASELINE_BENCHMARK_SHARDS,  # 3
    )
    _submit_and_wait_capacity_phase(
        session, "AutoGluon capacity probe",
        AUTOGLUON_CPU_POOL, runtimes["AUTOGLUON_RUNTIME_ENVIRONMENT"],
        AUTOGLUON_MAX_CONCURRENT_SHARDS,  # 30
    )
    return (
        f"Capacity probe complete. Validated concurrency envelope: "
        f"GPU={GPU_BENCHMARK_SHARDS}, CPU={CPU_BASELINE_BENCHMARK_SHARDS}, "
        f"AutoGluon={AUTOGLUON_MAX_CONCURRENT_SHARDS}."
    )


def run_evaluation_prep(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Fetch/validate benchmark manifest and index. Runs on DEEPSET_CPU_POOL."""
    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pool_list(session, [CPU_POOL])
    _run_prep_phase(session, runtimes)
    return "Evaluation prep complete. Benchmark manifest validated."


def run_deepset_evaluation(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Run synthetic eval and 10 DeepSet GPU benchmark shards on DEEPSET_GPU_POOL."""
    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "best.pt"):
        raise FileNotFoundError(f"{MODEL_STAGE}/checkpoints/best.pt is required.")
    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pool_list(session, [GPU_POOL])
    _run_deepset_phase(session, runtimes)
    return f"DeepSet evaluation complete. {GPU_BENCHMARK_SHARDS} benchmark shards finished."


def run_baseline_evaluation(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Run 3 CPU baseline benchmark shards on DEEPSET_CPU_POOL."""
    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pool_list(session, [CPU_POOL])
    _run_baseline_phase(session, runtimes)
    return f"CPU baseline evaluation complete. {CPU_BASELINE_BENCHMARK_SHARDS} shards finished."


def run_autogluon_evaluation(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Run 30 AutoGluon benchmark shards on AUTOGLUON_CPU_POOL."""
    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pool_list(session, [AUTOGLUON_CPU_POOL])
    _run_autogluon_phase(session, runtimes)
    return (
        f"AutoGluon evaluation complete. {AUTOGLUON_BENCHMARK_SHARDS} total shards "
        f"(batched at {AUTOGLUON_MAX_CONCURRENT_SHARDS} concurrent)."
    )


def run_evaluation_aggregation(
    session,
    prep_runtime_environment: str = None,
    benchmark_runtime_environment: str = None,
    autogluon_runtime_environment: str = None,
) -> str:
    """Run benchmark aggregation on DEEPSET_CPU_POOL and return results listing."""
    runtimes = _resolve_runtime_environments(
        prep_runtime_environment=prep_runtime_environment,
        benchmark_runtime_environment=benchmark_runtime_environment,
        autogluon_runtime_environment=autogluon_runtime_environment,
    )
    _preflight_compute_pool_list(session, [CPU_POOL])
    _run_aggregate_phase(session, runtimes)
    eval_contents = _list_stage(session, f"{EVAL_RESULTS_STAGE}/")
    return (
        "Benchmark aggregation complete.\n\nEVALUATION_RESULTS_STAGE:\n"
        + "\n".join(f"  {p}" for p in eval_contents)
    )
