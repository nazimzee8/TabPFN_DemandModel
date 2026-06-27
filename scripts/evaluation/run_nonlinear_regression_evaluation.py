"""
run_nonlinear_regression_evaluation.py
=========================================
Orchestration handlers for the nonlinear synthetic regression evaluation suite
(suite_id='nonlinear', 6 families, 420 datasets).

Imports shared helpers from run_linear_regression_evaluation rather than
duplicating them. Every job submission injects
  SYNREG_INDEX_TABLE=NONLINEAR_REGRESSION_DATASET_INDEX
so that evaluate_nonlinear_regression.py and autogluon_ray.py query the
nonlinear index table, leaving LINEAR_REGRESSION_DATASET_INDEX untouched.

Stored procedure handlers (all callable from SQL):
  run_nonlinear_regression_prep
  run_nonlinear_regression_deepset_evaluation
  run_nonlinear_regression_baseline_evaluation       (0/1/2 extra-arg overloads)
  run_nonlinear_regression_autogluon_evaluation      (5-arg and 10-arg overloads)
  run_nonlinear_regression_aggregation               (2-arg and 5-arg overloads)
  SPCS probe helpers                                   (4 functions × 2 overloads)

All 5 evaluation procs take IS_MIXED_CATEGORICAL BOOLEAN as their first SQL
parameter. Pass FALSE for the numeric suite, TRUE for the mixed suite.
"""

from __future__ import annotations

import os
import time

from run_linear_regression_evaluation import (
    _submit_synreg,
    _wait_done,
    _wait_job_group,
    _ensure_compute_pool_usable,
    _synreg_shard_env,
    _run_autogluon_shards_single_wave,
    _stage_file_exists,
    _resolve_positive_int_runtime_param,
    _spcs_run_id,
    _spcs_dns_domain,
    _spcs_session_context_env,
    _execute_spcs_job_service,
    _build_spcs_job_spec,
    _wait_spcs_job_group,
    _cancel_spcs_job_group,
    _verify_spcs_image_in_repository,
    _resolve_combined_autogluon_execution_plan,
    _resolve_runtime_string_param,
    _resolve_baseline_shard_count,
    _spcs_ray_port_env_vars,
    _spcs_ray_coordinator_endpoints,
    _spcs_ray_worker_endpoints,
    DEEPSET_GPU_POOL,
    DEEPSET_CPU_POOL,
    AUTOGLUON_CPU_POOL,
    SYNREG_GPU_SHARDS,
    SYNREG_CPU_SHARDS,
    SYNREG_AUTOGLUON_SHARDS,
    SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT,
    SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT,
    SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT,
    SYNREG_AUTOGLUON_MAX_IN_FLIGHT_DEFAULT,
    SYNREG_RAY_EVALUATION_READY_TIMEOUT_SECONDS_DEFAULT,
    SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS_DEFAULT,
    SYNREG_DEEPSET_CKPT_STAGE,
    SYNREG_BASELINE_PIP,
    SYNREG_AG_PIP,
    SYNREG_AG_RAY_PIP,
    SYNREG_AUTOGLUON_SPCS_IMAGE,
    SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT,
    SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE,
    SYNREG_SPCS_WORKER_SUBMIT_STAGGER_SECONDS,
    SYNREG_SPCS_WORKER_PLACEMENT_TIMEOUT_SECONDS,
    SYNREG_AUTOGLUON_SPCS_RAY_START_TIMEOUT_SECONDS,
    _wait_spcs_workers_ready,
    _spcs_worker_connect_timeout,
    SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT,
    SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT,
    SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT,
    SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT,
    SYNREG_PYPI_EAI,
    SPCS_RAY_COORDINATOR_RESOURCES,
    SPCS_RAY_WORKER_RESOURCES,
    SPCS_SINGLE_NODE_RESOURCES,
    _SYNREG_COORDINATOR_OBJ_STORE_ENV,
    _SYNREG_WORKER_OBJ_STORE_ENV,
)

# ---------------------------------------------------------------------------
# nonlinear_v2 suite constants
# ---------------------------------------------------------------------------

NONLINEAR_SUITE_ID    = "nonlinear"
NONLINEAR_INDEX_TABLE = "NONLINEAR_REGRESSION_DATASET_INDEX"
NONLINEAR_N_DATASETS  = 420
NONLINEAR_GPU_SHARDS  = SYNREG_GPU_SHARDS          # 10
NONLINEAR_PARTS_PREFIX = "@EVALUATION_RESULTS_STAGE/nonlinear/regression/numeric/nonlinear"
NONLINEAR_OUTPUT_STAGE = "@EVALUATION_RESULTS_STAGE/nonlinear/regression/numeric"

# Mixed-categorical nonlinear regression eval constants
NONLINEAR_MIXED_SUITE_ID    = "nonlinear_mixed_regression"
NONLINEAR_MIXED_INDEX_TABLE = "NONLINEAR_MIXED_REGRESSION_DATASET_INDEX"
NONLINEAR_MIXED_PARTS_PREFIX = "@EVALUATION_RESULTS_STAGE/nonlinear/regression/mixed/nonlinear_mixed_regression"
NONLINEAR_MIXED_OUTPUT_STAGE = "@EVALUATION_RESULTS_STAGE/nonlinear/regression/mixed"

# Env var injected into every job to redirect index queries to the nonlinear table
_NONLINEAR_INDEX_ENV  = {"SYNREG_INDEX_TABLE": NONLINEAR_INDEX_TABLE}

# Mixed-categorical env dict — overrides index table and enables the flag in evaluators
_NONLINEAR_MIXED_INDEX_ENV = {
    "SYNREG_INDEX_TABLE": NONLINEAR_MIXED_INDEX_TABLE,
    "SYNREG_IS_MIXED_CATEGORICAL": "true",
}


def _nonlinear_regression_suite_params(is_mixed_categorical: bool) -> dict:
    """Return the 4 suite-level variables that differ between standard and mixed evals."""
    if is_mixed_categorical:
        return {
            "suite_id":     NONLINEAR_MIXED_SUITE_ID,
            "parts_prefix": NONLINEAR_MIXED_PARTS_PREFIX,
            "output_stage": NONLINEAR_MIXED_OUTPUT_STAGE,
            "idx_env":      _NONLINEAR_MIXED_INDEX_ENV,
        }
    return {
        "suite_id":     NONLINEAR_SUITE_ID,
        "parts_prefix": NONLINEAR_PARTS_PREFIX,
        "output_stage": NONLINEAR_OUTPUT_STAGE,
        "idx_env":      _NONLINEAR_INDEX_ENV,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Prep (index creation + population)
# ---------------------------------------------------------------------------

def run_nonlinear_regression_prep(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """Phase 1: Index nonlinear_v2 datasets into the appropriate NONLINEAR_*_DATASET_INDEX."""
    _sp = _nonlinear_regression_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(f"[INFO] nonlinear_v2 Phase 1: prep suite_id={_sp['suite_id']} …", flush=True)
    env_vars = {
        "NONLINEAR_SUITE_ID":   _sp["suite_id"],
        "NONLINEAR_N_DATASETS": str(NONLINEAR_N_DATASETS),
        **_sp["idx_env"],
    }
    job = _submit_synreg(
        session=session,
        label="nonlinear_prep",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars=env_vars,
        runtime_environment=bench_rt,
        entrypoint="prepare_nonlinear_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(job, label="nonlinear_prep", session=session)
    return f"run_nonlinear_regression_prep: ok suite_id={_sp['suite_id']}"


# ---------------------------------------------------------------------------
# Phase 2 — DeepSet GPU evaluation
# ---------------------------------------------------------------------------

def run_nonlinear_regression_deepset_evaluation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """Phase 2: DeepSet — 10 GPU shards on DEEPSET_GPU_POOL."""
    _sp = _nonlinear_regression_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)

    _ckpt_filename = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[-1]
    _ckpt_dir      = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[0] + "/"
    if not _stage_file_exists(session, _ckpt_dir, _ckpt_filename):
        raise RuntimeError(
            f"[run_nonlinear_regression_deepset_evaluation] Checkpoint not found: "
            f"{SYNREG_DEEPSET_CKPT_STAGE!r}. "
            f"Verify with: LIST {_ckpt_dir}; — upload before running DeepSet evaluation."
        )

    print(
        f"[INFO] nonlinear_v2 Phase 2: DeepSet suite_id={_sp['suite_id']} "
        f"({NONLINEAR_GPU_SHARDS} shards) …",
        flush=True,
    )
    deepset_jobs = [
        _submit_synreg(
            session=session,
            label=f"nonlinear_deepset_shard_{i}",
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=_synreg_shard_env(
                mode="deepset",
                suite_id=_sp["suite_id"],
                num_shards=NONLINEAR_GPU_SHARDS,
                shard_index=i,
                results_stage=_sp["parts_prefix"],
                extra_env={
                    "MC_K": "8",
                    "SYNTHETIC_REGRESSION_CONTEXT_SIZE": "200",
                    "SYNTHETIC_REGRESSION_CONTEXT_ENSEMBLES": "5",
                    "SYNTHETIC_REGRESSION_TEST_BATCH_SIZE": "128",
                    "SYNTHETIC_REGRESSION_FEATURE_SELECTOR": "train_f_regression",
                    "BENCHMARK_REQUIRE_CUDA": "true",
                    "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH": SYNREG_DEEPSET_CKPT_STAGE,
                    "SYNREG_RUN_CHECKPOINT_GATES": "true",
                    "SYNREG_CHECKPOINT_GATE_STRICT": "true",
                    **_sp["idx_env"],
                },
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_nonlinear_regression.py",
            target_instances=1,
            pip_requirements=None,
            external_access_integrations=None,
        )
        for i in range(NONLINEAR_GPU_SHARDS)
    ]
    _wait_job_group(
        [(f"nonlinear_deepset_shard_{i}", job) for i, job in enumerate(deepset_jobs)],
        session,
    )
    return (
        f"run_nonlinear_regression_deepset_evaluation: ok "
        f"suite_id={_sp['suite_id']} shards={NONLINEAR_GPU_SHARDS}"
    )


# ---------------------------------------------------------------------------
# Phase 3 — Baseline CPU evaluation
# ---------------------------------------------------------------------------

def run_nonlinear_regression_baseline_evaluation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
    baseline_shards=None,
    baseline_concurrent_nodes=None,
) -> str:
    """Phase 3: Baselines — CPU shards on DEEPSET_CPU_POOL."""
    _sp = _nonlinear_regression_suite_params(is_mixed_categorical)
    proc = "run_nonlinear_regression_baseline_evaluation"
    shard_count = _resolve_baseline_shard_count(proc, baseline_shards)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(
        f"[INFO] nonlinear_v2 Phase 3: baselines suite_id={_sp['suite_id']} "
        f"({shard_count} shards) …",
        flush=True,
    )
    jobs = []
    for i in range(shard_count):
        lbl = f"nonlinear_baseline_shard_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=_sp["suite_id"],
                num_shards=shard_count,
                shard_index=i,
                results_stage=_sp["parts_prefix"],
                extra_env={
                    **_sp["idx_env"],
                    "SYNREG_NONLINEAR_BASELINES": "true",
                },
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_nonlinear_regression.py",
            target_instances=1,
            pip_requirements=SYNREG_BASELINE_PIP,
            external_access_integrations=SYNREG_PYPI_EAI,
        )
        jobs.append((lbl, job))
    _wait_job_group(jobs, session)
    return (
        f"run_nonlinear_regression_baseline_evaluation: ok "
        f"suite_id={_sp['suite_id']} shards={shard_count}"
    )


# ---------------------------------------------------------------------------
# Phase 4 — AutoGluon SPCS evaluation
# ---------------------------------------------------------------------------

def run_nonlinear_regression_autogluon_evaluation(
    session,
    is_mixed_categorical: bool,
    ag_rt: str = "spcs_job",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_concurrent_clusters=None,
    autogluon_task_cpus=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
    ray_ready_timeout_seconds=None,
    worker_submit_stagger_seconds=None,
) -> str:
    """AutoGluon evaluation using SPCS custom-image backend."""
    _sp  = _nonlinear_regression_suite_params(is_mixed_categorical)
    proc = "run_nonlinear_regression_autogluon_evaluation"

    image = (
        str(ag_rt).strip()
        if (ag_rt and str(ag_rt).strip().lower() != "spcs_job")
        else SYNREG_AUTOGLUON_SPCS_IMAGE
    )

    if autogluon_cluster_shards and autogluon_cluster_shards > 0:
        _concurrent = autogluon_concurrent_clusters or autogluon_cluster_shards
        if _concurrent > 0 and _concurrent != autogluon_cluster_shards:
            raise ValueError(
                f"In Ray mode, autogluon_concurrent_clusters ({_concurrent}) "
                f"must equal autogluon_cluster_shards ({autogluon_cluster_shards})."
            )

    plan = _resolve_combined_autogluon_execution_plan(
        procedure_name=proc,
        cluster_shards_arg=autogluon_cluster_shards,
        workers_per_shard_arg=autogluon_workers_per_shard,
        concurrent_clusters_arg=autogluon_concurrent_clusters,
    )
    task_cpus = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_TASK_CPUS",
        sql_arg=autogluon_task_cpus, env_var="AUTOGLUON_TASK_CPUS",
        default=SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT,
    )
    time_limit = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_TIME_LIMIT",
        sql_arg=autogluon_time_limit, env_var="AUTOGLUON_TIME_LIMIT",
        default=300,
    )
    presets = _resolve_runtime_string_param(
        procedure_name=proc, name="AUTOGLUON_PRESETS",
        sql_arg=autogluon_presets, env_var="AUTOGLUON_PRESETS",
        default="best_quality",
    )

    ag_min_tmp           = os.getenv("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", str(SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT))
    ag_max_features      = os.getenv("BENCHMARK_CPU_MAX_PROCESSED_FEATURES",   str(SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT))
    ag_max_matrix_bytes  = os.getenv("BENCHMARK_CPU_MAX_MATRIX_BYTES",         str(SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT))
    ag_max_dataset_bytes = os.getenv("BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES",  str(SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT))
    worker_access_mode   = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE",         "driver_presigned_url")
    max_work_item_bytes  = os.getenv("SYNREG_MAX_WORK_ITEM_BYTES",             "8192")
    presigned_url_expiry_seconds        = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_SECONDS",        "86400")
    presigned_url_expiry_policy         = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_POLICY",         "strict")
    presigned_url_expiry_buffer_seconds = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_BUFFER_SECONDS", "3600")

    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    _verify_spcs_image_in_repository(session, image)
    run_id  = _spcs_run_id()
    _stagger = (
        worker_submit_stagger_seconds
        if worker_submit_stagger_seconds is not None
        else SYNREG_SPCS_WORKER_SUBMIT_STAGGER_SECONDS
    )

    if plan.mode == "single_node_shards":
        output_shards = plan.output_shards
        print(
            f"[INFO] {proc}: suite_id={_sp['suite_id']} backend=spcs_job "
            f"mode=single_node_shards output_shards={output_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}",
            flush=True,
        )
        jobs = []
        for shard_index in range(output_shards):
            lbl = f"spcs_nl_ag_shard_{run_id}_{shard_index}"
            full_env = {"HOME": "/tmp"}
            full_env.update(_spcs_session_context_env(session))
            full_env.update(
                _synreg_shard_env(
                    mode="autogluon",
                    suite_id=_sp["suite_id"],
                    num_shards=output_shards,
                    shard_index=shard_index,
                    results_stage=_sp["parts_prefix"],
                    extra_env={
                        "AUTOGLUON_TIME_LIMIT": str(time_limit),
                        "AUTOGLUON_PRESETS": presets,
                        "AUTOGLUON_TASK_CPUS": str(task_cpus),
                        "SYNREG_EXPECTED_AG_SHARDS": str(output_shards),
                        "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                        "BENCHMARK_CPU_MAX_PROCESSED_FEATURES": ag_max_features,
                        "BENCHMARK_CPU_MAX_MATRIX_BYTES": ag_max_matrix_bytes,
                        "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES": ag_max_dataset_bytes,
                        **_sp["idx_env"],
                    },
                )
            )
            spec = _build_spcs_job_spec(
                image=image,
                args=["/app/scripts/evaluate_nonlinear_regression.py"],
                env_vars=full_env,
                resource_role=SPCS_SINGLE_NODE_RESOURCES,
            )
            job_name = _execute_spcs_job_service(
                session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                spec=spec,
            )
            jobs.append((lbl, job_name))
        _wait_spcs_job_group(jobs, session)
        return (
            f"{proc}: ok backend=spcs_job suite_id={_sp['suite_id']} "
            f"mode=single_node_shards output_shards={output_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )

    else:
        # Ray distributed mode
        cluster_shards    = plan.output_shards
        workers_per_shard = plan.workers_per_shard
        head_port         = SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT
        print(
            f"[INFO] {proc}: suite_id={_sp['suite_id']} backend=spcs_job "
            f"mode=ray_clusters cluster_shards={cluster_shards} "
            f"workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}",
            flush=True,
        )
        support_jobs: list = []
        coordinator_jobs: list = []
        dns_suffix = _spcs_dns_domain(session)
        coord_obj_store  = os.getenv(_SYNREG_COORDINATOR_OBJ_STORE_ENV, "268435456")
        worker_obj_store = os.getenv(_SYNREG_WORKER_OBJ_STORE_ENV, "268435456")

        _eval_ok = False
        try:
            for shard_index in range(cluster_shards):
                coord_label      = f"spcs_nl_ray_coord_{run_id}_{shard_index}"
                safe_coord_label = "".join(c if c.isalnum() else "_" for c in coord_label).upper()
                dns_service_name = safe_coord_label.lower().replace("_", "-")
                coord_hostname   = (
                    f"{dns_service_name}.{dns_suffix}" if dns_suffix else dns_service_name
                )
                head_address = f"{coord_hostname}:{head_port}"

                coord_env = {"HOME": "/tmp"}
                coord_env.update(_spcs_session_context_env(session))
                coord_env.update(
                    _synreg_shard_env(
                        mode="autogluon",
                        suite_id=_sp["suite_id"],
                        num_shards=cluster_shards,
                        shard_index=shard_index,
                        results_stage=_sp["parts_prefix"],
                        extra_env={
                            _SYNREG_COORDINATOR_OBJ_STORE_ENV: coord_obj_store,
                            "SPCS_RAY_RUN_ID": run_id,
                            "SPCS_RAY_SHARD_INDEX": str(shard_index),
                            "SYNREG_AUTOGLUON_DISTRIBUTED_MODE": "ray_work_items",
                            "SYNREG_AUTOGLUON_CLUSTER_SHARDS": str(cluster_shards),
                            "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(workers_per_shard),
                            "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS": str(cluster_shards),
                            "AUTOGLUON_TIME_LIMIT": str(time_limit),
                            "AUTOGLUON_PRESETS": presets,
                            "AUTOGLUON_TASK_CPUS": str(task_cpus),
                            "SYNREG_WORKER_DATA_ACCESS_MODE": worker_access_mode,
                            "SYNREG_MAX_WORK_ITEM_BYTES": max_work_item_bytes,
                            "SYNREG_PRESIGNED_URL_EXPIRY_SECONDS": presigned_url_expiry_seconds,
                            "SYNREG_PRESIGNED_URL_EXPIRY_POLICY": presigned_url_expiry_policy,
                            "SYNREG_PRESIGNED_URL_EXPIRY_BUFFER_SECONDS": presigned_url_expiry_buffer_seconds,
                            "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                            "BENCHMARK_CPU_MAX_PROCESSED_FEATURES": ag_max_features,
                            "BENCHMARK_CPU_MAX_MATRIX_BYTES": ag_max_matrix_bytes,
                            "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES": ag_max_dataset_bytes,
                            **({"SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS": str(ray_ready_timeout_seconds)} if ray_ready_timeout_seconds is not None else {}),
                            **_spcs_ray_port_env_vars(),
                            **_sp["idx_env"],
                        },
                    )
                )

                worker_env_base = {
                    "HOME": "/tmp",
                    "RAY_HEAD_ADDRESS": head_address,
                    "AUTOGLUON_TASK_CPUS": str(task_cpus),
                    _SYNREG_WORKER_OBJ_STORE_ENV: worker_obj_store,
                    "SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS": str(_spcs_worker_connect_timeout()),
                    **_spcs_ray_port_env_vars(),
                }
                shard_workers: list = []
                for w in range(workers_per_shard):
                    w_label = f"spcs_nl_ray_worker_{run_id}_{shard_index}_{w}"
                    w_spec  = _build_spcs_job_spec(
                        image=image,
                        args=["/app/scripts/spcs_ray_worker.py"],
                        env_vars=worker_env_base,
                        resource_role=SPCS_RAY_WORKER_RESOURCES,
                        endpoints=_spcs_ray_worker_endpoints(),
                    )
                    w_job = _execute_spcs_job_service(
                        session,
                        label=w_label,
                        compute_pool=AUTOGLUON_CPU_POOL,
                        spec=w_spec,
                    )
                    shard_workers.append((w_label, w_job))
                    support_jobs.append((w_label, w_job))
                    if _stagger > 0 and w < workers_per_shard - 1:
                        time.sleep(_stagger)

                _wait_spcs_workers_ready(
                    session,
                    shard_workers,
                    timeout_seconds=SYNREG_SPCS_WORKER_PLACEMENT_TIMEOUT_SECONDS,
                )

                coord_spec = _build_spcs_job_spec(
                    image=image,
                    args=["/app/scripts/spcs_ray_coordinator.py"],
                    env_vars=coord_env,
                    resource_role=SPCS_RAY_COORDINATOR_RESOURCES,
                    endpoints=_spcs_ray_coordinator_endpoints(head_port),
                )
                coord_job = _execute_spcs_job_service(
                    session,
                    label=coord_label,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    spec=coord_spec,
                )
                coordinator_jobs.append((coord_label, coord_job))

            _wait_spcs_job_group(coordinator_jobs, session)
            _eval_ok = True
        finally:
            if not _eval_ok and SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE:
                _keep_labels = [lbl for lbl, _ in support_jobs]
                print(
                    f"[WARNING] SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE=true: "
                    f"coordinator jobs failed; support jobs left running: {_keep_labels}.",
                    flush=True,
                )
            else:
                _cancel_spcs_job_group(support_jobs, session)

        return (
            f"{proc}: ok backend=spcs_job suite_id={_sp['suite_id']} "
            f"mode=ray_clusters cluster_shards={cluster_shards} "
            f"workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )


def run_nonlinear_regression_autogluon_evaluation_default(
    session,
    is_mixed_categorical: bool,
    ag_image: str,
    autogluon_cluster_shards: int = 0,
    autogluon_workers_per_shard: int = 1,
    autogluon_concurrent_clusters: int = 0,
) -> str:
    """SQL handler: 5-arg overload (IS_MIXED_CATEGORICAL + 4 AG params)."""
    return run_nonlinear_regression_autogluon_evaluation(
        session,
        is_mixed_categorical=is_mixed_categorical,
        ag_rt=ag_image,
        autogluon_cluster_shards=autogluon_cluster_shards,
        autogluon_workers_per_shard=autogluon_workers_per_shard,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters or autogluon_cluster_shards or None,
    )


def run_nonlinear_regression_autogluon_evaluation_full(
    session,
    is_mixed_categorical: bool,
    ag_image: str,
    autogluon_cluster_shards: int,
    autogluon_workers_per_shard: int,
    autogluon_concurrent_clusters: int,
    autogluon_time_limit: int,
    autogluon_presets: str,
    autogluon_task_cpus: int,
    ray_ready_timeout_seconds: int,
    worker_submit_stagger_seconds: int,
) -> str:
    """SQL handler: 10-arg full overload (IS_MIXED_CATEGORICAL + 9 AG params)."""
    return run_nonlinear_regression_autogluon_evaluation(
        session,
        is_mixed_categorical=is_mixed_categorical,
        ag_rt=ag_image,
        autogluon_cluster_shards=autogluon_cluster_shards,
        autogluon_workers_per_shard=autogluon_workers_per_shard,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters,
        autogluon_time_limit=autogluon_time_limit,
        autogluon_presets=autogluon_presets,
        autogluon_task_cpus=autogluon_task_cpus,
        ray_ready_timeout_seconds=ray_ready_timeout_seconds,
        worker_submit_stagger_seconds=worker_submit_stagger_seconds,
    )


# ---------------------------------------------------------------------------
# Phase 5 — Aggregation
# ---------------------------------------------------------------------------

def run_nonlinear_regression_aggregation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
    expected_deepset_shards=None,
    expected_baseline_shards=None,
    expected_ag_shards=None,
) -> str:
    """Phase 5: Aggregation — 1 CPU job; outputs to the suite's output stage."""
    _sp  = _nonlinear_regression_suite_params(is_mixed_categorical)
    proc = "run_nonlinear_regression_aggregation"
    resolved_deepset = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="EXPECTED_DEEPSET_SHARDS",
        sql_arg=expected_deepset_shards, env_var="SYNREG_EXPECTED_DEEPSET_SHARDS",
        default=SYNREG_GPU_SHARDS,
    )
    resolved_baseline = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="EXPECTED_BASELINE_SHARDS",
        sql_arg=expected_baseline_shards, env_var="SYNREG_EXPECTED_BASELINE_SHARDS",
        default=SYNREG_CPU_SHARDS,
    )
    resolved_ag = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="EXPECTED_AG_SHARDS",
        sql_arg=expected_ag_shards, env_var="SYNREG_EXPECTED_AG_SHARDS",
        default=SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT,
    )
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(
        f"[INFO] nonlinear_v2 Phase 5: aggregation suite_id={_sp['suite_id']} "
        f"(expected deepset={resolved_deepset} baseline={resolved_baseline} "
        f"ag={resolved_ag}) …",
        flush=True,
    )
    agg_job = _submit_synreg(
        session=session,
        label="nonlinear_aggregate",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_MODE":       "aggregate",
            "SYNTHETIC_REGRESSION_SUITE_ID":   _sp["suite_id"],
            "SYNREG_RESULTS_STAGE":            _sp["parts_prefix"],
            "SYNREG_OUTPUT_STAGE":             _sp["output_stage"],
            "SYNREG_EXPECTED_DEEPSET_SHARDS":  str(resolved_deepset),
            "SYNREG_EXPECTED_BASELINE_SHARDS": str(resolved_baseline),
            "SYNREG_EXPECTED_AG_SHARDS":       str(resolved_ag),
            **_sp["idx_env"],
        },
        runtime_environment=bench_rt,
        entrypoint="evaluate_nonlinear_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(agg_job, label="nonlinear_aggregate", session=session)
    return (
        f"run_nonlinear_regression_aggregation: ok "
        f"suite_id={_sp['suite_id']} output={_sp['output_stage']} "
        f"expected_deepset={resolved_deepset} "
        f"expected_baseline={resolved_baseline} "
        f"expected_ag={resolved_ag}"
    )


def run_nonlinear_regression_aggregation_default(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """SQL handler: 2-arg overload (IS_MIXED_CATEGORICAL + bench_rt)."""
    return run_nonlinear_regression_aggregation(session, is_mixed_categorical, bench_rt)


def run_nonlinear_regression_aggregation_full(
    session,
    is_mixed_categorical: bool,
    bench_rt: str,
    expected_deepset_shards: int,
    expected_baseline_shards: int,
    expected_ag_shards: int,
) -> str:
    """SQL handler: 5-arg overload (IS_MIXED_CATEGORICAL + bench_rt + 3 shard counts)."""
    return run_nonlinear_regression_aggregation(
        session, is_mixed_categorical, bench_rt,
        expected_deepset_shards=expected_deepset_shards,
        expected_baseline_shards=expected_baseline_shards,
        expected_ag_shards=expected_ag_shards,
    )


# ---------------------------------------------------------------------------
# SPCS probe functions
# ---------------------------------------------------------------------------

def run_nonlinear_regression_autogluon_spcs_import_probe(
    session,
    ag_image: str,
    autogluon_cluster_shards: int = 1,
) -> str:
    """Probe: test AG import in SPCS container."""
    _verify_spcs_image_in_repository(session, ag_image)
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    run_id  = _spcs_run_id()
    lbl = f"spcs_nl_import_probe_{run_id}"
    full_env = {"HOME": "/tmp"}
    full_env.update(_spcs_session_context_env(session))
    full_env["SYNREG_PROBE_MODE"] = "import"
    spec = _build_spcs_job_spec(
        image=ag_image,
        args=["/app/scripts/autogluon_import_timing_probe.py"],
        env_vars=full_env,
        resource_role=SPCS_SINGLE_NODE_RESOURCES,
    )
    job_name = _execute_spcs_job_service(
        session, label=lbl, compute_pool=AUTOGLUON_CPU_POOL, spec=spec,
    )
    _wait_spcs_job_group([(lbl, job_name)], session)
    return f"run_nonlinear_regression_autogluon_spcs_import_probe: ok image={ag_image!r}"


def run_nonlinear_regression_autogluon_spcs_session_probe(
    session,
    ag_image: str,
    autogluon_cluster_shards: int = 1,
) -> str:
    """Probe: test Snowpark session context in SPCS container."""
    _verify_spcs_image_in_repository(session, ag_image)
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    run_id = _spcs_run_id()
    lbl = f"spcs_nl_session_probe_{run_id}"
    full_env = {"HOME": "/tmp"}
    full_env.update(_spcs_session_context_env(session))
    spec = _build_spcs_job_spec(
        image=ag_image,
        args=["/app/scripts/spcs_snowpark_session_probe.py"],
        env_vars=full_env,
        resource_role=SPCS_SINGLE_NODE_RESOURCES,
    )
    job_name = _execute_spcs_job_service(
        session, label=lbl, compute_pool=AUTOGLUON_CPU_POOL, spec=spec,
    )
    _wait_spcs_job_group([(lbl, job_name)], session)
    return f"run_nonlinear_regression_autogluon_spcs_session_probe: ok image={ag_image!r}"


def run_nonlinear_regression_autogluon_spcs_capacity_probe(
    session,
    ag_image: str,
    autogluon_cluster_shards: int,
    autogluon_workers_per_shard: int,
    autogluon_concurrent_clusters: int,
) -> str:
    """Probe: test SPCS cluster capacity."""
    _verify_spcs_image_in_repository(session, ag_image)
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    run_id = _spcs_run_id()
    lbl = f"spcs_nl_capacity_probe_{run_id}"
    full_env = {"HOME": "/tmp"}
    full_env.update(_spcs_session_context_env(session))
    full_env.update({
        "SYNREG_AUTOGLUON_CLUSTER_SHARDS": str(autogluon_cluster_shards),
        "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(autogluon_workers_per_shard),
        "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS": str(autogluon_concurrent_clusters),
    })
    spec = _build_spcs_job_spec(
        image=ag_image,
        args=["/app/scripts/capacity_probe.py"],
        env_vars=full_env,
        resource_role=SPCS_SINGLE_NODE_RESOURCES,
    )
    job_name = _execute_spcs_job_service(
        session, label=lbl, compute_pool=AUTOGLUON_CPU_POOL, spec=spec,
    )
    _wait_spcs_job_group([(lbl, job_name)], session)
    return f"run_nonlinear_regression_autogluon_spcs_capacity_probe: ok image={ag_image!r}"


def run_nonlinear_regression_autogluon_spcs_worker_access_probe(
    session,
    ag_image: str,
    autogluon_cluster_shards: int,
    autogluon_workers_per_shard: int,
    autogluon_concurrent_clusters: int,
) -> str:
    """Probe: test SPCS worker data-access path."""
    _verify_spcs_image_in_repository(session, ag_image)
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    run_id = _spcs_run_id()
    lbl = f"spcs_nl_worker_access_probe_{run_id}"
    full_env = {"HOME": "/tmp"}
    full_env.update(_spcs_session_context_env(session))
    full_env.update({
        "SYNREG_AUTOGLUON_CLUSTER_SHARDS": str(autogluon_cluster_shards),
        "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(autogluon_workers_per_shard),
        "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS": str(autogluon_concurrent_clusters),
        **_NONLINEAR_INDEX_ENV,
    })
    spec = _build_spcs_job_spec(
        image=ag_image,
        args=["/app/scripts/autogluon_worker_access_probe.py"],
        env_vars=full_env,
        resource_role=SPCS_SINGLE_NODE_RESOURCES,
    )
    job_name = _execute_spcs_job_service(
        session, label=lbl, compute_pool=AUTOGLUON_CPU_POOL, spec=spec,
    )
    _wait_spcs_job_group([(lbl, job_name)], session)
    return f"run_nonlinear_regression_autogluon_spcs_worker_access_probe: ok image={ag_image!r}"

