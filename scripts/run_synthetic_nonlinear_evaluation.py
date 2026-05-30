"""
run_synthetic_nonlinear_evaluation.py
=======================================
Orchestration handlers for the nonlinear synthetic regression evaluation suite
(nonlinear_v1, regimes I–L).

Imports shared helpers from run_synthetic_regression_evaluation rather than
duplicating them. Every job submission injects
  SYNREG_INDEX_TABLE=SYNTHETIC_NONLINEAR_DATASET_INDEX
so that evaluate_synthetic_regression.py and autogluon_ray.py query the
nonlinear index table, leaving SYNTHETIC_REGRESSION_DATASET_INDEX untouched.

Stored procedure handlers (all callable from SQL):
  run_synthetic_nonlinear_prep
  run_synthetic_nonlinear_deepset_evaluation
  run_synthetic_nonlinear_baseline_evaluation       (0/1/2 extra-arg overloads)
  run_synthetic_nonlinear_autogluon_spcs_evaluation (4-arg and 9-arg overloads)
  run_synthetic_nonlinear_aggregation
"""

from __future__ import annotations

import os
import time

from run_synthetic_regression_evaluation import (
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
# Nonlinear suite constants
# ---------------------------------------------------------------------------

NONLINEAR_SUITE_ID      = "nonlinear_v1"
NONLINEAR_INDEX_TABLE   = "SYNTHETIC_NONLINEAR_DATASET_INDEX"
NONLINEAR_N_DATASETS    = 400
NONLINEAR_GPU_SHARDS    = SYNREG_GPU_SHARDS          # 10
NONLINEAR_PARTS_PREFIX  = f"@EVALUATION_RESULTS_STAGE/regression/{NONLINEAR_SUITE_ID}"
NONLINEAR_OUTPUT_STAGE  = "@EVALUATION_RESULTS_STAGE/nonlinear"

# Env var injected into every job to redirect index queries to the nonlinear table
_NONLINEAR_INDEX_ENV    = {"SYNREG_INDEX_TABLE": NONLINEAR_INDEX_TABLE}


# ---------------------------------------------------------------------------
# Phase 1 — Prep (index creation + population)
# ---------------------------------------------------------------------------

def run_synthetic_nonlinear_prep(
    session,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """Phase 1: Index 400 nonlinear datasets into SYNTHETIC_NONLINEAR_DATASET_INDEX."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print("[INFO] Nonlinear Phase 1: prep …", flush=True)
    env_vars = {
        "NONLINEAR_SUITE_ID":   NONLINEAR_SUITE_ID,
        "NONLINEAR_N_DATASETS": str(NONLINEAR_N_DATASETS),
        **_NONLINEAR_INDEX_ENV,
    }
    job = _submit_synreg(
        session=session,
        label="nonlinear_prep",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars=env_vars,
        runtime_environment=bench_rt,
        entrypoint="evaluate_synthetic_nonlinear.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(job, label="nonlinear_prep", session=session)
    return f"run_synthetic_nonlinear_prep: ok suite_id={NONLINEAR_SUITE_ID}"


# ---------------------------------------------------------------------------
# Phase 2 — DeepSet GPU evaluation
# ---------------------------------------------------------------------------

def run_synthetic_nonlinear_deepset_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """Phase 2: DeepSet — 10 GPU shards on DEEPSET_GPU_POOL."""
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)

    # Preflight: verify checkpoint exists before wasting GPU quota.
    _ckpt_filename = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[-1]
    _ckpt_dir      = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[0] + "/"
    if not _stage_file_exists(session, _ckpt_dir, _ckpt_filename):
        raise RuntimeError(
            f"[run_synthetic_nonlinear_deepset_evaluation] Checkpoint not found: "
            f"{SYNREG_DEEPSET_CKPT_STAGE!r}. "
            f"Verify with: LIST {_ckpt_dir}; — upload checkpoint before running DeepSet evaluation."
        )

    print(f"[INFO] Nonlinear Phase 2: DeepSet ({NONLINEAR_GPU_SHARDS} shards) …", flush=True)
    deepset_jobs = [
        _submit_synreg(
            session=session,
            label=f"nonlinear_deepset_shard_{i}",
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=_synreg_shard_env(
                mode="deepset",
                suite_id=NONLINEAR_SUITE_ID,
                num_shards=NONLINEAR_GPU_SHARDS,
                shard_index=i,
                results_stage=NONLINEAR_PARTS_PREFIX,
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
                    **_NONLINEAR_INDEX_ENV,
                },
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
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
    return f"run_synthetic_nonlinear_deepset_evaluation: ok shards={NONLINEAR_GPU_SHARDS}"


# ---------------------------------------------------------------------------
# Phase 3 — Baseline CPU evaluation
# ---------------------------------------------------------------------------

def run_synthetic_nonlinear_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    baseline_shards=None,
    baseline_concurrent_nodes=None,
) -> str:
    """Phase 3: Baselines — CPU shards on DEEPSET_CPU_POOL.

    Mirrors the OOD full baseline pattern (inline, does not call
    _run_baseline_shards_single_wave from the combined module).
    """
    proc = "run_synthetic_nonlinear_baseline_evaluation"
    shard_count = _resolve_baseline_shard_count(proc, baseline_shards)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(f"[INFO] Nonlinear Phase 3: baselines ({shard_count} shards) …", flush=True)
    jobs = []
    for i in range(shard_count):
        lbl = f"nonlinear_baseline_shard_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=NONLINEAR_SUITE_ID,
                num_shards=shard_count,
                shard_index=i,
                results_stage=NONLINEAR_PARTS_PREFIX,
                extra_env=_NONLINEAR_INDEX_ENV,
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=SYNREG_BASELINE_PIP,
            external_access_integrations=SYNREG_PYPI_EAI,
        )
        jobs.append((lbl, job))
    _wait_job_group(jobs, session)
    return (
        f"run_synthetic_nonlinear_baseline_evaluation: ok "
        f"suite_id={NONLINEAR_SUITE_ID} shards={shard_count}"
    )


def run_synthetic_nonlinear_baseline_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """SQL handler: 1-arg overload (bench_rt only)."""
    return run_synthetic_nonlinear_baseline_evaluation(session, bench_rt)


def run_synthetic_nonlinear_baseline_evaluation_with_shards(
    session,
    bench_rt: str = "2.5.0-py311",
    baseline_shards: int = SYNREG_CPU_SHARDS,
) -> str:
    """SQL handler: 2-arg overload (bench_rt, baseline_shards)."""
    return run_synthetic_nonlinear_baseline_evaluation(
        session, bench_rt, baseline_shards=baseline_shards
    )


def run_synthetic_nonlinear_baseline_evaluation_with_shards_and_concurrency(
    session,
    bench_rt: str = "2.5.0-py311",
    baseline_shards: int = SYNREG_CPU_SHARDS,
    baseline_concurrent_nodes: int = SYNREG_CPU_SHARDS,
) -> str:
    """SQL handler: 3-arg overload (bench_rt, baseline_shards, concurrent_nodes)."""
    return run_synthetic_nonlinear_baseline_evaluation(
        session, bench_rt,
        baseline_shards=baseline_shards,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )


# ---------------------------------------------------------------------------
# Phase 4 — AutoGluon SPCS evaluation (single-node and Ray distributed)
# ---------------------------------------------------------------------------

def run_synthetic_nonlinear_autogluon_spcs_evaluation(
    session,
    ag_rt: str = "spcs_job",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
    ray_ready_timeout_seconds=None,
    worker_submit_stagger_seconds=None,
) -> str:
    """AutoGluon evaluation using SPCS custom-image backend.

    Single-node mode (autogluon_cluster_shards == 0):
        One SPCS job service per shard. No Ray.
        Entrypoint: /app/src/evaluate_synthetic_regression.py

    Ray distributed mode (autogluon_cluster_shards > 0):
        Per cluster shard: one coordinator SPCS service + N worker SPCS services.
        Entrypoint: /app/scripts/spcs_ray_coordinator.py

    Injects SYNREG_INDEX_TABLE=SYNTHETIC_NONLINEAR_DATASET_INDEX in all coordinator
    and driver env vars so autogluon_ray.py queries the nonlinear index table.
    """
    proc = "run_synthetic_nonlinear_autogluon_spcs_evaluation"

    # Resolve image — reads SYNREG_AUTOGLUON_SPCS_IMAGE from this module's namespace
    # (patchable in tests via monkeypatch.setattr(mod, "SYNREG_AUTOGLUON_SPCS_IMAGE", ...))
    image = (
        str(ag_rt).strip()
        if (ag_rt and str(ag_rt).strip().lower() != "spcs_job")
        else SYNREG_AUTOGLUON_SPCS_IMAGE
    )

    # Ray-mode invariant: concurrent_clusters must equal cluster_shards (single-wave).
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

    ag_min_tmp            = os.getenv("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", str(SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT))
    ag_max_features       = os.getenv("BENCHMARK_CPU_MAX_PROCESSED_FEATURES",   str(SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT))
    ag_max_matrix_bytes   = os.getenv("BENCHMARK_CPU_MAX_MATRIX_BYTES",         str(SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT))
    ag_max_dataset_bytes  = os.getenv("BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES",  str(SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT))
    worker_access_mode    = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE",         "driver_presigned_url")
    max_work_item_bytes   = os.getenv("SYNREG_MAX_WORK_ITEM_BYTES",             "8192")
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
            f"[INFO] {proc}: suite_id={NONLINEAR_SUITE_ID} backend=spcs_job "
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
                    suite_id=NONLINEAR_SUITE_ID,
                    num_shards=output_shards,
                    shard_index=shard_index,
                    results_stage=NONLINEAR_PARTS_PREFIX,
                    extra_env={
                        "AUTOGLUON_TIME_LIMIT": str(time_limit),
                        "AUTOGLUON_PRESETS": presets,
                        "AUTOGLUON_TASK_CPUS": str(task_cpus),
                        "SYNREG_EXPECTED_AG_SHARDS": str(output_shards),
                        "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                        "BENCHMARK_CPU_MAX_PROCESSED_FEATURES": ag_max_features,
                        "BENCHMARK_CPU_MAX_MATRIX_BYTES": ag_max_matrix_bytes,
                        "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES": ag_max_dataset_bytes,
                        **_NONLINEAR_INDEX_ENV,
                    },
                )
            )
            spec = _build_spcs_job_spec(
                image=image,
                args=["/app/src/evaluate_synthetic_regression.py"],
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
            f"{proc}: ok backend=spcs_job suite_id={NONLINEAR_SUITE_ID} "
            f"mode=single_node_shards output_shards={output_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )

    else:
        # Ray distributed mode — coordinator topology
        cluster_shards    = plan.output_shards
        workers_per_shard = plan.workers_per_shard
        head_port         = SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT
        print(
            f"[INFO] {proc}: suite_id={NONLINEAR_SUITE_ID} backend=spcs_job "
            f"mode=ray_clusters cluster_shards={cluster_shards} "
            f"workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}",
            flush=True,
        )
        support_jobs: list = []
        coordinator_jobs: list = []
        dns_suffix = _spcs_dns_domain(session)
        coord_obj_store = os.getenv(_SYNREG_COORDINATOR_OBJ_STORE_ENV, "268435456")
        worker_obj_store = os.getenv(_SYNREG_WORKER_OBJ_STORE_ENV, "268435456")

        for shard_index in range(cluster_shards):
            coord_label       = f"spcs_nl_ray_coord_{run_id}_{shard_index}"
            safe_coord_label  = "".join(c if c.isalnum() else "_" for c in coord_label).upper()
            dns_service_name  = safe_coord_label.lower().replace("_", "-")
            coord_hostname    = (
                f"{dns_service_name}.{dns_suffix}" if dns_suffix else dns_service_name
            )
            head_address = f"{coord_hostname}:{head_port}"
            print(
                f"[INFO] {proc}: shard={shard_index} coord_label={coord_label!r} "
                f"head_address={head_address!r}",
                flush=True,
            )

            coord_env = {"HOME": "/tmp"}
            coord_env.update(_spcs_session_context_env(session))
            coord_env.update(
                _synreg_shard_env(
                    mode="autogluon",
                    suite_id=NONLINEAR_SUITE_ID,
                    num_shards=cluster_shards,
                    shard_index=shard_index,
                    results_stage=NONLINEAR_PARTS_PREFIX,
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
                        **_NONLINEAR_INDEX_ENV,
                    },
                )
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

            # Workers — connect to coordinator's DNS address
            worker_env_base = {
                "HOME": "/tmp",
                "RAY_HEAD_ADDRESS": head_address,
                "AUTOGLUON_TASK_CPUS": str(task_cpus),
                _SYNREG_WORKER_OBJ_STORE_ENV: worker_obj_store,
                "SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS": str(600),
                **_spcs_ray_port_env_vars(),
            }
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
                support_jobs.append((w_label, w_job))
                if _stagger > 0 and w < workers_per_shard - 1:
                    time.sleep(_stagger)

        _eval_ok = False
        try:
            _wait_spcs_job_group(coordinator_jobs, session)
            _eval_ok = True
        finally:
            if not _eval_ok and SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE:
                _keep_labels = [lbl for lbl, _ in support_jobs]
                print(
                    f"[WARNING] SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE=true: coordinator "
                    f"jobs failed; support jobs left running: {_keep_labels}.",
                    flush=True,
                )
            else:
                _cancel_spcs_job_group(support_jobs, session)
        return (
            f"{proc}: ok backend=spcs_job suite_id={NONLINEAR_SUITE_ID} "
            f"mode=ray_clusters cluster_shards={cluster_shards} "
            f"workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )


def run_synthetic_nonlinear_autogluon_spcs_evaluation_default(
    session,
    ag_image: str,
    autogluon_cluster_shards: int = 0,
    autogluon_workers_per_shard: int = 1,
    autogluon_concurrent_clusters: int = 0,
) -> str:
    """SQL handler: 4-arg overload (ag_image, cluster_shards, workers_per_shard, concurrent_clusters)."""
    return run_synthetic_nonlinear_autogluon_spcs_evaluation(
        session,
        ag_rt=ag_image,
        autogluon_cluster_shards=autogluon_cluster_shards,
        autogluon_workers_per_shard=autogluon_workers_per_shard,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters or autogluon_cluster_shards or None,
    )


def run_synthetic_nonlinear_autogluon_spcs_evaluation_full(
    session,
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
    """SQL handler: 9-arg full overload."""
    return run_synthetic_nonlinear_autogluon_spcs_evaluation(
        session,
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

def run_synthetic_nonlinear_aggregation(
    session,
    bench_rt: str = "2.5.0-py311",
    expected_deepset_shards=None,
    expected_baseline_shards=None,
    expected_ag_shards=None,
) -> str:
    """Phase 5: Aggregation — 1 CPU job; outputs to NONLINEAR_OUTPUT_STAGE."""
    proc = "run_synthetic_nonlinear_aggregation"
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
        f"[INFO] Nonlinear Phase 5: aggregation "
        f"(expected deepset={resolved_deepset} baseline={resolved_baseline} ag={resolved_ag}) …",
        flush=True,
    )
    agg_job = _submit_synreg(
        session=session,
        label="nonlinear_aggregate",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_MODE":       "aggregate",
            "SYNTHETIC_REGRESSION_SUITE_ID":   NONLINEAR_SUITE_ID,
            "SYNREG_RESULTS_STAGE":            NONLINEAR_PARTS_PREFIX,
            "SYNREG_OUTPUT_STAGE":             NONLINEAR_OUTPUT_STAGE,
            "SYNREG_EXPECTED_DEEPSET_SHARDS":  str(resolved_deepset),
            "SYNREG_EXPECTED_BASELINE_SHARDS": str(resolved_baseline),
            "SYNREG_EXPECTED_AG_SHARDS":       str(resolved_ag),
            **_NONLINEAR_INDEX_ENV,
        },
        runtime_environment=bench_rt,
        entrypoint="evaluate_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(agg_job, label="nonlinear_aggregate", session=session)
    return (
        f"run_synthetic_nonlinear_aggregation: ok output={NONLINEAR_OUTPUT_STAGE} "
        f"expected_deepset={resolved_deepset} expected_baseline={resolved_baseline} "
        f"expected_ag={resolved_ag}"
    )


def run_synthetic_nonlinear_aggregation_default(
    session,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """SQL handler: 1-arg overload."""
    return run_synthetic_nonlinear_aggregation(session, bench_rt)


def run_synthetic_nonlinear_aggregation_full(
    session,
    bench_rt: str,
    expected_deepset_shards: int,
    expected_baseline_shards: int,
    expected_ag_shards: int,
) -> str:
    """SQL handler: 4-arg overload."""
    return run_synthetic_nonlinear_aggregation(
        session, bench_rt,
        expected_deepset_shards=expected_deepset_shards,
        expected_baseline_shards=expected_baseline_shards,
        expected_ag_shards=expected_ag_shards,
    )
