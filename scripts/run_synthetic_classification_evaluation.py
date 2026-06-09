"""
run_synthetic_classification_evaluation.py
==========================================
Snowflake stored-procedure handler orchestrator for the linear classification
evaluation suite.

Mirrors run_synthetic_regression_evaluation.py patterns for classification.
All public handler functions are registered as Snowflake stored procedure
HANDLER targets in sql/synthetic_linear_pipeline.sql.

Path guard: every classification handler asserts SYNCLS_RESULTS_STAGE
contains '/classification/linear/'.

Handler functions (SQL HANDLER targets):
  run_synthetic_classification_linear_prep
  run_synthetic_classification_linear_deepset_evaluation
  run_synthetic_classification_linear_baseline_evaluation
  run_synthetic_classification_linear_autogluon_evaluation
  run_synthetic_classification_linear_aggregation
  run_synthetic_classification_linear_pipeline
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Pool / stage constants (reuse same pools as regression)
# ---------------------------------------------------------------------------

DEEPSET_GPU_POOL    = "DEEPSET_GPU_POOL"
DEEPSET_CPU_POOL    = "DEEPSET_CPU_POOL"
AUTOGLUON_CPU_POOL  = "AUTOGLUON_CPU_POOL"

SYNCLS_DEFAULT_SUITE_ID          = "linear_classification_stat_aware"
SYNCLS_DEFAULT_CKPT              = "@MODEL_STAGE/checkpoints/best_classification.pt"
SYNCLS_DEFAULT_FEATURE_SELECTOR  = "train_f_classif"

# Baseline pip requirements (catboost requires explicit pin)
_SYNCLS_BASELINE_PIP = ["catboost==1.2.10"]
_SYNCLS_AG_PIP       = ["autogluon.tabular==1.3.0"]
_SYNCLS_PYPI_EAI     = ["TABPFN_PYPI_EAI"]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _set_classification_linear_env(suite_id: str, *, is_mixed_categorical: bool = False) -> str:
    """
    Set env vars for classification-linear suite and return the results stage path.
    Must be called at the start of each classification-linear handler.
    When is_mixed_categorical=True, switches to the mixed-categorical data family
    and evaluation index table.
    """
    if is_mixed_categorical:
        results_stage = f"@EVALUATION_RESULTS_STAGE/linear/classification/mixed/{suite_id}"
        os.environ["TRAINING_DATA_FAMILY"] = "synthetic_linear_classification_mixed_categorical"
        os.environ["SYNCLS_INDEX_TABLE"] = "LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX"
        os.environ["SYNCLS_IS_MIXED_CATEGORICAL"] = "true"
    else:
        results_stage = f"@EVALUATION_RESULTS_STAGE/linear/classification/numeric/{suite_id}"
        os.environ["TRAINING_DATA_FAMILY"] = "synthetic_linear_classification"
        os.environ["SYNCLS_IS_MIXED_CATEGORICAL"] = "false"
        os.environ["SYNCLS_INDEX_TABLE"] = "LINEAR_CLASSIFICATION_DATASET_INDEX"
    os.environ["SYNTHETIC_CLASSIFICATION_SUITE_ID"]        = suite_id
    os.environ["SYNCLS_RESULTS_STAGE"]                     = results_stage
    os.environ["SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH"]     = SYNCLS_DEFAULT_CKPT
    os.environ["SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR"] = SYNCLS_DEFAULT_FEATURE_SELECTOR
    if is_mixed_categorical:
        assert "/linear/classification/mixed/" in results_stage, (
            f"SYNCLS_RESULTS_STAGE must contain '/linear/classification/mixed/', got: {results_stage!r}"
        )
    else:
        assert "/linear/classification/numeric/" in results_stage, (
            f"SYNCLS_RESULTS_STAGE must contain '/linear/classification/numeric/', got: {results_stage!r}"
        )
    return results_stage


def _resolve_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_str(value, default: str) -> str:
    if value is None:
        return default
    return str(value)


# ---------------------------------------------------------------------------
# Shared orchestration helpers (re-implemented to avoid circular imports)
# ---------------------------------------------------------------------------

def _ensure_compute_pool_usable(session, pool_name: str) -> None:
    """Resume compute pool if suspended."""
    try:
        session.sql(f"ALTER COMPUTE POOL {pool_name} RESUME IF SUSPENDED").collect()
    except Exception as exc:
        print(f"[WARN] Could not resume pool {pool_name}: {exc}", flush=True)


def _submit_cls_job(
    session,
    label: str,
    compute_pool: str,
    env_vars: dict,
    runtime_environment: str,
    entrypoint: str = "evaluate_linear_classification.py",
    target_instances: int = 1,
    pip_requirements=None,
    external_access_integrations=None,
):
    """Submit a classification evaluation MLJob."""
    try:
        from snowflake.ml.jobs import MLJob
        job = MLJob.submit_from_stage(
            session=session,
            name=label,
            compute_pool=compute_pool,
            stage_path=f"@MODEL_STAGE/scripts/{entrypoint}",
            runtime_environment=runtime_environment,
            env_vars=env_vars,
            target_instances=target_instances,
            pip_requirements=pip_requirements or [],
            external_access_integrations=external_access_integrations or [],
        )
    except Exception:
        # Fallback for environments where MLJob API differs slightly
        from snowflake.ml.jobs import submit_from_stage
        job = submit_from_stage(
            session=session,
            compute_pool=compute_pool,
            script_path=f"@MODEL_STAGE/scripts/{entrypoint}",
            env_vars=env_vars,
            name=label,
            num_instances=target_instances,
            pip_requirements=pip_requirements or [],
        )
    return job


def _wait_done(job, label: str, session) -> None:
    """Wait for a single job to complete. Raises RuntimeError on failure."""
    import time
    poll_interval = 15
    timeout = 7200
    elapsed = 0
    while elapsed < timeout:
        try:
            status = job.status if hasattr(job, "status") else "DONE"
            if status in ("DONE", "SUCCEEDED", "FAILED", "CANCELLED", "ERROR"):
                if status in ("FAILED", "CANCELLED", "ERROR"):
                    raise RuntimeError(f"Job {label!r} finished with status={status}.")
                return
        except AttributeError:
            return  # job has no status attr → assumed done
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Job {label!r} timed out after {timeout}s.")


def _wait_job_group(labeled_jobs: list, session) -> None:
    """Wait for all jobs in a [(label, job)] list."""
    for label, job in labeled_jobs:
        _wait_done(job, label, session)


# ---------------------------------------------------------------------------
# Shard submission helpers
# ---------------------------------------------------------------------------

def _run_baseline_shards(
    session,
    suite_id: str,
    results_stage: str,
    runtime_environment: str,
    baseline_shards: int,
    baseline_concurrent_nodes: int,
) -> int:
    """Submit baseline evaluation shards and wait for completion."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    jobs = []
    for shard_index in range(baseline_shards):
        label = f"cls_baseline_shard_{shard_index}"
        env = _classification_shard_env(
            mode="baselines",
            suite_id=suite_id,
            num_shards=baseline_shards,
            shard_index=shard_index,
            results_stage=results_stage,
        )
        job = _submit_cls_job(
            session=session,
            label=label,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=env,
            runtime_environment=runtime_environment,
            target_instances=1,
            pip_requirements=_SYNCLS_BASELINE_PIP,
            external_access_integrations=_SYNCLS_PYPI_EAI,
        )
        jobs.append((label, job))
        # Stagger concurrency
        if len(jobs) >= baseline_concurrent_nodes:
            _wait_job_group(jobs[:baseline_concurrent_nodes], session)
            jobs = jobs[baseline_concurrent_nodes:]

    if jobs:
        _wait_job_group(jobs, session)
    return baseline_shards


def _classification_shard_env(
    mode: str,
    suite_id: str,
    num_shards: int,
    shard_index: int,
    results_stage: str,
    extra_env: dict | None = None,
) -> dict:
    """Build env vars dict for a classification shard job."""
    import os as _os
    run_id = _os.getenv("SYNCLS_EVALUATION_RUN_ID", "")
    env = {
        "SYNTHETIC_CLASSIFICATION_MODE":            mode,
        "SYNTHETIC_CLASSIFICATION_SUITE_ID":        suite_id,
        "SYNTHETIC_CLASSIFICATION_NUM_SHARDS":      str(num_shards),
        "SYNTHETIC_CLASSIFICATION_SHARD_INDEX":     str(shard_index),
        "SYNCLS_RESULTS_STAGE":                     results_stage,
        "SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH":     SYNCLS_DEFAULT_CKPT,
        "SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR": SYNCLS_DEFAULT_FEATURE_SELECTOR,
        "SYNTHETIC_CLASSIFICATION_CONTEXT_SIZE":    "200",
        "SYNTHETIC_CLASSIFICATION_CONTEXT_ENSEMBLES": "5",
        "SYNTHETIC_CLASSIFICATION_TEST_BATCH_SIZE": "128",
        "TRAINING_DATA_FAMILY":                     os.environ.get(
            "TRAINING_DATA_FAMILY", "synthetic_linear_classification"
        ),
        "SYNCLS_EVALUATION_RUN_ID":                 run_id,
        # Explicit flag propagation (mixed-cat + index table)
        "SYNCLS_IS_MIXED_CATEGORICAL": os.environ.get(
            "SYNCLS_IS_MIXED_CATEGORICAL", "false"
        ),
        "SYNCLS_INDEX_TABLE": os.environ.get(
            "SYNCLS_INDEX_TABLE", "LINEAR_CLASSIFICATION_DATASET_INDEX"
        ),
    }
    if extra_env:
        env.update(extra_env)
    return env


def _classification_aggregate_env(
    suite_id: str,
    results_stage: str,
    expected_deepset_shards: int,
    expected_baseline_shards: int,
    expected_ag_shards: int,
) -> dict:
    """Build env vars dict for the aggregate job."""
    import os as _os
    run_id = _os.getenv("SYNCLS_EVALUATION_RUN_ID", "")
    env = {
        "SYNTHETIC_CLASSIFICATION_MODE":            "aggregate",
        "SYNTHETIC_CLASSIFICATION_SUITE_ID":        suite_id,
        "SYNCLS_RESULTS_STAGE":                     results_stage,
        "SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH":     SYNCLS_DEFAULT_CKPT,
        "SYNTHETIC_CLASSIFICATION_FEATURE_SELECTOR": SYNCLS_DEFAULT_FEATURE_SELECTOR,
        "SYNCLS_EXPECTED_DEEPSET_SHARDS":           str(expected_deepset_shards),
        "SYNCLS_EXPECTED_BASELINE_SHARDS":          str(expected_baseline_shards),
        "SYNCLS_EXPECTED_AG_SHARDS":                str(expected_ag_shards),
        "TRAINING_DATA_FAMILY":                     os.environ.get(
            "TRAINING_DATA_FAMILY", "synthetic_linear_classification"
        ),
        "SYNCLS_EVALUATION_RUN_ID":                 run_id,
        # Explicit flag propagation (mixed-cat + index table)
        "SYNCLS_IS_MIXED_CATEGORICAL": os.environ.get(
            "SYNCLS_IS_MIXED_CATEGORICAL", "false"
        ),
        "SYNCLS_INDEX_TABLE": os.environ.get(
            "SYNCLS_INDEX_TABLE", "LINEAR_CLASSIFICATION_DATASET_INDEX"
        ),
    }
    return env


# ---------------------------------------------------------------------------
# Public handler functions (Snowflake HANDLER targets)
# ---------------------------------------------------------------------------

def run_synthetic_classification_linear_prep(
    session,
    prep_rt: str,
    bench_rt: str,
    ag_rt: str,
    suite_id: str = SYNCLS_DEFAULT_SUITE_ID,
    is_mixed_categorical=False,
) -> str:
    """Phase 1: Index classification parquets into LINEAR_CLASSIFICATION_DATASET_INDEX."""
    results_stage = _set_classification_linear_env(suite_id, is_mixed_categorical=is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    label = "cls_linear_prep"
    env = {
        "SYNTHETIC_CLASSIFICATION_SUITE_ID": suite_id,
        "SYNTHETIC_CLASSIFICATION_FORCE_REBUILD": os.getenv(
            "SYNTHETIC_CLASSIFICATION_FORCE_REBUILD", "false"
        ),
    }
    job = _submit_cls_job(
        session=session,
        label=label,
        compute_pool=DEEPSET_CPU_POOL,
        env_vars=env,
        runtime_environment=bench_rt,
        entrypoint="prepare_synthetic_classification.py",
        target_instances=1,
    )
    _wait_done(job, label, session)
    return f"run_synthetic_classification_linear_prep: ok suite_id={suite_id}"


def run_synthetic_classification_linear_deepset_evaluation(
    session,
    bench_rt: str,
    suite_id: str = SYNCLS_DEFAULT_SUITE_ID,
    is_mixed_categorical=False,
) -> str:
    """Phase 2: DeepSet GPU inference on all classification datasets."""
    results_stage = _set_classification_linear_env(suite_id, is_mixed_categorical=is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)

    deepset_shards = int(os.getenv("SYNTHETIC_CLASSIFICATION_NUM_SHARDS", "10"))
    jobs = []
    for i in range(deepset_shards):
        label = f"cls_deepset_shard_{i}"
        env = _classification_shard_env(
            mode="deepset",
            suite_id=suite_id,
            num_shards=deepset_shards,
            shard_index=i,
            results_stage=results_stage,
            extra_env={
                "BENCHMARK_REQUIRE_CUDA": "true",
                # Use SYNCLS_* prefix for classification (SYNREG_* is regression-only).
                # Deprecation shim: also honour legacy SYNREG_* reads for one release.
                "SYNCLS_RUN_CHECKPOINT_GATES": "true",
                "SYNCLS_CHECKPOINT_GATE_STRICT": "true",
            },
        )
        job = _submit_cls_job(
            session=session,
            label=label,
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=env,
            runtime_environment=bench_rt,
            entrypoint="evaluate_linear_classification.py",
            target_instances=1,
        )
        jobs.append((label, job))
    _wait_job_group(jobs, session)
    return f"run_synthetic_classification_linear_deepset_evaluation: ok shards={deepset_shards}"


def run_synthetic_classification_linear_baseline_evaluation(
    session,
    bench_rt: str,
    suite_id: str = SYNCLS_DEFAULT_SUITE_ID,
    baseline_shards: int = 10,
    baseline_concurrent_nodes: int = 10,
    is_mixed_categorical=False,
) -> str:
    """Phase 3: CPU baseline models on all classification datasets."""
    results_stage = _set_classification_linear_env(suite_id, is_mixed_categorical=is_mixed_categorical)
    baseline_shards = _resolve_int(baseline_shards, 10)
    baseline_concurrent_nodes = _resolve_int(baseline_concurrent_nodes, 10)
    shard_count = _run_baseline_shards(
        session=session,
        suite_id=suite_id,
        results_stage=results_stage,
        runtime_environment=bench_rt,
        baseline_shards=baseline_shards,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )
    return (
        f"run_synthetic_classification_linear_baseline_evaluation: ok "
        f"shards={shard_count}"
    )


def run_synthetic_classification_linear_autogluon_evaluation(
    session,
    bench_rt: str,
    ag_rt: str,
    suite_id: str = SYNCLS_DEFAULT_SUITE_ID,
    autogluon_cluster_shards: int = 0,
    autogluon_workers_per_shard: int = 1,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
    is_mixed_categorical=False,
) -> str:
    """Phase 4: AutoGluon evaluation (single-node default for classification)."""
    results_stage = _set_classification_linear_env(suite_id, is_mixed_categorical=is_mixed_categorical)
    cluster_shards   = _resolve_int(autogluon_cluster_shards, 0)
    concurrent_units = _resolve_int(autogluon_concurrent_clusters, 10)
    task_cpus        = _resolve_int(autogluon_task_cpus, 1)
    time_limit       = _resolve_int(autogluon_time_limit, 300)
    presets          = _resolve_str(autogluon_presets, "high_quality")

    if cluster_shards > 0:
        raise ValueError(
            "Classification AutoGluon evaluation uses single-node mode. "
            "Set autogluon_cluster_shards=0 (default). "
            "Ray distributed classification has not been validated."
        )

    # Single-node shards: concurrent_units = number of shards
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    jobs = []
    for i in range(concurrent_units):
        label = f"cls_autogluon_shard_{i}"
        env = _classification_shard_env(
            mode="autogluon",
            suite_id=suite_id,
            num_shards=concurrent_units,
            shard_index=i,
            results_stage=results_stage,
            extra_env={
                "SYNCLS_AUTOGLUON_TIME_LIMIT":  str(time_limit),
                "SYNCLS_AUTOGLUON_PRESETS":     presets,
                "SYNCLS_AUTOGLUON_TASK_CPUS":   str(task_cpus),
                "SYNCLS_CLUSTER_SHARDS":        "0",
            },
        )
        job = _submit_cls_job(
            session=session,
            label=label,
            compute_pool=AUTOGLUON_CPU_POOL,
            env_vars=env,
            runtime_environment=ag_rt,
            entrypoint="evaluate_linear_classification.py",
            target_instances=1,
            pip_requirements=_SYNCLS_AG_PIP,
            external_access_integrations=_SYNCLS_PYPI_EAI,
        )
        jobs.append((label, job))
    _wait_job_group(jobs, session)
    return (
        f"run_synthetic_classification_linear_autogluon_evaluation: ok "
        f"shards={concurrent_units} mode=single_node"
    )


def run_synthetic_classification_linear_aggregation(
    session,
    bench_rt: str,
    ag_rt: str,
    suite_id: str = SYNCLS_DEFAULT_SUITE_ID,
    expected_ag_shards: int = 0,
    expected_baseline_shards: int = 10,
    expected_deepset_shards: int = 10,
    is_mixed_categorical=False,
) -> str:
    """Phase 5: Aggregate shards, produce summary CSVs and manifest."""
    results_stage = _set_classification_linear_env(suite_id, is_mixed_categorical=is_mixed_categorical)
    expected_ag_shards       = _resolve_int(expected_ag_shards,       0)
    expected_baseline_shards = _resolve_int(expected_baseline_shards, 10)
    expected_deepset_shards  = _resolve_int(expected_deepset_shards,  10)

    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    label = "cls_aggregation"
    env = _classification_aggregate_env(
        suite_id=suite_id,
        results_stage=results_stage,
        expected_deepset_shards=expected_deepset_shards,
        expected_baseline_shards=expected_baseline_shards,
        expected_ag_shards=expected_ag_shards,
    )
    job = _submit_cls_job(
        session=session,
        label=label,
        compute_pool=DEEPSET_CPU_POOL,
        env_vars=env,
        runtime_environment=bench_rt,
        entrypoint="evaluate_linear_classification.py",
        target_instances=1,
    )
    _wait_done(job, label, session)
    return (
        f"run_synthetic_classification_linear_aggregation: ok suite_id={suite_id} "
        f"expected_deepset={expected_deepset_shards} "
        f"expected_baseline={expected_baseline_shards} "
        f"expected_ag={expected_ag_shards}"
    )


def run_synthetic_classification_linear_pipeline(
    session,
    prep_rt: str,
    bench_rt: str,
    ag_rt: str,
    suite_id: str = SYNCLS_DEFAULT_SUITE_ID,
    baseline_shards: int = 10,
    baseline_concurrent_nodes: int = 10,
    autogluon_cluster_shards: int = 0,
    autogluon_workers_per_shard: int = 1,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
    is_mixed_categorical=False,
) -> str:
    """End-to-end classification pipeline: prep → deepset → baselines → ag → aggregate."""
    concurrent_clusters = _resolve_int(autogluon_concurrent_clusters, 10)
    results = []
    results.append(
        run_synthetic_classification_linear_prep(
            session, prep_rt, bench_rt, ag_rt, suite_id,
            is_mixed_categorical=is_mixed_categorical,
        )
    )
    results.append(
        run_synthetic_classification_linear_deepset_evaluation(
            session, bench_rt, suite_id,
            is_mixed_categorical=is_mixed_categorical,
        )
    )
    results.append(
        run_synthetic_classification_linear_baseline_evaluation(
            session, bench_rt, suite_id,
            baseline_shards=baseline_shards,
            baseline_concurrent_nodes=baseline_concurrent_nodes,
            is_mixed_categorical=is_mixed_categorical,
        )
    )
    results.append(
        run_synthetic_classification_linear_autogluon_evaluation(
            session, bench_rt, ag_rt, suite_id,
            autogluon_cluster_shards=autogluon_cluster_shards,
            autogluon_workers_per_shard=autogluon_workers_per_shard,
            autogluon_task_cpus=autogluon_task_cpus,
            autogluon_concurrent_clusters=autogluon_concurrent_clusters,
            autogluon_time_limit=autogluon_time_limit,
            autogluon_presets=autogluon_presets,
            is_mixed_categorical=is_mixed_categorical,
        )
    )
    results.append(
        run_synthetic_classification_linear_aggregation(
            session, bench_rt, ag_rt, suite_id,
            expected_ag_shards=concurrent_clusters,
            expected_baseline_shards=baseline_shards,
            expected_deepset_shards=int(os.getenv("SYNTHETIC_CLASSIFICATION_NUM_SHARDS", "10")),
            is_mixed_categorical=is_mixed_categorical,
        )
    )
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic classification orchestrator")
    parser.add_argument("--phase", default="pipeline",
                        choices=["prep", "deepset_evaluation", "baseline_evaluation",
                                 "autogluon_evaluation", "aggregation", "pipeline"])
    parser.add_argument("--prep-rt",  default="2.5.0-py311")
    parser.add_argument("--bench-rt", default="2.5.0-py311")
    parser.add_argument("--ag-rt",    default="2.5.0-py311")
    parser.add_argument("--suite-id", default=SYNCLS_DEFAULT_SUITE_ID)
    args = parser.parse_args()

    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()

    phase_map = {
        "prep":                   lambda: run_synthetic_classification_linear_prep(
                                      session, args.prep_rt, args.bench_rt, args.ag_rt, args.suite_id),
        "deepset_evaluation":     lambda: run_synthetic_classification_linear_deepset_evaluation(
                                      session, args.bench_rt, args.suite_id),
        "baseline_evaluation":    lambda: run_synthetic_classification_linear_baseline_evaluation(
                                      session, args.bench_rt, args.suite_id),
        "autogluon_evaluation":   lambda: run_synthetic_classification_linear_autogluon_evaluation(
                                      session, args.bench_rt, args.ag_rt, args.suite_id),
        "aggregation":            lambda: run_synthetic_classification_linear_aggregation(
                                      session, args.bench_rt, args.ag_rt, args.suite_id),
        "pipeline":               lambda: run_synthetic_classification_linear_pipeline(
                                      session, args.prep_rt, args.bench_rt, args.ag_rt, args.suite_id),
    }
    result = phase_map[args.phase]()
    print(f"[RESULT] {result}")


if __name__ == "__main__":
    main()
