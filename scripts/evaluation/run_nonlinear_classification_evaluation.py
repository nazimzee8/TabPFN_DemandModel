"""
run_nonlinear_classification_evaluation.py
==========================================
Orchestration handlers for the nonlinear synthetic classification evaluation
suite (suite_id='nonlinear_classification', 11 families).

Imports shared helpers from run_linear_classification_evaluation rather than
duplicating them. Every job submission injects
  SYNCLS_INDEX_TABLE=NONLINEAR_CLASSIFICATION_DATASET_INDEX
so that evaluate_nonlinear_classification.py queries the nonlinear index table,
leaving LINEAR_CLASSIFICATION_DATASET_INDEX untouched.

Stored procedure handlers (all callable from SQL):
  run_nonlinear_classification_prep
  run_nonlinear_classification_deepset_evaluation
  run_nonlinear_classification_baseline_evaluation
  run_nonlinear_classification_autogluon_evaluation
  run_nonlinear_classification_aggregation
"""

from __future__ import annotations

import os

from run_linear_classification_evaluation import (
    _submit_cls_job,
    _wait_done,
    _wait_job_group,
    _ensure_compute_pool_usable,
    _classification_shard_env,
    _classification_aggregate_env,
    _SYNCLS_BASELINE_PIP,
    _SYNCLS_AG_PIP,
    _SYNCLS_PYPI_EAI,
    DEEPSET_GPU_POOL,
    DEEPSET_CPU_POOL,
    AUTOGLUON_CPU_POOL,
    SYNCLS_DEFAULT_CKPT,
    SYNCLS_DEFAULT_FEATURE_SELECTOR,
    _resolve_int,
    _resolve_str,
)

# ---------------------------------------------------------------------------
# Nonlinear classification uses a distinct checkpoint from linear classification.
# _classification_shard_env and _classification_aggregate_env are defined in
# run_linear_classification_evaluation and capture SYNCLS_DEFAULT_CKPT from that
# module's namespace at call time.  A plain module-level re-assignment here would
# not affect them, so we wrap both functions and override the checkpoint key in
# the returned dict instead.
# ---------------------------------------------------------------------------

SYNCLS_DEFAULT_CKPT = "@MODEL_STAGE/checkpoints/best_nonlinear_cls.pt"  # nonlinear override


def _nonlinear_cls_shard_env(*args, **kwargs) -> dict:
    """Wrapper: calls the shared shard-env builder then pins the nonlinear checkpoint."""
    env = _classification_shard_env(*args, **kwargs)
    env["SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH"] = SYNCLS_DEFAULT_CKPT
    return env


def _nonlinear_cls_aggregate_env(*args, **kwargs) -> dict:
    """Wrapper: calls the shared aggregate-env builder then pins the nonlinear checkpoint."""
    env = _classification_aggregate_env(*args, **kwargs)
    env["SYNCLS_DEEPSET_CHECKPOINT_STAGE_PATH"] = SYNCLS_DEFAULT_CKPT
    return env


# ---------------------------------------------------------------------------
# nonlinear classification suite constants
# ---------------------------------------------------------------------------

NONLINEAR_CLS_SUITE_ID      = "nonlinear_classification"
NONLINEAR_CLS_INDEX_TABLE   = "NONLINEAR_CLASSIFICATION_DATASET_INDEX"
NONLINEAR_CLS_RESULTS_STAGE = (
    "@EVALUATION_RESULTS_STAGE/nonlinear/classification/numeric/nonlinear_classification"
)
NONLINEAR_CLS_DEEPSET_SHARDS   = 10
NONLINEAR_CLS_BASELINE_SHARDS  = 10
NONLINEAR_CLS_AUTOGLUON_SHARDS = 10

# Injected into every job so evaluate_nonlinear_classification.py
# queries the nonlinear index table instead of the linear one.
_NONLINEAR_CLS_INDEX_ENV = {"SYNCLS_INDEX_TABLE": NONLINEAR_CLS_INDEX_TABLE}

# Mixed-categorical nonlinear classification eval constants
NONLINEAR_MIXED_CLS_SUITE_ID      = "nonlinear_mixed_classification"
NONLINEAR_MIXED_CLS_INDEX_TABLE   = "NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX"
NONLINEAR_MIXED_CLS_RESULTS_STAGE = (
    "@EVALUATION_RESULTS_STAGE/nonlinear/classification/mixed/nonlinear_mixed_classification"
)

# Mixed-categorical env dict — overrides index table and enables the flag in evaluators
_NONLINEAR_MIXED_CLS_INDEX_ENV = {
    "SYNCLS_INDEX_TABLE":           NONLINEAR_MIXED_CLS_INDEX_TABLE,
    "SYNCLS_IS_MIXED_CATEGORICAL":  "true",
}


def _nonlinear_classification_suite_params(is_mixed_categorical: bool) -> dict:
    """Return suite-level variables that differ between standard and mixed cls evals."""
    if is_mixed_categorical:
        return {
            "suite_id":      NONLINEAR_MIXED_CLS_SUITE_ID,
            "results_stage": NONLINEAR_MIXED_CLS_RESULTS_STAGE,
            "idx_env":       _NONLINEAR_MIXED_CLS_INDEX_ENV,
        }
    return {
        "suite_id":      NONLINEAR_CLS_SUITE_ID,
        "results_stage": NONLINEAR_CLS_RESULTS_STAGE,
        "idx_env":       _NONLINEAR_CLS_INDEX_ENV,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Prep (index creation + population)
# ---------------------------------------------------------------------------

def run_nonlinear_classification_prep(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """Phase 1: Index nonlinear classification datasets into the appropriate eval index table."""
    _sp = _nonlinear_classification_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(
        f"[INFO] nonlinear_classification Phase 1: prep suite_id={_sp['suite_id']} …",
        flush=True,
    )
    env = {
        "NONLINEAR_CLS_SUITE_ID": _sp["suite_id"],
        **_sp["idx_env"],
    }
    job = _submit_cls_job(
        session=session,
        label="nonlinear_cls_prep",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars=env,
        runtime_environment=bench_rt,
        entrypoint="prepare_nonlinear_classification.py",
        target_instances=1,
    )
    _wait_done(job, label="nonlinear_cls_prep", session=session)
    return f"run_nonlinear_classification_prep: ok suite_id={_sp['suite_id']}"


# ---------------------------------------------------------------------------
# Phase 2 — DeepSet GPU evaluation
# ---------------------------------------------------------------------------

def run_nonlinear_classification_deepset_evaluation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
) -> str:
    """Phase 2: DeepSet GPU inference — 10 shards on DEEPSET_GPU_POOL."""
    _sp = _nonlinear_classification_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)
    print(
        f"[INFO] nonlinear_classification Phase 2: deepset suite_id={_sp['suite_id']} …",
        flush=True,
    )

    jobs = []
    for i in range(NONLINEAR_CLS_DEEPSET_SHARDS):
        label = f"nonlinear_cls_deepset_{i}"
        env = _nonlinear_cls_shard_env(
            mode="deepset",
            suite_id=_sp["suite_id"],
            num_shards=NONLINEAR_CLS_DEEPSET_SHARDS,
            shard_index=i,
            results_stage=_sp["results_stage"],
            extra_env={
                "BENCHMARK_REQUIRE_CUDA": "true",
                "SYNCLS_RUN_CHECKPOINT_GATES": "true",
                "SYNCLS_CHECKPOINT_GATE_STRICT": "true",
                **_sp["idx_env"],
            },
        )
        job = _submit_cls_job(
            session=session,
            label=label,
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=env,
            runtime_environment=bench_rt,
            entrypoint="evaluate_nonlinear_classification.py",
            target_instances=1,
        )
        jobs.append((label, job))

    _wait_job_group(jobs, session)
    return (
        f"run_nonlinear_classification_deepset_evaluation: ok "
        f"suite_id={_sp['suite_id']} shards={NONLINEAR_CLS_DEEPSET_SHARDS}"
    )


# ---------------------------------------------------------------------------
# Phase 3 — Baseline CPU evaluation
# ---------------------------------------------------------------------------

def run_nonlinear_classification_baseline_evaluation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
    baseline_shards_or_is_mixed=None,
) -> str:
    """Phase 3: CPU baseline models — sharded across DEEPSET_CPU_POOL.

    Overload dispatch (Snowpark positional binding — IS_MIXED_CATEGORICAL is the first SQL arg):
      Overload 1 (is_mixed_categorical, bench_rt)        -> baseline_shards_or_is_mixed=None (use default shards)
      Overload 2 (is_mixed_categorical, bench_rt, shards) -> baseline_shards_or_is_mixed receives int
    """
    if isinstance(baseline_shards_or_is_mixed, bool):
        is_mixed_categorical = baseline_shards_or_is_mixed
        baseline_shards = NONLINEAR_CLS_BASELINE_SHARDS
    elif baseline_shards_or_is_mixed is not None:
        baseline_shards = _resolve_int(baseline_shards_or_is_mixed, NONLINEAR_CLS_BASELINE_SHARDS)
    else:
        baseline_shards = NONLINEAR_CLS_BASELINE_SHARDS
    _sp = _nonlinear_classification_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(
        f"[INFO] nonlinear_classification Phase 3: baselines "
        f"suite_id={_sp['suite_id']} shards={baseline_shards} …",
        flush=True,
    )

    jobs = []
    for shard_index in range(baseline_shards):
        label = f"nonlinear_cls_baseline_{shard_index}"
        env = _nonlinear_cls_shard_env(
            mode="baselines",
            suite_id=_sp["suite_id"],
            num_shards=baseline_shards,
            shard_index=shard_index,
            results_stage=_sp["results_stage"],
            extra_env=_sp["idx_env"],
        )
        job = _submit_cls_job(
            session=session,
            label=label,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=env,
            runtime_environment=bench_rt,
            entrypoint="evaluate_nonlinear_classification.py",
            target_instances=1,
            pip_requirements=_SYNCLS_BASELINE_PIP,
            external_access_integrations=_SYNCLS_PYPI_EAI,
        )
        jobs.append((label, job))

    if jobs:
        _wait_job_group(jobs, session)
    return (
        f"run_nonlinear_classification_baseline_evaluation: ok "
        f"suite_id={_sp['suite_id']} shards={baseline_shards}"
    )


# ---------------------------------------------------------------------------
# Phase 4 — AutoGluon evaluation
# ---------------------------------------------------------------------------

def run_nonlinear_classification_autogluon_evaluation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_shards_or_is_mixed=None,
    autogluon_task_cpus=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
) -> str:
    """Phase 4: AutoGluon single-node shards on AUTOGLUON_CPU_POOL.

    Overload dispatch (Snowpark positional binding — IS_MIXED_CATEGORICAL is the first SQL arg):
      Overload 1 (is_mixed_categorical, bench_rt, ag_rt)               -> autogluon_shards_or_is_mixed=None (use default shards)
      Overload 2 (is_mixed_categorical, bench_rt, ag_rt, shards, ...)  -> autogluon_shards_or_is_mixed receives int
    """
    if isinstance(autogluon_shards_or_is_mixed, bool):
        is_mixed_categorical = autogluon_shards_or_is_mixed
        autogluon_shards = NONLINEAR_CLS_AUTOGLUON_SHARDS
    elif autogluon_shards_or_is_mixed is not None:
        autogluon_shards = _resolve_int(autogluon_shards_or_is_mixed, NONLINEAR_CLS_AUTOGLUON_SHARDS)
    else:
        autogluon_shards = NONLINEAR_CLS_AUTOGLUON_SHARDS
    _sp = _nonlinear_classification_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    task_cpus  = _resolve_int(autogluon_task_cpus, 1)
    time_limit = _resolve_int(autogluon_time_limit, 300)
    presets    = _resolve_str(autogluon_presets, "high_quality")
    print(
        f"[INFO] nonlinear_classification Phase 4: autogluon "
        f"suite_id={_sp['suite_id']} shards={autogluon_shards} …",
        flush=True,
    )

    jobs = []
    for i in range(autogluon_shards):
        label = f"nonlinear_cls_ag_{i}"
        env = _nonlinear_cls_shard_env(
            mode="autogluon",
            suite_id=_sp["suite_id"],
            num_shards=autogluon_shards,
            shard_index=i,
            results_stage=_sp["results_stage"],
            extra_env={
                "SYNCLS_AUTOGLUON_TIME_LIMIT":  str(time_limit),
                "SYNCLS_AUTOGLUON_PRESETS":     presets,
                "SYNCLS_AUTOGLUON_TASK_CPUS":   str(task_cpus),
                "SYNCLS_CLUSTER_SHARDS":        "0",
                **_sp["idx_env"],
            },
        )
        job = _submit_cls_job(
            session=session,
            label=label,
            compute_pool=AUTOGLUON_CPU_POOL,
            env_vars=env,
            runtime_environment=ag_rt,
            entrypoint="evaluate_nonlinear_classification.py",
            target_instances=1,
            pip_requirements=_SYNCLS_AG_PIP,
            external_access_integrations=_SYNCLS_PYPI_EAI,
        )
        jobs.append((label, job))

    _wait_job_group(jobs, session)
    return (
        f"run_nonlinear_classification_autogluon_evaluation: ok "
        f"suite_id={_sp['suite_id']} shards={autogluon_shards} mode=single_node"
    )


# ---------------------------------------------------------------------------
# Phase 5 — Aggregation
# ---------------------------------------------------------------------------

def run_nonlinear_classification_aggregation(
    session,
    is_mixed_categorical: bool,
    bench_rt: str = "2.5.0-py311",
    expected_deepset_shards_or_is_mixed=None,
    expected_baseline_shards: int = NONLINEAR_CLS_BASELINE_SHARDS,
    expected_ag_shards: int = NONLINEAR_CLS_AUTOGLUON_SHARDS,
) -> str:
    """Phase 5: Aggregate shards, produce summary CSVs.

    Overload dispatch (Snowpark positional binding — IS_MIXED_CATEGORICAL is the first SQL arg):
      Overload 1 (is_mixed_categorical, bench_rt)                           -> expected_deepset_shards_or_is_mixed=None (use default shard counts)
      Overload 2 (is_mixed_categorical, bench_rt, ds_shards, bl_shards, ag_shards) -> expected_deepset_shards_or_is_mixed receives int
    """
    if isinstance(expected_deepset_shards_or_is_mixed, bool):
        is_mixed_categorical = expected_deepset_shards_or_is_mixed
        expected_deepset_shards = NONLINEAR_CLS_DEEPSET_SHARDS
    elif expected_deepset_shards_or_is_mixed is not None:
        expected_deepset_shards = _resolve_int(expected_deepset_shards_or_is_mixed, NONLINEAR_CLS_DEEPSET_SHARDS)
    else:
        expected_deepset_shards = NONLINEAR_CLS_DEEPSET_SHARDS
    _sp = _nonlinear_classification_suite_params(is_mixed_categorical)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    expected_baseline_shards = _resolve_int(expected_baseline_shards, NONLINEAR_CLS_BASELINE_SHARDS)
    expected_ag_shards       = _resolve_int(expected_ag_shards,       NONLINEAR_CLS_AUTOGLUON_SHARDS)
    print(
        f"[INFO] nonlinear_classification Phase 5: aggregation suite_id={_sp['suite_id']} …",
        flush=True,
    )

    env = _nonlinear_cls_aggregate_env(
        suite_id=_sp["suite_id"],
        results_stage=_sp["results_stage"],
        expected_deepset_shards=expected_deepset_shards,
        expected_baseline_shards=expected_baseline_shards,
        expected_ag_shards=expected_ag_shards,
    )
    env.update(_sp["idx_env"])

    job = _submit_cls_job(
        session=session,
        label="nonlinear_cls_aggregation",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars=env,
        runtime_environment=bench_rt,
        entrypoint="evaluate_nonlinear_classification.py",
        target_instances=1,
    )
    _wait_done(job, label="nonlinear_cls_aggregation", session=session)
    return (
        f"run_nonlinear_classification_aggregation: ok suite_id={_sp['suite_id']} "
        f"expected_deepset={expected_deepset_shards} "
        f"expected_baseline={expected_baseline_shards} "
        f"expected_ag={expected_ag_shards}"
    )

