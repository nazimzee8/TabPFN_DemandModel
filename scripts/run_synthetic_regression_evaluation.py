"""
run_synthetic_regression_evaluation.py
========================================
Orchestrator for the split-phase synthetic regression evaluation suite.

Mirrors run_evaluation_test.py exactly:
  - submit_from_stage with explicit runtime_environment, env_vars,
    pip_requirements, external_access_integrations
  - Phase-gated parallelism: prep → DeepSet GPU → CPU baselines → AutoGluon → aggregate
  - Node quota error handling (395034)
  - Capacity probes in non-overlapping batches

Stored procedure handlers (all with signature
  (session, prep_rt, bench_rt, ag_rt) -> str):

  run_synthetic_regression_runtime_probes
  run_synthetic_regression_capacity_probe
  run_synthetic_regression_prep
  run_synthetic_regression_deepset_evaluation
  run_synthetic_regression_baseline_evaluation
  run_synthetic_regression_autogluon_evaluation
  run_synthetic_regression_aggregation
  run_synthetic_regression_pipeline
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Pool / stage / shard constants
# ---------------------------------------------------------------------------

SCRIPTS_STAGE = "@MODEL_STAGE/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"

SYNREG_GPU_SHARDS = 10
SYNREG_CPU_SHARDS = 6
SYNREG_AUTOGLUON_SHARDS = 60
SYNREG_BASELINE_CONCURRENT_NODES_DEFAULT = 6
SYNREG_AUTOGLUON_CONCURRENT_NODES_DEFAULT = 60

# Distributed AutoGluon defaults (combined suite, ray_work_items mode)
SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT = int(
    os.getenv("SYNREG_AUTOGLUON_CLUSTER_SHARDS", "6")
)
SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT = int(
    os.getenv("SYNREG_AUTOGLUON_WORKERS_PER_SHARD", "4")
)
SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT = int(
    os.getenv("AUTOGLUON_TASK_CPUS", "1")
)
SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS_DEFAULT = int(
    os.getenv(
        "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS",
        str(SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT),
    )
)
SYNREG_AUTOGLUON_DISTRIBUTED_MODE_DEFAULT = os.getenv(
    "SYNREG_AUTOGLUON_DISTRIBUTED_MODE",
    "ray_work_items",
)
SYNREG_AUTOGLUON_ENTRYPOINT_DEFAULT = os.getenv(
    "SYNREG_AUTOGLUON_ENTRYPOINT",
    "evaluate_synthetic_regression_autogluon_ray.py",
)

DEEPSET_GPU_POOL = "DEEPSET_GPU_POOL"
DEEPSET_CPU_POOL = "DEEPSET_CPU_POOL"
AUTOGLUON_CPU_POOL = "AUTOGLUON_CPU_POOL"

# pip requirements (pinned)
SYNREG_BASELINE_PIP = ["catboost==1.2.10"]
SYNREG_AG_PIP = ["autogluon.tabular==1.3.0"]
SYNREG_PYPI_EAI = ["TABPFN_PYPI_EAI"]

SYNREG_CHECKPOINT_LOADING_MODES = {"deepset", "baselines", "autogluon"}
SYNREG_DEEPSET_CKPT_STAGE = os.getenv(
    "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH",
    "@MODEL_STAGE/checkpoints/best.pt",
)

# Runtime environment keys (passed as positional args to handlers)
_PREP_RT_KEY = "prep"
_BENCH_RT_KEY = "bench"
_AG_RT_KEY = "ag"

# Import probes
_BENCH_PROBE_IMPORTS = [
    "torch", "numpy", "pandas", "sklearn", "scipy", "pyarrow", "matplotlib",
    "snowflake.snowpark",
]
_BASELINE_PROBE_IMPORTS = _BENCH_PROBE_IMPORTS + ["xgboost", "lightgbm", "catboost"]
_AG_PROBE_IMPORTS = [
    "autogluon.tabular", "numpy", "pandas", "sklearn", "torch", "pyarrow",
    "matplotlib", "snowflake.snowpark",
]
_PREP_PROBE_IMPORTS = ["torch", "numpy", "scipy", "pyarrow", "snowflake.snowpark"]

# ---------------------------------------------------------------------------
# Internal helpers (mirror run_evaluation_test.py)
# ---------------------------------------------------------------------------

_UNSET = object()


def _wait_done(job, label: str, session) -> None:
    """Wait for job completion. Raise RuntimeError with diagnostics if failed.
    Mirrors run_evaluation_test.py._wait_done exactly."""
    try:
        job.wait()
    except Exception as wait_exc:
        if "300002" in str(wait_exc) or "000603" in str(wait_exc):
            raise RuntimeError(
                f"[FATAL] Job '{label}' terminated with Snowflake internal error "
                "300002/000603. The container likely crashed before reaching a clean "
                "terminal state. Inspect the job's container logs in Snowsight."
            ) from wait_exc
        raise RuntimeError(
            f"[FATAL] Job '{label}' raised during wait.\n"
            f"Original exception: {wait_exc}"
        ) from wait_exc

    if job.status == "DONE":
        print(f"[INFO] Job '{label}' complete.", flush=True)
        return

    try:
        logs = job.logs(node_id=0)
    except Exception:
        logs = "<logs unavailable>"
    raise RuntimeError(
        f"[FATAL] Job '{label}' failed with status {job.status!r}.\n"
        f"Logs:\n{logs}"
    )


COMPUTE_POOL_USABLE_STATES = {"ACTIVE", "IDLE"}
COMPUTE_POOL_FAILED_STATES = {"FAILED", "ERROR", "DELETING", "STOPPING", "UNKNOWN"}


def _list_stage(session, stage_path: str) -> list[str]:
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _ensure_compute_pool_usable(session, compute_pool: str) -> None:
    rows = session.sql(f"SHOW COMPUTE POOLS LIKE '{compute_pool}'").collect()
    for row in rows:
        name = str(row[0])
        if name.upper() != compute_pool.upper():
            continue
        state = str(row[1]).upper() if len(row) > 1 else "UNKNOWN"
        if state in COMPUTE_POOL_FAILED_STATES:
            raise RuntimeError(
                f"Compute pool {compute_pool} is unusable: state={state}."
            )
        if state == "SUSPENDED":
            print(
                f"Compute pool {compute_pool} is SUSPENDED; issuing RESUME.",
                flush=True,
            )
            session.sql(f"ALTER COMPUTE POOL {compute_pool} RESUME").collect()
            time.sleep(10)
        return
    raise RuntimeError(f"Compute pool {compute_pool} does not exist.")


def _is_node_quota_error(exc: Exception) -> bool:
    """Return True if this is a Snowflake node quota error (395034)."""
    msg = str(exc)
    return "395034" in msg or (
        "Requested number of nodes" in msg and "exceeds the node limit" in msg
    )


def _batched(items: list, n: int):
    """Yield successive n-item chunks from items."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _parse_positive_int(value, *, name: str, procedure_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{procedure_name}: {name} must be a positive integer; got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"{procedure_name}: {name} must be a positive integer; got {value!r}."
        )
    return parsed


def _resolve_positive_int_runtime_param(
    *,
    procedure_name: str,
    name: str,
    sql_arg,
    env_var: str,
    default: int,
) -> int:
    raw_value = sql_arg if sql_arg is not None else os.getenv(env_var, default)
    return _parse_positive_int(
        raw_value,
        name=name,
        procedure_name=procedure_name,
    )


def _resolve_runtime_string_param(
    *,
    procedure_name: str,
    name: str,
    sql_arg,
    env_var: str,
    default: str,
) -> str:
    value = sql_arg if sql_arg is not None else os.getenv(env_var, default)
    value = str(value).strip()
    if not value:
        raise ValueError(f"{procedure_name}: {name} / {env_var} must not be empty.")
    return value


def _resolve_concurrent_nodes(
    *,
    procedure_name: str,
    sql_arg,
    env_var: str,
    default: int,
    shard_count: int,
    compute_pool: str,
    arg_name: str,
) -> int:
    raw_value = sql_arg if sql_arg is not None else os.getenv(env_var, default)
    requested = _parse_positive_int(
        raw_value,
        name=arg_name,
        procedure_name=procedure_name,
    )
    if requested > shard_count:
        raise ValueError(
            f"{procedure_name}: requested concurrency {requested} exceeds shard "
            f"count {shard_count} for compute pool {compute_pool}. Raise the shard "
            "count explicitly or lower the requested concurrent node count."
        )
    return requested


def _resolve_baseline_concurrent_nodes(
    procedure_name: str,
    sql_arg=None,
    *,
    shard_count: int = SYNREG_CPU_SHARDS,
) -> int:
    return _resolve_concurrent_nodes(
        procedure_name=procedure_name,
        sql_arg=sql_arg,
        env_var="SYNREG_BASELINE_CONCURRENT_NODES",
        default=SYNREG_BASELINE_CONCURRENT_NODES_DEFAULT,
        shard_count=shard_count,
        compute_pool=DEEPSET_CPU_POOL,
        arg_name="BASELINE_CONCURRENT_NODES",
    )


def _resolve_autogluon_concurrent_nodes(
    procedure_name: str,
    sql_arg=None,
    *,
    shard_count: int = SYNREG_AUTOGLUON_SHARDS,
) -> int:
    return _resolve_concurrent_nodes(
        procedure_name=procedure_name,
        sql_arg=sql_arg,
        env_var="SYNREG_AUTOGLUON_CONCURRENT_NODES",
        default=SYNREG_AUTOGLUON_CONCURRENT_NODES_DEFAULT,
        shard_count=shard_count,
        compute_pool=AUTOGLUON_CPU_POOL,
        arg_name="AUTOGLUON_CONCURRENT_NODES",
    )


def _wait_job_group(labeled_jobs: list[tuple[str, object]], session) -> None:
    """Wait for every (label, job) pair; propagate first failure."""
    for label, job in labeled_jobs:
        _wait_done(job, label, session)


def _synreg_shard_env(
    *,
    mode: str,
    suite_id: str,
    num_shards: int,
    shard_index: int,
    results_stage: str,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build env vars for synthetic regression evaluation shard jobs."""
    env = {
        "SYNTHETIC_REGRESSION_MODE": mode,
        "SYNTHETIC_REGRESSION_SUITE_ID": suite_id,
        "SYNTHETIC_REGRESSION_NUM_SHARDS": str(num_shards),
        "SYNTHETIC_REGRESSION_SHARD_INDEX": str(shard_index),
        "SYNREG_RESULTS_STAGE": results_stage,
    }
    if extra_env:
        env.update(extra_env)
    if mode in SYNREG_CHECKPOINT_LOADING_MODES:
        env["ALLOW_UNSAFE_TORCH_LOAD"] = "true"
    return env


def _validate_synreg_submission_env(label: str, env_vars: dict[str, str]) -> None:
    mode = env_vars.get("SYNTHETIC_REGRESSION_MODE")
    if (
        mode in SYNREG_CHECKPOINT_LOADING_MODES
        and env_vars.get("ALLOW_UNSAFE_TORCH_LOAD") != "true"
    ):
        raise RuntimeError(
            "Synthetic regression shard submission contract violated: "
            f"job '{label}' mode={mode!r} must set "
            "ALLOW_UNSAFE_TORCH_LOAD=true for trusted internal staged checkpoints."
        )


# ---------------------------------------------------------------------------
# Core submission helper
# ---------------------------------------------------------------------------

def _submit_synreg(
    session,
    label: str,
    compute_pool: str,
    env_vars: dict,
    runtime_environment: str,
    entrypoint: str = "evaluate_synthetic_regression.py",
    target_instances: int = 1,
    pip_requirements: list[str] | None = None,
    external_access_integrations: list[str] | None = None,
):
    """
    Submit a synthetic regression MLJob via submit_from_stage.

    Most evaluator jobs use target_instances=1. Distributed AutoGluon work-item cluster jobs
    may set target_instances > 1, but only with an entrypoint that coordinates workers safely
    and prevents duplicate shard writes.
    """
    if target_instances > 1 and "autogluon_ray" not in entrypoint:
        raise RuntimeError(
            f"Refusing target_instances={target_instances} for entrypoint={entrypoint!r}. "
            "Multi-instance synthetic regression jobs require the Ray work-item entrypoint "
            "to avoid duplicate independent writers."
        )
    full_env_vars = {"HOME": "/tmp"}
    full_env_vars.update(env_vars)
    _validate_synreg_submission_env(label, full_env_vars)

    from snowflake.ml.jobs import submit_from_stage

    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint=entrypoint,
        compute_pool=compute_pool,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=target_instances,
        runtime_environment=runtime_environment,
        env_vars=full_env_vars,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
        session=session,
    )
    print(f"[INFO] Submitted job '{label}' on pool '{compute_pool}'", flush=True)
    return job


def _submit_capacity_probe(
    session,
    label: str,
    compute_pool: str,
    runtime_environment: str,
):
    """Submit a lightweight capacity probe (sleep 30s) to test pool availability."""
    return _submit_synreg(
        session=session,
        label=label,
        compute_pool=compute_pool,
        env_vars={"CAPACITY_PROBE": "true"},
        runtime_environment=runtime_environment,
        entrypoint="capacity_probe.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )


def _submit_and_wait_capacity_phase(
    session,
    phase_label: str,
    compute_pool: str,
    runtime_environment: str,
    count: int,
) -> None:
    """Submit `count` capacity probe jobs and wait for all of them."""
    jobs = []
    for i in range(count):
        lbl = f"{phase_label}_{i}"
        try:
            job = _submit_capacity_probe(session, lbl, compute_pool, runtime_environment)
            jobs.append((lbl, job))
        except Exception as e:
            if _is_node_quota_error(e):
                raise RuntimeError(
                    f"[QUOTA] Node quota exceeded during {phase_label} capacity probe.\n"
                    f"Requested {count} nodes on pool '{compute_pool}'.\n"
                    "Remediation: reduce node count, request quota increase, or wait for pool capacity."
                ) from e
            raise
    _wait_job_group(jobs, session)


def _submit_runtime_probe(
    session,
    label: str,
    compute_pool: str,
    runtime_environment: str,
    imports_to_check: list[str],
    pip_requirements: list[str] | None = None,
    external_access_integrations: list[str] | None = None,
) -> object:
    """Submit a runtime import probe job."""
    return _submit_synreg(
        session=session,
        label=label,
        compute_pool=compute_pool,
        env_vars={"RUNTIME_PROBE": "true", "PROBE_IMPORTS": ",".join(imports_to_check)},
        runtime_environment=runtime_environment,
        entrypoint="runtime_probe.py",
        target_instances=1,
        pip_requirements=pip_requirements,
        external_access_integrations=external_access_integrations,
    )


# ---------------------------------------------------------------------------
# Stored procedure handlers
# ---------------------------------------------------------------------------

def run_synthetic_regression_runtime_probes(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    """
    Serialized (not parallel): 4 probe jobs covering all runtimes + dependency combos.
    Probe 1: bench_rt, DeepSet imports (no pip)
    Probe 2: bench_rt, baseline imports + catboost pip
    Probe 3: ag_rt, AutoGluon imports + autogluon pip
    Probe 4: prep_rt, preparation imports (no pip)
    """
    status_parts = []

    # Probe 1: DeepSet (benchmark runtime, no pip)
    print("[INFO] Runtime probe 1/4: DeepSet imports …", flush=True)
    job1 = _submit_runtime_probe(
        session,
        label="synreg_probe_deepset",
        compute_pool=DEEPSET_GPU_POOL,
        runtime_environment=benchmark_runtime_environment,
        imports_to_check=_BENCH_PROBE_IMPORTS,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(job1, "synreg_probe_deepset", session)
    status_parts.append("probe1:ok")

    # Probe 2: Baselines (benchmark runtime + catboost pip)
    print("[INFO] Runtime probe 2/4: Baseline imports (catboost) …", flush=True)
    job2 = _submit_runtime_probe(
        session,
        label="synreg_probe_baselines",
        compute_pool=DEEPSET_CPU_POOL,
        runtime_environment=benchmark_runtime_environment,
        imports_to_check=_BASELINE_PROBE_IMPORTS,
        pip_requirements=SYNREG_BASELINE_PIP,
        external_access_integrations=SYNREG_PYPI_EAI,
    )
    _wait_done(job2, "synreg_probe_baselines", session)
    status_parts.append("probe2:ok")

    # Probe 3: AutoGluon
    print("[INFO] Runtime probe 3/4: AutoGluon imports …", flush=True)
    job3 = _submit_runtime_probe(
        session,
        label="synreg_probe_autogluon",
        compute_pool=AUTOGLUON_CPU_POOL,
        runtime_environment=autogluon_runtime_environment,
        imports_to_check=_AG_PROBE_IMPORTS,
        pip_requirements=SYNREG_AG_PIP,
        external_access_integrations=SYNREG_PYPI_EAI,
    )
    _wait_done(job3, "synreg_probe_autogluon", session)
    status_parts.append("probe3:ok")

    # Probe 4: Prep runtime
    print("[INFO] Runtime probe 4/4: Prep imports …", flush=True)
    job4 = _submit_runtime_probe(
        session,
        label="synreg_probe_prep",
        compute_pool=DEEPSET_CPU_POOL,
        runtime_environment=prep_runtime_environment,
        imports_to_check=_PREP_PROBE_IMPORTS,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(job4, "synreg_probe_prep", session)
    status_parts.append("probe4:ok")

    result = "run_synthetic_regression_runtime_probes: " + " ".join(status_parts)
    print(f"[INFO] {result}", flush=True)
    return result


def run_synthetic_regression_capacity_probe(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    baseline_concurrent_nodes=None,
    autogluon_concurrent_nodes=None,
) -> str:
    baseline_concurrency = _resolve_baseline_concurrent_nodes(
        "run_synthetic_regression_capacity_probe",
        baseline_concurrent_nodes,
    )
    autogluon_concurrency = _resolve_autogluon_concurrent_nodes(
        "run_synthetic_regression_capacity_probe",
        autogluon_concurrent_nodes,
    )
    print(
        f"[INFO] Capacity probe Phase 1: DEEPSET_CPU_POOL "
        f"({baseline_concurrency} nodes) ...",
        flush=True,
    )
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_baseline",
        DEEPSET_CPU_POOL,
        benchmark_runtime_environment,
        baseline_concurrency,
    )
    print(
        f"[INFO] Capacity probe Phase 2: AUTOGLUON_CPU_POOL "
        f"({autogluon_concurrency} nodes) ...",
        flush=True,
    )
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_autogluon",
        AUTOGLUON_CPU_POOL,
        autogluon_runtime_environment,
        autogluon_concurrency,
    )
    result = (
        "run_synthetic_regression_capacity_probe: ok "
        f"baseline={baseline_concurrency} ag={autogluon_concurrency}"
    )
    print(f"[INFO] {result}", flush=True)
    return result

    """
    3 phases, strictly non-overlapping:
    Phase 1: 10 capacity probes on DEEPSET_GPU_POOL (bench_rt)
    Phase 2: 3 capacity probes on DEEPSET_CPU_POOL (bench_rt)
    Phase 3: 30 capacity probes on AUTOGLUON_CPU_POOL (ag_rt)
    """
    print("[INFO] Capacity probe Phase 1: DEEPSET_GPU_POOL (10 nodes) …", flush=True)
    _submit_and_wait_capacity_phase(
        session,
        "retired_synreg_cap_gpu",
        DEEPSET_GPU_POOL,
        benchmark_runtime_environment,
        SYNREG_GPU_SHARDS,
    )

    print("[INFO] Capacity probe Phase 2: DEEPSET_CPU_POOL (3 nodes) …", flush=True)
    _submit_and_wait_capacity_phase(
        session,
        "retired_synreg_cap_cpu",
        DEEPSET_CPU_POOL,
        benchmark_runtime_environment,
        SYNREG_CPU_SHARDS,
    )

    print("[INFO] Capacity probe Phase 3: AUTOGLUON_CPU_POOL (30 nodes) …", flush=True)
    _submit_and_wait_capacity_phase(
        session,
        "retired_synreg_cap_ag",
        AUTOGLUON_CPU_POOL,
        autogluon_runtime_environment,
        SYNREG_AUTOGLUON_SHARDS,
    )

    result = (
        f"run_synthetic_regression_capacity_probe: ok "
        f"gpu={SYNREG_GPU_SHARDS} cpu={SYNREG_CPU_SHARDS} ag={SYNREG_AUTOGLUON_SHARDS}"
    )
    print(f"[INFO] {result}", flush=True)
    return result

def run_synthetic_regression_baseline_capacity_probe(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    baseline_concurrent_nodes=None,
) -> str:
    baseline_concurrency = _resolve_baseline_concurrent_nodes(
        "run_synthetic_regression_baseline_capacity_probe",
        baseline_concurrent_nodes,
    )
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_baseline",
        DEEPSET_CPU_POOL,
        benchmark_runtime_environment,
        baseline_concurrency,
    )
    return (
        "run_synthetic_regression_baseline_capacity_probe: ok "
        f"baseline={baseline_concurrency}"
    )


def run_synthetic_regression_autogluon_capacity_probe(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    autogluon_concurrent_nodes=None,
) -> str:
    autogluon_concurrency = _resolve_autogluon_concurrent_nodes(
        "run_synthetic_regression_autogluon_capacity_probe",
        autogluon_concurrent_nodes,
    )
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_autogluon",
        AUTOGLUON_CPU_POOL,
        autogluon_runtime_environment,
        autogluon_concurrency,
    )
    return (
        "run_synthetic_regression_autogluon_capacity_probe: ok "
        f"ag={autogluon_concurrency}"
    )


def run_synthetic_regression_prep(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    """Submit prepare_synthetic_regression.py on DEEPSET_CPU_POOL (prep_rt).

    Indexes the in-distribution suite (default suite_id = linear_poisson_v1_recommended,
    200 datasets × 5 split seeds) into SYNTHETIC_REGRESSION_DATASET_INDEX.

    OOD indexing is handled exclusively by run_synthetic_regression_ood_deepset_pilot.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
    base_seed = os.getenv("SYNTHETIC_REGRESSION_BASE_SEED", "20260512")
    force_rebuild = os.getenv("SYNTHETIC_REGRESSION_FORCE_REBUILD", "false")

    print("[INFO] Submitting in-distribution prep job …", flush=True)
    job_prod = _submit_synreg(
        session=session,
        label="synreg_prep",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_SUITE_ID": suite_id,
            "SYNTHETIC_REGRESSION_BASE_SEED": base_seed,
            "SYNTHETIC_REGRESSION_FORCE_REBUILD": force_rebuild,
        },
        runtime_environment=prep_runtime_environment,
        entrypoint="prepare_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )

    _wait_done(job_prod, "synreg_prep", session)
    return "run_synthetic_regression_prep: ok"


def run_synthetic_regression_deepset_evaluation(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    """
    Submit 10 independent single-instance GPU jobs (bench_rt, DEEPSET_GPU_POOL).
    No pip, no EAI.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")

    jobs = []
    for i in range(SYNREG_GPU_SHARDS):
        lbl = f"synreg_deepset_shard_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=_synreg_shard_env(
                mode="deepset",
                suite_id=suite_id,
                num_shards=SYNREG_GPU_SHARDS,
                shard_index=i,
                results_stage=f"@EVALUATION_RESULTS_STAGE/regression/{suite_id}",
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
                },
            ),
            runtime_environment=benchmark_runtime_environment,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=None,
            external_access_integrations=None,
        )
        jobs.append((lbl, job))

    print(f"[INFO] Waiting for {SYNREG_GPU_SHARDS} DeepSet GPU shards …", flush=True)
    _wait_job_group(jobs, session)
    return f"run_synthetic_regression_deepset_evaluation: ok shards={SYNREG_GPU_SHARDS}"


def run_synthetic_regression_baseline_evaluation(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    baseline_concurrent_nodes=None,
) -> str:
    """
    Submit baseline CPU shards (bench_rt, DEEPSET_CPU_POOL) in concurrency-limited waves.
    pip=catboost==1.2.10, EAI=TABPFN_PYPI_EAI.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
    baseline_concurrency = _resolve_baseline_concurrent_nodes(
        "run_synthetic_regression_baseline_evaluation",
        baseline_concurrent_nodes,
    )

    all_shards = list(range(SYNREG_CPU_SHARDS))
    total_submitted = 0
    for batch in _batched(all_shards, baseline_concurrency):
        jobs = []
        for i in batch:
            lbl = f"synreg_baseline_shard_{i}"
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=DEEPSET_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="baselines",
                    suite_id=suite_id,
                    num_shards=SYNREG_CPU_SHARDS,
                    shard_index=i,
                    results_stage=f"@EVALUATION_RESULTS_STAGE/regression/{suite_id}",
                ),
                runtime_environment=benchmark_runtime_environment,
                entrypoint="evaluate_synthetic_regression.py",
                target_instances=1,
                pip_requirements=SYNREG_BASELINE_PIP,
                external_access_integrations=SYNREG_PYPI_EAI,
            )
            jobs.append((lbl, job))
        _wait_job_group(jobs, session)
        total_submitted += len(batch)
        print(
            f"[INFO] Baseline batch done: {total_submitted}/{SYNREG_CPU_SHARDS}",
            flush=True,
        )
    return (
        f"run_synthetic_regression_baseline_evaluation: ok "
        f"shards={SYNREG_CPU_SHARDS} concurrency={baseline_concurrency}"
    )

    jobs = []
    for i in range(SYNREG_CPU_SHARDS):
        lbl = f"synreg_baseline_shard_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=suite_id,
                num_shards=SYNREG_CPU_SHARDS,
                shard_index=i,
                results_stage=f"@EVALUATION_RESULTS_STAGE/regression/{suite_id}",
            ),
            runtime_environment=benchmark_runtime_environment,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=SYNREG_BASELINE_PIP,
            external_access_integrations=SYNREG_PYPI_EAI,
        )
        jobs.append((lbl, job))

    print(f"[INFO] Waiting for {SYNREG_CPU_SHARDS} baseline CPU shards …", flush=True)
    _wait_job_group(jobs, session)
    return f"run_synthetic_regression_baseline_evaluation: ok shards={SYNREG_CPU_SHARDS}"


def run_synthetic_regression_autogluon_evaluation(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    autogluon_concurrent_nodes=None,
) -> str:
    """
    Submit AutoGluon CPU shards (ag_rt, AUTOGLUON_CPU_POOL) in concurrency-limited waves.
    pip=autogluon.tabular==1.3.0, EAI=TABPFN_PYPI_EAI.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
    ag_time_limit = os.getenv("AUTOGLUON_TIME_LIMIT", "300")
    ag_presets = os.getenv("AUTOGLUON_PRESETS", "best_quality")
    autogluon_concurrency = _resolve_autogluon_concurrent_nodes(
        "run_synthetic_regression_autogluon_evaluation",
        autogluon_concurrent_nodes,
    )

    all_shards = list(range(SYNREG_AUTOGLUON_SHARDS))
    total_submitted = 0

    for batch in _batched(all_shards, autogluon_concurrency):
        jobs = []
        for i in batch:
            lbl = f"synreg_ag_shard_{i}"
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="autogluon",
                    suite_id=suite_id,
                    num_shards=SYNREG_AUTOGLUON_SHARDS,
                    shard_index=i,
                    results_stage=f"@EVALUATION_RESULTS_STAGE/regression/{suite_id}",
                    extra_env={
                        "AUTOGLUON_TIME_LIMIT": ag_time_limit,
                        "AUTOGLUON_PRESETS": ag_presets,
                    },
                ),
                runtime_environment=autogluon_runtime_environment,
                entrypoint="evaluate_synthetic_regression.py",
                target_instances=1,
                pip_requirements=SYNREG_AG_PIP,
                external_access_integrations=SYNREG_PYPI_EAI,
            )
            jobs.append((lbl, job))
        _wait_job_group(jobs, session)
        total_submitted += len(batch)
        print(f"[INFO] AutoGluon batch done: {total_submitted}/{SYNREG_AUTOGLUON_SHARDS}", flush=True)

    return (
        f"run_synthetic_regression_autogluon_evaluation: ok "
        f"shards={SYNREG_AUTOGLUON_SHARDS} concurrency={autogluon_concurrency}"
    )


def run_synthetic_regression_aggregation(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    """Submit 1 CPU aggregation job on DEEPSET_CPU_POOL and return stage listing.
    Mirrors run_evaluation_aggregation from run_evaluation_test.py."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)

    suite_id = os.getenv(
        "SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended"
    )
    job = _submit_synreg(
        session=session,
        label="synreg_aggregate",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_MODE": "aggregate",
            "SYNTHETIC_REGRESSION_SUITE_ID": suite_id,
            "SYNREG_RESULTS_STAGE": f"@EVALUATION_RESULTS_STAGE/regression/{suite_id}",
            "SYNREG_EXPECTED_DEEPSET_SHARDS":  str(SYNREG_GPU_SHARDS),
            "SYNREG_EXPECTED_BASELINE_SHARDS": str(SYNREG_CPU_SHARDS),
            "SYNREG_EXPECTED_AG_SHARDS":       str(SYNREG_AUTOGLUON_SHARDS),
        },
        runtime_environment=benchmark_runtime_environment,
        entrypoint="evaluate_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(job, "synreg_aggregate", session)

    eval_contents = _list_stage(session, "@EVALUATION_RESULTS_STAGE/")
    return (
        "Synthetic regression aggregation complete.\n\nEVALUATION_RESULTS_STAGE:\n"
        + "\n".join(f"  {p}" for p in eval_contents)
    )


# ---------------------------------------------------------------------------
# OOD pilot — DeepSet only, no baselines, no AutoGluon
# ---------------------------------------------------------------------------

OOD_PILOT_SUITE_ID  = "ood_linear_pilot_v1"    # pilot (80 datasets, DeepSet only)
OOD_PILOT_PARTS_PREFIX = "@EVALUATION_RESULTS_STAGE/ood_parity"
OOD_PILOT_GPU_SHARDS = 5

OOD_FULL_SUITE_ID      = "ood_linear_full_v1"   # full suite (200 datasets, all methods)
OOD_FULL_GPU_SHARDS    = SYNREG_GPU_SHARDS       # 10, same as main pipeline
OOD_FULL_N_DATASETS    = 200                     # all 50/regime × 4 regimes
OOD_FULL_PARTS_PREFIX  = f"@EVALUATION_RESULTS_STAGE/regression/{OOD_FULL_SUITE_ID}"
OOD_FULL_OUTPUT_STAGE  = "@EVALUATION_RESULTS_STAGE/ood_full"


def _submit_ood_prep(session, suite_id: str, n_datasets: int, bench_rt: str):
    """Submit prepare_ood_regression.py on DEEPSET_CPU_POOL."""
    return _submit_synreg(
        session=session,
        label="ood_prep",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "OOD_REGRESSION_SUITE_ID": suite_id,
            "OOD_REGRESSION_N_DATASETS": str(n_datasets),   # preferred
            "OOD_REGRESSION_N_PILOT": str(n_datasets),      # backward compat fallback
        },
        runtime_environment=bench_rt,
        entrypoint="prepare_ood_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )


def run_synthetic_regression_ood_deepset_pilot(session, bench_rt: str) -> str:
    """Prep + DeepSet-only OOD pilot. DeepSet shards only — no CPU evaluators, no external AutoML.

    Phase 1 — OOD prep: indexes 80 pilot datasets (20 per regime) into
    SYNTHETIC_REGRESSION_DATASET_INDEX under suite_id=ood_linear_pilot_v1.

    Phase 2 — DeepSet shards: 5 GPU shards on DEEPSET_GPU_POOL, each writing
    result parts to @EVALUATION_RESULTS_STAGE/ood_parity/.
    """
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)

    # Phase 1: prep
    _wait_done(_submit_ood_prep(session, OOD_PILOT_SUITE_ID, 80, bench_rt), label="ood_prep", session=session)

    # Phase 2: DeepSet shards only
    jobs = [
        _submit_synreg(
            session=session,
            label=f"ood_deepset_shard_{i}",
            compute_pool=DEEPSET_GPU_POOL,
            env_vars={
                "SYNTHETIC_REGRESSION_MODE": "deepset",
                "SYNTHETIC_REGRESSION_SUITE_ID": OOD_PILOT_SUITE_ID,
                "SYNTHETIC_REGRESSION_NUM_SHARDS": str(OOD_PILOT_GPU_SHARDS),
                "SYNTHETIC_REGRESSION_SHARD_INDEX": str(i),
                "SYNREG_RESULTS_STAGE": OOD_PILOT_PARTS_PREFIX,
                "ALLOW_UNSAFE_TORCH_LOAD": "true",
                "SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH": SYNREG_DEEPSET_CKPT_STAGE,
                "SYNREG_RUN_CHECKPOINT_GATES": "true",
                "SYNREG_CHECKPOINT_GATE_STRICT": "true",
            },
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
        )
        for i in range(OOD_PILOT_GPU_SHARDS)
    ]
    for i, job in enumerate(jobs):
        _wait_done(job, label=f"ood_deepset_shard_{i}", session=session)

    return f"OK suite_id={OOD_PILOT_SUITE_ID} shards={OOD_PILOT_GPU_SHARDS}"


def run_synthetic_regression_ood_full_prep(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 1: OOD prep — indexes all 200 datasets under suite_id=ood_linear_full_v1."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print("[INFO] OOD full Phase 1: prep …", flush=True)
    _wait_done(
        _submit_ood_prep(session, OOD_FULL_SUITE_ID, OOD_FULL_N_DATASETS, bench_rt),
        label="ood_full_prep",
        session=session,
    )
    return f"run_synthetic_regression_ood_full_prep: ok suite_id={OOD_FULL_SUITE_ID}"


def run_synthetic_regression_ood_full_deepset_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 2: DeepSet — 10 GPU shards on DEEPSET_GPU_POOL."""
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)
    print(f"[INFO] OOD full Phase 2: DeepSet ({OOD_FULL_GPU_SHARDS} shards) …", flush=True)
    deepset_jobs = [
        _submit_synreg(
            session=session,
            label=f"ood_full_deepset_shard_{i}",
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=_synreg_shard_env(
                mode="deepset",
                suite_id=OOD_FULL_SUITE_ID,
                num_shards=OOD_FULL_GPU_SHARDS,
                shard_index=i,
                results_stage=OOD_FULL_PARTS_PREFIX,
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
                },
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=None,
            external_access_integrations=None,
        )
        for i in range(OOD_FULL_GPU_SHARDS)
    ]
    _wait_job_group([(f"ood_full_deepset_shard_{i}", job) for i, job in enumerate(deepset_jobs)], session)
    return f"run_synthetic_regression_ood_full_deepset_evaluation: ok shards={OOD_FULL_GPU_SHARDS}"


def run_synthetic_regression_ood_full_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
) -> str:
    """Phase 3: Baselines — 3 CPU shards on DEEPSET_CPU_POOL."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(f"[INFO] OOD full Phase 3: baselines ({SYNREG_CPU_SHARDS} shards) …", flush=True)
    baseline_jobs = [
        _submit_synreg(
            session=session,
            label=f"ood_full_baseline_shard_{i}",
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=OOD_FULL_SUITE_ID,
                num_shards=SYNREG_CPU_SHARDS,
                shard_index=i,
                results_stage=OOD_FULL_PARTS_PREFIX,
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=SYNREG_BASELINE_PIP,
            external_access_integrations=SYNREG_PYPI_EAI,
        )
        for i in range(SYNREG_CPU_SHARDS)
    ]
    _wait_job_group([(f"ood_full_baseline_shard_{i}", job) for i, job in enumerate(baseline_jobs)], session)
    return f"run_synthetic_regression_ood_full_baseline_evaluation: ok shards={SYNREG_CPU_SHARDS}"


def run_synthetic_regression_ood_full_autogluon_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 4: AutoGluon — 30 CPU shards on AUTOGLUON_CPU_POOL (batched)."""
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    print(f"[INFO] OOD full Phase 4: AutoGluon ({SYNREG_AUTOGLUON_SHARDS} shards) …", flush=True)
    all_ag_shards = list(range(SYNREG_AUTOGLUON_SHARDS))
    total_ag_submitted = 0
    for batch in _batched(all_ag_shards, SYNREG_AUTOGLUON_CONCURRENT_NODES_DEFAULT):
        ag_batch_jobs = []
        for i in batch:
            lbl = f"ood_full_ag_shard_{i}"
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="autogluon",
                    suite_id=OOD_FULL_SUITE_ID,
                    num_shards=SYNREG_AUTOGLUON_SHARDS,
                    shard_index=i,
                    results_stage=OOD_FULL_PARTS_PREFIX,
                ),
                runtime_environment=ag_rt,
                entrypoint="evaluate_synthetic_regression.py",
                target_instances=1,
                pip_requirements=SYNREG_AG_PIP,
                external_access_integrations=SYNREG_PYPI_EAI,
            )
            ag_batch_jobs.append((lbl, job))
        _wait_job_group(ag_batch_jobs, session)
        total_ag_submitted += len(batch)
        print(f"[INFO] OOD full AG batch done: {total_ag_submitted}/{SYNREG_AUTOGLUON_SHARDS}", flush=True)
    return f"run_synthetic_regression_ood_full_autogluon_evaluation: ok shards={SYNREG_AUTOGLUON_SHARDS}"


def run_synthetic_regression_ood_full_aggregation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 5: Aggregation — 1 CPU job; outputs to OOD_FULL_OUTPUT_STAGE."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print("[INFO] OOD full Phase 5: aggregation …", flush=True)
    agg_job = _submit_synreg(
        session=session,
        label="ood_full_aggregate",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_MODE": "aggregate",
            "SYNTHETIC_REGRESSION_SUITE_ID": OOD_FULL_SUITE_ID,
            "SYNREG_RESULTS_STAGE": OOD_FULL_PARTS_PREFIX,
            "SYNREG_OUTPUT_STAGE": OOD_FULL_OUTPUT_STAGE,
            "SYNREG_EXPECTED_DEEPSET_SHARDS":  str(OOD_FULL_GPU_SHARDS),
            "SYNREG_EXPECTED_BASELINE_SHARDS": str(SYNREG_CPU_SHARDS),
            "SYNREG_EXPECTED_AG_SHARDS":       str(SYNREG_AUTOGLUON_SHARDS),
        },
        runtime_environment=bench_rt,
        entrypoint="evaluate_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(agg_job, label="ood_full_aggregate", session=session)
    return f"run_synthetic_regression_ood_full_aggregation: ok output={OOD_FULL_OUTPUT_STAGE}"


def run_synthetic_regression_ood_full_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """All-in-one convenience wrapper. Calls all 5 phase functions in sequence."""
    run_synthetic_regression_ood_full_prep(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_deepset_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_baseline_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_autogluon_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_aggregation(session, bench_rt, ag_rt)
    return (f"run_synthetic_regression_ood_full_evaluation: ok suite_id={OOD_FULL_SUITE_ID} "
            f"deepset={OOD_FULL_GPU_SHARDS} baselines={SYNREG_CPU_SHARDS} "
            f"ag={SYNREG_AUTOGLUON_SHARDS} output={OOD_FULL_OUTPUT_STAGE}")


def run_synthetic_regression_pipeline(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    """
    Run the full pipeline in sequence:
    1. runtime_probes
    2. capacity_probe
    3. prep
    4. deepset_evaluation
    5. baseline_evaluation
    6. autogluon_evaluation
    7. aggregation
    """
    status_parts = []

    def _phase(fn_name: str, fn) -> None:
        print(f"\n[PIPELINE] === {fn_name} ===", flush=True)
        result = fn(session, prep_runtime_environment, benchmark_runtime_environment,
                    autogluon_runtime_environment)
        status_parts.append(f"{fn_name}: {result}")
        print(f"[PIPELINE] {fn_name} done.", flush=True)

    _phase("runtime_probes", run_synthetic_regression_runtime_probes)
    _phase("capacity_probe", run_synthetic_regression_capacity_probe)
    _phase("prep", run_synthetic_regression_prep)
    _phase("deepset_evaluation", run_synthetic_regression_deepset_evaluation)
    _phase("baseline_evaluation", run_synthetic_regression_baseline_evaluation)
    _phase("autogluon_evaluation", run_synthetic_regression_autogluon_evaluation)
    _phase("aggregation", run_synthetic_regression_aggregation)

    return "\n".join(status_parts)


# ---------------------------------------------------------------------------
# Combined suite — primary A/B/C/D + OOD E/F/G/H → linear_all_v1 (400 datasets)
# ---------------------------------------------------------------------------

COMBINED_SUITE_ID      = "linear_all_v1"
COMBINED_N_DATASETS    = 400
COMBINED_PARTS_PREFIX  = f"@EVALUATION_RESULTS_STAGE/regression/{COMBINED_SUITE_ID}"
COMBINED_OUTPUT_STAGE  = "@EVALUATION_RESULTS_STAGE/combined"


def run_synthetic_regression_combined_prep(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 1: Submit prepare_synthetic_regression.py with SYNTHETIC_REGRESSION_SUITE_ID=linear_all_v1.

    The script detects the combined suite ID and calls prepare_combined_suite(), which
    copies index rows from both the primary in-distribution suite and the OOD full suite.
    Both source suites must be indexed before calling this procedure.
    """
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    job = _submit_synreg(
        session=session,
        label="combined_prep",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_SUITE_ID": COMBINED_SUITE_ID,
        },
        runtime_environment=bench_rt,
        entrypoint="prepare_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(job, label="combined_prep", session=session)
    return f"run_synthetic_regression_combined_prep: ok suite_id={COMBINED_SUITE_ID}"


def run_synthetic_regression_combined_deepset_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 2: DeepSet — 10 GPU shards on DEEPSET_GPU_POOL."""
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)
    print(f"[INFO] Combined Phase 2: DeepSet ({SYNREG_GPU_SHARDS} shards) …", flush=True)
    deepset_jobs = [
        _submit_synreg(
            session=session,
            label=f"combined_deepset_shard_{i}",
            compute_pool=DEEPSET_GPU_POOL,
            env_vars=_synreg_shard_env(
                mode="deepset",
                suite_id=COMBINED_SUITE_ID,
                num_shards=SYNREG_GPU_SHARDS,
                shard_index=i,
                results_stage=COMBINED_PARTS_PREFIX,
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
                },
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=None,
            external_access_integrations=None,
        )
        for i in range(SYNREG_GPU_SHARDS)
    ]
    _wait_job_group(
        [(f"combined_deepset_shard_{i}", job) for i, job in enumerate(deepset_jobs)],
        session,
    )
    return f"run_synthetic_regression_combined_deepset_evaluation: ok shards={SYNREG_GPU_SHARDS}"


def run_synthetic_regression_combined_aggregation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    expected_ag_shards=None,
    expected_baseline_shards=None,
    expected_deepset_shards=None,
) -> str:
    """Phase 5: Aggregation — 1 CPU job; outputs to COMBINED_OUTPUT_STAGE."""
    proc = "run_synthetic_regression_combined_aggregation"
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
        f"[INFO] Combined Phase 5: aggregation "
        f"(expected deepset={resolved_deepset} baseline={resolved_baseline} ag={resolved_ag}) …",
        flush=True,
    )
    agg_job = _submit_synreg(
        session=session,
        label="combined_aggregate",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_MODE": "aggregate",
            "SYNTHETIC_REGRESSION_SUITE_ID": COMBINED_SUITE_ID,
            "SYNREG_RESULTS_STAGE": COMBINED_PARTS_PREFIX,
            "SYNREG_OUTPUT_STAGE": COMBINED_OUTPUT_STAGE,
            "SYNREG_EXPECTED_DEEPSET_SHARDS":  str(resolved_deepset),
            "SYNREG_EXPECTED_BASELINE_SHARDS": str(resolved_baseline),
            "SYNREG_EXPECTED_AG_SHARDS":       str(resolved_ag),
        },
        runtime_environment=bench_rt,
        entrypoint="evaluate_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(agg_job, label="combined_aggregate", session=session)
    return (
        f"run_synthetic_regression_combined_aggregation: ok output={COMBINED_OUTPUT_STAGE} "
        f"expected_deepset={resolved_deepset} expected_baseline={resolved_baseline} "
        f"expected_ag={resolved_ag}"
    )


def _run_baseline_shards_batched(
    *,
    session,
    procedure_name: str,
    label_prefix: str,
    suite_id: str,
    results_stage: str,
    runtime_environment: str,
    baseline_concurrent_nodes=None,
) -> int:
    concurrency = _resolve_baseline_concurrent_nodes(
        procedure_name,
        baseline_concurrent_nodes,
    )
    submitted = 0
    for batch in _batched(list(range(SYNREG_CPU_SHARDS)), concurrency):
        jobs = []
        for i in batch:
            lbl = f"{label_prefix}_{i}"
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=DEEPSET_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="baselines",
                    suite_id=suite_id,
                    num_shards=SYNREG_CPU_SHARDS,
                    shard_index=i,
                    results_stage=results_stage,
                ),
                runtime_environment=runtime_environment,
                entrypoint="evaluate_synthetic_regression.py",
                target_instances=1,
                pip_requirements=SYNREG_BASELINE_PIP,
                external_access_integrations=SYNREG_PYPI_EAI,
            )
            jobs.append((lbl, job))
        _wait_job_group(jobs, session)
        submitted += len(batch)
        print(
            f"[INFO] {procedure_name} baseline batch done: "
            f"{submitted}/{SYNREG_CPU_SHARDS}",
            flush=True,
        )
    return concurrency


def _run_autogluon_shards_batched(
    *,
    session,
    procedure_name: str,
    label_prefix: str,
    suite_id: str,
    results_stage: str,
    runtime_environment: str,
    autogluon_concurrent_nodes=None,
) -> int:
    concurrency = _resolve_autogluon_concurrent_nodes(
        procedure_name,
        autogluon_concurrent_nodes,
    )
    submitted = 0
    for batch in _batched(list(range(SYNREG_AUTOGLUON_SHARDS)), concurrency):
        jobs = []
        for i in batch:
            lbl = f"{label_prefix}_{i}"
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="autogluon",
                    suite_id=suite_id,
                    num_shards=SYNREG_AUTOGLUON_SHARDS,
                    shard_index=i,
                    results_stage=results_stage,
                ),
                runtime_environment=runtime_environment,
                entrypoint="evaluate_synthetic_regression.py",
                target_instances=1,
                pip_requirements=SYNREG_AG_PIP,
                external_access_integrations=SYNREG_PYPI_EAI,
            )
            jobs.append((lbl, job))
        _wait_job_group(jobs, session)
        submitted += len(batch)
        print(
            f"[INFO] {procedure_name} AutoGluon batch done: "
            f"{submitted}/{SYNREG_AUTOGLUON_SHARDS}",
            flush=True,
        )
    return concurrency


def run_synthetic_regression_ood_full_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
) -> str:
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    concurrency = _run_baseline_shards_batched(
        session=session,
        procedure_name="run_synthetic_regression_ood_full_baseline_evaluation",
        label_prefix="ood_full_baseline_shard",
        suite_id=OOD_FULL_SUITE_ID,
        results_stage=OOD_FULL_PARTS_PREFIX,
        runtime_environment=bench_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )
    return (
        f"run_synthetic_regression_ood_full_baseline_evaluation: ok "
        f"shards={SYNREG_CPU_SHARDS} concurrency={concurrency}"
    )


def run_synthetic_regression_ood_full_autogluon_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_concurrent_nodes=None,
) -> str:
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    concurrency = _run_autogluon_shards_batched(
        session=session,
        procedure_name="run_synthetic_regression_ood_full_autogluon_evaluation",
        label_prefix="ood_full_ag_shard",
        suite_id=OOD_FULL_SUITE_ID,
        results_stage=OOD_FULL_PARTS_PREFIX,
        runtime_environment=ag_rt,
        autogluon_concurrent_nodes=autogluon_concurrent_nodes,
    )
    return (
        f"run_synthetic_regression_ood_full_autogluon_evaluation: ok "
        f"shards={SYNREG_AUTOGLUON_SHARDS} concurrency={concurrency}"
    )


def run_synthetic_regression_ood_full_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
    autogluon_concurrent_nodes=None,
) -> str:
    run_synthetic_regression_ood_full_prep(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_deepset_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_baseline_evaluation(
        session,
        bench_rt,
        ag_rt,
        baseline_concurrent_nodes,
    )
    run_synthetic_regression_ood_full_autogluon_evaluation(
        session,
        bench_rt,
        ag_rt,
        autogluon_concurrent_nodes,
    )
    run_synthetic_regression_ood_full_aggregation(session, bench_rt, ag_rt)
    return (
        f"run_synthetic_regression_ood_full_evaluation: ok suite_id={OOD_FULL_SUITE_ID} "
        f"deepset={OOD_FULL_GPU_SHARDS} baselines={SYNREG_CPU_SHARDS} "
        f"ag={SYNREG_AUTOGLUON_SHARDS} output={OOD_FULL_OUTPUT_STAGE}"
    )


def run_synthetic_regression_combined_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
) -> str:
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    concurrency = _run_baseline_shards_batched(
        session=session,
        procedure_name="run_synthetic_regression_combined_baseline_evaluation",
        label_prefix="combined_baseline_shard",
        suite_id=COMBINED_SUITE_ID,
        results_stage=COMBINED_PARTS_PREFIX,
        runtime_environment=bench_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )
    return (
        f"run_synthetic_regression_combined_baseline_evaluation: ok "
        f"shards={SYNREG_CPU_SHARDS} concurrency={concurrency}"
    )


def run_synthetic_regression_combined_autogluon_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
    autogluon_entrypoint=None,
) -> str:
    """Phase 4: AutoGluon — distributed work-item clusters on AUTOGLUON_CPU_POOL.

    Each logical shard is submitted as a Snowflake MLJob with target_instances=workers_per_shard,
    backed by the Ray work-item entrypoint. The driver inside each MLJob owns exactly one shard
    index and writes exactly one AutoGluon_shard{i}_of_{N}_detailed.csv file.

    SQL callers:
      CALL run_synthetic_regression_combined_autogluon_evaluation(bench_rt, ag_rt,
           cluster_shards, workers_per_shard, task_cpus, concurrent_clusters,
           time_limit_secs, presets, entrypoint);
    """
    proc = "run_synthetic_regression_combined_autogluon_evaluation"

    cluster_shards = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_CLUSTER_SHARDS",
        sql_arg=autogluon_cluster_shards, env_var="SYNREG_AUTOGLUON_CLUSTER_SHARDS",
        default=SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT,
    )
    workers_per_shard = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_WORKERS_PER_SHARD",
        sql_arg=autogluon_workers_per_shard, env_var="SYNREG_AUTOGLUON_WORKERS_PER_SHARD",
        default=SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT,
    )
    task_cpus = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_TASK_CPUS",
        sql_arg=autogluon_task_cpus, env_var="AUTOGLUON_TASK_CPUS",
        default=SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT,
    )
    concurrent_clusters = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_CONCURRENT_CLUSTERS",
        sql_arg=autogluon_concurrent_clusters, env_var="SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS",
        default=SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS_DEFAULT,
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
    resolved_entrypoint = _resolve_runtime_string_param(
        procedure_name=proc, name="AUTOGLUON_ENTRYPOINT",
        sql_arg=autogluon_entrypoint, env_var="SYNREG_AUTOGLUON_ENTRYPOINT",
        default=SYNREG_AUTOGLUON_ENTRYPOINT_DEFAULT,
    )

    if concurrent_clusters > cluster_shards:
        raise ValueError(
            f"{proc}: autogluon_concurrent_clusters={concurrent_clusters} > "
            f"autogluon_cluster_shards={cluster_shards}. "
            "Concurrent clusters cannot exceed total cluster shards."
        )

    max_requested_nodes = concurrent_clusters * workers_per_shard
    print(
        f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} "
        f"cluster_shards={cluster_shards} workers_per_shard={workers_per_shard} "
        f"task_cpus={task_cpus} concurrent_clusters={concurrent_clusters} "
        f"max_requested_nodes={max_requested_nodes} "
        f"time_limit={time_limit} presets={presets!r} entrypoint={resolved_entrypoint!r}",
        flush=True,
    )

    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)

    ag_min_tmp = os.getenv(
        "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", "5368709120"
    )

    all_shard_indices = list(range(cluster_shards))
    total_submitted = 0

    for batch in _batched(all_shard_indices, concurrent_clusters):
        jobs = []
        for shard_index in batch:
            lbl = f"combined_ag_cluster_{shard_index}"
            try:
                job = _submit_synreg(
                    session=session,
                    label=lbl,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars=_synreg_shard_env(
                        mode="autogluon",
                        suite_id=COMBINED_SUITE_ID,
                        num_shards=cluster_shards,
                        shard_index=shard_index,
                        results_stage=COMBINED_PARTS_PREFIX,
                        extra_env={
                            "SYNREG_AUTOGLUON_DISTRIBUTED_MODE": "ray_work_items",
                            "SYNREG_AUTOGLUON_CLUSTER_SHARDS": str(cluster_shards),
                            "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(workers_per_shard),
                            "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS": str(concurrent_clusters),
                            "AUTOGLUON_TIME_LIMIT": str(time_limit),
                            "AUTOGLUON_PRESETS": presets,
                            "AUTOGLUON_TASK_CPUS": str(task_cpus),
                            "SYNREG_OUTPUT_STAGE": COMBINED_OUTPUT_STAGE,
                            "SYNREG_EXPECTED_AG_SHARDS": str(cluster_shards),
                            "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                        },
                    ),
                    runtime_environment=ag_rt,
                    entrypoint=resolved_entrypoint,
                    target_instances=workers_per_shard,
                    pip_requirements=SYNREG_AG_PIP,
                    external_access_integrations=SYNREG_PYPI_EAI,
                )
                jobs.append((lbl, job))
            except Exception as e:
                if _is_node_quota_error(e):
                    raise RuntimeError(
                        f"[QUOTA] Node quota exceeded during {proc}.\n"
                        f"Requested clusters={concurrent_clusters}, "
                        f"workers_per_shard={workers_per_shard}, "
                        f"total_nodes={max_requested_nodes} on {AUTOGLUON_CPU_POOL}.\n"
                        "Remediation: lower SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS, "
                        "lower SYNREG_AUTOGLUON_WORKERS_PER_SHARD, suspend idle pools, "
                        "or request a higher Snowflake node quota."
                    ) from e
                raise
        _wait_job_group(jobs, session)
        total_submitted += len(batch)
        print(
            f"[INFO] Combined AG cluster batch done: {total_submitted}/{cluster_shards}",
            flush=True,
        )

    return (
        f"run_synthetic_regression_combined_autogluon_evaluation: ok "
        f"suite_id={COMBINED_SUITE_ID} cluster_shards={cluster_shards} "
        f"workers_per_shard={workers_per_shard} task_cpus={task_cpus} "
        f"concurrent_clusters={concurrent_clusters} max_requested_nodes={max_requested_nodes} "
        f"time_limit={time_limit} presets={presets!r} entrypoint={resolved_entrypoint!r}"
    )


def run_synthetic_regression_combined_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
    autogluon_entrypoint=None,
) -> str:
    """All-in-one combined suite wrapper — runs all 5 phases in sequence.

    Prerequisites: linear_poisson_v1_recommended and ood_linear_full_v1 must be
    indexed before calling this procedure.
    """
    proc = "run_synthetic_regression_combined_evaluation"
    # Resolve cluster_shards early so aggregation gets the same value.
    resolved_cluster_shards = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_CLUSTER_SHARDS",
        sql_arg=autogluon_cluster_shards, env_var="SYNREG_AUTOGLUON_CLUSTER_SHARDS",
        default=SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT,
    )
    resolved_concurrency = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="SYNREG_BASELINE_CONCURRENT_NODES",
        sql_arg=baseline_concurrent_nodes, env_var="SYNREG_BASELINE_CONCURRENT_NODES",
        default=SYNREG_BASELINE_CONCURRENT_NODES_DEFAULT,
    )

    run_synthetic_regression_combined_prep(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_deepset_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_baseline_evaluation(
        session, bench_rt, ag_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )
    run_synthetic_regression_combined_autogluon_evaluation(
        session, bench_rt, ag_rt,
        autogluon_cluster_shards=autogluon_cluster_shards,
        autogluon_workers_per_shard=autogluon_workers_per_shard,
        autogluon_task_cpus=autogluon_task_cpus,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters,
        autogluon_time_limit=autogluon_time_limit,
        autogluon_presets=autogluon_presets,
        autogluon_entrypoint=autogluon_entrypoint,
    )
    run_synthetic_regression_combined_aggregation(
        session, bench_rt, ag_rt,
        expected_ag_shards=resolved_cluster_shards,
    )
    return (
        f"run_synthetic_regression_combined_evaluation: ok "
        f"suite_id={COMBINED_SUITE_ID} "
        f"deepset={SYNREG_GPU_SHARDS} baselines={SYNREG_CPU_SHARDS} "
        f"baseline_concurrency={resolved_concurrency} "
        f"ag_cluster_shards={resolved_cluster_shards} output={COMBINED_OUTPUT_STAGE}"
    )


def run_synthetic_regression_combined_baseline_capacity_probe(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
) -> str:
    """Capacity probe: submit N capacity_probe.py jobs to DEEPSET_CPU_POOL.

    Tests whether the baseline CPU pool can scale to the requested node count
    before committing to a full combined baseline evaluation run.
    """
    proc = "run_synthetic_regression_combined_baseline_capacity_probe"
    n_probes = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="BASELINE_CONCURRENT_NODES",
        sql_arg=baseline_concurrent_nodes, env_var="SYNREG_BASELINE_CONCURRENT_NODES",
        default=SYNREG_BASELINE_CONCURRENT_NODES_DEFAULT,
    )
    print(
        f"[INFO] {proc}: submitting {n_probes} capacity probes to {DEEPSET_CPU_POOL}",
        flush=True,
    )
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    jobs = []
    for i in range(n_probes):
        lbl = f"combined_baseline_cap_probe_{i}"
        try:
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=DEEPSET_CPU_POOL,
                env_vars={"CAPACITY_PROBE_INDEX": str(i), "CAPACITY_PROBE_TOTAL": str(n_probes)},
                runtime_environment=bench_rt,
                entrypoint="capacity_probe.py",
                target_instances=1,
                pip_requirements=None,
                external_access_integrations=None,
            )
            jobs.append((lbl, job))
        except Exception as e:
            if _is_node_quota_error(e):
                raise RuntimeError(
                    f"[QUOTA] Node quota exceeded during {proc}.\n"
                    f"Requested baseline_concurrent_nodes={n_probes} on {DEEPSET_CPU_POOL}.\n"
                    "Remediation: lower SYNREG_BASELINE_CONCURRENT_NODES, "
                    "suspend idle pools, or request a higher Snowflake node quota."
                ) from e
            raise
    _wait_job_group(jobs, session)
    return (
        f"{proc}: ok baseline_concurrent_nodes={n_probes} pool={DEEPSET_CPU_POOL}"
    )


def run_synthetic_regression_combined_autogluon_capacity_probe(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_concurrent_clusters=None,
) -> str:
    """Capacity probe: test the concurrent node envelope for distributed AutoGluon.

    Submits autogluon_concurrent_clusters capacity_probe.py jobs to AUTOGLUON_CPU_POOL,
    each requesting autogluon_workers_per_shard target instances. This verifies the pool
    can satisfy concurrent_clusters * workers_per_shard nodes simultaneously.

    Recommended default: 6 clusters x 4 workers = 24 CPU_X64_M nodes.
    """
    proc = "run_synthetic_regression_combined_autogluon_capacity_probe"
    cluster_shards = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_CLUSTER_SHARDS",
        sql_arg=autogluon_cluster_shards, env_var="SYNREG_AUTOGLUON_CLUSTER_SHARDS",
        default=SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT,
    )
    workers_per_shard = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_WORKERS_PER_SHARD",
        sql_arg=autogluon_workers_per_shard, env_var="SYNREG_AUTOGLUON_WORKERS_PER_SHARD",
        default=SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT,
    )
    concurrent_clusters = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="AUTOGLUON_CONCURRENT_CLUSTERS",
        sql_arg=autogluon_concurrent_clusters, env_var="SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS",
        default=SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS_DEFAULT,
    )
    if concurrent_clusters > cluster_shards:
        raise ValueError(
            f"{proc}: autogluon_concurrent_clusters={concurrent_clusters} > "
            f"autogluon_cluster_shards={cluster_shards}."
        )
    total_requested_nodes = concurrent_clusters * workers_per_shard
    print(
        f"[INFO] {proc}: submitting {concurrent_clusters} probes to {AUTOGLUON_CPU_POOL} "
        f"each with target_instances={workers_per_shard} "
        f"(total_requested_nodes={total_requested_nodes})",
        flush=True,
    )
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    jobs = []
    for i in range(concurrent_clusters):
        lbl = f"combined_ag_cap_probe_{i}"
        try:
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars={
                    "CAPACITY_PROBE_INDEX": str(i),
                    "CAPACITY_PROBE_TOTAL": str(concurrent_clusters),
                },
                runtime_environment=ag_rt,
                entrypoint="capacity_probe.py",
                target_instances=workers_per_shard,
                pip_requirements=None,
                external_access_integrations=None,
            )
            jobs.append((lbl, job))
        except Exception as e:
            if _is_node_quota_error(e):
                raise RuntimeError(
                    f"[QUOTA] Node quota exceeded during {proc}.\n"
                    f"Requested clusters={concurrent_clusters}, "
                    f"workers_per_shard={workers_per_shard}, "
                    f"total_nodes={total_requested_nodes} on {AUTOGLUON_CPU_POOL}.\n"
                    "Remediation: lower SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS, "
                    "lower SYNREG_AUTOGLUON_WORKERS_PER_SHARD, suspend idle pools, "
                    "or request a higher Snowflake node quota."
                ) from e
            raise
    _wait_job_group(jobs, session)
    return (
        f"{proc}: ok cluster_shards={cluster_shards} workers_per_shard={workers_per_shard} "
        f"concurrent_clusters={concurrent_clusters} "
        f"total_requested_nodes={total_requested_nodes} pool={AUTOGLUON_CPU_POOL}"
    )


# ---------------------------------------------------------------------------
# Main (for direct invocation / stored proc driver entry)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entrypoint when called directly as a Snowpark stored procedure driver.
    The stored procedure SQL passes runtime names as arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic regression orchestrator")
    parser.add_argument("--phase", default="pipeline",
                        choices=[
                            "runtime_probes", "capacity_probe", "prep",
                            "deepset_evaluation", "baseline_evaluation",
                            "autogluon_evaluation", "aggregation", "pipeline",
                        ])
    parser.add_argument("--prep-rt", default="2.5.0-py311")
    parser.add_argument("--bench-rt", default="2.5.0-py311")
    parser.add_argument("--ag-rt", default="2.5.0-py311")
    args = parser.parse_args()

    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()

    phase_map = {
        "runtime_probes": run_synthetic_regression_runtime_probes,
        "capacity_probe": run_synthetic_regression_capacity_probe,
        "prep": run_synthetic_regression_prep,
        "deepset_evaluation": run_synthetic_regression_deepset_evaluation,
        "baseline_evaluation": run_synthetic_regression_baseline_evaluation,
        "autogluon_evaluation": run_synthetic_regression_autogluon_evaluation,
        "aggregation": run_synthetic_regression_aggregation,
        "pipeline": run_synthetic_regression_pipeline,
    }
    fn = phase_map[args.phase]
    result = fn(session, args.prep_rt, args.bench_rt, args.ag_rt)
    print(f"[RESULT] {result}")


if __name__ == "__main__":
    main()
