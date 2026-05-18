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
SYNREG_CPU_SHARDS = 3
SYNREG_AUTOGLUON_SHARDS = 30
SYNREG_AUTOGLUON_MAX_CONCURRENT = 30

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

    Note: Do NOT use distributed process group (no dist.init_process_group).
    target_instances=1 for all jobs.
    """
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
) -> str:
    """
    3 phases, strictly non-overlapping:
    Phase 1: 10 capacity probes on DEEPSET_GPU_POOL (bench_rt)
    Phase 2: 3 capacity probes on DEEPSET_CPU_POOL (bench_rt)
    Phase 3: 30 capacity probes on AUTOGLUON_CPU_POOL (ag_rt)
    """
    print("[INFO] Capacity probe Phase 1: DEEPSET_GPU_POOL (10 nodes) …", flush=True)
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_gpu",
        DEEPSET_GPU_POOL,
        benchmark_runtime_environment,
        SYNREG_GPU_SHARDS,
    )

    print("[INFO] Capacity probe Phase 2: DEEPSET_CPU_POOL (3 nodes) …", flush=True)
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_cpu",
        DEEPSET_CPU_POOL,
        benchmark_runtime_environment,
        SYNREG_CPU_SHARDS,
    )

    print("[INFO] Capacity probe Phase 3: AUTOGLUON_CPU_POOL (30 nodes) …", flush=True)
    _submit_and_wait_capacity_phase(
        session,
        "synreg_cap_ag",
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
) -> str:
    """
    Submit 3 independent single-instance CPU jobs (bench_rt, DEEPSET_CPU_POOL).
    pip=catboost==1.2.10, EAI=TABPFN_PYPI_EAI.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")

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
) -> str:
    """
    Submit 30 independent single-instance CPU jobs (ag_rt, AUTOGLUON_CPU_POOL).
    Batch at SYNREG_AUTOGLUON_MAX_CONCURRENT concurrent.
    pip=autogluon.tabular==1.3.0, EAI=TABPFN_PYPI_EAI.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
    ag_time_limit = os.getenv("AUTOGLUON_TIME_LIMIT", "300")
    ag_presets = os.getenv("AUTOGLUON_PRESETS", "best_quality")

    all_shards = list(range(SYNREG_AUTOGLUON_SHARDS))
    total_submitted = 0

    for batch in _batched(all_shards, SYNREG_AUTOGLUON_MAX_CONCURRENT):
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

    return f"run_synthetic_regression_autogluon_evaluation: ok shards={SYNREG_AUTOGLUON_SHARDS}"


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
    for batch in _batched(all_ag_shards, SYNREG_AUTOGLUON_MAX_CONCURRENT):
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


def run_synthetic_regression_combined_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 3: Baselines — 3 CPU shards on DEEPSET_CPU_POOL."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print(f"[INFO] Combined Phase 3: baselines ({SYNREG_CPU_SHARDS} shards) …", flush=True)
    baseline_jobs = [
        _submit_synreg(
            session=session,
            label=f"combined_baseline_shard_{i}",
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=COMBINED_SUITE_ID,
                num_shards=SYNREG_CPU_SHARDS,
                shard_index=i,
                results_stage=COMBINED_PARTS_PREFIX,
            ),
            runtime_environment=bench_rt,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            pip_requirements=SYNREG_BASELINE_PIP,
            external_access_integrations=SYNREG_PYPI_EAI,
        )
        for i in range(SYNREG_CPU_SHARDS)
    ]
    _wait_job_group(
        [(f"combined_baseline_shard_{i}", job) for i, job in enumerate(baseline_jobs)],
        session,
    )
    return f"run_synthetic_regression_combined_baseline_evaluation: ok shards={SYNREG_CPU_SHARDS}"


def run_synthetic_regression_combined_autogluon_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 4: AutoGluon — 30 CPU shards on AUTOGLUON_CPU_POOL (batched)."""
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    print(
        f"[INFO] Combined Phase 4: AutoGluon ({SYNREG_AUTOGLUON_SHARDS} shards) …",
        flush=True,
    )
    all_ag_shards = list(range(SYNREG_AUTOGLUON_SHARDS))
    total_ag_submitted = 0
    for batch in _batched(all_ag_shards, SYNREG_AUTOGLUON_MAX_CONCURRENT):
        ag_batch_jobs = []
        for i in batch:
            lbl = f"combined_ag_shard_{i}"
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="autogluon",
                    suite_id=COMBINED_SUITE_ID,
                    num_shards=SYNREG_AUTOGLUON_SHARDS,
                    shard_index=i,
                    results_stage=COMBINED_PARTS_PREFIX,
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
        print(
            f"[INFO] Combined AG batch done: {total_ag_submitted}/{SYNREG_AUTOGLUON_SHARDS}",
            flush=True,
        )
    return f"run_synthetic_regression_combined_autogluon_evaluation: ok shards={SYNREG_AUTOGLUON_SHARDS}"


def run_synthetic_regression_combined_aggregation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Phase 5: Aggregation — 1 CPU job; outputs to COMBINED_OUTPUT_STAGE."""
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    print("[INFO] Combined Phase 5: aggregation …", flush=True)
    agg_job = _submit_synreg(
        session=session,
        label="combined_aggregate",
        compute_pool=DEEPSET_CPU_POOL,
        env_vars={
            "SYNTHETIC_REGRESSION_MODE": "aggregate",
            "SYNTHETIC_REGRESSION_SUITE_ID": COMBINED_SUITE_ID,
            "SYNREG_RESULTS_STAGE": COMBINED_PARTS_PREFIX,
            "SYNREG_OUTPUT_STAGE": COMBINED_OUTPUT_STAGE,
            "SYNREG_EXPECTED_DEEPSET_SHARDS":  str(SYNREG_GPU_SHARDS),
            "SYNREG_EXPECTED_BASELINE_SHARDS": str(SYNREG_CPU_SHARDS),
            "SYNREG_EXPECTED_AG_SHARDS":       str(SYNREG_AUTOGLUON_SHARDS),
        },
        runtime_environment=bench_rt,
        entrypoint="evaluate_synthetic_regression.py",
        target_instances=1,
        pip_requirements=None,
        external_access_integrations=None,
    )
    _wait_done(agg_job, label="combined_aggregate", session=session)
    return f"run_synthetic_regression_combined_aggregation: ok output={COMBINED_OUTPUT_STAGE}"


def run_synthetic_regression_combined_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    """All-in-one convenience wrapper. Calls all 5 phase functions in sequence.

    Prerequisites: both linear_poisson_v1_recommended and ood_linear_full_v1 must be
    indexed in SYNTHETIC_REGRESSION_DATASET_INDEX before calling this procedure.
    """
    run_synthetic_regression_combined_prep(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_deepset_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_baseline_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_autogluon_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_aggregation(session, bench_rt, ag_rt)
    return (
        f"run_synthetic_regression_combined_evaluation: ok "
        f"suite_id={COMBINED_SUITE_ID} "
        f"deepset={SYNREG_GPU_SHARDS} baselines={SYNREG_CPU_SHARDS} "
        f"ag={SYNREG_AUTOGLUON_SHARDS} output={COMBINED_OUTPUT_STAGE}"
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
