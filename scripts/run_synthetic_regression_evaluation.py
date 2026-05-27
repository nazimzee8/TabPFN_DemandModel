"""
run_synthetic_regression_evaluation.py
========================================
Orchestrator for the split-phase synthetic regression evaluation suite.

Mirrors run_evaluation_test.py exactly:
  - submit_from_stage with explicit runtime_environment, env_vars,
    pip_requirements, external_access_integrations
  - Phase-gated parallelism: prep → DeepSet GPU → CPU baselines → AutoGluon → aggregate
  - Node quota error handling (395034)
  - Capacity probes in non-overlapping phases

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


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {raw!r}.")
    return value


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
SYNREG_AUTOGLUON_MAX_IN_FLIGHT_DEFAULT = int(
    os.getenv(
        "SYNREG_AUTOGLUON_MAX_IN_FLIGHT",
        str(SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT),
    )
)
SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT = int(
    os.getenv("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", "5368709120")
)
SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT = int(
    os.getenv("BENCHMARK_CPU_MAX_PROCESSED_FEATURES", "512")
)
SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT = int(
    os.getenv("BENCHMARK_CPU_MAX_MATRIX_BYTES", "2147483648")
)
SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT = int(
    os.getenv(
        "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES",
        str(SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT),
    )
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
SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS_DEFAULT = _env_positive_int(
    "SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS",
    _env_positive_int("SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS", 300),
)
SYNREG_RAY_CAPACITY_READY_POLL_SECONDS_DEFAULT = _env_positive_int(
    "SYNREG_RAY_CAPACITY_READY_POLL_SECONDS",
    _env_positive_int("SYNREG_RAY_CLUSTER_READY_POLL_SECONDS", 10),
)
SYNREG_RAY_EVALUATION_READY_TIMEOUT_SECONDS_DEFAULT = _env_positive_int(
    "SYNREG_RAY_EVALUATION_READY_TIMEOUT_SECONDS",
    _env_positive_int("SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS", 600),
)
SYNREG_RAY_EVALUATION_READY_POLL_SECONDS_DEFAULT = _env_positive_int(
    "SYNREG_RAY_EVALUATION_READY_POLL_SECONDS",
    _env_positive_int("SYNREG_RAY_CLUSTER_READY_POLL_SECONDS", 10),
)

DEEPSET_GPU_POOL = "DEEPSET_GPU_POOL"
DEEPSET_CPU_POOL = "DEEPSET_CPU_POOL"
AUTOGLUON_CPU_POOL = "AUTOGLUON_CPU_POOL"

# pip requirements (pinned)
SYNREG_BASELINE_PIP = ["catboost==1.2.10"]
SYNREG_AG_PIP = ["autogluon.tabular==1.3.0"]
SYNREG_RAY_PIP = ["ray"]
SYNREG_AG_RAY_PIP = SYNREG_AG_PIP + SYNREG_RAY_PIP
SYNREG_PYPI_EAI = ["TABPFN_PYPI_EAI"]

# ---------------------------------------------------------------------------
# SPCS custom-image backend configuration
# ---------------------------------------------------------------------------

SYNREG_AUTOGLUON_EXECUTION_BACKEND = os.getenv(
    "SYNREG_AUTOGLUON_EXECUTION_BACKEND",
    "mljob",
)
"""AutoGluon execution backend. Values: 'mljob' (default), 'spcs_job'."""

SYNREG_AUTOGLUON_SPCS_IMAGE = os.getenv(
    "SYNREG_AUTOGLUON_SPCS_IMAGE",
    "",
)
"""Full OCI image reference for SPCS AutoGluon containers.
Example: <account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY/tabpfn-autogluon-ray:1.0.0
Required when SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job.
"""

SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT = int(
    os.getenv("SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT", "6379")
)
SYNREG_AUTOGLUON_SPCS_RAY_START_TIMEOUT_SECONDS = int(
    os.getenv("SYNREG_AUTOGLUON_SPCS_RAY_START_TIMEOUT_SECONDS", "600")
)
SYNREG_AUTOGLUON_SPCS_RAY_POLL_SECONDS = int(
    os.getenv("SYNREG_AUTOGLUON_SPCS_RAY_POLL_SECONDS", "10")
)

SPCS_RAY_HEAD_RESOURCES = "ray_head"
SPCS_RAY_DRIVER_RESOURCES = "ray_driver"
SPCS_RAY_COORDINATOR_RESOURCES = "ray_coordinator"
SPCS_RAY_WORKER_RESOURCES = "ray_worker"
SPCS_SINGLE_NODE_RESOURCES = "single_node"
SPCS_PROBE_RESOURCES = "probe"
SPCS_WORKER_CAPACITY_PROBE_RESOURCES = "worker_capacity_probe"

# Env var names for object-store memory caps (passed into container specs).
# Workers and coordinators each cap Ray's object store to avoid OOM on fixed-memory nodes.
_SYNREG_COORDINATOR_OBJ_STORE_ENV = "SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES"
_SYNREG_WORKER_OBJ_STORE_ENV = "SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES"

_VALID_AUTOGLUON_EXECUTION_BACKENDS = {"mljob", "spcs_job"}

SYNREG_CHECKPOINT_LOADING_MODES = {"deepset", "baselines"}
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
    "matplotlib", "snowflake.snowpark", "ray",
]
_PREP_PROBE_IMPORTS = ["torch", "numpy", "scipy", "pyarrow", "snowflake.snowpark"]

# ---------------------------------------------------------------------------
# Backend validation helper
# ---------------------------------------------------------------------------


def _validate_autogluon_backend(backend: str, image: str, procedure_name: str) -> None:
    if backend not in _VALID_AUTOGLUON_EXECUTION_BACKENDS:
        raise ValueError(
            f"{procedure_name}: SYNREG_AUTOGLUON_EXECUTION_BACKEND={backend!r} is not valid. "
            f"Supported values: {sorted(_VALID_AUTOGLUON_EXECUTION_BACKENDS)!r}."
        )
    if backend == "spcs_job" and not image:
        raise ValueError(
            f"{procedure_name}: SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job requires "
            "SYNREG_AUTOGLUON_SPCS_IMAGE to be set to the full OCI image reference."
        )


def _resolve_spcs_image(procedure_name: str, image_arg=None) -> str:
    """Resolve the SPCS image from the procedure arg, with env fallback for legacy calls."""
    if image_arg is not None:
        candidate = str(image_arg).strip()
        if candidate and candidate.lower() != "spcs_job":
            _validate_autogluon_backend("spcs_job", candidate, procedure_name)
            return candidate

    image = os.getenv("SYNREG_AUTOGLUON_SPCS_IMAGE", SYNREG_AUTOGLUON_SPCS_IMAGE).strip()
    _validate_autogluon_backend("spcs_job", image, procedure_name)
    return image


def _parse_spcs_image_reference(image: str) -> dict[str, str | None]:
    """Parse a Snowflake image reference into repository URL and image tag/digest."""
    normalized = image.strip().rstrip("/")
    if "@" in normalized:
        repo_and_image, digest = normalized.split("@", 1)
    else:
        repo_and_image, digest = normalized, None
    repo_url, sep, image_name = repo_and_image.rpartition("/")
    if not sep or not repo_url or not image_name:
        raise ValueError(
            "SYNREG_AUTOGLUON_SPCS_IMAGE must be a full image reference like "
            "<repository_url>/tabpfn-autogluon-ray:1.0.0 or <repository_url>/name@sha256:..."
        )
    tag = None
    if digest is None and ":" in image_name:
        image_name, tag = image_name.rsplit(":", 1)
    return {
        "repository_url": repo_url,
        "image_name": image_name,
        "tag": tag,
        "digest": digest,
    }


def _row_dict_lower(row) -> dict[str, object]:
    if hasattr(row, "as_dict"):
        return {str(k).lower(): v for k, v in row.as_dict().items()}
    return {}


def _verify_spcs_image_in_repository(session, image: str) -> None:
    """Operational verification: confirm the image reference exists in a Snowflake image repository.

    Queries ``SHOW IMAGE REPOSITORIES`` and checks whether any known repository URL is a
    substring of the supplied image reference. Issues a warning (not a fatal error) if the
    query fails or no repository matches — SHOW IMAGE REPOSITORIES may require privileges
    that the stored-procedure role does not have.

    Call this at the start of each SPCS orchestration function so that a missing push is
    surfaced before containers are submitted, not after a long wait.
    """
    try:
        parsed = _parse_spcs_image_reference(image)
        rows = session.sql("SHOW IMAGE REPOSITORIES").collect()
        matched_repo_name = None
        matched_repo_url = None
        seen_repo_urls: list[str] = []
        for row in rows:
            d = _row_dict_lower(row)
            url = d.get("repository_url")
            name = d.get("name")
            if url is None and len(row) > 0:
                url = str(row[0])
            if url:
                normalized_url = str(url).rstrip("/")
                seen_repo_urls.append(normalized_url)
                if normalized_url == parsed["repository_url"]:
                    matched_repo_url = normalized_url
                    matched_repo_name = str(name) if name else None
                    break
        if not matched_repo_url:
            print(
                f"[WARN] Image repository URL {parsed['repository_url']!r} was not found in "
                f"SHOW IMAGE REPOSITORIES results ({seen_repo_urls}). Run "
                "SHOW IMAGES IN IMAGE REPOSITORY <repository_name> manually before evaluation.",
                flush=True,
            )
            return
        if not matched_repo_name:
            print(
                f"[WARN] Repository URL {matched_repo_url!r} matched, but SHOW IMAGE REPOSITORIES "
                "did not expose a repository name. Run SHOW IMAGES IN IMAGE REPOSITORY manually "
                f"to confirm {image!r}.",
                flush=True,
            )
            return
        try:
            image_rows = session.sql(f"SHOW IMAGES IN IMAGE REPOSITORY {matched_repo_name}").collect()
            expected_name = str(parsed["image_name"])
            expected_tag = parsed["tag"]
            expected_digest = parsed["digest"]
            for row in image_rows:
                d = _row_dict_lower(row)
                row_values = {str(v) for v in d.values() if v is not None}
                row_name = str(d.get("image_name") or d.get("name") or "")
                row_tag = str(d.get("tag") or d.get("image_tag") or "") or None
                row_digest = str(d.get("digest") or d.get("image_digest") or "") or None
                if expected_digest:
                    if row_name.endswith(expected_name) and row_digest == expected_digest:
                        print(f"[INFO] SPCS image digest verified exactly: {image!r}", flush=True)
                        return
                    if expected_name in row_values and expected_digest in row_values:
                        print(f"[INFO] SPCS image digest verified exactly: {image!r}", flush=True)
                        return
                elif row_name.endswith(expected_name) and row_tag == expected_tag:
                    print(f"[INFO] SPCS image tag verified exactly: {image!r}", flush=True)
                    return
                elif expected_name in row_values and expected_tag in row_values:
                    print(f"[INFO] SPCS image tag verified exactly: {image!r}", flush=True)
                    return
            print(
                f"[WARN] Repository {matched_repo_name!r} exists, but image {image!r} was not "
                "matched by exact name/tag or digest in SHOW IMAGES output. Run "
                f"SHOW IMAGES IN IMAGE REPOSITORY {matched_repo_name}; before evaluation.",
                flush=True,
            )
            return
        except Exception as exc:
            print(
                f"[WARN] Could not query SHOW IMAGES for repository {matched_repo_name!r}: {exc}. "
                f"Run SHOW IMAGES IN IMAGE REPOSITORY {matched_repo_name}; manually to confirm {image!r}.",
                flush=True,
            )
            return
    except Exception as exc:
        print(
            f"[WARN] Exact SPCS image verification could not run: {exc}. "
            "Falling back to repository URL prefix verification.",
            flush=True,
        )
    try:
        rows = session.sql("SHOW IMAGE REPOSITORIES").collect()
        repo_urls: list[str] = []
        for row in rows:
            url = None
            if hasattr(row, "as_dict"):
                d = {str(k).lower(): v for k, v in row.as_dict().items()}
                url = d.get("repository_url") or d.get("name")
            if url is None and len(row) > 0:
                url = str(row[0])
            if url:
                repo_urls.append(str(url).rstrip("/"))
        matched = any(image.startswith(r) or r in image for r in repo_urls if r)
        if matched:
            print(
                f"[INFO] SPCS image repository verified: {image!r} matches a known repository.",
                flush=True,
            )
        else:
            print(
                f"[WARN] Image {image!r} was not matched in Snowflake image repositories "
                f"({repo_urls}). Ensure the image has been pushed with "
                "'docker push <repository_url>/tabpfn-autogluon-ray:<tag>' before running.",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[WARN] Could not query image repositories (may lack privilege): {exc}. "
            "Proceeding without image verification — run SHOW IMAGE REPOSITORIES manually "
            "to confirm the image is present.",
            flush=True,
        )


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


def _stage_file_exists(session, stage_path: str, filename: str) -> bool:
    rows = session.sql(f"LIST {stage_path}").collect()
    return any(str(row[0]).rstrip("/").endswith(f"/{filename}") for row in rows)


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


def _single_wave_concurrency_error(
    *,
    procedure_name: str,
    compute_pool: str,
    arg_name: str,
    requested: int,
    required: int,
    required_name: str,
    remediation: str,
) -> ValueError:
    return ValueError(
        f"{procedure_name}: {arg_name}={requested} is invalid for single-wave "
        f"execution; required {arg_name}={required} to match {required_name}. "
        f"Compute pool: {compute_pool}. Remediation: {remediation}"
    )


def _resolve_single_wave_baseline_concurrency(
    procedure_name: str,
    sql_arg=None,
    *,
    shard_count: int = SYNREG_CPU_SHARDS,
) -> int:
    raw_value = (
        sql_arg
        if sql_arg is not None
        else os.getenv("SYNREG_BASELINE_CONCURRENT_NODES", SYNREG_BASELINE_CONCURRENT_NODES_DEFAULT)
    )
    requested = _parse_positive_int(
        raw_value,
        name="BASELINE_CONCURRENT_NODES",
        procedure_name=procedure_name,
    )
    if requested != shard_count:
        raise _single_wave_concurrency_error(
            procedure_name=procedure_name,
            compute_pool=DEEPSET_CPU_POOL,
            arg_name="BASELINE_CONCURRENT_NODES",
            requested=requested,
            required=shard_count,
            required_name=f"BASELINE_SHARDS={shard_count}",
            remediation=(
                "request enough Snowflake quota for the full baseline shard count, "
                "or increase BASELINE_SHARDS through the SYNREG_BASELINE_SHARDS env var "
                "or the BASELINE_SHARDS SQL argument; "
                "lower concurrency values are rejected rather than batched."
            ),
        )
    return requested


def _resolve_single_wave_autogluon_concurrency(
    procedure_name: str,
    sql_arg=None,
    *,
    shard_count: int = SYNREG_AUTOGLUON_SHARDS,
) -> int:
    raw_value = (
        sql_arg
        if sql_arg is not None
        else os.getenv("SYNREG_AUTOGLUON_CONCURRENT_NODES", SYNREG_AUTOGLUON_CONCURRENT_NODES_DEFAULT)
    )
    requested = _parse_positive_int(
        raw_value,
        name="AUTOGLUON_CONCURRENT_NODES",
        procedure_name=procedure_name,
    )
    if requested != shard_count:
        raise _single_wave_concurrency_error(
            procedure_name=procedure_name,
            compute_pool=AUTOGLUON_CPU_POOL,
            arg_name="AUTOGLUON_CONCURRENT_NODES",
            requested=requested,
            required=shard_count,
            required_name=f"SYNREG_AUTOGLUON_SHARDS={shard_count}",
            remediation=(
                "request enough Snowflake quota for the full legacy AutoGluon "
                "shard count or change the shard count through a supported runtime "
                "argument; lower concurrency values are rejected rather than batched."
            ),
        )
    return requested


def _resolve_single_wave_autogluon_clusters(
    *,
    procedure_name: str,
    concurrent_clusters_arg,
    cluster_shards: int,
) -> int:
    concurrent_clusters = _resolve_positive_int_runtime_param(
        procedure_name=procedure_name,
        name="AUTOGLUON_CONCURRENT_CLUSTERS",
        sql_arg=concurrent_clusters_arg,
        env_var="SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS",
        default=SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS_DEFAULT,
    )
    if concurrent_clusters != cluster_shards:
        raise _single_wave_concurrency_error(
            procedure_name=procedure_name,
            compute_pool=AUTOGLUON_CPU_POOL,
            arg_name="AUTOGLUON_CONCURRENT_CLUSTERS",
            requested=concurrent_clusters,
            required=cluster_shards,
            required_name=f"AUTOGLUON_CLUSTER_SHARDS={cluster_shards}",
            remediation=(
                "request enough Snowflake quota for all distributed AutoGluon "
                "clusters or lower AUTOGLUON_CLUSTER_SHARDS through the combined "
                "AutoGluon API; lower concurrent cluster values are rejected rather "
                "than batched."
            ),
        )
    return concurrent_clusters


def _parse_nonneg_int(value, *, name: str, procedure_name: str) -> int:
    """Like _parse_positive_int but allows 0 (used for AUTOGLUON_CLUSTER_SHARDS=0 → single-node mode)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{procedure_name}: {name} must be a non-negative integer; got {value!r}."
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"{procedure_name}: {name} must be a non-negative integer; got {value!r}."
        )
    return parsed


def _resolve_nonneg_int_runtime_param(
    *,
    procedure_name: str,
    name: str,
    sql_arg,
    env_var: str,
    default: int,
) -> int:
    raw_value = sql_arg if sql_arg is not None else os.getenv(env_var, default)
    return _parse_nonneg_int(
        raw_value,
        name=name,
        procedure_name=procedure_name,
    )


class _AutoGluonExecutionPlan:
    """Resolved execution plan for combined AutoGluon evaluation and capacity probes.

    mode="ray_clusters":
        output_shards = cluster_shards > 0
        workers_per_shard >= 1 (typically 4)
        entrypoint = autogluon_ray.py
        target_instances = workers_per_shard
        capacity_probe_entrypoint = ray_capacity_probe.py
        uses_ray = True

    mode="single_node_shards":
        output_shards = concurrent_units (= AUTOGLUON_CONCURRENT_CLUSTERS)
        workers_per_shard = 1 (enforced)
        entrypoint = evaluate_synthetic_regression.py
        target_instances = 1
        capacity_probe_entrypoint = capacity_probe.py
        uses_ray = False
    """

    __slots__ = (
        "mode", "output_shards", "workers_per_shard", "concurrent_units",
        "entrypoint", "target_instances", "capacity_probe_entrypoint", "uses_ray",
    )

    def __init__(
        self, *, mode, output_shards, workers_per_shard, concurrent_units,
        entrypoint, target_instances, capacity_probe_entrypoint, uses_ray,
    ):
        self.mode = mode
        self.output_shards = output_shards
        self.workers_per_shard = workers_per_shard
        self.concurrent_units = concurrent_units
        self.entrypoint = entrypoint
        self.target_instances = target_instances
        self.capacity_probe_entrypoint = capacity_probe_entrypoint
        self.uses_ray = uses_ray

    def __repr__(self) -> str:
        return (
            f"_AutoGluonExecutionPlan(mode={self.mode!r}, output_shards={self.output_shards}, "
            f"workers_per_shard={self.workers_per_shard}, concurrent_units={self.concurrent_units}, "
            f"entrypoint={self.entrypoint!r}, target_instances={self.target_instances}, "
            f"capacity_probe_entrypoint={self.capacity_probe_entrypoint!r}, uses_ray={self.uses_ray})"
        )


def _resolve_combined_autogluon_execution_plan(
    *,
    procedure_name: str,
    cluster_shards_arg,
    workers_per_shard_arg,
    concurrent_clusters_arg,
) -> _AutoGluonExecutionPlan:
    """Resolve and validate the combined AutoGluon execution plan.

    Returns an _AutoGluonExecutionPlan with mode="ray_clusters" or "single_node_shards".
    Entrypoints are derived internally from the mode and are not accepted as arguments.

    Ray distributed cluster-shard mode (cluster_shards > 0):
        - AUTOGLUON_CLUSTER_SHARDS > 0
        - AUTOGLUON_WORKERS_PER_SHARD >= 1
        - AUTOGLUON_CONCURRENT_CLUSTERS must equal AUTOGLUON_CLUSTER_SHARDS
        - entrypoint is always autogluon_ray.py (derived internally)
        - target_instances = workers_per_shard
        - output_shards = cluster_shards

    Single-node shard mode (cluster_shards == 0):
        - AUTOGLUON_CLUSTER_SHARDS == 0
        - AUTOGLUON_WORKERS_PER_SHARD must equal 1
        - AUTOGLUON_CONCURRENT_CLUSTERS is interpreted as concurrent single-node shard count
        - entrypoint is always evaluate_synthetic_regression.py (derived internally)
        - target_instances = 1
        - output_shards = concurrent_clusters

    Raises ValueError for invalid combinations.
    """
    cluster_shards = _resolve_nonneg_int_runtime_param(
        procedure_name=procedure_name,
        name="AUTOGLUON_CLUSTER_SHARDS",
        sql_arg=cluster_shards_arg,
        env_var="SYNREG_AUTOGLUON_CLUSTER_SHARDS",
        default=SYNREG_AUTOGLUON_CLUSTER_SHARDS_DEFAULT,
    )

    if cluster_shards == 0:
        # ------------------------------------------------------------------ #
        # Single-node shard mode                                               #
        # ------------------------------------------------------------------ #
        workers_per_shard = _resolve_positive_int_runtime_param(
            procedure_name=procedure_name,
            name="AUTOGLUON_WORKERS_PER_SHARD",
            sql_arg=workers_per_shard_arg,
            env_var="SYNREG_AUTOGLUON_WORKERS_PER_SHARD",
            default=1,
        )
        if workers_per_shard != 1:
            raise ValueError(
                f"{procedure_name}: AUTOGLUON_CLUSTER_SHARDS=0 selects single-node shard mode; "
                f"AUTOGLUON_WORKERS_PER_SHARD must be 1, got {workers_per_shard}. "
                "Single-node mode does not support multi-instance MLJobs. "
                "Set AUTOGLUON_CLUSTER_SHARDS > 0 to use Ray distributed mode."
            )
        # In single-node mode, AUTOGLUON_CONCURRENT_CLUSTERS is the shard/concurrency count.
        concurrent_shards = _resolve_positive_int_runtime_param(
            procedure_name=procedure_name,
            name="AUTOGLUON_CONCURRENT_CLUSTERS",
            sql_arg=concurrent_clusters_arg,
            env_var="SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS",
            default=SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS_DEFAULT,
        )
        # Entrypoint is derived internally — single-node mode always uses evaluate_synthetic_regression.py.
        return _AutoGluonExecutionPlan(
            mode="single_node_shards",
            output_shards=concurrent_shards,
            workers_per_shard=1,
            concurrent_units=concurrent_shards,
            entrypoint="evaluate_synthetic_regression.py",
            target_instances=1,
            capacity_probe_entrypoint="capacity_probe.py",
            uses_ray=False,
        )

    else:
        # ------------------------------------------------------------------ #
        # Ray distributed cluster-shard mode                                  #
        # ------------------------------------------------------------------ #
        workers_per_shard = _resolve_positive_int_runtime_param(
            procedure_name=procedure_name,
            name="AUTOGLUON_WORKERS_PER_SHARD",
            sql_arg=workers_per_shard_arg,
            env_var="SYNREG_AUTOGLUON_WORKERS_PER_SHARD",
            default=SYNREG_AUTOGLUON_WORKERS_PER_SHARD_DEFAULT,
        )
        concurrent_clusters = _resolve_single_wave_autogluon_clusters(
            procedure_name=procedure_name,
            concurrent_clusters_arg=concurrent_clusters_arg,
            cluster_shards=cluster_shards,
        )
        # Entrypoint is derived internally — Ray mode always uses autogluon_ray.py.
        return _AutoGluonExecutionPlan(
            mode="ray_clusters",
            output_shards=cluster_shards,
            workers_per_shard=workers_per_shard,
            concurrent_units=concurrent_clusters,
            entrypoint="autogluon_ray.py",
            target_instances=workers_per_shard,
            capacity_probe_entrypoint="ray_capacity_probe.py",
            uses_ray=True,
        )


def _resolve_baseline_concurrent_nodes(
    procedure_name: str,
    sql_arg=None,
    *,
    shard_count: int = SYNREG_CPU_SHARDS,
) -> int:
    return _resolve_single_wave_baseline_concurrency(
        procedure_name,
        sql_arg,
        shard_count=shard_count,
    )


def _resolve_baseline_shard_count(
    procedure_name: str,
    sql_arg=None,
) -> int:
    """Resolve BASELINE_SHARDS from SQL arg, env var SYNREG_BASELINE_SHARDS, or SYNREG_CPU_SHARDS."""
    return _resolve_positive_int_runtime_param(
        procedure_name=procedure_name,
        name="BASELINE_SHARDS",
        sql_arg=sql_arg,
        env_var="SYNREG_BASELINE_SHARDS",
        default=SYNREG_CPU_SHARDS,
    )


def _resolve_autogluon_concurrent_nodes(
    procedure_name: str,
    sql_arg=None,
    *,
    shard_count: int = SYNREG_AUTOGLUON_SHARDS,
) -> int:
    return _resolve_single_wave_autogluon_concurrency(
        procedure_name,
        sql_arg,
        shard_count=shard_count,
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
# Multi-instance entrypoint allowlist
# ---------------------------------------------------------------------------

_MULTI_INSTANCE_SYNREG_ENTRYPOINTS = {
    "autogluon_ray.py",
    "autogluon_worker_access_probe.py",
    "ray_capacity_probe.py",
}


def _entrypoint_basename(entrypoint: str) -> str:
    return entrypoint.replace("\\", "/").rsplit("/", 1)[-1]


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
    if (
        target_instances > 1
        and _entrypoint_basename(entrypoint) not in _MULTI_INSTANCE_SYNREG_ENTRYPOINTS
    ):
        raise RuntimeError(
            f"Refusing target_instances={target_instances} for entrypoint={entrypoint!r}. "
            "Multi-instance synthetic regression jobs are only permitted for Ray-coordinated "
            "entrypoints: " + ", ".join(sorted(_MULTI_INSTANCE_SYNREG_ENTRYPOINTS))
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
                    "Remediation: request quota increase, suspend idle pools, or wait for pool capacity."
                ) from e
            raise
    _wait_job_group(jobs, session)


# ---------------------------------------------------------------------------
# SPCS job service helpers
# ---------------------------------------------------------------------------

def _quote_spcs_env_value(value: str) -> str:
    """Escape a value for inclusion in a SPCS YAML env block (double-quoted string)."""
    # Escape backslash and double-quote; wrap in double quotes.
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _env_nonempty(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def _resolve_spcs_resource_profile(
    prefix: str,
    *,
    default_cpu_request: str,
    default_cpu_limit: str,
    default_memory_request: str,
    default_memory_limit: str,
) -> dict[str, str]:
    """Resolve SPCS resource requests/limits for one container role."""
    cpu = os.getenv(f"{prefix}_CPU")
    memory = os.getenv(f"{prefix}_MEMORY")
    cpu_request_default = str(cpu).strip() if cpu and str(cpu).strip() else default_cpu_request
    cpu_limit_default = str(cpu).strip() if cpu and str(cpu).strip() else default_cpu_limit
    memory_request_default = (
        str(memory).strip() if memory and str(memory).strip() else default_memory_request
    )
    memory_limit_default = (
        str(memory).strip() if memory and str(memory).strip() else default_memory_limit
    )
    return {
        "cpu_request": _env_nonempty(f"{prefix}_CPU_REQUEST", cpu_request_default),
        "cpu_limit": _env_nonempty(f"{prefix}_CPU_LIMIT", cpu_limit_default),
        "memory_request": _env_nonempty(f"{prefix}_MEMORY_REQUEST", memory_request_default),
        "memory_limit": _env_nonempty(f"{prefix}_MEMORY_LIMIT", memory_limit_default),
    }


def _spcs_resources_for_role(role: str) -> dict[str, str]:
    """Return role-specific SPCS resources for custom-image AutoGluon jobs."""
    if role == SPCS_RAY_HEAD_RESOURCES:
        return _resolve_spcs_resource_profile(
            "SYNREG_SPCS_RAY_HEAD",
            default_cpu_request="0.5",
            default_cpu_limit="0.5",
            default_memory_request="2Gi",
            default_memory_limit="4Gi",
        )
    if role == SPCS_RAY_DRIVER_RESOURCES:
        return _resolve_spcs_resource_profile(
            "SYNREG_SPCS_RAY_DRIVER",
            default_cpu_request="0.5",
            default_cpu_limit="1",
            default_memory_request="2Gi",
            default_memory_limit="4Gi",
        )
    if role == SPCS_RAY_COORDINATOR_RESOURCES:
        # Coordinator runs the Ray head (minimal CPU) AND the AutoGluon driver Python process.
        # Slightly more CPU/memory than a pure head, but far less than workers which run fits.
        return _resolve_spcs_resource_profile(
            "SYNREG_SPCS_RAY_COORDINATOR",
            default_cpu_request="1",
            default_cpu_limit="2",
            default_memory_request="4Gi",
            default_memory_limit="8Gi",
        )
    if role == SPCS_RAY_WORKER_RESOURCES:
        return _resolve_spcs_resource_profile(
            "SYNREG_SPCS_RAY_WORKER",
            default_cpu_request="4",
            default_cpu_limit="4",
            default_memory_request="16Gi",
            default_memory_limit="16Gi",
        )
    if role == SPCS_SINGLE_NODE_RESOURCES:
        return _resolve_spcs_resource_profile(
            "SYNREG_SPCS_SINGLE_NODE",
            default_cpu_request="4",
            default_cpu_limit="4",
            default_memory_request="16Gi",
            default_memory_limit="16Gi",
        )
    if role == SPCS_WORKER_CAPACITY_PROBE_RESOURCES:
        return _spcs_resources_for_role(SPCS_RAY_WORKER_RESOURCES)
    if role == SPCS_PROBE_RESOURCES:
        return _resolve_spcs_resource_profile(
            "SYNREG_SPCS_PROBE",
            default_cpu_request="0.5",
            default_cpu_limit="0.5",
            default_memory_request="2Gi",
            default_memory_limit="2Gi",
        )
    raise ValueError(f"Unknown SPCS resource role: {role!r}")


def _build_spcs_job_spec(
    *,
    image: str,
    args: list,
    env_vars: dict,
    cpu: str | None = None,
    memory: str | None = None,
    cpu_request: str | None = None,
    cpu_limit: str | None = None,
    memory_request: str | None = None,
    memory_limit: str | None = None,
    resource_role: str = SPCS_RAY_WORKER_RESOURCES,
    enable_snowflake_service_token: bool = True,
    endpoints: list[dict] | None = None,
) -> str:
    """Build a SPCS job service YAML spec string.

    Args:
        image:    Full OCI image reference.
        args:     Arguments passed to the ENTRYPOINT (python). First element is the script path.
        env_vars: Dict of environment variable name -> value (all coerced to str).
        cpu:      Backward-compatible CPU override that sets request and limit.
        memory:   Backward-compatible memory override that sets request and limit.
        cpu_request/cpu_limit/memory_request/memory_limit: Explicit resource values.
        resource_role: Role-specific defaults used when explicit values are omitted.
        enable_snowflake_service_token: When True (default), adds
            ``snowflakeService: {enabled: true}`` so SPCS mounts the OAuth token
            at ``/snowflake/session/token``. Required for any container that creates
            a Snowpark session via ``create_snowpark_session()`` / ``spcs_snowpark_session_probe.py``.
        endpoints: Optional list of TCP/HTTP endpoint dicts to expose from the container.
            Each dict must have ``name`` (str) and ``port`` (int); ``protocol`` defaults to "TCP".
            Example: ``[{"name": "ray-head", "port": 6379, "protocol": "TCP"}]``
            Required for coordinator containers so workers can reach the Ray head port.

    Returns:
        YAML spec string suitable for use in EXECUTE JOB SERVICE FROM SPECIFICATION $$.
    """
    # Build env block
    env_lines = []
    for k, v in sorted(env_vars.items()):
        env_lines.append(f"        {k}: {_quote_spcs_env_value(v)}")
    env_block = "\n".join(env_lines)

    # Build args block
    args_block = "\n".join(f'        - "{a}"' for a in args)
    resources = _spcs_resources_for_role(resource_role)
    cpu_request_value = str(cpu_request or cpu or resources["cpu_request"])
    cpu_limit_value = str(cpu_limit or cpu or resources["cpu_limit"])
    memory_request_value = str(memory_request or memory or resources["memory_request"])
    memory_limit_value = str(memory_limit or memory or resources["memory_limit"])

    spec = (
        "spec:\n"
        "  containers:\n"
        "    - name: main\n"
        f"      image: {image}\n"
        "      args:\n"
        f"{args_block}\n"
    )
    if env_lines:
        spec += "      env:\n" + env_block + "\n"
    spec += (
        "      resources:\n"
        "        requests:\n"
        f'          cpu: "{cpu_request_value}"\n'
        f'          memory: "{memory_request_value}"\n'
        "        limits:\n"
        f'          cpu: "{cpu_limit_value}"\n'
        f'          memory: "{memory_limit_value}"\n'
    )
    if endpoints:
        spec += "  endpoints:\n"
        for ep in endpoints:
            ep_name = ep["name"]
            ep_port = int(ep["port"])
            ep_proto = ep.get("protocol", "TCP")
            spec += f"    - name: {ep_name}\n"
            spec += f"      port: {ep_port}\n"
            spec += f"      protocol: {ep_proto}\n"
    if enable_snowflake_service_token:
        # Enables OAuth token injection at /snowflake/session/token.
        # Required for any container that calls create_snowpark_session() inside SPCS.
        spec += "  snowflakeService:\n    enabled: true\n"
    return spec


def _execute_spcs_job_service(
    session,
    *,
    label: str,
    compute_pool: str,
    spec: str,
) -> str:
    """Submit a SPCS job service via EXECUTE JOB SERVICE and return the job service name.

    Returns the job service name (label) for use in status polling.
    The label is normalized to uppercase and non-alphanumeric chars replaced with underscores
    to satisfy Snowflake identifier constraints.
    """
    # Snowflake job service names must be valid identifiers
    safe_label = "".join(c if c.isalnum() else "_" for c in label).upper()
    sql = (
        f"EXECUTE JOB SERVICE\n"
        f"  IN COMPUTE POOL {compute_pool}\n"
        f"  NAME = {safe_label}\n"
        f"  ASYNC = TRUE\n"
        f"  FROM SPECIFICATION $$\n"
        f"{spec}\n"
        f"$$"
    )
    print(f"[INFO] Submitting SPCS job service '{safe_label}' on pool '{compute_pool}'", flush=True)
    try:
        session.sql(sql).collect()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to submit SPCS job service '{safe_label}': {exc}"
        ) from exc
    print(f"[INFO] SPCS job service '{safe_label}' submitted.", flush=True)
    return safe_label


def _cancel_spcs_job_service(session, job_name: str) -> None:
    """Cancel a running SPCS job service, ignoring already-finished jobs."""
    try:
        session.sql(f"SELECT {job_name}!SPCS_CANCEL_JOB()").collect()
        print(f"[INFO] SPCS job service '{job_name}' canceled.", flush=True)
    except Exception as exc:
        print(
            f"[WARN] Could not cancel SPCS job service '{job_name}': {exc}",
            flush=True,
        )


def _cancel_spcs_job_group(labeled_jobs: list, session) -> None:
    for _label, job_name in labeled_jobs:
        _cancel_spcs_job_service(session, job_name)


def _spcs_dns_domain(session=None) -> str:
    """Resolve the SPCS service DNS domain for the current schema.

    Priority:
    1. SPCS_RAY_HEAD_DNS_SUFFIX env var (explicit override / fallback).
    2. SYSTEM$GET_SERVICE_DNS_DOMAIN(CURRENT_DATABASE(), CURRENT_SCHEMA()) via session.
    3. Empty string — DNS names will be unqualified (only valid if workers can resolve without suffix).

    Snowflake documents the service DNS name format as:
      <service-name-dashes>.<domain>
    where <domain> is returned by SYSTEM$GET_SERVICE_DNS_DOMAIN.
    Underscores in service names are replaced by dashes per SPCS naming rules.
    """
    override = os.getenv("SPCS_RAY_HEAD_DNS_SUFFIX", "").strip()
    if override:
        return override
    if session is not None:
        try:
            rows = session.sql(
                "SELECT SYSTEM$GET_SERVICE_DNS_DOMAIN(CURRENT_DATABASE(), CURRENT_SCHEMA())"
            ).collect()
            if rows and rows[0][0]:
                domain = str(rows[0][0]).strip()
                if domain:
                    return domain
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] SYSTEM$GET_SERVICE_DNS_DOMAIN failed; coordinator DNS will be "
                f"unqualified. Set SPCS_RAY_HEAD_DNS_SUFFIX to fix. error={exc!r}",
                flush=True,
            )
    return ""


def _spcs_run_id() -> str:
    """Generate a short unique identifier for a SPCS orchestration run.

    Embedding this in all job service labels prevents 'service name already exists'
    errors when the same orchestration function is called more than once (e.g., on
    retry or rerun), since Snowflake requires service names to be unique within a
    schema until the service is dropped.
    """
    import uuid
    return uuid.uuid4().hex[:8]


def _wait_spcs_job(session, job_name: str, poll_seconds: int = 10, timeout_seconds: int = 7200) -> None:
    """Poll SHOW JOB SERVICES until the named job service reaches DONE or FAILED."""
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while True:
        rows = session.sql(f"SHOW SERVICES LIKE '{job_name}'").collect()
        if rows:
            row = rows[0]
            # Status is typically in column index 5 or named STATUS
            status = None
            if hasattr(row, "as_dict"):
                d = {str(k).upper(): v for k, v in row.as_dict().items()}
                status = d.get("STATUS") or d.get("STATE")
            if status is None and len(row) > 5:
                status = str(row[5])
            status = str(status).upper() if status else "UNKNOWN"
        else:
            status = "NOT_FOUND"

        if status != last_status:
            print(f"[INFO] SPCS job service '{job_name}': status={status}", flush=True)
            last_status = status

        if status in ("DONE", "SUCCEEDED"):
            return
        if status in ("FAILED", "INTERNAL_ERROR", "NOT_FOUND"):
            try:
                log_rows = session.sql(
                    f"CALL SYSTEM$GET_SERVICE_LOGS('{job_name}', '0', 1000)"
                ).collect()
                logs = str(log_rows[0][0]) if log_rows else "<logs unavailable>"
            except Exception:
                logs = "<logs unavailable>"
            raise RuntimeError(
                f"[FATAL] SPCS job service '{job_name}' failed with status={status}.\n"
                f"Logs:\n{logs}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"[TIMEOUT] SPCS job service '{job_name}' did not complete within "
                f"{timeout_seconds}s. Last status: {status}."
            )
        time.sleep(poll_seconds)


def _wait_spcs_job_group(labeled_jobs: list, session) -> None:
    """Wait for all (label, job_name) SPCS job services to complete."""
    for label, job_name in labeled_jobs:
        _wait_spcs_job(session, job_name)


def _spcs_session_context_env(session) -> dict[str, str]:
    """Propagate the caller's Snowflake context into SPCS OAuth sessions."""
    try:
        rows = session.sql(
            "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE(), CURRENT_WAREHOUSE()"
        ).collect()
        if not rows:
            return {}
        row = rows[0]
        values = [row[i] for i in range(4)]
    except Exception:
        return {}
    keys = [
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
    ]
    return {
        key: str(value)
        for key, value in zip(keys, values)
        if value is not None and str(value).strip()
    }


def _submit_spcs_synreg(
    session,
    label: str,
    compute_pool: str,
    env_vars: dict,
    image: str,
    entrypoint_path: str,
    resource_role: str = SPCS_RAY_WORKER_RESOURCES,
    cpu: str | None = None,
    memory: str | None = None,
    cpu_request: str | None = None,
    cpu_limit: str | None = None,
    memory_request: str | None = None,
    memory_limit: str | None = None,
    endpoints: list[dict] | None = None,
) -> str:
    """Submit a single-container SPCS job service for synthetic regression.

    No runtime_environment, no pip_requirements, no external_access_integrations.
    AutoGluon, Ray, and all dependencies are preinstalled in the image.

    Returns the job service name for status polling.
    """
    full_env = {"HOME": "/tmp"}
    full_env.update(_spcs_session_context_env(session))
    full_env.update(env_vars)
    spec = _build_spcs_job_spec(
        image=image,
        args=[entrypoint_path],
        env_vars=full_env,
        resource_role=resource_role,
        cpu=cpu,
        memory=memory,
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        memory_request=memory_request,
        memory_limit=memory_limit,
        endpoints=endpoints,
    )
    return _execute_spcs_job_service(
        session,
        label=label,
        compute_pool=compute_pool,
        spec=spec,
    )


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


def run_synthetic_regression_capacity_probe_default(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    return run_synthetic_regression_capacity_probe(
        session,
        prep_runtime_environment,
        benchmark_runtime_environment,
        autogluon_runtime_environment,
    )


def run_synthetic_regression_baseline_capacity_probe(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    baseline_concurrent_nodes=None,
    baseline_shards=None,
) -> str:
    proc = "run_synthetic_regression_baseline_capacity_probe"
    shard_count = _resolve_baseline_shard_count(proc, baseline_shards)
    baseline_concurrency = _resolve_baseline_concurrent_nodes(
        proc, baseline_concurrent_nodes, shard_count=shard_count,
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

    # Preflight: verify checkpoint exists before wasting GPU quota.
    _ckpt_filename = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[-1]
    _ckpt_dir = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[0] + "/"
    if not _stage_file_exists(session, _ckpt_dir, _ckpt_filename):
        raise RuntimeError(
            f"[run_synthetic_regression_deepset_evaluation] Checkpoint not found: "
            f"{SYNREG_DEEPSET_CKPT_STAGE!r}. "
            f"Compute pool: {DEEPSET_GPU_POOL}. "
            f"Verify with: LIST {_ckpt_dir}; — upload checkpoint before running DeepSet evaluation."
        )

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
    baseline_shards=None,
) -> str:
    """
    Submit all baseline CPU shards (bench_rt, DEEPSET_CPU_POOL) in one wave.
    pip=catboost==1.2.10, EAI=TABPFN_PYPI_EAI.

    BASELINE_SHARDS controls how many shard files are written (default SYNREG_CPU_SHARDS=6).
    BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
    """
    proc = "run_synthetic_regression_baseline_evaluation"
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
    shard_count = _resolve_baseline_shard_count(proc, baseline_shards)
    _resolve_baseline_concurrent_nodes(proc, baseline_concurrent_nodes, shard_count=shard_count)

    jobs = []
    for i in range(shard_count):
        lbl = f"synreg_baseline_shard_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=suite_id,
                num_shards=shard_count,
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

    print(f"[INFO] Waiting for {shard_count} baseline CPU shards …", flush=True)
    _wait_job_group(jobs, session)
    return (
        f"run_synthetic_regression_baseline_evaluation: ok "
        f"shards={shard_count} concurrency={shard_count}"
    )


def run_synthetic_regression_baseline_evaluation_default(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    return run_synthetic_regression_baseline_evaluation(
        session,
        prep_runtime_environment,
        benchmark_runtime_environment,
        autogluon_runtime_environment,
    )


def run_synthetic_regression_baseline_evaluation_with_shards(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    baseline_shards,
    baseline_concurrent_nodes,
) -> str:
    """SQL handler for the (prep_rt, bench_rt, ag_rt, BASELINE_SHARDS, BASELINE_CONCURRENT_NODES) overload."""
    return run_synthetic_regression_baseline_evaluation(
        session,
        prep_runtime_environment,
        benchmark_runtime_environment,
        autogluon_runtime_environment,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        baseline_shards=baseline_shards,
    )


def run_synthetic_regression_autogluon_evaluation(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
    autogluon_concurrent_nodes=None,
) -> str:
    """
    Submit all legacy AutoGluon CPU shards (ag_rt, AUTOGLUON_CPU_POOL) in one wave.
    pip=autogluon.tabular==1.3.0, EAI=TABPFN_PYPI_EAI.
    """
    suite_id = os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", "linear_poisson_v1_recommended")
    ag_time_limit = os.getenv("AUTOGLUON_TIME_LIMIT", "300")
    ag_presets = os.getenv("AUTOGLUON_PRESETS", "best_quality")
    autogluon_concurrency = _resolve_autogluon_concurrent_nodes(
        "run_synthetic_regression_autogluon_evaluation",
        autogluon_concurrent_nodes,
    )

    jobs = []
    for i in range(SYNREG_AUTOGLUON_SHARDS):
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
    print(
        f"[INFO] Waiting for {SYNREG_AUTOGLUON_SHARDS} AutoGluon CPU shards "
        f"(single wave concurrency={autogluon_concurrency}) ...",
        flush=True,
    )
    _wait_job_group(jobs, session)

    return (
        f"run_synthetic_regression_autogluon_evaluation: ok "
        f"shards={SYNREG_AUTOGLUON_SHARDS} concurrency={autogluon_concurrency}"
    )


def run_synthetic_regression_autogluon_evaluation_default(
    session,
    prep_runtime_environment: str,
    benchmark_runtime_environment: str,
    autogluon_runtime_environment: str,
) -> str:
    return run_synthetic_regression_autogluon_evaluation(
        session,
        prep_runtime_environment,
        benchmark_runtime_environment,
        autogluon_runtime_environment,
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

    # Preflight: verify checkpoint exists before wasting GPU quota.
    _ckpt_filename = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[-1]
    _ckpt_dir = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[0] + "/"
    if not _stage_file_exists(session, _ckpt_dir, _ckpt_filename):
        raise RuntimeError(
            f"[run_synthetic_regression_ood_deepset_pilot] Checkpoint not found: "
            f"{SYNREG_DEEPSET_CKPT_STAGE!r}. "
            f"Compute pool: {DEEPSET_GPU_POOL}. "
            f"Verify with: LIST {_ckpt_dir}; — upload checkpoint before running DeepSet evaluation."
        )

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

    # Preflight: verify checkpoint exists before wasting GPU quota.
    _ckpt_filename = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[-1]
    _ckpt_dir = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[0] + "/"
    if not _stage_file_exists(session, _ckpt_dir, _ckpt_filename):
        raise RuntimeError(
            f"[run_synthetic_regression_ood_full_deepset_evaluation] Checkpoint not found: "
            f"{SYNREG_DEEPSET_CKPT_STAGE!r}. "
            f"Compute pool: {DEEPSET_GPU_POOL}. "
            f"Verify with: LIST {_ckpt_dir}; — upload checkpoint before running DeepSet evaluation."
        )

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
) -> str:
    """Phase 2: DeepSet — 10 GPU shards on DEEPSET_GPU_POOL."""
    _ensure_compute_pool_usable(session, DEEPSET_GPU_POOL)

    # Preflight: verify checkpoint exists before wasting GPU quota.
    _ckpt_filename = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[-1]
    _ckpt_dir = SYNREG_DEEPSET_CKPT_STAGE.rsplit("/", 1)[0] + "/"
    if not _stage_file_exists(session, _ckpt_dir, _ckpt_filename):
        raise RuntimeError(
            f"[run_synthetic_regression_combined_deepset_evaluation] Checkpoint not found: "
            f"{SYNREG_DEEPSET_CKPT_STAGE!r}. "
            f"Compute pool: {DEEPSET_GPU_POOL}. "
            f"Verify with: LIST {_ckpt_dir}; — upload checkpoint before running DeepSet evaluation."
        )

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


def _run_baseline_shards_single_wave(
    *,
    session,
    procedure_name: str,
    label_prefix: str,
    suite_id: str,
    results_stage: str,
    runtime_environment: str,
    baseline_shards=None,
    baseline_concurrent_nodes=None,
) -> int:
    shard_count = _resolve_baseline_shard_count(procedure_name, baseline_shards)
    _resolve_baseline_concurrent_nodes(
        procedure_name,
        baseline_concurrent_nodes,
        shard_count=shard_count,
    )
    jobs = []
    for i in range(shard_count):
        lbl = f"{label_prefix}_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=DEEPSET_CPU_POOL,
            env_vars=_synreg_shard_env(
                mode="baselines",
                suite_id=suite_id,
                num_shards=shard_count,
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
    return shard_count


def _run_autogluon_shards_single_wave(
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
    jobs = []
    for i in range(SYNREG_AUTOGLUON_SHARDS):
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
    return concurrency


def run_synthetic_regression_ood_full_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
    baseline_shards=None,
) -> str:
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    shard_count = _run_baseline_shards_single_wave(
        session=session,
        procedure_name="run_synthetic_regression_ood_full_baseline_evaluation",
        label_prefix="ood_full_baseline_shard",
        suite_id=OOD_FULL_SUITE_ID,
        results_stage=OOD_FULL_PARTS_PREFIX,
        runtime_environment=bench_rt,
        baseline_shards=baseline_shards,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )
    return (
        f"run_synthetic_regression_ood_full_baseline_evaluation: ok "
        f"shards={shard_count} concurrency={shard_count}"
    )


def run_synthetic_regression_ood_full_baseline_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_ood_full_baseline_evaluation(session, bench_rt, ag_rt)


def run_synthetic_regression_ood_full_baseline_evaluation_with_shards(
    session,
    bench_rt: str,
    ag_rt: str,
    baseline_shards,
    baseline_concurrent_nodes,
) -> str:
    """SQL handler for the (bench_rt, ag_rt, BASELINE_SHARDS, BASELINE_CONCURRENT_NODES) overload."""
    return run_synthetic_regression_ood_full_baseline_evaluation(
        session, bench_rt, ag_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        baseline_shards=baseline_shards,
    )


def run_synthetic_regression_ood_full_autogluon_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_concurrent_nodes=None,
) -> str:
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    concurrency = _run_autogluon_shards_single_wave(
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


def run_synthetic_regression_ood_full_autogluon_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_ood_full_autogluon_evaluation(session, bench_rt, ag_rt)


def run_synthetic_regression_ood_full_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
    autogluon_concurrent_nodes=None,
    baseline_shards=None,
) -> str:
    run_synthetic_regression_ood_full_prep(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_deepset_evaluation(session, bench_rt, ag_rt)
    run_synthetic_regression_ood_full_baseline_evaluation(
        session,
        bench_rt,
        ag_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        baseline_shards=baseline_shards,
    )
    run_synthetic_regression_ood_full_autogluon_evaluation(
        session,
        bench_rt,
        ag_rt,
        autogluon_concurrent_nodes,
    )
    run_synthetic_regression_ood_full_aggregation(session, bench_rt, ag_rt)
    resolved_baseline = _resolve_baseline_shard_count(
        "run_synthetic_regression_ood_full_evaluation", baseline_shards
    )
    return (
        f"run_synthetic_regression_ood_full_evaluation: ok suite_id={OOD_FULL_SUITE_ID} "
        f"deepset={OOD_FULL_GPU_SHARDS} baselines={resolved_baseline} "
        f"ag={SYNREG_AUTOGLUON_SHARDS} output={OOD_FULL_OUTPUT_STAGE}"
    )


def run_synthetic_regression_ood_full_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_ood_full_evaluation(session, bench_rt, ag_rt)


def run_synthetic_regression_combined_baseline_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
    baseline_shards=None,
) -> str:
    proc = "run_synthetic_regression_combined_baseline_evaluation"
    _resolved_shards = _resolve_baseline_shard_count(proc, baseline_shards)
    _resolve_baseline_concurrent_nodes(proc, baseline_concurrent_nodes, shard_count=_resolved_shards)
    _ensure_compute_pool_usable(session, DEEPSET_CPU_POOL)
    shard_count = _run_baseline_shards_single_wave(
        session=session,
        procedure_name="run_synthetic_regression_combined_baseline_evaluation",
        label_prefix="combined_baseline_shard",
        suite_id=COMBINED_SUITE_ID,
        results_stage=COMBINED_PARTS_PREFIX,
        runtime_environment=bench_rt,
        baseline_shards=baseline_shards,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )
    return (
        f"run_synthetic_regression_combined_baseline_evaluation: ok "
        f"shards={shard_count} concurrency={shard_count}"
    )


def run_synthetic_regression_combined_baseline_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_baseline_evaluation(session, bench_rt)


def run_synthetic_regression_combined_baseline_evaluation_with_shards(
    session,
    bench_rt: str,
    baseline_shards,
    baseline_concurrent_nodes,
) -> str:
    """SQL handler for the (bench_rt, BASELINE_SHARDS, BASELINE_CONCURRENT_NODES) overload."""
    return run_synthetic_regression_combined_baseline_evaluation(
        session, bench_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        baseline_shards=baseline_shards,
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
    ray_ready_timeout_seconds=None,
    ray_ready_poll_seconds=None,
) -> str:
    """Phase 4: AutoGluon evaluation on AUTOGLUON_CPU_POOL.

    Supports two execution modes controlled by AUTOGLUON_CLUSTER_SHARDS:

    Ray distributed cluster-shard mode (AUTOGLUON_CLUSTER_SHARDS > 0):
        Each logical shard is submitted as a Snowflake MLJob with
        target_instances=AUTOGLUON_WORKERS_PER_SHARD, backed by autogluon_ray.py.
        The driver inside each MLJob owns exactly one shard index and writes exactly
        one AutoGluon_shard{i}_of_{N}_detailed.csv file. N = cluster_shards.
        Entrypoint is always autogluon_ray.py (derived internally).

    Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS == 0):
        Each logical shard is submitted as a single-container MLJob with
        target_instances=1, using evaluate_synthetic_regression.py (mode=autogluon).
        AUTOGLUON_WORKERS_PER_SHARD must equal 1.
        AUTOGLUON_CONCURRENT_CLUSTERS is interpreted as the number of single-node shards.
        N = concurrent_clusters (= number of single-node shards). No Ray, no object store.
        Entrypoint is always evaluate_synthetic_regression.py (derived internally).

    SQL callers:
      CALL run_synthetic_regression_combined_autogluon_evaluation(bench_rt, ag_rt,
           cluster_shards, workers_per_shard, task_cpus, concurrent_clusters,
           time_limit_secs, presets);
      Optional extended form adds ray_ready_timeout_seconds and ray_ready_poll_seconds.
    """
    proc = "run_synthetic_regression_combined_autogluon_evaluation"

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
    ray_ready_timeout = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS",
        sql_arg=ray_ready_timeout_seconds,
        env_var="SYNREG_RAY_EVALUATION_READY_TIMEOUT_SECONDS",
        default=SYNREG_RAY_EVALUATION_READY_TIMEOUT_SECONDS_DEFAULT,
    )
    ray_ready_poll = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="SYNREG_RAY_CLUSTER_READY_POLL_SECONDS",
        sql_arg=ray_ready_poll_seconds,
        env_var="SYNREG_RAY_EVALUATION_READY_POLL_SECONDS",
        default=SYNREG_RAY_EVALUATION_READY_POLL_SECONDS_DEFAULT,
    )

    ag_min_tmp = os.getenv(
        "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES",
        str(SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT),
    )
    ag_max_features = os.getenv(
        "BENCHMARK_CPU_MAX_PROCESSED_FEATURES",
        str(SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT),
    )
    ag_max_matrix_bytes = os.getenv(
        "BENCHMARK_CPU_MAX_MATRIX_BYTES",
        str(SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT),
    )
    ag_max_dataset_bytes = os.getenv(
        "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES",
        str(SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT),
    )
    ag_max_in_flight = os.getenv(
        "SYNREG_AUTOGLUON_MAX_IN_FLIGHT",
        str(SYNREG_AUTOGLUON_MAX_IN_FLIGHT_DEFAULT),
    )
    worker_access_mode = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE", "driver_presigned_url")
    max_work_item_bytes = os.getenv("SYNREG_MAX_WORK_ITEM_BYTES", "8192")
    presigned_url_expiry_seconds = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_SECONDS", "86400")
    presigned_url_expiry_policy = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_POLICY", "strict")
    presigned_url_expiry_buffer_seconds = os.getenv(
        "SYNREG_PRESIGNED_URL_EXPIRY_BUFFER_SECONDS",
        "3600",
    )

    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)

    if plan.mode == "single_node_shards":
        # ------------------------------------------------------------------ #
        # Single-node shard mode: one container per shard, no Ray             #
        # ------------------------------------------------------------------ #
        autogluon_single_node_shards = plan.concurrent_units   # == plan.output_shards
        print(
            f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} "
            f"mode=single_node_shards "
            f"autogluon_single_node_shards={autogluon_single_node_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r} "
            f"entrypoint={plan.entrypoint!r}",
            flush=True,
        )
        jobs = []
        for shard_index in range(autogluon_single_node_shards):
            lbl = f"combined_ag_shard_{shard_index}"
            try:
                job = _submit_synreg(
                    session=session,
                    label=lbl,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars=_synreg_shard_env(
                        mode="autogluon",
                        suite_id=COMBINED_SUITE_ID,
                        num_shards=autogluon_single_node_shards,
                        shard_index=shard_index,
                        results_stage=COMBINED_PARTS_PREFIX,
                        extra_env={
                            "AUTOGLUON_TIME_LIMIT": str(time_limit),
                            "AUTOGLUON_PRESETS": presets,
                            "AUTOGLUON_TASK_CPUS": str(task_cpus),
                            "SYNREG_EXPECTED_AG_SHARDS": str(autogluon_single_node_shards),
                            "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                            "BENCHMARK_CPU_MAX_PROCESSED_FEATURES": ag_max_features,
                            "BENCHMARK_CPU_MAX_MATRIX_BYTES": ag_max_matrix_bytes,
                            "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES": ag_max_dataset_bytes,
                        },
                    ),
                    runtime_environment=ag_rt,
                    entrypoint=plan.entrypoint,
                    target_instances=1,
                    pip_requirements=SYNREG_AG_PIP,
                    external_access_integrations=SYNREG_PYPI_EAI,
                )
                jobs.append((lbl, job))
            except Exception as e:
                if _is_node_quota_error(e):
                    raise RuntimeError(
                        f"[QUOTA] Node quota exceeded during {proc} (single_node_shards mode).\n"
                        f"Requested autogluon_single_node_shards={autogluon_single_node_shards} "
                        f"on {AUTOGLUON_CPU_POOL}.\n"
                        "Remediation: lower AUTOGLUON_CONCURRENT_CLUSTERS (single-node shard count), "
                        "suspend idle pools, or request a higher Snowflake node quota."
                    ) from e
                raise
        _wait_job_group(jobs, session)
        return (
            f"run_synthetic_regression_combined_autogluon_evaluation: ok "
            f"suite_id={COMBINED_SUITE_ID} mode=single_node_shards "
            f"autogluon_single_node_shards={autogluon_single_node_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )

    else:
        # ------------------------------------------------------------------ #
        # Ray distributed cluster-shard mode                                  #
        # ------------------------------------------------------------------ #
        cluster_shards = plan.output_shards
        workers_per_shard = plan.workers_per_shard
        concurrent_clusters = plan.concurrent_units
        resolved_entrypoint = plan.entrypoint
        max_requested_nodes = concurrent_clusters * workers_per_shard
        print(
            f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} "
            f"mode=ray_clusters "
            f"cluster_shards={cluster_shards} workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} concurrent_clusters={concurrent_clusters} "
            f"max_requested_nodes={max_requested_nodes} "
            f"time_limit={time_limit} presets={presets!r} "
            f"ray_ready_timeout={ray_ready_timeout}s ray_ready_poll={ray_ready_poll}s "
            f"entrypoint={resolved_entrypoint!r}",
            flush=True,
        )
        jobs = []
        for shard_index in range(cluster_shards):
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
                            "SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS": str(ray_ready_timeout),
                            "SYNREG_RAY_CLUSTER_READY_POLL_SECONDS": str(ray_ready_poll),
                            "SYNREG_AUTOGLUON_MAX_IN_FLIGHT": ag_max_in_flight,
                            "SYNREG_WORKER_DATA_ACCESS_MODE": worker_access_mode,
                            "SYNREG_MAX_WORK_ITEM_BYTES": max_work_item_bytes,
                            "SYNREG_OUTPUT_STAGE": COMBINED_OUTPUT_STAGE,
                            "SYNREG_EXPECTED_AG_SHARDS": str(cluster_shards),
                            "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                            "BENCHMARK_CPU_MAX_PROCESSED_FEATURES": ag_max_features,
                            "BENCHMARK_CPU_MAX_MATRIX_BYTES": ag_max_matrix_bytes,
                            "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES": ag_max_dataset_bytes,
                        },
                    ),
                    runtime_environment=ag_rt,
                    entrypoint=resolved_entrypoint,
                    target_instances=workers_per_shard,
                    pip_requirements=SYNREG_AG_RAY_PIP,
                    external_access_integrations=SYNREG_PYPI_EAI,
                )
                jobs.append((lbl, job))
            except Exception as e:
                if _is_node_quota_error(e):
                    raise RuntimeError(
                        f"[QUOTA] Node quota exceeded during {proc} (ray_clusters mode).\n"
                        f"Requested clusters={concurrent_clusters}, "
                        f"workers_per_shard={workers_per_shard}, "
                        f"total_nodes={max_requested_nodes} on {AUTOGLUON_CPU_POOL}.\n"
                        "Remediation: lower AUTOGLUON_CLUSTER_SHARDS through the "
                        "combined AutoGluon API, lower SYNREG_AUTOGLUON_WORKERS_PER_SHARD, "
                        "suspend idle pools, or request a higher Snowflake node quota."
                    ) from e
                raise
        _wait_job_group(jobs, session)
        return (
            f"run_synthetic_regression_combined_autogluon_evaluation: ok "
            f"suite_id={COMBINED_SUITE_ID} mode=ray_clusters "
            f"cluster_shards={cluster_shards} workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} concurrent_clusters={concurrent_clusters} "
            f"max_requested_nodes={max_requested_nodes} "
            f"time_limit={time_limit} presets={presets!r} entrypoint={resolved_entrypoint!r}"
        )


def run_synthetic_regression_combined_autogluon_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_autogluon_evaluation(session, bench_rt, ag_rt)


def run_synthetic_regression_combined_evaluation(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_concurrent_nodes=None,
    baseline_shards=None,
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
) -> str:
    """All-in-one combined suite wrapper — runs all 5 phases in sequence.

    Prerequisites: linear_poisson_v1_recommended and ood_linear_full_v1 must be
    indexed before calling this procedure.

    BASELINE_SHARDS controls how many baseline shard files are written (default 6).
    BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS. Aggregation automatically
    expects the same resolved baseline shard count.
    """
    proc = "run_synthetic_regression_combined_evaluation"
    # Resolve baseline shard count early so aggregation gets the same value.
    resolved_baseline_shards = _resolve_baseline_shard_count(proc, baseline_shards)
    # Resolve the AutoGluon execution plan early so aggregation gets plan.output_shards.
    # In Ray mode:         output_shards = cluster_shards.
    # In single-node mode: output_shards = concurrent_clusters (= single-node shard count).
    ag_plan = _resolve_combined_autogluon_execution_plan(
        procedure_name=proc,
        cluster_shards_arg=autogluon_cluster_shards,
        workers_per_shard_arg=autogluon_workers_per_shard,
        concurrent_clusters_arg=autogluon_concurrent_clusters,
    )
    resolved_concurrency = _resolve_single_wave_baseline_concurrency(
        proc,
        baseline_concurrent_nodes,
        shard_count=resolved_baseline_shards,
    )

    run_synthetic_regression_combined_prep(session, bench_rt, ag_rt)
    run_synthetic_regression_combined_deepset_evaluation(session, bench_rt)
    run_synthetic_regression_combined_baseline_evaluation(
        session, bench_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        baseline_shards=baseline_shards,
    )
    run_synthetic_regression_combined_autogluon_evaluation(
        session, bench_rt, ag_rt,
        autogluon_cluster_shards=autogluon_cluster_shards,
        autogluon_workers_per_shard=autogluon_workers_per_shard,
        autogluon_task_cpus=autogluon_task_cpus,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters,
        autogluon_time_limit=autogluon_time_limit,
        autogluon_presets=autogluon_presets,
    )
    run_synthetic_regression_combined_aggregation(
        session, bench_rt,
        expected_ag_shards=ag_plan.output_shards,
        expected_baseline_shards=resolved_baseline_shards,
    )
    return (
        f"run_synthetic_regression_combined_evaluation: ok "
        f"suite_id={COMBINED_SUITE_ID} "
        f"deepset={SYNREG_GPU_SHARDS} baselines={resolved_baseline_shards} "
        f"baseline_concurrency={resolved_concurrency} "
        f"ag_mode={ag_plan.mode} ag_output_shards={ag_plan.output_shards} "
        f"output={COMBINED_OUTPUT_STAGE}"
    )


def run_synthetic_regression_combined_evaluation_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_evaluation(session, bench_rt, ag_rt)


def run_synthetic_regression_combined_evaluation_legacy_concurrency(
    session,
    bench_rt: str,
    ag_rt: str,
    baseline_concurrent_nodes,
    autogluon_concurrent_clusters,
) -> str:
    return run_synthetic_regression_combined_evaluation(
        session,
        bench_rt,
        ag_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters,
    )


def run_synthetic_regression_combined_evaluation_with_baseline_shards(
    session,
    bench_rt: str,
    ag_rt: str,
    baseline_shards,
    baseline_concurrent_nodes,
    autogluon_cluster_shards,
    autogluon_workers_per_shard,
    autogluon_task_cpus,
    autogluon_concurrent_clusters,
    autogluon_time_limit,
    autogluon_presets,
) -> str:
    """SQL handler for the full combined evaluation overload with explicit BASELINE_SHARDS."""
    return run_synthetic_regression_combined_evaluation(
        session, bench_rt, ag_rt,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
        baseline_shards=baseline_shards,
        autogluon_cluster_shards=autogluon_cluster_shards,
        autogluon_workers_per_shard=autogluon_workers_per_shard,
        autogluon_task_cpus=autogluon_task_cpus,
        autogluon_concurrent_clusters=autogluon_concurrent_clusters,
        autogluon_time_limit=autogluon_time_limit,
        autogluon_presets=autogluon_presets,
    )


def run_synthetic_regression_combined_baseline_capacity_probe(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    baseline_shards=None,
    baseline_concurrent_nodes=None,
) -> str:
    """Capacity probe: submit one probe per baseline shard to DEEPSET_CPU_POOL.

    Tests whether the baseline CPU pool can scale to the required single-wave node count
    before committing to a full combined baseline evaluation run.

    BASELINE_SHARDS controls how many probes are submitted (default SYNREG_CPU_SHARDS=6).
    BASELINE_CONCURRENT_NODES must equal BASELINE_SHARDS.
    """
    proc = "run_synthetic_regression_combined_baseline_capacity_probe"
    shard_count = _resolve_baseline_shard_count(proc, baseline_shards)
    n_probes = _resolve_single_wave_baseline_concurrency(
        proc,
        baseline_concurrent_nodes,
        shard_count=shard_count,
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
                    "Remediation: suspend idle pools, request a higher Snowflake node quota, "
                    "or change the baseline shard count through a supported runtime argument."
                ) from e
            raise
    _wait_job_group(jobs, session)
    return (
        f"{proc}: ok baseline_concurrent_nodes={n_probes} pool={DEEPSET_CPU_POOL}"
    )


def run_synthetic_regression_combined_baseline_capacity_probe_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_baseline_capacity_probe(session, bench_rt, ag_rt)


def run_synthetic_regression_combined_baseline_capacity_probe_with_shards(
    session,
    bench_rt: str,
    ag_rt: str,
    baseline_shards,
    baseline_concurrent_nodes,
) -> str:
    """SQL handler for the (bench_rt, ag_rt, BASELINE_SHARDS, BASELINE_CONCURRENT_NODES) overload."""
    return run_synthetic_regression_combined_baseline_capacity_probe(
        session, bench_rt, ag_rt,
        baseline_shards=baseline_shards,
        baseline_concurrent_nodes=baseline_concurrent_nodes,
    )


def run_synthetic_regression_combined_autogluon_capacity_probe(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_concurrent_clusters=None,
    ray_ready_timeout_seconds=None,
    ray_ready_poll_seconds=None,
) -> str:
    """Capacity probe: test the node envelope for combined AutoGluon execution.

    Supports two modes mirroring run_synthetic_regression_combined_autogluon_evaluation:

    Ray distributed mode (AUTOGLUON_CLUSTER_SHARDS > 0):
        Submits one ray_capacity_probe.py job per cluster shard to AUTOGLUON_CPU_POOL,
        each requesting AUTOGLUON_WORKERS_PER_SHARD target instances. Verifies the pool
        can satisfy cluster_shards * workers_per_shard nodes simultaneously.
        Recommended default: 6 clusters x 4 workers = 24 CPU_X64_M nodes.

    Single-node shard mode (AUTOGLUON_CLUSTER_SHARDS == 0):
        Submits one capacity_probe.py job per single-node shard (target_instances=1).
        AUTOGLUON_CONCURRENT_CLUSTERS specifies the concurrent shard count.
        No Ray, no multi-instance MLJobs.
    """
    proc = "run_synthetic_regression_combined_autogluon_capacity_probe"
    plan = _resolve_combined_autogluon_execution_plan(
        procedure_name=proc,
        cluster_shards_arg=autogluon_cluster_shards,
        workers_per_shard_arg=autogluon_workers_per_shard,
        concurrent_clusters_arg=autogluon_concurrent_clusters,
    )
    ray_ready_timeout = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS",
        sql_arg=ray_ready_timeout_seconds,
        env_var="SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS",
        default=SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS_DEFAULT,
    )
    ray_ready_poll = _resolve_positive_int_runtime_param(
        procedure_name=proc, name="SYNREG_RAY_CLUSTER_READY_POLL_SECONDS",
        sql_arg=ray_ready_poll_seconds,
        env_var="SYNREG_RAY_CAPACITY_READY_POLL_SECONDS",
        default=SYNREG_RAY_CAPACITY_READY_POLL_SECONDS_DEFAULT,
    )
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)

    if plan.mode == "single_node_shards":
        # ------------------------------------------------------------------ #
        # Single-node capacity probe: capacity_probe.py, target_instances=1   #
        # ------------------------------------------------------------------ #
        autogluon_concurrent_shards = plan.concurrent_units
        print(
            f"[INFO] {proc}: mode=single_node_shards "
            f"submitting {autogluon_concurrent_shards} probes to {AUTOGLUON_CPU_POOL} "
            f"each with target_instances=1",
            flush=True,
        )
        jobs = []
        for i in range(autogluon_concurrent_shards):
            lbl = f"combined_ag_cap_probe_{i}"
            try:
                job = _submit_synreg(
                    session=session,
                    label=lbl,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars={
                        "CAPACITY_PROBE_INDEX": str(i),
                        "CAPACITY_PROBE_TOTAL": str(autogluon_concurrent_shards),
                    },
                    runtime_environment=ag_rt,
                    entrypoint="capacity_probe.py",
                    target_instances=1,
                    pip_requirements=None,
                    external_access_integrations=None,
                )
                jobs.append((lbl, job))
            except Exception as e:
                if _is_node_quota_error(e):
                    raise RuntimeError(
                        f"[QUOTA] Node quota exceeded during {proc} (single_node_shards mode).\n"
                        f"Requested autogluon_concurrent_shards={autogluon_concurrent_shards} "
                        f"on {AUTOGLUON_CPU_POOL}.\n"
                        "Remediation: lower AUTOGLUON_CONCURRENT_CLUSTERS (single-node shard count), "
                        "suspend idle pools, or request a higher Snowflake node quota."
                    ) from e
                raise
        _wait_job_group(jobs, session)
        return (
            f"{proc}: ok mode=single_node_shards "
            f"autogluon_concurrent_shards={autogluon_concurrent_shards} "
            f"pool={AUTOGLUON_CPU_POOL}"
        )

    else:
        # ------------------------------------------------------------------ #
        # Ray distributed capacity probe: ray_capacity_probe.py               #
        # ------------------------------------------------------------------ #
        concurrent_clusters = plan.concurrent_units
        workers_per_shard = plan.workers_per_shard
        cluster_shards = plan.output_shards
        total_requested_nodes = concurrent_clusters * workers_per_shard
        print(
            f"[INFO] {proc}: mode=ray_clusters "
            f"submitting {concurrent_clusters} probes to {AUTOGLUON_CPU_POOL} "
            f"each with target_instances={workers_per_shard} "
            f"(total_requested_nodes={total_requested_nodes}) "
            f"ray_ready_timeout={ray_ready_timeout}s ray_ready_poll={ray_ready_poll}s",
            flush=True,
        )
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
                        "SYNTHETIC_REGRESSION_MODE": "autogluon_capacity_probe",
                        "SYNTHETIC_REGRESSION_SUITE_ID": COMBINED_SUITE_ID,
                        "SYNTHETIC_REGRESSION_NUM_SHARDS": str(cluster_shards),
                        "SYNTHETIC_REGRESSION_SHARD_INDEX": str(i),
                        "SYNREG_RESULTS_STAGE": COMBINED_PARTS_PREFIX,
                        "SYNREG_AUTOGLUON_DISTRIBUTED_MODE": "ray_work_items",
                        "SYNREG_AUTOGLUON_CLUSTER_SHARDS": str(cluster_shards),
                        "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(workers_per_shard),
                        "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS": str(concurrent_clusters),
                        "AUTOGLUON_TASK_CPUS": str(SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT),
                        "EXPECTED_RAY_NODES": str(workers_per_shard),
                        "EXPECTED_RAY_CPUS_MIN": str(workers_per_shard),
                        "SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS": str(ray_ready_timeout),
                        "SYNREG_RAY_CLUSTER_READY_POLL_SECONDS": str(ray_ready_poll),
                    },
                    runtime_environment=ag_rt,
                    entrypoint="ray_capacity_probe.py",
                    target_instances=workers_per_shard,
                    pip_requirements=SYNREG_AG_RAY_PIP,
                    external_access_integrations=SYNREG_PYPI_EAI,
                )
                jobs.append((lbl, job))
            except Exception as e:
                if _is_node_quota_error(e):
                    raise RuntimeError(
                        f"[QUOTA] Node quota exceeded during {proc} (ray_clusters mode).\n"
                        f"Requested clusters={concurrent_clusters}, "
                        f"workers_per_shard={workers_per_shard}, "
                        f"total_nodes={total_requested_nodes} on {AUTOGLUON_CPU_POOL}.\n"
                        "Remediation: lower AUTOGLUON_CLUSTER_SHARDS through the "
                        "combined AutoGluon API, lower SYNREG_AUTOGLUON_WORKERS_PER_SHARD, "
                        "suspend idle pools, or request a higher Snowflake node quota."
                    ) from e
                raise
        _wait_job_group(jobs, session)
        return (
            f"{proc}: ok mode=ray_clusters "
            f"cluster_shards={cluster_shards} workers_per_shard={workers_per_shard} "
            f"concurrent_clusters={concurrent_clusters} "
            f"total_requested_nodes={total_requested_nodes} "
            f"ray_ready_timeout={ray_ready_timeout}s ray_ready_poll={ray_ready_poll}s "
            f"pool={AUTOGLUON_CPU_POOL}"
        )


def run_synthetic_regression_combined_autogluon_capacity_probe_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_autogluon_capacity_probe(session, bench_rt, ag_rt)


def run_synthetic_regression_combined_autogluon_worker_access_probe(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_concurrent_clusters=None,
) -> str:
    """Validate the AutoGluon worker-local dataset access path.

    Uses the same topology resolver as combined AutoGluon capacity and evaluation:
    Ray mode submits one multi-instance MLJob per cluster shard, while single-node
    mode submits one one-instance probe per resolved shard.
    """
    proc = "run_synthetic_regression_combined_autogluon_worker_access_probe"
    plan = _resolve_combined_autogluon_execution_plan(
        procedure_name=proc,
        cluster_shards_arg=autogluon_cluster_shards,
        workers_per_shard_arg=autogluon_workers_per_shard,
        concurrent_clusters_arg=autogluon_concurrent_clusters,
    )
    access_mode = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE", "driver_presigned_url")
    probe_items = os.getenv(
        "SYNREG_WORKER_ACCESS_PROBE_ITEMS",
        str(max(1, plan.workers_per_shard)),
    )
    max_work_item_bytes = os.getenv("SYNREG_MAX_WORK_ITEM_BYTES", "8192")
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)

    if plan.mode == "single_node_shards":
        autogluon_concurrent_shards = plan.concurrent_units
        print(
            f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} "
            f"mode=single_node_shards submitting {autogluon_concurrent_shards} "
            f"worker-access probes to {AUTOGLUON_CPU_POOL} "
            f"access_mode={access_mode!r}",
            flush=True,
        )
        jobs = []
        for shard_index in range(autogluon_concurrent_shards):
            lbl = f"combined_ag_worker_access_probe_{shard_index}"
            try:
                job = _submit_synreg(
                    session=session,
                    label=lbl,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars=_synreg_shard_env(
                        mode="autogluon_worker_access_probe",
                        suite_id=COMBINED_SUITE_ID,
                        num_shards=autogluon_concurrent_shards,
                        shard_index=shard_index,
                        results_stage=COMBINED_PARTS_PREFIX,
                        extra_env={
                            "SYNREG_WORKER_ACCESS_PROBE_USE_RAY": "false",
                            "SYNREG_WORKER_ACCESS_PROBE_ITEMS": probe_items,
                            "SYNREG_WORKER_DATA_ACCESS_MODE": access_mode,
                            "SYNREG_MAX_WORK_ITEM_BYTES": max_work_item_bytes,
                        },
                    ),
                    runtime_environment=ag_rt,
                    entrypoint="autogluon_worker_access_probe.py",
                    target_instances=1,
                    pip_requirements=None,
                    external_access_integrations=None,
                )
                jobs.append((lbl, job))
            except Exception as e:
                if _is_node_quota_error(e):
                    raise RuntimeError(
                        f"[QUOTA] Node quota exceeded during {proc} (single_node_shards mode).\n"
                        f"Requested autogluon_concurrent_shards={autogluon_concurrent_shards} "
                        f"on {AUTOGLUON_CPU_POOL}.\n"
                        "Remediation: lower AUTOGLUON_CONCURRENT_CLUSTERS (single-node shard count), "
                        "suspend idle pools, or request a higher Snowflake node quota."
                    ) from e
                raise
        _wait_job_group(jobs, session)
        return (
            f"{proc}: ok mode=single_node_shards "
            f"autogluon_concurrent_shards={autogluon_concurrent_shards} "
            f"access_mode={access_mode!r} pool={AUTOGLUON_CPU_POOL}"
        )

    cluster_shards = plan.output_shards
    workers_per_shard = plan.workers_per_shard
    concurrent_clusters = plan.concurrent_units
    total_requested_nodes = concurrent_clusters * workers_per_shard
    print(
        f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} mode=ray_clusters "
        f"submitting {concurrent_clusters} worker-access probes to {AUTOGLUON_CPU_POOL} "
        f"each with target_instances={workers_per_shard} "
        f"access_mode={access_mode!r} total_requested_nodes={total_requested_nodes}",
        flush=True,
    )
    jobs = []
    for shard_index in range(cluster_shards):
        lbl = f"combined_ag_worker_access_probe_{shard_index}"
        try:
            job = _submit_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=_synreg_shard_env(
                    mode="autogluon_worker_access_probe",
                    suite_id=COMBINED_SUITE_ID,
                    num_shards=cluster_shards,
                    shard_index=shard_index,
                    results_stage=COMBINED_PARTS_PREFIX,
                    extra_env={
                        "SYNREG_WORKER_ACCESS_PROBE_USE_RAY": "true",
                        "SYNREG_WORKER_ACCESS_PROBE_ITEMS": probe_items,
                        "SYNREG_WORKER_DATA_ACCESS_MODE": access_mode,
                        "SYNREG_MAX_WORK_ITEM_BYTES": max_work_item_bytes,
                        "SYNREG_AUTOGLUON_DISTRIBUTED_MODE": "ray_work_items",
                        "SYNREG_AUTOGLUON_CLUSTER_SHARDS": str(cluster_shards),
                        "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(workers_per_shard),
                        "SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS": str(concurrent_clusters),
                        "AUTOGLUON_TASK_CPUS": str(SYNREG_AUTOGLUON_TASK_CPUS_DEFAULT),
                        "EXPECTED_RAY_NODES": str(workers_per_shard),
                        "EXPECTED_RAY_CPUS_MIN": str(workers_per_shard),
                        "SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS": str(
                            SYNREG_RAY_CAPACITY_READY_TIMEOUT_SECONDS_DEFAULT
                        ),
                        "SYNREG_RAY_CLUSTER_READY_POLL_SECONDS": str(
                            SYNREG_RAY_CAPACITY_READY_POLL_SECONDS_DEFAULT
                        ),
                    },
                ),
                runtime_environment=ag_rt,
                entrypoint="autogluon_worker_access_probe.py",
                target_instances=workers_per_shard,
                pip_requirements=SYNREG_AG_RAY_PIP,
                external_access_integrations=SYNREG_PYPI_EAI,
            )
            jobs.append((lbl, job))
        except Exception as e:
            if _is_node_quota_error(e):
                raise RuntimeError(
                    f"[QUOTA] Node quota exceeded during {proc} (ray_clusters mode).\n"
                    f"Requested clusters={concurrent_clusters}, "
                    f"workers_per_shard={workers_per_shard}, "
                    f"total_nodes={total_requested_nodes} on {AUTOGLUON_CPU_POOL}.\n"
                    "Remediation: lower AUTOGLUON_CLUSTER_SHARDS through the "
                    "combined AutoGluon API, lower SYNREG_AUTOGLUON_WORKERS_PER_SHARD, "
                    "suspend idle pools, or request a higher Snowflake node quota."
                ) from e
            raise
    _wait_job_group(jobs, session)
    return (
        f"{proc}: ok mode=ray_clusters "
        f"cluster_shards={cluster_shards} workers_per_shard={workers_per_shard} "
        f"concurrent_clusters={concurrent_clusters} "
        f"total_requested_nodes={total_requested_nodes} access_mode={access_mode!r} "
        f"pool={AUTOGLUON_CPU_POOL}"
    )


def run_synthetic_regression_combined_autogluon_worker_access_probe_default(
    session,
    bench_rt: str = "2.5.0-py311",
    ag_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_autogluon_worker_access_probe(session, bench_rt, ag_rt)


def run_synthetic_regression_combined_aggregation_default(
    session,
    bench_rt: str = "2.5.0-py311",
) -> str:
    return run_synthetic_regression_combined_aggregation(session, bench_rt)


def run_synthetic_regression_combined_aggregation_ag(
    session,
    bench_rt: str,
    expected_ag_shards,
) -> str:
    return run_synthetic_regression_combined_aggregation(
        session,
        bench_rt,
        expected_ag_shards=expected_ag_shards,
    )


# ---------------------------------------------------------------------------
def run_synthetic_regression_autogluon_import_timing_probe(
    session,
    ag_rt: str = "2.5.0-py311",
    with_pip: bool = True,
    probe_count: int = 1,
) -> str:
    """Submit lightweight AutoGluon import-timing probes to AUTOGLUON_CPU_POOL.

    Each probe is a single-instance MLJob running autogluon_import_timing_probe.py.
    The probe emits structured JSON log lines to stdout:
      - python_entrypoint_started: emitted immediately when the entrypoint starts.
        Time from MLJob submission to this event approximates scheduling + image
        startup + pip install (with_pip=True) or just scheduling + image startup
        (with_pip=False).
      - autogluon_import_complete: time spent importing autogluon.tabular in pip/preinstalled mode.
      - ray_import_complete: time spent importing ray in pip/preinstalled mode.
      - *_import_skipped: emitted in no_pip_baseline mode; the job should still succeed.
      - import_failed: emitted (then exception re-raised) if an import fails.

    with_pip=True  — pip_requirements=SYNREG_AG_PIP, external_access_integrations=SYNREG_PYPI_EAI.
                     Measures the full bootstrap overhead under production conditions.
    with_pip=False — no pip install and no AutoGluon/Ray import. Provides a no-pip
                     scheduling + startup baseline that should succeed even when
                     AutoGluon is not preinstalled.
    """
    proc = "run_synthetic_regression_autogluon_import_timing_probe"
    if probe_count < 1:
        raise ValueError(f"{proc}: probe_count must be >= 1; got {probe_count!r}.")

    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)

    deps_mode = "pip" if with_pip else "no_pip_baseline"
    pip_reqs = SYNREG_AG_PIP if with_pip else None
    eai = SYNREG_PYPI_EAI if with_pip else None

    print(
        f"[INFO] {proc}: submitting {probe_count} probe(s) to {AUTOGLUON_CPU_POOL} "
        f"with_pip={with_pip} deps_mode={deps_mode!r} runtime={ag_rt!r}",
        flush=True,
    )

    jobs = []
    for i in range(probe_count):
        lbl = f"ag_import_timing_probe_{i}"
        job = _submit_synreg(
            session=session,
            label=lbl,
            compute_pool=AUTOGLUON_CPU_POOL,
            env_vars={
                "SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE": deps_mode,
                "SYNREG_AG_IMPORT_PROBE_LABEL": f"ag_import_timing_probe_{i}_of_{probe_count}",
                "EVAL_RUNTIME_ENVIRONMENT": ag_rt,
            },
            runtime_environment=ag_rt,
            entrypoint="autogluon_import_timing_probe.py",
            target_instances=1,
            pip_requirements=pip_reqs,
            external_access_integrations=eai,
        )
        jobs.append((lbl, job))

    _wait_job_group(jobs, session)
    return (
        f"{proc}: ok with_pip={with_pip} deps_mode={deps_mode!r} "
        f"probe_count={probe_count} runtime={ag_rt!r} pool={AUTOGLUON_CPU_POOL}"
    )


def run_synthetic_regression_autogluon_import_timing_probe_default(
    session,
    ag_rt: str = "2.5.0-py311",
) -> str:
    """Default single-probe pip-mode import timing probe."""
    return run_synthetic_regression_autogluon_import_timing_probe(
        session, ag_rt=ag_rt, with_pip=True, probe_count=1
    )


# ---------------------------------------------------------------------------
# SPCS AutoGluon evaluation and probe functions
# ---------------------------------------------------------------------------

def run_synthetic_regression_combined_autogluon_spcs_evaluation(
    session,
    ag_rt: str = "spcs_job",
    autogluon_cluster_shards=None,
    autogluon_workers_per_shard=None,
    autogluon_task_cpus=None,
    autogluon_concurrent_clusters=None,
    autogluon_time_limit=None,
    autogluon_presets=None,
) -> str:
    """AutoGluon evaluation using SPCS custom image backend.

    This is the SPCS equivalent of run_synthetic_regression_combined_autogluon_evaluation
    with SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job. No runtime_environment or
    pip_requirements are used — AutoGluon, Ray, and all dependencies are preinstalled
    in the custom OCI image (SYNREG_AUTOGLUON_SPCS_IMAGE).

    Single-node mode (AUTOGLUON_CLUSTER_SHARDS == 0):
        One SPCS job service per shard. No Ray.
        Entrypoint: /app/src/evaluate_synthetic_regression.py

    Ray distributed mode (AUTOGLUON_CLUSTER_SHARDS > 0):
        Per cluster shard: one head SPCS service + N worker SPCS services + one driver job.
        Self-managed Ray: driver connects with RAY_HEAD_ADDRESS (explicit mode).
        Entrypoint: /app/scripts/autogluon_ray.py (with SYNREG_RAY_ADDRESS_MODE=explicit)
    """
    proc = "run_synthetic_regression_combined_autogluon_spcs_evaluation"

    image = _resolve_spcs_image(proc, ag_rt)

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

    ag_min_tmp = os.getenv("BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES", str(SYNREG_AUTOGLUON_MIN_TMP_FREE_BYTES_DEFAULT))
    ag_max_features = os.getenv("BENCHMARK_CPU_MAX_PROCESSED_FEATURES", str(SYNREG_AUTOGLUON_MAX_FEATURES_DEFAULT))
    ag_max_matrix_bytes = os.getenv("BENCHMARK_CPU_MAX_MATRIX_BYTES", str(SYNREG_AUTOGLUON_MAX_MATRIX_BYTES_DEFAULT))
    ag_max_dataset_bytes = os.getenv("BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES", str(SYNREG_AUTOGLUON_MAX_DATASET_BYTES_DEFAULT))
    worker_access_mode = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE", "driver_presigned_url")
    max_work_item_bytes = os.getenv("SYNREG_MAX_WORK_ITEM_BYTES", "8192")
    presigned_url_expiry_seconds = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_SECONDS", "86400")
    presigned_url_expiry_policy = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_POLICY", "strict")
    presigned_url_expiry_buffer_seconds = os.getenv(
        "SYNREG_PRESIGNED_URL_EXPIRY_BUFFER_SECONDS",
        "3600",
    )

    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    _verify_spcs_image_in_repository(session, image)
    run_id = _spcs_run_id()

    if plan.mode == "single_node_shards":
        output_shards = plan.output_shards
        print(
            f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} backend=spcs_job "
            f"mode=single_node_shards output_shards={output_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}",
            flush=True,
        )
        jobs = []
        for shard_index in range(output_shards):
            lbl = f"spcs_ag_shard_{run_id}_{shard_index}"
            env = _synreg_shard_env(
                mode="autogluon",
                suite_id=COMBINED_SUITE_ID,
                num_shards=output_shards,
                shard_index=shard_index,
                results_stage=COMBINED_PARTS_PREFIX,
                extra_env={
                    "AUTOGLUON_TIME_LIMIT": str(time_limit),
                    "AUTOGLUON_PRESETS": presets,
                    "AUTOGLUON_TASK_CPUS": str(task_cpus),
                    "SYNREG_EXPECTED_AG_SHARDS": str(output_shards),
                    "BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES": ag_min_tmp,
                    "BENCHMARK_CPU_MAX_PROCESSED_FEATURES": ag_max_features,
                    "BENCHMARK_CPU_MAX_MATRIX_BYTES": ag_max_matrix_bytes,
                    "BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES": ag_max_dataset_bytes,
                },
            )
            job_name = _submit_spcs_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=env,
                image=image,
                entrypoint_path="/app/src/evaluate_synthetic_regression.py",
                resource_role=SPCS_SINGLE_NODE_RESOURCES,
            )
            jobs.append((lbl, job_name))
        _wait_spcs_job_group(jobs, session)
        return (
            f"{proc}: ok backend=spcs_job suite_id={COMBINED_SUITE_ID} "
            f"mode=single_node_shards output_shards={output_shards} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )
    else:
        # Ray distributed mode — coordinator topology (30 containers for 6×4).
        # Each shard gets 1 coordinator (Ray head + AutoGluon driver merged) + N workers.
        # This replaces the old 3-service-per-shard model (head + workers + driver = 36).
        cluster_shards = plan.output_shards
        workers_per_shard = plan.workers_per_shard
        head_port = SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT
        print(
            f"[INFO] {proc}: suite_id={COMBINED_SUITE_ID} backend=spcs_job "
            f"mode=ray_clusters cluster_shards={cluster_shards} "
            f"workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r} "
            f"topology=coordinator+workers containers={cluster_shards + cluster_shards * workers_per_shard}",
            flush=True,
        )
        support_jobs: list = []
        coordinator_jobs: list = []
        # Resolve DNS domain: checks SPCS_RAY_HEAD_DNS_SUFFIX override first, then
        # calls SYSTEM$GET_SERVICE_DNS_DOMAIN to auto-resolve the current schema's domain.
        dns_suffix = _spcs_dns_domain(session)
        # Object store memory limits: explicit caps prevent Ray from over-allocating.
        coord_obj_store = os.getenv(_SYNREG_COORDINATOR_OBJ_STORE_ENV, "500000000")
        worker_obj_store = os.getenv(_SYNREG_WORKER_OBJ_STORE_ENV, "2000000000")

        for shard_index in range(cluster_shards):
            coord_label = f"spcs_ray_coord_{run_id}_{shard_index}"
            safe_coord_label = "".join(c if c.isalnum() else "_" for c in coord_label).upper()
            # SPCS replaces underscores with dashes in DNS names (service name portion only).
            dns_service_name = safe_coord_label.lower().replace("_", "-")
            if dns_suffix:
                coord_hostname = f"{dns_service_name}.{dns_suffix}"
            else:
                coord_hostname = dns_service_name
            # Workers connect to this address to join the per-shard Ray cluster.
            head_address = f"{coord_hostname}:{head_port}"
            print(
                f"[INFO] {proc}: shard={shard_index} coord_label={coord_label!r} "
                f"head_address={head_address!r}",
                flush=True,
            )

            # Coordinator env: Ray head config + AutoGluon driver config (all-in-one).
            # The coordinator script sets SYNREG_RAY_ADDRESS_MODE=explicit and
            # RAY_HEAD_ADDRESS=localhost:{head_port} internally before running autogluon_ray.py,
            # so we do NOT set those here (they must point to localhost, not the DNS address).
            coord_env = _synreg_shard_env(
                mode="autogluon",
                suite_id=COMBINED_SUITE_ID,
                num_shards=cluster_shards,
                shard_index=shard_index,
                results_stage=COMBINED_PARTS_PREFIX,
                extra_env={
                    # Ray head config (consumed by spcs_ray_coordinator.py)
                    "SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT": str(head_port),
                    _SYNREG_COORDINATOR_OBJ_STORE_ENV: coord_obj_store,
                    # Cluster identity (Finding 6): autogluon_ray.py verifies after ray.init()
                    "SPCS_RAY_RUN_ID": run_id,
                    "SPCS_RAY_SHARD_INDEX": str(shard_index),
                    # Driver config (passed through to autogluon_ray.py subprocess)
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
                },
            )
            coord_job = _submit_spcs_synreg(
                session=session,
                label=coord_label,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=coord_env,
                image=image,
                entrypoint_path="/app/scripts/spcs_ray_coordinator.py",
                resource_role=SPCS_RAY_COORDINATOR_RESOURCES,
                # Expose Ray head TCP port so workers in separate SPCS services can connect.
                endpoints=[{"name": "ray-head", "port": head_port, "protocol": "TCP"}],
            )
            coordinator_jobs.append((coord_label, coord_job))

            # Workers — connect to coordinator's DNS address, advertise explicit CPU capacity.
            worker_env_base = {
                "HOME": "/tmp",
                "RAY_HEAD_ADDRESS": head_address,
                "AUTOGLUON_TASK_CPUS": str(task_cpus),
                _SYNREG_WORKER_OBJ_STORE_ENV: worker_obj_store,
                "SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS": str(SYNREG_AUTOGLUON_SPCS_RAY_START_TIMEOUT_SECONDS),
            }
            for w in range(workers_per_shard):
                w_label = f"spcs_ray_worker_{run_id}_{shard_index}_{w}"
                w_job = _submit_spcs_synreg(
                    session=session,
                    label=w_label,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars=worker_env_base,
                    image=image,
                    entrypoint_path="/app/scripts/spcs_ray_worker.py",
                    resource_role=SPCS_RAY_WORKER_RESOURCES,
                )
                support_jobs.append((w_label, w_job))

        try:
            _wait_spcs_job_group(coordinator_jobs, session)
        finally:
            _cancel_spcs_job_group(support_jobs, session)
        return (
            f"{proc}: ok backend=spcs_job suite_id={COMBINED_SUITE_ID} "
            f"mode=ray_clusters cluster_shards={cluster_shards} "
            f"workers_per_shard={workers_per_shard} "
            f"task_cpus={task_cpus} time_limit={time_limit} presets={presets!r}"
        )


def run_synthetic_regression_autogluon_spcs_import_probe(
    session,
    ag_rt: str = "spcs_job",
    probe_count: int = 1,
) -> str:
    """SPCS backend import timing probe.

    Submits probe_count SPCS job services each running autogluon_import_timing_probe.py.
    Unlike the MLJob version, no pip install occurs — dependencies are preinstalled in
    the custom image. This measures pure scheduling + container startup latency.
    """
    proc = "run_synthetic_regression_autogluon_spcs_import_probe"
    image = _resolve_spcs_image(proc, ag_rt)
    if probe_count < 1:
        raise ValueError(f"{proc}: probe_count must be >= 1; got {probe_count!r}.")
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    _verify_spcs_image_in_repository(session, image)
    jobs = []
    for i in range(probe_count):
        lbl = f"spcs_ag_import_probe_{i}"
        job_name = _submit_spcs_synreg(
            session=session,
            label=lbl,
            compute_pool=AUTOGLUON_CPU_POOL,
            env_vars={
                "SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE": "preinstalled",
                "SYNREG_AG_IMPORT_PROBE_LABEL": f"spcs_ag_import_probe_{i}_of_{probe_count}",
            },
            image=image,
            entrypoint_path="/app/scripts/autogluon_import_timing_probe.py",
            resource_role=SPCS_PROBE_RESOURCES,
        )
        jobs.append((lbl, job_name))
    _wait_spcs_job_group(jobs, session)
    return (
        f"{proc}: ok probe_count={probe_count} backend=spcs_job "
        f"deps_mode=preinstalled pool={AUTOGLUON_CPU_POOL}"
    )


def run_synthetic_regression_autogluon_spcs_session_probe(
    session,
    ag_rt: str = "spcs_job",
    probe_count: int = 1,
) -> str:
    """SPCS Snowpark OAuth session probe — step 2 in the SPCS preflight runbook.

    Submits probe_count SPCS job services each running spcs_snowpark_session_probe.py.
    Validates that:
    - The SPCS spec's ``snowflakeService.enabled=true`` causes the OAuth token to be
      injected at ``/snowflake/session/token``.
    - The container can open a Snowpark session with the correct database/schema/role
      context propagated via SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA / SNOWFLAKE_ROLE.

    Run this after the import probe and before the capacity probe.
    A successful run confirms that the AutoGluon driver containers can create Snowpark
    sessions inside SPCS — required for querying SYNTHETIC_REGRESSION_DATASET_INDEX and
    calling GET_PRESIGNED_URL().
    """
    proc = "run_synthetic_regression_autogluon_spcs_session_probe"
    image = _resolve_spcs_image(proc, ag_rt)
    if probe_count < 1:
        raise ValueError(f"{proc}: probe_count must be >= 1; got {probe_count!r}.")
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    _verify_spcs_image_in_repository(session, image)
    run_id = _spcs_run_id()
    jobs = []
    for i in range(probe_count):
        lbl = f"spcs_session_probe_{run_id}_{i}"
        job_name = _submit_spcs_synreg(
            session=session,
            label=lbl,
            compute_pool=AUTOGLUON_CPU_POOL,
            env_vars={},
            image=image,
            entrypoint_path="/app/scripts/spcs_snowpark_session_probe.py",
            resource_role=SPCS_PROBE_RESOURCES,
        )
        jobs.append((lbl, job_name))
    _wait_spcs_job_group(jobs, session)
    return f"{proc}: ok backend=spcs_job probe_count={probe_count}"


def run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
    session,
    ag_rt: str = "spcs_job",
    cluster_shards=None,
    workers_per_shard=None,
    concurrent_clusters=None,
) -> str:
    """SPCS backend capacity probe: verify that custom-image containers start and Ray imports.

    Submits SPCS job services that run ray_capacity_probe.py. Validates that the
    custom image can start on the compute pool, Ray is importable, and the cluster
    reaches the expected node count.
    """
    proc = "run_synthetic_regression_combined_autogluon_spcs_capacity_probe"
    image = _resolve_spcs_image(proc, ag_rt)

    plan = _resolve_combined_autogluon_execution_plan(
        procedure_name=proc,
        cluster_shards_arg=cluster_shards,
        workers_per_shard_arg=workers_per_shard,
        concurrent_clusters_arg=concurrent_clusters,
    )
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    run_id = _spcs_run_id()

    probe_count = plan.concurrent_units
    print(
        f"[INFO] {proc}: backend=spcs_job probe_count={probe_count} "
        f"mode={plan.mode}",
        flush=True,
    )

    if not plan.uses_ray:
        jobs = []
        for i in range(probe_count):
            lbl = f"spcs_cap_probe_{run_id}_{i}"
            job_name = _submit_spcs_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars={
                    "SYNREG_AUTOGLUON_RUNTIME_DEPS_MODE": "preinstalled",
                    "SYNREG_AG_IMPORT_PROBE_LABEL": f"{lbl}_of_{probe_count}",
                },
                image=image,
                entrypoint_path="/app/scripts/autogluon_import_timing_probe.py",
                resource_role=SPCS_PROBE_RESOURCES,
            )
            jobs.append((lbl, job_name))
        _wait_spcs_job_group(jobs, session)
    else:
        support_jobs = []
        coordinator_jobs = []
        head_port = SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT
        dns_suffix = _spcs_dns_domain(session)
        coord_obj_store = os.getenv(_SYNREG_COORDINATOR_OBJ_STORE_ENV, "500000000")
        worker_obj_store = os.getenv(_SYNREG_WORKER_OBJ_STORE_ENV, "2000000000")

        for shard_index in range(plan.output_shards):
            coord_label = f"spcs_cap_ray_coord_{run_id}_{shard_index}"
            safe_coord_label = "".join(c if c.isalnum() else "_" for c in coord_label).upper()
            dns_service_name = safe_coord_label.lower().replace("_", "-")
            coord_hostname = f"{dns_service_name}.{dns_suffix}" if dns_suffix else dns_service_name
            head_address = f"{coord_hostname}:{head_port}"

            coord_env = {
                "SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT": str(head_port),
                _SYNREG_COORDINATOR_OBJ_STORE_ENV: coord_obj_store,
                "SPCS_RAY_RUN_ID": run_id,
                "SPCS_RAY_SHARD_INDEX": str(shard_index),
                # Coordinator runs ray_capacity_probe.py instead of autogluon_ray.py
                "SPCS_RAY_DRIVER_SCRIPT": "/app/scripts/ray_capacity_probe.py",
                # Probe-specific config (passed through to ray_capacity_probe.py subprocess)
                "CAPACITY_PROBE_LABEL": coord_label,
                "EXPECTED_RAY_NODES": str(plan.workers_per_shard + 1),
                "EXPECTED_RAY_CPUS_MIN": str(plan.workers_per_shard),
                "CAPACITY_PROBE_SLEEP_SECONDS": "10",
            }
            coord_job = _submit_spcs_synreg(
                session=session,
                label=coord_label,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=coord_env,
                image=image,
                entrypoint_path="/app/scripts/spcs_ray_coordinator.py",
                resource_role=SPCS_RAY_COORDINATOR_RESOURCES,
                endpoints=[{"name": "ray-head", "port": head_port, "protocol": "TCP"}],
            )
            coordinator_jobs.append((coord_label, coord_job))

            worker_env_base = {
                "RAY_HEAD_ADDRESS": head_address,
                "AUTOGLUON_TASK_CPUS": "1",
                _SYNREG_WORKER_OBJ_STORE_ENV: worker_obj_store,
                "SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS": str(SYNREG_AUTOGLUON_SPCS_RAY_START_TIMEOUT_SECONDS),
            }
            for worker_index in range(plan.workers_per_shard):
                worker_label = f"spcs_cap_ray_worker_{run_id}_{shard_index}_{worker_index}"
                worker_job = _submit_spcs_synreg(
                    session=session,
                    label=worker_label,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars=worker_env_base,
                    image=image,
                    entrypoint_path="/app/scripts/spcs_ray_worker.py",
                    resource_role=SPCS_RAY_WORKER_RESOURCES,
                )
                support_jobs.append((worker_label, worker_job))

        try:
            _wait_spcs_job_group(coordinator_jobs, session)
        finally:
            _cancel_spcs_job_group(support_jobs, session)
    return f"{proc}: ok backend=spcs_job probe_count={probe_count} mode={plan.mode}"


def run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
    session,
    ag_rt: str = "spcs_job",
    cluster_shards=None,
    workers_per_shard=None,
    concurrent_clusters=None,
) -> str:
    """SPCS backend worker dataset-access probe.

    Driver queries SYNTHETIC_REGRESSION_DATASET_INDEX and builds compact item dicts.
    Workers validate dataset access using the same production path.
    No AutoGluon training.
    """
    proc = "run_synthetic_regression_combined_autogluon_spcs_worker_access_probe"
    image = _resolve_spcs_image(proc, ag_rt)

    plan = _resolve_combined_autogluon_execution_plan(
        procedure_name=proc,
        cluster_shards_arg=cluster_shards,
        workers_per_shard_arg=workers_per_shard,
        concurrent_clusters_arg=concurrent_clusters,
    )
    _ensure_compute_pool_usable(session, AUTOGLUON_CPU_POOL)
    run_id = _spcs_run_id()
    worker_access_mode = os.getenv("SYNREG_WORKER_DATA_ACCESS_MODE", "driver_presigned_url")
    presigned_url_expiry_seconds = os.getenv("SYNREG_PRESIGNED_URL_EXPIRY_SECONDS", "86400")
    print(
        f"[INFO] {proc}: backend=spcs_job mode={plan.mode} "
        f"concurrent_units={plan.concurrent_units} "
        f"workers_per_shard={plan.workers_per_shard} "
        f"access_mode={worker_access_mode!r}",
        flush=True,
    )

    shard_count = plan.output_shards if plan.uses_ray else plan.concurrent_units
    if plan.uses_ray:
        support_jobs = []
        driver_jobs = []
        head_port = SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT
        dns_suffix = _spcs_dns_domain(session)
        coord_obj_store = os.getenv(_SYNREG_COORDINATOR_OBJ_STORE_ENV, "500000000")
        worker_obj_store = os.getenv(_SYNREG_WORKER_OBJ_STORE_ENV, "2000000000")
        for shard_index in range(shard_count):
            coord_label = f"spcs_worker_probe_ray_coord_{run_id}_{shard_index}"
            safe_coord_label = "".join(c if c.isalnum() else "_" for c in coord_label).upper()
            dns_service_name = safe_coord_label.lower().replace("_", "-")
            coord_hostname = f"{dns_service_name}.{dns_suffix}" if dns_suffix else dns_service_name
            head_address = f"{coord_hostname}:{head_port}"

            coord_env = {
                "SYNREG_AUTOGLUON_SPCS_RAY_HEAD_PORT": str(head_port),
                _SYNREG_COORDINATOR_OBJ_STORE_ENV: coord_obj_store,
                "SPCS_RAY_RUN_ID": run_id,
                "SPCS_RAY_SHARD_INDEX": str(shard_index),
                "SPCS_RAY_DRIVER_SCRIPT": "/app/scripts/autogluon_worker_access_probe.py",
                # Probe env vars (inherited by autogluon_worker_access_probe.py subprocess)
                "SYNTHETIC_REGRESSION_SUITE_ID": os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", COMBINED_SUITE_ID),
                "SYNTHETIC_REGRESSION_NUM_SHARDS": str(shard_count),
                "SYNTHETIC_REGRESSION_SHARD_INDEX": str(shard_index),
                "SYNREG_WORKER_DATA_ACCESS_MODE": worker_access_mode,
                "SYNREG_PRESIGNED_URL_EXPIRY_SECONDS": presigned_url_expiry_seconds,
                "SYNREG_WORKER_ACCESS_PROBE_USE_RAY": "true",
                "EXPECTED_RAY_NODES": str(plan.workers_per_shard + 1),
                "EXPECTED_RAY_CPUS_MIN": str(plan.workers_per_shard),
                "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(plan.workers_per_shard),
            }
            coord_job = _submit_spcs_synreg(
                session=session,
                label=coord_label,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=coord_env,
                image=image,
                entrypoint_path="/app/scripts/spcs_ray_coordinator.py",
                resource_role=SPCS_RAY_COORDINATOR_RESOURCES,
                endpoints=[{"name": "ray-head", "port": head_port, "protocol": "TCP"}],
            )
            driver_jobs.append((coord_label, coord_job))

            worker_env = {
                "RAY_HEAD_ADDRESS": head_address,
                _SYNREG_WORKER_OBJ_STORE_ENV: worker_obj_store,
                "SPCS_RAY_WORKER_CONNECT_TIMEOUT_SECONDS": str(SYNREG_AUTOGLUON_SPCS_RAY_START_TIMEOUT_SECONDS),
            }
            for worker_index in range(plan.workers_per_shard):
                worker_label = f"spcs_worker_probe_ray_worker_{run_id}_{shard_index}_{worker_index}"
                worker_job = _submit_spcs_synreg(
                    session=session,
                    label=worker_label,
                    compute_pool=AUTOGLUON_CPU_POOL,
                    env_vars=worker_env,
                    image=image,
                    entrypoint_path="/app/scripts/spcs_ray_worker.py",
                    resource_role=SPCS_RAY_WORKER_RESOURCES,
                )
                support_jobs.append((worker_label, worker_job))
        try:
            _wait_spcs_job_group(driver_jobs, session)
        finally:
            _cancel_spcs_job_group(support_jobs, session)
    else:
        jobs = []
        for shard_index in range(shard_count):
            lbl = f"spcs_ag_worker_probe_{run_id}_{shard_index}"
            env = {
                "SYNTHETIC_REGRESSION_SUITE_ID": os.getenv("SYNTHETIC_REGRESSION_SUITE_ID", COMBINED_SUITE_ID),
                "SYNTHETIC_REGRESSION_NUM_SHARDS": str(shard_count),
                "SYNTHETIC_REGRESSION_SHARD_INDEX": str(shard_index),
                "SYNREG_WORKER_DATA_ACCESS_MODE": worker_access_mode,
                "SYNREG_PRESIGNED_URL_EXPIRY_SECONDS": presigned_url_expiry_seconds,
                "SYNREG_WORKER_ACCESS_PROBE_USE_RAY": "false",
                "EXPECTED_RAY_NODES": str(plan.workers_per_shard),
                "EXPECTED_RAY_CPUS_MIN": str(plan.workers_per_shard),
                "SYNREG_AUTOGLUON_WORKERS_PER_SHARD": str(plan.workers_per_shard),
            }
            job_name = _submit_spcs_synreg(
                session=session,
                label=lbl,
                compute_pool=AUTOGLUON_CPU_POOL,
                env_vars=env,
                image=image,
                entrypoint_path="/app/scripts/autogluon_worker_access_probe.py",
                resource_role=SPCS_PROBE_RESOURCES,
            )
            jobs.append((lbl, job_name))
        _wait_spcs_job_group(jobs, session)
    return (
        f"{proc}: ok backend=spcs_job mode={plan.mode} "
        f"concurrent_units={plan.concurrent_units} access_mode={worker_access_mode!r}"
    )


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
