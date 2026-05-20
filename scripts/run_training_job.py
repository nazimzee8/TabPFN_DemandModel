"""
Orchestrator for the MODEL3 training pipeline.

Handler for the run_training_pipeline() Snowpark stored procedure.
The Snowpark session is injected automatically by the stored procedure framework.
"""
import json
import os
import tempfile

from snowflake.ml.jobs import submit_from_stage

GPU_POOL = "DEEPSET_GPU_POOL"
CPU_POOL = "DEEPSET_CPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
KAGGLE_STAGE = "@META_DATASET_STAGE/kaggle/"
TRAIN_NUM_NODES = 10
LOCAL_TMP_DIR = tempfile.gettempdir()

# Training data family — identifies the synthetic data suite used for this training run.
# Production synthetic regression evaluation checkpoints use synthetic_regression_combined
# (combined suite linear_all_v1, which includes primary + OOD data).
# Override via TRAINING_DATA_FAMILY env var when launching a different training mode.
DEFAULT_TRAINING_DATA_FAMILY = os.getenv(
    "TRAINING_DATA_FAMILY", "synthetic_regression_combined"
)

# MODEL3 architecture selectors — propagated to all training/HPO MLJob env_vars.
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
DEFAULT_MODEL_DESIGN_PATTERN = os.getenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")


def _wait_done(job, label, session):
    try:
        job.wait()
    except Exception as exc:
        if "300002" in str(exc) or "000603" in str(exc):
            raise RuntimeError(
                f"{label} job terminated with Snowflake internal error 300002 "
                "(service status unavailable — container likely crashed before reaching "
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


def _list_stage(session, stage_path):
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _stage_file_exists(session, stage_path, filename):
    rows = session.sql(f"LIST {stage_path}").collect()
    return any(str(row[0]).rstrip("/").endswith(f"/{filename}") for row in rows)


def _kaggle_secret_spec_overrides(secret_name):
    return {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "secrets": [
                        {
                            "snowflakeSecret": secret_name,
                            "secretKeyRef": "username",
                            "envVarName": "KAGGLE_USERNAME",
                        },
                        {
                            "snowflakeSecret": secret_name,
                            "secretKeyRef": "password",
                            "envVarName": "KAGGLE_KEY",
                        },
                    ],
                }
            ]
        }
    }


def run_kaggle_download(session) -> str:
    print("Submitting Kaggle benchmark download job ...")
    kaggle_secret_name = "KAGGLE_API_SECRET"
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="download_kaggle_to_stage.py",
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["kaggle"],
        external_access_integrations=["BENCHMARK_EXTERNAL_ACCESS"],
        env_vars={
            "KAGGLE_STAGE": KAGGLE_STAGE,
            "KAGGLE_MAX_SAMPLES": "10000",
            "HOME": "/tmp",
        },
        spec_overrides=_kaggle_secret_spec_overrides(kaggle_secret_name),
        session=session,
    )
    _wait_done(job, "Kaggle benchmark download", session)

    kaggle_contents = _list_stage(session, KAGGLE_STAGE)
    return (
        "Kaggle benchmark download complete.\n\n"
        f"{KAGGLE_STAGE}:\n"
        + "\n".join(f"  {p}" for p in kaggle_contents)
    )


def build_meta_dataset_index(session) -> str:
    print("Submitting META_DATASET_INDEX build job ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="build_meta_dataset_index.py",
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["pyarrow"],
        env_vars={"HOME": "/tmp"},
        session=session,
    )
    _wait_done(job, "META_DATASET_INDEX build", session)

    counts = session.sql(
        """
        SELECT split, COUNT(*) AS task_count
        FROM META_DATASET_INDEX
        GROUP BY split
        ORDER BY split
        """
    ).collect()
    subset_counts = session.sql(
        """
        WITH ranked AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY split, hpo_bucket
              ORDER BY prior_regime, p, n_train, task_id
            ) AS bucket_rank
          FROM META_DATASET_INDEX
          WHERE split IN ('train', 'val')
        ),
        selected AS (
          SELECT *
          FROM ranked
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY split
            ORDER BY bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id
          ) <= IFF(split = 'train', 200, 40)
        )
        SELECT split, COUNT(*) AS selected_rows
        FROM selected
        GROUP BY split
        ORDER BY split
        """
    ).collect()
    subset_count_map = {str(row[0]): int(row[1]) for row in subset_counts}
    expected_subset_counts = {"train": 200, "val": 40}
    if subset_count_map != expected_subset_counts:
        raise ValueError(
            "META_DATASET_INDEX HPO subset validation failed: "
            f"expected {expected_subset_counts}, got {subset_count_map}"
        )
    return (
        "META_DATASET_INDEX build complete.\n\n"
        "Full split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        + "\n\nHPO subset counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in subset_counts)
    )


def _validate_meta_dataset_index(session):
    """Pre-flight check: raises RuntimeError if META_DATASET_INDEX is missing or has wrong counts."""
    _EXPECTED_INDEX_COUNTS = {"train": 800, "val": 100, "test": 100}
    try:
        rows = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_DATASET_INDEX GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            "META_DATASET_INDEX does not exist or cannot be queried. "
            "Run CALL build_meta_dataset_index(); first."
        ) from exc

    counts = {str(row[0]).lower(): int(row[1]) for row in rows}
    mismatches = {
        split: {"expected": expected, "actual": counts.get(split, 0)}
        for split, expected in _EXPECTED_INDEX_COUNTS.items()
        if counts.get(split, 0) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"META_DATASET_INDEX has wrong split counts: {mismatches}. "
            "Run CALL build_meta_dataset_index(); to rebuild."
        )
    print(
        "META_DATASET_INDEX validated: "
        + ", ".join(f"{s}={counts[s]}" for s in ("train", "val", "test"))
    )

    # Column existence check
    _REQUIRED_COLUMNS = "split, task_id, stage_path, p, n_train, hpo_bucket, prior_regime"
    try:
        session.sql(
            f"SELECT {_REQUIRED_COLUMNS} FROM META_DATASET_INDEX LIMIT 1"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"META_DATASET_INDEX is missing one or more required columns "
            f"({_REQUIRED_COLUMNS}). "
            "Rebuild with CALL build_meta_dataset_index(); "
            f"Error: {exc}"
        ) from exc

    # Stage file accessibility spot-check
    for _split in ("train", "val"):
        try:
            _files = session.sql(f"LIST @META_DATASET_STAGE/{_split}/").collect()
            if not _files:
                raise RuntimeError(
                    f"No staged files found in @META_DATASET_STAGE/{_split}/. "
                    "Re-upload training data before starting a GPU job."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"META_DATASET_INDEX references @META_DATASET_STAGE/{_split}/ "
                f"but the stage directory is inaccessible: {exc}. "
                "Verify stage permissions and re-upload data."
            ) from exc


def run_pipeline(session) -> str:
    """Full two-sweep training pipeline:

    Step 1: Validate META_DATASET_INDEX (counts + columns + stage access)
    Step 2: Pre-training (pretrain.pt)
    Step 3: HPO sweep 1 — ridge_residual (best_config_ridge_residual.json)
    Step 4: MODEL3 DDP memory probe (worst-case: d_phi=256, n_blocks=2)
    Step 5: HPO sweep 2 — architecture with baseline from sweep 1
            (best_config_architecture.json + merged best_config.json)
    Step 6: Load merged best_config.json
    Step 7: Final training with best_config + pretrain warm-start (best.pt)
    """
    # ── Step 1: Validate META_DATASET_INDEX ──────────────────────────────────
    print("Step 1: Validating META_DATASET_INDEX ...")
    _validate_meta_dataset_index(session)

    # ── Step 2: Pre-train with default hyperparameters ────────────────────────
    print("Step 2: Submitting pre-training job ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "CHECKPOINT_OUTPUT_NAME": "pretrain.pt",
            "TRAIN_NUM_NODES": str(TRAIN_NUM_NODES),
            "HOME": "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK": "true",
            "MODEL_FAMILY":          DEFAULT_MODEL_FAMILY,
            "TRAINING_DATA_FAMILY":  DEFAULT_TRAINING_DATA_FAMILY,
            "MODEL_DESIGN_PATTERN": DEFAULT_MODEL_DESIGN_PATTERN,
        },
        session=session,
    )
    _wait_done(pretrain_job, "Pre-training", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "pretrain.pt"):
        raise RuntimeError(
            "Step 2 (pre-training) did not produce pretrain.pt in "
            f"{MODEL_STAGE}/checkpoints/. Check container logs before proceeding."
        )

    # ── Step 3: HPO sweep 1 — ridge_residual ─────────────────────────────────
    print("Step 3: Submitting HPO ridge_residual sweep ...")
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,
        env_vars={
            "HOME": "/tmp",
            "MODEL_FAMILY":          DEFAULT_MODEL_FAMILY,
            "TRAINING_DATA_FAMILY":  DEFAULT_TRAINING_DATA_FAMILY,
            "MODEL_DESIGN_PATTERN": DEFAULT_MODEL_DESIGN_PATTERN,
            "HPO_SWEEP_MODE":        "ridge_residual",
        },
        session=session,
    )
    _wait_done(hpo_job, "HPO ridge_residual", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/hpo/", "best_config_ridge_residual.json"):
        raise RuntimeError(
            "Step 3 (HPO ridge_residual) did not produce best_config_ridge_residual.json in "
            f"{MODEL_STAGE}/hpo/. Check container logs before proceeding."
        )

    # ── Step 4: MODEL3 DDP memory probe (worst-case architecture candidates) ──
    print("Step 4: Submitting MODEL3 DDP memory probe (pre-architecture HPO gate) ...")
    probe_env = {
        "HOME": "/tmp",
        "TRAIN_NUM_NODES": str(TRAIN_NUM_NODES),
        "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
        "STRICT_WORLD_SIZE_CHECK": "true",
        "MODEL_DESIGN_PATTERN": "inductive_forecasting",
        "MODEL_FAMILY": "market_exchangeable_icl",
        "MODEL_PROBE_N_CONTEXT": "200",
        "MODEL_PROBE_P_FEATURES": "128",
        "MODEL_PROBE_M_QUERY": "128",
        "MODEL_PROBE_D_PHI": "256",    # max ARCH_D_PHI_CANDIDATES
        "MODEL_PROBE_N_BLOCKS": "2",   # max ARCH_N_SAB_FEAT_CANDIDATES
        "MODEL_PROBE_RUN_BACKWARD": "true",
        "MODEL_PROBE_DTYPE": "float32",
        "MODEL_PROBE_MAX_GPU_MEMORY_FRACTION": "0.9",
        "MODEL_PROBE_STRICT_MEMORY_GUARD": "true",
        "MODEL_PROBE_MEMORY_SAFETY_FACTOR": "1.5",
        "MODEL_PROBE_OUTPUT_STAGE": f"{MODEL_STAGE}/diagnostics/",
    }
    probe_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="model_ddp_memory_probe.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars=probe_env,
        session=session,
    )
    _wait_done(probe_job, "MODEL3DDPMemoryProbe (pre-architecture HPO gate)", session)

    # ── Step 5: HPO sweep 2 — architecture with baseline from sweep 1 ─────────
    print("Step 5: Submitting HPO architecture sweep ...")
    arch_hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,
        env_vars={
            "HOME": "/tmp",
            "MODEL_FAMILY":          DEFAULT_MODEL_FAMILY,
            "TRAINING_DATA_FAMILY":  DEFAULT_TRAINING_DATA_FAMILY,
            "MODEL_DESIGN_PATTERN": DEFAULT_MODEL_DESIGN_PATTERN,
            "HPO_SWEEP_MODE":        "architecture",
            "HPO_BASELINE_CONFIG_STAGE_PATH": f"{MODEL_STAGE}/hpo/best_config_ridge_residual.json",
        },
        session=session,
    )
    _wait_done(arch_hpo_job, "HPO architecture", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/hpo/", "best_config.json"):
        raise RuntimeError(
            "Step 5 (HPO architecture) did not produce best_config.json in "
            f"{MODEL_STAGE}/hpo/. Check container logs before proceeding."
        )

    # ── Step 6: Load merged best_config.json ─────────────────────────────────
    print("Step 6: Loading merged best_config.json ...")
    session.file.get(f"{MODEL_STAGE}/hpo/best_config.json", LOCAL_TMP_DIR)
    with open(os.path.join(LOCAL_TMP_DIR, "best_config.json")) as f:
        best_config = json.load(f)
    print("Merged best config:", best_config)

    # ── Step 7: Final training with best_config + pretrain warm-start ─────────
    print("Step 7: Submitting final training job ...")
    train_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "BEST_CONFIG": json.dumps(best_config),
            "PRETRAIN_CHECKPOINT_PATH": f"{MODEL_STAGE}/checkpoints/pretrain.pt",
            "PRETRAIN_LOAD_POLICY": "allow_cold_start_on_arch_mismatch",
            "TRAIN_NUM_NODES": str(TRAIN_NUM_NODES),
            "HOME": "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK": "true",
            "MODEL_FAMILY":          DEFAULT_MODEL_FAMILY,
            "TRAINING_DATA_FAMILY":  DEFAULT_TRAINING_DATA_FAMILY,
            "MODEL_DESIGN_PATTERN": DEFAULT_MODEL_DESIGN_PATTERN,
        },
        session=session,
    )
    _wait_done(train_job, "Final training", session)

    hpo_contents        = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Training pipeline complete "
        "(Validate → Pretrain → HPO ridge_residual → Memory probe → "
        "HPO architecture → Merge config → Final training).\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
        + "\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )

