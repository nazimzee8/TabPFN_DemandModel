"""
Orchestrator for the DeepSet training pipeline.

Handler for the run_training_pipeline() Snowpark stored procedure.
The Snowpark session is injected automatically by the stored procedure framework.
"""
import json

from snowflake.ml.jobs import submit_from_stage

GPU_POOL = "DEEPSET_GPU_POOL"
CPU_POOL = "DEEPSET_CPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
KAGGLE_STAGE = "@META_DATASET_STAGE/kaggle/"
TRAIN_NUM_NODES = 10


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


def run_pipeline(session) -> str:
    # ── Phase 1: Pre-train with default hyperparameters ───────────────────────
    print("Submitting pre-training job (Phase 1) ...")
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
        },
        session=session,
    )
    _wait_done(pretrain_job, "Pre-training", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "pretrain.pt"):
        raise RuntimeError(
            "Phase 1 (pre-training) did not produce pretrain.pt in "
            f"{MODEL_STAGE}/checkpoints/. Check container logs before proceeding."
        )

    # ── Phase 2: HPO (each trial warm-starts from pretrain.pt) ───────────────
    print("Submitting HPO job (Phase 2) ...")
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,
        env_vars={"HOME": "/tmp"},
        session=session,
    )
    _wait_done(hpo_job, "HPO", session)

    session.file.get(f"{MODEL_STAGE}/hpo/best_config.json", "/tmp/")
    with open("/tmp/best_config.json") as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    # ── Phase 3: Final training (fine-tunes from pretrain.pt with best_config) ─
    print("Submitting final training job (Phase 3) ...")
    train_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "BEST_CONFIG": json.dumps(best_config),
            "PRETRAIN_CHECKPOINT_PATH": f"{MODEL_STAGE}/checkpoints/pretrain.pt",
            "TRAIN_NUM_NODES": str(TRAIN_NUM_NODES),
            "HOME": "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK": "true",
        },
        session=session,
    )
    _wait_done(train_job, "Final training", session)

    hpo_contents        = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Training pipeline complete (Pretrain → HPO → Final training).\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
        + "\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )


