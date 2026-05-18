"""
Stored procedure handler for submitting only the pre-training MLJob (Phase 1).

Trains with default hyperparameters over the full train/val splits and writes
@MODEL_STAGE/checkpoints/pretrain.pt. This checkpoint is consumed by run_hpo_pipeline()
(warm-start) and run_model_training() (fine-tune with best_config).
"""
import os

from snowflake.ml.jobs import submit_from_stage

EXPECTED_INDEX_COUNTS = {"train": 800, "val": 100, "test": 100}

GPU_POOL            = "DEEPSET_GPU_POOL"
MODEL_STAGE         = "@MODEL_STAGE"
SCRIPTS_STAGE       = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
TRAIN_NUM_NODES             = 10

# Model selector env vars — propagated to the pretrain MLJob so train.py builds
# the correct model family. Default values preserve MODEL2 production behavior.
DEFAULT_DEEPSET_MODEL_FAMILY  = os.getenv("DEEPSET_MODEL_FAMILY",  "market_aware")
DEFAULT_TRAINING_DATA_FAMILY  = os.getenv("TRAINING_DATA_FAMILY",  "synthetic_regression_combined")
DEFAULT_MODEL_ARCH_VERSION    = os.getenv("MODEL_ARCH_VERSION",    "model2")
DEFAULT_MODEL3_DESIGN_PATTERN = os.getenv("MODEL3_DESIGN_PATTERN", "inductive_forecasting")


def _wait_done(job, label):
    job.wait()
    if job.status == "DONE":
        print(f"{label} complete.")
        return
    try:
        logs = job.get_logs()
    except Exception as exc:
        logs = f"(job.get_logs() failed: {exc}.)"
    print(f"{label} container logs:\n", logs)
    raise RuntimeError(f"{label} failed with status {job.status!r}\n--- logs ---\n{logs}")


def _list_stage(session, stage_path):
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _validate_meta_dataset_index(session):
    """Pre-flight check: raises RuntimeError if META_DATASET_INDEX is missing or has wrong counts."""
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
        for split, expected in EXPECTED_INDEX_COUNTS.items()
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


def _run_pretrain_impl(
    session,
    deepset_model_family: str,
    training_data_family: str,
    model_arch_version: str,
    model3_design_pattern: str,
) -> str:
    _validate_meta_dataset_index(session)
    print("Submitting pre-training job ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "CHECKPOINT_OUTPUT_NAME":    "pretrain.pt",
            "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
            "HOME":                      "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK":   "true",
            "DEEPSET_MODEL_FAMILY":      deepset_model_family,
            "TRAINING_DATA_FAMILY":      training_data_family,
            "MODEL_ARCH_VERSION":        model_arch_version,
            "MODEL3_DESIGN_PATTERN":     model3_design_pattern,
        },
        session=session,
    )
    _wait_done(pretrain_job, "Pre-training")

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Pre-training pipeline complete.\n\n"
        "MODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )


def run_pretrain_pipeline(session) -> str:
    """Zero-arg entrypoint: uses env-var defaults (MODEL2 production behavior)."""
    return _run_pretrain_impl(
        session,
        DEFAULT_DEEPSET_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_ARCH_VERSION,
        DEFAULT_MODEL3_DESIGN_PATTERN,
    )


def run_pretrain_pipeline_m3(
    session,
    deepset_model_family: str,
    training_data_family: str,
    model_arch_version: str,
    model3_design_pattern: str,
) -> str:
    """Parameterized entrypoint: explicit selectors for MODEL3 training.

    Usage:
        CALL run_pretrain_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'model3',
            'inductive_forecasting'
        );
    """
    return _run_pretrain_impl(
        session,
        deepset_model_family,
        training_data_family,
        model_arch_version,
        model3_design_pattern,
    )
