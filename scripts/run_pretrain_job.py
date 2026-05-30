"""
Stored procedure handler for submitting only the pre-training MLJob (Phase 1).

Trains with default hyperparameters over the full train/val splits and writes
@MODEL_STAGE/checkpoints/pretrain.pt. This checkpoint is consumed by run_hpo_pipeline()
(warm-start) and run_model_training() (fine-tune with best_config).
"""
import json
import os

from snowflake.ml.jobs import submit_from_stage

EXPECTED_INDEX_COUNTS = {"train": 800, "val": 100, "test": 100}

# Nonlinear pretrain defaults: enable latent ridge expert, disable linear ridge expert.
NONLINEAR_PRETRAIN_DEFAULTS = {
    "use_latent_ridge_expert": True,
    "latent_ridge_dim": 64,
    "latent_ridge_lambda": 1.0,
    "latent_ridge_jitter": 1e-3,
    "latent_ridge_use_bias": True,
    "use_ridge_expert": False,
}
NONLINEAR_PRETRAIN_CHECKPOINT = "pretrain_nonlinear_meta.pt"

GPU_POOL            = "DEEPSET_GPU_POOL"
MODEL_STAGE         = "@MODEL_STAGE"
SCRIPTS_STAGE       = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
TRAIN_NUM_NODES             = 10

# Model selector env vars — propagated to the pretrain MLJob so train.py builds
# the correct model family.
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
DEFAULT_TRAINING_DATA_FAMILY  = os.getenv("TRAINING_DATA_FAMILY",  "synthetic_regression_combined")
DEFAULT_MODEL_DESIGN_PATTERN = os.getenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")


def _get_session():
    from snowflake.snowpark import Session
    return Session.builder.getOrCreate()


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

    # Column existence check — catches missing columns before long GPU jobs
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

    # Stage file accessibility spot-check — catches empty/missing staged data
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


def _run_pretrain_impl(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
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
            "MODEL_FAMILY":              model_family,
            "TRAINING_DATA_FAMILY":      training_data_family,
            "MODEL_DESIGN_PATTERN":     model_design_pattern,
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


def _run_pretrain_gate_impl(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    gate_hidden_dim: int,
) -> str:
    """Run pretrain for one gate candidate, writing pretrain_gate<N>.pt.

    gate_hidden_dim must be one of the HPO candidates: 32, 64, or 128.
    This must be called for all three candidates before run_hpo_pipeline().
    """
    _validate_meta_dataset_index(session)
    gate_dim = int(gate_hidden_dim)
    if gate_dim not in (32, 64, 128):
        raise ValueError(
            f"gate_hidden_dim={gate_dim} is not a valid HPO candidate. "
            "Allowed values: 32, 64, 128."
        )
    checkpoint_name = f"pretrain_gate{gate_dim}.pt"
    print(f"Submitting pre-training job for gate_hidden_dim={gate_dim} ({checkpoint_name}) ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "CHECKPOINT_OUTPUT_NAME":    checkpoint_name,
            "GATE_HIDDEN_DIM":           str(gate_dim),
            "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
            "HOME":                      "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK":   "true",
            "MODEL_FAMILY":              model_family,
            "TRAINING_DATA_FAMILY":      training_data_family,
            "MODEL_DESIGN_PATTERN":     model_design_pattern,
        },
        session=session,
    )
    _wait_done(pretrain_job, f"Pre-training gate_hidden_dim={gate_dim}")

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        f"Pre-training pipeline complete (gate_hidden_dim={gate_dim}, "
        f"checkpoint={checkpoint_name}).\n\n"
        "MODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )


def run_pretrain_pipeline() -> str:
    """Zero-arg entrypoint: uses env-var defaults."""
    session = _get_session()
    return _run_pretrain_impl(
        session,
        DEFAULT_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_DESIGN_PATTERN,
    )


def run_pretrain_pipeline_model(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Parameterized entrypoint: explicit selectors for MODEL3 training.

    Usage:
        CALL run_pretrain_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'inductive_forecasting'
        );
    """
    session = _get_session()
    return _run_pretrain_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
    )


def run_pretrain_pipeline_model_gate(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    gate_hidden_dim: int,
) -> str:
    """Gate-specific pretrain: writes pretrain_gate<N>.pt for one HPO candidate.

    Must be called for each gate candidate (32, 64, 128) before run_hpo_pipeline().
    The HPO search space tunes gate_hidden_dim and each trial needs a matching
    pretrain checkpoint; one shared checkpoint cannot serve all gate widths.

    Usage:
        CALL run_pretrain_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'inductive_forecasting',
            64
        );
    """
    session = _get_session()
    return _run_pretrain_gate_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        int(gate_hidden_dim),
    )


def _validate_nonlinear_dataset_index(session):
    """Pre-flight check: raises RuntimeError if META_NONLINEAR_DATASET_INDEX is missing or has wrong counts."""
    try:
        rows = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_NONLINEAR_DATASET_INDEX GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            "META_NONLINEAR_DATASET_INDEX does not exist or cannot be queried. "
            "Run CALL build_meta_nonlinear_dataset_index(); first."
        ) from exc

    counts = {str(row[0]).lower(): int(row[1]) for row in rows}
    mismatches = {
        split: {"expected": expected, "actual": counts.get(split, 0)}
        for split, expected in EXPECTED_INDEX_COUNTS.items()
        if counts.get(split, 0) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"META_NONLINEAR_DATASET_INDEX has wrong split counts: {mismatches}. "
            "Run CALL build_meta_nonlinear_dataset_index(); to rebuild."
        )
    print(
        "META_NONLINEAR_DATASET_INDEX validated: "
        + ", ".join(f"{s}={counts[s]}" for s in ("train", "val", "test"))
    )

    _REQUIRED_COLUMNS = "split, task_id, stage_path, p, n_train, hpo_bucket, prior_regime"
    try:
        session.sql(
            f"SELECT {_REQUIRED_COLUMNS} FROM META_NONLINEAR_DATASET_INDEX LIMIT 1"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"META_NONLINEAR_DATASET_INDEX is missing one or more required columns "
            f"({_REQUIRED_COLUMNS}). "
            "Rebuild with CALL build_meta_nonlinear_dataset_index(); "
            f"Error: {exc}"
        ) from exc

    for _split in ("train", "val"):
        try:
            _files = session.sql(f"LIST @META_NONLINEAR_DATASET_STAGE/{_split}/").collect()
            if not _files:
                raise RuntimeError(
                    f"No staged files found in @META_NONLINEAR_DATASET_STAGE/{_split}/. "
                    "Re-upload nonlinear training data before starting a GPU job."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"META_NONLINEAR_DATASET_INDEX references @META_NONLINEAR_DATASET_STAGE/{_split}/ "
                f"but the stage directory is inaccessible: {exc}. "
                "Verify stage permissions and re-upload data."
            ) from exc


def _run_pretrain_nonlinear_impl(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    _validate_nonlinear_dataset_index(session)
    print("Submitting nonlinear pre-training job ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "CHECKPOINT_OUTPUT_NAME":    NONLINEAR_PRETRAIN_CHECKPOINT,
            "BEST_CONFIG":               json.dumps(NONLINEAR_PRETRAIN_DEFAULTS),
            "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
            "HOME":                      "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK":   "true",
            "MODEL_FAMILY":              model_family,
            "TRAINING_DATA_FAMILY":      training_data_family,
            "MODEL_DESIGN_PATTERN":     model_design_pattern,
        },
        session=session,
    )
    _wait_done(pretrain_job, "Nonlinear pre-training")

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Nonlinear pre-training pipeline complete.\n\n"
        "MODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )


def run_pretrain_pipeline_nonlinear() -> str:
    """Zero-arg nonlinear entrypoint: uses env-var defaults."""
    session = _get_session()
    return _run_pretrain_nonlinear_impl(
        session,
        DEFAULT_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_DESIGN_PATTERN,
    )


def run_pretrain_pipeline_nonlinear_model(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Three-arg nonlinear entrypoint: writes pretrain_nonlinear_meta.pt.

    Usage:
        CALL run_pretrain_pipeline_nonlinear(
            'market_exchangeable_icl',
            'synthetic_regression_nonlinear',
            'inductive_forecasting'
        );
    """
    session = _get_session()
    return _run_pretrain_nonlinear_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
    )
