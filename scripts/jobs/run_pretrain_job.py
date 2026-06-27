"""
Stored procedure handler for submitting only the pre-training MLJob (Phase 1).

Trains with default hyperparameters over the full train/val splits and writes
@MODEL_STAGE/checkpoints/pretrain.pt. This checkpoint is consumed by run_hpo_pipeline()
(warm-start) and run_model_training() (fine-tune with best_config).
"""
import json
import os

from snowflake.ml.jobs import submit_from_stage

def _derive_split_counts(total: int) -> dict:
    train = int(0.8 * total)
    val = int(0.1 * total)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


_LINEAR_EXPECTED_TOTAL = int(os.getenv("META_DATASET_EXPECTED_TOTAL", "1000"))
# Canonical name matches task_routing.py expected_total_env = "META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL"
# (used by _validate_nonlinear_index_by_spec). Legacy alias "META_NONLINEAR_DATASET_EXPECTED_TOTAL"
# is checked as fallback for backward compatibility.
_NONLINEAR_EXPECTED_TOTAL = int(os.getenv(
    "META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL",
    os.getenv("META_NONLINEAR_DATASET_EXPECTED_TOTAL", "1000"),
))
_CLASSIFICATION_EXPECTED_TOTAL = int(
    os.getenv("META_CLASSIFICATION_DATASET_EXPECTED_TOTAL", "1000")
)
EXPECTED_INDEX_COUNTS = _derive_split_counts(_LINEAR_EXPECTED_TOTAL)
_NONLINEAR_EXPECTED_INDEX_COUNTS = _derive_split_counts(_NONLINEAR_EXPECTED_TOTAL)
_CLASSIFICATION_EXPECTED_INDEX_COUNTS = _derive_split_counts(
    _CLASSIFICATION_EXPECTED_TOTAL
)

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
NONLINEAR_TRAINING_DATA_FAMILY = "synthetic_regression_nonlinear"
CLASSIFICATION_TRAINING_DATA_FAMILY = "synthetic_linear_classification"

# All nonlinear families accepted by the nonlinear pretrain handler.
# Includes the historical alias ("synthetic_regression_nonlinear") and the three
# additional nonlinear families (classification, mixed-regression, mixed-classification).
_NONLINEAR_TRAINING_DATA_FAMILIES = frozenset({
    "synthetic_regression_nonlinear",                       # alias (legacy)
    "synthetic_nonlinear_regression",                       # canonical
    "synthetic_nonlinear_classification",
    "synthetic_nonlinear_regression_mixed_categorical",
    "synthetic_nonlinear_classification_mixed_categorical",
})


def _is_nonlinear_family(training_data_family: str) -> bool:
    """Return True if training_data_family belongs to a nonlinear task suite."""
    return (
        training_data_family in _NONLINEAR_TRAINING_DATA_FAMILIES
        or "nonlinear" in training_data_family
    )

GPU_POOL            = "DEEPSET_GPU_POOL"
MODEL_STAGE         = "@MODEL_STAGE"
SCRIPTS_STAGE       = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
TRAIN_NUM_NODES             = 10

# Model selector env vars — propagated to the pretrain MLJob so train.py builds
# the correct model family.
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
DEFAULT_TRAINING_DATA_FAMILY = os.getenv(
    "PRETRAIN_TRAINING_DATA_FAMILY",
    os.getenv("TRAINING_DATA_FAMILY", "synthetic_linear_regression"),
)

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
    """Pre-flight check: raises RuntimeError if META_REGRESSION_DATASET_INDEX is missing or has wrong counts."""
    try:
        rows = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_REGRESSION_DATASET_INDEX GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            "META_REGRESSION_DATASET_INDEX does not exist or cannot be queried. "
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
            f"META_REGRESSION_DATASET_INDEX has wrong split counts: {mismatches}. "
            "Run CALL build_meta_dataset_index(); to rebuild."
        )
    print(
        "META_REGRESSION_DATASET_INDEX validated: "
        + ", ".join(f"{s}={counts[s]}" for s in ("train", "val", "test"))
    )

    # Column existence check — catches missing columns before long GPU jobs
    _REQUIRED_COLUMNS = "split, task_id, stage_path, p, n_train, hpo_bucket, prior_regime"
    try:
        session.sql(
            f"SELECT {_REQUIRED_COLUMNS} FROM META_REGRESSION_DATASET_INDEX LIMIT 1"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"META_REGRESSION_DATASET_INDEX is missing one or more required columns "
            f"({_REQUIRED_COLUMNS}). "
            "Rebuild with CALL build_meta_dataset_index(); "
            f"Error: {exc}"
        ) from exc

    # Stage file accessibility spot-check — catches empty/missing staged data
    for _split in ("train", "val"):
        try:
            _files = session.sql(f"LIST @META_DATASET_STAGE/linear/regression/numeric/{_split}/").collect()
            if not _files:
                raise RuntimeError(
                    f"No staged files found in @META_DATASET_STAGE/linear/regression/numeric/{_split}/. "
                    "Re-upload training data before starting a GPU job."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"META_REGRESSION_DATASET_INDEX references @META_DATASET_STAGE/linear/regression/numeric/{_split}/ "
                f"but the stage directory is inaccessible: {exc}. "
                "Verify stage permissions and re-upload data."
            ) from exc


def _validate_generic_nonlinear_dataset_index(
    session, training_data_family: str
) -> None:
    """Existence check for nonlinear families that lack a dedicated validator.

    For the regression-nonlinear family, the stricter _validate_nonlinear_dataset_index
    is used. For nonlinear classification and mixed-categorical families the table name
    is resolved via task_routing (available on the SPCS script path) and a quick
    existence + row-count check is run.
    """
    from task_routing import get_training_data_spec
    spec = get_training_data_spec(training_data_family)
    table = spec.index_table
    try:
        rows = session.sql(
            f"SELECT split, COUNT(*) AS task_count FROM {table} GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"{table} does not exist or cannot be queried. "
            f"Build the index for TRAINING_DATA_FAMILY={training_data_family!r} first."
        ) from exc
    counts = {str(row[0]).lower(): int(row[1]) for row in rows}
    print(
        f"[pretrain] {table} validated: "
        + ", ".join(f"{s}={counts.get(s, 0)}" for s in ("train", "val", "test"))
    )


def _validate_training_dataset_index(session, training_data_family: str) -> None:
    """Route index validation to the table selected by TRAINING_DATA_FAMILY."""
    if training_data_family == NONLINEAR_TRAINING_DATA_FAMILY:
        _validate_nonlinear_dataset_index(session)
    elif _is_nonlinear_family(training_data_family):
        _validate_generic_nonlinear_dataset_index(session, training_data_family)
    elif training_data_family == CLASSIFICATION_TRAINING_DATA_FAMILY:
        _validate_classification_dataset_index(session)
    else:
        _validate_meta_dataset_index(session)


def _validate_classification_dataset_index(session) -> None:
    table = "META_CLASSIFICATION_DATASET_INDEX"
    stage = "@META_DATASET_STAGE"
    try:
        rows = session.sql(
            f"SELECT split, COUNT(*) AS task_count FROM {table} GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"{table} does not exist or cannot be queried. "
            "Build the classification index before starting training."
        ) from exc
    counts = {str(row[0]).lower(): int(row[1]) for row in rows}
    mismatches = {
        split: {"expected": expected, "actual": counts.get(split, 0)}
        for split, expected in _CLASSIFICATION_EXPECTED_INDEX_COUNTS.items()
        if counts.get(split, 0) != expected
    }
    if mismatches:
        raise RuntimeError(f"{table} has wrong split counts: {mismatches}.")
    required = (
        "split, task_id, stage_path, p, n_train, hpo_bucket, "
        "prior_regime, num_classes"
    )
    try:
        session.sql(f"SELECT {required} FROM {table} LIMIT 1").collect()
    except Exception as exc:
        raise RuntimeError(
            f"{table} is missing required columns ({required}): {exc}"
        ) from exc
    for split in ("train", "val"):
        files = session.sql(f"LIST {stage}/linear/classification/numeric/{split}/").collect()
        if not files:
            raise RuntimeError(f"No staged files found in {stage}/linear/classification/numeric/{split}/.")


def _run_pretrain_impl(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    _validate_training_dataset_index(session, training_data_family)
    # Family-qualified checkpoint name prevents linear-numeric and linear-mixed from
    # overwriting each other at pretrain.pt (L5 fix). Uses the same data_subdir convention
    # as run_model_training_job.py M4.
    from task_routing import get_training_data_spec as _get_spec
    _spec = _get_spec(training_data_family)
    if training_data_family == CLASSIFICATION_TRAINING_DATA_FAMILY:
        checkpoint_name = "pretrain_classification.pt"
    elif _spec.is_nonlinear:
        # Nonlinear families use their own dedicated pretrain entrypoints; guard against misuse.
        raise ValueError(
            f"_run_pretrain_impl does not support nonlinear families "
            f"(training_data_family={training_data_family!r}). "
            "Use run_pretrain_pipeline_nonlinear_model() instead."
        )
    else:
        # Family-qualified: e.g. pretrain_linear_regression_numeric.pt
        checkpoint_name = f"pretrain_{_spec.data_subdir.replace('/', '_')}.pt"
    print("Submitting pre-training job ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "CHECKPOINT_OUTPUT_NAME":                checkpoint_name,
            "TRAIN_NUM_NODES":                       str(TRAIN_NUM_NODES),
            "HOME":                                  "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE":             str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK":               "true",
            "MODEL_FAMILY":                          model_family,
            "TRAINING_DATA_FAMILY":                  training_data_family,
            "MODEL_DESIGN_PATTERN":                  model_design_pattern,
            "META_DATASET_EXPECTED_TOTAL":           str(_LINEAR_EXPECTED_TOTAL),
            "META_NONLINEAR_DATASET_EXPECTED_TOTAL": str(_NONLINEAR_EXPECTED_TOTAL),
            "META_CLASSIFICATION_DATASET_EXPECTED_TOTAL": str(
                _CLASSIFICATION_EXPECTED_TOTAL
            ),
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
    _validate_training_dataset_index(session, training_data_family)
    gate_dim = int(gate_hidden_dim)
    # HPO search space is pinned to GATE_HIDDEN_DIM_CANDIDATES=[64]; only gate=64 is meaningful.
    # train.py reads fusion_gate_hidden_dim from BEST_CONFIG, not from the GATE_HIDDEN_DIM env var,
    # so the multi-gate loop previously produced three identical checkpoints. Collapse to gate=64.
    if gate_dim != 64:
        raise ValueError(
            f"gate_hidden_dim={gate_dim} is not supported. Only gate_dim=64 is used "
            "(HPO pinned GATE_HIDDEN_DIM_CANDIDATES=[64]; train.py reads gate dim from BEST_CONFIG)."
        )
    checkpoint_name = (
        "pretrain_classification_gate64.pt"
        if training_data_family == CLASSIFICATION_TRAINING_DATA_FAMILY
        else "pretrain_gate64.pt"
    )
    print(f"Submitting pre-training job for gate_hidden_dim=64 ({checkpoint_name}) ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "CHECKPOINT_OUTPUT_NAME":    checkpoint_name,
            # GATE_HIDDEN_DIM removed: train.py reads fusion_gate_hidden_dim from BEST_CONFIG only.
            "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
            "HOME":                      "/tmp",
            "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK":   "true",
            "MODEL_FAMILY":              model_family,
            "TRAINING_DATA_FAMILY":      training_data_family,
            "MODEL_DESIGN_PATTERN":     model_design_pattern,
            "META_CLASSIFICATION_DATASET_EXPECTED_TOTAL": str(
                _CLASSIFICATION_EXPECTED_TOTAL
            ),
        },
        session=session,
    )
    _wait_done(pretrain_job, "Pre-training gate_hidden_dim=64")

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        f"Pre-training pipeline complete (gate_hidden_dim=64, "
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
            'synthetic_linear_regression',
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
    """Gate-specific pretrain: writes pretrain_gate64.pt.

    Only gate_hidden_dim=64 is accepted. HPO search space is pinned to
    GATE_HIDDEN_DIM_CANDIDATES=[64] and train.py reads fusion_gate_hidden_dim
    from BEST_CONFIG (not from GATE_HIDDEN_DIM env var), so only one checkpoint
    is needed.

    Usage:
        CALL run_pretrain_pipeline(
            'market_exchangeable_icl',
            'synthetic_linear_regression',
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
    """Pre-flight check: raises RuntimeError if META_NONLINEAR_REGRESSION_DATASET_INDEX is missing or has wrong counts."""
    try:
        rows = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_NONLINEAR_REGRESSION_DATASET_INDEX GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            "META_NONLINEAR_REGRESSION_DATASET_INDEX does not exist or cannot be queried. "
            "Run CALL build_meta_nonlinear_dataset_index(); first."
        ) from exc

    counts = {str(row[0]).lower(): int(row[1]) for row in rows}
    mismatches = {
        split: {"expected": expected, "actual": counts.get(split, 0)}
        for split, expected in _NONLINEAR_EXPECTED_INDEX_COUNTS.items()
        if counts.get(split, 0) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"META_NONLINEAR_REGRESSION_DATASET_INDEX has wrong split counts: {mismatches}. "
            "Run CALL build_meta_nonlinear_dataset_index(); to rebuild."
        )
    print(
        "META_NONLINEAR_REGRESSION_DATASET_INDEX validated: "
        + ", ".join(f"{s}={counts[s]}" for s in ("train", "val", "test"))
    )

    _REQUIRED_COLUMNS = "split, task_id, stage_path, p, n_train, hpo_bucket, prior_regime"
    try:
        session.sql(
            f"SELECT {_REQUIRED_COLUMNS} FROM META_NONLINEAR_REGRESSION_DATASET_INDEX LIMIT 1"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"META_NONLINEAR_REGRESSION_DATASET_INDEX is missing one or more required columns "
            f"({_REQUIRED_COLUMNS}). "
            "Rebuild with CALL build_meta_nonlinear_dataset_index(); "
            f"Error: {exc}"
        ) from exc

    for _split in ("train", "val"):
        try:
            _files = session.sql(f"LIST @META_DATASET_STAGE/nonlinear/regression/numeric/{_split}/").collect()
            if not _files:
                raise RuntimeError(
                    f"No staged files found in @META_DATASET_STAGE/nonlinear/regression/numeric/{_split}/. "
                    "Re-upload nonlinear training data before starting a GPU job."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"META_NONLINEAR_REGRESSION_DATASET_INDEX references @META_DATASET_STAGE/nonlinear/regression/numeric/{_split}/ "
                f"but the stage directory is inaccessible: {exc}. "
                "Verify stage permissions and re-upload data."
            ) from exc


def _run_pretrain_nonlinear_impl(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    # Route to the appropriate validator for this nonlinear family.
    _validate_training_dataset_index(session, training_data_family)
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
        NONLINEAR_TRAINING_DATA_FAMILY,   # hardcoded — not env-var chain
        DEFAULT_MODEL_DESIGN_PATTERN,
    )


def run_pretrain_pipeline_nonlinear_model(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Three-arg nonlinear entrypoint: writes pretrain_nonlinear_meta.pt.

    Accepts all four nonlinear training-data families:
      - synthetic_regression_nonlinear (alias) / synthetic_nonlinear_regression (canonical)
      - synthetic_nonlinear_classification
      - synthetic_nonlinear_regression_mixed_categorical
      - synthetic_nonlinear_classification_mixed_categorical

    Usage:
        CALL run_pretrain_pipeline_nonlinear(
            'market_exchangeable_icl',
            'synthetic_regression_nonlinear',
            'inductive_forecasting'
        );
    """
    if not _is_nonlinear_family(training_data_family):
        raise ValueError(
            f"run_pretrain_pipeline_nonlinear_model only supports nonlinear "
            f"training families. Got {training_data_family!r}. "
            f"Accepted families: {sorted(_NONLINEAR_TRAINING_DATA_FAMILIES)}. "
            "Use run_pretrain_pipeline() for linear families."
        )
    session = _get_session()
    return _run_pretrain_nonlinear_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
    )
