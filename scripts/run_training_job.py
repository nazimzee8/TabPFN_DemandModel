"""
Orchestrator for the MODEL3 training pipeline.

Handler for the run_training_pipeline() Snowpark stored procedure.
Each public handler creates its own Snowpark session via Session.builder.getOrCreate().
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
KAGGLE_STAGE = "@META_REGRESSION_DATASET_STAGE/kaggle/"
TRAIN_NUM_NODES = 10
LOCAL_TMP_DIR = tempfile.gettempdir()

# Training data family — identifies the synthetic data suite used for this training run.
# Production synthetic regression evaluation checkpoints use synthetic_linear_regression
# (combined suite linear_all_v1, which includes primary + OOD data).
# Override via TRAINING_DATA_FAMILY env var when launching a different training mode.
DEFAULT_TRAINING_DATA_FAMILY = os.getenv(
    "TRAINING_DATA_FAMILY", "synthetic_linear_regression"
)

# MODEL3 architecture selectors — propagated to all training/HPO MLJob env_vars.
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
DEFAULT_MODEL_DESIGN_PATTERN = os.getenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")

def _derive_split_counts(total: int) -> dict:
    train = int(0.8 * total)
    val = int(0.1 * total)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


_LINEAR_EXPECTED_TOTAL = int(os.getenv("META_REGRESSION_DATASET_EXPECTED_TOTAL", "1000"))
_NONLINEAR_EXPECTED_TOTAL = int(os.getenv("META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL", "1000"))
# Module-level constant shared by both linear and nonlinear index validators.
EXPECTED_INDEX_COUNTS = _derive_split_counts(_LINEAR_EXPECTED_TOTAL)
_NONLINEAR_EXPECTED_INDEX_COUNTS = _derive_split_counts(_NONLINEAR_EXPECTED_TOTAL)
_NONLINEAR_TRAINING_FAMILY = "synthetic_regression_nonlinear"


def _get_session():
    from snowflake.snowpark import Session
    return Session.builder.getOrCreate()


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


def run_kaggle_download() -> str:
    session = _get_session()
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


def _build_meta_regression_index_impl(
    session, is_mixed_categorical: bool, expected_total: int
) -> None:
    """Submit the appropriate regression-index build MLJob and wait for completion.

    Routes on *is_mixed_categorical*:
      False → build_meta_dataset_index.py      → META_REGRESSION_DATASET_INDEX
      True  → build_meta_mixed_regression_dataset_index.py → META_MIXED_REGRESSION_DATASET_INDEX
    """
    if is_mixed_categorical:
        entrypoint = "build_meta_mixed_regression_dataset_index.py"
        env_key    = "META_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL"
        label      = "META_MIXED_REGRESSION_DATASET_INDEX build"
    else:
        entrypoint = "build_meta_dataset_index.py"
        env_key    = "META_REGRESSION_DATASET_EXPECTED_TOTAL"
        label      = "META_REGRESSION_DATASET_INDEX build"
    print(f"Submitting {label} job (expected_total={expected_total}) ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint=entrypoint,
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["pyarrow"],
        env_vars={"HOME": "/tmp", env_key: str(expected_total)},
        session=session,
    )
    _wait_done(job, label, session)


def _validate_regression_index(session, is_mixed_categorical: bool) -> str:
    """Validate and return a summary string for the regression training index."""
    if is_mixed_categorical:
        counts = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_MIXED_REGRESSION_DATASET_INDEX GROUP BY split ORDER BY split"
        ).collect()
        return (
            "META_MIXED_REGRESSION_DATASET_INDEX build complete.\n\nFull split counts:\n"
            + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        )
    # Numeric regression — also validate HPO subset counts
    counts = session.sql(
        "SELECT split, COUNT(*) AS task_count "
        "FROM META_REGRESSION_DATASET_INDEX GROUP BY split ORDER BY split"
    ).collect()
    subset_counts = session.sql(
        """
        WITH ranked AS (
          SELECT *,
            ROW_NUMBER() OVER (
              PARTITION BY split, hpo_bucket
              ORDER BY prior_regime, p, n_train, task_id
            ) AS bucket_rank
          FROM META_REGRESSION_DATASET_INDEX
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
        SELECT split, COUNT(*) AS selected_rows FROM selected GROUP BY split ORDER BY split
        """
    ).collect()
    subset_count_map = {str(row[0]): int(row[1]) for row in subset_counts}
    expected_subset_counts = {"train": 200, "val": 40}
    if subset_count_map != expected_subset_counts:
        raise ValueError(
            "META_REGRESSION_DATASET_INDEX HPO subset validation failed: "
            f"expected {expected_subset_counts}, got {subset_count_map}"
        )
    return (
        "META_REGRESSION_DATASET_INDEX build complete.\n\n"
        "Full split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        + "\n\nHPO subset counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in subset_counts)
    )


def build_meta_dataset_index_with_flag(is_mixed_categorical: bool) -> str:
    """Boolean-routed handler — rebuild regression training index.

    is_mixed_categorical=FALSE → META_REGRESSION_DATASET_INDEX (numeric)
    is_mixed_categorical=TRUE  → META_MIXED_REGRESSION_DATASET_INDEX (mixed-categorical)

    Expected total is read from the corresponding env var default (1000).

    Usage:
        CALL build_meta_dataset_index(FALSE);
        CALL build_meta_dataset_index(TRUE);
    """
    session = _get_session()
    env_key = (
        "META_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL"
        if is_mixed_categorical
        else "META_REGRESSION_DATASET_EXPECTED_TOTAL"
    )
    expected_total = int(os.getenv(env_key, str(_LINEAR_EXPECTED_TOTAL)))
    _build_meta_regression_index_impl(session, is_mixed_categorical, expected_total)
    return _validate_regression_index(session, is_mixed_categorical)


def build_meta_dataset_index_with_flag_and_total(
    is_mixed_categorical: bool, expected_total: int
) -> str:
    """Boolean-routed handler — rebuild regression training index with explicit total.

    Usage:
        CALL build_meta_dataset_index(FALSE, 1000);
        CALL build_meta_dataset_index(TRUE, 500);
    """
    session = _get_session()
    _build_meta_regression_index_impl(session, is_mixed_categorical, int(expected_total))
    return _validate_regression_index(session, is_mixed_categorical)


def _build_meta_classification_index_impl(
    session, is_mixed_categorical: bool, expected_total: int
) -> None:
    """Submit the appropriate classification-index build MLJob and wait for completion."""
    if is_mixed_categorical:
        entrypoint = "build_meta_mixed_classification_dataset_index.py"
        env_key    = "META_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL"
        label      = "META_MIXED_CATEGORICAL_DATASET_INDEX build"
    else:
        entrypoint = "build_meta_classification_dataset_index.py"
        env_key    = "META_CLASSIFICATION_DATASET_EXPECTED_TOTAL"
        label      = "META_CLASSIFICATION_DATASET_INDEX build"
    print(f"Submitting {label} job (expected_total={expected_total}) ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint=entrypoint,
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["pyarrow"],
        env_vars={"HOME": "/tmp", env_key: str(expected_total)},
        session=session,
    )
    _wait_done(job, label, session)


def _validate_classification_index(session, is_mixed_categorical: bool) -> str:
    """Validate and return a summary string for the classification training index."""
    if is_mixed_categorical:
        counts = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_MIXED_CATEGORICAL_DATASET_INDEX GROUP BY split ORDER BY split"
        ).collect()
        return (
            "META_MIXED_CATEGORICAL_DATASET_INDEX build complete.\n\nFull split counts:\n"
            + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        )
    counts = session.sql(
        "SELECT split, COUNT(*) AS task_count "
        "FROM META_CLASSIFICATION_DATASET_INDEX GROUP BY split ORDER BY split"
    ).collect()
    return (
        "META_CLASSIFICATION_DATASET_INDEX build complete.\n\nFull split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
    )


def build_meta_classification_dataset_index_with_flag(is_mixed_categorical: bool) -> str:
    """Boolean-routed handler — rebuild classification training index.

    is_mixed_categorical=FALSE → META_CLASSIFICATION_DATASET_INDEX (numeric)
    is_mixed_categorical=TRUE  → META_MIXED_CATEGORICAL_DATASET_INDEX (mixed-categorical)

    Usage:
        CALL build_meta_classification_dataset_index(FALSE);
        CALL build_meta_classification_dataset_index(TRUE);
    """
    session = _get_session()
    env_key = (
        "META_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL"
        if is_mixed_categorical
        else "META_CLASSIFICATION_DATASET_EXPECTED_TOTAL"
    )
    expected_total = int(os.getenv(env_key, "1000"))
    _build_meta_classification_index_impl(session, is_mixed_categorical, expected_total)
    return _validate_classification_index(session, is_mixed_categorical)


def build_meta_classification_dataset_index_with_flag_and_total(
    is_mixed_categorical: bool, expected_total: int
) -> str:
    """Boolean-routed handler — rebuild classification training index with explicit total.

    Usage:
        CALL build_meta_classification_dataset_index(FALSE, 1000);
        CALL build_meta_classification_dataset_index(TRUE, 500);
    """
    session = _get_session()
    _build_meta_classification_index_impl(session, is_mixed_categorical, int(expected_total))
    return _validate_classification_index(session, is_mixed_categorical)


def build_meta_nonlinear_classification_dataset_index() -> str:
    """Rebuild META_NONLINEAR_CLASSIFICATION_DATASET_INDEX using env-var defaults."""
    session = _get_session()
    expected_total = os.getenv("META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL", "1000")
    print("Submitting META_NONLINEAR_CLASSIFICATION_DATASET_INDEX build job ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="build_meta_nonlinear_classification_dataset_index.py",
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["pyarrow"],
        env_vars={
            "HOME": "/tmp",
            "META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL": expected_total,
        },
        session=session,
    )
    _wait_done(job, "META_NONLINEAR_CLASSIFICATION_DATASET_INDEX build", session)
    counts = session.sql(
        "SELECT split, COUNT(*) AS task_count "
        "FROM META_NONLINEAR_CLASSIFICATION_DATASET_INDEX GROUP BY split ORDER BY split"
    ).collect()
    return (
        "META_NONLINEAR_CLASSIFICATION_DATASET_INDEX build complete.\n\nFull split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
    )


def build_meta_nonlinear_classification_dataset_index_with_total(expected_total: int) -> str:
    """Set the expected staged nonlinear-classification count and rebuild its metadata index."""
    os.environ["META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL"] = str(int(expected_total))
    return build_meta_nonlinear_classification_dataset_index()


def _build_meta_nonlinear_regression_index_impl(
    session, is_mixed_categorical: bool, expected_total: int
) -> None:
    """Submit the appropriate nonlinear-regression-index build MLJob and wait."""
    if is_mixed_categorical:
        entrypoint = "build_meta_nonlinear_mixed_regression_dataset_index.py"
        env_key    = "META_NONLINEAR_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL"
        label      = "META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX build"
    else:
        entrypoint = "build_meta_nonlinear_dataset_index.py"
        env_key    = "META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL"
        label      = "META_NONLINEAR_REGRESSION_DATASET_INDEX build"
    print(f"Submitting {label} job (expected_total={expected_total}) ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint=entrypoint,
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["pyarrow"],
        env_vars={"HOME": "/tmp", env_key: str(expected_total)},
        session=session,
    )
    _wait_done(job, label, session)


def build_meta_nonlinear_dataset_index_with_flag(is_mixed_categorical: bool) -> str:
    """Boolean-routed handler — rebuild nonlinear-regression training index.

    is_mixed_categorical=FALSE → META_NONLINEAR_REGRESSION_DATASET_INDEX (numeric)
    is_mixed_categorical=TRUE  → META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX (mixed-cat)

    Usage:
        CALL build_meta_nonlinear_dataset_index(FALSE);
        CALL build_meta_nonlinear_dataset_index(TRUE);
    """
    session = _get_session()
    env_key = (
        "META_NONLINEAR_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL"
        if is_mixed_categorical
        else "META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL"
    )
    expected_total = int(os.getenv(env_key, str(_NONLINEAR_EXPECTED_TOTAL)))
    _build_meta_nonlinear_regression_index_impl(session, is_mixed_categorical, expected_total)
    if is_mixed_categorical:
        counts = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX GROUP BY split ORDER BY split"
        ).collect()
        return (
            "META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX build complete.\n\nFull split counts:\n"
            + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        )
    counts = session.sql(
        "SELECT split, COUNT(*) AS task_count "
        "FROM META_NONLINEAR_REGRESSION_DATASET_INDEX GROUP BY split ORDER BY split"
    ).collect()
    return (
        "META_NONLINEAR_REGRESSION_DATASET_INDEX build complete.\n\nFull split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
    )


def build_meta_nonlinear_dataset_index_with_flag_and_total(
    is_mixed_categorical: bool, expected_total: int
) -> str:
    """Boolean-routed handler — rebuild nonlinear-regression training index with explicit total.

    Usage:
        CALL build_meta_nonlinear_dataset_index(FALSE, 1000);
        CALL build_meta_nonlinear_dataset_index(TRUE, 500);
    """
    session = _get_session()
    _build_meta_nonlinear_regression_index_impl(session, is_mixed_categorical, int(expected_total))
    if is_mixed_categorical:
        counts = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX GROUP BY split ORDER BY split"
        ).collect()
        return (
            "META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX build complete.\n\nFull split counts:\n"
            + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        )
    counts = session.sql(
        "SELECT split, COUNT(*) AS task_count "
        "FROM META_NONLINEAR_REGRESSION_DATASET_INDEX GROUP BY split ORDER BY split"
    ).collect()
    return (
        "META_NONLINEAR_REGRESSION_DATASET_INDEX build complete.\n\nFull split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
    )


def _build_meta_nonlinear_classification_index_impl(
    session, is_mixed_categorical: bool, expected_total: int
) -> None:
    """Submit the appropriate nonlinear-classification-index build MLJob and wait."""
    if is_mixed_categorical:
        entrypoint = "build_meta_nonlinear_mixed_classification_dataset_index.py"
        env_key    = "META_NONLINEAR_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL"
        label      = "META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX build"
    else:
        entrypoint = "build_meta_nonlinear_classification_dataset_index.py"
        env_key    = "META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL"
        label      = "META_NONLINEAR_CLASSIFICATION_DATASET_INDEX build"
    print(f"Submitting {label} job (expected_total={expected_total}) ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint=entrypoint,
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["pyarrow"],
        env_vars={"HOME": "/tmp", env_key: str(expected_total)},
        session=session,
    )
    _wait_done(job, label, session)


def build_meta_nonlinear_classification_dataset_index_with_flag(
    is_mixed_categorical: bool,
) -> str:
    """Boolean-routed handler — rebuild nonlinear-classification training index.

    is_mixed_categorical=FALSE → META_NONLINEAR_CLASSIFICATION_DATASET_INDEX (numeric)
    is_mixed_categorical=TRUE  → META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX (mixed-cat)

    Usage:
        CALL build_meta_nonlinear_classification_dataset_index(FALSE);
        CALL build_meta_nonlinear_classification_dataset_index(TRUE);
    """
    session = _get_session()
    env_key = (
        "META_NONLINEAR_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL"
        if is_mixed_categorical
        else "META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL"
    )
    expected_total = int(os.getenv(env_key, "1000"))
    _build_meta_nonlinear_classification_index_impl(session, is_mixed_categorical, expected_total)
    if is_mixed_categorical:
        counts = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX GROUP BY split ORDER BY split"
        ).collect()
        return (
            "META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX build complete.\n\nFull split counts:\n"
            + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
        )
    counts = session.sql(
        "SELECT split, COUNT(*) AS task_count "
        "FROM META_NONLINEAR_CLASSIFICATION_DATASET_INDEX GROUP BY split ORDER BY split"
    ).collect()
    return (
        "META_NONLINEAR_CLASSIFICATION_DATASET_INDEX build complete.\n\nFull split counts:\n"
        + "\n".join(f"  {row[0]}: {row[1]}" for row in counts)
    )


def build_meta_nonlinear_classification_dataset_index_with_flag_and_total(
    is_mixed_categorical: bool, expected_total: int
) -> str:
    """Boolean-routed handler — rebuild nonlinear-classification index with explicit total.

    Usage:
        CALL build_meta_nonlinear_classification_dataset_index(FALSE, 1000);
        CALL build_meta_nonlinear_classification_dataset_index(TRUE, 500);
    """
    session = _get_session()
    _build_meta_nonlinear_classification_index_impl(session, is_mixed_categorical, int(expected_total))
    return build_meta_nonlinear_classification_dataset_index_with_flag(is_mixed_categorical)


def _validate_meta_dataset_index(session):
    """Pre-flight check: raises RuntimeError if META_REGRESSION_DATASET_INDEX is missing or wrong."""
    # (uses module-level EXPECTED_INDEX_COUNTS)
    try:
        rows = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_REGRESSION_DATASET_INDEX GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            "META_REGRESSION_DATASET_INDEX does not exist or cannot be queried. "
            "Run CALL build_meta_dataset_index(FALSE); first."
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
            "Run CALL build_meta_dataset_index(FALSE); to rebuild."
        )
    print(
        "META_REGRESSION_DATASET_INDEX validated: "
        + ", ".join(f"{s}={counts[s]}" for s in ("train", "val", "test"))
    )

    # Column existence check
    _REQUIRED_COLUMNS = "split, task_id, stage_path, p, n_train, hpo_bucket, prior_regime"
    try:
        session.sql(
            f"SELECT {_REQUIRED_COLUMNS} FROM META_REGRESSION_DATASET_INDEX LIMIT 1"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"META_REGRESSION_DATASET_INDEX is missing one or more required columns "
            f"({_REQUIRED_COLUMNS}). "
            "Rebuild with CALL build_meta_dataset_index(FALSE); "
            f"Error: {exc}"
        ) from exc

    # Stage file accessibility spot-check (numeric/ subdir)
    for _split in ("train", "val"):
        try:
            _files = session.sql(
                f"LIST @META_REGRESSION_DATASET_STAGE/numeric/{_split}/"
            ).collect()
            if not _files:
                raise RuntimeError(
                    f"No staged files found in @META_REGRESSION_DATASET_STAGE/numeric/{_split}/. "
                    "Re-upload training data before starting a GPU job."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"META_REGRESSION_DATASET_INDEX references "
                f"@META_REGRESSION_DATASET_STAGE/numeric/{_split}/ "
                f"but the stage directory is inaccessible: {exc}. "
                "Verify stage permissions and re-upload data."
            ) from exc


def _validate_nonlinear_training_index(session) -> None:
    """Pre-flight for META_NONLINEAR_REGRESSION_DATASET_INDEX + @META_NONLINEAR_REGRESSION_DATASET_STAGE."""
    try:
        rows = session.sql(
            "SELECT split, COUNT(*) AS task_count "
            "FROM META_NONLINEAR_REGRESSION_DATASET_INDEX GROUP BY split"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            "META_NONLINEAR_REGRESSION_DATASET_INDEX does not exist or cannot be queried. "
            "Run CALL build_meta_nonlinear_dataset_index(FALSE); first."
        ) from exc
    counts = {str(row[0]).lower(): int(row[1]) for row in rows}
    mismatches = {
        split: {"expected": exp, "actual": counts.get(split, 0)}
        for split, exp in _NONLINEAR_EXPECTED_INDEX_COUNTS.items()
        if counts.get(split, 0) != exp
    }
    if mismatches:
        raise RuntimeError(
            f"META_NONLINEAR_REGRESSION_DATASET_INDEX has wrong split counts: {mismatches}. "
            "Run CALL build_meta_nonlinear_dataset_index(FALSE); to rebuild."
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
            f"META_NONLINEAR_REGRESSION_DATASET_INDEX missing required columns "
            f"({_REQUIRED_COLUMNS}): {exc}"
        ) from exc
    for _split in ("train", "val"):
        try:
            _files = session.sql(
                f"LIST @META_NONLINEAR_REGRESSION_DATASET_STAGE/numeric/{_split}/"
            ).collect()
            if not _files:
                raise RuntimeError(
                    f"No staged files in "
                    f"@META_NONLINEAR_REGRESSION_DATASET_STAGE/numeric/{_split}/. "
                    "Re-upload nonlinear training data before starting a GPU job."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"@META_NONLINEAR_REGRESSION_DATASET_STAGE/numeric/{_split}/ inaccessible: {exc}"
            ) from exc


def _validate_training_index(session, training_data_family: str) -> None:
    """Route to the correct index validation based on training_data_family."""
    from task_routing import get_training_data_spec
    spec = get_training_data_spec(training_data_family)
    if spec.is_nonlinear:
        _validate_nonlinear_training_index(session)
    else:
        _validate_meta_dataset_index(session)


_GATE_HIDDEN_DIM_CANDIDATES = [32, 64, 128]


def run_pipeline() -> str:
    """Full two-sweep training pipeline (linear families only).

    Enforces linear-only: raises RuntimeError immediately if
    DEFAULT_TRAINING_DATA_FAMILY == "synthetic_regression_nonlinear". For nonlinear
    production training use the dedicated sequence:
      1. CALL build_meta_nonlinear_dataset_index();
      2. CALL run_pretrain_pipeline_nonlinear(...);
      3. CALL run_hpo_pipeline(..., 'nonlinear_model', ...);
      4. CALL run_model_training(...);

    Step 1: Validate training dataset index (META_REGRESSION_DATASET_INDEX for linear families)
    Step 2: Pre-training (pretrain.pt, no BEST_CONFIG — establishes warm-start baseline)
    Step 3: HPO sweep — linear_model (tunes optimizer/regularization/Ridge Expert)
            Writes best_config_linear_model.json.
    Step 4: MODEL3 DDP memory probe (worst-case d_phi=256, n_blocks=2)
            Guards against OOM before the architecture HPO allocates GPU cluster.
    Step 5: HPO sweep — architecture (tunes d_phi and n_sab_feat)
            Reads best_config_linear_model.json via HPO_BASELINE_CONFIG_STAGE_PATH.
            Writes best_config_linear_model_architecture.json and merged best_config.json.
    Step 6: Load merged best_config.json
    Step 7: Final training with best_config + pretrain warm-start (best_regression.pt)
    """
    if DEFAULT_TRAINING_DATA_FAMILY == _NONLINEAR_TRAINING_FAMILY:
        raise RuntimeError(
            "run_pipeline() is the linear synthetic regression training pipeline and rejects "
            f"TRAINING_DATA_FAMILY={_NONLINEAR_TRAINING_FAMILY!r}. "
            "For nonlinear production training use the dedicated sequence:\n"
            "  1. CALL build_meta_nonlinear_dataset_index();\n"
            "  2. CALL run_pretrain_pipeline_nonlinear('market_exchangeable_icl', "
            "'synthetic_regression_nonlinear', 'inductive_forecasting');\n"
            "  3. CALL run_hpo_pipeline(..., 'nonlinear_model', '', "
            "'@MODEL_STAGE/checkpoints/pretrain_nonlinear_model.pt');\n"
            "  4. -- optional: CALL run_hpo_pipeline(..., 'nonlinear_model_architecture', "
            "'@MODEL_STAGE/hpo/best_config_nonlinear_model.json', "
            "'@MODEL_STAGE/checkpoints/pretrain_nonlinear_model.pt');\n"
            "  5. CALL run_model_training('market_exchangeable_icl', "
            "'synthetic_regression_nonlinear', 'inductive_forecasting');\n"
            "Set TRAINING_DATA_FAMILY to a linear family (e.g. synthetic_regression_combined) "
            "to use run_pipeline() for linear training."
        )
    session = _get_session()
    _common_env = {
        "HOME":                      "/tmp",
        "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
        "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),
        "STRICT_WORLD_SIZE_CHECK":   "true",
        "MODEL_FAMILY":              DEFAULT_MODEL_FAMILY,
        "TRAINING_DATA_FAMILY":      DEFAULT_TRAINING_DATA_FAMILY,
        "HPO_TRAINING_DATA_FAMILY":  DEFAULT_TRAINING_DATA_FAMILY,
        "MODEL_DESIGN_PATTERN":     DEFAULT_MODEL_DESIGN_PATTERN,
    }

    # ── Step 1: Validate training dataset index ──────────────────────────────
    print("Step 1: Validating training dataset index ...")
    _validate_training_index(session, DEFAULT_TRAINING_DATA_FAMILY)

    # ── Step 2: Pre-training ──────────────────────────────────────────────────
    print("Step 2: Submitting pre-training job (pretrain.pt) ...")
    pretrain_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            **_common_env,
            "CHECKPOINT_OUTPUT_NAME": "pretrain.pt",
        },
        session=session,
    )
    _wait_done(pretrain_job, "Pre-training", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "pretrain.pt"):
        raise RuntimeError(
            "Step 2 (pre-training) did not produce pretrain.pt in "
            f"{MODEL_STAGE}/checkpoints/. Check container logs before proceeding."
        )

    # ── Step 3: HPO sweep — linear_model ───────────────────────────────────
    print("Step 3: Submitting HPO linear_model sweep ...")
    hpo_rr_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,
        env_vars={
            **_common_env,
            "HPO_SWEEP_MODE": "linear_model",
            "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH": f"{MODEL_STAGE}/checkpoints/pretrain.pt",
        },
        session=session,
    )
    _wait_done(hpo_rr_job, "HPO linear_model", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/hpo/", "best_config_linear_model.json"):
        raise RuntimeError(
            "Step 3 (HPO linear_model) did not produce best_config_linear_model.json in "
            f"{MODEL_STAGE}/hpo/. Check container logs before proceeding."
        )

    # ── Step 4: MODEL3 DDP memory probe (pre-architecture HPO gate) ───────────
    print("Step 4: Submitting MODEL3 DDP memory probe (d_phi=256, n_blocks=2) ...")
    probe_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="model_ddp_memory_probe.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "HOME":                             "/tmp",
            "TRAIN_NUM_NODES":                  str(TRAIN_NUM_NODES),
            "EXPECTED_TRAIN_WORLD_SIZE":         str(TRAIN_NUM_NODES * 4),
            "STRICT_WORLD_SIZE_CHECK":           "true",
            "MODEL_ARCH_VERSION":               "model4",
            "MODEL_DESIGN_PATTERN":             "inductive_forecasting",
            "MODEL_FAMILY":                     "market_exchangeable_icl",
            "MODEL_PROBE_N_CONTEXT":            "200",
            "MODEL_PROBE_P_FEATURES":           "128",
            "MODEL_PROBE_M_QUERY":              "128",
            "MODEL_PROBE_D_PHI":                "256",
            "MODEL_PROBE_N_BLOCKS":             "2",
            "MODEL_PROBE_RUN_BACKWARD":         "true",
            "MODEL_PROBE_DTYPE":                "float32",
            "MODEL_PROBE_MAX_GPU_MEMORY_FRACTION": "0.9",
            "MODEL_PROBE_STRICT_MEMORY_GUARD":  "true",
            "MODEL_PROBE_MEMORY_SAFETY_FACTOR": "1.5",
            "MODEL_PROBE_OUTPUT_STAGE":         f"{MODEL_STAGE}/diagnostics/",
        },
        session=session,
    )
    _wait_done(probe_job, "MODEL3DDPMemoryProbe (pre-architecture HPO gate)", session)

    # ── Step 5: HPO sweep — architecture ─────────────────────────────────────
    print("Step 5: Submitting HPO linear_model_architecture sweep ...")
    hpo_arch_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,
        env_vars={
            **_common_env,
            "HPO_SWEEP_MODE":                    "linear_model_architecture",
            "HPO_BASELINE_CONFIG_STAGE_PATH":    f"{MODEL_STAGE}/hpo/best_config_linear_model.json",
        },
        session=session,
    )
    _wait_done(hpo_arch_job, "HPO linear_model_architecture", session)

    if not _stage_file_exists(session, f"{MODEL_STAGE}/hpo/", "best_config.json"):
        raise RuntimeError(
            "Step 5 (HPO linear_model_architecture) did not produce best_config.json in "
            f"{MODEL_STAGE}/hpo/. Check container logs before proceeding."
        )

    # ── Step 6: Load merged best_config.json ──────────────────────────────────
    print("Step 6: Loading merged best_config.json ...")
    session.file.get(f"{MODEL_STAGE}/hpo/best_config.json", LOCAL_TMP_DIR)
    with open(os.path.join(LOCAL_TMP_DIR, "best_config.json")) as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    # ── Step 7: Final training ─────────────────────────────────────────────────
    print("Step 7: Submitting final training job ...")
    train_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            **_common_env,
            "BEST_CONFIG":              json.dumps(best_config),
            "PRETRAIN_CHECKPOINT_PATH": f"{MODEL_STAGE}/checkpoints/pretrain.pt",
            "PRETRAIN_LOAD_POLICY":     "allow_cold_start_on_arch_mismatch",
            "CHECKPOINT_OUTPUT_NAME":   "best_regression.pt",
        },
        session=session,
    )
    _wait_done(train_job, "Final training", session)

    hpo_contents        = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Training pipeline complete "
        "(Validate → Pretrain → HPO linear_model → MemProbe → HPO architecture → Final training).\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
        + "\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )
