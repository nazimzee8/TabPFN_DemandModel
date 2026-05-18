"""
Stored procedure handler for submitting only the HPO MLJob.
"""

import os

from snowflake.ml.jobs import submit_from_stage

GPU_POOL = "DEEPSET_GPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"

# Training data family for traceability — passed through to the HPO job for logging.
# Does not affect search behavior or Ray Tune metric selection.
DEFAULT_TRAINING_DATA_FAMILY = os.getenv(
    "TRAINING_DATA_FAMILY", "synthetic_regression_combined"
)

# MODEL3 architecture selectors — propagated to the HPO MLJob env_vars.
# Default values preserve MODEL2 production behavior.
DEFAULT_DEEPSET_MODEL_FAMILY  = os.getenv("DEEPSET_MODEL_FAMILY",  "market_aware")
DEFAULT_MODEL_ARCH_VERSION    = os.getenv("MODEL_ARCH_VERSION",    "model2")
DEFAULT_MODEL3_DESIGN_PATTERN = os.getenv("MODEL3_DESIGN_PATTERN", "inductive_forecasting")


def _wait_done(job, label):
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


def _run_hpo_impl(
    session,
    deepset_model_family: str,
    training_data_family: str,
    model_arch_version: str,
    model3_design_pattern: str,
) -> str:
    print("Submitting HPO job ...")
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,   # 5 nodes x 4 GPUs/node = 20 GPUs; Ray Tune schedules 1 GPU/trial → 20 concurrent trials
        env_vars={
            "HOME": "/tmp",
            "DEEPSET_MODEL_FAMILY":  deepset_model_family,
            "TRAINING_DATA_FAMILY":  training_data_family,
            "MODEL_ARCH_VERSION":    model_arch_version,
            "MODEL3_DESIGN_PATTERN": model3_design_pattern,
        },
        session=session,
    )
    _wait_done(hpo_job, "HPO")

    hpo_contents = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    return (
        "HPO pipeline complete.\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
    )


def run_hpo_pipeline(session) -> str:
    """Zero-arg entrypoint: uses env-var defaults (MODEL2 production behavior)."""
    return _run_hpo_impl(
        session,
        DEFAULT_DEEPSET_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_ARCH_VERSION,
        DEFAULT_MODEL3_DESIGN_PATTERN,
    )


def run_hpo_pipeline_m3(
    session,
    deepset_model_family: str,
    training_data_family: str,
    model_arch_version: str,
    model3_design_pattern: str,
) -> str:
    """Parameterized entrypoint: explicit selectors for MODEL3 HPO.

    Only supports inductive_forecasting; transductive_completion raises in hpo.py.

    Usage:
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'model3',
            'inductive_forecasting'
        );
    """
    return _run_hpo_impl(
        session,
        deepset_model_family,
        training_data_family,
        model_arch_version,
        model3_design_pattern,
    )
