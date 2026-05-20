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
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
DEFAULT_MODEL_DESIGN_PATTERN = os.getenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")

# HPO sweep mode — selects search space in hpo.py.
# ridge_residual: tunes optimizer/regularization/Ridge Expert; architecture fixed (default).
# architecture:   tunes d_phi/n_sab_feat; allows cold-start on pretrain mismatch.
DEFAULT_HPO_SWEEP_MODE = os.getenv("HPO_SWEEP_MODE", "ridge_residual")

# Baseline config stage path for architecture sweep.
# Must point to best_config_ridge_residual.json from sweep 1 when HPO_SWEEP_MODE=architecture.
DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH = os.getenv(
    "HPO_BASELINE_CONFIG_STAGE_PATH", ""
)

_ALLOWED_HPO_SWEEP_MODES = {"ridge_residual", "architecture"}
if DEFAULT_HPO_SWEEP_MODE not in _ALLOWED_HPO_SWEEP_MODES:
    raise ValueError(
        f"Invalid HPO_SWEEP_MODE={DEFAULT_HPO_SWEEP_MODE!r}. "
        f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
    )


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
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    hpo_sweep_mode: str,
    hpo_baseline_config_stage_path: str = "",
) -> str:
    target_instances = 5   # 5 nodes x 4 GPUs/node = 20 GPUs; Ray Tune schedules 1 GPU/trial → 20 concurrent trials
    print(
        "Submitting HPO job:",
        {
            "model_family":         model_family,
            "training_data_family": training_data_family,
            "model_design_pattern": model_design_pattern,
            "hpo_sweep_mode":       hpo_sweep_mode,
            "hpo_baseline_config_stage_path": hpo_baseline_config_stage_path or "(none)",
            "target_instances":     target_instances,
            "compute_pool":         GPU_POOL,
            "entrypoint":           "hpo.py",
        },
    )
    env_vars = {
        "HOME":                 "/tmp",
        "MODEL_FAMILY":          model_family,
        "TRAINING_DATA_FAMILY":  training_data_family,
        "MODEL_DESIGN_PATTERN":  model_design_pattern,
        "HPO_SWEEP_MODE":        hpo_sweep_mode,
    }
    if hpo_baseline_config_stage_path:
        env_vars["HPO_BASELINE_CONFIG_STAGE_PATH"] = hpo_baseline_config_stage_path
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=target_instances,
        env_vars=env_vars,
        session=session,
    )
    _wait_done(hpo_job, "HPO")

    hpo_contents = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    return (
        f"HPO pipeline complete (hpo_sweep_mode={hpo_sweep_mode!r}).\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
    )


def run_hpo_pipeline(session) -> str:
    """Zero-arg entrypoint: uses env-var defaults including DEFAULT_HPO_SWEEP_MODE."""
    return _run_hpo_impl(
        session,
        DEFAULT_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_DESIGN_PATTERN,
        DEFAULT_HPO_SWEEP_MODE,
        DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH,
    )


def run_hpo_pipeline_model(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Three-arg entrypoint: explicit model selectors; uses DEFAULT_HPO_SWEEP_MODE.

    Only supports inductive_forecasting; transductive_completion raises in hpo.py.

    Usage:
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'inductive_forecasting'
        );
    """
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        DEFAULT_HPO_SWEEP_MODE,
        DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH,
    )


def run_hpo_pipeline_model_sweep(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    hpo_sweep_mode: str,
) -> str:
    """Four-arg entrypoint: explicit model selectors + HPO sweep mode.

    hpo_sweep_mode must be one of: 'ridge_residual', 'architecture'.

    ridge_residual (default):
        Fixed architecture. Tunes lr, weight_decay, dropout, ridge_lambda,
        gate_hidden_dim, use_huber, huber_delta, lambda_l1.

    architecture:
        Tunes d_phi and n_sab_feat. Allows cold-start on pretrain mismatch.
        Run only after ridge_residual sweep and DDP memory probe for d_phi > 128.

    Usage:
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'inductive_forecasting',
            'ridge_residual'
        );
    """
    _mode = str(hpo_sweep_mode).strip().lower()
    if _mode not in _ALLOWED_HPO_SWEEP_MODES:
        raise ValueError(
            f"Invalid HPO_SWEEP_MODE={_mode!r}. "
            f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
        )
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        _mode,
        DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH,
    )


def run_hpo_pipeline_model_sweep_with_baseline(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    hpo_sweep_mode: str,
    hpo_baseline_config_stage_path: str,
) -> str:
    """Five-arg entrypoint: explicit model selectors + HPO sweep mode + baseline config path.

    hpo_sweep_mode must be one of: 'ridge_residual', 'architecture'.
    hpo_baseline_config_stage_path: stage path to best_config_ridge_residual.json.
      Required when hpo_sweep_mode='architecture'; ignored (pass '') for 'ridge_residual'.

    Usage (architecture sweep with baseline):
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'inductive_forecasting',
            'architecture',
            '@MODEL_STAGE/hpo/best_config_ridge_residual.json'
        );
    """
    _mode = str(hpo_sweep_mode).strip().lower()
    if _mode not in _ALLOWED_HPO_SWEEP_MODES:
        raise ValueError(
            f"Invalid HPO_SWEEP_MODE={_mode!r}. "
            f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
        )
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        _mode,
        hpo_baseline_config_stage_path.strip(),
    )
