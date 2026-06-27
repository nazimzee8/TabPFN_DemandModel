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
    "HPO_TRAINING_DATA_FAMILY",
    os.getenv("TRAINING_DATA_FAMILY", "synthetic_linear_regression"),
)

# MODEL3 architecture selectors — propagated to the HPO MLJob env_vars.
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
DEFAULT_MODEL_DESIGN_PATTERN = os.getenv("MODEL_DESIGN_PATTERN", "inductive_forecasting")

# HPO sweep mode — selects search space in hpo.py.
# linear_model: tunes optimizer/regularization/Ridge Expert; architecture fixed (default).
# linear_model_architecture: tunes d_phi/n_sab_feat; allows cold-start on pretrain mismatch.
# lbacnp_model: MODEL5 LBACNP search space — handles linear, nonlinear, and classification
#   families. Deprecated nonlinear_* modes are aliased here for backward compatibility.
_HPO_SWEEP_MODE_ALIASES = {
    "ridge_residual": "linear_model",
    "architecture": "linear_model_architecture",
    "linear_stats": "linear_model",
    # deprecated nonlinear modes → lbacnp_model (mirrors hpo.py alias table)
    "nonlinear_meta": "lbacnp_model",
    "nonlinear_model": "lbacnp_model",
    "nonlinear_architecture": "lbacnp_model",
    "nonlinear_model_architecture": "lbacnp_model",
    # additional deprecated aliases
    "attentive_cnp": "lbacnp_model",
    "model5": "lbacnp_model",
}
_ALLOWED_HPO_SWEEP_MODES = {
    "linear_model",
    "linear_model_architecture",
    "lbacnp_model",
}


def _canonical_hpo_sweep_mode(value: str) -> str:
    mode = str(value).strip().lower()
    return _HPO_SWEEP_MODE_ALIASES.get(mode, mode)


DEFAULT_HPO_SWEEP_MODE = _canonical_hpo_sweep_mode(os.getenv("HPO_SWEEP_MODE", "linear_model"))

# Baseline config stage path for architecture sweep.
# Must point to best_config_linear_model.json from sweep 1 when HPO_SWEEP_MODE=architecture.
DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH = os.getenv(
    "HPO_BASELINE_CONFIG_STAGE_PATH", ""
)

if DEFAULT_HPO_SWEEP_MODE not in _ALLOWED_HPO_SWEEP_MODES:
    raise ValueError(
        f"Invalid HPO_SWEEP_MODE={DEFAULT_HPO_SWEEP_MODE!r}. "
        f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)} "
        f"(deprecated aliases: {sorted(_HPO_SWEEP_MODE_ALIASES)})"
    )


def _get_session():
    from snowflake.snowpark import Session
    return Session.builder.getOrCreate()


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
    hpo_pretrain_checkpoint_stage_path: str = "",
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
    _canonical_mode = _canonical_hpo_sweep_mode(hpo_sweep_mode)
    env_vars = {
        "HOME":                 "/tmp",
        "MODEL_FAMILY":          model_family,
        "TRAINING_DATA_FAMILY":  training_data_family,
        "HPO_TRAINING_DATA_FAMILY": training_data_family,
        "MODEL_DESIGN_PATTERN":  model_design_pattern,
        "HPO_SWEEP_MODE":        _canonical_mode,
        # Propagate arch version so hpo.py writes the correct label in best_config.json.
        # lbacnp_model → model5_lbacnp; all other modes remain model4.
        "MODEL_ARCH_VERSION": (
            "model5_lbacnp" if _canonical_mode == "lbacnp_model" else "model4"
        ),
    }
    if hpo_baseline_config_stage_path:
        env_vars["HPO_BASELINE_CONFIG_STAGE_PATH"] = hpo_baseline_config_stage_path
    if hpo_pretrain_checkpoint_stage_path:
        env_vars["HPO_PRETRAIN_CHECKPOINT_STAGE_PATH"] = hpo_pretrain_checkpoint_stage_path
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


def run_hpo_pipeline() -> str:
    """Zero-arg entrypoint: uses env-var defaults including DEFAULT_HPO_SWEEP_MODE."""
    session = _get_session()
    return _run_hpo_impl(
        session,
        DEFAULT_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_DESIGN_PATTERN,
        DEFAULT_HPO_SWEEP_MODE,
        DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH,
    )


def run_hpo_pipeline_model(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Three-arg entrypoint: explicit model selectors; uses DEFAULT_HPO_SWEEP_MODE.

    Only supports inductive_forecasting; transductive_completion raises in hpo.py.

    Usage:
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_linear_regression',
            'inductive_forecasting'
        );
    """
    session = _get_session()
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        DEFAULT_HPO_SWEEP_MODE,
        DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH,
    )


def run_hpo_pipeline_model_sweep(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    hpo_sweep_mode: str,
) -> str:
    """Four-arg entrypoint: explicit model selectors + HPO sweep mode.

    hpo_sweep_mode must be one of: 'linear_model', 'linear_model_architecture',
    'lbacnp_model'. Deprecated aliases (nonlinear_model, nonlinear_model_architecture,
    model5, attentive_cnp) are accepted and remapped to 'lbacnp_model'.

    linear_model (default):
        Fixed architecture. Tunes lr, weight_decay, dropout, ridge_lambda,
        gate_hidden_dim, use_huber, huber_delta, lambda_l1.

    linear_model_architecture:
        Tunes d_phi and n_sab_feat. Allows cold-start on pretrain mismatch.
        Run only after linear_model sweep and DDP memory probe for d_phi > 128.

    lbacnp_model:
        MODEL5 LBACNP search space. Handles linear, nonlinear, and classification
        families via TRAINING_DATA_FAMILY. Sets MODEL_ARCH_VERSION=model5_lbacnp.

    Usage:
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_linear_regression',
            'inductive_forecasting',
            'lbacnp_model'
        );
    """
    _mode = _canonical_hpo_sweep_mode(hpo_sweep_mode)
    if _mode not in _ALLOWED_HPO_SWEEP_MODES:
        raise ValueError(
            f"Invalid HPO_SWEEP_MODE={_mode!r}. "
            f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
        )
    session = _get_session()
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        _mode,
        DEFAULT_HPO_BASELINE_CONFIG_STAGE_PATH,
    )


def run_hpo_pipeline_model_sweep_with_baseline(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    hpo_sweep_mode: str,
    hpo_baseline_config_stage_path: str,
) -> str:
    """Five-arg entrypoint: explicit model selectors + HPO sweep mode + baseline config path.

    hpo_sweep_mode must be one of: 'linear_model', 'linear_model_architecture', 'lbacnp_model'.
    hpo_baseline_config_stage_path: stage path to best_config_linear_model.json.
      Required when hpo_sweep_mode='linear_model_architecture'; ignored (pass '') for other modes.

    Usage (architecture sweep with baseline):
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_linear_regression',
            'inductive_forecasting',
            'linear_model_architecture',
            '@MODEL_STAGE/hpo/best_config_linear_model.json'
        );
    """
    _mode = _canonical_hpo_sweep_mode(hpo_sweep_mode)
    if _mode not in _ALLOWED_HPO_SWEEP_MODES:
        raise ValueError(
            f"Invalid HPO_SWEEP_MODE={_mode!r}. "
            f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
        )
    session = _get_session()
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        _mode,
        hpo_baseline_config_stage_path.strip(),
    )


def run_hpo_pipeline_model_sweep_with_baseline_and_pretrain(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
    hpo_sweep_mode: str,
    hpo_baseline_config_stage_path: str,
    hpo_pretrain_checkpoint_stage_path: str,
) -> str:
    """Six-arg SQL-callable entrypoint: sweep mode + baseline config + pretrain checkpoint.

    hpo_pretrain_checkpoint_stage_path: stage path to a pretrain checkpoint (e.g.
      pretrain_nonlinear_meta.pt). Used for lbacnp_model warm-start; auto-detected
      from @MODEL_STAGE/checkpoints/ when empty. Pass '' to rely on auto-detection.

    Usage (nonlinear HPO with pretrain warm-start):
        CALL run_hpo_pipeline(
            'market_exchangeable_icl',
            'synthetic_regression_nonlinear',
            'inductive_forecasting',
            'nonlinear_model',
            '',
            '@MODEL_STAGE/checkpoints/pretrain_nonlinear_meta.pt'
        );
    """
    _mode = _canonical_hpo_sweep_mode(hpo_sweep_mode)
    if _mode not in _ALLOWED_HPO_SWEEP_MODES:
        raise ValueError(
            f"Invalid HPO_SWEEP_MODE={_mode!r}. "
            f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
        )
    session = _get_session()
    return _run_hpo_impl(
        session,
        model_family,
        training_data_family,
        model_design_pattern,
        _mode,
        hpo_baseline_config_stage_path.strip(),
        hpo_pretrain_checkpoint_stage_path.strip(),
    )

