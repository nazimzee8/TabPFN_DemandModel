"""
Stored procedure handler for submitting only the model training MLJob.
"""

import json
import os
import tempfile

from snowflake.ml.jobs import submit_from_stage

EXPECTED_INDEX_COUNTS = {"train": 800, "val": 100, "test": 100}

GPU_POOL = "DEEPSET_GPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
TRAIN_NUM_NODES = 10
LOCAL_TMP_DIR = tempfile.gettempdir()

# Training data family — identifies the synthetic data suite used for this training run.
# Production synthetic regression evaluation checkpoints use synthetic_regression_combined
# (combined suite linear_all_v1, which includes primary + OOD data).
# Override via TRAINING_DATA_FAMILY env var when launching a different training mode.
DEFAULT_TRAINING_DATA_FAMILY = os.getenv(
    "TRAINING_DATA_FAMILY", "synthetic_regression_combined"
)

# MODEL3 architecture selectors — propagated to the training MLJob env_vars.
DEFAULT_MODEL_FAMILY          = os.getenv("MODEL_FAMILY",          "market_exchangeable_icl")
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
        logs = (
            f"(job.get_logs() failed: {exc}. Use Snowflake service/job log "
            "retrieval from Snowsight or the MLJob object for details.)"
        )
    print(f"{label} container logs:\n", logs)
    print(
        f"\n[DIAGNOSTIC] {label} failed. Run these queries to investigate:\n"
        "  LIST @MODEL_STAGE/checkpoints/;\n"
        "  LIST @MODEL_STAGE/checkpoints/ PATTERN='.*train_failure[.]json';\n"
        "  LIST @MODEL_STAGE/checkpoints/ PATTERN='.*training_submission_started[.]json';\n"
        "  LIST @MODEL_STAGE/checkpoints/ PATTERN='.*best[.]pt';\n"
        "If train_failure.json is absent, the failure happened before Python-side diagnostics\n"
        "completed, OR the Snowpark upload itself failed — search container logs for\n"
        "'[TRAINING FAILURE JSON]' to find the payload printed before upload was attempted.\n"
        "If training_submission_started.json is absent, failure happened before train.py main().\n"
        "Search logs for '[train.py main] starting PyTorchDistributor.run' and\n"
        "'[train_fn] entered train_fn' to isolate the failure boundary.",
        flush=True,
    )
    if _detect_prometheus_mmap_failure(logs if isinstance(logs, str) else ""):
        print(
            "\n[DIAGNOSTIC] Root-cause boundary detected:\n"
            "Prometheus active query tracker mmap panic occurred during Snowflake MLJob/Ray "
            "runtime startup.\n\n"
            "This is a runtime/infrastructure startup failure, not a train.py/model/DDP/dataset "
            "failure unless train.py boundary markers appear in the logs above.\n\n"
            "Expected markers proving Python training reached execution:\n"
            "  [train.py main] entered main\n"
            "  [train.py main] starting PyTorchDistributor.run\n"
            "  [train_fn] entered train_fn\n"
            "  [train_fn] topology\n\n"
            "If these markers are absent, do not debug model code.\n\n"
            "train_failure.json may not exist because train.py did not reach its Python-side "
            "exception handler — absence is expected when the managed runtime fails before "
            "Python diagnostics could run.\n\n"
            "Recommended next step:\n"
            "  CALL run_training_runtime_probe(1);   -- single-node probe\n"
            "  CALL run_training_runtime_probe(10);  -- full-topology probe\n\n"
            "If the probe fails before printing '[runtime_probe] entered Python', escalate to "
            "Snowflake Support as a managed MLJob/Ray/Prometheus runtime issue.",
            flush=True,
        )
    raise RuntimeError(f"{label} failed with status {job.status!r}\n--- logs ---\n{logs}")


def _list_stage(session, stage_path):
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _stage_file_exists(session, stage_path, filename):
    rows = session.sql(f"LIST {stage_path}").collect()
    return any(str(row[0]).rstrip("/").endswith(f"/{filename}") for row in rows)


def _detect_prometheus_mmap_failure(logs: str) -> bool:
    """Return True if logs contain evidence of the Prometheus active query tracker mmap panic."""
    markers = [
        "Unable to create mmap-ed active query log",
        "Failed to mmap",
        "data/queries.active",
        "activeQueryTracker",
    ]
    return any(marker in logs for marker in markers)


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


def _apply_nonlinear_cold_start_guard() -> None:
    """Raise RuntimeError unless ALLOW_NONLINEAR_COLD_START=true.

    Called when a nonlinear best_config has no pretrain_checkpoint_stage_path.
    This prevents silent cold-start of nonlinear final training; the intended flow is:

        1. CALL run_pretrain_pipeline_nonlinear(...)  → pretrain_nonlinear_meta.pt
        2. CALL run_hpo_pipeline(... , pretrain_checkpoint_stage_path=...)  → best_config.json
        3. CALL run_model_training(...)

    For development-only cold-start, set ALLOW_NONLINEAR_COLD_START=true.
    """
    allow_cold_start = os.getenv("ALLOW_NONLINEAR_COLD_START", "").lower() == "true"
    if not allow_cold_start:
        raise RuntimeError(
            "[run_model_training] Nonlinear config has no pretrain checkpoint in "
            "best_config._meta.pretrain_checkpoint_stage_path. "
            "Run CALL run_pretrain_pipeline_nonlinear(...) then rerun HPO, "
            "or set ALLOW_NONLINEAR_COLD_START=true to override (dev only)."
        )
    print("[run_model_training] ALLOW_NONLINEAR_COLD_START=true: cold-starting.", flush=True)


def _run_model_training_impl(
    session,
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Core implementation for model training submission.

    All logic is here; run_model_training() and run_model_training_model() delegate to this.
    """
    if not _stage_file_exists(session, f"{MODEL_STAGE}/hpo/", "best_config.json"):
        raise FileNotFoundError(
            f"{MODEL_STAGE}/hpo/best_config.json is required before training. "
            f"Run CALL run_hpo_pipeline() and inspect {MODEL_STAGE}/hpo/hpo_failure.json "
            "if the config is missing."
        )

    session.file.get(f"{MODEL_STAGE}/hpo/best_config.json", LOCAL_TMP_DIR)
    with open(os.path.join(LOCAL_TMP_DIR, "best_config.json")) as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    _validate_meta_dataset_index(session)

    # Pretrain load policy: architecture sweep allows cold-start on mismatch
    # (d_phi/n_sab_feat may differ from pretrain checkpoint); ridge_residual
    # requires exact match because gate-specific checkpoints are built to match.
    hpo_sweep_mode = best_config.get("hpo_sweep_mode", "ridge_residual")
    is_nonlinear_config = bool(best_config.get("use_latent_ridge_expert", False)) or (
        str(hpo_sweep_mode).startswith("nonlinear_")
    )
    pretrain_policy = (
        "allow_cold_start_on_arch_mismatch"
        if hpo_sweep_mode in ("architecture", "nonlinear_meta", "nonlinear_architecture")
        else "require_match"
    )

    # Resolve pretrain checkpoint (strict — no cold-start, no legacy pretrain.pt fallback):
    #   1. best_config._meta.pretrain_checkpoint_stage_path (written by HPO, most accurate)
    #   2. Fallback: @MODEL_STAGE/checkpoints/pretrain_gate<gate_dim>.pt
    # FileNotFoundError is raised before submit_from_stage() if no valid checkpoint exists.
    _meta = best_config.get("_meta", {})
    _meta_ckpt = str(_meta.get("pretrain_checkpoint_stage_path", "")).strip()
    gate_dim = int(best_config.get("gate_hidden_dim", 64))

    pretrain_checkpoint_path = ""
    if _meta_ckpt:
        _ckpt_filename = _meta_ckpt.rsplit("/", 1)[-1]
        if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", _ckpt_filename):
            raise FileNotFoundError(
                f"[run_model_training] Pretrain checkpoint from best_config._meta not found: "
                f"{_meta_ckpt!r} (gate_hidden_dim={gate_dim}). "
                f"Check @MODEL_STAGE/checkpoints/ and rerun the gate-specific pretrain: "
                f"CALL run_pretrain_pipeline('...', '...', '...', {gate_dim});"
            )
        pretrain_checkpoint_path = _meta_ckpt
        print(
            f"[run_model_training] Using pretrain checkpoint from _meta: "
            f"{pretrain_checkpoint_path!r}",
            flush=True,
        )
    elif is_nonlinear_config:
        _apply_nonlinear_cold_start_guard()
    else:
        _gate_ckpt_name = f"pretrain_gate{gate_dim}.pt"
        _gate_ckpt_path = f"{MODEL_STAGE}/checkpoints/{_gate_ckpt_name}"
        if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", _gate_ckpt_name):
            raise FileNotFoundError(
                f"[run_model_training] No pretrain checkpoint found for "
                f"gate_hidden_dim={gate_dim}. "
                f"Expected: {_gate_ckpt_path} in @MODEL_STAGE/checkpoints/. "
                "Run the gate-specific pretrain first: "
                f"CALL run_pretrain_pipeline('...', '...', '...', {gate_dim});"
            )
        pretrain_checkpoint_path = _gate_ckpt_path
        print(
            f"[run_model_training] Using pretrain_gate{gate_dim}.pt "
            f"(gate_hidden_dim={gate_dim}, no _meta path found)",
            flush=True,
        )

    env_vars = {
        "BEST_CONFIG":               json.dumps(best_config),
        "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
        "CHECKPOINT_OUTPUT_NAME":    "best.pt",
        "HOME":                      "/tmp",
        "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),   # 10 × 4 = 40
        "STRICT_WORLD_SIZE_CHECK":   "true",
        "MODEL_FAMILY":               model_family,
        "TRAINING_DATA_FAMILY":       training_data_family,
        "MODEL_DESIGN_PATTERN":      model_design_pattern,
        "PRETRAIN_LOAD_POLICY":      pretrain_policy,
        "PRETRAIN_CHECKPOINT_PATH":  pretrain_checkpoint_path,
    }
    # Topology preflight: EXPECTED_TRAIN_WORLD_SIZE must equal TRAIN_NUM_NODES × 4.
    _expected_ws = int(env_vars["EXPECTED_TRAIN_WORLD_SIZE"])
    if _expected_ws != TRAIN_NUM_NODES * 4:
        raise RuntimeError(
            f"Topology mismatch before submit: EXPECTED_TRAIN_WORLD_SIZE={_expected_ws} "
            f"!= TRAIN_NUM_NODES({TRAIN_NUM_NODES}) × 4. Fix the constant before submitting."
        )

    print(
        "Training submission config:",
        {
            "target_instances":           TRAIN_NUM_NODES,
            "TRAIN_NUM_NODES":            env_vars["TRAIN_NUM_NODES"],
            "EXPECTED_TRAIN_WORLD_SIZE":  env_vars["EXPECTED_TRAIN_WORLD_SIZE"],
            "STRICT_WORLD_SIZE_CHECK":    env_vars["STRICT_WORLD_SIZE_CHECK"],
            "CHECKPOINT_OUTPUT_NAME":     env_vars["CHECKPOINT_OUTPUT_NAME"],
            "TRAINING_DATA_FAMILY":       env_vars["TRAINING_DATA_FAMILY"],
            "PRETRAIN_LOAD_POLICY":       env_vars["PRETRAIN_LOAD_POLICY"],
            "compute_pool":               GPU_POOL,
            "entrypoint":                 "train.py",
            "source":                     SCRIPTS_STAGE,
            "stage_name":                 MLJOB_PAYLOAD_STAGE,
            "has_pretrain":               "PRETRAIN_CHECKPOINT_PATH" in env_vars,
        },
        flush=True,
    )

    import json as _json_sp
    import time as _time_sp
    _sp_start_payload = {
        "time_utc":                  _time_sp.strftime("%Y-%m-%dT%H:%M:%SZ", _time_sp.gmtime()),
        "train_num_nodes":           TRAIN_NUM_NODES,
        "expected_train_world_size": int(env_vars["EXPECTED_TRAIN_WORLD_SIZE"]),
        "strict_world_size_check":   env_vars["STRICT_WORLD_SIZE_CHECK"],
        "checkpoint_output_name":    env_vars["CHECKPOINT_OUTPUT_NAME"],
        "training_data_family":      training_data_family,
        "has_best_config":           True,
        "has_pretrain":              "PRETRAIN_CHECKPOINT_PATH" in env_vars,
        "pretrain_load_policy":      pretrain_policy,
        "compute_pool":              GPU_POOL,
        "target_instances":          TRAIN_NUM_NODES,
        "entrypoint":                "train.py",
        "source":                    SCRIPTS_STAGE,
        "stage_name":                MLJOB_PAYLOAD_STAGE,
        "best_config":               best_config,
    }
    _sp_start_local = os.path.join(LOCAL_TMP_DIR, "training_submission_started.json")
    with open(_sp_start_local, "w", encoding="utf-8") as _sp_f:
        _json_sp.dump(_sp_start_payload, _sp_f, indent=2, sort_keys=True)
    try:
        session.file.put(
            _sp_start_local, "@MODEL_STAGE/checkpoints/",
            overwrite=True, auto_compress=False,
        )
        print("Uploaded training_submission_started.json to @MODEL_STAGE/checkpoints/", flush=True)
    except Exception as _sp_exc:
        print(f"[WARNING] Could not upload training_submission_started.json: {_sp_exc}", flush=True)
        print(
            "[SUBMISSION START JSON]",
            _json_sp.dumps(_sp_start_payload, indent=2, sort_keys=True),
            flush=True,
        )

    print("Submitting training job ...")
    train_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars=env_vars,
        session=session,
    )
    _wait_done(train_job, "Training")

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Model training complete.\n\n"
        "MODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )


def run_model_training() -> str:
    """Zero-arg entrypoint: uses env-var defaults."""
    session = _get_session()
    return _run_model_training_impl(
        session,
        DEFAULT_MODEL_FAMILY,
        DEFAULT_TRAINING_DATA_FAMILY,
        DEFAULT_MODEL_DESIGN_PATTERN,
    )


def run_model_training_model(
    model_family: str,
    training_data_family: str,
    model_design_pattern: str,
) -> str:
    """Parameterized model training handler.

    Matches the explicit runtime lineage variable pattern used by run_pretrain_pipeline_model()
    and run_hpo_pipeline_model(). Passes MODEL_FAMILY, TRAINING_DATA_FAMILY, and
    MODEL_DESIGN_PATTERN directly to the training MLJob env_vars (does not mutate os.environ).

    MODEL_ARCH_VERSION is not a parameter here — the default 'model3' is set internally
    by train.py based on MODEL_FAMILY. Do not expose it as a SQL/runtime selector.

    SQL call:
        CALL run_model_training(
            'market_exchangeable_icl',
            'synthetic_regression_combined',
            'inductive_forecasting'
        );
    """
    print(
        f"[run_model_training_model] model_family={model_family!r} "
        f"training_data_family={training_data_family!r} "
        f"model_design_pattern={model_design_pattern!r}",
        flush=True,
    )
    session = _get_session()
    return _run_model_training_impl(
        session, model_family, training_data_family, model_design_pattern
    )


def run_model_ddp_memory_probe(
    model_design_pattern: str,
    model_family: str,
    n_context: int,
    p_features: int,
    m_query: int,
    d_phi: int,
    n_blocks: int,
    run_backward: bool,
) -> str:
    """Launch the MODEL3 DDP memory probe on DEEPSET_GPU_POOL.

    Measures peak CUDA memory per DDP worker for the given MODEL3 ICL shape before
    pretrain / HPO / final training.  Because MODEL3 meta-training uses back-propagation,
    run_backward should always be True for training-regime validation.

    The probe uses the same DDP topology as training: TRAIN_NUM_NODES nodes ×
    4 workers/GPUs = world_size 40.  Results are uploaded as structured JSON to
    @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json.

    Call via:
        CALL run_model_ddp_memory_probe(
            'inductive_forecasting', 'market_exchangeable_icl',
            200, 128, 128, 128, 1, TRUE
        );

    After the call:
        LIST @MODEL_STAGE/diagnostics/;
        SELECT $1 FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json
          (FILE_FORMAT => (TYPE = JSON));
    """
    # ---- Validate arguments before submitting ----
    if model_design_pattern == "transductive_completion":
        raise ValueError(
            "run_model_ddp_memory_probe does not support "
            "model_design_pattern='transductive_completion'. "
            "Use model_design_pattern='inductive_forecasting'."
        )
    if model_design_pattern != "inductive_forecasting":
        raise ValueError(
            f"Unsupported model_design_pattern={model_design_pattern!r}. "
            "Only 'inductive_forecasting' is currently supported."
        )
    if model_family != "market_exchangeable_icl":
        raise ValueError(
            f"Unsupported model_family={model_family!r}. "
            "Only 'market_exchangeable_icl' is currently supported."
        )
    for _name, _val in [
        ("n_context",  n_context),
        ("p_features", p_features),
        ("m_query",    m_query),
        ("d_phi",      d_phi),
        ("n_blocks",   n_blocks),
    ]:
        if _val <= 0:
            raise ValueError(
                f"Shape parameter {_name}={_val!r} must be a positive integer."
            )

    expected_world_size = TRAIN_NUM_NODES * 4   # 10 × 4 = 40

    env_vars = {
        "HOME":                                "/tmp",
        "TRAIN_NUM_NODES":                     str(TRAIN_NUM_NODES),
        "EXPECTED_TRAIN_WORLD_SIZE":           str(expected_world_size),
        "STRICT_WORLD_SIZE_CHECK":             "true",
        "MODEL_DESIGN_PATTERN":               model_design_pattern,
        "MODEL_FAMILY":                        model_family,
        "MODEL_PROBE_N_CONTEXT":              str(n_context),
        "MODEL_PROBE_P_FEATURES":             str(p_features),
        "MODEL_PROBE_M_QUERY":                str(m_query),
        "MODEL_PROBE_D_PHI":                  str(d_phi),
        "MODEL_PROBE_N_BLOCKS":               str(n_blocks),
        "MODEL_PROBE_RUN_BACKWARD":           "true" if run_backward else "false",
        "MODEL_PROBE_DTYPE":                  "float32",
        "MODEL_PROBE_MAX_GPU_MEMORY_FRACTION": "0.9",
        "MODEL_PROBE_STRICT_MEMORY_GUARD":    "true",
        "MODEL_PROBE_MEMORY_SAFETY_FACTOR":   "1.5",
        "MODEL_PROBE_OUTPUT_STAGE":           f"{MODEL_STAGE}/diagnostics/",
    }

    print(
        "Submitting MODEL3 DDP memory probe:",
        {
            "entrypoint":            "model_ddp_memory_probe.py",
            "compute_pool":          GPU_POOL,
            "target_instances":      TRAIN_NUM_NODES,
            "TRAIN_NUM_NODES":       env_vars["TRAIN_NUM_NODES"],
            "EXPECTED_WORLD_SIZE":   env_vars["EXPECTED_TRAIN_WORLD_SIZE"],
            "model_design_pattern": model_design_pattern,
            "model_family":          model_family,
            "shape": {
                "n_context":    n_context,
                "p_features":   p_features,
                "m_query":      m_query,
                "d_phi":        d_phi,
                "n_blocks":     n_blocks,
                "run_backward": run_backward,
            },
        },
        flush=True,
    )

    session = _get_session()
    probe_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="model_ddp_memory_probe.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars=env_vars,
        session=session,
    )
    _wait_done(probe_job, "MODEL3DDPMemoryProbe")

    diagnostics = _list_stage(session, f"{MODEL_STAGE}/diagnostics/")
    return (
        "MODEL3 DDP memory probe complete.\n\n"
        "MODEL_STAGE diagnostics:\n"
        + "\n".join(f"  {p}" for p in diagnostics)
        + "\n\nTo read results:\n"
        "  SELECT $1\n"
        "  FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json\n"
        "    (FILE_FORMAT => (TYPE = JSON));"
    )


def run_training_runtime_probe(target_instances: int) -> str:
    """Submit a minimal runtime probe MLJob to verify Python entrypoint reachability.

    Args:
        target_instances: Number of nodes to use.
            1 = single-node probe (fast, minimal resources).
            10 = full-topology probe (same size as final training).

    Call with:
        CALL run_training_runtime_probe(1);
        CALL run_training_runtime_probe(2);
        CALL run_training_runtime_probe(5);
        CALL run_training_runtime_probe(10);
    """
    if target_instances not in (1, 2, 5, 10):
        raise ValueError(
            f"target_instances must be 1, 2, 5, or 10, got {target_instances}."
        )

    env_vars = {
        "TRAIN_NUM_NODES":           str(target_instances),
        "EXPECTED_TRAIN_WORLD_SIZE": str(target_instances * 4),
        "STRICT_WORLD_SIZE_CHECK":   "false",   # probe does not validate DDP world size
        "CHECKPOINT_OUTPUT_NAME":    "runtime_probe.txt",
        "HOME":                      "/tmp",
    }

    print(
        f"Submitting runtime probe: target_instances={target_instances} ...",
        flush=True,
    )
    print(
        "Expected success marker: '[runtime_probe] entered Python'\n"
        "Expected completion marker: '[runtime_probe] completed'\n"
        "If the Prometheus mmap panic appears before '[runtime_probe] entered Python', "
        "escalate to Snowflake Support as a managed MLJob/Ray/Prometheus runtime issue.",
        flush=True,
    )

    session = _get_session()
    probe_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="runtime_probe.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=target_instances,
        env_vars=env_vars,
        session=session,
    )
    _wait_done(probe_job, f"RuntimeProbe(target_instances={target_instances})")

    return f"Runtime probe complete (target_instances={target_instances})."
