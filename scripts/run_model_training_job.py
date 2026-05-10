"""
Stored procedure handler for submitting only the model training MLJob.
"""

import json

from snowflake.ml.jobs import submit_from_stage

EXPECTED_INDEX_COUNTS = {"train": 800, "val": 100, "test": 100}

GPU_POOL = "DEEPSET_GPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
TRAIN_NUM_NODES = 10


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


def run_model_training(session) -> str:
    if not _stage_file_exists(session, f"{MODEL_STAGE}/hpo/", "best_config.json"):
        raise FileNotFoundError(
            f"{MODEL_STAGE}/hpo/best_config.json is required before training. "
            f"Run CALL run_hpo_pipeline() and inspect {MODEL_STAGE}/hpo/hpo_failure.json "
            "if the config is missing."
        )

    session.file.get(f"{MODEL_STAGE}/hpo/best_config.json", "/tmp/")
    with open("/tmp/best_config.json") as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    _validate_meta_dataset_index(session)

    env_vars = {
        "BEST_CONFIG":               json.dumps(best_config),
        "TRAIN_NUM_NODES":           str(TRAIN_NUM_NODES),
        "CHECKPOINT_OUTPUT_NAME":    "best.pt",
        "HOME":                      "/tmp",
        "EXPECTED_TRAIN_WORLD_SIZE": str(TRAIN_NUM_NODES * 4),   # 10 × 4 = 40
        "STRICT_WORLD_SIZE_CHECK":   "true",
    }
    if _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "pretrain.pt"):
        env_vars["PRETRAIN_CHECKPOINT_PATH"] = f"{MODEL_STAGE}/checkpoints/pretrain.pt"
        print("pretrain.pt found; final training will warm-start from it.")
    else:
        print("No pretrain.pt found; final training starts from random initialization.")

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
        "has_best_config":           True,
        "has_pretrain":              "PRETRAIN_CHECKPOINT_PATH" in env_vars,
        "compute_pool":              GPU_POOL,
        "target_instances":          TRAIN_NUM_NODES,
        "entrypoint":                "train.py",
        "source":                    SCRIPTS_STAGE,
        "stage_name":                MLJOB_PAYLOAD_STAGE,
        "best_config":               best_config,
    }
    _sp_start_local = "/tmp/training_submission_started.json"
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


def run_training_runtime_probe(session, target_instances: int) -> str:
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
