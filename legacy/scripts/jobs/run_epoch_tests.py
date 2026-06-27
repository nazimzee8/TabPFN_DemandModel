"""
Stored procedure handlers for GPU epoch calibration jobs.

Run before production training to measure per-epoch wall-clock time.
Results are written to @EPOCH_STAGE and can be read with:
  SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json   (FILE_FORMAT => (TYPE = JSON));
  SELECT $1 FROM @EPOCH_STAGE/train_timing.json  (FILE_FORMAT => (TYPE = JSON));
"""

from snowflake.ml.jobs import submit_from_stage

GPU_POOL = "DEEPSET_GPU_POOL"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
EPOCH_STAGE = "@EPOCH_STAGE"
TRAIN_NUM_NODES = 10  # 10-node production DDP topology (4 workers/node = world_size 40)


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
    raise RuntimeError(f"{label} failed with status {job.status!r}\n--- logs ---\n{logs}")


def _list_stage(session, stage_path):
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _preflight_scripts(session, entrypoint):
    required = ["train.py", "model.py", "snowflake_io.py", entrypoint]
    try:
        rows = session.sql(
            "LIST @MODEL_STAGE/scripts/ "
            "PATTERN='.*(hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py'"
        ).collect()
    except Exception as exc:
        raise RuntimeError(
            f"Could not list {SCRIPTS_STAGE} before submitting {entrypoint}: {exc}"
        ) from exc

    staged = {str(row[0]).rstrip("/").split("/")[-1].lower() for row in rows}
    missing = [name for name in required if name.lower() not in staged]
    if missing:
        raise FileNotFoundError(
            f"{SCRIPTS_STAGE} is missing required MLJob source files for {entrypoint}: "
            f"{', '.join(missing)}. Re-upload scripts from SnowSQL, then recreate the "
            "stored procedure:\n"
            "  PUT file://C:/Documents/TabPFN_DemandModel/src/*.py "
            "@MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
            "  PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py "
            "@MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
        )


def run_hpo_epoch_test(session) -> str:
    """Submit hpo_epoch_test.py as a 1-node GPU job to measure HPO epoch sweep timings.
    Output: @EPOCH_STAGE/hpo_timing.json
    Read result: SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json (FILE_FORMAT => (TYPE = JSON));
    """
    _preflight_scripts(session, "hpo_epoch_test.py")
    print("Submitting HPO epoch test job ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo_epoch_test.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        env_vars={"HOME": "/tmp"},
        session=session,
    )
    _wait_done(job, "HPO epoch test")
    timing_contents = _list_stage(session, EPOCH_STAGE)
    return (
        "HPO epoch test complete.\n"
        "Read timing sweep summary: SELECT $1 FROM @EPOCH_STAGE/hpo_timing.json "
        "(FILE_FORMAT => (TYPE = JSON));\n\n"
        f"{EPOCH_STAGE}:\n" + "\n".join(f"  {p}" for p in timing_contents)
    )


def run_train_epoch_test(session) -> str:
    """Submit train_epoch_test.py as a 10-node DDP GPU job to measure training epoch time.
    Tests production GPU_NV_M topology (4 workers/node, world_size=40).
    Output: @EPOCH_STAGE/train_timing.json
    Read result: SELECT $1 FROM @EPOCH_STAGE/train_timing.json (FILE_FORMAT => (TYPE = JSON));
    """
    _preflight_scripts(session, "train_epoch_test.py")
    print("Submitting train epoch test job ...")
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train_epoch_test.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={"HOME": "/tmp", "TRAIN_NUM_NODES": str(TRAIN_NUM_NODES)},
        session=session,
    )
    _wait_done(job, "Train epoch test")
    timing_contents = _list_stage(session, EPOCH_STAGE)
    return (
        "Train epoch test complete.\n"
        "Read timing: SELECT $1 FROM @EPOCH_STAGE/train_timing.json "
        "(FILE_FORMAT => (TYPE = JSON));\n\n"
        f"{EPOCH_STAGE}:\n" + "\n".join(f"  {p}" for p in timing_contents)
    )
