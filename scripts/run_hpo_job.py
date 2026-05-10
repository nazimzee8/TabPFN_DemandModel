"""
Stored procedure handler for submitting only the HPO MLJob.
"""

from snowflake.ml.jobs import submit_from_stage

GPU_POOL = "DEEPSET_GPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"


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


def run_hpo_pipeline(session) -> str:
    print("Submitting HPO job ...")
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=5,   # 5 nodes x 4 GPUs/node = 20 GPUs; Ray Tune schedules 1 GPU/trial → 20 concurrent trials
        env_vars={"HOME": "/tmp"},
        session=session,
    )
    _wait_done(hpo_job, "HPO")

    hpo_contents = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    return (
        "HPO pipeline complete.\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
    )
