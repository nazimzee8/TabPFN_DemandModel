"""
Orchestrator for the DeepSet training pipeline.

Handler for the run_training_pipeline() Snowpark stored procedure.
The Snowpark session is injected automatically by the stored procedure framework.
"""
import json

from snowflake.ml.jobs import submit_from_stage   # Container Runtime ML Jobs API

COMPUTE_POOL  = "DEEPSET_GPU_POOL"
SCRIPTS_STAGE = "@MODEL_STAGE/scripts"


def run_pipeline(session) -> str:
    # ── Phase 1: HPO ──────────────────────────────────────────────────────────
    print("Submitting HPO job …")
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=COMPUTE_POOL,
        stage_name="@MODEL_STAGE",
        target_instances=2,
        session=session,
    )
    hpo_job.wait()
    if hpo_job.status != "DONE":
        raise RuntimeError(f"HPO job failed with status {hpo_job.status!r}")
    print("HPO complete.")

    # Read best config from stage
    session.file.get("@MODEL_STAGE/hpo/best_config.json", "/tmp/")
    with open("/tmp/best_config.json") as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    # ── Phase 2: Full Training ────────────────────────────────────────────────
    print("Submitting training job …")
    train_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=COMPUTE_POOL,
        stage_name="@MODEL_STAGE",
        target_instances=4,
        env_vars={"BEST_CONFIG": json.dumps(best_config)},
        session=session,
    )
    train_job.wait()
    if train_job.status != "DONE":
        raise RuntimeError(f"Training job failed with status {train_job.status!r}")
    print("Training complete.")

    # ── Phase 3: Evaluation ───────────────────────────────────────────────────
    print("Submitting evaluation job …")
    eval_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="evaluate.py",
        compute_pool=COMPUTE_POOL,
        stage_name="@MODEL_STAGE",
        target_instances=1,
        env_vars={"MODEL_PATH": "best.pt", "DATA_DIR": "/data",
                  "RESULTS_DIR": "results/"},
        session=session,
    )
    eval_job.wait()
    if eval_job.status != "DONE":
        raise RuntimeError(f"Evaluation job failed with status {eval_job.status!r}")
    print("Evaluation complete.")

    # ── Verify stage ──────────────────────────────────────────────────────────
    contents = [row[0] for row in session.sql("LIST @MODEL_STAGE").collect()]
    return "Pipeline complete.\n" + "\n".join(f"  {p}" for p in contents)
