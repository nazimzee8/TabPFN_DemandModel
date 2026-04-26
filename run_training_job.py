"""
Orchestrator for the DeepSet training pipeline.

Handler for the run_training_pipeline() Snowpark stored procedure.
The Snowpark session is injected automatically by the stored procedure framework.
"""
import json
import os

from snowflake.ml.jobs import MLJob   # Container Runtime ML Jobs API

COMPUTE_POOL  = "DEEPSET_GPU_POOL"
RUNTIME_IMAGE = "snowflake/ml-runtime-gpu:latest"   # Snowflake-managed GPU image
SCRIPTS_STAGE = "@MODEL_STAGE/scripts"
SCRIPTS_LOCAL = "/tmp/scripts"


def run_pipeline(session) -> str:
    # Download scripts from stage to the stored procedure's local execution environment
    # so MLJob can re-upload them into each job container.
    os.makedirs(SCRIPTS_LOCAL, exist_ok=True)
    session.file.get(SCRIPTS_STAGE, SCRIPTS_LOCAL)

    # ── Phase 1: HPO ──────────────────────────────────────────────────────────
    print("Submitting HPO job …")
    hpo_job = MLJob.submit_job(
        session=session,
        entrypoint="hpo.py",
        compute_pool=COMPUTE_POOL,
        num_instances=2,          # 2 GPU_NV_S nodes → 2 parallel trials
        runtime_image=RUNTIME_IMAGE,
        upload_dir=SCRIPTS_LOCAL,
    )
    hpo_job.wait()
    print("HPO complete.")

    # Read best config from stage
    session.file.get("@MODEL_STAGE/hpo/best_config.json", "/tmp/")
    with open("/tmp/best_config.json") as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    # ── Phase 2: Full Training ────────────────────────────────────────────────
    print("Submitting training job …")
    train_job = MLJob.submit_job(
        session=session,
        entrypoint="train.py",
        compute_pool=COMPUTE_POOL,
        num_instances=4,          # DDP across 4 A10G GPUs
        runtime_image=RUNTIME_IMAGE,
        upload_dir=SCRIPTS_LOCAL,
        env_vars={"BEST_CONFIG": json.dumps(best_config)},
    )
    train_job.wait()
    print("Training complete.")

    # ── Phase 3: Evaluation ───────────────────────────────────────────────────
    print("Submitting evaluation job …")
    eval_job = MLJob.submit_job(
        session=session,
        entrypoint="evaluate.py",
        compute_pool=COMPUTE_POOL,
        num_instances=1,
        runtime_image=RUNTIME_IMAGE,
        upload_dir=SCRIPTS_LOCAL,
        env_vars={"MODEL_PATH": "best.pt", "DATA_DIR": "/data",
                  "RESULTS_DIR": "results/"},
    )
    eval_job.wait()
    print("Evaluation complete.")

    # ── Verify stage ──────────────────────────────────────────────────────────
    contents = [row[0] for row in session.sql("LIST @MODEL_STAGE").collect()]
    return "Pipeline complete.\n" + "\n".join(f"  {p}" for p in contents)
