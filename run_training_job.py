"""
Submit HPO + training jobs to Snowflake Container Runtime for ML.

Usage:
    set SNOWFLAKE_ACCOUNT=...  SNOWFLAKE_USER=...  SNOWFLAKE_PASSWORD=...
    python run_training_job.py
"""
import json, os

from snowflake.snowpark import Session
from snowflake.ml.jobs import MLJob   # Container Runtime ML Jobs API

COMPUTE_POOL   = "DEEPSET_GPU_POOL"
RUNTIME_IMAGE  = "snowflake/ml-runtime-gpu:latest"   # Snowflake-managed GPU image

connection_params = {
    "account":   os.environ["SNOWFLAKE_ACCOUNT"],
    "user":      os.environ["SNOWFLAKE_USER"],
    "password":  os.environ["SNOWFLAKE_PASSWORD"],
    "database":  "TABPFN_DB",
    "schema":    "TABPFN_SCHEMA",
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
}

with Session.builder.configs(connection_params).create() as session:

    # ── Phase 1: HPO ──────────────────────────────────────────────────────────
    print("Submitting HPO job …")
    hpo_job = MLJob.submit_job(
        session=session,
        entrypoint="hpo.py",
        compute_pool=COMPUTE_POOL,
        num_instances=2,          # 2 GPU_NV_S nodes → 2 parallel trials
        runtime_image=RUNTIME_IMAGE,
        upload_dir=".",           # uploads model.py, train.py, hpo.py, evaluate.py
    )
    hpo_job.wait()
    print("HPO complete.")

    # Read best config from stage
    session.file.get("@MODEL_STAGE/hpo/best_config.json", ".")
    with open("best_config.json") as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    # ── Phase 2: Full Training ────────────────────────────────────────────────
    print("Submitting training job …")
    train_job = MLJob.submit_job(
        session=session,
        entrypoint="train.py",
        compute_pool=COMPUTE_POOL,
        num_instances=2,          # DDP across 2 A10G GPUs
        runtime_image=RUNTIME_IMAGE,
        upload_dir=".",
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
        upload_dir=".",
        env_vars={"MODEL_PATH": "best.pt", "DATA_DIR": "/data",
                  "RESULTS_DIR": "results/"},
    )
    eval_job.wait()
    print("Evaluation complete.")

    # ── Verify stage ──────────────────────────────────────────────────────────
    print("\nStage contents:")
    for row in session.sql("LIST @MODEL_STAGE").collect():
        print(f"  {row[0]}")
