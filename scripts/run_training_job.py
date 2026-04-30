"""
Orchestrator for the DeepSet training pipeline.

Handler for the run_training_pipeline() Snowpark stored procedure.
The Snowpark session is injected automatically by the stored procedure framework.
"""
import json

from snowflake.ml.jobs import submit_from_stage

GPU_POOL = "DEEPSET_GPU_POOL"
CPU_POOL = "DEEPSET_CPU_POOL"
AUTOGLUON_CPU_POOL = "AUTOGLUON_CPU_POOL"
MODEL_STAGE = "@MODEL_STAGE"
SCRIPTS_STAGE = f"{MODEL_STAGE}/scripts/"
MLJOB_PAYLOAD_STAGE = "MLJOB_PAYLOAD_STAGE"
EVAL_RESULTS_STAGE = "@EVALUATION_RESULTS_STAGE"
KAGGLE_STAGE = "@META_DATASET_STAGE/kaggle/"
TRAIN_NUM_NODES = 4
BENCHMARK_PIP_REQUIREMENTS = [
    "openml",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "pandas",
    "scipy",
]
AUTOGLUON_PIP_REQUIREMENTS = BENCHMARK_PIP_REQUIREMENTS + ["autogluon.tabular[all]==1.0.0"]

BASELINE_METHODS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "KNN",
    "LinearRegression",
    "Ridge",
    "SVR",
    "MLP",
]
AUTOGLUON_METHOD = "AutoGluon"
MAX_BASELINE_CONCURRENCY = 3


def _wait_done(job, label, session):
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


def _submit_eval(session, label, compute_pool, env_vars):
    print(f"Submitting {label} ...")
    job_kwargs = {
        "source": SCRIPTS_STAGE,
        "entrypoint": "evaluate.py",
        "compute_pool": compute_pool,
        "stage_name": MLJOB_PAYLOAD_STAGE,
        "target_instances": 1,
        "env_vars": {
            "MODEL_PATH": "best.pt",
            "DATA_DIR": "/tmp/data",
            "RESULTS_DIR": "results/",
            "EVAL_RESULTS_STAGE": EVAL_RESULTS_STAGE,
            "HOME": "/tmp",
            **env_vars,
        },
        "session": session,
    }
    if env_vars.get("EVAL_MODE") == "benchmark":
        job_kwargs["external_access_integrations"] = ["BENCHMARK_EXTERNAL_ACCESS"]
        if env_vars.get("BENCHMARK_METHOD") == AUTOGLUON_METHOD:
            job_kwargs["pip_requirements"] = AUTOGLUON_PIP_REQUIREMENTS
        else:
            job_kwargs["pip_requirements"] = BENCHMARK_PIP_REQUIREMENTS
    return submit_from_stage(**job_kwargs)


def _list_stage(session, stage_path):
    try:
        return [row[0] for row in session.sql(f"LIST {stage_path}").collect()]
    except Exception as exc:
        return [f"{stage_path}: LIST failed: {exc}"]


def _stage_file_exists(session, stage_path, filename):
    rows = session.sql(f"LIST {stage_path}").collect()
    return any(str(row[0]).rstrip("/").endswith(f"/{filename}") for row in rows)


def _kaggle_secret_spec_overrides(secret_name):
    return {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "secrets": [
                        {
                            "snowflakeSecret": secret_name,
                            "secretKeyRef": "username",
                            "envVarName": "KAGGLE_USERNAME",
                        },
                        {
                            "snowflakeSecret": secret_name,
                            "secretKeyRef": "password",
                            "envVarName": "KAGGLE_KEY",
                        },
                    ],
                }
            ]
        }
    }


def run_kaggle_download(session) -> str:
    print("Submitting Kaggle benchmark download job ...")
    kaggle_secret_name = "KAGGLE_API_SECRET"
    job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="download_kaggle_to_stage.py",
        compute_pool=CPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=1,
        pip_requirements=["kaggle"],
        external_access_integrations=["BENCHMARK_EXTERNAL_ACCESS"],
        env_vars={
            "KAGGLE_STAGE": KAGGLE_STAGE,
            "KAGGLE_MAX_SAMPLES": "10000",
            "HOME": "/tmp",
        },
        spec_overrides=_kaggle_secret_spec_overrides(kaggle_secret_name),
        session=session,
    )
    _wait_done(job, "Kaggle benchmark download", session)

    kaggle_contents = _list_stage(session, KAGGLE_STAGE)
    return (
        "Kaggle benchmark download complete.\n\n"
        f"{KAGGLE_STAGE}:\n"
        + "\n".join(f"  {p}" for p in kaggle_contents)
    )


def run_pipeline(session) -> str:
    # Phase 1: HPO
    print("Submitting HPO job ...")
    hpo_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="hpo.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=4,
        env_vars={"HOME": "/tmp"},
        session=session,
    )
    _wait_done(hpo_job, "HPO", session)

    session.file.get(f"{MODEL_STAGE}/hpo/best_config.json", "/tmp/")
    with open("/tmp/best_config.json") as f:
        best_config = json.load(f)
    print("Best config:", best_config)

    # Phase 2: Full training
    print("Submitting training job ...")
    train_job = submit_from_stage(
        source=SCRIPTS_STAGE,
        entrypoint="train.py",
        compute_pool=GPU_POOL,
        stage_name=MLJOB_PAYLOAD_STAGE,
        target_instances=TRAIN_NUM_NODES,
        env_vars={
            "BEST_CONFIG": json.dumps(best_config),
            "TRAIN_NUM_NODES": str(TRAIN_NUM_NODES),
            "HOME": "/tmp",
        },
        session=session,
    )
    _wait_done(train_job, "Training", session)

    hpo_contents = _list_stage(session, f"{MODEL_STAGE}/hpo/")
    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    return (
        "Training pipeline complete.\n\n"
        "MODEL_STAGE hpo:\n"
        + "\n".join(f"  {p}" for p in hpo_contents)
        + "\n\nMODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
    )


def run_evaluation_pipeline(session) -> str:
    if not _stage_file_exists(session, f"{MODEL_STAGE}/checkpoints/", "best.pt"):
        raise FileNotFoundError(f"{MODEL_STAGE}/checkpoints/best.pt is required before evaluation.")

    synthetic_job = _submit_eval(
        session,
        "synthetic evaluation job",
        GPU_POOL,
        {"EVAL_MODE": "synthetic"},
    )
    deepset_job = _submit_eval(
        session,
        "DeepSetModel-MC benchmark job",
        GPU_POOL,
        {"EVAL_MODE": "benchmark", "BENCHMARK_METHOD": "DeepSetModel-MC"},
    )

    baseline_jobs = []
    for i in range(0, len(BASELINE_METHODS), MAX_BASELINE_CONCURRENCY):
        batch = BASELINE_METHODS[i:i + MAX_BASELINE_CONCURRENCY]
        running = [
            (
                method,
                _submit_eval(
                    session,
                    f"{method} benchmark job",
                    CPU_POOL,
                    {"EVAL_MODE": "benchmark", "BENCHMARK_METHOD": method},
                ),
            )
            for method in batch
        ]
        for method, job in running:
            _wait_done(job, f"{method} benchmark", session)
            baseline_jobs.append(job)

    autogluon_job = _submit_eval(
        session,
        "AutoGluon benchmark job",
        AUTOGLUON_CPU_POOL,
        {
            "EVAL_MODE": "benchmark",
            "BENCHMARK_METHOD": AUTOGLUON_METHOD,
            "AUTOGLUON_TIME_LIMIT": "300",
        },
    )

    _wait_done(synthetic_job, "Synthetic evaluation", session)
    _wait_done(deepset_job, "DeepSetModel-MC benchmark", session)
    _wait_done(autogluon_job, "AutoGluon benchmark", session)

    aggregate_job = _submit_eval(
        session,
        "benchmark aggregate job",
        CPU_POOL,
        {"EVAL_MODE": "aggregate"},
    )
    _wait_done(aggregate_job, "Benchmark aggregate", session)

    checkpoint_contents = _list_stage(session, f"{MODEL_STAGE}/checkpoints/")
    eval_contents = _list_stage(session, f"{EVAL_RESULTS_STAGE}/")
    return (
        "Evaluation pipeline complete.\n\n"
        "MODEL_STAGE checkpoints:\n"
        + "\n".join(f"  {p}" for p in checkpoint_contents)
        + "\n\nEVALUATION_RESULTS_STAGE:\n"
        + "\n".join(f"  {p}" for p in eval_contents)
    )
