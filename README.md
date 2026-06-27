# TabPFN_DemandModel

A TabPFN-style tabular foundation model for demand forecasting, trained on
synthetic data-generating processes (DGPs) and evaluated on regression and
classification benchmarks. Runs on Snowflake SPCS with Ray for distributed
training and HPO.

---

## Repository Layout

```
TabPFN_DemandModel/
├── src/                        # Core library
│   ├── model/                  # Model, training, HPO, inference
│   │   ├── model.py            # DeepSetICLModel (MODEL4 architecture)
│   │   ├── train.py            # DDP training loop
│   │   ├── hpo.py              # Ray Tune HPO driver
│   │   ├── classification.py   # Classification head + targets
│   │   ├── deepset_inference.py
│   │   ├── constants.py        # Shared hyper-parameter constants
│   │   ├── task_routing.py     # Task-family → head routing
│   │   └── support_augmentation.py
│   ├── data_generation/        # Synthetic DGP libraries
│   │   ├── dgp_helpers.py      # Shared regime/profile helpers
│   │   ├── generate_dgp.py     # Linear meta-dataset generator (CLI)
│   │   └── generate_nonlinear_dgp.py
│   ├── dataset_index/          # Meta-dataset index builders
│   │   └── build_meta_*_dataset_index.py
│   ├── evaluation/             # Evaluation engine and baselines
│   │   ├── evaluate.py         # Main evaluation entry point
│   │   ├── evaluation_metrics.py
│   │   ├── baseline_models.py  # XGBoost/LightGBM/etc. wrappers
│   │   ├── autogluon_models.py
│   │   └── prepare_benchmark_datasets.py
│   ├── correctness/            # Invariance contracts and sanity checks
│   │   ├── sanity_checks.py
│   │   ├── permutation_contracts.py
│   │   └── tolerance_policy.py
│   ├── snowflake_io/           # Snowflake stage I/O helpers
│   │   └── snowflake_io.py
│   └── epoch_calibration/      # Epoch timing calibration (SPCS diagnostics)
│       ├── hpo_epoch_test.py
│       ├── train_epoch_test.py
│       └── runtime_probe.py
│
├── scripts/                    # Orchestration scripts
│   ├── jobs/                   # Snowflake stored-procedure handlers (MLJob entry points)
│   │   ├── run_training_job.py
│   │   ├── run_hpo_job.py
│   │   ├── run_model_training_job.py
│   │   ├── run_pretrain_job.py
│   │   ├── run_epoch_tests.py
│   │   ├── run_evaluation_test.py
│   │   └── download_kaggle_to_stage.py
│   ├── generation/             # Local synthetic data generation
│   │   ├── generate_linear_regression.py
│   │   ├── generate_linear_classification.py
│   │   ├── generate_nonlinear_regression.py
│   │   ├── generate_nonlinear_classification.py
│   │   └── run_generate_linear_numeric.py
│   ├── preparation/            # Dataset preparation and staging
│   │   ├── prepare_linear_{regression,classification}.py
│   │   ├── prepare_nonlinear_{regression,classification}.py
│   │   └── index_linear_eval_data.py
│   ├── evaluation/             # Evaluation runners and orchestrators
│   │   ├── evaluate_{linear,nonlinear}_{regression,classification}.py
│   │   └── run_{linear,nonlinear}_{regression,classification}_evaluation.py
│   ├── ray/                    # Ray / SPCS distributed infra
│   │   ├── autogluon_ray.py
│   │   ├── spcs_ray_coordinator.py
│   │   ├── spcs_ray_head.py
│   │   └── spcs_ray_worker.py
│   ├── probes/                 # Capacity and diagnostic probes
│   │   └── *_probe.py
│   ├── maintenance/            # Checkpoint migration, result transfers
│   │   ├── migrate_checkpoint.py
│   │   ├── download_results.py
│   │   ├── upload_results.py
│   │   └── download_kaggle_benchmark.py
│   └── ood_regression/         # OOD regression evaluation package
│
├── sql/                        # Snowflake SQL pipelines (stored procedures)
├── tests/                      # Pytest test suite (~71 files)
├── docs/                       # Architecture and design documentation
│   ├── MODEL5_LBACNP.md        # MODEL5-LBACNP design spec
│   ├── MODEL_REVISION.md       # Architecture revision history
│   ├── regression_evaluation.md
│   ├── cursor_dataset_generation.md
│   └── CUDA_GRAPHS.md
├── skills/                     # Claude agent skill definitions
│   ├── architecture/
│   ├── linear-synthetic-data/
│   ├── linear-evaluation-pipeline/
│   ├── nonlinear-synthetic-data/
│   ├── nonlinear-evaluation-pipeline/
│   └── machine-learning-pipeline/
├── docker/                     # AutoGluon SPCS container image
│   └── autogluon/
├── data/                       # Synthetic training data (local; not restructured)
│   ├── linear/regression/numeric/{train,val,test}/
│   ├── linear/classification/numeric/{train,val,test}/
│   ├── nonlinear/regression/numeric/{train,val,test}/
│   ├── nonlinear/classification/numeric/{train,val,test}/
│   ├── nonlinear_{regression,classification}/
│   ├── ood_regression/
│   └── synthetic_regression_prepared/
├── models/                     # Saved model checkpoints
│   ├── best_regression.pt / best_classification.pt / best_nonlinear_cls.pt
│   └── best_config.json
├── results/                    # Evaluation outputs (CSVs)
├── artifacts/                  # Sanity-check audit outputs
├── _bootstrap.py               # Recursive sys.path installer (local use only)
├── conftest.py                 # Pytest root config (auto-loads _bootstrap)
├── preload.py                  # Generates preload.sql for Snowflake deployment
├── preload.sql                 # Generated Snowflake setup script (run with snowsql)
├── requirements.txt            # Python dependencies
└── CLAUDE.md                   # Agent instructions and project conventions
```

---

## Dual-Environment Design

All `.py` files in `src/` and `scripts/` are **deployed to Snowflake** by flattening every
subdir into a single stage (`@MODEL_STAGE/scripts/`). Stored procedures resolve handlers by
bare filename. This means:

- **Filenames must stay globally unique** across all of `src/` and `scripts/`.
- Subdirectories are **organizational only** — they do not become Python packages.
- `preload.py` regenerates `preload.sql` with one `PUT` per code-bearing subdir. Run it
  after any file additions or moves: `python preload.py`

`_bootstrap.py` provides flat-import support locally by recursively adding every `src/` and
`scripts/` subdir to `sys.path`. It is **not** deployed to Snowflake.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Regenerate Snowflake deployment script
python preload.py

# Generate linear regression training data
python scripts/generation/generate_linear_regression.py --help
```

---

## Key Documentation

| Document | Location |
|---|---|
| Model architecture (current) | [`MODEL4.md`](MODEL4.md) |
| Agent instructions | [`CLAUDE.md`](CLAUDE.md) |
| Snowflake training guide | [`legacy/docs/Snowflake_Training.md`](legacy/docs/Snowflake_Training.md) |
| Evaluation pipeline | [`docs/regression_evaluation.md`](docs/regression_evaluation.md) |
| Project memory / decisions | [`MEMORY_TabPFN.md`](MEMORY_TabPFN.md) |
| ML pipeline runbook | [`.claude/commands/machine-learning-pipeline.md`](.claude/commands/machine-learning-pipeline.md) |
