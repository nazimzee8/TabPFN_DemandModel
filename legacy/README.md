# Legacy Archive

This directory contains the legacy OpenML/Kaggle benchmark pipeline that predates
the 8-suite synthetic evaluation architecture. These files are not referenced by
any of the 8 active SQL pipeline files.

Archived 2026-06-22 as part of the post-refactor consistency cleanup.

## Contents

- `src/evaluation/evaluate.py` — OpenML/Kaggle benchmark evaluator (loads best.pt)
- `src/evaluation/prepare_benchmark_datasets.py` — benchmark dataset preparation
- `scripts/jobs/run_evaluation_test.py` — benchmark eval job orchestrator
- `scripts/jobs/run_epoch_tests.py` — epoch calibration job
- `scripts/jobs/download_kaggle_to_stage.py` — Kaggle data download job
- `scripts/maintenance/download_kaggle_benchmark.py` — Kaggle benchmark utilities
- `scripts/maintenance/download_results.py` — results download helper
- `scripts/maintenance/upload_results.py` — results upload helper
- `scripts/ray/spcs_ray_head.py` — SPCS Ray head-node process wrapper
- `docs/Snowflake_Training.md` — legacy pipeline runbook
