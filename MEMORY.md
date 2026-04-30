# Project Memory

## Snowflake Stage Ownership

- `@META_DATASET_STAGE`: train/val/test synthetic parquet datasets and staged benchmark datasets.
- `@MODEL_STAGE`: Python scripts, HPO config, and model checkpoints.
- `@EVALUATION_RESULTS_STAGE`: all evaluation reports, per-method benchmark parts, and comparison CSVs.

Canonical benchmark output: `@EVALUATION_RESULTS_STAGE/model_comparison.csv`.

## Snowflake-Only Training Guardrails

- Never download or materialize `@META_DATASET_STAGE` to the local workstation.
- MLJobs may materialize staged parquet only inside Snowflake container-local `/tmp/data`.
- Use `auto_compress=False` for JSON, checkpoint, CSV, and NPZ stage uploads.
- Pass Snowflake secrets into MLJob containers through `spec_overrides`; do not fetch secret values inside scripts.
- Never use `MIN_NODES = 0` for Snowflake compute pools; use `MIN_NODES = 1` with suspend settings for cost control.
- `submit_from_stage.stage_name` is a bare payload stage name such as `MLJOB_PAYLOAD_STAGE`, not an `@STAGE` path.
- Snowflake MLJob secrets use `spec.containers[].secrets[]`, not Kubernetes-style `env.valueFrom`.
- Benchmark jobs must install their dependencies and fail loudly if any dependency is unavailable.
- Benchmark ranking broke once when `predict_autogluon()` was inserted inside `add_rank_columns()` before the rank loop; keep aggregation smoke tests.

## Kaggle Snowflake Download Troubleshooting

- Log line `Loaded Kaggle credentials from MLJob secret environment.` means Snowflake injected non-empty secret values at runtime.
- `401 Unauthorized` from `DownloadDataFiles` means Kaggle rejected the username/token; recreate `KAGGLE_API_SECRET` using the exact Kaggle username and API token.
- `403 Forbidden` from `DownloadDataFiles` means Kaggle authenticated the account but blocked competition file download; accept the competition rules while logged into the same Kaggle account used by the token.
- Accept rules for these Kaggle Playground Series competitions before downloading:
  - https://www.kaggle.com/competitions/playground-series-s3e3/rules
  - https://www.kaggle.com/competitions/playground-series-s3e5/rules
  - https://www.kaggle.com/competitions/playground-series-s3e9/rules
  - https://www.kaggle.com/competitions/playground-series-s3e22/rules
  - https://www.kaggle.com/competitions/playground-series-s3e26/rules
- After accepting rules, rerun:

```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
```
