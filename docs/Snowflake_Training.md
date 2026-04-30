# Snowflake Training — DeepSet TabPFN

Describes how to run the DeepSet training pipeline inside a Snowflake environment
using Snowpark Container Services (SPCS) and Snowflake Model Registry.

---

## Snowflake Environment Construction

Ordered setup: from local machine to a running SPCS training job with model checkpoint
written back to Snowflake.

**Steps:**

1. **Create database, schema, and stages** with `run_training_job.sql`: `@META_DATASET_STAGE`, `@MODEL_STAGE`, `@EVALUATION_RESULTS_STAGE`, and `@MLJOB_PAYLOAD_STAGE`.
2. **Create compute pools** with `run_training_job.sql`: `DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, and `AUTOGLUON_CPU_POOL`. Verify the pools reach `ACTIVE` state before submitting jobs.
3. **Create network rules, `KAGGLE_API_SECRET`, and `BENCHMARK_EXTERNAL_ACCESS`**. The committed SQL uses placeholders only; never commit real Kaggle credentials.
4. **Upload scripts and Parquet data** with SnowSQL `PUT`: scripts to `@MODEL_STAGE/scripts/` and local synthetic datasets to `@META_DATASET_STAGE/{train,val,test}/`.
5. **Call `download_kaggle_to_stage()`**. This submits an MLJob that receives Kaggle credentials at runtime and writes benchmark `.npz` files to `@META_DATASET_STAGE/kaggle/`.
6. **Call `run_training_pipeline()`**. HPO writes `@MODEL_STAGE/hpo/best_config.json`; training consumes that JSON through `BEST_CONFIG` and writes `@MODEL_STAGE/checkpoints/best.pt`.
7. **Verify `best.pt`** with `LIST @MODEL_STAGE/checkpoints/;`.
8. **Call `run_evaluation_pipeline()`**. Evaluation reads `best.pt`, runs synthetic, DeepSet, baseline, AutoGluon, and aggregate jobs, then writes CSV outputs to `@EVALUATION_RESULTS_STAGE`.
9. **Download results locally** from the client with SnowSQL `GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';`.

## Current Snowflake Guardrails

- `run_training_job.sql` creates `@MLJOB_PAYLOAD_STAGE` in addition to
  `@META_DATASET_STAGE`, `@MODEL_STAGE`, and `@EVALUATION_RESULTS_STAGE`.
- CPU compute pools must use one minimum node. Use `AUTO_SUSPEND_SECS` and
  `INITIALLY_SUSPENDED` for cost control instead of a zero-node minimum.
- `submit_from_stage(source="@MODEL_STAGE/scripts/", stage_name="MLJOB_PAYLOAD_STAGE")`
  uses a staged script source and a bare payload stage name. Do not pass
  `stage_name="@MODEL_STAGE"`.
- MLJob secrets use Snowflake service spec syntax:
  `spec.containers[].secrets[]` with `snowflakeSecret`, `secretKeyRef`, and
  `envVarName`. Do not use Kubernetes-style `env.valueFrom`.
- Kaggle credentials live only in the Snowflake `KAGGLE_API_SECRET`. They are
  injected into the Kaggle download MLJob at runtime as `KAGGLE_USERNAME` and
  `KAGGLE_KEY`; they are not stored in the repo, staged as files, printed, or baked
  into a container image.
- AutoGluon is a full benchmark method. Its job installs the shared benchmark
  dependencies plus `autogluon.tabular[all]==1.0.0`, sets
  `BENCHMARK_METHOD=AutoGluon`, `AUTOGLUON_TIME_LIMIT=300`, and writes
  `@EVALUATION_RESULTS_STAGE/benchmark_parts/AutoGluon_detailed.csv`.

Recommended smoke order: compile the Python scripts, create Snowflake objects with
`run_training_job.sql`, run `CALL download_kaggle_to_stage()`, run
`CALL run_training_pipeline()`, verify `LIST @MODEL_STAGE/hpo/` and
`LIST @MODEL_STAGE/checkpoints/` include `best.pt`, run a one-dataset AutoGluon smoke
where supported, then run `CALL run_evaluation_pipeline()` and verify
`AutoGluon_detailed.csv`, `model_comparison.csv`, and
`model_comparison_summary.csv`.

## Secure Kaggle Token Handling

The Snowflake `SECRET` is the source of truth for Kaggle credentials. If a Kaggle
token has been pasted into chat, logs, SQL files, stage files, or Snowflake query
history where it should not be visible, revoke it in Kaggle and create a new API
token before continuing.

### Secret lifecycle

`KAGGLE_API_SECRET` is a Snowflake schema object, usually
`TABPFN_DB.TABPFN_SCHEMA.KAGGLE_API_SECRET`. Enter the real Kaggle username and API
token once in SnowSQL or Snowsight when you create the `SECRET`; do not pass those
values to `CALL download_kaggle_to_stage()`.

`CALL download_kaggle_to_stage()` only references the existing secret name; it does
not prompt for or receive the token manually. In `run_training_job.py`, the Kaggle
download MLJob references `KAGGLE_API_SECRET` and maps the secret fields to
container environment variables through `spec_overrides`: `USERNAME` becomes
`KAGGLE_USERNAME`, and `PASSWORD` becomes `KAGGLE_KEY`. Snowflake injects those
values into the MLJob container at job startup.

Use placeholder values in committed SQL and docs:

```sql
CREATE OR REPLACE SECRET KAGGLE_API_SECRET
  TYPE = PASSWORD
  USERNAME = '<kaggle_username>'
  PASSWORD = '<kaggle_api_key>';
```

Grant only the job-submitting role the ability to read the secret:

```sql
GRANT READ ON SECRET KAGGLE_API_SECRET TO ROLE <job_submitter_role>;
GRANT USAGE ON INTEGRATION BENCHMARK_EXTERNAL_ACCESS TO ROLE <job_submitter_role>;
```

`BENCHMARK_EXTERNAL_ACCESS` must list `KAGGLE_API_SECRET` in
`ALLOWED_AUTHENTICATION_SECRETS`. `run_training_job.py` then injects the secret into
the Kaggle download MLJob through `spec_overrides` using
`spec.containers[].secrets[]`; the container sees only runtime environment variables
`KAGGLE_USERNAME` and `KAGGLE_KEY`. Do not commit, print, stage, or bake Kaggle
credentials into images or containers.

Verify the existing secret and external access integration before running the
download job:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

DESC SECRET KAGGLE_API_SECRET;
DESC INTEGRATION BENCHMARK_EXTERNAL_ACCESS;
```

`DESC SECRET` may show the username and metadata, but it should not show the Kaggle
token. If the real token was pasted into source files, chat, screenshots, shared
logs, or a shared worksheet, revoke and rotate the Kaggle API token before using
the pipeline again.

## Snowflake Environment Setup Checklist

Use this checklist before running `CALL run_training_pipeline()` and
`CALL run_evaluation_pipeline()`. It covers synthetic data staging, Snowflake-native
Kaggle benchmark ingestion, benchmark external network access,
`EVALUATION_RESULTS_STAGE`, CPU compute pools, and local result retrieval.

### 1. Connect with the Snowflake account identifier

Use the account identifier from the Snowflake account URL or Snowsight connection
details:

```bash
snowsql -a <account_identifier> -u <username>
```

`<account_identifier>` is the value from
`<account_identifier>.snowflakecomputing.com`. For `YOUR_ACCOUNT_NAME` in scripts or
environment variables, use this SnowSQL/connector account identifier, not a database,
schema, warehouse, or username. Snowflake's preferred client identifier format is
`organization-account_name`; the legacy locator-region form still works for older
accounts. Snowsight's account URL is the safest copy source. After connecting, this
query helps confirm the organization and account currently in session:

```sql
SELECT CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT();
```

Reference: Snowflake [Account identifiers](https://docs.snowflake.com/en/user-guide/admin-account-identifier.html).

### 2. Prepare benchmark inputs

Synthetic meta-datasets are staged as Parquet files:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE;
PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE;
```

For Kaggle benchmark data, the preferred production path is Snowflake-native:
store the Kaggle API token in a Snowflake `SECRET`, run the one-off
`download_kaggle_to_stage()` setup procedure, and let the MLJob upload `.npz`
files directly to `@META_DATASET_STAGE/kaggle/`.

Local `kaggle.json` is only needed when generating `.npz` files on your workstation
for development or fallback testing:

```bash
# Windows path: C:/Users/<you>/.kaggle/kaggle.json
# POSIX-style path used by Kaggle tooling: ~/.kaggle/kaggle.json
python download_kaggle_benchmark.py --out_dir ./data/kaggle
```

Then upload the locally downloaded benchmark files to the dataset stage:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file://C:/Documents/TabPFN_DemandModel/data/kaggle/*.npz @META_DATASET_STAGE/kaggle/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

The Snowflake-native Kaggle path replaces this local upload for normal operations.
OpenML remains fetched from inside Snowflake by benchmark MLJobs at runtime; do not
pre-stage OpenML files unless you intentionally change the evaluation mode.

### 3. Enable external network access and Kaggle secret

OpenML benchmark jobs need Snowflake external access so `evaluate.py` can reach
OpenML from the container runtime. The Kaggle setup job also needs external access
to Kaggle/download hosts and a Snowflake `SECRET` holding the API credentials.
Create the network rules and secret in `TABPFN_SCHEMA`:

```sql
USE ROLE SYSADMIN;
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

CREATE OR REPLACE NETWORK RULE openml_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'www.openml.org',
    'openml.org',
    'api.openml.org',
    'pypi.org',
    'files.pythonhosted.org'
  );

CREATE OR REPLACE NETWORK RULE kaggle_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'www.kaggle.com',
    'kaggle.com',
    'storage.googleapis.com',
    'pypi.org',
    'files.pythonhosted.org'
  );

CREATE OR REPLACE SECRET KAGGLE_API_SECRET
  TYPE = PASSWORD
  USERNAME = '<kaggle_username>'
  PASSWORD = '<kaggle_api_key>';
```

Create the external access integration at the account level:

```sql
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION benchmark_external_access
  ALLOWED_NETWORK_RULES = (openml_network_rule, kaggle_network_rule)
  ALLOWED_AUTHENTICATION_SECRETS = (KAGGLE_API_SECRET)
  ENABLED = TRUE;
```

The admin/security role needs privileges to create network rules in the schema and to
create external access integrations at the account level. If a non-admin role submits
jobs, grant it least-privilege access to the integration and secret:

```sql
GRANT READ ON SECRET KAGGLE_API_SECRET TO ROLE <job_submitter_role>;
GRANT USAGE ON INTEGRATION benchmark_external_access TO ROLE <job_submitter_role>;
```

`run_training_job.py` submits the Kaggle setup job with
`external_access_integrations=["BENCHMARK_EXTERNAL_ACCESS"]` and
`pip_requirements=["kaggle"]`; Kaggle credentials are exposed to the container as
`KAGGLE_USERNAME` and `KAGGLE_KEY` through MLJob `spec_overrides`, not read by the
script through `snowflake.snowpark.secrets`. Benchmark evaluation jobs also name
`BENCHMARK_EXTERNAL_ACCESS` for OpenML fetches and install `openml`, `scikit-learn`,
`xgboost`, `lightgbm`, `catboost`, `pandas`, and `scipy` with `pip_requirements`.
Creating the environment objects only makes the route available; jobs will not use
it unless the MLJob configuration names the integration and dependencies.

After uploading scripts, run the setup procedure and verify staged `.npz` files:

```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
```

References: Snowflake [`CREATE NETWORK RULE`](https://docs.snowflake.com/en/sql-reference/sql/create-network-rule) and [`CREATE EXTERNAL ACCESS INTEGRATION`](https://docs.snowflake.com/en/sql-reference/sql/create-external-access-integration).

### 4. Confirm stages and compute pools

All project stages must exist before jobs are submitted. `MLJOB_PAYLOAD_STAGE` is the
bare stage name passed to `submit_from_stage(stage_name=...)`; results are written
to `@EVALUATION_RESULTS_STAGE` before any local download:

```sql
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

Resource split:

- GPU (`DEEPSET_GPU_POOL`): HPO, 4-node DDP training, synthetic DeepSet evaluation, and DeepSet benchmark.
- CPU_X64_XS (`DEEPSET_CPU_POOL`): Kaggle download, classic baseline benchmark jobs, and aggregation.
- CPU_X64_M (`AUTOGLUON_CPU_POOL`): AutoGluon benchmark.

```sql
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 4
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_SUSPEND_SECS = 300;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 3
  INSTANCE_FAMILY = CPU_X64_XS
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;
```

### 5. Download benchmark outputs

Use SnowSQL for a direct stage check and one-off download:

```sql
LIST @EVALUATION_RESULTS_STAGE/;
GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
```

Or use the Python helper. Set the connector environment variables first:

```bash
set SNOWFLAKE_ACCOUNT=<account_identifier>
set SNOWFLAKE_USER=<username>
set SNOWFLAKE_PASSWORD=<password>
set SNOWFLAKE_WAREHOUSE=COMPUTE_WH
python download_results.py
```

`SNOWFLAKE_WAREHOUSE` is optional if `COMPUTE_WH` is correct for your account.

**Data and artifact flow:**

```
Local machine
  └── *.parquet ──PUT──→ @META_DATASET_STAGE
  └── *.py ──PUT──→ @MODEL_STAGE/scripts/ ──vol mount──→ /opt/app/

Snowsight / SnowSQL
  └── CALL run_training_pipeline() ──→ Snowpark stored procedure
        └── run_training_job.run_pipeline(session) submits:

Container Runtime — Phase 1: HPO (2 nodes)
  └── hpo.py → @MODEL_STAGE/hpo/best_config.json

Container Runtime — Phase 2: Training (4 nodes, DDP)
  ├── materialize_meta_dataset_stage() copies staged parquet to /tmp/data/
  ├── DataLoader (4 workers, prefetch_factor=2) reads /tmp/data/train/*.parquet
  ├── trains DeepSet (phi, rho, psi + 4 equivariant scalars)
  │     BF16 autocast + GradScaler, batched forward over all m test rows
  │     torch.compile(mode="reduce-overhead") fuses GPU kernels
  └── writes best.pt → @MODEL_STAGE/checkpoints/best.pt

Snowsight / SnowSQL
  └── CALL run_evaluation_pipeline() ──→ Snowpark stored procedure
        └── run_training_job.run_evaluation_pipeline(session) submits:

Container Runtime — Evaluation
  └── evaluate.py → @EVALUATION_RESULTS_STAGE/synthetic/*.csv
                  → @EVALUATION_RESULTS_STAGE/benchmark_parts/*.csv
                  → @EVALUATION_RESULTS_STAGE/model_comparison.csv

Model Registry
  └── DEEPSET_TABPFN_V1!PREDICT() ← loads from @MODEL_STAGE/checkpoints/best.pt
```

---

## Data Storage

Variable-shape datasets (pickle) cannot be stored in flat Snowflake tables efficiently.
Use an **internal named stage** with Parquet files:

```
@META_DATASET_STAGE/
  train/   ← 800 parquet files (one per meta-task)
  val/     ← 100 parquet files
  test/    ← 100 parquet files
```

Each Parquet file contains: `X_train`, `y_train`, `X_test`, `betaX_test`,
`n`, `p`, `prior_regime`, stored as nested arrays via VARIANT or PyArrow list types.

### Generating datasets locally

Prerequisites: `pip install numpy pyarrow`

```bash
cd C:/Documents/TabPFN_DemandModel
python generate_dgp.py --n_datasets 1000 --out_dir data/
# Writes: data/train/ (800 files), data/val/ (100), data/test/ (100)
```

### Uploading to Snowflake via SnowSQL

> **Important**: `PUT` is a client-side command that streams files from your local disk
> directly to Snowflake's internal stage. It is **not supported** in the Snowsight web UI
> ("Unsupported feature 'unsupported_requested_format:snowflake'"). Use SnowSQL instead.

Install SnowSQL: https://docs.snowflake.com/en/user-guide/snowsql-install-config

Connect:

```bash
snowsql -a <account_identifier> -u <username>
```

Use the same `<account_identifier>` described in the setup checklist. It is the
SnowSQL/connector account identifier from `<account_identifier>.snowflakecomputing.com`.

Then run the three `PUT` commands inside SnowSQL:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE;
PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE;
PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE;
```

Verify the upload:

```sql
LIST @META_DATASET_STAGE/train/;
LIST @META_DATASET_STAGE/val/;
LIST @META_DATASET_STAGE/test/;
```

### Uploading Python scripts via SnowSQL

Run once, and re-run whenever any script changes:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file://C:/Documents/TabPFN_DemandModel/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

Verify:

```sql
LIST @MODEL_STAGE/scripts/;
```

---

## Prerequisite SQL Objects

Run these once in Snowsight (or SnowSQL) before any other step.

```sql
-- Dedicated database and schema that own all project objects
-- (stages, compute pools, procedures, and model registry entries).
CREATE DATABASE IF NOT EXISTS TABPFN_DB;
USE DATABASE TABPFN_DB;
CREATE SCHEMA IF NOT EXISTS TABPFN_SCHEMA;
USE SCHEMA TABPFN_SCHEMA;

-- Internal stage for raw Parquet meta-datasets (train / val / test splits).
-- MLJobs materialize this stage into ephemeral container-local /tmp/data.
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Internal stage for scripts, HPO config, and model checkpoints.
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Internal stage for all evaluation CSVs and benchmark comparison outputs.
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Internal stage used by submit_from_stage(stage_name=...) for MLJob payloads.
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

---

## Compute: Container Runtime for ML

### 1. Compute Pool

```sql
-- GPU_NV_S: 1 A10G per node. MAX_NODES=4 supports 4-node DDP training.
-- This can exceed the earlier $5/hr budget cap when all GPU nodes are active.
-- CPU_X64_XS handles baseline benchmark jobs with bounded concurrency.
-- CPU_X64_M handles the separate AutoGluon stacked-ensemble benchmark.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY; drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 4
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_SUSPEND_SECS = 300;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 3
  INSTANCE_FAMILY = CPU_X64_XS
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;
```

### 2. Job Submission (MLJob)

`run_training_job.py` is deployed as Snowpark Python stored procedures. The training
procedure submits HPO and training MLJobs using scripts already on
`@MODEL_STAGE/scripts/`; the evaluation procedure separately submits synthetic,
DeepSet benchmark, baseline benchmark, AutoGluon benchmark, and aggregate MLJobs. No
local Python environment is needed, and no dataset stage contents are materialized
outside Snowflake.

#### What is an MLJob container?

An MLJob container is a short-lived compute environment that Snowflake starts on
one or more nodes in your GPU compute pool to run a single Python script. When
`submit_from_stage()` is called, Snowflake pulls the managed ML runtime image onto
the requested nodes, runs your entrypoint (e.g. `train.py`), writes outputs to the
stage, then shuts the container down. The `stage_name` argument is the MLJob payload
stage for scripts/artifacts; it is not a dataset mount. Training scripts explicitly
materialize `@META_DATASET_STAGE/{train,val,test}/` into container-local `/tmp/data`
with Snowpark `session.file.get()`. PyTorch, Ray, and `snowflake-ml-python` are
pre-installed — no Docker build or image management is required.

Create the procedures only after the scripts are staged. Otherwise `IMPORTS =
('@MODEL_STAGE/scripts/run_training_job.py')` can fail because Snowflake cannot
resolve the staged file:

```sql
PUT file://C:/Documents/TabPFN_DemandModel/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
LIST @MODEL_STAGE/scripts/;
```

Create the procedures, and re-run these statements after uploading an updated
`run_training_job.py`:

```sql
CREATE OR REPLACE PROCEDURE download_kaggle_to_stage()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_kaggle_download';

CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

CREATE OR REPLACE PROCEDURE run_evaluation_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_evaluation_pipeline';
```

Then stage Kaggle data once, run training, verify the checkpoint, and run evaluation:

```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
CALL run_training_pipeline();
LIST @MODEL_STAGE/hpo/;
LIST @MODEL_STAGE/checkpoints/;
CALL run_evaluation_pipeline();
```

For a setup smoke check before the full run:

```sql
DESC SECRET KAGGLE_API_SECRET;
DESC INTEGRATION BENCHMARK_EXTERNAL_ACCESS;
SHOW COMPUTE POOLS LIKE 'DEEPSET_GPU_POOL';
SHOW COMPUTE POOLS LIKE 'DEEPSET_CPU_POOL';
SHOW COMPUTE POOLS LIKE 'AUTOGLUON_CPU_POOL';
LIST @MODEL_STAGE/scripts/;
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
```

For full acceptance:

```sql
CALL run_training_pipeline();
LIST @MODEL_STAGE/checkpoints/;
CALL run_evaluation_pipeline();
LIST @EVALUATION_RESULTS_STAGE/;
GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
```

Kaggle download uses `compute_pool="DEEPSET_CPU_POOL"` and writes `.npz` files to
`@META_DATASET_STAGE/kaggle/`. HPO, training, synthetic evaluation, and
DeepSetModel-MC benchmark use `compute_pool="DEEPSET_GPU_POOL"`; baseline benchmark
jobs use `compute_pool="DEEPSET_CPU_POOL"` with bounded concurrency. AutoGluon runs
as a separate stacked-ensemble benchmark job on `AUTOGLUON_CPU_POOL` with
`autogluon.tabular[all]==1.0.0`, `presets="best_quality"`,
`AUTOGLUON_TIME_LIMIT=300`, `num_cpus=1`, `num_gpus=0`, and temporary model
artifacts under `/tmp` cleaned after each fit:

| Phase | Entrypoint | Instances | Output |
|---|---|---|---|
| Kaggle download | `download_kaggle_to_stage.py` | 1 CPU | `@META_DATASET_STAGE/kaggle/*.npz` |
| HPO | `hpo.py` | 2 | `@MODEL_STAGE/hpo/best_config.json` |
| Training | `train.py` | 4 (DDP) | `@MODEL_STAGE/checkpoints/best.pt` |
| Synthetic evaluation | `evaluate.py` | 1 GPU | `@EVALUATION_RESULTS_STAGE/synthetic/test_report.csv` and `mc_report.csv` |
| DeepSet benchmark | `evaluate.py` | 1 GPU | `@EVALUATION_RESULTS_STAGE/benchmark_parts/DeepSetModel-MC_detailed.csv` |
| Baseline benchmarks | `evaluate.py` | bounded CPU jobs | `@EVALUATION_RESULTS_STAGE/benchmark_parts/<method>_detailed.csv` |
| AutoGluon benchmark | `evaluate.py` | 1 CPU_X64_M | `@EVALUATION_RESULTS_STAGE/benchmark_parts/AutoGluon_detailed.csv` |
| Aggregate comparison | `evaluate.py` | 1 CPU | `@EVALUATION_RESULTS_STAGE/model_comparison.csv` |

### 3. Checkpoint Output

Write best checkpoint back to the model stage on improvement:

```python
session.file.put(
    local_file_name="best.pt",
    stage_location="@MODEL_STAGE/checkpoints/",
    overwrite=True,
    auto_compress=False,
)
```

Use `auto_compress=False` for runtime artifacts whose exact names are consumed by
procedures or later jobs: `best_config.json` for training, `best.pt` for evaluation,
evaluation CSVs for aggregation/download, and Kaggle `.npz` files for benchmarks.

---

## Distributed Training & Hyperparameter Optimization

### Container Runtime for ML

Snowflake Container Runtime for ML provides a managed, GPU-enabled image with PyTorch,
Ray, and `snowflake-ml-python` pre-installed.

- No container image build or push needed — scripts are read directly from
  `@MODEL_STAGE/scripts/` via `submit_from_stage()`.
- Runtime image: `snowflake/ml-runtime-gpu:latest` (Snowflake-managed).
- Jobs submitted from the stored procedure via `run_training_job.py`.

> Scripts are referenced directly from the stage via `source=` in
> `submit_from_stage()`. No Docker image is required or maintained.

### Distributed Training — PyTorchDistributor

- Class: `snowflake.ml.modeling.distributors.pytorch.PyTorchDistributor`
- Manages Ray cluster setup, DDP process group initialization, and result collection
  internally — no manual `torchrun` or rank-environment setup required.
- `PyTorchScalingConfig(num_nodes=4, num_workers_per_node=1, ...)` maps to
  4× GPU_NV_S nodes (one A10G each); `run_training_job.py` submits training with
  matching `target_instances=4`.
- `get_context()` inside `train_fn` provides `local_rank`, `rank`, and `world_size`.
- `DistributedSampler` with `rank` / `world_size` splits the 800 training tasks
  across 4 GPU processes (200 tasks/GPU/epoch); future train split sizes must divide
  evenly by `world_size`.
- Validation uses a no-padding rank slice, reduces `(sum_loss, total_count)` across
  ranks, and computes exact weighted global MSE before the early-stop check;
  `dist.broadcast(stop, src=0)` propagates the stop signal.

### Hyperparameter Optimization — Tuner + BayesOpt

- Class: `snowflake.ml.modeling.tune.Tuner`
- Algorithm: `BayesOpt` (Gaussian-process surrogate; minimizes trials needed vs.
  grid or random search).
- Search space: `lr`, `weight_decay`, `d_phi`, `d_rho`, `dropout`, `pool`.
- 20 trials, 30-epoch runs each; best config written to
  `@MODEL_STAGE/hpo/best_config.json` on completion.
- To use a simpler baseline, swap `BayesOpt()` → `RandomSearch()` in `hpo.py`.

### Compute Pool & Cost

| Configuration | Credits/node/hr | Nodes | Total cost/hr |
|---|---|---|---|
| GPU_NV_S (this design) | 0.57 | 2 | ~$2.28–3.42 |
| previous single-node | 2.68 | 1 | ~$5.36–8.04 |

- 2-node GPU_NV_S pool: 1.14 cr/hr ≈ **$2.28–3.42/hr** (Standard/Enterprise) — ~80%
  cheaper than the previous single-node configuration.
- Pool suspends when idle; no charges in `SUSPENDED` state.

### Estimated End-to-End Cost

| Phase | Nodes | Cost/hr | Duration | Total |
|---|---|---|---|---|
| HPO (20 trials × 30 epochs) | 2 × GPU_NV_S | ~$2.28–3.42 | ~40–60 min | ~$1.52–3.42 |
| Full training (DDP) | 4 × GPU_NV_S | ~$4.56–6.84 | ~8–13 min | ~$0.61–1.48 |
| Evaluation | 1 × GPU_NV_S | ~$1.14–1.71 | ~5–10 min | ~$0.10–0.29 |
| **Total** | | | **~53–83 min** | **~$2.23–5.19** |

---

## Training Loop Adaptation

Inside the container, `train.py` follows this loop:

```python
import os, glob
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DeepSetModel

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "/tmp/data"
PATIENCE   = 10
MAX_EPOCHS = 200
USE_AMP    = DEVICE == "cuda"

# --- Dataset and DataLoader ---
class ParquetMetaDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path  = self.files[idx]
        table = pq.read_table(path)
        d     = table.to_pydict()
        return (
            torch.tensor(np.array(d["X_train"][0]),    dtype=torch.float32),
            torch.tensor(np.array(d["y_train"][0]),    dtype=torch.float32),
            torch.tensor(np.array(d["X_test"][0]),     dtype=torch.float32),
            torch.tensor(np.array(d["betaX_test"][0]), dtype=torch.float32),
        )

def identity_collate(batch):
    return batch[0]   # batch_size=1; skip default list-wrapping

materialize_meta_dataset_stage(DATA_DIR, splits=("train", "val"))
train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.parquet")))
val_files   = sorted(glob.glob(os.path.join(DATA_DIR, "val",   "*.parquet")))

train_loader = DataLoader(
    ParquetMetaDataset(train_files), batch_size=1, shuffle=True,
    num_workers=4, prefetch_factor=2, pin_memory=USE_AMP, collate_fn=identity_collate,
)
val_loader = DataLoader(
    ParquetMetaDataset(val_files), batch_size=1, shuffle=False,
    num_workers=4, prefetch_factor=2, pin_memory=USE_AMP, collate_fn=identity_collate,
)

# --- Model, compiler, optimizer, scaler ---
from model import DeepSetModel, ModelConfig

cfg = ModelConfig(
    d_phi=128, d_rho=256, pool="pna", n_heads=4,
    n_sab_feat=1, n_sab_samp=1,
    norm_feat=True, norm_target=True, dropout=0.1,
)
model     = DeepSetModel(cfg=cfg).to(DEVICE)
model     = torch.compile(model, mode="reduce-overhead")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# --- Training loop ---
best_val_mse   = float("inf")
patience_count = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    for X_train, y_train, X_test, betaX_test in train_loader:
        X_train    = X_train.to(DEVICE)
        y_train    = y_train.to(DEVICE)
        X_test     = X_test.to(DEVICE)
        betaX_test = betaX_test.to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
            y_hat = model(X_train, y_train, X_test)          # batched: (m,)
            loss  = nn.functional.mse_loss(y_hat, betaX_test)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # --- Validation ---
    model.eval()
    val_mses = []
    with torch.no_grad():
        for X_train, y_train, X_test, betaX_test in val_loader:
            X_train    = X_train.to(DEVICE)
            y_train    = y_train.to(DEVICE)
            X_test     = X_test.to(DEVICE)
            betaX_test = betaX_test.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
                y_hat = model(X_train, y_train, X_test)
            val_mses.append(nn.functional.mse_loss(y_hat.float(), betaX_test.float()).item())
    val_mse = float(np.mean(val_mses))
    print(f"Epoch {epoch:3d}  val_mse={val_mse:.4f}")

    if val_mse < best_val_mse:
        best_val_mse   = val_mse
        patience_count = 0
        # torch.compile wraps the module; _orig_mod holds the original state dict
        ckpt_module = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({"state_dict": ckpt_module.state_dict(), "cfg": ckpt_module.cfg}, "best.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

# Upload best checkpoint to Snowflake stage
# (run via Snowpark session inside container)
```

---

## Performance Optimizations

The original training loop ran row-by-row: for each meta-dataset, `forward()` was called
once per test row (`X_test[k]`), producing ~3.6 M serial GPU kernel launches across 800
training files × ~4,500 test rows. This serialized inference saturated the CPU and left
GPU Tensor Cores idle.

### 1. Batched Forward Pass

`forward()` accepts `x_test` of shape `(m, p)` (all test rows at once) and returns
predictions of shape `(m,)`. A single GPU kernel dispatch replaces the per-row loop.

### 2. BF16 Autocast

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward and
backward passes. BF16 halves tensor bandwidth and fully activates Tensor Cores on A10G,
roughly doubling throughput. `GradScaler` is included for safety with larger gradient
magnitudes.

### 3. DataLoader with Background Workers

```python
DataLoader(..., num_workers=4, prefetch_factor=2, pin_memory=True)
```

Four OS processes read and decode Parquet files while the GPU trains on the previous
batch, removing I/O from the critical path. `pin_memory=True` enables DMA transfers
between page-locked host RAM and GPU memory.

### 4. `torch.compile`

```python
model = torch.compile(model, mode="reduce-overhead")
```

`torch.compile` traces the graph and fuses repeated small kernels (linear + ReLU,
residual add) into single fused kernels, reducing Python interpreter overhead and GPU
kernel launch latency. The compiled model is saved via `model._orig_mod` to avoid
`torch.compile` wrapper artefacts in the checkpoint.

### 5. GPU_NV_S Compute Pool

`GPU_NV_S` provides 1× A10G GPU per node and ~12 vCPUs. 4 DataLoader worker processes
fit comfortably within available host RAM, leaving headroom for the main training
process. 4 nodes are used for DDP training; 2 nodes for parallel HPO trials.

### Cost Comparison

| Configuration | Estimated wall-clock | Notes |
|---|---|---|
| GPU_NV_S, row-by-row, FP32 | ~4 hours | Original |
| GPU_NV_S × 2, batched, BF16, DDP, compile | ~15–25 minutes | Previous optimized |
| GPU_NV_S × 4, batched, BF16, DDP, compile | ~8–13 minutes | Current (4-node DDP) |

Estimates assume 800 training files × 200 epochs with early stopping at epoch ~100.

---

## Architecture Design

### Latent Space Dimensions

DeepSet universality (Zaheer et al. 2017) requires:

- **`d_phi >= p`** (number of features): `phi` maps each `(y_i, x_ij, x_test_j)` triple
  into R^{d_phi}. For the feature-level aggregation to represent all possible set
  functions over p feature vectors, d_phi must span at least p dimensions.
- **`d_rho >= n`** (number of training samples): `rho` maps each aggregated sample
  embedding into R^{d_rho}. For the sample-level aggregation to represent all possible
  sample-set functions, d_rho must span at least n dimensions.

Current defaults: `d_phi=128`, `d_rho=256`. If your tasks have more than 128 features
or more than 256 training samples, increase these accordingly.

### Phi Injectivity

`phi: R^3 → R^{d_phi}` must be injective — different input triples must produce
different embeddings — so that no two training examples collapse to the same vector
before aggregation. With `d_phi=128` (far larger than the 3-dimensional input), a
trained ReLU MLP is injective on the training manifold by standard covering arguments.

### Continuity

`phi` and `rho` must be continuous: a small perturbation in the input must produce
a small change in the output, so that the aggregated representation varies smoothly.
ReLU networks are piecewise linear and therefore Lipschitz continuous — this
requirement is satisfied by the architecture as-is.

### PNA Pooling and the Sum/Mean Collision Problem

**The problem:** even with an injective `phi`, two *different* multisets can satisfy

```
mean(phi(x) for x in S1)  ==  mean(phi(x) for x in S2),   S1 ≠ S2
```

This "multiset collision" causes the model to map distinct training contexts to the
same latent representation, losing information that is relevant for the prediction.

**The fix — Principal Neighbourhood Aggregation (PNA):** instead of aggregating with
mean alone, concatenate four statistics over the set dimension:

```
pool(S) = cat[ sum_phi, mean_phi, max_phi, std_phi ]   ∈ R^{4·d_phi}
```

Two sets that share the same mean will generally differ in at least one of sum, max,
or std, yielding a distinct joint embedding. PNA is applied at *both* pooling stages
(feature-level and sample-level), so collisions are suppressed throughout the network.
The learnable equivariance layers (λ, γ) continue to operate *before* pooling and are
unaffected by this change.

PNA increases the rho input from `d_phi → 4·d_phi` and the psi input from
`d_rho → 4·d_rho`. The extra parameters are absorbed by rho and psi without changing
the output interface.

### Self-Attention Blocks (SAB)

The simple linear equivariance layer (λI + γ/n·11ᵀ) is replaced by one or more
**Self-Attention Blocks** from the Set Transformer (Lee et al. 2019), applied at both
the feature level (features attend to each other per sample) and the sample level
(samples attend to each other before final pooling):

```
X → φ → SAB_feat → pool_feat → ρ → SAB_samp → pool_samp → ψ
```

`SAB(X) = MAB(X, X)` where `MAB(Q, K) = LayerNorm(H + FFN(H))` and
`H = LayerNorm(Q + Dropout(MHA(Q, K, K)))`. SAB is permutation equivariant:
`SAB(X[π]) = SAB(X)[π]` for any permutation π — a strictly more expressive
generalisation of the original λ/γ equivariance. The number of SAB layers is
controlled by `n_sab_feat` and `n_sab_samp` in `ModelConfig`.

Setting `n_sab_feat=0, n_sab_samp=0` recovers the original linear equivariance
layers exactly (backward-compatible with old checkpoints).

### Pooling Modes

`SetPool` provides a unified interface for seven permutation-invariant pooling modes:

| Mode | Output dim | Description |
|---|---|---|
| `sum` | d | Element sum |
| `mean` | d | Element mean |
| `max` | d | Element max |
| `pna` | 4d | Concat[sum, mean, max, std] |
| `learned` | d | Softmax-weighted sum (learned scores) |
| `attn` | d | PMA: single-seed cross-attention |
| `multipool` | 5d | Concat[pna, attn] — for ablation |

PNA and multipool are the most expressive. Use `pool="multipool"` to run ablations
comparing all statistics simultaneously. Use `pool="attn"` for the Set Transformer
canonical pooling.

### Normalization Strategy

Two per-context normalizations are applied inside `forward()`:
- **Feature normalization** (`norm_feat=True`): each column of X_train is
  standardised to zero mean and unit variance; the same statistics are applied to
  x_test. This makes the model scale-invariant to feature magnitudes.
- **Target normalization** (`norm_target=True`): y_train is standardised before
  being fed to φ; the final prediction is denormalized back to the original scale.
  This removes sensitivity to the absolute scale of the regression target.

Both normalizations use per-context statistics (computed from X_train / y_train
of the current task), not global running statistics — the model requires no warm-up
and works immediately on any new task.

Batch normalization is not used: SPCS runs each meta-dataset as a batch of 1, so
BN statistics would be degenerate, and BN over the set dimension would break
permutation invariance with small sets.

### ModelConfig Hyperparameterization

All hyperparameters are bundled in `ModelConfig` (a `dataclasses.dataclass`):

| Field | Default | Description |
|---|---|---|
| `d_phi` | 128 | phi output dim (≥ p for universality) |
| `d_rho` | 256 | rho output dim (≥ n for universality) |
| `pool` | `"pna"` | Pooling mode (see table above) |
| `n_heads` | 4 | Attention heads for SAB / AttentionPool |
| `n_sab_feat` | 1 | SAB layers at feature level |
| `n_sab_samp` | 1 | SAB layers at sample level |
| `norm_feat` | `True` | Feature standardization |
| `norm_target` | `True` | Target standardization |
| `dropout` | 0.1 | Dropout in MLPs and SAB |

---

## Model Registry

After training, register the model via Snowflake Model Registry for versioned
deployment and SQL-callable inference:

```python
from snowflake.ml.registry import Registry

reg = Registry(session=session, database="TABPFN_DB", schema="TABPFN_SCHEMA")
reg.log_model(
    model=deepset_model,
    model_name="DEEPSET_TABPFN_V1",
    version_name="v1",
    sample_input_data=sample_batch,
)
```

SQL-callable inference after registration:

```sql
SELECT DEEPSET_TABPFN_V1!PREDICT(X_TRAIN, Y_TRAIN, X_TEST)
FROM INFERENCE_TABLE;
```

---

## Model Output

`best.pt` is the pretrained model artifact. It encodes the learned PPD approximation
procedure — the full state dict of the DeepSet (phi, rho, psi MLPs and the four
equivariant scalars λ_1, γ_1, λ_2, γ_2).

**Key properties:**
- Stored at `@MODEL_STAGE/checkpoints/best.pt`.
- Created by `torch.save({"state_dict": ..., "cfg": ...}, "best.pt")` whenever val MSE improves.
- Uploaded from the training container via `session.file.put("best.pt", "@MODEL_STAGE/checkpoints/", overwrite=True)`.
- This is the handoff artifact from training to evaluation. Evaluation does not read `@MODEL_STAGE/hpo/best_config.json`.

**Used for inference on any new synthetic dataset without retraining:**

```python
import torch
from model import DeepSetModel, ModelConfig

ckpt = torch.load("best.pt", map_location="cpu")
if isinstance(ckpt, dict) and "cfg" in ckpt:
    cfg, state_dict = ckpt["cfg"], ckpt["state_dict"]
else:                                       # legacy bare state_dict
    cfg = ModelConfig(d_phi=128, d_rho=256, pool="pna",
                      n_sab_feat=0, n_sab_samp=0,
                      norm_feat=False, norm_target=False)
    state_dict = ckpt
model = DeepSetModel(cfg=cfg)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    y_hat = model(X_train_new, y_train_new, x_test_new)
```

**SQL-callable via Model Registry:**

```sql
SELECT DEEPSET_TABPFN_V1!PREDICT(X_TRAIN, Y_TRAIN, X_TEST)
FROM INFERENCE_TABLE;
```

The registered model loads from `@MODEL_STAGE/checkpoints/best.pt` at inference time,
so no Docker image rebuild is required to update predictions after retraining.

---

## OOD Evaluation Stored Procedure

Run after training as a Snowpark Python stored procedure. Returns a result table
stratified by regime, feature count quartile, and set size quartile.

```python
from snowflake.snowpark import functions as F

def evaluate_ood(session):
    test_files = session.file.list("@META_DATASET_STAGE/test/")
    results = []

    for f in test_files:
        dataset = load_parquet(f)
        y_hat   = model.predict(dataset)
        mse     = float(((y_hat - dataset["betaX_test"]) ** 2).mean())
        results.append({
            "prior_regime": dataset["prior_regime"],
            "p_quartile":   quartile(int(dataset["p"])),
            "n_quartile":   quartile(int(dataset["n"])),
            "mse":          mse,
        })

    return (
        session.create_dataframe(results)
        .group_by("prior_regime", "p_quartile", "n_quartile")
        .agg(F.mean("mse").alias("mean_mse"), F.stddev("mse").alias("std_mse"))
        .sort("prior_regime", "p_quartile", "n_quartile")
    )
```

Expected output shape: one row per `(regime, p_quartile, n_quartile)` combination,
with `mean_mse` and `std_mse` columns. Regime A should have the lowest `mean_mse`;
Regimes C (heavy-tail noise) and D (correlated X) higher but not catastrophic.

---

## Downloading Results Locally

Stage ownership is split by artifact type:

| Stage path | File | Written by |
|---|---|---|
| `@MODEL_STAGE/checkpoints/best.pt` | Model checkpoint | `train.py` (on each val MSE improvement) |
| `@EVALUATION_RESULTS_STAGE/synthetic/test_report.csv` | Synthetic point-prediction report | `evaluate.py` |
| `@EVALUATION_RESULTS_STAGE/synthetic/mc_report.csv` | Synthetic MC dropout report | `evaluate.py` |
| `@EVALUATION_RESULTS_STAGE/benchmark_parts/<method>_detailed.csv` | Per-method benchmark detail | `evaluate.py` |
| `@EVALUATION_RESULTS_STAGE/model_comparison.csv` | Canonical benchmark comparison across DeepSet and all baselines | aggregate evaluation job |
| `@EVALUATION_RESULTS_STAGE/model_comparison_summary.csv` | Summary ranks and metrics | aggregate evaluation job |

Use SnowSQL for a direct stage check and one-off download:

```sql
LIST @EVALUATION_RESULTS_STAGE/;
GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
```

Use `download_results.py` to pull the checkpoint and evaluation outputs to your
local machine.

### Prerequisites

```bash
pip install snowflake-snowpark-python   # already in requirements.txt
```

### Set credentials

Set the following environment variables before running `download_results.py`:

| Variable | Required | Default |
|---|---|---|
| `SNOWFLAKE_ACCOUNT` | yes | — |
| `SNOWFLAKE_USER` | yes | — |
| `SNOWFLAKE_PASSWORD` | yes | — |
| `SNOWFLAKE_WAREHOUSE` | no | `COMPUTE_WH` |

### Run

```bash
cd C:/Documents/TabPFN_DemandModel
python download_results.py
```

The script:
1. Connects using the env-var credentials (same `TABPFN_DB` / `TABPFN_SCHEMA` / `COMPUTE_WH` defaults as the training job).
2. Lists `@MODEL_STAGE/checkpoints/` and `@EVALUATION_RESULTS_STAGE/`.
3. Downloads `@MODEL_STAGE/checkpoints/` and `@EVALUATION_RESULTS_STAGE/` into the local `models/` directory (created automatically).

### Expected output

```
Connected to Snowflake.
Stage contents:
  @model_stage/checkpoints/best.pt.gz
  @EVALUATION_RESULTS_STAGE/model_comparison.csv.gz

Downloading @MODEL_STAGE/checkpoints/ ...
Downloading @EVALUATION_RESULTS_STAGE/ ...

Done. Files saved to ./models/
  models/best.pt
  models/model_comparison.csv
```

> **Note:** Snowflake automatically decompresses `.gz` files on `GET`/`session.file.get()`,
> so the files land as `best.pt` and CSV files without `.gz`.

### Load the checkpoint

```python
import torch
from model import DeepSetModel

ckpt  = torch.load("models/best.pt", map_location="cpu")
model = DeepSetModel(cfg=ckpt["cfg"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

### Inspect evaluation results

```python
import pandas as pd
df = pd.read_csv("models/model_comparison.csv")
print(df)
```
