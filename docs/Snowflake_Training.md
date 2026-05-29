# Snowflake Training - DeepSet TabPFN

Describes how to run the DeepSet training pipeline inside a Snowflake environment
using Snowpark Container Services (SPCS) and Snowflake Model Registry.

---

## Snowflake Environment Construction

Ordered setup: from local machine to a running SPCS training job with model checkpoint
written back to Snowflake.

**Steps:**

1. **Create database, schema, and stages** with `run_training_job.sql`: `@META_DATASET_STAGE`, `@MODEL_STAGE`, `@EVALUATION_DATASET_STAGE`, `@EVALUATION_RESULTS_STAGE`, and `@MLJOB_PAYLOAD_STAGE`.
2. **Create compute pools** with `run_training_job.sql`: `DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, and `AUTOGLUON_CPU_POOL`. Verify the pools reach `ACTIVE` state before submitting jobs.
3. **Create network rules, `KAGGLE_API_SECRET`, and `BENCHMARK_EXTERNAL_ACCESS`**. The committed SQL uses placeholders only; never commit real Kaggle credentials.
4. **Upload scripts and Parquet data** with SnowSQL `PUT`: `src/*.py` and `scripts/*.py` to `@MODEL_STAGE/scripts/`, and local synthetic datasets to `@META_DATASET_STAGE/{train,val,test}/`.
5. **Call `build_meta_dataset_index()`**. This submits a CPU MLJob that lists staged synthetic parquet, reads scalar metadata inside Snowflake, rebuilds `META_DATASET_INDEX`, and validates `train=800`, `val=100`, `test=100`.
6. **Optionally call `download_kaggle_to_stage()`**. This submits an MLJob that receives Kaggle credentials at runtime and stages raw Kaggle benchmark inputs under `@META_DATASET_STAGE/kaggle/`.
7. **Call `prepare_benchmark_datasets()`**. This fetches/normalizes OpenML and staged Kaggle data once, then writes prepared `.npz` files and `benchmark_manifest.json` under `@META_DATASET_STAGE/benchmark_prepared/`.
8. **Run gate-specific pretrain jobs** — one per `gate_hidden_dim` candidate (32, 64, 128).
   HPO tunes `gate_hidden_dim` across all three; each HPO trial warm-starts from its matching
   `pretrain_gate<N>.pt`. All three checkpoints must exist before HPO starts.

   ```sql
   CALL run_pretrain_pipeline(
       'market_exchangeable_icl', 'synthetic_regression_combined',
       'inductive_forecasting', 32
   );
   CALL run_pretrain_pipeline(
       'market_exchangeable_icl', 'synthetic_regression_combined',
       'inductive_forecasting', 64
   );
   CALL run_pretrain_pipeline(
       'market_exchangeable_icl', 'synthetic_regression_combined',
       'inductive_forecasting', 128
   );
   LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain_gate.*[.]pt';
   ```

   Each job writes `@MODEL_STAGE/checkpoints/pretrain_gate<N>.pt` with the gate MLP of
   the matching width. Verify all three exist before starting HPO.

9. **Call `run_hpo_pipeline()` — ridge_residual sweep (single sweep, production-ready).**
   HPO requires all three gate-specific pretrain checkpoints and samples `gate_hidden_dim` across
   them. Fixed architecture: `d_phi=128`, `n_sab_feat=1` (not tuned in this release).

   ```sql
   CALL run_hpo_pipeline(
       'market_exchangeable_icl', 'synthetic_regression_combined',
       'inductive_forecasting', 'ridge_residual'
   );
   SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));
   ```

   Tuned parameters: `lr`, `weight_decay`, `dropout`, `ridge_lambda`, `gate_hidden_dim`,
   `use_huber`, `huber_delta`, `lambda_l1`. Fails hard on pretrain checkpoint mismatch.

   Writes `@MODEL_STAGE/hpo/best_config.json` with `_meta.pretrain_checkpoint_stage_path`
   recording which `pretrain_gate<N>.pt` the winning trial used.

   **Architecture HPO (`HPO_SWEEP_MODE=architecture`) is intentionally disabled for this
   release.** Architecture is fixed at `d_phi=128`, `n_sab_feat=1`. Re-enable only after
   adding matching pretrain checkpoint matrices for every `d_phi`/`n_sab_feat` candidate.

10. **Call `run_model_training()`**. Reads `best_config.json` from `@MODEL_STAGE/hpo/` and
    resolves the pretrain checkpoint strictly — no cold-start, no legacy fallback:
    - Preferred: `best_config._meta.pretrain_checkpoint_stage_path` (set by HPO).
    - Fallback: `@MODEL_STAGE/checkpoints/pretrain_gate<gate_hidden_dim>.pt`.
    - Raises `FileNotFoundError` before job submission if no valid checkpoint is found.

    Always uses `PRETRAIN_LOAD_POLICY=require_match` — raises `RuntimeError` on any
    architecture mismatch. Writes `@MODEL_STAGE/checkpoints/best.pt` (v4 format).

    The saved checkpoint `metadata` includes `best_val_mse`, `train_mse_at_best`,
    `best_epoch`, `pretrain_loaded`, `pretrain_policy`.

11. **Verify `best.pt`** with `LIST @MODEL_STAGE/checkpoints/ PATTERN='.*best[.]pt';`.
12. **Run evaluation** — two options depending on Snowflake account node quota:
    - **Split-phase (recommended under tight quota)**: call each phase independently
      and suspend its pool before starting the next (see *Split-Phase Evaluation Under
      Tight Node Quota* below). This prevents `Requested number of nodes exceeds node
      limit` errors when all three pools cannot be held simultaneously.
    - **Monolithic (legacy convenience)**: call `run_evaluation_pipeline('<prep>', '<benchmark>', '<autogluon>')`.
      Evaluation reads `best.pt`, preflights all compute pools and runtime images,
      validates prepared benchmark manifest/index metadata, launches prepared-dataset
      benchmark shards, then aggregates CSV outputs to `@EVALUATION_RESULTS_STAGE`.
      Requires holding quota across all three pools for the full ~2-hour run.
13. **Download results locally** from the client with SnowSQL `GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';`.

## Split-Phase Evaluation Under Tight Node Quota

When the Snowflake account has limited node quota, holding `DEEPSET_GPU_POOL` (10 nodes),
`DEEPSET_CPU_POOL` (6 nodes), and `AUTOGLUON_CPU_POOL` (60 nodes) simultaneously during the
full ~2-hour `run_evaluation_pipeline()` run can cause `Requested number of nodes exceeds node
limit` failures mid-pipeline. The split-phase procedures expose each benchmark phase as its own
stored procedure so quota is released between phases.

### Recommended SQL sequence

```sql
-- Step 1: Validate runtime images (run once before all phases)
CALL run_evaluation_runtime_probes('<prep>', '<benchmark>', '<autogluon>');

-- Step 2: Validate and prepare benchmark manifest/index (DEEPSET_CPU_POOL)
CALL run_evaluation_prep('<prep>', '<benchmark>', '<autogluon>');

-- Step 3: Synthetic eval + 10 DeepSet GPU shards (DEEPSET_GPU_POOL)
CALL run_deepset_evaluation('<prep>', '<benchmark>', '<autogluon>');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;   -- release GPU quota

-- Step 4: 3 CPU baseline benchmark shards (DEEPSET_CPU_POOL)
CALL run_baseline_evaluation('<prep>', '<benchmark>', '<autogluon>');
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;   -- release CPU quota

-- Step 5: 30 AutoGluon shards, max 30 concurrent (AUTOGLUON_CPU_POOL)
CALL run_autogluon_evaluation('<prep>', '<benchmark>', '<autogluon>');
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND; -- release AutoGluon quota

-- Step 6: Aggregate all benchmark_parts/ into model_comparison.csv
CALL run_evaluation_aggregation('<prep>', '<benchmark>', '<autogluon>');
```

### Key notes

- **Manual suspend is required**: completing a phase does not automatically release quota.
  The caller must issue `ALTER COMPUTE POOL ... SUSPEND` after each phase.
- **No runtime probes in split procedures**: run `run_evaluation_runtime_probes()` once
  before starting any split-phase procedure.
- **Retrying individual phases**: if a phase fails, re-run only that phase without re-running
  completed ones. The prep and aggregation jobs are idempotent.
- **Aggregation re-run**: `run_evaluation_aggregation()` can be re-run without re-running
  prior phases as long as the `benchmark_parts/` part files already exist on
  `@EVALUATION_RESULTS_STAGE`.
- **AutoGluon batching**: `run_autogluon_evaluation()` submits 30 shards in batches of up to 30
  (`AUTOGLUON_MAX_CONCURRENT_SHARDS`), so all AutoGluon shards can run concurrently when
  `AUTOGLUON_CPU_POOL MAX_NODES` and account quota allow it.

## Current Snowflake Guardrails

- `run_training_job.sql` creates `@MLJOB_PAYLOAD_STAGE` in addition to
  `@META_DATASET_STAGE`, `@MODEL_STAGE`, `@EVALUATION_DATASET_STAGE`, and
  `@EVALUATION_RESULTS_STAGE`.
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
- AutoGluon is a full benchmark method. Its shard jobs use
  `AUTOGLUON_RUNTIME_ENVIRONMENT`, set
  `BENCHMARK_METHOD=AutoGluon`, `AUTOGLUON_TIME_LIMIT=300`, and write unique part
  CSVs under `@EVALUATION_RESULTS_STAGE/benchmark_parts/AutoGluon_shard{i}_of_{n}_detailed.csv`.
  The AutoGluon runtime probe (`AUTOGLUON_REQUIRED_IMPORTS`) checks:
  `autogluon.tabular`, `numpy`, `pandas`, `sklearn`, `scipy`, `pyarrow`,
  `torch`, and `snowflake.snowpark`. It intentionally excludes `xgboost`,
  `lightgbm`, and `catboost`, which are not required for AutoGluon shards.
  `scipy`, `pyarrow`, and `torch` must be present because `evaluate.py`
  hard-imports them at module startup regardless of selected benchmark method.
- `run_evaluation_pipeline()` requires three STRING arguments: the prep, benchmark,
  and AutoGluon runtime image names. Do not rely on local shell variables or
  Snowsight worksheet variables; they are not propagated into the Python stored
  procedure runtime. Quick-fail diagnostic: if the procedure errors in under a few
  seconds with `PREP_RUNTIME_ENVIRONMENT is required`, the old zero-argument
  procedure is still installed. Recreate it with three arguments, then:
  `CALL run_evaluation_pipeline('<prep>', '<benchmark>', '<autogluon>');`
- MODEL3-ICL benchmark shard jobs run `MODEL3-ICL-MC bounded-context ensemble`, not
  exact full-context inference. The OOM boundary was forwarding the full
  90% processed training split against the full test split inside MC dropout on
  Snowflake GPU nodes. The remediation preserves dataset, seed, split, and test-row
  coverage by splitting 90/10 first, fitting preprocessing on train only, applying
  deterministic train-only `train_f_regression` feature selection capped by
  `BENCHMARK_DEEPSET_FEATURE_CAP` (default `model.cfg.d_phi`), sampling five
  deterministic non-overlapping train-only context windows capped at 200 rows,
  predicting the same capped processed full test split in 128-row chunks,
  averaging the five prediction vectors, and computing metrics once. First stable
  full benchmark runs use `MC_K=8`; move to `MC_K=16` only after memory and
  runtime stability are proven.

Recommended smoke order: compile the Python scripts, create Snowflake objects with
`run_training_job.sql`, run `CALL build_meta_dataset_index()`, optionally run
`CALL download_kaggle_to_stage()` for raw Kaggle staging, run
`CALL prepare_benchmark_datasets()`, verify
`LIST @META_DATASET_STAGE/benchmark_prepared/ PATTERN='.*benchmark_manifest[.]json';`,
run `CALL run_hpo_pipeline()`, verify or read
`@MODEL_STAGE/hpo/best_config.json`, run `CALL run_model_training()`, verify
`LIST @MODEL_STAGE/checkpoints/` includes `best.pt`, optionally run
`CALL run_evaluation_runtime_probes('<prep>', '2.5.0-py311', '<autogluon>')` to
validate all 5 runtime probes (including the CPU baseline probe with catboost) before
the full run, then run
`CALL run_evaluation_capacity_probe('<prep>', '2.5.0-py311', '<autogluon>')` to verify
the account can currently accept the planned concurrency envelope (GPU=10, CPU=3,
AutoGluon=30) — this is a quota check only and does not load models or benchmark data;
if it fails with a node limit error, run `SHOW COMPUTE POOLS`, suspend idle pools, wait
for active jobs to finish, or request a higher Snowflake account node quota before
retrying. Then run a one-dataset AutoGluon smoke where supported, then run
`CALL run_evaluation_pipeline('<prep>', '2.5.0-py311', '<autogluon>')` and verify
`benchmark_parts/AutoGluon_shard*_detailed.csv`, `model_comparison.csv`, and
`model_comparison_summary.csv`.

## CatBoost Dependency on Snowflake-Managed Container Runtime

CatBoost is NOT preinstalled in the `2.5.0-py311` Snowflake-managed Container Runtime image.
It is installed per-MLJob via `pip_requirements` + `external_access_integrations`. This requires
a Snowflake EAI that allows outbound PyPI access.

### Admin bootstrap (one-time, ACCOUNTADMIN required)

Run the Step 2c block in `sql/run_training_job.sql`:
```sql
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TABPFN_CATBOOST_PYPI_EAI ...
GRANT USAGE ON INTEGRATION TABPFN_CATBOOST_PYPI_EAI TO ROLE <job_submitter_role>;
```

### MLJob dependency wiring

Every CatBoost-dependent MLJob submission must pass both:
```python
pip_requirements=[f"catboost=={CATBOOST_VERSION}"]
external_access_integrations=["TABPFN_CATBOOST_PYPI_EAI"]
```
The EAI opens the PyPI network path; `pip_requirements` triggers the install.
Passing only one of the two will fail silently or with a network error.

### Validation

1. Verify EAI exists: `SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'TABPFN_CATBOOST_PYPI_EAI';`
2. Verify grant: `SHOW GRANTS ON INTEGRATION TABPFN_CATBOOST_PYPI_EAI;`
3. Run probes: `CALL run_evaluation_runtime_probes('<prep>', '2.5.0-py311', '<autogluon>');`
4. Confirm logs: `[runtime_probe] required import ok: catboost version=<pinned_version>`

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No module named 'catboost'` with `pip_requirements` set | Missing EAI — pip cannot reach PyPI | Create EAI and pass `external_access_integrations` |
| Network error during pip install | EAI exists but USAGE not granted to job role | `GRANT USAGE ON INTEGRATION ... TO ROLE ...` |
| Notebook imports catboost but probe cannot | Notebook EAI setting does not propagate to submitted MLJobs | Configure submission args, not only notebook |
| EAI creation fails | Role lacks `CREATE INTEGRATION` or PyPI rule access | Run bootstrap with ACCOUNTADMIN |

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

Use this checklist before running `CALL run_hpo_pipeline()`,
`CALL run_model_training()`, and `CALL run_evaluation_pipeline()`. It covers synthetic data staging, Snowflake-native
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

REMOVE @META_DATASET_STAGE/train/;
REMOVE @META_DATASET_STAGE/val/;
REMOVE @META_DATASET_STAGE/test/;
PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

For Kaggle benchmark data, the preferred production raw-staging path is Snowflake-native:
store the Kaggle API token in a Snowflake `SECRET`, run the one-off
`download_kaggle_to_stage()` setup procedure, and let the MLJob upload raw Kaggle
`.npz` files directly to `@META_DATASET_STAGE/kaggle/`. Those files are not the
final benchmark inputs consumed by shard jobs.

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

After optional Kaggle raw staging, run the canonical preparation procedure. It
fetches OpenML datasets, normalizes OpenML and Kaggle into prepared `.npz` files,
and writes the manifest plus `BENCHMARK_DATASET_INDEX` metadata table used by
all benchmark shards:

```sql
CALL prepare_benchmark_datasets();
LIST @META_DATASET_STAGE/benchmark_prepared/;
LIST @META_DATASET_STAGE/benchmark_prepared/ PATTERN='.*benchmark_manifest[.]json';
```

Production benchmark shard jobs consume prepared staged datasets only:
`@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json` and prepared
`.npz` files under `@META_DATASET_STAGE/benchmark_prepared/{openml,kaggle}/`.
They load prepared files with `np.load(..., allow_pickle=False)` and must not
fetch OpenML or Kaggle data at shard runtime. `evaluate.py` must not import,
require, or call OpenML APIs; OpenML is a dataset-preparation dependency only,
owned by `prepare_benchmark_datasets.py`. `evaluate.py` reads only prepared
manifest/index metadata plus staged `.npz` payloads. Each shard owns datasets by
`dataset_index` and evaluates every owned dataset across all configured seeds,
loading one `.npz` file at a time. The default assignment is deterministic
modulo; `BENCHMARK_SHARD_STRATEGY=balanced` uses
`BENCHMARK_DATASET_INDEX.benchmark_weight` for cost-aware assignment without
inspecting or transforming the staged `.npz` payloads.

### 3. Enable external network access and Kaggle secret

Benchmark dataset preparation needs Snowflake external access so
`prepare_benchmark_datasets.py` can fetch OpenML and read staged Kaggle inputs.
The optional Kaggle setup job also needs external access to Kaggle/download hosts
and a Snowflake `SECRET` holding the API credentials. Benchmark shard jobs must
not use OpenML or Kaggle network fetch paths in production. Create the network
rules and secret in `TABPFN_SCHEMA`:

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
script through `snowflake.snowpark.secrets`.

`run_evaluation_pipeline()` always submits the benchmark dataset preparation job
before benchmark shards as a lightweight manifest and `BENCHMARK_DATASET_INDEX`
validation step. Existing valid manifests exit early inside
`prepare_benchmark_datasets.py`; invalid or incomplete manifests are rebuilt. The
prep job uses `pip_requirements=["openml==0.15.1"]` and
`external_access_integrations=["BENCHMARK_EXTERNAL_ACCESS", "TABPFN_PYPI_EAI"]`;
benchmark shard jobs do not receive the benchmark integration or OpenML dependency.
`openml` is not a shard dependency, and `evaluate.py` must remain OpenML-free. Creating the
environment objects only makes the route available; jobs will not use it unless
the MLJob configuration names the integration and dependencies.

The evaluation submission layer requires prebuilt managed runtime images and adds
only the narrow per-job pip dependencies that are missing from those images.
Configure all three controls before calling `run_evaluation_pipeline()`:

- `PREP_RUNTIME_ENVIRONMENT` for `prepare_benchmark_datasets.py`.
- `BENCHMARK_RUNTIME_ENVIRONMENT` for non-AutoGluon `evaluate.py` jobs,
  including synthetic evaluation, DeepSet benchmark shards, baseline benchmark
  shards, and aggregation.
- `AUTOGLUON_RUNTIME_ENVIRONMENT` for AutoGluon shards.

`run_evaluation_test.py` passes these values to
`submit_from_stage(runtime_environment=...)`. It fails fast with a
`RuntimeError` if any required runtime image env var is missing. It exposes
the selected runtime inside each container as `EVAL_RUNTIME_ENVIRONMENT` and runs
5 `runtime_probe.py` preflight probes: benchmark GPU, benchmark aggregate CPU,
CPU baseline (with `pip_requirements=["catboost"]`), prep CPU (with
`pip_requirements=["openml==0.15.1"]`), and AutoGluon CPU.
Both `run_evaluation_pipeline()` and `run_evaluation_runtime_probes()` submit
these five probes serially: submit one probe, wait for completion, then submit
the next. This is intentional because `target_instances=1` still consumes
account node quota; do not restore concurrent probe fan-out across the GPU, CPU,
and AutoGluon pools unless Snowflake node quota has been raised and verified.

Probe import requirements (`BENCHMARK_REQUIRED_IMPORTS` / `BASELINE_REQUIRED_IMPORTS`):
- **Benchmark GPU and aggregate CPU probes** (`BENCHMARK_REQUIRED_IMPORTS`): `torch`,
  `pyarrow`, `pandas`, `scipy`, `sklearn`, Snowpark, and Snowflake ML jobs.
  These probes do **not** include XGBoost, LightGBM, or CatBoost.
- **CPU baseline probe** (`BASELINE_REQUIRED_IMPORTS`): same as above plus
  `xgboost`, `lightgbm`, and `catboost`. This probe additionally passes
  `pip_requirements=["catboost"]` because `2.5.0-py311` does not include `catboost`
  in the managed image.
- **Prep probe**: `openml`, `numpy`, and Snowpark. This probe passes
  `pip_requirements=["openml==0.15.1"]` with `TABPFN_PYPI_EAI`.
- **AutoGluon probe**: `autogluon.tabular`, `numpy`, `pandas`, `sklearn`, `scipy`,
  `pyarrow`, `torch`, and Snowpark. Intentionally excludes XGBoost, LightGBM, and
  CatBoost. `scipy`, `pyarrow`, and `torch` are required because `evaluate.py`
  hard-imports them at module startup regardless of benchmark method.

CPU baseline shard jobs pass `pip_requirements=["catboost"]` **explicitly at
the call site** in the Phase 4 loop of `run_evaluation_pipeline()`. The
`_submit_eval()` helper accepts an explicit `pip_requirements` parameter (default
`_UNSET`); when `_UNSET` the helper falls back to `_pip_requirements_for_eval()`
inference, but Phase 4 always passes `pip_requirements=list(BASELINE_EXTRA_PIP_REQUIREMENTS)`
directly so the guarantee is visible at the submission site and does not depend on
`BENCHMARK_METHODS` parsing. Prep jobs pass only `openml==0.15.1`; AutoGluon jobs
pass only `autogluon.tabular==1.3.0`; synthetic, DeepSet, and aggregate jobs pass
no `pip_requirements`.

Verify available runtime image names in the target account. Snowflake documents
the argument on the
[`submit_from_stage` API reference](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/jobs/snowflake.ml.jobs.submit_from_stage).

After uploading scripts, optionally run the raw Kaggle setup procedure, then run
dataset preparation and verify the prepared benchmark manifest:

```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
CALL prepare_benchmark_datasets();
LIST @META_DATASET_STAGE/benchmark_prepared/;
LIST @META_DATASET_STAGE/benchmark_prepared/ PATTERN='.*benchmark_manifest[.]json';
```

References: Snowflake [`CREATE NETWORK RULE`](https://docs.snowflake.com/en/sql-reference/sql/create-network-rule) and [`CREATE EXTERNAL ACCESS INTEGRATION`](https://docs.snowflake.com/en/sql-reference/sql/create-external-access-integration).

### 4. Confirm stages and compute pools

All project stages must exist before jobs are submitted. `MLJOB_PAYLOAD_STAGE` is the
bare stage name passed to `submit_from_stage(stage_name=...)`; results are written
to `@EVALUATION_RESULTS_STAGE` before any local download:

```sql
CREATE STAGE IF NOT EXISTS META_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS MODEL_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

Resource split:

- GPU (`DEEPSET_GPU_POOL`): 5-node HPO, 10-node DDP training, synthetic MODEL3-ICL evaluation, and MODEL3-ICL-MC benchmark (10 single-node GPU shard jobs).
- CPU_X64_M (`DEEPSET_CPU_POOL`): Kaggle download, 6 single-node synthetic regression baseline dataset shard jobs, and aggregation. Each baseline shard owns a dataset subset and runs all baseline methods inside the dataset-first loop.
- CPU_X64_M (`AUTOGLUON_CPU_POOL`): Synthetic regression AutoGluon evaluation. Two modes:
  - **Ray distributed** (`AUTOGLUON_CLUSTER_SHARDS > 0`, combined suite default): 6 logical
    work-item clusters × 4 target instances = up to 24 concurrent CPU_X64_M nodes. Each cluster
    runs `autogluon_ray.py`; Ray distributes independent dataset/seed/condition work items.
    Each Ray worker loads its own dataset locally — the driver does **not** call `ray.put(dataset)`.
    `MAX_IN_FLIGHT` bounds concurrent worker-loaded fits (not object-store payload count).
    `CONCURRENT_CLUSTERS` must equal `CLUSTER_SHARDS` (single-wave enforcement).
  - **Single-node shards** (`AUTOGLUON_CLUSTER_SHARDS = 0`): N independent single-instance MLJobs
    (`target_instances=1`), each running `evaluate_synthetic_regression.py` with `mode=autogluon`.
    No Ray cluster; `CONCURRENT_CLUSTERS` determines the shard count and expected aggregation files.
  Main and OOD suites use single-node sharded paths, so the pool remains capped at 60 nodes to
  support those strict single-wave runs.

Evaluation dependency checks are method-aware. The shared prepared-benchmark
path requires scikit-learn for splitting, preprocessing, and metrics for
DeepSet, CPU baselines, and AutoGluon. XGBoost, LightGBM, and CatBoost are
required only when their exact baseline methods are selected, so AutoGluon
runtime images do not need those three baseline packages.

```sql
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_SUSPEND_SECS = 300;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 6
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 60
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
  â””â”€â”€ *.parquet â”€â”€PUTâ”€â”€â†’ @META_DATASET_STAGE
  â””â”€â”€ *.py â”€â”€PUTâ”€â”€â†’ @MODEL_STAGE/scripts/ â”€â”€vol mountâ”€â”€â†’ /opt/app/

Snowsight / SnowSQL
  â””â”€â”€ CALL run_hpo_pipeline() â”€â”€â†’ Snowpark stored procedure
        â””â”€â”€ run_hpo_job.run_hpo_pipeline(session) submits:

Container Runtime - Phase 1: HPO (5 nodes, GPU_NV_M)
  â””â”€â”€ hpo.py â†’ @MODEL_STAGE/hpo/best_config.json

Snowsight / SnowSQL
  â””â”€â”€ CALL run_model_training() â”€â”€â†’ Snowpark stored procedure
        â””â”€â”€ run_model_training_job.run_model_training(session) submits:

Container Runtime - Phase 2: Training (10 nodes, DDP)
  â”œâ”€â”€ reads @MODEL_STAGE/hpo/best_config.json into BEST_CONFIG
  â”œâ”€â”€ META_DATASET_INDEX selects full train/val stage_path rows
  â”œâ”€â”€ materialize_indexed_meta_dataset() copies selected parquet to /tmp/data/
  â”œâ”€â”€ DataLoader (4 workers, prefetch_factor=2) reads /tmp/data/train/*.parquet
  â”œâ”€â”€ trains DeepSet (phi, rho, psi + 4 equivariant scalars)
  â”‚     BF16 autocast + GradScaler, batched forward over all m test rows
  â”‚     torch.compile(mode="reduce-overhead") fuses GPU kernels
  â””â”€â”€ writes best.pt â†’ @MODEL_STAGE/checkpoints/best.pt

Snowsight / SnowSQL
  â””â”€â”€ CALL run_evaluation_pipeline() â”€â”€â†’ Snowpark stored procedure
        â””â”€â”€ run_evaluation_test.run_evaluation_pipeline(session) submits:

Container Runtime - Evaluation
  â”œâ”€â”€ prepare_benchmark_datasets.py â†’ @META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json
  â””â”€â”€ evaluate.py â†’ @EVALUATION_RESULTS_STAGE/synthetic/*.csv
                  â†’ @EVALUATION_RESULTS_STAGE/benchmark_parts/*_shard*_detailed.csv
                  â†’ @EVALUATION_RESULTS_STAGE/model_comparison.csv
                  â†’ @EVALUATION_RESULTS_STAGE/model_comparison_summary.csv

Model Registry
  â””â”€â”€ DEEPSET_TABPFN_V1!PREDICT() â† loads from @MODEL_STAGE/checkpoints/best.pt
```

---

## Data Storage

Variable-shape datasets (pickle) cannot be stored in flat Snowflake tables efficiently.
Use an **internal named stage** with Parquet files:

```
@META_DATASET_STAGE/
  train/   â† 800 parquet files (one per meta-task)
  val/     â† 100 parquet files
  test/    â† 100 parquet files
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

REMOVE @META_DATASET_STAGE/train/;
REMOVE @META_DATASET_STAGE/val/;
REMOVE @META_DATASET_STAGE/test/;
PUT file://C:/Documents/TabPFN_DemandModel/data/train/*.parquet @META_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/data/val/*.parquet   @META_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/data/test/*.parquet  @META_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

Verify the upload:

```sql
LIST @META_DATASET_STAGE/train/;
LIST @META_DATASET_STAGE/val/;
LIST @META_DATASET_STAGE/test/;
```

`META_DATASET_INDEX` is required before HPO and training. It is a metadata
pruning table over staged parquet payloads, not a copy of the payloads. Rebuild
it after every synthetic parquet regeneration or restaging:

```sql
CALL build_meta_dataset_index();

SELECT split, COUNT(*) AS task_count
FROM META_DATASET_INDEX
GROUP BY split
ORDER BY split;
-- Expected: train=800, val=100, test=100
```

The HPO subset is deterministic and should return 200 train rows and 40
validation rows:

```sql
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY split, hpo_bucket
      ORDER BY prior_regime, p, n_train, task_id
    ) AS bucket_rank
  FROM META_DATASET_INDEX
  WHERE split IN ('train', 'val')
),
selected AS (
  SELECT *
  FROM ranked
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY split
    ORDER BY bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id
  ) <= IFF(split = 'train', 200, 40)
)
SELECT split, COUNT(*) AS selected_rows
FROM selected
GROUP BY split
ORDER BY split;
```

### Uploading Python scripts via SnowSQL

Run once, and re-run whenever any script changes:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

The current local canonical data has verified HPO cardinality
`max_p=24` and `max_n_train=200`. HPO uses a fixed warm-start architecture
`d_phi=128`, `d_rho=256`, and `pool="pna"`; regenerated datasets must keep
selected HPO rows within `max(p) <= 128` and `max(n_train) <= 256` or HPO fails
before launching trials.

After the ShardedDataConnector failure fix, restage at least the changed runtime
files before rerunning pretrain or final training:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;
USE WAREHOUSE COMPUTE_WH;

PUT file://C:/Documents/TabPFN_DemandModel/src/train.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/src/snowflake_io.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

LIST @MODEL_STAGE/scripts/ PATTERN='.*(train|snowflake_io)[.]py';
```

For a safer full restage that avoids stale script drift:

```sql
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;
USE WAREHOUSE COMPUTE_WH;

PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

LIST @MODEL_STAGE/scripts/ PATTERN='.*(train|snowflake_io|model|run_pretrain_job|run_model_training_job)[.]py';
```

All runnable MLJob code, including `hpo_epoch_test.py` and `train_epoch_test.py`,
is loaded from `@MODEL_STAGE/scripts/`. `@EPOCH_STAGE` is output-only for epoch
calibration JSON. `hpo_timing.json` contains metadata selection time,
materialization time, a baseline run, marginal sweep runs, and derived HPO
wall-clock estimates; `train_timing.json` contains production topology timing
and full train/val materialization time. Error files are written as
`hpo_epoch_error.json` and `train_epoch_error.json`.

Read `hpo_timing.json` by phase. HPO wall time includes MLJob startup, Ray/Tuner
scheduling, metadata selection, per-node stage materialization, and epoch
compute; the single `epoch_time_s` field is only the model-compute component.

Verify:

```sql
LIST @MODEL_STAGE/scripts/;
LIST @MODEL_STAGE/scripts/ PATTERN='.*(hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py';
```

Before rerunning a canceled HPO job, confirm `hpo.py`, `train.py`, `model.py`,
`snowflake_io.py`, `run_hpo_job.py`, `run_model_training_job.py`, and
`run_training_job.py` are all present under
`@MODEL_STAGE/scripts/` and are not only `.gz` duplicates. HPO is expected to
produce either `@MODEL_STAGE/hpo/best_config.json` on success or
`@MODEL_STAGE/hpo/hpo_failure.json` if Python starts and then fails.

**hpo.py guardrails:**

- `debug/hpo_failure.json` is a **locally-downloaded snapshot** and is not
  automatically synced from the stage. An unchanged local file does not mean
  `main()` was not reached — always check the stage version first:
  `LIST @MODEL_STAGE/hpo/;` and compare the `last_modified` timestamp.
- `TunerConfig` does not accept `uses_snowflake_trainer` in its documented API
  (`metric`, `mode`, `search_alg`, `num_trials`, `max_concurrent_trials`,
  `resource_per_trial`). Do not add undocumented kwargs; a library version bump
  will raise `TypeError: __init__() got an unexpected keyword argument` inside
  `main()`, which the `except` block catches but may not successfully upload to
  `hpo_failure.json` if the session is also disrupted.
- Legacy Snowflake ML Tuner path only: `ctx.report()` inside `train_for_hpo()`
  **must** pass `model=model.to("cpu")`. Omitting the model argument causes
  `TypeError: Path must be a string` inside `tuner.run()` when Snowflake tries
  to load `TunerResults.best_model` from an empty path after all trials complete.
- `hpo.py` emits three diagnostic prints that confirm normal startup sequence in
  job logs: `"hpo.py: module load started"` (before imports), `"hpo.py: all
  imports OK"` (after all imports), and `"hpo.py: entered main()"` (first line
  of `main()`). If any of these are absent, the crash happened before that point.

Before running epoch calibration, confirm `hpo_epoch_test.py`,
`train_epoch_test.py`, `train.py`, `model.py`, and `snowflake_io.py` are all
present under `@MODEL_STAGE/scripts/`.

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

-- Internal stage for synthetic regression and OOD evaluation input datasets.
CREATE STAGE IF NOT EXISTS EVALUATION_DATASET_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Internal stage for all evaluation CSVs and benchmark comparison outputs.
CREATE STAGE IF NOT EXISTS EVALUATION_RESULTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- Internal stage used by submit_from_stage(stage_name=...) for MLJob payloads.
CREATE STAGE IF NOT EXISTS MLJOB_PAYLOAD_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

---

## Compute: Container Runtime for ML

### 1. Compute Pool

```sql
-- GPU_NV_M: 4 A10G GPUs per node. MAX_NODES=10 supports 5-node HPO (20 concurrent
-- one-GPU trials, 1 round) and 10-node DDP training (world_size=40).
-- Evaluation: synthetic=1 GPU node; DeepSet benchmark=10 GPU dataset shard jobs;
--   baselines=6 CPU dataset shard jobs, each running all baseline methods;
--   AutoGluon=60 CPU shard jobs. CPU_X64_M: MAX_NODES=6 supports the combined
--   baseline shard topology.
-- SPCS does not support ALTER COMPUTE POOL to change INSTANCE_FAMILY; drop and recreate.
DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;
CREATE COMPUTE POOL DEEPSET_GPU_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = GPU_NV_M
  AUTO_SUSPEND_SECS = 300;

DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;
CREATE COMPUTE POOL DEEPSET_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 6
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;
CREATE COMPUTE POOL AUTOGLUON_CPU_POOL
  MIN_NODES = 1
  MAX_NODES = 60
  INSTANCE_FAMILY = CPU_X64_M
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;
```

### 2. Job Submission (MLJob)

`run_hpo_job.py`, `run_model_training_job.py`, and `run_training_job.py` are
deployed as Snowpark Python stored procedures. The split training procedures submit
HPO and training MLJobs using scripts already on `@MODEL_STAGE/scripts/`; the
combined wrapper still submits both phases in one call. The evaluation procedure
separately submits synthetic, DeepSet benchmark, baseline benchmark, AutoGluon
benchmark, and aggregate MLJobs. No local Python environment is needed, and no
dataset stage contents are materialized outside Snowflake.

#### What is an MLJob container?

An MLJob container is a short-lived compute environment that Snowflake starts on
one or more nodes in your GPU compute pool to run a single Python script. When
`submit_from_stage()` is called, Snowflake pulls the managed ML runtime image onto
the requested nodes, runs your entrypoint (e.g. `train.py`), writes outputs to the
stage, then shuts the container down. The `stage_name` argument is the MLJob payload
stage for scripts/artifacts; it is not a dataset mount. Training scripts explicitly
materialize `@META_DATASET_STAGE/{train,val,test}/` into container-local `/tmp/data`
with Snowpark `session.file.get()`. PyTorch, Ray, and `snowflake-ml-python` are
pre-installed - no Docker build or image management is required.

Create the procedures only after the scripts are staged. Otherwise `IMPORTS =
('@MODEL_STAGE/scripts/run_training_job.py')` can fail because Snowflake cannot
resolve the staged file:

```sql
PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
LIST @MODEL_STAGE/scripts/;
```

Create the procedures, and re-run these statements after uploading updated
procedure scripts:

```sql
CREATE OR REPLACE PROCEDURE download_kaggle_to_stage()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_kaggle_download';

CREATE OR REPLACE PROCEDURE build_meta_dataset_index()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.build_meta_dataset_index';

CREATE OR REPLACE PROCEDURE run_hpo_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_hpo_job.py')
  HANDLER = 'run_hpo_job.run_hpo_pipeline';

CREATE OR REPLACE PROCEDURE run_model_training()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_model_training_job.py')
  HANDLER = 'run_model_training_job.run_model_training';

CREATE OR REPLACE PROCEDURE run_training_pipeline()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_training_job.py')
  HANDLER = 'run_training_job.run_pipeline';

CREATE OR REPLACE PROCEDURE prepare_benchmark_datasets()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  ARTIFACT_REPOSITORY = snowflake.snowpark.pypi_shared_repository
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python', 'openml==0.15.1')
  EXTERNAL_ACCESS_INTEGRATIONS = (BENCHMARK_EXTERNAL_ACCESS)
  IMPORTS = ('@MODEL_STAGE/scripts/prepare_benchmark_datasets.py')
  HANDLER = 'prepare_benchmark_datasets.prepare_datasets';

CREATE OR REPLACE PROCEDURE run_evaluation_pipeline(
  PREP_RUNTIME_ENVIRONMENT STRING,
  BENCHMARK_RUNTIME_ENVIRONMENT STRING,
  AUTOGLUON_RUNTIME_ENVIRONMENT STRING
)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python')
  IMPORTS = ('@MODEL_STAGE/scripts/run_evaluation_test.py')
  HANDLER = 'run_evaluation_test.run_evaluation_pipeline';
```

Procedure responsibilities:

- `build_meta_dataset_index()` imports `run_training_job.py` and submits `build_meta_dataset_index.py` on `DEEPSET_CPU_POOL`; it rebuilds `META_DATASET_INDEX` from `@META_DATASET_STAGE/{train,val,test}/`.
- `run_hpo_pipeline()` imports `run_hpo_job.py` and submits only the `hpo.py` MLJob/container.
- `run_model_training()` imports `run_model_training_job.py`, reads `@MODEL_STAGE/hpo/best_config.json`, passes it to `train.py` as `BEST_CONFIG`, and submits only the training MLJob/container.
- `run_training_pipeline()` imports `run_training_job.py` and remains a one-call convenience wrapper for HPO plus training.
- `prepare_benchmark_datasets()` imports `prepare_benchmark_datasets.py` and submits a single CPU preparation job that fetches/normalizes OpenML and Kaggle once into `@META_DATASET_STAGE/benchmark_prepared/`.
- `run_evaluation_pipeline()` imports `run_evaluation_test.py`, verifies `best.pt`, `runtime_probe.py`, compute pools, and runtime-image imports by submitting and waiting on 5 serial probes including the CPU baseline probe with `pip_requirements=["catboost"]`, runs single-node GPU synthetic evaluation, always submits benchmark preparation as manifest/index validation, then launches prepared benchmark shards and aggregation.
- `run_evaluation_runtime_probes()` imports `run_evaluation_test.py` and runs only the 5 serial preflight probes without submitting evaluation jobs. Use this to validate runtime environments before a full evaluation run.

Then optionally stage raw Kaggle data once, prepare benchmark datasets, run HPO,
inspect the HPO artifact, run training, verify the checkpoint, and run evaluation:

```sql
CALL download_kaggle_to_stage();
LIST @META_DATASET_STAGE/kaggle/;
CALL prepare_benchmark_datasets();
LIST @META_DATASET_STAGE/benchmark_prepared/;
LIST @META_DATASET_STAGE/benchmark_prepared/ PATTERN='.*benchmark_manifest[.]json';
CALL build_meta_dataset_index();
CALL run_hpo_pipeline();
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/best_config.json (FILE_FORMAT => (TYPE = JSON));
CALL run_model_training();
LIST @MODEL_STAGE/checkpoints/;
CALL run_evaluation_runtime_probes('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');
CALL run_evaluation_pipeline('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');
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
CALL prepare_benchmark_datasets();
LIST @META_DATASET_STAGE/benchmark_prepared/ PATTERN='.*benchmark_manifest[.]json';
```

If `best_config.json` is missing, training has not started; the issue is in HPO.
First confirm the mandatory pretrain checkpoint exists, then check
`@MODEL_STAGE/hpo/hpo_failure.json`:

```sql
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain[.]pt';
LIST @MODEL_STAGE/hpo/ PATTERN='.*hpo_failure[.]json';
SELECT $1 FROM @MODEL_STAGE/hpo/hpo_failure.json (FILE_FORMAT => (TYPE = JSON));
```

Check service logs only if `hpo_failure.json` is missing or incomplete. First
confirm `DEEPSET_GPU_POOL` has capacity for five `GPU_NV_M` nodes, then inspect
the SPCS job service state and container logs:

```sql
SHOW JOB SERVICES IN ACCOUNT;

-- Replace with the exact HPO service name from SHOW JOB SERVICES.
DESC SERVICE TABPFN_DB.TABPFN_SCHEMA.<HPO_SERVICE_NAME>;

SELECT SYSTEM$GET_SERVICE_LOGS(
  'TABPFN_DB.TABPFN_SCHEMA.<HPO_SERVICE_NAME>',
  0,
  'main',
  500
);

LIST @MODEL_STAGE/scripts/;
```

Ray scrape or TSDB compaction warnings are usually noise unless they appear with
real worker failures, trial failures, Python tracebacks, or resource allocation
errors.

For a healthy HPO startup, the logs should include `[HPO] Pretrain checkpoint
found`, `Ray object-store preflight success`, `HPO trial received in-memory records`,
and `[HPO trial] Loaded pretrain checkpoint from Ray object store.`. If driver
materialization fails before trials start, the likely failure domain is driver
Snowpark/session/stage access to `@META_DATASET_STAGE`, not GPU capacity.

If logs are unavailable, record the `SHOW JOB SERVICES` row fields `name`,
`status`, `created_on`, `updated_on`, `current_instances`, `target_instances`,
`compute_pool`, `query_warehouse`, `managing_object_domain`, and
`managing_object_name`. Use `sql/diagnose_training_pipeline_failure.sql` as the
full worksheet before rerunning.

For training failures after `best_config.json` exists, check for
`@MODEL_STAGE/checkpoints/train_failure.json` before service logs:

```sql
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*train_failure[.]json';
SELECT $1 FROM @MODEL_STAGE/checkpoints/train_failure.json (FILE_FORMAT => (TYPE = JSON));
```

For pretrain or final-training startup failures, the Snowflake message
`Multi-node training requires a stage... SF_PYTORCH` is not the fatal error; it
means `artifact_stage_location` was missing or inferred. The production path now
passes `artifact_stage_location="TABPFN_DB.TABPFN_SCHEMA.MODEL_STAGE"` explicitly.
If service logs stop after `Loading data into a pandas dataframe`, suspect
worker-side ShardedDataConnector shard conversion/materialization around
`shard.to_pandas()`. The canonical path is SQL rank sharding over
`META_DATASET_INDEX`; inspect `@MODEL_STAGE/checkpoints/train_failure.json` first,
then service logs if that artifact is missing or incomplete.

If `CALL run_hpo_epoch_test()` fails with
`ModuleNotFoundError: No module named 'train'`, the epoch MLJob source stage is
missing the shared Python modules. Re-upload both source trees to
`@MODEL_STAGE/scripts/`, recreate `run_hpo_epoch_test()` and
`run_train_epoch_test()` from `sql/run_training_job.sql`, then rerun the epoch
call:

```sql
PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
LIST @MODEL_STAGE/scripts/ PATTERN='.*(hpo_epoch_test|train_epoch_test|train|model|snowflake_io)[.]py';
```

`@EPOCH_STAGE` is not a code source for these MLJobs. It only receives
`hpo_timing.json`, `hpo_epoch_error.json`, `train_timing.json`, and
`train_epoch_error.json`. `hpo_timing.json` is a sweep summary with `baseline`,
`runs`, and `summary`, not one scalar epoch-time object. If `hpo_epoch_error.json`
was absent during this failure, Python failed while importing project modules
before the script's older error handler could run.

For full acceptance:

```sql
CALL run_hpo_pipeline();
LIST @MODEL_STAGE/hpo/ PATTERN='.*best_config[.]json';
CALL run_model_training();
LIST @MODEL_STAGE/checkpoints/;
CALL prepare_benchmark_datasets();
LIST @META_DATASET_STAGE/benchmark_prepared/ PATTERN='.*benchmark_manifest[.]json';
CALL run_evaluation_runtime_probes('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');
CALL run_evaluation_pipeline('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');
LIST @EVALUATION_RESULTS_STAGE/;
GET @EVALUATION_RESULTS_STAGE 'file://C:/Documents/TabPFN_DemandModel/results/';
```

Kaggle download uses `compute_pool="DEEPSET_CPU_POOL"` and writes raw `.npz` files
to `@META_DATASET_STAGE/kaggle/`. Benchmark dataset preparation uses one CPU node
and writes `@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json` plus
prepared `.npz` files under `benchmark_prepared/{openml,kaggle}/`. HPO, training,
synthetic evaluation, and MODEL3-ICL-MC benchmark use
`compute_pool="DEEPSET_GPU_POOL"`; baseline benchmark jobs use
`compute_pool="DEEPSET_CPU_POOL"` as 3 combined dataset shard jobs with
`BENCHMARK_METHODS=<all 9 baseline methods>` and bounded concurrency. AutoGluon runs as
separate stacked-ensemble benchmark shard jobs on `AUTOGLUON_CPU_POOL` with
`autogluon.tabular[all]==1.0.0`, `presets="best_quality"`,
`AUTOGLUON_TIME_LIMIT=300`, `num_cpus=BENCHMARK_NUM_CPUS` (default `1`),
`num_gpus=0`, and temporary model
artifacts under `/tmp` cleaned after each fit:
the AutoGluon runtime probe requires `autogluon.tabular`, `numpy`, `pandas`,
`sklearn`, `scipy`, `pyarrow`, `torch`, and `snowflake.snowpark`. It
intentionally excludes XGBoost, LightGBM, and CatBoost. `scipy`, `pyarrow`, and
`torch` are required because AutoGluon shards still execute `evaluate.py`, which
imports those modules at startup.

Benchmark shard jobs are dataset-first: a shard owns a subset of manifest
datasets, loads one prepared `.npz`, evaluates all seeds for that dataset,
releases it, and then continues to the next owned dataset.

| Phase | Entrypoint | Instances | Output |
|---|---|---|---|
| Optional Kaggle raw staging | `download_kaggle_to_stage.py` | 1 CPU | `@META_DATASET_STAGE/kaggle/*.npz` |
| Benchmark dataset preparation | `prepare_benchmark_datasets.py` | 1 CPU | `@META_DATASET_STAGE/benchmark_prepared/benchmark_manifest.json` and prepared `.npz` files |
| HPO | `hpo.py` | 5 | `@MODEL_STAGE/hpo/best_config.json` |
| Training | `train.py` | 10 (DDP, 40 workers) | `@MODEL_STAGE/checkpoints/best.pt` |
| Synthetic evaluation | `evaluate.py` | 1 GPU single-node job | `@EVALUATION_RESULTS_STAGE/synthetic/test_report.csv` and `mc_report.csv` |
| MODEL3-ICL benchmark | `evaluate.py` | 10 GPU shard jobs | `@EVALUATION_RESULTS_STAGE/benchmark_parts/MODEL3-ICL-MC_shard{i}_of_{n}_detailed.csv` |
| Baseline benchmarks | `evaluate.py` | 3 CPU shard jobs; each receives `BENCHMARK_METHODS=<all 9 baseline methods>` | `@EVALUATION_RESULTS_STAGE/benchmark_parts/<method>_shard{i}_of_{n}_detailed.csv` |
| AutoGluon benchmark | `evaluate.py` | 30 CPU_X64_M shard jobs | `@EVALUATION_RESULTS_STAGE/benchmark_parts/AutoGluon_shard{i}_of_{n}_detailed.csv` |
| Aggregate comparison | `evaluate.py` | 1 CPU | `@EVALUATION_RESULTS_STAGE/model_comparison.csv` and `model_comparison_summary.csv` |

Benchmark preparation is always submitted before shards. When the prepared
manifest/index and staged `.npz` payloads already exist, the prep job validates
and exits early, so the expected evaluation work is: five serial runtime probes,
synthetic evaluation, prep validation, 10 MODEL3-ICL benchmark shards, 3 combined
baseline shards, 30 AutoGluon shards, aggregate comparison, and the parent
submission/orchestration job.

#### Evaluation Runtime Reduction Plan

The old baseline topology parallelized by method and dataset shard:
`9 methods × 3 CPU shard jobs = 27` jobs. That repeated dataset load and
preprocessing work for each method. The planned topology parallelizes only by
dataset shard for baselines: 3 CPU shard jobs receive
`BENCHMARK_METHODS=<all 9 baseline methods>`, each shard selects its assigned
dataset indices first, and then runs every baseline method inside that
dataset-first loop.

This is still full-fidelity benchmark execution: no change to datasets, seeds,
method list, `AUTOGLUON_TIME_LIMIT=300`, or AutoGluon `presets="best_quality"`.
It does not batch all datasets into memory, and all benchmark shards remain
single-node jobs. It does not introduce PyTorch distributed execution for
benchmark shards. `BENCHMARK_NUM_CPUS` is for
AutoGluon single-node internal parallelism only and defaults to `1`; sklearn
baselines keep their existing `n_jobs=1` behavior.

Keep the benchmark worker architecture invariant:

- Assigned dataset indices are selected first.
- `load_prepared_dataset(ds_meta)` stays inside the assigned dataset loop.
- Seeds run inside that dataset scope.
- The dataset is deleted and garbage-collected before the next dataset.

Keep the dependency invariant:

- `openml` belongs only to preparation.
- AutoGluon remains lazy-imported only for the AutoGluon method.
- Baseline and DeepSet benchmark shards should not load API/data-fetch dependencies.
- `run_evaluation_test.py` submits all MLJobs with `runtime_environment`.
  Prep probe and prep MLJob pass `pip_requirements=["openml==0.15.1"]` with
  `TABPFN_PYPI_EAI`; the prep MLJob also keeps `BENCHMARK_EXTERNAL_ACCESS` for
  OpenML/Kaggle API access during rebuilds.
  CPU baseline shard jobs pass `pip_requirements=["catboost"]` because the managed
  benchmark runtime (`2.5.0-py311`) does not include `catboost`. This is passed
  **explicitly** in the Phase 4 baseline loop via
  `pip_requirements=list(BASELINE_EXTRA_PIP_REQUIREMENTS)`, not inferred from
  `BENCHMARK_METHODS`. The `_submit_eval()` helper uses a module-level `_UNSET`
  sentinel as the default so that other callers continue using auto-detection
  unchanged. Synthetic, DeepSet, and aggregate jobs pass no `pip_requirements`.
- DeepSet benchmark rows include `raw_features`, `processed_features`,
  `selected_features`, `feature_selector`, and `feature_cap`. Feature capping is
  DeepSet-only; baseline and AutoGluon comparison inputs remain the full processed
  train/test matrices.
- CPU baseline and AutoGluon rows do not cap comparison inputs. If
  `BENCHMARK_CPU_MAX_PROCESSED_FEATURES` or `BENCHMARK_CPU_MAX_MATRIX_BYTES` is
  exceeded, they emit NaN result rows with `skip_reason` before model
  construction. AutoGluon also checks `/tmp` free space with
  `BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES` before creating DataFrames or temp
  model directories. DeepSet feature selection applies the same matrix-risk
  guard before calling `f_regression`.

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

- No container image build or push needed - scripts are read directly from
  `@MODEL_STAGE/scripts/` via `submit_from_stage()`.
- Runtime image: `snowflake/ml-runtime-gpu:latest` (Snowflake-managed).
- Jobs submitted from stored procedures via `run_hpo_job.py`,
  `run_model_training_job.py`, and `run_training_job.py`.

> Scripts are referenced directly from the stage via `source=` in
> `submit_from_stage()`. No Docker image is required or maintained.

### Distributed Training - PyTorchDistributor

- Class: `snowflake.ml.modeling.distributors.pytorch.PyTorchDistributor`
- Manages Ray cluster setup, DDP process group initialization, and result collection
  internally - no manual `torchrun` or rank-environment setup required.
- `PyTorchScalingConfig(num_nodes=10, num_workers_per_node=4, ...)` maps to
  10 x GPU_NV_M nodes (four A10G GPUs each); `run_model_training_job.py` submits training with
  matching `target_instances=10`. Total world_size=40 (10 nodes x 4 workers/node).
- `get_context()` inside `train_fn` provides `local_rank`, `rank`, and `world_size`.
- `train_fn` shards `META_DATASET_INDEX` directly in SQL by DDP `rank` and
  `world_size`, using `ROW_NUMBER() OVER (PARTITION BY split ORDER BY task_id) - 1`
  and `MOD(rn, world_size) = rank`. Each worker materializes only its selected
  train/val stage paths with `materialize_indexed_meta_dataset()`.
- Validation uses a no-padding rank slice, reduces `(sum_loss, total_count)` across
  ranks, and computes exact weighted global MSE before the early-stop check;
  `dist.broadcast(stop, src=0)` propagates the stop signal.

### Hyperparameter Optimization - Ray Tune RandomSearch

- Class: Ray Tune functional API (`tune.run()`).
- Algorithm: Random sampling with FIFO scheduling.
- Search space: `lr`, `weight_decay`, `dropout`.
- Fixed architecture for every trial: `d_phi=128`, `d_rho=256`, `pool="pna"`.
- Ray Tune metric reporting must use metrics-dict style for every trial, e.g.
  `tune.report({"val_mse": value})` or the local compatibility helper. Do not
  use keyword-style `tune.report(val_mse=value)`; Snowflake's Ray runtime can
  raise `TypeError: report() got an unexpected keyword argument 'val_mse'`.
  Keep `tune.run(metric="val_mse", mode="min")` unchanged so best-trial
  selection and downstream artifacts continue to use the `val_mse` key.
- 20 trials, 30-epoch runs each; best config written to
  `@MODEL_STAGE/hpo/best_config.json` on completion. The file includes tuned
  optimizer/dropout values plus the fixed architecture so final training uses
  the same model shape.
- Parallel layout: `run_training_job.py` and `run_hpo_job.py` both submit HPO with
  `target_instances=5`; Ray Tune uses `resources_per_trial={"gpu": 1}`.
  With 5 nodes × 4 A10G GPUs = 20 GPUs total, Ray Tune schedules all 20 trials
  concurrently (no explicit max_concurrent_trials needed); 20 trials / 20 = 1 round x 30 epochs.
  Calibrated with `CALL run_hpo_epoch_test()`, which writes a baseline plus
  marginal sweep summary to `@EPOCH_STAGE/hpo_timing.json`.
- HPO uses a deterministic balanced search subset by default: 200 train tasks
  and 40 validation tasks. Full production training still consumes the full
  train/validation splits after HPO writes `best_config.json`.
- HPO requires `@MODEL_STAGE/checkpoints/pretrain.pt`. Missing checkpoint,
  failed driver download, or checkpoint architecture mismatch is a hard failure;
  trials must log `[HPO trial] Loaded pretrain checkpoint from Ray object store.`.
- Before `tune.run()`, the HPO driver materializes the deterministic 200 train
  and 40 validation parquet payloads, loads them as CPU tensors, downloads
  `pretrain.pt`, and publishes both payloads to the Ray object store. Ray workers
  consume only these object-store payloads and do not create Snowpark sessions.
  A successful startup logs `Ray object-store preflight success`.
- `META_DATASET_INDEX` is the pruning layer over staged parquet payloads, not a
  replacement for `@META_DATASET_STAGE`. It stores one row per task payload and
  is clustered by `(split, hpo_bucket, prior_regime, p, n_train)` so HPO can
  select balanced subsets before materializing data on each node.
- `CALL build_meta_dataset_index();` is the canonical Snowflake-native population
  step. It reads metadata from staged parquet inside an MLJob, truncates/rebuilds
  `META_DATASET_INDEX`, validates `train=800`, `val=100`, `test=100`, and should
  be re-run after synthetic parquet regeneration or restaging.
- Runtime selection is index-backed. HPO ranks rows within `(split, hpo_bucket)`
  by `prior_regime, p, n_train, task_id`, then chooses each split by
  `bucket_rank, hpo_bucket, prior_regime, p, n_train, task_id`. Production
  training selects every `train` and `val` row ordered by `split, task_id`.
- Snowflake jobs fail fast if `META_DATASET_INDEX` is missing, empty, lacks
  required runtime fields (`split`, `task_id`, `stage_path`, `p`, `n_train`),
  or cannot provide the required HPO subset. Local developer runs with no active
  Snowflake session use existing local parquet files instead of downloading
  `@META_DATASET_STAGE`.
- HPO also fails before trials if selected HPO cardinality exceeds the fixed
  architecture. Current local data was verified at `max_p=24`,
  `max_n_train=200`, which is within the fixed `d_phi=128`, `d_rho=256` guard.
- HPO query time is broader than one model epoch. It includes MLJob startup,
  Ray/Tuner scheduling, metadata selection, driver stage materialization,
  object-store payload publication, and epoch compute. The observed
  `CALL run_hpo_epoch_test()` query time was 6m 24s
  (384s), which is the current fixed overhead budget until a newer calibration
  run proves otherwise.
- SPCS MLJob services (`Service Type: JOB`) cannot be dynamically re-scaled after
  submission - `scale_cluster()` was removed from `hpo.py` because it always raises
  error 517003. All GPU parallelism is set at submission time via `target_instances`.
- GPU_NV_M has 4 A10G GPUs per node; `max_concurrent_trials=4` with
  `resource_per_trial={"GPU": 1}` assigns one GPU to each concurrent trial.

Expected HPO success logs:

```text
[HPO] Pretrain checkpoint found
Ray object-store preflight success
HPO trial received in-memory records
[HPO trial] Loaded pretrain checkpoint from Ray object store.
```

If HPO fails before `[HPO trial] Loaded pretrain checkpoint from Ray object store.`
and the logs show parquet materialization errors, investigate driver
Snowpark/session/stage access first. Do not treat this symptom as GPU capacity failure when
`target_instances=5` and `resources_per_trial={"gpu": 1}` are unchanged.
If every Ray trial fails with `TypeError: report() got an unexpected keyword
argument 'val_mse'`, restage patched `src/hpo.py` to `@MODEL_STAGE/scripts/`
and rerun HPO. No `CALL build_meta_dataset_index();` rerun is required for
this reporting-only patch, and no procedure recreation is required if only
`src/hpo.py` changed.
- `TunerResults.best_result` is handled as a one-row DataFrame. Hyperparameters
  are read from `config/<param>` columns, with raw parameter names only as a
  compatibility fallback, then written with the `best_config.json` schema consumed
  by `train.py`: `{lr, weight_decay, d_phi, d_rho, dropout, pool}`.

### Compute Pool & Cost

| Configuration | Credits/node/hr | Nodes | Total cost/hr |
|---|---|---|---|
| GPU_NV_M | 1.42 | 5 (HPO) / 5 (train) | $7.10 / $7.10 |
| previous GPU_NV_S | 0.57 | 2 | ~$2.28-3.42 |
| previous single-node | 2.68 | 1 | ~$5.36-8.04 |

- GPU_NV_M (this design): 1.42 cr/node/hr; 5-node HPO = 7.10 cr/hr; 5-node training = 7.10 cr/hr.
- Pool suspends when idle; no charges in `SUSPENDED` state.

### Estimated End-to-End Cost

| Phase | Nodes | Cost/hr | Duration | Total |
|---|---|---|---|---|
| HPO (20 trials x 30 epochs) | 5 x GPU_NV_M | ~$7.10 | ~31.6 min* | ~$3.74 |
| Full training (DDP) | 5 x GPU_NV_M | ~$7.10 | roughly 25% of HPO wall time | ~$1.87 |
| Evaluation | 1 x GPU_NV_M | ~$1.42 | ~5-10 min | ~$0.12-0.24 |
| **Total** | | | **~45 min plus evaluation** | **~$9.47 plus evaluation** |

\* Calibrated against `@EPOCH_STAGE/hpo_timing.json`; re-estimate from
`summary.mean_epoch_time_s` and `summary.max_epoch_time_s` if sweep timings differ.
Observed 12-way calibration was about 48.1 min mean with overhead and 107.4 min
conservative with overhead. The target 40-way conservative estimate is
`30 * 50.48 + 384 = 1898.4s`, or about 31.6 min.
† With world_size=40 (10 nodes x 4 workers/node), the 800-file train split maps to 20 files/GPU/epoch.

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

from model import ModelConfig, _instantiate_model

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

files_by_split = materialize_indexed_meta_dataset(DATA_DIR, splits=("train", "val"))
train_files = files_by_split["train"]
val_files   = files_by_split["val"]

train_loader = DataLoader(
    ParquetMetaDataset(train_files), batch_size=1, shuffle=True,
    num_workers=4, prefetch_factor=2, pin_memory=USE_AMP, collate_fn=identity_collate,
)
val_loader = DataLoader(
    ParquetMetaDataset(val_files), batch_size=1, shuffle=False,
    num_workers=4, prefetch_factor=2, pin_memory=USE_AMP, collate_fn=identity_collate,
)

# --- Model, compiler, optimizer, scaler ---
from model import ModelConfig, _instantiate_model

cfg = ModelConfig(
    d_phi=128, d_rho=256, pool="pna", n_heads=4,
    n_sab_feat=1,
    norm_feat=True, norm_target=True, dropout=0.1,
    model_family="market_exchangeable_icl",
    model_arch_version="model3",
    model_design_pattern="inductive_forecasting",
)
model     = _instantiate_model(cfg).to(DEVICE)
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
training files x ~4,500 test rows. This serialized inference saturated the CPU and left
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
`torch.compile` wrapper artifacts in the checkpoint.

### 5. Historical GPU_NV_S Compute Pool

This section describes the previous GPU_NV_S design for comparison only. The
current HPO/training runbook uses `GPU_NV_M`: 5 nodes for HPO with 20 concurrent
one-GPU trials (1 round), and 10 nodes for full DDP training with `world_size=40`.

### Cost Comparison

| Configuration | Estimated wall-clock | Notes |
|---|---|---|
| GPU_NV_S, row-by-row, FP32 | ~4 hours | Original |
| GPU_NV_S x 2, batched, BF16, DDP, compile | ~15-25 minutes | Previous optimized |
| GPU_NV_S x 4, batched, BF16, DDP, compile | ~8-13 minutes | Previous 4-node DDP design |

Estimates assume 800 training files x 200 epochs with early stopping at epoch ~100.

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

`phi: R^3 â†’ R^{d_phi}` must be injective - different input triples must produce
different embeddings - so that no two training examples collapse to the same vector
before aggregation. With `d_phi=128` (far larger than the 3-dimensional input), a
trained ReLU MLP is injective on the training manifold by standard covering arguments.

### Continuity

`phi` and `rho` must be continuous: a small perturbation in the input must produce
a small change in the output, so that the aggregated representation varies smoothly.
ReLU networks are piecewise linear and therefore Lipschitz continuous - this
requirement is satisfied by the architecture as-is.

### PNA Pooling and the Sum/Mean Collision Problem

**The problem:** even with an injective `phi`, two *different* multisets can satisfy

```
mean(phi(x) for x in S1)  ==  mean(phi(x) for x in S2),   S1 â‰  S2
```

This "multiset collision" causes the model to map distinct training contexts to the
same latent representation, losing information that is relevant for the prediction.

**The fix - Principal Neighbourhood Aggregation (PNA):** instead of aggregating with
mean alone, concatenate four statistics over the set dimension:

```
pool(S) = cat[ sum_phi, mean_phi, max_phi, std_phi ]   âˆˆ R^{4Â·d_phi}
```

Two sets that share the same mean will generally differ in at least one of sum, max,
or std, yielding a distinct joint embedding. PNA is applied at *both* pooling stages
(feature-level and sample-level), so collisions are suppressed throughout the network.
The learnable equivariance layers (Î», Î³) continue to operate *before* pooling and are
unaffected by this change.

PNA increases the rho input from `d_phi â†’ 4Â·d_phi` and the psi input from
`d_rho â†’ 4Â·d_rho`. The extra parameters are absorbed by rho and psi without changing
the output interface.

### Self-Attention Blocks (SAB)

The simple linear equivariance layer (Î»I + Î³/nÂ·11áµ€) is replaced by one or more
**Self-Attention Blocks** from the Set Transformer (Lee et al. 2019), applied at both
the feature level (features attend to each other per sample) and the sample level
(samples attend to each other before final pooling):

```
X â†’ Ï† â†’ SAB_feat â†’ pool_feat â†’ Ï â†’ SAB_samp â†’ pool_samp â†’ Ïˆ
```

`SAB(X) = MAB(X, X)` where `MAB(Q, K) = LayerNorm(H + FFN(H))` and
`H = LayerNorm(Q + Dropout(MHA(Q, K, K)))`. SAB is permutation equivariant:
`SAB(X[Ï€]) = SAB(X)[Ï€]` for any permutation Ï€ - a strictly more expressive
generalisation of the original Î»/Î³ equivariance. The number of SAB layers is
controlled by `n_sab_feat` in `ModelConfig`.

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
| `multipool` | 5d | Concat[pna, attn] - for ablation |

PNA and multipool are the most expressive. Use `pool="multipool"` to run ablations
comparing all statistics simultaneously. Use `pool="attn"` for the Set Transformer
canonical pooling.

### Normalization Strategy

Two per-context normalizations are applied inside `forward()`:
- **Feature normalization** (`norm_feat=True`): each column of X_train is
  standardised to zero mean and unit variance; the same statistics are applied to
  x_test. This makes the model scale-invariant to feature magnitudes.
- **Target normalization** (`norm_target=True`): y_train is standardised before
  being fed to Ï†; the final prediction is denormalized back to the original scale.
  This removes sensitivity to the absolute scale of the regression target.

Both normalizations use per-context statistics (computed from X_train / y_train
of the current task), not global running statistics - the model requires no warm-up
and works immediately on any new task.

Batch normalization is not used: SPCS runs each meta-dataset as a batch of 1, so
BN statistics would be degenerate, and BN over the set dimension would break
permutation invariance with small sets.

### ModelConfig Hyperparameterization

All hyperparameters are bundled in `ModelConfig` (a `dataclasses.dataclass`):

| Field | Default | Description |
|---|---|---|
| `d_phi` | 128 | phi output dim (â‰¥ p for universality) |
| `d_rho` | 256 | rho output dim (â‰¥ n for universality) |
| `pool` | `"pna"` | Pooling mode (see table above) |
| `n_heads` | 4 | Attention heads for SAB / AttentionPool |
| `n_sab_feat` | 1 | Number of ExchangeableMatrixBlocks |
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
    model=model,
    model_name="MODEL3_ICL_TABPFN_V1",
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
procedure — the full state dict of the MODEL3 ICL model (ColumnEncoder, CellEncoder,
ExchangeableMatrixBlocks, and prediction head).

**Key properties:**
- Stored at `@MODEL_STAGE/checkpoints/best.pt`.
- Created by `torch.save({"state_dict": ..., "cfg": ...}, "best.pt")` whenever val MSE improves.
- Uploaded from the training container via `session.file.put("best.pt", "@MODEL_STAGE/checkpoints/", overwrite=True)`.
- This is the handoff artifact from training to evaluation. Evaluation does not read `@MODEL_STAGE/hpo/best_config.json`.

**Used for inference on any new synthetic dataset without retraining:**

```python
import torch
from model import ModelConfig, _instantiate_model

ckpt = torch.load("best.pt", map_location="cpu", weights_only=True)
cfg_payload = ckpt.get("cfg")
cfg = ModelConfig(**cfg_payload) if isinstance(cfg_payload, dict) else cfg_payload
model = _instantiate_model(cfg)
model.load_state_dict(ckpt["state_dict"])
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
| `@EVALUATION_RESULTS_STAGE/benchmark_parts/<method>_shard{i}_of_{n}_detailed.csv` | Per-method benchmark shard detail | `evaluate.py` |
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
| `SNOWFLAKE_ACCOUNT` | yes | - |
| `SNOWFLAKE_USER` | yes | - |
| `SNOWFLAKE_PASSWORD` | yes | - |
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
from model import ModelConfig, _instantiate_model

ckpt = torch.load("models/best.pt", map_location="cpu", weights_only=True)
cfg_payload = ckpt.get("cfg")
cfg = ModelConfig(**cfg_payload) if isinstance(cfg_payload, dict) else cfg_payload
model = _instantiate_model(cfg)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

### Inspect evaluation results

```python
import pandas as pd
df = pd.read_csv("models/model_comparison.csv")
print(df)
```

## Troubleshooting: Prometheus mmap Panic During MLJob Startup

### Symptom

The Snowflake MLJob fails with status `FAILED`. Container logs contain:

```text
level=ERROR source=query_logger.go msg="Failed to mmap"
component=activeQueryTracker  file=data/queries.active  err="invalid argument"
panic: Unable to create mmap-ed active query log
```

Observed runtime context:
```text
Snowflake Connector for Python: 4.0.0
Python: 3.11.14 / Snowpark: 1.49.0 / Prometheus: 3.5.1
Head Instance IP: 10.244.31.139  Dashboard enabled: true
```

### Root-cause interpretation

The managed Prometheus process panics during Snowflake MLJob/Ray runtime startup. This is a
runtime/infrastructure failure, not a model, DDP, NCCL, dataset, or HPO failure. Do not debug
those layers unless Python training boundary markers prove execution reached them.

### Boundary markers

```text
[train.py main] entered main              → train.py executing in container
[train.py main] starting PyTorchDistributor.run  → distributor about to launch
[train_fn] entered train_fn               → valid to debug DDP/model/dataset layers
[train_fn] topology                       → valid to debug world-size issues
[TRAINING FAILURE JSON]                   → Python exception handler ran
[runtime_probe] entered Python            → probe reached user Python code
[runtime_probe] completed                 → probe finished successfully
```

If `[train.py main] entered main` is absent, do not debug Python training code.

### Stage investigation SQL

```sql
LIST @MODEL_STAGE/checkpoints/;
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*train_failure[.]json';
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*training_submission_started[.]json';
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*best[.]pt';
LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain[.]pt';
```

`training_submission_started.json` present → stored procedure or train.py startup reached upload.
`train_failure.json` absent with Prometheus mmap markers → expected; Python handler did not run.

### Runtime Pinning

Training/HPO/runtime-probe submissions still use the account-managed default
unless explicitly changed. Evaluation submissions are different:
`run_evaluation_test.py` requires `PREP_RUNTIME_ENVIRONMENT`,
`BENCHMARK_RUNTIME_ENVIRONMENT`, and `AUTOGLUON_RUNTIME_ENVIRONMENT` and passes
those values as `runtime_environment`. The 3-runtime architecture is preserved;
`BENCHMARK_RUNTIME_ENVIRONMENT` is used for synthetic eval, DeepSet, CPU baseline
shards, and aggregate. `2.5.0-py311` is the known-good Snowflake-managed benchmark
runtime. It does not include `catboost`, so CPU baseline shard jobs additionally
pass `pip_requirements=["catboost"]`. The prep runtime path additionally installs
`openml==0.15.1` for the prep probe and prep MLJob only. Synthetic, DeepSet, and
aggregate jobs pass no `pip_requirements`. It verifies staged `runtime_probe.py`,
preflights compute pools, and runs 5
`runtime_probe.py` preflight probes (benchmark GPU, benchmark aggregate CPU, CPU
baseline with `pip_requirements=["catboost"]`, prep CPU with
`pip_requirements=["openml==0.15.1"]`, and AutoGluon CPU) before launching
synthetic, prep, benchmark shard, or aggregate work. The probes are deliberately
serialized in both `run_evaluation_pipeline()` and
`run_evaluation_runtime_probes()` because each single-node probe still counts
against the account node quota.

Compute pool preflight (`_preflight_compute_pools`) accepts `SUSPENDED + AUTO_RESUME=TRUE`
without waiting. If all three pools are suspended when `run_evaluation_runtime_probes()` or
`run_evaluation_pipeline()` is called, the preflight passes immediately and `submit_from_stage()`
triggers auto-resume when each probe job is submitted. `SUSPENDED + AUTO_RESUME=FALSE` fails
with a message instructing the user to enable `AUTO_RESUME` or manually resume.

Development cost-control pattern:
- Keep all pools suspended when not actively running jobs.
- Keep `AUTO_RESUME=TRUE` on all pools.
- Use `AUTO_SUSPEND_SECS = 300` (or shorter) for development.
- After each run: `ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;` etc.

Use `CALL run_evaluation_runtime_probes('<prep>', '2.5.0-py311', '<autogluon>');`
to run only the 5 preflight probes without submitting evaluation jobs.

Node quota troubleshooting:
- If a probe fails with `Requested number of nodes 1 exceeds the node limit for the account`,
  first suspend idle compute pools and check current pool state with
  `SHOW COMPUTE POOLS;`.
- Avoid concurrent evaluation probe fan-out under the current quota. A successful
  10-node training run does not imply enough spare quota to run simultaneous
  probes across `DEEPSET_GPU_POOL`, `DEEPSET_CPU_POOL`, and
  `AUTOGLUON_CPU_POOL`.
- Do not change the preflight back to submit-all-then-wait unless the Snowflake
  account node quota has been increased and verified.

### Operational change notes

- 2026-05-09: Evaluation runtime reduction: required prebuilt runtime images via
  [`submit_from_stage(runtime_environment=...)`](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/jobs/snowflake.ml.jobs.submit_from_stage),
  manifest/index validation preflight, 27 baseline jobs consolidated to 3 multi-method dataset
  shards, and single-node CPU parallelism via `BENCHMARK_NUM_CPUS`. Architecture
  remains dataset-first and single-node shard based.
- 2026-05-09: DeepSet evaluation hardening: runtime image probe jobs now run
  before expensive work; bounded contexts are deterministic non-overlapping
  train-only windows; DeepSet-only `train_f_regression` feature selection is
  capped by `model.cfg.d_phi` unless overridden; benchmark rows record feature
  cap metadata.

### Runtime probe workflow

```sql
-- Step 1: single-node probe
CALL run_training_runtime_probe(1);    -- single-node probe

-- Step 2: optional intermediate probes
CALL run_training_runtime_probe(2);    -- 2-node probe (optional)
CALL run_training_runtime_probe(5);    -- 5-node probe (optional)

-- Step 3: full-topology probe (only if step 1 passes)
CALL run_training_runtime_probe(10);   -- full-topology probe

-- Step 4: rerun final training (only if probes pass)
CALL run_model_training();
```

Look for `[runtime_probe] entered Python` and `[runtime_probe] completed` in job logs.
If the probe fails before `[runtime_probe] entered Python` and Prometheus mmap markers appear,
escalate to Snowflake Support.

### Snowflake Support escalation template

```
Snowflake MLJob / Ray runtime startup failure — GPU compute pool.

Job fails before Python entrypoint execution. Container logs show Prometheus panic:
  Failed to mmap / component=activeQueryTracker / file=data/queries.active / err="invalid argument"
  panic: Unable to create mmap-ed active query log

No Python boundary markers appear ([train.py main] entered main, [train_fn] entered train_fn).

Compute pool: DEEPSET_GPU_POOL (GPU_NV_M)
Target instances: 10 (final training) / 1 (single-node probe)
Training topology: 10 nodes × 4 workers = 40 world size
Connector: 4.0.0 / Python: 3.11.14 / Snowpark: 1.49.0 / Prometheus: 3.5.1
runtime_environment: training not pinned here; evaluation uses required runtime image env vars

Request: Please confirm whether the managed Prometheus data directory supports mmap for
active query tracking, and whether there is a supported way to disable Ray Dashboard /
Prometheus or redirect data/queries.active to an mmap-compatible path such as /tmp.
```

### Canonical topology (do not change without instruction)

- Final training: 10 nodes × 4 workers per node = 40 expected workers
- Pretraining: 10 nodes × 4 workers per node = 40 expected workers
- HPO: 5 nodes × 4 concurrent trials per node = 20 total trial slots / 20 total trials

---

## pip Dependencies on Snowflake-Managed Container Runtime (2.5.0-py311)

### What is missing from the managed image

The `2.5.0-py311` Snowflake-managed runtime does not include several third-party packages
required by this project. They must be installed per-job via `pip_requirements` +
`external_access_integrations`.

| Package | Version pinned | Required by |
|---------|---------------|-------------|
| `catboost` | `1.2.10` | CPU baseline probe (probe index 2), all CPU baseline shard jobs |
| `openml` | `0.15.1` | prep CPU runtime probe (probe index 3), benchmark dataset prep MLJob |
| `autogluon.tabular` | `1.3.0` | AutoGluon CPU runtime probe (probe index 4), all AutoGluon shard jobs |

The direct `CALL prepare_benchmark_datasets()` stored procedure still lists
`openml==0.15.1` in its `PACKAGES` clause. The evaluation MLJob path installs the
same pinned version with `pip_requirements` so the prep runtime probe and prep
MLJob can import OpenML without baking it into every runtime image.

### TABPFN_PYPI_EAI design

A single general-purpose External Access Integration (`TABPFN_PYPI_EAI`) covers all
pip installs. This replaces the former `TABPFN_CATBOOST_PYPI_EAI` (which was
CatBoost-specific). The integration uses `SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE`.

`BENCHMARK_EXTERNAL_ACCESS` is a **separate** concern: it allows runtime calls to
external APIs (OpenML, Kaggle). The dataset prep MLJob needs `BENCHMARK_EXTERNAL_ACCESS`
for API downloads and `TABPFN_PYPI_EAI` to install `openml==0.15.1`. Benchmark
shard and aggregate jobs must not receive the OpenML dependency.

### Job × dependency table

| Job | `pip_requirements` | `external_access_integrations` |
|-----|--------------------|-------------------------------|
| GPU benchmark probe (0) | — | — |
| CPU benchmark probe (1) | — | — |
| CPU baseline probe (2) | `catboost==1.2.10` | `TABPFN_PYPI_EAI` |
| prep CPU runtime probe (3) | `openml==0.15.1` | `TABPFN_PYPI_EAI` |
| AutoGluon CPU runtime probe (4) | `autogluon.tabular==1.3.0` | `TABPFN_PYPI_EAI` |
| `prepare_benchmark_datasets.py` (ML Job) | `openml==0.15.1` | `BENCHMARK_EXTERNAL_ACCESS`, `TABPFN_PYPI_EAI` |
| CPU baseline shard jobs (×3) | `catboost==1.2.10` | `TABPFN_PYPI_EAI` |
| AutoGluon shard jobs (×30) | `autogluon.tabular==1.3.0` | `TABPFN_PYPI_EAI` |
| DeepSet GPU shard jobs (×10) | — | — |
| Synthetic eval job | — | — |
| Aggregate job | — | — |

### Creating and granting the integration

Run once as ACCOUNTADMIN (or a role with `CREATE INTEGRATION` and access to
`SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE`):

```sql
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI
  ALLOWED_NETWORK_RULES = (SNOWFLAKE.EXTERNAL_ACCESS.PYPI_RULE)
  ENABLED = TRUE
  COMMENT = 'Allows TabPFN ML Jobs to install approved PyPI dependencies in Snowflake-managed Container Runtime.';

GRANT USAGE ON INTEGRATION TABPFN_PYPI_EAI TO ROLE ACCOUNTADMIN;
```

The full DDL is in `sql/run_training_job.sql` (Step 2c).

### Validation steps

After staging the updated script and recreating stored procedures:

```sql
-- 1. Verify the integration exists and is enabled.
SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'TABPFN_PYPI_EAI';
DESC EXTERNAL ACCESS INTEGRATION TABPFN_PYPI_EAI;
SHOW GRANTS ON INTEGRATION TABPFN_PYPI_EAI;

-- 2. Run all 5 preflight probes end-to-end.
CALL run_evaluation_runtime_probes('<prep_runtime>', '2.5.0-py311', '<autogluon_runtime>');
-- All 5 probes must pass, including:
--   probe 3 (prep): openml,numpy,snowflake.snowpark must import successfully
--     (openml is installed by prep-probe pip_requirements)
--   probe 4 (AutoGluon): autogluon.tabular 1.3.0 must import successfully
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'openml'` in `CALL prepare_benchmark_datasets()` | `openml` missing from stored procedure `PACKAGES` clause | Re-run `CREATE PROCEDURE prepare_benchmark_datasets()` DDL with `'openml==0.15.1'` in `PACKAGES`; the fix is in `sql/run_training_job.sql` |
| `ModuleNotFoundError: No module named 'openml'` in prep probe or evaluation prep MLJob | Prep job missing `pip_requirements`/EAI | Verify prep probe and prep MLJob pass `openml==0.15.1` with `TABPFN_PYPI_EAI` |
| `ModuleNotFoundError: No module named 'autogluon'` in shard job | AutoGluon shard job missing `pip_requirements`/EAI | Verify AutoGluon shard loop passes both kwargs |
| `Integration 'TABPFN_PYPI_EAI' does not exist` | EAI not created or wrong name | Re-run Step 2c in `sql/run_training_job.sql` |
| Probe passes but shard job fails install | Version pin mismatch between constants and PyPI availability | Check `AUTOGLUON_VERSION` / `OPENML_VERSION` constants |

---

## OOD Full Suite Evaluation

The OOD full suite (`ood_linear_full_v1`) runs the complete evaluation pipeline (DeepSet +
baselines + AutoGluon) on 200 OOD datasets across 4 regimes (E/F/G/H, 50 per regime).

### Dataset counts

| Pool | Count | Description |
|------|-------|-------------|
| Source pool | 200 staged parquet files | 50 per regime E/F/G/H; generated locally, staged to `@EVALUATION_DATASET_STAGE/ood_parity/` |
| Pilot indexed | 80 (20/regime) | Indexed under `ood_linear_pilot_v1`; DeepSet only |
| Full suite indexed | 200 (50/regime) | Indexed under `ood_linear_full_v1`; all methods |

### Step 1 — Generate 200 OOD parquet files locally

`generate_ood_eval_data.py` is a **local-only CLI** and must **never** be staged to Snowflake.

```bash
python scripts/ood_regression/generate_ood_eval_data.py --n_datasets 200
```

Output: `data/ood_regression/{E,F,G,H}/dataset_NNNN.parquet` and `data/ood_regression/ood_manifest.json`

### Step 2 — Stage all 200 OOD parquet files

```sql
PUT file://data/ood_regression/E/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/E/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://data/ood_regression/F/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/F/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://data/ood_regression/G/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/G/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://data/ood_regression/H/*.parquet @EVALUATION_DATASET_STAGE/ood_parity/H/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://data/ood_regression/ood_manifest.json @EVALUATION_DATASET_STAGE/ood_parity/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

### Step 3 — Stage updated Python scripts

```sql
PUT file://C:/Documents/TabPFN_DemandModel/src/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://C:/Documents/TabPFN_DemandModel/scripts/*.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://scripts/ood_regression/prepare_ood_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://scripts/prepare_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

### Step 4 — Call the OOD full evaluation procedure

```sql
CALL run_synthetic_regression_ood_full_evaluation('2.5.0-py311', '2.5.0-py311');
```

This runs 5 phases sequentially:
1. **OOD prep** — indexes 200 datasets under `ood_linear_full_v1`
2. **DeepSet** — 10 GPU shards on `DEEPSET_GPU_POOL`
3. **Baselines** — 6 CPU shards on `DEEPSET_CPU_POOL`
4. **AutoGluon** — 60 CPU single-node shards on `AUTOGLUON_CPU_POOL` (legacy path)
5. **Aggregation** — 1 CPU job; outputs to `@EVALUATION_RESULTS_STAGE/ood_full/`

> Note: The combined suite (`linear_all_v1`) uses a different AutoGluon path: 6 distributed
> work-item clusters × 4 workers via `autogluon_ray.py`.
> See the Combined Suite section below.

### Step 5 — Verify index

```sql
SELECT prior_regime, COUNT(*) AS n
FROM SYNTHETIC_REGRESSION_DATASET_INDEX
WHERE suite_id = 'ood_linear_full_v1'
GROUP BY prior_regime ORDER BY prior_regime;
-- Expected: E/F/G/H each with 50 rows
```

### Step 6 — Verify outputs

```sql
LIST @EVALUATION_RESULTS_STAGE/ood_full/;
-- Expected files:
--   synthetic_regression_model_comparison.csv
--   synthetic_regression_model_comparison_summary.csv
--   synthetic_regression_summary_by_regime.csv
--   synthetic_regression_chart_data_model_rank.csv
```

### Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OOD_REGRESSION_N_DATASETS` | 80 | Preferred: number of OOD datasets to index (must be divisible by 4) |
| `OOD_REGRESSION_N_PILOT` | — | Legacy fallback for `OOD_REGRESSION_N_DATASETS`; still accepted |
| `OOD_REGRESSION_SUITE_ID` | `ood_linear_pilot_v1` | Suite ID for indexed rows |

---

## Combined Suite Evaluation (linear_all_v1)

The combined suite (`linear_all_v1`, 400 datasets) is an index-level composition of the primary
in-distribution suite (`linear_poisson_v1_recommended`, regimes A/B/C/D, 200 datasets) and the
OOD full suite (`ood_linear_full_v1`, regimes E/F/G/H, 200 datasets). No parquet files are merged
or rewritten; `prepare_combined_suite()` copies rows into `SYNTHETIC_REGRESSION_DATASET_INDEX`.

**Prerequisites:** both source suites must be indexed before running combined prep.

### Baseline shard count (runtime-configurable)

Baseline shards are runtime-configurable. Default remains 6. 1 baseline shard = 1 single-node
MLJob = 1 output shard file.

| Parameter / env var | Default | Description |
|---------------------|---------|-------------|
| `SYNREG_BASELINE_SHARDS` / `BASELINE_SHARDS` SQL arg | `SYNREG_CPU_SHARDS` = 6 | Number of baseline shard files written; controls MLJob count |
| `SYNREG_BASELINE_CONCURRENT_NODES` / `BASELINE_CONCURRENT_NODES` SQL arg | 6 | Required single-wave CPU nodes; **must equal `BASELINE_SHARDS`** |

**Guardrails:** `BASELINE_CONCURRENT_NODES` must exactly equal `BASELINE_SHARDS`. Lower values are
rejected (no silent batching). Higher values are also rejected unless `BASELINE_SHARDS` is raised
to match. Aggregation must expect `SYNREG_EXPECTED_BASELINE_SHARDS` matching the resolved shard
count; the all-in-one `run_synthetic_regression_combined_evaluation` wires this automatically.

**To run with 10 baseline shards:**

```sql
CALL run_synthetic_regression_combined_baseline_capacity_probe(
  '2.5.0-py311', '2.5.0-py311', 10, 10
);
CALL run_synthetic_regression_combined_baseline_evaluation(
  '2.5.0-py311', 10, 10
);
-- Aggregation must expect 10 baseline shard files:
CALL run_synthetic_regression_combined_aggregation(
  '2.5.0-py311', 6, 10, 10
);
```

### AutoGluon distributed work-item architecture

The combined suite uses a distributed AutoGluon evaluation path instead of the legacy 60-shard
single-node path used by the main and OOD suites.

**Default topology (6 clusters × 4 workers = 24 concurrent CPU_X64_M nodes):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYNREG_AUTOGLUON_CLUSTER_SHARDS` | 6 | Number of logical shard files written |
| `SYNREG_AUTOGLUON_WORKERS_PER_SHARD` | 4 | `target_instances` per MLJob cluster |
| `AUTOGLUON_TASK_CPUS` | 1 | CPUs per individual AutoGluon fit task |
| `SYNREG_AUTOGLUON_CONCURRENT_CLUSTERS` | 6 | Required single-wave MLJob clusters; must equal cluster shards |
| `SYNREG_AUTOGLUON_MAX_IN_FLIGHT` | 4 | Max Ray work items submitted before collecting results |
| `BENCHMARK_AUTOGLUON_MIN_TMP_FREE_BYTES` | 5368709120 | Per-task `/tmp` free-space guard before fitting |
| `BENCHMARK_AUTOGLUON_MAX_DATASET_BYTES` | 2147483648 | Per-worker dataset size guard (worker-local load) |
| `BENCHMARK_CPU_MAX_PROCESSED_FEATURES` | 512 | Per-task processed feature cap |
| `BENCHMARK_CPU_MAX_MATRIX_BYTES` | 2147483648 | Per-task train+holdout matrix byte cap |
| `SYNREG_AUTOGLUON_DISTRIBUTED_MODE` | `ray_work_items` | Distribution strategy |
| `SYNREG_WORKER_DATA_ACCESS_MODE` | `driver_presigned_url` | Driver-derived presigned HTTPS URL; workers download via urllib without a Snowpark session |
| `SYNREG_MAX_WORK_ITEM_BYTES` | 8192 | Compact Ray item metadata size guard |

Each MLJob cluster runs `autogluon_ray.py` (derived internally — not a runtime argument). The
entrypoint calls `ray.init(address="auto")` to attach to the Snowflake-provisioned multi-node
Ray cluster and fails before writing a CSV if fewer than `SYNREG_AUTOGLUON_WORKERS_PER_SHARD`
Ray nodes are alive. The driver process:
1. Loads `SYNTHETIC_REGRESSION_DATASET_INDEX` for `suite_id=linear_all_v1`
2. Expands rows to explicit `(dataset, split_seed, condition)` work items
3. Assigns this cluster's shard using `assign_synthetic_regression_shard(items, shard_index, num_shards)`
4. Builds compact JSON item dicts and derives `dataset_access.scoped_url` with `BUILD_SCOPED_FILE_URL`
5. Distributes only compact item dicts across Ray tasks; each worker loads its own dataset with `SnowflakeFile.open(scoped_url)` (no `ray.put`, no worker Snowpark session)
6. Writes exactly one file: `AutoGluon_shard{shard_index}_of_{num_shards}_detailed.csv`

Workers never query `SYNTHETIC_REGRESSION_DATASET_INDEX` and do not create Snowpark sessions.
The only worker data access surface is the scoped URL in the item dict produced by the driver.

Aggregation expects `SYNREG_EXPECTED_AG_SHARDS=6` (matching `SYNREG_AUTOGLUON_CLUSTER_SHARDS`).

### Step-by-step runbook

```sql
-- Prerequisites: linear_poisson_v1_recommended and ood_linear_full_v1 must be indexed.
-- Required operational sequence:
--   1. runtime probes
--   2. combined AutoGluon capacity probe
--   3. combined AutoGluon worker-access probe
--   4. combined AutoGluon evaluation
--   5. aggregation

-- Step 0: Stage updated scripts (SnowSQL only)
-- PUT file://scripts/run_synthetic_regression_evaluation.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/autogluon_ray.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/ray_capacity_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/autogluon_worker_access_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/autogluon_import_timing_probe.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://scripts/prepare_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file://src/evaluate_synthetic_regression.py @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- Verify:
-- LIST @MODEL_STAGE/scripts/ PATTERN='.*(run_synthetic_regression_evaluation|evaluate_synthetic_regression|autogluon_ray|capacity_probe|ray_capacity_probe|autogluon_worker_access_probe|autogluon_import_timing_probe)[.]py';

-- Step 0a: Runtime probes
CALL run_synthetic_regression_runtime_probes('2.5.0-py311', '2.5.0-py311', '2.5.0-py311');

-- Step 1: Capacity probes (verify node envelope before evaluation)
-- Ray readiness knobs:
--   capacity probe default: SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS=300, POLL_SECONDS=10
--   evaluation default:     SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS=600, POLL_SECONDS=10
-- Use the extended overloads below to override these per run.
CALL run_synthetic_regression_combined_baseline_capacity_probe(
  '2.5.0-py311', '2.5.0-py311', 6
);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

CALL run_synthetic_regression_combined_autogluon_capacity_probe(
  '2.5.0-py311', '2.5.0-py311',
  6,     -- AUTOGLUON_CLUSTER_SHARDS
  4,     -- AUTOGLUON_WORKERS_PER_SHARD
  6,     -- AUTOGLUON_CONCURRENT_CLUSTERS
  300,   -- RAY_READY_TIMEOUT_SECONDS
  10     -- RAY_READY_POLL_SECONDS
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

CALL run_synthetic_regression_combined_autogluon_worker_access_probe(
  '2.5.0-py311', '2.5.0-py311', 6, 4, 6
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Optional: AutoGluon import timing probe — measures dependency bootstrap latency.
-- Time from MLJob submission to python_entrypoint_started approximates scheduling +
-- image startup + pip install (pip mode) or just scheduling + image startup (no-pip).
-- No-pip mode skips AutoGluon/Ray imports and emits *_import_skipped events.
-- autogluon_import_complete.import_seconds measures import overhead only in pip/preinstalled modes.
-- Compare pip vs no-pip waves to estimate bootstrap overhead under concurrency.
--
-- Single pip-mode probe (default):
CALL run_synthetic_regression_autogluon_import_timing_probe('2.5.0-py311');
--
-- 8 concurrent pip-mode probes (simulates full evaluation wave concurrency):
-- CALL run_synthetic_regression_autogluon_import_timing_probe('2.5.0-py311', TRUE, 8);
--
-- 8 concurrent no-pip probes (scheduling + image startup baseline; skips AutoGluon/Ray imports):
-- CALL run_synthetic_regression_autogluon_import_timing_probe('2.5.0-py311', FALSE, 8);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Step 2: Combined prep (index composition)
CALL run_synthetic_regression_combined_prep('2.5.0-py311', '2.5.0-py311');
-- Verify (expect A/B/C/D/E/F/G/H each with 50 rows, total 400):
-- SELECT prior_regime, COUNT(*) AS n
-- FROM SYNTHETIC_REGRESSION_DATASET_INDEX
-- WHERE suite_id = 'linear_all_v1'
-- GROUP BY prior_regime ORDER BY prior_regime;

-- Step 3: DeepSet evaluation (10 GPU shards → 10 MODEL3-ICL shard files)
CALL run_synthetic_regression_combined_deepset_evaluation('2.5.0-py311');
ALTER COMPUTE POOL DEEPSET_GPU_POOL SUSPEND;

-- Step 4: Baseline evaluation (6 CPU shards)
CALL run_synthetic_regression_combined_baseline_evaluation('2.5.0-py311', 6);
ALTER COMPUTE POOL DEEPSET_CPU_POOL SUSPEND;

-- Step 5: Distributed AutoGluon evaluation (6 clusters × 4 workers → 6 shard files)
CALL run_synthetic_regression_combined_autogluon_evaluation(
  '2.5.0-py311',
  '2.5.0-py311',
  6,     -- AUTOGLUON_CLUSTER_SHARDS
  4,     -- AUTOGLUON_WORKERS_PER_SHARD
  1,     -- AUTOGLUON_TASK_CPUS
  6,     -- AUTOGLUON_CONCURRENT_CLUSTERS
  300,   -- AUTOGLUON_TIME_LIMIT_SECONDS
  'best_quality',
  600,   -- RAY_READY_TIMEOUT_SECONDS
  10     -- RAY_READY_POLL_SECONDS
);
ALTER COMPUTE POOL AUTOGLUON_CPU_POOL SUSPEND;

-- Step 6: Aggregation (expects N=6 AutoGluon shard files)
CALL run_synthetic_regression_combined_aggregation('2.5.0-py311', 6);

-- Step 7: Final model training with explicit runtime lineage
CALL run_model_training(
  'market_exchangeable_icl',
  'synthetic_regression_combined',
  'inductive_forecasting'
);
```

### Output verification

```sql
-- Verify 6 AutoGluon shard files were written:
LIST @EVALUATION_RESULTS_STAGE/regression/linear_all_v1/
  PATTERN='.*AutoGluon_shard[0-9]+_of_6_detailed[.]csv';

-- Verify combined aggregation outputs:
LIST @EVALUATION_RESULTS_STAGE/combined/;
-- Expected:
--   synthetic_regression_model_comparison.csv
--   synthetic_regression_model_comparison_summary.csv
--   synthetic_regression_summary_by_regime.csv
--   synthetic_regression_chart_data_model_rank.csv

-- Read combined summary:
SELECT $1
FROM @EVALUATION_RESULTS_STAGE/combined/synthetic_regression_model_comparison_summary.csv;
```

---

## MODEL3 DDP Memory Probe

`run_model_ddp_memory_probe()` is a deployment-safety probe that measures peak CUDA
memory per DDP worker for representative MODEL3 ICL shapes **before** pretrain / HPO /
final training.  Because MODEL3 meta-training uses back-propagation through the full
forward graph, always run with `RUN_BACKWARD=TRUE` to get a faithful peak-memory
measurement that covers gradient tensors and activation storage during backprop.

### When to run

Run before every new MODEL3 training experiment when:
- Increasing `N_CONTEXT`, `P_FEATURES`, `M_QUERY`, `D_PHI`, or `N_BLOCKS`.
- Testing a new `d_phi` value from the HPO search space.
- Deploying MODEL3 on a new compute pool configuration.

### Procedure signature

```sql
CALL run_model_ddp_memory_probe(
    MODEL_ARCH_VERSION   STRING,    -- must be 'model3'
    MODEL_DESIGN_PATTERN STRING,   -- 'inductive_forecasting' (transductive not yet supported)
    MODEL_FAMILY          STRING,   -- 'market_exchangeable_icl'
    N_CONTEXT  INTEGER,             -- context rows (n)
    P_FEATURES INTEGER,             -- features (p)
    M_QUERY    INTEGER,             -- query batch size (m)
    D_PHI      INTEGER,             -- channel dimension (d_phi / d_model)
    N_BLOCKS   INTEGER,             -- number of ExchangeableMatrixBlocks
    RUN_BACKWARD BOOLEAN            -- always TRUE for training-regime validation
);
```

### Canonical call (production defaults)

```sql
CALL run_model_ddp_memory_probe(
    'model3',
    'inductive_forecasting',
    'market_exchangeable_icl',
    200,    -- N_CONTEXT
    128,    -- P_FEATURES
    128,    -- M_QUERY
    128,    -- D_PHI
    1,      -- N_BLOCKS
    TRUE    -- RUN_BACKWARD
);
```

### Reading results

```sql
LIST @MODEL_STAGE/diagnostics/;

SELECT $1
FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json
  (FILE_FORMAT => (TYPE = JSON));
```

### JSON result schema

| Field | Description |
|-------|-------------|
| `status` | `"ok"` \| `"error"` \| `"skipped_static_memory_guard"` |
| `shape.n_context` … `shape.run_backward` | Probe input shape and flags |
| `static_estimate.h_tensor_bytes` | Bytes for H: (m, n, p, d_phi) |
| `static_estimate.estimated_reserved_bytes` | H × activation_factor × safety_factor |
| `static_estimate.activation_factor` | 20 (backward) or 8 (forward-only) |
| `summary.max_peak_reserved_bytes` | Max peak reserved across all 40 workers |
| `summary.max_reserved_fraction` | max_peak_reserved / cuda_total |
| `ranks[*].peak_memory_reserved_bytes` | Per-worker peak reserved bytes |
| `ranks[*].backward_ok` | Whether backward pass succeeded on that worker |

### Static memory guard

Before allocating any tensors the probe estimates
`H_bytes = m * n * p * d_phi * 4` (float32) and then
`estimated_reserved = H_bytes * 20 * 1.5` (forward+backward, safety_factor=1.5).

If this estimate exceeds `CUDA_total × MODEL_PROBE_MAX_GPU_MEMORY_FRACTION` (default 0.9),
the probe emits `status="skipped_static_memory_guard"` without allocating tensors and
raises `RuntimeError` if `MODEL_PROBE_STRICT_MEMORY_GUARD=true` (default).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROBE_N_CONTEXT` | `200` | Context rows |
| `MODEL_PROBE_P_FEATURES` | `128` | Features |
| `MODEL_PROBE_M_QUERY` | `128` | Query batch size |
| `MODEL_PROBE_D_PHI` | `128` | Channel dimension |
| `MODEL_PROBE_N_BLOCKS` | `1` | ExchangeableMatrixBlocks |
| `MODEL_PROBE_RUN_BACKWARD` | `true` | Run backward pass (always true for training-regime validation) |
| `MODEL_PROBE_DTYPE` | `float32` | Tensor dtype (`float32` or `bfloat16`) |
| `MODEL_PROBE_MAX_GPU_MEMORY_FRACTION` | `0.9` | Static guard threshold as fraction of GPU total |
| `MODEL_PROBE_MAX_TENSOR_BYTES` | — | Hard byte cap (optional override) |
| `MODEL_PROBE_MEMORY_SAFETY_FACTOR` | `1.5` | Overhead multiplier for reserved estimate |
| `MODEL_PROBE_STRICT_MEMORY_GUARD` | `true` | Raise if guard triggers |
| `MODEL_PROBE_OUTPUT_STAGE` | `@MODEL_STAGE/diagnostics/` | Upload destination |

## AutoGluon SPCS Custom Image Backend

### Migration rationale

The default MLJob backend installs AutoGluon via `pip_requirements` at every container startup.
On a 6-shard × 4-worker deployment this means 24 concurrent pip installs of `autogluon.tabular==1.3.0`,
each taking 3–8 minutes. The SPCS custom-image backend eliminates this overhead by preinstalling
AutoGluon, Ray, and all dependencies into a Docker image at build time.

Key differences from the MLJob backend:

| Property | MLJob (`mljob`) | SPCS (`spcs_job`) |
|---|---|---|
| `runtime_environment` | Required (e.g. `2.5.0-py311`) | **Not used** |
| `pip_requirements` | `autogluon.tabular==1.3.0`, `ray` | **Not used** |
| AutoGluon source | pip install at startup | Preinstalled in OCI image |
| Ray topology | Snowflake-managed multi-instance | **Self-managed** head + workers |
| Ray address mode | `auto` | `explicit` (RAY_HEAD_ADDRESS) |
| SPCS image required | No | Yes (`SYNREG_AUTOGLUON_SPCS_IMAGE`) |

### Architecture (text diagram)

```
SPCS Job Services per shard (Ray distributed mode):

  [SPCS coordinator]       spcs_ray_coordinator.py
    subprocess 1: ray start --head --num-cpus=0 --object-store-memory=...
                            --resources={"spcs_cluster_id_<run_id>_<shard>": 1}
    subprocess 2: autogluon_ray.py
                            (SYNREG_RAY_ADDRESS_MODE=explicit,
                             RAY_HEAD_ADDRESS=localhost:<port>)
    TCP endpoint: ray-head → port 6379 (exposed in SPCS spec)

          | <-- RAY_HEAD_ADDRESS (<coordinator_dns>.<SPCS_RAY_HEAD_DNS_SUFFIX>:<port>)
          |     SPCS DNS rule: underscores in service name → dashes
          |     e.g. spcs_ray_coord_r0_0 → spcs-ray-coord-r0-0.<suffix>

  [SPCS worker 0]          spcs_ray_worker.py
                            (--num-cpus=<AUTOGLUON_TASK_CPUS>
                             --object-store-memory=<SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES>)
  [SPCS worker 1]          spcs_ray_worker.py
  ...
```

Single-node mode (AUTOGLUON_CLUSTER_SHARDS=0): one SPCS container per shard running
`evaluate_synthetic_regression.py`. No Ray, no coordinator/worker containers.

Default coordinator topology: 6 coordinators + 24 workers = **30 SPCS containers**
for a 6×4 deployment. The coordinator merges the Ray head and AutoGluon driver into
one container: it starts `ray start --head --num-cpus=0` as a subprocess, waits for
the head to become reachable on localhost, then runs `autogluon_ray.py` with
`RAY_HEAD_ADDRESS=localhost:<port>`. Workers connect to the coordinator's external
DNS address. Only the 24 worker containers provide schedulable Ray CPUs.

Default SPCS resource profiles:

| Role | CPU request | CPU limit | Memory request | Memory limit |
| --- | ---: | ---: | ---: | ---: |
| Ray coordinator | 1 | 2 | 4Gi | 8Gi |
| Ray worker | 4 | 4 | 16Gi | 16Gi |
| Single-node AutoGluon | 4 | 4 | 16Gi | 16Gi |
| Import/session probe | 0.5 | 0.5 | 2Gi | 2Gi |

Override with `SYNREG_SPCS_RAY_COORDINATOR_CPU`, `SYNREG_SPCS_RAY_COORDINATOR_MEMORY`,
`SYNREG_SPCS_RAY_WORKER_CPU`, `SYNREG_SPCS_RAY_WORKER_MEMORY`,
`SYNREG_SPCS_SINGLE_NODE_CPU`, or `SYNREG_SPCS_SINGLE_NODE_MEMORY`. To separate
requests from limits, use the same prefixes with `_CPU_REQUEST`, `_CPU_LIMIT`,
`_MEMORY_REQUEST`, and `_MEMORY_LIMIT`.

Object store memory overrides: `SYNREG_SPCS_RAY_COORDINATOR_OBJECT_STORE_MEMORY_BYTES`
(default 500 MB) and `SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES` (default 2 GB).

### Step 0 — Create the image repository (once)

```sql
-- See sql/create_autogluon_spcs_image_repository.sql
CREATE IMAGE REPOSITORY IF NOT EXISTS AUTOGLUON_IMAGE_REPOSITORY;
SHOW IMAGE REPOSITORIES;  -- note repository_url
```

### Step 1 — Build and push the Docker image

```bash
# Build for linux/amd64 (required for Snowflake SPCS)
docker build --platform linux/amd64 \
  -f docker/autogluon/Dockerfile \
  -t tabpfn-autogluon-ray:1.0.0 .

# Health check (optional)
docker run --rm tabpfn-autogluon-ray:1.0.0 \
  -c "import ray, autogluon.tabular; print('ok')"

# Push to Snowflake image repository
docker login <account>.registry.snowflakecomputing.com

docker tag tabpfn-autogluon-ray:1.0.0 \
  <repository_url>/tabpfn-autogluon-ray:1.0.0

docker push <repository_url>/tabpfn-autogluon-ray:1.0.0
```

Verify the image is available:

```sql
SHOW IMAGES IN IMAGE REPOSITORY AUTOGLUON_IMAGE_REPOSITORY;
```

### Step 2 — Configure environment variables

```bash
export SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job
export SYNREG_AUTOGLUON_SPCS_IMAGE=<repository_url>/tabpfn-autogluon-ray:1.0.0
```

### Step 3 — SPCS import timing probe

Validates that the custom image starts and all imports succeed. Unlike the MLJob probe,
no pip install occurs; this measures pure scheduling + container startup latency.

```sql
CALL run_synthetic_regression_autogluon_spcs_import_probe('spcs_job', 1);
```

### Step 4 — SPCS session probe (mandatory)

Validates that the SPCS job service receives the Snowflake OAuth token automatically at
`/snowflake/session/token` and that Snowpark session creation succeeds inside a container.
This is required before the capacity and worker-access probes — if the session probe
fails, all subsequent probes that need dataset-stage access will also fail.

```sql
CALL run_synthetic_regression_autogluon_spcs_session_probe('spcs_job', 1);
```

Expected output: `session probe ok` with `account=<account>` and `role=<role>` in the container
log. If the probe fails with a token error, verify that the compute pool and service network
policy allow the container to reach the Snowflake token endpoint. The spec must NOT contain a
`snowflakeService` block — Snowflake rejects it with error 395018.

### Step 5 — SPCS capacity probe

Verifies that the custom image starts correctly on the compute pool and Ray is importable.

```sql
-- Single-node mode (AUTOGLUON_CLUSTER_SHARDS=0): 2 concurrent containers
CALL run_synthetic_regression_combined_autogluon_spcs_capacity_probe(
  'spcs_job',   -- AUTOGLUON_RUNTIME_ENVIRONMENT (ignored; present for signature compat)
  0,            -- AUTOGLUON_CLUSTER_SHARDS (0 = single-node mode)
  1,            -- AUTOGLUON_WORKERS_PER_SHARD
  2             -- AUTOGLUON_CONCURRENT_CLUSTERS
);
```

### Step 6 — SPCS worker access probe

Validates dataset access from SPCS containers using the same production path.

```sql
CALL run_synthetic_regression_combined_autogluon_spcs_worker_access_probe(
  'spcs_job', 0, 1, 2
);
```

### Step 7 — SPCS evaluation (single-node mode)

Single-node mode: one SPCS job service per shard. No self-managed Ray.
The entrypoint is `/app/src/evaluate_synthetic_regression.py`.

```sql
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
  'spcs_job',   -- AUTOGLUON_RUNTIME_ENVIRONMENT (ignored)
  0,            -- AUTOGLUON_CLUSTER_SHARDS (0 = single-node)
  1,            -- AUTOGLUON_WORKERS_PER_SHARD
  1,            -- AUTOGLUON_TASK_CPUS
  6,            -- AUTOGLUON_CONCURRENT_CLUSTERS (number of shards)
  300,          -- AUTOGLUON_TIME_LIMIT
  'best_quality'-- AUTOGLUON_PRESETS
);
```

### Step 7b — SPCS evaluation (Ray distributed mode)

Ray distributed mode: per shard, one coordinator SPCS service + N worker SPCS services.
The coordinator merges the Ray head and AutoGluon driver into a single container:
it starts `ray start --head --num-cpus=0` locally, then runs `autogluon_ray.py` with
`RAY_HEAD_ADDRESS=localhost:<port>`. Workers connect to the coordinator via its external
DNS address. This is self-managed Ray — Snowflake does not manage the Ray cluster topology.

```sql
CALL run_synthetic_regression_combined_autogluon_spcs_evaluation(
  'spcs_job',   -- AUTOGLUON_RUNTIME_ENVIRONMENT (ignored)
  6,            -- AUTOGLUON_CLUSTER_SHARDS (Ray clusters; 6 coordinators)
  4,            -- AUTOGLUON_WORKERS_PER_SHARD (4 workers per coordinator)
  1,            -- AUTOGLUON_TASK_CPUS
  6,            -- AUTOGLUON_CONCURRENT_CLUSTERS (must equal AUTOGLUON_CLUSTER_SHARDS)
  300,          -- AUTOGLUON_TIME_LIMIT
  'best_quality'-- AUTOGLUON_PRESETS
);
```

**Per-shard coordinator DNS:** Each shard's head address is derived automatically from the
shard's coordinator SPCS service name and `SPCS_RAY_HEAD_DNS_SUFFIX`. Set the suffix to the
SPCS internal DNS domain for your Snowflake account:

```bash
# SPCS DNS rule: underscores in the service name are replaced by dashes in DNS
# Format: <service_name_lower_with_dashes>.<suffix>:<port>
# Example: shard 0 coordinator SPCS_RAY_COORD_R0_0 → spcs-ray-coord-r0-0.<suffix>:6379
export SPCS_RAY_HEAD_DNS_SUFFIX=<spcs_internal_dns_suffix>
```

The suffix is typically the SPCS service endpoint domain shown in `SHOW SERVICES` or in
the Snowflake SPCS documentation for your account region. Each shard has a unique derived
address — do not use a single global override for multi-shard deployments.

### Rollback to MLJob backend

To revert to the MLJob backend, unset or change the backend env var:

```bash
export SYNREG_AUTOGLUON_EXECUTION_BACKEND=mljob
# SYNREG_AUTOGLUON_SPCS_IMAGE is not required for mljob
```

Then use the original procedures:

```sql
CALL run_synthetic_regression_combined_autogluon_evaluation(...);
```

### Operational caveats

- The SPCS backend uses `EXECUTE JOB SERVICE` which does not support `pip_requirements`
  or `runtime_environment` — all dependencies must be preinstalled in the image.
- **SPCS OAuth token:** SPCS job services automatically receive the OAuth token at
  `/snowflake/session/token`, `SNOWFLAKE_ACCOUNT`, and `SNOWFLAKE_HOST`. No `snowflakeService`
  YAML block is required or supported — Snowflake rejects specs containing it with error 395018.
  Container code must read the token from the file path and use `authenticator='oauth'`.
- **Ray requires deterministic ports on SPCS:** Use `--node-manager-port=6380`,
  `--object-manager-port=6381`, `--runtime-env-agent-port=6382`,
  `--min-worker-port=10002 --max-worker-port=10010` on both head and worker `ray start` commands.
  SPCS specs must declare these as TCP endpoints — traffic to undeclared ports is silently dropped.
- **Capacity probe uses reduced object-store memory:** The SPCS capacity probe defaults coordinator
  and worker object-store to 256 MB each (vs 500 MB / 2 GB for production). `/dev/shm` in SPCS
  containers is only 64 MiB; Ray falls back to `/tmp` but stays stable.
- **Raylet failure diagnostics:** If raylet exits, the coordinator/worker scripts now dump the last
  32 KB of `/tmp/ray/session_latest/logs/raylet.err`, `gcs_server.err`, and related log files as
  structured JSON log events for inspection in SPCS job service logs.

### Capacity probe failure diagnostics

When a capacity probe reports partial worker join (e.g. `live_nodes=2/5`), use this sequence:

1. **Check timeout first** — The default readiness timeout is now 900 s. If the probe timed out
   at 300 s, simply rerun: cold-start of a custom image can take 3–5 minutes.
2. **Reduce burst pressure** — If partial join persists, rerun with
   `SYNREG_SPCS_WORKER_SUBMIT_STAGGER_SECONDS=10` to stagger worker submission by 10 s per
   worker and reduce simultaneous scheduling pressure.
3. **Capture missing worker logs** — If workers still show no logs, rerun with
   `SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE=true`. On coordinator timeout the worker jobs are
   left running so their logs can be retrieved via SPCS job service log inspection.
4. **Cancel leftover jobs** — After inspection, manually cancel each leftover worker:
   `SELECT <job_service_name>!SPCS_CANCEL_JOB();`

Env vars summary:

| Env var | Default | Purpose |
|---------|---------|---------|
| `SYNREG_RAY_CLUSTER_READY_TIMEOUT_SECONDS` | 900 | Capacity probe readiness timeout |
| `SYNREG_SPCS_WORKER_SUBMIT_STAGGER_SECONDS` | 0 | Sleep (s) between worker submissions |
| `SYNREG_SPCS_KEEP_SUPPORT_JOBS_ON_FAILURE` | false | Leave workers running on coordinator failure |

### SPCS Ray cancellation diagnostics

When Snowflake cancels a worker or coordinator (CANCELLED: Job was cancelled while running),
the SPCS wrapper scripts now emit structured JSON log events before exiting:

- **`spcs_ray_worker_signal_received`** — logged immediately on SIGTERM or SIGINT; includes
  `signal_name`, `uptime_seconds`, `ray_head_address`, configured Ray ports, `run_id`,
  `shard_index`, `ray_proc_pid`, and `ray_proc_returncode` (if Ray subprocess was running).
- **`spcs_ray_worker_exit_after_signal`** — logged after graceful Ray subprocess termination;
  includes `final_ray_returncode` and `exit_code` (128 + signal number).
- **`spcs_ray_coordinator_signal_received`** / **`spcs_ray_coordinator_exit_after_signal`** —
  equivalent events for the coordinator, adding `driver_pid` and `ray_head_pid`.
- Ray log tails (`ray_log_file` events) are dumped on signal if `/tmp/ray/session_latest/logs`
  exists at the time of cancellation.

**Distinguishing cancellation causes:**
- Signal log present → Python was running; Snowflake sent SIGTERM. Check `uptime_seconds` to
  see if the worker was alive long enough to attempt Ray connection.
- No signal log, no startup log → Snowflake terminated before Python started (cold-start race
  or scheduling failure). Use `SYNREG_SPCS_WORKER_SUBMIT_STAGGER_SECONDS=10` to reduce burst.
- `spcs_ray_worker_ray_exited` with non-zero returncode → Ray process crashed; check
  `ray_log_file` events for raylet/GCS errors.

**Capacity probe JSON logs:**
`ray_capacity_probe.py` now emits structured JSON events (`ray_capacity_probe_started`,
`ray_capacity_probe_readiness`, `ray_capacity_probe_ready`, `ray_capacity_probe_timeout`,
`ray_capacity_probe_sleeping`, `ray_capacity_probe_complete`) that can be filtered alongside
coordinator/worker events in SPCS job service logs.

- **Coordinator merges head + driver:** `spcs_ray_coordinator.py` starts `ray start --head
  --num-cpus=0 --object-store-memory=<bytes>` as a subprocess, polls localhost until the
  port is reachable, then executes `autogluon_ray.py` with `SYNREG_RAY_ADDRESS_MODE=explicit`
  and `RAY_HEAD_ADDRESS=localhost:<port>`. The coordinator's exit code propagates from the
  driver subprocess. The Ray head subprocess is always terminated in a `finally` block.
- **Ray head starts with `--num-cpus=0`** (inside coordinator) so it does not consume
  schedulable CPU from the pool. The driver's readiness check uses
  `expected_nodes = WORKERS_PER_SHARD + 1` (head counts as a live node with zero CPUs) and
  `expected_cpus = WORKERS_PER_SHARD * TASK_CPUS`.
- **Workers advertise explicit `--num-cpus` and `--object-store-memory`** on the `ray start`
  command so Ray's resource accounting is accurate. `AUTOGLUON_TASK_CPUS` controls
  `--num-cpus`; `SYNREG_SPCS_RAY_WORKER_OBJECT_STORE_MEMORY_BYTES` controls object store.
- **TCP endpoints for all Ray ports:** The coordinator SPCS spec exposes five TCP endpoints:
  `ray-head` (6379), `ray-node-manager` (6380), `ray-object-manager` (6381),
  `ray-runtime-env-agent` (6382), and `ray-worker-ports` (portRange 10002–10010). Worker specs
  expose the same four ports except `ray-head`. All ports are deterministic — passed via
  `--node-manager-port` etc. on `ray start` — so they can be declared in SPCS specs.
- **Resource sizing:** Worker and single-node containers keep worker-sized resources because
  they run AutoGluon fits. Coordinators use a smaller profile (1/2 CPU, 4Gi/8Gi memory).
  In the default 6×4 topology this keeps 24 schedulable worker containers while avoiding
  worker-sized reservations for the 6 coordinator containers.
- **Cluster identity verification:** The coordinator announces a custom Ray resource
  `spcs_cluster_id_<run_id>_<shard_index>=1`. The driver (running inside coordinator) checks
  for this resource after `ray.init()` to confirm it joined the correct shard's cluster. If
  the check fails, the driver raises `RuntimeError` rather than submitting work to the wrong cluster.
- **Per-shard DNS with `SPCS_RAY_HEAD_DNS_SUFFIX` and underscore→dash normalization:**
  SPCS replaces underscores in service names with dashes in DNS. Each shard's coordinator
  address is derived as `<service_name_with_underscores_to_dashes>.<suffix>:<port>`. For
  example, coordinator service `SPCS_RAY_COORD_R0_0` → DNS hostname `spcs-ray-coord-r0-0.<suffix>`.
  Set `SPCS_RAY_HEAD_DNS_SUFFIX` to the SPCS internal DNS domain. Do not set
  `SPCS_RAY_HEAD_DNS_OVERRIDE` — it is no longer supported and will be ignored.
- **Presigned URL expiry:** The default expiry is 86400 s (24 h). The driver uses
  `SYNREG_PRESIGNED_URL_EXPIRY_POLICY=strict` by default and fails before submitting Ray work
  if the conservative shard runtime estimate exceeds the expiry minus
  `SYNREG_PRESIGNED_URL_EXPIRY_BUFFER_SECONDS` (default 3600). Set the policy to `warn` only
  when you accept late-worker HTTP 403 risk.
- **Image verification:** At job submission the driver first matches the exact repository URL
  from `SYNREG_AUTOGLUON_SPCS_IMAGE`, then attempts exact image name/tag or digest matching via
  `SHOW IMAGES IN IMAGE REPOSITORY`. If the procedure role cannot query those metadata views,
  it logs a warning and you must manually run `SHOW IMAGES IN IMAGE REPOSITORY AUTOGLUON_IMAGE_REPOSITORY;`.
- The Snowpark session probe (`run_synthetic_regression_autogluon_spcs_session_probe`) uses
  the SPCS-injected OAuth token at `/snowflake/session/token` with `SNOWFLAKE_ACCOUNT` and
  `SNOWFLAKE_HOST`. Run it as a SQL stored procedure call (Step 4), not as a bash command.
- AUTOGLUON_IMAGE_REPOSITORY images are private to the Snowflake account. Each push
  requires re-authentication with `docker login`.
- The `SYNREG_AUTOGLUON_SPCS_IMAGE` env var must be set to the full OCI image reference
  including the registry hostname, database, schema, repository, image name, and tag.

**Module-level torch dependency removed:**
`evaluate_synthetic_regression.py` no longer imports `torch`, `deepset_inference`, or `model`
at module level. These are imported lazily inside DeepSet/checkpoint functions only. The SPCS
AutoGluon custom image does not need torch for worker-access probes or distributed AutoGluon
evaluation. If a DeepSet path is invoked in a torch-free runtime, `_import_torch()` raises a
clear `RuntimeError` instead of a bare `ModuleNotFoundError`.
