---
name: architecture
description: >
  Reference for the intended training and evaluation pipeline architecture of the DeepSet ICL
  model: how synthetic data is generated, how TRAINING_DATA_FAMILY routes pretrain/HPO/final-
  training to shared checkpoints, and how the regression and classification evaluation suites
  are each parameterized for categorical features rather than duplicated into separate scripts.
  Use when adding a task family, wiring a new runtime flag, adding categorical coverage to an
  eval generator, or reasoning about which script/checkpoint/env-var a change belongs in.
  Does not restate the model internals (see MODEL4.md) or Snowflake infrastructure details
  (see CLAUDE.md). Does not cover the OpenML/Kaggle benchmark (see evaluate.py).
---

# Architecture

The guiding principle is **one script per concern, parameterized — never duplicated**:
one training-data generator per task type, one family-routing variable (`TRAINING_DATA_FAMILY`),
one checkpoint name per task (independent of categorical-ness), and one shared evaluation script
per task that is gated by a runtime flag for categorical features rather than forked into a
separate script.

All previously identified implementation gaps have been resolved. The architecture described below
reflects the current state of the codebase.

---

## Source-of-truth files

| Concern | Authoritative file(s) |
|---|---|
| Family routing | `src/model/task_routing.py` — `_FAMILY_SPECS`, `get_training_data_spec`, `allowed_training_data_families` |
| Family + constant names | `src/model/constants.py` — `MIXED_CAT_REGRESSION_TRAINING_FAMILY`, `MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY` |
| Training data generation (linear, categorical) | `src/generate_dgp.py` + `src/dgp_helpers.py` |
| Training data generation (nonlinear) | `src/generate_nonlinear_dgp.py` |
| Eval-suite generation (linear) | `scripts/generate_synthetic_regression.py`, `scripts/generate_synthetic_classification.py` |
| Eval-suite generation (nonlinear) | `scripts/generate_nonlinear_regression.py`, `scripts/generate_nonlinear_classification.py` |
| Training entry | `src/model/train.py` |
| Pretrain orchestrator | `scripts/run_pretrain_job.py` |
| HPO orchestrator | `scripts/run_hpo_job.py` + `src/model/hpo.py` |
| Final-training orchestrator | `scripts/run_model_training_job.py` |
| Eval prep | `scripts/prepare_synthetic_regression.py`, `scripts/prepare_synthetic_classification.py` |
| Eval run (linear) | `scripts/evaluate_linear_regression.py`, `scripts/evaluate_linear_classification.py` |
| Eval run (nonlinear) | `scripts/evaluate_nonlinear_regression.py`, `scripts/evaluate_nonlinear_classification.py` |
| Eval orchestration | `scripts/run_synthetic_regression_evaluation.py`, `scripts/run_synthetic_classification_evaluation.py` |
| Eval SQL procedures (DDL) | `sql/04_synthetic_regression_evaluation_pipeline.sql` |
| Model internals (out of scope here) | `MODEL4.md` |

---

## Data Generation

### Training-data generators (`src/`)

**`src/generate_dgp.py`** is the canonical training-data generator for all linear task families.
It dispatches on `--task_family`:

| `--task_family` | Task objective | Categorical features |
|---|---|---|
| `linear_regression` | regression | No |
| `linear_classification` | classification | No |
| `linear_regression_mixed_categorical` | regression | Yes |
| `linear_classification_mixed_categorical` | classification | Yes |

The constant names for the two mixed families live in `src/model/constants.py`:
`MIXED_CAT_REGRESSION_TRAINING_FAMILY = "synthetic_linear_regression_mixed_categorical"` and
`MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY = "synthetic_linear_classification_mixed_categorical"`.

Mixed datasets carry a `training_data_family` field inside the dataset dict. All families use a
fixed 80/10/10 train/val/test split. Shared helpers (regime allocation, parquet writing, teacher
calibration) live in `src/dgp_helpers.py`.

**`src/generate_nonlinear_dgp.py`** generates all four nonlinear training task families. It
writes training-format parquet into `train/val/test/` splits (80/10/10) via
`main_nonlinear_training()` (gated by `--task_family`) and evaluation-format parquet via
`main()`. The `--task_family` values accepted by `main_nonlinear_training()` are:

| `--task_family` | Task objective | Categorical features |
|---|---|---|
| `synthetic_nonlinear_regression` | regression | No |
| `synthetic_nonlinear_classification` | classification | No |
| `synthetic_nonlinear_regression_mixed_categorical` | regression | Yes |
| `synthetic_nonlinear_classification_mixed_categorical` | classification | Yes |

Evaluation parquet (420 datasets, 6 families × 7 regimes) is written by `main()` unchanged.
Seed magic constants (`_NONLINEAR_*_SEED_MAGIC`) are applied per family to prevent seed collision.

### Evaluation-suite generators (`scripts/`)

`scripts/generate_synthetic_regression.py` and `scripts/generate_synthetic_classification.py`
are the evaluation-suite generators. They are sibling scripts that share `src/dgp_helpers.py`
with `generate_dgp.py` but are NOT wrappers of it — they generate independent, bias-resistant
evaluation payloads fed to the `prepare_synthetic_*.py` stage.

Their current dispatch is across robustness **suite families**, not categorical-ness:

- Regression families (`_FAMILY_ORDER`): `primary`, `feature_noise`, `target_noise`,
  `training_size`, `sparsity`, `correlation`, `dimensionality`, `ood`, `eval_only_unseen`,
  `hidden_holdout`, `stress`.
- Classification families (`_FAMILY_GENERATORS`): `primary`, `feature_noise`, `label_noise`,
  `training_size`, `class_imbalance`, `margin`, `num_classes`, `ood`, `eval_only_unseen`,
  `hidden_holdout`, `stress`.

Output destinations:
- Regression: `@EVALUATION_DATASET_STAGE/synthetic_regression_prepared/{suite_id}/{family}/`
- Classification: `@EVALUATION_DATASET_STAGE/synthetic_classification_prepared/{suite_id}/{family}/`

**Categorical eval generation is now implemented.** Each eval generator produces BOTH categorical
and non-categorical eval datasets within the same script, gated by the runtime flags
`SYNREG_IS_MIXED_CATEGORICAL` (regression) / `SYNCLS_IS_MIXED_CATEGORICAL` (classification).
This mirrors how `generate_dgp.py` handles both for training. The mixed builders in
`src/dgp_helpers.py` (`allocate_mixed_regression_tasks`, `allocate_mixed_classification_tasks`,
`build_mixed_regression_dataset`, `build_mixed_classification_dataset`) are reused.
The mixed output destinations are:
- `@EVALUATION_DATASET_STAGE/mixed_regression_prepared/{suite_id}/{family}/`
- `@EVALUATION_DATASET_STAGE/mixed_classification_prepared/{suite_id}/{family}/`

Do not create separate `generate_synthetic_regression_categorical.py` /
`generate_synthetic_classification_categorical.py` scripts. The existing ones are parameterized.

---

## Training: `TRAINING_DATA_FAMILY` routing

`TRAINING_DATA_FAMILY` is the single environment variable that routes every stage of the training
pipeline. It is validated against `allowed_training_data_families()` in `src/task_routing.py`.

Full family table (from `_FAMILY_SPECS` in `src/model/task_routing.py`):

All 8 canonical families share `stage = @META_DATASET_STAGE`. The stage subdir
(`{linear|nonlinear}/{regression|classification}/{numeric|mixed}`) fully encodes the family.

| Family | `task_objective` | Index table | Stage subdir | HPO metric |
|---|---|---|---|---|
| `synthetic_linear_regression` (default) | `inductive_regression` | `META_REGRESSION_DATASET_INDEX` | `linear/regression/numeric` | `val_mse` |
| `synthetic_regression_primary` | `inductive_regression` | `META_REGRESSION_DATASET_INDEX` | `linear/regression/numeric` | `val_mse` |
| `synthetic_regression_ood` | `inductive_regression` | `META_REGRESSION_DATASET_INDEX` | `linear/regression/numeric` | `val_mse` |
| `market_mental_model` | `inductive_regression` | `META_REGRESSION_DATASET_INDEX` | `linear/regression/numeric` | `val_mse` |
| `synthetic_nonlinear_regression` | `inductive_regression` | `META_NONLINEAR_REGRESSION_DATASET_INDEX` | `nonlinear/regression/numeric` | `val_mse` |
| `synthetic_nonlinear_classification` | `inductive_classification` | `META_NONLINEAR_CLASSIFICATION_DATASET_INDEX` | `nonlinear/classification/numeric` | `val_cross_entropy` |
| `synthetic_nonlinear_regression_mixed_categorical` | `inductive_regression` | `META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX` | `nonlinear/regression/mixed` | `val_mse` |
| `synthetic_nonlinear_classification_mixed_categorical` | `inductive_classification` | `META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX` | `nonlinear/classification/mixed` | `val_cross_entropy` |
| `synthetic_linear_classification` | `inductive_classification` | `META_CLASSIFICATION_DATASET_INDEX` | `linear/classification/numeric` | `val_cross_entropy` |
| `synthetic_linear_regression_mixed_categorical` | `inductive_regression` | `META_MIXED_REGRESSION_DATASET_INDEX` | `linear/regression/mixed` | `val_mse` |
| `synthetic_linear_classification_mixed_categorical` | `inductive_classification` | `META_MIXED_CATEGORICAL_DATASET_INDEX` | `linear/classification/mixed` | `val_cross_entropy` |

Back-compat alias: `synthetic_regression_nonlinear` maps to the nonlinear-regression spec
(`task_routing.py:122`). The canonical family name is `synthetic_nonlinear_regression`.

#### Stage path convention

All training families share `@META_DATASET_STAGE`. The full path is
`@META_DATASET_STAGE/{subdir}/{split}/{task_id}.parquet` where `{subdir}` = `data_subdir`
(the three-segment canonical derived from linearity/objective/feature-type). Helper functions
`canonical_meta_subdir(family, split)` and `canonical_eval_subdir(family)` in
`src/model/task_routing.py` are the single source of truth.

Example for linear regression:
```
@META_DATASET_STAGE/linear/regression/numeric/train/{task_id}.parquet
@META_DATASET_STAGE/linear/regression/numeric/val/{task_id}.parquet
@META_DATASET_STAGE/linear/regression/numeric/test/{task_id}.parquet
```

Example for nonlinear mixed-categorical classification:
```
@META_DATASET_STAGE/nonlinear/classification/mixed/train/{task_id}.parquet
@META_DATASET_STAGE/nonlinear/classification/mixed/val/{task_id}.parquet
@META_DATASET_STAGE/nonlinear/classification/mixed/test/{task_id}.parquet
```

The index builder scripts (`src/dataset_index/build_meta_*_dataset_index.py`) LIST and GET
from the matching subdir paths in `@META_DATASET_STAGE`.

`is_nonlinear` for a `TrainingDataSpec` is true when its `family` is in `_NONLINEAR_FAMILIES`
(a frozenset in `task_routing.py` containing all 4 nonlinear family strings).
`is_classification` derives from `task_objective` only — never from categorical-ness or
nonlinear-ness. `CHECKPOINT_OUTPUT_NAME` follows from `is_classification` alone and is unchanged.

**`is_classification` derives from `task_objective` only — never from categorical-ness.**
`TrainingDataSpec.is_classification` is `task_objective == "inductive_classification"`.
Mixed-categorical regression resolves `is_classification = False`; mixed-categorical
classification resolves `is_classification = True`. See `src/task_routing.py`.

**Pipeline flow:**

1. **Pretrain** (`scripts/run_pretrain_job.py` → `src/train.py`): reads `TRAINING_DATA_FAMILY`,
   resolves `is_classification` → writes `pretrain.pt` / `pretrain_classification.pt`
   (+ `pretrain_gate{dim}.pt` / `pretrain_classification_gate{dim}.pt` variants).
2. **HPO** (`scripts/run_hpo_job.py` → `src/hpo.py`): warm-starts from pretrain checkpoint;
   embeds `training_data_family` in `best_config._meta` → writes `@MODEL_STAGE/hpo/best_config.json`.
3. **Final training** (`scripts/run_model_training_job.py` → `src/train.py`): enforces
   `TRAINING_DATA_FAMILY` matches `best_config._meta.training_data_family` (raises on mismatch);
   resolves `is_classification` → writes the final checkpoint (see §Checkpoint naming).

SQL orchestration procedures: `run_pretrain_pipeline()`, `run_hpo_pipeline()`, `run_model_training()`
(defined in `preload.sql`).

---

## Checkpoint naming contract

The checkpoint base name is determined solely by `is_classification`. Categorical-ness never
changes the name. This holds because `is_classification` is derived from `task_objective`, which
is the same for `synthetic_linear_regression_mixed_categorical` as it is for
`synthetic_linear_regression`.

| Phase | Regression output | Classification output |
|---|---|---|
| Pretrain | `pretrain.pt` / `pretrain_gate{dim}.pt` | `pretrain_classification.pt` / `pretrain_classification_gate{dim}.pt` |
| HPO | `best_config.json` (not a checkpoint) | same |
| Final training | `best_regression.pt` | `best_classification.pt` |

`CHECKPOINT_OUTPUT_NAME` in `run_model_training_job.py` (lines 437–439):
```python
"CHECKPOINT_OUTPUT_NAME": "best_classification.pt" if is_classification else "best_regression.pt"
```

The regression eval pipeline reads `@MODEL_STAGE/checkpoints/best_regression.pt`. The env var
`SYNREG_DEEPSET_CHECKPOINT_STAGE_PATH` overrides this default.

The final training job now writes `best_regression.pt` directly, so no manual aliasing is needed.

---

## Evaluation: shared suite, parameterized by categorical flag

### Design contract

Each regression and classification evaluation stage is implemented in **one** script that handles
both categorical and non-categorical synthetic datasets via a boolean runtime flag. Do not fork
separate scripts per categorical-ness.

| Stage | Script | Categorical flag |
|---|---|---|
| Eval-suite generation | `scripts/generate_synthetic_regression.py` | `SYNREG_IS_MIXED_CATEGORICAL` |
| Eval-suite generation | `scripts/generate_synthetic_classification.py` | `SYNCLS_IS_MIXED_CATEGORICAL` |
| Index preparation | `scripts/prepare_synthetic_regression.py` | `SYNREG_IS_MIXED_CATEGORICAL` |
| Index preparation | `scripts/prepare_synthetic_classification.py` | `SYNCLS_IS_MIXED_CATEGORICAL` |
| Eval prep (nonlinear reg) | `scripts/prepare_nonlinear_regression.py` | `SYNREG_IS_MIXED_CATEGORICAL` (via `SYNREG_INDEX_TABLE` env redirect) |
| Eval prep (nonlinear cls) | `scripts/prepare_nonlinear_classification.py` | `SYNCLS_IS_MIXED_CATEGORICAL` (via `SYNCLS_INDEX_TABLE` env redirect) |
| Evaluation run (linear) | `scripts/evaluate_linear_regression.py` | `SYNREG_IS_MIXED_CATEGORICAL` |
| Evaluation run (linear) | `scripts/evaluate_linear_classification.py` | `SYNCLS_IS_MIXED_CATEGORICAL` |
| Evaluation run (nonlinear) | `scripts/evaluate_nonlinear_regression.py` | `SYNREG_IS_MIXED_CATEGORICAL` |
| Evaluation run (nonlinear) | `scripts/evaluate_nonlinear_classification.py` | `SYNCLS_IS_MIXED_CATEGORICAL` |

Nonlinear orchestrators inject both the `*_IS_MIXED_CATEGORICAL=true` flag and
`*_INDEX_TABLE=NONLINEAR_MIXED_REGRESSION_DATASET_INDEX` (or `NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX`)
via `_NONLINEAR_MIXED_INDEX_ENV` / `_NONLINEAR_MIXED_CLS_INDEX_ENV`. The prep scripts read `SYNREG_INDEX_TABLE` /
`SYNCLS_INDEX_TABLE` (with fallback to the standard nonlinear index) to select the correct
output table. See §Nonlinear mixed-categorical eval (below).

### Flag naming convention

The canonical form is `<SUBSYSTEM>_IS_MIXED_CATEGORICAL` with the subsystem prefix. Do not
introduce `USE_CATEGORICAL`, `WITH_CATEGORICAL`, `*_CAT`, or any other variant. Parsed by the
`_env_flag` helper (`os.getenv(name, "false").strip().lower() in ("1", "true", "yes")`).

### What the flag does in prep scripts

When `SYNREG_IS_MIXED_CATEGORICAL=true`, `prepare_linear_regression.py` dispatches to
`_prepare_mixed_regression()`, which reads from
`@EVALUATION_DATASET_STAGE/linear/regression/mixed/{suite_id}/` and writes to
`LINEAR_MIXED_REGRESSION_DATASET_INDEX` (includes extra columns: `p_num`, `p_cat`,
`categorical_cardinalities`, etc.).
The classification analogue is `_prepare_mixed_classification()` in `prepare_linear_classification.py`.

### SQL procedures and orchestrators

The `SYNREG_IS_MIXED_CATEGORICAL` / `SYNCLS_IS_MIXED_CATEGORICAL` flags are read by the prep and
evaluator scripts and are now injected into the MLJob `env_vars` by the orchestrators. The wiring
points are:

**Regression** (`scripts/evaluation/run_linear_regression_evaluation.py`):
- `_synreg_shard_env()` — includes `"SYNREG_IS_MIXED_CATEGORICAL": os.getenv("SYNREG_IS_MIXED_CATEGORICAL", "false")` in the base env dict so it propagates to all phases including `deepset`.
- `run_linear_regression_prep()` env_vars — same flag so the prep job builds the correct (mixed or standard) index table.

**Classification** (`scripts/evaluation/run_linear_classification_evaluation.py`):
- `_classification_shard_env()` — includes `"SYNCLS_IS_MIXED_CATEGORICAL": os.getenv("SYNCLS_IS_MIXED_CATEGORICAL", "false")`.
- `run_linear_classification_prep()` env_vars — same flag.

Each SQL procedure DDL (`sql/linear_regression_{numeric,mixed}_pipeline.sql` and the
classification counterparts) exposes `IS_MIXED_CATEGORICAL BOOLEAN` as the **first** proc parameter.
Numeric procs pass `FALSE`; mixed procs pass `TRUE`. Both bind the same shared Python handler
(`run_linear_regression_evaluation.run_linear_regression_prep`, etc.). The handler calls
`_set_regression_linear_env(suite_id, is_mixed_categorical=<arg>)` which sets
`SYNREG_IS_MIXED_CATEGORICAL`, `SYNREG_INDEX_TABLE`, and the results-stage path internally.

### Nonlinear mixed-categorical eval wiring

Nonlinear evaluation uses **index-table injection** rather than the `os.getenv` propagation used
by the linear orchestrators (because `_synreg_shard_env` / `_classification_shard_env` propagate
their caller's env, not a fresh `os.getenv` from the orchestrator process itself). The orchestrator
injects both env vars explicitly on every SPCS job:

**Regression** (`scripts/evaluation/run_nonlinear_regression_evaluation.py`):
- `_NONLINEAR_MIXED_INDEX_ENV = {"SYNREG_INDEX_TABLE": "NONLINEAR_REGRESSION_MIXED_DATASET_INDEX", "SYNREG_IS_MIXED_CATEGORICAL": "true"}`
- Core phase handlers accept `is_mixed_categorical: bool = False`; when `True`, suite params are
  fetched from `_nonlinear_regression_suite_params(True)`.
- SQL procs: `run_nonlinear_regression_{prep,deepset_evaluation,baseline_evaluation,autogluon_evaluation,aggregation}`
  with `IS_MIXED_CATEGORICAL BOOLEAN` as trailing param; mixed pipeline passes `TRUE`.
  Files: `sql/nonlinear_regression_{numeric,mixed}_pipeline.sql`.

**Classification** (`scripts/evaluation/run_nonlinear_classification_evaluation.py`):
- `_NONLINEAR_MIXED_CLS_INDEX_ENV = {"SYNCLS_INDEX_TABLE": "NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX", "SYNCLS_IS_MIXED_CATEGORICAL": "true"}`
- Core phase handlers accept `is_mixed_categorical: bool = False`.
- SQL procs: `run_nonlinear_classification_{prep,deepset_evaluation,baseline_evaluation,autogluon_evaluation,aggregation}`
  with `IS_MIXED_CATEGORICAL BOOLEAN` as trailing param.
  Files: `sql/nonlinear_classification_{numeric,mixed}_pipeline.sql`.

Prep scripts read `SYNREG_INDEX_TABLE` / `SYNCLS_INDEX_TABLE` (with fallback to standard table) so
the same `prepare_nonlinear_regression.py` / `prepare_nonlinear_classification.py` script writes to
the correct eval index without forking.

### Linear vs nonlinear: fork + shared harness

The linear and nonlinear evaluation pipelines are **parallel forks** of the same 4-mode structure
(`SYNTHETIC_REGRESSION_MODE` = `deepset` / `baselines` / `autogluon` / `aggregate`), not a single
parameterized script. This is deliberate: nonlinear suites have different defaults, different index
tables, and nonlinear-specific baseline flags.

**Evaluator-script pairs (source forks):**

| Linear script | Nonlinear fork | Key differences |
|---|---|---|
| `evaluate_linear_regression.py` | `evaluate_nonlinear_regression.py` | Suite id (`linear_poisson_v1_recommended` vs `nonlinear`), index table (`LINEAR_REGRESSION_DATASET_INDEX` vs `NONLINEAR_REGRESSION_DATASET_INDEX`), `SYNREG_NONLINEAR_BASELINES=true` |
| `evaluate_linear_classification.py` | `evaluate_nonlinear_classification.py` | Suite id, index table (`LINEAR_CLASSIFICATION_DATASET_INDEX` vs `NONLINEAR_CLASSIFICATION_DATASET_INDEX`), nonlinear baseline flag |

Both forks share the same `_run_deepset_mode`, `_run_baselines_mode`, `_run_autogluon_mode`,
and `_run_aggregate_mode` logic inherited from `evaluate_linear_regression.py`. Any evaluator-logic
change must be mirrored across the fork pair.

**Orchestrator relationship:**

- `scripts/run_synthetic_regression_evaluation.py` is the **canonical shared orchestrator** for
  both linear and combined suites. It exposes all split-phase helpers used by both pipelines.
- `scripts/run_nonlinear_regression_evaluation.py` is a **thin wrapper** that imports ~70 helpers
  from `run_synthetic_regression_evaluation.py`, overrides suite constants
  (`NONLINEAR_INDEX_TABLE`, `NONLINEAR_PARTS_PREFIX`, `NONLINEAR_OUTPUT_STAGE`, etc.), and injects
  `SYNREG_INDEX_TABLE=NONLINEAR_REGRESSION_DATASET_INDEX` on every submitted job.
  New shared orchestration logic belongs in `run_synthetic_regression_evaluation.py`, not copied
  into the nonlinear wrapper.
- The classification orchestrators follow the same pattern (`run_synthetic_classification_evaluation.py`
  → shared; `run_nonlinear_classification_evaluation.py` → thin wrapper).

**Result stage layout** (`@EVALUATION_RESULTS_STAGE/{suite}/{task}/{numeric|mixed}/{suite_id}`):

| Suite | Task | Categorical | Example path |
|---|---|---|---|
| `linear` | `regression` | No | `@EVALUATION_RESULTS_STAGE/linear/regression/numeric/{suite_id}` |
| `linear` | `regression` | Yes | `@EVALUATION_RESULTS_STAGE/linear/regression/mixed/{suite_id}` |
| `linear` | `classification` | No | `@EVALUATION_RESULTS_STAGE/linear/classification/numeric/{suite_id}` |
| `linear` | `classification` | Yes | `@EVALUATION_RESULTS_STAGE/linear/classification/mixed/{suite_id}` |
| `nonlinear` | `regression` | No | `@EVALUATION_RESULTS_STAGE/nonlinear/regression/numeric/{suite_id}` |
| `nonlinear` | `regression` | Yes | `@EVALUATION_RESULTS_STAGE/nonlinear/regression/mixed/{suite_id}` |
| `nonlinear` | `classification` | No | `@EVALUATION_RESULTS_STAGE/nonlinear/classification/numeric/{suite_id}` |
| `nonlinear` | `classification` | Yes | `@EVALUATION_RESULTS_STAGE/nonlinear/classification/mixed/{suite_id}` |

---

## Guardrails

- **Do not duplicate scripts for categorical.** The single-script-per-concern rule applies to
  generators, prep, and evaluators. Categorical-ness is a runtime parameter, not a code fork.
- **Categorical-ness never changes the checkpoint name or `task_objective`.** Mixed-categorical
  regression still produces `best_regression.pt`; mixed-categorical classification
  still produces `best_classification.pt`.
- **Do not bypass `task_routing.py`.** All family-to-spec lookups go through
  `get_training_data_spec()`. Hard-coding family strings in orchestrators is a defect.
- **`is_classification` is read from `TrainingDataSpec`, not inferred.** Do not check family
  string equality for classification logic outside of `run_model_training_job.py`'s bootstrap
  (which uses the `_CLASSIFICATION_TRAINING_FAMILY` constant for legacy reasons).
- **Mirror evaluator logic across the fork pair.** `evaluate_nonlinear_regression.py` /
  `evaluate_nonlinear_classification.py` are source forks of their linear counterparts. Bug fixes
  and feature additions to the evaluator runtime must be applied to both the linear and nonlinear
  fork. Shared orchestration helpers belong in `run_synthetic_regression_evaluation.py` or
  `run_synthetic_classification_evaluation.py`, not duplicated into the thin nonlinear wrappers.
- **Model internals** — architecture, forward-pass math, checkpoint format versions — live in
  `MODEL4.md` and `src/model.py`. Do not restate them here.
- **Snowflake infrastructure** (compute pools, stage DDL, SPCS service specs, session patterns)
  lives in `CLAUDE.md`. Do not restate it here.
