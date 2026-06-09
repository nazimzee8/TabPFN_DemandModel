# TabPFN DeepSet Project Memory

## HPO Guardrails

- `max_concurrent_trials=2` (GPU_NV_M has 2 GPUs/node; each trial uses 1 GPU)
- `target_instances=6` for HPO — applies to both `run_training_job.py` (`run_pipeline`)
  and `run_hpo_job.py` (`run_hpo_pipeline`). 6 nodes × 2 concurrent = 12 parallel trials.
- GPU_NV_M replaces GPU_NV_S: 2× A10G/node, MAX_NODES=6.
  Training uses 4 nodes with `num_workers_per_node=2` → world_size=8.
- Training `target_instances=TRAIN_NUM_NODES=4`; HPO `target_instances=6`.
- Never call `scale_cluster()` from within an MLJob — raises error 517003.
- Epoch calibration scripts: `hpo_epoch_test.py` / `train_epoch_test.py` in `src/`.
  Uploaded to `@EPOCH_STAGE` (contains all `src/*.py`).
  Run before pool upgrade via `CALL run_hpo_epoch_test()` / `CALL run_train_epoch_test()`.
  Results: `@EPOCH_STAGE/hpo_timing.json`, `@EPOCH_STAGE/train_timing.json`.

## Stage Layout

| Stage | Contents |
|---|---|
| `@META_REGRESSION_DATASET_STAGE` | numeric/ + mixed/ subdirs ({train,val,test}/) + kaggle/*.npz |
| `@MODEL_STAGE/scripts/` | all src/*.py + scripts/*.py |
| `@MODEL_STAGE/hpo/` | best_config.json, hpo_failure.json |
| `@MODEL_STAGE/checkpoints/` | best.pt |
| `@EVALUATION_RESULTS_STAGE` | synthetic/, benchmark_parts/, model_comparison.csv |
| `@EPOCH_STAGE` | src/*.py + hpo_timing.json + train_timing.json |
| `@MLJOB_PAYLOAD_STAGE` | MLJob payload (managed by submit_from_stage) |

## Known Bugs Fixed

- `run_hpo_job.py`: was `target_instances=2`, fixed to `6`.
- `run_model_training_job.py`: `train_job = submit_from_stage(...)` correctly assigned.
- `hpo.py`: was `max_concurrent_trials=1`, fixed to `2`.
- `train.py`: was `num_workers_per_node=1`, fixed to `2`.
