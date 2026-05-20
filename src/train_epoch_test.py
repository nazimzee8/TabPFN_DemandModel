"""
train_epoch_test.py - DDP training epoch timing calibration.
Runs 1 DDP training epoch + 1 validation epoch
(world_size = TRAIN_NUM_NODES x 4 workers/node).
Rank 0 writes train_timing.json to @EPOCH_STAGE/.
Submit via: CALL run_train_epoch_test()
Output: @EPOCH_STAGE/train_timing.json
"""
import os
os.environ["HOME"] = "/tmp"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime, json, tempfile, time, traceback as _tb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def _upload_error_json(session, filename, script_name):
    payload = {
        "script":     script_name,
        "error_type": type(sys.exc_info()[1]).__name__ if sys.exc_info()[1] else "Unknown",
        "error":      str(sys.exc_info()[1]),
        "traceback":  _tb.format_exc(),
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
    }
    print(f"[ERROR] {script_name}: {payload['error_type']}: {payload['error']}", flush=True)
    print(f"[ERROR] Traceback:\n{payload['traceback']}", flush=True)
    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as _tmp:
            err_path = os.path.join(_tmp, filename)
            with open(err_path, "w") as f:
                json.dump(payload, f, indent=2)
            session.file.put(err_path, "@EPOCH_STAGE/", overwrite=True, auto_compress=False)
        print(f"Uploaded {filename} to @EPOCH_STAGE/", flush=True)
    except Exception as _ue:
        print(f"[WARNING] Could not upload {filename}: {_ue}", flush=True)


def train_timing_fn():
    # PyTorchDistributor calls this with zero args; all config comes from os.environ.
    from snowflake.ml.modeling.distributors.pytorch import get_context

    ctx        = get_context()
    local_rank = ctx.get_local_rank() if hasattr(ctx, "get_local_rank") else ctx.local_rank
    rank       = ctx.get_rank()       if hasattr(ctx, "get_rank")       else ctx.rank
    world_size = ctx.get_world_size() if hasattr(ctx, "get_world_size") else ctx.world_size
    device     = f"cuda:{local_rank}"
    is_main    = (rank == 0)
    use_amp    = True

    try:
        from train import (
            run_epoch, identity_collate, ParquetMetaDataset,
            reduce_loss_sum_count,
            DATA_DIR, LR, WEIGHT_DECAY, D_PHI, D_RHO, POOL,
            N_HEADS, N_SAB_FEAT, NORM_FEAT, NORM_TARGET,
        )
        from model import ModelConfig, _instantiate_model
        from snowflake_io import materialize_connector_shard

        # Pre-training: BEST_CONFIG absent → hyper_params={} → defaults apply.
        hyper_params = json.loads(os.environ.get("BEST_CONFIG", "{}"))

        lr           = float(hyper_params.get("lr",           LR))
        weight_decay = float(hyper_params.get("weight_decay", WEIGHT_DECAY))
        d_phi        = int(hyper_params.get("d_phi",          D_PHI))
        d_rho        = int(hyper_params.get("d_rho",          D_RHO))
        dropout      = float(hyper_params.get("dropout",      0.1))
        pool         = hyper_params.get("pool",               POOL)

        materialize_t0 = time.perf_counter()
        dataset_map = ctx.get_dataset_map()
        train_files = materialize_connector_shard(
            dataset_map["train"].get_shard(), DATA_DIR, "train"
        )
        val_files = materialize_connector_shard(
            dataset_map["val"].get_shard(), DATA_DIR, "val"
        )
        materialization_time_s = time.perf_counter() - materialize_t0
        if not train_files or not val_files:
            raise FileNotFoundError(
                f"train_epoch_test requires train and val parquet files under {DATA_DIR}; "
                f"found train={len(train_files)}, val={len(val_files)}"
            )

        train_loader = DataLoader(
            ParquetMetaDataset(train_files), batch_size=1, shuffle=True,
            num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=identity_collate,
        )
        val_loader = DataLoader(
            ParquetMetaDataset(val_files), batch_size=1, shuffle=False,
            num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=identity_collate,
        )

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool,
                            n_heads=N_HEADS, n_sab_feat=N_SAB_FEAT,
                            norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout)
        model     = _instantiate_model(cfg).to(device)
        model     = torch.compile(model, mode="reduce-overhead")
        model     = DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        run_epoch(model, train_loader, optimizer, scaler, True, device, use_amp)
        with torch.no_grad():
            val_loss_sum, val_count = run_epoch(
                model, val_loader, None, scaler, False, device, use_amp,
                return_sum_count=True,
            )
        val_mse = reduce_loss_sum_count(val_loss_sum, val_count, device, dist)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        if is_main:
            result = {
                "epoch_time_s": round(epoch_time, 2),
                "val_mse":      round(float(val_mse), 6),
                "num_nodes":    int(os.environ.get("TRAIN_NUM_NODES", "10")),
                "num_workers_per_node": 4,
                "world_size":   world_size,
                "train_files":  len(train_files),
                "val_files":    len(val_files),
                "materialization_time_s": round(materialization_time_s, 2),
                "lr": lr, "weight_decay": weight_decay,
                "d_phi": d_phi, "d_rho": d_rho, "dropout": dropout, "pool": pool,
            }
            print(f"[TIMING] Train epoch wall-clock (rank 0): {epoch_time:.2f}s", flush=True)
            print("train_epoch_test result:", result, flush=True)
            from snowflake.snowpark import Session
            session = Session.builder.getOrCreate()
            with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                out_path = os.path.join(tmp_dir, "train_timing.json")
                with open(out_path, "w") as f:
                    json.dump(result, f)
                session.file.put(out_path, "@EPOCH_STAGE/", overwrite=True, auto_compress=False)
            print("Uploaded train_timing.json to @EPOCH_STAGE/", flush=True)

        return {"epoch_time_s": epoch_time, "world_size": world_size}

    except Exception:
        if is_main:
            try:
                from snowflake.snowpark import Session
                _session = Session.builder.getOrCreate()
                _upload_error_json(_session, "train_epoch_error.json", "train_epoch_test.py (train_timing_fn)")
            except Exception as _se:
                print(f"[WARNING] Could not get session for error upload: {_se}", flush=True)
        raise


def main():
    num_nodes = int(os.environ.get("TRAIN_NUM_NODES", "10"))
    num_workers_per_node = 4
    print(f"train_epoch_test: {num_nodes} nodes x {num_workers_per_node} workers/node "
          f"(world_size={num_nodes * num_workers_per_node})",
          flush=True)
    try:
        from snowflake.ml.modeling.distributors.pytorch import (
            PyTorchDistributor, PyTorchScalingConfig, WorkerResourceConfig,
        )

        from snowflake.ml.data.sharded_data_connector import ShardedDataConnector
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
        train_df = session.sql(
            "SELECT split, task_id, stage_path, p, n_train "
            "FROM META_DATASET_INDEX WHERE split = 'train' ORDER BY task_id"
        )
        val_df = session.sql(
            "SELECT split, task_id, stage_path, p, n_train "
            "FROM META_DATASET_INDEX WHERE split = 'val' ORDER BY task_id"
        )
        train_connector = ShardedDataConnector.from_dataframe(train_df, equal=True)
        val_connector   = ShardedDataConnector.from_dataframe(val_df,   equal=False)

        distributor = PyTorchDistributor(
            train_func=train_timing_fn,
            scaling_config=PyTorchScalingConfig(
                num_nodes=num_nodes,
                num_workers_per_node=num_workers_per_node,
                resource_requirements_per_worker=WorkerResourceConfig(
                    num_cpus=4,
                    num_gpus=1,
                ),
            ),
        )
        result = distributor.run(dataset_map={"train": train_connector, "val": val_connector})
        print("train_epoch_test complete:", result, flush=True)
    except Exception:
        try:
            from snowflake.snowpark import Session
            _session = Session.builder.getOrCreate()
            _upload_error_json(_session, "train_epoch_error.json", "train_epoch_test.py (main)")
        except Exception as _se:
            print(f"[WARNING] Could not get session for error upload: {_se}", flush=True)
        raise


if __name__ == "__main__":
    main()
