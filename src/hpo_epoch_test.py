"""
hpo_epoch_test.py - HPO epoch timing calibration.
Runs a baseline timing pass and marginal sweeps for runtime-relevant HPO
parameters. Submit via: CALL run_hpo_epoch_test()
Output: @EPOCH_STAGE/hpo_timing.json
"""
import os
os.environ["HOME"] = "/tmp"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime, json, tempfile, time, traceback as _tb
import torch


BASELINE_CONFIG = {
    "d_phi": 128,
    "d_rho": 256,
    "pool": "pna",
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
POOL_CANDIDATES = ["pna", "attn", "multipool"]
NUM_TRIALS = 20
EPOCHS_PER_TRIAL = 30
PARALLEL_TRIALS = 20   # GPU_NV_M: 4 concurrent/node × 5 nodes = 20 concurrent


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


def _make_model(config, device):
    from train import N_HEADS, N_SAB_FEAT, NORM_FEAT, NORM_TARGET
    from model import ModelConfig, _instantiate_model

    cfg = ModelConfig(
        d_phi=int(config["d_phi"]),
        d_rho=int(config["d_rho"]),
        pool=config["pool"],
        n_heads=N_HEADS,
        n_sab_feat=N_SAB_FEAT,
        norm_feat=NORM_FEAT,
        norm_target=NORM_TARGET,
        dropout=float(config["dropout"]),
    )
    return _instantiate_model(cfg).to(device)


def _time_one_epoch(label, parameter, value, config, train_loader, val_loader, device, use_amp):
    from train import run_epoch

    model = _make_model(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    run_epoch(model, train_loader, optimizer, scaler, True, device, use_amp)
    with torch.no_grad():
        val_mse = run_epoch(model, val_loader, None, scaler, False, device, use_amp)

    if device == "cuda":
        torch.cuda.synchronize()
    epoch_time = time.perf_counter() - t0

    result = {
        "label": label,
        "parameter": parameter,
        "value": value,
        "epoch_time_s": round(epoch_time, 2),
        "val_mse": round(float(val_mse), 6),
        "config": dict(config),
    }
    print(f"[TIMING] {label}: {epoch_time:.2f}s", flush=True)
    print("hpo_epoch_test run:", result, flush=True)
    return result


def _build_sweep_runs(d_phi_candidates, d_rho_candidates):
    runs = []
    for d_phi in d_phi_candidates:
        config = dict(BASELINE_CONFIG, d_phi=int(d_phi))
        runs.append(("d_phi", int(d_phi), config))
    for d_rho in d_rho_candidates:
        config = dict(BASELINE_CONFIG, d_rho=int(d_rho))
        runs.append(("d_rho", int(d_rho), config))
    for pool in POOL_CANDIDATES:
        config = dict(BASELINE_CONFIG, pool=pool)
        runs.append(("pool", pool, config))
    return runs


def _hpo_wall_time(epoch_time_s):
    rounds = -(-NUM_TRIALS // PARALLEL_TRIALS)
    return epoch_time_s * EPOCHS_PER_TRIAL * rounds


def main():
    from train import make_loader, DATA_DIR, N_HEADS
    from hpo import (
        BASE_D_PHI_CANDIDATES, BASE_D_RHO_CANDIDATES,
        cardinality_aware_candidates, scan_hpo_cardinalities,
        select_hpo_index_rows, HPO_SPLIT_LIMITS,
    )
    from snowflake_io import materialize_indexed_meta_dataset

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print("hpo_epoch_test startup:", {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": device,
    }, flush=True)

    metadata_t0 = time.perf_counter()
    hpo_rows = select_hpo_index_rows()
    metadata_selection_time_s = time.perf_counter() - metadata_t0

    materialize_t0 = time.perf_counter()
    files_by_split = materialize_indexed_meta_dataset(
        DATA_DIR,
        splits=("train", "val"),
        split_limits=HPO_SPLIT_LIMITS,
        hpo_subset=True,
        rows=hpo_rows,
    )
    materialization_time_s = time.perf_counter() - materialize_t0
    train_files = files_by_split.get("train", [])
    val_files   = files_by_split.get("val", [])
    if not train_files or not val_files:
        raise FileNotFoundError(
            f"hpo_epoch_test requires train and val parquet files under {DATA_DIR}; "
            f"found train={len(train_files)}, val={len(val_files)}"
        )
    print(f"Dataset: {len(train_files)} train, {len(val_files)} val files", flush=True)

    max_p, max_n_train = scan_hpo_cardinalities(hpo_rows)
    d_phi_candidates = cardinality_aware_candidates(
        "d_phi", BASE_D_PHI_CANDIDATES, "max(p)", max_p
    )
    d_rho_candidates = cardinality_aware_candidates(
        "d_rho", BASE_D_RHO_CANDIDATES, "max(n_train)", max_n_train
    )
    print(
        "Cardinality-aware timing dimensions:",
        {
            "max_p": max_p,
            "max_n_train": max_n_train,
            "n_heads": N_HEADS,
            "d_phi_candidates": d_phi_candidates,
            "d_rho_candidates": d_rho_candidates,
            "pool_candidates": POOL_CANDIDATES,
        },
        flush=True,
    )

    train_loader = make_loader(train_files, shuffle=True)
    val_loader   = make_loader(val_files,   shuffle=False)

    baseline = _time_one_epoch(
        "baseline",
        "baseline",
        None,
        BASELINE_CONFIG,
        train_loader,
        val_loader,
        device,
        use_amp,
    )

    runs = []
    for parameter, value, config in _build_sweep_runs(d_phi_candidates, d_rho_candidates):
        label = f"{parameter}={value}"
        runs.append(
            _time_one_epoch(label, parameter, value, config, train_loader, val_loader, device, use_amp)
        )

    epoch_times = [run["epoch_time_s"] for run in runs]
    mean_epoch_time = sum(epoch_times) / len(epoch_times)
    max_epoch_time = max(epoch_times)
    result = {
        "baseline": baseline,
        "runs": runs,
        "summary": {
            "min_epoch_time_s": round(min(epoch_times), 2),
            "mean_epoch_time_s": round(mean_epoch_time, 2),
            "max_epoch_time_s": round(max_epoch_time, 2),
            "estimated_hpo_wall_time_s_mean": round(_hpo_wall_time(mean_epoch_time), 2),
            "estimated_hpo_wall_time_s_conservative": round(_hpo_wall_time(max_epoch_time), 2),
            "num_trials": NUM_TRIALS,
            "epochs_per_trial": EPOCHS_PER_TRIAL,
            "parallel_trials": PARALLEL_TRIALS,
            "hpo_rounds": -(-NUM_TRIALS // PARALLEL_TRIALS),
        },
        "metadata": {
            "device": device,
            "train_files": len(train_files),
            "val_files": len(val_files),
            "metadata_selection_time_s": round(metadata_selection_time_s, 2),
            "materialization_time_s": round(materialization_time_s, 2),
            "baseline_config": BASELINE_CONFIG,
            "d_phi_candidates": d_phi_candidates,
            "d_rho_candidates": d_rho_candidates,
            "pool_candidates": POOL_CANDIDATES,
            "fixed_parameters": ["lr", "weight_decay", "dropout"],
        },
    }
    print("hpo_epoch_test result:", result, flush=True)

    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        out_path = os.path.join(tmp_dir, "hpo_timing.json")
        with open(out_path, "w") as f:
            json.dump(result, f)
        session.file.put(out_path, "@EPOCH_STAGE/", overwrite=True, auto_compress=False)
    print("Uploaded hpo_timing.json to @EPOCH_STAGE/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] hpo_epoch_test.py: {type(exc).__name__}: {exc}", flush=True)
        print(_tb.format_exc(), flush=True)
        try:
            from snowflake.snowpark import Session
            _session = Session.builder.getOrCreate()
            _upload_error_json(_session, "hpo_epoch_error.json", "hpo_epoch_test.py")
        except Exception as _se:
            print(f"[WARNING] Could not upload hpo_epoch_error.json: {_se}", flush=True)
        raise
