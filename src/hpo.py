"""
Hyperparameter optimization using Ray Tune on a Snowflake ML Job multi-node cluster.

Run before train.py to find the best config. Writes best_config.json to
@MODEL_STAGE/hpo/ on completion.

Usage (via run_hpo_pipeline stored procedure or directly):
    python hpo.py
"""
print("hpo.py: module load started", flush=True)
import os
os.environ["HOME"] = "/tmp"   # SPCS: redirect ~ to writable path before any connector import

import json
import tempfile
import traceback

print("hpo.py: stdlib imports OK", flush=True)


FIXED_D_PHI           = 128
FIXED_D_RHO           = 256
FIXED_POOL            = "pna"
BASE_D_PHI_CANDIDATES = [64, 128, 256, 512]      # kept for hpo_epoch_test.py
BASE_D_RHO_CANDIDATES = [128, 256, 512, 1024]    # kept for hpo_epoch_test.py
HPO_SPLIT_LIMITS      = {"train": 200, "val": 40}
NUM_TRIALS            = 20
HPO_MODEL_FAMILY      = os.environ.get("DEEPSET_MODEL_FAMILY", "market_aware")
TRIAL_MAX_EPOCHS      = 30   # max epochs per HPO trial (early stopping via PATIENCE)

# MODEL3 runtime selectors — propagated to HPO workers via env vars.
# Default: MODEL_ARCH_VERSION="model2" preserves existing MODEL2 HPO behavior.
# MODEL3 HPO is activated only when MODEL_ARCH_VERSION="model3".
MODEL_ARCH_VERSION    = os.environ.get("MODEL_ARCH_VERSION",    "model2")
MODEL3_DESIGN_PATTERN = os.environ.get("MODEL3_DESIGN_PATTERN", "inductive_forecasting")


# IMPORTANT:
# Snowpark sessions are only available in the HPO MLJob driver process.
# Ray workers do not inherit the Snowflake active session or connector defaults.
# Therefore, all Snowflake I/O must happen before tune.run(), and Ray workers
# must consume in-memory payloads through Ray object store.


# ── stage upload helper ───────────────────────────────────────────────────────

def _upload_json_to_hpo(filename, payload):
    """Upload a JSON payload to @MODEL_STAGE/hpo/.

    Uses Session.builder.getOrCreate() so it works in the head-node driver
    process regardless of whether get_active_session() is available.
    """
    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        tmp = os.path.join(tmp_dir, filename)
        with open(tmp, "w") as f:
            json.dump(payload, f)
        session.file.put(tmp, "@MODEL_STAGE/hpo/", overwrite=True, auto_compress=False)


# ── pretrain checkpoint check ─────────────────────────────────────────────────

def _check_pretrain_checkpoint():
    """Return @MODEL_STAGE/checkpoints/pretrain.pt or fail before HPO trials."""
    stage_path = "@MODEL_STAGE/checkpoints/pretrain.pt"
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
        rows = session.sql("LIST @MODEL_STAGE/checkpoints/").collect()
    except Exception as exc:
        raise RuntimeError(
            "[HPO] Could not verify mandatory pretrain checkpoint at "
            f"{stage_path}: {exc}"
        ) from exc

    if any(str(r[0]).rstrip("/").endswith("/pretrain.pt") for r in rows):
        print(f"[HPO] Pretrain checkpoint found: {stage_path}", flush=True)
        return stage_path

    raise FileNotFoundError(
        "[HPO] Mandatory pretrain checkpoint is missing. Expected "
        f"{stage_path}. Run CALL run_pretrain_pipeline() before HPO and verify "
        "with: LIST @MODEL_STAGE/checkpoints/ PATTERN='.*pretrain[.]pt';"
    )


# ── metadata / cardinality helpers (no heavy imports) ────────────────────────

def select_hpo_index_rows():
    from snowflake_io import select_meta_dataset_index_rows
    return select_meta_dataset_index_rows(
        splits=("train", "val"),
        split_limits=HPO_SPLIT_LIMITS,
        hpo_subset=True,
    )


def _read_positive_cardinality(row, column):
    if column not in row or row[column] is None:
        raise ValueError(f"HPO index metadata found missing {column!r} in {row}")
    value = int(row[column])
    if value <= 0:
        raise ValueError(f"HPO index metadata found non-positive {column!r}={value}")
    return value


def scan_hpo_cardinalities(rows):
    max_p       = max(_read_positive_cardinality(row, "p")       for row in rows)
    max_n_train = max(_read_positive_cardinality(row, "n_train") for row in rows)
    if max_p <= 0 or max_n_train <= 0:
        raise ValueError(
            f"HPO cardinality scan failed: max_p={max_p}, max_n_train={max_n_train}"
        )
    return max_p, max_n_train


def enforce_fixed_architecture_cardinality(max_p, max_n_train):
    if max_p > FIXED_D_PHI or max_n_train > FIXED_D_RHO:
        raise ValueError(
            "Selected HPO rows exceed the fixed warm-start architecture: "
            f"max_p={max_p} (limit {FIXED_D_PHI}), "
            f"max_n_train={max_n_train} (limit {FIXED_D_RHO}). "
            "Regenerate data within these bounds or create a matching pretrain "
            "checkpoint before changing the architecture."
        )


def cardinality_aware_candidates(name, base_candidates, observed_name, observed_value, n_heads=None):
    # n_heads is optional for backward compatibility: hpo_epoch_test.py calls this
    # with 4 positional args. When omitted, N_HEADS is imported from train at call time.
    if n_heads is None:
        from train import N_HEADS
        n_heads = N_HEADS
    candidates = [d for d in base_candidates if d >= observed_value and d % n_heads == 0]
    if not candidates:
        raise ValueError(
            f"No HPO candidates for {name} satisfy {observed_name}={observed_value} "
            f"with n_heads={n_heads}. Expand the base {name} candidate list beyond "
            f"{base_candidates} before launching HPO."
        )
    return candidates


def normalize_checkpoint_model_config(saved_cfg, checkpoint_name="checkpoint"):
    from model import ModelConfig

    if isinstance(saved_cfg, ModelConfig):
        return saved_cfg
    if isinstance(saved_cfg, dict):
        try:
            return ModelConfig(**saved_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{checkpoint_name} has invalid cfg payload: {saved_cfg!r}"
            ) from exc
    if saved_cfg is None:
        raise ValueError(f"{checkpoint_name} is missing required cfg payload")
    raise TypeError(
        f"{checkpoint_name} cfg must be a dict or ModelConfig, "
        f"got {type(saved_cfg).__name__}"
    )


def checkpoint_architecture_mismatches(saved_cfg, current_cfg):
    saved_cfg = normalize_checkpoint_model_config(saved_cfg, "saved checkpoint")
    current_cfg = normalize_checkpoint_model_config(current_cfg, "current model")
    fields = (
        "d_phi",
        "d_rho",
        "pool",
        "n_heads",
        "n_sab_feat",
        "n_sab_samp",
        "norm_feat",
        "norm_target",
        "model_family",
    )
    return {
        field: {
            "saved": getattr(saved_cfg, field, None),
            "current": getattr(current_cfg, field, None),
        }
        for field in fields
        if getattr(saved_cfg, field, None) != getattr(current_cfg, field, None)
    }


def _run_ray_object_store_preflight(ray):
    """Verify Ray workers can read object-store payloads on each alive node."""
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    if not alive_nodes:
        raise RuntimeError("Ray object-store preflight found no alive Ray nodes")

    marker_ref = ray.put({"ok": True, "phase": "hpo_object_store_preflight"})

    @ray.remote(num_cpus=0.1)
    def _object_store_worker_probe(marker_refs):
        import socket as _socket
        import ray as _ray

        value = _ray.get(marker_refs[0])
        return {"host": _socket.gethostname(), "ok": bool(value.get("ok"))}

    refs = []
    for node in alive_nodes:
        try:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            refs.append(
                _object_store_worker_probe.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"], soft=False
                    )
                ).remote([marker_ref])
            )
        except Exception:
            refs.append(_object_store_worker_probe.remote([marker_ref]))

    results = ray.get(refs)
    failed = [result for result in results if not result.get("ok")]
    if failed:
        raise RuntimeError(f"Ray object-store preflight failed: {failed}")
    print(
        "Ray object-store preflight success:",
        {
            "nodes_checked": len(results),
            "hosts": sorted({result["host"] for result in results}),
        },
        flush=True,
    )


def _torch_dataset_base():
    from torch.utils.data import Dataset
    return Dataset


class InMemoryMetaDataset(_torch_dataset_base()):
    """HPO-only dataset over preloaded CPU tensor records."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            record["X_train"],
            record["y_train"],
            record["X_test"],
            record["betaX_test"],
        )


def make_in_memory_loader(records, shuffle):
    import torch
    from torch.utils.data import DataLoader
    from train import identity_collate

    return DataLoader(
        InMemoryMetaDataset(records),
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=identity_collate,
    )


def _load_torch_checkpoint_cpu(path):
    """
    Load a checkpoint in a PyTorch 2.6+ compatible way.
    Prefers weights_only=True; falls back to safe_globals([ModelConfig])
    for legacy pretrain checkpoints; last resort is weights_only=False
    (acceptable here since pretrain.pt is a trusted internally generated artifact).
    """
    import torch
    from model import ModelConfig

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        msg = str(exc)
        if ("ModelConfig" in msg or "Weights only load failed" in msg
                or "Unsupported global" in msg or "UnpicklingError" in msg):
            try:
                from torch.serialization import safe_globals
                with safe_globals([ModelConfig]):
                    return torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                pass
            # Last resort for trusted internally generated pretrain checkpoint
            return torch.load(path, map_location="cpu", weights_only=False)
        if "weights_only" not in msg:
            raise
        # Older PyTorch without weights_only parameter
        return torch.load(path, map_location="cpu")


def _download_pretrain_checkpoint_on_driver(stage_path, local_root):
    import glob as _glob
    from snowflake.snowpark import Session

    os.makedirs(local_root, exist_ok=True)
    local_path = os.path.join(local_root, "pretrain.pt")
    session = Session.builder.getOrCreate()
    session.file.get(stage_path, local_root)
    if not os.path.exists(local_path):
        candidates = sorted(_glob.glob(local_path + "*"))
        if candidates:
            local_path = candidates[0]
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"[HPO driver] Failed to download mandatory pretrain checkpoint "
            f"{stage_path} to {local_root}"
        )
    return local_path


def _prepare_hpo_payload_on_driver(hpo_rows, pretrain_checkpoint_path):
    from snowflake_io import materialize_indexed_meta_dataset
    from train import load_parquet

    files_by_split = materialize_indexed_meta_dataset(
        "/tmp/hpo_driver_data",
        splits=("train", "val"),
        split_limits=HPO_SPLIT_LIMITS,
        hpo_subset=True,
        rows=hpo_rows,
    )
    payload = {}
    for split in ("train", "val"):
        records = []
        for path in files_by_split.get(split, []):
            X_train, y_train, X_test, betaX_test = load_parquet(path)
            records.append(
                {
                    "X_train": X_train.cpu(),
                    "y_train": y_train.cpu(),
                    "X_test": X_test.cpu(),
                    "betaX_test": betaX_test.cpu(),
                    "source": path,
                }
            )
        payload[split] = records

    counts = {split: len(payload.get(split, [])) for split in ("train", "val")}
    print("[HPO driver] materialized in-memory records:", counts, flush=True)
    if counts["train"] != HPO_SPLIT_LIMITS["train"] or counts["val"] != HPO_SPLIT_LIMITS["val"]:
        raise ValueError(
            "[HPO driver] HPO payload has unexpected split counts: "
            f"{counts}; expected {HPO_SPLIT_LIMITS}"
        )

    local_checkpoint = _download_pretrain_checkpoint_on_driver(
        pretrain_checkpoint_path,
        "/tmp/hpo_driver_pretrain",
    )
    print(
        "[HPO driver] downloaded pretrain checkpoint:",
        {"stage_path": pretrain_checkpoint_path, "local_path": local_checkpoint},
        flush=True,
    )
    checkpoint = _load_torch_checkpoint_cpu(local_checkpoint)
    print("[HPO driver] loaded pretrain checkpoint on CPU", flush=True)
    return payload, checkpoint


def _report_hpo_metric(metrics):
    """
    Report metrics from a Ray Tune trainable in a Ray-version-compatible way.

    Preferred:
    - ray.train.report({"val_mse": ...}) for modern Ray AIR/Tune versions.

    Fallbacks:
    - ray.tune.report({"val_mse": ...}) for dictionary-style Tune reporting.
    - ray.tune.report(**metrics) only as a legacy fallback.

    This helper must be called only inside Ray workers.
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"HPO metric report expected dict, got {type(metrics)}")
    if "val_mse" not in metrics:
        raise KeyError(f"HPO metric report missing 'val_mse': {metrics}")

    try:
        from ray import train
        train.report(metrics)
        return
    except Exception as train_report_exc:
        train_report_error = repr(train_report_exc)

    try:
        import ray.tune as tune
        tune.report(metrics)
        return
    except TypeError as dict_report_exc:
        dict_report_error = repr(dict_report_exc)
    except Exception:
        raise

    try:
        import ray.tune as tune
        tune.report(**metrics)
        return
    except Exception as legacy_report_exc:
        raise RuntimeError(
            "Failed to report HPO metric through ray.train.report, "
            "ray.tune.report(dict), and ray.tune.report(**metrics). "
            f"ray.train.report error: {train_report_error}; "
            f"ray.tune.report(dict) error: {dict_report_error}; "
            f"ray.tune.report(**metrics) error: {repr(legacy_report_exc)}"
        ) from legacy_report_exc


# ── Ray Tune trainable ────────────────────────────────────────────────────────

def _build_ray_trainable(hpo_data_ref, pretrain_ckpt_ref):
    """Return a Ray Tune function trainable over Ray object-store payloads."""
    def ray_trainable(config):
        import sys
        import torch
        import ray
        import ray.tune as tune
        from train import (run_epoch, PATIENCE, N_HEADS, N_SAB_FEAT, N_SAB_SAMP,
                           NORM_FEAT, NORM_TARGET)
        from model import DeepSetModel, ModelConfig, _instantiate_model

        if "snowflake.snowpark" in sys.modules:
            print(
                "[WARN] snowflake.snowpark is loaded in an HPO Ray worker; "
                "workers must not perform Snowflake I/O.",
                flush=True,
            )

        lr           = float(config["lr"])
        weight_decay = float(config["weight_decay"])
        dropout      = float(config["dropout"])
        d_phi        = FIXED_D_PHI
        d_rho        = FIXED_D_RHO
        pool         = FIXED_POOL

        device  = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = device == "cuda"
        print(
            "HPO trial runtime:",
            {
                "cuda_available":    torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "device": device,
                "config": config,
            },
            flush=True,
        )

        hpo_payload = ray.get(hpo_data_ref)
        pretrain_ckpt = ray.get(pretrain_ckpt_ref)
        train_records = hpo_payload.get("train", [])
        val_records   = hpo_payload.get("val",   [])
        record_counts = {"train": len(train_records), "val": len(val_records)}
        print(
            "HPO trial received in-memory records:",
            record_counts,
            flush=True,
        )
        if not train_records or not val_records:
            raise FileNotFoundError(
                "HPO requires non-empty train and val in-memory records; "
                f"found {record_counts}"
            )

        cfg = ModelConfig(
            d_phi=d_phi, d_rho=d_rho, pool=pool,
            n_heads=N_HEADS, n_sab_feat=N_SAB_FEAT, n_sab_samp=N_SAB_SAMP,
            norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout,
            model_family=config.get("model_family", HPO_MODEL_FAMILY),
            model_arch_version=MODEL_ARCH_VERSION,
            model3_design_pattern=MODEL3_DESIGN_PATTERN,
        )
        model     = _instantiate_model(cfg).to(device)

        if not pretrain_ckpt:
            raise RuntimeError("[HPO trial] Missing mandatory pretrain checkpoint payload")
        _saved_cfg = pretrain_ckpt.get("cfg")
        _arch_mismatches = checkpoint_architecture_mismatches(_saved_cfg, cfg)
        if _arch_mismatches:
            raise RuntimeError(
                "[HPO trial] Pretrain checkpoint architecture mismatch: "
                f"{_arch_mismatches}; saved={_saved_cfg}, current={cfg}"
            )
        model.load_state_dict(pretrain_ckpt["state_dict"])
        print("[HPO trial] Loaded pretrain checkpoint from Ray object store.", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler    = torch.cuda.amp.GradScaler(enabled=False)   # BF16 needs no loss scaling

        train_loader = make_in_memory_loader(train_records, shuffle=True)
        val_loader   = make_in_memory_loader(val_records,   shuffle=False)

        best_val_mse   = float("inf")
        patience_count = 0
        print(
            f"[HPO trial] starting epoch loop: max_epochs={TRIAL_MAX_EPOCHS}, patience={PATIENCE}",
            flush=True,
        )

        for epoch in range(1, TRIAL_MAX_EPOCHS + 1):
            run_epoch(model, train_loader, optimizer, scaler, True,  device, use_amp)
            with torch.no_grad():
                val_mse = run_epoch(model, val_loader, None, scaler, False, device, use_amp)
            if val_mse < best_val_mse:
                best_val_mse   = val_mse
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                break

        # Metric reporting is Ray-version-sensitive. Always report a dict through
        # _report_hpo_metric() so Ray AIR/Tune versions are handled safely.
        try:
            final_metrics = {"val_mse": float(best_val_mse)}
            print("[HPO trial] final best_val_mse:", final_metrics, flush=True)
            _report_hpo_metric(final_metrics)
        except Exception as report_exc:
            raise RuntimeError(
                f"[HPO trial] failed during metric reporting: {repr(report_exc)}"
            ) from report_exc

    return ray_trainable


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("hpo.py: entered main()", flush=True)

    # Heavy imports inside main() so module-level import failures are caught by
    # the __main__ exception handler and still produce hpo_failure.json.
    import torch
    import ray
    import ray.tune as tune

    print("Ray version:", getattr(ray, "__version__", "unknown"), flush=True)
    print(
        "HPO startup:",
        {
            "cuda_available":    torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        flush=True,
    )
    print("hpo.py: all imports OK", flush=True)

    # ── Ray cluster initialization ────────────────────────────────────────────
    pretrain_checkpoint_path = _check_pretrain_checkpoint()

    # ── metadata selection ────────────────────────────────────────────────────
    hpo_rows           = select_hpo_index_rows()
    max_p, max_n_train = scan_hpo_cardinalities(hpo_rows)
    selected_train_rows = sum(1 for row in hpo_rows if row["split"] == "train")
    selected_val_rows   = sum(1 for row in hpo_rows if row["split"] == "val")

    enforce_fixed_architecture_cardinality(max_p, max_n_train)
    print(
        f"[HPO driver] selected index rows: train={selected_train_rows} "
        f"val={selected_val_rows}",
        flush=True,
    )
    print(
        "Fixed HPO architecture:",
        {
            "max_p":              max_p,
            "max_n_train":        max_n_train,
            "indexed_train_rows": selected_train_rows,
            "indexed_val_rows":   selected_val_rows,
            "d_phi":              FIXED_D_PHI,
            "d_rho":              FIXED_D_RHO,
            "pool":               FIXED_POOL,
        },
        flush=True,
    )
    hpo_payload, pretrain_ckpt = _prepare_hpo_payload_on_driver(
        hpo_rows,
        pretrain_checkpoint_path,
    )

    # address="auto" connects to the Ray cluster that submit_from_stage() already
    # provisioned across all target_instances nodes.
    ray.init(address="auto", ignore_reinit_error=True)
    cluster_resources = ray.cluster_resources()
    print("Ray cluster resources:", cluster_resources, flush=True)

    total_gpus = int(cluster_resources.get("GPU", 0))
    if total_gpus < NUM_TRIALS:
        print(
            f"[WARN] Cluster has {total_gpus} GPUs but NUM_TRIALS={NUM_TRIALS}. "
            "Trials will run in multiple rounds.",
            flush=True,
        )

    hpo_data_ref = ray.put(hpo_payload)
    pretrain_ckpt_ref = ray.put(pretrain_ckpt)
    print("[HPO driver] published HPO payload and pretrain checkpoint to Ray object store", flush=True)
    _run_ray_object_store_preflight(ray)

    # ── Guard: HPO only supports inductive training ────────────────────────────
    if MODEL3_DESIGN_PATTERN == "transductive_completion":
        raise ValueError(
            "HPO does not support MODEL3_DESIGN_PATTERN='transductive_completion'. "
            "Transductive completion requires a different training objective and cannot "
            "be optimized through the inductive MSE objective in hpo.py. "
            "Set MODEL3_DESIGN_PATTERN='inductive_forecasting' to use HPO, or train "
            "a completion model directly via run_model_training()."
        )

    # ── Ray Tune search space ─────────────────────────────────────────────────
    search_space = {
        "lr":           tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "dropout":      tune.uniform(0.0, 0.3),
        "model_family": HPO_MODEL_FAMILY,
    }
    print(
        "HPO Ray Tune config:",
        {
            "num_samples":         NUM_TRIALS,
            "metric":              "val_mse",
            "mode":                "min",
            "resources_per_trial": {"gpu": 1},
            "search_alg":          "random (FIFO, no early stopping)",
            "fixed_architecture":   {
                "d_phi": FIXED_D_PHI,
                "d_rho": FIXED_D_RHO,
                "pool":  FIXED_POOL,
            },
        },
        flush=True,
    )

    # ── run trials ────────────────────────────────────────────────────────────
    # tune.run() functional API: stable across Ray 1.x and 2.x.
    # FIFO scheduler + no search_alg = random sampling, matching prior RandomSearch behavior.
    # resources_per_trial={"gpu": 1}: Ray uses lowercase keys.
    trainable = _build_ray_trainable(hpo_data_ref, pretrain_ckpt_ref)
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=NUM_TRIALS,
        metric="val_mse",
        mode="min",
        resources_per_trial={"gpu": 1},
        verbose=1,
    )

    # ── extract and upload best config ────────────────────────────────────────
    best_config_raw = analysis.best_config
    if best_config_raw is None:
        raise RuntimeError(
            "tune.run() completed but analysis.best_config is None — all trials "
            "likely failed before reporting 'val_mse'. Check per-trial Ray worker "
            "logs; search for '[HPO trial] failed during metric reporting'."
        )

    best_val_mse = analysis.best_result.get("val_mse")
    print("Best hyperparameters:", best_config_raw, flush=True)
    print("Best val_mse:", best_val_mse, flush=True)

    best_config = {
        "lr":                   float(best_config_raw["lr"]),
        "weight_decay":         float(best_config_raw["weight_decay"]),
        "d_phi":                FIXED_D_PHI,
        "d_rho":                FIXED_D_RHO,
        "dropout":              float(best_config_raw["dropout"]),
        "pool":                 FIXED_POOL,
        "model_family":         best_config_raw.get("model_family", HPO_MODEL_FAMILY),
        "model_arch_version":   MODEL_ARCH_VERSION,
        "model3_design_pattern": MODEL3_DESIGN_PATTERN,
    }

    _upload_json_to_hpo("best_config.json", best_config)
    print("Uploaded best_config.json to @MODEL_STAGE/hpo/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("HPO failed:", repr(exc), flush=True)
        print(traceback.format_exc(), flush=True)
        try:
            _upload_json_to_hpo(
                "hpo_failure.json",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "phase": "hpo",
                    "snowflake_io_policy": "driver_only",
                    "ray_worker_snowpark_calls_allowed": False,
                },
            )
            print("Uploaded hpo_failure.json to @MODEL_STAGE/hpo/", flush=True)
        except Exception as upload_exc:
            print("Failed to upload hpo_failure.json:", repr(upload_exc), flush=True)
        raise
