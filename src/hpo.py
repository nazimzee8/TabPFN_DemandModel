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
FIXED_N_SAB_FEAT      = 1
BASE_D_PHI_CANDIDATES = [64, 128, 256, 512]      # kept for hpo_epoch_test.py
BASE_D_RHO_CANDIDATES = [128, 256, 512, 1024]    # kept for hpo_epoch_test.py
HPO_SPLIT_LIMITS      = {"train": 200, "val": 40}
NUM_TRIALS            = 20
MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "market_exchangeable_icl")
TRIAL_MAX_EPOCHS      = 30   # max epochs per HPO trial (early stopping via PATIENCE)

# Architecture sweep candidates — memory-safe defaults.
# d_phi=512 requires explicit DDP memory probe before production use.
ARCH_D_PHI_CANDIDATES      = [64, 128, 192, 256]
ARCH_N_SAB_FEAT_CANDIDATES = [1, 2]

# Gate hidden dim candidates — each requires its own pretrain checkpoint.
# Must match the tune.choice([...]) values in the ridge_residual search space.
GATE_HIDDEN_DIM_CANDIDATES = [32, 64, 128]

# MODEL3 runtime selectors — propagated to HPO workers via env vars.
# MODEL_ARCH_VERSION is hardcoded to "model3".
MODEL_ARCH_VERSION = "model3"
MODEL_DESIGN_PATTERN = os.environ.get("MODEL_DESIGN_PATTERN", "inductive_forecasting")

# HPO sweep mode — controls which search space is used.
# ridge_residual (default): tunes optimizer/regularization/Ridge Expert; architecture fixed.
# architecture: tunes d_phi/n_sab_feat with Ridge Expert fixed; cold-start allowed on mismatch.
HPO_SWEEP_MODE = os.environ.get("HPO_SWEEP_MODE", "ridge_residual").strip().lower()

_ALLOWED_HPO_SWEEP_MODES = {"ridge_residual", "architecture"}
if HPO_SWEEP_MODE not in _ALLOWED_HPO_SWEEP_MODES:
    raise ValueError(
        f"Invalid HPO_SWEEP_MODE={HPO_SWEEP_MODE!r}. "
        f"Allowed values: {sorted(_ALLOWED_HPO_SWEEP_MODES)}"
    )

HPO_BASELINE_CONFIG_STAGE_PATH = os.environ.get(
    "HPO_BASELINE_CONFIG_STAGE_PATH", ""
).strip()


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


# ── baseline config loader (architecture sweep only) ─────────────────────────

def _load_baseline_config_from_stage(stage_path):
    """Download and parse the ridge_residual best_config.json for architecture sweep.

    stage_path must be a non-empty Snowflake stage path such as
    @MODEL_STAGE/hpo/best_config_ridge_residual.json.
    """
    import glob as _glob
    if not stage_path:
        raise ValueError(
            "architecture HPO requires HPO_BASELINE_CONFIG_STAGE_PATH. Run ridge_residual HPO "
            "first, then pass HPO_BASELINE_CONFIG_STAGE_PATH=@MODEL_STAGE/hpo/best_config_ridge_residual.json"
        )
    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        session.file.get(stage_path, tmp_dir)
        # session.file.get may rename the file; glob for any .json
        candidates = sorted(_glob.glob(os.path.join(tmp_dir, "*.json")))
        if not candidates:
            raise FileNotFoundError(
                f"[HPO driver] Failed to download baseline config from {stage_path!r} to {tmp_dir}"
            )
        with open(candidates[0]) as f:
            return json.load(f)


def _merge_sweep_configs(baseline_config, arch_config):
    """Merge ridge_residual baseline config with architecture sweep best config.

    Returns a new dict with:
    - All keys from baseline_config as the base
    - d_phi, n_sab_feat, hpo_sweep_mode overridden from arch_config
    - _meta.sweeps recording both sweeps' provenance
    - _meta.best_val_mse from the architecture sweep (final gate)
    """
    merged = {**baseline_config}
    # Override architecture-specific keys from arch sweep
    for key in ("d_phi", "n_sab_feat", "hpo_sweep_mode"):
        if key in arch_config:
            merged[key] = arch_config[key]

    baseline_meta = baseline_config.get("_meta", {})
    arch_meta = arch_config.get("_meta", {})
    merged["_meta"] = {
        "sweeps": {
            "ridge_residual": {
                "stage_path": "@MODEL_STAGE/hpo/best_config_ridge_residual.json",
                "best_val_mse": baseline_meta.get("best_val_mse"),
                "pretrain_warm_start_policy": baseline_meta.get("pretrain_warm_start_policy"),
            },
            "architecture": {
                "stage_path": "@MODEL_STAGE/hpo/best_config_architecture.json",
                "best_val_mse": arch_meta.get("best_val_mse"),
                "pretrain_warm_start_policy": arch_meta.get("pretrain_warm_start_policy"),
                "d_phi": arch_config.get("d_phi"),
                "n_sab_feat": arch_config.get("n_sab_feat"),
            },
        },
        "merged_from": [
            "@MODEL_STAGE/hpo/best_config_ridge_residual.json",
            "@MODEL_STAGE/hpo/best_config_architecture.json",
        ],
        "best_val_mse": arch_meta.get("best_val_mse"),
    }
    return merged


# ── pretrain checkpoint check ─────────────────────────────────────────────────

def _check_pretrain_checkpoints():
    """Return {32: stage_path, 64: stage_path, 128: stage_path} or fail before HPO trials.

    Requires one pretrain_gate<N>.pt per GATE_HIDDEN_DIM_CANDIDATES entry.
    Call CALL run_pretrain_pipeline(..., gate_hidden_dim) for each candidate first.
    """
    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    try:
        rows = session.sql("LIST @MODEL_STAGE/checkpoints/").collect()
    except Exception as exc:
        raise RuntimeError(
            f"[HPO] Could not verify mandatory pretrain checkpoints: {exc}"
        ) from exc

    found_names = {str(r[0]).rstrip("/").rsplit("/", 1)[-1] for r in rows}
    checkpoint_map = {}
    missing = []
    for gate_dim in GATE_HIDDEN_DIM_CANDIDATES:
        filename = f"pretrain_gate{gate_dim}.pt"
        stage_path = f"@MODEL_STAGE/checkpoints/{filename}"
        if filename in found_names:
            checkpoint_map[gate_dim] = stage_path
            print(f"[HPO] Pretrain checkpoint found: {stage_path}", flush=True)
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"[HPO] Mandatory pretrain checkpoints missing: {missing}. "
            "Run CALL run_pretrain_pipeline(MODEL_FAMILY, TRAINING_DATA_FAMILY, "
            "MODEL_DESIGN_PATTERN, gate_hidden_dim) for each candidate before HPO. "
            f"Expected files: {[f'pretrain_gate{d}.pt' for d in GATE_HIDDEN_DIM_CANDIDATES]}. "
            "Verify with: LIST @MODEL_STAGE/checkpoints/;"
        )

    return checkpoint_map


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
        "use_ridge_expert",
        "gate_hidden_dim",
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
    # Derive expected local filename from the stage path
    expected_filename = stage_path.rsplit("/", 1)[-1]  # e.g. "pretrain_gate64.pt"
    local_path = os.path.join(local_root, expected_filename)
    session = Session.builder.getOrCreate()
    session.file.get(stage_path, local_root)
    if not os.path.exists(local_path):
        # session.file.get may rename; try glob variants of the expected name
        candidates = sorted(_glob.glob(local_path + "*"))
        if candidates:
            local_path = candidates[0]
    if not os.path.exists(local_path):
        # Last resort: any .pt file present in the directory
        all_pts = sorted(_glob.glob(os.path.join(local_root, "*.pt")))
        if all_pts:
            local_path = all_pts[0]
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"[HPO driver] Failed to download mandatory pretrain checkpoint "
            f"{stage_path} to {local_root}"
        )
    return local_path


def _prepare_hpo_payload_on_driver(hpo_rows, checkpoint_map):
    """Materialise HPO data and load all gate-specific pretrain checkpoints.

    checkpoint_map: {gate_dim: stage_path} as returned by _check_pretrain_checkpoints().
    Returns (payload, gate_ckpt_map) where gate_ckpt_map: {gate_dim: checkpoint_dict}.
    """
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

    gate_ckpt_map = {}
    for gate_dim, stage_path in checkpoint_map.items():
        local_root = f"/tmp/hpo_driver_pretrain_{gate_dim}"
        local_checkpoint = _download_pretrain_checkpoint_on_driver(stage_path, local_root)
        checkpoint = _load_torch_checkpoint_cpu(local_checkpoint)
        gate_ckpt_map[gate_dim] = checkpoint
        print(
            f"[HPO driver] loaded pretrain checkpoint gate_hidden_dim={gate_dim}:",
            {"stage_path": stage_path, "local_path": local_checkpoint},
            flush=True,
        )

    print(
        f"[HPO driver] loaded {len(gate_ckpt_map)} gate pretrain checkpoints: "
        f"{sorted(gate_ckpt_map.keys())}",
        flush=True,
    )
    return payload, gate_ckpt_map


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


# ── HPO search space builder ──────────────────────────────────────────────────

def build_hpo_search_space(tune, baseline_config=None) -> dict:
    """Return the Ray Tune search space for the active HPO_SWEEP_MODE.

    ridge_residual (default):
        Fixed architecture (d_phi, d_rho, n_sab_feat, pool).
        Tunes: lr, weight_decay, dropout, ridge_lambda, gate_hidden_dim,
               use_huber, huber_delta, lambda_l1.

    architecture:
        Requires baseline_config (from ridge_residual sweep).
        Fixes all optimizer/regularization params from baseline_config.
        Tunes d_phi from ARCH_D_PHI_CANDIDATES and n_sab_feat from
        ARCH_N_SAB_FEAT_CANDIDATES. Fixed d_rho and pool.
    """
    if HPO_SWEEP_MODE == "ridge_residual":
        return {
            "lr":                   tune.loguniform(1e-4, 3e-3),
            "weight_decay":         tune.loguniform(1e-6, 1e-3),
            "dropout":              tune.uniform(0.0, 0.25),
            "use_ridge_expert":     True,
            "ridge_lambda":         tune.loguniform(1e-3, 1e2),
            "gate_hidden_dim":      tune.choice([32, 64, 128]),
            "use_huber":            tune.choice([False, True]),
            "huber_delta":          tune.choice([0.5, 1.0, 2.0]),
            "lambda_l1":            tune.choice([0.0, 1e-6, 1e-5, 1e-4]),
            "d_phi":                FIXED_D_PHI,
            "n_sab_feat":           FIXED_N_SAB_FEAT,
            "d_rho":                FIXED_D_RHO,
            "pool":                 FIXED_POOL,
            "model_family":         MODEL_FAMILY,
            "model_design_pattern": MODEL_DESIGN_PATTERN,
            "hpo_sweep_mode":       HPO_SWEEP_MODE,
        }
    # architecture sweep: freeze optimizer/regularization from ridge_residual baseline
    if baseline_config is None:
        raise ValueError(
            "architecture HPO requires HPO_BASELINE_CONFIG_STAGE_PATH. Run ridge_residual HPO "
            "first, then pass HPO_BASELINE_CONFIG_STAGE_PATH=@MODEL_STAGE/hpo/best_config_ridge_residual.json"
        )
    return {
        "lr":               float(baseline_config["lr"]),
        "weight_decay":     float(baseline_config["weight_decay"]),
        "dropout":          float(baseline_config["dropout"]),
        "use_ridge_expert": True,
        "ridge_lambda":     float(baseline_config["ridge_lambda"]),
        "gate_hidden_dim":  int(baseline_config["gate_hidden_dim"]),
        "use_huber":        bool(baseline_config.get("use_huber", False)),
        "huber_delta":      float(baseline_config.get("huber_delta", 1.0)),
        "lambda_l1":        float(baseline_config.get("lambda_l1", 0.0)),
        "d_rho":            FIXED_D_RHO,
        "pool":             FIXED_POOL,
        "model_family":     MODEL_FAMILY,
        "model_design_pattern": MODEL_DESIGN_PATTERN,
        "hpo_sweep_mode":   HPO_SWEEP_MODE,
        "d_phi":            tune.choice(ARCH_D_PHI_CANDIDATES),
        "n_sab_feat":       tune.choice(ARCH_N_SAB_FEAT_CANDIDATES),
    }


# ── Ray Tune trainable ────────────────────────────────────────────────────────

def _build_ray_trainable(hpo_data_ref, pretrain_ckpt_map_ref):
    """Return a Ray Tune function trainable over Ray object-store payloads.

    pretrain_ckpt_map_ref: Ray object store ref to {gate_dim: checkpoint_dict}.
    Each trial selects the checkpoint matching its sampled gate_hidden_dim.
    A mismatch is a hard error (not a cold-start fallback).
    """
    def ray_trainable(config):
        import sys
        import torch
        import torch.nn as nn
        import ray
        import ray.tune as tune
        from train import (run_epoch, PATIENCE, N_HEADS, NORM_FEAT, NORM_TARGET)
        from model import ModelConfig, _instantiate_model

        if "snowflake.snowpark" in sys.modules:
            print(
                "[WARN] snowflake.snowpark is loaded in an HPO Ray worker; "
                "workers must not perform Snowflake I/O.",
                flush=True,
            )

        lr               = float(config["lr"])
        weight_decay     = float(config["weight_decay"])
        dropout          = float(config["dropout"])
        d_phi            = int(config.get("d_phi",            FIXED_D_PHI))
        d_rho            = int(config.get("d_rho",            FIXED_D_RHO))
        pool             = config.get("pool",                  FIXED_POOL)
        n_sab_feat       = int(config.get("n_sab_feat",       FIXED_N_SAB_FEAT))
        use_ridge_expert = bool(config.get("use_ridge_expert", True))
        ridge_lambda     = float(config.get("ridge_lambda",   1.0))
        gate_hidden_dim  = int(config.get("gate_hidden_dim",  64))
        use_huber        = bool(config.get("use_huber",        False))
        huber_delta      = float(config.get("huber_delta",    1.0))
        lambda_l1        = float(config.get("lambda_l1",      0.0))

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
        pretrain_ckpt_map = ray.get(pretrain_ckpt_map_ref)
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

        # Select the pretrain checkpoint matching this trial's gate_hidden_dim
        if gate_hidden_dim not in pretrain_ckpt_map:
            raise RuntimeError(
                f"[HPO trial] No pretrain checkpoint for gate_hidden_dim={gate_hidden_dim}. "
                f"Available gate dims: {sorted(pretrain_ckpt_map.keys())}. "
                "Run CALL run_pretrain_pipeline(..., gate_hidden_dim) for all candidates."
            )
        pretrain_ckpt = pretrain_ckpt_map[gate_hidden_dim]

        cfg = ModelConfig(
            d_phi=d_phi, d_rho=d_rho, pool=pool,
            n_heads=N_HEADS, n_sab_feat=n_sab_feat,
            norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout,
            model_family=config.get("model_family", MODEL_FAMILY),
            model_arch_version=MODEL_ARCH_VERSION,
            model_design_pattern=config.get("model_design_pattern", MODEL_DESIGN_PATTERN),
            use_ridge_expert=use_ridge_expert,
            ridge_lambda=ridge_lambda,
            gate_hidden_dim=gate_hidden_dim,
        )
        model = _instantiate_model(cfg).to(device)

        if not pretrain_ckpt:
            raise RuntimeError(
                f"[HPO trial] Missing mandatory pretrain checkpoint for gate_hidden_dim={gate_hidden_dim}"
            )
        _saved_cfg = pretrain_ckpt.get("cfg")
        _arch_mismatches = checkpoint_architecture_mismatches(_saved_cfg, cfg)
        if _arch_mismatches:
            raise RuntimeError(
                f"[HPO trial] Pretrain checkpoint architecture mismatch for "
                f"gate_hidden_dim={gate_hidden_dim}: {_arch_mismatches}; "
                f"saved={_saved_cfg}, current={cfg}. "
                f"The pretrain_gate{gate_hidden_dim}.pt checkpoint must exactly "
                "match the trial architecture (d_phi, gate_hidden_dim, etc.)."
            )
        model.load_state_dict(pretrain_ckpt["state_dict"])
        print(
            f"[HPO trial] Loaded pretrain_gate{gate_hidden_dim}.pt from Ray object store.",
            flush=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler    = torch.cuda.amp.GradScaler(enabled=False)   # BF16 needs no loss scaling

        _huber_loss = nn.HuberLoss(delta=huber_delta) if use_huber else None
        loss_fn     = (lambda y_hat, y: _huber_loss(y_hat, y)) if use_huber else None

        train_loader = make_in_memory_loader(train_records, shuffle=True)
        val_loader   = make_in_memory_loader(val_records,   shuffle=False)

        best_val_mse   = float("inf")
        patience_count = 0
        print(
            f"[HPO trial] starting epoch loop: max_epochs={TRIAL_MAX_EPOCHS}, patience={PATIENCE}, "
            f"use_huber={use_huber}, lambda_l1={lambda_l1}",
            flush=True,
        )

        for epoch in range(1, TRIAL_MAX_EPOCHS + 1):
            run_epoch(model, train_loader, optimizer, scaler, True,  device, use_amp,
                      loss_fn=loss_fn, l1_lambda=lambda_l1)
            with torch.no_grad():
                val_mse = run_epoch(model, val_loader, None, scaler, False, device, use_amp,
                                    loss_fn=loss_fn)
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
    # Gate-specific pretrain checkpoints are required for ridge_residual mode.
    # Architecture sweep uses cold-start (allow_cold_start_on_arch_mismatch) so
    # gate-specific checkpoints are optional; an empty map means all trials cold-start.
    if HPO_SWEEP_MODE == "ridge_residual":
        checkpoint_map = _check_pretrain_checkpoints()
    else:
        # architecture sweep: gate checkpoints optional; cold-start on mismatch
        try:
            checkpoint_map = _check_pretrain_checkpoints()
        except FileNotFoundError:
            checkpoint_map = {}
            print(
                "[HPO architecture] No gate-specific pretrain checkpoints found; "
                "all trials will cold-start (PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch).",
                flush=True,
            )

    # ── metadata selection ────────────────────────────────────────────────────
    hpo_rows           = select_hpo_index_rows()
    max_p, max_n_train = scan_hpo_cardinalities(hpo_rows)
    selected_train_rows = sum(1 for row in hpo_rows if row["split"] == "train")
    selected_val_rows   = sum(1 for row in hpo_rows if row["split"] == "val")

    if HPO_SWEEP_MODE == "ridge_residual":
        enforce_fixed_architecture_cardinality(max_p, max_n_train)
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
    else:
        # architecture sweep: validate that at least one d_phi candidate satisfies max_p
        valid_d_phi = cardinality_aware_candidates(
            "d_phi", ARCH_D_PHI_CANDIDATES, "max_p", max_p
        )
        print(
            f"[HPO architecture] valid d_phi candidates for max_p={max_p}: {valid_d_phi}",
            flush=True,
        )
    print(
        f"[HPO driver] selected index rows: train={selected_train_rows} "
        f"val={selected_val_rows}",
        flush=True,
    )

    # Load baseline config for architecture sweep (driver only, before Ray init)
    baseline_config = None
    if HPO_SWEEP_MODE == "architecture":
        baseline_config = _load_baseline_config_from_stage(HPO_BASELINE_CONFIG_STAGE_PATH)
        print("[HPO driver] loaded baseline config from", HPO_BASELINE_CONFIG_STAGE_PATH, flush=True)

    hpo_payload, gate_ckpt_map = _prepare_hpo_payload_on_driver(
        hpo_rows,
        checkpoint_map,
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
    pretrain_ckpt_map_ref = ray.put(gate_ckpt_map)
    print(
        "[HPO driver] published HPO payload and gate pretrain checkpoint map to Ray object store:",
        {"gate_dims": sorted(gate_ckpt_map.keys())},
        flush=True,
    )
    _run_ray_object_store_preflight(ray)

    # ── Guard: HPO only supports inductive training ────────────────────────────
    if MODEL_DESIGN_PATTERN == "transductive_completion":
        raise ValueError(
            "HPO does not support MODEL_DESIGN_PATTERN='transductive_completion'. "
            "Transductive completion requires a different training objective and cannot "
            "be optimized through the inductive MSE objective in hpo.py. "
            "Set MODEL_DESIGN_PATTERN='inductive_forecasting' to use HPO, or train "
            "a completion model directly via run_model_training()."
        )

    # ── Ray Tune search space ─────────────────────────────────────────────────
    search_space = build_hpo_search_space(tune, baseline_config=baseline_config)
    _arch_info = (
        {
            "d_phi_candidates":      ARCH_D_PHI_CANDIDATES,
            "n_sab_feat_candidates": ARCH_N_SAB_FEAT_CANDIDATES,
            "pretrain_mismatch_policy": "cold_start",
        }
        if HPO_SWEEP_MODE == "architecture"
        else {
            "d_phi":       FIXED_D_PHI,
            "n_sab_feat":  FIXED_N_SAB_FEAT,
            "pretrain_mismatch_policy": "fail_trial",
        }
    )
    print(
        "HPO Ray Tune config:",
        {
            "hpo_sweep_mode":       HPO_SWEEP_MODE,
            "model_family":         MODEL_FAMILY,
            "model_design_pattern": MODEL_DESIGN_PATTERN,
            "num_samples":          NUM_TRIALS,
            "metric":               "val_mse",
            "mode":                 "min",
            "resources_per_trial":  {"gpu": 1},
            "search_alg":           "random (FIFO, no early stopping)",
            "fixed":                {"d_rho": FIXED_D_RHO, "pool": FIXED_POOL},
            "architecture_info":    _arch_info,
        },
        flush=True,
    )

    # ── run trials ────────────────────────────────────────────────────────────
    # tune.run() functional API: stable across Ray 1.x and 2.x.
    # FIFO scheduler + no search_alg = random sampling, matching prior RandomSearch behavior.
    # resources_per_trial={"gpu": 1}: Ray uses lowercase keys.
    trainable = _build_ray_trainable(hpo_data_ref, pretrain_ckpt_map_ref)
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
        "dropout":              float(best_config_raw["dropout"]),

        "d_phi":                int(best_config_raw.get("d_phi",       FIXED_D_PHI)),
        "d_rho":                int(best_config_raw.get("d_rho",       FIXED_D_RHO)),
        "pool":                 best_config_raw.get("pool",             FIXED_POOL),
        "n_sab_feat":           int(best_config_raw.get("n_sab_feat",  FIXED_N_SAB_FEAT)),

        "use_ridge_expert":     bool(best_config_raw.get("use_ridge_expert", True)),
        "ridge_lambda":         float(best_config_raw.get("ridge_lambda",    1.0)),
        "gate_hidden_dim":      int(best_config_raw.get("gate_hidden_dim",   64)),

        "use_huber":            bool(best_config_raw.get("use_huber",    False)),
        "huber_delta":          float(best_config_raw.get("huber_delta", 1.0)),
        "lambda_l1":            float(best_config_raw.get("lambda_l1",   0.0)),

        "model_family":         best_config_raw.get("model_family",         MODEL_FAMILY),
        "model_arch_version":   MODEL_ARCH_VERSION,
        "model_design_pattern": best_config_raw.get("model_design_pattern", MODEL_DESIGN_PATTERN),
        "hpo_sweep_mode":       HPO_SWEEP_MODE,

        "_meta": {
            "best_val_mse":               float(best_val_mse) if best_val_mse is not None else None,
            "num_trials":                 NUM_TRIALS,
            "trial_max_epochs":           TRIAL_MAX_EPOCHS,
            "pretrain_warm_start_policy": "fail_on_mismatch",
            "pretrain_checkpoint_map": {
                str(gate_dim): path for gate_dim, path in checkpoint_map.items()
            },
            "pretrain_checkpoint_stage_path": checkpoint_map.get(
                int(best_config_raw.get("gate_hidden_dim", 64)), ""
            ),
        },
    }

    sweep_filename = f"best_config_{HPO_SWEEP_MODE}.json"
    _upload_json_to_hpo(sweep_filename, best_config)
    print(f"Uploaded {sweep_filename} to @MODEL_STAGE/hpo/", flush=True)

    if HPO_SWEEP_MODE == "architecture" and baseline_config is not None:
        merged = _merge_sweep_configs(baseline_config, best_config)
        _upload_json_to_hpo("best_config.json", merged)
        print("Uploaded merged best_config.json to @MODEL_STAGE/hpo/", flush=True)
    else:
        # ridge_residual: best_config.json == best_config_ridge_residual.json
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
