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
NONLINEAR_TARGET_REGIMES = tuple(
    item.strip()
    for item in os.environ.get("NONLINEAR_TARGET_REGIMES", "I,J,K,L").split(",")
    if item.strip()
)
NONLINEAR_FEATURE_REGIMES = tuple(
    item.strip()
    for item in os.environ.get(
        "NONLINEAR_FEATURE_REGIMES",
        "gaussian_dense,gaussian_sparse,noise_features,high_dim_dense,"
        "high_dim_sparse,correlated_ar,block_correlated,equicorrelated,feature_noise",
    ).split(",")
    if item.strip()
)

# Architecture sweep candidates — memory-safe defaults.
# d_phi=512 requires explicit DDP memory probe before production use.
ARCH_D_PHI_CANDIDATES      = [64, 128, 192, 256]
ARCH_N_SAB_FEAT_CANDIDATES = [1, 2]

# Gate hidden dim candidates — each requires its own pretrain checkpoint.
# Must match the tune.choice([...]) values in the ridge_residual search space.
GATE_HIDDEN_DIM_CANDIDATES = [32, 64, 128]

# MODEL4 runtime selectors — propagated to HPO workers via env vars.
MODEL_ARCH_VERSION = "model4"
MODEL_DESIGN_PATTERN = os.environ.get("MODEL_DESIGN_PATTERN", "inductive_forecasting")

# HPO sweep mode — controls which search space is used.
# ridge_residual (default): tunes optimizer/regularization/Ridge Expert; architecture fixed.
# architecture: tunes d_phi/n_sab_feat with Ridge Expert fixed; cold-start allowed on mismatch.
# nonlinear_meta: tunes latent nonlinear ridge/head/pooling behavior; latent_ridge_dim ∈ {32,64,128}.
# linear_stats (MODEL4): tunes MODEL4 head dims and auxiliary loss weights; backbone fixed.
_HPO_SWEEP_MODE_ALIASES = {
    "ridge_residual": "linear_model",
    "architecture": "linear_model_architecture",
    "linear_stats": "linear_model",
    "nonlinear_meta": "nonlinear_model",
    "nonlinear_architecture": "nonlinear_model_architecture",
}
_CANONICAL_HPO_SWEEP_MODES = {
    "linear_model",
    "linear_model_architecture",
    "nonlinear_model",
    "nonlinear_model_architecture",
}
_HPO_SWEEP_MODE_RAW = os.environ.get("HPO_SWEEP_MODE", "linear_model").strip().lower()
HPO_SWEEP_MODE = _HPO_SWEEP_MODE_ALIASES.get(_HPO_SWEEP_MODE_RAW, _HPO_SWEEP_MODE_RAW)

if HPO_SWEEP_MODE not in _CANONICAL_HPO_SWEEP_MODES:
    raise ValueError(
        f"Invalid HPO_SWEEP_MODE={_HPO_SWEEP_MODE_RAW!r}. "
        f"Allowed values: {sorted(_CANONICAL_HPO_SWEEP_MODES)} "
        f"(deprecated aliases: {sorted(_HPO_SWEEP_MODE_ALIASES)})"
    )
HPO_TRAINING_DATA_FAMILY = os.environ.get(
    "HPO_TRAINING_DATA_FAMILY",
    os.environ.get("TRAINING_DATA_FAMILY", "synthetic_linear_regression"),
).strip()
os.environ["TRAINING_DATA_FAMILY"] = HPO_TRAINING_DATA_FAMILY
from task_routing import (
    CLASSIFICATION_OBJECTIVE,
    get_training_data_spec,
)
HPO_DATA_SPEC = get_training_data_spec(HPO_TRAINING_DATA_FAMILY)
HPO_TASK_OBJECTIVE = HPO_DATA_SPEC.task_objective
HPO_METRIC = HPO_DATA_SPEC.hpo_metric
HPO_METRIC_MODE = HPO_DATA_SPEC.hpo_mode

try:
    from constants import MODEL4_MAX_N_TRAIN, MODEL4_MAX_P
except ImportError:
    from src.constants import MODEL4_MAX_N_TRAIN, MODEL4_MAX_P

DEFAULT_MAX_INDEX_P = "512" if HPO_SWEEP_MODE.startswith("nonlinear_model") else str(MODEL4_MAX_P)
DEFAULT_MAX_INDEX_N_TRAIN = "1024" if HPO_SWEEP_MODE.startswith("nonlinear_model") else str(MODEL4_MAX_N_TRAIN)
HPO_MAX_INDEX_P = int(os.environ.get("HPO_MAX_INDEX_P", DEFAULT_MAX_INDEX_P))
HPO_MAX_INDEX_N_TRAIN = int(os.environ.get(
    "HPO_MAX_INDEX_N_TRAIN",
    DEFAULT_MAX_INDEX_N_TRAIN,
))

HPO_BASELINE_CONFIG_STAGE_PATH = os.environ.get(
    "HPO_BASELINE_CONFIG_STAGE_PATH", ""
).strip()
HPO_PRETRAIN_CHECKPOINT_STAGE_PATH = os.environ.get(
    "HPO_PRETRAIN_CHECKPOINT_STAGE_PATH", ""
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
    """Download and parse the baseline best_config.json for an architecture sweep.

    stage_path must be a non-empty Snowflake stage path such as
    @MODEL_STAGE/hpo/best_config_linear_model.json.
    """
    import glob as _glob
    if not stage_path:
        raise ValueError(
            "architecture HPO requires HPO_BASELINE_CONFIG_STAGE_PATH. Run the task baseline HPO "
            "first, then pass the matching best_config_<mode>.json stage path."
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
    for key in (
        "d_phi",
        "n_sab_feat",
        "hpo_sweep_mode",
        "latent_ridge_dim",
        "icl_pool_mode",
        "feature_pool_mode",
        "use_query_context_attention",
        "query_context_heads",
    ):
        if key in arch_config:
            merged[key] = arch_config[key]

    baseline_meta = baseline_config.get("_meta", {})
    arch_meta = arch_config.get("_meta", {})
    baseline_sweep_name = baseline_config.get("hpo_sweep_mode", "linear_model")
    arch_sweep_name = arch_config.get("hpo_sweep_mode", "linear_model_architecture")
    merged["_meta"] = {
        "sweeps": {
            baseline_sweep_name: {
                "stage_path": f"@MODEL_STAGE/hpo/best_config_{baseline_sweep_name}.json",
                HPO_METRIC: baseline_meta.get(HPO_METRIC),
                "pretrain_warm_start_policy": baseline_meta.get("pretrain_warm_start_policy"),
            },
            arch_sweep_name: {
                "stage_path": f"@MODEL_STAGE/hpo/best_config_{arch_sweep_name}.json",
                HPO_METRIC: arch_meta.get(HPO_METRIC),
                "pretrain_warm_start_policy": arch_meta.get("pretrain_warm_start_policy"),
                "d_phi": arch_config.get("d_phi"),
                "n_sab_feat": arch_config.get("n_sab_feat"),
                "latent_ridge_dim": arch_config.get("latent_ridge_dim"),
                "icl_pool_mode": arch_config.get("icl_pool_mode"),
                "feature_pool_mode": arch_config.get("feature_pool_mode"),
                "use_query_context_attention": arch_config.get("use_query_context_attention"),
            },
        },
        "merged_from": [
            f"@MODEL_STAGE/hpo/best_config_{baseline_sweep_name}.json",
            f"@MODEL_STAGE/hpo/best_config_{arch_sweep_name}.json",
        ],
        HPO_METRIC: arch_meta.get(HPO_METRIC),
    }

    # Preserve pretrain checkpoint metadata through merge (precedence: arch > baseline)
    _arch_ckpt = arch_meta.get("pretrain_checkpoint_stage_path", "")
    _base_ckpt = baseline_meta.get("pretrain_checkpoint_stage_path", "")
    merged["_meta"]["pretrain_checkpoint_stage_path"] = _arch_ckpt if _arch_ckpt else _base_ckpt

    # Merge checkpoint maps: baseline as base, arch overrides on key collision
    merged["_meta"]["pretrain_checkpoint_map"] = {
        **baseline_meta.get("pretrain_checkpoint_map", {}),
        **arch_meta.get("pretrain_checkpoint_map", {}),
    }

    # Propagate training_data_family if either sweep recorded it
    merged["_meta"]["training_data_family"] = (
        arch_meta.get("training_data_family") or baseline_meta.get("training_data_family") or ""
    )

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
    filename_prefix = (
        "pretrain_classification_gate"
        if HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
        else "pretrain_gate"
    )
    for gate_dim in GATE_HIDDEN_DIM_CANDIDATES:
        filename = f"{filename_prefix}{gate_dim}.pt"
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
            f"Expected prefix: {filename_prefix!r}. "
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
    if max_p > HPO_MAX_INDEX_P or max_n_train > HPO_MAX_INDEX_N_TRAIN:
        raise ValueError(
            "Selected HPO rows exceed the configured dataset cardinality contract: "
            f"max_p={max_p} (limit {HPO_MAX_INDEX_P}), "
            f"max_n_train={max_n_train} (limit {HPO_MAX_INDEX_N_TRAIN}). "
            "Adjust HPO_MAX_INDEX_P/HPO_MAX_INDEX_N_TRAIN or regenerate data "
            "within the configured bounds."
        )


def validate_nonlinear_hpo_coverage(rows):
    """Validate selected nonlinear HPO rows cover the intended curriculum."""
    by_split = {}
    for row in rows:
        split = str(row.get("split", ""))
        by_split.setdefault(split, []).append(row)

    errors = []
    for split, limit in HPO_SPLIT_LIMITS.items():
        split_rows = by_split.get(split, [])
        if not split_rows:
            errors.append(f"{split}: no selected rows")
            continue

        target_regimes = {str(row.get("prior_regime", "")) for row in split_rows}
        missing_targets = sorted(set(NONLINEAR_TARGET_REGIMES) - target_regimes)
        if missing_targets:
            errors.append(f"{split}: missing target regimes {missing_targets}")

        feature_regimes = {
            str(row.get("feature_regime", "legacy_gaussian"))
            for row in split_rows
            if row.get("feature_regime", "legacy_gaussian") != "legacy_gaussian"
        }
        if feature_regimes and int(limit) >= len(NONLINEAR_FEATURE_REGIMES):
            missing_features = sorted(set(NONLINEAR_FEATURE_REGIMES) - feature_regimes)
            if missing_features:
                errors.append(f"{split}: missing feature regimes {missing_features}")

    if errors:
        raise ValueError(
            "Selected nonlinear HPO rows do not cover the configured curriculum: "
            + "; ".join(errors)
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


# Fields that are new in MODEL4 — not present in MODEL3 pretrain checkpoints.
# These are excluded from arch-mismatch checks when loading a MODEL3 checkpoint
# for a MODEL4 trial (allow_cold_start_on_arch_mismatch handles the new heads).
_MODEL4_NEW_FIELDS = frozenset({
    "use_linear_stats", "linear_stat_dim", "max_moment_p", "use_sxx_triangle",
    "use_coefficient_head", "coeff_head_hidden_dim",
    "use_lambda_head", "lambda_head_hidden_dim",
    "use_residual_head",
    "beta_aux_loss_weight", "pred_aux_loss_weight", "cos_aux_loss_weight",
    "lambda_aux_loss_weight", "teacher_ols_threshold",
    "linear_encoder_hidden_dim", "fusion_gate_hidden_dim",
})


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
        "use_latent_ridge_expert",
        "latent_ridge_dim",
        "latent_ridge_use_bias",
        "use_query_context_attention",
        "query_context_heads",
        "icl_pool_mode",
        "feature_pool_mode",
        "ridge_mixture_mode",
        "nonlinear_head_hidden_mult",
    )
    all_mismatches = {
        field: {
            "saved": getattr(saved_cfg, field, None),
            "current": getattr(current_cfg, field, None),
        }
        for field in fields
        if getattr(saved_cfg, field, None) != getattr(current_cfg, field, None)
    }
    # MODEL4-only fields are absent from MODEL3 pretrain checkpoints — not real mismatches.
    return {k: v for k, v in all_mismatches.items() if k not in _MODEL4_NEW_FIELDS}


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
        if record.get("task_objective") == CLASSIFICATION_OBJECTIVE:
            return record
        return (
            record["X_train"],
            record["y_train"],
            record["X_test"],
            record["betaX_test"],
            record.get("beta_true"),   # None for old parquet; present for new parquet
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
    from train import load_classification_parquet, load_parquet

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
            if HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE:
                record = load_classification_parquet(path)
                record = {
                    key: value.cpu() if hasattr(value, "cpu") else value
                    for key, value in record.items()
                }
                record["source"] = path
            else:
                X_train, y_train, X_test, betaX_test, beta_true = load_parquet(path)
                record = {
                    "X_train":    X_train.cpu(),
                    "y_train":    y_train.cpu(),
                    "X_test":     X_test.cpu(),
                    "betaX_test": betaX_test.cpu(),
                    "source":     path,
                }
                if beta_true is not None:
                    record["beta_true"] = beta_true.cpu()
            records.append(record)
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
    if HPO_METRIC not in metrics:
        raise KeyError(f"HPO metric report missing {HPO_METRIC!r}: {metrics}")

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

    linear_model (default):
        Fixed architecture (d_phi, d_rho, n_sab_feat, pool).
        Tunes: lr, weight_decay, dropout, ridge_lambda, gate_hidden_dim,
               use_huber, huber_delta, lambda_l1, MODEL4 head dims and
               auxiliary loss weights.

    nonlinear_model:
        Fixed architecture (d_phi, d_rho, n_sab_feat, pool).
        Tunes: lr, weight_decay, dropout, latent_ridge_dim, latent_ridge_lambda,
               latent_ridge_jitter, latent_ridge_use_bias, use_query_context_attention,
               icl_pool_mode, feature_pool_mode, ridge_mixture_mode,
               nonlinear_head_hidden_mult, use_huber, huber_delta, lambda_l1.
        The pretrain checkpoint (pretrain_nonlinear_meta.pt, built with dim=64)
        warm-starts dim=64 trials; dim=32/128 trials cold-start on architecture mismatch.

    linear_model_architecture:
        Requires baseline_config (from linear_model sweep).
        Fixes all optimizer/regularization params from baseline_config.
        Tunes d_phi from ARCH_D_PHI_CANDIDATES and n_sab_feat from
        ARCH_N_SAB_FEAT_CANDIDATES. Fixed d_rho and pool.

    nonlinear_model_architecture:
        Same as linear_model_architecture but for nonlinear models; requires
        baseline_config from nonlinear_model sweep.
    """
    if (
        HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
        and HPO_SWEEP_MODE == "linear_model"
    ):
        return {
            "lr": tune.loguniform(1e-4, 3e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "dropout": tune.uniform(0.0, 0.20),
            "lambda_l1": tune.choice([0.0, 1e-6, 1e-5]),
            "d_phi": FIXED_D_PHI,
            "n_sab_feat": FIXED_N_SAB_FEAT,
            "d_rho": FIXED_D_RHO,
            "pool": FIXED_POOL,
            "model_family": MODEL_FAMILY,
            "model_arch_version": MODEL_ARCH_VERSION,
            "model_design_pattern": MODEL_DESIGN_PATTERN,
            "task_objective": CLASSIFICATION_OBJECTIVE,
            "hpo_sweep_mode": HPO_SWEEP_MODE,
            "use_classification_path": True,
            "use_class_label_embeddings": True,
            "use_classification_stats": True,
            "use_class_stat_fusion": True,
            "use_class_coefficient_head": True,
            "use_class_bias_head": True,
            "use_class_residual_head": False,
            "max_num_classes": 10,
            "class_embedding_dim": tune.choice([32, 64, 128]),
            "class_stat_dim": tune.choice([64, 128, 256]),
            "class_stat_hidden_dim": tune.choice([128, 256]),
            "class_head_hidden_dim": tune.choice([32, 64, 128]),
            "class_fusion_gate_hidden_dim": tune.choice([32, 64]),
            "class_ce_loss_weight": 1.0,
            "class_logit_kl_loss_weight": tune.choice([0.0, 0.025, 0.05]),
            "class_coef_aux_loss_weight": tune.choice([0.0, 0.025, 0.05]),
            "class_margin_aux_loss_weight": tune.choice([0.0, 0.01]),
            "class_prior_aux_loss_weight": tune.choice([0.0, 0.01]),
            "class_calibration_aux_loss_weight": 0.0,
            "class_imbalance_reweighting": tune.choice([False, True]),
            "class_label_smoothing": tune.choice([0.0, 0.05]),
            # Regression-only modules remain present but inactive.
            "use_linear_stats": True,
            "use_coefficient_head": True,
            "use_lambda_head": True,
        }
    if HPO_SWEEP_MODE == "linear_model":
        return {
            "lr":                   tune.loguniform(1e-4, 3e-3),
            "weight_decay":         tune.loguniform(1e-6, 1e-3),
            "dropout":              tune.uniform(0.0, 0.20),
            "use_ridge_expert":     False,
            "ridge_lambda":         1.0,
            "gate_hidden_dim":      64,
            "use_huber":            tune.choice([False, True]),
            "huber_delta":          tune.choice([0.5, 1.0, 2.0]),
            "lambda_l1":            tune.choice([0.0, 1e-6, 1e-5, 1e-4]),
            "d_phi":                FIXED_D_PHI,
            "n_sab_feat":           FIXED_N_SAB_FEAT,
            "d_rho":                FIXED_D_RHO,
            "pool":                 FIXED_POOL,
            "model_family":         MODEL_FAMILY,
            "model_arch_version":   MODEL_ARCH_VERSION,
            "model_design_pattern": MODEL_DESIGN_PATTERN,
            "hpo_sweep_mode":       HPO_SWEEP_MODE,
            "use_linear_stats":     True,
            "use_coefficient_head": True,
            "use_lambda_head":      True,
            "linear_stat_dim":           tune.choice([64, 128, 256]),
            "coeff_head_hidden_dim":     tune.choice([32, 64, 128]),
            "lambda_head_hidden_dim":    tune.choice([16, 32, 64]),
            "linear_encoder_hidden_dim": tune.choice([128, 256]),
            "fusion_gate_hidden_dim":    tune.choice([32, 64]),
            "beta_aux_loss_weight":      tune.loguniform(0.01, 0.5),
            "pred_aux_loss_weight":      tune.loguniform(0.005, 0.2),
            "cos_aux_loss_weight":       tune.loguniform(0.001, 0.1),
            "lambda_aux_loss_weight":    tune.loguniform(0.001, 0.05),
        }
    if HPO_SWEEP_MODE == "nonlinear_model":
        return {
            "lr":                   tune.loguniform(1e-4, 3e-3),
            "weight_decay":         tune.loguniform(1e-6, 1e-3),
            "dropout":              tune.uniform(0.0, 0.25),
            "use_ridge_expert":     False,
            "ridge_lambda":         1.0,
            "gate_hidden_dim":      64,
            "use_latent_ridge_expert": True,
            "latent_ridge_dim":     tune.choice([32, 64, 128]),
            "latent_ridge_lambda":  tune.loguniform(1e-3, 1e2),
            "latent_ridge_jitter":  tune.choice([1e-8, 1e-6, 1e-4]),
            "latent_ridge_use_bias": tune.choice([True, False]),
            "use_query_context_attention": tune.choice([False, True]),
            "query_context_heads":  4,
            "icl_pool_mode":        tune.choice(["mean", "pna", "multipool"]),
            "feature_pool_mode":    tune.choice(["mean", "pna", "attn"]),
            "ridge_mixture_mode":   tune.choice(["residual", "convex"]),
            "nonlinear_head_hidden_mult": tune.choice([1, 2, 4]),
            "use_huber":            tune.choice([False, True]),
            "huber_delta":          tune.choice([0.5, 1.0, 2.0]),
            "lambda_l1":            tune.choice([0.0, 1e-6, 1e-5, 1e-4]),
            "d_phi":                FIXED_D_PHI,
            "n_sab_feat":           FIXED_N_SAB_FEAT,
            "d_rho":                FIXED_D_RHO,
            "pool":                 FIXED_POOL,
            "model_family":         MODEL_FAMILY,
            "model_arch_version":   MODEL_ARCH_VERSION,
            "model_design_pattern": MODEL_DESIGN_PATTERN,
            "hpo_sweep_mode":       HPO_SWEEP_MODE,
            "use_linear_stats":     False,
            "use_coefficient_head": False,
            "use_lambda_head":      False,
        }
    # Architecture sweeps freeze optimizer/head params from the task-specific baseline.
    if baseline_config is None:
        raise ValueError(
            f"{HPO_SWEEP_MODE} HPO requires HPO_BASELINE_CONFIG_STAGE_PATH. Run the baseline HPO "
            "first, then pass the matching best_config stage path."
        )
    is_nonlinear_architecture = HPO_SWEEP_MODE == "nonlinear_model_architecture"
    arch_space = {
        "lr":               float(baseline_config["lr"]),
        "weight_decay":     float(baseline_config["weight_decay"]),
        "dropout":          float(baseline_config["dropout"]),
        "use_ridge_expert": bool(baseline_config.get("use_ridge_expert", False)),
        "ridge_lambda":     float(baseline_config.get("ridge_lambda", 1.0)),
        "gate_hidden_dim":  int(baseline_config.get("gate_hidden_dim", 64)),
        "use_huber":        bool(baseline_config.get("use_huber", False)),
        "huber_delta":      float(baseline_config.get("huber_delta", 1.0)),
        "lambda_l1":        float(baseline_config.get("lambda_l1", 0.0)),
        "d_rho":            FIXED_D_RHO,
        "pool":             FIXED_POOL,
        "model_family":     MODEL_FAMILY,
        "model_arch_version": MODEL_ARCH_VERSION,
        "model_design_pattern": MODEL_DESIGN_PATTERN,
        "hpo_sweep_mode":   HPO_SWEEP_MODE,
        "d_phi":            tune.choice(ARCH_D_PHI_CANDIDATES),
        "n_sab_feat":       tune.choice(ARCH_N_SAB_FEAT_CANDIDATES),
        "use_linear_stats": bool(baseline_config.get("use_linear_stats", not is_nonlinear_architecture)),
        "use_coefficient_head": bool(baseline_config.get("use_coefficient_head", not is_nonlinear_architecture)),
        "use_lambda_head": bool(baseline_config.get("use_lambda_head", not is_nonlinear_architecture)),
        "linear_stat_dim": int(baseline_config.get("linear_stat_dim", 128)),
        "coeff_head_hidden_dim": int(baseline_config.get("coeff_head_hidden_dim", 64)),
        "lambda_head_hidden_dim": int(baseline_config.get("lambda_head_hidden_dim", 32)),
        "linear_encoder_hidden_dim": int(baseline_config.get("linear_encoder_hidden_dim", 256)),
        "fusion_gate_hidden_dim": int(baseline_config.get("fusion_gate_hidden_dim", 64)),
        "beta_aux_loss_weight": float(baseline_config.get("beta_aux_loss_weight", 0.10)),
        "pred_aux_loss_weight": float(baseline_config.get("pred_aux_loss_weight", 0.05)),
        "cos_aux_loss_weight": float(baseline_config.get("cos_aux_loss_weight", 0.02)),
        "lambda_aux_loss_weight": float(baseline_config.get("lambda_aux_loss_weight", 0.01)),
    }
    if HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE:
        arch_space.update({
            "task_objective": CLASSIFICATION_OBJECTIVE,
            "use_classification_path": True,
            "use_class_label_embeddings": bool(
                baseline_config.get("use_class_label_embeddings", True)
            ),
            "use_classification_stats": bool(
                baseline_config.get("use_classification_stats", True)
            ),
            "use_class_stat_fusion": bool(
                baseline_config.get("use_class_stat_fusion", True)
            ),
            "use_class_coefficient_head": bool(
                baseline_config.get("use_class_coefficient_head", True)
            ),
            "use_class_bias_head": bool(
                baseline_config.get("use_class_bias_head", True)
            ),
            "use_class_residual_head": bool(
                baseline_config.get("use_class_residual_head", False)
            ),
            "max_num_classes": int(baseline_config.get("max_num_classes", 10)),
            "class_embedding_dim": int(
                baseline_config.get("class_embedding_dim", 64)
            ),
            "class_stat_dim": int(baseline_config.get("class_stat_dim", 128)),
            "class_stat_hidden_dim": int(
                baseline_config.get("class_stat_hidden_dim", 256)
            ),
            "class_head_hidden_dim": int(
                baseline_config.get("class_head_hidden_dim", 64)
            ),
            "class_fusion_gate_hidden_dim": int(
                baseline_config.get("class_fusion_gate_hidden_dim", 64)
            ),
            "class_ce_loss_weight": float(
                baseline_config.get("class_ce_loss_weight", 1.0)
            ),
            "class_logit_kl_loss_weight": float(
                baseline_config.get("class_logit_kl_loss_weight", 0.05)
            ),
            "class_coef_aux_loss_weight": float(
                baseline_config.get("class_coef_aux_loss_weight", 0.05)
            ),
            "class_margin_aux_loss_weight": float(
                baseline_config.get("class_margin_aux_loss_weight", 0.01)
            ),
            "class_prior_aux_loss_weight": float(
                baseline_config.get("class_prior_aux_loss_weight", 0.01)
            ),
            "class_calibration_aux_loss_weight": float(
                baseline_config.get("class_calibration_aux_loss_weight", 0.0)
            ),
            "class_imbalance_reweighting": bool(
                baseline_config.get("class_imbalance_reweighting", True)
            ),
            "class_label_smoothing": float(
                baseline_config.get("class_label_smoothing", 0.0)
            ),
        })
    if is_nonlinear_architecture:
        arch_space.update({
            "use_latent_ridge_expert": True,
            "latent_ridge_dim": int(baseline_config.get("latent_ridge_dim", 64)),
            "latent_ridge_lambda": float(baseline_config.get("latent_ridge_lambda", 1.0)),
            "latent_ridge_jitter": float(baseline_config.get("latent_ridge_jitter", 1e-6)),
            "latent_ridge_use_bias": bool(baseline_config.get("latent_ridge_use_bias", True)),
            "use_query_context_attention": bool(baseline_config.get("use_query_context_attention", False)),
            "query_context_heads": int(baseline_config.get("query_context_heads", 4)),
            "icl_pool_mode": baseline_config.get("icl_pool_mode", "mean"),
            "feature_pool_mode": baseline_config.get("feature_pool_mode", "mean"),
            "ridge_mixture_mode": baseline_config.get("ridge_mixture_mode", "residual"),
            "nonlinear_head_hidden_mult": int(baseline_config.get("nonlinear_head_hidden_mult", 2)),
        })
    return arch_space


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
        from train import (
            run_classification_epoch,
            run_epoch,
            PATIENCE,
            N_HEADS,
            NORM_FEAT,
            NORM_TARGET,
        )
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
        use_ridge_expert = bool(config.get("use_ridge_expert", False))   # MODEL4 default: False
        ridge_lambda     = float(config.get("ridge_lambda",   1.0))
        gate_hidden_dim  = int(config.get("gate_hidden_dim",  64))
        use_huber        = bool(config.get("use_huber",        False))
        huber_delta      = float(config.get("huber_delta",    1.0))
        lambda_l1        = float(config.get("lambda_l1",      0.0))
        use_latent_ridge_expert = bool(config.get("use_latent_ridge_expert", False))
        latent_ridge_dim        = int(config.get("latent_ridge_dim", 64))
        latent_ridge_lambda     = float(config.get("latent_ridge_lambda", 1.0))
        latent_ridge_jitter     = float(config.get("latent_ridge_jitter", 1e-6))
        latent_ridge_use_bias   = bool(config.get("latent_ridge_use_bias", True))
        use_query_context_attention = bool(config.get("use_query_context_attention", False))
        query_context_heads     = int(config.get("query_context_heads", 4))
        icl_pool_mode           = config.get("icl_pool_mode", "mean")
        feature_pool_mode       = config.get("feature_pool_mode", "mean")
        ridge_mixture_mode      = config.get("ridge_mixture_mode", "residual")
        nonlinear_head_hidden_mult = int(config.get("nonlinear_head_hidden_mult", 2))
        # MODEL4 fields
        use_linear_stats          = bool(config.get("use_linear_stats",          True))
        use_coefficient_head      = bool(config.get("use_coefficient_head",      True))
        use_lambda_head_flag      = bool(config.get("use_lambda_head",           True))
        use_residual_head         = bool(config.get("use_residual_head",         False))
        linear_stat_dim           = int(config.get("linear_stat_dim",           128))
        coeff_head_hidden_dim     = int(config.get("coeff_head_hidden_dim",      64))
        lambda_head_hidden_dim    = int(config.get("lambda_head_hidden_dim",     32))
        beta_aux_loss_weight      = float(config.get("beta_aux_loss_weight",     0.10))
        pred_aux_loss_weight      = float(config.get("pred_aux_loss_weight",     0.05))
        cos_aux_loss_weight       = float(config.get("cos_aux_loss_weight",      0.02))
        lambda_aux_loss_weight    = float(config.get("lambda_aux_loss_weight",   0.01))
        teacher_ols_threshold     = float(config.get("teacher_ols_threshold",    5.0))
        max_moment_p              = int(config.get("max_moment_p",               128))
        linear_encoder_hidden_dim = int(config.get("linear_encoder_hidden_dim",  256))
        fusion_gate_hidden_dim    = int(config.get("fusion_gate_hidden_dim",     64))
        is_classification = HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
        use_classification_path = bool(
            config.get("use_classification_path", is_classification)
        )
        use_class_label_embeddings = bool(
            config.get("use_class_label_embeddings", True)
        )
        use_classification_stats = bool(
            config.get("use_classification_stats", True)
        )
        use_class_stat_fusion = bool(config.get("use_class_stat_fusion", True))
        use_class_coefficient_head = bool(
            config.get("use_class_coefficient_head", True)
        )
        use_class_bias_head = bool(config.get("use_class_bias_head", True))
        use_class_residual_head = bool(
            config.get("use_class_residual_head", False)
        )
        max_num_classes = int(config.get("max_num_classes", 10))
        class_embedding_dim = int(config.get("class_embedding_dim", 64))
        class_stat_dim = int(config.get("class_stat_dim", 128))
        class_stat_hidden_dim = int(config.get("class_stat_hidden_dim", 256))
        class_head_hidden_dim = int(config.get("class_head_hidden_dim", 64))
        class_fusion_gate_hidden_dim = int(
            config.get("class_fusion_gate_hidden_dim", 64)
        )

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

        cfg = ModelConfig(
            d_phi=d_phi, d_rho=d_rho, pool=pool,
            n_heads=N_HEADS, n_sab_feat=n_sab_feat,
            norm_feat=NORM_FEAT,
            norm_target=(NORM_TARGET if not is_classification else False),
            dropout=dropout,
            model_family=config.get("model_family", MODEL_FAMILY),
            model_arch_version=MODEL_ARCH_VERSION,
            model_design_pattern=config.get("model_design_pattern", MODEL_DESIGN_PATTERN),
            task_objective=HPO_TASK_OBJECTIVE,
            use_ridge_expert=use_ridge_expert,
            ridge_lambda=ridge_lambda,
            gate_hidden_dim=gate_hidden_dim,
            use_latent_ridge_expert=use_latent_ridge_expert,
            latent_ridge_dim=latent_ridge_dim,
            latent_ridge_lambda=latent_ridge_lambda,
            latent_ridge_jitter=latent_ridge_jitter,
            latent_ridge_use_bias=latent_ridge_use_bias,
            use_query_context_attention=use_query_context_attention,
            query_context_heads=query_context_heads,
            icl_pool_mode=icl_pool_mode,
            feature_pool_mode=feature_pool_mode,
            ridge_mixture_mode=ridge_mixture_mode,
            nonlinear_head_hidden_mult=nonlinear_head_hidden_mult,
            # MODEL4
            use_linear_stats=use_linear_stats,
            use_coefficient_head=use_coefficient_head,
            use_lambda_head=use_lambda_head_flag,
            use_residual_head=use_residual_head,
            linear_stat_dim=linear_stat_dim,
            coeff_head_hidden_dim=coeff_head_hidden_dim,
            lambda_head_hidden_dim=lambda_head_hidden_dim,
            beta_aux_loss_weight=beta_aux_loss_weight,
            pred_aux_loss_weight=pred_aux_loss_weight,
            cos_aux_loss_weight=cos_aux_loss_weight,
            lambda_aux_loss_weight=lambda_aux_loss_weight,
            teacher_ols_threshold=teacher_ols_threshold,
            max_moment_p=max_moment_p,
            linear_encoder_hidden_dim=linear_encoder_hidden_dim,
            fusion_gate_hidden_dim=fusion_gate_hidden_dim,
            use_classification_path=use_classification_path,
            use_class_label_embeddings=use_class_label_embeddings,
            use_classification_stats=use_classification_stats,
            use_class_stat_fusion=use_class_stat_fusion,
            use_class_coefficient_head=use_class_coefficient_head,
            use_class_bias_head=use_class_bias_head,
            use_class_residual_head=use_class_residual_head,
            max_num_classes=max_num_classes,
            class_embedding_dim=class_embedding_dim,
            class_stat_dim=class_stat_dim,
            class_stat_hidden_dim=class_stat_hidden_dim,
            class_head_hidden_dim=class_head_hidden_dim,
            class_fusion_gate_hidden_dim=class_fusion_gate_hidden_dim,
            class_ce_loss_weight=float(config.get("class_ce_loss_weight", 1.0)),
            class_logit_kl_loss_weight=float(
                config.get("class_logit_kl_loss_weight", 0.05)
            ),
            class_coef_aux_loss_weight=float(
                config.get("class_coef_aux_loss_weight", 0.05)
            ),
            class_margin_aux_loss_weight=float(
                config.get("class_margin_aux_loss_weight", 0.01)
            ),
            class_prior_aux_loss_weight=float(
                config.get("class_prior_aux_loss_weight", 0.01)
            ),
            class_calibration_aux_loss_weight=float(
                config.get("class_calibration_aux_loss_weight", 0.0)
            ),
            class_imbalance_reweighting=bool(
                config.get("class_imbalance_reweighting", True)
            ),
            class_label_smoothing=float(
                config.get("class_label_smoothing", 0.0)
            ),
        )
        model = _instantiate_model(cfg).to(device)

        pretrain_ckpt = None
        if use_latent_ridge_expert:
            pretrain_ckpt = pretrain_ckpt_map.get("nonlinear")
            if pretrain_ckpt:
                _saved_cfg = pretrain_ckpt.get("cfg")
                _arch_mismatches = checkpoint_architecture_mismatches(_saved_cfg, cfg)
                if _arch_mismatches:
                    print(
                        f"[HPO nonlinear] Exact nonlinear pretrain mismatch; cold-starting trial: "
                        f"{_arch_mismatches}",
                        flush=True,
                    )
                    pretrain_ckpt = None
            else:
                print("[HPO nonlinear] No explicit nonlinear pretrain checkpoint; cold-starting.", flush=True)
        elif is_classification:
            if "classification" in pretrain_ckpt_map:
                pretrain_ckpt = pretrain_ckpt_map["classification"]
            else:
                pretrain_ckpt = pretrain_ckpt_map.get(gate_hidden_dim)
            if pretrain_ckpt is None:
                print(
                    "[HPO classification] No compatible pretrain checkpoint; "
                    "cold-starting trial.",
                    flush=True,
                )
        else:
            if "linear" in pretrain_ckpt_map:
                pretrain_ckpt = pretrain_ckpt_map["linear"]
                _saved_cfg = pretrain_ckpt.get("cfg")
                _arch_mismatches = checkpoint_architecture_mismatches(_saved_cfg, cfg)
                if _arch_mismatches:
                    print(
                        f"[HPO linear] Explicit pretrain mismatch; cold-starting trial: "
                        f"{_arch_mismatches}",
                        flush=True,
                    )
                    pretrain_ckpt = None
            elif gate_hidden_dim not in pretrain_ckpt_map:
                if use_linear_stats and not use_ridge_expert:
                    print("[HPO linear] No explicit linear pretrain checkpoint; cold-starting.", flush=True)
                    pretrain_ckpt = None
                else:
                    raise RuntimeError(
                        f"[HPO trial] No pretrain checkpoint for gate_hidden_dim={gate_hidden_dim}. "
                        f"Available gate dims: {sorted(pretrain_ckpt_map.keys())}. "
                        "Run CALL run_pretrain_pipeline(..., gate_hidden_dim) for all candidates."
                    )
            else:
                pretrain_ckpt = pretrain_ckpt_map[gate_hidden_dim]
            if pretrain_ckpt is None:
                pass
            elif not pretrain_ckpt:
                raise RuntimeError(
                    f"[HPO trial] Missing mandatory pretrain checkpoint for gate_hidden_dim={gate_hidden_dim}"
                )
            elif "linear" not in pretrain_ckpt_map:
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
        if pretrain_ckpt:
            current_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in pretrain_ckpt["state_dict"].items()
                if key in current_state and current_state[key].shape == value.shape
            }
            model.load_state_dict(compatible_state, strict=False)
            print(
                "[HPO trial] Loaded shape-compatible pretrain parameters from "
                f"Ray object store ({len(compatible_state)}/{len(current_state)} tensors).",
                flush=True,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler    = torch.cuda.amp.GradScaler(enabled=False)   # BF16 needs no loss scaling

        _huber_loss = nn.HuberLoss(delta=huber_delta) if use_huber else None
        loss_fn     = (lambda y_hat, y: _huber_loss(y_hat, y)) if use_huber else None

        train_loader = make_in_memory_loader(train_records, shuffle=True)
        val_loader   = make_in_memory_loader(val_records,   shuffle=False)

        best_val_metric = float("inf")
        patience_count = 0
        print(
            f"[HPO trial] starting epoch loop: max_epochs={TRIAL_MAX_EPOCHS}, patience={PATIENCE}, "
            f"use_huber={use_huber}, lambda_l1={lambda_l1}",
            flush=True,
        )

        for epoch in range(1, TRIAL_MAX_EPOCHS + 1):
            if is_classification:
                run_classification_epoch(
                    model, train_loader, optimizer, scaler, True, device, use_amp,
                    l1_lambda=lambda_l1, cfg=cfg,
                )
            else:
                run_epoch(
                    model, train_loader, optimizer, scaler, True, device, use_amp,
                    loss_fn=loss_fn, l1_lambda=lambda_l1, cfg=cfg,
                )
            with torch.no_grad():
                if is_classification:
                    val_metric = run_classification_epoch(
                        model, val_loader, None, scaler, False, device, use_amp,
                        cfg=cfg,
                    )
                else:
                    val_metric = run_epoch(
                        model, val_loader, None, scaler, False, device, use_amp,
                        loss_fn=loss_fn, cfg=cfg,
                    )
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                break

        # Metric reporting is Ray-version-sensitive. Always report a dict through
        # _report_hpo_metric() so Ray AIR/Tune versions are handled safely.
        try:
            final_metrics = {HPO_METRIC: float(best_val_metric)}
            print("[HPO trial] final metric:", final_metrics, flush=True)
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
    # HPO can use an explicit task checkpoint, but MODEL4 task sweeps are allowed
    # to cold-start when no exact-compatible pretrain is available.
    if HPO_SWEEP_MODE in ("linear_model", "linear_model_architecture"):
        checkpoint_map = {}
        if HPO_PRETRAIN_CHECKPOINT_STAGE_PATH:
            checkpoint_key = (
                "classification"
                if HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
                else "linear"
            )
            checkpoint_map[checkpoint_key] = HPO_PRETRAIN_CHECKPOINT_STAGE_PATH
        # Deprecated gate-specific checkpoints are still accepted when present.
        try:
            checkpoint_map.update(_check_pretrain_checkpoints())
        except FileNotFoundError:
            print(
                "[HPO linear] No gate-specific pretrain checkpoints found; "
                "trials will cold-start unless HPO_PRETRAIN_CHECKPOINT_STAGE_PATH was provided.",
                flush=True,
            )
    else:
        checkpoint_map = {}
        if HPO_PRETRAIN_CHECKPOINT_STAGE_PATH:
            checkpoint_map["nonlinear"] = HPO_PRETRAIN_CHECKPOINT_STAGE_PATH
            print(
                "[HPO nonlinear] Using explicit nonlinear pretrain candidate: "
                f"{HPO_PRETRAIN_CHECKPOINT_STAGE_PATH}",
                flush=True,
            )

    # ── metadata selection ────────────────────────────────────────────────────
    hpo_rows           = select_hpo_index_rows()
    max_p, max_n_train = scan_hpo_cardinalities(hpo_rows)
    selected_train_rows = sum(1 for row in hpo_rows if row["split"] == "train")
    selected_val_rows   = sum(1 for row in hpo_rows if row["split"] == "val")

    if HPO_SWEEP_MODE.startswith("nonlinear_model"):
        validate_nonlinear_hpo_coverage(hpo_rows)

    if HPO_SWEEP_MODE in ("linear_model", "nonlinear_model"):
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
                "max_index_p_limit":  HPO_MAX_INDEX_P,
                "max_index_n_train_limit": HPO_MAX_INDEX_N_TRAIN,
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
    if HPO_SWEEP_MODE in ("linear_model_architecture", "nonlinear_model_architecture"):
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
    if HPO_SWEEP_MODE in ("linear_model_architecture", "nonlinear_model_architecture"):
        _arch_info = {
            "d_phi_candidates":      ARCH_D_PHI_CANDIDATES,
            "n_sab_feat_candidates": ARCH_N_SAB_FEAT_CANDIDATES,
            "pretrain_mismatch_policy": "cold_start",
        }
    elif HPO_SWEEP_MODE == "nonlinear_model":
        _arch_info = {
            "d_phi":       FIXED_D_PHI,
            "n_sab_feat":  FIXED_N_SAB_FEAT,
            "pretrain_mismatch_policy": "cold_start",
            "latent_ridge_dim_candidates": [32, 64, 128],
        }
    else:
        _arch_info = {
            "d_phi":       FIXED_D_PHI,
            "n_sab_feat":  FIXED_N_SAB_FEAT,
            "pretrain_mismatch_policy": "fail_trial",
        }
    print(
        "HPO Ray Tune config:",
        {
            "hpo_sweep_mode":       HPO_SWEEP_MODE,
            "model_family":         MODEL_FAMILY,
            "model_design_pattern": MODEL_DESIGN_PATTERN,
            "num_samples":          NUM_TRIALS,
            "metric":               HPO_METRIC,
            "mode":                 HPO_METRIC_MODE,
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
        metric=HPO_METRIC,
        mode=HPO_METRIC_MODE,
        resources_per_trial={"gpu": 1},
        verbose=1,
    )

    # ── extract and upload best config ────────────────────────────────────────
    best_config_raw = analysis.best_config
    if best_config_raw is None:
        raise RuntimeError(
            "tune.run() completed but analysis.best_config is None — all trials "
            f"likely failed before reporting {HPO_METRIC!r}. Check per-trial Ray worker "
            "logs; search for '[HPO trial] failed during metric reporting'."
        )

    best_val_metric = analysis.best_result.get(HPO_METRIC)
    print("Best hyperparameters:", best_config_raw, flush=True)
    print(f"Best {HPO_METRIC}:", best_val_metric, flush=True)

    best_config = {
        "lr":                   float(best_config_raw["lr"]),
        "weight_decay":         float(best_config_raw["weight_decay"]),
        "dropout":              float(best_config_raw["dropout"]),

        "d_phi":                int(best_config_raw.get("d_phi",       FIXED_D_PHI)),
        "d_rho":                int(best_config_raw.get("d_rho",       FIXED_D_RHO)),
        "pool":                 best_config_raw.get("pool",             FIXED_POOL),
        "n_sab_feat":           int(best_config_raw.get("n_sab_feat",  FIXED_N_SAB_FEAT)),

        "use_ridge_expert":     bool(best_config_raw.get("use_ridge_expert", False)),
        "ridge_lambda":         float(best_config_raw.get("ridge_lambda",    1.0)),
        "gate_hidden_dim":      int(best_config_raw.get("gate_hidden_dim",   64)),
        "use_latent_ridge_expert": bool(best_config_raw.get("use_latent_ridge_expert", False)),
        "latent_ridge_dim":     int(best_config_raw.get("latent_ridge_dim", 64)),
        "latent_ridge_lambda":  float(best_config_raw.get("latent_ridge_lambda", 1.0)),
        "latent_ridge_jitter":  float(best_config_raw.get("latent_ridge_jitter", 1e-6)),
        "latent_ridge_use_bias": bool(best_config_raw.get("latent_ridge_use_bias", True)),
        "use_query_context_attention": bool(best_config_raw.get("use_query_context_attention", False)),
        "query_context_heads":  int(best_config_raw.get("query_context_heads", 4)),
        "icl_pool_mode":        best_config_raw.get("icl_pool_mode", "mean"),
        "feature_pool_mode":    best_config_raw.get("feature_pool_mode", "mean"),
        "ridge_mixture_mode":   best_config_raw.get("ridge_mixture_mode", "residual"),
        "nonlinear_head_hidden_mult": int(best_config_raw.get("nonlinear_head_hidden_mult", 2)),

        "use_huber":            bool(best_config_raw.get("use_huber",    False)),
        "huber_delta":          float(best_config_raw.get("huber_delta", 1.0)),
        "lambda_l1":            float(best_config_raw.get("lambda_l1",   0.0)),

        "model_family":         best_config_raw.get("model_family",         MODEL_FAMILY),
        "model_arch_version":   MODEL_ARCH_VERSION,
        "model_design_pattern": best_config_raw.get("model_design_pattern", MODEL_DESIGN_PATTERN),
        "hpo_sweep_mode":       HPO_SWEEP_MODE,

        # MODEL4 fields
        "use_linear_stats":          bool(best_config_raw.get("use_linear_stats",          True)),
        "use_coefficient_head":      bool(best_config_raw.get("use_coefficient_head",      True)),
        "use_lambda_head":           bool(best_config_raw.get("use_lambda_head",           True)),
        "use_residual_head":         bool(best_config_raw.get("use_residual_head",         False)),
        "linear_stat_dim":           int(best_config_raw.get("linear_stat_dim",           128)),
        "coeff_head_hidden_dim":     int(best_config_raw.get("coeff_head_hidden_dim",      64)),
        "lambda_head_hidden_dim":    int(best_config_raw.get("lambda_head_hidden_dim",     32)),
        "beta_aux_loss_weight":      float(best_config_raw.get("beta_aux_loss_weight",     0.10)),
        "pred_aux_loss_weight":      float(best_config_raw.get("pred_aux_loss_weight",     0.05)),
        "cos_aux_loss_weight":       float(best_config_raw.get("cos_aux_loss_weight",      0.02)),
        "lambda_aux_loss_weight":    float(best_config_raw.get("lambda_aux_loss_weight",   0.01)),
        "teacher_ols_threshold":     float(best_config_raw.get("teacher_ols_threshold",    5.0)),
        "linear_encoder_hidden_dim": int(best_config_raw.get("linear_encoder_hidden_dim",  256)),
        "fusion_gate_hidden_dim":    int(best_config_raw.get("fusion_gate_hidden_dim",     64)),

        # MODEL4 classification fields
        "task_objective": HPO_TASK_OBJECTIVE,
        "use_classification_path": bool(
            best_config_raw.get(
                "use_classification_path",
                HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE,
            )
        ),
        "use_class_label_embeddings": bool(
            best_config_raw.get("use_class_label_embeddings", True)
        ),
        "use_classification_stats": bool(
            best_config_raw.get("use_classification_stats", True)
        ),
        "use_class_stat_fusion": bool(
            best_config_raw.get("use_class_stat_fusion", True)
        ),
        "use_class_coefficient_head": bool(
            best_config_raw.get("use_class_coefficient_head", True)
        ),
        "use_class_bias_head": bool(
            best_config_raw.get("use_class_bias_head", True)
        ),
        "use_class_residual_head": bool(
            best_config_raw.get("use_class_residual_head", False)
        ),
        "max_num_classes": int(best_config_raw.get("max_num_classes", 10)),
        "class_embedding_dim": int(
            best_config_raw.get("class_embedding_dim", 64)
        ),
        "class_stat_dim": int(best_config_raw.get("class_stat_dim", 128)),
        "class_stat_hidden_dim": int(
            best_config_raw.get("class_stat_hidden_dim", 256)
        ),
        "class_head_hidden_dim": int(
            best_config_raw.get("class_head_hidden_dim", 64)
        ),
        "class_fusion_gate_hidden_dim": int(
            best_config_raw.get("class_fusion_gate_hidden_dim", 64)
        ),
        "class_ce_loss_weight": float(
            best_config_raw.get("class_ce_loss_weight", 1.0)
        ),
        "class_logit_kl_loss_weight": float(
            best_config_raw.get("class_logit_kl_loss_weight", 0.05)
        ),
        "class_coef_aux_loss_weight": float(
            best_config_raw.get("class_coef_aux_loss_weight", 0.05)
        ),
        "class_margin_aux_loss_weight": float(
            best_config_raw.get("class_margin_aux_loss_weight", 0.01)
        ),
        "class_prior_aux_loss_weight": float(
            best_config_raw.get("class_prior_aux_loss_weight", 0.01)
        ),
        "class_calibration_aux_loss_weight": float(
            best_config_raw.get("class_calibration_aux_loss_weight", 0.0)
        ),
        "class_imbalance_reweighting": bool(
            best_config_raw.get("class_imbalance_reweighting", True)
        ),
        "class_label_smoothing": float(
            best_config_raw.get("class_label_smoothing", 0.0)
        ),

        "_meta": {
            HPO_METRIC: (
                float(best_val_metric) if best_val_metric is not None else None
            ),
            "num_trials":                 NUM_TRIALS,
            "trial_max_epochs":           TRIAL_MAX_EPOCHS,
            "pretrain_warm_start_policy": (
                "cold_start_default_exact_match_optional"
                if (
                    bool(best_config_raw.get("use_latent_ridge_expert", False))
                    or (
                        bool(best_config_raw.get("use_linear_stats", True))
                        and not bool(best_config_raw.get("use_ridge_expert", False))
                    )
                )
                else "fail_on_mismatch"
            ),
            "pretrain_checkpoint_map": {
                str(gate_dim): path for gate_dim, path in checkpoint_map.items()
            },
            "pretrain_checkpoint_stage_path": checkpoint_map.get(
                "nonlinear"
                if bool(best_config_raw.get("use_latent_ridge_expert", False))
                else (
                    "classification"
                    if HPO_TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
                    else (
                        "linear"
                        if (
                        bool(best_config_raw.get("use_linear_stats", True))
                        and not bool(best_config_raw.get("use_ridge_expert", False))
                        )
                        else int(best_config_raw.get("gate_hidden_dim", 64))
                    )
                ),
                ""
            ),
            "training_data_family": HPO_TRAINING_DATA_FAMILY,
        },
    }

    sweep_filename = f"best_config_{HPO_SWEEP_MODE}.json"
    _upload_json_to_hpo(sweep_filename, best_config)
    print(f"Uploaded {sweep_filename} to @MODEL_STAGE/hpo/", flush=True)

    if HPO_SWEEP_MODE in ("linear_model_architecture", "nonlinear_model_architecture") and baseline_config is not None:
        merged = _merge_sweep_configs(baseline_config, best_config)
        _upload_json_to_hpo("best_config.json", merged)
        print("Uploaded merged best_config.json to @MODEL_STAGE/hpo/", flush=True)
    else:
        # ridge_residual / nonlinear_meta: best_config.json == best_config_{mode}.json
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
