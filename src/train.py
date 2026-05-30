"""
train.py

Training loop for the DeepSetICLModel inside a Snowpark Container Services (SPCS)
environment.  Reads meta-datasets from Parquet files, trains with early stopping,
and uploads the best checkpoint to a Snowflake model stage.

Usage (inside container):
    python train.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
try:
    from snowflake.ml.modeling.distributors.pytorch import (
        PyTorchDistributor,
        PyTorchScalingConfig,
        WorkerResourceConfig,
        get_context,
    )
except ImportError:
    PyTorchDistributor = None
    PyTorchScalingConfig = None
    WorkerResourceConfig = None
    get_context = None

from model import ModelConfig, _instantiate_model

# ---------------------------------------------------------------------------
# Key constants
# ---------------------------------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR     = "/tmp/data"
PATIENCE     = 10
MAX_EPOCHS   = 200
D_PHI        = 128
D_RHO        = 256
POOL         = "pna"      # "sum"|"mean"|"max"|"pna"|"learned"|"attn"|"multipool"
N_HEADS      = 4
N_SAB_FEAT   = 1
NORM_FEAT    = True
NORM_TARGET  = True
LR           = 1e-3
WEIGHT_DECAY = 1e-4
USE_AMP      = DEVICE == "cuda"

USE_HUBER    = False   # off by default; toggle via hyper_params["use_huber"]
LAMBDA_L1    = 0.0     # L1 penalty coefficient; helps Regime B sparse β
HUBER_DELTA  = 1.0     # δ for Huber loss; robustness to Regime C heavy-tailed ε

# Gate hidden dim — controls Ridge Expert gate MLP width.
# Resolution order in train_fn: BEST_CONFIG["gate_hidden_dim"] > GATE_HIDDEN_DIM env var > 64.
_GATE_HIDDEN_DIM_ENV = os.environ.get("GATE_HIDDEN_DIM", "").strip()
DEFAULT_GATE_HIDDEN_DIM = int(_GATE_HIDDEN_DIM_ENV) if _GATE_HIDDEN_DIM_ENV else 64
if DEFAULT_GATE_HIDDEN_DIM <= 0:
    raise ValueError(
        f"GATE_HIDDEN_DIM env var must be a positive integer, got {_GATE_HIDDEN_DIM_ENV!r}"
    )

MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "market_exchangeable_icl")
TRAINING_DATA_FAMILY = os.environ.get(
    "TRAINING_DATA_FAMILY",
    "unknown"
)
# Valid values: synthetic_regression_primary | synthetic_regression_ood
#               | synthetic_regression_combined | market_mental_model | unknown
_TRAINING_DATA_FAMILY_ALLOWED = frozenset({
    "synthetic_regression_primary",
    "synthetic_regression_ood",
    "synthetic_regression_combined",
    "synthetic_regression_nonlinear",
    "market_mental_model",
    "unknown",
})
if TRAINING_DATA_FAMILY not in _TRAINING_DATA_FAMILY_ALLOWED:
    raise ValueError(
        f"Invalid TRAINING_DATA_FAMILY={TRAINING_DATA_FAMILY!r}. "
        f"Allowed values: {sorted(_TRAINING_DATA_FAMILY_ALLOWED)}"
    )
if TRAINING_DATA_FAMILY == "unknown" and (
    os.environ.get("SNOWFLAKE_HOST", "") or os.environ.get("SF_PYTORCH_DISTRIBUTOR", "")
):
    print(
        "[WARNING] TRAINING_DATA_FAMILY=unknown in a Snowflake environment. "
        "Set TRAINING_DATA_FAMILY explicitly for production auditability. "
        "Production synthetic regression evaluation checkpoints should use "
        "TRAINING_DATA_FAMILY=synthetic_regression_combined.",
        flush=True,
    )

TRAIN_RUN_SANITY_CHECKS    = os.environ.get("TRAIN_RUN_SANITY_CHECKS",    "true").lower() == "true"
TRAIN_SANITY_CHECK_STRICT  = os.environ.get("TRAIN_SANITY_CHECK_STRICT",  "true").lower() == "true"
TRAIN_SANITY_OUT_DIR       = os.environ.get("TRAIN_SANITY_OUT_DIR",       "/tmp/tabpfn_sanity")
TRAIN_SANITY_WRITE_ALL_RANKS = os.environ.get("TRAIN_SANITY_WRITE_ALL_RANKS", "false").lower() == "true"

# MODEL3 runtime selectors
MODEL_ARCH_VERSION    = "model3"
MODEL_DESIGN_PATTERN = os.environ.get("MODEL_DESIGN_PATTERN", "inductive_forecasting")
if MODEL_DESIGN_PATTERN not in ("inductive_forecasting", "transductive_completion"):
    raise ValueError(
        f"Invalid MODEL_DESIGN_PATTERN={MODEL_DESIGN_PATTERN!r}. "
        "Valid: 'inductive_forecasting', 'transductive_completion'"
    )


# ---------------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------------

def load_parquet(path):
    """
    Load a single-row meta-dataset Parquet file.

    Returns:
        X_train    : torch.FloatTensor  (n_train, p)
        y_train    : torch.FloatTensor  (n_train,)
        X_test     : torch.FloatTensor  (n_test,  p)
        betaX_test : torch.FloatTensor  (n_test,)
    """
    table = pq.read_table(path)
    d = table.to_pydict()
    X_train    = torch.tensor(np.array(d["X_train"][0]),    dtype=torch.float32)
    y_train    = torch.tensor(np.array(d["y_train"][0]),    dtype=torch.float32)
    X_test     = torch.tensor(np.array(d["X_test"][0]),     dtype=torch.float32)
    betaX_test = torch.tensor(np.array(d["betaX_test"][0]), dtype=torch.float32)
    return X_train, y_train, X_test, betaX_test


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

class ParquetMetaDataset(Dataset):
    """Each item is one meta-dataset (X_train, y_train, X_test, betaX_test)."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load_parquet(self.files[idx])


def identity_collate(batch):
    """batch_size=1; return the item directly without default list-wrapping."""
    return batch[0]


def make_loader(files, shuffle):
    return DataLoader(
        ParquetMetaDataset(files),
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=USE_AMP,
        persistent_workers=True,
        collate_fn=identity_collate,
    )


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scaler, training: bool, device, use_amp,
              loss_fn=None, l1_lambda=0.0, return_sum_count=False):
    """
    Iterate over all meta-datasets in `loader`.  If training=True, backprop per dataset.
    Returns mean loss across all test-row predictions in the epoch.

    Args:
        loss_fn:   callable(y_hat, y) â†’ scalar loss, or None for MSE.
        l1_lambda: L1 penalty coefficient on model parameters (training only).
    """
    model.train(training)
    total_loss  = 0.0
    total_count = 0

    for X_train, y_train, X_test, betaX_test in loader:
        X_train    = X_train.to(device, non_blocking=True)
        y_train    = y_train.to(device, non_blocking=True)
        X_test     = X_test.to(device, non_blocking=True)
        betaX_test = betaX_test.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            y_hat = model(X_train, y_train, X_test)          # batched: (m,)
            loss  = F.mse_loss(y_hat, betaX_test) if loss_fn is None else loss_fn(y_hat, betaX_test)

        if l1_lambda > 0.0 and training:
            loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss  += loss.item() * X_test.shape[0]
        total_count += X_test.shape[0]

    if return_sum_count:
        return total_loss, total_count
    return total_loss / max(total_count, 1)


def make_no_padding_rank_indices(n_items, rank, world_size):
    """Shard validation items without sampler padding or duplicate examples."""
    return list(range(rank, n_items, world_size))


def loss_sum_count_to_mse(loss_sum, total_count):
    """Compute a weighted MSE from summed squared loss and prediction count."""
    return float(loss_sum) / max(int(total_count), 1)


def reduce_loss_sum_count(loss_sum, total_count, device, dist_module):
    """All-reduce validation loss sums and counts, then return exact global MSE."""
    stats = torch.tensor([float(loss_sum), float(total_count)], device=device)
    dist_module.all_reduce(stats, op=dist_module.ReduceOp.SUM)
    return loss_sum_count_to_mse(stats[0].item(), stats[1].item())


# ---------------------------------------------------------------------------
# Pretrain checkpoint loader
# ---------------------------------------------------------------------------

def _normalize_checkpoint_model_config(saved_cfg, checkpoint_name="checkpoint"):
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


def _checkpoint_architecture_mismatches(saved_cfg, current_cfg):
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
    return {
        field: {
            "saved": getattr(saved_cfg, field, None),
            "current": getattr(current_cfg, field, None),
        }
        for field in fields
        if getattr(saved_cfg, field, None) != getattr(current_cfg, field, None)
    }


def _load_pretrain_checkpoint(model, stage_path, cfg, device, rank,
                              pretrain_load_policy="require_match"):
    """Download and warm-start model from a pretrain checkpoint.

    Called before torch.compile() and DDP wrapping so the plain nn.Module
    receives the state dict. Raises RuntimeError if PRETRAIN_CHECKPOINT_PATH
    is set but the file cannot be downloaded (missing stage file).

    Args:
        pretrain_load_policy: One of:
            - "require_match": raise RuntimeError on architecture mismatch (default).
            - "allow_cold_start_on_arch_mismatch": log and skip load_state_dict on mismatch.

    Returns:
        tuple (pretrain_loaded: bool, pretrain_mismatch_reason: str|None)
    """
    import glob as _glob
    local_dir = f"/tmp/pretrain_ckpt_rank{rank}"
    os.makedirs(local_dir, exist_ok=True)
    # Derive expected local filename from the stage path (e.g. "pretrain_gate64.pt")
    expected_filename = stage_path.rsplit("/", 1)[-1]
    local_path = os.path.join(local_dir, expected_filename)
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
        session.file.get(stage_path, local_dir)
        # session.file.get may rename with a suffix; try exact path first, then glob
        if not os.path.exists(local_path):
            candidates = sorted(_glob.glob(local_path + "*"))
            if candidates:
                local_path = candidates[0]
        if not os.path.exists(local_path):
            # Last resort: any .pt file in the download directory
            all_pts = sorted(_glob.glob(os.path.join(local_dir, "*.pt")))
            if all_pts:
                local_path = all_pts[0]
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"[PRETRAIN] rank {rank}: PRETRAIN_CHECKPOINT_PATH was set to "
                f"{stage_path!r} but {expected_filename!r} could not be downloaded "
                f"to {local_dir!r}. "
                "Verify the file exists with: LIST @MODEL_STAGE/checkpoints/;"
            )
        # PyTorch 2.6+ compat: try weights_only=True first, fall back for legacy ModelConfig
        try:
            ckpt = torch.load(local_path, map_location=device, weights_only=True)
        except Exception as _exc:
            _msg = str(_exc)
            if ("ModelConfig" in _msg or "Weights only load failed" in _msg
                    or "Unsupported global" in _msg or "UnpicklingError" in _msg):
                try:
                    from torch.serialization import safe_globals
                    from model import ModelConfig as _MC
                    with safe_globals([_MC]):
                        ckpt = torch.load(local_path, map_location=device, weights_only=True)
                except Exception:
                    ckpt = torch.load(local_path, map_location=device, weights_only=False)
            else:
                raise
        saved_cfg = _normalize_checkpoint_model_config(ckpt.get("cfg"), "pretrain checkpoint")
        arch_mismatches = _checkpoint_architecture_mismatches(saved_cfg, cfg)
        if arch_mismatches:
            if pretrain_load_policy == "allow_cold_start_on_arch_mismatch":
                print(
                    f"[PRETRAIN] rank {rank}: PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch: "
                    f"mismatch {arch_mismatches}; starting from scratch.",
                    flush=True,
                )
                return False, repr(arch_mismatches)
            else:  # require_match
                raise RuntimeError(
                    f"[PRETRAIN] rank {rank}: Pretrain checkpoint architecture mismatch "
                    f"(PRETRAIN_LOAD_POLICY=require_match): {arch_mismatches}; "
                    f"saved={saved_cfg}, current={cfg}. "
                    "Fix the architecture or use PRETRAIN_LOAD_POLICY=allow_cold_start_on_arch_mismatch."
                )
        model.load_state_dict(ckpt["state_dict"])
        print(f"[PRETRAIN] rank {rank}: loaded pretrain checkpoint from {stage_path}", flush=True)
        return True, None
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"[PRETRAIN] rank {rank}: could not load {stage_path}: {exc}; starting from scratch.", flush=True)
        return False, repr(exc)


# ---------------------------------------------------------------------------
# Snowpark upload
# ---------------------------------------------------------------------------

def upload_to_snowflake(local_path: str, stage_path: str, raise_on_error: bool = False):
    """
    Upload a local file to a Snowflake internal stage via Snowpark.
    Wrapped in try/except so it degrades gracefully when running locally.
    Set raise_on_error=True for diagnostic uploads where silent failure is unacceptable.
    """
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
        session.file.put(local_path, stage_path, overwrite=True, auto_compress=False)
        print(f"Uploaded {local_path} to {stage_path}", flush=True)
    except Exception as exc:
        print(f"[WARNING] Snowpark upload failed (skipping): {exc}", flush=True)
        if raise_on_error:
            raise


def upload_training_failure(exc, stage_path="@MODEL_STAGE/checkpoints/"):
    """
    Upload uncaught training failure details for Snowflake-first diagnostics.
    """
    import json as _json
    import socket as _socket
    import time as _time
    import traceback as _traceback

    payload = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": _traceback.format_exc(),
        "time_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "host": _socket.gethostname(),
        "checkpoint_output_name": os.environ.get("CHECKPOINT_OUTPUT_NAME", ""),
        "train_num_nodes": os.environ.get("TRAIN_NUM_NODES", ""),
        "expected_train_world_size": os.environ.get("EXPECTED_TRAIN_WORLD_SIZE", ""),
    }
    local_path = "train_failure.json"
    with open(local_path, "w", encoding="utf-8") as handle:
        _json.dump(payload, handle, indent=2, sort_keys=True)
    print("[TRAINING FAILURE JSON]", _json.dumps(payload, indent=2, sort_keys=True), flush=True)
    try:
        upload_to_snowflake(local_path, stage_path, raise_on_error=True)
    except Exception as upload_exc:
        print(
            "[TRAINING FAILURE UPLOAD FAILED] "
            f"{type(upload_exc).__name__}: {upload_exc}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Train-time sanity gate
# ---------------------------------------------------------------------------

def _run_train_sanity_gate(model, device, rank, is_main):
    """Run structural sanity checks on this rank's device before compile/DDP."""
    if not TRAIN_RUN_SANITY_CHECKS:
        return
    from sanity_checks import run_all_checks, save_results

    write_results = is_main or TRAIN_SANITY_WRITE_ALL_RANKS
    out_dir = (
        os.path.join(TRAIN_SANITY_OUT_DIR, f"rank{rank}")
        if TRAIN_SANITY_WRITE_ALL_RANKS
        else TRAIN_SANITY_OUT_DIR
    )

    print(f"[train_fn] running model sanity checks on device={device}", flush=True)
    results = run_all_checks(model=model, device=torch.device(device))
    all_passed = results.get("all_passed", False)

    if write_results:
        save_results(results, out_dir)

    if all_passed:
        print("[train_fn] sanity checks passed", flush=True)
    else:
        failed = [k for k, v in results.items()
                  if isinstance(v, dict) and not v.get("passed", True)]
        msg = f"[train_fn] sanity checks FAILED: {failed}"
        print(msg, flush=True)
        if TRAIN_SANITY_CHECK_STRICT:
            raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Distributed training function (invoked by PyTorchDistributor on each worker)
# ---------------------------------------------------------------------------

def train_fn():
    """
    Invoked by PyTorchDistributor on each worker.
    DDP process group is initialized automatically by the distributor.
    All config is read from os.environ â€” PyTorchDistributor calls this with zero args.
    """
    print("[train_fn] entered train_fn", flush=True)
    import json as _json
    import torch.distributed as dist
    from snowflake_io import (
        materialize_indexed_meta_dataset,
        select_rank_sharded_index_rows,
    )

    ctx       = get_context()
    local_rank = ctx.get_local_rank() if hasattr(ctx, "get_local_rank") else ctx.local_rank
    rank = ctx.get_rank() if hasattr(ctx, "get_rank") else ctx.rank
    world_size = ctx.get_world_size() if hasattr(ctx, "get_world_size") else ctx.world_size
    device    = f"cuda:{local_rank}"
    is_main   = (rank == 0)
    import socket as _socket
    print(
        f"[train_fn] worker: rank={rank} local_rank={local_rank} world_size={world_size} "
        f"device={device} host={_socket.gethostname()} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()} "
        f"TRAIN_NUM_NODES={os.environ.get('TRAIN_NUM_NODES', '?')} "
        f"EXPECTED_TRAIN_WORLD_SIZE={os.environ.get('EXPECTED_TRAIN_WORLD_SIZE', '?')} "
        f"STRICT_WORLD_SIZE_CHECK={os.environ.get('STRICT_WORLD_SIZE_CHECK', '?')} "
        f"CHECKPOINT_OUTPUT_NAME={os.environ.get('CHECKPOINT_OUTPUT_NAME', '?')}",
        flush=True,
    )
    use_amp   = True
    pretrain_ckpt_stage_path = os.environ.get("PRETRAIN_CHECKPOINT_PATH", "").strip()
    checkpoint_output_name   = os.environ.get("CHECKPOINT_OUTPUT_NAME", "best.pt")

    # Pretrain load policy — controls behaviour on architecture mismatch
    pretrain_load_policy = os.environ.get("PRETRAIN_LOAD_POLICY", "require_match").strip().lower()
    _VALID_PRETRAIN_POLICIES = {"require_match", "allow_cold_start_on_arch_mismatch"}
    if pretrain_load_policy not in _VALID_PRETRAIN_POLICIES:
        raise ValueError(
            f"Invalid PRETRAIN_LOAD_POLICY={pretrain_load_policy!r}. "
            f"Allowed: {sorted(_VALID_PRETRAIN_POLICIES)}"
        )

    # Tracking variables — initialised before the pretrain block so they are always defined
    pretrain_loaded = False
    pretrain_mismatch_reason = None

    # Pre-training: BEST_CONFIG absent â†’ hyper_params={} â†’ all .get() fall back to
    # module-level defaults (LR=1e-3, D_PHI=128, MAX_EPOCHS=200, etc.). Correct.
    hyper_params = _json.loads(os.environ.get("BEST_CONFIG", "{}"))

    lr           = float(hyper_params.get("lr",           LR))
    weight_decay = float(hyper_params.get("weight_decay", WEIGHT_DECAY))
    d_phi        = int(hyper_params.get("d_phi",          D_PHI))
    d_rho        = int(hyper_params.get("d_rho",          D_RHO))
    dropout      = float(hyper_params.get("dropout",      0.1))
    pool         = hyper_params.get("pool",               POOL)
    max_epochs   = int(hyper_params.get("max_epochs",     MAX_EPOCHS))

    use_huber        = bool(hyper_params.get("use_huber",        USE_HUBER))
    lambda_l1        = float(hyper_params.get("lambda_l1",       LAMBDA_L1))
    huber_delta      = float(hyper_params.get("huber_delta",     HUBER_DELTA))
    use_ridge_expert = bool(hyper_params.get("use_ridge_expert", True))
    ridge_lambda     = float(hyper_params.get("ridge_lambda",    1.0))
    gate_hidden_dim  = int(hyper_params.get("gate_hidden_dim",   DEFAULT_GATE_HIDDEN_DIM))
    n_sab_feat       = int(hyper_params.get("n_sab_feat",        N_SAB_FEAT))
    use_latent_ridge_expert = bool(hyper_params.get("use_latent_ridge_expert", False))
    latent_ridge_dim        = int(hyper_params.get("latent_ridge_dim", 64))
    latent_ridge_lambda     = float(hyper_params.get("latent_ridge_lambda", 1.0))
    latent_ridge_jitter     = float(hyper_params.get("latent_ridge_jitter", 1e-6))
    latent_ridge_use_bias   = bool(hyper_params.get("latent_ridge_use_bias", True))
    use_query_context_attention = bool(hyper_params.get("use_query_context_attention", False))
    query_context_heads     = int(hyper_params.get("query_context_heads", 4))
    icl_pool_mode           = hyper_params.get("icl_pool_mode", "mean")
    feature_pool_mode       = hyper_params.get("feature_pool_mode", "mean")
    ridge_mixture_mode      = hyper_params.get("ridge_mixture_mode", "residual")
    nonlinear_head_hidden_mult = int(hyper_params.get("nonlinear_head_hidden_mult", 2))

    _huber_loss = nn.HuberLoss(delta=huber_delta) if use_huber else None
    loss_fn     = (lambda y_hat, y: _huber_loss(y_hat, y)) if use_huber else None

    # ---- World-size topology check (fires before data materialization) ----
    _expected_ws   = int(os.environ.get("EXPECTED_TRAIN_WORLD_SIZE", "0"))
    _strict_ws     = os.environ.get("STRICT_WORLD_SIZE_CHECK", "").lower() == "true"
    if is_main:
        print(
            f"[train_fn] topology: actual_world_size={world_size}  "
            f"expected_world_size={_expected_ws if _expected_ws > 0 else '(not set)'}  "
            f"TRAIN_NUM_NODES={os.environ.get('TRAIN_NUM_NODES', '?')}  "
            f"num_workers_per_node=4  "
            f"STRICT_WORLD_SIZE_CHECK={_strict_ws}",
            flush=True,
        )
    if _expected_ws > 0 and _strict_ws and world_size != _expected_ws:
        raise RuntimeError(
            f"[train_fn] World-size mismatch: EXPECTED_TRAIN_WORLD_SIZE={_expected_ws} "
            f"but PyTorchDistributor reports world_size={world_size}. "
            f"TRAIN_NUM_NODES={os.environ.get('TRAIN_NUM_NODES', '?')}, "
            "num_workers_per_node=4. "
            "Verify that target_instances in submit_from_stage() equals num_nodes in "
            "PyTorchScalingConfig, and that all nodes in the compute pool are healthy."
        )

    # --- Data: query this rank's rows directly; avoid ShardedDataConnector shard conversion. ---
    train_rows = select_rank_sharded_index_rows("train", rank=rank, world_size=world_size)
    val_rows = select_rank_sharded_index_rows("val", rank=rank, world_size=world_size)
    print(
        f"[train_fn] rank={rank} world_size={world_size} selected_rows "
        f"train={len(train_rows)} val={len(val_rows)} "
        f"train_stage_sample={[row['stage_path'] for row in train_rows[:3]]} "
        f"val_stage_sample={[row['stage_path'] for row in val_rows[:3]]}",
        flush=True,
    )
    files_by_split = materialize_indexed_meta_dataset(
        DATA_DIR,
        splits=("train", "val"),
        rows=train_rows + val_rows,
    )
    train_files = files_by_split["train"]
    val_files = files_by_split["val"]

    # ---- Startup diagnostics (printed before any guard so failures are self-explanatory) ----
    _num_workers_per_node = 4  # mirrors main() PyTorchScalingConfig; logged for cross-check
    print(
        f"[train_fn] rank={rank} local_rank={local_rank} world_size={world_size}  "
        f"TRAIN_NUM_NODES={os.environ.get('TRAIN_NUM_NODES','?')}  "
        f"num_workers_per_node={_num_workers_per_node}  "
        f"checkpoint_output_name={checkpoint_output_name!r}  "
        f"is_pretrain={checkpoint_output_name == 'pretrain.pt'}  "
        f"train_files={len(train_files)}  val_files={len(val_files)}  "
        f"train_sample={train_files[:3]}  val_sample={val_files[:3]}",
        flush=True,
    )

    if not train_files or not val_files:
        raise FileNotFoundError(
            f"Training requires non-empty train and val parquet splits under {DATA_DIR}; "
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

    # --- Model ---
    model_family = hyper_params.get("model_family", MODEL_FAMILY)
    _design_pattern = hyper_params.get("model_design_pattern", MODEL_DESIGN_PATTERN)
    cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool,
                        n_heads=N_HEADS, n_sab_feat=n_sab_feat,
                        norm_feat=NORM_FEAT, norm_target=NORM_TARGET, dropout=dropout,
                        model_family=model_family,
                        model_arch_version="model3",
                        model_design_pattern=_design_pattern,
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
                        nonlinear_head_hidden_mult=nonlinear_head_hidden_mult)
    print(
        f"[train_fn] model_family={cfg.model_family} "
        f"training_data_family={TRAINING_DATA_FAMILY} "
        f"task_type=regression",
        flush=True,
    )
    model = _instantiate_model(cfg).to(device)
    if pretrain_ckpt_stage_path:
        pretrain_loaded, pretrain_mismatch_reason = _load_pretrain_checkpoint(
            model, pretrain_ckpt_stage_path, cfg, device, rank,
            pretrain_load_policy=pretrain_load_policy,
        )
    _run_train_sanity_gate(model, device, rank, is_main)
    model = torch.compile(model, mode="reduce-overhead")
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_mse      = float("inf")
    patience_count    = 0
    best_epoch        = 0
    train_mse_at_best = float("inf")

    for epoch in range(1, max_epochs + 1):
        train_mse = run_epoch(model, train_loader, optimizer, scaler, True,  device, use_amp,
                              loss_fn=loss_fn, l1_lambda=lambda_l1)
        with torch.no_grad():
            val_loss_sum, val_count = run_epoch(
                model, val_loader, None, scaler, False, device, use_amp,
                loss_fn=loss_fn, l1_lambda=0.0, return_sum_count=True,
            )

        val_mse = reduce_loss_sum_count(val_loss_sum, val_count, device, dist)

        if is_main:
            print(f"Epoch {epoch:3d}  val_mse={val_mse:.4f}")
            if val_mse < best_val_mse:
                best_val_mse      = val_mse
                best_epoch        = epoch
                train_mse_at_best = train_mse
                patience_count    = 0
                ckpt = model.module if isinstance(model, DistributedDataParallel) else model
                ckpt = ckpt._orig_mod if hasattr(ckpt, "_orig_mod") else ckpt
                import dataclasses as _dc
                format_version = 4
                _metadata = {
                    "source": "train.py",
                    "checkpoint_name": os.path.basename(checkpoint_output_name),
                    "pytorch_version": torch.__version__,
                    "model_family": cfg.model_family,
                    "task_type": "regression",
                    "training_entrypoint": "train.py",
                    "training_data_family": TRAINING_DATA_FAMILY,
                    "best_val_mse": float(best_val_mse),
                    "train_mse_at_best": float(train_mse_at_best),
                    "best_epoch": int(best_epoch),
                }
                _metadata["model_arch_version"]    = MODEL_ARCH_VERSION
                _metadata["model_design_pattern"] = cfg.model_design_pattern
                _metadata["task_objective"] = (
                    "inductive_regression"
                    if cfg.model_design_pattern == "inductive_forecasting"
                    else "transductive_completion"
                )
                # Include Ridge Expert config and HPO sweep mode from best_config
                _metadata["use_ridge_expert"] = cfg.use_ridge_expert
                _metadata["ridge_lambda"]     = cfg.ridge_lambda
                _metadata["gate_hidden_dim"]  = cfg.gate_hidden_dim
                _metadata["n_sab_feat"]       = cfg.n_sab_feat
                _metadata["use_latent_ridge_expert"] = cfg.use_latent_ridge_expert
                _metadata["latent_ridge_dim"] = cfg.latent_ridge_dim
                _metadata["latent_ridge_lambda"] = cfg.latent_ridge_lambda
                _metadata["latent_ridge_use_bias"] = cfg.latent_ridge_use_bias
                _metadata["use_query_context_attention"] = cfg.use_query_context_attention
                _metadata["query_context_heads"] = cfg.query_context_heads
                _metadata["icl_pool_mode"] = cfg.icl_pool_mode
                _metadata["feature_pool_mode"] = cfg.feature_pool_mode
                _metadata["ridge_mixture_mode"] = cfg.ridge_mixture_mode
                if hyper_params.get("hpo_sweep_mode"):
                    _metadata["hpo_sweep_mode"] = hyper_params["hpo_sweep_mode"]
                # Pretrain load policy tracking
                _metadata["pretrain_loaded"]           = pretrain_loaded
                _metadata["pretrain_checkpoint_path"]  = pretrain_ckpt_stage_path
                _metadata["pretrain_policy"]           = pretrain_load_policy
                if pretrain_mismatch_reason:
                    _metadata["pretrain_mismatch_reason"] = pretrain_mismatch_reason
                torch.save({
                    "checkpoint_format_version": format_version,
                    "cfg": _dc.asdict(ckpt.cfg),
                    "state_dict": ckpt.state_dict(),
                    "metadata": _metadata,
                }, checkpoint_output_name)
            else:
                patience_count += 1

        stop = torch.tensor(int(patience_count >= PATIENCE), device=device)
        dist.broadcast(stop, src=0)
        if stop.item():
            if is_main:
                print("Early stopping.")
            break

    if is_main:
        upload_to_snowflake(checkpoint_output_name, "@MODEL_STAGE/checkpoints/")

    return {"val_mse": best_val_mse}


# ---------------------------------------------------------------------------
# Entry point â€” submits train_fn via PyTorchDistributor
# ---------------------------------------------------------------------------

def main():
    print("[train.py main] entered main", flush=True)
    if PyTorchDistributor is None:
        raise RuntimeError(
            "snowflake.ml is required to submit distributed training. "
            "Run train.py inside the Snowflake ML runtime or install snowflake-ml-python."
        )

    distributor = PyTorchDistributor(
        train_func=train_fn,
        scaling_config=PyTorchScalingConfig(
            num_nodes=int(os.environ.get("TRAIN_NUM_NODES", "10")),
            num_workers_per_node=4,   # 4 A10G GPUs per GPU_NV_M node
            resource_requirements_per_worker=WorkerResourceConfig(
                num_cpus=4,
                num_gpus=1,
            ),
        ),
    )
    # Write a startup artifact before launching workers so we can confirm train.py main()
    # was reached even if distributor.run() fails before Python diagnostics run.
    import json as _json_main
    import time as _time_main
    _startup_payload = {
        "time_utc":                   _time_main.strftime("%Y-%m-%dT%H:%M:%SZ", _time_main.gmtime()),
        "TRAIN_NUM_NODES":            os.environ.get("TRAIN_NUM_NODES", ""),
        "EXPECTED_TRAIN_WORLD_SIZE":  os.environ.get("EXPECTED_TRAIN_WORLD_SIZE", ""),
        "STRICT_WORLD_SIZE_CHECK":    os.environ.get("STRICT_WORLD_SIZE_CHECK", ""),
        "has_best_config":            bool(os.environ.get("BEST_CONFIG", "")),
        "has_pretrain":               bool(os.environ.get("PRETRAIN_CHECKPOINT_PATH", "")),
        "checkpoint_output_name":     os.environ.get("CHECKPOINT_OUTPUT_NAME", "best.pt"),
    }
    print("[train.py main] startup:", _startup_payload, flush=True)
    _startup_local = "training_submission_started.json"
    with open(_startup_local, "w", encoding="utf-8") as _f:
        _json_main.dump(_startup_payload, _f, indent=2, sort_keys=True)
    upload_to_snowflake(_startup_local, "@MODEL_STAGE/checkpoints/")

    print("[train.py main] starting PyTorchDistributor.run", flush=True)
    try:
        result = distributor.run(
            artifact_stage_location="TABPFN_DB.TABPFN_SCHEMA.MODEL_STAGE"
        )
        print("[train.py main] PyTorchDistributor.run completed", flush=True)
    except Exception as exc:
        print("[train.py main] PyTorchDistributor.run failed:", repr(exc), flush=True)
        upload_training_failure(exc)
        raise
    print("Training result:", result)


if __name__ == "__main__":
    main()
