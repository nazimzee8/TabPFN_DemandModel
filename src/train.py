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
from classification import (
    LogisticTeacher,
    canonicalize_class_coefficients,
    compute_classification_losses,
    inverse_frequency_class_weight,
    permute_class_labels,
)
from support_augmentation import permute_support_rows
from task_routing import (
    CLASSIFICATION_OBJECTIVE,
    allowed_training_data_families,
    get_training_data_spec,
)
from constants import (
    MIXED_CAT_REGRESSION_TRAINING_FAMILY,
    MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
    CHECKPOINT_VERSION_MIXED_REGRESSION,
    CHECKPOINT_VERSION_MIXED_CLASSIFICATION,
)

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
# Valid values: synthetic_linear_regression | synthetic_regression_primary
#               | synthetic_regression_ood | synthetic_regression_combined
#               | market_mental_model | unknown
_TRAINING_DATA_FAMILY_ALLOWED = allowed_training_data_families()
if TRAINING_DATA_FAMILY not in _TRAINING_DATA_FAMILY_ALLOWED:
    raise ValueError(
        f"Invalid TRAINING_DATA_FAMILY={TRAINING_DATA_FAMILY!r}. "
        f"Allowed values: {sorted(_TRAINING_DATA_FAMILY_ALLOWED)}"
    )
TRAINING_DATA_SPEC = get_training_data_spec(TRAINING_DATA_FAMILY)
TASK_OBJECTIVE = TRAINING_DATA_SPEC.task_objective
if TRAINING_DATA_FAMILY == "unknown" and (
    os.environ.get("SNOWFLAKE_HOST", "") or os.environ.get("SF_PYTORCH_DISTRIBUTOR", "")
):
    print(
        "[WARNING] TRAINING_DATA_FAMILY=unknown in a Snowflake environment. "
        "Set TRAINING_DATA_FAMILY explicitly for production auditability. "
        "Production synthetic regression evaluation checkpoints should use "
        "TRAINING_DATA_FAMILY=synthetic_linear_regression.",
        flush=True,
    )

TRAIN_RUN_SANITY_CHECKS    = os.environ.get("TRAIN_RUN_SANITY_CHECKS",    "true").lower() == "true"
TRAIN_SANITY_CHECK_STRICT  = os.environ.get("TRAIN_SANITY_CHECK_STRICT",  "true").lower() == "true"
TRAIN_SANITY_OUT_DIR       = os.environ.get("TRAIN_SANITY_OUT_DIR",       "/tmp/tabpfn_sanity")
TRAIN_SANITY_WRITE_ALL_RANKS = os.environ.get("TRAIN_SANITY_WRITE_ALL_RANKS", "false").lower() == "true"

# MODEL4 runtime selectors
MODEL_ARCH_VERSION    = "model4"
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
        beta       : torch.FloatTensor  (p,) or None — ground-truth beta (MODEL4 aux loss)

    Field priority: 'beta' (canonical, written by generate_dgp.py) >
                    'beta_true' (legacy fallback) > None.
    """
    table = pq.read_table(path)
    d = table.to_pydict()
    X_train    = torch.tensor(np.array(d["X_train"][0]),    dtype=torch.float32)
    y_train    = torch.tensor(np.array(d["y_train"][0]),    dtype=torch.float32)
    X_test     = torch.tensor(np.array(d["X_test"][0]),     dtype=torch.float32)
    betaX_test = torch.tensor(np.array(d["betaX_test"][0]), dtype=torch.float32)
    # Read 'beta' first (canonical), fall back to 'beta_true' (legacy), then None
    if "beta" in d:
        beta = torch.tensor(np.array(d["beta"][0]), dtype=torch.float32)
    elif "beta_true" in d:
        beta = torch.tensor(np.array(d["beta_true"][0]), dtype=torch.float32)
    else:
        beta = None
    return X_train, y_train, X_test, betaX_test, beta


def _first_present(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name][0]
    return None


def load_classification_parquet(path, *, min_support_per_class: int = 1, global_idx: int | None = None):
    """Load one `linear_classification_v1` task as a validated dictionary.

    Parameters
    ----------
    min_support_per_class : int
        Minimum number of training samples required per class (default 1).
        Pass 0 to disable the check (e.g., for stress-regime evaluation tasks).
    global_idx : int | None
        Optional task index included in error messages.
    """
    table = pq.read_table(path)
    d = table.to_pydict()
    required = ("X_train", "y_train", "X_test", "y_test", "num_classes")
    missing = [name for name in required if name not in d]
    if missing:
        raise ValueError(
            f"Classification parquet {path!r} is missing required fields: {missing}"
        )

    X_train = torch.tensor(np.asarray(d["X_train"][0]), dtype=torch.float32)
    X_test = torch.tensor(np.asarray(d["X_test"][0]), dtype=torch.float32)
    y_train = torch.tensor(np.asarray(d["y_train"][0]), dtype=torch.long)
    y_test = torch.tensor(np.asarray(d["y_test"][0]), dtype=torch.long)
    num_classes = int(d["num_classes"][0])
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Classification X_train and X_test must be rank-2 arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Classification train/query feature widths must match.")
    if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
        raise ValueError("Classification y_train must align with X_train rows.")
    if y_test.ndim != 1 or y_test.shape[0] != X_test.shape[0]:
        raise ValueError("Classification y_test must align with X_test rows.")
    if num_classes < 2:
        raise ValueError("Classification tasks require num_classes >= 2.")

    # F8: derive canonical label map from y_train only (no query-label leakage).
    unique_labels = torch.unique(y_train, sorted=True)
    if unique_labels.numel() < 2:
        raise ValueError("Classification tasks require at least two realized classes.")
    if int(unique_labels.min()) < 0:
        raise ValueError("Classification labels must be non-negative.")

    W_value = _first_present(d, "W_true", "w_true")
    b_value = _first_present(d, "b_true")
    W_true = (
        torch.tensor(np.asarray(W_value), dtype=torch.float32)
        if W_value is not None else None
    )
    b_true = (
        torch.tensor(np.asarray(b_value), dtype=torch.float32)
        if b_value is not None else None
    )

    contiguous = torch.arange(
        unique_labels.numel(), dtype=unique_labels.dtype
    )
    unseen_query_classes: list[int] = []
    if not torch.equal(unique_labels.cpu(), contiguous):
        label_map = {int(old): new for new, old in enumerate(unique_labels.tolist())}
        y_train = torch.tensor(
            [label_map[int(value)] for value in y_train.tolist()], dtype=torch.long
        )
        # Query labels not in label_map (unseen at train time) go to OOD bucket.
        _ood_bucket = unique_labels.numel() - 1
        unseen_query_classes = sorted({
            int(v) for v in y_test.tolist() if int(v) not in label_map
        })
        y_test = torch.tensor(
            [label_map.get(int(value), _ood_bucket) for value in y_test.tolist()],
            dtype=torch.long,
        )
        if W_true is not None and W_true.ndim == 2:
            if W_true.shape[1] > int(unique_labels.max()):
                W_true = W_true[:, unique_labels.long()]
            elif W_true.shape[0] > int(unique_labels.max()):
                W_true = W_true[unique_labels.long(), :]
        if b_true is not None and b_true.ndim == 1 and b_true.numel() > int(unique_labels.max()):
            b_true = b_true[unique_labels.long()]
        num_classes = int(unique_labels.numel())
    else:
        if int(unique_labels.max()) >= num_classes:
            raise ValueError(
                f"Classification label {int(unique_labels.max())} exceeds "
                f"num_classes={num_classes}."
            )
        # Check for query labels outside the train-derived label set.
        unseen_query_classes = sorted({
            int(v) for v in y_test.tolist()
            if int(v) not in set(unique_labels.tolist())
        })

    # F3: minimum support count check — must run after label remapping.
    support_class_counts = {
        cls: int((y_train == cls).sum()) for cls in range(num_classes)
    }
    missing_support = [cls for cls, cnt in support_class_counts.items() if cnt == 0]
    if missing_support and min_support_per_class > 0:
        raise ValueError(
            f"Classes {missing_support} have 0 support samples "
            f"(min_support_per_class={min_support_per_class}). "
            f"Task global_idx={global_idx}"
        )

    class_prior_value = _first_present(d, "class_prior")
    class_prior = (
        torch.tensor(np.asarray(class_prior_value), dtype=torch.float32)
        if class_prior_value is not None else None
    )
    return {
        "task_objective": CLASSIFICATION_OBJECTIVE,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "num_classes": num_classes,
        "W_true": W_true,
        "b_true": b_true,
        "class_prior": class_prior,
        "unseen_query_classes": unseen_query_classes,
        "support_class_counts": support_class_counts,
        "missing_support_classes": missing_support,
    }


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

class ParquetMetaDataset(Dataset):
    """Each item is one meta-dataset (X_train, y_train, X_test, betaX_test)."""

    def __init__(self, files, task_objective="inductive_regression"):
        self.files = files
        self.task_objective = task_objective

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.task_objective == CLASSIFICATION_OBJECTIVE:
            return load_classification_parquet(self.files[idx])
        return load_parquet(self.files[idx])


def identity_collate(batch):
    """batch_size=1; return the item directly without default list-wrapping."""
    return batch[0]


# ---------------------------------------------------------------------------
# Mixed-categorical Parquet loaders
# ---------------------------------------------------------------------------

def _first_col(d, key, default=None):
    """Read the first row of a parquet dict column, or return default."""
    if key in d and d[key] is not None and len(d[key]) > 0:
        return d[key][0]
    return default


def load_mixed_regression_parquet(path):
    """Load mixed-categorical regression training parquet.

    Returns dict with all numeric and categorical arrays as tensors.
    """
    table = pq.read_table(path)
    d = table.to_pydict()
    X_num_train = torch.tensor(np.array(d["X_num_train"][0]), dtype=torch.float32)
    X_num_test  = torch.tensor(np.array(d["X_num_test"][0]),  dtype=torch.float32)
    y_train     = torch.tensor(np.array(d["y_train"][0]),     dtype=torch.float32)
    y_test      = torch.tensor(np.array(d["y_test"][0]),      dtype=torch.float32)
    X_cat_train = torch.tensor(np.array(d["X_cat_train"][0]), dtype=torch.long)
    X_cat_test  = torch.tensor(np.array(d["X_cat_test"][0]),  dtype=torch.long)
    categorical_cardinalities = torch.tensor(
        np.array(d["categorical_cardinalities"][0]), dtype=torch.long
    )
    cat_missing_mask_train = torch.tensor(
        np.array(d["cat_missing_mask_train"][0]), dtype=torch.bool
    )
    cat_missing_mask_test = torch.tensor(
        np.array(d["cat_missing_mask_test"][0]), dtype=torch.bool
    )
    cat_unknown_mask_test = torch.tensor(
        np.array(d["cat_unknown_mask_test"][0]), dtype=torch.bool
    )
    beta_num = None
    if "beta_num" in d and d["beta_num"][0] is not None:
        beta_num = torch.tensor(np.array(d["beta_num"][0]), dtype=torch.float32)
    cat_effects = None
    if "cat_effects" in d and d["cat_effects"][0] is not None:
        cat_effects = [
            torch.tensor(np.array(eff), dtype=torch.float32)
            for eff in d["cat_effects"][0]
        ]
    return {
        "task_objective": "inductive_regression",
        "X_train": X_num_train,
        "y_train": y_train,
        "X_test": X_num_test,
        "y_test": y_test,
        "X_cat_train": X_cat_train,
        "X_cat_test": X_cat_test,
        "categorical_cardinalities": categorical_cardinalities,
        "cat_missing_mask_train": cat_missing_mask_train,
        "cat_missing_mask_test": cat_missing_mask_test,
        "cat_unknown_mask_test": cat_unknown_mask_test,
        "beta_num": beta_num,
        "cat_effects": cat_effects,
    }


def load_mixed_classification_parquet(path):
    """Load mixed-categorical classification training parquet."""
    table = pq.read_table(path)
    d = table.to_pydict()
    X_num_train = torch.tensor(np.array(d["X_num_train"][0]), dtype=torch.float32)
    X_num_test  = torch.tensor(np.array(d["X_num_test"][0]),  dtype=torch.float32)
    y_train     = torch.tensor(np.array(d["y_train"][0]),     dtype=torch.long)
    y_test      = torch.tensor(np.array(d["y_test"][0]),      dtype=torch.long)
    X_cat_train = torch.tensor(np.array(d["X_cat_train"][0]), dtype=torch.long)
    X_cat_test  = torch.tensor(np.array(d["X_cat_test"][0]),  dtype=torch.long)
    categorical_cardinalities = torch.tensor(
        np.array(d["categorical_cardinalities"][0]), dtype=torch.long
    )
    cat_missing_mask_train = torch.tensor(
        np.array(d["cat_missing_mask_train"][0]), dtype=torch.bool
    )
    cat_missing_mask_test = torch.tensor(
        np.array(d["cat_missing_mask_test"][0]), dtype=torch.bool
    )
    cat_unknown_mask_test = torch.tensor(
        np.array(d["cat_unknown_mask_test"][0]), dtype=torch.bool
    )
    num_classes = int(d["num_classes"][0])
    W_num = None
    if "W_num" in d and d["W_num"][0] is not None:
        W_num = torch.tensor(np.array(d["W_num"][0]), dtype=torch.float32)
    b = None
    if "b" in d and d["b"][0] is not None:
        b = torch.tensor(np.array(d["b"][0]), dtype=torch.float32)
    cat_class_effects = None
    if "cat_class_effects" in d and d["cat_class_effects"][0] is not None:
        cat_class_effects = [
            torch.tensor(np.array(eff), dtype=torch.float32)
            for eff in d["cat_class_effects"][0]
        ]
    return {
        "task_objective": CLASSIFICATION_OBJECTIVE,
        "X_train": X_num_train,
        "y_train": y_train,
        "X_test": X_num_test,
        "y_test": y_test,
        "X_cat_train": X_cat_train,
        "X_cat_test": X_cat_test,
        "categorical_cardinalities": categorical_cardinalities,
        "cat_missing_mask_train": cat_missing_mask_train,
        "cat_missing_mask_test": cat_missing_mask_test,
        "cat_unknown_mask_test": cat_unknown_mask_test,
        "num_classes": num_classes,
        "W_num": W_num,
        "b": b,
        "cat_class_effects": cat_class_effects,
    }


class MixedCategoricalMetaDataset(Dataset):
    """Dataset over mixed-categorical meta-tasks.

    Each item = one task dict (variable p_num, p_cat, n).
    """
    def __init__(self, files, task_objective="inductive_regression"):
        self.files = files
        self.task_objective = task_objective

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.task_objective == CLASSIFICATION_OBJECTIVE:
            return load_mixed_classification_parquet(self.files[idx])
        return load_mixed_regression_parquet(self.files[idx])


def make_loader(files, shuffle, task_objective="inductive_regression"):
    return DataLoader(
        ParquetMetaDataset(files, task_objective=task_objective),
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
              loss_fn=None, l1_lambda=0.0, return_sum_count=False, cfg=None,
              epoch_idx: int = 0, rank: int = 0):
    """
    Iterate over all meta-datasets in `loader`.  If training=True, backprop per dataset.
    Returns mean loss across all test-row predictions in the epoch.

    Args:
        loss_fn:   callable(y_hat, y) → scalar loss, or None for MSE.
        l1_lambda: L1 penalty coefficient on model parameters (training only).
        cfg:       ModelConfig instance; when provided and use_coefficient_head=True,
                   enables debug forward pass and MODEL4 three-tier auxiliary losses.
        epoch_idx: Current epoch index (for deterministic augmentation seeds).
        rank:      Distributed rank (for deterministic augmentation seeds).
    """
    model.train(training)
    total_loss  = 0.0
    total_count = 0
    _support_aug = (
        training
        and cfg is not None
        and getattr(cfg, "support_permutation_augmentation", False)
    )
    _support_aug_seed = (
        int(getattr(cfg, "support_permutation_base_seed", 0)) if cfg else 0
    )

    for _batch_idx, batch_item in enumerate(loader):
        # Mixed-categorical batches are dicts; pure numeric batches are tuples
        cat_kwargs = {}
        if isinstance(batch_item, dict):
            X_train = batch_item["X_train"].to(device, non_blocking=True)
            y_train = batch_item["y_train"].to(device, non_blocking=True)
            X_test  = batch_item["X_test"].to(device, non_blocking=True)
            betaX_test = batch_item.get("y_test")
            if betaX_test is not None:
                betaX_test = betaX_test.to(device, non_blocking=True)
            else:
                betaX_test = y_train.new_zeros(X_test.shape[0])
            beta = batch_item.get("beta_num")
            if beta is not None:
                beta = beta.to(device, non_blocking=True)
            # Categorical inputs
            for cat_key in ("X_cat_train", "X_cat_test", "categorical_cardinalities",
                            "cat_missing_mask_train", "cat_unknown_mask_test"):
                v = batch_item.get(cat_key)
                if v is not None:
                    cat_kwargs[cat_key] = v.to(device, non_blocking=True)
        else:
            # Handle both 4-tuple (old parquet) and 5-tuple (new parquet with beta)
            if len(batch_item) == 5:
                X_train, y_train, X_test, betaX_test, beta = batch_item
            else:
                X_train, y_train, X_test, betaX_test = batch_item
                beta = None

            X_train    = X_train.to(device, non_blocking=True)
            y_train    = y_train.to(device, non_blocking=True)
            X_test     = X_test.to(device, non_blocking=True)
            betaX_test = betaX_test.to(device, non_blocking=True)
            if beta is not None:
                beta = beta.to(device, non_blocking=True)

        # Support-row permutation augmentation (training only)
        if _support_aug:
            X_train, y_train, _ = permute_support_rows(
                X_train, y_train,
                base_seed=_support_aug_seed, epoch=epoch_idx,
                rank=rank, batch_idx=_batch_idx,
            )

        if training:
            optimizer.zero_grad()

        use_debug = cfg is not None and (
            getattr(cfg, "use_coefficient_head", False)
            or getattr(cfg, "use_lambda_head", False)
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            if use_debug:
                y_hat, debug = model(X_train, y_train, X_test,
                                     return_debug=True, beta=beta,
                                     **cat_kwargs)
            else:
                y_hat = model(X_train, y_train, X_test, **cat_kwargs)
                debug = {}

            loss = F.mse_loss(y_hat, betaX_test) if loss_fn is None else loss_fn(y_hat, betaX_test)

            # -----------------------------------------------------------------------
            # METHOD 2: Three-tier auxiliary losses (soft penalization toward OLS)
            # Teacher: OLS (unbiased; training data guarantees n >= 5*p)
            # beta_norm takes priority when available (new parquet with beta)
            # -----------------------------------------------------------------------
            if use_debug and cfg is not None:
                beta_hat    = debug.get("beta_hat_norm")       # (p,) or None
                y_coeff     = debug.get("y_coeff_norm")        # (m,) or None
                beta_ols    = debug.get("beta_ols_teacher")    # (p,) or None, detached
                y_ols       = debug.get("y_ols_norm_teacher")  # (m,) or None, detached
                beta_norm   = debug.get("beta_norm")           # (p,) or None, detached

                # Preferred teacher: ground-truth beta_norm if available, else OLS
                beta_teacher = beta_norm if beta_norm is not None else beta_ols

                # Tier 1 — Scale-invariant coefficient MSE
                if (getattr(cfg, "beta_aux_loss_weight", 0.0) > 0
                        and beta_hat is not None and beta_teacher is not None):
                    diff_sq = (beta_hat - beta_teacher.detach()).pow(2).mean()
                    scale   = beta_teacher.detach().pow(2).mean() + 1e-6
                    L_beta  = diff_sq / scale
                    loss    = loss + cfg.beta_aux_loss_weight * L_beta

                # Tier 2 — Normalized prediction MSE
                if (getattr(cfg, "pred_aux_loss_weight", 0.0) > 0
                        and y_coeff is not None and y_ols is not None):
                    y_t   = y_ols.detach()
                    var_t = y_t.var() + 1e-6
                    L_pred = (y_coeff - y_t).pow(2).mean() / var_t
                    loss   = loss + cfg.pred_aux_loss_weight * L_pred

                # Tier 3 — Cosine direction alignment
                if (getattr(cfg, "cos_aux_loss_weight", 0.0) > 0
                        and beta_hat is not None and beta_teacher is not None):
                    bt       = beta_teacher.detach()
                    dot      = (beta_hat * bt).sum()
                    norm_prod = beta_hat.norm() * bt.norm() + 1e-8
                    L_cos   = 1.0 - dot / norm_prod
                    loss    = loss + cfg.cos_aux_loss_weight * L_cos

                # Lambda soft prior (heuristic: lambda ~ p/n)
                lam_hat = debug.get("lambda_hat")
                if (getattr(cfg, "lambda_aux_loss_weight", 0.0) > 0
                        and lam_hat is not None):
                    p_over_n  = X_train.shape[1] / X_train.shape[0]
                    lam_target = torch.tensor(p_over_n, device=device, dtype=lam_hat.dtype)
                    L_lambda  = F.mse_loss(
                        lam_hat.log().clamp(min=-10.0),
                        lam_target.log().clamp(min=-10.0),
                    )
                    loss = loss + cfg.lambda_aux_loss_weight * L_lambda

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


def _classification_teacher_for_batch(
    batch: dict,
    output: dict,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    cfg: ModelConfig,
):
    W_true = batch.get("W_true")
    b_true = batch.get("b_true")
    num_classes = int(batch["num_classes"])
    if W_true is not None:
        W_raw, b_raw = canonicalize_class_coefficients(
            W_true.to(X_train.device),
            b_true.to(X_train.device) if b_true is not None else None,
            num_classes,
        )
        if cfg.norm_feat:
            col_mean = X_train.mean(dim=0)
            col_std = X_train.std(dim=0, unbiased=False).clamp(min=1e-8)
            W_teacher = col_std.unsqueeze(1) * W_raw
            b_teacher = b_raw + col_mean @ W_raw
            X_query_norm = (X_test - col_mean) / col_std
        else:
            W_teacher = W_raw
            b_teacher = b_raw
            X_query_norm = X_test
        teacher_logits = X_query_norm @ W_teacher + b_teacher
        return {
            "W_teacher_norm": W_teacher.detach(),
            "b_teacher": b_teacher.detach(),
            "teacher_logits": teacher_logits.detach(),
            "teacher_probs": torch.softmax(teacher_logits.detach(), dim=-1),
        }

    teacher = LogisticTeacher(
        C=cfg.teacher_logreg_C,
        max_iter=cfg.teacher_logreg_max_iter,
        solver=cfg.teacher_logreg_solver,
    )
    if cfg.norm_feat:
        col_mean = X_train.mean(dim=0)
        col_std = X_train.std(dim=0, unbiased=False).clamp(min=1e-8)
        X_teacher = (X_train - col_mean) / col_std
        X_query_teacher = (X_test - col_mean) / col_std
    else:
        X_teacher = X_train
        X_query_teacher = X_test
    return teacher.fit_predict(
        X_teacher,
        y_train,
        X_query_teacher,
        num_classes,
    )


def run_classification_epoch(
    model,
    loader,
    optimizer,
    scaler,
    training: bool,
    device,
    use_amp,
    *,
    l1_lambda=0.0,
    return_sum_count=False,
    cfg: ModelConfig,
    epoch_idx: int = 0,
    rank: int = 0,
):
    """Run one classification epoch and return mean CE per query row.

    When return_sum_count=True returns (total_ce, total_loss, total_count)
    so callers can track true CE separately from the full optimisation objective.
    """
    model.train(training)
    total_ce = 0.0    # pure query cross-entropy only
    total_loss = 0.0  # full optimisation objective (ce + aux terms)
    total_count = 0
    _perm_aug = getattr(cfg, "class_permutation_augmentation", False)
    _perm_base_seed = getattr(cfg, "class_permutation_base_seed", 0)
    for _batch_idx, batch in enumerate(loader):
        X_train = batch["X_train"].to(device, non_blocking=True)
        y_train = batch["y_train"].to(device, non_blocking=True).long()
        X_test = batch["X_test"].to(device, non_blocking=True)
        y_test = batch["y_test"].to(device, non_blocking=True).long()
        num_classes = int(batch["num_classes"])
        W_true = batch.get("W_true")
        b_true = batch.get("b_true")
        if W_true is not None:
            batch["W_true"] = W_true.to(device, non_blocking=True)
        if b_true is not None:
            batch["b_true"] = b_true.to(device, non_blocking=True)

        # Support-row permutation augmentation (training only)
        _support_aug = (
            training
            and getattr(cfg, "support_permutation_augmentation", False)
        )
        if _support_aug:
            _support_seed = int(getattr(cfg, "support_permutation_base_seed", 0))
            X_train, y_train, _ = permute_support_rows(
                X_train, y_train,
                base_seed=_support_seed, epoch=epoch_idx,
                rank=rank, batch_idx=_batch_idx,
            )

        if training:
            optimizer.zero_grad()

        # F2: episodic class-label permutation augmentation (opt-in).
        if _perm_aug and training:
            import numpy as _np_perm
            _perm_rng = _np_perm.random.default_rng(int(_perm_base_seed) ^ _batch_idx)
            _teacher_in = batch if any(
                k in batch for k in ("teacher_probs", "teacher_logits", "W_teacher_norm")
            ) else None
            y_train, y_test, _teacher_perm = permute_class_labels(
                y_train, y_test, _teacher_in, _perm_rng, num_classes
            )
            if _teacher_perm is not None:
                batch.update(_teacher_perm)

        class_weight = None
        if cfg.class_imbalance_reweighting:
            class_weight = inverse_frequency_class_weight(y_train, num_classes)

        # Extract categorical kwargs when present in batch
        cat_kwargs = {}
        if batch.get("X_cat_train") is not None:
            cat_kwargs["X_cat_train"] = batch["X_cat_train"].to(device, non_blocking=True)
            cat_kwargs["X_cat_test"] = batch["X_cat_test"].to(device, non_blocking=True)
            cat_kwargs["categorical_cardinalities"] = batch["categorical_cardinalities"].to(
                device, non_blocking=True
            )

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
        ):
            output = model(
                X_train,
                y_train,
                X_test,
                task_objective=CLASSIFICATION_OBJECTIVE,
                num_classes=num_classes,
                return_aux=True,
                **cat_kwargs,
            )
            teacher = None
            teacher_active = (
                cfg.class_logit_kl_loss_weight > 0
                or cfg.class_coef_aux_loss_weight > 0
            )
            if teacher_active:
                teacher = _classification_teacher_for_batch(
                    batch, output, X_train, y_train, X_test, cfg
                )
            losses = compute_classification_losses(
                output,
                y_test,
                y_train=y_train,
                teacher=teacher,
                cfg=cfg,
                class_weight=class_weight,
            )
            loss = losses["total_loss"]

        if l1_lambda > 0.0 and training:
            loss = loss + l1_lambda * sum(
                parameter.abs().sum() for parameter in model.parameters()
            )
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        n_query = X_test.shape[0]
        # F6: track pure CE separately from the full optimisation objective.
        total_ce   += losses.get("ce", losses["total_loss"]).item() * n_query
        total_loss += loss.item() * n_query
        total_count += n_query

    if return_sum_count:
        return total_ce, total_loss, total_count
    return total_ce / max(total_count, 1)


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
        "task_objective",
        "use_classification_path",
        "max_num_classes",
        "class_embedding_dim",
        "class_stat_dim",
        "class_stat_hidden_dim",
        "class_head_hidden_dim",
        "class_fusion_gate_hidden_dim",
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
            - "load_compatible_backbone": load only tensors whose names and shapes match.

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
            elif pretrain_load_policy == "load_compatible_backbone":
                current_state = model.state_dict()
                compatible_state = {
                    key: value
                    for key, value in ckpt["state_dict"].items()
                    if key in current_state
                    and current_state[key].shape == value.shape
                }
                model.load_state_dict(compatible_state, strict=False)
                print(
                    f"[PRETRAIN] rank {rank}: loaded "
                    f"{len(compatible_state)}/{len(current_state)} compatible tensors "
                    f"from {stage_path}; mismatches={arch_mismatches}",
                    flush=True,
                )
                return True, repr(arch_mismatches)
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
    checkpoint_output_name   = os.environ.get("CHECKPOINT_OUTPUT_NAME", "best_regression.pt")

    # Pretrain load policy — controls behaviour on architecture mismatch
    pretrain_load_policy = os.environ.get("PRETRAIN_LOAD_POLICY", "require_match").strip().lower()
    _VALID_PRETRAIN_POLICIES = {
        "require_match",
        "allow_cold_start_on_arch_mismatch",
        "load_compatible_backbone",
    }
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
    use_ridge_expert = bool(hyper_params.get("use_ridge_expert", False))   # MODEL4 default: False
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

    # MODEL4 hyperparams
    use_linear_stats          = bool(hyper_params.get("use_linear_stats",          True))
    use_coefficient_head      = bool(hyper_params.get("use_coefficient_head",      True))
    use_lambda_head           = bool(hyper_params.get("use_lambda_head",           True))
    use_residual_head         = bool(hyper_params.get("use_residual_head",         False))
    linear_stat_dim           = int(hyper_params.get("linear_stat_dim",           128))
    coeff_head_hidden_dim     = int(hyper_params.get("coeff_head_hidden_dim",      64))
    lambda_head_hidden_dim    = int(hyper_params.get("lambda_head_hidden_dim",     32))
    beta_aux_loss_weight      = float(hyper_params.get("beta_aux_loss_weight",     0.10))
    pred_aux_loss_weight      = float(hyper_params.get("pred_aux_loss_weight",     0.05))
    cos_aux_loss_weight       = float(hyper_params.get("cos_aux_loss_weight",      0.02))
    lambda_aux_loss_weight    = float(hyper_params.get("lambda_aux_loss_weight",   0.01))
    teacher_ols_threshold     = float(hyper_params.get("teacher_ols_threshold",    5.0))
    max_moment_p              = int(hyper_params.get("max_moment_p",               128))
    linear_encoder_hidden_dim = int(hyper_params.get("linear_encoder_hidden_dim",  256))
    fusion_gate_hidden_dim    = int(hyper_params.get("fusion_gate_hidden_dim",     64))

    is_classification = TASK_OBJECTIVE == CLASSIFICATION_OBJECTIVE
    use_classification_path = bool(
        hyper_params.get("use_classification_path", is_classification)
    )
    use_class_label_embeddings = bool(
        hyper_params.get("use_class_label_embeddings", True)
    )
    use_classification_stats = bool(
        hyper_params.get("use_classification_stats", True)
    )
    use_class_stat_fusion = bool(
        hyper_params.get("use_class_stat_fusion", True)
    )
    use_class_coefficient_head = bool(
        hyper_params.get("use_class_coefficient_head", True)
    )
    use_class_bias_head = bool(hyper_params.get("use_class_bias_head", True))
    use_class_residual_head = bool(
        hyper_params.get("use_class_residual_head", False)
    )
    max_num_classes = int(hyper_params.get("max_num_classes", 10))
    class_embedding_dim = int(hyper_params.get("class_embedding_dim", 64))
    class_stat_dim = int(hyper_params.get("class_stat_dim", 128))
    class_stat_hidden_dim = int(hyper_params.get("class_stat_hidden_dim", 256))
    class_head_hidden_dim = int(hyper_params.get("class_head_hidden_dim", 64))
    class_fusion_gate_hidden_dim = int(
        hyper_params.get("class_fusion_gate_hidden_dim", 64)
    )
    classification_teacher_type = hyper_params.get(
        "classification_teacher_type", "logistic_regression"
    )
    teacher_logreg_C = float(hyper_params.get("teacher_logreg_C", 1.0))
    teacher_logreg_max_iter = int(
        hyper_params.get("teacher_logreg_max_iter", 200)
    )
    teacher_logreg_solver = hyper_params.get("teacher_logreg_solver", "lbfgs")
    class_ce_loss_weight = float(
        hyper_params.get("class_ce_loss_weight", 1.0)
    )
    class_logit_kl_loss_weight = float(
        hyper_params.get("class_logit_kl_loss_weight", 0.05)
    )
    class_coef_aux_loss_weight = float(
        hyper_params.get("class_coef_aux_loss_weight", 0.05)
    )
    class_margin_aux_loss_weight = float(
        hyper_params.get("class_margin_aux_loss_weight", 0.01)
    )
    class_prior_aux_loss_weight = float(
        hyper_params.get("class_prior_aux_loss_weight", 0.01)
    )
    class_calibration_aux_loss_weight = float(
        hyper_params.get("class_calibration_aux_loss_weight", 0.0)
    )
    class_imbalance_reweighting = bool(
        hyper_params.get("class_imbalance_reweighting", True)
    )
    class_label_smoothing = float(
        hyper_params.get("class_label_smoothing", 0.0)
    )

    # Mixed-categorical hyperparams
    is_mixed_categorical = TRAINING_DATA_FAMILY in (
        MIXED_CAT_REGRESSION_TRAINING_FAMILY,
        MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY,
    )
    use_categorical_features = bool(
        hyper_params.get("use_categorical_features", is_mixed_categorical)
    )
    cat_embed_dim = int(hyper_params.get("cat_embed_dim", 32))
    cat_feat_id_embed_dim = int(hyper_params.get("cat_feat_id_embed_dim", 16))
    cat_cardinality_embed_dim = int(hyper_params.get("cat_cardinality_embed_dim", 8))
    cat_stat_dim = int(hyper_params.get("cat_stat_dim", 64))
    cat_stat_hidden_dim = int(hyper_params.get("cat_stat_hidden_dim", 128))
    cat_head_hidden_dim = int(hyper_params.get("cat_head_hidden_dim", 64))
    cat_effect_aux_loss_weight = float(hyper_params.get("cat_effect_aux_loss_weight", 0.05))
    cat_class_effect_aux_loss_weight = float(
        hyper_params.get("cat_class_effect_aux_loss_weight", 0.05)
    )

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

    _DatasetCls = MixedCategoricalMetaDataset if is_mixed_categorical else ParquetMetaDataset
    train_loader = DataLoader(
        _DatasetCls(train_files, task_objective=TASK_OBJECTIVE),
        batch_size=1, shuffle=True,
        num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=identity_collate,
    )
    val_loader = DataLoader(
        _DatasetCls(val_files, task_objective=TASK_OBJECTIVE),
        batch_size=1, shuffle=False,
        num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=identity_collate,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # --- Model ---
    model_family = hyper_params.get("model_family", MODEL_FAMILY)
    _design_pattern = hyper_params.get("model_design_pattern", MODEL_DESIGN_PATTERN)
    cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool,
                        n_heads=N_HEADS, n_sab_feat=n_sab_feat,
                        norm_feat=NORM_FEAT,
                        norm_target=(NORM_TARGET if not is_classification else False),
                        dropout=dropout,
                        model_family=model_family,
                        model_arch_version=MODEL_ARCH_VERSION,
                        model_design_pattern=_design_pattern,
                        task_objective=TASK_OBJECTIVE,
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
                        use_lambda_head=use_lambda_head,
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
                        classification_teacher_type=classification_teacher_type,
                        teacher_logreg_C=teacher_logreg_C,
                        teacher_logreg_max_iter=teacher_logreg_max_iter,
                        teacher_logreg_solver=teacher_logreg_solver,
                        class_ce_loss_weight=class_ce_loss_weight,
                        class_logit_kl_loss_weight=class_logit_kl_loss_weight,
                        class_coef_aux_loss_weight=class_coef_aux_loss_weight,
                        class_margin_aux_loss_weight=class_margin_aux_loss_weight,
                        class_prior_aux_loss_weight=class_prior_aux_loss_weight,
                        class_calibration_aux_loss_weight=class_calibration_aux_loss_weight,
                        class_imbalance_reweighting=class_imbalance_reweighting,
                        class_label_smoothing=class_label_smoothing,
                        # Mixed-categorical
                        use_categorical_features=use_categorical_features,
                        cat_embed_dim=cat_embed_dim,
                        cat_feat_id_embed_dim=cat_feat_id_embed_dim,
                        cat_cardinality_embed_dim=cat_cardinality_embed_dim,
                        cat_stat_dim=cat_stat_dim,
                        cat_stat_hidden_dim=cat_stat_hidden_dim,
                        cat_head_hidden_dim=cat_head_hidden_dim,
                        cat_effect_aux_loss_weight=cat_effect_aux_loss_weight,
                        cat_class_effect_aux_loss_weight=cat_class_effect_aux_loss_weight)
    print(
        f"[train_fn] model_family={cfg.model_family} "
        f"training_data_family={TRAINING_DATA_FAMILY} "
        f"task_type={'classification' if is_classification else 'regression'}",
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

    best_val_metric   = float("inf")
    patience_count    = 0
    best_epoch        = 0
    train_metric_at_best = float("inf")

    for epoch in range(1, max_epochs + 1):
        epoch_fn = run_classification_epoch if is_classification else run_epoch
        if is_classification:
            train_metric = epoch_fn(
                model, train_loader, optimizer, scaler, True, device, use_amp,
                l1_lambda=lambda_l1, cfg=cfg,
            )
        else:
            train_metric = epoch_fn(
                model, train_loader, optimizer, scaler, True, device, use_amp,
                loss_fn=loss_fn, l1_lambda=lambda_l1, cfg=cfg,
            )
        with torch.no_grad():
            if is_classification:
                # F6: three-tuple (ce_sum, total_loss_sum, count)
                val_ce_sum, val_total_loss_sum, val_count = epoch_fn(
                    model, val_loader, None, scaler, False, device, use_amp,
                    l1_lambda=0.0, return_sum_count=True, cfg=cfg,
                )
            else:
                val_loss_sum, val_count = epoch_fn(
                    model, val_loader, None, scaler, False, device, use_amp,
                    loss_fn=loss_fn, l1_lambda=0.0,
                    return_sum_count=True, cfg=cfg,
                )

        if is_classification:
            val_metric = reduce_loss_sum_count(val_ce_sum, val_count, device, dist)
            val_total_metric = reduce_loss_sum_count(val_total_loss_sum, val_count, device, dist)
        else:
            val_metric = reduce_loss_sum_count(val_loss_sum, val_count, device, dist)
        metric_name = "val_cross_entropy" if is_classification else "val_mse"

        if is_main:
            print(f"Epoch {epoch:3d}  {metric_name}={val_metric:.4f}")
            if val_metric < best_val_metric:
                best_val_metric   = val_metric
                best_epoch        = epoch
                train_metric_at_best = train_metric
                patience_count    = 0
                ckpt = model.module if isinstance(model, DistributedDataParallel) else model
                ckpt = ckpt._orig_mod if hasattr(ckpt, "_orig_mod") else ckpt
                import dataclasses as _dc
                if TRAINING_DATA_FAMILY == MIXED_CAT_REGRESSION_TRAINING_FAMILY:
                    format_version = CHECKPOINT_VERSION_MIXED_REGRESSION
                elif TRAINING_DATA_FAMILY == MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY:
                    format_version = CHECKPOINT_VERSION_MIXED_CLASSIFICATION
                else:
                    format_version = 5 if is_classification else 4
                _metadata = {
                    "source": "train.py",
                    "checkpoint_name": os.path.basename(checkpoint_output_name),
                    "pytorch_version": torch.__version__,
                    "model_family": cfg.model_family,
                    "task_type": "classification" if is_classification else "regression",
                    "training_entrypoint": "train.py",
                    "training_data_family": TRAINING_DATA_FAMILY,
                    metric_name: float(best_val_metric),
                    (
                        "train_cross_entropy_at_best"
                        if is_classification else "train_mse_at_best"
                    ): float(train_metric_at_best),
                    "best_epoch": int(best_epoch),
                }
                # F6: record full objective alongside true CE.
                if is_classification:
                    _metadata["val_total_loss"] = float(val_total_metric)
                # F10: evidence chain fields.
                if is_classification:
                    _metadata["checkpoint_task_objective"]  = cfg.task_objective
                    _metadata["checkpoint_max_k"]           = int(getattr(cfg, "max_k", 0))
                    _metadata["checkpoint_best_epoch"]      = int(best_epoch)
                    _metadata["checkpoint_training_family"] = getattr(cfg, "training_data_family", "")
                    _metadata["checkpoint_val_ce_by_k"]     = {}
                _metadata["model_arch_version"]    = MODEL_ARCH_VERSION
                _metadata["model_design_pattern"] = cfg.model_design_pattern
                _metadata["task_objective"] = cfg.task_objective
                if is_classification:
                    if is_mixed_categorical:
                        _metadata["classification_path_version"] = "class_linear_mixed_categorical_v1"
                    else:
                        _metadata["classification_path_version"] = "class_linear_v1"
                    _metadata["supports_variable_p"] = True
                    _metadata["supports_variable_k"] = True
                if is_mixed_categorical:
                    _metadata["feature_contract_version"] = "mixed_categorical_linear_v1"
                    _metadata["uses_entity_embeddings"] = True
                    _metadata["uses_context_only_categorical_statistics"] = True
                    _metadata["use_categorical_features"] = True
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

    return {
        ("val_cross_entropy" if is_classification else "val_mse"):
            best_val_metric
    }


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
        "checkpoint_output_name":     os.environ.get("CHECKPOINT_OUTPUT_NAME", "best_regression.pt"),
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
