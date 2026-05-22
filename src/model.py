"""
model.py

DeepSetICLModel: a Hartford-style exchangeable neural network for
in-context regression (MODEL3 inductive forecasting), operating over a training
set (X_train, y_train) and one or more test points x_test.
"""

import dataclasses

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------

def build_mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """
    Returns a 2-layer MLP:
        Linear(in_dim, hidden_dim) -> ReLU -> Dropout(dropout) -> Linear(hidden_dim, out_dim)
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VALID_POOLS = {"sum", "mean", "max", "pna", "learned", "attn", "multipool"}

# How many times larger is the output dim vs input dim after pooling?
POOL_SCALE = {
    "sum": 1, "mean": 1, "max": 1, "learned": 1, "attn": 1,
    "pna": 4,        # sum + mean + max + std
    "multipool": 5,  # pna + attn
}


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelConfig:
    d_phi:       int   = 128   # channel dimension d_model for ExchangeableMatrixBlocks
    d_rho:       int   = 256   # kept for checkpoint compatibility
    pool:        str   = "pna"
    n_heads:     int   = 4     # attention heads in SAB / AttentionPool
    n_sab_feat:  int   = 1     # number of ExchangeableMatrixBlocks (>=1)
    norm_feat:   bool  = True  # standardize X_train columns per-context
    norm_target: bool  = True  # standardize y_train per-context; denormalize output
    dropout:     float = 0.1

    # Model family and design pattern
    model_family:          str   = "market_exchangeable_icl"   # "market_exchangeable_icl" | "market_exchangeable_completion"
    use_ridge_expert:      bool  = True       # enable RidgeExpert + gate
    ridge_lambda:          float = 1.0        # ridge regularisation λ (must be > 0)
    gate_hidden_dim:       int   = 64         # gate_head MLP hidden dim

    # MODEL3 selectors
    model_arch_version:    str = "model3"                # always "model3"
    model_design_pattern: str = "inductive_forecasting" # "inductive_forecasting" | "transductive_completion"

    def __post_init__(self):
        if self.pool not in VALID_POOLS:
            raise ValueError(f"Invalid pool '{self.pool}'. Valid: {sorted(VALID_POOLS)}")
        # n_heads divisibility only matters when attention is actually used
        uses_heads = (self.n_sab_feat > 0 or self.pool in ("attn", "multipool"))
        if uses_heads:
            if self.d_phi % self.n_heads != 0:
                raise ValueError(f"d_phi={self.d_phi} must be divisible by n_heads={self.n_heads}")
            if self.d_rho % self.n_heads != 0:
                raise ValueError(f"d_rho={self.d_rho} must be divisible by n_heads={self.n_heads}")
        _VALID_FAMILIES = frozenset({
            "market_exchangeable_icl", "market_exchangeable_completion",
        })
        if self.model_family not in _VALID_FAMILIES:
            raise ValueError(
                f"Invalid model_family: {self.model_family!r}. Valid: {sorted(_VALID_FAMILIES)}"
            )
        if self.model_arch_version != "model3":
            raise ValueError(
                f"Invalid model_arch_version: {self.model_arch_version!r}. "
                "Valid: 'model3'"
            )
        if self.model_design_pattern not in ("inductive_forecasting", "transductive_completion"):
            raise ValueError(
                f"Invalid model_design_pattern: {self.model_design_pattern!r}. "
                "Valid: 'inductive_forecasting', 'transductive_completion'"
            )
        # MODEL3 family/pattern consistency
        if (self.model_design_pattern == "inductive_forecasting"
                and self.model_family != "market_exchangeable_icl"):
            raise ValueError(
                f"model_design_pattern='inductive_forecasting' requires "
                f"model_family='market_exchangeable_icl', got {self.model_family!r}"
            )
        if (self.model_design_pattern == "transductive_completion"
                and self.model_family != "market_exchangeable_completion"):
            raise ValueError(
                f"model_design_pattern='transductive_completion' requires "
                f"model_family='market_exchangeable_completion', got {self.model_family!r}"
            )
        if self.use_ridge_expert and self.ridge_lambda <= 0:
            raise ValueError(
                f"ridge_lambda must be > 0 when use_ridge_expert=True, got {self.ridge_lambda}"
            )
        if self.gate_hidden_dim <= 0:
            raise ValueError(
                f"gate_hidden_dim must be positive, got {self.gate_hidden_dim}"
            )


# ---------------------------------------------------------------------------
# MAB (Multihead Attention Block)
# ---------------------------------------------------------------------------

class MAB(nn.Module):
    """MAB(Q, K) = LayerNorm(H + FFN(H)), H = LayerNorm(Q + Dropout(MHA(Q,K,K)))"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(),
            nn.Dropout(p=dropout), nn.Linear(2 * d_model, d_model),
        )
        self.drop  = nn.Dropout(p=dropout)

    def forward(self, Q, K):  # Q,K: (batch, seq, d)
        h, _ = self.mha(Q, K, K)
        Q     = self.norm1(Q + self.drop(h))
        return self.norm2(Q + self.ffn(Q))


# ---------------------------------------------------------------------------
# SAB (Self-Attention Block)
# ---------------------------------------------------------------------------

class SAB(nn.Module):
    """SAB(X) = MAB(X, X) — permutation equivariant, (batch, set, d) → same shape."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(self, x):
        return self.mab(x, x)


# ---------------------------------------------------------------------------
# AttentionPool (PMA with k=1 seed)
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Single-seed cross-attention pooling: (batch, set, d) → (batch, d)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):            # x: (batch, set, d)
        q      = self.seed.expand(x.size(0), -1, -1)   # (batch, 1, d)
        out, _ = self.mha(q, x, x)                     # (batch, 1, d)
        return self.norm(out).squeeze(1)                # (batch, d)


# ---------------------------------------------------------------------------
# LearnedPool
# ---------------------------------------------------------------------------

class LearnedPool(nn.Module):
    """Softmax-weighted sum with a learned 2-layer score network: (batch, set, d) → (batch, d)."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(),
            nn.Dropout(p=dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):            # x: (batch, set, d)
        w = torch.softmax(self.score(x), dim=1)   # (batch, set, 1)
        return (w * x).sum(dim=1)                 # (batch, d)


# ---------------------------------------------------------------------------
# SetPool (unified interface)
# ---------------------------------------------------------------------------

class SetPool(nn.Module):
    """
    Unified permutation-invariant pooling: (batch, set_size, d_model) → (batch, out_dim)
    where out_dim = POOL_SCALE[mode] * d_model.
    """
    def __init__(self, d_model: int, mode: str, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.learned   = LearnedPool(d_model, dropout)
        elif mode in ("attn", "multipool"):
            self.attn_pool = AttentionPool(d_model, n_heads, dropout)

    @staticmethod
    def _pna(x):                     # x: (batch, set, d)
        s, mu = x.sum(dim=1), x.mean(dim=1)
        mx    = x.max(dim=1).values
        std   = x.std(dim=1, unbiased=False)
        return torch.cat([s, mu, mx, std], dim=-1)    # (batch, 4*d)

    def forward(self, x):            # x: (batch, set, d)
        if self.mode == "sum":       return x.sum(dim=1)
        if self.mode == "mean":      return x.mean(dim=1)
        if self.mode == "max":       return x.max(dim=1).values
        if self.mode == "pna":       return self._pna(x)
        if self.mode == "learned":   return self.learned(x)
        if self.mode == "attn":      return self.attn_pool(x)
        if self.mode == "multipool":
            return torch.cat([self._pna(x), self.attn_pool(x)], dim=-1)  # (batch, 5*d)


# ---------------------------------------------------------------------------
# RidgeExpert (stateless, no nn.Module parameters)
# ---------------------------------------------------------------------------

class RidgeExpert:
    """Closed-form ridge regression. Stateless; no nn.Module parameters.
    Primal form (n >= p): β = (XᵀX + λI)⁻¹ Xᵀy
    Dual form  (n < p):   α = (XXᵀ + λI)⁻¹ y,  β = Xᵀα
    """

    def predict(self, X_norm, y_norm, x_test_norm, lam: float):
        """
        Args:
            X_norm       : (n, p) float tensor — normalized training features
            y_norm       : (n,)   float tensor — normalized training labels
            x_test_norm  : (m, p) float tensor — normalized test features
            lam          : float  — ridge regularisation λ (must be > 0)

        Returns:
            (m,) float tensor — ridge predictions cast back to x_test_norm.dtype
        """
        original_dtype = x_test_norm.dtype
        device_type = X_norm.device.type

        # Disable AMP locally so torch.linalg.solve always receives matching
        # float32 tensors.  Mixed-precision inference (bfloat16) on the
        # DeepSet / encoder / SAB path is unaffected — only this closed-form
        # solve runs in float32.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            X  = X_norm.to(dtype=torch.float32)
            y  = y_norm.to(dtype=torch.float32)
            xq = x_test_norm.to(dtype=torch.float32)

            n, p = X.shape
            kw = dict(device=X.device, dtype=torch.float32)

            if n >= p:
                A    = X.T @ X + lam * torch.eye(p, **kw)
                beta = torch.linalg.solve(A, X.T @ y)   # (p,)
            else:
                K     = X @ X.T + lam * torch.eye(n, **kw)
                alpha = torch.linalg.solve(K, y)          # (n,)
                beta  = X.T @ alpha                        # (p,)

            pred = xq @ beta                               # (m,)

        return pred.to(dtype=original_dtype)


# ---------------------------------------------------------------------------
# MODEL3 shared utilities
# ---------------------------------------------------------------------------

def _masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    keepdim: bool = True,
) -> torch.Tensor:
    """
    Mean of x over `dim`, ignoring positions where mask is False.

    x:    shape with d_model as last axis, e.g. (B, R, C, d)
    mask: same shape as x except no last axis, e.g. (B, R, C)
    Returns tensor with `dim` reduced (kept if keepdim=True).
    """
    mask_f = mask.unsqueeze(-1).float()            # (..., dim_size, ..., 1)
    numerator = (x * mask_f).sum(dim=dim, keepdim=keepdim)
    denominator = mask_f.sum(dim=dim, keepdim=keepdim).clamp(min=1e-6)
    return numerator / denominator


# ---------------------------------------------------------------------------
# MODEL3 embedding primitives
# ---------------------------------------------------------------------------

class ColumnEncoder(nn.Module):
    """
    Encodes per-column statistics (mean and std from X_train) into d_model embeddings.
    Input: col_mean (p,), col_std (p,)  ->  (p, d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, col_mean: torch.Tensor, col_std: torch.Tensor) -> torch.Tensor:
        """col_mean, col_std: (p,) -> (p, d_model)"""
        return self.net(torch.stack([col_mean, col_std], dim=-1))  # (p, 2) -> (p, d_model)


class CellEncoder(nn.Module):
    """
    Encodes per-cell values and their interactions into d_model embeddings.
    d_in=3 for inductive ICL: [x_val_norm, y_val_norm, query_val_norm]
    d_in=2 for completion:    [value * is_observed, is_observed]
    Input: (..., d_in) -> (..., d_model)
    """
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) -> (..., d_model)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# MODEL3 ExchangeableMatrixBlock
# ---------------------------------------------------------------------------

class ExchangeableMatrixBlock(nn.Module):
    """
    Hartford-style exchangeable block operating on H: (batch, rows, cols, d_model).

    Update order:
      1. Row update:  for each row r, aggregate column evidence -> update H[:, r, :, :]
      2. Column update: for each col c, aggregate row evidence  -> update H[:, :, c, :]

    Both updates use identity residual connections and LayerNorm.
    Permutation-equivariant over the rows axis AND the cols axis independently.

    Optional mask: (batch, rows, cols) bool — True = observed/valid.
    When mask is provided, aggregations exclude unmasked positions via _masked_mean.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.row_update = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.col_update = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm_row = nn.LayerNorm(d_model)
        self.norm_col = nn.LayerNorm(d_model)

    def forward(
        self,
        H: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        H:    (batch, rows, cols, d_model)
        mask: (batch, rows, cols) bool — True = valid (optional)
        Returns: (batch, rows, cols, d_model)
        """
        # Step 1: Row update — aggregate over cols, inject into each (batch, row, col, :)
        if mask is not None:
            col_agg = _masked_mean(H, mask, dim=2, keepdim=True)  # (B, R, 1, d)
        else:
            col_agg = H.mean(dim=2, keepdim=True)                 # (B, R, 1, d)
        row_in = torch.cat([H, col_agg.expand_as(H)], dim=-1)    # (B, R, C, 2d)
        H = self.norm_row(H + self.row_update(row_in))

        # Step 2: Column update — aggregate over rows, inject into each (batch, row, col, :)
        if mask is not None:
            row_agg = _masked_mean(H, mask, dim=1, keepdim=True)  # (B, 1, C, d)
        else:
            row_agg = H.mean(dim=1, keepdim=True)                 # (B, 1, C, d)
        col_in = torch.cat([H, row_agg.expand_as(H)], dim=-1)    # (B, R, C, 2d)
        H = self.norm_col(H + self.col_update(col_in))

        return H


# ---------------------------------------------------------------------------
# DeepSetICLModel - MODEL3 inductive forecasting
# ---------------------------------------------------------------------------

class DeepSetICLModel(nn.Module):
    """
    MODEL3 inductive in-context learning model using Hartford-style exchangeable blocks.

    Architecture:
      1. ColumnEncoder:           per-feature stats -> (p, d_model) embeddings
      2. CellEncoder:             [x_norm, y_norm, q_norm] -> H: (m, n, p, d_model)
      3. ExchangeableMatrixBlocks over (n=context samples, p=features)
      4. Pool over n (mean), optional SAB over p
      5. Pool over p (mean) -> prediction head -> (m,)
      6. Optional gated ridge expert residual

    Required config:
      cfg.model_family             == "market_exchangeable_icl"
      cfg.model_arch_version       == "model3"
      cfg.model_design_pattern    == "inductive_forecasting"

    Shared config fields used:
      cfg.d_phi             — channel dimension d_model for blocks
      cfg.n_sab_feat        — number of ExchangeableMatrixBlocks (>=1)
      cfg.n_heads           — unused in this architecture (no SAB attention heads)
      cfg.dropout           — dropout rate
      cfg.norm_feat         — normalise X_train columns per context
      cfg.norm_target       — normalise y_train per context
      cfg.use_ridge_expert  — enable RidgeExpert gated residual path
      cfg.ridge_lambda      — ridge λ (when use_ridge_expert=True)
      cfg.gate_hidden_dim   — hidden dim for gate MLP (when use_ridge_expert=True)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.model_family != "market_exchangeable_icl":
            raise ValueError(
                f"DeepSetICLModel requires model_family='market_exchangeable_icl', "
                f"got {cfg.model_family!r}"
            )
        self.cfg = cfg
        d_model = cfg.d_phi
        n_blocks = max(1, cfg.n_sab_feat)

        self.col_encoder  = ColumnEncoder(d_model)
        self.cell_encoder = CellEncoder(d_in=3, d_model=d_model)
        self.blocks = nn.ModuleList([
            ExchangeableMatrixBlock(d_model, cfg.dropout)
            for _ in range(n_blocks)
        ])
        self.pred_head = nn.Linear(d_model, 1)

        if cfg.use_ridge_expert:
            self._ridge    = RidgeExpert()
            self.gate_head = build_mlp(d_model, 1, cfg.gate_hidden_dim, cfg.dropout)

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test:  torch.Tensor,
    ) -> torch.Tensor:
        """
        X_train: (n, p)       — training features
        y_train: (n,)         — training labels
        x_test:  (p,) or (m, p) — query features
        Returns: scalar if x_test is (p,); (m,) if x_test is (m, p)
        """
        n, p   = X_train.shape
        single = x_test.ndim == 1
        if single:
            x_test = x_test.unsqueeze(0)   # (1, p)
        m  = x_test.shape[0]
        EPS = 1e-8

        # --- Column statistics (always computed; used for ColumnEncoder + optional norm) ---
        col_mean = X_train.mean(dim=0)                              # (p,)
        col_std  = X_train.std(dim=0, unbiased=False).clamp(min=EPS)  # (p,)
        col_emb  = self.col_encoder(col_mean, col_std)             # (p, d_model)

        # --- Feature normalisation ---
        if self.cfg.norm_feat:
            X_norm  = (X_train - col_mean) / col_std               # (n, p)
            xq_norm = (x_test  - col_mean) / col_std               # (m, p)
        else:
            X_norm  = X_train
            xq_norm = x_test

        # --- Target normalisation ---
        y_mean = y_std = None
        if self.cfg.norm_target:
            y_mean  = y_train.mean()
            y_std   = y_train.std(unbiased=False).clamp(min=EPS)
            y_norm  = (y_train - y_mean) / y_std                   # (n,)
        else:
            y_norm = y_train

        # --- Build cell inputs (m, n, p, 3): [x_norm, y_norm, query_norm] ---
        X_exp  = X_norm.unsqueeze(0).expand(m, -1, -1)            # (m, n, p)
        y_exp  = y_norm.view(1, n, 1).expand(m, n, p)             # (m, n, p)
        q_exp  = xq_norm.unsqueeze(1).expand(-1, n, -1)           # (m, n, p)
        cell_in = torch.stack([X_exp, y_exp, q_exp], dim=-1)      # (m, n, p, 3)

        # --- Encode cells -> H + broadcast column embeddings ---
        H = self.cell_encoder(cell_in)                             # (m, n, p, d_model)
        H = H + col_emb.unsqueeze(0).unsqueeze(0)                 # broadcast (m, n, p, d_model)

        # --- Hartford-style exchangeable blocks ---
        for block in self.blocks:
            H = block(H)                                           # (m, n, p, d_model)

        # --- Pool: sample axis (n) then feature axis (p) ---
        h_feat = H.mean(dim=1)                                     # (m, p, d_model)
        h_glob = h_feat.mean(dim=1)                                # (m, d_model)

        # --- Prediction ---
        neural = self.pred_head(h_glob).squeeze(-1)                # (m,)

        # --- Optional gated ridge expert ---
        if self.cfg.use_ridge_expert:
            ridge = self._ridge.predict(X_norm, y_norm, xq_norm, self.cfg.ridge_lambda)  # (m,)
            gate  = torch.sigmoid(self.gate_head(h_glob).squeeze(-1))                    # (m,)
            pred_norm = ridge + gate * neural
        else:
            pred_norm = neural

        # --- Denormalise ---
        if self.cfg.norm_target:
            y_hat = pred_norm * y_std + y_mean
        else:
            y_hat = pred_norm

        return y_hat.squeeze(0) if single else y_hat


# ---------------------------------------------------------------------------
# DeepSetCompletionModel - MODEL3 transductive completion
# ---------------------------------------------------------------------------

class DeepSetCompletionModel(nn.Module):
    """
    MODEL3 transductive completion model for sparse market matrix/tensor completion.

    Operates on a partially observed matrix with an explicit missingness mask.
    Architecture:
      1. CellEncoder:             [value * observed, is_observed] -> H: (1, R, C, d_model)
      2. ExchangeableMatrixBlocks with masked aggregations
      3. Decoder head:            (R, C, d_model) -> (R, C) predictions

    Required config:
      cfg.model_family             == "market_exchangeable_completion"
      cfg.model_arch_version       == "model3"
      cfg.model_design_pattern    == "transductive_completion"

    Shared config fields used:
      cfg.d_phi     — channel dimension d_model
      cfg.n_sab_feat — number of ExchangeableMatrixBlocks (>=1)
      cfg.dropout   — dropout rate
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.model_family != "market_exchangeable_completion":
            raise ValueError(
                f"DeepSetCompletionModel requires "
                f"model_family='market_exchangeable_completion', got {cfg.model_family!r}"
            )
        self.cfg = cfg
        d_model  = cfg.d_phi
        n_blocks = max(1, cfg.n_sab_feat)

        # Input: [value * observed_mask, is_observed_flag] -> d_model
        self.cell_encoder = CellEncoder(d_in=2, d_model=d_model)
        self.blocks = nn.ModuleList([
            ExchangeableMatrixBlock(d_model, cfg.dropout)
            for _ in range(n_blocks)
        ])
        self.decoder_head = nn.Linear(d_model, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        X:    (rows, cols) — sparse matrix; unobserved cells may be 0
        mask: (rows, cols) — bool: True = observed, False = missing/target
        Returns: (rows, cols) — predicted values for all cells
        """
        mask_f   = mask.float()                                    # (R, C)
        cell_in  = torch.stack([X * mask_f, mask_f], dim=-1)      # (R, C, 2)
        H        = self.cell_encoder(cell_in)                     # (R, C, d_model)
        H        = H.unsqueeze(0)                                  # (1, R, C, d_model)
        mask_b   = mask.unsqueeze(0)                               # (1, R, C)

        for block in self.blocks:
            H = block(H, mask=mask_b)                              # (1, R, C, d_model)

        H   = H.squeeze(0)                                         # (R, C, d_model)
        out = self.decoder_head(H).squeeze(-1)                     # (R, C)
        return out


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _instantiate_model(cfg: ModelConfig) -> nn.Module:
    """Route to the correct model class based on cfg.model_family."""
    if cfg.model_family == "market_exchangeable_icl":
        return DeepSetICLModel(cfg=cfg)
    if cfg.model_family == "market_exchangeable_completion":
        return DeepSetCompletionModel(cfg=cfg)
    raise ValueError(f"Unknown model_family: {cfg.model_family!r}")
