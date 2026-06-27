"""
model.py

DeepSetICLModel: a Hartford-style exchangeable neural network for
in-context regression (MODEL3 inductive forecasting), operating over a training
set (X_train, y_train) and one or more test points x_test.
"""

import dataclasses
import os
from torch.utils.checkpoint import checkpoint as _checkpoint

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

# MODEL4 linear-statistic constants
N_FEAT_STATS   = 6   # per-feature statistics: sxy, diag_sxx, x_mean, x_std, corr, norm_sxy
N_GLOBAL_STATS = 7   # global statistics: y_mean, y_std, y_var, p/n, n/p, cond_num, eff_rank
N_CLASS_FEAT_STATS = 9
N_CLASS_GLOBAL_STATS = 12

REGRESSION_TASK_OBJECTIVES = frozenset({"inductive_regression", "linear_regression"})
CLASSIFICATION_TASK_OBJECTIVES = frozenset({
    "inductive_classification", "linear_classification",
})


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

    # ICL pooling modes (core architecture, used by all forward paths)
    icl_pool_mode:         str   = "mean"     # sample-axis pool: any VALID_POOLS key
    feature_pool_mode:     str   = "mean"     # feature-axis pool: any VALID_POOLS key

    # Model family and design pattern
    model_family:          str   = "market_exchangeable_icl"   # "market_exchangeable_icl" | "market_exchangeable_completion"
    # Architecture version selectors
    model_arch_version:    str = "model4"                # "model3" | "model4"
    model_design_pattern: str = "inductive_forecasting" # "inductive_forecasting" | "transductive_completion"
    task_objective: str = "inductive_regression"

    # MODEL4: Linear-statistic-aware modules (all True by default)
    use_linear_stats:          bool  = True    # enable LinearStatisticExtractor + encoder + fusion
    use_coefficient_head:      bool  = True    # enable CoefficientHead as primary prediction path
    use_lambda_head:           bool  = True    # enable LambdaHead for dataset-specific λ estimate
    use_residual_head:         bool  = False   # add neural residual on top of coefficient prediction
    linear_stat_dim:           int   = 128     # embedding dim for linear statistics
    max_moment_p:              int   = 128     # reserved for future p-capping in extractor
    use_sxx_triangle:          bool  = False   # reserved for future upper-triangle Sxx feature
    coeff_head_hidden_dim:     int   = 64      # CoefficientHead MLP hidden dim
    lambda_head_hidden_dim:    int   = 32      # LambdaHead MLP hidden dim
    # Auxiliary losses (Method 2 — soft penalization toward OLS structure)
    beta_aux_loss_weight:      float = 0.10   # Tier 1: scale-invariant coefficient MSE
    pred_aux_loss_weight:      float = 0.05   # Tier 2: normalized prediction MSE
    cos_aux_loss_weight:       float = 0.02   # Tier 3: cosine direction alignment
    lambda_aux_loss_weight:    float = 0.01   # Lambda soft prior (p/n heuristic)
    teacher_ols_threshold:     float = 5.0    # use OLS teacher when n/p >= this
    linear_encoder_hidden_dim: int   = 256    # LinearStatisticEncoder MLP hidden dim
    fusion_gate_hidden_dim:    int   = 64     # FusionModule gate MLP hidden dim

    # MODEL4 linear-classification path. Disabled for regression checkpoints.
    use_classification_path:          bool = False
    use_class_label_embeddings:       bool = True
    use_classification_stats:         bool = True
    use_class_stat_fusion:            bool = True
    use_class_coefficient_head:       bool = True
    use_class_bias_head:              bool = True
    use_class_residual_head:          bool = False
    max_num_classes:                  int = 10
    class_embedding_dim:              int = 64
    class_stat_dim:                   int = 128
    class_stat_hidden_dim:            int = 256
    class_head_hidden_dim:            int = 64
    class_fusion_gate_hidden_dim:     int = 64
    classification_teacher_type:      str = "logistic_regression"
    teacher_logreg_C:                 float = 1.0
    teacher_logreg_max_iter:          int = 200
    teacher_logreg_solver:            str = "lbfgs"
    class_ce_loss_weight:             float = 1.0
    class_logit_kl_loss_weight:       float = 0.05
    class_coef_aux_loss_weight:       float = 0.05
    class_margin_aux_loss_weight:     float = 0.01
    class_prior_aux_loss_weight:      float = 0.01
    class_calibration_aux_loss_weight: float = 0.0
    class_imbalance_reweighting:      bool = True
    class_label_smoothing:            float = 0.0

    # Mixed-categorical path (Step 5 additions)
    use_categorical_features:          bool  = False
    cat_max_vocab_size:                int   = 53    # MIXED_CAT_VOCAB_SIZE from constants
    cat_embed_dim:                     int   = 32    # entity embed dim before projection
    cat_feat_id_embed_dim:             int   = 16    # feature identity positional embed dim
    cat_cardinality_embed_dim:         int   = 8     # cardinality bucket embed dim
    cat_stat_dim:                      int   = 64    # categorical statistic embedding dim
    cat_stat_hidden_dim:               int   = 128   # MLP hidden dim for cat stat encoder
    cat_head_hidden_dim:               int   = 64    # cat effect head MLP hidden dim
    cat_effect_aux_loss_weight:        float = 0.05  # cat effect MSE auxiliary loss
    cat_class_effect_aux_loss_weight:  float = 0.05  # cat class effect MSE loss (classification)
    cat_max_p_cat:                     int   = 64    # max number of categorical features

    # MODEL5-LBACNP controls
    use_lbacnp:                        bool  = False
    lbacnp_latent_dim:                 int   = 128
    lbacnp_latent_max:                 int   = 64
    lbacnp_latent_frac:                float = 0.25
    lbacnp_n_blocks:                   int   = 4
    lbacnp_heads:                      int   = 4
    lbacnp_dropout:                    float = 0.1
    lbacnp_output_mode:                str   = "last"  # "last" only for v1

    # Numeric PLE controls
    use_numeric_ple:                   bool  = False
    ple_num_bins:                      int   = 32
    ple_boundary_mode:                 str   = "context_quantile"
    ple_target_aware:                  bool  = False
    ple_projection_dim:                int   = 128

    # Linear-task LBACNP residual controls
    linear_lbacnp_residual:            bool  = True
    linear_lbacnp_residual_gate_init:  float = -3.0
    lbacnp_residual_penalty_weight:    float = 1e-4

    # Regression distribution head controls
    regression_distribution_head:      bool  = True
    nll_loss_weight:                   float = 0.0
    mse_loss_weight:                   float = 1.0
    min_log_var:                       float = -8.0
    max_log_var:                       float = 4.0

    def __post_init__(self):
        if self.pool not in VALID_POOLS:
            raise ValueError(f"Invalid pool '{self.pool}'. Valid: {sorted(VALID_POOLS)}")
        if self.icl_pool_mode not in VALID_POOLS:
            raise ValueError(
                f"Invalid icl_pool_mode '{self.icl_pool_mode}'. Valid: {sorted(VALID_POOLS)}"
            )
        if self.feature_pool_mode not in VALID_POOLS:
            raise ValueError(
                f"Invalid feature_pool_mode '{self.feature_pool_mode}'. Valid: {sorted(VALID_POOLS)}"
            )
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
        if self.model_arch_version not in ("model3", "model4", "model5_lbacnp"):
            raise ValueError(
                f"Invalid model_arch_version: {self.model_arch_version!r}. "
                "Valid: 'model3', 'model4', 'model5_lbacnp'"
            )
        valid_task_objectives = REGRESSION_TASK_OBJECTIVES | CLASSIFICATION_TASK_OBJECTIVES | {
            "transductive_completion",
        }
        if self.task_objective not in valid_task_objectives:
            raise ValueError(
                f"Invalid task_objective={self.task_objective!r}. "
                f"Valid: {sorted(valid_task_objectives)}"
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
        # MODEL4 consistency
        if self.use_coefficient_head and not self.use_linear_stats:
            raise ValueError(
                "use_coefficient_head=True requires use_linear_stats=True "
                "(the coefficient head requires feat_stats from the extractor)."
            )
        classification_enabled = (
            self.use_classification_path
            or self.task_objective in CLASSIFICATION_TASK_OBJECTIVES
        )
        if classification_enabled and self.model_arch_version not in ("model4", "model5_lbacnp"):
            raise ValueError(
                "The classification path requires model_arch_version='model4' or 'model5_lbacnp'."
            )
        if self.max_num_classes < 2:
            raise ValueError("max_num_classes must be at least 2.")
        for field_name in (
            "class_embedding_dim",
            "class_stat_dim",
            "class_stat_hidden_dim",
            "class_head_hidden_dim",
            "class_fusion_gate_hidden_dim",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if not 0.0 <= self.class_label_smoothing < 1.0:
            raise ValueError("class_label_smoothing must be in [0, 1).")
        if self.use_lbacnp:
            if self.lbacnp_latent_dim != self.d_phi:
                raise ValueError("MODEL5 v1 requires lbacnp_latent_dim == d_phi")
            if self.lbacnp_latent_dim % self.lbacnp_heads != 0:
                raise ValueError(
                    f"lbacnp_latent_dim={self.lbacnp_latent_dim} must be divisible by "
                    f"lbacnp_heads={self.lbacnp_heads}"
                )
            if not (0.0 < self.lbacnp_latent_frac <= 1.0):
                raise ValueError("lbacnp_latent_frac must be in (0, 1]")
            if self.lbacnp_latent_max < 1:
                raise ValueError("lbacnp_latent_max must be positive")
            if self.lbacnp_n_blocks < 1:
                raise ValueError("lbacnp_n_blocks must be positive")
        if self.use_numeric_ple:
            if self.ple_num_bins < 2:
                raise ValueError("ple_num_bins must be >= 2")
            if self.ple_boundary_mode not in {"context_quantile", "fixed", "context_supervised"}:
                raise ValueError(
                    f"Invalid ple_boundary_mode: {self.ple_boundary_mode!r}. "
                    "Valid: 'context_quantile', 'fixed', 'context_supervised'"
                )
            if self.ple_target_aware:
                raise ValueError(
                    "ple_target_aware must remain False until context-only supervised PLE "
                    "is implemented and tested"
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
# MODEL5-LBACNP modules
# ---------------------------------------------------------------------------

class NumericPLEEncoder(nn.Module):
    """
    Context-quantile Piecewise Linear Encoding for numeric features.

    Fits T+1 quantile boundaries from X_context only, then maps each scalar to a
    T-dim PLE vector and projects to d_out via a shared Linear(T, d_out).

    Input:  X_context (n, p), X_query (m, p)
    Output: E_context (n, p, d_out), E_query (m, p, d_out)
    """

    def __init__(self, ple_num_bins: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.ple_num_bins = ple_num_bins
        self.projection = nn.Linear(ple_num_bins, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p=dropout)

    @staticmethod
    def _encode(x: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """
        x:          (batch, p)
        boundaries: (p, T+1)  — T+1 quantile points (including min and max from context)
        Returns:    (batch, p, T)  — piecewise-linear thermometer encoding
        """
        left  = boundaries[:, :-1].unsqueeze(0)           # (1, p, T)
        right = boundaries[:, 1:].unsqueeze(0)             # (1, p, T)
        width = (right - left).clamp(min=1e-8)             # (1, p, T)
        x_exp = x.unsqueeze(-1)                            # (batch, p, 1)
        return ((x_exp - left) / width).clamp(0.0, 1.0)   # (batch, p, T)

    def forward(
        self,
        X_context: torch.Tensor,   # (n, p)
        X_query:   torch.Tensor,   # (m, p)
    ):
        T = self.ple_num_bins
        # Fit T+1 quantile boundaries from context only (no query, no target)
        pts = torch.linspace(0.0, 1.0, T + 1, device=X_context.device, dtype=torch.float32)
        boundaries = torch.quantile(
            X_context.float(), pts, dim=0
        ).T.to(X_context.dtype)                            # (p, T+1)
        # Ensure monotone (handles constant or near-constant columns)
        boundaries = boundaries.cummax(dim=-1).values

        ple_ctx = self._encode(X_context, boundaries)      # (n, p, T)
        ple_qry = self._encode(X_query,   boundaries)      # (m, p, T)

        E_ctx = self.drop(self.norm(self.projection(ple_ctx)))   # (n, p, d_out)
        E_qry = self.drop(self.norm(self.projection(ple_qry)))   # (m, p, d_out)
        return E_ctx, E_qry


class ContextPairEncoder(nn.Module):
    """
    Maps per-feature tokens + target scalar to a per-row context embedding.

    Input:  tokens (n, p, d), y (n,)
    Output: (n, d)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.feature_pool = AttentionPool(d_model, n_heads, dropout)
        self.y_proj = nn.Linear(1, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """tokens: (n, p, d), y: (n,) -> (n, d)"""
        row_emb = self.feature_pool(tokens)               # (n, d)
        y_emb   = self.y_proj(y.unsqueeze(-1).float())    # (n, d)
        return self.drop(self.norm(row_emb + y_emb.to(row_emb.dtype)))


class QueryFeatureEncoder(nn.Module):
    """
    Maps per-feature tokens to a per-query-row embedding.

    Input:  tokens (m, p, d)
    Output: (m, d)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.feature_pool = AttentionPool(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (m, p, d) -> (m, d)"""
        return self.drop(self.norm(self.feature_pool(tokens)))


class LatentBottleneckBlock(nn.Module):
    """
    One LBACNP block:
      latent = MAB(latent, context)   — context → latent cross-attention
      latent = SAB(latent)            — latent self-attention
      query  = MAB(query,  latent)    — query  → latent cross-attention

    All attention sequences must be presented with a batch dimension;
    this block adds/removes it internally.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.context_cross_attn = MAB(d_model, n_heads, dropout)
        self.latent_self_attn   = SAB(d_model, n_heads, dropout)
        self.query_cross_attn   = MAB(d_model, n_heads, dropout)

    def forward(
        self,
        latent:  torch.Tensor,   # (L, d)
        context: torch.Tensor,   # (n, d)
        query:   torch.Tensor,   # (m, d)
    ):
        latent = self.context_cross_attn(
            latent.unsqueeze(0), context.unsqueeze(0)
        ).squeeze(0)
        latent = self.latent_self_attn(latent.unsqueeze(0)).squeeze(0)
        query  = self.query_cross_attn(
            query.unsqueeze(0), latent.unsqueeze(0)
        ).squeeze(0)
        return latent, query


class LatentBottleneckACNP(nn.Module):
    """
    Stack of LatentBottleneckBlocks with a learned latent seed bank.

    At runtime L_eff = min(latent_max, max(1, floor(latent_frac * n_context)))
    seed bank rows [:L_eff] are used as the initial latent.

    Complexity: O(n*L_eff) + O(L_eff^2) + O(m*L_eff) — no O(m*n) attention.
    """

    def __init__(
        self,
        d_model:    int,
        n_blocks:   int,
        n_heads:    int,
        latent_max: int,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.latent_max = latent_max
        self.latent_seed_bank = nn.Parameter(torch.randn(latent_max, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            LatentBottleneckBlock(d_model, n_heads, dropout)
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        context:      torch.Tensor,   # (n, d)
        query:        torch.Tensor,   # (m, d)
        latent_frac:  float = 0.25,
    ) -> torch.Tensor:
        """Returns updated query embeddings: (m, d)"""
        n = context.shape[0]
        L_eff = min(self.latent_max, max(1, int(latent_frac * n)))
        latent = self.latent_seed_bank[:L_eff]   # (L_eff, d)
        for block in self.blocks:
            latent, query = block(latent, context, query)
        return query


class RegressionDistributionHead(nn.Module):
    """
    Maps query embeddings to (mu, log_var) for a Gaussian prediction distribution.

    Input:  qemb (m, d)
    Output: mu (m,), log_var (m,)
    """

    def __init__(
        self,
        d_model:     int,
        hidden_dim:  int,
        dropout:     float = 0.0,
        min_log_var: float = -8.0,
        max_log_var: float = 4.0,
    ):
        super().__init__()
        self.mu_head      = build_mlp(d_model, 1, hidden_dim, dropout)
        self.log_var_head = build_mlp(d_model, 1, hidden_dim, dropout)
        self.min_log_var  = min_log_var
        self.max_log_var  = max_log_var

    def forward(self, qemb: torch.Tensor):
        """qemb: (m, d) -> mu (m,), log_var (m,)"""
        mu      = self.mu_head(qemb).squeeze(-1)
        log_var = self.log_var_head(qemb).squeeze(-1).clamp(
            self.min_log_var, self.max_log_var
        )
        return mu, log_var


class ClassificationLBACNPHead(nn.Module):
    """
    Maps query embeddings to multi-class logits.

    Input:  qemb (m, d), num_classes K
    Output: logits (m, K)
    """

    def __init__(self, d_model: int, hidden_dim: int, max_num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.head = build_mlp(d_model, max_num_classes, hidden_dim, dropout)

    def forward(self, qemb: torch.Tensor, num_classes: int) -> torch.Tensor:
        """qemb: (m, d) -> logits (m, K)"""
        return self.head(qemb)[:, :num_classes]


class ResidualGate(nn.Module):
    """
    Sigmoid gate initialised near zero to suppress LBACNP residual at training start.

    gate_init: initial last-layer bias; negative → gate ≈ sigmoid(gate_init) ≈ 0.
    """

    def __init__(self, d_model: int, hidden_dim: int, gate_init: float = -3.0, dropout: float = 0.0):
        super().__init__()
        self.gate_mlp = build_mlp(d_model, 1, hidden_dim, dropout)
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(gate_init)

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        """query_emb: (m, d) -> gate (m,) in [0, 1]"""
        return torch.sigmoid(self.gate_mlp(query_emb).squeeze(-1))



# ---------------------------------------------------------------------------
# MODEL4 modules: LinearStatisticExtractor, LinearStatisticEncoder,
#                 FusionModule, CoefficientHead, LambdaHead
# ---------------------------------------------------------------------------

class LinearStatisticExtractor(nn.Module):
    """
    Extracts sufficient linear statistics from normalized (X_train, y_train).

    No trainable parameters. Runs in float32 (caller wraps in autocast(enabled=False)).

    Returns:
        feat_stats  (p, N_FEAT_STATS)  — per-feature: sxy, diag_sxx, x_mean, x_std, corr, norm_sxy
        global_stats (N_GLOBAL_STATS,) — global: y_mean, y_std, y_var, p/n, n/p, cond, eff_rank
    """

    def __init__(self, max_moment_p: int = 128):
        super().__init__()
        # No parameters — feature extraction only.

    @staticmethod
    def _condition_number(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s = torch.linalg.svdvals(X.float())
        return (s[0] / s[-1].clamp(min=eps)).clamp(max=1e6).log1p()

    @staticmethod
    def _effective_rank(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s = torch.linalg.svdvals(X.float())
        s2 = s ** 2
        p_i = s2 / s2.sum().clamp(min=eps)
        return (-(p_i.clamp(min=eps) * p_i.clamp(min=eps).log()).sum()).exp()

    def forward(
        self,
        X_norm: torch.Tensor,   # (n, p) float32
        y_norm: torch.Tensor,   # (n,)   float32
    ):
        EPS = 1e-8
        n, p = X_norm.shape

        # Per-feature statistics (vectorized, no per-feature loop)
        sxy      = (X_norm * y_norm.unsqueeze(1)).sum(0)             # (p,)  X.T @ y
        diag_sxx = (X_norm ** 2).sum(0)                              # (p,)  diag(X.T @ X)
        x_mean   = X_norm.mean(0)                                    # (p,)
        x_std    = X_norm.std(0, unbiased=False).clamp(min=EPS)      # (p,)
        corr     = sxy / (diag_sxx.sqrt() * y_norm.norm() + EPS)    # (p,)
        norm_sxy = sxy / (n * x_std + EPS)                           # (p,)
        feat_stats = torch.stack(
            [sxy, diag_sxx, x_mean, x_std, corr, norm_sxy], dim=1   # (p, 6)
        )

        # Global statistics
        cond = self._condition_number(X_norm)
        eff_rank = self._effective_rank(X_norm)
        global_stats = torch.stack([
            y_norm.mean(),
            y_norm.std(unbiased=False).clamp(min=EPS),
            y_norm.var(unbiased=False),
            torch.tensor(p / max(n, 1), device=X_norm.device, dtype=torch.float32),
            torch.tensor(n / max(p, 1), device=X_norm.device, dtype=torch.float32),
            cond,
            eff_rank,
        ])                                                           # (7,)

        return feat_stats, global_stats


class LinearStatisticEncoder(nn.Module):
    """
    Encodes (feat_stats, global_stats) into a single fixed-dim embedding.

    feat_proj:   shared MLP over each feature row (p, n_feat) → (p, linear_stat_dim)
    global_proj: MLP over (pooled_feat ++ global_stats) → (linear_stat_dim,)
    """

    def __init__(
        self,
        n_feat_stats: int,
        n_global_stats: int,
        linear_stat_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.feat_proj   = build_mlp(n_feat_stats, linear_stat_dim, hidden_dim, dropout)
        self.global_proj = build_mlp(linear_stat_dim + n_global_stats, linear_stat_dim,
                                     hidden_dim, dropout)

    def forward(
        self,
        feat_stats:   torch.Tensor,   # (p, N_FEAT_STATS)
        global_stats: torch.Tensor,   # (N_GLOBAL_STATS,)
    ) -> torch.Tensor:
        """Returns: (linear_stat_dim,)"""
        feat_pooled = self.feat_proj(feat_stats).mean(0)                    # (linear_stat_dim,)
        combined    = torch.cat([feat_pooled, global_stats], dim=0)         # (linear_stat_dim + 7,)
        return self.global_proj(combined)                                   # (linear_stat_dim,)


class FusionModule(nn.Module):
    """
    Gated fusion of DeepSet neural embedding and linear-statistic embedding.

    gate = sigmoid(MLP([h_glob, z_linear]))
    output = gate * h_glob + (1 - gate) * z_linear_proj

    When neural_out_dim != linear_stat_dim, z_linear is projected first.
    """

    def __init__(
        self,
        neural_out_dim: int,
        linear_stat_dim: int,
        gate_hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_proj = (
            nn.Linear(linear_stat_dim, neural_out_dim)
            if linear_stat_dim != neural_out_dim
            else None
        )
        self.gate_mlp = build_mlp(
            neural_out_dim + linear_stat_dim,
            neural_out_dim,
            gate_hidden_dim,
            dropout,
        )

    def forward(
        self,
        h_glob:       torch.Tensor,   # (m, neural_out_dim)
        z_linear_exp: torch.Tensor,   # (m, linear_stat_dim)
    ) -> torch.Tensor:
        """Returns: (m, neural_out_dim)"""
        gate = torch.sigmoid(
            self.gate_mlp(torch.cat([h_glob, z_linear_exp], dim=-1))   # (m, neural_out_dim)
        )
        z = self.z_proj(z_linear_exp) if self.z_proj is not None else z_linear_exp
        return gate * h_glob + (1.0 - gate) * z


class CoefficientHead(nn.Module):
    """
    Per-feature shared MLP that estimates beta_hat in normalized space.

    Input per feature row: [feat_stats_j (N_FEAT_STATS,), h_global (global_dim,)]
    Output: scalar beta_hat_j
    Handles any p with the same weights.
    """

    def __init__(
        self,
        n_feat_stats: int,
        global_dim:   int,
        hidden_dim:   int,
        dropout:      float,
    ):
        super().__init__()
        self.mlp = build_mlp(n_feat_stats + global_dim, 1, hidden_dim, dropout)

    def forward(
        self,
        feat_stats: torch.Tensor,   # (p, N_FEAT_STATS)
        h_global:   torch.Tensor,   # (global_dim,)
    ) -> torch.Tensor:
        """Returns: (p,) beta_hat in normalized space"""
        p = feat_stats.shape[0]
        h_exp    = h_global.unsqueeze(0).expand(p, -1)              # (p, global_dim)
        combined = torch.cat([feat_stats, h_exp], dim=1)            # (p, N_FEAT_STATS + global_dim)
        return self.mlp(combined).squeeze(-1)                       # (p,)


class LambdaHead(nn.Module):
    """
    Small MLP that outputs a dataset-specific positive regularization strength.
    Linear → ReLU → Linear → Softplus  (guarantees positive scalar output)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, h_global: torch.Tensor) -> torch.Tensor:
        """h_global: (input_dim,) → scalar"""
        return self.mlp(h_global).squeeze(-1)


# ---------------------------------------------------------------------------
# MODEL4 linear-classification modules
# ---------------------------------------------------------------------------

class ClassLabelEncoder(nn.Module):
    """Encode support labels and a learned masked query-label token."""

    def __init__(self, max_num_classes: int, class_embedding_dim: int, d_phi: int):
        super().__init__()
        self.max_num_classes = int(max_num_classes)
        self.class_embed = nn.Embedding(max_num_classes, class_embedding_dim)
        self.mask_label = nn.Parameter(torch.zeros(class_embedding_dim))
        self.proj = nn.Linear(class_embedding_dim, d_phi)

    def forward_train_labels(self, y_train: torch.Tensor) -> torch.Tensor:
        return self.proj(self.class_embed(y_train.long()))

    def forward_query_mask(
        self,
        m: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = self.mask_label.to(device=device, dtype=dtype).unsqueeze(0).expand(m, -1)
        return self.proj(mask)

    def class_embeddings(self, num_classes: int, *, dtype: torch.dtype) -> torch.Tensor:
        return self.class_embed.weight[:num_classes].to(dtype=dtype)


class ClassificationStatisticExtractor(nn.Module):
    """Parameter-free class-conditional statistics computed in float32."""

    @staticmethod
    def _effective_rank(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        singular_values = torch.linalg.svdvals(X)
        squared = singular_values.square()
        probs = squared / squared.sum().clamp(min=eps)
        return (-(probs.clamp(min=eps) * probs.clamp(min=eps).log()).sum()).exp()

    def forward(
        self,
        X_norm: torch.Tensor,
        y_train: torch.Tensor,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_dtype = X_norm.dtype
        with torch.amp.autocast(device_type=X_norm.device.type, enabled=False):
            X = X_norm.float()
            y = y_train.long()
            n, p = X.shape
            K = int(num_classes)
            eps = 1e-8
            global_mean = X.mean(dim=0)
            global_var = X.var(dim=0, unbiased=False)
            global_std = global_var.sqrt().clamp(min=eps)
            feat_stats = torch.zeros(
                p, K, N_CLASS_FEAT_STATS, device=X.device, dtype=torch.float32
            )
            counts = torch.bincount(y, minlength=K).to(torch.float32)
            present = counts > 0
            priors = counts / max(n, 1)

            for class_id in range(K):
                count = counts[class_id]
                feat_stats[:, class_id, 0] = count
                feat_stats[:, class_id, 1] = priors[class_id]
                if not bool(present[class_id].item()):
                    continue
                indicator = (y == class_id).to(torch.float32)
                X_class = X[y == class_id]
                class_mean = X_class.mean(dim=0)
                class_var = X_class.var(dim=0, unbiased=False)
                class_std = class_var.sqrt()
                mean_diff = class_mean - global_mean
                standardized_diff = mean_diff / global_std
                indicator_centered = indicator - indicator.mean()
                X_centered = X - global_mean
                numerator = (X_centered * indicator_centered.unsqueeze(1)).sum(dim=0)
                denominator = (
                    X_centered.square().sum(dim=0).sqrt()
                    * indicator_centered.square().sum().sqrt()
                ).clamp(min=eps)
                corr = numerator / denominator
                feat_stats[:, class_id, 2:] = torch.stack(
                    [
                        class_mean,
                        class_var,
                        class_std,
                        mean_diff,
                        standardized_diff,
                        corr,
                        corr.abs(),
                    ],
                    dim=-1,
                )

            present_priors = priors[present]
            if present_priors.numel():
                minority = present_priors.min()
                majority = present_priors.max()
                imbalance = majority / minority.clamp(min=eps)
            else:
                minority = majority = imbalance = torch.tensor(
                    0.0, device=X.device, dtype=torch.float32
                )
            entropy = -(
                priors[present].clamp(min=eps)
                * priors[present].clamp(min=eps).log()
            ).sum()
            normalizer = present.sum().clamp(min=1) * max(p, 1)
            mean_abs_corr = feat_stats[:, :, 8].sum() / normalizer
            mean_separation = feat_stats[:, :, 6].abs().sum() / normalizer
            global_stats = torch.stack(
                [
                    torch.tensor(float(n), device=X.device),
                    torch.tensor(float(p), device=X.device),
                    torch.tensor(float(K), device=X.device),
                    torch.tensor(p / max(n, 1), device=X.device),
                    torch.tensor(n / max(p, 1), device=X.device),
                    entropy,
                    imbalance,
                    minority,
                    majority,
                    self._effective_rank(X),
                    mean_abs_corr,
                    mean_separation,
                ]
            )
        return feat_stats.to(original_dtype), global_stats.to(original_dtype), present


class ClassificationStatisticEncoder(nn.Module):
    """Pool class-feature statistics invariantly over variable p and K."""

    def __init__(
        self,
        n_feat_stats: int,
        n_global_stats: int,
        stat_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.feat_class_proj = build_mlp(
            n_feat_stats, stat_dim, hidden_dim, dropout
        )
        self.global_proj = build_mlp(
            stat_dim + n_global_stats, stat_dim, hidden_dim, dropout
        )

    def forward(
        self,
        class_feat_stats: torch.Tensor,
        class_global_stats: torch.Tensor,
        class_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.feat_class_proj(class_feat_stats)
        mask = class_present_mask.to(encoded.dtype).view(1, -1, 1)
        denominator = (mask.sum() * max(encoded.shape[0], 1)).clamp(min=1.0)
        pooled = (encoded * mask).sum(dim=(0, 1)) / denominator
        return self.global_proj(torch.cat([pooled, class_global_stats], dim=0))


class ClassFusionModule(FusionModule):
    """Classification-specific gated fusion with independent parameters."""


class ClassCoefficientHead(nn.Module):
    """Shared per-(feature,class) head producing normalized coefficients."""

    def __init__(
        self,
        n_feat_stats: int,
        global_dim: int,
        class_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.mlp = build_mlp(
            n_feat_stats + global_dim + class_embedding_dim,
            1,
            hidden_dim,
            dropout,
        )

    def forward(
        self,
        class_feat_stats: torch.Tensor,
        h_global: torch.Tensor,
        class_embeddings: torch.Tensor,
        class_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        p, K, _ = class_feat_stats.shape
        h = h_global.view(1, 1, -1).expand(p, K, -1)
        class_emb = class_embeddings.view(1, K, -1).expand(p, K, -1)
        coefficients = self.mlp(
            torch.cat([class_feat_stats, h, class_emb], dim=-1)
        ).squeeze(-1)
        return coefficients - coefficients[:, :1]


class ClassBiasHead(nn.Module):
    """Per-class intercept head using global and class-prior information."""

    def __init__(
        self,
        n_global_stats: int,
        global_dim: int,
        class_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.mlp = build_mlp(
            n_global_stats + global_dim + class_embedding_dim + 2,
            1,
            hidden_dim,
            dropout,
        )

    def forward(
        self,
        class_feat_stats: torch.Tensor,
        class_global_stats: torch.Tensor,
        h_global: torch.Tensor,
        class_embeddings: torch.Tensor,
        class_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        K = class_embeddings.shape[0]
        global_exp = class_global_stats.unsqueeze(0).expand(K, -1)
        h_exp = h_global.unsqueeze(0).expand(K, -1)
        count_prior = class_feat_stats[:, :, :2].mean(dim=0)
        bias = self.mlp(
            torch.cat(
                [global_exp, h_exp, class_embeddings, count_prior],
                dim=-1,
            )
        ).squeeze(-1)
        return bias - bias[:1]


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


def _no_compile(fn):
    """Decorator that excludes `fn` from torch.compile capture (graph break).

    Used to keep host-syncing float32 linear algebra (svdvals / lstsq / solve, which
    allocate dynamic LAPACK workspaces and synchronise the host) OUT of any captured
    CUDA-graph region.  When the CUDA-graph training path runs these statistics on a
    variable-sized valid sub-block, the graph break guarantees the captured core stays
    static-shaped and sync-free.  No-op when torch._dynamo is unavailable.
    """
    _dyn = getattr(torch, "_dynamo", None)
    if _dyn is not None and hasattr(_dyn, "disable"):
        return _dyn.disable(fn)
    return fn


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

    def __init__(self, d_model: int, dropout: float = 0.0, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
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

    def _row_fwd(self, H: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            col_agg = _masked_mean(H, mask, dim=2, keepdim=True)
        else:
            col_agg = H.mean(dim=2, keepdim=True)
        x = torch.cat([H, col_agg.expand_as(H)], dim=-1)
        for module in self.row_update:
            x = module(x)
        return x

    def _col_fwd(self, H: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            row_agg = _masked_mean(H, mask, dim=1, keepdim=True)
        else:
            row_agg = H.mean(dim=1, keepdim=True)
        x = torch.cat([H, row_agg.expand_as(H)], dim=-1)
        for module in self.col_update:
            x = module(x)
        return x

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
        if self.use_checkpoint and torch.is_grad_enabled():
            H = self.norm_row(H + _checkpoint(self._row_fwd, H, mask, use_reentrant=False))
            H = self.norm_col(H + _checkpoint(self._col_fwd, H, mask, use_reentrant=False))
        else:
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
        _use_ckpt = os.environ.get("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"
        self.blocks = nn.ModuleList([
            ExchangeableMatrixBlock(d_model, cfg.dropout, use_checkpoint=_use_ckpt)
            for _ in range(n_blocks)
        ])
        self.icl_sample_pool = (
            SetPool(d_model, cfg.icl_pool_mode, cfg.n_heads, cfg.dropout)
            if cfg.icl_pool_mode != "mean"
            else None
        )
        sample_out_dim = d_model * POOL_SCALE[cfg.icl_pool_mode]
        self.icl_feature_pool = (
            SetPool(sample_out_dim, cfg.feature_pool_mode, cfg.n_heads, cfg.dropout)
            if cfg.feature_pool_mode != "mean"
            else None
        )
        neural_out_dim = sample_out_dim * POOL_SCALE[cfg.feature_pool_mode]
        self.pred_head = nn.Linear(neural_out_dim, 1)

        # MODEL4 modules (active when use_linear_stats=True, the new default)
        if cfg.use_linear_stats:
            self.stat_extractor = LinearStatisticExtractor(cfg.max_moment_p)
            self.stat_encoder   = LinearStatisticEncoder(
                N_FEAT_STATS, N_GLOBAL_STATS, cfg.linear_stat_dim,
                cfg.linear_encoder_hidden_dim, cfg.dropout,
            )
            self.fusion = FusionModule(
                neural_out_dim, cfg.linear_stat_dim, cfg.fusion_gate_hidden_dim, cfg.dropout,
            )
        else:
            self.stat_extractor = self.stat_encoder = self.fusion = None

        if cfg.use_coefficient_head:
            self.coeff_head = CoefficientHead(
                N_FEAT_STATS, neural_out_dim, cfg.coeff_head_hidden_dim, cfg.dropout,
            )
        else:
            self.coeff_head = None

        if cfg.use_lambda_head:
            self.lambda_head = LambdaHead(neural_out_dim, cfg.lambda_head_hidden_dim)
        else:
            self.lambda_head = None

        self.classification_enabled = (
            cfg.use_classification_path
            or cfg.task_objective in CLASSIFICATION_TASK_OBJECTIVES
        )
        if self.classification_enabled:
            self.class_label_encoder = ClassLabelEncoder(
                cfg.max_num_classes,
                cfg.class_embedding_dim,
                cfg.d_phi,
            )
            self.class_cell_encoder = CellEncoder(d_in=2, d_model=cfg.d_phi)
            self.class_stat_extractor = ClassificationStatisticExtractor()
            self.class_stat_encoder = ClassificationStatisticEncoder(
                N_CLASS_FEAT_STATS,
                N_CLASS_GLOBAL_STATS,
                cfg.class_stat_dim,
                cfg.class_stat_hidden_dim,
                cfg.dropout,
            )
            self.class_fusion = ClassFusionModule(
                neural_out_dim,
                cfg.class_stat_dim,
                cfg.class_fusion_gate_hidden_dim,
                cfg.dropout,
            )
            self.class_coeff_head = ClassCoefficientHead(
                N_CLASS_FEAT_STATS,
                neural_out_dim,
                cfg.class_embedding_dim,
                cfg.class_head_hidden_dim,
                cfg.dropout,
            )
            self.class_bias_head = ClassBiasHead(
                N_CLASS_GLOBAL_STATS,
                neural_out_dim,
                cfg.class_embedding_dim,
                cfg.class_head_hidden_dim,
                cfg.dropout,
            )
            self.class_neural_head = nn.Linear(
                neural_out_dim, cfg.max_num_classes
            )
        else:
            self.class_label_encoder = None
            self.class_cell_encoder = None
            self.class_stat_extractor = None
            self.class_stat_encoder = None
            self.class_fusion = None
            self.class_coeff_head = None
            self.class_bias_head = None
            self.class_neural_head = None

        # Mixed-categorical modules (only when use_categorical_features=True)
        if cfg.use_categorical_features:
            self.cat_token_encoder = CategoricalTokenEncoder(cfg)
            self.cat_stat_extractor = CategoricalStatisticExtractor()
            self.cat_stat_encoder = CategoricalStatisticEncoder(cfg)
            self.cat_reg_head = RegressionCategoryEffectHead(cfg, neural_out_dim)
            self.cat_cls_head = ClassificationCategoryEffectHead(
                cfg, neural_out_dim, cfg.max_num_classes
            )
        else:
            self.cat_token_encoder = None
            self.cat_stat_extractor = None
            self.cat_stat_encoder = None
            self.cat_reg_head = None
            self.cat_cls_head = None

        # MODEL5-LBACNP modules (only when use_lbacnp=True)
        if cfg.use_lbacnp:
            if cfg.use_numeric_ple:
                self.ple_encoder = NumericPLEEncoder(
                    ple_num_bins=cfg.ple_num_bins,
                    d_out=cfg.ple_projection_dim,
                    dropout=cfg.lbacnp_dropout,
                )
            else:
                self.ple_encoder = None
            self.lbacnp_context_encoder = ContextPairEncoder(
                d_model=d_model,
                n_heads=cfg.lbacnp_heads,
                dropout=cfg.lbacnp_dropout,
            )
            self.lbacnp_query_encoder = QueryFeatureEncoder(
                d_model=d_model,
                n_heads=cfg.lbacnp_heads,
                dropout=cfg.lbacnp_dropout,
            )
            self.lbacnp = LatentBottleneckACNP(
                d_model=d_model,
                n_blocks=cfg.lbacnp_n_blocks,
                n_heads=cfg.lbacnp_heads,
                latent_max=cfg.lbacnp_latent_max,
                dropout=cfg.lbacnp_dropout,
            )
            self.lbacnp_reg_head = RegressionDistributionHead(
                d_model=d_model,
                hidden_dim=d_model,
                dropout=cfg.lbacnp_dropout,
                min_log_var=cfg.min_log_var,
                max_log_var=cfg.max_log_var,
            )
            self.lbacnp_cls_head = ClassificationLBACNPHead(
                d_model=d_model,
                hidden_dim=d_model,
                max_num_classes=cfg.max_num_classes,
                dropout=cfg.lbacnp_dropout,
            )
            if cfg.linear_lbacnp_residual:
                self.lbacnp_residual_gate = ResidualGate(
                    d_model=d_model,
                    hidden_dim=max(1, d_model // 2),
                    gate_init=cfg.linear_lbacnp_residual_gate_init,
                    dropout=cfg.lbacnp_dropout,
                )
            else:
                self.lbacnp_residual_gate = None
        else:
            self.ple_encoder = None
            self.lbacnp_context_encoder = None
            self.lbacnp_query_encoder = None
            self.lbacnp = None
            self.lbacnp_reg_head = None
            self.lbacnp_cls_head = None
            self.lbacnp_residual_gate = None

    @staticmethod
    def _pool_sequence(x: torch.Tensor, pool: SetPool | None) -> torch.Tensor:
        return x.mean(dim=1) if pool is None else pool(x)

    def forward(
        self,
        X_train:     torch.Tensor,
        y_train:     torch.Tensor,
        x_test:      torch.Tensor,
        return_debug: bool = False,
        beta:        torch.Tensor | None = None,
        task_objective: str | None = None,
        num_classes: int | None = None,
        y_test: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        return_aux: bool = False,
        **kwargs,
    ):
        objective = task_objective or self.cfg.task_objective
        if objective in REGRESSION_TASK_OBJECTIVES:
            if self.cfg.use_lbacnp:
                return self.forward_regression_lbacnp(
                    X_train, y_train, x_test,
                    return_debug=return_debug, beta=beta, **kwargs,
                )
            return self.forward_regression(
                X_train,
                y_train,
                x_test,
                return_debug=return_debug,
                beta=beta,
                **kwargs,
            )
        if objective in CLASSIFICATION_TASK_OBJECTIVES:
            if num_classes is None:
                raise ValueError(
                    "num_classes is required for explicit classification routing."
                )
            if self.cfg.use_lbacnp:
                return self.forward_classification_lbacnp(
                    X_train, y_train, x_test,
                    num_classes=num_classes,
                    y_test=y_test,
                    class_weight=class_weight,
                    return_aux=return_aux,
                    **kwargs,
                )
            return self.forward_classification(
                X_train,
                y_train,
                x_test,
                num_classes=num_classes,
                y_test=y_test,
                class_weight=class_weight,
                return_aux=return_aux,
                **kwargs,
            )
        raise ValueError(
            "Unknown task_objective; require explicit task metadata. "
            f"Got {objective!r}."
        )

    @staticmethod
    def _masked_col_stats(X, row_mask, eps):
        """Per-column mean/std over valid rows only (biased var, matches std(unbiased=False)).

        Tensor-only (no host sync, static shape) so it stays inside the captured graph.
        """
        rmf = row_mask.to(X.dtype).unsqueeze(1)                    # (n, 1)
        cnt = rmf.sum().clamp(min=1.0)                             # == n_valid
        col_mean = (X * rmf).sum(dim=0) / cnt                      # (p,)
        col_var  = (((X - col_mean) ** 2) * rmf).sum(dim=0) / cnt  # (p,)
        col_std  = col_var.clamp(min=eps * eps).sqrt().clamp(min=eps)
        return col_mean, col_std

    @staticmethod
    def _masked_vec_stats(y, row_mask, eps):
        """Scalar mean/std of a 1-D vector over valid entries only (tensor-only)."""
        rmf = row_mask.to(y.dtype)                                 # (n,)
        cnt = rmf.sum().clamp(min=1.0)
        y_mean = (y * rmf).sum() / cnt
        y_var  = (((y - y_mean) ** 2) * rmf).sum() / cnt
        y_std  = y_var.clamp(min=eps * eps).sqrt().clamp(min=eps)
        return y_mean, y_std

    @_no_compile
    def _linear_stats_padded(self, X_norm, y_norm, n_valid, p_valid):
        """Eager (graph-broken) linear-statistic preamble for the CUDA-graph path.

        Operates on the variable-sized VALID sub-block `X_norm[:n_valid, :p_valid]`,
        so the host-syncing svdvals (cond / eff_rank) see real shapes and stay OUTSIDE
        the captured graph.  Returns fixed-shape tensors padded back to P_PAD so the
        compiled core that consumes them is static-shaped.
        """
        nv, pv = int(n_valid), int(p_valid)
        p = X_norm.shape[1]
        with torch.amp.autocast(device_type=X_norm.device.type, enabled=False):
            feat_v, global_stats = self.stat_extractor(
                X_norm[:nv, :pv].float(), y_norm[:nv].float()
            )                                                       # (pv, 6), (7,)
        feat_v       = feat_v.to(X_norm.dtype)
        global_stats = global_stats.to(X_norm.dtype)
        feat_full = feat_v.new_zeros((p, feat_v.shape[1]))          # (p, 6), padded rows = 0
        feat_full[:pv] = feat_v
        # Encode using ONLY the valid feature rows (mean-pool would be polluted by pad rows)
        z_linear = self.stat_encoder(feat_v, global_stats)          # (linear_stat_dim,)
        return feat_full, global_stats, z_linear

    @_no_compile
    def _ols_teacher_padded(self, X_norm, y_norm, xq_norm, n_valid, p_valid):
        """Eager (graph-broken) OLS teacher on the valid sub-block (lstsq/solve host-sync)."""
        nv, pv = int(n_valid), int(p_valid)
        p = X_norm.shape[1]
        with torch.no_grad():
            with torch.amp.autocast(device_type=X_norm.device.type, enabled=False):
                Xf  = X_norm[:nv, :pv].float()
                yf  = y_norm[:nv].float()
                xqf = xq_norm[:, :pv].float()
                kw  = dict(device=Xf.device, dtype=torch.float32)
                if nv / max(pv, 1) >= self.cfg.teacher_ols_threshold:
                    beta_v = torch.linalg.lstsq(Xf, yf, rcond=None).solution
                else:
                    A = Xf.T @ Xf + 1e-4 * torch.eye(pv, **kw)
                    beta_v = torch.linalg.solve(A, Xf.T @ yf)
                y_ols = xqf @ beta_v                                # (m,)
        beta_full = beta_v.new_zeros(p)
        beta_full[:pv] = beta_v
        return beta_full.to(X_norm.dtype), y_ols.to(X_norm.dtype)

    def forward_regression(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        return_debug: bool = False,
        beta: torch.Tensor | None = None,
        n_valid: torch.Tensor | None = None,
        p_valid: torch.Tensor | None = None,
        m_valid: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        X_train:    (n, p)         — training features
        y_train:    (n,)           — training labels
        x_test:     (p,) or (m, p) — query features
        beta:       (p,) optional  — ground-truth coefficient vector (for aux loss supervision)
        n_valid/p_valid/m_valid: 0-dim long tensors — when provided, (X_train, y_train,
            x_test) are zero-padded to a STATIC (N_PAD, P_PAD) / (M_PAD, P_PAD) shape and
            only the leading [:n_valid, :p_valid] / [:m_valid] sub-block is real.  Enables
            the CUDA-graph training path: all reductions over the support (n), feature (p),
            and query (m) axes are masked so the padded result is numerically identical to
            the unpadded forward.  When all None, this is the exact legacy path.
        Returns: scalar if x_test is (p,); (m,) if x_test is (m, p)
        """
        n, p   = X_train.shape
        EPS = 1e-8

        # --- CUDA-graph padded/masked mode -----------------------------------
        masked = n_valid is not None
        row_mask = col_mask = block_mask = None
        if masked:
            if self.icl_sample_pool is not None or self.icl_feature_pool is not None:
                raise ValueError(
                    "CUDA-graph padded path requires mean/mean pooling "
                    "(icl_pool_mode=feature_pool_mode='mean')."
                )
            if x_test.ndim == 1:
                raise ValueError("CUDA-graph padded path requires a 2-D (m, p) x_test.")
            if p_valid is None:
                p_valid = torch.tensor(p, device=X_train.device)
            row_mask = torch.arange(n, device=X_train.device) < n_valid   # (n,) bool
            col_mask = torch.arange(p, device=X_train.device) < p_valid   # (p,) bool
            # m_valid: if None when n_valid provided, fall back to all m rows valid.
            # Callers should always supply m_valid for exact parity with the unpadded path.

        single = x_test.ndim == 1
        if single:
            x_test = x_test.unsqueeze(0)   # (1, p)
        m  = x_test.shape[0]

        # --- Column statistics (always computed; used for ColumnEncoder + optional norm) ---
        if masked:
            col_mean, col_std = self._masked_col_stats(X_train, row_mask, EPS)
        else:
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
            if masked:
                y_mean, y_std = self._masked_vec_stats(y_train, row_mask, EPS)
            else:
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

        # Build the (m, n, p) validity mask once for the exchangeable blocks / pooling.
        if masked:
            block_mask = (row_mask.view(1, n, 1) & col_mask.view(1, 1, p)).expand(m, n, p)

        # --- Encode cells -> H + broadcast column embeddings ---
        _use_ckpt = self.blocks[0].use_checkpoint if self.blocks else False
        if _use_ckpt and torch.is_grad_enabled():
            H = _checkpoint(self.cell_encoder, cell_in, use_reentrant=False)
        else:
            H = self.cell_encoder(cell_in)                         # (m, n, p, d_model)
        H = H + col_emb.unsqueeze(0).unsqueeze(0)                 # broadcast (m, n, p, d_model)

        # --- Hartford-style exchangeable blocks ---
        for block in self.blocks:
            H = block(H, mask=block_mask)                          # (m, n, p, d_model)

        # --- MODEL4: Compute linear statistics (float32, outside AMP) ---
        feat_stats, z_linear = None, None
        if self.cfg.use_linear_stats and self.stat_extractor is not None:
            if masked:
                feat_stats, global_stats, z_linear = self._linear_stats_padded(
                    X_norm, y_norm, n_valid, p_valid
                )
            else:
                with torch.amp.autocast(device_type=X_train.device.type, enabled=False):
                    feat_stats_f32, global_stats_f32 = self.stat_extractor(
                        X_norm.float(), y_norm.float()
                    )
                feat_stats   = feat_stats_f32.to(X_norm.dtype)       # (p, 6)
                global_stats = global_stats_f32.to(X_norm.dtype)     # (7,)
                z_linear     = self.stat_encoder(feat_stats, global_stats)  # (linear_stat_dim,)

        # --- Pool: sample axis (n) then feature axis (p) ---
        if self.icl_sample_pool is None:
            if masked:
                h_feat = _masked_mean(H, block_mask, dim=1, keepdim=False)  # (m, p, d_model)
            else:
                h_feat = H.mean(dim=1)                                 # (m, p, d_model)
        else:
            H_mp = H.permute(0, 2, 1, 3).reshape(m * p, n, self.cfg.d_phi)
            h_feat = self.icl_sample_pool(H_mp).view(m, p, -1)
        if masked:
            cmf = col_mask.to(h_feat.dtype).view(1, p, 1)             # (1, p, 1)
            h_glob = (h_feat * cmf).sum(dim=1) / cmf.sum().clamp(min=1.0)  # (m, d_model)
        else:
            h_glob = self._pool_sequence(h_feat, self.icl_feature_pool)

        # --- MODEL4: Fuse DeepSet embedding with linear-stat embedding ---
        if self.cfg.use_linear_stats and self.fusion is not None and z_linear is not None:
            z_lin_exp = z_linear.unsqueeze(0).expand(m, -1)       # (m, linear_stat_dim)
            h_fused   = self.fusion(h_glob, z_lin_exp)            # (m, neural_out_dim)
        else:
            h_fused = h_glob                                       # MODEL3 unchanged path

        # --- MODEL4: Coefficient head → beta_hat + coefficient prediction ---
        # h_global_mean: task-level summary used by coeff_head and lambda_head.
        # In the padded path, padded query rows (q >= m_valid) get a non-zero
        # normalised value (-col_mean/col_std) because they are zeros in input space.
        # We must exclude them from the mean so the coefficient estimate matches the
        # unpadded path.  Masking is tensor-only (arange < m_valid) — stays in graph.
        if masked and m_valid is not None:
            _qmf = (torch.arange(m, device=h_fused.device) < m_valid).to(h_fused.dtype)
            h_global_mean = (h_fused * _qmf.unsqueeze(-1)).sum(0) / _qmf.sum().clamp(min=1.0)
        else:
            h_global_mean = h_fused.mean(0)                       # (neural_out_dim,)

        beta_hat_norm, y_coeff_norm = None, None
        if self.cfg.use_coefficient_head and self.coeff_head is not None:
            beta_hat_norm = self.coeff_head(feat_stats, h_global_mean)  # (p,)
            if masked:
                # Zero padded-feature coefficients so xq_norm @ beta is exact (pad cols → 0).
                beta_hat_norm = beta_hat_norm * col_mask.to(beta_hat_norm.dtype)
            y_coeff_norm  = xq_norm @ beta_hat_norm               # (m,)

        # --- MODEL4: Lambda head → dataset-specific regularization estimate ---
        lambda_hat = None
        if self.cfg.use_lambda_head and self.lambda_head is not None:
            lambda_hat = self.lambda_head(h_global_mean)          # scalar

        # --- MODEL4: On-the-fly OLS teacher (no_grad, float32, detached) ---
        # OLS is unbiased and avoids lambda-selection bias when n/p >= teacher_ols_threshold.
        # Training data guarantees n >= 5*p, so OLS is always used by default.
        beta_ols_teacher, y_ols_norm_teacher = None, None
        any_aux_active = (
            self.cfg.beta_aux_loss_weight > 0
            or self.cfg.pred_aux_loss_weight > 0
            or self.cfg.cos_aux_loss_weight > 0
        )
        if any_aux_active and (self.cfg.use_coefficient_head or self.cfg.use_lambda_head):
            if masked:
                beta_ols_teacher, y_ols_norm_teacher = self._ols_teacher_padded(
                    X_norm, y_norm, xq_norm, n_valid, p_valid
                )
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type=X_train.device.type, enabled=False):
                        n_tr, p_f = X_norm.shape
                        Xf  = X_norm.float()
                        yf  = y_norm.float()
                        kw  = dict(device=Xf.device, dtype=torch.float32)
                        if n_tr / max(p_f, 1) >= self.cfg.teacher_ols_threshold:
                            result = torch.linalg.lstsq(Xf, yf, rcond=None)
                            beta_ols_teacher = result.solution          # (p,)
                        else:
                            jitter = 1e-4
                            A = Xf.T @ Xf + jitter * torch.eye(p_f, **kw)
                            beta_ols_teacher = torch.linalg.solve(A, Xf.T @ yf)  # (p,)
                        y_ols_norm_teacher = (xq_norm.float() @ beta_ols_teacher).to(X_norm.dtype)
                beta_ols_teacher = beta_ols_teacher.to(X_norm.dtype)

        # --- MODEL4: beta in normalized space for teacher loss ---
        beta_norm = None
        if beta is not None and y_std is not None:
            beta_norm = (beta * col_std / y_std).detach()   # (p,)

        # --- Prediction ---
        neural_norm = self.pred_head(h_fused).squeeze(-1)             # (m,)
        debug = {}

        # --- MODEL4: Coefficient head primary path ---
        if self.cfg.use_coefficient_head and y_coeff_norm is not None:
            if self.cfg.use_residual_head:
                pred_norm = y_coeff_norm + neural_norm
            else:
                pred_norm = y_coeff_norm
            debug = {
                "feat_stats":           feat_stats,
                "z_linear":             z_linear,
                "beta_hat_norm":        beta_hat_norm,
                "y_coeff_norm":         y_coeff_norm,
                "lambda_hat":           lambda_hat,
                "beta_ols_teacher":     beta_ols_teacher,
                "y_ols_norm_teacher":   y_ols_norm_teacher,
                "beta_norm":            beta_norm,
                "neural_prediction":    neural_norm,
                "final_normalized_prediction": pred_norm,
            }
        else:
            pred_norm = neural_norm
            debug = {
                "neural_prediction": neural_norm,
                "final_normalized_prediction": pred_norm,
            }

        # --- Mixed-categorical path: add categorical effect to pred_norm ---
        if (self.cfg.use_categorical_features
                and self.cat_token_encoder is not None
                and "X_cat_train" in kwargs and kwargs["X_cat_train"] is not None):
            X_cat_train = kwargs["X_cat_train"]
            X_cat_test = kwargs.get("X_cat_test")
            categorical_cardinalities = kwargs.get("categorical_cardinalities")
            if X_cat_test is not None and categorical_cardinalities is not None:
                objective = kwargs.get("task_objective") or self.cfg.task_objective
                ctx_cat_tokens = self.cat_token_encoder(X_cat_train, categorical_cardinalities)
                feat_cat_stats, global_cat_stats = self.cat_stat_extractor(
                    X_cat_train, y_norm.float(), categorical_cardinalities, objective
                )
                z_cat_stat = self.cat_stat_encoder(
                    feat_cat_stats, global_cat_stats, categorical_cardinalities
                )
                query_cat_tokens = self.cat_token_encoder(X_cat_test, categorical_cardinalities)
                h_global_mean = h_fused.mean(0)
                cat_effect = self.cat_reg_head(query_cat_tokens, z_cat_stat, h_global_mean)
                pred_norm = pred_norm + cat_effect

        # --- Denormalise ---
        if self.cfg.norm_target:
            y_hat = pred_norm * y_std + y_mean
        else:
            y_hat = pred_norm

        if single:
            y_hat = y_hat.squeeze(0)
            debug = {
                key: val.squeeze(0) if torch.is_tensor(val) and val.ndim > 0 else val
                for key, val in debug.items()
            }
        return (y_hat, debug) if return_debug else y_hat

    def forward_classification(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        num_classes: int,
        y_test: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        return_aux: bool = False,
        **kwargs,
    ) -> dict:
        """Predict binary or multiclass query labels with `(m, K)` logits."""
        if not self.classification_enabled:
            raise RuntimeError(
                "Classification modules are disabled. Set "
                "use_classification_path=True and task_objective="
                "'inductive_classification' in ModelConfig."
            )
        if X_train.ndim != 2:
            raise ValueError(f"X_train must have shape (n,p), got {tuple(X_train.shape)}.")
        if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
            raise ValueError("y_train must have shape (n,) matching X_train.")
        if y_train.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        ):
            raise TypeError("Classification y_train must contain integer class labels.")
        K = int(num_classes)
        if K < 2 or K > self.cfg.max_num_classes:
            raise ValueError(
                f"num_classes must be in [2, {self.cfg.max_num_classes}], got {K}."
            )
        if y_train.numel() == 0:
            raise ValueError("Classification support set must not be empty.")
        if int(y_train.min()) < 0 or int(y_train.max()) >= K:
            raise ValueError(f"Classification labels must be contiguous IDs in 0..{K - 1}.")

        if x_test.ndim == 1:
            x_test = x_test.unsqueeze(0)
        if x_test.ndim != 2 or x_test.shape[1] != X_train.shape[1]:
            raise ValueError("x_test must have shape (m,p) with the support feature width.")

        n, p = X_train.shape
        m = x_test.shape[0]
        eps = 1e-8
        col_mean = X_train.mean(dim=0)
        col_std = X_train.std(dim=0, unbiased=False).clamp(min=eps)
        col_emb = self.col_encoder(col_mean, col_std)
        if self.cfg.norm_feat:
            X_norm = (X_train - col_mean) / col_std
            xq_norm = (x_test - col_mean) / col_std
        else:
            X_norm = X_train
            xq_norm = x_test

        X_exp = X_norm.unsqueeze(0).expand(m, -1, -1)
        q_exp = xq_norm.unsqueeze(1).expand(-1, n, -1)
        H = self.class_cell_encoder(torch.stack([X_exp, q_exp], dim=-1))
        H = H + col_emb.unsqueeze(0).unsqueeze(0)
        if self.cfg.use_class_label_embeddings:
            support_labels = self.class_label_encoder.forward_train_labels(y_train)
            query_mask = self.class_label_encoder.forward_query_mask(
                m, device=X_train.device, dtype=H.dtype
            )
            H = H + support_labels.unsqueeze(0).unsqueeze(2)
            H = H + query_mask.unsqueeze(1).unsqueeze(2)

        for block in self.blocks:
            H = block(H)

        if self.icl_sample_pool is None:
            h_feat = H.mean(dim=1)
        else:
            H_mp = H.permute(0, 2, 1, 3).reshape(m * p, n, self.cfg.d_phi)
            h_feat = self.icl_sample_pool(H_mp).view(m, p, -1)
        h_glob = self._pool_sequence(h_feat, self.icl_feature_pool)

        class_feat_stats, class_global_stats, class_present_mask = (
            self.class_stat_extractor(X_norm, y_train, K)
        )
        if self.cfg.use_classification_stats:
            z_class = self.class_stat_encoder(
                class_feat_stats,
                class_global_stats,
                class_present_mask,
            )
        else:
            z_class = torch.zeros(
                self.cfg.class_stat_dim, device=X_train.device, dtype=X_norm.dtype
            )
        if self.cfg.use_class_stat_fusion:
            h_fused = self.class_fusion(
                h_glob, z_class.unsqueeze(0).expand(m, -1)
            )
        else:
            h_fused = h_glob

        h_global = h_fused.mean(dim=0)
        class_embeddings = self.class_label_encoder.class_embeddings(
            K, dtype=h_global.dtype
        )
        if self.cfg.use_class_coefficient_head:
            W_hat_norm = self.class_coeff_head(
                class_feat_stats,
                h_global,
                class_embeddings,
                class_present_mask,
            )
        else:
            W_hat_norm = torch.zeros(
                p, K, device=X_train.device, dtype=X_norm.dtype
            )
        if self.cfg.use_class_bias_head:
            b_hat = self.class_bias_head(
                class_feat_stats,
                class_global_stats,
                h_global,
                class_embeddings,
                class_present_mask,
            )
        else:
            b_hat = torch.zeros(K, device=X_train.device, dtype=X_norm.dtype)

        logits_linear = xq_norm @ W_hat_norm + b_hat
        logits_neural = self.class_neural_head(h_fused)[:, :K]
        if self.cfg.use_class_residual_head:
            logits = logits_linear + logits_neural
        elif self.cfg.use_class_coefficient_head:
            logits = logits_linear
        else:
            logits = logits_neural

        # --- Mixed-categorical classification path ---
        if (self.cfg.use_categorical_features
                and self.cat_cls_head is not None
                and kwargs.get("X_cat_train") is not None):
            X_cat_train_cls = kwargs["X_cat_train"]
            X_cat_test_cls = kwargs.get("X_cat_test")
            cat_cards_cls = kwargs.get("categorical_cardinalities")
            if X_cat_test_cls is not None and cat_cards_cls is not None:
                ctx_cat_tokens = self.cat_token_encoder(X_cat_train_cls, cat_cards_cls)
                feat_cat_stats, global_cat_stats = self.cat_stat_extractor(
                    X_cat_train_cls, y_train.float(), cat_cards_cls,
                    "inductive_classification", num_classes=K,
                )
                z_cat_stat = self.cat_stat_encoder(
                    feat_cat_stats, global_cat_stats, cat_cards_cls
                )
                query_cat_tokens = self.cat_token_encoder(X_cat_test_cls, cat_cards_cls)
                cat_logit_shift = self.cat_cls_head(
                    query_cat_tokens, z_cat_stat, h_global, num_classes=K,
                )
                logits = logits + cat_logit_shift

        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        aux = {
            "logits_linear": logits_linear,
            "logits_neural": logits_neural,
            "z_class": z_class,
            "h_global": h_global,
            "col_mean": col_mean,
            "col_std": col_std,
        }
        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "W_hat_norm": W_hat_norm,
            "b_hat": b_hat,
            "class_feat_stats": class_feat_stats,
            "class_global_stats": class_global_stats,
            "class_present_mask": class_present_mask,
            "aux": aux if return_aux else {},
        }

    # ------------------------------------------------------------------
    # MODEL5-LBACNP helpers
    # ------------------------------------------------------------------

    def _lbacnp_feature_tokens(
        self,
        X_norm:   torch.Tensor,   # (n_or_m, p) — already normalised
        X_ref:    torch.Tensor,   # (n, p) context reference for PLE boundaries
        col_mean: torch.Tensor,
        col_std:  torch.Tensor,
    ):
        """Return per-feature token matrix using PLE or ColumnEncoder fallback.

        Returns (batch, p, d_phi).
        """
        n_or_m, p = X_norm.shape
        if self.cfg.use_numeric_ple and self.ple_encoder is not None:
            # PLE encoder needs context reference to fit boundaries
            # We pass X_ref (context) and X_norm (target rows) to _encode
            T = self.cfg.ple_num_bins
            pts = torch.linspace(0.0, 1.0, T + 1, device=X_ref.device, dtype=torch.float32)
            boundaries = torch.quantile(
                X_ref.float(), pts, dim=0
            ).T.to(X_ref.dtype)
            boundaries = boundaries.cummax(dim=-1).values
            ple = NumericPLEEncoder._encode(X_norm, boundaries)   # (n_or_m, p, T)
            tokens = self.ple_encoder.drop(
                self.ple_encoder.norm(self.ple_encoder.projection(ple))
            )  # (n_or_m, p, d_phi)
        else:
            col_emb = self.col_encoder(col_mean, col_std)          # (p, d_phi)
            tokens = col_emb.unsqueeze(0).expand(n_or_m, -1, -1)  # (n_or_m, p, d_phi)
        return tokens

    def forward_regression_lbacnp(
        self,
        X_train:      torch.Tensor,
        y_train:      torch.Tensor,
        x_test:       torch.Tensor,
        return_debug: bool = False,
        beta:         torch.Tensor | None = None,
        **kwargs,
    ):
        """
        MODEL5-LBACNP regression forward.

        Linear tasks  (use_coefficient_head=True):
            y_linear = X_query @ beta_hat  (MODEL4 coefficient path)
            y_hat    = y_linear + gate(qemb) * mu_delta(qemb)
            Returns  {"mu": y_hat, "log_var": log_var_delta, ...}

        Nonlinear tasks (use_coefficient_head=False):
            mu, log_var = RegressionDistributionHead(LBACNP(context, query))
            Returns {"mu": mu, "log_var": log_var}
        """
        n, p = X_train.shape
        single = x_test.ndim == 1
        if single:
            x_test = x_test.unsqueeze(0)
        m   = x_test.shape[0]
        EPS = 1e-8

        # -- Normalisation --
        col_mean = X_train.mean(dim=0)
        col_std  = X_train.std(dim=0, unbiased=False).clamp(min=EPS)
        if self.cfg.norm_feat:
            X_norm  = (X_train - col_mean) / col_std
            xq_norm = (x_test  - col_mean) / col_std
        else:
            X_norm  = X_train
            xq_norm = x_test

        y_mean = y_std = None
        if self.cfg.norm_target:
            y_mean = y_train.mean()
            y_std  = y_train.std(unbiased=False).clamp(min=EPS)
            y_norm = (y_train - y_mean) / y_std
        else:
            y_norm = y_train

        # -- Feature tokens --
        ctx_tokens = self._lbacnp_feature_tokens(X_norm,  X_norm, col_mean, col_std)
        qry_tokens = self._lbacnp_feature_tokens(xq_norm, X_norm, col_mean, col_std)

        # -- Row embeddings --
        ctx_rows = self.lbacnp_context_encoder(ctx_tokens, y_norm)   # (n, d)
        qry_rows = self.lbacnp_query_encoder(qry_tokens)              # (m, d)

        # -- Latent bottleneck --
        qemb = self.lbacnp(ctx_rows, qry_rows, self.cfg.lbacnp_latent_frac)  # (m, d)

        # -- Distribution head --
        mu_norm, log_var = self.lbacnp_reg_head(qemb)   # (m,), (m,)

        if self.cfg.use_coefficient_head and self.coeff_head is not None:
            # Linear task: MODEL4 coefficient path is primary, LBACNP is residual
            col_emb = self.col_encoder(col_mean, col_std)
            X_exp   = X_norm.unsqueeze(0).expand(m, -1, -1)
            y_exp   = y_norm.view(1, n, 1).expand(m, n, p)
            q_exp   = xq_norm.unsqueeze(1).expand(-1, n, -1)
            _cell_in = torch.stack([X_exp, y_exp, q_exp], dim=-1)
            _use_ckpt = self.blocks[0].use_checkpoint if self.blocks else False
            if _use_ckpt and torch.is_grad_enabled():
                H = _checkpoint(self.cell_encoder, _cell_in, use_reentrant=False)
            else:
                H = self.cell_encoder(_cell_in)
            H = H + col_emb.unsqueeze(0).unsqueeze(0)
            for block in self.blocks:
                H = block(H)
            if self.icl_sample_pool is None:
                h_feat = H.mean(dim=1)
            else:
                H_mp   = H.permute(0, 2, 1, 3).reshape(m * p, n, self.cfg.d_phi)
                h_feat = self.icl_sample_pool(H_mp).view(m, p, -1)
            h_glob = self._pool_sequence(h_feat, self.icl_feature_pool)

            with torch.amp.autocast(device_type=X_train.device.type, enabled=False):
                feat_stats_f32, global_stats_f32 = self.stat_extractor(
                    X_norm.float(), y_norm.float()
                )
            feat_stats   = feat_stats_f32.to(X_norm.dtype)
            global_stats = global_stats_f32.to(X_norm.dtype)
            z_linear     = self.stat_encoder(feat_stats, global_stats)

            z_lin_exp    = z_linear.unsqueeze(0).expand(m, -1)
            h_fused      = self.fusion(h_glob, z_lin_exp)
            h_global_mean = h_fused.mean(0)
            beta_hat_norm = self.coeff_head(feat_stats, h_global_mean)
            y_coeff_norm  = xq_norm @ beta_hat_norm                   # (m,)

            gate = (
                self.lbacnp_residual_gate(qemb)
                if self.lbacnp_residual_gate is not None
                else torch.ones(m, device=qemb.device, dtype=qemb.dtype)
            )
            pred_norm = y_coeff_norm + gate * mu_norm
        else:
            # Nonlinear task: LBACNP is primary
            pred_norm = mu_norm
            gate      = None
            beta_hat_norm = y_coeff_norm = None

        # -- Denormalise --
        if self.cfg.norm_target:
            mu = pred_norm * y_std + y_mean
            log_var_scaled = log_var + 2.0 * torch.log(y_std.clamp(min=EPS))
        else:
            mu = pred_norm
            log_var_scaled = log_var

        if single:
            mu             = mu.squeeze(0)
            log_var_scaled = log_var_scaled.squeeze(0)

        result = {"mu": mu, "log_var": log_var_scaled}
        if return_debug:
            result["gate"]          = gate
            result["y_coeff_norm"]  = y_coeff_norm
            result["beta_hat_norm"] = beta_hat_norm
        return result

    def forward_classification_lbacnp(
        self,
        X_train:      torch.Tensor,
        y_train:      torch.Tensor,
        x_test:       torch.Tensor,
        num_classes:  int,
        y_test:       torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        return_aux:   bool = False,
        **kwargs,
    ) -> dict:
        """
        MODEL5-LBACNP classification forward.

        Linear tasks  (use_class_coefficient_head=True):
            logits = logits_linear + gate * delta_logits_lbacnp

        Nonlinear tasks (use_class_coefficient_head=False):
            logits = ClassificationLBACNPHead(QEMB)
        """
        if x_test.ndim == 1:
            x_test = x_test.unsqueeze(0)
        EPS = 1e-8
        n, p = X_train.shape
        m    = x_test.shape[0]
        K    = int(num_classes)

        col_mean = X_train.mean(dim=0)
        col_std  = X_train.std(dim=0, unbiased=False).clamp(min=EPS)
        if self.cfg.norm_feat:
            X_norm  = (X_train - col_mean) / col_std
            xq_norm = (x_test  - col_mean) / col_std
        else:
            X_norm  = X_train
            xq_norm = x_test

        # Run MODEL4 backbone to get logits_linear, h_fused, class stats
        m4_out = self.forward_classification(
            X_train, y_train, x_test, num_classes=K,
            y_test=y_test, class_weight=class_weight, return_aux=True, **kwargs,
        )

        # Build LBACNP feature tokens
        ctx_tokens = self._lbacnp_feature_tokens(X_norm,  X_norm, col_mean, col_std)
        qry_tokens = self._lbacnp_feature_tokens(xq_norm, X_norm, col_mean, col_std)

        # Normalise y as a float scalar (class label / (K-1)) for the context encoder
        y_float = y_train.float() / max(K - 1, 1)

        ctx_rows = self.lbacnp_context_encoder(ctx_tokens, y_float)   # (n, d)
        qry_rows = self.lbacnp_query_encoder(qry_tokens)               # (m, d)

        qemb = self.lbacnp(ctx_rows, qry_rows, self.cfg.lbacnp_latent_frac)  # (m, d)

        delta_logits = self.lbacnp_cls_head(qemb, K)   # (m, K)

        if self.cfg.use_class_coefficient_head:
            # Linear classification: LBACNP is residual correction
            gate = (
                self.lbacnp_residual_gate(qemb).unsqueeze(-1)
                if self.lbacnp_residual_gate is not None
                else torch.ones(m, 1, device=qemb.device, dtype=qemb.dtype)
            )
            logits = m4_out["logits"] + gate * delta_logits
        else:
            # Nonlinear classification: LBACNP is primary
            logits = delta_logits

        probs = torch.softmax(logits, dim=-1)
        pred  = logits.argmax(dim=-1)

        return {
            "logits":               logits,
            "probs":                probs,
            "pred":                 pred,
            "W_hat_norm":           m4_out.get("W_hat_norm"),
            "b_hat":                m4_out.get("b_hat"),
            "class_feat_stats":     m4_out.get("class_feat_stats"),
            "class_global_stats":   m4_out.get("class_global_stats"),
            "class_present_mask":   m4_out.get("class_present_mask"),
            "aux":                  m4_out.get("aux", {}) if return_aux else {},
        }


# ---------------------------------------------------------------------------
# Mixed-categorical modules (Step 5 additions)
# ---------------------------------------------------------------------------

_CAT_CARDINALITY_BINS = [2, 5, 10, 20, 50]   # 6 bins: <=2, <=5, <=10, <=20, <=50, >50


def _cardinality_bucket(cardinality: int) -> int:
    """Map a cardinality integer to a bin index 0..5."""
    for idx, threshold in enumerate(_CAT_CARDINALITY_BINS):
        if cardinality <= threshold:
            return idx
    return len(_CAT_CARDINALITY_BINS)   # bin 5: >50


class CategoricalTokenEncoder(nn.Module):
    """Entity Embedding-based categorical tokenizer.

    Produces a d_phi embedding for each (sample, categorical feature) cell.

    Components per cell:
      entity_embed(category_id)    → (cat_embed_dim,)
      feat_id_embed(feature_index) → (cat_feat_id_embed_dim,)   # positional
      cardinality_embed(bucket)    → (cat_cardinality_embed_dim,)
      concat → Linear → d_phi

    Special tokens (PAD=0, MISSING=1, UNKNOWN=2) have their own embedding rows.
    Vocabulary: fixed at cat_max_vocab_size (53).
    Cardinality: bucketed into 6 bins: [<=2, <=5, <=10, <=20, <=50, >50].
    """

    _N_CARDINALITY_BINS = len(_CAT_CARDINALITY_BINS) + 1  # 6

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        vocab_size = cfg.cat_max_vocab_size       # 53
        n_bins = self._N_CARDINALITY_BINS         # 6
        self.entity_embed = nn.Embedding(vocab_size, cfg.cat_embed_dim, padding_idx=0)
        self.feat_id_embed = nn.Embedding(cfg.cat_max_p_cat, cfg.cat_feat_id_embed_dim)
        self.card_embed = nn.Embedding(n_bins, cfg.cat_cardinality_embed_dim)
        in_dim = cfg.cat_embed_dim + cfg.cat_feat_id_embed_dim + cfg.cat_cardinality_embed_dim
        self.proj = nn.Linear(in_dim, cfg.d_phi)
        self.cfg = cfg

    def forward(
        self,
        X_cat: torch.LongTensor,          # (n, p_cat)
        cardinalities: torch.LongTensor,  # (p_cat,)
    ) -> torch.Tensor:                    # (n, p_cat, d_phi)
        n, p_cat = X_cat.shape
        # Entity embeddings — clamp IDs to vocab range
        ids = X_cat.clamp(0, self.cfg.cat_max_vocab_size - 1)   # (n, p_cat)
        entity = self.entity_embed(ids)                          # (n, p_cat, cat_embed_dim)

        # Feature positional embeddings
        feat_idx = torch.arange(p_cat, device=X_cat.device).clamp(0, self.cfg.cat_max_p_cat - 1)
        feat = self.feat_id_embed(feat_idx)                      # (p_cat, cat_feat_id_embed_dim)
        feat = feat.unsqueeze(0).expand(n, -1, -1)               # (n, p_cat, cat_feat_id_embed_dim)

        # Cardinality bucket embeddings
        buckets = torch.tensor(
            [_cardinality_bucket(int(c)) for c in cardinalities.tolist()],
            device=X_cat.device,
            dtype=torch.long,
        )  # (p_cat,)
        card = self.card_embed(buckets)                          # (p_cat, cat_cardinality_embed_dim)
        card = card.unsqueeze(0).expand(n, -1, -1)               # (n, p_cat, cat_cardinality_embed_dim)

        combined = torch.cat([entity, feat, card], dim=-1)       # (n, p_cat, in_dim)
        return self.proj(combined)                               # (n, p_cat, d_phi)


class CategoricalStatisticExtractor(nn.Module):
    """Parameter-free context-only categorical statistics.

    Computes statistics using context rows (X_cat_context, y_context) ONLY.
    Never uses test labels.

    Regression statistics per (feature j, category c):
      count, frequency, mean_y_by_category, shrunk_effect, missing_rate

    Classification statistics per (feature j, category c):
      count, frequency, class_counts (K), shrunk_class_log_odds, missing_rate

    Returns feat_cat_stats (p_cat, max_card, n_stat_features) and
            global_cat_stats (n_global_stats,).
    """

    N_REG_STAT_FEATURES = 5
    N_CLS_STAT_BASE = 5   # count, freq, mean_y, shrunk_effect, missing_rate

    def __init__(self):
        super().__init__()
        # No parameters — statistics only

    @staticmethod
    def _shrink(counts: torch.Tensor, values: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        """Count-based shrinkage toward zero: shrunk = counts / (counts + alpha) * values."""
        return counts / (counts + alpha) * values

    @torch.no_grad()
    def forward(
        self,
        X_cat_context: torch.LongTensor,    # (n_ctx, p_cat)
        y_context: torch.Tensor,             # (n_ctx,) or (n_ctx, K)
        cardinalities: torch.LongTensor,     # (p_cat,)
        task_objective: str,
        num_classes: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from constants import ENTITY_EMBED_MISSING_ID
        device = X_cat_context.device
        original_dtype = y_context.dtype
        X = X_cat_context.long()
        n_ctx, p_cat = X.shape
        max_card = int(cardinalities.max().item()) if p_cat > 0 else 1
        is_regression = "regression" in task_objective or task_objective == "inductive_regression"

        if is_regression:
            n_stats = self.N_REG_STAT_FEATURES
        else:
            n_stats = self.N_REG_STAT_FEATURES  # same shape, different semantics

        feat_cat_stats = torch.zeros(
            p_cat, max_card, n_stats, device=device, dtype=torch.float32
        )
        global_parts = []

        from constants import ENTITY_EMBED_FIRST_REAL_ID
        y_f = y_context.float()
        if is_regression:
            y_mean_global = y_f.mean() if n_ctx > 0 else torch.zeros(1, device=device)

        total_missing = torch.zeros(p_cat, device=device, dtype=torch.float32)

        for j in range(p_cat):
            col = X[:, j]
            card_j = int(cardinalities[j].item())
            missing_mask = (col == ENTITY_EMBED_MISSING_ID)
            missing_count = float(missing_mask.sum().item())
            missing_rate = missing_count / max(n_ctx, 1)
            total_missing[j] = missing_rate
            valid_mask = ~missing_mask
            col_valid = col[valid_mask]
            real_ids = (col_valid - ENTITY_EMBED_FIRST_REAL_ID).clamp(0, card_j - 1)

            for c in range(min(card_j, max_card)):
                cat_mask = (real_ids == c)
                count = float(cat_mask.sum().item())
                freq = count / max(n_ctx, 1)
                if is_regression:
                    if count > 0:
                        y_valid = y_f[valid_mask]
                        mean_y = float(y_valid[cat_mask].mean().item())
                        shrunk = float(count / (count + 10.0) * mean_y)
                    else:
                        mean_y = 0.0
                        shrunk = 0.0
                    feat_cat_stats[j, c, 0] = count
                    feat_cat_stats[j, c, 1] = freq
                    feat_cat_stats[j, c, 2] = mean_y
                    feat_cat_stats[j, c, 3] = shrunk
                    feat_cat_stats[j, c, 4] = missing_rate
                else:
                    # Classification: use mean_y as class distribution proxy
                    if count > 0 and y_f.ndim == 1:
                        y_valid = y_f[valid_mask]
                        mean_y = float(y_valid[cat_mask].mean().item())
                        shrunk = float(count / (count + 10.0) * mean_y)
                    else:
                        mean_y = 0.0
                        shrunk = 0.0
                    feat_cat_stats[j, c, 0] = count
                    feat_cat_stats[j, c, 1] = freq
                    feat_cat_stats[j, c, 2] = mean_y
                    feat_cat_stats[j, c, 3] = shrunk
                    feat_cat_stats[j, c, 4] = missing_rate

        avg_missing = total_missing.mean()
        max_missing = total_missing.max() if p_cat > 0 else torch.zeros(1, device=device)
        global_cat_stats = torch.stack([
            torch.tensor(float(n_ctx), device=device, dtype=torch.float32),
            torch.tensor(float(p_cat), device=device, dtype=torch.float32),
            torch.tensor(float(max_card), device=device, dtype=torch.float32),
            avg_missing.float(),
            max_missing.float(),
        ])  # (5,)

        return feat_cat_stats.to(original_dtype), global_cat_stats.to(original_dtype)


N_CAT_FEAT_STATS = CategoricalStatisticExtractor.N_REG_STAT_FEATURES  # 5
N_CAT_GLOBAL_STATS = 5


class CategoricalStatisticEncoder(nn.Module):
    """Encode categorical statistics → fixed embedding via masked mean-pooling."""

    def __init__(self, cfg: ModelConfig, n_stat_features: int = N_CAT_FEAT_STATS):
        super().__init__()
        self.feat_proj = build_mlp(
            n_stat_features, cfg.cat_stat_dim, cfg.cat_stat_hidden_dim, cfg.dropout
        )
        self.global_proj = build_mlp(
            cfg.cat_stat_dim + N_CAT_GLOBAL_STATS, cfg.cat_stat_dim,
            cfg.cat_stat_hidden_dim, cfg.dropout
        )

    def forward(
        self,
        feat_cat_stats: torch.Tensor,     # (p_cat, max_cards, n_stat_features)
        global_cat_stats: torch.Tensor,   # (N_CAT_GLOBAL_STATS,)
        cardinalities: torch.LongTensor,  # (p_cat,) for masking
    ) -> torch.Tensor:                    # (cat_stat_dim,)
        p_cat, max_card, _ = feat_cat_stats.shape
        if p_cat == 0:
            return torch.zeros(
                self.global_proj.mlp[-1].out_features if hasattr(self.global_proj, "mlp")
                else global_cat_stats.shape[0],
                device=feat_cat_stats.device, dtype=feat_cat_stats.dtype
            )
        # Build validity mask (p_cat, max_card)
        card_range = torch.arange(max_card, device=cardinalities.device)
        mask = card_range.unsqueeze(0) < cardinalities.unsqueeze(1).float()   # (p_cat, max_card)
        encoded = self.feat_proj(feat_cat_stats)   # (p_cat, max_card, cat_stat_dim)
        denom = mask.float().sum().clamp(min=1.0)
        masked_sum = (encoded * mask.unsqueeze(-1).to(encoded.dtype)).sum(dim=(0, 1))
        pooled = masked_sum / denom                  # (cat_stat_dim,)
        combined = torch.cat([pooled, global_cat_stats], dim=0)
        return self.global_proj(combined)            # (cat_stat_dim,)


class RegressionCategoryEffectHead(nn.Module):
    """Predict scalar category effects for regression.

    Input per query row: cat_token_embed (m, p_cat, d_phi), z_cat_stat (cat_stat_dim,),
    h_global (neural_out_dim,).
    Output: (m,) summed over p_cat.
    """

    def __init__(self, cfg: ModelConfig, neural_out_dim: int):
        super().__init__()
        in_dim = cfg.d_phi + cfg.cat_stat_dim + neural_out_dim
        self.mlp = build_mlp(in_dim, 1, cfg.cat_head_hidden_dim, cfg.dropout)

    def forward(
        self,
        query_cat_tokens: torch.Tensor,   # (m, p_cat, d_phi)
        z_cat_stat: torch.Tensor,         # (cat_stat_dim,)
        h_global: torch.Tensor,           # (neural_out_dim,)
    ) -> torch.Tensor:                    # (m,)
        m, p_cat, d_phi = query_cat_tokens.shape
        z_exp = z_cat_stat.unsqueeze(0).unsqueeze(0).expand(m, p_cat, -1)
        h_exp = h_global.unsqueeze(0).unsqueeze(0).expand(m, p_cat, -1)
        combined = torch.cat([query_cat_tokens, z_exp, h_exp], dim=-1)  # (m, p_cat, in_dim)
        effects = self.mlp(combined).squeeze(-1)                          # (m, p_cat)
        return effects.sum(dim=1)                                         # (m,)


class ClassificationCategoryEffectHead(nn.Module):
    """Predict K-dimensional category logit shifts for classification.

    Returns: (m, K) summed over p_cat.
    Reference class column (col 0) is zeroed after prediction.
    """

    def __init__(self, cfg: ModelConfig, neural_out_dim: int, max_num_classes: int):
        super().__init__()
        in_dim = cfg.d_phi + cfg.cat_stat_dim + neural_out_dim
        self.mlp = build_mlp(in_dim, max_num_classes, cfg.cat_head_hidden_dim, cfg.dropout)
        self.max_num_classes = max_num_classes

    def forward(
        self,
        query_cat_tokens: torch.Tensor,   # (m, p_cat, d_phi)
        z_cat_stat: torch.Tensor,         # (cat_stat_dim,)
        h_global: torch.Tensor,           # (neural_out_dim,)
        num_classes: int,
    ) -> torch.Tensor:                    # (m, K)
        m, p_cat, d_phi = query_cat_tokens.shape
        K = int(num_classes)
        z_exp = z_cat_stat.unsqueeze(0).unsqueeze(0).expand(m, p_cat, -1)
        h_exp = h_global.unsqueeze(0).unsqueeze(0).expand(m, p_cat, -1)
        combined = torch.cat([query_cat_tokens, z_exp, h_exp], dim=-1)  # (m, p_cat, in_dim)
        raw = self.mlp(combined)[..., :K]    # (m, p_cat, K)
        raw[..., 0] = 0.0                   # zero out reference class
        return raw.sum(dim=1)               # (m, K)


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
        _use_ckpt = os.environ.get("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"
        self.blocks = nn.ModuleList([
            ExchangeableMatrixBlock(d_model, cfg.dropout, use_checkpoint=_use_ckpt)
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
