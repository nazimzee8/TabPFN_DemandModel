"""
model.py

DeepSetModel: a permutation-equivariant neural network for in-context
regression, operating over a training set (X_train, y_train) and one or
more test points x_test.
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
    d_phi:       int   = 128   # phi output dim; must be >= p
    d_rho:       int   = 256   # rho output dim; must be >= n
    pool:        str   = "pna"
    n_heads:     int   = 4     # attention heads in SAB / AttentionPool
    n_sab_feat:  int   = 1     # SAB layers at feature level (0 = linear equivariance)
    n_sab_samp:  int   = 1     # SAB layers at sample level (0 = linear equivariance)
    norm_feat:   bool  = True  # standardize X_train columns per-context
    norm_target: bool  = True  # standardize y_train per-context; denormalize output
    dropout:     float = 0.1
    feature_aggregation_order: str = "legacy_feature_pool_first"
    # "legacy_feature_pool_first" | "sample_evidence_first"
    # Missing field in old cfg dicts maps to "legacy_feature_pool_first" (default).

    # New fields — MarketAwareDeepSetModel only
    model_family:             str   = "deepset"  # "deepset" | "market_aware" | "market_exchangeable_icl" | "market_exchangeable_completion"
    d_sample:                 int   = 64         # phi_sample output dim (must be divisible by n_heads when sample attn used)
    n_sab_sample_per_feature: int   = 0          # SAB layers over n within each (q,j) slot — keep 0 (memory: see §Memory Guard)
    sample_pool:              str   = "attn"     # pooling over n: "attn" | "pna" | "mean"
    use_ridge_expert:         bool  = False      # enable RidgeExpert + gate
    ridge_lambda:             float = 1.0        # ridge regularisation λ (must be > 0)
    residual_scale_init:      float = 0.1        # init value of learned residual_scale scalar
    gate_hidden_dim:          int   = 64         # gate_head MLP hidden dim

    # MODEL3 selectors — default values preserve MODEL2 behavior
    model_arch_version:    str = "model2"               # "model2" | "model3"
    model3_design_pattern: str = "inductive_forecasting" # "inductive_forecasting" | "transductive_completion"

    def __post_init__(self):
        if self.pool not in VALID_POOLS:
            raise ValueError(f"Invalid pool '{self.pool}'. Valid: {sorted(VALID_POOLS)}")
        # n_heads divisibility only matters when attention is actually used
        uses_heads = (self.n_sab_feat > 0 or self.n_sab_samp > 0
                      or self.pool in ("attn", "multipool"))
        if uses_heads:
            if self.d_phi % self.n_heads != 0:
                raise ValueError(f"d_phi={self.d_phi} must be divisible by n_heads={self.n_heads}")
            if self.d_rho % self.n_heads != 0:
                raise ValueError(f"d_rho={self.d_rho} must be divisible by n_heads={self.n_heads}")
        _VALID_FAMILIES = frozenset({
            "deepset", "market_aware",
            "market_exchangeable_icl", "market_exchangeable_completion",
        })
        if self.model_family not in _VALID_FAMILIES:
            raise ValueError(
                f"Invalid model_family: {self.model_family!r}. Valid: {sorted(_VALID_FAMILIES)}"
            )
        if self.sample_pool not in POOL_SCALE:
            raise ValueError(f"Invalid sample_pool: {self.sample_pool!r}")
        if self.model_family == "market_aware" and self.n_sab_sample_per_feature > 0:
            if self.d_sample % self.n_heads != 0:
                raise ValueError(
                    f"d_sample={self.d_sample} must be divisible by n_heads={self.n_heads}"
                )
        if self.feature_aggregation_order not in (
            "legacy_feature_pool_first", "sample_evidence_first"
        ):
            raise ValueError(
                f"Invalid feature_aggregation_order: {self.feature_aggregation_order!r}"
            )
        # MODEL3 selector validation
        if self.model_arch_version not in ("model2", "model3"):
            raise ValueError(
                f"Invalid model_arch_version: {self.model_arch_version!r}. "
                "Valid: 'model2', 'model3'"
            )
        if self.model3_design_pattern not in ("inductive_forecasting", "transductive_completion"):
            raise ValueError(
                f"Invalid model3_design_pattern: {self.model3_design_pattern!r}. "
                "Valid: 'inductive_forecasting', 'transductive_completion'"
            )
        # MODEL3 families require model_arch_version="model3"
        _MODEL3_FAMILIES = frozenset({"market_exchangeable_icl", "market_exchangeable_completion"})
        if self.model_family in _MODEL3_FAMILIES and self.model_arch_version != "model3":
            raise ValueError(
                f"model_family={self.model_family!r} requires model_arch_version='model3', "
                f"got model_arch_version={self.model_arch_version!r}"
            )
        # MODEL3 family/pattern consistency
        if self.model_arch_version == "model3":
            if (self.model3_design_pattern == "inductive_forecasting"
                    and self.model_family != "market_exchangeable_icl"):
                raise ValueError(
                    f"model3_design_pattern='inductive_forecasting' requires "
                    f"model_family='market_exchangeable_icl', got {self.model_family!r}"
                )
            if (self.model3_design_pattern == "transductive_completion"
                    and self.model_family != "market_exchangeable_completion"):
                raise ValueError(
                    f"model3_design_pattern='transductive_completion' requires "
                    f"model_family='market_exchangeable_completion', got {self.model_family!r}"
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
# DeepSetModel
# ---------------------------------------------------------------------------

class DeepSetModel(nn.Module):
    """
    Permutation-equivariant deep-set model for in-context regression.

    Instantiate via ModelConfig for full control:
        cfg = ModelConfig(d_phi=128, d_rho=256, pool="pna", ...)
        model = DeepSetModel(cfg=cfg)

    Legacy flat-kwargs path (backward compat, no SAB, no normalization):
        model = DeepSetModel(d_phi=128, d_rho=256, pool="pna")
    """

    def __init__(self, cfg: ModelConfig = None, *,
                 d_phi=128, d_rho=256, pool="pna", dropout=0.1):
        super().__init__()
        if cfg is None:
            # Backward-compat path: replicates original behavior exactly
            cfg = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool, dropout=dropout,
                              n_sab_feat=0, n_sab_samp=0,
                              norm_feat=False, norm_target=False, n_heads=4)
        self.cfg = cfg

        self.phi = build_mlp(3, cfg.d_phi, cfg.d_phi, cfg.dropout)

        # Feature equivariance
        if cfg.n_sab_feat > 0:
            self.sab_feat = nn.Sequential(
                *[SAB(cfg.d_phi, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_sab_feat)])
        else:
            self.lambda_feat = nn.Parameter(torch.tensor(1.0))
            self.gamma_feat  = nn.Parameter(torch.tensor(0.1))

        self.feat_pool = SetPool(cfg.d_phi, cfg.pool, cfg.n_heads, cfg.dropout)
        rho_in = POOL_SCALE[cfg.pool] * cfg.d_phi

        self.rho = build_mlp(rho_in, cfg.d_rho, cfg.d_rho, cfg.dropout)

        # Sample equivariance
        if cfg.n_sab_samp > 0:
            self.sab_samp = nn.Sequential(
                *[SAB(cfg.d_rho, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_sab_samp)])
        else:
            self.lambda_samp = nn.Parameter(torch.tensor(1.0))
            self.gamma_samp  = nn.Parameter(torch.tensor(0.1))

        self.samp_pool = SetPool(cfg.d_rho, cfg.pool, cfg.n_heads, cfg.dropout)
        psi_in = POOL_SCALE[cfg.pool] * cfg.d_rho

        self.psi = build_mlp(psi_in, 1, cfg.d_rho // 2, cfg.dropout)

    @staticmethod
    def _pna_pool(x: torch.Tensor, dim: int) -> torch.Tensor:
        """Backward-compat static method. Concatenate [sum, mean, max, std] over `dim`."""
        s   = x.sum(dim=dim)
        mu  = x.mean(dim=dim)
        mx  = x.max(dim=dim).values
        std = x.std(dim=dim, unbiased=False)
        return torch.cat([s, mu, mx, std], dim=-1)

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor,
                x_test: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_train : (n, p) float tensor — training features
            y_train : (n,)   float tensor — training labels
            x_test  : (p,) or (m, p) float tensor — test feature vector(s)

        Returns:
            Scalar tensor if x_test is (p,); shape (m,) if x_test is (m, p).
        """
        n, p   = X_train.shape
        single = x_test.ndim == 1
        if single:
            x_test = x_test.unsqueeze(0)      # (1, p)
        m = x_test.shape[0]
        EPS = 1e-8

        # --- Feature normalization (per-context) ---
        if self.cfg.norm_feat:
            f_mean  = X_train.mean(dim=0)                              # (p,)
            f_std   = X_train.std(dim=0, unbiased=False).clamp(min=EPS)
            X_train = (X_train - f_mean) / f_std                      # (n, p)
            x_test  = (x_test  - f_mean) / f_std                      # (m, p)

        # --- Target normalization (per-context) ---
        y_mean = y_std = None
        if self.cfg.norm_target:
            y_mean  = y_train.mean()
            y_std   = y_train.std(unbiased=False).clamp(min=EPS)
            y_train = (y_train - y_mean) / y_std                      # (n,)

        # --- Step 1: Build inp and apply phi ---
        y_e  = y_train.view(1, n, 1).expand(m, n, p)
        X_e  = X_train.view(1, n, p).expand(m, n, p)
        xt_e = x_test.view(m, 1, p).expand(m, n, p)
        inp  = torch.stack([y_e, X_e, xt_e], dim=3)                  # (m, n, p, 3)
        h    = self.phi(inp)                                           # (m, n, p, d_phi)

        if self.cfg.feature_aggregation_order == "sample_evidence_first":
            # Step 3 (new): aggregate sample evidence per feature via simple mean,
            # keeping feature identity. No new modules — uses mean(dim=1 of n-axis).
            h_perm = h.permute(0, 2, 1, 3)           # (m, p, n, d_phi)
            h_ev   = h_perm.mean(dim=2)              # (m, p, d_phi) — feature identity preserved

            # Step 4 (new): Feature SAB/equivariance on collapsed evidence
            if self.cfg.n_sab_feat > 0:
                h_ev = self.sab_feat(h_ev)           # (m, p, d_phi)
            else:
                h_ev = (self.lambda_feat * h_ev
                        + self.gamma_feat * h_ev.mean(1, keepdim=True))  # (m, p, d_phi)

            # Step 5 (new): Feature pool — feat_pool treats p as set, d_phi as feature dim
            h_feat = self.feat_pool(h_ev)            # (m, rho_in)

            # Step 6 (new): rho (MLP, applies per-element, works on (m, rho_in))
            r = self.rho(h_feat)                     # (m, d_rho)

            # Step 7 (new): psi — samp_pool needs (batch, set, d); use n=1 to match psi_in
            r_pool = self.samp_pool(r.unsqueeze(1))  # (m, psi_in); pna: [sum,mean,max,std] of n=1
            raw = self.psi(r_pool).squeeze(-1)       # (m,)

        else:  # legacy_feature_pool_first — existing code unchanged
            # --- Step 2: Feature equivariance ---
            if self.cfg.n_sab_feat > 0:
                h_flat = h.contiguous().view(m * n, p, self.cfg.d_phi)
                h_flat = self.sab_feat(h_flat)                            # (m*n, p, d_phi)
                h      = h_flat.view(m, n, p, self.cfg.d_phi)
            else:
                mean_i = h.mean(dim=2, keepdim=True)
                h      = self.lambda_feat * h + self.gamma_feat * mean_i  # (m, n, p, d_phi)

            # --- Step 3: Feature pool ---
            h_flat = h.contiguous().view(m * n, p, self.cfg.d_phi)
            h_feat = self.feat_pool(h_flat)                               # (m*n, rho_in)
            rho_in = POOL_SCALE[self.cfg.pool] * self.cfg.d_phi
            h_feat = h_feat.view(m, n, rho_in)                           # (m, n, rho_in)

            # --- Step 4: rho per sample ---
            r = self.rho(h_feat)                                          # (m, n, d_rho)

            # --- Step 5: Sample equivariance ---
            if self.cfg.n_sab_samp > 0:
                r = self.sab_samp(r)                                      # (m, n, d_rho)
            else:
                mean_j = r.mean(dim=1, keepdim=True)
                r      = self.lambda_samp * r + self.gamma_samp * mean_j  # (m, n, d_rho)

            # --- Step 6: Sample pool ---
            r_pool = self.samp_pool(r)                                    # (m, psi_in)

            # --- Step 7: psi ---
            raw = self.psi(r_pool).squeeze(-1)                            # (m,)

        # --- Step 8: Denormalize ---
        y_hat = raw * y_std + y_mean if self.cfg.norm_target else raw

        return y_hat.squeeze(0) if single else y_hat


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
            (m,) float tensor — ridge predictions
        """
        n, p = X_norm.shape
        kw = dict(device=X_norm.device, dtype=X_norm.dtype)
        if n >= p:
            A    = X_norm.T @ X_norm + lam * torch.eye(p, **kw)
            beta = torch.linalg.solve(A, X_norm.T @ y_norm)   # (p,)
        else:
            K     = X_norm @ X_norm.T + lam * torch.eye(n, **kw)
            alpha = torch.linalg.solve(K, y_norm)              # (n,)
            beta  = X_norm.T @ alpha                           # (p,)
        return x_test_norm @ beta                              # (m,)


# ---------------------------------------------------------------------------
# MarketAwareDeepSetModel
# ---------------------------------------------------------------------------

class MarketAwareDeepSetModel(nn.Module):
    """
    Permutation-equivariant deep-set model that aggregates sample evidence
    per feature before pooling features, preserving feature identity for
    cross-feature interactions.

    Key difference from DeepSetModel: pooling order is reversed. Sample
    evidence is aggregated within each (query, feature) slot first, then
    features interact via SAB or linear equivariance.

    Instantiate via ModelConfig with model_family="market_aware":
        cfg = ModelConfig(model_family="market_aware", d_sample=64, ...)
        model = MarketAwareDeepSetModel(cfg=cfg)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # TOKEN_DIM = 6: (y, X_ij, xq_j, X_ij*xq_j, X_ij*y, |X_ij-xq_j|)
        self.phi_sample = build_mlp(6, cfg.d_sample, cfg.d_sample, cfg.dropout)

        # Optional SAB over n (n_sab_sample_per_feature > 0)
        if cfg.n_sab_sample_per_feature > 0:
            self.sab_sample = nn.Sequential(*[
                SAB(cfg.d_sample, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_sab_sample_per_feature)
            ])

        # Pool over n
        self.sample_pool_layer = SetPool(cfg.d_sample, cfg.sample_pool, cfg.n_heads, cfg.dropout)
        d_feat = POOL_SCALE[cfg.sample_pool] * cfg.d_sample
        self._d_feat = d_feat

        # Feature SAB over p
        if cfg.n_sab_feat > 0:
            self.sab_feat = nn.Sequential(*[
                SAB(d_feat, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_sab_feat)
            ])
        else:
            self.lambda_feat = nn.Parameter(torch.tensor(1.0))
            self.gamma_feat  = nn.Parameter(torch.tensor(0.1))

        # Prediction head
        self.beta_head         = build_mlp(d_feat, 1, d_feat // 2, cfg.dropout)
        self.residual_scale    = nn.Parameter(torch.tensor(cfg.residual_scale_init))
        self.feat_summary_pool = SetPool(d_feat, "pna", cfg.n_heads, cfg.dropout)
        self.residual_head     = build_mlp(POOL_SCALE["pna"] * d_feat, 1,
                                           POOL_SCALE["pna"] * d_feat // 2, cfg.dropout)
        self.gate_head         = build_mlp(d_feat, 1, cfg.gate_hidden_dim, cfg.dropout)

        # Ridge expert
        if cfg.use_ridge_expert:
            self._ridge = RidgeExpert()

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor,
                x_test: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_train : (n, p) float tensor — training features
            y_train : (n,)   float tensor — training labels
            x_test  : (p,) or (m, p) float tensor — test feature vector(s)

        Returns:
            Scalar tensor if x_test is (p,); shape (m,) if x_test is (m, p).
        """
        n, p   = X_train.shape
        single = x_test.ndim == 1
        if single:
            x_test = x_test.unsqueeze(0)   # (1, p)
        m = x_test.shape[0]
        EPS = 1e-8

        # --- Step 1: Normalize (identical to DeepSetModel) ---
        if self.cfg.norm_feat:
            f_mean  = X_train.mean(dim=0)
            f_std   = X_train.std(dim=0, unbiased=False).clamp(min=EPS)
            X_norm  = (X_train - f_mean) / f_std
            xq_norm = (x_test  - f_mean) / f_std
        else:
            X_norm  = X_train
            xq_norm = x_test

        y_mean = y_std = None
        if self.cfg.norm_target:
            y_mean  = y_train.mean()
            y_std   = y_train.std(unbiased=False).clamp(min=EPS)
            y_norm  = (y_train - y_mean) / y_std
        else:
            y_norm  = y_train

        # --- Step 2: Build 6-feature tokens per (query, feature, sample) ---
        # y_e:  (m, p, n)
        y_e  = y_norm.view(1, 1, n).expand(m, p, n)
        # Xi_e: (m, p, n) — X_norm.T is (p, n)
        Xi_e = X_norm.T.unsqueeze(0).expand(m, p, n)
        # xq_e: (m, p, n) — x_test feature j broadcast over samples
        xq_e = xq_norm.unsqueeze(2).expand(m, p, n)

        tokens = torch.stack([
            y_e,
            Xi_e,
            xq_e,
            Xi_e * xq_e,
            Xi_e * y_e,
            (Xi_e - xq_e).abs(),
        ], dim=3)                                    # (m, p, n, 6)
        flat = tokens.view(m * p, n, 6)              # (m*p, n, 6)

        # --- Step 3: Sample evidence per feature ---
        h  = self.phi_sample(flat)                   # (m*p, n, d_sample)
        if self.cfg.n_sab_sample_per_feature > 0:
            h = self.sab_sample(h)                   # (m*p, n, d_sample)
        ev = self.sample_pool_layer(h)               # (m*p, d_feat)
        ev = ev.view(m, p, self._d_feat)             # (m, p, d_feat)  ← feature identity preserved

        # --- Step 4: Cross-feature interaction ---
        if self.cfg.n_sab_feat > 0:
            ctx = self.sab_feat(ev)                  # (m, p, d_feat)
        else:
            ctx = self.lambda_feat * ev + self.gamma_feat * ev.mean(1, keepdim=True)  # (m, p, d_feat)

        # --- Step 5: Neural prediction ---
        beta_like = self.beta_head(ctx).squeeze(-1)  # (m, p)
        lin_pred  = (beta_like * xq_norm).sum(dim=1) # (m,)
        summary   = self.feat_summary_pool(ctx)       # (m, 4*d_feat)
        resid     = self.residual_head(summary).squeeze(-1)  # (m,)
        neural    = lin_pred + self.residual_scale * resid   # (m,)

        # --- Step 6: Ridge expert (optional) ---
        if self.cfg.use_ridge_expert:
            ridge = self._ridge.predict(X_norm, y_norm, xq_norm, self.cfg.ridge_lambda)  # (m,)

        # --- Step 7: Gate + combine ---
        ctx_mean = ctx.mean(dim=1)                   # (m, d_feat)
        gate     = torch.sigmoid(self.gate_head(ctx_mean).squeeze(-1))  # (m,)
        if self.cfg.use_ridge_expert:
            pred_norm = ridge + gate * neural
        else:
            pred_norm = neural

        # --- Step 8: Denormalize (identical to DeepSetModel) ---
        y_hat = pred_norm * y_std + y_mean if self.cfg.norm_target else pred_norm

        return y_hat.squeeze(0) if single else y_hat


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
# MarketExchangeableICLModel — MODEL3 inductive forecasting
# ---------------------------------------------------------------------------

class MarketExchangeableICLModel(nn.Module):
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
      cfg.model3_design_pattern    == "inductive_forecasting"

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
                f"MarketExchangeableICLModel requires model_family='market_exchangeable_icl', "
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
# MarketExchangeableCompletionModel — MODEL3 transductive completion
# ---------------------------------------------------------------------------

class MarketExchangeableCompletionModel(nn.Module):
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
      cfg.model3_design_pattern    == "transductive_completion"

    Shared config fields used:
      cfg.d_phi     — channel dimension d_model
      cfg.n_sab_feat — number of ExchangeableMatrixBlocks (>=1)
      cfg.dropout   — dropout rate
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.model_family != "market_exchangeable_completion":
            raise ValueError(
                f"MarketExchangeableCompletionModel requires "
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
    family = getattr(cfg, "model_family", "deepset")
    if family == "deepset":
        return DeepSetModel(cfg=cfg)
    if family == "market_aware":
        return MarketAwareDeepSetModel(cfg=cfg)
    if family == "market_exchangeable_icl":
        return MarketExchangeableICLModel(cfg=cfg)
    if family == "market_exchangeable_completion":
        return MarketExchangeableCompletionModel(cfg=cfg)
    raise ValueError(f"Unknown model_family: {family!r}")
