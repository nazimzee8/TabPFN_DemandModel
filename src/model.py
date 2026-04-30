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
