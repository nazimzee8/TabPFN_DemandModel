"""
model.py

DeepSetModel: a permutation-equivariant neural network for in-context
regression, operating over a training set (X_train, y_train) and a
single test point x_test.
"""

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
# DeepSetModel
# ---------------------------------------------------------------------------

class DeepSetModel(nn.Module):
    """
    Permutation-equivariant deep-set model for in-context regression.

    Args:
        d_phi   : hidden/output dimension of the phi MLP (default 64)
        d_rho   : hidden/output dimension of the rho MLP (default 64)
        pool    : pooling strategy — only 'mean' is implemented (default 'mean')
        dropout : dropout probability applied inside each MLP (default 0.1)
    """

    def __init__(self, d_phi: int = 64, d_rho: int = 64,
                 pool: str = "mean", dropout: float = 0.1):
        super().__init__()

        if pool != "mean":
            raise ValueError(f"Unsupported pool mode '{pool}'. Only 'mean' is supported.")

        self.d_phi = d_phi
        self.d_rho = d_rho
        self.pool  = pool

        # phi: R^3 -> R^d_phi  (one triple per training-sample x feature pair)
        self.phi = build_mlp(in_dim=3, out_dim=d_phi, hidden_dim=d_phi, dropout=dropout)

        # rho: R^d_phi -> R^d_rho  (one vector per training sample)
        self.rho = build_mlp(in_dim=d_phi, out_dim=d_rho, hidden_dim=d_rho, dropout=dropout)

        # psi: R^d_rho -> R  (final scalar prediction)
        self.psi = build_mlp(in_dim=d_rho, out_dim=1, hidden_dim=d_rho // 2, dropout=dropout)

        # Learnable equivariant-layer scalars
        # Feature-level equivariance (step 2)
        self.lambda_feat = nn.Parameter(torch.tensor(1.0))
        self.gamma_feat  = nn.Parameter(torch.tensor(0.1))
        # Sample-level equivariance (step 5)
        self.lambda_samp = nn.Parameter(torch.tensor(1.0))
        self.gamma_samp  = nn.Parameter(torch.tensor(0.1))

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor,
                x_test: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_train : (n, p) float tensor — training features
            y_train : (n,)   float tensor — training labels
            x_test  : (p,)   float tensor — single test feature vector

        Returns:
            Scalar tensor — predicted noiseless linear response at x_test.
        """
        n, p = X_train.shape

        # ------------------------------------------------------------------
        # Step 1 — Build (n, p, 3) input tensor and apply phi
        # ------------------------------------------------------------------
        y_expand      = y_train.unsqueeze(1).expand(n, p)        # (n, p)
        x_test_expand = x_test.unsqueeze(0).expand(n, p)         # (n, p)
        inp = torch.stack([y_expand, X_train, x_test_expand], dim=2)  # (n, p, 3)

        h = self.phi(inp)                                         # (n, p, d_phi)

        # ------------------------------------------------------------------
        # Step 2 — Equivariant feature layer (across i=features, per j=sample)
        # ------------------------------------------------------------------
        mean_i  = h.mean(dim=1, keepdim=True)                    # (n, 1, d_phi)
        h_prime = self.lambda_feat * h + self.gamma_feat * mean_i  # (n, p, d_phi)

        # ------------------------------------------------------------------
        # Step 3 — Feature pooling (mean over i=features)
        # ------------------------------------------------------------------
        g = h_prime.mean(dim=1)                                   # (n, d_phi)

        # ------------------------------------------------------------------
        # Step 4 — Apply rho per sample
        # ------------------------------------------------------------------
        r = self.rho(g)                                           # (n, d_rho)

        # ------------------------------------------------------------------
        # Step 5 — Equivariant sample layer (across j=samples)
        # ------------------------------------------------------------------
        mean_j  = r.mean(dim=0, keepdim=True)                    # (1, d_rho)
        r_prime = self.lambda_samp * r + self.gamma_samp * mean_j  # (n, d_rho)

        # ------------------------------------------------------------------
        # Step 6 — Sample pooling (mean over j=samples)
        # ------------------------------------------------------------------
        z = r_prime.mean(dim=0)                                   # (d_rho,)

        # ------------------------------------------------------------------
        # Step 7 — psi: produce scalar prediction
        # ------------------------------------------------------------------
        y_hat = self.psi(z)                                       # (1,)
        return y_hat.squeeze()                                    # scalar
