"""
tolerance_policy.py

Centrally owned tolerance thresholds for permutation invariance and
equivariance checks.  Every comparison in permutation_contracts.py and
sanity_checks.py should resolve its tolerances through this module.

Tolerances are keyed by:
  - dtype        (float32, float64, bfloat16, float16)
  - device_type  ("cpu", "cuda")
  - inference    ("deterministic", "stochastic")   # stochastic = MC dropout / ensembles
  - output_type  ("scalar_prediction", "logits", "probabilities",
                  "coefficients", "bias", "embedding")
  - model_path   ("regression", "classification")

The module exposes a single entry point: ``get_tolerance(**kw) -> Tolerance``.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

# ── version for provenance / artifact stamping ──────────────────────────
TOLERANCE_POLICY_VERSION = "1.0.0"


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """Absolute and relative tolerance pair with metadata."""
    atol: float
    rtol: float
    note: str = ""

    def allclose(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return bool(torch.allclose(a, b, atol=self.atol, rtol=self.rtol))

    def max_abs_delta(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item() if a.numel() > 0 else 0.0

    def mean_abs_delta(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().mean().item() if a.numel() > 0 else 0.0

    def max_rel_delta(self, a: torch.Tensor, b: torch.Tensor) -> float:
        denom = a.abs().clamp(min=1e-12)
        return ((a - b).abs() / denom).max().item() if a.numel() > 0 else 0.0


# ── base tolerance tables ───────────────────────────────────────────────

# dtype → (atol, rtol)
_DTYPE_BASE: dict[torch.dtype, tuple[float, float]] = {
    torch.float64:  (1e-10, 1e-8),
    torch.float32:  (1e-5,  1e-5),
    torch.bfloat16: (5e-3,  5e-3),
    torch.float16:  (1e-3,  1e-3),
}

# Multipliers applied on top of dtype base
_STOCHASTIC_MULTIPLIER = 5.0     # MC dropout introduces extra variance
_CUDA_MULTIPLIER       = 2.0     # nondeterministic reductions on GPU

# output_type adjustments (multiplicative)
_OUTPUT_MULTIPLIER: dict[str, float] = {
    "scalar_prediction": 1.0,
    "logits":            1.0,
    "probabilities":     2.0,   # softmax amplifies near boundaries
    "coefficients":      1.0,
    "bias":              1.0,
    "embedding":         1.0,
}


def get_tolerance(
    dtype: torch.dtype = torch.float32,
    device_type: Literal["cpu", "cuda"] = "cpu",
    inference: Literal["deterministic", "stochastic"] = "deterministic",
    output_type: str = "scalar_prediction",
    model_path: Literal["regression", "classification"] = "regression",
) -> Tolerance:
    """Resolve the tolerance for the given combination of factors.

    Returns a ``Tolerance`` dataclass with ``atol``, ``rtol``, and ``note``.
    """
    base_atol, base_rtol = _DTYPE_BASE.get(dtype, (1e-5, 1e-5))

    mult = 1.0
    notes = [f"dtype={dtype}"]

    if device_type == "cuda":
        mult *= _CUDA_MULTIPLIER
        notes.append("cuda×2")

    if inference == "stochastic":
        mult *= _STOCHASTIC_MULTIPLIER
        notes.append("stochastic×5")

    out_mult = _OUTPUT_MULTIPLIER.get(output_type, 1.0)
    if out_mult != 1.0:
        mult *= out_mult
        notes.append(f"{output_type}×{out_mult}")

    atol = base_atol * mult
    rtol = base_rtol * mult

    return Tolerance(atol=atol, rtol=rtol, note="; ".join(notes))
