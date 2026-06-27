"""
support_augmentation.py

Support-row permutation augmentation for training episodes.

Applies a deterministic random permutation to the support (training) rows of
each episode.  All row-aligned fields are permuted together:
  - X_train, y_train (always)
  - Any additional tensor whose first dimension equals n_train (opt-in)

Seeds are derived from: base_seed ⊕ epoch ⊕ rank ⊕ batch_idx to guarantee
deterministic, collision-resistant permutations across distributed workers.

Never moves observations across support/query boundaries.
Never uses query labels to construct a support permutation.
"""

from __future__ import annotations

import hashlib

import torch


def _permutation_seed(
    base_seed: int,
    epoch: int = 0,
    rank: int = 0,
    batch_idx: int = 0,
) -> int:
    """Derive a deterministic, collision-resistant seed for one episode."""
    payload = f"{base_seed}\x1f{epoch}\x1f{rank}\x1f{batch_idx}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def permute_support_rows(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    base_seed: int,
    epoch: int = 0,
    rank: int = 0,
    batch_idx: int = 0,
    extra_row_tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Permute support rows deterministically.

    Parameters
    ----------
    X_train : (n, p) tensor
    y_train : (n,) tensor
    base_seed : int
        Per-run base seed.
    epoch, rank, batch_idx : int
        Combined with base_seed to produce a unique permutation per episode.
    extra_row_tensors : dict, optional
        Additional tensors whose first dimension == n to permute in lockstep.

    Returns
    -------
    X_perm, y_perm, extras_perm
        Permuted tensors.  extras_perm is an empty dict if extra_row_tensors
        was None.
    """
    n = X_train.shape[0]
    seed = _permutation_seed(base_seed, epoch, rank, batch_idx)
    g = torch.Generator(device=X_train.device)
    g.manual_seed(seed % (2**63))
    pi = torch.randperm(n, device=X_train.device, generator=g)

    X_perm = X_train[pi]
    y_perm = y_train[pi]

    extras_perm: dict[str, torch.Tensor] = {}
    if extra_row_tensors:
        for key, tensor in extra_row_tensors.items():
            if tensor is not None and tensor.ndim >= 1 and tensor.shape[0] == n:
                extras_perm[key] = tensor[pi]
            else:
                extras_perm[key] = tensor

    return X_perm, y_perm, extras_perm
