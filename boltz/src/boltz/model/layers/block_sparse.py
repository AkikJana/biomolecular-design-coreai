"""Block-sparse attention: compute attention only over active query/key block
pairs, skipping the rest.

This is *exact* with respect to a dense attention that masks the same inactive
block pairs to -inf -- i.e. it is a faithful, cheaper implementation of a
block-masked attention, not an approximation of it. Relative to FULL dense
attention it is lossy (it drops the masked block pairs), so using it to
accelerate a model trained with full attention requires accuracy validation /
fine-tuning. It is provided as a building block, not enabled in the live path.

Savings: with ``nb`` blocks and ``A`` active pairs, score/value compute drops
from O(nb^2) to O(A) block-matmuls.
"""

import math

import torch
from torch import Tensor


def block_sparse_attention(
    q: Tensor,         # (B, H, N, d)
    k: Tensor,         # (B, H, N, d)
    v: Tensor,         # (B, H, N, d)
    block_mask: Tensor,  # (nb, nb) bool: which (query-block, key-block) pairs are active
    bias: Tensor = None,   # (B, H, N, N) additive bias, optional
    key_mask: Tensor = None,  # (B, N) 1=keep, 0=pad, optional
    block_size: int = 64,
    inf: float = 1e6,
) -> Tensor:
    """Attention restricted to active blocks. Returns (B, H, N, d).

    Requires N divisible by block_size and every query block to have at least one
    active key block (a True diagonal in ``block_mask`` guarantees this).
    """
    B, H, N, d = q.shape
    nb = N // block_size
    assert nb * block_size == N, "N must be divisible by block_size"
    assert tuple(block_mask.shape) == (nb, nb), "block_mask must be (nb, nb)"
    scale = 1.0 / math.sqrt(d)

    out = torch.zeros_like(q)
    for bi in range(nb):
        active = torch.nonzero(block_mask[bi], as_tuple=False).flatten().tolist()
        if not active:
            continue
        qs = slice(bi * block_size, (bi + 1) * block_size)
        q_i = q[:, :, qs] * scale  # (B, H, bs, d)

        # Gather active key/value blocks.
        k_cat = torch.cat([k[:, :, j * block_size:(j + 1) * block_size] for j in active], dim=2)
        v_cat = torch.cat([v[:, :, j * block_size:(j + 1) * block_size] for j in active], dim=2)

        logits = torch.einsum("bhid,bhjd->bhij", q_i, k_cat)  # (B, H, bs, A*bs)

        if bias is not None:
            bias_cat = torch.cat(
                [bias[:, :, qs, j * block_size:(j + 1) * block_size] for j in active], dim=-1
            )
            logits = logits + bias_cat
        if key_mask is not None:
            km = torch.cat([key_mask[:, j * block_size:(j + 1) * block_size] for j in active], dim=1)
            logits = logits + (1 - km[:, None, None, :].float()) * -inf

        attn = logits.softmax(dim=-1)
        out[:, :, qs] = torch.einsum("bhij,bhjd->bhid", attn, v_cat)
    return out


def dense_block_masked_attention(
    q: Tensor, k: Tensor, v: Tensor, block_mask: Tensor,
    bias: Tensor = None, key_mask: Tensor = None, block_size: int = 64, inf: float = 1e6,
) -> Tensor:
    """Reference: full dense attention with inactive block pairs masked to -inf.

    ``block_sparse_attention`` is numerically equal to this (used for testing).
    """
    B, H, N, d = q.shape
    nb = N // block_size
    scale = 1.0 / math.sqrt(d)
    logits = torch.einsum("bhid,bhjd->bhij", q * scale, k)
    if bias is not None:
        logits = logits + bias
    if key_mask is not None:
        logits = logits + (1 - key_mask[:, None, None, :].float()) * -inf
    # Expand block_mask to full N x N and mask inactive pairs.
    full = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    logits = logits + (1 - full.float())[None, None] * -inf
    attn = logits.softmax(dim=-1)
    return torch.einsum("bhij,bhjd->bhid", attn, v)
