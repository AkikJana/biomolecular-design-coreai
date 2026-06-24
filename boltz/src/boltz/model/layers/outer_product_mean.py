import torch
from torch import Tensor, nn

from boltz.model.layers.low_rank_pair_representation import LowRankPairUpdater


class _OuterProductMeanFunction(torch.autograd.Function):
    """Memory-efficient S-contracted outer-product mean.

    forward inputs:
        Xm, Ym : (B, S, N, rank)  -- mask-weighted low-rank factors
        W      : (c_out, rank)
        num    : (B, N, N)        -- per-pair valid-sequence counts (>= 1)
    output:
        z      : (B, N, N, c_out)

    Only ``(Xm, Ym, W, num)`` are saved for backward. The contracted
    ``G`` of shape ``(B, N, N, rank)`` is recomputed inside ``backward`` instead
    of being persisted on the autograd tape, so the saved-activation footprint is
    O(B * S * N * rank) rather than O(B * N^2 * rank).
    """

    @staticmethod
    def forward(ctx, Xm, Ym, W, num):
        # Contract over the MSA depth S -> (B, N, N, rank). Transient: freed
        # once sum_U is formed (not saved for backward).
        G = torch.einsum("bsir,bsjr->bijr", Xm, Ym)
        sum_U = torch.einsum("bijr,cr->bijc", G, W)  # (B, N, N, c_out)
        z = sum_U / num.unsqueeze(-1)
        ctx.save_for_backward(Xm, Ym, W, num)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        Xm, Ym, W, num = ctx.saved_tensors

        # dL/d(sum_U): fold the normalization back in.
        gz = grad_z / num.unsqueeze(-1)  # (B, N, N, c_out)

        # dL/dG = gz @ W   (sum_U = G x W^T over the rank axis).
        grad_G = torch.einsum("bijc,cr->bijr", gz, W)  # (B, N, N, rank)

        # dL/dXm and dL/dYm flow back through G = sum_s Xm Ym.
        grad_Xm = torch.einsum("bijr,bsjr->bsir", grad_G, Ym)
        grad_Ym = torch.einsum("bijr,bsir->bsjr", grad_G, Xm)

        grad_W = None
        if ctx.needs_input_grad[2]:
            # Recompute G from the saved factors (avoids persisting it).
            G = torch.einsum("bsir,bsjr->bijr", Xm, Ym)
            grad_W = torch.einsum("bijc,bijr->cr", gz, G)

        # ``num`` is derived from a non-differentiable mask -> no gradient.
        return grad_Xm, grad_Ym, grad_W, None


class OuterProductMean(nn.Module):
    """Outer product mean layer, implemented using LowRankPairUpdater."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        """Initialize the outer product mean layer.

        Parameters
        ----------
        c_in : int
            The input dimension.
        c_hidden : int
            The hidden dimension.
        c_out : int
            The output dimension.

        """
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        # Instantiate LowRankPairUpdater with rank = c_hidden
        self.low_rank_updater = LowRankPairUpdater(d_seq=c_in, d_pair=c_out, rank=c_hidden, use_bias=True)

    def forward(self, m: Tensor, mask: Tensor, chunk_size: int = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The sequence tensor (B, S, N, c_in).
        mask : torch.Tensor
            The mask tensor (B, S, N).

        Returns
        -------
        torch.Tensor
            The output tensor (B, N, N, c_out).

        """
        # m: (B, S, N, c_in)
        # mask: (B, S, N)
        #
        # The outer-product mean over the MSA depth S is computed by contracting
        # over S *before* forming any (N, N) tensor. This is mathematically
        # identical to materializing the per-row update U of shape
        # (B, S, N, N, c_out) and then mean-reducing over S, but it never
        # allocates an S * N^2 tensor, so peak memory is independent of MSA depth.
        #
        #   U[b, s, i, j, c] = sum_r X[b, s, i, r] * Y[b, s, j, r] * W[c, r]
        #   z[b, i, j, c]    = (sum_s mask_i * mask_j * U[..]) / (sum_s mask_i * mask_j)
        #                    = sum_r W[c, r] * G[b, i, j, r] / num_mask
        #   where G[b, i, j, r] = sum_s (mask * X)[b, s, i, r] (mask * Y)[b, s, j, r]
        #
        # ``chunk_size`` is accepted for call-site compatibility with the
        # original OuterProductMean but is unnecessary here: the S dimension is
        # contracted analytically rather than tiled.
        m_normed = self.norm(m)  # (B, S, N, c_in)

        updater = self.low_rank_updater
        X = updater.proj_x(m_normed)  # (B, S, N, rank)
        Y = updater.proj_y(m_normed)  # (B, S, N, rank)

        mask = mask.to(m)  # (B, S, N)
        Xm = X * mask.unsqueeze(-1)  # (B, S, N, rank)
        Ym = Y * mask.unsqueeze(-1)  # (B, S, N, rank)

        # Mask-weighted normalization (number of valid sequences per (i, j) pair).
        num_mask = torch.einsum("bsi,bsj->bij", mask, mask).clamp(min=1)  # (B, N, N)

        # The S-contraction, rank->channel projection, and normalization are
        # fused in a custom autograd Function whose backward recomputes G,
        # keeping saved activations O(B * S * N * rank) instead of O(B * N^2 * rank).
        z_out = _OuterProductMeanFunction.apply(Xm, Ym, updater.W, num_mask)

        return z_out

