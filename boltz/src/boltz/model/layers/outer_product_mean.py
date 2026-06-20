import torch
from torch import Tensor, nn

from boltz.model.layers.low_rank_pair_representation import LowRankPairUpdater

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
        B, S, N, C_in = m.shape
        
        # Apply layer norm
        m_normed = self.norm(m) # (B, S, N, c_in)
        
        # Reshape to (B * S, N, c_in) to apply LowRankPairUpdater
        m_flat = m_normed.reshape(B * S, N, C_in)
        
        # LowRankPairUpdater returns (B * S, N, N, c_out)
        U_flat = self.low_rank_updater(m_flat)
        U = U_flat.reshape(B, S, N, N, -1)
        
        # Expand mask to compute weighted mean over S
        mask = mask.unsqueeze(-1).to(m) # (B, S, N, 1)
        pairwise_mask = mask[:, :, :, None, :] * mask[:, :, None, :, :] # (B, S, N, N, 1)
        
        # Sum over the sequences (S) dimension
        weighted_U = U * pairwise_mask # (B, S, N, N, c_out)
        sum_U = weighted_U.sum(dim=1) # (B, N, N, c_out)
        
        # Normalize by num_mask
        num_mask = pairwise_mask.sum(dim=1).clamp(min=1) # (B, N, N, 1)
        z_out = sum_U / num_mask
        
        return z_out

