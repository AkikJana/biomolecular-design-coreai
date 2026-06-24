from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

import boltz.model.layers.initialize as init
from boltz.model.modules.utils import autocast_device_type


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
        use_fold_cp: bool = False,
        num_devices: int = 4,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.use_fold_cp = use_fold_cp
        self.num_devices = num_devices

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor or bias (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        # Fold-CP sharded ring attention is numerically identical to the dense
        # path (online softmax). It only applies to self-attention (q and k of
        # equal length); otherwise fall back to dense.
        if self.use_fold_cp and q.shape[1] == k.shape[1]:
            o = self._fold_cp_attention(q, k, v, bias, mask)
        else:
            with torch.autocast(autocast_device_type(q.device.type), enabled=False):
                # Compute attention weights
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (self.head_dim**0.5) + bias.float()
                attn = attn + (1 - mask[:, None, None].float()) * -self.inf
                attn = attn.softmax(dim=-1)

                # Compute output
                o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
            o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o

    def _fold_cp_attention(
        self, q: Tensor, k: Tensor, v: Tensor, bias: Tensor, mask: Tensor
    ) -> Tensor:
        """Sharded ring attention with online (flash-style) softmax.

        Mathematically identical to the dense path; computes attention across
        ``num_devices`` shards as it would on multiple devices. q, k, v are
        (B, N, H, head_dim); ``bias`` is (B, H, N, N); ``mask`` is (B, N) over
        keys. The sequence is zero-padded to a multiple of ``num_devices`` so the
        shard grid divides evenly for arbitrary N (padded keys are masked out and
        padded query rows are sliced off), so the result is unchanged.
        """
        B = q.shape[0]
        P = self.num_devices
        H, hd = self.num_heads, self.head_dim
        N = q.shape[1]

        N_pad = ((N + P - 1) // P) * P
        if N_pad != N:
            pad_seq = N_pad - N
            # q, k, v are (B, N, H, hd) -> pad the sequence dim (dim 1)
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_seq))
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_seq))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_seq))
            mask = torch.nn.functional.pad(mask, (0, pad_seq))  # padded keys -> 0
            bias = torch.nn.functional.pad(bias, (0, pad_seq, 0, pad_seq))

        N_full = N_pad
        N_shard = N_full // P
        scale = 1.0 / (hd**0.5)

        q_shards = q.view(B, P, N_shard, H, hd)
        k_shards = k.view(B, P, N_shard, H, hd)
        v_shards = v.view(B, P, N_shard, H, hd)
        mask_shards = mask.view(B, P, N_shard)
        z_shards = bias.permute(0, 2, 3, 1).view(B, P, N_shard, N_full, H)

        out_shards = []
        for b in range(B):
            q_s = q_shards[b]
            z_s = z_shards[b]
            mask_s = mask_shards[b]

            o_s = torch.zeros_like(q_s)
            m_run = torch.full((P, N_shard, H, 1), -float("inf"), device=q.device)
            d_run = torch.zeros((P, N_shard, H, 1), device=q.device)

            current_k = k_shards[b].clone()
            current_v = v_shards[b].clone()
            q_scaled = q_s * scale

            for step in range(P):
                for rank in range(P):
                    kv_rank = (rank - step) % P

                    q_h = q_scaled[rank].permute(1, 0, 2)
                    k_h = current_k[rank].permute(1, 0, 2)
                    logits = torch.bmm(q_h, k_h.transpose(-1, -2)).permute(1, 2, 0)

                    bias_slice = z_s[rank, :, kv_rank * N_shard : (kv_rank + 1) * N_shard, :]
                    logits = logits + bias_slice

                    mask_slice = mask_s[kv_rank]
                    logits = logits + (1 - mask_slice[None, :, None].float()) * -self.inf

                    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                    logits_max = logits_max.permute(0, 2, 1)
                    m_new = torch.maximum(m_run[rank], logits_max)

                    exp_logits = torch.exp(logits - logits_max.permute(0, 2, 1))
                    exp_sum = torch.sum(exp_logits, dim=1, keepdim=True).permute(0, 2, 1)

                    alpha = torch.exp(m_run[rank] - m_new)
                    alpha = torch.where(
                        torch.isinf(m_run[rank]), torch.ones_like(alpha), alpha
                    )
                    exp_scale = torch.exp(logits_max - m_new)
                    d_new = alpha * d_run[rank] + exp_scale * exp_sum

                    exp_h = exp_logits.permute(2, 0, 1)
                    v_h = current_v[rank].permute(1, 0, 2)
                    local_out = torch.bmm(exp_h, v_h).permute(1, 0, 2)

                    o_s[rank] = alpha * o_s[rank] + exp_scale * local_out
                    m_run[rank] = m_new
                    d_run[rank] = d_new

                current_k = torch.roll(current_k, shifts=1, dims=0)
                current_v = torch.roll(current_v, shifts=1, dims=0)

            o_s = o_s / d_run
            out_shards.append(o_s.view(N_full, H, hd))

        o = torch.stack(out_shards, dim=0).reshape(B, N_full, self.c_s)
        if N_pad != N:
            o = o[:, :N, :]
        return o
