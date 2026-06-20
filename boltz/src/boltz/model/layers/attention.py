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
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
        use_mla: bool = False,
        latent_dim: int = 32,
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
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.use_mla = use_mla
        self.latent_dim = latent_dim
        self.use_fold_cp = use_fold_cp
        self.num_devices = num_devices

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        
        if self.use_mla:
            # MLA Key/Value down/up projections
            self.kv_down_proj = nn.Linear(c_s, latent_dim, bias=False)
            self.k_up_proj = nn.Linear(latent_dim, c_s, bias=False)
            self.v_up_proj = nn.Linear(latent_dim, c_s, bias=False)
        else:
            self.proj_k = nn.Linear(c_s, c_s, bias=False)
            self.proj_v = nn.Linear(c_s, c_s, bias=False)

        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
            Rearrange("b ... h -> b h ..."),
        )

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N)
        multiplicity : int, optional
            The diffusion batch size, by default 1

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Layer norms
        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        if self.use_mla:
            latent_kv = self.kv_down_proj(k_in)
            k = self.k_up_proj(latent_kv).view(B, -1, self.num_heads, self.head_dim)
            v = self.v_up_proj(latent_kv).view(B, -1, self.num_heads, self.head_dim)
        else:
            k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
            v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        # Caching z projection during diffusion roll-out
        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)

            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        if self.use_fold_cp:
            P = self.num_devices
            N_full = q.shape[1]
            N_shard = N_full // P
            
            q_shards = q.view(B, P, N_shard, self.num_heads, self.head_dim)
            k_shards = k.view(B, P, N_shard, self.num_heads, self.head_dim)
            v_shards = v.view(B, P, N_shard, self.num_heads, self.head_dim)
            mask_shards = mask.view(B, P, N_shard)
            
            z_p = z.permute(0, 2, 3, 1) # [B, N, N, H]
            z_shards = z_p.view(B, P, N_shard, N_full, self.num_heads)
            
            out_shards = []
            for b in range(B):
                q_s = q_shards[b]
                k_s = k_shards[b]
                v_s = v_shards[b]
                z_s = z_shards[b]
                mask_s = mask_shards[b]
                
                o_s = torch.zeros_like(q_s)
                m_run = torch.full((P, N_shard, self.num_heads, 1), -float('inf'), device=s.device)
                d_run = torch.zeros((P, N_shard, self.num_heads, 1), device=s.device)
                
                current_k = k_s.clone()
                current_v = v_s.clone()
                
                scale = 1.0 / (self.head_dim ** 0.5)
                q_scaled = q_s * scale
                
                for step in range(P):
                    for rank in range(P):
                        kv_rank = (rank - step) % P
                        
                        q_local = q_scaled[rank]
                        k_local = current_k[rank]
                        v_local = current_v[rank]
                        
                        q_h = q_local.permute(1, 0, 2)
                        k_h = k_local.permute(1, 0, 2)
                        
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
                        alpha = torch.where(torch.isinf(m_run[rank]), torch.ones_like(alpha), alpha)
                        
                        exp_scale = torch.exp(logits_max - m_new)
                        d_new = alpha * d_run[rank] + exp_scale * exp_sum
                        
                        exp_h = exp_logits.permute(2, 0, 1)
                        v_h = v_local.permute(1, 0, 2)
                        
                        local_out = torch.bmm(exp_h, v_h).permute(1, 0, 2)
                        local_out_scaled = exp_scale * local_out
                        
                        o_s[rank] = alpha * o_s[rank] + local_out_scaled
                        m_run[rank] = m_new
                        d_run[rank] = d_new
                        
                    current_k = torch.roll(current_k, shifts=1, dims=0)
                    current_v = torch.roll(current_v, shifts=1, dims=0)
                    
                o_s = o_s / d_run
                out_shards.append(o_s.view(N_full, self.num_heads, self.head_dim))
                
            o = torch.stack(out_shards, dim=0)
            o = o.reshape(B, -1, self.c_s)
        else:
            with torch.autocast(autocast_device_type(q.device.type), enabled=False):
                # Compute attention weights
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (self.head_dim**0.5) + z.float()
                # The pairwise mask tensor (B, N) is broadcasted to (B, 1, 1, N) and (B, H, N, N)
                attn = attn + (1 - mask[:, None, None].float()) * -self.inf
                attn = attn.softmax(dim=-1)

                # Compute output
                o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
            o = o.reshape(B, -1, self.c_s)

        o = self.proj_o(g * o)

        return o
