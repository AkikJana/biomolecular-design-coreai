import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

class CacheAlignedAttention(nn.Module):
    """
    Optimized Cache-Aligned Attention module for edge hardware (MPS/ANE).
    
    This module addresses two critical issues on edge accelerators:
    1. Dynamic Shape Recompilation: Changes in KV cache shape trigger expensive
       JIT compiler graph updates (recompilation) on Apple Silicon MPS/ANE.
       We solve this by pre-allocating a static, fixed-size contiguous memory cache.
    2. Memory Thrashing & Unaligned Strides: Dynamic memory allocation and 
       unaligned memory accesses degrade memory bandwidth. We solve this by
       padding the head and sequence dimensions to configurable block sizes,
       ensuring all memory offsets and strides align with hardware cache lines (64 bytes).
    
    Features:
    - Standard Attention with Static KV Cache & In-place aligned updates via index_copy_.
    - Linear-Scaling Attention (O(N) complexity) via recurrence kernel trick,
      providing a naturally constant state size for long sequence contexts.
    - Zero-recompilation during generation under torch.compile.
    """
    def __init__(
        self,
        c_s: int,
        num_heads: int,
        max_seq_len: int = 1024,
        block_size: int = 16,
        head_dim: Optional[int] = None,
        attention_type: str = "standard",  # "standard" or "linear"
        device: str = "mps",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.c_s = c_s
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.attention_type = attention_type.lower()
        self.device = device
        self.dtype = dtype
        
        # Calculate and align head dimension
        self.head_dim = head_dim if head_dim is not None else (c_s // num_heads)
        # Pad head dimension to a multiple of block_size (align to hardware cache lines)
        self.head_dim_padded = ((self.head_dim + block_size - 1) // block_size) * block_size
        
        # Pad maximum sequence length to a multiple of block_size
        self.max_seq_len_padded = ((max_seq_len + block_size - 1) // block_size) * block_size
        
        # Projections
        self.proj_q = nn.Linear(c_s, c_s, device=device, dtype=dtype)
        self.proj_k = nn.Linear(c_s, c_s, bias=False, device=device, dtype=dtype)
        self.proj_v = nn.Linear(c_s, c_s, bias=False, device=device, dtype=dtype)
        self.proj_o = nn.Linear(c_s, c_s, bias=False, device=device, dtype=dtype)
        
        # Initialize Cache buffers
        self.cache_k = None
        self.cache_v = None
        
        # Linear attention state buffers
        self.linear_cache_S = None  # shape: (B, H, D, D)
        self.linear_cache_z = None  # shape: (B, H, D)
        
        # Registers for position tracking (buffers to avoid host-device synchronization)
        self.register_buffer(
            "current_pos",
            torch.zeros((), dtype=torch.long, device=device),
            persistent=False
        )
        self.register_buffer(
            "actual_len",
            torch.zeros((), dtype=torch.long, device=device),
            persistent=False
        )
        
    def reset_cache(self, batch_size: int):
        """
        Pre-allocates the fixed-size contiguous memory blocks for keys and values,
        ensuring alignment to cache line boundaries.
        """
        if self.attention_type == "standard":
            # Cache shape: (B, H, S_padded, D_padded)
            # Strides are multiples of head_dim_padded, ensuring memory alignment.
            self.cache_k = torch.zeros(
                (batch_size, self.num_heads, self.max_seq_len_padded, self.head_dim_padded),
                device=self.device,
                dtype=self.dtype
            )
            self.cache_v = torch.zeros(
                (batch_size, self.num_heads, self.max_seq_len_padded, self.head_dim_padded),
                device=self.device,
                dtype=self.dtype
            )
        elif self.attention_type == "linear":
            # Linear cache: state matrix S of shape (B, H, D_padded, D_padded)
            # and normalizer z of shape (B, H, D_padded)
            self.linear_cache_S = torch.zeros(
                (batch_size, self.num_heads, self.head_dim_padded, self.head_dim_padded),
                device=self.device,
                dtype=self.dtype
            )
            self.linear_cache_z = torch.zeros(
                (batch_size, self.num_heads, self.head_dim_padded),
                device=self.device,
                dtype=self.dtype
            )
            
        self.current_pos.zero_()
        self.actual_len.zero_()
        
    def _update_standard_cache(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the key-value cache using block-aligned in-place copy.
        """
        B, H, S_k, D = k.shape
        device = k.device
        
        # 1. Pad head dimension to aligned size if necessary
        if D < self.head_dim_padded:
            k = torch.nn.functional.pad(k, (0, self.head_dim_padded - D))
            v = torch.nn.functional.pad(v, (0, self.head_dim_padded - D))
            
        # 2. Round up update sequence length to a multiple of block_size
        S_k_padded = ((S_k + self.block_size - 1) // self.block_size) * self.block_size
        
        # Pad sequence dimension to match aligned block size
        if S_k_padded > S_k:
            pad_len = S_k_padded - S_k
            k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
            v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
        else:
            k_padded = k
            v_padded = v
            
        # 3. Perform in-place aligned copy using index_copy_
        # We write static-sized blocks of size S_k_padded to the cache.
        indices = self.current_pos + torch.arange(S_k_padded, device=device)
        self.cache_k.index_copy_(2, indices, k_padded)
        self.cache_v.index_copy_(2, indices, v_padded)
        
        # 4. Update track variables
        self.current_pos.copy_(self.current_pos + S_k_padded)
        self.actual_len.copy_(self.actual_len + S_k)
        
        return self.cache_k, self.cache_v

    def _forward_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        custom_mask: Optional[torch.Tensor] = None,
        pair_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard attention with static cache shapes and in-place updates.
        """
        B, S_q, H, D = q.shape
        device = q.device
        
        # Pre-align Query head dimension
        if D < self.head_dim_padded:
            q = torch.nn.functional.pad(q, (0, self.head_dim_padded - D))
            
        # Transpose to (B, H, S, D) for standard attention dot products
        q = q.transpose(1, 2)
        k_trans = k.transpose(1, 2)
        v_trans = v.transpose(1, 2)
        
        # Lazy initialization or dynamic batch resize
        if self.cache_k is None or self.cache_k.shape[0] != B:
            self.reset_cache(B)
            
        # Update Cache
        k_cache, v_cache = self._update_standard_cache(k_trans, v_trans)
        
        # Compute dot product attention over the entire cached block (completely static shape)
        # q: (B, H, S_q, D_padded)
        # k_cache: (B, H, S_max_padded, D_padded)
        attn_scores = torch.matmul(q, k_cache.transpose(-2, -1)) / math.sqrt(self.head_dim_padded)
        # attn_scores shape: (B, H, S_q, S_max_padded)
        
        # Apply Pair Bias if provided.
        # pair_bias shape: (B, H, S_q, S_k) or (B, H, S_q, S_max_padded)
        if pair_bias is not None:
            # Pad or slice pair bias to match S_max_padded
            S_pb = pair_bias.shape[-1]
            if S_pb < self.max_seq_len_padded:
                pair_bias = torch.nn.functional.pad(pair_bias, (0, self.max_seq_len_padded - S_pb))
            elif S_pb > self.max_seq_len_padded:
                pair_bias = pair_bias[..., :self.max_seq_len_padded]
            attn_scores = attn_scores + pair_bias
            
        # Create attention mask to hide future/padded keys
        col_indices = torch.arange(self.max_seq_len_padded, device=device)
        valid_mask = col_indices < self.actual_len
        
        if causal:
            # For causal mask: key_index < (actual_len - S_k + q_index + 1)
            # S_k is the number of newly added keys (tokens)
            q_indices = torch.arange(S_q, device=device).unsqueeze(1)
            k_indices = col_indices.unsqueeze(0)
            causal_mask = k_indices < (self.actual_len - S_q + q_indices + 1)
            mask = valid_mask.unsqueeze(0) & causal_mask
        else:
            mask = valid_mask.view(1, 1, 1, self.max_seq_len_padded)
            
        if custom_mask is not None:
            # Custom mask is combined with cache mask
            # Pad custom mask to S_max_padded
            S_m = custom_mask.shape[-1]
            if S_m < self.max_seq_len_padded:
                custom_mask_padded = torch.nn.functional.pad(custom_mask, (0, self.max_seq_len_padded - S_m), value=0)
            else:
                custom_mask_padded = custom_mask[..., :self.max_seq_len_padded]
            mask = mask & custom_mask_padded.to(torch.bool)
            
        # Apply large negative value to masked elements to zero them out in softmax
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        
        # Softmax and compute output
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # v_cache: (B, H, S_max_padded, D_padded)
        o = torch.matmul(attn_probs, v_cache)
        # shape: (B, H, S_q, D_padded)
        
        # Transpose back to (B, S_q, H, D_padded) and project out
        o = o.transpose(1, 2)
        # Crop back to original c_s dimension (if padded)
        if o.shape[-1] > self.head_dim:
            o = o[..., :self.head_dim]
            
        o = o.reshape(B, S_q, self.c_s)
        return self.proj_o(o)

    def _forward_linear(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        custom_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Linear attention with recurrent updates.
        Computes attention in O(N) complexity using the kernel trick:
        phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T)
        """
        B, S_q, H, D = q.shape
        S_k = k.shape[1]
        
        # Transpose to (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if D < self.head_dim_padded:
            q = torch.nn.functional.pad(q, (0, self.head_dim_padded - D))
            k = torch.nn.functional.pad(k, (0, self.head_dim_padded - D))
            v = torch.nn.functional.pad(v, (0, self.head_dim_padded - D))
            
        # Lazy initialization
        if self.linear_cache_S is None or self.linear_cache_S.shape[0] != B:
            self.reset_cache(B)
            
        # Apply Feature Map (ELU + 1 is standard for Linear Attention)
        # This ensures all query/key elements are non-negative.
        phi_q = torch.nn.functional.elu(q) + 1.0
        phi_k = torch.nn.functional.elu(k) + 1.0
        
        # Apply custom mask to keys if present
        if custom_mask is not None:
            # custom_mask shape: (B, S_k) or (B, H, S_k)
            # Expand mask to (B, H, S_k, 1)
            if custom_mask.dim() == 2:
                custom_mask = custom_mask.unsqueeze(1).unsqueeze(-1)
            elif custom_mask.dim() == 3:
                custom_mask = custom_mask.unsqueeze(-1)
            phi_k = phi_k * custom_mask.to(phi_k.dtype)
            
        # Update Recurrent state
        # state update: S_t = S_{t-1} + phi(K_t)^T @ V_t
        # phi_k: (B, H, S_k, D_padded)
        # v: (B, H, S_k, D_padded)
        # phi_k.transpose(-2, -1) @ v: (B, H, D_padded, D_padded)
        update_S = torch.matmul(phi_k.transpose(-2, -1), v)
        self.linear_cache_S.copy_(self.linear_cache_S + update_S)
        
        # normalizer update: z_t = z_{t-1} + phi(K_t).sum(dim=S)
        update_z = phi_k.sum(dim=-2) # shape: (B, H, D_padded)
        self.linear_cache_z.copy_(self.linear_cache_z + update_z)
        
        # Compute Output:
        # Numerator: phi_q @ S  -> (B, H, S_q, D_padded)
        # Denominator: phi_q @ z^T -> (B, H, S_q, 1)
        num = torch.matmul(phi_q, self.linear_cache_S)
        
        # phi_q: (B, H, S_q, D_padded)
        # z: (B, H, D_padded) -> unsqueeze to (B, H, D_padded, 1)
        den = torch.matmul(phi_q, self.linear_cache_z.unsqueeze(-1))
        # Add epsilon to denominator to avoid division by zero
        den = den + 1e-6
        
        o = num / den
        
        # Transpose back and project out
        o = o.transpose(1, 2)
        if o.shape[-1] > self.head_dim:
            o = o[..., :self.head_dim]
            
        o = o.reshape(B, S_q, self.c_s)
        return self.proj_o(o)

    def forward(
        self,
        s: torch.Tensor,
        k_in: torch.Tensor,
        causal: bool = False,
        custom_mask: Optional[torch.Tensor] = None,
        pair_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        - s: input sequence tensor for queries (B, S_q, c_s)
        - k_in: input sequence tensor for keys/values (B, S_k, c_s)
        - causal: apply causal masking (only standard attention)
        - custom_mask: optional attention mask (B, S_k)
        - pair_bias: optional pair bias (B, H, S_q, S_k)
        """
        B = s.shape[0]
        
        # Compute linear projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        
        if self.attention_type == "standard":
            return self._forward_standard(q, k, v, causal, custom_mask, pair_bias)
        elif self.attention_type == "linear":
            return self._forward_linear(q, k, v, custom_mask)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
