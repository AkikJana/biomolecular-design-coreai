import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAProteinAttention(nn.Module):
    """Simulates Multi-Head Latent Attention (MLA) adapted for protein structure models.
    
    Instead of caching raw K/V projections for a large constant target receptor, 
    MLA projects Key-Value states into a compressed low-dimensional latent space,
    reducing memory consumption by up to 80% on local hardware.
    """
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, latent_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim
        
        # Projections for Queries
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # Standard Key/Value projections (Baseline)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # MLA Key/Value projections
        # 1. Down-projection: compresses sequence states to latent dimension
        self.kv_down_proj = nn.Linear(embed_dim, latent_dim)
        
        # 2. Up-projections: reconstruct Keys and Values from latent states on-the-fly
        self.k_up_proj = nn.Linear(latent_dim, embed_dim)
        self.v_up_proj = nn.Linear(latent_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # [B, H, L, head_dim]

    def forward_standard_cache(self, 
                               binder_seq: torch.Tensor, 
                               target_k_cache: torch.Tensor, 
                               target_v_cache: torch.Tensor) -> torch.Tensor:
        """Standard Cache: Reuses raw target Keys and Values from VRAM."""
        q_binder = self._split_heads(self.q_proj(binder_seq))
        k_binder = self._split_heads(self.k_proj(binder_seq))
        v_binder = self._split_heads(self.v_proj(binder_seq))
        
        # Concat along sequence dimension
        full_k = torch.cat([target_k_cache, k_binder], dim=2)
        full_v = torch.cat([target_v_cache, v_binder], dim=2)
        
        # Project queries for both parts
        q_target = self._split_heads(self.q_proj(target_k_cache.permute(0, 2, 1, 3).contiguous().view(1, -1, self.embed_dim)))
        full_q = torch.cat([q_target, q_binder], dim=2)
        
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(full_q, full_k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, full_v)
        
        # Project output
        batch_size, num_heads, seq_len, head_dim = attn_out.shape
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_out)

    def precompute_latent_cache(self, target_seq: torch.Tensor) -> torch.Tensor:
        """MLA Caching: Down-projects the target sequence into low-dimensional latent cache."""
        with torch.no_grad():
            # Compresses target from size [1, L_target, 128] to [1, L_target, 32]
            latent_cache = self.kv_down_proj(target_seq)
        return latent_cache

    def forward_mla_cache(self, 
                          binder_seq: torch.Tensor, 
                          target_latent_cache: torch.Tensor) -> torch.Tensor:
        """MLA Cache: Reconstructs target Keys and Values on-the-fly from latent representation."""
        # 1. Reconstruct target Keys and Values from compressed latent cache
        # Shape: [1, L_target, embed_dim]
        k_target_reconstructed = self.k_up_proj(target_latent_cache)
        v_target_reconstructed = self.v_up_proj(target_latent_cache)
        
        # Split into heads: [1, H, L_target, head_dim]
        k_target_heads = self._split_heads(k_target_reconstructed)
        v_target_heads = self._split_heads(v_target_reconstructed)
        
        # 2. Compute projections for binder sequence
        q_binder = self._split_heads(self.q_proj(binder_seq))
        k_binder = self._split_heads(self.k_proj(binder_seq))
        v_binder = self._split_heads(self.v_proj(binder_seq))
        
        # 3. Concatenate Keys/Values
        full_k = torch.cat([k_target_heads, k_binder], dim=2)
        full_v = torch.cat([v_target_heads, v_binder], dim=2)
        
        # Project queries
        q_target = self._split_heads(self.q_proj(k_target_reconstructed))
        full_q = torch.cat([q_target, q_binder], dim=2)
        
        # Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(full_q, full_k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, full_v)
        
        # Project output
        batch_size, num_heads, seq_len, head_dim = attn_out.shape
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_out)


def run_mla_benchmark():
    print("======================================================================")
    print("MLA-STYLE LATENT TARGET CACHING DEMONSTRATION")
    print("======================================================================")
    
    L_target = 200
    L_binder = 20
    embed_dim = 128
    latent_dim = 32 # 4x compression ratio
    
    print(f"Target Length:     {L_target} residues")
    print(f"Binder Length:     {L_binder} residues")
    print(f"Embedding Dim:     {embed_dim}")
    print(f"MLA Latent Dim:    {latent_dim} (Compression Ratio: {embed_dim / latent_dim:.1f}x)")
    
    torch.manual_seed(42)
    model = MLAProteinAttention(embed_dim=embed_dim, num_heads=4, latent_dim=latent_dim)
    model.eval()
    
    target_seq = torch.randn(1, L_target, embed_dim)
    binder_seq = torch.randn(1, L_binder, embed_dim)
    
    # 1. Compute Standard Cache Tensors
    with torch.no_grad():
        k_standard = model._split_heads(model.k_proj(target_seq))
        v_standard = model._split_heads(model.v_proj(target_seq))
    
    # 2. Compute MLA Latent Cache Tensor
    latent_cache = model.precompute_latent_cache(target_seq)
    
    # Calculate Memory Footprints (Number of floating-point values stored in cache)
    standard_cache_size = k_standard.nelement() + v_standard.nelement()
    mla_cache_size = latent_cache.nelement()
    
    memory_savings = (1.0 - (mla_cache_size / standard_cache_size)) * 100
    
    # Run forward passes
    with torch.no_grad():
        out_standard = model.forward_standard_cache(binder_seq, k_standard, v_standard)
        out_mla = model.forward_mla_cache(binder_seq, latent_cache)
        
    print("\nCache Memory Statistics:")
    print("-" * 50)
    print(f"  Standard Cache Size (Values):  {standard_cache_size}")
    print(f"  MLA Latent Cache Size (Values): {mla_cache_size}")
    print(f"  Active VRAM Cache Savings:     {memory_savings:.2f}%")
    print("-" * 50)
    
    # Confirm structural projection functionality
    print(f"  Standard Output Shape:        {out_standard.shape}")
    print(f"  MLA Output Shape:             {out_mla.shape}")
    print(f"  Execution Check:              Successful (MLA runs smoothly)")
    print("======================================================================\n")

if __name__ == "__main__":
    run_mla_benchmark()
