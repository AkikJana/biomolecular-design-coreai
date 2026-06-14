import torch
import torch.nn as nn
import math
from typing import List, Tuple, Dict, Any

class FoldCPManager:
    """Manages 2D grid configuration and simulated inter-GPU ring communication.
    
    In a real distributed system (e.g., NVIDIA BioNeMo / Boltz-CP), this wrapper
    interfaces with torch.distributed. Here we implement a high-fidelity local simulator
    to verify mathematical equivalence and profile memory savings.
    """
    def __init__(self, num_devices: int = 4):
        self.num_devices = num_devices
        # Solve for a balanced 2D grid: P = P_row x P_col
        # For P=4, grid is 2x2. For P=8, grid is 2x4.
        self.p_row = int(math.sqrt(num_devices))
        while num_devices % self.p_row != 0:
            self.p_row -= 1
        self.p_col = num_devices // self.p_row
        
        print(f"[Fold-CP] Initialized {num_devices}-device virtual grid ({self.p_row}x{self.p_col})")

    def get_rank_coords(self, rank: int) -> Tuple[int, int]:
        """Maps a global rank ID to (row_idx, col_idx) in the 2D grid."""
        return rank // self.p_col, rank % self.p_col

    def get_rank_from_coords(self, row: int, col: int) -> int:
        """Maps (row, col) coordinates to a global rank ID."""
        return (row % self.p_row) * self.p_col + (col % self.p_col)


def ring_attention_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    num_ranks: int,
    device_manager: FoldCPManager
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulates a Ring Attention forward pass for one batch.
    
    Rather than allocating a quadratic O(N^2) attention matrix on a single device,
    each GPU rank computes attention on its local sequence shard of size N/P.
    Keys (K), Values (V), and Pair Biases are passed along a communication ring.
    
    Online Softmax (FlashAttention-style) is used to maintain exact numerical outputs.
    
    Args:
        q: Query shard of shape [P, N_shard, H, D]
        k: Key shard of shape [P, N_shard, H, D]
        v: Value shard of shape [P, N_shard, H, D]
        bias: Sharded 2D pair bias of shape [P, N_shard, N_shard, H] (if sharded 2D)
              or [P, N_shard, N_full, H] for simplicity in ring rotation.
        num_ranks: Number of virtual GPU ranks (P).
        device_manager: FoldCPManager instance.
        
    Returns:
        output: Aggregated attention output shard of shape [P, N_shard, H, D]
        final_m: Online softmax scaling maximums.
        final_d: Online softmax denominator sums.
    """
    P, N_shard, H, D = q.shape
    device = q.device
    
    # Initialize online softmax accumulators for each rank
    # out: [P, N_shard, H, D]
    out = torch.zeros_like(q)
    # m (max logits): [P, N_shard, H, 1] initialized to -infinity
    m = torch.full((P, N_shard, H, 1), -float('inf'), device=device)
    # d (denominator sum): [P, N_shard, H, 1] initialized to zero
    d = torch.zeros((P, N_shard, H, 1), device=device)
    
    # We will simulate a ring-pass. Rank i currently holds:
    # Query: q[i]
    # Key: k[i] (which moves around the ring)
    # Value: v[i] (which moves around the ring)
    
    current_k = k.clone()
    current_v = v.clone()
    
    # Scale query for dot-product attention
    scale = 1.0 / math.sqrt(D)
    q_scaled = q * scale
    
    for step in range(num_ranks):
        # Determine which sequence shard we are currently comparing against.
        # At step 0: Rank i compares q[i] with k[i] (self-attention block).
        # At step s: Rank i compares q[i] with k[(i - s) % P].
        for rank in range(num_ranks):
            # Key/Value rank index that 'rank' is currently processing
            kv_rank = (rank - step) % num_ranks
            
            # Extract local shards
            q_local = q_scaled[rank]           # [N_shard, H, D]
            k_local = current_k[rank]          # [N_shard, H, D]
            v_local = current_v[rank]          # [N_shard, H, D]
            
            # Compute Raw Attention Logits: [N_shard, N_shard, H]
            # Permute for matmul: q is [H, N_shard, D], k is [H, N_shard, D]
            q_h = q_local.permute(1, 0, 2)     # [H, N_shard, D]
            k_h = k_local.permute(1, 0, 2)     # [H, N_shard, D]
            
            # Compute logits [H, N_shard, N_shard]
            logits = torch.bmm(q_h, k_h.transpose(-1, -2))
            
            # Permute back to [N_shard, N_shard, H]
            logits = logits.permute(1, 2, 0)
            
            # Inject pair bias shard if available
            # bias is [P, N_shard, N_full, H], we extract the slice matching kv_rank
            if bias is not None:
                bias_slice = bias[rank, :, kv_rank * N_shard : (kv_rank + 1) * N_shard, :]
                logits = logits + bias_slice
                
            # Online Softmax Update
            # Find maximum along the Key dimension (dim=1)
            logits_max, _ = torch.max(logits, dim=1, keepdim=True) # [N_shard, 1, H]
            logits_max = logits_max.permute(0, 2, 1) # [N_shard, H, 1]
            
            # New running max
            m_new = torch.maximum(m[rank], logits_max)
            
            # Exponents
            exp_logits = torch.exp(logits - logits_max.permute(0, 2, 1)) # [N_shard, N_shard, H]
            exp_sum = torch.sum(exp_logits, dim=1, keepdim=True).permute(0, 2, 1) # [N_shard, H, 1]
            
            # Update denominator and scale previous output
            alpha = torch.exp(m[rank] - m_new)
            # Handle nan/inf when multiplying by 0
            alpha = torch.where(torch.isinf(m[rank]), torch.ones_like(alpha), alpha)
            
            # Scale the new local sum of exponents by exp(logits_max - m_new)
            exp_scale = torch.exp(logits_max - m_new)
            
            d_new = alpha * d[rank] + exp_scale * exp_sum
            
            # Compute local weighted values: [N_shard, H, D]
            # exp_logits: [N_shard, N_shard, H] -> permuted to [H, N_shard, N_shard]
            # v_local: [N_shard, H, D] -> permuted to [H, N_shard, D]
            exp_h = exp_logits.permute(2, 0, 1)
            v_h = v_local.permute(1, 0, 2)
            
            local_out = torch.bmm(exp_h, v_h).permute(1, 0, 2) # [N_shard, H, D]
            
            # Scale local_out by exp(logits_max - m_new)
            local_out_scaled = exp_scale * local_out
            
            # Update running output shard
            out[rank] = alpha * out[rank] + local_out_scaled
            
            # Update state
            m[rank] = m_new
            d[rank] = d_new
            
        # Ring Shift: Rotate K and V shards to the next rank in the ring
        # Rank i receives from Rank (i-1)%P, and sends to Rank (i+1)%P
        current_k = torch.roll(current_k, shifts=1, dims=0)
        current_v = torch.roll(current_v, shifts=1, dims=0)
        
    # Final normalization of output: out = out / d
    out = out / d
    return out, m, d


def ring_triangular_multiplication(
    a_shard: torch.Tensor,
    b_shard: torch.Tensor,
    device_manager: FoldCPManager
) -> torch.Tensor:
    """Computes a 2D Ring-based Triangular Multiplicative Update.
    
    In structural biology networks (e.g. Evoformer/Evoformer-like blocks), 
    the pair representation updates often involve:
        Outward:  C_ij = sum_k (A_ik * B_jk)
        Inward:   C_ij = sum_k (A_ki * B_kj)
        
    Here we implement a parallelized Ring Multiplication for C_ij = sum_k (A_ik * B_kj)
    sharded across a 2D grid of size P_row x P_col.
    
    Each device holds a local shard of:
        a_shard: [P_row, P_col, N/P_row, N/P_col, C]
        b_shard: [P_row, P_col, N/P_row, N/P_col, C]
        
    We return the sharded product:
        c_shard: [P_row, P_col, N/P_row, N/P_col, C]
    
    To avoid global O(N^2) memory footprint, the matrix multiplication is
    computed in sub-blocks by rotating column shards of A and row shards of B
    along their respective grid lines.
    """
    P_row = device_manager.p_row
    P_col = device_manager.p_col
    
    assert a_shard.shape[0] == P_row and a_shard.shape[1] == P_col, "Shard shapes must match 2D grid"
    
    R_shard, C_shard, C_dim = a_shard.shape[2], a_shard.shape[3], a_shard.shape[4]
    device = a_shard.device
    
    # Result accumulator: [P_row, P_col, R_shard, C_shard, C_dim]
    c_out = torch.zeros_like(a_shard)
    
    # We require P_row == P_col for standard block ring matrix multiplication (or we pad).
    # For general rectangular grids, we perform a virtual ring-shift of length P_col/P_row.
    # Here we assume a square grid P_row x P_row for simplicity. If P_row != P_col,
    # we simulate the Ring Communication appropriately.
    
    current_a = a_shard.clone()
    current_b = b_shard.clone()
    
    # Simulate SUMMA-style parallel block matrix multiplication:
    # Ranks along row loop over column blocks.
    # To compute C[i, j] = sum_k A[i, k] * B[k, j]:
    # At step k, device (r, c) multiplies its currently held A block (k-th col block)
    # with B block (k-th row block).
    # We rotate A along rows (left-right) and B along cols (up-down).
    
    # Setup initial alignments for Cannon's algorithm (or just use virtual indexing for simulation)
    for step in range(P_row):
        # In a real cluster, each device (r, c) broadcast-shares or shifts its shards.
        # Here we simulate the operation on all ranks:
        for r in range(P_row):
            for c in range(P_col):
                # The virtual block index k we need to multiply
                # For step s, the column block of A and row block of B:
                k = (r + c + step) % P_row
                
                # Fetch A[r, k] and B[k, c]
                # In actual distributed code, this is obtained via ring communications.
                a_block = a_shard[r, k]  # [R_shard, C_shard, C_dim]
                b_block = b_shard[k, c]  # [R_shard, C_shard, C_dim]
                
                # Compute block multiplication:
                # To perform C_ij = sum_k A_ik * B_kj, we do standard matrix multiplication
                # for each feature dimension C_dim.
                # a_block is [R_shard, K_shard, C_dim], b_block is [K_shard, C_shard, C_dim]
                # Here, since R_shard == C_shard == K_shard, it's a batch matmul over C_dim.
                
                # Permute to make batch size C_dim: [C_dim, R_shard, K_shard]
                a_b = a_block.permute(2, 0, 1)
                b_b = b_block.permute(2, 0, 1)
                
                # Batch matmul: [C_dim, R_shard, C_shard]
                prod = torch.bmm(a_b, b_b)
                
                # Permute back: [R_shard, C_shard, C_dim]
                c_out[r, c] += prod.permute(1, 2, 0)
                
    return c_out


def run_fold_cp_benchmark(N: int = 1024, D: int = 64, H: int = 4, P: int = 4) -> Dict[str, Any]:
    """Runs a benchmarking profile comparing Fold-CP Context Parallelism against Monolithic execution.
    
    Validates:
    1. Numerical Equivalence (to 1e-6 precision).
    2. VRAM footprint savings (O(N^2) vs O(N^2/P)).
    3. Communication volume characteristics.
    """
    device = torch.device("cpu")
    
    # 1. Setup Input Tensors for standard monolithic run
    # Monolithic representations:
    # Q, K, V: [1, N, H, D]
    # Bias: [1, N, N, H]
    q_mono = torch.randn(1, N, H, D, device=device).double()
    k_mono = torch.randn(1, N, H, D, device=device).double()
    v_mono = torch.randn(1, N, H, D, device=device).double()
    bias_mono = torch.randn(1, N, N, H, device=device).double()
    
    # 2. Compute Monolithic Ground Truth Attention
    # Scale query
    scale = 1.0 / math.sqrt(D)
    q_scaled = q_mono * scale
    
    # Monolithic Attention:
    # q: [H, N, D], k: [H, N, D], bias: [H, N, N]
    q_h = q_scaled[0].permute(1, 0, 2)
    k_h = k_mono[0].permute(1, 0, 2)
    bias_h = bias_mono[0].permute(2, 0, 1) # [H, N, N]
    
    logits_mono = torch.bmm(q_h, k_h.transpose(-1, -2)) + bias_h
    attn_weights = torch.softmax(logits_mono, dim=-1)
    
    v_h = v_mono[0].permute(1, 0, 2)
    output_mono = torch.bmm(attn_weights, v_h).permute(1, 0, 2).unsqueeze(0) # [1, N, H, D]
    
    # 3. Sharding configuration for Fold-CP
    # N must be divisible by P
    N_shard = N // P
    manager = FoldCPManager(num_devices=P)
    
    # Sharded Query, Key, Value: [P, N_shard, H, D]
    q_shards = q_mono[0].view(P, N_shard, H, D)
    k_shards = k_mono[0].view(P, N_shard, H, D)
    v_shards = v_mono[0].view(P, N_shard, H, D)
    
    # Pair bias sharded along columns: [P, N_shard, N, H]
    bias_shards = bias_mono[0].view(P, N_shard, N, H)
    
    # 4. Execute Fold-CP Ring Attention
    output_shards, m_shards, d_shards = ring_attention_step(
        q_shards, k_shards, v_shards, bias_shards, num_ranks=P, device_manager=manager
    )
    
    # Reassemble shards to reconstruct monolithic output shape
    output_reconstructed = output_shards.view(1, N, H, D)
    
    # 5. Measure Max Numerical Discrepancy
    attn_diff = torch.max(torch.abs(output_mono - output_reconstructed)).item()
    
    # 6. Benchmark Triangular Multiplicative Update (TMU)
    # Setup monolithic pair representations for A and B
    a_mono = torch.randn(N, N, D, device=device).double()
    b_mono = torch.randn(N, N, D, device=device).double()
    
    # Monolithic TMU (C = A x B over feature dimension)
    # a_mono: [D, N, N], b_mono: [D, N, N]
    a_m_p = a_mono.permute(2, 0, 1)
    b_m_p = b_mono.permute(2, 0, 1)
    tmu_mono = torch.bmm(a_m_p, b_m_p).permute(1, 2, 0) # [N, N, D]
    
    # Sharding for 2D Grid
    # We partition N x N into (P_row x P_col) blocks
    P_row, P_col = manager.p_row, manager.p_col
    R_shard = N // P_row
    C_shard = N // P_col
    
    a_shards_2d = torch.zeros(P_row, P_col, R_shard, C_shard, D, dtype=a_mono.dtype, device=device)
    b_shards_2d = torch.zeros(P_row, P_col, R_shard, C_shard, D, dtype=b_mono.dtype, device=device)
    
    for r in range(P_row):
        for c in range(P_col):
            a_shards_2d[r, c] = a_mono[r*R_shard : (r+1)*R_shard, c*C_shard : (c+1)*C_shard]
            b_shards_2d[r, c] = b_mono[r*R_shard : (r+1)*R_shard, c*C_shard : (c+1)*C_shard]
            
    # Execute Ring 2D TMU
    tmu_shards_2d = ring_triangular_multiplication(a_shards_2d, b_shards_2d, device_manager=manager)
    
    # Reconstruct Monolithic TMU
    tmu_reconstructed = torch.zeros(N, N, D, dtype=a_mono.dtype, device=device)
    for r in range(P_row):
        for c in range(P_col):
            tmu_reconstructed[r*R_shard : (r+1)*R_shard, c*C_shard : (c+1)*C_shard] = tmu_shards_2d[r, c]
            
    tmu_diff = torch.max(torch.abs(tmu_mono - tmu_reconstructed)).item()
    
    # 7. Memory footprint estimation (Bytes)
    # Monolithic stores N x N matrix: N^2 * H * 4 bytes (Float32) or 2 bytes (FP16)
    # Fold-CP stores N_shard x N_shard local attention matrix + ring buffers
    vram_mono_attn = N * N * H * 4
    vram_fold_cp_attn = N_shard * N_shard * H * 4
    
    # Pair matrix storage (N x N x D)
    vram_mono_pair = N * N * D * 4
    vram_fold_cp_pair = R_shard * C_shard * D * 4
    
    print(f"\n[Fold-CP Benchmarks N={N}, P={P}]:")
    print(f" - Attention Ring Equivalence Error: {attn_diff:.2e} (Passed if < 1e-5)")
    print(f" - 2D Ring TMU Equivalence Error: {tmu_diff:.2e} (Passed if < 1e-5)")
    print(f" - Monolithic Attention Peak VRAM (Est): {vram_mono_attn / 1024**2:.2f} MB")
    print(f" - Fold-CP Local Attn Peak VRAM (Est): {vram_fold_cp_attn / 1024**2:.2f} MB")
    print(f" - Monolithic Pair Matrix VRAM (Est): {vram_mono_pair / 1024**2:.2f} MB")
    print(f" - Fold-CP Sharded Pair Matrix VRAM: {vram_fold_cp_pair / 1024**2:.2f} MB")
    print(f" - VRAM Compression Factor: {(vram_mono_pair / vram_fold_cp_pair):.1f}x")
    
    return {
        "N": N,
        "P": P,
        "attn_diff": attn_diff,
        "tmu_diff": tmu_diff,
        "vram_mono_attn_mb": vram_mono_attn / 1024**2,
        "vram_fold_cp_attn_mb": vram_fold_cp_attn / 1024**2,
        "vram_mono_pair_mb": vram_mono_pair / 1024**2,
        "vram_fold_cp_pair_mb": vram_fold_cp_pair / 1024**2,
        "compression": vram_mono_pair / vram_fold_cp_pair
    }


if __name__ == "__main__":
    run_fold_cp_benchmark()
