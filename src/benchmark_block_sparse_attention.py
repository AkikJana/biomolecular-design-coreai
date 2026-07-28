import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
from src.block_sparse_attention import DynamicBlockSparseAttention

class FullAttention(nn.Module):
    """
    Standard full O(N^2) monolithic Self-Attention.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, X):
        batch_size, seq_len, dim = X.shape
        Q = self.q_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        O = torch.matmul(attn_weights, V)
        
        O = O.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out_proj(O)

def align_weights(sparse_model, dense_model):
    """
    Helper function to copy weights from sparse module to dense module to verify correctness.
    """
    dense_model.q_proj.weight.data.copy_(sparse_model.q_proj.weight.data)
    dense_model.q_proj.bias.data.copy_(sparse_model.q_proj.bias.data)
    dense_model.k_proj.weight.data.copy_(sparse_model.k_proj.weight.data)
    dense_model.k_proj.bias.data.copy_(sparse_model.k_proj.bias.data)
    dense_model.v_proj.weight.data.copy_(sparse_model.v_proj.weight.data)
    dense_model.v_proj.bias.data.copy_(sparse_model.v_proj.bias.data)
    dense_model.out_proj.weight.data.copy_(sparse_model.out_proj.weight.data)
    dense_model.out_proj.bias.data.copy_(sparse_model.out_proj.bias.data)

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def get_allocated_memory(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    elif device.type == "mps":
        return torch.mps.current_allocated_memory()
    else:
        return 0

def run_correctness_check(device):
    print("==================================================")
    print("           RUNNING CORRECTNESS CHECK              ")
    print("==================================================")
    
    batch_size = 4
    seq_len = 256
    dim = 128
    num_heads = 4
    block_size = 64
    
    X = torch.randn(batch_size, seq_len, dim, device=device)
    
    sparse_model = DynamicBlockSparseAttention(
        dim=dim, num_heads=num_heads, block_size=block_size
    ).to(device)
    
    dense_model = FullAttention(
        dim=dim, num_heads=num_heads
    ).to(device)
    
    align_weights(sparse_model, dense_model)
    
    # Forward pass forcing all blocks active
    with torch.no_grad():
        sparse_out = sparse_model(X, force_all_active=True)
        dense_out = dense_model(X)
        
    diff = torch.abs(sparse_out - dense_out)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    is_correct = max_diff < 1e-4
    if is_correct:
        print("Verification: SUCCESS (Block-Sparse Attention is mathematically equivalent to Full Attention when all blocks are active)")
    else:
        print("Verification: FAILED")
    return is_correct

def benchmark_performance(device, output_dir):
    print("\n==================================================")
    print("         BENCHMARKING WALL-CLOCK & MEMORY         ")
    print("==================================================")
    
    batch_size = 8
    dim = 256
    num_heads = 8
    block_size = 64
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        X = torch.randn(batch_size, seq_len, dim, device=device)
        
        # Instantiate models
        dense_model = FullAttention(dim=dim, num_heads=num_heads).to(device)
        sparse_model = DynamicBlockSparseAttention(
            dim=dim, num_heads=num_heads, block_size=block_size, threshold=0.5
        ).to(device)
        
        # Align weights for consistency
        align_weights(sparse_model, dense_model)
        
        # Profile Dense Forward
        # Warmup
        for _ in range(5):
            _ = dense_model(X)
        sync_device(device)
        
        mem_start_d = get_allocated_memory(device)
        t_start_d = time.perf_counter()
        for _ in range(30):
            _ = dense_model(X)
        sync_device(device)
        t_end_d = time.perf_counter()
        mem_end_d = get_allocated_memory(device)
        
        time_dense_fw = (t_end_d - t_start_d) / 30 * 1000.0
        mem_dense_fw = max(0, mem_end_d - mem_start_d) / (1024 * 1024)
        
        # Profile Dense Backward
        # Warmup
        dense_out = dense_model(X)
        loss_d = dense_out.sum()
        for _ in range(5):
            loss_d.backward(retain_graph=True)
        sync_device(device)
        
        t_start_d_bw = time.perf_counter()
        for _ in range(30):
            dense_model.zero_grad()
            dense_out = dense_model(X)
            loss_d = dense_out.sum()
            loss_d.backward(retain_graph=True)
        sync_device(device)
        t_end_d_bw = time.perf_counter()
        
        time_dense_bw = (t_end_d_bw - t_start_d_bw) / 30 * 1000.0
        
        # Profile Sparse Forward (Gated)
        # Warmup
        for _ in range(5):
            _ = sparse_model(X)
        sync_device(device)
        
        mem_start_s = get_allocated_memory(device)
        t_start_s = time.perf_counter()
        for _ in range(30):
            _ = sparse_model(X)
        sync_device(device)
        t_end_s = time.perf_counter()
        mem_end_s = get_allocated_memory(device)
        
        time_sparse_fw = (t_end_s - t_start_s) / 30 * 1000.0
        mem_sparse_fw = max(0, mem_end_s - mem_start_s) / (1024 * 1024)
        
        # Get active block details
        with torch.no_grad():
            _, _, gate_prob = sparse_model(X, return_gate_info=True)
            sparsity = sparse_model.last_sparsity_ratio
            
        # Profile Sparse Backward
        # Warmup
        sparse_out = sparse_model(X)
        loss_s = sparse_out.sum()
        for _ in range(5):
            loss_s.backward(retain_graph=True)
        sync_device(device)
        
        t_start_s_bw = time.perf_counter()
        for _ in range(30):
            sparse_model.zero_grad()
            sparse_out = sparse_model(X)
            loss_s = sparse_out.sum() + sparse_model.last_gate_loss
            loss_s.backward(retain_graph=True)
        sync_device(device)
        t_end_s_bw = time.perf_counter()
        
        time_sparse_bw = (t_end_s_bw - t_start_s_bw) / 30 * 1000.0
        
        # Calculate speedups
        fw_speedup = time_dense_fw / time_sparse_fw
        bw_speedup = time_dense_bw / time_sparse_bw
        total_speedup = (time_dense_fw + time_dense_bw) / (time_sparse_fw + time_sparse_bw)
        
        print(f"  Dense Attention  - FW: {time_dense_fw:6.2f} ms | BW: {time_dense_bw:6.2f} ms")
        print(f"  Sparse Attention - FW: {time_sparse_fw:6.2f} ms | BW: {time_sparse_bw:6.2f} ms (Sparsity: {sparsity*100:.1f}% active blocks)")
        print(f"  Speedup          - FW: {fw_speedup:5.2f}x | BW: {bw_speedup:5.2f}x | Total: {total_speedup:5.2f}x")
        
        results.append({
            "seq_len": seq_len,
            "dense_fw_ms": time_dense_fw,
            "dense_bw_ms": time_dense_bw,
            "dense_mem_mb": mem_dense_fw,
            "sparse_fw_ms": time_sparse_fw,
            "sparse_bw_ms": time_sparse_bw,
            "sparse_mem_mb": mem_sparse_fw,
            "sparsity_ratio": sparsity,
            "total_speedup": total_speedup
        })
        
    # Save results to json
    res_path = os.path.join(output_dir, "benchmark_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved benchmark results to {res_path}")
    return results

def train_gating_network(device, dim=128, num_heads=4, seq_len=512, block_size=64, epochs=50):
    print("\n==================================================")
    print("      TRAINING DYNAMIC GATING NETWORK              ")
    print("==================================================")
    
    # Generate some dummy data with a synthetic block-sparse correlation
    # We will simulate a situation where queries only pay attention to local blocks and a few key blocks
    batch_size = 16
    X = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Create a teacher model
    teacher = FullAttention(dim=dim, num_heads=num_heads).to(device)
    teacher.eval()
    
    # Create the block-sparse student
    student = DynamicBlockSparseAttention(
        dim=dim,
        num_heads=num_heads,
        block_size=block_size,
        gate_dim=32,
        threshold=0.5,
        sparsity_penalty=0.03 # weight to encourage sparsity
    ).to(device)
    
    # Initialize projection weights to be identical
    align_weights(student, teacher)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    
    print("Starting training loop to reconstruct full attention using a sparse model...")
    for epoch in range(1, epochs + 1):
        student.train()
        optimizer.zero_grad()
        
        # Teacher output is the target
        with torch.no_grad():
            target_out = teacher(X)
            
        # Student sparse output
        student_out, gate_decision, gate_prob = student(X, return_gate_info=True)
        
        # Reconstruction MSE loss + Gating sparsity penalty
        recon_loss = F.mse_loss(student_out, target_out)
        sparsity_loss = student.last_gate_loss
        
        loss = recon_loss + sparsity_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Total Loss: {loss.item():.4f} | MSE Loss: {recon_loss.item():.5f} | Sparsity Penalty: {sparsity_loss.item():.4f} | Sparsity Ratio: {student.last_sparsity_ratio*100:.1f}% active blocks")
            
    print("\nTraining completed!")
    
    # Evaluate final reconstruction error on a test set
    student.eval()
    X_test = torch.randn(batch_size, seq_len, dim, device=device)
    with torch.no_grad():
        target_test = teacher(X_test)
        student_test, final_gate, _ = student(X_test, return_gate_info=True)
        test_mse = F.mse_loss(student_test, target_test).item()
        
    print(f"Test Set Reconstruction MSE: {test_mse:.6f}")
    print(f"Final Sparsity Ratio: {student.last_sparsity_ratio*100:.1f}% active blocks")
    return test_mse, student.last_sparsity_ratio

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Output directory for results (artifacts folder)
    output_dir = "/Users/akikjana/.gemini/antigravity-cli/brain/4cdb7261-e55b-4efc-9ffa-c6509d76c9c2"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Correctness check
    is_correct = run_correctness_check(device)
    
    # 2. Benchmark performance
    bench_results = benchmark_performance(device, output_dir)
    
    # 3. Train gating network
    test_mse, final_sparsity = train_gating_network(device)
    
    # Save a summary report
    summary = {
        "device": str(device),
        "correctness_verified": is_correct,
        "final_test_mse": test_mse,
        "final_sparsity": final_sparsity,
        "benchmark": bench_results
    }
    
    summary_path = os.path.join(output_dir, "summary_block_sparse.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved summary report to {summary_path}")

if __name__ == "__main__":
    main()
