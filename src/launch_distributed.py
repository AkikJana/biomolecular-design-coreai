import os
import sys
import time
import math
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fold_cp_sharding import FoldCPManager, ring_attention_step, ring_triangular_multiplication

def find_free_port():
    """Finds a free port on localhost to avoid address collision."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def dist_ring_attention_step(q_local, k_local, v_local, bias_local, world_size, rank):
    """Executes a distributed Ring Attention step on the current rank.
    
    Communicates Key and Value shards in a ring using non-blocking send/receive
    while performing local attention computation.
    """
    N_shard, H, D = q_local.shape
    device = q_local.device
    
    # Initialize online softmax accumulators
    out_local = torch.zeros_like(q_local)
    m_local = torch.full((N_shard, H, 1), -float('inf'), device=device, dtype=q_local.dtype)
    d_local = torch.zeros((N_shard, H, 1), device=device, dtype=q_local.dtype)
    
    current_k = k_local.clone()
    current_v = v_local.clone()
    
    scale = 1.0 / math.sqrt(D)
    q_scaled = q_local * scale
    
    comp_time = 0.0
    comm_time = 0.0
    
    for step in range(world_size):
        kv_rank = (rank - step) % world_size
        
        # 1. Initiate ring communication for the next step (if not the last step)
        if step < world_size - 1:
            comm_start = time.perf_counter()
            next_k = torch.empty_like(current_k)
            next_v = torch.empty_like(current_v)
            reqs = []
            
            # Post non-blocking receives from previous rank
            reqs.append(dist.irecv(next_k, src=(rank - 1) % world_size))
            reqs.append(dist.irecv(next_v, src=(rank - 1) % world_size))
            
            # Post non-blocking sends to next rank
            reqs.append(dist.isend(current_k, dst=(rank + 1) % world_size))
            reqs.append(dist.isend(current_v, dst=(rank + 1) % world_size))
            comm_time += time.perf_counter() - comm_start
            
        # 2. Computation block
        comp_start = time.perf_counter()
        
        # Raw attention logits: [H, N_shard, D] x [H, D, N_shard] -> [H, N_shard, N_shard]
        q_h = q_scaled.permute(1, 0, 2)
        k_h = current_k.permute(1, 0, 2)
        logits = torch.bmm(q_h, k_h.transpose(-1, -2))
        logits = logits.permute(1, 2, 0) # [N_shard, N_shard, H]
        
        # Add bias slice if available
        if bias_local is not None:
            bias_slice = bias_local[:, kv_rank * N_shard : (kv_rank + 1) * N_shard, :]
            logits = logits + bias_slice
            
        # Online softmax update
        logits_max, _ = torch.max(logits, dim=1, keepdim=True) # [N_shard, 1, H]
        logits_max = logits_max.permute(0, 2, 1) # [N_shard, H, 1]
        
        m_new = torch.maximum(m_local, logits_max)
        exp_logits = torch.exp(logits - logits_max.permute(0, 2, 1)) # [N_shard, N_shard, H]
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True).permute(0, 2, 1) # [N_shard, H, 1]
        
        alpha = torch.exp(m_local - m_new)
        alpha = torch.where(torch.isinf(m_local), torch.ones_like(alpha), alpha)
        exp_scale = torch.exp(logits_max - m_new)
        
        d_new = alpha * d_local + exp_scale * exp_sum
        
        exp_h = exp_logits.permute(2, 0, 1) # [H, N_shard, N_shard]
        v_h = current_v.permute(1, 0, 2)   # [H, N_shard, D]
        local_out = torch.bmm(exp_h, v_h).permute(1, 0, 2) # [N_shard, H, D]
        local_out_scaled = exp_scale * local_out
        
        out_local = alpha * out_local + local_out_scaled
        m_local = m_new
        d_local = d_new
        
        comp_time += time.perf_counter() - comp_start
        
        # 3. Wait for communication to complete
        if step < world_size - 1:
            comm_wait_start = time.perf_counter()
            for req in reqs:
                req.wait()
            comm_time += time.perf_counter() - comm_wait_start
            current_k = next_k
            current_v = next_v
            
    out_local = out_local / d_local
    return out_local, comp_time, comm_time

def dist_ring_triangular_multiplication(a_local, b_local, row_group, col_group, P_row, P_col, r, c):
    """Executes a 2D Ring Triangular Multiplication step on the current rank.
    
    Gathers shards along rows and columns of the process grid, then computes the
    block multiplications locally.
    """
    # 1. Communication (All-gather along row/col process groups)
    comm_start = time.perf_counter()
    gathered_a = [torch.zeros_like(a_local) for _ in range(P_col)]
    gathered_b = [torch.zeros_like(b_local) for _ in range(P_row)]
    
    dist.all_gather(gathered_a, a_local, group=row_group)
    dist.all_gather(gathered_b, b_local, group=col_group)
    comm_time = time.perf_counter() - comm_start
    
    # 2. Computation
    comp_start = time.perf_counter()
    c_local = torch.zeros_like(a_local)
    
    for step in range(P_row):
        k = (r + c + step) % P_row
        a_block = gathered_a[k]
        b_block = gathered_b[k]
        
        # Permute to make batch size D: [D, S, S]
        a_b = a_block.permute(2, 0, 1)
        b_b = b_block.permute(2, 0, 1)
        prod = torch.bmm(a_b, b_b)
        
        c_local += prod.permute(1, 2, 0)
        
    comp_time = time.perf_counter() - comp_start
    
    return c_local, comp_time, comm_time

def run_benchmark_process(rank, world_size, port, results_dict):
    """Target function for spawned processes running the benchmark."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    if sys.platform == "darwin":
        os.environ['GLOO_SOCKET_IFNAME'] = 'lo0'
        os.environ['TP_SOCKET_IFNAME'] = 'lo0'
    
    # Check device and backend
    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        backend = "gloo"
        device = torch.device("cpu")
        
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Synchronize random seeds for data generation
    torch.manual_seed(42 + rank)
    
    # Benchmarking configurations
    N = 1024
    D = 64
    H = 4
    N_shard = N // world_size
    S = 128  # Block size for TMU
    
    # Setup sharded tensors for Ring Attention
    q_local = torch.randn(N_shard, H, D, device=device, dtype=torch.float64)
    k_local = torch.randn(N_shard, H, D, device=device, dtype=torch.float64)
    v_local = torch.randn(N_shard, H, D, device=device, dtype=torch.float64)
    bias_local = torch.randn(N_shard, N, H, device=device, dtype=torch.float64)
    
    # Setup sharded tensors for 2D TMU
    a_local = torch.randn(S, S, D, device=device, dtype=torch.float64)
    b_local = torch.randn(S, S, D, device=device, dtype=torch.float64)
    
    # Establish grid parameters
    P_row = int(math.sqrt(world_size))
    while world_size % P_row != 0:
        P_row -= 1
    P_col = world_size // P_row
    r = rank // P_col
    c = rank % P_col
    
    # Collective row/col process groups creation
    row_groups = []
    for i in range(P_row):
        ranks = [i * P_col + j for j in range(P_col)]
        group = dist.new_group(ranks)
        row_groups.append(group)

    col_groups = []
    for j in range(P_col):
        ranks = [i * P_col + j for i in range(P_row)]
        group = dist.new_group(ranks)
        col_groups.append(group)
        
    my_row_group = row_groups[r]
    my_col_group = col_groups[c]
    
    # ----------------------------------------------------
    # Correctness Verification against src/fold_cp_sharding.py
    # ----------------------------------------------------
    gathered_q = [torch.zeros_like(q_local) for _ in range(world_size)] if rank == 0 else None
    gathered_k = [torch.zeros_like(k_local) for _ in range(world_size)] if rank == 0 else None
    gathered_v = [torch.zeros_like(v_local) for _ in range(world_size)] if rank == 0 else None
    gathered_bias = [torch.zeros_like(bias_local) for _ in range(world_size)] if rank == 0 else None
    
    dist.gather(q_local, gathered_q, dst=0)
    dist.gather(k_local, gathered_k, dst=0)
    dist.gather(v_local, gathered_v, dst=0)
    dist.gather(bias_local, gathered_bias, dst=0)
    
    gathered_a = [torch.zeros_like(a_local) for _ in range(world_size)] if rank == 0 else None
    gathered_b = [torch.zeros_like(b_local) for _ in range(world_size)] if rank == 0 else None
    
    dist.gather(a_local, gathered_a, dst=0)
    dist.gather(b_local, gathered_b, dst=0)
    
    # Execute distributed steps
    out_local, _, _ = dist_ring_attention_step(q_local, k_local, v_local, bias_local, world_size, rank)
    tmu_local, _, _ = dist_ring_triangular_multiplication(a_local, b_local, my_row_group, my_col_group, P_row, P_col, r, c)
    
    gathered_out = [torch.zeros_like(out_local) for _ in range(world_size)] if rank == 0 else None
    dist.gather(out_local, gathered_out, dst=0)
    
    gathered_tmu = [torch.zeros_like(tmu_local) for _ in range(world_size)] if rank == 0 else None
    dist.gather(tmu_local, gathered_tmu, dst=0)
    
    if rank == 0:
        manager = FoldCPManager(num_devices=world_size)
        q_ref = torch.stack(gathered_q, dim=0)
        k_ref = torch.stack(gathered_k, dim=0)
        v_ref = torch.stack(gathered_v, dim=0)
        bias_ref = torch.stack(gathered_bias, dim=0)
        
        # Execute simulator Ring Attention
        ref_out, _, _ = ring_attention_step(q_ref, k_ref, v_ref, bias_ref, num_ranks=world_size, device_manager=manager)
        dist_out = torch.stack(gathered_out, dim=0)
        attn_error = torch.max(torch.abs(ref_out - dist_out)).item()
        
        # Execute simulator 2D Ring TMU
        a_shards_2d = torch.zeros(P_row, P_col, S, S, D, dtype=a_local.dtype)
        b_shards_2d = torch.zeros(P_row, P_col, S, S, D, dtype=b_local.dtype)
        for idx in range(world_size):
            r_idx = idx // P_col
            c_idx = idx % P_col
            a_shards_2d[r_idx, c_idx] = gathered_a[idx]
            b_shards_2d[r_idx, c_idx] = gathered_b[idx]
            
        ref_tmu = ring_triangular_multiplication(a_shards_2d, b_shards_2d, device_manager=manager)
        
        tmu_reconstructed = torch.zeros(P_row, P_col, S, S, D, dtype=tmu_local.dtype)
        for idx in range(world_size):
            r_idx = idx // P_col
            c_idx = idx % P_col
            tmu_reconstructed[r_idx, c_idx] = gathered_tmu[idx]
            
        tmu_error = torch.max(torch.abs(ref_tmu - tmu_reconstructed)).item()
        
        print(f"\n[Verification World Size = {world_size}]:")
        print(f" - Ring Attention Equivalence Error: {attn_error:.2e} (Passed if < 1e-6)")
        print(f" - 2D Ring TMU Equivalence Error: {tmu_error:.2e} (Passed if < 1e-6)")
        
        assert attn_error < 1e-6, "Ring Attention correctness validation failed."
        assert tmu_error < 1e-6, "2D Ring TMU correctness validation failed."
        
    # ----------------------------------------------------
    # Timing Benchmark
    # ----------------------------------------------------
    num_warmups = 5
    num_trials = 20
    
    # Warmups
    for _ in range(num_warmups):
        dist_ring_attention_step(q_local, k_local, v_local, bias_local, world_size, rank)
        dist_ring_triangular_multiplication(a_local, b_local, my_row_group, my_col_group, P_row, P_col, r, c)
        
    # Timed runs
    attn_comp_times = []
    attn_comm_times = []
    tmu_comp_times = []
    tmu_comm_times = []
    
    for _ in range(num_trials):
        _, comp_t, comm_t = dist_ring_attention_step(q_local, k_local, v_local, bias_local, world_size, rank)
        attn_comp_times.append(comp_t)
        attn_comm_times.append(comm_t)
        
        _, comp_t, comm_t = dist_ring_triangular_multiplication(a_local, b_local, my_row_group, my_col_group, P_row, P_col, r, c)
        tmu_comp_times.append(comp_t)
        tmu_comm_times.append(comm_t)
        
    # Compute rank averages
    mean_attn_comp = sum(attn_comp_times) / num_trials
    mean_attn_comm = sum(attn_comm_times) / num_trials
    mean_tmu_comp = sum(tmu_comp_times) / num_trials
    mean_tmu_comm = sum(tmu_comm_times) / num_trials
    
    # Sum timings across all ranks
    t_attn_comp = torch.tensor([mean_attn_comp], device=device)
    t_attn_comm = torch.tensor([mean_attn_comm], device=device)
    t_tmu_comp = torch.tensor([mean_tmu_comp], device=device)
    t_tmu_comm = torch.tensor([mean_tmu_comm], device=device)
    
    dist.reduce(t_attn_comp, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(t_attn_comm, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(t_tmu_comp, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(t_tmu_comm, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        results_dict[world_size] = {
            "attn_comp": t_attn_comp.item() / world_size,
            "attn_comm": t_attn_comm.item() / world_size,
            "tmu_comp": t_tmu_comp.item() / world_size,
            "tmu_comm": t_tmu_comm.item() / world_size
        }
        
    dist.destroy_process_group()

def plot_results(results):
    """Plots and saves the communication vs computation scaling curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    world_sizes = sorted(results.keys())
    
    attn_comp = [results[ws]["attn_comp"] * 1000 for ws in world_sizes]  # ms
    attn_comm = [results[ws]["attn_comm"] * 1000 for ws in world_sizes]  # ms
    tmu_comp = [results[ws]["tmu_comp"] * 1000 for ws in world_sizes]    # ms
    tmu_comm = [results[ws]["tmu_comm"] * 1000 for ws in world_sizes]    # ms
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(world_sizes))
    width = 0.35
    
    # 1. Ring Attention plot
    ax1.bar(x - width/2, attn_comp, width, label='Computation', color='#1f77b4')
    ax1.bar(x + width/2, attn_comm, width, label='Communication', color='#ff7f0e')
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Ring Attention Latency Scaling', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{ws} Ranks' for ws in world_sizes], fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. 2D Ring TMU plot
    ax2.bar(x - width/2, tmu_comp, width, label='Computation', color='#2ca02c')
    ax2.bar(x + width/2, tmu_comm, width, label='Communication', color='#d62728')
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('2D Ring TMU Latency Scaling', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{ws} Ranks' for ws in world_sizes], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Fold-CP Parallel Component Benchmarks (Computation vs Communication)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scaling_curves.png'))
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved scaling curves plot to: {plot_path}")
    
    # Print markdown table
    print("\nBenchmark Results Table (Latencies in milliseconds):")
    print("| World Size | Attention Comp (ms) | Attention Comm (ms) | TMU Comp (ms) | TMU Comm (ms) |")
    print("|------------|---------------------|---------------------|---------------|---------------|")
    for ws in world_sizes:
        print(f"| {ws:10d} | {results[ws]['attn_comp']*1000:19.3f} | {results[ws]['attn_comm']*1000:19.3f} | {results[ws]['tmu_comp']*1000:13.3f} | {results[ws]['tmu_comm']*1000:13.3f} |")

def main():
    manager = mp.Manager()
    results = manager.dict()
    
    world_sizes = [2, 4, 8]
    
    print("=========================================================================")
    print("Starting Distributed Benchmarks for Fold-CP Parallel Components")
    print("Environment: MacOS CPU (Gloo Fallback)")
    print("=========================================================================")
    
    for ws in world_sizes:
        port = find_free_port()
        print(f"\nLaunching {ws} processes...")
        # Use spawn method for multiprocess execution
        mp.spawn(
            run_benchmark_process,
            args=(ws, port, results),
            nprocs=ws,
            join=True
        )
        # Give sockets a brief moment to release
        time.sleep(1.0)
        
    # Build results copy
    res_copy = dict(results)
    plot_results(res_copy)
    print("Benchmarks finished successfully.")

if __name__ == "__main__":
    main()
