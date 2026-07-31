import time
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import gradcheck

# Import the modules we implemented
from low_rank_pair_representation import LowRankPairUpdater, FullRankPairUpdater, LowRankTensorProduct

def verify_gradients():
    print("=== Step 1: Verifying Gradients via gradcheck ===")
    # Use CPU and double precision for numerical stability in gradcheck
    device = torch.device("cpu")
    B, N, d, D_pair = 2, 5, 4, 6
    
    X = torch.randn(B, N, d, dtype=torch.float64, device=device, requires_grad=True)
    Y = torch.randn(B, N, d, dtype=torch.float64, device=device, requires_grad=True)
    W = torch.randn(D_pair, d, dtype=torch.float64, device=device, requires_grad=True)
    
    # Verify the custom autograd function
    try:
        test_passed = gradcheck(LowRankTensorProduct.apply, (X, Y, W), eps=1e-6, atol=1e-4)
        print(f"Gradcheck result: {test_passed} (Custom gradients match finite differences perfectly!)\n")
        return True
    except Exception as e:
        print(f"Gradcheck FAILED with error:\n{e}\n")
        return False

def benchmark_memory_and_speed(device):
    print("=== Step 2: Benchmarking Memory and Speed Scaling ===")
    
    # Parameters
    B = 1
    D_SEQ = 64
    D_PAIR = 128
    D_MID = 32
    RANK = 16
    
    seq_lengths = [100, 200, 500, 1000, 1500]
    results = []
    
    for N in seq_lengths:
        print(f"Running benchmark for N = {N}...")
        
        # Prepare inputs
        s = torch.randn(B, N, D_SEQ, device=device)
        
        # Initialize modules
        full_rank_mod = FullRankPairUpdater(d_seq=D_SEQ, d_pair=D_PAIR, d_mid=D_MID).to(device)
        low_rank_mod = LowRankPairUpdater(d_seq=D_SEQ, d_pair=D_PAIR, rank=RANK).to(device)
        
        # --- Full Rank Benchmark ---
        # Warmup
        for _ in range(3):
            out = full_rank_mod(s)
            loss = out.sum()
            loss.backward()
            full_rank_mod.zero_grad()
            
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        # Measure forward memory and speed
        m_start_f = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
        t0 = time.perf_counter()
        out_f = full_rank_mod(s)
        if device.type == 'mps':
            torch.mps.synchronize()
        t_f_forward = (time.perf_counter() - t0) * 1000.0
        m_forward_f = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
        
        # Calculate saved activations
        output_mem_f = out_f.element_size() * out_f.nelement()
        act_mem_f = max(0, m_forward_f - m_start_f - output_mem_f)
        
        # Measure backward speed
        loss_f = out_f.sum()
        t0 = time.perf_counter()
        loss_f.backward()
        if device.type == 'mps':
            torch.mps.synchronize()
        t_f_backward = (time.perf_counter() - t0) * 1000.0
        
        # Clean up full rank
        full_rank_mod.zero_grad()
        del out_f, loss_f
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
            
        # --- Low Rank Benchmark ---
        # Warmup
        for _ in range(3):
            out = low_rank_mod(s)
            loss = out.sum()
            loss.backward()
            low_rank_mod.zero_grad()
            
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
            
        # Measure forward memory and speed
        m_start_l = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
        t0 = time.perf_counter()
        out_l = low_rank_mod(s)
        if device.type == 'mps':
            torch.mps.synchronize()
        t_l_forward = (time.perf_counter() - t0) * 1000.0
        m_forward_l = torch.mps.current_allocated_memory() if device.type == 'mps' else 0
        
        # Calculate saved activations
        output_mem_l = out_l.element_size() * out_l.nelement()
        act_mem_l = max(0, m_forward_l - m_start_l - output_mem_l)
        
        # Measure backward speed
        loss_l = out_l.sum()
        t0 = time.perf_counter()
        loss_l.backward()
        if device.type == 'mps':
            torch.mps.synchronize()
        t_l_backward = (time.perf_counter() - t0) * 1000.0
        
        # Clean up low rank
        low_rank_mod.zero_grad()
        del out_l, loss_l
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
            
        # --- Fidelity (MSE) Measurement ---
        # Train low-rank updater to fit full-rank outputs for 50 steps
        optimizer = torch.optim.Adam(low_rank_mod.parameters(), lr=0.01)
        
        # Target
        with torch.no_grad():
            target = full_rank_mod(s)
            target_var = torch.var(target).item()
            
        initial_mse = 0.0
        for step in range(50):
            optimizer.zero_grad()
            pred = low_rank_mod(s)
            loss = torch.mean((pred - target)**2)
            if step == 0:
                initial_mse = loss.item()
            loss.backward()
            optimizer.step()
            
        final_mse = loss.item()
        rel_mse = final_mse / (target_var + 1e-8)
        
        results.append({
            'N': N,
            'Full_Fwd_Time_ms': t_f_forward,
            'Full_Bwd_Time_ms': t_f_backward,
            'Full_Act_Mem_MB': act_mem_f / (1024**2) if device.type == 'mps' else np.nan,
            'Low_Fwd_Time_ms': t_l_forward,
            'Low_Bwd_Time_ms': t_l_backward,
            'Low_Act_Mem_MB': act_mem_l / (1024**2) if device.type == 'mps' else np.nan,
            'Initial_MSE': initial_mse,
            'Final_MSE': final_mse,
            'Relative_MSE': rel_mse
        })
        
        print(f"  Full-Rank Saved Act Mem: {act_mem_f / (1024**2):.3f} MB | Low-Rank Saved Act Mem: {act_mem_l / (1024**2):.3f} MB")
        print(f"  Full-Rank Time (F/B): {t_f_forward:.1f}/{t_f_backward:.1f} ms | Low-Rank Time (F/B): {t_l_forward:.1f}/{t_l_backward:.1f} ms")
        print(f"  Fidelity (MSE): Initial={initial_mse:.6f} -> Final={final_mse:.6f} (Rel MSE: {rel_mse:.4%})")
        print("-" * 50)
        
    df = pd.DataFrame(results)
    return df

def generate_plots(df):
    print("=== Step 3: Generating Benchmark Visualizations ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Activation Memory Scaling
    axes[0].plot(df['N'], df['Full_Act_Mem_MB'], 'o-', label='Full-Rank (OPM)', color='tab:red')
    axes[0].plot(df['N'], df['Low_Act_Mem_MB'], 's-', label='Low-Rank (Ours)', color='tab:blue')
    axes[0].set_title('Saved Activation VRAM vs. Sequence Length')
    axes[0].set_xlabel('Sequence Length (N)')
    axes[0].set_ylabel('Activation VRAM (MB)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # 2. Computation Time Scaling (Forward + Backward)
    axes[1].plot(df['N'], df['Full_Fwd_Time_ms'] + df['Full_Bwd_Time_ms'], 'o-', label='Full-Rank (OPM)', color='tab:red')
    axes[1].plot(df['N'], df['Low_Fwd_Time_ms'] + df['Low_Bwd_Time_ms'], 's-', label='Low-Rank (Ours)', color='tab:blue')
    axes[1].set_title('Total Run Time (Fwd + Bwd) vs. Sequence Length')
    axes[1].set_xlabel('Sequence Length (N)')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    # 3. Approximation Fidelity
    axes[2].plot(df['N'], df['Relative_MSE'] * 100, 'o-', color='tab:green')
    axes[2].set_title('Low-Rank Approximation Relative MSE (%)')
    axes[2].set_xlabel('Sequence Length (N)')
    axes[2].set_ylabel('Relative MSE (%)')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = "benchmark_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved benchmark plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Check gradient correctness
    grad_ok = verify_gradients()
    if not grad_ok:
        print("Gradient check failed. Exiting.")
        exit(1)
        
    # Use MPS device if available for memory profiling, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running benchmarks on device: {device}\n")
    
    df = benchmark_memory_and_speed(device)
    
    # Save CSV
    csv_path = "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark results to {csv_path}")
    
    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(df.to_string(index=False, formatters={
        'Full_Fwd_Time_ms': '{:.2f}'.format,
        'Full_Bwd_Time_ms': '{:.2f}'.format,
        'Full_Act_Mem_MB': '{:.4f}'.format,
        'Low_Fwd_Time_ms': '{:.2f}'.format,
        'Low_Bwd_Time_ms': '{:.2f}'.format,
        'Low_Act_Mem_MB': '{:.4f}'.format,
        'Initial_MSE': '{:.6f}'.format,
        'Final_MSE': '{:.6f}'.format,
        'Relative_MSE': '{:.4%}'.format
    }))
    
    # Generate visualization
    generate_plots(df)
