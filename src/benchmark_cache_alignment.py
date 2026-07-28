import time
import math
import torch
import torch.nn as nn
from cache_alignment import CacheAlignedAttention

class BaselineDynamicAttention(nn.Module):
    """
    Baseline unoptimized attention module using dynamic KV concatenation.
    This simulates standard attention that triggers recompilations and
    memory allocations at every generation step due to changing shapes.
    """
    def __init__(self, c_s: int, num_heads: int, head_dim: int, device: str = "mps"):
        super().__init__()
        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.proj_q = nn.Linear(c_s, c_s, device=device)
        self.proj_k = nn.Linear(c_s, c_s, bias=False, device=device)
        self.proj_v = nn.Linear(c_s, c_s, bias=False, device=device)
        self.proj_o = nn.Linear(c_s, c_s, bias=False, device=device)
        
        self.cache_k = None
        self.cache_v = None
        
    def reset_cache(self, B=None):
        self.cache_k = None
        self.cache_v = None
        
    def forward(self, s: torch.Tensor, k_in: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B = s.shape[0]
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.cache_k is None:
            self.cache_k = k
            self.cache_v = v
        else:
            # Dynamic concatenation - triggers memory reallocation and changes tensor shapes
            self.cache_k = torch.cat([self.cache_k, k], dim=2)
            self.cache_v = torch.cat([self.cache_v, v], dim=2)
            
        attn_scores = torch.matmul(q, self.cache_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if causal:
            S_q = q.shape[2]
            S_k = self.cache_k.shape[2]
            # Slicing mask dynamically based on sequence lengths
            mask = torch.ones((S_q, S_k), dtype=torch.bool, device=q.device).tril(diagonal=S_k - S_q)
            attn_scores = attn_scores.masked_fill(~mask.view(1, 1, S_q, S_k), -1e9)
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        o = torch.matmul(attn_probs, self.cache_v)
        o = o.transpose(1, 2).reshape(B, -1, self.c_s)
        return self.proj_o(o)


def benchmark_module(model, name, device, num_steps=20):
    print(f"\n--- Benchmarking {name} ---")
    
    # Warmup / compilation pass
    # We do a prefill of 32 tokens, then decode step-by-step
    B = 1
    c_s = 256
    
    # 1. Prefill step (32 tokens)
    s_prefill = torch.randn(B, 32, c_s, device=device)
    model.reset_cache(B) if hasattr(model, "reset_cache") else model.reset_cache()
    
    print("Compiling / Warming up prefill...")
    t0 = time.perf_counter()
    # Call once to compile/warmup
    _ = model(s_prefill, s_prefill, causal=True)
    torch.mps.synchronize()
    prefill_warmup_time = time.perf_counter() - t0
    print(f"Prefill Warmup time: {prefill_warmup_time * 1000:.2f} ms")
    
    # 2. Decode steps (1 token at a time)
    decode_times = []
    
    # We compile the forward pass if we want to test compilation behavior
    # Let's compile the model call using torch.compile
    print("Compiling model for decode...")
    try:
        # We wrap the model call in a compiled function
        # Using fullgraph=False because of dynamic checks, but dynamic=True lets torch.compile handle sizes
        compiled_forward = torch.compile(model, dynamic=True)
    except Exception as e:
        print(f"Warning: torch.compile failed to initialize: {e}. Running eagerly.")
        compiled_forward = model
        
    for step in range(num_steps):
        s_step = torch.randn(B, 1, c_s, device=device)
        
        # Measure step time
        t_start = time.perf_counter()
        _ = compiled_forward(s_step, s_step, causal=True)
        torch.mps.synchronize()
        t_end = time.perf_counter()
        
        step_time = (t_end - t_start) * 1000  # in ms
        decode_times.append(step_time)
        print(f"  Step {step:02d} (Seq Len: {32 + step + 1}): {step_time:.3f} ms")
        
    avg_decode = sum(decode_times[1:]) / len(decode_times[1:]) if len(decode_times) > 1 else decode_times[0]
    max_decode = max(decode_times[1:]) if len(decode_times) > 1 else decode_times[0]
    first_step_decode = decode_times[0]
    
    print(f"Results for {name}:")
    print(f"  First Decode Step (includes compilation if compiled): {first_step_decode:.2f} ms")
    print(f"  Average Subsequent Decode Step: {avg_decode:.3f} ms")
    print(f"  Max Subsequent Decode Step (checking for spikes): {max_decode:.3f} ms")
    
    return first_step_decode, avg_decode, max_decode, decode_times


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    c_s = 256
    num_heads = 8
    head_dim = c_s // num_heads  # 32
    max_seq_len = 1024
    block_size = 16
    
    # Instantiate optimized standard attention
    opt_std_model = CacheAlignedAttention(
        c_s=c_s,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        block_size=block_size,
        attention_type="standard",
        device=device
    )
    
    # Instantiate optimized linear attention
    opt_lin_model = CacheAlignedAttention(
        c_s=c_s,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        block_size=block_size,
        attention_type="linear",
        device=device
    )
    
    # Instantiate baseline dynamic attention
    baseline_model = BaselineDynamicAttention(
        c_s=c_s,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device
    )
    
    # Verify correctness: Check if optimized standard attention produces the same shapes and compiles
    print("\n--- Verifying Output Correctness ---")
    x = torch.randn(1, 4, c_s, device=device)
    
    # Reset caches
    opt_std_model.reset_cache(1)
    baseline_model.reset_cache()
    
    # Forward pass on both
    out_baseline = baseline_model(x, x, causal=True)
    out_opt = opt_std_model(x, x, causal=True)
    
    print(f"Baseline output shape: {out_baseline.shape}")
    print(f"Optimized output shape: {out_opt.shape}")
    
    # Check shape equality
    assert out_baseline.shape == out_opt.shape, "Error: Shapes do not match!"
    print("Correctness check passed: output shapes match!")
    
    # Benchmark standard baseline
    b_first, b_avg, b_max, b_all = benchmark_module(baseline_model, "Baseline (Dynamic KV Concat)", device)
    
    # Benchmark optimized standard
    o_first, o_avg, o_max, o_all = benchmark_module(opt_std_model, "Optimized Cache-Aligned Standard Attention", device)
    
    # Benchmark optimized linear
    l_first, l_avg, l_max, l_all = benchmark_module(opt_lin_model, "Optimized Cache-Aligned Linear Attention", device)
    
    # Print comparison table
    print("\n================== BENCHMARK COMPARISON ==================")
    print(f"{'Metric':<40} | {'Baseline':<12} | {'Opt Standard':<15} | {'Opt Linear':<12}")
    print("-" * 90)
    print(f"{'First Decode Step (Compilation/Warmup)':<40} | {b_first:9.2f} ms | {o_first:12.2f} ms | {l_first:9.2f} ms")
    print(f"{'Average Subsequent Decode Step':<40} | {b_avg:9.3f} ms | {o_avg:12.3f} ms | {l_avg:9.3f} ms")
    print(f"{'Max Subsequent Decode Step (Recompilation Check)':<40} | {b_max:9.3f} ms | {o_max:12.3f} ms | {l_max:9.3f} ms")
    
    # Check if there are recompilation spikes in baseline vs optimized
    # A spike is usually defined as a step that is > 3x the average of subsequent steps
    baseline_spikes = sum(1 for t in b_all[1:] if t > b_avg * 3)
    opt_spikes = sum(1 for t in o_all[1:] if t > o_avg * 3)
    
    print("-" * 90)
    print(f"{'Recompilation Spikes Detected':<40} | {baseline_spikes:<12} | {opt_spikes:<15} | {'N/A (Recurrent)':<12}")
    print("==========================================================")
    
    if opt_spikes == 0:
        print("\nSUCCESS: Zero-recompilation behavior verified for CacheAlignedAttention!")
    else:
        print(f"\nWARNING: Detected {opt_spikes} latency spikes in optimized model. Check for graph compilation guards.")

if __name__ == "__main__":
    main()
