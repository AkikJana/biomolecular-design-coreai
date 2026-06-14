import os
import sys
import time
import json
import torch
import math
from typing import Dict, Any, List

# Ensure src path is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from speculative_flow_matching import SpeculativeFlowMatchingSampler, FlowMatchingODE

# -------------------------------------------------------------
# SIMULATED BIOMOLECULAR MODELS WITH ARTIFICIAL LATENCY
# -------------------------------------------------------------
TARGET_LATENCY = 0.010  # 10ms simulated GPU latency per target model evaluation
DRAFT_LATENCY = 0.001   # 1ms simulated GPU/CPU latency per draft model evaluation

def target_vector_field(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    # Simulate heavy model calculation delay
    time.sleep(TARGET_LATENCY * (1 if t.ndim == 0 else len(t.unique())))
    t_expanded = t.view(-1, 1, 1)
    return -x / (2.0 - t_expanded)

def draft_vector_field(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    # Simulate draft model calculation delay
    time.sleep(DRAFT_LATENCY)
    t_expanded = t.view(-1, 1, 1)
    draft_v = -x / (2.0 - t_expanded)
    # Simulate systematic draft error: sine-wave structural perturbation
    draft_error = 0.015 * torch.sin(x * 4.0)
    return draft_v + draft_error

# -------------------------------------------------------------
# BENCHMARK RUNNER
# -------------------------------------------------------------
def run_grid_benchmark():
    torch.manual_seed(42)
    print("Initializing simulated noise coordinates (batch=2, residues=50, dim=3)...")
    x_init = torch.randn(2, 50, 3)
    step_size = 0.02
    
    # 1. Run Baseline (Standard ODE)
    print("\n[Baseline] Running Standard Flow Matching ODE...")
    baseline_solver = FlowMatchingODE(step_size=step_size)
    start_time = time.time()
    baseline_result = baseline_solver.solve(x_init, target_vector_field)
    baseline_time = time.time() - start_time
    baseline_evals = int(1.0 / step_size)
    
    print(f"Baseline completed in {baseline_time:.4f} seconds (evals: {baseline_evals}).")
    
    # Grid search parameters
    lookaheads = [2, 4, 6, 8]
    tolerances = [0.001, 0.005, 0.01, 0.03, 0.05, 0.10]
    
    results = []
    
    print("\n[Grid Sweep] Beginning parameter grid sweep...")
    for K in lookaheads:
        for epsilon in tolerances:
            sampler = SpeculativeFlowMatchingSampler(
                draft_vf_fn=draft_vector_field,
                target_vf_fn=target_vector_field,
                step_size=step_size,
                speculative_lookahead=K,
                tolerance=epsilon
            )
            
            # Reset seeds for consistency
            torch.manual_seed(42)
            
            # Benchmark execution time and accuracy
            start_t = time.time()
            res_coords, stats = sampler.sample(x_init)
            duration = time.time() - start_t
            
            l2_error = torch.norm(baseline_result - res_coords).item()
            wall_speedup = baseline_time / max(1e-6, duration)
            
            grid_result = {
                "K": K,
                "tolerance": epsilon,
                "target_evals": stats["total_target_evaluations"],
                "draft_proposed": stats["total_drafts_proposed"],
                "draft_accepted": stats["total_drafts_accepted"],
                "acceptance_rate": stats["acceptance_rate"],
                "theoretical_speedup": stats["estimated_speedup_factor"],
                "wall_time_sec": duration,
                "wall_speedup": wall_speedup,
                "l2_discrepancy": l2_error
            }
            results.append(grid_result)
            print(f"K={K:1d} | eps={epsilon:.3f} | Acc={stats['acceptance_rate']*100:6.2f}% | "
                  f"Theory Speedup={stats['estimated_speedup_factor']:.2f}x | "
                  f"Wall Speedup={wall_speedup:.2f}x | L2 Err={l2_error:.8f}")
            
    # Save results as JSON
    out_dir = "/tmp/biomolecular_design"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "benchmark_grid_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] Grid benchmark results saved to {json_path}")
    
    # 2. Print Markdown Table
    print("\n==========================================================================================")
    print("GRID SWEEP BENCHMARK REPORT (MARKDOWN TABLE)")
    print("==========================================================================================")
    print("| Lookahead (K) | Tolerance (ε) | Evals (Target) | Accept Rate (%) | Theoretical Speedup | Wall-Clock Speedup | L2 Discrepancy |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['K']} | {r['tolerance']:.3f} | {r['target_evals']} | {r['acceptance_rate']*100:.2f}% | {r['theoretical_speedup']:.2f}x | {r['wall_speedup']:.2f}x | {r['l2_discrepancy']:.2e} |")
    print("==========================================================================================\n")
    
    # 3. Create ECharts line chart configuration to plot speedup vs tolerance
    # Group series by K
    series_data = {}
    for K in lookaheads:
        series_data[K] = []
        
    for r in results:
        series_data[r["K"]].append([r["tolerance"], r["wall_speedup"]])
        
    echarts_series = []
    for K, data in series_data.items():
        # sort data by tolerance
        data = sorted(data, key=lambda x: x[0])
        echarts_series.append({
            "name": f"Lookahead K={K}",
            "type": "line",
            "data": data,
            "smooth": True,
            "symbolSize": 8
        })
        
    echarts_spec = {
        "title": {
            "text": "Speculative Flow Matching: Speedup vs Tolerance (ε)",
            "left": "center",
            "textStyle": {
                "color": "#0c2340",
                "fontSize": 14
            }
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": { "type": "cross" }
        },
        "legend": {
            "bottom": 10,
            "data": [f"Lookahead K={K}" for K in lookaheads]
        },
        "xAxis": {
            "name": "Tolerance (ε)",
            "nameLocation": "middle",
            "nameGap": 25,
            "type": "value",
            "splitLine": { "lineStyle": { "type": "dashed" } }
        },
        "yAxis": {
            "name": "Wall-Clock Speedup Factor",
            "nameLocation": "middle",
            "nameGap": 30,
            "type": "value",
            "splitLine": { "lineStyle": { "type": "dashed" } }
        },
        "series": echarts_series
    }
    
    spec_path = os.path.join(out_dir, "benchmark_grid_echarts_spec.json")
    with open(spec_path, 'w') as f:
        json.dump(echarts_spec, f, indent=2)
    print(f"[Save] ECharts specification saved to {spec_path}")

if __name__ == "__main__":
    run_grid_benchmark()
