import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
import math
from speculative_flow_matching import SpeculativeFlowMatchingSampler, FlowMatchingODE

# -------------------------------------------------------------
# MOCK MODELS FOR TESTING
# -------------------------------------------------------------

# True target vector field pulls coordinates from random noise towards origin (0, 0, 0)
def target_vector_field(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    # Vector field pulling to 0: v(x, t) = -x / (1.0 - t + epsilon)
    # Plus some target characteristics (e.g., structural folding trajectory)
    t_expanded = t.view(-1, 1, 1)
    target_v = -x / (2.0 - t_expanded)
    return target_v

# Draft vector field is a fast approximation with some added noise/imperfections
def draft_vector_field(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    t_expanded = t.view(-1, 1, 1)
    # Approximate vector field
    draft_v = -x / (2.0 - t_expanded)
    
    # Add a small systematic draft error (simulating draft model limitations)
    draft_error = 0.02 * torch.sin(x * 5.0)
    return draft_v + draft_error

# -------------------------------------------------------------
# TEST RUN
# -------------------------------------------------------------

def run_test():
    torch.manual_seed(42)
    print("Initializing test coordinates (simulating noise)...")
    batch_size = 2
    num_residues = 50
    coord_dim = 3
    x_init = torch.randn(batch_size, num_residues, coord_dim)

    # 1. Standard Flow Matching Sampler (Baseline)
    print("\n=======================================================")
    print("1. RUNNING STANDARD FLOW MATCHING ODE (BASELINE)")
    print("=======================================================")
    step_size = 0.02
    standard_solver = FlowMatchingODE(step_size=step_size)
    baseline_result = standard_solver.solve(x_init, target_vector_field)
    print(f"Standard ODE completed. Final coordinates shape: {baseline_result.shape}")
    print(f"Required target evaluations: {int(1.0 / step_size)}")

    # 2. Speculative Flow Matching: Case A - Perfect Draft Model (no error)
    print("\n=======================================================")
    print("2. SPECULATIVE FLOW MATCHING: CASE A (PERFECT DRAFT MODEL)")
    print("=======================================================")
    spec_sampler_perfect = SpeculativeFlowMatchingSampler(
        draft_vf_fn=target_vector_field,  # Use same function, no error
        target_vf_fn=target_vector_field,
        step_size=step_size,
        speculative_lookahead=4,
        tolerance=0.05
    )
    res_perfect, stats_perfect = spec_sampler_perfect.sample(x_init)
    print("Execution Statistics (Perfect Draft):")
    for k, v in stats_perfect.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    l2_perfect = torch.norm(baseline_result - res_perfect).item()
    print(f"Discrepancy (L2 distance) with baseline: {l2_perfect:.8f}")
    assert math.isclose(l2_perfect, 0.0, abs_tol=1e-5), "Perfect draft model must have 0.0 discrepancy!"

    # 3. Speculative Flow Matching: Case B - Imperfect Draft with High Tolerance (0.08)
    print("\n=======================================================")
    print("3. SPECULATIVE FLOW MATCHING: CASE B (IMPERFECT DRAFT, HIGH TOLERANCE)")
    print("=======================================================")
    spec_sampler_high = SpeculativeFlowMatchingSampler(
        draft_vf_fn=draft_vector_field,
        target_vf_fn=target_vector_field,
        step_size=step_size,
        speculative_lookahead=4,
        tolerance=0.08
    )
    res_high, stats_high = spec_sampler_high.sample(x_init)
    print("Execution Statistics (High Tolerance):")
    for k, v in stats_high.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    l2_high = torch.norm(baseline_result - res_high).item()
    print(f"Discrepancy (L2 distance) with baseline: {l2_high:.6f}")

    # 4. Speculative Flow Matching: Case C - Imperfect Draft with Low Tolerance (0.005)
    print("\n=======================================================")
    print("4. SPECULATIVE FLOW MATCHING: CASE C (IMPERFECT DRAFT, LOW TOLERANCE)")
    print("=======================================================")
    spec_sampler_low = SpeculativeFlowMatchingSampler(
        draft_vf_fn=draft_vector_field,
        target_vf_fn=target_vector_field,
        step_size=step_size,
        speculative_lookahead=4,
        tolerance=0.005
    )
    res_low, stats_low = spec_sampler_low.sample(x_init)
    print("Execution Statistics (Low Tolerance):")
    for k, v in stats_low.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    l2_low = torch.norm(baseline_result - res_low).item()
    print(f"Discrepancy (L2 distance) with baseline: {l2_low:.6f}")

if __name__ == "__main__":
    run_test()
