import torch
import time
import math
from src.parallel_picard_solver import MockVectorField, ParallelPicardSolver

def benchmark_parallel_picard():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running benchmarks on device: {device}")
    
    num_residues = 500
    num_steps = 20
    dt = 1.0 / num_steps
    
    # 1. Instantiate the mock vector field and move to device
    vector_field = MockVectorField(num_residues=num_residues).to(device)
    vector_field.eval() # Eval mode
    
    # 2. Generate random starting coordinates
    x0 = torch.randn(num_residues, 3, device=device)
    
    # 3. Instantiate the Parallel Picard Solver
    solver = ParallelPicardSolver(
        vector_field=vector_field,
        dt=dt,
        num_steps=num_steps,
        max_iters=5,
        tol=1e-4
    )
    
    # Warmup
    print("Warming up...")
    _ = solver.solve_sequential(x0)
    _, _ = solver.solve_parallel(x0)
    
    # Benchmark Sequential Euler
    print("Benchmarking Sequential Euler Solver...")
    start_time = time.perf_counter()
    traj_seq = solver.solve_sequential(x0)
    seq_time = (time.perf_counter() - start_time) * 1000 # in ms
    
    # Benchmark Parallel Picard
    print("Benchmarking Parallel Picard Iterative Solver...")
    start_time = time.perf_counter()
    traj_par, iters = solver.solve_parallel(x0)
    par_time = (time.perf_counter() - start_time) * 1000 # in ms
    
    # Compute error (MSE) between the two trajectories
    mse = torch.mean((traj_seq - traj_par) ** 2).item()
    
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Number of Residues: {num_residues}")
    print(f"Trajectory Length (steps): {num_steps}")
    print(f"Sequential Euler Latency: {seq_time:.2f} ms")
    print(f"Parallel Picard Latency:   {par_time:.2f} ms")
    print(f"Picard Iterations Taken:   {iters}")
    print(f"Speedup Factor:            {seq_time / par_time:.2f}x")
    print(f"Trajectory Approximation MSE: {mse:.6f}")
    print("===================================================\n")
    
    # Check if the speedup is positive and error is reasonably small
    if mse < 1e-2:
        print("Verification: PASSED (Approximation error is small)")
    else:
        print("Verification: FAILED (Approximation error too high)")

if __name__ == "__main__":
    benchmark_parallel_picard()
