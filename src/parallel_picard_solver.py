import torch
import torch.nn as nn
import time
from typing import Callable, Tuple

class MockVectorField(nn.Module):
    """A mock vector field network for structural flow matching.
    Simulates a network that takes coordinate states [B, N, 3] and time t [B]
    and predicts the update velocity [B, N, 3].
    """
    def __init__(self, num_residues: int = 100, embed_dim: int = 64):
        super().__init__()
        self.num_residues = num_residues
        self.coordinate_proj = nn.Linear(3, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 3)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, seq_features: torch.Tensor = None) -> torch.Tensor:
        # x: [B, N, 3]
        # t: [B] or [B, 1]
        B, N, _ = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(-1) # [B, 1]
            
        x_embed = self.coordinate_proj(x) # [B, N, embed_dim]
        t_embed = self.time_proj(t).unsqueeze(1) # [B, 1, embed_dim]
        
        h = x_embed + t_embed # [B, N, embed_dim]
        if seq_features is not None:
            h = h + seq_features
            
        vel = self.net(h) # [B, N, 3]
        return vel

class ParallelPicardSolver:
    """Parallel Picard/Jacobi Iterative Trajectory Solver.
    
    Rather than evaluating the ODE sequentially (which takes T sequential steps),
    we initialize a guess trajectory of shape [T+1, N, 3] and iteratively refine
    the entire trajectory in parallel. Each iteration evaluates the vector field
    across all T time steps in a single batched forward pass.
    """
    def __init__(
        self, 
        vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dt: float,
        num_steps: int,
        max_iters: int = 5,
        tol: float = 1e-4
    ):
        self.vector_field = vector_field
        self.dt = dt
        self.num_steps = num_steps
        self.max_iters = max_iters
        self.tol = tol
        
    def solve_sequential(self, x0: torch.Tensor) -> torch.Tensor:
        """Standard sequential Euler integration."""
        x = x0.clone()
        trajectory = [x.clone()]
        
        for step in range(self.num_steps):
            t = torch.tensor([step * self.dt], device=x0.device, dtype=x0.dtype)
            v = self.vector_field(x.unsqueeze(0), t) # [1, N, 3]
            x = x + self.dt * v.squeeze(0)
            trajectory.append(x.clone())
            
        return torch.stack(trajectory, dim=0) # [T+1, N, 3]

    def solve_parallel(self, x0: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Parallel Picard/Jacobi Iterative solver.
        
        Args:
            x0: Initial coordinate tensor of shape [N, 3]
            
        Returns:
            trajectory: Solved trajectory of shape [T+1, N, 3]
            iters: Number of iterations taken to converge
        """
        N, D = x0.shape
        T = self.num_steps
        
        # 1. Initialize trajectory guess: repeat x0 across all steps.
        trajectory = x0.unsqueeze(0).repeat(T + 1, 1, 1) # [T+1, N, 3]
        
        # Precompute times for each step
        times = torch.linspace(0.0, T * self.dt, T, device=x0.device, dtype=x0.dtype) # [T]
        
        for iter_idx in range(self.max_iters):
            prev_trajectory = trajectory.clone()
            
            # Batch the evaluations across the trajectory steps [0, ..., T-1]
            x_batch = trajectory[:-1] # [T, N, 3]
            v_batch = self.vector_field(x_batch, times) # [T, N, 3]
            
            # Picard integration update:
            # x_t = x_0 + \int_0^t v_s(x_s) ds
            # In discrete steps:
            # x_i = x_0 + \sum_{j=0}^{i-1} dt * v_j(x_j)
            cumsum_v = torch.cumsum(v_batch * self.dt, dim=0) # [T, N, 3]
            
            # Update trajectory steps 1 to T
            trajectory[1:] = x0.unsqueeze(0) + cumsum_v
            
            # Check convergence
            diff = torch.norm(trajectory - prev_trajectory, dim=-1).mean().item()
            if diff < self.tol:
                return trajectory, iter_idx + 1
                
        return trajectory, self.max_iters
