import torch
import torch.nn as nn
from typing import Callable, Tuple, Dict, Any, List

class FlowMatchingODE:
    """Standard Flow Matching ODE Solver using Euler integration."""
    
    def __init__(self, step_size: float = 0.02):
        self.step_size = step_size

    @torch.no_grad()
    def solve(self, 
              x_init: torch.Tensor, 
              vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              extra_args: Dict[str, Any] = {}) -> torch.Tensor:
        """Integrates the vector field from t=0 to t=1.
        
        Args:
            x_init: Initial noise coordinates, shape [B, N, D] or similar.
            vector_field_fn: Callable mapping (x, t) -> v(x, t).
            extra_args: Any additional arguments to pass to the vector field (e.g., sequence embeddings).
        """
        x = x_init.clone()
        t_steps = torch.arange(0.0, 1.0, self.step_size, device=x.device)
        
        for t in t_steps:
            t_tensor = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            v = vector_field_fn(x, t_tensor, **extra_args)
            x = x + v * self.step_size
            
        return x


class SpeculativeFlowMatchingSampler:
    """Speculative Flow Matching Sampler.
    
    Accelerates flow-matching structure generation by drafting multiple steps 
    with a cheap draft model and verifying them in parallel using an expensive target model.
    """
    
    def __init__(
        self,
        draft_vf_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target_vf_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        step_size: float = 0.02,
        speculative_lookahead: int = 4,
        tolerance: float = 0.05
    ):
        """
        Args:
            draft_vf_fn: Vector field function for the fast draft model (e.g., pruned Boltz-1).
            target_vf_fn: Vector field function for the full target model (e.g., Boltz-1/2).
            step_size: Integration step size (dt).
            speculative_lookahead (K): Number of draft steps to speculate before verification.
            tolerance (epsilon): Maximum allowed difference between draft and target vector fields.
        """
        self.draft_vf_fn = draft_vf_fn
        self.target_vf_fn = target_vf_fn
        self.step_size = step_size
        self.K = speculative_lookahead
        self.tolerance = tolerance

    @torch.no_grad()
    def sample(self, 
               x_init: torch.Tensor, 
               extra_args: Dict[str, Any] = {}) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs speculative flow-matching integration from t=0 to t=1.
        
        Returns:
            final_x: The generated 3D coordinates.
            stats: A dictionary with execution statistics (e.g., acceptance rate, target steps saved).
        """
        device = x_init.device
        dtype = x_init.dtype
        x = x_init.clone()
        
        t = 0.0
        dt = self.step_size
        
        # Track statistics
        total_evals_target = 0
        total_drafts_proposed = 0
        total_drafts_accepted = 0
        
        while t < 1.0 - 1e-5:
            # 1. GENERATE DRAFT TRAJECTORY (K steps lookahead)
            draft_x = [x.clone()]
            draft_t = []
            
            curr_x = x.clone()
            curr_t = t
            
            # Run the draft model sequentially to get proposal states
            for k in range(self.K):
                if curr_t >= 1.0 - 1e-5:
                    break
                t_tensor = torch.full((x.shape[0],), curr_t, device=device, dtype=dtype)
                draft_t.append(curr_t)
                
                v_draft = self.draft_vf_fn(curr_x, t_tensor, **extra_args)
                curr_x = curr_x + v_draft * dt
                draft_x.append(curr_x.clone())
                curr_t += dt
                total_drafts_proposed += 1

            # Number of actual proposed steps
            actual_k = len(draft_t)
            if actual_k == 0:
                break
                
            # 2. PARALLEL VERIFICATION BY TARGET MODEL
            # Batch the verification queries together: shape [actual_k * B, N, D]
            batch_size = x.shape[0]
            verify_x_batch = torch.cat(draft_x[:-1], dim=0) # Exclude the final point
            verify_t_batch = torch.tensor(draft_t, device=device, dtype=dtype).repeat_interleave(batch_size)
            
            # Expand extra_args for the batched verification call
            batched_extra_args = {}
            for key, val in extra_args.items():
                if isinstance(val, torch.Tensor):
                    # Repeat along batch dimension
                    dims = [1] * len(val.shape)
                    dims[0] = actual_k
                    batched_extra_args[key] = val.repeat(*dims)
                else:
                    batched_extra_args[key] = val
            
            # Single parallel evaluation of the target model
            v_target_batch = self.target_vf_fn(verify_x_batch, verify_t_batch, **batched_extra_args)
            total_evals_target += 1
            
            # Reshape target outputs back to list of steps: [actual_k, B, N, D]
            v_target_steps = v_target_batch.chunk(actual_k, dim=0)
            
            # 3. VERIFY STEP-BY-STEP
            accepted_k = 0
            curr_verified_x = x.clone()
            
            for k in range(actual_k):
                x_k = draft_x[k]
                t_k = draft_t[k]
                t_k_tensor = torch.full((batch_size,), t_k, device=device, dtype=dtype)
                
                # Retrieve target vector field evaluated at x_k
                v_target = v_target_steps[k]
                
                # Re-evaluate draft vector field at x_k to check divergence
                v_draft = self.draft_vf_fn(x_k, t_k_tensor, **extra_args)
                
                # Measure L2 difference between vector fields (normalized by magnitude)
                diff = torch.norm(v_target - v_draft, p=2, dim=-1) / (torch.norm(v_target, p=2, dim=-1) + 1e-8)
                mean_diff = diff.mean().item()
                
                if mean_diff <= self.tolerance:
                    # Accept step: update state using the target vector field (semi-correction)
                    curr_verified_x = x_k + v_target * dt
                    accepted_k += 1
                    total_drafts_accepted += 1
                else:
                    # Reject step: correct the current step using the target model's trajectory
                    curr_verified_x = x_k + v_target * dt
                    # Reject subsequent steps in this speculative block
                    break
            
            # Move simulation time forward
            x = curr_verified_x
            if accepted_k == actual_k:
                t += accepted_k * dt
            else:
                t += (accepted_k + 1) * dt
            
        acceptance_rate = total_drafts_accepted / max(1, total_drafts_proposed)
        speedup_factor = (1.0 / self.step_size) / (total_evals_target + (total_drafts_proposed - total_drafts_accepted))
        
        stats = {
            "total_target_evaluations": total_evals_target,
            "total_drafts_proposed": total_drafts_proposed,
            "total_drafts_accepted": total_drafts_accepted,
            "acceptance_rate": acceptance_rate,
            "estimated_speedup_factor": speedup_factor
        }
        
        return x, stats
