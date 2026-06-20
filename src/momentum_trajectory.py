import torch
import torch.nn as nn
import math

class AdaptiveMomentumSpeculativeSolver:
    def __init__(self, vector_field, tol=1e-3, beta=0.9, k_max=5, var_threshold=0.05, 
                 alpha=0.8, forecast_mode='quadratic', use_momentum=False):
        """
        Adaptive Momentum-based Speculative Flow Matching Solver.
        
        Args:
            vector_field: A callable mapping (x, t) -> velocity tensor of shape like x.
            tol: Tolerance for accepting speculative steps.
            beta: Momentum decay coefficient for running velocity EMA.
            k_max: Maximum speculative lookahead step K.
            var_threshold: Running curvature threshold below which speculation is triggered/increased.
            alpha: EMA decay coefficient for running curvature tracking.
            forecast_mode: 'linear' or 'quadratic' forecasting.
            use_momentum: Whether to use the running momentum vector instead of raw velocity for stepping.
        """
        self.vector_field = vector_field
        self.tol = tol
        self.beta = beta
        self.k_max = k_max
        self.var_threshold = var_threshold
        self.alpha = alpha
        self.forecast_mode = forecast_mode
        self.use_momentum = use_momentum

    def solve(self, x0, t_span):
        """
        Solve the ODE from t_span[0] to t_span[-1].
        
        Args:
            x0: Initial state tensor of shape [batch_size, ...].
            t_span: 1D tensor of time steps, must be sorted.
            
        Returns:
            x_final: Final state.
            trajectory: Tensor of shape [num_steps, batch_size, ...] containing states at each t in t_span.
            info: Dictionary containing solver statistics.
        """
        n_steps = len(t_span)
        device = x0.device
        dtype = x0.dtype
        
        trajectory = torch.empty((n_steps, *x0.shape), device=device, dtype=dtype)
        trajectory[0] = x0
        
        x = x0.clone()
        t = t_span[0]
        v = self.vector_field(x, t)
        nfe = 1
        
        m = v.clone()
        v_prev = None
        m_prev = None
        a = torch.zeros_like(v)
        
        running_curvature = 0.0
        K = 1
        
        accepted_specs = 0
        rejected_specs = 0
        total_specs = 0
        skipped_steps_total = 0
        
        n = 0
        while n < n_steps - 1:
            dt = t_span[n+1] - t_span[n]
            
            # Compute curvature and update running curvature
            if (not self.use_momentum and v_prev is not None) or (self.use_momentum and m_prev is not None):
                active = m if self.use_momentum else v
                active_prev = m_prev if self.use_momentum else v_prev
                diff_norm = torch.norm(active - active_prev, p=2, dim=-1)
                active_norm = torch.norm(active, p=2, dim=-1)
                # Average relative change across batch
                step_curvature = torch.mean(diff_norm / (active_norm + 1e-8)).item()
                running_curvature = self.alpha * running_curvature + (1 - self.alpha) * step_curvature
            else:
                running_curvature = 0.0
            
            # Decide target speculative step K_target
            # We need at least one step of history to compute acceleration and curvature
            history_exists = m_prev is not None if self.use_momentum else v_prev is not None
            if history_exists and running_curvature < self.var_threshold:
                K_target = min(K + 1, self.k_max)
            else:
                K_target = 1
            
            # We can only speculate if we have enough steps left
            if K_target > 1 and n + K_target < n_steps:
                K = K_target
                total_specs += 1
                
                # Forecast states for steps n+1 to n+K
                x_forecast = [x]
                v_active = m if self.use_momentum else v
                
                # Forecast loop
                x_curr = x.clone()
                for j in range(1, K + 1):
                    step_dt = t_span[n+j] - t_span[n+j-1]
                    time_elapsed = t_span[n+j-1] - t_span[n]
                    if self.forecast_mode == 'quadratic':
                        v_j = v_active + a * time_elapsed
                    else:
                        v_j = v_active
                    x_curr = x_curr + step_dt * v_j
                    x_forecast.append(x_curr)
                
                # The final forecasted state is at index n+K
                x_final_forecast = x_forecast[-1]
                t_final_spec = t_span[n+K]
                
                # Evaluate actual velocity at the final state
                v_actual = self.vector_field(x_final_forecast, t_final_spec)
                nfe += 1
                
                # Forecasted velocity at the end of the speculative window:
                total_time = t_span[n+K] - t_span[n]
                if self.forecast_mode == 'quadratic':
                    v_forecast_end = v_active + a * total_time
                else:
                    v_forecast_end = v_active
                
                # Compute verification error
                error_norm = torch.norm(v_actual - v_forecast_end, p=2, dim=-1)
                v_actual_norm = torch.norm(v_actual, p=2, dim=-1)
                error = torch.mean(error_norm / (v_actual_norm + 1e-8)).item()
                
                if error < self.tol:
                    # ACCEPT
                    accepted_specs += 1
                    skipped_steps_total += (K - 1)
                    
                    # Fill intermediate states in the trajectory
                    for j in range(1, K + 1):
                        trajectory[n+j] = x_forecast[j]
                    
                    # Update solver state
                    x = x_final_forecast
                    v_prev = v.clone() if v is not None else None
                    v = v_actual
                    
                    m_prev = m.clone()
                    m = self.beta * m + (1.0 - self.beta) * v
                    
                    # Update acceleration based on active updates
                    active = m if self.use_momentum else v
                    active_prev = m_prev if self.use_momentum else v_prev
                    a = (active - active_prev) / total_time
                    
                    n += K
                else:
                    # REJECT
                    rejected_specs += 1
                    
                    # Fallback to the first forecasted step (Euler step from t_n)
                    x_next = x_forecast[1]
                    trajectory[n+1] = x_next
                    
                    # Evaluate velocity at x_next
                    v_next = self.vector_field(x_next, t_span[n+1])
                    nfe += 1
                    
                    # Update solver state
                    v_prev = v.clone()
                    v = v_next
                    
                    m_prev = m.clone()
                    m = self.beta * m + (1.0 - self.beta) * v
                    
                    active = m if self.use_momentum else v
                    active_prev = m_prev if self.use_momentum else v_prev
                    a = (active - active_prev) / dt
                    
                    x = x_next
                    n += 1
                    K = 1 # reset speculation lookahead
            else:
                # Standard step (either K_target == 1 or not enough steps remaining)
                v_active = m if self.use_momentum else v
                x_next = x + dt * v_active
                trajectory[n+1] = x_next
                
                # Evaluate velocity at x_next
                v_next = self.vector_field(x_next, t_span[n+1])
                nfe += 1
                
                # Update solver state
                v_prev = v.clone() if v is not None else None
                v = v_next
                
                m_prev = m.clone() if m is not None else None
                m = self.beta * m + (1.0 - self.beta) * v
                
                active = m if self.use_momentum else v
                active_prev = m_prev if self.use_momentum else v_prev
                if active_prev is not None:
                    a = (active - active_prev) / dt
                else:
                    a = torch.zeros_like(active)
                
                x = x_next
                n += 1
                K = 1 # reset speculation lookahead
                
        info = {
            'nfe': nfe,
            'total_specs': total_specs,
            'accepted_specs': accepted_specs,
            'rejected_specs': rejected_specs,
            'skipped_steps': skipped_steps_total,
            'acceptance_rate': accepted_specs / total_specs if total_specs > 0 else 0.0
        }
        return x, trajectory, info


class EulerSolver:
    def __init__(self, vector_field):
        self.vector_field = vector_field
        
    def solve(self, x0, t_span):
        n_steps = len(t_span)
        trajectory = torch.empty((n_steps, *x0.shape), device=x0.device, dtype=x0.dtype)
        trajectory[0] = x0
        
        x = x0.clone()
        nfe = 0
        for n in range(n_steps - 1):
            t = t_span[n]
            dt = t_span[n+1] - t
            v = self.vector_field(x, t)
            nfe += 1
            x = x + dt * v
            trajectory[n+1] = x
            
        return x, trajectory, {'nfe': nfe}


class HeunSolver:
    def __init__(self, vector_field):
        self.vector_field = vector_field
        
    def solve(self, x0, t_span):
        n_steps = len(t_span)
        trajectory = torch.empty((n_steps, *x0.shape), device=x0.device, dtype=x0.dtype)
        trajectory[0] = x0
        
        x = x0.clone()
        nfe = 0
        for n in range(n_steps - 1):
            t = t_span[n]
            t_next = t_span[n+1]
            dt = t_next - t
            
            v1 = self.vector_field(x, t)
            nfe += 1
            
            x_pred = x + dt * v1
            v2 = self.vector_field(x_pred, t_next)
            nfe += 1
            
            x = x + dt * 0.5 * (v1 + v2)
            trajectory[n+1] = x
            
        return x, trajectory, {'nfe': nfe}


class MockFoldingField(nn.Module):
    def __init__(self, target_coords, barrier_center, barrier_scale=5.0, barrier_strength=8.0):
        super().__init__()
        self.target_coords = target_coords  # [D] or [B, D]
        self.barrier_center = barrier_center  # [D] or [B, D]
        self.barrier_scale = barrier_scale
        self.barrier_strength = barrier_strength
        
    def forward(self, x, t):
        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
            
        # 1. Drive force towards native state: pull force
        v_pull = self.target_coords - x
        
        # 2. Nonlinear transition barrier (active mainly around t = 0.4 to 0.6)
        # Represents steric clashes or transition state barrier that decays as folding proceeds
        dist = x - self.barrier_center
        dist_sq = (dist ** 2).sum(dim=-1, keepdim=True)
        
        # Strength of barrier follows a Gaussian peak at t=0.5
        t_barrier_factor = torch.exp(-((t - 0.5) / 0.15) ** 2) * self.barrier_strength
        v_barrier = dist * torch.exp(-dist_sq / self.barrier_scale) * t_barrier_factor
        
        # 3. High-frequency torsional or local landscape bumps (fluctuations)
        # Decays linearly as t approaches 1.0 (folding system converges/locks in)
        v_torsion = 1.5 * torch.sin(3.0 * x) * (1.0 - t)
        
        # Total velocity field
        v = v_pull + v_barrier + v_torsion
        return v
