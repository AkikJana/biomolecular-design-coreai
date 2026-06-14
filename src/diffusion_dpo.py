import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DiffusionDPOLoss(nn.Module):
    """Direct Preference Optimization Loss for 3D Coordinate Diffusion / Flow Matching Models.
    
    Optimizes the model's coordinate denoising trajectory steps directly, encouraging 
    the policy model to make smaller errors along winning structural pathways (high affinity)
    and larger errors along losing pathways.
    """
    
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_mse_win: torch.Tensor,
        policy_mse_loss: torch.Tensor,
        ref_mse_win: torch.Tensor,
        ref_mse_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the DPO loss on diffusion denoising trajectories.
        
        Args:
            policy_mse_win: Mean Squared Errors of the policy model along the winning trajectory.
                            Shape: [T, B] where T is diffusion steps, B is batch size.
            policy_mse_loss: Mean Squared Errors of the policy model along the losing trajectory.
                            Shape: [T, B].
            ref_mse_win: Mean Squared Errors of the reference model along the winning trajectory.
                         Shape: [T, B].
            ref_mse_loss: Mean Squared Errors of the reference model along the losing trajectory.
                          Shape: [T, B].
                          
        Returns:
            loss: Scaled loss tensor.
            metrics: Diagnostic metrics (accuracies, implicit rewards, margins).
        """
        # Sum the errors across the diffusion steps (T dimension) to get the total trajectory log-likelihoods
        # Since log-probability is proportional to -MSE:
        # log_pi(x) = -sum_t (MSE_t)
        policy_logp_win = -policy_mse_win.sum(dim=0)
        policy_logp_loss = -policy_mse_loss.sum(dim=0)
        ref_logp_win = -ref_mse_win.sum(dim=0)
        ref_logp_loss = -ref_mse_loss.sum(dim=0)
        
        # Calculate the log-ratio difference (implicit reward) for winning and losing trajectories
        implicit_reward_win = self.beta * (policy_logp_win - ref_logp_win)
        implicit_reward_loss = self.beta * (policy_logp_loss - ref_logp_loss)
        
        # Calculate logits: reward difference
        logits = implicit_reward_win - implicit_reward_loss
        
        # Compute DPO logsigmoid loss
        loss = -F.logsigmoid(logits)
        mean_loss = loss.mean()
        
        # Compute diagnostics
        with torch.no_grad():
            accuracy = (logits > 0).float().mean().item()
            margin = logits.mean().item()
            mean_rew_win = implicit_reward_win.mean().item()
            mean_rew_loss = implicit_reward_loss.mean().item()
            
        metrics = {
            "loss": mean_loss.item(),
            "accuracy": accuracy,
            "margin": margin,
            "reward_win": mean_rew_win,
            "reward_loss": mean_rew_loss
        }
        
        return mean_loss, metrics
