import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import torch
from diffusion_dpo import DiffusionDPOLoss

def test_diffusion_dpo():
    print("=== Testing 3D Coordinate Diffusion DPO Loss ===")
    
    # 1. Setup mock dimensions
    steps = 20 # 20 diffusion steps
    batch_size = 2 # batch size of 2
    
    # 2. Simulate MSE coordinate errors for the policy and reference models
    # The reference model has similar errors on both trajectories (no preference)
    ref_mse_win = torch.full((steps, batch_size), 0.15)
    ref_mse_loss = torch.full((steps, batch_size), 0.15)
    
    # The policy model has been trained, so it makes:
    # - SMALLER error on the winning trajectory (policy is aligned with the winner)
    # - LARGER error on the losing trajectory
    policy_mse_win = torch.full((steps, batch_size), 0.08, requires_grad=True)
    policy_mse_loss = torch.full((steps, batch_size), 0.22, requires_grad=True)
    
    # 3. Instantiate and run Diffusion DPO Loss
    dpo_loss_module = DiffusionDPOLoss(beta=0.1)
    loss, metrics = dpo_loss_module(policy_mse_win, policy_mse_loss, ref_mse_win, ref_mse_loss)
    
    print("\nLoss Diagnostics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    print(f"\nComputed Diffusion DPO Loss: {loss.item():.6f}")
    
    # 4. Backward Pass
    loss.backward()
    
    # Verify values
    print(f"\nPolicy Win gradients sum: {policy_mse_win.grad.sum().item():.6f}")
    print(f"Policy Loss gradients sum: {policy_mse_loss.grad.sum().item():.6f}")
    
    assert metrics["accuracy"] == 1.0, "Policy has lower MSE (better likelihood) on winner than reference, accuracy should be 1.0!"
    assert policy_mse_win.grad.sum().item() > 0, "Win gradients should be positive for MSE errors (minimizing error -> positive grad on error value)!"
    assert policy_mse_loss.grad.sum().item() < 0, "Loss gradients should be negative for MSE errors!"
    print("\nSuccess: Diffusion DPO Loss module runs forward and backward passes flawlessly!")

if __name__ == "__main__":
    test_diffusion_dpo()
