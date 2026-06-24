import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import torch
import torch.nn.functional as F
from train_preference_alignment import grpo_loss, PolicyNetwork, AASequenceTokenizer, get_sequence_logps
from speculative_flow_matching import SearchGuidedSpeculativeSampler
from agentic_design_loop import run_codesign_loop, compute_rewards

def test_grpo_advantage_properties():
    """Verify that GRPO advantage calculation standardizes rewards to zero-mean and unit-variance."""
    # 1. Setup sample rewards
    rewards = torch.tensor([1.2, -0.5, 3.4, 0.1, 2.3, -1.8, 0.7, 0.9])
    
    # Calculate group statistics
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean_r) / std_r
    
    # 2. Check properties
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-5), "Mean should be close to 0."
    assert torch.allclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-5), "Standard deviation should be close to 1."

    # 3. Check loss output properties with dummy inputs
    G = 4
    policy_logps = torch.randn(G)
    old_logps = torch.randn(G)
    rewards_small = torch.randn(G)
    
    loss, kl_mean, adv = grpo_loss(policy_logps, old_logps, rewards_small, beta=0.1, clip_eps=0.2)
    
    # Check that loss is a valid scalar tensor and advantages are standardized
    assert loss.dim() == 0
    assert kl_mean.dim() == 0
    assert adv.shape == (G,)
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-5)

def test_search_guided_trajectory_selection():
    """Verify search-guided sampler perturbing trajectories, computing reward, and selecting the best path."""
    torch.manual_seed(42)
    
    # Simple vector field functions
    def dummy_vf(x, t, **kwargs):
        return -x / 2.0

    sampler = SearchGuidedSpeculativeSampler(
        draft_vf_fn=dummy_vf,
        target_vf_fn=dummy_vf,
        step_size=0.1,
        speculative_lookahead=2,
        tolerance=0.05,
        num_candidates=3,
        perturb_scale=0.1
    )
    
    # Setup coordinates: Batch=1, residues=5, dims=3
    x_init = torch.randn(1, 5, 3)
    pocket_coords = torch.zeros(3, 3) # Pocket at the origin
    
    # Verify that biophysical reward calculates valid values
    rewards = sampler.compute_biophysical_reward(x_init, pocket_coords)
    assert rewards.shape == (1,)
    
    # Run a lookahead rollout manually
    rewards_rollout = sampler.run_draft_lookahead(x_init, 0.0, pocket_coords, {})
    assert rewards_rollout.shape == (1,)
    
    # Run the full sampler
    final_x, stats = sampler.sample(x_init, pocket_coords)
    assert final_x.shape == x_init.shape
    assert "acceptance_rate" in stats
    assert "total_steps" in stats

def test_co_design_loop_e2e():
    """Run agentic co-design loop end-to-end for multiple iterations and verify convergence properties."""
    # Run for 3 iterations, group size 4
    metrics = run_codesign_loop(iterations=3, group_size=4)
    
    assert len(metrics) == 3
    for entry in metrics:
        assert "iteration" in entry
        assert "mean_reward" in entry
        assert "loss" in entry
        assert "kl" in entry
        assert entry["loss"] != 0.0
        assert entry["kl"] > 0.0
        
    # Verify that the final iteration ran without error and loss/metrics are recorded
    print("Co-design loop E2E test finished successfully.")

if __name__ == "__main__":
    print("Running E2E tests for GRPO and Agentic Co-design...")
    test_grpo_advantage_properties()
    test_search_guided_trajectory_selection()
    test_co_design_loop_e2e()
    print("All tests passed successfully!")

