import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
from g_dpo_alignment import cluster_by_union_mask, select_group_preference_pairs, GDPOLoss

def test_union_mask_clustering():
    print("--- Testing Union Mask Clustering ---")
    # Wild-type sequence (15 residues)
    wt = "MATEVLADIGSAKLR"
    
    # Generate mock mutants with mutations at specific positions
    # Group 1: mutations at position 1 (A->G) and position 3 (T->Y)
    m1 = "MGTEVLADIGSAKLR"
    m2 = "MAYEVLADIGSAKLR"
    m3 = "MGYEVLADIGSAKLR"
    
    # Group 2: mutations at position 10 (S->T) and position 12 (A->V)
    m4 = "MATEVLADIGTAKLV"
    m5 = "MATEVLADIGSAKLV"
    
    sequences = [wt, m1, m2, m3, m4, m5]
    print(f"Sequences to cluster (total: {len(sequences)}):")
    for idx, seq in enumerate(sequences):
        print(f"  {idx}: {seq}")
        
    # Cluster with max_positions_in_union = 2
    groups = cluster_by_union_mask(sequences, max_positions_in_union=2)
    print("\nResulting Clusters:")
    for cluster_idx, group in enumerate(groups):
        cluster_seqs = [sequences[i] for i in group]
        print(f"  Cluster {cluster_idx}: Indices {group} -> Sequences {cluster_seqs}")
        
    assert len(groups) == 2, "Expected exactly 2 clusters!"
    print("Success: Union Mask Clustering grouped sequences by local mutations correctly!")


def test_g_dpo_loss():
    print("\n--- Testing g-DPO PyTorch Loss ---")
    # Simulate a group of 5 mutant sequences
    group_size = 5
    
    # Mock scores (higher is better binding affinity)
    scores = torch.tensor([1.2, 3.5, 0.8, 2.1, 1.9])
    
    # Mock log-likelihoods under policy model (M)
    # The policy model has higher likelihoods for high-scoring mutants (good alignment)
    policy_logps = torch.tensor([-5.2, -3.1, -6.8, -4.5, -4.7], requires_grad=True)
    
    # Mock log-likelihoods under reference model (M)
    ref_logps = torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0])
    
    # Instantiate g-DPO Loss
    loss_module = GDPOLoss(beta=0.1)
    
    # Calculate loss using "best_vs_all" pairing
    print("Running g-DPO Loss (best_vs_all)...")
    loss, metrics = loss_module(policy_logps, ref_logps, scores, pairing_strategy="best_vs_all")
    
    print("Loss Calculation Diagnostics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    print(f"\nComputed Loss: {loss.item():.6f}")
    
    # Backpropagate to test gradient calculations
    loss.backward()
    print(f"Policy gradients: {policy_logps.grad.tolist()}")
    
    # Verify values
    assert metrics["num_pairs"] == 4, "best_vs_all for size 5 should yield exactly 4 pairs!"
    assert metrics["accuracy"] == 1.0, "Policy has higher ratio for winner than reference, accuracy should be 1.0!"
    assert policy_logps.grad is not None, "Gradients should successfully propagate to policy logps!"
    print("Success: g-DPO Loss module works perfectly with backward gradient passes!")


if __name__ == "__main__":
    test_union_mask_clustering()
    test_g_dpo_loss()
