# src/agentic_design_loop.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from train_preference_alignment import PolicyNetwork, AASequenceTokenizer, get_sequence_logps, grpo_loss
from speculative_flow_matching import SearchGuidedSpeculativeSampler
import time

def compute_rewards(coords: torch.Tensor, pocket_coords: torch.Tensor) -> torch.Tensor:
    """
    Computes joint reward: pocket affinity - steric clash penalties.
    coords: [B, N, 3] - Binder coordinates
    pocket_coords: [P, 3] or [B, P, 3] - Target pocket coordinates
    """
    B, N, _ = coords.shape
    
    # 1. Pocket Affinity Score
    if len(pocket_coords.shape) == 2:
        p_coords = pocket_coords.unsqueeze(0) # [1, P, 3]
    else:
        p_coords = pocket_coords # [B, P, 3]
        
    dists_pocket = torch.cdist(coords, p_coords)
    affinity = torch.exp(-dists_pocket / 5.0).sum(dim=(1, 2)) # [B]
    
    # 2. Steric clashes
    dists_self = torch.cdist(coords, coords)
    mask = torch.eye(N, device=coords.device).bool()
    mask |= torch.diag(torch.ones(N - 1, device=coords.device), 1).bool()
    mask |= torch.diag(torch.ones(N - 1, device=coords.device), -1).bool()
    
    clash_vals = torch.clamp(2.0 - dists_self, min=0.0)
    clash_vals[..., mask] = 0.0
    clash_penalty = -10.0 * (clash_vals ** 2).sum(dim=(1, 2)) # [B]
    
    return affinity + clash_penalty

def sample_sequences(policy: PolicyNetwork, tokenizer: AASequenceTokenizer, base_seq: str, group_size: int, interface_pos: list) -> list:
    """Samples mutated sequences using policy model probabilities."""
    policy.eval()
    device = next(policy.parameters()).device
    sequences = []
    
    base_tokens = tokenizer.encode(base_seq).to(device)
    L = len(base_tokens)
    
    with torch.no_grad():
        for _ in range(group_size):
            tokens = base_tokens.clone()
            
            # Compute logits across the sequence
            logits = policy(tokens.unsqueeze(0))[0] # (L, vocab_size)
            
            # Apply mutations at interface positions
            for pos in interface_pos:
                pos_logits = logits[pos].clone()
                pos_logits[:4] = -float('inf')  # Mask out special tokens (<pad>, <unk>, <bos>, <eos>)
                probs = torch.softmax(pos_logits / 1.0, dim=-1) # Temperature 1.0
                sampled_token = torch.multinomial(probs, 1).item()
                tokens[pos] = sampled_token
                
            sequences.append(tokenizer.decode(tokens))
            
    return sequences

def run_codesign_loop(iterations=5, group_size=4):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AASequenceTokenizer()
    
    # 1. Policy initialization
    policy = PolicyNetwork(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=1e-4)
    
    # 2. Search-Guided speculative sampler initialization
    # Simple linear vector fields for demonstration
    sampler = SearchGuidedSpeculativeSampler(
        draft_vf_fn=lambda x, t, **k: -x / (2.0 - t.view(-1, 1, 1)),
        target_vf_fn=lambda x, t, **k: -x / (2.0 - t.view(-1, 1, 1)),
        step_size=0.1,
        speculative_lookahead=2,
        tolerance=0.1,
        num_candidates=4,
        perturb_scale=0.05
    )
    
    # Define reference WT sequence and target pocket coordinates (simulated)
    wt_sequence = "MATEVLADIGSAKLR"
    pocket_coords = torch.randn(10, 3, device=device)
    interface_positions = [2, 4, 8, 12]
    
    print("Starting closed-loop Agentic Co-Design...")
    metrics_history = []
    
    for iter_idx in range(iterations):
        # A. Policy Generation Phase
        seq_candidates = sample_sequences(policy, tokenizer, wt_sequence, group_size, interface_positions)
        
        # B. Speculative Folding Phase
        folded_coords_list = []
        for seq in seq_candidates:
            # Predict initial structure coordinates wrapper
            x_init = torch.randn(1, len(seq), 3, device=device)
            coords, _ = sampler.sample(x_init, pocket_coords)
            folded_coords_list.append(coords)
            
        folded_coords = torch.cat(folded_coords_list, dim=0) # [G, N, 3]
        
        # C. Biophysical Scoring Phase
        rewards = compute_rewards(folded_coords, pocket_coords) # [G]
        
        # D. GRPO Update Phase
        policy.train()
        encoded_tokens = torch.stack([tokenizer.encode(seq) for seq in seq_candidates]).to(device)
        
        # Compute and cache old log probabilities once
        with torch.no_grad():
            old_logits = policy(encoded_tokens)
            old_logps = get_sequence_logps(old_logits, encoded_tokens, length_normalize=True).detach()
            
        # Loop 3 times (inner steps)
        for inner_step in range(3):
            # Forward pass on active policy
            policy_logits = policy(encoded_tokens)
            policy_logps = get_sequence_logps(policy_logits, encoded_tokens, length_normalize=True)
            
            # Calculate loss using the GRPO training step
            loss, kl, advantages = grpo_loss(
                policy_logps=policy_logps,
                old_logps=old_logps,
                rewards=rewards,
                beta=0.1,
                clip_eps=0.2
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        mean_r = rewards.mean().item()
        loss_val = loss.item()
        kl_val = kl.item()
        
        print(f"Iteration {iter_idx+1:02d} | Mean Reward: {mean_r:.4f} | Loss: {loss_val:.4f} | Avg KL: {kl_val:.4f}")
        
        metrics_history.append({
            "iteration": iter_idx + 1,
            "mean_reward": mean_r,
            "loss": loss_val,
            "kl": kl_val
        })
        
    return metrics_history

if __name__ == "__main__":
    run_codesign_loop(iterations=5, group_size=4)
