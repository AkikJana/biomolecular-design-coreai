import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

def compute_union_mask(seq_a: str, seq_b: str) -> List[int]:
    """Finds indices where two sequences differ."""
    return [i for i, (char_a, char_b) in enumerate(zip(seq_a, seq_b)) if char_a != char_b]


def cluster_by_union_mask(sequences: List[str], max_positions_in_union: int = 5) -> List[List[int]]:
    """Groups sequence indices into clusters where the union of mutated positions is small.
    
    This ensures that sequences in a group can share the same masked context (union mask),
    minimizing computation for log-likelihood calculations.
    
    Args:
        sequences: List of mutated protein sequences (assumed to be of equal length).
        max_positions_in_union: Max number of different positions allowed in a single group's union mask.
        
    Returns:
        A list of groups, where each group is a list of sequence indices.
    """
    if not sequences:
        return []
        
    ref_seq = sequences[0] # Take the first sequence (typically wild-type) as the reference template
    n_seqs = len(sequences)
    
    # Calculate mutations for each sequence relative to the reference sequence
    mut_positions = [set(compute_union_mask(ref_seq, seq)) for seq in sequences]
    
    unassigned = set(range(n_seqs))
    clusters = []
    
    while unassigned:
        # Start a new cluster with the first unassigned sequence
        seed_idx = min(unassigned)
        current_cluster = [seed_idx]
        current_union = set(mut_positions[seed_idx])
        unassigned.remove(seed_idx)
        
        # Grep other sequences that fit within the union mask limit
        to_check = list(unassigned)
        for idx in to_check:
            potential_union = current_union.union(mut_positions[idx])
            if len(potential_union) <= max_positions_in_union:
                current_cluster.append(idx)
                current_union = potential_union
                unassigned.remove(idx)
                
        clusters.append(current_cluster)
        
    return clusters


def select_group_preference_pairs(
    scores: torch.Tensor, 
    pairing_strategy: str = "best_vs_all"
) -> List[Tuple[int, int]]:
    """Selects preference pairs within a group of sequences using a linear-scaling strategy.
    
    Args:
        scores: 1D Tensor of rewards/scores for each sequence in the group (higher is better).
        pairing_strategy: "best_vs_all" (O(M)) or "top_vs_bottom" (O(M)).
        
    Returns:
        A list of tuples (winner_idx, loser_idx) relative to the group.
    """
    m = len(scores)
    if m < 2:
        return []
        
    pairs = []
    
    if pairing_strategy == "best_vs_all":
        # Find the single best sequence
        best_idx = torch.argmax(scores).item()
        for i in range(m):
            if i != best_idx:
                # The best sequence is the winner, others are losers
                # (Only add pair if there is a score difference)
                if scores[best_idx] > scores[i]:
                    pairs.append((best_idx, i))
                    
    elif pairing_strategy == "top_vs_bottom":
        # Sort sequences by score
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        mid = m // 2
        top_half = sorted_indices[:mid]
        bottom_half = sorted_indices[mid:]
        
        # Pair them up sequentially
        for w, l in zip(top_half, bottom_half):
            if scores[w] > scores[l]:
                pairs.append((w, l))
                
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")
        
    return pairs


class GDPOLoss(nn.Module):
    """Group-based Direct Preference Optimization Loss (g-DPO).
    
    Formulates DPO over group-based preference comparisons to avoid the O(N^2) pairing explosion.
    """
    
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        scores: torch.Tensor,
        pairing_strategy: str = "best_vs_all"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the g-DPO loss for a group of sequences.
        
        Args:
            policy_logps: Log-likelihoods of the sequences under the policy model. Shape [M].
            ref_logps: Log-likelihoods of the sequences under the reference model. Shape [M].
            scores: Reward/affinity scores of the sequences. Shape [M].
            pairing_strategy: Pairing logic ("best_vs_all" or "top_vs_bottom").
            
        Returns:
            loss: Scaled loss tensor.
            metrics: Diagnostic metrics (accuracies, margins).
        """
        # 1. Select preference pairs within the group
        pairs = select_group_preference_pairs(scores, pairing_strategy)
        
        if not pairs:
            # Fallback for groups with no distinct preferences
            return torch.tensor(0.0, device=policy_logps.device, requires_grad=True), {"accuracy": 0.0, "margin": 0.0}
            
        # Extract indices of winners and losers
        winners = [w for w, l in pairs]
        losers = [l for w, l in pairs]
        
        # 2. Gather log likelihoods for pairs
        policy_logps_w = policy_logps[winners]
        policy_logps_l = policy_logps[losers]
        ref_logps_w = ref_logps[winners]
        ref_logps_l = ref_logps[losers]
        
        # 3. Compute log-ratio differences
        policy_ratio = policy_logps_w - policy_logps_l
        ref_ratio = ref_logps_w - ref_logps_l
        logits = policy_ratio - ref_ratio
        
        # 4. Compute DPO Loss
        loss = -F.logsigmoid(self.beta * logits)
        
        if self.label_smoothing > 0.0:
            loss = loss * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            
        mean_loss = loss.mean()
        
        # 5. Compute diagnostics
        with torch.no_grad():
            accuracy = (logits > 0).float().mean().item()
            margin = logits.mean().item()
            
        metrics = {
            "loss": mean_loss.item(),
            "accuracy": accuracy,
            "margin": margin,
            "num_pairs": len(pairs)
        }
        
        return mean_loss, metrics
