# src/agentic_design_loop.py
"""Closed-loop GRPO co-design driven by real Boltz rewards.

Each iteration: the policy proposes mutant sequences at interface positions, a
RewardModel scores them (real Boltz confidence + clash geometry via
boltz_reward), and GRPO updates the policy from the group-standardized rewards.

The structure scoring is fully delegated to a RewardModel (boltz_reward.py): the
previous placeholder "fold" (random init coords + an identity vector field that
ignored the sequence) and its synthetic reward are gone. Pass a BoltzRewardModel
wrapping real Boltz inference for production; the demo uses
SyntheticSequenceBoltzReward, which scores fabricated-but-sequence-dependent
Boltz outputs through the genuine reward formula so the loop runs and learns.
"""

import torch
import torch.optim as optim

from train_preference_alignment import (
    PolicyNetwork,
    AASequenceTokenizer,
    get_sequence_logps,
    grpo_loss,
)
from boltz_reward import RewardModel, SyntheticSequenceBoltzReward


def sample_sequences(policy, tokenizer, base_seq, group_size, interface_pos):
    """Sample mutated sequences from the policy at the interface positions."""
    policy.eval()
    device = next(policy.parameters()).device
    sequences = []
    base_tokens = tokenizer.encode(base_seq).to(device)

    with torch.no_grad():
        for _ in range(group_size):
            tokens = base_tokens.clone()
            logits = policy(tokens.unsqueeze(0))[0]  # (L, vocab)
            for pos in interface_pos:
                pos_logits = logits[pos].clone()
                pos_logits[:4] = -float("inf")  # mask special tokens
                probs = torch.softmax(pos_logits, dim=-1)
                tokens[pos] = torch.multinomial(probs, 1).item()
            sequences.append(tokenizer.decode(tokens))
    return sequences


def run_codesign_loop(
    reward_model: RewardModel,
    wt_sequence: str,
    interface_positions: list,
    iterations: int = 15,
    group_size: int = 8,
    lr: float = 1e-3,
    grpo_beta: float = 0.1,
    grpo_clip_eps: float = 0.2,
    inner_steps: int = 3,
    device: str = None,
    verbose: bool = True,
):
    dev = torch.device(
        device
        or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    tokenizer = AASequenceTokenizer()
    policy = PolicyNetwork(vocab_size=tokenizer.vocab_size).to(dev)
    optimizer = optim.AdamW(policy.parameters(), lr=lr)

    history = []
    for it in range(1, iterations + 1):
        # 1. Policy proposes mutant sequences.
        seqs = sample_sequences(policy, tokenizer, wt_sequence, group_size, interface_positions)

        # 2. Real Boltz reward pathway: confidence + clash geometry.
        rewards = reward_model.score(seqs).to(dev)

        # 3. GRPO update from group-standardized rewards.
        policy.train()
        tokens = torch.stack([tokenizer.encode(s) for s in seqs]).to(dev)
        with torch.no_grad():
            old_logps = get_sequence_logps(policy(tokens), tokens, length_normalize=True).detach()

        loss = kl = None
        for _ in range(inner_steps):
            logps = get_sequence_logps(policy(tokens), tokens, length_normalize=True)
            loss, kl, _ = grpo_loss(
                policy_logps=logps, old_logps=old_logps, rewards=rewards,
                beta=grpo_beta, clip_eps=grpo_clip_eps,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_r = rewards.mean().item()
        history.append({"iteration": it, "mean_reward": mean_r, "loss": loss.item(), "kl": kl.item()})
        if verbose:
            print(f"Iter {it:02d} | mean reward {mean_r:.4f} | loss {loss.item():.4f} | KL {kl.item():.4f}")

    return history


def main():
    wt_sequence = "MATEVLADIGSAKLR"
    interface_positions = [2, 4, 8, 12]
    # Hidden optimal motif the policy should discover at the interface positions.
    target = list(wt_sequence)
    for p, aa in zip(interface_positions, "WYFM"):
        target[p] = aa
    reward_model = SyntheticSequenceBoltzReward(
        target_seq="".join(target), interface_positions=interface_positions, clash_weight=1.0
    )

    print("Closed-loop GRPO co-design with Boltz-style rewards...")
    hist = run_codesign_loop(
        reward_model=reward_model, wt_sequence=wt_sequence,
        interface_positions=interface_positions, iterations=20, group_size=8,
    )
    first = sum(h["mean_reward"] for h in hist[:3]) / 3
    last = sum(h["mean_reward"] for h in hist[-3:]) / 3
    print(f"\nMean reward: first 3 iters {first:.4f} -> last 3 iters {last:.4f} ({last - first:+.4f})")


if __name__ == "__main__":
    main()
