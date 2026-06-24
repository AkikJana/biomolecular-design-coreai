"""Rewards for GRPO co-design, computed from real Boltz model outputs.

The reward pathway here is the genuine Boltz confidence metric (the same formula
Boltz uses to rank predictions) combined with a geometric clash penalty on the
predicted atom coordinates:

    confidence_score = (4 * complex_plddt + (iptm or ptm)) / 5      # Boltz formula
    reward           = confidence_score - clash_weight * clash_penalty(coords)

``RewardModel.score(sequences) -> (G,)`` is the only thing the GRPO loop needs.
``BoltzRewardModel`` wraps a real Boltz predictor via an injected ``predict_fn``
(sequence -> Boltz output dict), so featurization/inference is decoupled and the
reward computation can be tested without model weights. For real co-design,
supply a predict_fn that featurizes the sequence and runs Boltz2.forward,
returning its output dict (keys: complex_plddt, iptm, ptm, sample_atom_coords).
"""

from typing import Callable, Dict, List

import torch


def boltz_confidence_score(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Boltz's own confidence score: (4 * complex_plddt + iptm_or_ptm) / 5.

    Falls back to ptm when iptm is all-zero (single chain), matching boltz2.py.
    """
    complex_plddt = out["complex_plddt"]
    iptm = out.get("iptm", torch.zeros_like(complex_plddt))
    if torch.allclose(iptm, torch.zeros_like(iptm)):
        score_term = out.get("ptm", torch.zeros_like(complex_plddt))
    else:
        score_term = iptm
    return (4.0 * complex_plddt + score_term) / 5.0


def clash_penalty(
    coords: torch.Tensor, atom_mask: torch.Tensor = None, threshold: float = 2.0
) -> torch.Tensor:
    """Mean squared steric overlap over non-adjacent atom pairs, per batch element.

    coords: (B, M, 3); atom_mask: (B, M) or None. Returns (B,).
    """
    B, M, _ = coords.shape
    if atom_mask is None:
        atom_mask = coords.new_ones(B, M)
    dist = torch.cdist(coords, coords)  # (B, M, M)
    # exclude self + sequence-adjacent neighbors (bonded), and padded atoms
    eye = torch.eye(M, device=coords.device)
    adj = torch.diag(torch.ones(M - 1, device=coords.device), 1)
    adj = adj + adj.T + eye
    pair_valid = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2) * (1 - adj).unsqueeze(0)
    overlap = torch.clamp(threshold - dist, min=0.0) ** 2 * pair_valid
    return overlap.sum(dim=(1, 2)) / (pair_valid.sum(dim=(1, 2)) + 1e-8)


def compute_design_reward(
    out: Dict[str, torch.Tensor], clash_weight: float = 1.0
) -> torch.Tensor:
    """Scalar design reward per prediction from a Boltz output dict. Returns (B,)."""
    reward = boltz_confidence_score(out)  # (B,)
    if clash_weight > 0 and "sample_atom_coords" in out:
        coords = out["sample_atom_coords"]
        mask = out.get("atom_mask", None)
        reward = reward - clash_weight * clash_penalty(coords, mask)
    return reward


class RewardModel:
    """Interface: map a list of sequences to a reward tensor of shape (G,)."""

    def score(self, sequences: List[str]) -> torch.Tensor:
        raise NotImplementedError


class BoltzRewardModel(RewardModel):
    """Scores sequences with real Boltz outputs via an injected predictor.

    predict_fn(sequence: str) -> Boltz output dict. Typically this featurizes the
    sequence (tokenize / MSA / atom features) and runs Boltz2.forward, returning
    the dict with complex_plddt, iptm/ptm and sample_atom_coords.
    """

    def __init__(
        self,
        predict_fn: Callable[[str], Dict[str, torch.Tensor]],
        clash_weight: float = 1.0,
    ):
        self.predict_fn = predict_fn
        self.clash_weight = clash_weight

    @torch.no_grad()
    def score(self, sequences: List[str]) -> torch.Tensor:
        rewards = []
        for seq in sequences:
            out = self.predict_fn(seq)
            r = compute_design_reward(out, clash_weight=self.clash_weight)
            rewards.append(r.reshape(-1)[0])
        return torch.stack(rewards)


class SyntheticSequenceBoltzReward(RewardModel):
    """Test/demo stand-in for BoltzRewardModel.

    Fabricates Boltz-like outputs whose confidence rises with matches to a hidden
    target motif at the interface positions, then scores them with the REAL Boltz
    reward formula (boltz_confidence_score / compute_design_reward). This lets the
    GRPO co-design loop run end-to-end and actually learn, while exercising the
    genuine reward pathway. Swap in BoltzRewardModel for real predictions.
    """

    def __init__(
        self,
        target_seq: str,
        interface_positions: List[int],
        num_atoms: int = 48,
        clash_weight: float = 1.0,
        seed: int = 0,
    ):
        self.target = target_seq
        self.interface = interface_positions
        self.num_atoms = num_atoms
        self.clash_weight = clash_weight
        self._g = torch.Generator().manual_seed(seed)

    def _fabricate(self, seq: str) -> Dict[str, torch.Tensor]:
        match = sum(1 for p in self.interface if p < len(seq) and seq[p] == self.target[p])
        frac = match / max(1, len(self.interface))
        complex_plddt = torch.tensor([0.40 + 0.55 * frac])
        iptm = torch.tensor([0.30 + 0.65 * frac])
        # Well-spaced backbone (~3.8 A) -> negligible clashes; jitter shrinks with fit.
        idx = torch.arange(self.num_atoms, dtype=torch.float32).unsqueeze(-1)
        coords = idx * torch.tensor([3.8, 0.0, 0.0])
        coords = coords + (1.0 - frac) * 0.2 * torch.randn(
            self.num_atoms, 3, generator=self._g
        )
        return {
            "complex_plddt": complex_plddt,
            "iptm": iptm,
            "ptm": iptm,
            "sample_atom_coords": coords.unsqueeze(0),
        }

    @torch.no_grad()
    def score(self, sequences: List[str]) -> torch.Tensor:
        return torch.stack(
            [compute_design_reward(self._fabricate(s), self.clash_weight)[0] for s in sequences]
        )
