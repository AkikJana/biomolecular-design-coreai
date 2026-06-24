"""Edge surrogate with an affinity head + a benchmark Scorer adapter.

Extends the coords-only surrogate (convert_surrogate_coreai.SurrogateModel) with a
scalar affinity head so the edge model can *rank binders*, which is what the
surrogate-vs-Boltz-2 benchmark needs. Pure PyTorch (no coreai dependency) so it
trains/runs in the base env and can later be FP8/FP4-quantized and CoreAI-exported.

Outputs mirror Boltz-2's affinity keys:
    affinity_pred_value          (regression value; lower ~ tighter, convention-dependent)
    affinity_probability_binary  (P(binder); higher = better)  <- default ranking signal
plus sample_atom_coords for structure.

Untrained it is meaningless; train it by distilling Boltz-2 affinities
(predict_fn from boltz2_predict) into the affinity head.
"""

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_surrogate_vs_reference import Scorer


class AffinitySurrogate(nn.Module):
    """Sequence -> (coords, affinity) with a cacheable receptor cross-attention."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, hidden: int = 128,
                 alphabet: str = "ACDEFGHIKLMNPQRSTVWY", seed: int = 42):
        super().__init__()
        self.alphabet = alphabet
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(len(alphabet), embed_dim)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(len(alphabet), embed_dim, generator=g))

        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.coord_head = nn.Linear(embed_dim, 3)
        # Per-position affinity contributions, summed over the binder. Binding is
        # ~additive over contacts, so summing position-wise terms (each aware of
        # its residue + position via the PE in embed_seq) is the right inductive
        # bias -- it lets the head represent "match at pos 2 + match at pos 4 + ..."
        # which a single pooled vector cannot.
        self.affinity_head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def _split(self, x):  # (1, L, D) -> (1, H, L, hd)
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _positional_encoding(self, length: int, device) -> torch.Tensor:
        pe = torch.zeros(length, self.embed_dim, device=device)
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, L, D)

    def embed_seq(self, seq: str) -> torch.Tensor:
        idx = torch.tensor(
            [self.alphabet.find(c) if c in self.alphabet else 0 for c in seq],
            device=self.embed.weight.device,
        )
        emb = self.embed(idx).unsqueeze(0)  # (1, L, D)
        return emb + self._positional_encoding(emb.shape[1], emb.device)

    def target_kv(self, target_seq: str):
        """Precompute (and cache) receptor key/value once per target."""
        t = self.embed_seq(target_seq)
        return self._split(self.k_proj(t)), self._split(self.v_proj(t))

    def forward(self, binder_emb, target_k, target_v) -> Dict[str, torch.Tensor]:
        h = F.relu(self.conv1(binder_emb.transpose(1, 2))).transpose(1, 2)  # (1, Lb, D)
        q = self._split(self.q_proj(h))                                     # (1, H, Lb, hd)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = (q @ target_k.transpose(-2, -1) * scale).softmax(dim=-1)
        o = (attn @ target_v).permute(0, 2, 1, 3).reshape(1, -1, self.embed_dim)
        # Residual: keep the binder's own (PE-aware) features so the affinity head
        # sees binder residue identity + position directly, not only the
        # target-derived cross-attention output.
        o = self.out_proj(o) + h                                           # (1, Lb, D)

        coords = self.coord_head(o)                                        # (1, Lb, 3)

        # Sum of per-position affinity contributions (additive over contacts).
        raw = self.affinity_head(o).sum(dim=1)                             # (1, 1)
        return {
            "sample_atom_coords": coords,
            "affinity_pred_value": raw.reshape(-1),
            "affinity_probability_binary": torch.sigmoid(raw).reshape(-1),
        }

    def predict(self, target_seq: str, binder_seq: str, target_kv=None) -> Dict[str, torch.Tensor]:
        k, v = target_kv if target_kv is not None else self.target_kv(target_seq)
        return self.forward(self.embed_seq(binder_seq), k, v)


class SurrogateAffinityScorer(Scorer):
    """Ranks (target, binder) pairs with the surrogate's affinity head.

    Caches receptor K/V per unique target (the screening optimization: many
    binders vs one target), so cost is ~O(L_binder * L_target) per candidate.
    """

    name = "surrogate(edge)"

    def __init__(self, surrogate: AffinitySurrogate,
                 rank_key: str = "affinity_probability_binary"):
        self.surrogate = surrogate
        self.rank_key = rank_key

    @torch.no_grad()
    def score(self, pairs: Sequence) -> torch.Tensor:
        kv_cache: Dict[str, tuple] = {}
        scores = []
        for target, binder in pairs:
            if target not in kv_cache:
                kv_cache[target] = self.surrogate.target_kv(target)
            out = self.surrogate.predict(target, binder, target_kv=kv_cache[target])
            scores.append(out[self.rank_key].reshape(-1)[0])
        return torch.stack(scores)

    def model_size_bytes(self) -> int:
        # fp32 params; FP8/FP4 weight-only quant would cut this ~4x / ~8x.
        return sum(p.numel() for p in self.surrogate.parameters()) * 4
