"""Benchmark harness: edge surrogate vs full reference (Boltz-2) for binder ranking.

Produces the numbers that turn the edge pipeline into a defensible claim:

    "The surrogate reproduces Boltz-2's affinity ranking with <top-k recall>%
     (Spearman <rho>) at <ms>/candidate in <MB>."

The harness is scorer-agnostic. Plug in:
  - reference: full Boltz-2 affinity (see BoltzReferenceScorer below / boltz_reward.BoltzRewardModel)
  - surrogate: your edge model's affinity proxy
and it reports ranking agreement, latency, and model size. A runnable synthetic
pair (ground-truth reference + noisy surrogate) is included so the metric layer
is testable today; swap in the real scorers for real numbers.
"""

import math
import time
from typing import List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# Scorer interface
# --------------------------------------------------------------------------- #
class Scorer:
    """Maps candidates to affinity scores (higher = better binder)."""

    name: str = "scorer"

    def score(self, pairs: Sequence) -> torch.Tensor:  # (N,)
        raise NotImplementedError

    def model_size_bytes(self) -> Optional[int]:
        return None


# --------------------------------------------------------------------------- #
# Rank-agreement metrics
# --------------------------------------------------------------------------- #
def _avg_ranks(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (1-based), ties share the mean rank."""
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based mean of positions i..j
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double(); b = b.double()
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return (a @ b).item() / denom if denom > 1e-12 else 0.0


def spearman(ref: torch.Tensor, surr: torch.Tensor) -> float:
    return _pearson(_avg_ranks(ref), _avg_ranks(surr))


def kendall_tau(ref: torch.Tensor, surr: torch.Tensor) -> float:
    """Tau-a (no tie correction); O(n^2), fine for benchmark sizes."""
    n = ref.numel()
    if n < 2:
        return 0.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (ref[i] - ref[j]) * (surr[i] - surr[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    total = 0.5 * n * (n - 1)
    return (c - d) / total if total > 0 else 0.0


def topk_recall(ref: torch.Tensor, surr: torch.Tensor, k: int) -> float:
    """Fraction of the reference top-k that the surrogate also ranks top-k."""
    k = min(k, ref.numel())
    ref_top = set(torch.topk(ref, k).indices.tolist())
    surr_top = set(torch.topk(surr, k).indices.tolist())
    return len(ref_top & surr_top) / k


def top1_in_topk(ref: torch.Tensor, surr: torch.Tensor, k: int) -> bool:
    """Is the reference best candidate within the surrogate's top-k?"""
    k = min(k, ref.numel())
    ref_best = int(torch.argmax(ref).item())
    return ref_best in set(torch.topk(surr, k).indices.tolist())


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def _time_scorer(scorer: Scorer, pairs, reps: int = 3):
    scorer.score(pairs)  # warmup
    t0 = time.perf_counter()
    scores = None
    for _ in range(reps):
        scores = scorer.score(pairs)
    per_candidate_ms = (time.perf_counter() - t0) / reps / max(1, len(pairs)) * 1000
    return scores, per_candidate_ms


def benchmark(
    pairs: Sequence,
    reference: Scorer,
    surrogate: Scorer,
    ks: Sequence[int] = (1, 5, 10),
    reps: int = 3,
    verbose: bool = True,
) -> dict:
    ref_scores, ref_ms = _time_scorer(reference, pairs, reps)
    surr_scores, surr_ms = _time_scorer(surrogate, pairs, reps)

    metrics = {
        "n_candidates": len(pairs),
        "spearman": spearman(ref_scores, surr_scores),
        "kendall_tau": kendall_tau(ref_scores, surr_scores),
        "topk_recall": {k: topk_recall(ref_scores, surr_scores, k) for k in ks},
        "top1_in_topk": {k: top1_in_topk(ref_scores, surr_scores, k) for k in ks},
        "latency_ms_per_candidate": {reference.name: ref_ms, surrogate.name: surr_ms},
        "speedup": ref_ms / surr_ms if surr_ms > 0 else float("inf"),
        "model_size_bytes": {
            reference.name: reference.model_size_bytes(),
            surrogate.name: surrogate.model_size_bytes(),
        },
    }

    if verbose:
        _print_report(reference, surrogate, metrics, ks)
    return metrics


def _fmt_mb(b):
    return f"{b / 1024**2:.2f} MB" if b else "n/a"


def _print_report(reference, surrogate, m, ks):
    print("=" * 64)
    print(f"Surrogate-vs-reference benchmark  (N={m['n_candidates']})")
    print("=" * 64)
    print(f"  reference: {reference.name}   surrogate: {surrogate.name}")
    print("-" * 64)
    print(f"  Spearman rho      : {m['spearman']:+.3f}")
    print(f"  Kendall  tau      : {m['kendall_tau']:+.3f}")
    for k in ks:
        print(f"  top-{k:<2d} recall      : {m['topk_recall'][k]*100:5.1f}%   "
              f"(best-in-top{k}: {'yes' if m['top1_in_topk'][k] else 'no'})")
    print("-" * 64)
    rms = m["latency_ms_per_candidate"]
    print(f"  latency/candidate : {reference.name} {rms[reference.name]:.3f} ms | "
          f"{surrogate.name} {rms[surrogate.name]:.3f} ms  ({m['speedup']:.0f}x)")
    sz = m["model_size_bytes"]
    print(f"  model size        : {reference.name} {_fmt_mb(sz[reference.name])} | "
          f"{surrogate.name} {_fmt_mb(sz[surrogate.name])}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Reference adapter for real Boltz-2 (documented; needs weights)
# --------------------------------------------------------------------------- #
class BoltzReferenceScorer(Scorer):
    """Wraps a real Boltz-2 affinity predictor.

    predict_fn(target_seq, binder_seq) -> Boltz output dict; affinity is read via
    boltz_reward.compute_design_reward (or use the model's affinity head directly).
    """

    name = "boltz2"

    def __init__(self, predict_fn, size_bytes: Optional[int] = None):
        from boltz_reward import compute_design_reward
        self._predict = predict_fn
        self._reward = compute_design_reward
        self._size = size_bytes

    @torch.no_grad()
    def score(self, pairs):
        return torch.stack(
            [self._reward(self._predict(t, b)).reshape(-1)[0] for (t, b) in pairs]
        )

    def model_size_bytes(self):
        return self._size


# --------------------------------------------------------------------------- #
# Synthetic scorers (runnable demo / tests of the metric layer)
# --------------------------------------------------------------------------- #
class SyntheticReferenceScorer(Scorer):
    """Ground-truth affinity = match of binder to a hidden motif at interface positions."""

    name = "ref(synthetic)"

    def __init__(self, target_motif: str, interface: List[int], size_bytes=400_000_000):
        self.motif = target_motif
        self.interface = interface
        self._size = size_bytes

    def score(self, pairs):
        out = []
        for (_, binder) in pairs:
            match = sum(1 for i, p in enumerate(self.interface)
                        if p < len(binder) and binder[p] == self.motif[i])
            out.append(float(match) / max(1, len(self.interface)))
        return torch.tensor(out)

    def model_size_bytes(self):
        return self._size


class NoisySurrogateScorer(Scorer):
    """Approximates a reference scorer with additive noise (models surrogate error)."""

    name = "surrogate(edge)"

    def __init__(self, reference: Scorer, noise: float = 0.1, latency_pad: float = 0.0,
                 size_bytes=8_000_000, seed: int = 0):
        self.reference = reference
        self.noise = noise
        self.latency_pad = latency_pad
        self._size = size_bytes
        self._g = torch.Generator().manual_seed(seed)

    def score(self, pairs):
        base = self.reference.score(pairs)
        if self.latency_pad:
            time.sleep(self.latency_pad)
        return base + self.noise * torch.randn(base.shape, generator=self._g)

    def model_size_bytes(self):
        return self._size


def main():
    import random
    random.seed(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    interface = [2, 4, 8, 12, 15]
    motif = "WYFML"
    target = "MATEVLADIGSAKLRPQ"
    wt = list("MATEVLADIGSAKLRPQ")
    pairs = []
    for _ in range(40):  # 40 binder variants vs one target
        b = wt.copy()
        for p in interface:
            if random.random() < 0.5:
                b[p] = random.choice(aa)
        pairs.append((target, "".join(b)))

    reference = SyntheticReferenceScorer(motif, interface)
    surrogate = NoisySurrogateScorer(reference, noise=0.15)
    benchmark(pairs, reference, surrogate, ks=(1, 5, 10))


if __name__ == "__main__":
    main()
