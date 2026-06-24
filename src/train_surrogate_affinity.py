"""Affinity-distillation trainer for the edge surrogate's affinity head.

Distills a reference (Boltz-2) binder-affinity signal into AffinitySurrogate so
the edge model reproduces Boltz-2's *ranking* of binders. Progress is reported
with the benchmark's own metrics (Spearman, top-k recall), so "training improved"
== "the benchmark number improved".

Teacher abstraction
-------------------
AffinityTeacher.label(target, binder) -> (value, prob_binder).
  - BoltzAffinityTeacher wraps a real Boltz predict_fn (boltz2_predict) -- run
    `boltz predict --affinity` over your pairs, then distill those labels.
  - SyntheticAffinityTeacher (motif-match) makes the trainer runnable/testable now.

Loss: BCE on P(binder) vs the reference probability + a pairwise ranking margin
(directly optimizes the ordering the benchmark measures).
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from surrogate_affinity import AffinitySurrogate, SurrogateAffinityScorer
from benchmark_surrogate_vs_reference import spearman, topk_recall


# --------------------------------------------------------------------------- #
# Teachers
# --------------------------------------------------------------------------- #
class AffinityTeacher:
    def label(self, target: str, binder: str):
        raise NotImplementedError


class SyntheticAffinityTeacher(AffinityTeacher):
    """Continuous, additive, position-specific affinity proxy for real Boltz-2.

    Each interface position has a fixed per-amino-acid contribution; the label is
    the sum over interface positions (binding ~ additive over contacts). This is
    non-degenerate (continuous, few ties) and learnable, unlike a sparse
    exact-motif-match label -- a faithful stand-in for distilling real affinities.
    """

    _AA = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, interface, seed: int = 123):
        self.interface = list(interface)
        g = torch.Generator().manual_seed(seed)
        # contribution[i, aa] in [0, 1] for interface position i, amino acid aa
        self.contrib = torch.rand(len(self.interface), len(self._AA), generator=g)

    def label(self, target, binder):
        s = 0.0
        for i, p in enumerate(self.interface):
            if p < len(binder):
                aa = self._AA.find(binder[p])
                if aa >= 0:
                    s += self.contrib[i, aa].item()
        return s, s / max(1, len(self.interface))  # value, P(binder) in [0,1]


class BoltzAffinityTeacher(AffinityTeacher):
    """Reads reference affinity from a Boltz predict_fn (boltz2_predict)."""

    def __init__(self, predict_fn, value_key="affinity_pred_value",
                 prob_key="affinity_probability_binary"):
        self.predict_fn = predict_fn
        self.value_key, self.prob_key = value_key, prob_key

    def label(self, target, binder):
        out = self.predict_fn(target, binder)
        return out[self.value_key].reshape(-1)[0].item(), out[self.prob_key].reshape(-1)[0].item()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_AA = "ACDEFGHIKLMNPQRSTVWY"


def make_dataset(wt, interface, teacher, n, rng):
    pairs, probs = [], []
    for _ in range(n):
        b = list(wt)
        for p in interface:
            if rng.random() < 0.5:
                b[p] = rng.choice(_AA)
        binder = "".join(b)
        _, prob = teacher.label(wt, binder)
        pairs.append((wt, binder))
        probs.append(prob)
    return pairs, torch.tensor(probs)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def _forward_probs(surrogate, pairs, target_kv):
    raws, probs = [], []
    for _, binder in pairs:
        out = surrogate.forward(surrogate.embed_seq(binder), *target_kv)
        raws.append(out["affinity_pred_value"].reshape(-1)[0])
        probs.append(out["affinity_probability_binary"].reshape(-1)[0])
    return torch.stack(raws), torch.stack(probs)


def _pairwise_rank_loss(raw, ref_prob, margin=0.1):
    diff_pred = raw.unsqueeze(0) - raw.unsqueeze(1)        # (N, N)
    sign = torch.sign(ref_prob.unsqueeze(0) - ref_prob.unsqueeze(1))
    mask = (sign != 0).float()
    return (F.relu(margin - sign * diff_pred) * mask).sum() / (mask.sum() + 1e-8)


def evaluate(surrogate, pairs, ref_prob):
    scorer = SurrogateAffinityScorer(surrogate)
    surr = scorer.score(pairs)
    return spearman(ref_prob, surr), topk_recall(ref_prob, surr, min(5, len(pairs)))


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_surrogate_affinity(
    teacher: AffinityTeacher = None,
    target: str = "MATEVLADIGSAKLRPQ",
    interface=(2, 4, 8, 12, 15),
    motif: str = "WYFML",
    epochs: int = 60,
    n_train: int = 96,
    n_eval: int = 48,
    embed_dim: int = 64,
    num_heads: int = 4,
    hidden: int = 64,
    lr: float = 3e-3,
    ranking_weight: float = 1.0,
    device: str = None,
    ckpt_path: str = "surrogate_affinity.pt",
    seed: int = 0,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = pick_device(device)
    interface = list(interface)
    teacher = teacher or SyntheticAffinityTeacher(interface)

    train_pairs, train_prob = make_dataset(target, interface, teacher, n_train, rng)
    eval_pairs, eval_prob = make_dataset(target, interface, teacher, n_eval, rng)
    train_prob, eval_prob = train_prob.to(dev), eval_prob.to(dev)

    surrogate = AffinitySurrogate(embed_dim=embed_dim, num_heads=num_heads, hidden=hidden).to(dev)
    opt = torch.optim.AdamW(surrogate.parameters(), lr=lr)

    init_rho, init_recall = evaluate(surrogate, eval_pairs, eval_prob)
    if verbose:
        print(f"[affinity-distill] device={dev}")
        print(f"[affinity-distill] initial: Spearman {init_rho:+.3f} | top-5 recall {init_recall*100:.1f}%")

    final_rho, final_recall = init_rho, init_recall
    for epoch in range(1, epochs + 1):
        surrogate.train()
        kv = surrogate.target_kv(target)  # recomputed each step so grads flow to k/v proj
        raw, prob = _forward_probs(surrogate, train_pairs, kv)
        loss = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), train_prob)
        loss = loss + ranking_weight * _pairwise_rank_loss(raw, train_prob)
        opt.zero_grad(); loss.backward(); opt.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            surrogate.eval()
            final_rho, final_recall = evaluate(surrogate, eval_pairs, eval_prob)
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"Spearman {final_rho:+.3f} | top-5 recall {final_recall*100:.1f}%")

    surrogate.eval()
    final_rho, final_recall = evaluate(surrogate, eval_pairs, eval_prob)
    torch.save(
        {"state_dict": surrogate.state_dict(),
         "config": {"embed_dim": embed_dim, "num_heads": num_heads, "hidden": hidden}},
        ckpt_path,
    )
    if verbose:
        print(f"[affinity-distill] final: Spearman {final_rho:+.3f} | top-5 recall {final_recall*100:.1f}% "
              f"(Spearman {init_rho:+.3f}->{final_rho:+.3f})")
        print(f"[affinity-distill] checkpoint saved -> {ckpt_path}")

    return {"initial_spearman": init_rho, "final_spearman": final_rho,
            "initial_recall": init_recall, "final_recall": final_recall, "ckpt_path": ckpt_path}


def main():
    p = argparse.ArgumentParser(description="Distill Boltz-2 affinities into the surrogate head")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--n-eval", type=int, default=48)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--ranking-weight", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckpt-path", type=str, default="surrogate_affinity.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    train_surrogate_affinity(
        epochs=args.epochs, n_train=args.n_train, n_eval=args.n_eval,
        embed_dim=args.embed_dim, hidden=args.hidden, lr=args.lr,
        ranking_weight=args.ranking_weight, device=args.device,
        ckpt_path=args.ckpt_path, seed=args.seed,
    )
    print(f"[affinity-distill] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
