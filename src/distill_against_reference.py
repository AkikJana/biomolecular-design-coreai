"""Distill AffinitySurrogate against real Boltz reference scores, with a held-out split.

The surrogate is only useful if it reproduces the reference's *ranking* on
binders it was not trained on. Training and evaluating on the same candidates
would report a high correlation that says nothing, so this script folds a
candidate set once, splits it, trains on the training half only, and reports
rank agreement on the held-out half alongside the untrained baseline.

Usage:
    python src/distill_against_reference.py --num-binders 48
    python src/distill_against_reference.py --skip-predict --results-dir <dir> --num-binders 48
"""

import argparse
import json
import random
from pathlib import Path

import torch

from benchmark_surrogate_vs_reference import spearman, kendall_tau, topk_recall
from boltz2_predict import BoltzCliPredictFn
from run_reference_benchmark import (
    DEFAULT_BINDER,
    DEFAULT_TARGET,
    REPO_ROOT,
    make_binder_variants,
    run_boltz,
)
from surrogate_affinity import AffinitySurrogate
from train_surrogate_affinity import _forward_probs, _pairwise_rank_loss


class MultiDirPredictFn:
    """Read predictions that were folded across several batch directories."""

    def __init__(self, results_dirs, name_fn):
        self.readers = [BoltzCliPredictFn(str(d), name_fn) for d in results_dirs]

    def __call__(self, target, binder):
        last = None
        for reader in self.readers:
            try:
                return reader(target, binder)
            except FileNotFoundError as exc:
                last = exc
        raise last


def fold_in_batches(work: Path, target: str, binders, names, batch_size: int,
                    recycling: int, sampling: int):
    """Fold candidates in fixed-size batches.

    A single `boltz predict` over 48 complexes drove this machine into ~11 GB of
    swap, which grew the swapfiles until the volume hit 100%. Memory stays
    bounded per batch, and each batch's results directory is kept separately.
    """
    results_dirs, total_elapsed = [], 0.0
    for start in range(0, len(binders), batch_size):
        chunk = list(zip(binders, names))[start:start + batch_size]
        batch_dir = work / f"batch_{start // batch_size:02d}"
        input_dir = batch_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for binder, name in chunk:
            (input_dir / f"{name}.yaml").write_text(
                "version: 1\nsequences:\n"
                f"  - protein:\n      id: A\n      sequence: {target}\n      msa: empty\n"
                f"  - protein:\n      id: B\n      sequence: {binder}\n      msa: empty\n"
            )
        results_dir, elapsed = run_boltz(input_dir, batch_dir, recycling, sampling, "boltz1")
        total_elapsed += elapsed
        results_dirs.append(results_dir)
        print(f"[boltz] batch {start // batch_size}: {len(chunk)} complexes in "
              f"{elapsed:.0f}s", flush=True)
    return results_dirs, total_elapsed


def rank_metrics(ref: torch.Tensor, surr: torch.Tensor) -> dict:
    k = min(5, ref.numel())
    return {
        "spearman": spearman(ref, surr),
        "kendall_tau": kendall_tau(ref, surr),
        f"top{k}_recall": topk_recall(ref, surr, k),
    }


def score_all(surrogate: AffinitySurrogate, target: str, binders) -> torch.Tensor:
    """Rank scores for every binder, reusing one cached receptor K/V."""
    with torch.no_grad():
        target_kv = surrogate.target_kv(target)
        _, probs = _forward_probs(surrogate, [(target, b) for b in binders], target_kv)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-binders", type=int, default=48)
    ap.add_argument("--test-frac", type=float, default=0.33)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rank-key", default="iptm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=6,
                    help="complexes per boltz invocation; bounds peak memory")
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "distill"))
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    target = DEFAULT_TARGET
    binders = make_binder_variants(DEFAULT_BINDER, args.num_binders)
    names = [f"pair_{i:03d}" for i in range(len(binders))]
    name_for = dict(zip(binders, names))

    if args.skip_predict:
        if not args.results_dir:
            ap.error("--skip-predict requires --results-dir")
        results_dirs = [Path(d) for d in args.results_dir.split(",")]
    else:
        results_dirs, elapsed = fold_in_batches(
            work, target, binders, names, args.batch_size,
            args.recycling_steps, args.sampling_steps,
        )
        print(f"[boltz] folded {len(binders)} complexes in {elapsed:.1f}s "
              f"({elapsed / len(binders) * 1000:.0f} ms/candidate)", flush=True)

    predict_fn = MultiDirPredictFn(results_dirs, lambda t, b: name_for[b])
    ref_all = torch.tensor(
        [predict_fn(target, b)[args.rank_key].reshape(-1)[0].item() for b in binders]
    )
    print(f"reference {args.rank_key}: min={ref_all.min():.4f} max={ref_all.max():.4f} "
          f"std={ref_all.std():.4f} over {len(binders)} candidates")
    if float(ref_all.max() - ref_all.min()) == 0.0:
        raise SystemExit("reference is constant; nothing to learn or measure")

    # Held-out split. The test half is never seen during training.
    rng = random.Random(args.seed)
    order = list(range(len(binders)))
    rng.shuffle(order)
    n_test = max(2, int(round(len(binders) * args.test_frac)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    train_binders = [binders[i] for i in train_idx]
    test_binders = [binders[i] for i in test_idx]
    ref_train, ref_test = ref_all[train_idx], ref_all[test_idx]
    print(f"split: {len(train_binders)} train / {len(test_binders)} held-out test")

    torch.manual_seed(args.seed)
    surrogate = AffinitySurrogate(embed_dim=args.embed_dim, hidden=args.hidden)

    baseline_test = rank_metrics(ref_test, score_all(surrogate, target, test_binders))
    print(f"\nUNTRAINED  held-out: rho={baseline_test['spearman']:+.3f} "
          f"tau={baseline_test['kendall_tau']:+.3f}")

    opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr)
    train_pairs = [(target, b) for b in train_binders]
    for epoch in range(args.epochs):
        opt.zero_grad()
        target_kv = surrogate.target_kv(target)
        raw, _ = _forward_probs(surrogate, train_pairs, target_kv)
        loss = _pairwise_rank_loss(raw, ref_train)
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            tr = rank_metrics(ref_train, score_all(surrogate, target, train_binders))
            te = rank_metrics(ref_test, score_all(surrogate, target, test_binders))
            print(f"  epoch {epoch+1:4d}  loss={loss.item():.4f}  "
                  f"train rho={tr['spearman']:+.3f}  held-out rho={te['spearman']:+.3f}",
                  flush=True)

    trained_train = rank_metrics(ref_train, score_all(surrogate, target, train_binders))
    trained_test = rank_metrics(ref_test, score_all(surrogate, target, test_binders))

    print("\n" + "=" * 64)
    print(f"Distillation vs boltz1({args.rank_key})   N={len(binders)}"
          f"  ({len(train_binders)} train / {len(test_binders)} held-out)")
    print("=" * 64)
    print(f"  {'':22}{'train':>10}{'held-out':>12}")
    for key in ("spearman", "kendall_tau"):
        print(f"  {key:22}{trained_train[key]:>+10.3f}{trained_test[key]:>+12.3f}")
    print(f"  {'held-out rho (before)':22}{'':>10}{baseline_test['spearman']:>+12.3f}")
    print("-" * 64)
    print("  The held-out column is the only one that means anything. A train")
    print("  score far above it is memorisation, not distillation.")
    print("=" * 64)

    ckpt = work / "surrogate_affinity_distilled.pt"
    torch.save(surrogate.state_dict(), ckpt)
    summary = {
        "n_candidates": len(binders),
        "n_train": len(train_binders),
        "n_test": len(test_binders),
        "rank_key": args.rank_key,
        "reference_spread": {"min": ref_all.min().item(), "max": ref_all.max().item(),
                             "std": ref_all.std().item()},
        "untrained_heldout": baseline_test,
        "trained_train": trained_train,
        "trained_heldout": trained_test,
        "epochs": args.epochs,
    }
    (work / "distill_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"checkpoint: {ckpt}\nsummary:    {work / 'distill_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
