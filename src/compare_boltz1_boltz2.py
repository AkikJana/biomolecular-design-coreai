"""Compare Boltz-1 and Boltz-2 on the PDB cognate/decoy/scrambled benchmark.

Both runs fold identical pairs with identical seeds and byte-identical cached
MSAs, so the checkpoint is the only variable.

Two questions, deliberately kept apart:

* **Sensitivity** -- do real complexes score above scrambled sequence? Boltz-1
  already passed this (AUC 0.881). It is the easy question: a scrambled peptide
  is not a peptide-shaped object at all.
* **Specificity** -- among peptides that are all real binders *of some other
  receptor*, does the cognate one win? This is what a screening reference has
  to do, and it is where Boltz-1 failed (cognate #1 for 2/11 receptors,
  Wilcoxon p=0.746).

The pooled cognate-vs-decoy AUC that the benchmark prints is confounded by
receptor identity -- receptors differ several-fold in baseline ipTM, so a
pooled test partly measures "which receptor is this" rather than "is this the
right peptide". The within-receptor rank test below removes that by comparing
each cognate only against decoys on its own receptor.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path):
    pairs = json.loads(Path(path).read_text())
    return [p for p in pairs if not np.isnan(p.get("score", float("nan")))]


def pooled_stats(pairs, label_a, label_b):
    a = np.array([p["score"] for p in pairs if p["label"] == label_a])
    b = np.array([p["score"] for p in pairs if p["label"] == label_b])
    if len(a) < 3 or len(b) < 3:
        return None
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    return {"auc": u / (len(a) * len(b)), "p": p,
            "mean_a": a.mean(), "mean_b": b.mean(), "n_a": len(a), "n_b": len(b)}


def within_receptor(pairs, include_scrambled=False):
    """For each receptor, where does its cognate rank among its own competitors?

    Returns (ranks, n_competitors, n_first). rank 1 = cognate scored highest.
    """
    by_rid = {}
    for p in pairs:
        by_rid.setdefault(p["receptor_id"], []).append(p)

    ranks, sizes, first = [], [], 0
    for rid, group in sorted(by_rid.items()):
        cog = [p for p in group if p["label"] == "cognate"]
        comp = [p for p in group if p["label"] == "decoy"]
        if include_scrambled:
            comp += [p for p in group if p["label"] == "scrambled"]
        if len(cog) != 1 or not comp:
            continue
        scores = [cog[0]["score"]] + [c["score"] for c in comp]
        # rank 1 = highest score; ties broken against the cognate (conservative)
        rank = 1 + sum(s >= scores[0] for s in scores[1:])
        ranks.append(rank)
        sizes.append(len(scores))
        first += rank == 1
    return np.array(ranks), np.array(sizes), first


def rank_test(ranks, sizes):
    """Is the cognate ranked better than chance?

    Under the null the cognate is a uniform draw from 1..n, so its expected
    rank is (n+1)/2. Wilcoxon signed-rank on (observed - expected).
    """
    if len(ranks) < 5:
        return None
    expected = (sizes + 1) / 2
    diff = ranks - expected
    if np.allclose(diff, 0):
        return {"p": 1.0, "mean_rank": ranks.mean(), "mean_expected": expected.mean()}
    stat, p = stats.wilcoxon(diff, alternative="less")   # better = lower rank
    return {"p": p, "stat": stat, "mean_rank": ranks.mean(),
            "mean_expected": expected.mean()}


def describe(name, pairs):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    for label in ("cognate", "decoy", "scrambled"):
        v = np.array([p["score"] for p in pairs if p["label"] == label])
        if len(v):
            print(f"  {label:10} n={len(v):3d}  mean {v.mean():.4f}  "
                  f"sd {v.std():.4f}  range {v.min():.4f}-{v.max():.4f}")

    print("\n  -- sensitivity (pooled) --")
    for a, b in (("cognate", "scrambled"), ("cognate", "decoy")):
        s = pooled_stats(pairs, a, b)
        if s:
            verdict = "SEPARATES" if (s["p"] < 0.05 and s["auc"] > 0.5) else "does NOT separate"
            print(f"  {a} > {b:10}  AUC {s['auc']:.3f}  p {s['p']:.4g}  -> {verdict}")

    print("\n  -- specificity (within receptor, cognate vs its own decoys) --")
    ranks, sizes, first = within_receptor(pairs)
    if len(ranks):
        print(f"  cognate ranked #1 for {first}/{len(ranks)} receptors "
              f"(chance ~{np.mean(1 / sizes) * len(ranks):.1f})")
        print(f"  mean rank {ranks.mean():.2f} of {sizes.mean():.1f} competitors "
              f"(chance {((sizes + 1) / 2).mean():.2f})")
        t = rank_test(ranks, sizes)
        if t:
            verdict = "better than chance" if t["p"] < 0.05 else "NOT better than chance"
            print(f"  Wilcoxon p {t['p']:.4f}  -> {verdict}")
    return ranks, sizes, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boltz1", default=str(REPO_ROOT / "artifacts" / "pdb_binders" /
                                            "pdb_binder_scores.json"))
    ap.add_argument("--boltz2", default=str(REPO_ROOT / "artifacts" / "pdb_binders_b2" /
                                            "pdb_binder_scores.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "boltz1_vs_boltz2.json"))
    args = ap.parse_args()

    p1, p2 = load(args.boltz1), load(args.boltz2)
    r1, s1, f1 = describe("Boltz-1", p1)
    r2, s2, f2 = describe("Boltz-2", p2)

    print(f"\n{'=' * 72}\nPaired comparison\n{'=' * 72}")

    # Paired on receptor: same receptor, same decoys, different checkpoint.
    common = sorted({p["receptor_id"] for p in p1} & {p["receptor_id"] for p in p2})
    print(f"  {len(common)} receptors in both runs")

    def rank_map(pairs):
        by = {}
        ranks, sizes, _ = within_receptor(pairs)
        rids = sorted({p["receptor_id"] for p in pairs})
        # within_receptor iterates sorted rids and skips incomplete groups;
        # rebuild the mapping explicitly to stay aligned.
        i = 0
        for rid in rids:
            grp = [p for p in pairs if p["receptor_id"] == rid]
            cog = [p for p in grp if p["label"] == "cognate"]
            comp = [p for p in grp if p["label"] == "decoy"]
            if len(cog) != 1 or not comp:
                continue
            by[rid] = (ranks[i], sizes[i])
            i += 1
        return by

    m1, m2 = rank_map(p1), rank_map(p2)
    both = [r for r in common if r in m1 and r in m2]
    if both:
        print(f"\n  {'receptor':10} {'boltz1':>14} {'boltz2':>14}")
        for rid in both:
            print(f"  {rid:10} {f'#{m1[rid][0]} of {m1[rid][1]}':>14} "
                  f"{f'#{m2[rid][0]} of {m2[rid][1]}':>14}")
        a = np.array([m1[r][0] for r in both], dtype=float)
        b = np.array([m2[r][0] for r in both], dtype=float)
        print(f"\n  mean cognate rank: boltz1 {a.mean():.2f}  boltz2 {b.mean():.2f}")
        if not np.allclose(a - b, 0):
            _, pp = stats.wilcoxon(a, b)
            print(f"  paired Wilcoxon (boltz1 vs boltz2 ranks) p {pp:.4f}")
        else:
            print("  identical ranks under both models")

    out = {
        "boltz1": {"n": len(p1), "cognate_first": int(f1), "n_receptors": len(r1),
                   "mean_rank": float(r1.mean()) if len(r1) else None},
        "boltz2": {"n": len(p2), "cognate_first": int(f2), "n_receptors": len(r2),
                   "mean_rank": float(r2.mean()) if len(r2) else None},
        "per_receptor": {r: {"boltz1_rank": int(m1[r][0]), "boltz2_rank": int(m2[r][0]),
                             "n_competitors": int(m1[r][1])} for r in both},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
