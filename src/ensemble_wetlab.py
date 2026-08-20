"""Does ensembling across architectures help, and does lineage explain it?

Section 7.9.4 measured a cross-model ensemble worth +0.017 and originally read
that as "ensembling does not help". The three arms ensembled there were Boltz-2,
Boltz-1 and DeCAF, and DeCAF is distilled from Boltz-1, so two shared weights and
all three shared an architecture. The correction to that section asserts a
mechanism -- that averaging correlated predictors buys nothing -- and supports it
by citing [25] rather than by measuring it.

The data to measure it is now on disk. [25] released ipSAE from ten
independently developed predictors on the same 1,320 designs, with binding
measured by two CROs. Two of the ten pairs share a lineage by construction:

    ESMFold2 and ESMFold2-Fast        the same model and its fast variant
    OpenFold3 and AlphaFold3/OF3      the same weights under two inference codes

The rest are unrelated architectures. That gives a pre-specified contrast --
chosen by lineage, not by which pairs happen to score well -- between what a
within-lineage ensemble buys and what a cross-lineage one buys.

Scores are z-scored within target before averaging, which is [25]'s convention
and Section 7.10's. Differences are bootstrapped paired over targets, because
comparing two intervals is not a test of their difference.

Usage:
    python src/ensemble_wetlab.py
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
DATA = ART / "anthropic_binder" / "design_summary.csv"
warnings.filterwarnings("ignore")

PRED = {"ef2fast": "ESMFold2-Fast", "ef2full": "ESMFold2", "ptxv2": "Protenix v2",
        "odde": "OpenDDE", "afm3": "AlphaFold-Multimer v3", "boltz2": "Boltz-2",
        "chai1": "Chai-1", "of3": "OpenFold3", "rf3": "RoseTTAFold3",
        "af3of3": "AlphaFold3/OF3 weights"}

# Declared from what the models are, before any score was looked at.
SAME_LINEAGE = [("ef2full", "ef2fast"), ("of3", "af3of3")]


def zcols(df, keys):
    """Within-target z for each predictor, which is what makes them averageable."""
    out = pd.DataFrame(index=df.index)
    for k in keys:
        c = f"ipsae_min_{k}"
        out[k] = df.groupby("target")[c].transform(
            lambda s: (s - s.mean()) / (s.std() or 1))
    return out


def score(df, vals):
    d = pd.DataFrame({"target": df["target"], "y": df["y"], "v": vals}).dropna()
    aps = [average_precision_score(g["y"], g["v"])
           for _, g in d.groupby("target") if g["y"].nunique() > 1]
    z = d.groupby("target")["v"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    return float(np.mean(aps)), float(roc_auc_score(d["y"], z))


def paired_ci(df, keys_a, keys_b, n=2000, seed=0):
    """Bootstrap the difference between two ensembles, resampling targets."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("target")]
    diffs = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            a = score(s, zcols(s, keys_a).mean(axis=1))[0]
            b = score(s, zcols(s, keys_b).mean(axis=1))[0]
            diffs.append(a - b)
        except Exception:                                          # noqa: BLE001
            pass
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ART / "ensemble_wetlab.json"))
    args = ap.parse_args()

    d = pd.read_csv(DATA, low_memory=False)
    d = d[d["binder_final"].isin([True, False])].copy()
    d["y"] = d["binder_final"].astype(int)
    keys = list(PRED)
    Z = zcols(d, keys)

    print(f"{len(d)} designs, {d['y'].sum()} binders, {d['target'].nunique()} targets\n")

    singles = {k: score(d, Z[k]) for k in keys}
    best_k = max(singles, key=lambda k: singles[k][0])
    print(f"  best single predictor: {PRED[best_k]} "
          f"(macro-AP {singles[best_k][0]:.3f})\n")

    print("  correlation of within-target z scores")
    corr = Z.corr(method="spearman")
    cross = [corr.loc[a, b] for a, b in combinations(keys, 2)
             if (a, b) not in SAME_LINEAGE and (b, a) not in SAME_LINEAGE]
    for a, b in SAME_LINEAGE:
        print(f"    same lineage   {PRED[a]:24} + {PRED[b]:24} rho {corr.loc[a,b]:.3f}")
    print(f"    cross lineage  mean rho over the other {len(cross)} pairs      "
          f"{np.mean(cross):.3f}  (range {min(cross):.2f} to {max(cross):.2f})")

    print("\n  ensembles, macro-AP / within-target AUC")
    rows, result = [], {}
    all_ap, all_auc = score(d, Z[keys].mean(axis=1))
    rows.append(("all ten predictors", all_ap, all_auc))
    for a, b in SAME_LINEAGE:
        s = score(d, Z[[a, b]].mean(axis=1))
        rows.append((f"same lineage: {PRED[a]} + {PRED[b]}", *s))
    # cross-lineage pairs matched to the same-lineage members, so the contrast is
    # between lineages and not between which models happen to be strong
    for a, b in [("ef2full", "rf3"), ("of3", "ef2full"), ("boltz2", "ptxv2")]:
        s = score(d, Z[[a, b]].mean(axis=1))
        rows.append((f"cross lineage: {PRED[a]} + {PRED[b]}", *s))
    rows.append((f"best single ({PRED[best_k]})", *singles[best_k]))
    for label, a, u in rows:
        print(f"    {label:52}{a:.3f}  {u:.3f}")

    print("\n  paired differences against the best single predictor")
    for label, keys_a in [("all ten", keys)] + \
            [(f"same lineage {PRED[a]}+{PRED[b]}", [a, b]) for a, b in SAME_LINEAGE] + \
            [(f"cross lineage {PRED[a]}+{PRED[b]}", [a, b])
             for a, b in [("ef2full", "rf3"), ("of3", "ef2full"), ("boltz2", "ptxv2")]]:
        obs = score(d, Z[keys_a].mean(axis=1))[0] - singles[best_k][0]
        lo, hi = paired_ci(d, keys_a, [best_k], args.n_boot)
        flag = "excludes 0" if lo * hi > 0 else "includes 0"
        print(f"    {label:44}{obs:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {flag}")
        result[label] = {"delta_ap": obs, "ci": [lo, hi]}

    # A pair compared against the best of ten is compared against a baseline
    # chosen by taking a maximum, which is optimistic. The question a pair
    # actually poses is whether averaging beats its own better member.
    print("\n  each pair against the better of its own two members")
    for a, b in SAME_LINEAGE + [("ef2full", "rf3"), ("of3", "ef2full"),
                                ("boltz2", "ptxv2")]:
        better = a if singles[a][0] >= singles[b][0] else b
        obs = score(d, Z[[a, b]].mean(axis=1))[0] - singles[better][0]
        lo, hi = paired_ci(d, [a, b], [better], args.n_boot)
        kind = "same " if (a, b) in SAME_LINEAGE else "cross"
        flag = "excludes 0" if lo * hi > 0 else "includes 0"
        print(f"    {kind} {PRED[a]:22}+{PRED[b]:22}{obs:+.3f}"
              f"  [{lo:+.3f}, {hi:+.3f}]  {flag}")
        result[f"pair_vs_better_{a}+{b}"] = {"kind": kind.strip(),
                                             "delta_ap": obs, "ci": [lo, hi]}

    Path(args.out).write_text(json.dumps(
        {"best_single": best_k, "singles": {k: v for k, v in singles.items()},
         "same_lineage_rho": {f"{a}+{b}": float(corr.loc[a, b])
                              for a, b in SAME_LINEAGE},
         "cross_lineage_rho_mean": float(np.mean(cross)),
         "deltas": result}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
