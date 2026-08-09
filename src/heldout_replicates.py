"""How much of Section 7.10 was one draw of the dice?

Section 7.5 established that a single unseeded fold does not reproduce its own
per-receptor ranking, and Section 8.2 recommends replicate averaging for any
per-receptor claim. Section 7.10 was then written from a single draw of the
held-out panel. A second, independently drawn set of the same 132 folds moved
the interface-pLDDT cognate-versus-scramble p-value from 0.089 to 0.00008 -- the
same panel, the same settings, a different seed.

That is the dissertation's own finding applied to its own headline, so the
held-out result is re-estimated here from every available draw rather than from
whichever one was run first.

The two families of test behave differently under replication, and the
difference is the point:

  order sensitivity      (cognate minus its own scramble) is unstable across
                         draws and needs averaging before it means anything
  receptor specificity   (cognate ranked among its own decoys) is stable, and
                         says the same thing in every draw

Usage:
    python src/heldout_replicates.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
PANEL = ART / "heldout_panel"
warnings.filterwarnings("ignore")

FLIP = {"iface_pae", "mpae"}
BASE_METRICS = ("iptm", "iface_plddt", "receptor_side")
PAE_METRICS = ("iface_pae", "mpae", "pae_frac_lt10", "ipsae", "pdockq2")


def draws():
    """Every completed independent draw of the held-out panel, oldest first."""
    found = []
    for f in sorted(PANEL.glob("heldout_scores*.json")):
        recs = json.loads(f.read_text())
        if len(recs) >= 100:
            found.append((f.stem.replace("heldout_scores", "") or "_1", recs))
    return found


def val(r, m):
    return -r[m] if m in FLIP else r[m]


def effect(recs, m):
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s:
            diffs += [val(c[0], m) - val(x, m) for x in s]
    if len(diffs) < 3:
        return None
    d = np.array(diffs, float)
    return {"effect": float(d.mean()),
            "p": float(stats.ttest_1samp(d, 0).pvalue)}


def rank(recs, m):
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    ranks, sizes = [], []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        dd = [x for x in g if x["label"] == "decoy"]
        if not c or not dd:
            continue
        sc = [val(c[0], m)] + [val(x, m) for x in dd]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    if len(ranks) < 5:
        return None
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    return {"mean_rank": float(r.mean()), "chance": float(exp.mean()),
            "first": int((r == 1).sum()), "n": len(r),
            "p": float(stats.wilcoxon(r - exp)[1])}


def averaged(sets, metrics):
    """Fold-wise mean across draws, keeping only folds present in all of them."""
    byname = {}
    for tag, recs in sets:
        for r in recs:
            byname.setdefault(r["name"], {})[tag] = r
    common = [n for n, v in byname.items() if len(v) == len(sets)]
    out = []
    for n in common:
        vs = list(byname[n].values())
        rec = {k: vs[0][k] for k in ("name", "receptor_id", "label", "peptide_from")
               if k in vs[0]}
        for m in metrics:
            if all(m in v for v in vs):
                rec[m] = float(np.mean([v[m] for v in vs]))
        out.append(rec)
    return out, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "heldout_replicates.json"))
    args = ap.parse_args()

    sets = draws()
    if len(sets) < 2:
        raise SystemExit(f"need at least two draws, found {len(sets)}")
    print(f"{len(sets)} independent draws: {', '.join(t for t, _ in sets)}")

    # a metric counts only if every draw carries it
    metrics = [m for m in BASE_METRICS + PAE_METRICS
               if all(any(m in r for r in recs) for _, recs in sets)]
    avg, n_common = averaged(sets, metrics)
    print(f"{n_common} folds present in every draw; metrics: {', '.join(metrics)}\n")

    result = {"n_draws": len(sets), "n_common_folds": n_common, "metrics": {}}

    print(f"{'ORDER SENSITIVITY':16} cognate minus its own scramble")
    head = "".join(f"{'draw ' + t:>20}" for t, _ in sets)
    print(f"{'metric':16}{head}{'mean of draws':>22}")
    print("-" * (16 + 20 * len(sets) + 22))
    for m in metrics:
        cells = ""
        for _, recs in sets:
            e = effect(recs, m)
            cells += f"{e['effect']:+11.3f} p{e['p']:8.5f}" if e else f"{'-':>20}"
        a = effect(avg, m)
        result["metrics"].setdefault(m, {})["effect_mean"] = a
        result["metrics"][m]["effect_per_draw"] = [effect(r, m) for _, r in sets]
        print(f"{m:16}{cells}{a['effect']:+13.3f} p{a['p']:8.5f}")

    print(f"\n{'RECEPTOR SPECIFICITY':16} cognate ranked among its own decoys "
          f"(chance 2.50)")
    print(f"{'metric':16}{head}{'mean of draws':>22}")
    print("-" * (16 + 20 * len(sets) + 22))
    for m in metrics:
        cells = ""
        for _, recs in sets:
            r = rank(recs, m)
            cells += f"{r['mean_rank']:9.2f} p{r['p']:7.4f} {r['first']:2d}#1" if r \
                else f"{'-':>20}"
        a = rank(avg, m)
        result["metrics"][m]["rank_mean"] = a
        result["metrics"][m]["rank_per_draw"] = [rank(r, m) for _, r in sets]
        print(f"{m:16}{cells}{a['mean_rank']:11.2f} p{a['p']:7.4f} {a['first']:2d}#1")

    # how much does a single draw move each family of test?
    print(f"\n{'=' * 70}\nSpread across draws\n{'=' * 70}")
    print(f"{'metric':16} {'effect range':>22} {'rank range':>18}")
    for m in metrics:
        es = [e["effect"] for e in result["metrics"][m]["effect_per_draw"] if e]
        rs = [r["mean_rank"] for r in result["metrics"][m]["rank_per_draw"] if r]
        print(f"{m:16} {min(es):+9.3f} to {max(es):+9.3f} "
              f"{min(rs):8.2f} to {max(rs):6.2f}")
    print("\nThe effect column is what moves; the rank column is what does not.")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
