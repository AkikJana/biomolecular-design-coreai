"""Require the match to hold in both directions before calling it.

Every test in this dissertation reads the panel one way: for a given receptor,
is the cognate the best of its candidate peptides? The panel also contains the
transpose -- each peptide is folded against its own receptor and against the
several that borrowed it as a decoy -- and that direction asks a different and
equally reasonable question: for a given peptide, is its own receptor the best
of the targets it was tried against?

Requiring both is the binding analogue of reciprocal best hits in homology
search. It is not a better score. It is the same scores read as a competition in
two directions, so it costs nothing beyond folds already run, and it trades
recall for precision in the direction a screener wants: fewer calls, each more
likely to be real, when every call costs a wet-lab experiment.

This validates a result that was first measured on one panel, one draw and
thirteen calls. Two claims died on replication earlier in this work, so the same
test is run here on both panels and on every available draw, with a permutation
null: the reciprocal filter must beat discarding the same number of calls at
random, or it is only shrinking the candidate set.

Usage:
    python src/reciprocal_match.py
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
ART = REPO_ROOT / "artifacts"
PANEL = ART / "heldout_panel"
warnings.filterwarnings("ignore")

MIN_TARGETS = 3    # a peptide needs several targets before "its best" means anything


def in_training_rows(metric):
    """(receptor, peptide, is_cognate, score) for the main panel, DeCAF."""
    main = json.loads((ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())
    dec = {r["job"]: r for r in json.loads(
        (ART / "decaf_scramble_result.json").read_text())["per_fold"]}
    return [{"rec": p["receptor_id"], "pep": p["peptide"],
             "cog": p["receptor_id"] == p["peptide_from"], "v": dec[p["name"]][metric]}
            for p in main
            if p["name"] in dec and p["label"] in ("cognate", "decoy")]


def heldout_rows(metric, store):
    """Same, for a held-out draw. Peptide sequences are rebuilt from the seed."""
    from heldout_panel import ALREADY, NEW
    from pdb_binder_benchmark import build_pairs
    seqdir = PANEL / "sequences"
    complexes = {}
    for pid in ALREADY + NEW:
        f = seqdir / f"{pid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            complexes[pid] = {"receptor": d["receptor"], "peptide": d["peptide"]}
    pairs = build_pairs(complexes, 3, 2, 0)
    for i, p in enumerate(pairs):
        p["name"] = f"ho_{i:03d}"
    scored = {r["name"]: r for r in json.loads(store.read_text())}
    return [{"rec": p["receptor_id"], "pep": p["peptide"],
             "cog": p["receptor_id"] == p["peptide_from"],
             "v": scored[p["name"]][metric]}
            for p in pairs
            if p["name"] in scored and p["label"] in ("cognate", "decoy")]


def reciprocal(rows, rng, n_perm=20000):
    """Precision of one-directional and reciprocal calls, and a permutation null."""
    byrec, bypep = defaultdict(list), defaultdict(list)
    for r in rows:
        byrec[r["rec"]].append(r)
        bypep[r["pep"]].append(r)
    best_rec = {k: max(v, key=lambda x: x["v"])["pep"] for k, v in byrec.items()}
    best_pep = {k: max(v, key=lambda x: x["v"])["rec"]
                for k, v in bypep.items() if len(v) >= MIN_TARGETS}
    truth = {(r["rec"], r["pep"]) for r in rows if r["cog"]}

    uni = [(r, p) for r, p in best_rec.items() if p in best_pep]
    rec = [(r, p) for r, p in uni if best_pep[p] == r]
    if not uni or not rec:
        return None
    tp_u = sum(x in truth for x in uni)
    tp_r = sum(x in truth for x in rec)
    base = len(truth) / len(rows)

    # null: discard the same number of calls at random
    obs = tp_r / len(rec)
    null = []
    for _ in range(n_perm):
        keep = rng.choice(len(uni), len(rec), replace=False)
        null.append(np.mean([uni[i] in truth for i in keep]))
    return {"n_uni": len(uni), "tp_uni": tp_u, "prec_uni": tp_u / len(uni),
            "n_rec": len(rec), "tp_rec": tp_r, "prec_rec": obs,
            "base_rate": base, "ef_uni": (tp_u / len(uni)) / base,
            "ef_rec": obs / base,
            "dropped": len(uni) - len(rec),
            "dropped_wrong": sum(x not in truth for x in uni if x not in rec),
            "dropped_right": sum(x in truth for x in uni if x not in rec),
            "p_vs_random_discard": float((np.array(null) >= obs).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "reciprocal_match.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    sets = [("in-training", lambda m: in_training_rows(m))]
    for f in sorted(PANEL.glob("heldout_scores*.json")):
        tag = f.stem.replace("heldout_scores", "") or "_1"
        sets.append((f"held-out{tag}", lambda m, f=f: heldout_rows(m, f)))

    print(f"{'panel':16} {'metric':13} {'one-dir':>16} {'reciprocal':>17} "
          f"{'dropped':>16} {'perm p':>8}")
    print("-" * 92)
    out = {}
    for name, loader in sets:
        for metric in ("iface_plddt", "iptm"):
            try:
                rows = loader(metric)
            except Exception:
                continue
            r = reciprocal(rows, rng)
            if not r:
                continue
            out.setdefault(name, {})[metric] = r
            print(f"{name:16} {metric:13} "
                  f"{r['tp_uni']:3d}/{r['n_uni']:<3d} {r['prec_uni']:5.0%}   "
                  f"{r['tp_rec']:3d}/{r['n_rec']:<3d} {r['prec_rec']:5.0%}      "
                  f"{r['dropped_wrong']:2d} wrong,{r['dropped_right']:2d} right "
                  f"{r['p_vs_random_discard']:8.4f}")
        print()

    print("A filter that only shrinks the candidate set would drop wrong and right")
    print("calls in proportion. The permutation p asks exactly that.\n")
    for metric in ("iface_plddt", "iptm"):
        got = [(k, v[metric]) for k, v in out.items() if metric in v]
        ho = [v for k, v in got if k.startswith("held-out")]
        if not ho:
            continue
        print(f"{metric}: held-out draws  "
              f"one-dir {np.mean([x['prec_uni'] for x in ho]):.0%} -> "
              f"reciprocal {np.mean([x['prec_rec'] for x in ho]):.0%}  "
              f"(mean of {len(ho)} draws), "
              f"perm p {np.mean([x['p_vs_random_discard'] for x in ho]):.4f}")

    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
