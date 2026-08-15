"""Does a real binder land in the same place twice?

Every readout tested so far is a number the model reports about itself, so all
of them are bounded by the variance decomposition of Section 7.9.2 and none of
them beat it -- combining readouts loses, combining models gains 0.017 with an
interval spanning zero, and a physics energy function costs 0.033.

This asks a different question, of the coordinates rather than the confidence
head. Fold the same pair several times with different sampling noise. A peptide
that genuinely binds has one favourable pose to find, so the draws should agree.
A peptide that does not has no such basin, so the draws should scatter. The
score is agreement, not confidence, and nothing in Section 7.9's decomposition
constrains it.

It only became testable once Section 7.11 established that a distilled model
produces physical backbones at ten steps: comparing poses between two point
clouds would measure nothing.

Method. For each fold, superpose every pair of draws on the **receptor** CA
atoms, then measure RMSD over the **peptide** CA atoms. Superposing on the
receptor is what makes the number a statement about where the peptide sits
relative to its target rather than about global drift. Lower spread is better,
so the score is negated to keep "higher is better" throughout.

Reported against the same two tests as every other readout: cognate versus its
own scramble, and cognate ranked among its own decoys.

Usage:
    python src/pose_convergence.py
"""

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
ART = REPO_ROOT / "artifacts"
PANEL = ART / "heldout_panel"
warnings.filterwarnings("ignore")


def kabsch_rmsd(P, Q):
    """RMSD of P onto Q after optimal superposition (Kabsch)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    diff = (Pc @ R) - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def apply_receptor_frame(rec_a, pep_a, rec_b, pep_b):
    """Superpose draw A on draw B by receptor, return peptide RMSD.

    The rotation is fitted on the receptor alone and then applied to the
    peptide, so the peptide RMSD measures displacement *relative to the target*
    rather than being minimised in its own right.
    """
    ca, cb = rec_a.mean(0), rec_b.mean(0)
    A, B = rec_a - ca, rec_b - cb
    V, S, Wt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    moved = (pep_a - ca) @ R
    target = pep_b - cb
    return float(np.sqrt(((moved - target) ** 2).sum() / len(pep_a)))


def chains_ca(path, parser):
    """(receptor CA, peptide CA) as arrays, shortest chain treated as peptide."""
    model = parser.get_structure("x", str(path))[0]
    ch = []
    for c in model:
        ca = np.array([r["CA"].coord for r in c if "CA" in r], dtype=float)
        if len(ca):
            ch.append(ca)
    if len(ch) != 2:
        return None
    ch.sort(key=len)
    return ch[1], ch[0]


def collect(tags):
    """{fold name: {tag: (receptor CA, peptide CA)}} across the given runs."""
    parser = PDBParser(QUIET=True)
    out = {}
    for tag in tags:
        for pdb in PANEL.glob(f"b{tag}*/boltz_results_inputs/predictions/*/*_model_0.pdb"):
            if pdb.stat().st_size == 0:
                continue
            name = pdb.parent.name
            got = chains_ca(pdb, parser)
            if got:
                out.setdefault(name, {})[tag] = got
    return out


def spread(entry):
    """Mean pairwise peptide RMSD across draws, and the receptor's own spread."""
    tags = sorted(entry)
    if len(tags) < 2:
        return None
    pep, rec = [], []
    for a, b in combinations(tags, 2):
        ra, pa = entry[a]
        rb, pb = entry[b]
        if ra.shape != rb.shape or pa.shape != pb.shape:
            continue
        pep.append(apply_receptor_frame(ra, pa, rb, pb))
        rec.append(kabsch_rmsd(ra, rb))
    if not pep:
        return None
    return float(np.mean(pep)), float(np.mean(rec)), len(pep)


def evaluate(rows, metric, label):
    by = {}
    for r in rows:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s:
            diffs += [c[0][metric] - x[metric] for x in s]
    ranks, sizes = [], []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        d = [x for x in g if x["label"] == "decoy"]
        if not c or not d:
            continue
        sc = [c[0][metric]] + [x[metric] for x in d]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    e = np.array(diffs, float)
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    out = {"effect": float(e.mean()) if len(e) else float("nan"),
           "p": float(stats.ttest_1samp(e, 0).pvalue) if len(e) > 2 else float("nan"),
           "mean_rank": float(r.mean()) if len(r) else float("nan"),
           "chance": float(exp.mean()) if len(r) else float("nan"),
           "first": int((r == 1).sum()) if len(r) else 0,
           "n_receptors": len(r),
           "rank_p": (float(stats.wilcoxon(r - exp)[1]) if len(r) > 4 else float("nan"))}
    print(f"{label:32} {out['effect']:+9.3f} {out['p']:9.5f} "
          f"{out['mean_rank']:6.2f} {out['rank_p']:8.4f} "
          f"{out['first']:3d}/{out['n_receptors']:<3d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="p1,p2",
                    help="run tags whose kept structures should be compared")
    ap.add_argument("--out", default=str(ART / "pose_convergence.json"))
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    entries = collect(tags)
    usable = {k: v for k, v in entries.items() if len(v) >= 2}
    print(f"tags {tags}: {len(entries)} folds seen, {len(usable)} with >= 2 draws")
    if len(usable) < 30:
        raise SystemExit("too few folds with repeated structures to test")

    meta = {}
    for f in PANEL.glob("heldout_scores*.json"):
        for r in json.loads(f.read_text()):
            meta.setdefault(r["name"], r)

    rows = []
    for name, entry in usable.items():
        s = spread(entry)
        if not s or name not in meta:
            continue
        pep, rec, npair = s
        m = meta[name]
        rows.append({"name": name, "receptor_id": m["receptor_id"],
                     "label": m["label"], "peptide_rmsd": pep,
                     "receptor_rmsd": rec, "n_pairs": npair,
                     # lower spread is better, so negate for "higher is better"
                     "pose_agreement": -pep,
                     "iface_plddt": m.get("iface_plddt"), "iptm": m.get("iptm")})
    print(f"{len(rows)} folds scored, {len({r['receptor_id'] for r in rows})} receptors\n")

    for lab in ("cognate", "decoy", "scrambled"):
        v = [r["peptide_rmsd"] for r in rows if r["label"] == lab]
        w = [r["receptor_rmsd"] for r in rows if r["label"] == lab]
        if v:
            print(f"  {lab:10} peptide RMSD across draws {np.mean(v):6.2f} A"
                  f"   (receptor {np.mean(w):5.2f} A)")
    print("\n  If the receptor RMSD is small and similar across classes, the")
    print("  peptide numbers are about the peptide and not about global drift.\n")

    print(f"{'readout':32} {'cog-scr':>9} {'p':>9} {'rank':>6} {'p':>8} {'#1':>7}")
    print("-" * 76)
    res = {"pose_agreement": evaluate(rows, "pose_agreement", "pose agreement (-RMSD)")}
    for m in ("iface_plddt", "iptm"):
        if all(r.get(m) is not None for r in rows):
            res[m] = evaluate(rows, m, f"{m} (same folds, reference)")

    Path(args.out).write_text(json.dumps(
        {"per_fold": rows, "summary": res, "tags": tags}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
