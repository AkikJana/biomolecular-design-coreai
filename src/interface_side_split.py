"""Which side of the interface carries the interface-pLDDT signal?

Section 7.6 found that interface pLDDT distinguishes a cognate peptide from its
own scramble where ipTM cannot. The obvious objection is that it may be reading
peptide *foldability* rather than binding: a scrambled peptide is often more
disordered, which would produce the result with no binding information in it.

Interface pLDDT averages over residues on both chains, so the two readings are
separable:

  peptide_side   mean pLDDT over the peptide residues at the interface
  receptor_side  mean pLDDT over the receptor residues at the interface

  peptide side only    the metric tracks how well the peptide itself is placed.
                       Consistent with the foldability objection; the binding
                       claim weakens.
  receptor side too    the receptor's own residues are placed more confidently
                       when given the right peptide. The receptor is responding
                       to the partner, which foldability cannot explain.

Costs nothing: it re-reads structures already on disk.

Usage:
    python src/interface_side_split.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy import stats

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_CUTOFF = 8.0


def rep_atom(res):
    if "CB" in res:
        return res["CB"]
    return res["CA"] if "CA" in res else None


def sides(model):
    """Interface pLDDT split by chain, plus the pooled value for reference."""
    chains = [c for c in model]
    if len(chains) < 2:
        return None
    ra = [(r, rep_atom(r)) for r in chains[0] if rep_atom(r) is not None]
    rb = [(r, rep_atom(r)) for r in chains[1] if rep_atom(r) is not None]
    if not ra or not rb:
        return None
    ca = np.array([a.coord for _, a in ra])
    cb = np.array([a.coord for _, a in rb])
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    mask = d < CONTACT_CUTOFF
    if not mask.any():
        return None

    rec_b = [ra[i][0]["CA"].get_bfactor() for i in range(len(ra))
             if mask[i].any() and "CA" in ra[i][0]]
    pep_b = [rb[j][0]["CA"].get_bfactor() for j in range(len(rb))
             if mask[:, j].any() and "CA" in rb[j][0]]
    if not rec_b or not pep_b:
        return None
    return {"receptor_side": float(np.mean(rec_b)),
            "peptide_side": float(np.mean(pep_b)),
            "iface_plddt": float(np.mean(rec_b + pep_b)),
            "n_rec_iface": len(rec_b), "n_pep_iface": len(pep_b),
            # whole-chain peptide pLDDT: the foldability quantity itself,
            # independent of whether those residues touch the receptor
            "peptide_whole": float(np.mean(
                [r["CA"].get_bfactor() for r, _ in rb if "CA" in r]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "pdb_binders_b2_n22"))
    ap.add_argument("--scores", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "iface_side_split.json"))
    args = ap.parse_args()

    pairs = [p for p in json.loads(Path(args.scores).read_text())
             if not np.isnan(p.get("score", float("nan")))]
    parser = PDBParser(QUIET=True)
    index = {}
    for pdb in sorted(Path(args.work).rglob("*_model_0.pdb")):
        if "batch_" in str(pdb):
            index.setdefault(pdb.parent.name, pdb)

    recs = []
    for p in pairs:
        pdb = index.get(p["name"])
        if pdb is None:
            continue
        try:
            s = sides(parser.get_structure("x", str(pdb))[0])
        except Exception:
            continue
        if s:
            recs.append({**{k: p[k] for k in ("receptor_id", "label", "name")}, **s})

    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)

    print(f"\n{'=' * 76}")
    print(f"Which side of the interface carries the signal?  ({len(recs)} complexes)")
    print("=" * 76)
    print(f"\n{'quantity':16} {'cognate':>9} {'scram':>9} {'decoy':>9}"
          f" {'cog-scr':>9} {'p':>9} {'rank':>6} {'rank p':>8}")
    print("-" * 76)

    out = {}
    for met in ("iface_plddt", "receptor_side", "peptide_side", "peptide_whole"):
        vals = {lab: np.array([r[met] for r in recs if r["label"] == lab])
                for lab in ("cognate", "scrambled", "decoy")}
        diffs = []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"]
            s = [x for x in g if x["label"] == "scrambled"]
            if c and s:
                diffs += [c[0][met] - x[met] for x in s]
        diffs = np.array(diffs)
        p_cs = stats.ttest_1samp(diffs, 0).pvalue

        ranks, sizes = [], []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"]
            d = [x for x in g if x["label"] == "decoy"]
            if not c or not d:
                continue
            sc = [c[0][met]] + [x[met] for x in d]
            ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
            sizes.append(len(sc))
        ranks = np.array(ranks, float); sizes = np.array(sizes, float)
        exp = (sizes + 1) / 2
        p_r = stats.wilcoxon(ranks - exp, alternative="two-sided")[1]

        ci = stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(),
                              scale=diffs.std(ddof=1) / np.sqrt(len(diffs)))
        out[met] = {"cognate": float(vals["cognate"].mean()),
                    "scrambled": float(vals["scrambled"].mean()),
                    "decoy": float(vals["decoy"].mean()),
                    "cognate_minus_own_scramble": float(diffs.mean()),
                    "ci95": [float(ci[0]), float(ci[1])],
                    "p_cognate_vs_own_scramble": float(p_cs),
                    "mean_rank": float(ranks.mean()), "chance": float(exp.mean()),
                    "p_rank": float(p_r)}
        print(f"{met:16} {vals['cognate'].mean():9.2f} {vals['scrambled'].mean():9.2f}"
              f" {vals['decoy'].mean():9.2f} {diffs.mean():+9.2f} {p_cs:9.5f}"
              f" {ranks.mean():6.2f} {p_r:8.4f}")

    rec_p = out["receptor_side"]["p_cognate_vs_own_scramble"]
    pep_p = out["peptide_side"]["p_cognate_vs_own_scramble"]
    print("\nVerdict")
    if rec_p < 0.05 and pep_p < 0.05:
        print("  BOTH sides shift. The receptor's own residues are placed more")
        print("  confidently when given the right peptide -- peptide foldability")
        print("  alone cannot account for that.")
    elif pep_p < 0.05 <= rec_p:
        print("  PEPTIDE SIDE ONLY. Consistent with the metric reading how well the")
        print("  peptide is placed rather than whether the interface is correct.")
        print("  The binding interpretation in Section 7.6 weakens accordingly.")
    elif rec_p < 0.05 <= pep_p:
        print("  RECEPTOR SIDE ONLY -- unexpected; worth inspecting directly.")
    else:
        print("  NEITHER side separates on its own; the pooled result rests on")
        print("  their combination and should be treated cautiously.")

    Path(args.out).write_text(json.dumps({"per_complex": recs, "summary": out}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
