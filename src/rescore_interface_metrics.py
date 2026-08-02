"""Re-score the already-folded complexes with interface metrics other than ipTM.

Section 7 established that ipTM tracks peptide composition rather than binding.
That is a statement about ipTM, not about the predicted structures -- which are
still on disk. This recomputes a set of interface measures from those same
structures and re-runs the identical tests, at no folding cost.

The decisive test is unchanged: a cognate must beat its **own scramble**, which
holds composition and length fixed and destroys only order. Any metric that
passes it is measuring something ipTM is not.

Metrics
  n_contacts      inter-chain residue pairs within 8 A (CB, or CA for glycine)
  iface_plddt     mean pLDDT over residues making those contacts
  pdockq          Bryant et al. 2022, on iface_plddt * log(n_contacts)
  delta_sasa      surface area buried on complex formation (Shrake-Rupley)
  contact_density n_contacts normalised by peptide length, since ipTM was
                  already shown to fall with peptide length
  iptm            the published value, carried through as the baseline

Usage:
    python src/rescore_interface_metrics.py
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy import stats

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_CUTOFF = 8.0


def rep_atom(res):
    """CB, or CA for glycine and anything lacking CB."""
    if "CB" in res:
        return res["CB"]
    return res["CA"] if "CA" in res else None


def interface(model):
    chains = [c for c in model]
    if len(chains) < 2:
        return None
    a, b = chains[0], chains[1]
    ra = [(r, rep_atom(r)) for r in a if rep_atom(r) is not None]
    rb = [(r, rep_atom(r)) for r in b if rep_atom(r) is not None]
    if not ra or not rb:
        return None
    ca = np.array([at.coord for _, at in ra])
    cb = np.array([at.coord for _, at in rb])
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    mask = d < CONTACT_CUTOFF
    n_contacts = int(mask.sum())
    ia = {i for i, _ in zip(range(len(ra)), range(len(ra))) if mask[i].any()}
    ib = {j for j in range(len(rb)) if mask[:, j].any()}
    plddts = ([ra[i][0]["CA"].get_bfactor() for i in ia if "CA" in ra[i][0]]
              + [rb[j][0]["CA"].get_bfactor() for j in ib if "CA" in rb[j][0]])
    iface_plddt = float(np.mean(plddts)) if plddts else 0.0
    return {"n_contacts": n_contacts, "iface_plddt": iface_plddt,
            "n_iface_res": len(ia) + len(ib), "len_b": len(rb)}


def pdockq(iface_plddt, n_contacts):
    if n_contacts <= 0:
        return 0.0
    x = iface_plddt * math.log(n_contacts)
    return 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018


def delta_sasa(structure_path, parser, sr):
    """Buried area = sum of isolated chain areas - complex area."""
    whole = parser.get_structure("c", str(structure_path))[0]
    sr.compute(whole, level="C")
    complex_area = sum(c.sasa for c in whole)
    total_alone = 0.0
    for cid in [c.id for c in whole]:
        sep = parser.get_structure("s", str(structure_path))[0]
        for other in [c.id for c in sep]:
            if other != cid:
                sep.detach_child(other)
        sr.compute(sep, level="C")
        total_alone += sum(c.sasa for c in sep)
    return float(total_alone - complex_area)


def collect(work, scores_path, with_sasa=True):
    pairs = [p for p in json.loads(Path(scores_path).read_text())
             if not np.isnan(p.get("score", float("nan")))]
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    # Only the benchmark folds under batch_*/. The msa_fetch probes are folded
    # at 0 recycling / 5 sampling steps purely to pull an alignment, and they
    # are *also* named pair_000 -- indexing them would silently score a
    # throwaway structure in place of a real pair. They are additionally
    # non-physical: 11 of them have coordinates past +/-1000 A, which run the
    # PDB columns together and do not parse.
    index = {}
    for pdb in sorted(Path(work).rglob("*_model_0.pdb")):
        if "batch_" not in str(pdb):
            continue
        index.setdefault(pdb.parent.name, pdb)

    out = []
    for p in pairs:
        pdb = index.get(p["name"])
        if pdb is None:
            continue
        try:
            model = parser.get_structure(p["name"], str(pdb))[0]
        except Exception as exc:
            print(f"  skipped {p['name']}: {str(exc)[:60]}")
            continue
        m = interface(model)
        if m is None:
            continue
        rec = {k: p[k] for k in ("receptor_id", "label", "name", "peptide_from")}
        rec["iptm"] = p["score"]
        rec.update(m)
        rec["pdockq"] = pdockq(m["iface_plddt"], m["n_contacts"])
        rec["contact_density"] = m["n_contacts"] / max(1, m["len_b"])
        rec["delta_sasa"] = delta_sasa(pdb, parser, sr) if with_sasa else float("nan")
        out.append(rec)
    return out


METRICS = ["iptm", "pdockq", "iface_plddt", "n_contacts", "contact_density",
           "delta_sasa"]


def report(recs):
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)

    print(f"\n{'=' * 78}")
    print(f"Re-scoring {len(recs)} folded complexes across {len(by)} receptors")
    print("=" * 78)

    print(f"\n{'metric':16} {'cognate':>9} {'scram':>9} {'decoy':>9}"
          f" {'cog-scr p':>10} {'rank':>6} {'chance':>7} {'rank p':>8}")
    print("-" * 78)
    results = {}
    for met in METRICS:
        vals = {lab: np.array([r[met] for r in recs if r["label"] == lab], float)
                for lab in ("cognate", "scrambled", "decoy")}
        if any(np.isnan(v).all() for v in vals.values()):
            continue
        # decisive control: cognate vs its OWN scrambles
        diffs = []
        for rid, g in by.items():
            c = [x for x in g if x["label"] == "cognate"]
            s = [x for x in g if x["label"] == "scrambled"]
            if c and s:
                diffs += [c[0][met] - x[met] for x in s]
        diffs = np.array(diffs, float)
        p_cs = stats.ttest_1samp(diffs, 0).pvalue if len(diffs) > 2 else float("nan")

        # within-receptor rank of the cognate among its decoys
        ranks, sizes = [], []
        for rid, g in by.items():
            c = [x for x in g if x["label"] == "cognate"]
            d = [x for x in g if x["label"] == "decoy"]
            if not c or not d:
                continue
            sc = [c[0][met]] + [x[met] for x in d]
            ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
            sizes.append(len(sc))
        ranks = np.array(ranks, float); sizes = np.array(sizes, float)
        exp = (sizes + 1) / 2
        p_rank = (stats.wilcoxon(ranks - exp, alternative="two-sided")[1]
                  if len(ranks) > 4 and not np.allclose(ranks - exp, 0) else float("nan"))
        results[met] = {"cognate_mean": float(vals["cognate"].mean()),
                        "scrambled_mean": float(vals["scrambled"].mean()),
                        "decoy_mean": float(vals["decoy"].mean()),
                        "cognate_minus_own_scramble": float(diffs.mean()),
                        "p_cognate_vs_own_scramble": float(p_cs),
                        "mean_rank": float(ranks.mean()),
                        "chance": float(exp.mean()),
                        "p_rank_vs_chance": float(p_rank),
                        "cognate_first": int((ranks == 1).sum()),
                        "n_receptors": int(len(ranks))}
        print(f"{met:16} {vals['cognate'].mean():9.3f} {vals['scrambled'].mean():9.3f}"
              f" {vals['decoy'].mean():9.3f} {p_cs:10.4f}"
              f" {ranks.mean():6.2f} {exp.mean():7.2f} {p_rank:8.4f}")

    print("\nThe column that decides it is 'cog-scr p': a metric that cannot beat a")
    print("cognate's own scramble is not responding to sequence order, and order is")
    print("what makes a binder a binder.")
    winners = [m for m, r in results.items()
               if r["p_cognate_vs_own_scramble"] < 0.05
               and r["cognate_minus_own_scramble"] > 0]
    print(f"\nmetrics beating their own scramble (p<0.05): {winners or 'NONE'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "pdb_binders_b2_n22"))
    ap.add_argument("--scores", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--no-sasa", action="store_true", help="skip the slow SASA pass")
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "rescore_metrics.json"))
    args = ap.parse_args()

    recs = collect(args.work, args.scores, with_sasa=not args.no_sasa)
    print(f"scored {len(recs)} complexes")
    results = report(recs)
    Path(args.out).write_text(json.dumps(
        {"per_complex": recs, "summary": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
