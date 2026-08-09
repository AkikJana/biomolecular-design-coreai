"""Is every "cognate" pair in the panel actually a binder?

The benchmark folds a receptor sequence and a peptide sequence. That is only a
faithful representation of the crystal if the crystallised peptide *is* its
canonical sequence. Two ways it is not, both silent:

**Non-standard residues read as X.** RCSB FASTA emits X for any component that
is not one of the twenty. 1NLO's peptide is XXXXPLPPLPX -- five of eleven
positions are ACE, MN1 (4-carboxypiperidine), MN2 and MN7 (benzene derivatives)
and NH2. It is a designed synthetic ligand, not a peptide, and what the
benchmark folds is not what SH3 binds. 1SEM and 2GBQ carry terminal ACE/NH2
caps only, which is a milder version of the same thing.

**Components bonded to the peptide chain.** 9GRF's peptide is O-glycosylated at
Ser3 and Thr4; StcE is a mucin-selective protease that reads the glycan, so
AASTTTPAPA on its own is not a substrate. 7F3S carries LBZ, a benzoyl-lysine, in
a histone H3 tail. audit_panel_ptms.py missed that one because it tests a fixed
allowlist of PTM codes and LBZ is not in it -- an allowlist cannot catch a
modification nobody thought of, whereas "is anything bonded to this chain"
can.

This is the same defect that excluded 1I8H, whose peptide needs
phosphothreonine, and whose "true binder ranked last of six" was correct
behaviour on a non-binder.

A cognate that does not bind should score like a scramble, so it *shrinks* the
cognate-versus-scramble contrast. Removing these members should therefore raise
every effect if the flags are real -- which is a prediction the script tests,
not an assumption. The exclusions are chosen by whether the sequence can be
folded faithfully, never by looking at the resulting p-value.

Usage:
    python src/panel_integrity.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

ART = REPO_ROOT / "artifacts"
CACHE = ART / "pdb_candidates"

# Peptides whose canonical sequence is not what was crystallised. Split by
# severity, because the two groups deserve different treatment: a peptide that
# is mostly synthetic is unusable, whereas a capped one is a real peptide with
# altered termini and only worth a sensitivity check.
UNFOLDABLE = {
    "1NLO": "XXXXPLPPLPX -- ACE/MN1/MN2/MN7/NH2, a synthetic ligand not a peptide",
    "9GRF": "O-glycosylated at Ser3/Thr4; StcE reads the glycan",
}
CAPPED = {
    "1SEM": "ACE cap (X at position 1)",
    "2GBQ": "ACE and NH2 caps (X at both termini)",
}


def load_arm(arm):
    if arm == "boltz2":
        side = json.loads((ART / "iface_side_split.json").read_text())["per_complex"]
        sc = {p["name"]: p["score"] for p in json.loads(
            (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}
        return [{**r, "iptm": sc.get(r["name"], np.nan)} for r in side]
    return json.loads((ART / f"{arm}_scramble_result.json").read_text())["per_fold"]


def effect(recs, drop, metric):
    """Cognate minus its own scrambles, with `drop` removed on both sides.

    A dropped receptor must also stop donating decoy peptides, or a peptide that
    is not a binder anywhere goes on acting as a negative for everyone else.
    """
    recs = [r for r in recs if r["receptor_id"] not in drop
            and r.get("peptide_from") not in drop]
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s:
            diffs += [c[0][metric] - x[metric] for x in s]
    if len(diffs) < 3:
        return None
    d = np.array(diffs, float)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                          scale=d.std(ddof=1) / np.sqrt(len(d)))
    return {"n_receptors": len(by), "n_pairs": len(d), "effect": float(d.mean()),
            "ci95": [float(ci[0]), float(ci[1])],
            "p": float(stats.ttest_1samp(d, 0).pvalue)}


def rank_test(recs, drop, metric):
    """Cognate's rank among its own decoys; chance is (n+1)/2.

    Reported alongside the effect because the two respond to exclusion in
    opposite ways, and quoting only the effect would misrepresent the result.
    Dropping receptors removes diluting non-binders, which helps the paired
    effect, but it also costs the Wilcoxon four of twenty-two samples, which
    hurts a test that already discards magnitude (Section 7.9).
    """
    recs = [r for r in recs if r["receptor_id"] not in drop
            and r.get("peptide_from") not in drop]
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    ranks, sizes = [], []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        d = [x for x in g if x["label"] == "decoy"]
        if not c or not d:
            continue
        sc = [c[0][metric]] + [x[metric] for x in d]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    if len(ranks) < 5:
        return None
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    p = stats.wilcoxon(r - exp)[1] if not np.allclose(r - exp, 0) else 1.0
    return {"mean_rank": float(r.mean()), "chance": float(exp.mean()),
            "p": float(p), "n_receptors": len(ranks)}


def audit(ids, label):
    """Peptide-attached components and X placeholders, per panel member."""
    from discover_pdb_binders import cofactors
    CACHE.mkdir(parents=True, exist_ok=True)
    seqdirs = [ART / d / "sequences" for d in
               ("pdb_binders_b2_n22", "heldout_panel", "pdb_binders_b2", "pdb_binders")]
    out = {}
    for pid in ids:
        pep = ""
        for sd in seqdirs:
            f = sd / f"{pid}.json"
            if f.exists():
                pep = json.loads(f.read_text())["peptide"]
                break
        cof = cofactors(pid, CACHE)
        nx = pep.count("X")
        if cof or nx:
            out[pid] = {"peptide": pep, "attached": cof, "n_placeholder_X": nx}
    print(f"\n{label}: {len(out)}/{len(ids)} members are not folded as crystallised")
    for pid, v in sorted(out.items()):
        bits = []
        if v["n_placeholder_X"]:
            bits.append(f"{v['n_placeholder_X']} x X")
        if v["attached"]:
            bits.append("bonded: " + ",".join(v["attached"]))
        print(f"   {pid}  {v['peptide']:26} {'; '.join(bits)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "panel_integrity.json"))
    args = ap.parse_args()

    from pdb_binder_benchmark import PDB_IDS
    result = {"audit": {"main": audit(PDB_IDS, "Main panel")}}
    try:
        from heldout_panel import ALREADY, NEW
        result["audit"]["heldout"] = audit(ALREADY + NEW, "Held-out panel")
    except Exception:
        pass

    variants = [("all 22", set()),
                ("minus unfoldable", set(UNFOLDABLE)),
                ("minus unfoldable + capped", set(UNFOLDABLE) | set(CAPPED))]
    print(f"\n{'=' * 78}\nCognate vs scramble, by what is excluded\n{'=' * 78}")
    print(f"{'arm':8} {'metric':14} {'variant':27} {'n':>3} {'effect':>9} {'p':>10}")
    print("-" * 78)
    eff = {}
    for arm in ("decaf", "boltz2", "boltz1"):
        try:
            recs = load_arm(arm)
        except Exception:
            continue
        for metric in ("iptm", "iface_plddt", "receptor_side"):
            for name, drop in variants:
                r = effect(recs, drop, metric)
                if not r:
                    continue
                eff.setdefault(arm, {}).setdefault(metric, {})[name] = r
                print(f"{arm:8} {metric:14} {name:27} {r['n_receptors']:3d} "
                      f"{r['effect']:+9.3f} {r['p']:10.6f}")
        print()

    print("Every effect should grow if the flags are real: a cognate that cannot")
    print("bind scores like a scramble and dilutes the contrast it belongs to.")
    cells = [(a, m) for a, ms in eff.items() for m in ms]
    grew = [(a, m) for a, m in cells
            if eff[a][m]["minus unfoldable"]["effect"] > eff[a][m]["all 22"]["effect"]]
    print(f"  grew in {len(grew)} of {len(cells)} arm-metric cells")

    # The rank tests are reported because they move the other way, and a result
    # that quoted only the effects would be selective.
    print(f"\n{'=' * 78}\nCognate ranked against its own decoys (chance 2.50)\n{'=' * 78}")
    print(f"{'arm':8} {'metric':14} {'variant':27} {'n':>3} {'rank':>7} {'p':>10}")
    print("-" * 78)
    rk = {}
    for arm in ("decaf", "boltz2", "boltz1"):
        try:
            recs = load_arm(arm)
        except Exception:
            continue
        for metric in ("iptm", "iface_plddt"):
            for name, drop in variants:
                r = rank_test(recs, drop, metric)
                if not r:
                    continue
                rk.setdefault(arm, {}).setdefault(metric, {})[name] = r
                print(f"{arm:8} {metric:14} {name:27} {r['n_receptors']:3d} "
                      f"{r['mean_rank']:7.2f} {r['p']:10.4f}")
        print()
    print("Mean ranks barely move; their p-values worsen because the test loses")
    print("four of twenty-two receptors. That is a power loss, not a weaker")
    print("effect, and it is why Section 7.9 prefers the mixed model.")

    result["effects"] = eff
    result["rank_tests"] = rk
    result["excluded"] = {"unfoldable": UNFOLDABLE, "capped": CAPPED}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
