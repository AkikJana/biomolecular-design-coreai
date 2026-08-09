"""Score the change in receptor confidence, not its level.

Every readout in this dissertation is receptor-dependent. Section 7.10 measures
the share of variance attributable to receptor identity at 0.33 for ipTM and
0.48 to 0.61 for the pLDDT-derived metrics, which is why every test is a
within-receptor contrast and why no absolute threshold exists. In a real screen
that is a genuine limitation: you have one target and no calibration set, so
"is this score high" has to mean something on its own.

An apo baseline removes the receptor term by construction. Fold the receptor
alone, take its pLDDT, and score each complex by

    delta = receptor_side(holo) - receptor_plddt(apo)

The receptor's intrinsic foldability -- how ordered it is, how deep its
alignment, how well the model knows the fold -- cancels, because it appears in
both terms. What survives is the change induced by the peptide, which is the
quantity Section 7.7 argued is the interpretable part of the signal.

This is a change to the *input*, not another way of reading the same head, so
it is not obviously bounded by the variance decomposition of Section 7.9.2 the
way mPAE, ipSAE and pDockQ2 are.

The test is whether the delta improves the **pooled** regime specifically.
Within-receptor z-scoring already cancels the receptor term empirically, so the
delta should add little there; if it helps anywhere it is where an absolute
threshold would be used.

Usage:
    python src/apo_baseline.py            # fold apo receptors, then analyse
    python src/apo_baseline.py --analyse-only
"""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

from decaf_scramble_test import fold  # noqa: E402

ART = REPO_ROOT / "artifacts"
DECAF_HOME = Path.home() / ".boltz" / "decaf"
WORK = ART / "apo_baseline"


def apo_plddt(results, name):
    """Mean CA pLDDT of a single-chain prediction."""
    d = results / "predictions" / name
    pdb = d / f"{name}_model_0.pdb"
    if not pdb.exists():
        return None
    model = PDBParser(QUIET=True).get_structure("x", str(pdb))[0]
    vals = [r["CA"].bfactor for c in model for r in c if "CA" in r]
    return float(np.mean(vals)) if vals else None


def fold_apo(ids, seqdir, msadir, ckpt, batch_size=8):
    WORK.mkdir(parents=True, exist_ok=True)
    store = WORK / "apo_plddt.json"
    done = json.loads(store.read_text()) if store.exists() else {}
    todo = [p for p in ids if p not in done]
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        bdir = WORK / f"b{start // batch_size:02d}"
        inputs = bdir / "inputs"
        if inputs.exists():
            shutil.rmtree(inputs)
        inputs.mkdir(parents=True)
        for pid in chunk:
            seq = json.loads((seqdir / f"{pid}.json").read_text())["receptor"]
            msa = msadir / f"{pid}.csv"
            # Same alignment as the holo fold. A different MSA would make the
            # difference partly an alignment difference rather than the effect
            # of the peptide.
            rline = f"      msa: {msa}\n" if msa.exists() else "      msa: empty\n"
            (inputs / f"{pid}.yaml").write_text(
                "version: 1\nsequences:\n"
                f"  - protein:\n      id: A\n      sequence: {seq}\n{rline}")
        res, el = fold(inputs, bdir, ckpt, 10, 1, "decaf")
        for pid in chunk:
            v = apo_plddt(res, pid)
            if v is not None:
                done[pid] = v
        store.write_text(json.dumps(done, indent=2))
        print(f"  apo batch {start // batch_size}: {len(chunk)} in {el:.0f}s",
              flush=True)
        shutil.rmtree(bdir, ignore_errors=True)
    return done


def evaluate(recs, apo):
    """Rank and enrichment for raw receptor_side against the apo-referenced delta."""
    rows = [r for r in recs if r["receptor_id"] in apo]
    for r in rows:
        r["delta_receptor"] = r["receptor_side"] - apo[r["receptor_id"]]
        r["delta_iface"] = r["iface_plddt"] - apo[r["receptor_id"]]

    def pooled_auc(metric):
        from sklearn.metrics import roc_auc_score
        y = [1 if r["label"] == "cognate" else 0 for r in rows]
        return float(roc_auc_score(y, [r[metric] for r in rows]))

    def ef5(metric, within=False):
        d = sorted(rows, key=lambda r: -r[metric])
        if within:
            by = {}
            for r in rows:
                by.setdefault(r["receptor_id"], []).append(r[metric])
            mu = {k: np.mean(v) for k, v in by.items()}
            sd = {k: (np.std(v) or 1) for k, v in by.items()}
            d = sorted(rows, key=lambda r: -((r[metric] - mu[r["receptor_id"]])
                                             / sd[r["receptor_id"]]))
        k = max(1, int(round(0.05 * len(d))))
        base = np.mean([r["label"] == "cognate" for r in rows])
        return float(np.mean([r["label"] == "cognate" for r in d[:k]]) / base)

    def effect(metric):
        by = {}
        for r in rows:
            by.setdefault(r["receptor_id"], []).append(r)
        diffs = []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"]
            s = [x for x in g if x["label"] == "scrambled"]
            if c and s:
                diffs += [c[0][metric] - x[metric] for x in s]
        d = np.array(diffs, float)
        return float(d.mean()), float(stats.ttest_1samp(d, 0).pvalue)

    print(f"\n{'metric':18} {'cog-scr':>9} {'p':>9} {'AUCpool':>8} "
          f"{'EFpool':>7} {'EFwithin':>9}")
    print("-" * 66)
    out = {}
    for m in ("receptor_side", "delta_receptor", "iface_plddt", "delta_iface"):
        e, p = effect(m)
        out[m] = {"effect": e, "p": p, "auc_pooled": pooled_auc(m),
                  "ef5_pooled": ef5(m), "ef5_within": ef5(m, True)}
        print(f"{m:18} {e:+9.3f} {p:9.5f} {out[m]['auc_pooled']:8.3f} "
              f"{out[m]['ef5_pooled']:7.2f} {out[m]['ef5_within']:9.2f}")
    print("\n  delta_* subtract the receptor's own apo pLDDT. The comparison that")
    print("  matters is the pooled column: within-receptor scoring already")
    print("  cancels the receptor term, so the delta should only help pooled.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DECAF_HOME / "decaf_conf_ckpt.ckpt"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    from heldout_panel import ALREADY, NEW
    ids = ALREADY + NEW
    seqdir = ART / "heldout_panel" / "sequences"
    msadir = ART / "heldout_panel" / "msa_cache"
    store = WORK / "apo_plddt.json"
    if args.analyse_only:
        apo = json.loads(store.read_text()) if store.exists() else {}
    else:
        print(f"folding {len(ids)} apo receptors")
        apo = fold_apo(ids, seqdir, msadir, args.ckpt, args.batch_size)
    print(f"apo baselines available for {len(apo)}/{len(ids)} receptors")

    recs = json.loads((ART / "heldout_panel" / "heldout_scores.json").read_text())
    res = evaluate(recs, apo)
    (ART / "apo_baseline_result.json").write_text(
        json.dumps({"apo_plddt": apo, "evaluation": res}, indent=2))
    print(f"\nwrote {ART}/apo_baseline_result.json")


if __name__ == "__main__":
    main()
