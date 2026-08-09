"""Two validity checks on the panel itself, from folds already on disk.

**1. Training-set contamination.** Every complex in the panel is a PDB entry, and
Boltz-1 was trained on PDB structures released before 2021-09-30 -- the same
cutoff as AlphaFold3. DeCAF distils Boltz-1 and inherits it. So for most of the
panel the model may be *recalling* a complex it was trained on rather than
predicting it, and "interface pLDDT is high for cognates" would partly mean "the
model has seen this structure".

Entries released after the cutoff form a genuine held-out set. If the
cognate-vs-scramble effect survives there, the signal is prediction; if it
collapses, much of Section 7 is measuring retrieval.

Boltz-2 postdates Boltz-1 and its cutoff is later and not stated in the same
terms, so the split is applied to the arms whose cutoff is known.

**2. Decoy label noise.** A decoy is another receptor's cognate peptide, assumed
to be a non-binder. Peptide binding is promiscuous, and the panel deliberately
keeps SH3 and PDZ folds recurring because discriminating similar domains is the
task. A decoy donated by a *similar* receptor may therefore genuinely bind,
making it a mislabelled negative that caps achievable AUC below 1.

Receptor similarity is computed from sequence directly rather than from an
external family database, so the test needs no annotation and no network.

Usage:
    python src/contamination_and_promiscuity.py
"""

import argparse
import json
import subprocess
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
CUTOFF = "2021-09-30"          # Boltz-1 / AlphaFold3 training cutoff


def release_dates(pdb_ids, cache=ART / "pdb_release_dates.json"):
    known = json.loads(cache.read_text()) if cache.exists() else {}
    for pid in pdb_ids:
        if pid in known:
            continue
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "30",
             f"https://data.rcsb.org/rest/v1/core/entry/{pid}"],
            capture_output=True, text=True)
        try:
            acc = json.loads(proc.stdout).get("rcsb_accession_info", {})
            known[pid] = (acc.get("initial_release_date") or "")[:10]
        except Exception:
            known[pid] = ""
    cache.write_text(json.dumps(known, indent=2))
    return known


def load_arm(arm):
    if arm == "boltz2":
        side = json.loads((ART / "iface_side_split.json").read_text())["per_complex"]
        sc = {p["name"]: p["score"] for p in json.loads(
            (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}
        return [{**r, "iptm": sc.get(r["name"], np.nan)} for r in side]
    return json.loads((ART / f"{arm}_scramble_result.json").read_text())["per_fold"]


def effect_by_group(recs, members, metric):
    """cognate minus its own scrambles, restricted to `members`."""
    by = {}
    for r in recs:
        if r["receptor_id"] in members:
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


def contamination(dates):
    pre = {k for k, v in dates.items() if v and v < CUTOFF}
    post = {k for k, v in dates.items() if v and v >= CUTOFF}
    print(f"\n{'=' * 76}\n1. Training-set contamination "
          f"(Boltz-1 cutoff {CUTOFF})\n{'=' * 76}")
    print(f"   in training  ({len(pre):2d}): {' '.join(sorted(pre))}")
    print(f"   HELD OUT     ({len(post):2d}): {' '.join(sorted(post))}")
    if len(post) < 3:
        print("   too few held-out receptors to test")
        return {"pre": sorted(pre), "post": sorted(post)}

    out = {"pre": sorted(pre), "post": sorted(post), "arms": {}}
    for arm in ("decaf", "boltz1", "boltz2"):
        try:
            recs = load_arm(arm)
        except Exception:
            continue
        note = "" if arm != "boltz2" else "   (cutoff differs; shown for reference)"
        print(f"\n   {arm}{note}")
        print(f"     {'metric':14} {'group':10} {'n_rec':>6} {'effect':>9} "
              f"{'95% CI':>20} {'p':>9}")
        for metric in ("iptm", "iface_plddt"):
            for label, grp in (("in training", pre), ("HELD OUT", post)):
                r = effect_by_group(recs, grp, metric)
                if not r:
                    continue
                out["arms"].setdefault(arm, {}).setdefault(metric, {})[label] = r
                print(f"     {metric:14} {label:10} {r['n_receptors']:6d} "
                      f"{r['effect']:+9.3f} [{r['ci95'][0]:+7.3f},{r['ci95'][1]:+7.3f}]"
                      f" {r['p']:9.5f}")
    return out


def promiscuity():
    """Do decoys donated by similar receptors score higher?"""
    seqs = {}
    for f in (ART / "pdb_binders_b2_n22" / "sequences").glob("*.json"):
        seqs[f.stem] = json.loads(f.read_text())["receptor"]
    # which receptor donated each decoy peptide is recorded only in the panel
    # definition, so it is joined in by fold name
    donor_of = {p["name"]: p.get("peptide_from") for p in json.loads(
        (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}

    print(f"\n{'=' * 76}\n2. Decoy label noise: are similar-receptor decoys "
          f"real binders?\n{'=' * 76}")
    out = {}
    for arm in ("decaf", "boltz2"):
        try:
            recs = load_arm(arm)
        except Exception:
            continue
        rows = []
        for r in recs:
            if r["label"] != "decoy":
                continue
            donor = r.get("peptide_from") or donor_of.get(r.get("name") or r.get("job"))
            tgt = r["receptor_id"]
            if not donor or donor not in seqs or tgt not in seqs:
                continue
            sim = SequenceMatcher(None, seqs[tgt], seqs[donor]).ratio()
            rows.append((sim, r))
        if len(rows) < 10:
            print(f"   {arm}: donor receptor not recorded in this arm; skipped")
            continue
        sims = np.array([s for s, _ in rows])
        print(f"\n   {arm}  ({len(rows)} decoys, receptor similarity "
              f"{sims.min():.2f}-{sims.max():.2f}, median {np.median(sims):.2f})")
        for metric in ("iptm", "iface_plddt"):
            vals = np.array([r[metric] for _, r in rows], float)
            rho, p = stats.spearmanr(sims, vals)
            hi = vals[sims >= np.median(sims)]
            lo = vals[sims < np.median(sims)]
            u, pu = stats.mannwhitneyu(hi, lo, alternative="greater")
            out.setdefault(arm, {})[metric] = {
                "spearman_sim_vs_score": float(rho), "p_spearman": float(p),
                "similar_mean": float(hi.mean()), "dissimilar_mean": float(lo.mean()),
                "p_similar_higher": float(pu)}
            print(f"     {metric:14} rho(similarity, score) {rho:+.3f} (p {p:.4f})"
                  f"   similar {hi.mean():.3f} vs dissimilar {lo.mean():.3f}"
                  f"  p {pu:.4f}")
    print("\n   A positive correlation would mean decoys from similar receptors")
    print("   score higher -- consistent with genuine promiscuity, i.e. some")
    print("   'negatives' bind, which caps the achievable AUC below 1.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "contamination_promiscuity.json"))
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from pdb_binder_benchmark import PDB_IDS

    dates = release_dates(PDB_IDS)
    print("PDB release dates:")
    for pid in PDB_IDS:
        flag = "HELD OUT" if dates.get(pid, "") >= CUTOFF else ""
        print(f"   {pid}  {dates.get(pid, '?')}  {flag}")

    result = {"cutoff": CUTOFF, "dates": dates,
              "contamination": contamination(dates),
              "promiscuity": promiscuity()}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
