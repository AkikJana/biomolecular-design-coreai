"""This work's readouts against binding measured in a wet lab.

Every number in Section 7 is scored against a proxy for binding: a cognate pair
taken from the PDB counts as a positive and a peptide borrowed from another
receptor counts as a negative. Section 7.10 spends its length on the resulting
worry -- that a model trained on those very complexes may be retrieving rather
than predicting -- and Section 7.14.6 concedes the tool's hit rate is unmeasured
because nothing here was ever synthesised.

Anthropic's autonomous binder-design study [25] removes both problems at once. It
released 1,320 de novo designs with binding measured by two independent contract
research organisations, together with co-folding scores from ten predictors. The
designs did not exist when any of those models were trained, so contamination is
not available as an explanation, and the labels are binding rather than a proxy
for it.

What this can and cannot test is worth stating plainly. The released confidence
score is ipSAE, which Section 7.11 examined, and not interface pLDDT, which is
this work's headline readout -- so the comparison below is of the metric they
released, on their designs. Boltz-2 is among their ten predictors, which makes
one row directly comparable to this work's own arm.

Two scoring conventions are reported because the two studies use different ones:

  within-target AUC   scores z-scored within each target, then one global ROC.
                      Sections 7.10 and 7.13 use this.
  macro-AP            average precision within each target, averaged over
                      targets. The comparison figures in [25] use this.

Usage:
    python src/anthropic_validation.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
DATA = ART / "anthropic_binder" / "design_summary.csv"
warnings.filterwarnings("ignore")

PREDICTORS = {
    "boltz2": "Boltz-2  (this work's model)",
    "ef2fast": "ESMFold2-Fast",
    "ef2full": "ESMFold2",
    "ptxv2": "Protenix v2",
    "afm3": "AlphaFold-Multimer v3",
    "af3of3": "AlphaFold3 / OpenFold3 weights",
    "chai1": "Chai-1",
    "of3": "OpenFold3",
    "rf3": "RoseTTAFold3",
    "odde": "OpenDDE",
}


def within_target_auc(df, col):
    """Section 7.13's convention: z within target, then one global ROC."""
    d = df.dropna(subset=[col])
    if d["y"].nunique() < 2:
        return None
    z = d.groupby("target")[col].transform(lambda x: (x - x.mean()) / (x.std() or 1))
    return float(roc_auc_score(d["y"], z))


def macro_ap(df, col):
    """Their convention: average precision within a target, averaged over targets.

    Targets where every design bound, or none did, carry no ranking information
    and are skipped rather than scored as 0 or 1.
    """
    aps = []
    for _, g in df.dropna(subset=[col]).groupby("target"):
        if g["y"].nunique() < 2:
            continue
        aps.append(average_precision_score(g["y"], g[col]))
    return (float(np.mean(aps)), len(aps)) if aps else (None, 0)


def boot_ci(df, col, fn, n=2000, seed=0):
    """Bootstrap over targets, which is the unit of independence here."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("target")]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in pick])
        v = fn(s, col)
        v = v[0] if isinstance(v, tuple) else v
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))) if vals else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "anthropic_validation.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} not found; download design_summary.csv from "
                         f"huggingface.co/datasets/Anthropic/claude-protein-binder-design")
    raw = pd.read_csv(DATA, low_memory=False)
    df = raw[raw["binder_final"].isin([True, False])].copy()
    df["y"] = df["binder_final"].astype(int)

    print(f"{len(df)} designs with a binding call, {df['y'].sum()} binders "
          f"({df['y'].mean():.1%}), {df['target'].nunique()} targets")
    print("labels are measured binding from two independent CROs; the designs "
          "postdate every model's training\n")

    rows = []
    # This work's own readouts, recomputed from the released Boltz-2 structures
    # by src/anthropic_iface_plddt.py. Section 7.18 could only compare ipSAE,
    # which is not what Section 8.2 recommends; these rows are the recommendation
    # itself, scored against the same measured binding.
    own = ART / "anthropic_binder" / "iface_plddt_boltz2.json"
    if own.exists():
        o = pd.DataFrame(json.loads(own.read_text()))
        df = df.merge(o[["uuid", "iface_plddt", "receptor_side", "peptide_side",
                         "peptide_whole"]], on="uuid", how="left")
        for col, lab in (("iface_plddt", "Interface pLDDT  (this work's readout)"),
                         ("receptor_side", "  receptor side"),
                         ("peptide_side", "  binder side"),
                         ("peptide_whole", "  binder whole-chain pLDDT")):
            if df[col].notna().sum() < 100:
                continue
            auc = within_target_auc(df, col)
            ap_val, n_t = macro_ap(df, col)
            lo, hi = boot_ci(df, col, macro_ap, args.n_boot)
            rows.append({"predictor": col, "label": lab, "within_auc": auc,
                         "macro_ap": ap_val, "ap_ci": [lo, hi], "n_targets": n_t,
                         "n_scored": int(df[col].notna().sum())})
        print(f"  interface pLDDT recomputed for {df['iface_plddt'].notna().sum()} "
              f"of {len(df)} designs\n")

    for key, label in PREDICTORS.items():
        col = f"ipsae_min_{key}"
        if col not in df.columns:
            continue
        auc = within_target_auc(df, col)
        ap_val, n_t = macro_ap(df, col)
        lo, hi = boot_ci(df, col, macro_ap, args.n_boot)
        rows.append({"predictor": key, "label": label, "within_auc": auc,
                     "macro_ap": ap_val, "ap_ci": [lo, hi], "n_targets": n_t})

    rows.sort(key=lambda r: -(r["macro_ap"] or 0))
    print(f"  {'predictor':34}{'within-target AUC':>18}{'macro-AP':>10}"
          f"{'95% CI':>16}")
    for r in rows:
        ci = (f"[{r['ap_ci'][0]:.2f},{r['ap_ci'][1]:.2f}]"
              if r["ap_ci"][0] is not None else "—")
        star = " *" if r["predictor"] == "boltz2" else ""
        print(f"  {r['label']:34}{r['within_auc']:>18.3f}{r['macro_ap']:>10.3f}"
              f"{ci:>16}{star}")

    base = df["y"].mean()
    print(f"\n  chance: AUC 0.500, macro-AP ~{base:.3f} (the base rate)")

    ip = next((r for r in rows if r["predictor"] == "iface_plddt"), None)
    if ip:
        bz = next(r for r in rows if r["predictor"] == "boltz2")
        print("\n  this work's readout vs the released score, same structures:")
        print(f"    interface pLDDT  AUC {ip['within_auc']:.3f}  AP {ip['macro_ap']:.3f}")
        print(f"    ipSAE (Boltz-2)  AUC {bz['within_auc']:.3f}  AP {bz['macro_ap']:.3f}")

    b = next(r for r in rows if r["predictor"] == "boltz2")
    print(f"\n{'=' * 78}")
    print(f"Boltz-2 ipSAE on measured binding: within-target AUC "
          f"{b['within_auc']:.3f}, macro-AP {b['macro_ap']:.3f}")
    print("This work's within-receptor AUC on its own PDB panel, for scale:")
    print("  in-training, full settings   0.909 - 0.943   (Section 7.13)")
    print("  held out,    full settings   0.682 - 0.803   (Section 7.10.6)")
    print()
    print("These are NOT the same measurement. The panels differ, the positives")
    print("differ -- a cognate crystal pair against a synthesised binder -- and the")
    print("readouts differ, ipSAE here against interface pLDDT there. What they")
    print("share is the question, and the ordering answers it: a readout scored")
    print("on complexes the model was trained on reads far higher than the same")
    print("family of readout scored against binding that was actually measured.")
    print()
    print("Note also that [25]'s own headline filtering figures (macro-AP 0.62,")
    print("0.61, 0.66 against AlphaFold3's 0.55) are computed on the Overath et")
    print("al. dataset -- 3,532 designs on 13 targets -- and NOT on the 1,320")
    print("designs used here, so they are not comparable to the table above.")

    Path(args.out).write_text(json.dumps(
        {"n_designs": int(len(df)), "n_binders": int(df["y"].sum()),
         "base_rate": float(base), "n_targets": int(df["target"].nunique()),
         "predictors": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
