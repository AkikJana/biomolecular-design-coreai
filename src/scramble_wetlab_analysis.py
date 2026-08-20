"""Does subtracting a candidate's own scrambles predict measured binding?

Written and committed before the folds finished, so the test is fixed in advance
rather than chosen once the numbers were visible. Section 7.15 measured what
happens otherwise: a search over 63 readout combinations found one beating the
headline, and both of its controls refused it.

The primary test is pre-specified and is a single comparison:

    raw     within-target AUC of interface pLDDT of the design as delivered
    margin  within-target AUC of (design - mean of its own scrambles)

If the scramble control adds information about binding, the margin separates
measured binders from measured non-binders better than the raw score does. The
difference is bootstrapped paired over targets. Everything else reported here is
secondary and is labelled as such.

The prediction registered in scramble_wetlab.py stands: these are 60 to 120
residue designed proteins, a permutation of which does not fold, so binders and
non-binders should both receive a ruined scramble and the subtraction may carry
nothing. A null result would bound the control's applicability -- it is
calibrated for short peptides, where a permutation is still a plausible ligand --
rather than refute it.

Usage:
    python src/scramble_wetlab_analysis.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
warnings.filterwarnings("ignore")


def within_auc(df, col):
    d = df.dropna(subset=[col])
    if d["y"].nunique() < 2:
        return None
    z = d.groupby("target")[col].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    return float(roc_auc_score(d["y"], z))


def paired_ci(df, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("target")]
    diffs = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            va, vb = within_auc(s, a), within_auc(s, b)
            if va is not None and vb is not None:
                diffs.append(va - vb)
        except Exception:                                          # noqa: BLE001
            pass
    return (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5))) if diffs else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(ART / "scramble_wetlab.json"))
    ap.add_argument("--out", default=str(ART / "scramble_wetlab_result.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    picks = json.loads(Path(args.store).read_text())["picks"]
    rows = []
    for p in picks:
        if not p["design"] or not p["scrambles"]:
            continue
        raw = float(np.mean(p["design"]))
        scr = float(np.mean(p["scrambles"]))
        rows.append({"target": p["target"], "y": p["y"], "raw": raw,
                     "scramble": scr, "margin": raw - scr,
                     "ipsae": p.get("ipsae"), "n_scr": len(p["scrambles"])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no design has both a design fold and a scramble fold")

    print(f"{len(df)} designs with both folds, {int(df['y'].sum())} measured binders, "
          f"{df['target'].nunique()} targets")
    inc = len(picks) - len(df)
    if inc:
        print(f"  ({inc} excluded for a missing fold)")

    print(f"\n  {'group':14}{'n':>4}{'raw':>9}{'scramble':>11}{'margin':>9}")
    for lab, name in ((1, "binders"), (0, "non-binders")):
        g = df[df["y"] == lab]
        print(f"  {name:14}{len(g):>4}{g['raw'].mean():>9.2f}"
              f"{g['scramble'].mean():>11.2f}{g['margin'].mean():>9.2f}")

    # Is the scramble ruined for both classes alike? That is the registered
    # prediction, and it is what would make the subtraction uninformative.
    b, nb = df[df["y"] == 1], df[df["y"] == 0]
    t, pv = stats.ttest_ind(b["margin"], nb["margin"], equal_var=False)
    print(f"\n  margin, binders vs non-binders: "
          f"{b['margin'].mean():+.2f} vs {nb['margin'].mean():+.2f}, "
          f"Welch p = {pv:.3f}")

    print("\n  PRIMARY TEST — within-target AUC against measured binding")
    res = {}
    for col, label in (("raw", "raw interface pLDDT"),
                       ("margin", "design minus its own scrambles"),
                       ("ipsae", "released ipSAE (reference)")):
        a = within_auc(df, col)
        res[col] = a
        if a is not None:
            print(f"    {label:34}{a:.3f}")
    lo, hi = paired_ci(df, "margin", "raw", args.n_boot)
    d_obs = (res["margin"] - res["raw"]) if res.get("raw") else None
    flag = ("excludes 0" if lo is not None and lo * hi > 0 else "includes 0")
    print(f"\n    margin - raw: {d_obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {flag}")

    print("\n  " + ("The scramble control adds information about measured binding."
                    if flag == "excludes 0" and d_obs > 0 else
                    "The scramble control does not add information here. On designed"
                    "\n  proteins a permutation does not fold, so binders and"
                    "\n  non-binders receive an equally ruined scramble and the"
                    "\n  subtraction is close to a constant."))

    Path(args.out).write_text(json.dumps(
        {"n": len(df), "n_binders": int(df["y"].sum()),
         "n_targets": int(df["target"].nunique()),
         "auc": res, "delta_margin_minus_raw": d_obs, "ci": [lo, hi],
         "margin_binders": float(b["margin"].mean()),
         "margin_nonbinders": float(nb["margin"].mean()),
         "margin_welch_p": float(pv)}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
