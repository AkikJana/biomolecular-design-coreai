"""Does combining models beat the best single model?

Section 7.9.2 found no headroom in combining thirteen readouts: cross-validated,
the combination scored worse than the best single one. Every one of those
thirteen came from the same forward pass, so they share their errors, and the
variance decomposition of 7.9.2 bounds any of them.

Combining *models* is a different proposition. Boltz-2, Boltz-1 and DeCAF-Boltz
have separately drawn sampling noise, and DeCAF has a different base and
sampling regime, so their errors are not the same errors. That bound is
per-model and says nothing about their combination. This is also the only
direction the recent literature suggests clears the numbers reported here --
a nanobody benchmark reaching ROC AUC 0.90 does it with a model ensemble plus
physics-based rescoring, against 0.77 for the best single readout in this work.

Testing it needs no new models: the same 132 pairs have already been folded on
all three arms, so the ensemble can be evaluated before deciding whether adding
IntFold or Chai-1 is worth the effort.

Scored under leave-one-receptor-out cross-validation. Folds of one receptor
share a baseline, so a random split leaks and would report an optimistic number
for the combination specifically -- which is the claim under test.

Usage:
    python src/model_ensemble.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
warnings.filterwarnings("ignore")

ARMS = ("boltz2", "boltz1", "decaf")
READOUTS = ("iptm", "iface_plddt", "receptor_side")


def frame():
    """One row per fold, with each arm's readouts as separate columns."""
    b2 = json.loads((ART / "iface_side_split.json").read_text())["per_complex"]
    sc = {p["name"]: p["score"] for p in json.loads(
        (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}
    donor = {p["name"]: p.get("peptide_from") for p in json.loads(
        (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}
    rows = {}
    for r in b2:
        rows[r["name"]] = {"name": r["name"], "receptor_id": r["receptor_id"],
                           "label": r["label"],
                           "peptide_from": donor.get(r["name"]),
                           "boltz2_iptm": sc.get(r["name"], np.nan),
                           "boltz2_iface_plddt": r["iface_plddt"],
                           "boltz2_receptor_side": r["receptor_side"]}
    for arm in ("boltz1", "decaf"):
        for r in json.loads((ART / f"{arm}_scramble_result.json").read_text())["per_fold"]:
            key = r.get("job")
            if key not in rows:
                continue
            for m in READOUTS:
                if m in r:
                    rows[key][f"{arm}_{m}"] = r[m]
    df = pd.DataFrame(rows.values())
    return df.dropna(subset=[f"{a}_{m}" for a in ARMS for m in READOUTS])


def loro_auc(df, cols):
    """Leave-one-receptor-out AUC for a logistic combination of `cols`."""
    y = (df["label"] == "cognate").astype(int).to_numpy()
    X = df[cols].to_numpy(float)
    groups = df["receptor_id"].to_numpy()
    pred = np.zeros(len(df))
    for r in np.unique(groups):
        te = groups == r
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        sx = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sx.transform(X[tr]), y[tr])
        pred[te] = clf.predict_proba(sx.transform(X[te]))[:, 1]
    within = pd.Series(pred).groupby(groups).transform(
        lambda x: (x - x.mean()) / (x.std() or 1)).to_numpy()
    return roc_auc_score(y, pred), roc_auc_score(y, within)


def single_auc(df, col):
    y = (df["label"] == "cognate").astype(int).to_numpy()
    v = df[col].to_numpy(float)
    within = pd.Series(v).groupby(df["receptor_id"].to_numpy()).transform(
        lambda x: (x - x.mean()) / (x.std() or 1)).to_numpy()
    return roc_auc_score(y, v), roc_auc_score(y, within)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "model_ensemble.json"))
    args = ap.parse_args()

    df = frame()
    print(f"{len(df)} folds on all three arms, "
          f"{df.receptor_id.nunique()} receptors\n")

    print(f"{'candidate':44} {'AUC pooled':>11} {'AUC within':>11}")
    print("-" * 68)
    out, best = {}, ("", 0.0)
    for arm in ARMS:
        for m in READOUTS:
            col = f"{arm}_{m}"
            p, w = single_auc(df, col)
            out[col] = {"pooled": p, "within": w}
            if w > best[1]:
                best = (col, w)
            print(f"{'single: ' + col:44} {p:11.3f} {w:11.3f}")

    print()
    combos = {
        "all 3 arms x interface pLDDT":
            [f"{a}_iface_plddt" for a in ARMS],
        "all 3 arms x ipTM":
            [f"{a}_iptm" for a in ARMS],
        "DeCAF ipTM + interface pLDDT":
            ["decaf_iptm", "decaf_iface_plddt"],
        "all 3 arms x all 3 readouts (9)":
            [f"{a}_{m}" for a in ARMS for m in READOUTS],
    }
    for label, cols in combos.items():
        p, w = loro_auc(df, cols)
        out[label] = {"pooled": p, "within": w, "n_features": len(cols)}
        print(f"{'LORO ensemble: ' + label:44} {p:11.3f} {w:11.3f}")

    print(f"\nbest single readout: {best[0]} at within-receptor AUC {best[1]:.3f}")
    ens = {k: v for k, v in out.items() if v.get("n_features")}
    better = [k for k, v in ens.items() if v["within"] > best[1]]
    print(f"ensembles beating it: {len(better)} of {len(ens)}"
          + (f" -> {better}" if better else ""))
    print("\nIf none beat it, cross-model ensembling does not clear the ceiling")
    print("either, and adding a fourth model is unlikely to be the answer.")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
