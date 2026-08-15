"""Can a sequence model do this without folding anything?

Everything in Section 7 reads a cofolding model's confidence head. Two 2026
results suggest that is the wrong instrument. King et al. fine-tuned Boltz-2 for
protein-protein affinity and found it "underperforms relative to sequence-based
alternatives in both small- and larger-scale data regimes", concluding that
current structure-based representations are not primed for affinity prediction
-- a conclusion that does not depend on how much data is thrown at it. BindPred
predicts binding affinity from ESM-2 embeddings and reports that adding explicit
structural information buys little over sequence alone.

If that transfers to peptide binders, the entire folding pipeline is an
expensive way to reach a number a single forward pass through a language model
also reaches. This tests it on the same panels, the same folds, the same labels
and the same leave-one-receptor-out protocol, so the comparison is exact. The
number to beat is 0.807, the best structural readout in this work (DeCAF
interface pLDDT, within-receptor AUC).

**The scramble control is what makes this honest.** A language model given a
peptide sees its composition, and Section 7.4 established that composition alone
reproduces most of the apparent discrimination between a cognate and a decoy. A
scramble has identical composition and length, so any embedding that is
effectively a weighted residue count *cannot* separate a cognate from its own
scramble. Reporting only AUC against decoys would let a composition detector
look like a binding predictor -- which is the exact error this dissertation
diagnosed in ipTM.

Receptor-only and peptide-only features are scored alongside the pair, because
if peptide-only does as well as the pair, the model is not reading an
interaction at all.

Usage:
    python src/plm_baseline.py --model facebook/esm2_t33_650M_UR50D
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
ART = REPO_ROOT / "artifacts"
warnings.filterwarnings("ignore")


def panels():
    """Every (receptor, peptide, label) the folding benchmark used, both panels."""
    rows = []
    main = json.loads((ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())
    for p in main:
        rows.append({"panel": "in_training", "receptor_id": p["receptor_id"],
                     "receptor": p["receptor"], "peptide": p["peptide"],
                     "label": p["label"], "peptide_from": p["peptide_from"]})
    # the held-out pairs are deterministic in the seed, so they are rebuilt
    # rather than stored -- same call, same seed, same peptides as were folded
    from heldout_panel import ALREADY, NEW
    from pdb_binder_benchmark import build_pairs
    seqdir = ART / "heldout_panel" / "sequences"
    complexes = {}
    for pid in ALREADY + NEW:
        f = seqdir / f"{pid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            complexes[pid] = {"receptor": d["receptor"], "peptide": d["peptide"]}
    for p in build_pairs(complexes, 3, 2, 0):
        rows.append({"panel": "held_out", **{k: p[k] for k in
                     ("receptor_id", "receptor", "peptide", "label", "peptide_from")}})
    return pd.DataFrame(rows)


def embed(seqs, model_name, batch=8):
    """Mean-pooled residue embeddings, one vector per sequence."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).to(dev).eval()
    uniq = sorted(set(seqs))
    out = {}
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            chunk = uniq[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
            h = mod(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            # mean over real residues only; padding would otherwise shrink the
            # vector in proportion to how short the sequence is
            pooled = (h * mask).sum(1) / mask.sum(1)
            for s, v in zip(chunk, pooled.float().cpu().numpy()):
                out[s] = v
            print(f"  embedded {min(i + batch, len(uniq))}/{len(uniq)}", flush=True)
    return out


def features(df, emb, kind):
    R = np.stack([emb[s] for s in df.receptor])
    P = np.stack([emb[s] for s in df.peptide])
    if kind == "peptide only":
        return P
    if kind == "receptor only":
        return R
    if kind == "concat":
        return np.hstack([R, P])
    if kind == "concat + product":
        return np.hstack([R, P, R * P])
    raise ValueError(kind)


def loro(df, X, C=0.01):
    """Leave-one-receptor-out, within-receptor standardised AUC.

    Regularisation is deliberately strong: 1280-dimensional embeddings against
    22 receptors will otherwise fit the training receptors perfectly and
    generalise to none of them.
    """
    y = (df.label == "cognate").astype(int).to_numpy()
    g = df.receptor_id.to_numpy()
    pred = np.zeros(len(df))
    for r in np.unique(g):
        te, tr = g == r, g != r
        if y[tr].sum() in (0, tr.sum()):
            continue
        sx = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=C).fit(sx.transform(X[tr]), y[tr])
        pred[te] = clf.predict_proba(sx.transform(X[te]))[:, 1]
    z = pd.Series(pred).groupby(g).transform(
        lambda x: (x - x.mean()) / (x.std() or 1)).to_numpy()
    return roc_auc_score(y, z), pred, y, g


def scramble_and_rank(df, pred):
    """The two tests every readout in Section 7 had to pass."""
    d = df.copy()
    d["v"] = pred
    diffs = []
    for _, grp in d.groupby("receptor_id"):
        c = grp[grp.label == "cognate"]
        s = grp[grp.label == "scrambled"]
        if len(c) and len(s):
            diffs += [c.iloc[0].v - r.v for _, r in s.iterrows()]
    e = np.array(diffs, float)
    ranks, sizes = [], []
    for _, grp in d.groupby("receptor_id"):
        c = grp[grp.label == "cognate"]
        dd = grp[grp.label == "decoy"]
        if not len(c) or not len(dd):
            continue
        sc = [c.iloc[0].v] + list(dd.v)
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    return ({"effect": float(e.mean()), "p": float(stats.ttest_1samp(e, 0).pvalue)},
            {"mean_rank": float(r.mean()), "chance": float(exp.mean()),
             "first": int((r == 1).sum()), "n": len(r),
             "p": float(stats.wilcoxon(r - exp)[1])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--out", default=str(ART / "plm_baseline.json"))
    args = ap.parse_args()

    df = panels()
    print(f"{len(df)} folds | {df.receptor_id.nunique()} receptors | "
          f"panels {df.panel.unique().tolist()}")
    seqs = sorted(set(df.receptor) | set(df.peptide))
    print(f"embedding {len(seqs)} unique sequences with {args.model}")
    emb = embed(seqs, args.model)

    result = {}
    print(f"\n{'features':22} {'panel':12} {'AUC':>7} {'cog-scr':>9} {'p':>9} "
          f"{'rank':>6} {'p':>8} {'#1':>7}")
    print("-" * 84)
    for kind in ("peptide only", "receptor only", "concat", "concat + product"):
        for panel in ("in_training", "held_out", "both"):
            sub = df if panel == "both" else df[df.panel == panel]
            X = features(sub, emb, kind)
            auc, pred, _, _ = loro(sub.reset_index(drop=True), X)
            sc, rk = scramble_and_rank(sub.reset_index(drop=True), pred)
            result.setdefault(kind, {})[panel] = {
                "auc_within": auc, "scramble": sc, "rank": rk,
                "n_receptors": int(sub.receptor_id.nunique())}
            print(f"{kind:22} {panel:12} {auc:7.3f} {sc['effect']:+9.3f} "
                  f"{sc['p']:9.5f} {rk['mean_rank']:6.2f} {rk['p']:8.4f} "
                  f"{rk['first']:3d}/{rk['n']:<3d}")

    print("\nReference, best structural readout (DeCAF interface pLDDT): "
          "within-receptor AUC 0.807")
    print("A model that separates cognate from decoy but NOT from its own")
    print("scramble is reading composition, which is Section 7.4's finding for ipTM.")
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "results": result}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
