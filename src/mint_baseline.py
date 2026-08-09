"""A sequence model that was actually trained on interactions, not fitted here.

An earlier attempt embedded receptors and peptides separately with ESM-2 and
fitted a classifier on the concatenation. That could not work, for two reasons
worth recording so the mistake is not repeated:

  * a linear model on [receptor, peptide] computes w_R.R + w_P.P, an *additive*
    score, and matching is not additive -- so the concatenation carried no more
    information than the peptide alone (AUC 0.093 against 0.095)
  * each distinct peptide appears once as a cognate and two to five times as a
    decoy, since it is a decoy for every other receptor, so the optimal
    peptide-only rule is "always decoy" and the AUC inverts below 0.5

MINT avoids both. It is ESM-2 650M with cross-chain attention blocks, trained on
96 million protein-protein interactions from STRING, so its embedding of a pair
is computed jointly and *depends on the pairing*. Swapping the peptide changes
the receptor's embedding, which is exactly the dependence the ESM-2 setup
lacked. It also ships an MLP trained on Bernett et al.'s gold-standard PPI set,
giving a genuinely zero-shot interaction probability with nothing fitted from
this project's 22 receptors.

Two scores are reported:

  zero-shot     the pretrained Bernett MLP's interaction probability. Nothing
                is fitted, so the 22-receptor limit does not apply at all.
  probe         logistic regression on MINT embeddings under leave-one-
                receptor-out. Legitimate here in a way it was not for ESM-2,
                because the embedding already encodes the pairing.

The number to beat is 0.807, the best structural readout in this work. And the
scramble control still governs: a cognate and its own scramble have identical
composition, so anything that separates cognate from decoy but not from scramble
is reading composition, which is Section 7.4's verdict on ipTM.

Usage:
    python src/mint_baseline.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
ART = REPO_ROOT / "artifacts"
MINT_HOME = Path.home() / ".cache" / "mint"
warnings.filterwarnings("ignore")


def mint_scores(df, device, batch=4, max_len=512):
    """Joint embeddings and the pretrained PPI probability for every pair."""
    from mint.helpers.extract import CollateFn, MINTWrapper, load_config
    cfg = load_config(str(MINT_HOME / "repo" / "data" / "esm2_t33_650M_UR50D.json"))
    wrapper = MINTWrapper(cfg, str(MINT_HOME / "mint.ckpt"),
                          sep_chains=True, device=device).eval()
    collate = CollateFn(max_len)

    mlp = None
    mlp_path = MINT_HOME / "bernett_mlp.pth"
    if mlp_path.exists():
        from mint.helpers.predict import SimpleMLP
        mlp = SimpleMLP(2560, 1)
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        mlp = mlp.to(device).eval()

    embs, probs = [], []
    pairs = list(zip(df.receptor, df.peptide))
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            chains, chain_ids = collate([(a, b) for a, b in chunk])
            e = wrapper(chains.to(device), chain_ids.to(device))
            embs.append(e.float().cpu().numpy())
            if mlp is not None:
                probs.append(torch.sigmoid(mlp(e)).squeeze(-1).float().cpu().numpy())
            print(f"  {min(i + batch, len(pairs))}/{len(pairs)}", flush=True)
    return np.vstack(embs), (np.concatenate(probs) if probs else None)


def loro(df, X, C=0.01):
    y = (df.label == "cognate").astype(int).to_numpy()
    g = df.receptor_id.to_numpy()
    pred = np.zeros(len(df))
    for r in np.unique(g):
        te, tr = g == r, g != r
        if y[tr].sum() in (0, tr.sum()):
            continue
        sx = StandardScaler().fit(X[tr])
        pred[te] = LogisticRegression(max_iter=3000, C=C).fit(
            sx.transform(X[tr]), y[tr]).predict_proba(sx.transform(X[te]))[:, 1]
    return pred


def evaluate(df, score, label):
    d = df.copy()
    d["v"] = score
    g = d.receptor_id.to_numpy()
    y = (d.label == "cognate").astype(int).to_numpy()
    z = pd.Series(d.v).groupby(g).transform(
        lambda x: (x - x.mean()) / (x.std() or 1)).to_numpy()
    auc = roc_auc_score(y, z)
    diffs = []
    for _, grp in d.groupby("receptor_id"):
        c, s = grp[grp.label == "cognate"], grp[grp.label == "scrambled"]
        if len(c) and len(s):
            diffs += [c.iloc[0].v - r.v for _, r in s.iterrows()]
    e = np.array(diffs, float)
    ranks, sizes = [], []
    for _, grp in d.groupby("receptor_id"):
        c, dd = grp[grp.label == "cognate"], grp[grp.label == "decoy"]
        if not len(c) or not len(dd):
            continue
        sc = [c.iloc[0].v] + list(dd.v)
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    out = {"auc_within": float(auc),
           "scramble_effect": float(e.mean()),
           "scramble_p": float(stats.ttest_1samp(e, 0).pvalue),
           "mean_rank": float(r.mean()), "chance": float(exp.mean()),
           "first": int((r == 1).sum()), "n_receptors": len(r),
           "rank_p": float(stats.wilcoxon(r - exp)[1])}
    print(f"{label:34} {auc:7.3f} {e.mean():+9.4f} {out['scramble_p']:9.5f} "
          f"{r.mean():6.2f} {out['rank_p']:8.4f} {out['first']:3d}/{len(r):<3d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "mint_baseline.json"))
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from plm_baseline import panels
    df = panels().reset_index(drop=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(df)} folds | {df.receptor_id.nunique()} receptors | device {device}")
    emb, prob = mint_scores(df, device, args.batch)
    print(f"\nembeddings {emb.shape}"
          + (" | zero-shot PPI probability available" if prob is not None else ""))

    print(f"\n{'score':34} {'AUC':>7} {'cog-scr':>9} {'p':>9} {'rank':>6} "
          f"{'p':>8} {'#1':>7}")
    print("-" * 88)
    res = {}
    for panel in ("in_training", "held_out", "both"):
        sub = df if panel == "both" else df[df.panel == panel]
        idx = sub.index.to_numpy()
        s = sub.reset_index(drop=True)
        if prob is not None:
            res[f"zero-shot PPI [{panel}]"] = evaluate(
                s, prob[idx], f"zero-shot Bernett MLP [{panel}]")
        res[f"probe [{panel}]"] = evaluate(
            s, loro(s, emb[idx]), f"LORO probe on MINT emb [{panel}]")

    print("\nReference: DeCAF interface pLDDT, within-receptor AUC 0.807")
    print("A score that beats decoys but not a peptide's own scramble is")
    print("reading composition -- Section 7.4's verdict on ipTM.")
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
