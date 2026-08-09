"""Does a physics energy function add anything the confidence head does not?

Every readout in this dissertation comes out of the model's own confidence head,
so all of them are bounded by the variance decomposition of Section 7.9.2, and
combining them buys nothing (7.9.2) -- nor does combining whole models, which
gains 0.0008 AUC with a bootstrap interval straddling zero. An empirical energy
function is different in kind: it reads the coordinates and knows nothing about
what the model believed.

That only works if the coordinates are physical, and for most of this project
they are not. Boltz-2 at 10 sampling steps produces 14% physically plausible
backbone bonds, a median CA-CA distance of 5.48 A against an ideal 3.80, and
56% of bonds beyond 5 A. PRODIGY on those structures returns femtomolar
affinities for arbitrary peptides -- the failure is the input, not the method.

DeCAF is distilled *for* ten steps and its backbones are 90.6% physical with a
median of 3.70 A. On those, an energy function has something to read. This is
also the direction the nanobody benchmark took to reach ROC AUC 0.90, where the
best confidence metric managed 0.86.

PRODIGY predicts binding dG from intermolecular contacts grouped by residue
polarity plus non-interacting surface. dG is negative for favourable binding, so
it is negated here to keep "higher is better" throughout.

The test is not whether dG separates cognates on its own -- it is whether adding
dG to the best confidence readout beats that readout alone, under
leave-one-receptor-out cross-validation.

Usage:
    python src/physics_rescore.py --structures DIR
"""

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
warnings.filterwarnings("ignore")

FLIP = {"iface_pae", "mpae"}
CONF = ("iptm", "iface_plddt", "receptor_side", "mpae", "ipsae", "pdockq2")


def prodigy_dg(path):
    """Predicted binding dG in kcal/mol, or None if PRODIGY cannot score it."""
    from prodigy_prot.modules.parsers import parse_structure
    from prodigy_prot.modules.prodigy import Prodigy
    try:
        # PRODIGY narrates chain breaks to stdout; the geometry audit is done
        # separately and far more precisely, so the commentary is discarded.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            st, _, _ = parse_structure(str(path))
            model = st[0] if isinstance(st, list) else st
            chains = [c.id for c in model]
            if len(chains) != 2:
                return None
            pr = Prodigy(model, path.stem, chains, 25.0)
            pr.predict()
            return {"dg": float(pr.ba_val), "kd": float(pr.kd_val),
                    "n_contacts": int(len(pr.ic_network))}
    except (Exception, SystemExit):
        # PRODIGY calls sys.exit() rather than raising when it rejects a
        # structure, and SystemExit descends from BaseException, so an
        # `except Exception` here silently aborted the whole run at the first
        # unscoreable file instead of skipping it.
        return None


def val(row, m):
    return -row[m] if m in FLIP else row[m]


def paired(df, metric):
    """Cognate minus its own scramble."""
    diffs = []
    for _, g in df.groupby("receptor_id"):
        c = g[g.label == "cognate"]
        s = g[g.label == "scrambled"]
        if len(c) and len(s):
            diffs += [val(c.iloc[0], metric) - val(r, metric)
                      for _, r in s.iterrows()]
    if len(diffs) < 3:
        return None
    d = np.array(diffs, float)
    return {"effect": float(d.mean()), "p": float(stats.ttest_1samp(d, 0).pvalue)}


def ranked(df, metric):
    ranks, sizes = [], []
    for _, g in df.groupby("receptor_id"):
        c = g[g.label == "cognate"]
        dd = g[g.label == "decoy"]
        if not len(c) or not len(dd):
            continue
        sc = [val(c.iloc[0], metric)] + [val(r, metric) for _, r in dd.iterrows()]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    if len(ranks) < 4:
        return None
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    return {"mean_rank": float(r.mean()), "chance": float(exp.mean()),
            "first": int((r == 1).sum()), "n": len(r),
            "p": float(stats.wilcoxon(r - exp)[1])}


def within_auc(df, cols):
    """LORO-CV AUC after within-receptor standardisation."""
    y = (df["label"] == "cognate").astype(int).to_numpy()
    X = df[cols].to_numpy(float)
    g = df["receptor_id"].to_numpy()
    pred = np.zeros(len(df))
    for r in np.unique(g):
        te, tr = g == r, g != r
        if y[tr].sum() in (0, tr.sum()):
            continue
        sx = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sx.transform(X[tr]), y[tr])
        pred[te] = clf.predict_proba(sx.transform(X[te]))[:, 1]
    z = pd.Series(pred).groupby(g).transform(
        lambda x: (x - x.mean()) / (x.std() or 1)).to_numpy()
    return roc_auc_score(y, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", required=True)
    ap.add_argument("--scores", default=str(
        ART / "heldout_panel" / "heldout_scores_pae2.json"))
    ap.add_argument("--out", default=str(ART / "physics_rescore.json"))
    args = ap.parse_args()

    by = {r["name"]: r for r in json.loads(Path(args.scores).read_text())}
    files = sorted(Path(args.structures).glob("*_model_0.pdb"))
    rows = []
    for i, f in enumerate(files, 1):
        name = f.name.replace("_model_0.pdb", "")
        if name not in by:
            continue
        e = prodigy_dg(f)
        if e:
            rows.append({**by[name], **e, "neg_dg": -e["dg"]})
        if i % 20 == 0:
            print(f"  scored {i}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("PRODIGY scored nothing")
    print(f"\n{len(df)} structures scored, {df.receptor_id.nunique()} receptors")
    print(f"  dG range {df.dg.min():.1f} to {df.dg.max():.1f} kcal/mol, "
          f"median {df.dg.median():.1f}")
    print(f"  contacts median {df.n_contacts.median():.0f}\n")

    have = [m for m in CONF if m in df.columns and df[m].notna().all()]
    print(f"{'readout':16} {'cog-scr':>10} {'p':>9} {'rank':>7} {'p':>8} "
          f"{'#1':>4} {'AUC within':>11}")
    print("-" * 74)
    out = {}
    for m in ["neg_dg"] + have:
        pr, rk = paired(df, m), ranked(df, m)
        a = within_auc(df, [m])
        out[m] = {"paired": pr, "rank": rk, "auc_within": a}
        rs = f"{rk['mean_rank']:7.2f} {rk['p']:8.4f} {rk['first']:4d}" if rk else " " * 20
        print(f"{m:16} {pr['effect']:+10.3f} {pr['p']:9.5f}{rs} {a:11.3f}")

    print(f"\n{'=' * 74}\nDoes physics add to the confidence head? (LORO-CV, "
          f"within-receptor)\n{'=' * 74}")
    best = max(have, key=lambda m: out[m]["auc_within"])
    print(f"{'combination':44} {'AUC':>8} {'vs best alone':>14}")
    print("-" * 70)
    base = out[best]["auc_within"]
    print(f"{'best confidence readout alone: ' + best:44} {base:8.3f} {'--':>14}")
    for label, cols in {
        f"{best} + PRODIGY dG": [best, "neg_dg"],
        "all confidence readouts": list(have),
        "all confidence readouts + PRODIGY dG": list(have) + ["neg_dg"],
    }.items():
        a = within_auc(df, cols)
        out[label] = {"auc_within": a, "n_features": len(cols)}
        print(f"{label:44} {a:8.3f} {a - base:+14.3f}")

    Path(args.out).write_text(json.dumps(
        {"per_fold": df.to_dict("records"), "summary": out}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
