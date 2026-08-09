"""Three re-analyses of folds already on disk. No new folding.

A. **Mixed models where rank tests were used.** Section 7.9 showed the estimator,
   not the evidence, was the binding constraint: Boltz-2 ipTM went p = 0.034 ->
   0.0042 on identical data. The alanine scan of 7.7 is the case that matters --
   it collapsed 88 folds into 22 receptor means, returned p = 0.244, and
   concluded site-localisation "would need n = 123 receptors". Fitted at fold
   level with receptor as a random effect, residual variance is estimated from
   ~63 degrees of freedom rather than 21.

B. **What replicate averaging actually buys.** Averaging k folds divides
   sampling noise by k while leaving the interaction term alone, so the arm with
   the *most* noise gains most. DeCAF has both the highest signal and the highest
   noise, which predicts it benefits disproportionately. The 96 replicate folds
   measure the curve directly instead of extrapolating it.

C. **How much is left to extract.** Section 7.9 called sigma2_delta/sigma2_eps an
   "information-theoretic ceiling on any score". That was too strong: it is
   per-metric, describing one readout's residual structure, not a bound over
   readouts. The honest version asks what the best *combination* of all 14
   available readouts achieves under leave-one-receptor-out cross-validation,
   and compares it to the best single metric. The gap is the headroom that
   further metric-hunting could win; if it is small, that search should stop.

   Cross-validation is by receptor, never by fold: folds of the same receptor
   share a baseline, so a random split would leak it and inflate the estimate.

Usage:
    python src/frontier_analysis.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"


# ----------------------------------------------------------------- part A
def mixed_alanine_scan():
    """7.7's site-localisation question, re-fitted at fold level."""
    recs = json.loads((ART / "receptor_controls_result.json").read_text())["per_fold"]
    df = pd.DataFrame([{"receptor": r["receptor_id"], "arm": r["arm"],
                        "receptor_side": r["receptor_side"],
                        "iface_plddt": r["iface_plddt"]} for r in recs])
    print(f"\n{'=' * 74}\nA. Alanine scan, mixed model vs paired t-test\n{'=' * 74}")
    print(f"   {len(df)} folds, {df['receptor'].nunique()} receptors, "
          f"{df['arm'].nunique()} arms")

    out = {}
    for metric in ("receptor_side", "iface_plddt"):
        d = df.dropna(subset=[metric]).copy()
        # surf_ala is the reference level: the question is whether mutating the
        # binding site costs MORE than mutating an equal number of exposed
        # residues elsewhere.
        d["arm"] = pd.Categorical(d["arm"],
                                  categories=["surf_ala", "iface_ala", "real",
                                              "scrambled"])
        try:
            fit = smf.mixedlm(f"{metric} ~ C(arm)", d, groups=d["receptor"]).fit(
                reml=True, method="lbfgs")
        except Exception as exc:
            print(f"   {metric}: fit failed ({str(exc)[:40]})")
            continue
        key = [k for k in fit.params.index if "iface_ala" in k]
        if not key:
            continue
        beta, p = float(fit.params[key[0]]), float(fit.pvalues[key[0]])
        se = float(fit.bse[key[0]])
        out[metric] = {"beta_iface_vs_surf": beta, "se": se, "p": p,
                       "ci95": [beta - 1.96 * se, beta + 1.96 * se]}
        print(f"   {metric:14} iface_ala - surf_ala = {beta:+.3f} "
              f"(SE {se:.3f})  p = {p:.4f}")
    print("   Section 7.7 paired t-test on 22 receptor means: -1.84, p = 0.244")
    return out


def mixed_all_arms():
    """Cognate-vs-scramble for every arm, fold level."""
    print(f"\n{'=' * 74}\nA2. Cognate vs OWN SCRAMBLE, mixed model, all arms\n{'=' * 74}")
    sources = {"boltz2": None, "boltz1": "boltz1_scramble_result.json",
               "decaf": "decaf_scramble_result.json"}
    # The reference values must answer the SAME question. These are the
    # cognate-vs-own-scramble t-tests of 7.6/7.8, not the cognate-vs-decoy rank
    # tests -- mixing the two would compare different hypotheses.
    ttest_p = {("boltz2", "iptm"): 0.416, ("boltz2", "iface_plddt"): 1e-5,
               ("boltz1", "iptm"): 0.0043, ("boltz1", "iface_plddt"): 0.067,
               ("decaf", "iptm"): 2e-5, ("decaf", "iface_plddt"): 1e-5}
    out = {}
    for arm, fname in sources.items():
        if arm == "boltz2":
            side = json.loads((ART / "iface_side_split.json").read_text())["per_complex"]
            sc = {p["name"]: p["score"] for p in json.loads(
                (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}
            rows = [{"receptor": r["receptor_id"], "label": r["label"],
                     "iface_plddt": r["iface_plddt"],
                     "iptm": sc.get(r["name"], np.nan)} for r in side]
        else:
            recs = json.loads((ART / fname).read_text())["per_fold"]
            rows = [{"receptor": r["receptor_id"], "label": r["label"],
                     "iface_plddt": r["iface_plddt"], "iptm": r["iptm"]}
                    for r in recs]
        df = pd.DataFrame(rows)
        df = df[df["label"].isin(["cognate", "scrambled"])]
        for metric in ("iptm", "iface_plddt"):
            d = df.dropna(subset=[metric]).copy()
            d["cognate"] = (d["label"] == "cognate").astype(float)
            try:
                fit = smf.mixedlm(f"{metric} ~ cognate", d,
                                  groups=d["receptor"]).fit(reml=True, method="lbfgs")
            except Exception:
                continue
            p = float(fit.pvalues.get("cognate", np.nan))
            out.setdefault(arm, {})[metric] = {"beta": float(fit.params["cognate"]),
                                               "p": p}
            print(f"   {arm:8} {metric:14} beta {fit.params['cognate']:+8.4f}"
                  f"  mixed p {p:9.6f}   (t-test p "
                  f"{ttest_p.get((arm, metric), float('nan')):.5f})")
    return out


# ----------------------------------------------------------------- part B
def replicate_curve():
    """Discriminability as a function of replicates averaged."""
    recs = json.loads((ART / "decaf_replicate_result.json").read_text())["per_fold"]
    reps = sorted({r["replicate"] for r in recs})
    by = {}
    for r in recs:
        by.setdefault(r["job"], {})[r["replicate"]] = r
    meta = {r["job"]: (r["receptor_id"], r["label"]) for r in recs}

    print(f"\n{'=' * 74}\nB. What replicate averaging buys (DeCAF, {len(by)} complexes)"
          f"\n{'=' * 74}")
    print(f"   {'metric':14} {'k':>2} {'effect':>9} {'noise SD':>10} {'d = eff/SD':>11}")
    out = {}
    rng = np.random.default_rng(0)
    for metric in ("iptm", "iface_plddt"):
        for k in range(1, len(reps) + 1):
            # average k randomly chosen replicates, repeated to stabilise
            effs = []
            for _ in range(200):
                avg = {}
                for job, per in by.items():
                    pick = rng.choice(list(per), size=k, replace=False)
                    avg[job] = float(np.mean([per[r][metric] for r in pick]))
                bycog = {}
                for job, v in avg.items():
                    rid, lab = meta[job]
                    bycog.setdefault(rid, {}).setdefault(lab, []).append(v)
                diffs = []
                for rid, g in bycog.items():
                    if "cognate" in g and "scrambled" in g:
                        diffs += [g["cognate"][0] - s for s in g["scrambled"]]
                if diffs:
                    effs.append(np.mean(diffs))
            eff = float(np.mean(effs))
            # noise of a k-average is sigma/sqrt(k)
            sd1 = float(np.sqrt(np.mean([np.var([per[r][metric] for r in per], ddof=1)
                                         for per in by.values() if len(per) > 1])))
            sdk = sd1 / np.sqrt(k)
            out.setdefault(metric, {})[k] = {"effect": eff, "noise_sd": sdk,
                                             "d": eff / sdk if sdk else float("nan")}
            print(f"   {metric:14} {k:2d} {eff:9.4f} {sdk:10.4f} {eff / sdk:11.2f}")
    return out


# ----------------------------------------------------------------- part C
def multivariate_headroom():
    """Best cross-validated linear readout vs the best single metric."""
    rescore = {r["name"]: r for r in
               json.loads((ART / "rescore_metrics.json").read_text())["per_complex"]}
    side = {r["name"]: r for r in
            json.loads((ART / "iface_side_split.json").read_text())["per_complex"]}
    pae = {r["name"]: r for r in json.loads((ART / "pae_metrics.json").read_text())}

    feats = ["iptm", "pdockq", "iface_plddt", "n_contacts", "contact_density",
             "delta_sasa", "receptor_side", "peptide_side", "peptide_whole",
             "iface_pae", "iface_pae_min", "pae_frac_lt10", "ipsae"]
    rows = []
    for name, r in rescore.items():
        if name not in side or name not in pae:
            continue
        row = {"name": name, "receptor": r["receptor_id"], "label": r["label"]}
        for f in feats:
            row[f] = r.get(f, side[name].get(f, pae[name].get(f, np.nan)))
        rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=feats)

    print(f"\n{'=' * 74}\nC. Headroom: best combination vs best single readout"
          f"\n{'=' * 74}")
    print(f"   {len(df)} folds, {df['receptor'].nunique()} receptors, "
          f"{len(feats)} readouts")

    out = {}
    for task, pos, neg in (("cognate vs own scramble", "cognate", "scrambled"),
                           ("cognate vs decoy", "cognate", "decoy")):
        d = df[df["label"].isin([pos, neg])].copy()
        y = (d["label"] == pos).astype(int).to_numpy()
        groups = d["receptor"].to_numpy()
        X = d[feats].to_numpy()

        # leave-one-receptor-out: folds of a receptor share a baseline, so a
        # random split would leak it
        def loro_scores(cols):
            pred = np.zeros(len(y), dtype=float)
            for rid in np.unique(groups):
                tr, te = groups != rid, groups == rid
                if len(np.unique(y[tr])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr][:, cols])
                clf = LogisticRegression(max_iter=2000, C=0.5).fit(
                    sc.transform(X[tr][:, cols]), y[tr])
                pred[te] = clf.predict_proba(sc.transform(X[te][:, cols]))[:, 1]
            return pred

        singles = {}
        for i, f in enumerate(feats):
            try:
                singles[f] = roc_auc_score(y, d[f].to_numpy())
            except Exception:
                continue
        # a readout may be informative in either direction
        singles = {f: max(a, 1 - a) for f, a in singles.items()}
        best_f = max(singles, key=singles.get)
        auc_all = roc_auc_score(y, loro_scores(list(range(len(feats)))))
        auc_best_single_cv = roc_auc_score(
            y, loro_scores([feats.index(best_f)]))

        out[task] = {"n": int(len(d)), "best_single": best_f,
                     "auc_best_single_raw": float(singles[best_f]),
                     "auc_best_single_loro": float(auc_best_single_cv),
                     "auc_all_readouts_loro": float(auc_all),
                     "headroom": float(auc_all - auc_best_single_cv)}
        print(f"\n   {task}  (n = {len(d)})")
        print(f"     best single readout      {best_f:16} AUC {singles[best_f]:.3f}"
              f"  (in-sample)")
        print(f"     same, cross-validated    {'':16} AUC {auc_best_single_cv:.3f}")
        print(f"     all {len(feats)} readouts combined {'':11} AUC {auc_all:.3f}")
        print(f"     headroom                 {'':16}     {auc_all - auc_best_single_cv:+.3f}")
    return out


def within_receptor_headroom():
    """Part C repeated with the receptor baseline removed.

    The pooled AUC of `multivariate_headroom` mixes two things: whether a
    readout separates cognate from negative *within* a receptor, and whether
    receptors happen to sit at different baselines. The second is exactly the
    nuisance the within-receptor rank tests were built to remove, and it is why
    peptide_side -- which Section 7.7 showed is largely peptide foldability --
    can top a pooled ranking without being the better discriminator.

    Here every comparison is made inside one receptor: concordance is the
    fraction of (cognate, negative) pairs from the same receptor that the readout
    orders correctly. Averaging that over receptors is the within-receptor AUC,
    and chance is 0.5 regardless of how receptor baselines differ.
    """
    rescore = {r["name"]: r for r in
               json.loads((ART / "rescore_metrics.json").read_text())["per_complex"]}
    side = {r["name"]: r for r in
            json.loads((ART / "iface_side_split.json").read_text())["per_complex"]}
    pae = {r["name"]: r for r in json.loads((ART / "pae_metrics.json").read_text())}
    feats = ["iptm", "pdockq", "iface_plddt", "n_contacts", "contact_density",
             "delta_sasa", "receptor_side", "peptide_side", "peptide_whole",
             "iface_pae", "iface_pae_min", "pae_frac_lt10", "ipsae"]
    rows = []
    for name, r in rescore.items():
        if name not in side or name not in pae:
            continue
        row = {"receptor": r["receptor_id"], "label": r["label"]}
        for f in feats:
            row[f] = r.get(f, side[name].get(f, pae[name].get(f, np.nan)))
        rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=feats)

    print(f"\n{'=' * 74}\nC2. Within-receptor concordance (baseline removed)"
          f"\n{'=' * 74}")
    rng = np.random.default_rng(0)
    out = {}
    for task, neg in (("cognate vs own scramble", "scrambled"),
                      ("cognate vs decoy", "decoy")):
        print(f"\n   {task}")
        print(f"     {'readout':16} {'within-rec AUC':>15} {'95% CI':>18} "
              f"{'pooled AUC':>11}")
        scores = {}
        for f in feats:
            per_rec = []
            for rid, g in df.groupby("receptor"):
                c = g[g.label == "cognate"][f].to_numpy()
                n = g[g.label == neg][f].to_numpy()
                if len(c) == 0 or len(n) == 0:
                    continue
                # concordance, ties counted as half
                wins = sum((cv > nv) + 0.5 * (cv == nv) for cv in c for nv in n)
                per_rec.append(wins / (len(c) * len(n)))
            if len(per_rec) < 5:
                continue
            a = np.array(per_rec, float)
            # a readout may be informative in either direction
            oriented = a if a.mean() >= 0.5 else 1 - a
            bs = [rng.choice(oriented, len(oriented), replace=True).mean()
                  for _ in range(5000)]
            pooled = roc_auc_score((df.label == "cognate").astype(int)[
                df.label.isin(["cognate", neg])],
                df[df.label.isin(["cognate", neg])][f])
            scores[f] = {"within_auc": float(oriented.mean()),
                         "ci95": [float(np.percentile(bs, 2.5)),
                                  float(np.percentile(bs, 97.5))],
                         "pooled_auc": float(max(pooled, 1 - pooled)),
                         "per_receptor": oriented.tolist()}
        for f, s in sorted(scores.items(), key=lambda x: -x[1]["within_auc"])[:6]:
            print(f"     {f:16} {s['within_auc']:15.3f} "
                  f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]{'':4} "
                  f"{s['pooled_auc']:11.3f}")

        # is the pooled winner actually better within receptor?
        if "peptide_side" in scores and "iface_plddt" in scores:
            a = np.array(scores["peptide_side"]["per_receptor"])
            b = np.array(scores["iface_plddt"]["per_receptor"])
            d = a - b
            pv = (stats.wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0)
            print(f"     peptide_side - iface_plddt = {d.mean():+.3f}  "
                  f"paired p = {pv:.4f}")
            scores["_peptide_vs_iface"] = {"diff": float(d.mean()), "p": float(pv)}
        out[task] = scores
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "frontier_analysis.json"))
    args = ap.parse_args()

    result = {"alanine_mixed": mixed_alanine_scan(),
              "order_mixed": mixed_all_arms(),
              "replicate_curve": replicate_curve(),
              "headroom": multivariate_headroom(),
              "within_receptor": within_receptor_headroom()}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
