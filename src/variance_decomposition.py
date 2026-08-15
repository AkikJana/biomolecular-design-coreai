"""How much binding signal is there to find, at most?

Every test so far collapses six folds per receptor into one integer rank. That
discards magnitude, which is why the results sit near p = 0.03 at n = 22. Model
the score instead:

    s(R,P) = mu + alpha_R + beta*cognate + gamma*scramble + delta_RP + eps

  alpha_R   receptor baseline -- the nuisance the within-receptor rank test was
            invented to remove. A random effect removes it without discarding
            magnitudes.
  delta_RP  the receptor-peptide interaction. **This is the binding signal.**
  eps       fold-to-fold sampling noise, measured directly from replicates.

The ratio that matters is sigma^2_delta / sigma^2_eps. It is an
information-theoretic ceiling on *any* score computed from this model:

  * sigma^2_delta small           no metric can rank well, and replicates will
                                  not help. Stop looking for better readouts.
  * sigma^2_delta large but
    sigma^2_eps larger            the signal is there and averaging recovers it.
                                  Replicates are exactly the fix.

Fitting one model per arm also answers a question the rank tests cannot: does
few-step distillation raise the signal, lower the noise, or both?

Usage:
    python src/variance_decomposition.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_arm(name):
    """(dataframe, metric list) for one model arm."""
    if name == "boltz2":
        recs = json.loads((REPO_ROOT / "artifacts" / "iface_side_split.json")
                          .read_text())["per_complex"]
        scores = {p["name"]: p["score"] for p in json.loads(
            (REPO_ROOT / "artifacts" / "pdb_binders_b2_n22" /
             "pdb_binder_scores.json").read_text())}
        for r in recs:
            r["iptm"] = scores.get(r["name"], np.nan)
    else:
        recs = json.loads((REPO_ROOT / "artifacts" /
                           f"{name}_scramble_result.json").read_text())["per_fold"]
    rows = [{"receptor": r["receptor_id"], "label": r["label"],
             "iptm": r.get("iptm", np.nan),
             "iface_plddt": r.get("iface_plddt", np.nan)} for r in recs]
    return pd.DataFrame(rows)


def noise_sd(metric):
    """Fold-to-fold SD, measured on identical re-runs (Section 7.5).

    The replicate store records ipTM under `score` and carries no structural
    metric, so interface pLDDT is recomputed from the replicate structures that
    the study left on disk.
    """
    rec = json.loads((REPO_ROOT / "artifacts" / "seed_variance" /
                      "seed_variance_scores.json").read_text())
    by = {}
    if metric == "iptm":
        for r in rec:
            by.setdefault(r["name"], []).append(r["score"])
    else:
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT / "src"))
        from Bio.PDB import PDBParser
        from interface_side_split import sides
        parser = PDBParser(QUIET=True)
        for pdb in sorted((REPO_ROOT / "artifacts" / "seed_variance").rglob(
                "*_model_0.pdb")):
            try:
                s = sides(parser.get_structure("x", str(pdb))[0])
            except Exception:
                continue
            if s:
                by.setdefault(pdb.parent.name, []).append(s[metric])
    sds = [np.std(v, ddof=1) for v in by.values()
           if len(v) > 1 and not np.isnan(v).any()]
    return float(np.sqrt(np.mean(np.square(sds)))) if sds else float("nan")


def decompose(df, metric, eps_sd):
    """Mixed model with receptor as a random effect; return variance parts."""
    d = df.dropna(subset=[metric]).copy()
    if d.empty or d["receptor"].nunique() < 5:
        return None
    d["cognate"] = (d["label"] == "cognate").astype(float)
    d["scramble"] = (d["label"] == "scrambled").astype(float)
    try:
        fit = smf.mixedlm(f"{metric} ~ cognate + scramble", d,
                          groups=d["receptor"]).fit(reml=True, method="lbfgs")
    except Exception as exc:
        print(f"    fit failed: {str(exc)[:60]}")
        return None

    var_receptor = float(fit.cov_re.iloc[0, 0])
    var_resid = float(fit.scale)
    # Residual variance is interaction plus sampling noise; the replicate study
    # measures the noise term directly, so the remainder is the interaction --
    # the part a scoring function could in principle exploit.
    var_eps = eps_sd ** 2 if not np.isnan(eps_sd) else np.nan
    var_delta = max(var_resid - var_eps, 0.0) if not np.isnan(var_eps) else np.nan
    return {"n": int(len(d)), "beta_cognate": float(fit.params.get("cognate", np.nan)),
            "p_cognate": float(fit.pvalues.get("cognate", np.nan)),
            "beta_scramble": float(fit.params.get("scramble", np.nan)),
            "var_receptor": var_receptor, "var_residual": var_resid,
            "var_noise": var_eps, "var_interaction": var_delta,
            "signal_to_noise": (var_delta / var_eps
                                if var_eps and not np.isnan(var_eps) and var_eps > 0
                                else float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" /
                                         "variance_decomposition.json"))
    args = ap.parse_args()

    eps = {m: noise_sd(m) for m in ("iptm", "iface_plddt")}
    # DeCAF's noise was measured separately and is NOT the same as its teacher's:
    # few-step sampling takes larger jumps and turns out to be ~1.6x noisier.
    # Borrowing the Boltz-2 term inflated DeCAF's signal-to-noise by ~2.8x.
    decaf_rep = REPO_ROOT / "artifacts" / "decaf_replicate_result.json"
    eps_by_arm = {"boltz2": eps, "boltz1": eps, "decaf": dict(eps)}
    if decaf_rep.exists():
        rep = json.loads(decaf_rep.read_text())["summary"]
        for m in eps:
            if m in rep:
                eps_by_arm["decaf"][m] = rep[m]["pooled_sd"]

    print("Fold-to-fold noise SD, measured on identical re-runs:")
    print(f"  {'metric':14} {'Boltz-2 / Boltz-1':>18} {'DeCAF':>10}")
    for m in eps:
        print(f"  {m:14} {eps[m]:18.4f} {eps_by_arm['decaf'][m]:10.4f}")
    print("\n  Boltz-1 has no replicate study of its own and borrows the Boltz-2")
    print("  term; its row is therefore an assumption, not a measurement.")

    out = {}
    for arm in ("boltz2", "boltz1", "decaf"):
        try:
            df = load_arm(arm)
        except Exception as exc:
            print(f"\n{arm}: unavailable ({str(exc)[:50]})")
            continue
        print(f"\n{'=' * 70}\n{arm}  ({len(df)} folds, "
              f"{df['receptor'].nunique()} receptors)\n{'=' * 70}")
        print(f"{'metric':14} {'beta_cog':>9} {'p':>9} {'var_recep':>10}"
              f" {'var_inter':>10} {'var_noise':>10} {'signal/noise':>13}")
        for m in ("iptm", "iface_plddt"):
            r = decompose(df, m, eps_by_arm[arm][m])
            if not r:
                continue
            out.setdefault(arm, {})[m] = r
            print(f"{m:14} {r['beta_cognate']:9.4f} {r['p_cognate']:9.5f}"
                  f" {r['var_receptor']:10.4f} {r['var_interaction']:10.4f}"
                  f" {r['var_noise']:10.4f} {r['signal_to_noise']:13.2f}")

    print(f"\n{'=' * 70}\nCeiling\n{'=' * 70}")
    for arm, mets in out.items():
        for m, r in mets.items():
            snr = r["signal_to_noise"]
            if np.isnan(snr):
                continue
            # a single fold's discriminability, and what averaging k folds buys
            d1 = np.sqrt(snr)
            verdict = ("noise-limited -- replicate averaging recovers it"
                       if snr < 1 else
                       "signal-limited -- averaging helps little")
            print(f"  {arm:7} {m:14} signal/noise {snr:6.2f}  "
                  f"d(1 fold) {d1:5.2f}  -> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"noise_sd_by_arm": eps_by_arm, "arms": out}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
