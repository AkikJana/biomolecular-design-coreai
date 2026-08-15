"""What would this actually be worth in a screen?

Every number in Section 7 is a hypothesis test: does the cognate beat its own
scramble, does it rank above chance among its decoys. Those answer whether a
signal exists. They do not answer the question a screener asks, which is: if I
fold a library against my target and take the top slice, how much better than
random is what I get?

Enrichment factor answers that. EF at x% is the hit rate in the top x% of the
ranked list divided by the overall hit rate. EF = 1 is random; the ceiling is
1/base_rate, reached when every true binder sorts to the top.

Two regimes are reported, because they are not the same task:

  pooled          one ranked list across every receptor, no per-target
                  calibration. This is the regime where an absolute threshold
                  would be used, and where Section 7.10's receptor-variance
                  shares (0.33-0.61) predict trouble.
  within-receptor scores z-scored per receptor first. This is a real screen:
                  one target, many candidates, ranked against each other.

The honest limitation is the panel: six folds per receptor, so the finest
per-target slice available is one in six. EF at small x is therefore computed on
the pooled list, and the per-target result is reported as top-1 accuracy, which
is what a six-candidate panel can actually support.

Usage:
    python src/screening_enrichment.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
warnings.filterwarnings("ignore")

FLIP = {"iface_pae", "mpae"}
FRACTIONS = (0.05, 0.10, 0.20)


def load(arm):
    if arm == "boltz2":
        side = json.loads((ART / "iface_side_split.json").read_text())["per_complex"]
        sc = {p["name"]: p["score"] for p in json.loads(
            (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json").read_text())}
        rows = [{**r, "iptm": sc.get(r["name"], np.nan)} for r in side]
        pae = {r["name"]: r for r in json.loads((ART / "pae_metrics.json").read_text())}
        for r in rows:
            p = pae.get(r.get("name"), {})
            for k in ("iface_pae", "iface_pae_min", "pae_frac_lt10"):
                if k in p:
                    r["mpae" if k == "iface_pae_min" else k] = p[k]
        return pd.DataFrame(rows)
    if arm == "heldout":
        return pd.DataFrame(json.loads(
            (ART / "heldout_panel" / "heldout_scores.json").read_text()))
    return pd.DataFrame(json.loads(
        (ART / f"{arm}_scramble_result.json").read_text())["per_fold"])


def _ef(hits, base, k):
    return float(hits[:k].mean() / base)


def enrichment(df, metric, within, n_boot=2000, seed=0):
    """EF at each fraction, with a bootstrap CI and a permutation p-value.

    At 132 folds the top 5% is seven items, so EF@5% moves in steps of about
    0.86 -- one more cognate in the slice. A point estimate alone would invite
    far more confidence than seven observations support, so every EF is reported
    with the interval and against a null built by shuffling the labels.
    """
    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return None
    d["v"] = -d[metric] if metric in FLIP else d[metric]
    if within:
        d["v"] = d.groupby("receptor_id")["v"].transform(
            lambda x: (x - x.mean()) / (x.std() or 1))
    d["hit"] = (d["label"] == "cognate").astype(int)
    d = d.sort_values("v", ascending=False)
    hits = d["hit"].to_numpy()
    base = hits.mean()
    rng = np.random.default_rng(seed)
    out = {"n": len(d), "base_rate": float(base), "ceiling": float(1 / base)}
    for f in FRACTIONS:
        k = max(1, int(round(f * len(d))))
        obs = _ef(hits, base, k)
        # Bootstrap over receptors, the unit of independence. The resampled
        # folds must be re-sorted by score before the top slice is taken --
        # concatenating the per-receptor hit arrays and slicing directly would
        # take an arbitrary subset rather than the highest-scoring one, and
        # produced an interval that sat below the point estimate.
        groups = [g[["v", "hit"]].to_numpy() for _, g in d.groupby("receptor_id")]
        boot = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            arr = np.concatenate([groups[i] for i in pick])
            arr = arr[np.argsort(-arr[:, 0], kind="stable")]
            hb = arr[:, 1]
            if hb.mean() > 0:
                boot.append(_ef(hb, hb.mean(), max(1, int(round(f * len(hb))))))
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        null = [_ef(rng.permutation(hits), base, k) for _ in range(n_boot)]
        out[f"ef@{int(f * 100)}"] = obs
        out[f"ef@{int(f * 100)}_ci"] = [float(lo), float(hi)]
        out[f"ef@{int(f * 100)}_p"] = float((np.array(null) >= obs).mean())
    # top-1 per receptor: the only per-target slice six folds can support
    firsts = 0
    for _, g in d.groupby("receptor_id"):
        if not g.empty and g.sort_values("v", ascending=False).iloc[0]["hit"] == 1:
            firsts += 1
    out["top1_receptors"] = firsts
    out["n_receptors"] = int(d["receptor_id"].nunique())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "screening_enrichment.json"))
    args = ap.parse_args()

    arms = [("boltz2", "Boltz-2 (in-training panel)"),
            ("decaf", "DeCAF (in-training panel)"),
            ("heldout", "DeCAF (held-out panel)")]
    metrics = ["iptm", "iface_plddt", "receptor_side", "mpae", "pae_frac_lt10"]
    result = {}
    for arm, label in arms:
        try:
            df = load(arm)
        except Exception as exc:
            print(f"{label}: unavailable ({exc})")
            continue
        have = [m for m in metrics if m in df.columns and df[m].notna().any()]
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        for within in (False, True):
            tag = "within-receptor" if within else "pooled"
            print(f"\n  {tag}")
            print(f"    {'metric':15} {'EF@5%':>7} {'95% CI':>16} {'perm p':>8}"
                  f" {'EF@10%':>7} {'ceiling':>8} {'top-1':>10}")
            for m in have:
                e = enrichment(df, m, within)
                if not e:
                    continue
                result.setdefault(arm, {}).setdefault(tag, {})[m] = e
                ci = e["ef@5_ci"]
                print(f"    {m:15} {e['ef@5']:7.2f} "
                      f"[{ci[0]:6.2f},{ci[1]:6.2f}] {e['ef@5_p']:8.3f}"
                      f" {e['ef@10']:7.2f} {e['ceiling']:8.2f} "
                      f"{e['top1_receptors']:4d}/{e['n_receptors']:<5d}")

    print(f"\n{'=' * 78}")
    print("EF = 1.00 is random. The ceiling is 1/base_rate, reached only if every")
    print("cognate sorts above every decoy and scramble. Top-1 counts receptors")
    print("whose cognate is the single highest-scoring of its six folds.")
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
