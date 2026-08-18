"""Can an automated search over readouts beat interface pLDDT on its own?

Sections 7.6 to 7.12 picked readouts by hand and Section 7.11.3 tried one
combination, which made things worse: adding PRODIGY's dG to interface pLDDT
took leave-one-receptor-out AUC from 0.856 to 0.823. This searches the
combinations systematically instead, over folds already on disk -- no folding,
no GPU.

The search is cheap and the honesty is the expensive part. Twenty-two receptors
and seven features will always yield a "best" combination; the question is
whether it is better than what searching noise produces. Three controls:

  LORO-CV inside the search    a random split leaks, because the folds of one
                               receptor share a baseline. Every candidate is
                               scored leave-one-receptor-out.

  a searched null              the entire search is re-run on labels permuted
                               within each receptor, and the best AUC it finds
                               is recorded. That distribution is what the real
                               winner has to beat -- not 0.5, and not the
                               single-feature baseline.

  one shot at the held-out panel   the winner is applied once to the
                               full-settings held-out folds. It is a report, not
                               a tuning signal, and nothing is selected on it.

A length control is reported alongside: decoys are other receptors' cognates and
so differ in length, and a combination that merely tracks peptide length would
look like a discovery. Section 7.6 already retired one vacuous length control.

Usage:
    python src/readout_search.py --n-null 200
"""

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

from model_ensemble import loro_auc, single_auc  # noqa: E402

ART = REPO_ROOT / "artifacts"
FULL_REGIME = "boltz1@200/3/full"
FEATURES = ["iface_plddt", "iptm", "receptor_side", "peptide_side",
            "peptide_whole", "n_rec_iface", "n_pep_iface"]
MAX_SUBSET = 3


def search_panel():
    """Section 7.13's full-settings in-training folds."""
    return pd.DataFrame(json.loads((ART / "settings_confound.json").read_text())
                        ["per_fold"])


def heldout_panel():
    """Fold-wise mean of every completed full-settings held-out draw."""
    import heldout_replicates as HR
    sets = HR.draws(FULL_REGIME)
    if not sets:
        return None, 0
    avg, _ = HR.averaged(sets, FEATURES)
    return pd.DataFrame(avg), len(sets)


def peptide_lengths(df):
    """Peptide length per fold, joined from the panel that defined the pairs."""
    for p in (ART / "pdb_binders_b2_n22" / "pdb_binder_scores.json",):
        try:
            rows = json.loads(p.read_text())
        except Exception:                                          # noqa: BLE001
            continue
        m = {r["name"]: len(r["peptide"]) for r in rows if "peptide" in r}
        if "name" in df.columns:
            got = df["name"].map(m)
            if got.notna().mean() > 0.9:
                return got.astype(float)
    return None


def subsets(feats, kmax):
    for k in range(1, kmax + 1):
        yield from combinations(feats, k)


def run_search(df, feats, kmax):
    """Best subset, and the best single feature, both leave-one-receptor-out.

    Both are returned because the quantity that matters is the *gain* the search
    buys over simply taking the best readout on its own. Nulling the winner
    against chance only asks whether the features carry signal, which is not in
    doubt; nulling the gain asks whether combining them buys anything a search
    over noise would not also appear to buy.
    """
    best = (-1.0, None)
    best_single = (-1.0, None)
    for cols in subsets(feats, kmax):
        cols = list(cols)
        auc = (single_auc(df, cols[0])[1] if len(cols) == 1
               else loro_auc(df, cols)[1])
        if len(cols) == 1 and auc > best_single[0]:
            best_single = (auc, cols[0])
        if auc > best[0]:
            best = (auc, tuple(cols))
    return best, best_single


def permute_within_receptor(df, rng):
    """Move the cognate label to a random fold of the same receptor.

    This keeps the design exactly -- one cognate, three decoys and two scrambles
    per receptor -- and destroys only the association between the label and the
    scores, which is what the null has to isolate.
    """
    out = df.copy()
    lab = out["label"].to_numpy().copy()
    for r in out["receptor_id"].unique():
        idx = np.where(out["receptor_id"].to_numpy() == r)[0]
        perm = rng.permutation(idx)
        lab[perm] = df["label"].to_numpy()[idx]
    out["label"] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--max-subset", type=int, default=MAX_SUBSET)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ART / "readout_search.json"))
    args = ap.parse_args()

    df = search_panel()
    feats = [f for f in FEATURES if f in df.columns and df[f].notna().all()]
    n_sub = sum(1 for _ in subsets(feats, args.max_subset))
    print(f"search panel : {len(df)} folds, {df['receptor_id'].nunique()} receptors")
    print(f"features     : {feats}")
    print(f"search space : {n_sub} subsets of size 1-{args.max_subset}\n")

    print("single-feature baselines (within-receptor AUC)")
    singles = {}
    for f in feats:
        singles[f] = single_auc(df, f)[1]
        print(f"  {f:16} {singles[f]:.3f}")
    base_feat = max(singles, key=singles.get)
    base = singles[base_feat]
    print(f"\nbest single feature: {base_feat} at {base:.3f}")

    L = peptide_lengths(df)
    len_auc = None
    if L is not None:
        d2 = df.copy(); d2["pep_len"] = L
        len_auc = single_auc(d2, "pep_len")[1]
        print(f"length control     : peptide length alone reaches {len_auc:.3f}")
    else:
        print("length control     : peptide lengths unavailable, control skipped")

    print(f"\nsearching {n_sub} subsets …")
    (best_auc, best_cols), _ = run_search(df, feats, args.max_subset)
    print(f"  best: {best_auc:.3f}  {' + '.join(best_cols)}")
    print(f"  gain over best single feature: {best_auc - base:+.3f}")

    print(f"\nsearched null: re-running the whole search on {args.n_null} "
          f"label permutations …")
    rng = np.random.default_rng(args.seed)
    null, null_gain = [], []
    for i in range(args.n_null):
        (nb, _), (ns, _) = run_search(permute_within_receptor(df, rng), feats,
                                      args.max_subset)
        null.append(nb)
        null_gain.append(nb - ns)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{args.n_null}  null best {np.mean(null):.3f}"
                  f"  null gain {np.mean(null_gain):+.3f}", flush=True)
    null, null_gain = np.array(null), np.array(null_gain)
    p = float((null >= best_auc).mean())
    gain = best_auc - base
    p_gain = float((null_gain >= gain).mean())
    print(f"\n  null best-of-search: mean {null.mean():.3f}, "
          f"95th pct {np.percentile(null, 95):.3f}, max {null.max():.3f}")
    print(f"  p(search noise >= {best_auc:.3f})            = {p:.3f}")
    print(f"  null GAIN over best single: mean {null_gain.mean():+.3f}, "
          f"95th pct {np.percentile(null_gain, 95):+.3f}")
    print(f"  p(search noise gains >= {gain:+.3f})        = {p_gain:.3f}"
          f"   <- the number that matters")

    result = {"features": feats, "n_subsets": n_sub, "singles": singles,
              "best_single": base_feat, "best_single_auc": base,
              "length_auc": len_auc,
              "best": {"auc": best_auc, "cols": list(best_cols)},
              "gain_over_best_single": gain,
              "null": {"n": int(args.n_null), "mean": float(null.mean()),
                       "p95": float(np.percentile(null, 95)),
                       "max": float(null.max()), "p_value": p,
                       "gain_mean": float(null_gain.mean()),
                       "gain_p95": float(np.percentile(null_gain, 95)),
                       "gain_p_value": p_gain}}

    ho, n_draws = heldout_panel()
    if ho is not None and all(c in ho.columns for c in best_cols):
        print(f"\nheld-out panel ({n_draws} full-settings draw(s), {len(ho)} folds)"
              f" — applied ONCE, not tuned on")
        ho_best = (single_auc(ho, best_cols[0])[1] if len(best_cols) == 1
                   else loro_auc(ho, list(best_cols))[1])
        ho_base = single_auc(ho, base_feat)[1]
        print(f"  {' + '.join(best_cols):40} {ho_best:.3f}")
        print(f"  {base_feat + ' (single baseline)':40} {ho_base:.3f}")
        print(f"  gain carried over: {ho_best - ho_base:+.3f}")
        result["heldout"] = {"n_draws": n_draws, "best": ho_best,
                             "baseline": ho_base, "gain": ho_best - ho_base}

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
