"""Regularised linear baseline for the binder-ranking task.

The neural surrogate reached held-out Spearman -0.034 (n=85) while fitting its
training set exactly. That leaves two explanations unseparated: the model is the
wrong shape for the data, or the reference ranking carries no learnable signal.
A ridge model plus a permutation null separates them.

  * Ridge beats the null  -> there is signal; the neural surrogate is the problem.
  * Ridge sits inside the null -> the reference ranking is (mostly) noise, and no
    amount of model work on this task will help.

Feature design matters here. The candidates are single-point mutants of one
15-mer, so a full position x amino-acid one-hot cannot generalise: every held-out
mutant carries a (position, residue) pair that appears in no training row, and
the model can only memorise. The primary featurisation is therefore **additive
main effects** -- which position was mutated, which residue it became, and the
physicochemical deltas -- exactly the additive model used for deep mutational
scanning data. The one-hot version is fitted too, as a demonstration of the
memorisation failure rather than a serious contender.

Reuses predictions already folded by distill_against_reference.py, so it needs no
Boltz runs. Uses the same split (seed, test fraction) so numbers are comparable.

Usage:
    python src/ridge_baseline.py --results-glob 'artifacts/distill_n258/batch_*/boltz_results_inputs'
"""

import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from benchmark_surrogate_vs_reference import spearman as torch_spearman
from distill_against_reference import MultiDirPredictFn
from run_reference_benchmark import DEFAULT_BINDER, DEFAULT_TARGET, REPO_ROOT, make_binder_variants

import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy, formal charge at pH 7, van der Waals volume (A^3),
# and a binary polarity flag. Enough to express "this substitution changes the
# chemistry a lot" without hand-tuning per residue.
HYDROPATHY = dict(zip(ALPHABET, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                                 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CHARGE = dict(zip(ALPHABET, [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
VOLUME = dict(zip(ALPHABET, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6,
                             166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0,
                             227.8, 193.6]))
POLAR = dict(zip(ALPHABET, [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]))


def additive_features(binder: str, wt: str) -> np.ndarray:
    """Main effects: mutated position, resulting residue, physicochemical deltas.

    Generalises to unseen (position, residue) pairs because position and residue
    enter separately rather than as a joint indicator.
    """
    pos_onehot = np.zeros(len(wt))
    aa_to = np.zeros(len(ALPHABET))
    aa_from = np.zeros(len(ALPHABET))
    deltas = np.zeros(4)
    for i, (w, b) in enumerate(zip(wt, binder)):
        if w == b:
            continue
        pos_onehot[i] = 1.0
        aa_to[ALPHABET.index(b)] = 1.0
        aa_from[ALPHABET.index(w)] = 1.0
        deltas = np.array([
            HYDROPATHY[b] - HYDROPATHY[w],
            CHARGE[b] - CHARGE[w],
            VOLUME[b] - VOLUME[w],
            POLAR[b] - POLAR[w],
        ])
    return np.concatenate([pos_onehot, aa_to, aa_from, deltas])


def onehot_features(binder: str) -> np.ndarray:
    """Full position x residue indicator. Cannot generalise on a single-mutant scan."""
    x = np.zeros(len(binder) * len(ALPHABET))
    for i, aa in enumerate(binder):
        x[i * len(ALPHABET) + ALPHABET.index(aa)] = 1.0
    return x


def spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    return torch_spearman(torch.tensor(a, dtype=torch.float64),
                          torch.tensor(b, dtype=torch.float64))


def fisher_ci(rho: float, n: int):
    if abs(rho) >= 1 or n < 4:
        return float("nan"), float("nan")
    z, se = math.atanh(rho), 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def fit_and_score(x_train, y_train, x_test, y_test):
    scaler = StandardScaler().fit(x_train)
    model = RidgeCV(alphas=np.logspace(-3, 4, 40)).fit(scaler.transform(x_train), y_train)
    train_rho = spearman_np(y_train, model.predict(scaler.transform(x_train)))
    test_rho = spearman_np(y_test, model.predict(scaler.transform(x_test)))
    return train_rho, test_rho, float(model.alpha_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob",
                    default=str(REPO_ROOT / "artifacts/distill_n258/batch_*/boltz_results_inputs"))
    ap.add_argument("--num-binders", type=int, default=258)
    ap.add_argument("--test-frac", type=float, default=0.33)
    ap.add_argument("--rank-key", default="iptm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=30,
                    help="repeated random splits; a single split is too noisy to trust "
                         "(sd ~0.10 across splits on this data)")
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "ridge_baseline.json"))
    args = ap.parse_args()

    wt = DEFAULT_BINDER
    binders = make_binder_variants(wt, args.num_binders)
    names = [f"pair_{i:03d}" for i in range(len(binders))]
    name_for = dict(zip(binders, names))

    dirs = sorted(glob.glob(args.results_glob))
    if not dirs:
        raise SystemExit(f"no reference directories matched {args.results_glob}")
    predict_fn = MultiDirPredictFn([Path(d) for d in dirs], lambda t, b: name_for[b])
    y = np.array([predict_fn(DEFAULT_TARGET, b)[args.rank_key].reshape(-1)[0].item()
                  for b in binders])
    print(f"reference {args.rank_key}: n={len(y)} min={y.min():.4f} max={y.max():.4f} "
          f"std={y.std():.4f}")

    # Same split as distill_against_reference.py so the numbers are comparable.
    rng = random.Random(args.seed)
    order = list(range(len(binders)))
    rng.shuffle(order)
    n_test = max(2, int(round(len(binders) * args.test_frac)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    print(f"split: {len(train_idx)} train / {len(test_idx)} held-out "
          f"(same seed/fraction as the neural run)")

    results = {}
    for label, featurise in (("additive main effects", lambda b: additive_features(b, wt)),
                             ("full position x residue one-hot", onehot_features)):
        x = np.stack([featurise(b) for b in binders])
        tr, te, alpha = fit_and_score(x[train_idx], y[train_idx], x[test_idx], y[test_idx])
        lo, hi = fisher_ci(te, len(test_idx))
        results[label] = {"train_rho": tr, "heldout_rho": te, "alpha": alpha,
                          "ci95": [lo, hi], "n_features": int(x.shape[1])}
        print(f"\n{label}  ({x.shape[1]} features, alpha={alpha:.3g})")
        print(f"  train rho    {tr:+.3f}")
        print(f"  held-out rho {te:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    # Empirical null: refit on shuffled training labels. If the real held-out rho
    # sits inside this distribution, the ranking carries no learnable signal.
    x = np.stack([additive_features(b, wt) for b in binders])
    null = []
    perm_rng = np.random.default_rng(args.seed)
    y_train = y[train_idx].copy()
    for _ in range(args.permutations):
        shuffled = perm_rng.permutation(y_train)
        _, te_null, _ = fit_and_score(x[train_idx], shuffled, x[test_idx], y[test_idx])
        null.append(te_null)
    null = np.array(null)
    real = results["additive main effects"]["heldout_rho"]
    p_emp = float((np.abs(null) >= abs(real)).mean())

    print(f"\npermutation null ({args.permutations} label shuffles, additive features)")
    print(f"  null held-out rho: mean {null.mean():+.3f}, sd {null.std():.3f}, "
          f"95% range [{np.percentile(null, 2.5):+.3f}, {np.percentile(null, 97.5):+.3f}]")
    print(f"  observed {real:+.3f}  ->  empirical p = {p_emp:.3f}")
    print(f"  {'SIGNAL' if p_emp < 0.05 else 'NO SIGNAL'}: the observed correlation is "
          f"{'outside' if p_emp < 0.05 else 'inside'} the null distribution")

    # Repeated splits against a matched null. A single split is unreliable here:
    # held-out rho ranged 0.04-0.22 across six seeds on identical data, so any one
    # number over- or under-states. Comparing the mean over many splits with the
    # mean over the same procedure on shuffled labels is the test that holds up.
    def repeated(labels, reps, offset):
        out = []
        for s in range(reps):
            r = random.Random(1000 + offset + s)
            o = list(range(len(binders)))
            r.shuffle(o)
            nt = int(round(len(binders) * args.test_frac))
            te, tr = o[:nt], o[nt:]
            _, rho, _ = fit_and_score(x[tr], labels[tr], x[te], labels[te])
            out.append(rho)
        return np.array(out)

    real = repeated(y, args.repeats, 0)
    pr = np.random.default_rng(args.seed)
    null_rep = np.concatenate(
        [repeated(pr.permutation(y), max(1, args.repeats // 6), 100 * (k + 1)) for k in range(6)]
    )
    diff = real.mean() - null_rep.mean()
    se = math.sqrt(real.var() / len(real) + null_rep.var() / len(null_rep))
    z = diff / se if se > 0 else float("nan")
    print(f"\nrepeated splits ({args.repeats} real vs {len(null_rep)} shuffled-label)")
    print(f"  real  mean rho {real.mean():+.3f} (sd {real.std():.3f}), "
          f"{int((real > 0).sum())}/{len(real)} positive")
    print(f"  null  mean rho {null_rep.mean():+.3f} (sd {null_rep.std():.3f})")
    print(f"  difference {diff:+.3f}  z = {z:.2f}  "
          f"-> {'signal' if abs(z) > 2 else 'no signal'}")

    summary = {"repeated_splits": {"repeats": int(len(real)),
                                   "real_mean": float(real.mean()),
                                   "real_sd": float(real.std()),
                                   "null_mean": float(null_rep.mean()),
                                   "null_sd": float(null_rep.std()),
                                   "z": float(z)},
               "n": len(binders), "n_train": len(train_idx), "n_test": len(test_idx),
               "rank_key": args.rank_key, "models": results,
               "permutation_null": {"n": args.permutations, "mean": float(null.mean()),
                                    "sd": float(null.std()),
                                    "p_empirical": p_emp},
               "neural_surrogate_heldout_rho": -0.034}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {args.out}")


if __name__ == "__main__":
    main()
