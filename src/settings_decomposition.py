"""Which of the three raised settings carries Section 7.13's effect?

Section 7.13 moved sampling steps 10 -> 200, recycling 1 -> 3 and alignment
depth 32 -> full together, and measured a three- to sevenfold larger
standardised effect on the scramble control. Raising three things at once cannot
say which of them mattered, and the report states that as an open question.

Section 7.11 supplies a prediction: at ten sampling steps only 14% of backbone
bonds are physically plausible against 99.7% at two hundred, and four separate
readout failures were traced to that. If geometry is the mechanism, sampling
steps should carry most of the effect and alignment depth little of it.

Five cells, each stock Boltz-1 on MPS, differing only in settings:

    reduced     10 steps, 1 recycling, MSA 32     Section 7.8
    sampling    200,      1,           32         this run
    alignment   10,       1,           full       this run
    recycling   10,       3,           32         this run
    full        200,      3,           full       Section 7.13

Each single-knob arm is scored against the same reduced baseline, and the share
attributed to a knob is its gain over reduced as a fraction of the full arm's
gain. Those shares need not sum to 100%: the settings interact, and a sum far
from 100 is itself the finding rather than an error.

Cohen's d is the paired form Section 7.13 published, verified against its 1.25
and 1.52 before use. Raw effects are shown too but the standardised column is
the honest one -- absolute confidence rises with sampling, so a larger raw gap
is partly scale.

Usage:
    python src/settings_decomposition.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

from settings_confound import tests  # noqa: E402

ART = REPO_ROOT / "artifacts"
WORK = ART / "settings_confound"
METRICS = ("iptm", "iface_plddt", "receptor_side")

ARMS = [
    ("reduced",   "10 / 1 / 32",    ART / "boltz1_scramble_result.json"),
    ("sampling",  "200 / 1 / 32",   WORK / "scores_samp.json"),
    ("alignment", "10 / 1 / full",  WORK / "scores_msa.json"),
    ("recycling", "10 / 3 / 32",    WORK / "scores_recyc.json"),
    ("full",      "200 / 3 / full", ART / "settings_confound.json"),
]


def load(path):
    try:
        d = json.loads(Path(path).read_text())
    except Exception:                                              # noqa: BLE001
        return None
    if isinstance(d, dict):
        d = d.get("per_fold")
    if not isinstance(d, list) or not d or not isinstance(d[0], dict):
        return None
    return d


def paired_d(recs, metric):
    """Cohen's d on within-receptor cognate-minus-scramble differences."""
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s and c[0].get(metric) is not None:
            diffs += [c[0][metric] - x[metric] for x in s
                      if x.get(metric) is not None]
    if len(diffs) < 3:
        return None
    e = np.array(diffs, float)
    sd = e.std(ddof=1)
    return float(e.mean() / sd) if sd > 1e-12 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "settings_decomposition.json"))
    ap.add_argument("--min-folds", type=int, default=60,
                    help="an arm below this is in progress, not a result. A "
                         "six-fold arm is one receptor and produced a -161%% "
                         "share on the first pass of this script.")
    args = ap.parse_args()

    cells, partial = {}, []
    for name, label, path in ARMS:
        recs = load(path)
        if recs is not None and len(recs) < args.min_folds:
            partial.append(f"{name} ({len(recs)} folds)")
            print(f"  {name:11} {label:16} {len(recs)} folds — IN PROGRESS, "
                  f"excluded")
            recs = None
        else:
            print(f"  {name:11} {label:16} "
                  f"{'missing' if recs is None else str(len(recs)) + ' folds'}")
        cells[name] = recs
    if cells["reduced"] is None or cells["full"] is None:
        raise SystemExit("both endpoints are needed")
    missing = [n for n in ("sampling", "alignment", "recycling")
               if cells[n] is None]
    if missing:
        print(f"\n*** arms not complete: {', '.join(missing)} — the share column "
              f"is partial and must not be read as a decomposition ***")
    if partial:
        print(f"    (in progress, excluded rather than reported: "
              f"{', '.join(partial)})")

    result = {}
    for metric in METRICS:
        print(f"\n{'=' * 78}\n{metric}\n{'=' * 78}")
        print(f"  {'arm':11}{'settings':16}{'cog-scr':>10}{'p':>10}{'d':>7}"
              f"{'share of full':>15}")
        base = tests(cells["reduced"], metric)
        base_d = paired_d(cells["reduced"], metric)
        fullt = tests(cells["full"], metric)
        full_d = paired_d(cells["full"], metric)
        _ = fullt["effect"] - base["effect"]   # raw span, shown per-arm below
        span_d = (full_d - base_d) if (full_d and base_d) else None
        m = {}
        for name, label, _ in ARMS:
            recs = cells[name]
            if recs is None:
                print(f"  {name:11}{label:16}{'—':>10}")
                continue
            t = tests(recs, metric)
            d = paired_d(recs, metric)
            share = ""
            if name not in ("reduced", "full") and span_d and d and base_d:
                frac = 100 * (d - base_d) / span_d
                share = f"{frac:>13.0f}%"
                m.setdefault("shares", {})[name] = frac
            elif name == "full":
                share = f"{'100%':>14}"
            m[name] = {"effect": t["effect"], "p": t["p"], "d": d,
                       "n": t.get("n_pairs")}
            print(f"  {name:11}{label:16}{t['effect']:>10.3f}{t['p']:>10.2g}"
                  f"{(d or float('nan')):>7.2f}{share}")
        if "shares" in m:
            tot = sum(m["shares"].values())
            got = len(m["shares"])
            if got == 3:
                print(f"\n    single-knob shares sum to {tot:.0f}% of the full "
                      f"gain in standardised terms")
                if abs(tot - 100) > 25:
                    print(f"    ({'super' if tot > 100 else 'sub'}-additive: the "
                          f"settings interact rather than contributing "
                          f"separately)")
            else:
                # additivity is a statement about all three knobs; one arm's
                # share is not evidence for or against it
                print(f"\n    {got} of 3 knobs measured; "
                      f"{tot:.0f}% of the full gain accounted for so far")
        result[metric] = m

    print(f"\n{'=' * 78}")
    print("Section 7.11 predicts sampling steps carry most of this: at ten steps")
    print("only 14% of backbone bonds are physically plausible, against 99.7% at")
    print("two hundred, and geometry-dependent readouts need a converged sampler.")
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
