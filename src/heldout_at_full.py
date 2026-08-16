"""Does the held-out penalty survive at the model's intended settings?

Section 7.10 measured that complexes released after the training cutoff cost
roughly half the effect, and Section 8 turns that into advice: halve your
expectations on a target the model has not seen. But 7.10 was folded entirely at
10 sampling steps, 1 recycling pass and MSA depth 32 -- and Section 7.13 then
showed those three reductions suppress the effect three- to sevenfold. The
practical advice therefore rests on a regime the report itself discredits, and
7.13 says so explicitly: whether contamination costs a factor of two at full
settings is unknown.

This closes the 2x2:

                    reduced settings          full settings
    in-training     7.8/7.13 reduced arm      7.13 full arm
    held-out        7.10                      folded for this analysis

The full row is the one that carries the argument, and it is clean: both cells
are stock Boltz-1 at 200 sampling steps, 3 recycling passes and undiminished
alignment depth, so the only thing varying is whether the model saw the complex.

The reduced row is NOT clean and is reported with that stated. Section 7.10's
held-out arm was folded on DeCAF while the in-training arm is stock Boltz-1, so
its retention figure confounds the model with the contamination it is meant to
isolate. It is shown for continuity with the published numbers, not as evidence.

Estimators are Section 7.13's, imported rather than reimplemented, and Cohen's d
is the paired form that reproduces the report's published 1.25 and 1.52.

Usage:
    python src/heldout_at_full.py
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
METRICS = ("iptm", "iface_plddt", "receptor_side")


# Each ROW must compare like with like. Section 7.10's held-out arm is DeCAF, so
# its in-training comparator has to be DeCAF too -- pairing it against stock
# Boltz-1 gives a "retention" above 100%, which says only that DeCAF beats a
# stock model run short, not anything about contamination. The model differs
# between rows; within a row it does not.
PATHS = {
    "in_reduced": ART / "decaf_scramble_result.json",       # DeCAF, in-training
    "held_reduced": ART / "heldout_panel" / "heldout_scores.json",   # DeCAF
    "in_full": ART / "settings_confound.json",              # Boltz-1, 200/3/full
    "held_full": ART / "heldout_panel" / "heldout_scores_full.json",  # Boltz-1
}


def load(which):
    """The four cells of the design, or None where a cell is not on disk.

    Two of these files are a bare list of folds and two wrap it in a dict under
    `per_fold`. Returning the dict unnoticed gives an iteration over its keys and
    a confusing `string indices must be integers` several frames away, so the
    shape is normalised here and anything else is refused outright.
    """
    try:
        d = json.loads(PATHS[which].read_text())
    except Exception:                                              # noqa: BLE001
        return None
    if isinstance(d, dict):
        d = d.get("per_fold")
    if not isinstance(d, list) or not d or not isinstance(d[0], dict):
        return None
    return d


def paired_d(recs, metric):
    """Cohen's d on the within-receptor cognate-minus-scramble differences.

    This is the paired form. It is chosen because it reproduces the d values
    Section 7.13 published (ipTM 1.25, interface pLDDT 1.52) to two decimals,
    where a pooled-SD d gives 1.21 and 1.40 -- close enough to be mistaken for
    the same quantity and different enough to make a comparison wrong.
    """
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


def within_auc(recs, metric):
    """P(cognate outranks a decoy of its own receptor), ties at half."""
    wins = n = 0
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    for g in by.values():
        c = [x[metric] for x in g
             if x["label"] == "cognate" and x.get(metric) is not None]
        d = [x[metric] for x in g
             if x["label"] == "decoy" and x.get(metric) is not None]
        for cv in c:
            for dv in d:
                n += 1
                wins += 1.0 if cv > dv else (0.5 if cv == dv else 0.0)
    return float(wins / n) if n else None


def draw_spread(metric):
    """How far one draw of this panel moves, measured at reduced settings.

    Section 7.10 folded the held-out panel five times and the effect ranges over
    38% of its mean for ipTM, 71% for interface pLDDT and 89% for the receptor
    side. That is the yardstick any single-draw figure has to be read against,
    and it is why a lone full-settings run cannot pin a retention percentage --
    it can only say whether the penalty is still there.
    """
    try:
        d = json.loads((ART / "heldout_replicates.json").read_text())
        v = d["metrics"][metric]
    except Exception:                                              # noqa: BLE001
        return None
    per = [x["effect"] for x in v["effect_per_draw"]]
    mean = v["effect_mean"]["effect"]
    return {"n_draws": d.get("n_draws"), "mean": mean,
            "lo": min(per), "hi": max(per),
            "spread_pct": 100 * (max(per) - min(per)) / abs(mean) if mean else None}


def summarise(recs, metric):
    t = tests(recs, metric) or {}
    return {"effect": t.get("effect"), "p": t.get("p"),
            "d": paired_d(recs, metric), "auc": within_auc(recs, metric),
            "mean_rank": t.get("mean_rank"), "chance": t.get("chance"),
            "first": t.get("first"), "n_receptors": t.get("n_receptors"),
            "n_pairs": t.get("n_pairs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "heldout_at_full.json"))
    ap.add_argument("--min-folds", type=int, default=120,
                    help="refuse to draw conclusions from a partial run")
    args = ap.parse_args()

    cells = {k: load(k) for k in
             ("in_reduced", "in_full", "held_reduced", "held_full")}
    for k, v in cells.items():
        print(f"  {k:14} {'missing' if v is None else str(len(v)) + ' folds'}")
    if cells["held_full"] is None:
        raise SystemExit("\nheldout_scores_full.json not found — run:\n"
                         "  python src/heldout_panel.py --base boltz1 "
                         "--sampling-steps 200 --recycling-steps 3 "
                         "--msa-depth 0 --run-tag _full --batch-size 6")
    n_full = len(cells["held_full"])
    partial = n_full < args.min_folds
    if partial:
        print(f"\n*** held-out full run is INCOMPLETE ({n_full} folds, "
              f"{len({r['receptor_id'] for r in cells['held_full']})} receptors). "
              f"Figures below are provisional and must not be quoted. ***")

    result = {"partial": partial, "n_held_full": n_full}
    for metric in METRICS:
        print(f"\n{'=' * 78}\n{metric}\n{'=' * 78}")
        print(f"  {'cell':16}{'cog-scr':>10}{'p':>10}{'d':>7}{'AUC':>7}"
              f"{'rank':>7}{'first':>9}")
        s = {}
        for key, label in (("in_reduced", "in-train red (D)"),
                           ("held_reduced", "held-out red (D)"),
                           ("in_full", "in-train full (B)"),
                           ("held_full", "held-out full (B)")):
            if cells[key] is None:
                continue
            s[key] = summarise(cells[key], metric)
            v = s[key]
            if v["effect"] is None:
                continue
            first = (f"{v['first']}/{v['n_receptors']}"
                     if v["first"] is not None else "—")
            print(f"  {label:16}{v['effect']:>10.3f}{v['p']:>10.2g}"
                  f"{(v['d'] or float('nan')):>7.2f}{(v['auc'] or float('nan')):>7.3f}"
                  f"{(v['mean_rank'] or float('nan')):>7.2f}{first:>9}")
        result[metric] = s

        sp = draw_spread(metric)
        if sp:
            print(f"\n    one draw at reduced settings moves this effect over "
                  f"{sp['lo']:+.3f} to {sp['hi']:+.3f} "
                  f"({sp['spread_pct']:.0f}% of the {sp['n_draws']}-draw mean "
                  f"{sp['mean']:+.3f})")
            result.setdefault("draw_spread", {})[metric] = sp

        # the question: what fraction of the in-training effect survives on
        # complexes the model was not trained on, at each setting?
        print()
        for tag, a, b, clean in (("reduced", "in_reduced", "held_reduced", True),
                                 ("full", "in_full", "held_full", True)):
            if a not in s or b not in s or not s[a]["effect"]:
                continue
            for field in ("effect", "d"):
                if s[a].get(field) and s[b].get(field):
                    keep = 100 * s[b][field] / s[a][field]
                    result.setdefault("retention", {}).setdefault(metric, {})[
                        f"{tag}_{field}"] = keep
                    note = "" if clean else "   (model confound)"
                    print(f"    {tag:8} held-out retains {keep:5.1f}% of the "
                          f"in-training {field}{note}")

    print(f"\n{'=' * 78}")
    print("(D) = DeCAF at 10 steps / 1 recycling / MSA 32, which is Section 7.10's")
    print("      regime.  (B) = stock Boltz-1 at 200 / 3 / full alignment depth,")
    print("      which is Section 7.13's full arm. Each ROW compares like with like;")
    print("      the model differs between rows, not within one.")
    print()
    print("The reduced cells here are ONE draw each, so they do not reproduce Section")
    print("7.10's published retention (44% / 52% / 43%), which averages five draws.")
    print("The full-settings cell is also one draw. Given the spread above, a single")
    print("draw supports 'the penalty is still there' and does NOT support a specific")
    print("percentage. Fold two more draws before any figure from this is quoted:")
    print("  python src/heldout_panel.py --base boltz1 --sampling-steps 200 \\")
    print("      --recycling-steps 3 --msa-depth 0 --run-tag _full2 --batch-size 6")
    if partial:
        print("\nRUN INCOMPLETE — provisional numbers, do not quote.")
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
