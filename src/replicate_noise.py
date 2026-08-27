"""Run-to-run SD for every metric, on whichever replicate store you point it at.

Section 7.5 measured fold-to-fold noise on 4 receptors and 96 folds. Three
numbers came out of it -- 0.0628 ipTM, 0.1498 pDockQ, 1.917 interface pLDDT --
and the preprint's Table 5 divides effects by them, including the 8.6x
effect-to-noise ratio that reaches the abstract.

Two problems this fixes.

The SD lived in two places and only one of them was a file. ipTM came from the
study's own summary; interface pLDDT and pDockQ were recomputed on demand inside
variance_decomposition.noise_sd, from PDBs under a hardcoded
artifacts/seed_variance. Those PDBs are no longer on disk, so the 1.917 in the
paper currently cannot be re-derived from anything the repo contains -- it is
carried in a constant. This writes all three to an artifact instead.

And the 4 receptors were chosen to span the outcome range (cognate ranked
#1/#2/#3/#4). That is a reasonable spanning sample and the wrong shape for
estimating a spread: selecting the extremes first is a mechanism for inflating
it. Pointing this at a 22-receptor store answers whether it did.

Usage:
    python src/replicate_noise.py --work artifacts/seed_variance_n22 \
        --out artifacts/replicate_noise_n22.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from Bio.PDB import PDBParser  # noqa: E402
from interface_side_split import sides  # noqa: E402
from rescore_interface_metrics import interface as iface_metrics  # noqa: E402
from rescore_interface_metrics import pdockq  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Section 7.5, on 4 receptors -- what this run is compared against.
#
# pDockQ is recomputed here after all. An earlier version of this script declared
# it unrecoverable because interface_side_split.sides does not return it -- true,
# but the wrong place to look. pDockQ is a function of interface pLDDT and the
# number of contacting residue PAIRS, and rescore_interface_metrics.interface
# computes exactly that at the same 8.0 A cutoff sides uses. "Not returned by the
# function I happened to be calling" is not the same as "left no artifact behind",
# and the difference cost this figure a row marked unreproducible.
SECTION_7_5 = {"iptm": 0.0628, "pdockq": 0.1498, "iface_plddt": 1.9172}
NOT_RECOMPUTED = set()

# The 4 receptors Section 7.5 measured, chosen to span the outcome range: cognate
# ranked #1, #2, #3 and #4 among its own decoys. A widened store contains them,
# so the same run answers two different questions -- whether the original 4 are
# unrepresentative (compare the two subsets of this run) and whether this run
# reproduces the published number at all (compare the 4-receptor subset to 7.5).
SECTION_7_5_RECEPTORS = ["1YCR", "9F6S", "8KDX", "6YOO"]


def pooled_sd(by_complex):
    """Pooled within-complex SD: RMS of the per-complex SDs.

    Each complex is folded R times, so every complex gives one SD on R draws and
    the pooled figure is their quadratic mean -- the same estimator 7.5 used, so
    the two numbers are comparable rather than merely similar.
    """
    sds = [np.std(v, ddof=1) for v in by_complex.values()
           if len(v) > 1 and not np.isnan(v).any()]
    if not sds:
        return float("nan"), 0
    return float(np.sqrt(np.mean(np.square(sds)))), len(sds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "seed_variance_n22"))
    ap.add_argument("--scores", default=None,
                    help="ipTM store; defaults to <work>/seed_variance_scores.json")
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "replicate_noise.json"))
    args = ap.parse_args()

    work = Path(args.work)
    scores = Path(args.scores) if args.scores else work / "seed_variance_scores.json"

    result = {"work_dir": str(work), "per_metric": {}}

    # ipTM comes from the study's own store, which records it under `score`.
    #
    # The store is also the only thing that maps a fold to its receptor: on disk
    # a fold is pair_NNN, and pair numbering is global within a replicate but
    # carries no receptor in the name. Reading that mapping from anywhere other
    # than this run's own store is how a previous build put one molecule's
    # results under another's name, so the mapping is taken from here and a pair
    # that names two different receptors is fatal rather than resolved.
    by_iptm = {}
    receptor_of = {}
    if not scores.exists():
        raise SystemExit(f"no ipTM store at {scores}; cannot map folds to receptors")
    for r in json.loads(scores.read_text()):
        by_iptm.setdefault(r["name"], []).append(r["score"])
        prev = receptor_of.setdefault(r["name"], r["receptor_id"])
        if prev != r["receptor_id"]:
            raise SystemExit(
                f"{r['name']} maps to both {prev} and {r['receptor_id']} in {scores}")
    sd, n = pooled_sd(by_iptm)
    result["per_metric"]["iptm"] = {"pooled_sd": sd, "n_complexes": n}
    result["n_receptors"] = len(set(receptor_of.values()))

    # The structural metrics are not in any store; they come off the structures
    # the folds left behind, scored exactly as Section 7.7 scores them.
    parser = PDBParser(QUIET=True)
    by_struct = {}
    pdbs = sorted(work.rglob("*_model_0.pdb"))
    print(f"  scoring {len(pdbs)} structures")
    for pdb in pdbs:
        try:
            s = sides(parser.get_structure("x", str(pdb))[0])
        except Exception:
            continue
        if not s:
            continue
        # The complex name is the batch-relative directory, matching how the
        # ipTM store keys its rows.
        key = pdb.parent.name
        for metric, val in s.items():
            by_struct.setdefault(metric, {}).setdefault(key, []).append(val)
        # pDockQ needs the contact-pair count, which sides() does not carry.
        try:
            m = iface_metrics(parser.get_structure("y", str(pdb))[0])
        except Exception:
            m = None
        if m:
            by_struct.setdefault("pdockq", {}).setdefault(key, []).append(
                pdockq(m["iface_plddt"], m["n_contacts"]))
            by_struct.setdefault("n_contacts", {}).setdefault(key, []).append(
                float(m["n_contacts"]))

    for metric, by in sorted(by_struct.items()):
        sd, n = pooled_sd(by)
        result["per_metric"][metric] = {"pooled_sd": sd, "n_complexes": n}

    # Same estimator, restricted to the 4 receptors 7.5 used, so the selection
    # question and the reproduction question do not have to share one number.
    subset = {}
    present = [r for r in SECTION_7_5_RECEPTORS if r in set(receptor_of.values())]
    if len(present) == len(SECTION_7_5_RECEPTORS):
        keep = {k for k, v in receptor_of.items() if v in set(SECTION_7_5_RECEPTORS)}
        sd_i, n_i = pooled_sd({k: v for k, v in by_iptm.items() if k in keep})
        subset["iptm"] = {"pooled_sd": sd_i, "n_complexes": n_i}
        for metric, by in sorted(by_struct.items()):
            sd_m, n_m = pooled_sd({k: v for k, v in by.items() if k in keep})
            subset[metric] = {"pooled_sd": sd_m, "n_complexes": n_m}
        result["section_7_5_receptor_subset"] = {
            "receptors": SECTION_7_5_RECEPTORS, "per_metric": subset}
    else:
        result["section_7_5_receptor_subset"] = {
            "receptors": SECTION_7_5_RECEPTORS, "per_metric": None,
            "note": f"store holds only {present} of the 4"}

    for metric in sorted(NOT_RECOMPUTED):
        result["per_metric"].setdefault(metric, {
            "pooled_sd": None, "n_complexes": 0,
            "section_7_5": SECTION_7_5[metric],
            "note": "not returned by sides(); left at its published value"})

    result["n_structures"] = len(pdbs)
    result["n_replicates_seen"] = max(
        (len(v) for by in by_struct.values() for v in by.values()), default=0)

    print(f"\n  {'metric':<16} {'this run':>10} {'Section 7.5':>12} {'ratio':>8}  n")
    for metric, d in sorted(result["per_metric"].items()):
        ref = SECTION_7_5.get(metric)
        sd = d["pooled_sd"]
        if sd is None:
            print(f"  {metric:<16} {'not recomputed':>10} {ref:>12.4f} {'--':>8}  -")
        elif ref:
            print(f"  {metric:<16} {sd:>10.4f} {ref:>12.4f} {sd / ref:>7.2f}x  {d['n_complexes']}")
            d["section_7_5"] = ref
            d["ratio_to_7_5"] = sd / ref if ref else None
        else:
            print(f"  {metric:<16} {sd:>10.4f} {'--':>12} {'--':>8}  {d['n_complexes']}")

    if subset:
        print("\n  restricted to the 4 receptors Section 7.5 used:")
        print(f"  {'metric':<16} {'4-rec':>10} {'all-rec':>10} {'7.5':>10}")
        for metric in sorted(subset):
            allv = result["per_metric"].get(metric, {}).get("pooled_sd")
            ref = SECTION_7_5.get(metric)
            a = f"{allv:.4f}" if allv is not None else "--"
            b = f"{ref:.4f}" if ref else "--"
            print(f"  {metric:<16} {subset[metric]['pooled_sd']:>10.4f} {a:>10} {b:>10}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
