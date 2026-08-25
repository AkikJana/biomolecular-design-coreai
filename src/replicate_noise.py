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

REPO_ROOT = Path(__file__).resolve().parents[1]

# Section 7.5, on 4 receptors -- what this run is compared against.
#
# pDockQ is listed for completeness and is not recomputed here. It is not one of
# the quantities interface_side_split.sides returns, so the 0.1498 came from some
# other path that left no artifact behind; variance_decomposition.noise_sd would
# raise KeyError if asked for it. That row of Table 5 stays at its published
# value, and it is the row the dissertation already discounts -- pDockQ is
# computed from CB-CB distances on backbones that are not connected.
SECTION_7_5 = {"iptm": 0.0628, "pdockq": 0.1498, "iface_plddt": 1.9172}
NOT_RECOMPUTED = {"pdockq"}


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
    by_iptm = {}
    if scores.exists():
        for r in json.loads(scores.read_text()):
            by_iptm.setdefault(r["name"], []).append(r["score"])
        sd, n = pooled_sd(by_iptm)
        result["per_metric"]["iptm"] = {"pooled_sd": sd, "n_complexes": n}
    else:
        print(f"  no ipTM store at {scores}")

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

    for metric, by in sorted(by_struct.items()):
        sd, n = pooled_sd(by)
        result["per_metric"][metric] = {"pooled_sd": sd, "n_complexes": n}

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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
