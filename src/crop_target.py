"""Crop a target construct to the region its designs actually bind.

Cofolding cost is roughly cubic in chain length -- measured today, a 1.9x longer
complex cost 8x the time, which is N^3 to within the noise. Most of a large
target is therefore paid for and never touched: EGFR's construct is 621 residues
and its designs contact a few dozen.

Cropping is only sound when every design against that target binds the same
neighbourhood. Where designs bind different faces, a per-design crop would give
each design a different target and destroy within-target ranking, and a crop to
the union of epitopes would remove nothing. This module measures which case a
target is in before cropping it, and refuses the ones it cannot help.

The four targets used in the first scramble run -- RBX1, TrkA, PD-L1, BHRF1 --
are 101 to 157 residues and are already too small to crop usefully. This exists
for the large ones.

Usage:
    python src/crop_target.py --pad 12
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts" / "anthropic_binder"
warnings.filterwarnings("ignore")


def constructs():
    txt = (ART / "target_constructs.fasta").read_text()
    out, name = {}, None
    for line in txt.splitlines():
        if line.startswith(">"):
            name = line
        elif name:
            out[name] = out.get(name, "") + line.strip()
    keyed = {}
    for header, seq in out.items():
        if "target=" not in header or "nucleic" in header:
            continue
        t = header.split("target=")[1].split()[0]
        copies = int(header.split("copies=")[1].split()[0]) if "copies=" in header else 1
        keyed.setdefault(t, {"seq": seq, "copies": copies})
    return keyed


def epitope_positions(s):
    """Residue numbers from 'B:THR18;B:TYR19;...', or [] if unparseable."""
    if not isinstance(s, str):
        return []
    return [int(m) for m in re.findall(r"[A-Z]:[A-Z]{3}(\d+)", s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=12,
                    help="residues kept either side of the epitope span")
    ap.add_argument("--max-frac", type=float, default=0.75,
                    help="refuse a crop that keeps more than this of the target")
    ap.add_argument("--out", default=str(ART / "cropped_targets.json"))
    args = ap.parse_args()

    cons = constructs()
    d = pd.read_csv(ART / "design_summary.csv", low_memory=False)
    d = d[d["binder_final"].isin([True, False])]

    print(f"  {'target':14}{'full':>6}{'epitope span':>14}{'crop':>7}{'kept':>7}"
          f"{'predicted':>11}  verdict")
    out = {}
    for t, g in d.groupby("target"):
        if t not in cons:
            continue
        seq, copies = cons[t]["seq"], cons[t]["copies"]
        pos = [p for s in g["epitope_residues"] for p in epitope_positions(s)]
        pos = [p for p in pos if 1 <= p <= len(seq)]
        if not pos:
            print(f"  {t:14}{len(seq):>6}{'no epitopes':>14}")
            continue
        lo, hi = max(1, min(pos) - args.pad), min(len(seq), max(pos) + args.pad)
        crop = hi - lo + 1
        kept = crop / len(seq)
        # a union spanning most of the chain means the designs do not agree on a
        # face; cropping it would remove nothing and pretend otherwise
        if kept > args.max_frac:
            verdict = "refused: designs bind too broadly"
        elif copies > 1:
            verdict = "refused: oligomeric, crop changes the interface"
        elif crop >= len(seq) - 10:
            verdict = "refused: nothing to remove"
        else:
            verdict = "OK"
            out[t] = {"lo": lo, "hi": hi, "seq": seq[lo - 1:hi],
                      "full_len": len(seq), "crop_len": crop,
                      "n_epitope_residues": len(set(pos))}
        binder = g["binder_length"].mean()
        speed = ((len(seq) + binder) / (crop + binder)) ** 3
        print(f"  {t:14}{len(seq):>6}{f'{min(pos)}-{max(pos)}':>14}{crop:>7}"
              f"{kept:>6.0%}{speed:>10.1f}x  {verdict}")

    # a crop is only valid if it still contains every epitope residue it was
    # built from; checked rather than assumed
    for t, c in out.items():
        pos = {p for s in d[d["target"] == t]["epitope_residues"]
               for p in epitope_positions(s) if 1 <= p <= c["full_len"]}
        missing = [p for p in pos if not (c["lo"] <= p <= c["hi"])]
        assert not missing, f"{t}: crop drops epitope residues {missing[:5]}"
    print(f"\n  {len(out)} target(s) croppable; every crop verified to retain "
          f"all of its epitope residues")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
