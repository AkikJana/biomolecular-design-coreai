"""How much does ipTM move between identical runs?

Every fold in the binder benchmarks was unseeded: Boltz's `--seed` defaults to
None, and the benchmark's own `--seed` controls only pair construction (which
decoys, which scrambles), not the diffusion sampler. So the reported ipTM values
are single draws from a distribution whose width was never measured.

That matters because the load-bearing claim -- cognates do not outscore their own
scrambles -- rests on a mean difference of **+0.0128**. If run-to-run SD is of
that order, the comparison is uninformative rather than evidence of
order-insensitivity. The cognate-decoy gap (+0.070) is larger but still worth
checking per complex.

Design: 4 receptors spanning the full range of outcomes in the n=22 run (cognate
ranked #1, #2, #3 and #4 among its decoys), all 6 of each receptor's complexes,
folded R times at the *same settings as the original run* -- so what is measured
is the variance of the numbers actually reported, not of some other
configuration.

Two quantities come out:

  per-complex SD   pooled across 24 complexes; compare against the effect sizes
  rank stability   does the cognate's rank among its own competitors flip
                   between replicates? This is what the conclusions rest on,
                   and it is not recoverable from the SD alone.

Usage:
    python src/seed_variance_study.py --replicates 3
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from boltz2_predict import BoltzCliPredictFn
from run_reference_benchmark import REPO_ROOT, run_boltz

# One receptor from each observed outcome in the n=22 powered run, so the
# variance estimate is not drawn only from cases that happened to work.
DEFAULT_RECEPTORS = ["1YCR", "9F6S", "8KDX", "6YOO"]   # cognate ranked #1/#2/#3/#4


def load_pairs(scores_path, receptors):
    pairs = json.loads(Path(scores_path).read_text())
    sel = [p for p in pairs if p["receptor_id"] in receptors]
    missing = set(receptors) - {p["receptor_id"] for p in sel}
    if missing:
        raise SystemExit(f"receptors not in {scores_path}: {sorted(missing)}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--receptors", nargs="*", default=DEFAULT_RECEPTORS)
    ap.add_argument("--source", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--msa-dir", default=str(REPO_ROOT / "artifacts" /
                                             "pdb_binders_b2_n22" / "msa_cache"))
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "seed_variance"))
    # The summary path used to be hardcoded, so a second run with a different
    # --receptors set wrote its SD over the 4-receptor one it was meant to be
    # compared against. Default is unchanged; widened runs pass their own.
    ap.add_argument("--summary", default=str(REPO_ROOT / "artifacts" /
                                             "seed_variance_summary.json"))
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    sel = load_pairs(args.source, args.receptors)
    print(f"{len(sel)} complexes x {args.replicates} replicates = "
          f"{len(sel) * args.replicates} folds")
    for rid in args.receptors:
        g = [p for p in sel if p["receptor_id"] == rid]
        print(f"  {rid}: {len(g)} complexes "
              f"({sum(x['label'] == 'cognate' for x in g)}C/"
              f"{sum(x['label'] == 'decoy' for x in g)}D/"
              f"{sum(x['label'] == 'scrambled' for x in g)}S)")

    out_path = work / "seed_variance_scores.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["name"], r["replicate"]) for r in records}

    for rep in range(args.replicates):
        todo = [p for p in sel if (p["name"], rep) not in done]
        if not todo:
            print(f"replicate {rep}: already complete")
            continue
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            bdir = work / f"rep{rep}_batch{start // args.batch_size:02d}"
            idir = bdir / "inputs"
            if idir.exists():
                shutil.rmtree(idir)
            idir.mkdir(parents=True, exist_ok=True)
            for p in chunk:
                msa = Path(args.msa_dir) / f"{p['receptor_id']}.csv"
                rline = (f"      msa: {msa}\n" if msa.exists() else "      msa: empty\n")
                (idir / f"{p['name']}.yaml").write_text(
                    "version: 1\nsequences:\n"
                    f"  - protein:\n      id: A\n      sequence: {p['receptor']}\n{rline}"
                    f"  - protein:\n      id: B\n      sequence: {p['peptide']}\n"
                    f"      msa: empty\n")
            # No --seed passed, exactly as the benchmarks ran: this measures the
            # spread of the numbers that were actually reported.
            res, el = run_boltz(idir, bdir, args.recycling_steps, args.sampling_steps,
                                args.model, max_msa_seqs=32)
            for p in chunk:
                try:
                    s = BoltzCliPredictFn(str(res), lambda t, b, n=p["name"]: n)(
                        "", "")["iptm"].reshape(-1)[0].item()
                except (FileNotFoundError, KeyError):
                    s = float("nan")
                records.append({**{k: p[k] for k in
                                   ("receptor_id", "label", "name", "peptide_from")},
                                "replicate": rep, "score": s})
            out_path.write_text(json.dumps(records, indent=2))
            print(f"  rep {rep} batch {start // args.batch_size}: "
                  f"{len(chunk)} in {el:.0f}s", flush=True)

    report(records, sel)


def report(records, sel):
    print("\n" + "=" * 72)
    print("Run-to-run spread of ipTM (identical inputs, unseeded sampling)")
    print("=" * 72)

    by_name = {}
    for r in records:
        by_name.setdefault(r["name"], []).append(r["score"])

    sds, ranges, means = [], [], []
    print(f"\n  {'complex':12} {'label':10} {'n':>2} {'mean':>7} {'sd':>7} {'range':>8}")
    orig = {p["name"]: p for p in sel}
    for name, vals in sorted(by_name.items()):
        v = np.array([x for x in vals if not np.isnan(x)])
        if len(v) < 2:
            continue
        sds.append(v.std(ddof=1)); ranges.append(v.max() - v.min()); means.append(v.mean())
        print(f"  {name:12} {orig[name]['label']:10} {len(v):2d} "
              f"{v.mean():7.4f} {v.std(ddof=1):7.4f} {v.max() - v.min():8.4f}")

    sds = np.array(sds)
    pooled = np.sqrt(np.mean(sds ** 2))
    print(f"\n  pooled within-complex SD : {pooled:.4f}")
    print(f"  median SD                : {np.median(sds):.4f}")
    print(f"  median range             : {np.median(ranges):.4f}")

    print("\n  against the effects the conclusions rest on:")
    for label, eff in (("cognate - decoy      ", 0.0698),
                       ("cognate - own scramble", 0.0128)):
        ratio = eff / pooled if pooled > 0 else float("inf")
        verdict = ("larger than noise" if ratio > 2 else
                   "COMPARABLE TO NOISE" if ratio > 0.5 else "BELOW NOISE")
        print(f"    {label} {eff:+.4f}  = {ratio:5.2f} x pooled SD  -> {verdict}")

    # Rank stability: does the cognate's rank among its own decoys change
    # between replicates? The SD alone does not answer this.
    print("\n  rank stability (cognate vs its 3 decoys, per replicate):")
    reps = sorted({r["replicate"] for r in records})
    for rid in sorted({r["receptor_id"] for r in records}):
        seq = []
        for rep in reps:
            g = [r for r in records if r["receptor_id"] == rid
                 and r["replicate"] == rep and r["label"] in ("cognate", "decoy")
                 and not np.isnan(r["score"])]
            c = [x for x in g if x["label"] == "cognate"]
            d = [x for x in g if x["label"] == "decoy"]
            if not c or not d:
                continue
            seq.append(1 + sum(x["score"] >= c[0]["score"] for x in d))
        if seq:
            flips = "STABLE" if len(set(seq)) == 1 else f"FLIPS {min(seq)}-{max(seq)}"
            print(f"    {rid}: ranks {seq}   {flips}")

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "pooled_sd": float(pooled), "median_sd": float(np.median(sds)),
        "median_range": float(np.median(ranges)),
        "n_complexes": int(len(sds)), "n_replicates": len(reps),
        "receptors": sorted(args.receptors), "n_receptors": len(args.receptors),
        "per_complex": {n: {"mean": float(np.nanmean(v)),
                            "sd": float(np.nanstd(v, ddof=1))}
                        for n, v in by_name.items() if len(v) > 1},
    }, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
