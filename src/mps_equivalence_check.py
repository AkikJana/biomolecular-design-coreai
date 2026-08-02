"""Is folding on MPS equivalent to folding on CPU?

Every benchmark in this project ran with --accelerator cpu, while the log
reported "GPU available: True (mps), used: False". A single trial fold showed
MPS about 5x faster on the fold itself, which would bring the replicate-averaged
work the project needs from ~30-50 h down to ~6-10 h on this machine. Before
switching, the two devices have to be shown to agree.

**Equivalence, not "no significant difference".** Folds are unseeded, so a
paired test failing to reject is expected whatever MPS does -- with enough noise
nothing is significant. This uses TOST (two one-sided tests) against a
pre-specified margin: MPS counts as equivalent only if the CPU-MPS difference is
demonstrably smaller than the run-to-run noise the pipeline already tolerates.

  margin(ipTM)            0.0628   measured pooled within-complex SD (Sec 7.5)
  margin(interface pLDDT) 1.9172   same, recomputed for the metric now
                                   recommended for ranking

Both metrics are tested: ipTM because it is what the benchmark recorded, and
interface pLDDT because it is what Section 7.6 recommends ranking on.

Usage:
    python src/mps_equivalence_check.py --replicates 4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from rescore_interface_metrics import interface  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MARGIN = {"iptm": 0.0628, "iface_plddt": 1.9172}


def pick(scores_path, n):
    """A spread across the ipTM range, so agreement is not tested at one point."""
    pairs = [p for p in json.loads(Path(scores_path).read_text())
             if not np.isnan(p.get("score", float("nan")))]
    pairs.sort(key=lambda p: p["score"])
    idx = np.linspace(0, len(pairs) - 1, n).round().astype(int)
    return [pairs[i] for i in idx]


def fold(inputs, out, device, sampling, recycling):
    cmd = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
           "--out_dir", str(out), "--model", "boltz2",
           "--accelerator", "gpu" if device == "mps" else "cpu",
           "--recycling_steps", str(recycling), "--sampling_steps", str(sampling),
           "--output_format", "pdb", "--override",
           "--subsample_msa", "--num_subsampled_msa", "32", "--max_msa_seqs", "32"]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print((proc.stdout + proc.stderr)[-1500:], file=sys.stderr)
        raise RuntimeError(f"{device} fold failed")
    return out / f"boltz_results_{inputs.name}", time.perf_counter() - t0


def score(results_dir, names):
    parser = PDBParser(QUIET=True)
    out = {}
    for name in names:
        d = results_dir / "predictions" / name
        conf = d / f"confidence_{name}_model_0.json"
        pdb = d / f"{name}_model_0.pdb"
        if not conf.exists() or not pdb.exists():
            continue
        rec = {"iptm": json.loads(conf.read_text())["iptm"]}
        try:
            m = interface(parser.get_structure("x", str(pdb))[0])
            rec["iface_plddt"] = m["iface_plddt"] if m else float("nan")
        except Exception:
            rec["iface_plddt"] = float("nan")
        out[name] = rec
    return out


def tost(diffs, margin):
    """Two one-sided tests. Equivalent if BOTH reject at 0.05."""
    n = len(diffs)
    mean, se = diffs.mean(), diffs.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0, abs(mean) < margin
    t_lo = (mean + margin) / se
    t_hi = (mean - margin) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)     # H0: diff <= -margin
    p_hi = stats.t.cdf(t_hi, n - 1)         # H0: diff >= +margin
    p = max(p_lo, p_hi)
    return p, p < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--n-complexes", type=int, default=6)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "mps_equivalence"))
    ap.add_argument("--scores", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--msa-dir", default=str(REPO_ROOT / "artifacts" /
                                             "pdb_binders_b2_n22" / "msa_cache"))
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    sel = pick(args.scores, args.n_complexes)
    names = [p["name"] for p in sel]
    print(f"{len(sel)} complexes x {args.replicates} replicates x 2 devices = "
          f"{len(sel) * args.replicates * 2} folds")
    for p in sel:
        print(f"  {p['name']}  {p['receptor_id']:6} {p['label']:9} ipTM {p['score']:.4f}")

    inputs = work / "inputs"
    if inputs.exists():
        shutil.rmtree(inputs)
    inputs.mkdir(parents=True)
    for p in sel:
        msa = Path(args.msa_dir) / f"{p['receptor_id']}.csv"
        rline = f"      msa: {msa}\n" if msa.exists() else "      msa: empty\n"
        (inputs / f"{p['name']}.yaml").write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {p['receptor']}\n{rline}"
            f"  - protein:\n      id: B\n      sequence: {p['peptide']}\n      msa: empty\n")

    store = work / "mps_equivalence_scores.json"
    recs = json.loads(store.read_text()) if store.exists() else []
    done = {(r["device"], r["replicate"], r["name"]) for r in recs}

    for device in ("cpu", "mps"):
        for rep in range(args.replicates):
            if all((device, rep, n) in done for n in names):
                print(f"  {device} rep{rep}: already done"); continue
            out = work / f"{device}_rep{rep}"
            res, el = fold(inputs, out, device, args.sampling_steps, args.recycling_steps)
            got = score(res, names)
            for n, v in got.items():
                recs.append({"device": device, "replicate": rep, "name": n, **v})
            store.write_text(json.dumps(recs, indent=2))
            print(f"  {device} rep{rep}: {len(got)} complexes in {el:.0f}s "
                  f"({el / max(1, len(got)):.0f}s each)", flush=True)
            shutil.rmtree(out, ignore_errors=True)

    report(recs, names)


def report(recs, names):
    print("\n" + "=" * 74)
    print("MPS vs CPU equivalence")
    print("=" * 74)

    for met, margin in MARGIN.items():
        cpu, mps, diffs = {}, {}, []
        for n in names:
            c = [r[met] for r in recs if r["device"] == "cpu" and r["name"] == n
                 and not np.isnan(r.get(met, float("nan")))]
            m = [r[met] for r in recs if r["device"] == "mps" and r["name"] == n
                 and not np.isnan(r.get(met, float("nan")))]
            if not c or not m:
                continue
            cpu[n], mps[n] = float(np.mean(c)), float(np.mean(m))
            diffs.append(mps[n] - cpu[n])
        diffs = np.array(diffs)
        if len(diffs) < 3:
            print(f"\n{met}: too few paired complexes"); continue

        within = []
        for n in names:
            for dev in ("cpu", "mps"):
                v = [r[met] for r in recs if r["device"] == dev and r["name"] == n
                     and not np.isnan(r.get(met, float("nan")))]
                if len(v) > 1:
                    within.append(np.std(v, ddof=1))
        within_sd = float(np.sqrt(np.mean(np.square(within)))) if within else float("nan")

        p_diff = stats.ttest_1samp(diffs, 0).pvalue
        p_tost, equiv = tost(diffs, margin)
        ci = stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(),
                              scale=diffs.std(ddof=1) / np.sqrt(len(diffs)))

        print(f"\n{met}   (equivalence margin +/-{margin})")
        print(f"  mean MPS - CPU      {diffs.mean():+.4f}   95% CI "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]   n={len(diffs)} complexes")
        print(f"  within-device SD    {within_sd:.4f}  (replicate spread on one device)")
        print(f"  difference test     p = {p_diff:.4f}  "
              f"({'no detectable difference' if p_diff > 0.05 else 'DIFFERENCE DETECTED'})")
        print(f"  TOST equivalence    p = {p_tost:.4f}  -> "
              f"{'EQUIVALENT within margin' if equiv else 'NOT established'}")

    print("\nEquivalence requires the TOST line, not the difference line: with")
    print("unseeded folds, failing to detect a difference is uninformative.")


if __name__ == "__main__":
    main()
