"""Run the surrogate-vs-reference benchmark against real Boltz predictions.

This closes the loop the benchmark harness was written for: it generates binder
variants, folds each (target, binder) complex with Boltz, and compares the
reference ranking to the edge surrogate's, reporting rank agreement alongside
latency and model size.

**Reference signal.** Boltz-2's affinity head predicts protein-*ligand* binding
affinity and needs the Boltz-2 weights plus an affinity checkpoint. Neither is
applicable here: the candidates are peptide binders, not small molecules. This
runner therefore ranks by **interface confidence (ipTM)** from Boltz-1, which is
a standard proxy for whether a binder engages its target and runs against the
checkpoint already in ~/.boltz. Pass --rank-key to use another confidence field.
Read the output as "does the surrogate reproduce Boltz's interface-confidence
ranking", not "does it reproduce Boltz-2 affinity".

**Latency.** The reference scorer reads predictions Boltz already wrote, so
timing its score() would measure JSON parsing. The wall clock of the actual
Boltz run is passed to benchmark() via reference_latency_ms so the reported
speedup reflects inference cost.

Usage:
    python src/run_reference_benchmark.py --num-binders 6
    python src/run_reference_benchmark.py --results-dir <existing> --skip-predict
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path


from benchmark_surrogate_vs_reference import benchmark
from boltz2_predict import BoltzCliPredictFn, BoltzAffinityScorer
from surrogate_affinity import AffinitySurrogate, SurrogateAffinityScorer

REPO_ROOT = Path(__file__).resolve().parent.parent

# A short target keeps CPU folding tractable; the binder is the 15-mer used
# elsewhere in this repo.
DEFAULT_TARGET = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
DEFAULT_BINDER = "MATEVLADIGSAKLR"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def make_binder_variants(base: str, n: int, seed: int = 0):
    """Single-point mutants of `base`, including the wild type."""
    rng = random.Random(seed)
    variants, seen = [base], {base}
    while len(variants) < n:
        pos = rng.randrange(len(base))
        sub = rng.choice(ALPHABET)
        cand = base[:pos] + sub + base[pos + 1:]
        if cand not in seen:
            seen.add(cand)
            variants.append(cand)
    return variants


def write_inputs(input_dir: Path, target: str, binders):
    """One YAML per (target, binder) complex, single-sequence mode (no MSA)."""
    input_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, binder in enumerate(binders):
        name = f"pair_{i:03d}"
        (input_dir / f"{name}.yaml").write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {target}\n"
            "      msa: empty\n"
            "  - protein:\n"
            "      id: B\n"
            f"      sequence: {binder}\n"
            "      msa: empty\n"
        )
        names.append(name)
    return names


def run_boltz(input_dir: Path, out_dir: Path, recycling: int, sampling: int, model: str):
    """Fold every input. Returns (results_dir, wall_clock_seconds)."""
    cmd = [
        sys.executable, "-m", "boltz.main", "predict", str(input_dir),
        "--out_dir", str(out_dir),
        "--model", model,
        "--accelerator", "cpu",
        "--recycling_steps", str(recycling),
        "--sampling_steps", str(sampling),
        "--output_format", "pdb",
        "--override",
    ]
    print(f"[boltz] {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        combined = proc.stdout + proc.stderr
        if "low_rank_updater" in combined and "Unexpected key" in combined:
            raise RuntimeError(
                "boltz predict failed: the vendored Boltz cannot load the stock "
                "checkpoint.\n\n"
                "This branch replaces OuterProductMean with the low-rank "
                "S-contracted implementation unconditionally, so its parameters "
                "(low_rank_updater.W / proj_x / proj_y) do not match the official "
                "boltz1_conf.ckpt (proj_a / proj_b / proj_o). No pretrained "
                "reference is loadable until the implementation is selectable.\n\n"
                "The fix exists on branch polly/opm-toggle (commit e41b767), which "
                "makes the choice a BOLTZMAC_OPM env toggle defaulting to stock. It "
                "is not merged here, and merging it also requires updating "
                "tests/test_boltz_modified_layers.py, which reaches into "
                "OuterProductMean.low_rank_updater directly."
            )
        print(combined[-3000:], file=sys.stderr)
        raise RuntimeError(f"boltz predict failed with exit code {proc.returncode}")
    return out_dir / f"boltz_results_{input_dir.name}", elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-binders", type=int, default=6)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--binder", default=DEFAULT_BINDER)
    ap.add_argument("--rank-key", default="iptm",
                    help="confidence field to rank by (iptm, complex_plddt, ptm, ...)")
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--model", default="boltz1", choices=["boltz1", "boltz2"])
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "reference_benchmark"))
    ap.add_argument("--results-dir", default=None,
                    help="reuse an existing boltz_results_* directory")
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--reference-latency-ms", type=float, default=None,
                    help="override the measured per-candidate inference cost")
    args = ap.parse_args()

    work = Path(args.work_dir)
    binders = make_binder_variants(args.binder, args.num_binders)
    pairs = [(args.target, b) for b in binders]

    input_dir = work / "inputs"
    names = write_inputs(input_dir, args.target, binders)

    ref_latency_ms = args.reference_latency_ms
    if args.skip_predict:
        if not args.results_dir:
            ap.error("--skip-predict requires --results-dir")
        results_dir = Path(args.results_dir)
    else:
        results_dir, elapsed = run_boltz(
            input_dir, work, args.recycling_steps, args.sampling_steps, args.model
        )
        per_candidate_ms = elapsed / len(pairs) * 1000.0
        print(f"[boltz] folded {len(pairs)} complexes in {elapsed:.1f}s "
              f"({per_candidate_ms:.0f} ms/candidate)")
        if ref_latency_ms is None:
            ref_latency_ms = per_candidate_ms

    name_for = dict(zip([b for b in binders], names))
    predict_fn = BoltzCliPredictFn(
        str(results_dir), name_fn=lambda target, binder: name_for[binder]
    )

    reference = BoltzAffinityScorer(predict_fn, rank_key=args.rank_key)
    reference.name = f"{args.model}({args.rank_key})"
    surrogate = SurrogateAffinityScorer(AffinitySurrogate())

    ref_raw = reference.score(pairs)
    print(f"\nreference {args.rank_key} per binder:")
    for binder, value in zip(binders, ref_raw.tolist()):
        print(f"  {binder}  {value:+.4f}")
    if float(ref_raw.max() - ref_raw.min()) == 0.0:
        print(f"\nWARNING: every candidate scored identically on '{args.rank_key}'. "
              f"Rank correlation is undefined against a constant reference; "
              f"try --rank-key complex_plddt or more sampling steps.")

    ks = tuple(k for k in (1, 3, 5) if k <= len(pairs))
    metrics = benchmark(pairs, reference, surrogate, ks=ks,
                        reference_latency_ms=ref_latency_ms)

    print("\nNOTE: AffinitySurrogate is untrained here, so rank agreement reflects "
          "an un-distilled model. Train it against these reference scores with "
          "src/train_surrogate_affinity.py before quoting the correlation.")
    return metrics


if __name__ == "__main__":
    main()
