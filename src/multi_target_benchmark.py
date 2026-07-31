"""Multi-target, diverse-binder version of the reference benchmark.

The single-target single-mutant scan produced a reference ranking with almost no
dynamic range (ipTM 0.052-0.103) and only ~1% of rank variance learnable. Two
things could be responsible: the binder set is too narrow, or the reference
cannot discriminate binders at all on this kind of input.

This script separates those by widening the task on both axes and adding a
control:

  * **Multiple targets.** Ranking is evaluated *within* each target and averaged,
    which is the actual screening use case, and pooled across targets as well.
  * **Diverse binders.** Single-point mutants, multi-point mutants (2-6
    substitutions), and scrambled sequences.
  * **Scrambled control.** A scramble keeps the exact amino-acid composition and
    destroys the order. If the reference cannot separate scrambled binders from
    designed ones, it is not responding to sequence structure at all, and no
    surrogate trained against it can be expected to rank anything. That check
    matters more than the correlation.

MSAs are fetched once per target and reused across all its binders.

Usage:
    python src/multi_target_benchmark.py --binders-per-target 60
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from boltz2_predict import BoltzCliPredictFn
from run_reference_benchmark import ALPHABET, REPO_ROOT, run_boltz

# Short, well-characterised, structurally distinct targets. Kept small so CPU
# folding stays tractable.
TARGETS = {
    "hba": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",       # haemoglobin alpha frag
    "ubq": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # ubiquitin
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",  # protein G B1 domain
}
BASE_BINDER = "MATEVLADIGSAKLR"


def make_diverse_binders(base: str, n: int, seed: int):
    """Single mutants, multi-point mutants, and composition-preserving scrambles."""
    rng = random.Random(seed)
    out, seen, kinds = [], set(), []

    def add(seq, kind):
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
            kinds.append(kind)

    add(base, "wildtype")
    n_single = n // 3
    n_multi = n // 3
    while sum(k == "single" for k in kinds) < n_single:
        p = rng.randrange(len(base))
        add(base[:p] + rng.choice(ALPHABET) + base[p + 1:], "single")
    while sum(k == "multi" for k in kinds) < n_multi:
        s = list(base)
        for p in rng.sample(range(len(base)), rng.randint(2, 6)):
            s[p] = rng.choice(ALPHABET)
        add("".join(s), "multi")
    guard = 0
    while len(out) < n and guard < 10000:      # scrambles: same composition, new order
        guard += 1
        s = list(base)
        rng.shuffle(s)
        add("".join(s), "scrambled")
    return out, kinds


def write_pair_yaml(path: Path, target: str, binder: str, target_msa: str | None):
    tline = f"      msa: {target_msa}\n" if target_msa else "      msa: empty\n"
    path.write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {target}\n{tline}"
        f"  - protein:\n      id: B\n      sequence: {binder}\n      msa: empty\n"
    )


def fetch_target_msa(work: Path, name: str, target: str) -> str | None:
    """Fold one throwaway complex with the MSA server to obtain the target alignment."""
    cache = work / "msa_cache" / f"{name}.csv"
    if cache.exists():
        return str(cache)
    probe = work / "msa_fetch" / name / "inputs"
    probe.mkdir(parents=True, exist_ok=True)
    write_pair_yaml(probe / "pair_000.yaml", target, BASE_BINDER, None)
    (probe / "pair_000.yaml").write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {target}\n"
        f"  - protein:\n      id: B\n      sequence: {BASE_BINDER}\n      msa: empty\n"
    )
    results, _ = run_boltz(probe, probe.parent, 0, 5, "boltz1", use_msa=True, max_msa_seqs=32)
    src = results / "msa" / "pair_000_0.csv"
    if not src.exists():
        print(f"  [msa] no alignment produced for {name}; falling back to single-sequence")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(src.read_text())
    print(f"  [msa] {name}: {len(src.read_text().splitlines()) - 1} homologs cached")
    return str(cache)


def fold_target(work: Path, name: str, target: str, binders, msa, batch_size, sampling, recycling):
    dirs = []
    for start in range(0, len(binders), batch_size):
        chunk = binders[start:start + batch_size]
        bdir = work / name / f"batch_{start // batch_size:02d}"
        idir = bdir / "inputs"
        idir.mkdir(parents=True, exist_ok=True)
        for j, b in enumerate(chunk):
            write_pair_yaml(idir / f"pair_{start + j:03d}.yaml", target, b, msa)
        res, elapsed = run_boltz(idir, bdir, recycling, sampling, "boltz1", max_msa_seqs=32)
        dirs.append(res)
        print(f"  [{name}] batch {start // batch_size}: {len(chunk)} in {elapsed:.0f}s", flush=True)
    return dirs


def read_scores(dirs, binders, rank_key):
    name_for = {b: f"pair_{i:03d}" for i, b in enumerate(binders)}
    vals = []
    for b in binders:
        for d in dirs:
            try:
                vals.append(BoltzCliPredictFn(str(d), lambda t, x: name_for[x])(
                    "", b)[rank_key].reshape(-1)[0].item())
                break
            except (FileNotFoundError, KeyError):
                continue
        else:
            vals.append(float("nan"))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binders-per-target", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--rank-key", default="iptm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "multi_target"))
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    per_target = {}

    for name, target in TARGETS.items():
        # Stable per-target seed: Python's hash() is salted per process, so
        # hash(name) would give a different binder set on every run.
        name_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100000
        binders, kinds = make_diverse_binders(BASE_BINDER, args.binders_per_target,
                                              args.seed + name_seed)
        msa = fetch_target_msa(work, name, target) if not args.skip_predict else \
            str(work / "msa_cache" / f"{name}.csv")
        if not args.skip_predict:
            dirs = fold_target(work, name, target, binders, msa, args.batch_size,
                               args.sampling_steps, args.recycling_steps)
        else:
            dirs = sorted((work / name).glob("batch_*/boltz_results_inputs"))
        scores = read_scores(dirs, binders, args.rank_key)
        per_target[name] = {"binders": binders, "kinds": kinds, "scores": scores.tolist()}
        ok = ~np.isnan(scores)
        print(f"[{name}] {ok.sum()}/{len(scores)} scored | "
              f"ipTM {np.nanmin(scores):.4f}-{np.nanmax(scores):.4f} "
              f"(sd {np.nanstd(scores):.4f})")

    # The control: can the reference tell designed binders from scrambles?
    print("\n" + "=" * 66)
    print("Reference discrimination: designed vs scrambled (same composition)")
    print("=" * 66)
    for name, d in per_target.items():
        s = np.array(d["scores"])
        kinds = np.array(d["kinds"])
        designed = s[(kinds != "scrambled") & ~np.isnan(s)]
        scram = s[(kinds == "scrambled") & ~np.isnan(s)]
        if len(scram) < 3 or len(designed) < 3:
            print(f"  {name}: too few in one group")
            continue
        pooled = np.sqrt((designed.var(ddof=1) + scram.var(ddof=1)) / 2)
        d_eff = (designed.mean() - scram.mean()) / pooled if pooled > 0 else float("nan")
        print(f"  {name}: designed {designed.mean():.4f} (n={len(designed)}) vs "
              f"scrambled {scram.mean():.4f} (n={len(scram)})  Cohen's d = {d_eff:+.2f}")

    out = work / "multi_target_scores.json"
    out.write_text(json.dumps(per_target, indent=2))
    print(f"\nscores: {out}")


if __name__ == "__main__":
    main()
