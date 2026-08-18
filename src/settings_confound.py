"""Do the negative results survive at the model's intended settings?

Every fold in Sections 7.2 to 7.12 was taken far below Boltz defaults: 10
sampling steps against 200, 1 recycling pass against 3, and an alignment
subsampled to 32 rows from the full depth. Section 1.5 states the confound and
Section 7.8 partly addresses it by running a model *trained* for ten steps, but
nothing has been folded at the intended settings, so "the settings confound is
stated, not resolved" has stood as the strongest available criticism of every
negative result in this work.

This resolves it. The same 132 pairs, the same model (stock Boltz-1), the same
device (MPS), with only the settings changed:

    reduced   10 sampling steps,  1 recycling,  MSA subsampled to 32
    full     200 sampling steps,  3 recycling,  full alignment depth

The reduced arm already exists as boltz1_scramble_result.json from Section 7.8,
so model and device are held constant by construction and settings are the only
difference. That is what makes this a test of the confound rather than another
arm.

A note on cost, since it is why this was not done earlier. The runner's own
comment records that a full alignment made the MSA module intractable on CPU --
a 40-complex run did not finish one batch in an hour and drove the machine to
~12 GB of swap. On MPS a full-depth fold at 200 steps takes about 106 seconds,
no slower than the same fold at depth 32. The obstacle was the device, not the
settings.

Two outcomes, both worth having. If the negatives hold, every conclusion in
Section 7 is strengthened and the confound is closed. If they do not, the
reduced-settings regime is doing the work and much of Section 7 describes a
degraded model rather than the metrics.

Usage:
    python src/settings_confound.py --batch-size 12
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

from interface_side_split import sides  # noqa: E402

ART = REPO_ROOT / "artifacts"
PANEL = ART / "pdb_binders_b2_n22"
WORK = ART / "settings_confound"


def free_gib(path=REPO_ROOT):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def fold(inputs, out, sampling, recycling, msa_depth, per_fold_budget=300):
    """One batch at the given settings. msa_depth=None means full alignment.

    Boltz gets its own TMPDIR and the cache size is reported per batch. The
    volume did fill during a run -- the fold died with "The volume Macintosh HD
    is out of space" -- but measurement afterwards did not support blaming the
    graph cache: the scratch directory stayed at 0 B and the Metal shader cache
    held steady at 180 MB while free space rose. The scratch directory is kept
    because it makes the claim measurable rather than assumed; the guard that
    actually addresses a full volume is --min-free-gib.
    """
    cmd = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
           "--out_dir", str(out), "--model", "boltz1",
           "--accelerator", "gpu", "--output_format", "pdb", "--override",
           "--recycling_steps", str(recycling),
           "--sampling_steps", str(sampling), "--diffusion_samples", "1",
           # Fork-based dataloader workers deadlock intermittently here: the
           # main thread parks in a condition wait at ~0% CPU and never returns,
           # after the run reports leaked loky semaphores. Three batches hung
           # this way. Loading in-process removes the fork entirely, and the
           # data pipeline is not the bottleneck at 200 sampling steps.
           "--num_workers", "0", "--preprocessing-threads", "1"]
    if msa_depth is not None:
        cmd += ["--subsample_msa", "--num_subsampled_msa", str(msa_depth),
                "--max_msa_seqs", str(msa_depth)]
    scratch = WORK / ".mpscache"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", TMPDIR=str(scratch))
    # A batch that hangs is worse than one that fails: it holds the lock, keeps
    # its process alive, and every liveness check reports it as healthy. One
    # batch blocked for three hours at 0.4% CPU after the volume filled, and was
    # reported as "running" twice before the score store's mtime gave it away.
    # A generous per-fold budget turns that silence into an exception.
    n_folds = len(list(inputs.glob("*.yaml")))
    budget = max(600, n_folds * per_fold_budget)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=budget)
    except subprocess.TimeoutExpired:
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError(
            f"boltz produced nothing for {budget}s on {n_folds} folds "
            f"(budget {per_fold_budget}s each) -- treating as hung, not slow")
    cache_gib = sum(f.stat().st_size for f in scratch.rglob("*") if f.is_file()) / 2**30
    shutil.rmtree(scratch, ignore_errors=True)
    if proc.returncode != 0:
        # 1200 characters twice cut off the line that mattered.
        print((proc.stdout + proc.stderr)[-6000:], file=sys.stderr)
        raise RuntimeError("boltz predict failed")
    return out / f"boltz_results_{inputs.name}", time.perf_counter() - t0, cache_gib


def score_dir(results, names):
    parser = PDBParser(QUIET=True)
    out = {}
    for n in names:
        d = results / "predictions" / n
        pdb, conf = d / f"{n}_model_0.pdb", d / f"confidence_{n}_model_0.json"
        if not pdb.exists():
            continue
        try:
            s = sides(parser.get_structure("x", str(pdb))[0])
        except Exception:
            continue
        if s:
            s["iptm"] = (json.loads(conf.read_text())["iptm"]
                         if conf.exists() else float("nan"))
            out[n] = s
    return out


def tests(recs, metric):
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s:
            diffs += [c[0][metric] - x[metric] for x in s]
    ranks, sizes = [], []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        d = [x for x in g if x["label"] == "decoy"]
        if not c or not d:
            continue
        sc = [c[0][metric]] + [x[metric] for x in d]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    # The two tests need different data and must be reported independently: the
    # scramble control needs only cognates and scrambles, and requiring decoys
    # too silently dropped the whole full-settings arm from the table.
    if len(diffs) < 3:
        return None
    e = np.array(diffs, float)
    out = {"effect": float(e.mean()),
           "p": float(stats.ttest_1samp(e, 0).pvalue),
           "n_pairs": len(e), "n_scramble_receptors": len(by)}
    if len(ranks) >= 5:
        r = np.array(ranks, float)
        exp = (np.array(sizes, float) + 1) / 2
        out.update({"mean_rank": float(r.mean()), "chance": float(exp.mean()),
                    "first": int((r == 1).sum()), "n_receptors": len(r),
                    "rank_p": float(stats.wilcoxon(r - exp)[1])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--msa-depth", type=int, default=0,
                    help="0 means the full cached alignment")
    ap.add_argument("--labels", default="",
                    help="comma-separated subset, e.g. cognate,scrambled. The "
                         "scramble control is the decisive test and needs only "
                         "66 of the 132 folds, so it can be run first.")
    ap.add_argument("--per-fold-budget", type=int, default=300,
                    help="seconds per fold before a batch is declared hung; "
                         "observed cost is 106-130s")
    ap.add_argument("--min-free-gib", type=float, default=6.0,
                    help="refuse to start a batch below this much free disk")
    ap.add_argument("--run-tag", default="",
                    help="suffix for the score store, batch dirs and output, so "
                         "one arm cannot be mistaken for another. Without it a "
                         "second arm finds every fold name already present, "
                         "skips all of them, and reports the FIRST arm's folds "
                         "under the second arm's settings.")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    depth = None if args.msa_depth == 0 else args.msa_depth

    pairs = json.loads((PANEL / "pdb_binder_scores.json").read_text())
    if args.labels:
        keep = set(args.labels.split(","))
        pairs = [p for p in pairs if p["label"] in keep]
    WORK.mkdir(parents=True, exist_ok=True)

    # Two instances computed the same batch directory and one deleted inputs/
    # while the other's boltz was reading it, which surfaced as a
    # non-deterministic "boltz predict failed" with no useful stderr. A run is
    # long enough that a second one is always a mistake, so refuse rather than
    # try to coexist.
    # Only folding touches batch directories, so analysis must not be blocked --
    # and a run that is still writing its final batch would otherwise lock out
    # the very command used to read its results.
    lock = WORK / f".lock{args.run_tag}"
    if args.analyse_only:
        lock = None
    elif lock.exists():
        try:
            other = int(lock.read_text().strip())
            os.kill(other, 0)
        except (ValueError, ProcessLookupError):
            lock.unlink()                      # stale lock from a killed run
        else:
            raise SystemExit(f"another run is active (pid {other}); refusing to "
                             f"start a second -- they collide on batch dirs")
    if lock is not None:
        lock.write_text(str(os.getpid()))
        import atexit
        atexit.register(lambda: lock.exists() and lock.unlink())
    store = WORK / f"scores{args.run_tag}.json"
    recs = json.loads(store.read_text()) if store.exists() else []

    # A resumed store must have been folded at the settings now being asked for,
    # or the resume silently mixes two regimes into one arm.
    side = WORK / f"arm{args.run_tag}.json"
    want = {"sampling": args.sampling_steps, "recycling": args.recycling_steps,
            "msa_depth": depth}
    if side.exists():
        have = json.loads(side.read_text())
        if have != want and recs:
            raise SystemExit(
                f"{store.name} holds {len(recs)} folds at {have}, but this run "
                f"asks for {want}. Use a different --run-tag.")
    side.write_text(json.dumps(want, indent=2))

    if not args.analyse_only:
        print(f"{len(pairs)} pairs | Boltz-1 | MPS | {args.sampling_steps} steps, "
              f"{args.recycling_steps} recycling, MSA "
              f"{'full' if depth is None else depth}")
        done = {r["name"] for r in recs}
        todo = [p for p in pairs if p["name"] not in done]
        skipped = []
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            bdir = WORK / f"b{args.run_tag}{start // args.batch_size:02d}"
            inputs = bdir / "inputs"
            if inputs.exists():
                shutil.rmtree(inputs)
            inputs.mkdir(parents=True)
            for p in chunk:
                msa = PANEL / "msa_cache" / f"{p['receptor_id']}.csv"
                rline = f"      msa: {msa}\n" if msa.exists() else "      msa: empty\n"
                (inputs / f"{p['name']}.yaml").write_text(
                    "version: 1\nsequences:\n"
                    f"  - protein:\n      id: A\n      sequence: {p['receptor']}\n{rline}"
                    f"  - protein:\n      id: B\n      sequence: {p['peptide']}\n"
                    f"      msa: empty\n")
            if free_gib() < args.min_free_gib:
                raise SystemExit(
                    f"only {free_gib():.1f} GiB free, need {args.min_free_gib} "
                    f"-- stopping cleanly at {len(recs)} folds rather than "
                    f"filling the volume mid-fold. Scores are saved; rerun to "
                    f"resume.")
            # Boltz hangs intermittently on this machine -- main thread parked
            # in a condition wait at ~0% CPU, no error, indefinitely. Neither
            # dataloader workers nor MSA depth explains it: the deepest
            # alignment in the panel (4Z8J, 14,159 rows) folds fine, and the
            # hang persists with --num_workers 0. Rather than keep guessing at
            # the cause, a timed-out batch is retried one fold at a time; folds
            # that hang alone are recorded and skipped so the run completes.
            try:
                res, el, cache_gib = fold(inputs, bdir, args.sampling_steps,
                                          args.recycling_steps, depth,
                                          args.per_fold_budget)
                got = score_dir(res, [p["name"] for p in chunk])
            except RuntimeError as exc:
                print(f"  batch {start // args.batch_size} failed ({exc}); "
                      f"retrying {len(chunk)} folds individually", flush=True)
                got, el, cache_gib = {}, 0.0, 0.0
                for one in chunk:
                    solo = bdir / f"solo_{one['name']}"
                    sin = solo / "inputs"
                    shutil.rmtree(solo, ignore_errors=True)
                    sin.mkdir(parents=True)
                    shutil.copy2(inputs / f"{one['name']}.yaml",
                                 sin / f"{one['name']}.yaml")
                    try:
                        r1, e1, c1 = fold(sin, solo, args.sampling_steps,
                                          args.recycling_steps, depth,
                                          args.per_fold_budget)
                        got.update(score_dir(r1, [one["name"]]))
                        el += e1
                    except RuntimeError:
                        skipped.append(one["name"])
                        print(f"    skipped {one['name']} "
                              f"({one['receptor_id']}, {one['label']}) -- hung alone",
                              flush=True)
                    shutil.rmtree(solo, ignore_errors=True)
            for p in chunk:
                if p["name"] in got:
                    recs.append({**{k: p[k] for k in
                                    ("name", "receptor_id", "label", "peptide_from")},
                                 **got[p["name"]]})
            store.write_text(json.dumps(recs, indent=2))
            print(f"  batch {start // args.batch_size}: {len(got)}/{len(chunk)} "
                  f"in {el:.0f}s | mps cache {cache_gib:.1f} GiB purged | "
                  f"{free_gib():.1f} GiB free", flush=True)
            shutil.rmtree(bdir, ignore_errors=True)
        if skipped:
            print(f"\n{len(skipped)} fold(s) skipped after hanging alone: "
                  f"{', '.join(skipped)}", flush=True)

    reduced = json.loads((ART / "boltz1_scramble_result.json").read_text())["per_fold"]
    for r in reduced:
        r.setdefault("receptor_id", r.get("receptor_id"))
    print(f"\n{'=' * 78}\nSame model, same device, settings varied\n{'=' * 78}")
    print(f"  reduced : 10 steps, 1 recycling, MSA 32   ({len(reduced)} folds, "
          f"Section 7.8)")
    print(f"  full    : {args.sampling_steps} steps, {args.recycling_steps} "
          f"recycling, MSA {'full' if depth is None else depth}   "
          f"({len(recs)} folds)\n")
    print(f"{'metric':16} {'arm':9} {'cog-scr':>9} {'p':>9} {'rank':>6} "
          f"{'p':>8} {'#1':>7}")
    print("-" * 74)
    out = {}
    for metric in ("iptm", "iface_plddt", "receptor_side"):
        for lab, arm in (("reduced", reduced), ("full", recs)):
            t = tests(arm, metric)
            if not t:
                continue
            out.setdefault(metric, {})[lab] = t
            rk = (f"{t['mean_rank']:6.2f} {t['rank_p']:8.4f} "
                  f"{t['first']:3d}/{t['n_receptors']:<3d}"
                  if "mean_rank" in t else f"{'-':>6} {'(no decoys)':>18}")
            print(f"{metric:16} {lab:9} {t['effect']:+9.3f} {t['p']:9.5f} {rk}")
        print()

    print("The question is whether the negatives are properties of the metrics or")
    print("of the reduced regime. If ipTM still fails its scramble control at full")
    print("settings, Section 7.4 stands on the model's own terms.")
    (ART / f"settings_confound{args.run_tag}.json").write_text(json.dumps(
        {"settings": {"sampling": args.sampling_steps,
                      "recycling": args.recycling_steps,
                      "msa_depth": depth}, "per_fold": recs, "summary": out},
        indent=2, default=float))
    print(f"\nwrote {ART}/settings_confound{args.run_tag}.json")


if __name__ == "__main__":
    main()
