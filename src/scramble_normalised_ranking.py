"""Rank binders on a score minus its own scramble, rather than on the raw score.

Section 7.4 showed the ranking signal is contaminated by peptide composition and
length. Every candidate therefore carries its own exact control: a scramble of
itself, which holds composition and length fixed and destroys only order. So
score the candidate against that control rather than against the population:

    delta(candidate) = score(candidate) - mean score(scrambles of it)

Unlike a global bias correction this cancels composition *per candidate*, with no
assumption that the bias is uniform across peptides.

**This is not free, and may not win.** delta subtracts two noisy quantities, so
with k scrambles its noise is sqrt(1 + 1/k) times that of the raw score -- about
1.22x at k=2. The bias it removes has to be worth that. The experiment measures
which effect dominates.

Design. Each receptor contributes 4 candidates (1 cognate + 3 decoys) and 2
scrambles of each, so 12 folds per receptor and 264 in total. Everything is
re-folded on one device: delta is a within-candidate difference, so a device
offset cancels inside it -- but only if the candidate and its scrambles share a
device, which rules out reusing the existing CPU scores.

Usage:
    python src/scramble_normalised_ranking.py --batch-size 12
"""

import argparse
import json
import os
import random
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
SCRAMBLES_PER_CANDIDATE = 2


def build(scores_path, seed):
    """Candidates (cognate + decoys) and independent scrambles of each."""
    pairs = [p for p in json.loads(Path(scores_path).read_text())
             if not np.isnan(p.get("score", float("nan")))]
    by = {}
    for p in pairs:
        by.setdefault(p["receptor_id"], []).append(p)

    rng = random.Random(seed)
    jobs, k = [], 0
    for rid in sorted(by):
        group = by[rid]
        cands = ([p for p in group if p["label"] == "cognate"]
                 + [p for p in group if p["label"] == "decoy"])
        for cand in cands:
            cid = f"c{k:03d}"
            k += 1
            jobs.append({"job": cid, "role": "candidate", "receptor_id": rid,
                         "label": cand["label"], "receptor": cand["receptor"],
                         "peptide": cand["peptide"], "parent": cid,
                         "src_name": cand["name"]})
            for s in range(SCRAMBLES_PER_CANDIDATE):
                chars = list(cand["peptide"])
                rng.shuffle(chars)
                jobs.append({"job": f"{cid}s{s}", "role": "scramble",
                             "receptor_id": rid, "label": cand["label"],
                             "receptor": cand["receptor"],
                             "peptide": "".join(chars), "parent": cid,
                             "src_name": cand["name"]})
    return jobs


def fold(inputs, out, sampling, recycling, device):
    cmd = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
           "--out_dir", str(out), "--model", "boltz2",
           "--accelerator", "gpu" if device == "mps" else "cpu",
           "--recycling_steps", str(recycling), "--sampling_steps", str(sampling),
           "--output_format", "pdb", "--override",
           "--subsample_msa", "--num_subsampled_msa", "32", "--max_msa_seqs", "32"]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    t0 = time.perf_counter()
    if os.environ.get("BOLTZ_NO_KERNELS") and "--no_kernels" not in cmd:
        cmd.append("--no_kernels")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print((proc.stdout + proc.stderr)[-1200:], file=sys.stderr)
        raise RuntimeError("fold failed")
    return out / f"boltz_results_{inputs.name}", time.perf_counter() - t0


def score_dir(results, names):
    parser = PDBParser(QUIET=True)
    out = {}
    for n in names:
        d = results / "predictions" / n
        conf, pdb = d / f"confidence_{n}_model_0.json", d / f"{n}_model_0.pdb"
        if not conf.exists() or not pdb.exists():
            continue
        rec = {"iptm": json.loads(conf.read_text())["iptm"]}
        try:
            m = interface(parser.get_structure("x", str(pdb))[0])
            rec["iface_plddt"] = m["iface_plddt"] if m else float("nan")
        except Exception:
            rec["iface_plddt"] = float("nan")
        out[n] = rec
    return out


def run(jobs, work, msa_dir, batch, sampling, recycling, device):
    store = work / "scramble_norm_scores.json"
    recs = json.loads(store.read_text()) if store.exists() else []
    done = {r["job"] for r in recs}
    todo = [j for j in jobs if j["job"] not in done]
    print(f"{len(jobs)} folds total, {len(todo)} remaining")

    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        bdir = work / f"b{start // batch:03d}"
        inputs = bdir / "inputs"
        if inputs.exists():
            shutil.rmtree(inputs)
        inputs.mkdir(parents=True)
        for j in chunk:
            msa = Path(msa_dir) / f"{j['receptor_id']}.csv"
            rline = f"      msa: {msa}\n" if msa.exists() else "      msa: empty\n"
            (inputs / f"{j['job']}.yaml").write_text(
                "version: 1\nsequences:\n"
                f"  - protein:\n      id: A\n      sequence: {j['receptor']}\n{rline}"
                f"  - protein:\n      id: B\n      sequence: {j['peptide']}\n"
                f"      msa: empty\n")
        res, el = fold(inputs, bdir, sampling, recycling, device)
        got = score_dir(res, [j["job"] for j in chunk])
        for j in chunk:
            if j["job"] in got:
                recs.append({**{k: j[k] for k in
                                ("job", "role", "receptor_id", "label", "parent")},
                             **got[j["job"]]})
        store.write_text(json.dumps(recs, indent=2))
        print(f"  batch {start // batch}: {len(got)}/{len(chunk)} in {el:.0f}s",
              flush=True)
        shutil.rmtree(bdir, ignore_errors=True)
    return recs


def analyse(recs):
    cands = {r["parent"]: r for r in recs if r["role"] == "candidate"}
    scr = {}
    for r in recs:
        if r["role"] == "scramble":
            scr.setdefault(r["parent"], []).append(r)

    print("\n" + "=" * 74)
    print("Raw score vs scramble-normalised score, as rankers")
    print("=" * 74)

    rows = []
    for pid, c in cands.items():
        s = scr.get(pid, [])
        if not s:
            continue
        row = {"receptor_id": c["receptor_id"], "label": c["label"]}
        for met in ("iptm", "iface_plddt"):
            vals = [x[met] for x in s if not np.isnan(x.get(met, float("nan")))]
            row[f"raw_{met}"] = c[met]
            row[f"delta_{met}"] = (c[met] - float(np.mean(vals))) if vals else float("nan")
        rows.append(row)

    by = {}
    for r in rows:
        by.setdefault(r["receptor_id"], []).append(r)

    print(f"\n{'metric':26} {'#1':>4} {'mean rank':>10} {'chance':>7} {'p':>8}")
    print("-" * 74)
    out = {}
    for met in ("iptm", "iface_plddt"):
        for kind in ("raw", "delta"):
            key = f"{kind}_{met}"
            ranks, sizes = [], []
            for rid, g in by.items():
                c = [x for x in g if x["label"] == "cognate"]
                d = [x for x in g if x["label"] == "decoy"]
                if not c or not d or np.isnan(c[0][key]):
                    continue
                sc = [c[0][key]] + [x[key] for x in d if not np.isnan(x[key])]
                ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
                sizes.append(len(sc))
            ranks = np.array(ranks, float); sizes = np.array(sizes, float)
            if len(ranks) < 5:
                continue
            exp = (sizes + 1) / 2
            p = (stats.wilcoxon(ranks - exp, alternative="two-sided")[1]
                 if not np.allclose(ranks - exp, 0) else 1.0)
            out[key] = {"cognate_first": int((ranks == 1).sum()),
                        "n_receptors": int(len(ranks)),
                        "mean_rank": float(ranks.mean()),
                        "chance": float(exp.mean()), "p": float(p)}
            label = f"{met} ({'scramble-normalised' if kind == 'delta' else 'raw'})"
            print(f"{label:26} {int((ranks == 1).sum()):4d} {ranks.mean():10.2f}"
                  f" {exp.mean():7.2f} {p:8.4f}")

    print("\nScramble normalisation wins only if delta beats raw on the same metric.")
    for met in ("iptm", "iface_plddt"):
        r, d = out.get(f"raw_{met}"), out.get(f"delta_{met}")
        if r and d:
            better = d["mean_rank"] < r["mean_rank"]
            print(f"  {met:12} raw {r['mean_rank']:.2f} -> delta {d['mean_rank']:.2f}"
                  f"   {'IMPROVES' if better else 'does not improve'}")
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "scramble_norm"))
    ap.add_argument("--scores", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--msa-dir", default=str(REPO_ROOT / "artifacts" /
                                             "pdb_binders_b2_n22" / "msa_cache"))
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    jobs = build(args.scores, args.seed)
    if args.analyse_only:
        recs = json.loads((work / "scramble_norm_scores.json").read_text())
    else:
        recs = run(jobs, work, args.msa_dir, args.batch_size,
                   args.sampling_steps, args.recycling_steps, args.device)
    rows, summary = analyse(recs)
    (REPO_ROOT / "artifacts" / "scramble_norm_result.json").write_text(
        json.dumps({"per_candidate": rows, "summary": summary}, indent=2))
    print(f"\nwrote {REPO_ROOT / 'artifacts' / 'scramble_norm_result.json'}")


if __name__ == "__main__":
    main()
