"""Does the interface-pLDDT signal survive on a model trained for few-step sampling?

Sections 7.6 and 7.7 rest on folds taken at 10 sampling steps from Boltz-2, a
model whose default is 200. Section 1.5 records that settings gap as stated but
unresolved, and it is the most obvious objection to the whole result: the signal
could be an artefact of under-sampled folding.

DeCAF-Boltz is distilled to be accurate *at* 10 steps. Running the identical
pairs through it separates the two possibilities. A smoke test on 1YCR gave
interface pLDDT 83.79 against 49.75 for Boltz-2 -- far higher confidence, but
confidence alone does not rank. What matters is whether a cognate still beats
its own scramble, which holds composition and length fixed.

Inputs are taken verbatim from the n=22 panel -- the same receptors, the same
cognate peptides and the same scrambled sequences already scored in 7.6 -- so
the model is the only thing that changes.

Caveat carried into the write-up: DeCAF distils Boltz-1, so this alters the base
model *and* the sampling regime at once. A difference cannot be attributed to
one without a Boltz-1 arm.

Usage:
    python src/decaf_scramble_test.py --batch-size 12
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

from interface_side_split import sides  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DECAF_HOME = Path.home() / ".boltz" / "decaf"


def build(scores_path):
    """Cognates, their scrambles and their decoys, exactly as scored in 7.6.

    Scrambles test order sensitivity; decoys test receptor specificity -- a
    decoy is a genuine binder of a *different* receptor, so ranking above it
    requires more than recognising a peptide-shaped object.
    """
    pairs = [p for p in json.loads(Path(scores_path).read_text())
             if not np.isnan(p.get("score", float("nan")))
             and p["label"] in ("cognate", "scrambled", "decoy")]
    return [{"job": p["name"], "receptor_id": p["receptor_id"],
             "label": p["label"], "receptor": p["receptor"],
             "peptide": p["peptide"], "boltz2_iptm": p["score"]}
            for p in pairs]


def fold(inputs, out, ckpt, sampling, recycling, base="decaf", write_pae=False,
         msa_depth=32):
    """Fold one batch under `base`.

    The stock arms exist to de-confound: DeCAF changes both the base model
    (Boltz-1 teacher rather than Boltz-2) and the sampling regime (trained for
    10 steps rather than run at 10 of an intended 200). Running stock Boltz-1
    on the same pairs, same device and same step count isolates the second.
    """
    if base == "decaf":
        cmd = [sys.executable, str(REPO_ROOT / "src" / "decaf_runner.py"), "predict",
               str(inputs), "--checkpoint", str(ckpt), "--model", "boltz1"]
    else:
        cmd = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
               "--model", base]
    cmd += ["--sampling_steps", str(sampling), "--diffusion_samples", "1",
            "--recycling_steps", str(recycling), "--accelerator", "gpu",
            "--output_format", "pdb", "--override", "--out_dir", str(out)]
    # msa_depth=0 means take the alignment whole. Subsampling to 32 rows is one
    # of the three reductions Section 7.13 found were suppressing the effect, so
    # it has to be switchable rather than baked in.
    if msa_depth:
        cmd += ["--subsample_msa", "--num_subsampled_msa", str(msa_depth),
                "--max_msa_seqs", str(msa_depth)]
    if base == "decaf":
        cmd.append("--no_kernels")
    if write_pae:
        # PAE-derived readouts (mPAE, ipSAE, pDockQ2) need the full matrix,
        # which boltz does not write by default.
        cmd.append("--write_full_pae")
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(combined[-1500:], file=sys.stderr)
        raise RuntimeError("decaf fold failed")
    # The fork falls back to the teacher sampler silently if the head is not
    # recognised, which would look like a DeCAF result rather than a bug.
    if base == "decaf" and "Detected Decaf checkpoint" not in combined:
        raise RuntimeError("DecafSampler did not engage -- refusing to score "
                           "teacher output as DeCAF")
    return out / f"boltz_results_{inputs.name}", time.perf_counter() - t0


def score(results, names):
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


def analyse(recs):
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)

    print("\n" + "=" * 74)
    print(f"{len(recs)} folds across {len(by)} receptors")
    print("=" * 74)
    print(f"\n{'metric':16} {'cognate':>9} {'scrambled':>10} {'cog-scr':>9}"
          f" {'95% CI':>20} {'p':>9}")
    print("-" * 74)

    summary = {}
    for met in ("iptm", "iface_plddt", "receptor_side", "peptide_side"):
        cog = np.array([r[met] for r in recs if r["label"] == "cognate"])
        scr = np.array([r[met] for r in recs if r["label"] == "scrambled"])
        diffs = []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"]
            s = [x for x in g if x["label"] == "scrambled"]
            if c and s:
                diffs += [c[0][met] - x[met] for x in s]
        diffs = np.array(diffs)
        if len(diffs) < 3:
            continue
        p = stats.ttest_1samp(diffs, 0).pvalue
        ci = stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(),
                              scale=diffs.std(ddof=1) / np.sqrt(len(diffs)))
        summary[met] = {"cognate": float(cog.mean()), "scrambled": float(scr.mean()),
                        "cognate_minus_own_scramble": float(diffs.mean()),
                        "ci95": [float(ci[0]), float(ci[1])], "p": float(p),
                        "n_pairs": int(len(diffs))}
        print(f"{met:16} {cog.mean():9.3f} {scr.mean():10.3f} {diffs.mean():+9.3f}"
              f" [{ci[0]:+7.3f},{ci[1]:+7.3f}] {p:9.5f}")

    print("\nSection 7.6 / 7.7 on Boltz-2, for comparison:")
    print("  iface_plddt    49.60      46.31     +3.30  [ +2.13, +4.47]  0.00000")
    print("  receptor_side  51.94      49.56     +2.38  [   —  ,   —  ]  0.00006")
    print("  iptm            0.502      0.489    +0.013 [ -0.019,+0.044] 0.41600")

    # -- receptor specificity: cognate against its own decoys -----------------
    have_decoys = any(r["label"] == "decoy" for r in recs)
    if have_decoys:
        print(f"\n{'metric':16} {'#1':>4} {'mean rank':>10} {'chance':>7} {'p':>9}"
              f"   {'Boltz-2 rank':>13}")
        print("-" * 74)
        b2 = {"iptm": 2.00, "iface_plddt": 1.91}
        for met in ("iptm", "iface_plddt", "receptor_side"):
            ranks, sizes = [], []
            for g in by.values():
                c = [x for x in g if x["label"] == "cognate"]
                d = [x for x in g if x["label"] == "decoy"]
                if not c or not d:
                    continue
                sc = [c[0][met]] + [x[met] for x in d]
                ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
                sizes.append(len(sc))
            if len(ranks) < 5:
                continue
            ranks = np.array(ranks, float); sizes = np.array(sizes, float)
            exp = (sizes + 1) / 2
            p = (stats.wilcoxon(ranks - exp, alternative="two-sided")[1]
                 if not np.allclose(ranks - exp, 0) else 1.0)
            summary.setdefault(met, {}).update({
                "mean_rank": float(ranks.mean()), "chance": float(exp.mean()),
                "cognate_first": int((ranks == 1).sum()),
                "p_rank": float(p), "n_receptors": int(len(ranks))})
            ref = f"{b2[met]:.2f}" if met in b2 else "-"
            print(f"{met:16} {int((ranks == 1).sum()):4d} {ranks.mean():10.2f}"
                  f" {exp.mean():7.2f} {p:9.4f}   {ref:>13}")
        print("\n  Boltz-2 reference: ipTM p = 0.034, interface pLDDT p = 0.027")
        print("  (both suggestive; neither cleared Bonferroni over six metrics)")

    ip = summary.get("iface_plddt")
    if ip:
        print("\nVerdict on the settings objection:")
        if ip["p"] < 0.05 and ip["cognate_minus_own_scramble"] > 0:
            print("  Interface pLDDT still separates a cognate from its own scramble on")
            print("  a model built for 10-step sampling. The signal is NOT an artefact")
            print("  of under-sampled folding.")
        else:
            print("  Interface pLDDT does NOT separate on a model built for 10-step")
            print("  sampling. This is evidence the Section 7.6 result was an artefact")
            print("  of running Boltz-2 far below its intended sampling budget.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--ckpt", default=str(DECAF_HOME / "decaf_conf_ckpt.ckpt"))
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "decaf_scramble"))
    ap.add_argument("--scores", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--msa-dir", default=str(REPO_ROOT / "artifacts" /
                                             "pdb_binders_b2_n22" / "msa_cache"))
    ap.add_argument("--base", default="decaf", choices=["decaf", "boltz1", "boltz2"],
                    help="decaf = few-step-trained; boltz1/boltz2 = stock teacher "
                         "run at the same reduced step count")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    store = work / f"{args.base}_scramble_scores.json"
    jobs = build(args.scores)
    print(f"{len(jobs)} folds: {sum(j['label']=='cognate' for j in jobs)} cognate, "
          f"{sum(j['label']=='scrambled' for j in jobs)} scrambled")

    recs = json.loads(store.read_text()) if store.exists() else []
    if not args.analyse_only:
        done = {r["job"] for r in recs}
        todo = [j for j in jobs if j["job"] not in done]
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            bdir = work / f"b{start // args.batch_size:02d}"
            inputs = bdir / "inputs"
            if inputs.exists():
                shutil.rmtree(inputs)
            inputs.mkdir(parents=True)
            for j in chunk:
                msa = Path(args.msa_dir) / f"{j['receptor_id']}.csv"
                rline = f"      msa: {msa}\n" if msa.exists() else "      msa: empty\n"
                (inputs / f"{j['job']}.yaml").write_text(
                    "version: 1\nsequences:\n"
                    f"  - protein:\n      id: A\n      sequence: {j['receptor']}\n{rline}"
                    f"  - protein:\n      id: B\n      sequence: {j['peptide']}\n"
                    f"      msa: empty\n")
            res, el = fold(inputs, bdir, args.ckpt, args.sampling_steps,
                           args.recycling_steps, args.base)
            got = score(res, [j["job"] for j in chunk])
            for j in chunk:
                if j["job"] in got:
                    recs.append({**{k: j[k] for k in
                                    ("job", "receptor_id", "label", "boltz2_iptm")},
                                 **got[j["job"]]})
            store.write_text(json.dumps(recs, indent=2))
            print(f"  batch {start // args.batch_size}: {len(got)}/{len(chunk)} "
                  f"in {el:.0f}s ({el / max(1, len(chunk)):.0f}s each)", flush=True)
            shutil.rmtree(bdir, ignore_errors=True)

    summary = analyse(recs)
    (REPO_ROOT / "artifacts" / f"{args.base}_scramble_result.json").write_text(
        json.dumps({"per_fold": recs, "summary": summary}, indent=2))
    print(f"\nwrote {REPO_ROOT / 'artifacts'}/{args.base}_scramble_result.json")


if __name__ == "__main__":
    main()
