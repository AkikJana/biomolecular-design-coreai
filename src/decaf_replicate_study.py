"""Is a few-step-distilled model more reproducible than its teacher?

Section 7.5 measured fold-to-fold noise entirely on Boltz-2 at 10 sampling
steps: pooled SD 0.0628 for ipTM, 1.917 for interface pLDDT, with 4 of 4
receptors changing the rank of their cognate across identical re-runs. Two
conclusions rest on it -- that single folds cannot support per-receptor claims,
and that 9-16 replicates are needed.

Both were measured in one regime. A model distilled to land accurately in ten
steps may be markedly less stochastic, and the variance decomposition now
depends on the answer: it currently applies the Boltz-2 noise term to the DeCAF
arm for want of a measurement, which is exactly the assumption this removes.

Design mirrors 7.5 exactly -- the same 24 complexes across the same 4 receptors,
4 identical unseeded re-runs -- so the two noise estimates are comparable.

Usage:
    python src/decaf_replicate_study.py --replicates 4
"""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from decaf_scramble_test import fold  # noqa: E402
from interface_side_split import sides  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DECAF_HOME = Path.home() / ".boltz" / "decaf"
# measured on Boltz-2 in Section 7.5, for comparison
BOLTZ2_SD = {"iptm": 0.0628, "iface_plddt": 1.9172}


def mirror_design(seed_store, scores_path):
    """The exact complexes of the Section 7.5 replicate study."""
    want = {r["name"] for r in json.loads(Path(seed_store).read_text())}
    pairs = [p for p in json.loads(Path(scores_path).read_text())
             if p["name"] in want]
    return [{"job": p["name"], "receptor_id": p["receptor_id"],
             "label": p["label"], "receptor": p["receptor"],
             "peptide": p["peptide"]} for p in pairs]


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


def report(recs):
    by = {}
    for r in recs:
        by.setdefault(r["job"], []).append(r)
    reps = sorted({r["replicate"] for r in recs})

    print("\n" + "=" * 74)
    print(f"DeCAF run-to-run variance ({len(by)} complexes x {len(reps)} re-runs)")
    print("=" * 74)
    print(f"\n{'metric':16} {'DeCAF SD':>10} {'Boltz-2 SD':>12} {'ratio':>8}"
          f" {'median range':>14}")
    print("-" * 74)

    summary = {}
    for met in ("iptm", "iface_plddt"):
        sds, rngs = [], []
        for vals in by.values():
            v = [x[met] for x in vals if not np.isnan(x.get(met, np.nan))]
            if len(v) > 1:
                sds.append(np.std(v, ddof=1)); rngs.append(max(v) - min(v))
        if not sds:
            continue
        pooled = float(np.sqrt(np.mean(np.square(sds))))
        ratio = pooled / BOLTZ2_SD[met]
        summary[met] = {"pooled_sd": pooled, "boltz2_sd": BOLTZ2_SD[met],
                        "ratio": float(ratio),
                        "median_range": float(np.median(rngs)),
                        "n_complexes": len(sds)}
        print(f"{met:16} {pooled:10.4f} {BOLTZ2_SD[met]:12.4f} {ratio:8.2f}"
              f" {np.median(rngs):14.4f}")

    # rank stability, the test that failed 4/4 on Boltz-2
    print("\nCognate rank among its own decoys, across identical re-runs:")
    byrec = {}
    for r in recs:
        byrec.setdefault(r["receptor_id"], {}).setdefault(r["replicate"], []).append(r)
    flips = {}
    for met in ("iptm", "iface_plddt"):
        stable = 0
        seqs = {}
        for rid, per_rep in sorted(byrec.items()):
            seq = []
            for rep in reps:
                g = per_rep.get(rep, [])
                c = [x for x in g if x["label"] == "cognate"]
                d = [x for x in g if x["label"] == "decoy"]
                if c and d:
                    sc = [c[0][met]] + [x[met] for x in d]
                    seq.append(1 + sum(v >= sc[0] for v in sc[1:]))
            if seq:
                seqs[rid] = seq
                stable += len(set(seq)) == 1
        flips[met] = {"stable": stable, "n": len(seqs), "sequences": seqs}
        detail = "  ".join(f"{k}:{v}" for k, v in seqs.items())
        print(f"  {met:14} stable for {stable}/{len(seqs)} receptors   {detail}")
    print("  Boltz-2 reference: ipTM stable for 0/4")

    summary["rank_stability"] = flips
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--ckpt", default=str(DECAF_HOME / "decaf_conf_ckpt.ckpt"))
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "decaf_replicates"))
    ap.add_argument("--seed-store", default=str(REPO_ROOT / "artifacts" /
                                                "seed_variance" / "seed_variance_scores.json"))
    ap.add_argument("--scores", default=str(REPO_ROOT / "artifacts" /
                                            "pdb_binders_b2_n22" / "pdb_binder_scores.json"))
    ap.add_argument("--msa-dir", default=str(REPO_ROOT / "artifacts" /
                                             "pdb_binders_b2_n22" / "msa_cache"))
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    store = work / "decaf_replicate_scores.json"
    jobs = mirror_design(args.seed_store, args.scores)
    print(f"{len(jobs)} complexes x {args.replicates} re-runs = "
          f"{len(jobs) * args.replicates} folds "
          f"(mirrors the Section 7.5 design exactly)")

    recs = json.loads(store.read_text()) if store.exists() else []
    if not args.analyse_only:
        done = {(r["job"], r["replicate"]) for r in recs}
        for rep in range(args.replicates):
            todo = [j for j in jobs if (j["job"], rep) not in done]
            for start in range(0, len(todo), args.batch_size):
                chunk = todo[start:start + args.batch_size]
                bdir = work / f"r{rep}b{start // args.batch_size}"
                inputs = bdir / "inputs"
                if inputs.exists():
                    shutil.rmtree(inputs)
                inputs.mkdir(parents=True)
                for j in chunk:
                    msa = Path(args.msa_dir) / f"{j['receptor_id']}.csv"
                    rline = (f"      msa: {msa}\n" if msa.exists()
                             else "      msa: empty\n")
                    (inputs / f"{j['job']}.yaml").write_text(
                        "version: 1\nsequences:\n"
                        f"  - protein:\n      id: A\n      sequence: {j['receptor']}\n"
                        f"{rline}"
                        f"  - protein:\n      id: B\n      sequence: {j['peptide']}\n"
                        f"      msa: empty\n")
                res, el = fold(inputs, bdir, args.ckpt, 10, 1, "decaf")
                got = score_dir(res, [j["job"] for j in chunk])
                for j in chunk:
                    if j["job"] in got:
                        recs.append({**{k: j[k] for k in
                                        ("job", "receptor_id", "label")},
                                     "replicate": rep, **got[j["job"]]})
                store.write_text(json.dumps(recs, indent=2))
                print(f"  rep {rep} batch {start // args.batch_size}: "
                      f"{len(got)}/{len(chunk)} in {el:.0f}s", flush=True)
                shutil.rmtree(bdir, ignore_errors=True)

    summary = report(recs)
    (REPO_ROOT / "artifacts" / "decaf_replicate_result.json").write_text(
        json.dumps({"per_fold": recs, "summary": summary}, indent=2))
    print(f"\nwrote {REPO_ROOT / 'artifacts'}/decaf_replicate_result.json")


if __name__ == "__main__":
    main()
