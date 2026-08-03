"""Does the receptor-side signal depend on the actual binding site?

The side-split analysis showed the receptor's own interface residues are placed
more confidently when the cognate peptide is present (+2.38 pLDDT, p = 6e-5) --
which peptide foldability cannot explain. This asks the follow-up: is that
because of the *binding site*, or would any receptor do?

Four arms, all with the same cognate peptide:

  real        receptor unchanged                              baseline
  iface_ala   interface residues mutated to alanine           site destroyed,
                                                              fold largely intact
  surf_ala    an equal number of the most solvent-exposed
              NON-interface residues mutated to alanine       control for "any
                                                              mutation lowers it"
  scrambled   receptor sequence shuffled                      fold destroyed

**iface_ala vs surf_ala is the comparison that matters.** Scrambling the
receptor also destroys its fold, so a drop there is uninformative on its own --
it is included because it was the originally proposed control, not because it
discriminates.

Every arm is folded in single-sequence mode: a mutated or shuffled receptor
cannot share the original MSA, so the baseline is run without one too rather
than letting MSA presence confound the comparison.

Usage:
    python src/receptor_site_controls.py
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
from Bio.PDB.SASA import ShrakeRupley
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from interface_side_split import rep_atom  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_CUTOFF = 8.0


MAX_SITE = 15


def interface_positions(model, top_n=MAX_SITE):
    """0-based receptor positions closest to the peptide.

    An 8 A CB-CB cutoff marks ~half the receptor as "interface" for a small
    domain (51 of 109 residues for 1YCR), and alanine-substituting that many
    destroys the fold -- reintroducing the confound this control exists to
    avoid. Taking the `top_n` nearest residues instead keeps the perturbation
    local to the binding groove.
    """
    chains = [c for c in model]
    ra = [(r, rep_atom(r)) for r in chains[0] if rep_atom(r) is not None]
    rb = [(r, rep_atom(r)) for r in chains[1] if rep_atom(r) is not None]
    if not ra or not rb:
        return None
    d = np.linalg.norm(np.array([a.coord for _, a in ra])[:, None, :]
                       - np.array([a.coord for _, a in rb])[None, :, :], axis=-1)
    closest = d.min(axis=1)
    contacting = [i for i in range(len(ra)) if closest[i] < CONTACT_CUTOFF]
    contacting.sort(key=lambda i: closest[i])
    return contacting[:top_n]


def exposed_non_interface(pdb, iface, n, parser):
    """The n most solvent-exposed receptor positions that are not at the interface."""
    struct = parser.get_structure("s", str(pdb))[0]
    chain = [c for c in struct][0]
    for other in [c.id for c in struct if c.id != chain.id]:
        struct.detach_child(other)
    ShrakeRupley().compute(struct, level="R")
    res = [r for r in chain if rep_atom(r) is not None]
    order = sorted((i for i in range(len(res)) if i not in set(iface)),
                   key=lambda i: -res[i].sasa)
    return order[:n]


def mutate(seq, positions, to="A"):
    chars = list(seq)
    for p in positions:
        if 0 <= p < len(chars):
            chars[p] = to
    return "".join(chars)


def build(scores_path, work, seed):
    pairs = [p for p in json.loads(Path(scores_path).read_text())
             if not np.isnan(p.get("score", float("nan")))
             and p["label"] == "cognate"]
    parser = PDBParser(QUIET=True)
    index = {}
    for pdb in sorted(Path(work).rglob("*_model_0.pdb")):
        if "batch_" in str(pdb):
            index.setdefault(pdb.parent.name, pdb)

    rng = random.Random(seed)
    jobs = []
    for p in pairs:
        pdb = index.get(p["name"])
        if pdb is None:
            continue
        try:
            model = parser.get_structure("x", str(pdb))[0]
        except Exception:
            continue
        iface = interface_positions(model)
        if not iface:
            continue
        surf = exposed_non_interface(pdb, iface, len(iface), parser)
        rec = p["receptor"]
        shuffled = list(rec); rng.shuffle(shuffled)
        arms = {"real": rec,
                "iface_ala": mutate(rec, iface),
                "surf_ala": mutate(rec, surf),
                "scrambled": "".join(shuffled)}
        for arm, seq in arms.items():
            jobs.append({"job": f"{p['receptor_id']}_{arm}", "arm": arm,
                         "receptor_id": p["receptor_id"], "receptor": seq,
                         "peptide": p["peptide"], "n_mutated": len(iface)})
    return jobs


def fold(inputs, out, device):
    cmd = [sys.executable, "-m", "boltz.main", "predict", str(inputs),
           "--out_dir", str(out), "--model", "boltz2",
           "--accelerator", "gpu" if device == "mps" else "cpu",
           "--recycling_steps", "1", "--sampling_steps", "10",
           "--output_format", "pdb", "--override"]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print((proc.stdout + proc.stderr)[-1200:], file=sys.stderr)
        raise RuntimeError("fold failed")
    return out / f"boltz_results_{inputs.name}", time.perf_counter() - t0


def score(results, names):
    from interface_side_split import sides
    parser = PDBParser(QUIET=True)
    out = {}
    for n in names:
        d = results / "predictions" / n
        pdb = d / f"{n}_model_0.pdb"
        conf = d / f"confidence_{n}_model_0.json"
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
    by_arm = {}
    for r in recs:
        by_arm.setdefault(r["arm"], {})[r["receptor_id"]] = r
    base = by_arm.get("real", {})

    print(f"\n{'=' * 74}\nReceptor-side controls (paired against the real receptor)\n{'=' * 74}")
    print(f"\n{'arm':12} {'n':>3} {'receptor_side':>14} {'vs real':>9} {'p':>9}"
          f" {'iface_plddt':>12} {'vs real':>9}")
    print("-" * 74)
    summary = {}
    for arm in ("real", "iface_ala", "surf_ala", "scrambled"):
        if arm not in by_arm:
            continue
        rids = [r for r in by_arm[arm] if r in base]
        rs = np.array([by_arm[arm][r]["receptor_side"] for r in rids])
        ip = np.array([by_arm[arm][r]["iface_plddt"] for r in rids])
        rs0 = np.array([base[r]["receptor_side"] for r in rids])
        ip0 = np.array([base[r]["iface_plddt"] for r in rids])
        if arm == "real":
            print(f"{arm:12} {len(rids):3d} {rs.mean():14.2f} {'-':>9} {'-':>9}"
                  f" {ip.mean():12.2f} {'-':>9}")
            summary[arm] = {"receptor_side": float(rs.mean()),
                            "iface_plddt": float(ip.mean()), "n": len(rids)}
            continue
        d_rs, d_ip = rs - rs0, ip - ip0
        p = stats.ttest_rel(rs, rs0).pvalue
        print(f"{arm:12} {len(rids):3d} {rs.mean():14.2f} {d_rs.mean():+9.2f}"
              f" {p:9.5f} {ip.mean():12.2f} {d_ip.mean():+9.2f}")
        summary[arm] = {"receptor_side": float(rs.mean()),
                        "delta_receptor_side": float(d_rs.mean()),
                        "p_vs_real": float(p),
                        "iface_plddt": float(ip.mean()),
                        "delta_iface_plddt": float(d_ip.mean()), "n": len(rids)}

    ia, sa = summary.get("iface_ala"), summary.get("surf_ala")
    if ia and sa:
        rids = [r for r in by_arm["iface_ala"] if r in by_arm["surf_ala"]]
        a = np.array([by_arm["iface_ala"][r]["receptor_side"] for r in rids])
        b = np.array([by_arm["surf_ala"][r]["receptor_side"] for r in rids])
        p = stats.ttest_rel(a, b).pvalue
        print("\nThe comparison that matters -- interface-Ala vs surface-Ala:")
        print(f"  receptor_side {a.mean():.2f} vs {b.mean():.2f}  "
              f"diff {a.mean() - b.mean():+.2f}  p = {p:.4f}")
        if p < 0.05 and a.mean() < b.mean():
            print("  -> Mutating the BINDING SITE costs more than mutating an equal")
            print("     number of exposed residues elsewhere. The receptor-side")
            print("     signal is site-specific.")
        else:
            print("  -> No site-specific penalty: alanine substitution at the")
            print("     interface is no worse than elsewhere, so the receptor-side")
            print("     signal is not demonstrably about the binding site.")
        summary["iface_vs_surf"] = {"iface_ala": float(a.mean()),
                                    "surf_ala": float(b.mean()),
                                    "diff": float(a.mean() - b.mean()),
                                    "p": float(p), "n": len(rids)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "receptor_controls"))
    ap.add_argument("--src", default=str(REPO_ROOT / "artifacts" / "pdb_binders_b2_n22"))
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    store = work / "receptor_control_scores.json"
    jobs = build(Path(args.src) / "pdb_binder_scores.json", args.src, args.seed)
    print(f"{len(jobs)} folds ({len(jobs) // 4} receptors x 4 arms), single-sequence")

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
                (inputs / f"{j['job']}.yaml").write_text(
                    "version: 1\nsequences:\n"
                    f"  - protein:\n      id: A\n      sequence: {j['receptor']}\n"
                    f"      msa: empty\n"
                    f"  - protein:\n      id: B\n      sequence: {j['peptide']}\n"
                    f"      msa: empty\n")
            res, el = fold(inputs, bdir, args.device)
            got = score(res, [j["job"] for j in chunk])
            for j in chunk:
                if j["job"] in got:
                    recs.append({**{k: j[k] for k in
                                    ("job", "arm", "receptor_id", "n_mutated")},
                                 **got[j["job"]]})
            store.write_text(json.dumps(recs, indent=2))
            print(f"  batch {start // args.batch_size}: {len(got)}/{len(chunk)} "
                  f"in {el:.0f}s", flush=True)
            shutil.rmtree(bdir, ignore_errors=True)

    summary = analyse(recs)
    (REPO_ROOT / "artifacts" / "receptor_controls_result.json").write_text(
        json.dumps({"per_fold": recs, "summary": summary}, indent=2))
    print(f"\nwrote {REPO_ROOT / 'artifacts' / 'receptor_controls_result.json'}")


if __name__ == "__main__":
    main()
