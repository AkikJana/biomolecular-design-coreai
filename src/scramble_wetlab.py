"""Does the scramble control separate binders that were actually measured?

The scramble control is this work's central methodological contribution. Section
7.4 showed a confidence score can rank a peptide above an unrelated decoy while
being blind to sequence order, and every result since is reported against a
candidate's own permutations. It has only ever been tested against PDB proxies:
a cognate crystal pair counts as a positive because it was crystallised, not
because anything was measured binding.

Anthropic's release [25] permits the real test. Its designs were synthesised and
their binding measured, so a design that binds and a design that does not are
both known, and the question becomes direct: does subtracting a candidate's own
scrambles separate measured binders better than the raw score does?

A prediction, recorded before the folds are run so that the outcome cannot be
rationalised afterwards. These binders are 60 to 120 residue designed proteins,
not the 5 to 25 residue peptides of Section 7.4's panel. A permutation of a
100-residue protein does not fold at all, so both binders and non-binders should
receive an equally ruined scramble, and the subtraction may add nothing. If that
is what happens, the finding is not that the control fails but that it is
calibrated for short peptides, where a permutation remains a plausible ligand,
and carries no information where it is not.

Four targets are used, all single-chain and comparable in size to Section 7.4's
receptors, with enough of both labels to measure anything: RBX1 (108 aa), PD-L1
(115), TrkA (101) and BHRF1 (157).

Usage:
    python src/scramble_wetlab.py --per-class 6 --scrambles 2
"""

import argparse
import json

import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

ART = REPO_ROOT / "artifacts"
WORK = ART / "scramble_wetlab"
BINDER_DATA = ART / "anthropic_binder"
TARGETS = ["RBX1", "PD-L1", "TrkA", "BHRF1"]


def target_constructs():
    """Single-chain target sequences, keyed by the summary table's target name."""
    txt = (BINDER_DATA / "target_constructs.fasta").read_text()
    out, name = {}, None
    for line in txt.splitlines():
        if line.startswith(">"):
            name = line[1:]
        elif name:
            out[name] = out.get(name, "") + line.strip()
    keyed = {}
    for header, seq in out.items():
        if "target=" not in header or "nucleic" in header:
            continue
        t = header.split("target=")[1].split()[0].strip()
        copies = int(header.split("copies=")[1].split()[0]) if "copies=" in header else 1
        if copies == 1:
            keyed[t] = seq
    return keyed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6,
                    help="designs sampled per target per label")
    ap.add_argument("--scrambles", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    # 7.19.5 names a full-settings repeat as an outstanding limitation: the
    # original ran DeCAF at ten sampling steps, and 7.16 measured that the
    # sampling budget carries most of the available effect. These make that
    # repeat runnable rather than aspirational.
    ap.add_argument("--base", default="decaf", choices=("decaf", "boltz1", "boltz2"))
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--out", default=str(ART / "scramble_wetlab.json"))
    args = ap.parse_args()

    from screen_server import fetch_msa, scrambles_of
    from decaf_scramble_test import fold
    from anthropic_iface_plddt import iface_plddt, split_chains
    from Bio.PDB import PDBParser

    WORK.mkdir(parents=True, exist_ok=True)
    cons = target_constructs()
    d = pd.read_csv(BINDER_DATA / "design_summary.csv", low_memory=False)
    d = d[d["binder_final"].isin([True, False])]

    picks = []
    for t in TARGETS:
        if t not in cons:
            print(f"  {t}: no single-chain construct, skipped")
            continue
        g = d[d["target"] == t]
        for label in (True, False):
            sub = g[g["binder_final"] == label]
            take = sub.sample(min(args.per_class, len(sub)), random_state=args.seed)
            for _, r in take.iterrows():
                picks.append({"uuid": r["uuid"], "target": t, "y": int(label),
                              "seq": r["sequence"], "receptor": cons[t],
                              "ipsae": r["ipsae_min_boltz2"]})
    print(f"{len(picks)} designs across {len({p['target'] for p in picks})} targets; "
          f"{sum(p['y'] for p in picks)} measured binders")

    # every design is folded as delivered and against permutations of itself
    pairs, index = [], {}
    for i, p in enumerate(picks):
        pairs.append((f"d{i:03d}", p["receptor"], p["seq"]))
        index[f"d{i:03d}"] = (i, "design", 0)
        for j, s in enumerate(scrambles_of(p["seq"], args.scrambles)):
            pairs.append((f"d{i:03d}_s{j}", p["receptor"], s))
            index[f"d{i:03d}_s{j}"] = (i, "scramble", j)
    print(f"{len(pairs)} folds "
          f"({len(picks)} designs x (1 + {args.scrambles} scrambles))")

    msas = {}
    for t in {p["target"] for p in picks}:
        seq = cons[t]
        m, cached = fetch_msa(seq, WORK / f"msa_{t}")
        msas[t] = m
        print(f"  msa {t}: {'cached' if cached else 'fetched'}")

    for p in picks:
        p.setdefault("design", []); p.setdefault("scrambles", [])
    store = Path(args.out)
    done = set()
    if store.exists():
        prev = json.loads(store.read_text())
        for a, b in zip(picks, prev.get("picks", [])):
            if a["uuid"] == b["uuid"]:
                a["design"], a["scrambles"] = b["design"], b["scrambles"]
        done = set(prev.get("done", []))

    todo = [x for x in pairs if x[0] not in done]
    parser = PDBParser(QUIET=True)
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start:start + args.batch_size]
        bdir = WORK / f"b{start // args.batch_size:03d}"
        inputs = bdir / "inputs"
        shutil.rmtree(inputs, ignore_errors=True)
        inputs.mkdir(parents=True)
        for name, rec, binder in chunk:
            t = picks[index[name][0]]["target"]
            line = f"      msa: {msas[t]}\n" if msas[t] else "      msa: empty\n"
            (inputs / f"{name}.yaml").write_text(
                "version: 1\nsequences:\n"
                f"  - protein:\n      id: A\n      sequence: {rec}\n{line}"
                f"  - protein:\n      id: B\n      sequence: {binder}\n"
                f"      msa: empty\n")
        ckpt = str(Path.home() / ".boltz" / "decaf" / "decaf_conf_ckpt.ckpt")
        res, el = fold(inputs, bdir, ckpt, args.sampling_steps,
                       args.recycling_steps, args.base)
        got = 0
        for name, _, binder in chunk:
            f = res / "predictions" / name / f"{name}_model_0.pdb"
            if not f.exists():
                continue
            try:
                model = parser.get_structure("x", str(f))[0]
                sp = split_chains(model, len(binder))
                s = iface_plddt(*sp) if sp else None
            except Exception:                                      # noqa: BLE001
                s = None
            if s is None:
                continue
            i, kind, _ = index[name]
            (picks[i]["design"] if kind == "design"
             else picks[i]["scrambles"]).append(s["iface_plddt"])
            done.add(name)
            got += 1
        store.write_text(json.dumps({"picks": picks, "done": sorted(done)}, indent=2))
        print(f"  batch {start // args.batch_size}: {got}/{len(chunk)} in {el:.0f}s "
              f"({len(done)}/{len(pairs)} folds)", flush=True)
        shutil.rmtree(bdir, ignore_errors=True)

    print(f"\nwrote {store}")


if __name__ == "__main__":
    main()
