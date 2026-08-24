"""The scramble control on a second model family.

Every result in this project comes from one model family. A reviewer's first
question is whether ipTM's indifference to sequence order is a property of
cofolding confidence or a property of Boltz, and nothing here can answer it.

This folds the identical 132-fold panel -- same cognates, same scrambles, same
decoys, same sequences -- under Chai-1, and scores the output with this project's
own interface-pLDDT implementation. The readout code is shared, so the only thing
that varies between this arm and Section 7.13's is the model.

Chai-1 does not run on Apple Silicon: MPSGraph cannot compile the broadcast
outer-product matmul in the pair representation ('mps.matmul' op contracting
dimensions differ), and on CPU one 66-residue complex took 2h47m. This arm needs
CUDA, which is why it did not exist until the panel moved to a rented GPU.

Usage:
    python src/chai_arm.py --panel hf_dataset/sequences.csv --out artifacts/chai_arm.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fasta(path, receptor, peptide):
    path.write_text(f">protein|name=receptor\n{receptor}\n"
                    f">protein|name=peptide\n{peptide}\n")


def interface_plddt(cif, cutoff=8.0):
    """Section 7.7's readout, unchanged: mean CA pLDDT over contacting residues.

    Representative atom is CB, or CA for glycine. Chai writes per-residue pLDDT
    into the B-factor column exactly as Boltz does, so the same code reads both.
    """
    import gemmi
    import numpy as np
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    if len(st) == 0 or len(st[0]) < 2:
        return None
    chains = sorted(st[0], key=len)
    pep, rec = chains[0], chains[-1]

    def rep(res):
        a = res.find_atom("CB", "*") or res.find_atom("CA", "*")
        return a

    pr = [(r, rep(r)) for r in pep]
    rr = [(r, rep(r)) for r in rec]
    pr = [(r, a) for r, a in pr if a is not None]
    rr = [(r, a) for r, a in rr if a is not None]

    pl, rl = set(), set()
    for i, (r1, a1) in enumerate(pr):
        for j, (r2, a2) in enumerate(rr):
            if a1.pos.dist(a2.pos) <= cutoff:
                pl.add(i); rl.add(j)
    if not pl or not rl:
        return None

    def ca_b(res):
        a = res.find_atom("CA", "*")
        return a.b_iso if a is not None else None

    vals = [ca_b(pr[i][0]) for i in pl] + [ca_b(rr[j][0]) for j in rl]
    vals = [v for v in vals if v is not None]
    rec_vals = [ca_b(rr[j][0]) for j in rl]
    rec_vals = [v for v in rec_vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.mean(rec_vals)) if rec_vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(REPO_ROOT / "hf_dataset" / "sequences.csv"))
    ap.add_argument("--labels", default=str(REPO_ROOT / "hf_dataset" / "folds.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "chai_arm.json"))
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "chai_arm"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arm", default="boltz2_reduced",
                    help="which arm to take labels from when the labels file "
                         "contains more than one")
    ap.add_argument("--labels-keep", default="cognate,scrambled",
                    help="which classes to fold. The scramble control needs only "
                         "cognate and scrambled; decoys serve the ranking test, "
                         "which is not what a second model family is here to "
                         "settle. Pass 'cognate,scrambled,decoy' for the lot.")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from chai_lab.chai1 import run_inference

    seqs = pd.read_csv(args.panel)
    lab = pd.read_csv(args.labels)
    # The combined folds.csv carries every arm, so one must be selected; a
    # purpose-built labels file carries one already and must not be filtered.
    if "arm" in lab.columns and lab["arm"].nunique() > 1:
        lab = lab[lab.arm == args.arm]
    lab = lab[["name", "receptor_id", "label"]]
    panel = seqs.merge(lab, on="name", how="inner")
    keep = [x.strip() for x in args.labels_keep.split(",") if x.strip()]
    panel = panel[panel.label.isin(keep)]
    if args.limit:
        panel = panel.head(args.limit)
    print(f"panel: {len(panel)} folds ({'+'.join(keep)}), "
          f"{panel.receptor_id.nunique()} receptors")

    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    store = Path(args.out)
    recs = json.loads(store.read_text()) if store.exists() else []
    done = {r["name"] for r in recs}

    t0 = time.time()
    for i, row in enumerate(panel.itertuples(), 1):
        if row.name in done:
            continue
        d = work / row.name
        d.mkdir(parents=True, exist_ok=True)
        # Chai refuses to write into an existing output directory, so a fold that
        # failed once fails forever on resume unless the directory goes with it.
        import shutil as _sh
        _sh.rmtree(d / "out", ignore_errors=True)
        fa = d / "in.fasta"
        write_fasta(fa, row.receptor_seq, row.peptide_seq)
        try:
            run_inference(fasta_file=fa, output_dir=d / "out",
                          num_trunk_recycles=args.recycles,
                          num_diffn_timesteps=args.steps,
                          num_diffn_samples=1, use_esm_embeddings=True,
                          use_msa_server=True, device=torch.device("cuda"),
                          seed=0, low_memory=True)
        except Exception as exc:
            print(f"  {row.name}: FAILED {str(exc)[:90]}", flush=True)
            continue
        cifs = sorted((d / "out").glob("*.cif"))
        if not cifs:
            print(f"  {row.name}: no structure written", flush=True)
            continue
        got = interface_plddt(cifs[0])
        if got is None:
            print(f"  {row.name}: no interface", flush=True)
            continue
        iface, rec_side = got
        recs.append({"name": row.name, "receptor_id": row.receptor_id,
                     "label": row.label, "iface_plddt": iface,
                     "receptor_side": rec_side})
        store.write_text(json.dumps(recs, indent=1))
        el = time.time() - t0
        print(f"  [{i}/{len(panel)}] {row.name} {row.label:9s} "
              f"iface {iface:6.2f}  ({el/60:.1f} min elapsed)", flush=True)

    print(f"\nwrote {store} with {len(recs)} folds")


if __name__ == "__main__":
    main()
