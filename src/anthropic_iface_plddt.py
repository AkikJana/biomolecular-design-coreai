"""Compute THIS work's readout on Anthropic's designs, against measured binding.

Section 7.18 compared ipSAE, because that is the score [25] released. It is not
this dissertation's recommendation. Section 8.2 recommends interface pLDDT, and
that readout is recoverable from the released structures: Boltz-2 writes its
per-residue pLDDT into the B-factor column, and there are 1,440 Boltz-2 models,
one per design. No folding is required.

The interface is computed exactly as Section 7.7 computes it -- an 8.0 A cutoff
between representative atoms, CB where present and CA otherwise, and interface
pLDDT as the mean CA pLDDT over the contacting residues of both sides. What
cannot be reused is `sides()` itself, which takes the first two chains as
receptor and peptide. Five of these targets are oligomeric: the 15-PGDH models
carry two 266-residue target chains and one 76-residue binder, so `sides()` would
have measured the target against itself and returned a confident number for it.

The binder chain is therefore identified by matching the released binder_length,
and every other polypeptide chain is pooled as the target.

Usage:
    python src/anthropic_iface_plddt.py --limit 1440
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts" / "anthropic_binder"
REPO = "Anthropic/claude-protein-binder-design"
CONTACT_CUTOFF = 8.0                     # Section 7.7's cutoff, unchanged
warnings.filterwarnings("ignore")


def rep_atom(res):
    if "CB" in res:
        return res["CB"]
    return res["CA"] if "CA" in res else None


def split_chains(model, binder_len, tol=2):
    """(target residues, binder residues), or None if the binder is ambiguous.

    The binder is the chain whose length matches the released binder_length. A
    small tolerance allows for terminal residues the predictor may drop. If two
    chains match equally well the design is skipped rather than guessed at.
    """
    chains = []
    for c in model:
        res = [r for r in c if r.id[0] == " " and "CA" in r]
        if res:
            chains.append((c.id, res))
    if len(chains) < 2:
        return None
    diffs = [abs(len(r) - binder_len) for _, r in chains]
    best = min(diffs)
    if best > tol or diffs.count(best) > 1:
        return None
    bi = diffs.index(best)
    binder = chains[bi][1]
    target = [r for i, (_, rs) in enumerate(chains) if i != bi for r in rs]
    return target, binder


def iface_plddt(target, binder):
    """Section 7.7's readout, with the target side pooled over its chains."""
    ta = [(r, rep_atom(r)) for r in target if rep_atom(r) is not None]
    ba = [(r, rep_atom(r)) for r in binder if rep_atom(r) is not None]
    if not ta or not ba:
        return None
    ca = np.array([a.coord for _, a in ta])
    cb = np.array([a.coord for _, a in ba])
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    mask = d < CONTACT_CUTOFF
    if not mask.any():
        return None
    tb = [ta[i][0]["CA"].get_bfactor() for i in range(len(ta)) if mask[i].any()]
    bb = [ba[j][0]["CA"].get_bfactor() for j in range(len(ba)) if mask[:, j].any()]
    if not tb or not bb:
        return None
    return {"receptor_side": float(np.mean(tb)),
            "peptide_side": float(np.mean(bb)),
            "iface_plddt": float(np.mean(tb + bb)),
            "n_rec_iface": len(tb), "n_pep_iface": len(bb),
            "peptide_whole": float(np.mean([r["CA"].get_bfactor() for r in binder]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all designs")
    ap.add_argument("--out", default=str(ART / "iface_plddt_boltz2.json"))
    args = ap.parse_args()

    from Bio.PDB import MMCIFParser
    from huggingface_hub import hf_hub_download

    d = pd.read_csv(ART / "design_summary.csv", low_memory=False)
    d = d[d["binder_final"].isin([True, False])].copy()
    if args.limit:
        d = d.head(args.limit)
    parser = MMCIFParser(QUIET=True)

    out, skipped = [], {"download": 0, "chains": 0, "iface": 0}
    for i, (_, r) in enumerate(d.iterrows(), 1):
        # structures_dir omits the repo's data/ prefix and carries a trailing
        # slash, and the filename encodes stoichiometry -- 1to1 for a monomeric
        # target, 1to2 and 1to3 for oligomers, rnp for the Cas9 ribonucleoprotein
        # -- so the variant is discovered rather than assumed.
        base = "data/" + str(r["structures_dir"]).strip("/")
        p = None
        for variant in ("1to1", "1to2", "1to3", "rnp"):
            try:
                p = hf_hub_download(REPO, f"{base}/predicted_boltz2_{variant}.cif",
                                    repo_type="dataset")
                break
            except Exception:                                      # noqa: BLE001
                continue
        if p is None:
            skipped["download"] += 1
            continue
        try:
            model = parser.get_structure("x", p)[0]
            sp = split_chains(model, int(r["binder_length"]))
        except Exception:                                          # noqa: BLE001
            sp = None
        if sp is None:
            skipped["chains"] += 1
            continue
        s = iface_plddt(*sp)
        if s is None:
            skipped["iface"] += 1
            continue
        out.append({"uuid": r["uuid"], "target": r["target"],
                    "y": int(r["binder_final"]),
                    "ipsae_min_boltz2": r["ipsae_min_boltz2"], **s})
        if i % 100 == 0:
            print(f"  {i}/{len(d)}  kept {len(out)}  skipped {sum(skipped.values())}",
                  flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nscored {len(out)} of {len(d)} designs")
    print(f"  skipped: {skipped}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
