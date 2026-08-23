"""Section 7.11's backbone finding, restated in PoseBusters' standard checks.

7.11.1 measured consecutive CA-CA distances directly and found 14% of them
physically plausible in Boltz-2 structures folded at ten sampling steps. That is
a bespoke metric. PoseBusters is the field's standard validity suite, so running
it on the same structures says the same thing in vocabulary a reviewer already
accepts -- or contradicts it, which would be worth knowing before the preprint
goes anywhere.

The peptide chain is treated as the molecule under test. PoseBusters' `mol`
config runs intramolecular checks only -- connectivity, bond lengths and angles,
internal clash, ring and double-bond flatness, internal energy -- which is
exactly the claim at issue: whether what comes back is a connected chain.

Scope, and it is a real limitation: the full-settings and DeCAF structures that
7.11 compares against did not survive the disk cleanup, so this characterises the
reduced-settings regime alone and cannot reproduce the contrast. Refolding a
subset at 200 steps would restore it.

Usage:
    python src/posebusters_check.py [--limit N] [--glob PATTERN]
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import gemmi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
DEFAULT_GLOB = "pdb_binders_b2_n22/**/predictions/**/*.pdb"
BATCH_ONLY = "batch_"


def peptide_chain(path, work):
    """Write the shorter chain of a two-chain prediction to its own PDB.

    Boltz writes receptor first and peptide second, but selecting by length
    rather than by position keeps this correct if that ever changes.
    """
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if len(st[0]) < 2:
        return None, None
    ch = min(st[0], key=len)
    sel = gemmi.Structure()
    sel.add_model(gemmi.Model("1"))
    sel[0].add_chain(ch.clone())
    sel.setup_entities()
    out = work / f"{path.parent.name}_{ch.name}.pdb"
    sel.write_pdb(str(out))
    return out, len(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ART / "posebusters.json"))
    args = ap.parse_args()

    from posebusters import PoseBusters

    paths = sorted(ART.glob(args.glob))
    # msa_fetch probes are 5-step throwaways folded only to trigger an
    # alignment download; scoring them as if they were benchmark structures
    # would report the wrong regime entirely.
    if "batch" not in args.glob:
        paths = [p for p in paths if BATCH_ONLY in str(p)]
    paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no structures matched {args.glob}")
    print(f"structures : {len(paths)}")

    work = REPO_ROOT / ".pb_work"
    work.mkdir(exist_ok=True)
    peps, lengths = [], []
    for p in paths:
        out, n = peptide_chain(p, work)
        if out:
            peps.append(out)
            lengths.append(n)
    print(f"peptides   : {len(peps)}  ({min(lengths)}-{max(lengths)} residues)")

    # Fragment count first, because it decides how the PoseBusters table may
    # be read. RDKit perceives bonds in a PDB by distance, so a backbone bond
    # stretched past bonding range is not perceived as a bond at all -- and a
    # bond that does not exist cannot fail a length or angle check. Those checks
    # are therefore vacuously satisfied here, and quoting them as passes would
    # assert the opposite of what the structures show.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    frags = []
    for q, n in zip(peps, lengths):
        m = Chem.MolFromPDBFile(str(q), sanitize=False, removeHs=False)
        frags.append({"file": q.name, "residues": n,
                      "atoms": m.GetNumAtoms() if m else None,
                      "bonds": m.GetNumBonds() if m else None,
                      "fragments": len(Chem.GetMolFrags(m)) if m else None})
    intact = sum(1 for f in frags if f["fragments"] == 1)
    med_frag = sorted(f["fragments"] for f in frags if f["fragments"])[len(frags) // 2]
    print(f"connectivity: {intact}/{len(frags)} peptides are a single fragment; "
          f"median {med_frag} fragments")

    print("running PoseBusters (mol config, intramolecular checks) …")
    df = PoseBusters(config="mol").bust(peps)

    checks = [c for c in df.columns if df[c].dtype == bool]
    rows = []
    for c in checks:
        rate = float(df[c].mean() * 100)
        rows.append({"check": c, "pass_rate": rate,
                     "n_pass": int(df[c].sum()), "n": int(len(df))})
    rows.sort(key=lambda r: r["pass_rate"])

    VACUOUS = {"bond_lengths", "bond_angles", "aromatic_ring_flatness",
               "non-aromatic_ring_non-flatness", "double_bond_flatness",
               "internal_energy"}
    print(f"\n  {'check':34s}{'pass':>8s}{'':4s}n = {len(df)}")
    for r in rows:
        mark = "  <- vacuous, see note" if r["check"] in VACUOUS else ""
        print(f"  {r['check']:34s}{r['pass_rate']:7.1f}%{mark}")
    print("\n  Vacuous checks are computed only over bonds RDKit perceived, and")
    print("  a broken backbone bond is not perceived. They are not evidence of")
    print("  good geometry; the connectivity line above is the finding.")
    all_pass = float(df[checks].all(axis=1).mean() * 100)
    print(f"\n  {'ALL CHECKS':34s}{all_pass:7.1f}%")

    result = {"n_structures": len(peps), "source_glob": args.glob,
              "checks": rows, "all_pass_rate": all_pass,
              "vacuous_checks": sorted(VACUOUS),
              "connectivity": {"single_fragment": intact, "n": len(frags),
                               "median_fragments": med_frag},
              "per_structure": frags,
              "peptide_len_min": min(lengths), "peptide_len_max": max(lengths)}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
