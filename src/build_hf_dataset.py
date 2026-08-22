"""Assemble the scramble-control panels and scores as a HuggingFace dataset.

The preprint's data-availability section is the weakest thing in it -- "available
on request" is not availability. This packages what a reader actually needs to
reproduce every control in the paper without folding anything: the two panels,
the per-fold scores for each inference arm, and the PoseBusters validity run.

Nothing here is a structure file. The predictions are 198 MB and most did not
survive a disk cleanup; the scores did, and the scores are what the controls are
computed from.

Arms are labelled from each artifact's own settings block where it records one,
and from the dissertation section that defines it otherwise. No arm's model or
settings are inferred.

Usage:
    python src/build_hf_dataset.py [--out DIR]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"

# artifact -> (arm name, panel, dissertation section defining it)
ARMS = [
    ("boltz1_scramble_result",      "boltz1_reduced",        "in-training", "7.8"),
    ("decaf_scramble_result",       "decaf_reduced",         "in-training", "7.8, 7.12"),
    ("rescore_metrics",             "boltz2_reduced",        "in-training", "7.4, 7.6"),
    ("settings_confound",           "boltz1_full",           "in-training", "7.13"),
    ("settings_confound_samp",      "boltz1_sampling_only",  "in-training", "7.16"),
    ("settings_confound_msa",       "boltz1_alignment_only", "in-training", "7.16"),
    ("settings_confound_recyc",     "boltz1_recycling_only", "in-training", "7.16"),
    ("heldout_panel_result",        "heldout_reduced",       "held-out",    "7.10"),
    ("heldout_panel_result_full",   "heldout_full_draw1",    "held-out",    "7.10.6"),
    ("heldout_panel_result_full2",  "heldout_full_draw2",    "held-out",    "7.10.6"),
    ("decaf_replicate_result",      "decaf_replicates",      "in-training", "7.5"),
]

COLS = ["arm", "panel", "section", "sampling_steps", "recycling", "msa_depth",
        "receptor_id", "name", "label", "peptide_from",
        "iptm", "iface_plddt", "receptor_side", "peptide_side", "peptide_whole",
        "n_rec_iface", "n_pep_iface"]


def sequences_from_inputs():
    """Receptor and peptide sequences for every in-training fold.

    The held-out panel's inputs did not survive a disk cleanup, so only the
    22-receptor in-training panel is covered here; the held-out receptors are
    listed by PDB ID in panels.csv and can be refetched from the RCSB.
    """
    # Restricted to the panel's own run directory. Pair names are global within
    # a run (batch_00 holds pair_000-003, batch_01 pair_004-007) but NOT across
    # runs: the distillation and reference-benchmark directories reuse pair_000
    # for a completely different complex, and so does the msa_fetch helper inside
    # this run. Globbing all of artifacts/ and deduplicating by filename silently
    # substituted those for the panel, so only batch_* is read.
    panel_run = REPO_ROOT / "artifacts" / "pdb_binders_b2_n22"
    rows, seen = [], set()
    for q in sorted(panel_run.glob("batch_*/inputs/pair_*.yaml")):
        if q.stem in seen:
            raise SystemExit(f"duplicate pair name within the panel run: {q}")
        seen.add(q.stem)
        try:
            d = yaml.safe_load(q.read_text())
        except Exception:
            continue
        chains = [c["protein"] for c in d.get("sequences", []) if "protein" in c]
        if len(chains) != 2:
            continue
        rec, pep = max(chains, key=lambda c: len(c["sequence"])), \
            min(chains, key=lambda c: len(c["sequence"]))
        rows.append({"name": q.stem, "receptor_seq": rec["sequence"],
                     "receptor_len": len(rec["sequence"]),
                     "peptide_seq": pep["sequence"],
                     "peptide_len": len(pep["sequence"])})
    return pd.DataFrame(rows) if rows else None


def load_arm(stem, arm, panel, section):
    d = json.loads((ART / f"{stem}.json").read_text())
    s = d.get("settings") or {}
    rows = []
    for r in (d.get("per_fold") or d["per_complex"]):
        row = {c: r.get(c) for c in COLS}
        row.update(arm=arm, panel=panel, section=section,
                   sampling_steps=s.get("sampling"), recycling=s.get("recycling"),
                   msa_depth=s.get("msa_depth"))
        rows.append(row)
    return pd.DataFrame(rows, columns=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "hf_dataset"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    frames = []
    for stem, arm, panel, section in ARMS:
        p = ART / f"{stem}.json"
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}")
        df = load_arm(stem, arm, panel, section)
        df.to_csv(out / "scores" / f"{arm}.csv", index=False)
        frames.append(df)
        print(f"  {arm:24s} {len(df):>4} folds  "
              f"{df.receptor_id.nunique()} receptors  -> scores/{arm}.csv")

    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(out / "folds.csv", index=False)
    print(f"\n  combined: {len(allf)} folds, {allf.arm.nunique()} arms -> folds.csv")

    # Sequences as they were actually folded, read from the Boltz input YAMLs.
    # Not from the deposited crystals: those give only the residues resolved in
    # the density, and the panel was built from full entity sequences, so the
    # two disagree on most receptors by a disordered terminus or two.
    seqs = sequences_from_inputs()
    if seqs is not None:
        seqs.to_csv(out / "sequences.csv", index=False)
        print(f"  sequences: {len(seqs)} chains over "
              f"{seqs.name.nunique()} folds -> sequences.csv")

    panels = (allf[["panel", "receptor_id"]].drop_duplicates()
              .sort_values(["panel", "receptor_id"]).reset_index(drop=True))
    both = panels.receptor_id.duplicated().sum()
    panels.to_csv(out / "panels.csv", index=False)
    print(f"  panels:   {len(panels)} rows, {panels.receptor_id.nunique()} unique "
          f"receptors ({both} in both panels) -> panels.csv")

    pb = ART / "posebusters.json"
    if pb.exists():
        d = json.loads(pb.read_text())
        pd.DataFrame(d["per_structure"]).to_csv(out / "posebusters.csv", index=False)
        (out / "posebusters_summary.json").write_text(
            json.dumps({k: v for k, v in d.items() if k != "per_structure"}, indent=2))
        print(f"  posebusters: {d['n_structures']} structures -> posebusters.csv")

    return out, allf


if __name__ == "__main__":
    main()
