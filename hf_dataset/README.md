---
license: cc-by-4.0
pretty_name: Scramble-control panels for cofolding confidence metrics
size_categories:
  - 1K<n<10K
tags:
  - protein-structure-prediction
  - cofolding
  - benchmarking
  - negative-controls
  - peptide-binders
configs:
  - config_name: folds
    data_files: folds.csv
    default: true
  - config_name: panels
    data_files: panels.csv
  - config_name: sequences
    data_files: sequences.csv
  - config_name: posebusters_reduced
    data_files: posebusters_reduced.csv
  - config_name: posebusters_converged
    data_files: posebusters_converged.csv
  - config_name: arm_boltz2_reduced
    data_files: scores/boltz2_reduced.csv
  - config_name: arm_boltz1_full
    data_files: scores/boltz1_full.csv
  - config_name: arm_panel59_full
    data_files: scores/panel59_full.csv
  - config_name: arm_panel59_reduced
    data_files: scores/panel59_reduced.csv
  - config_name: arm_chai1_full
    data_files: scores/chai1_full.csv
---

# Scramble-control panels for cofolding confidence metrics

Per-fold confidence scores for peptide–protein complexes, folded under Boltz-1,
Boltz-2, Chai-1 and a few-step-distilled model, with each cognate peptide scored
against **permutations of itself** as well as against unrelated decoys.

**2,456 folds across 16 inference arms and 75 receptors.**

A permutation — a *scramble* — preserves amino-acid composition and length
exactly and destroys only sequence order. Decoy comparisons cannot separate a
confidence metric that tracks binding from one that tracks composition; a
composition-matched permutation can. On these panels ipTM ranks cognates above
decoys while failing to distinguish them from their own permutations, and
permutations in fact outscore decoys.

## Files

| file | rows | what |
| :--- | ---: | :--- |
| `folds.csv` | 2,456 | every fold, every arm, one row each |
| `scores/<arm>.csv` | 96–132 | the same rows split by inference arm |
| `sequences.csv` | 132 | receptor and peptide sequences **as folded** |
| `panels.csv` | 103 | every panel by PDB ID, 75 unique receptors |
| `posebusters_reduced.csv` | 144 | structural validity, 10 sampling steps |
| `posebusters_converged.csv` | 354 | structural validity, 200 sampling steps |
| `posebusters_summary.json` | — | pass rates per check |

## Arms

`panel` is `in-training` (22 receptors, mostly pre-cutoff) or `held-out`
(22 receptors released after the model's training cutoff).

| arm | panel | sampling | recycling | MSA depth |
| :--- | :--- | ---: | ---: | :--- |
| `boltz2_reduced` | in-training | 10 | 1 | 32 |
| `boltz1_reduced` | in-training | 10 | 1 | 32 |
| `decaf_reduced` | in-training | 10 | 1 | 32 |
| `boltz1_full` | in-training | 200 | 3 | full |
| `boltz1_sampling_only` | in-training | 200 | 1 | 32 |
| `boltz1_alignment_only` | in-training | 10 | 1 | full |
| `boltz1_recycling_only` | in-training | 10 | 3 | 32 |
| `heldout_reduced` | held-out | 10 | 1 | 32 |
| `heldout_full_draw1` | held-out | 200 | 3 | full |
| `heldout_full_draw2` | held-out | 200 | 3 | full |
| `decaf_replicates` | in-training | 10 | 1 | 32 |
| `boltz1_full_gpu` | in-training | 200 | 3 | full |
| `heldout_full_draw3` | held-out | 200 | 3 | full |
| `panel59_full` | extended (59 rec.) | 200 | 3 | full |
| `panel59_reduced` | extended (59 rec.) | 10 | 1 | 32 |
| `chai1_full` | in-training | 200 | 3 | full |

The last five were folded on a rented RTX 4090 in August 2026. `chai1_full` is
**Chai-1**, not Boltz: the same panel and the same readout code, so the model is
the only variable. It carries cognates and scrambles only — its decoys were not
folded, so the ranking test is Boltz-only. Its interface pLDDT is calibrated
differently (near 95 where Boltz-1 sits near 90 on the same complexes), so only
standardised effects compare across the two.

Settings are taken from each run's own recorded configuration where it kept one;
the three arms without a settings block are labelled from the run that produced
them. `decaf_replicates` is 96 repeated folds of 4 receptors, used to measure
run-to-run spread.

## Labels

`cognate` — the peptide crystallised with that receptor.
`scrambled` — a permutation of that cognate, composition and length fixed.
`decoy` — a cognate peptide of a *different* receptor in the panel.

Each receptor contributes 1 cognate, 2 scrambles and 3 decoys.

## Readouts

`iptm`, `iface_plddt` (mean CA pLDDT over residues within 8 Å across the
interface, both sides), `receptor_side` / `peptide_side` (the same restricted to
one side), `peptide_whole` (whole-chain pLDDT), and interface residue counts.

## Suggested AUC convention

Scores are not comparable across receptors. Standardise within receptor, then
pool into one ROC, with the cognate positive and everything else folded against
that receptor negative.

## Known limitations

- **No structure files.** The predictions are ~198 MB and most did not survive a
  disk cleanup. The scores did, and the controls are computed from the scores.
- **`sequences.csv` covers the in-training panel only** — all 132 of its folds,
  read from the Boltz input files. The held-out panel's inputs were lost in the
  same cleanup; those 22 receptors are listed by PDB ID in `panels.csv` and can
  be refetched from the RCSB. Note that sequences recovered from deposited
  crystals contain only *observed* residues and will not exactly match what was
  folded, so they are not a drop-in substitute.

  Integrity of this file is checkable and was checked: all 44 scrambles have
  amino-acid composition identical to their own receptor's cognate, and all 66
  decoys are another receptor's cognate. A permutation is a permutation by
  construction, so a mis-joined table fails that test.
- **The panels overlap by 6 receptors** (7OKL, 7S7J, 8HLO, 8KDX, 9F6S, 9GRF).
  These postdate the training cutoff and appear in both.
- **Three arms do not record a per-fold `name`**, so their rows join on
  `receptor_id` and `label` rather than on fold identity.
- **PoseBusters is not usable at face value on peptides.** Its `bond_lengths`,
  `bond_angles` and flatness checks are **vacuous** here: RDKit perceives bonds
  in a PDB by distance, so a backbone bond stretched past bonding range is never
  perceived as a bond and cannot fail a length test. Those checks report 100%
  pass on structures that are in forty pieces. Worse, `all_atoms_connected`
  reports 0% pass on the *converged* set while RDKit's own `GetMolFrags` returns
  one fragment for the identical molecule. **Use the fragment counts and nothing
  else.** They are unambiguous: at 10 sampling steps 0 of 144 peptides is a
  single connected fragment (median 40.5); at 200 steps it is 354 of 354
  (median 1).

## Licence

CC BY 4.0.

Panel receptors are derived from the Protein Data Bank.

**No model weights are redistributed here, and none of the models is ours.** This
dataset contains scores computed from their outputs:

- **Boltz-1 / Boltz-2** — MIT-licensed (Wohlwend et al. 2024; Passaro et al. 2025)
- **Chai-1** — the `chai1_full` arm (Chai Discovery team, 2024)
- **DeCAF** — the few-step distilled arm, from Scarpellini et al.,
  *Few-step cofolding with all-atom flow maps*, arXiv:2606.08375

The designed-miniprotein comparison referenced in the boundary result uses the
per-design release of *Autonomous de novo protein binder design with Claude*
(CC BY 4.0), which is cited per its own `CITATION.cff` and is not redistributed
here.
