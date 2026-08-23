# GPU run, 23 August 2026

Six results folded on a rented RTX 4090, none of which was reachable on Apple
Silicon. Roughly 1,000 folds for about $3.50 of rental.

Nothing here has been written into the dissertation or the preprint. The
dissertation is submitted on the 28th and adding new numbers now would mean
re-verifying claims across 132 pages; these are stored for the paper.

| file | what |
| :--- | :--- |
| `panel59_readouts.json` | 59-receptor panel, full settings, all three readouts |
| `panel59_reduced_scores.json` | the same 59 receptors at 10 steps / 1 recycle |
| `panel59_scores.json` | the runner's own ipTM scoring of the full arm |
| `chai_arm.json` | the scramble control under Chai-1 |
| `scramble_wetlab_full.json` | Section 7.19 refolded at full settings, 48 designs |
| `scramble_wetlab_result_full.json` | its pre-specified analysis |
| `posebusters_converged.json` | PoseBusters on 354 converged structures |
| `heldout_panel_resultfull3.json` | a third full-settings held-out draw |
| `settings_confoundexpanded.json` | the 22-receptor panel refolded on the 4090 |
| `panel_expansion.json` | the 37 receptors `discover_pdb_binders.py` added |

## What they show

**The panel at 22 -> 59 receptors.** The scramble-control effect holds, but every
effect size falls: interface pLDDT Cohen's *d* 1.43 -> 1.04, receptor side
1.67 -> 0.77. The original 22 were a favourable draw and the dissertation's
figures are optimistic. This is the honest estimate.

**The settings confound at n = 59.** Reduced settings suppress the ipTM effect
12x raw and 3.6x standardised, against 7.4x and 2.8x at n = 22. Section 7.13's
"three to seven times" holds at 2.7x the panel, at p = 5.8e-12.

**A second model family.** Under Chai-1, interface pLDDT separates a cognate
peptide from a permutation of itself on 20 of 21 receptors, p = 1.8e-04,
*d* = 0.88 against Boltz-1's 1.43. The effect is a property of cofolding
confidence, not of Boltz -- which nothing in the dissertation could establish.
Weaker in Chai than in Boltz, and the claim should say so.

**Section 7.11's mechanism, externally confirmed.** PoseBusters on converged
structures: 354/354 peptides are a single connected fragment, median 1. At ten
sampling steps it was 0/144, median 41 fragments.

**Section 7.19 at full settings.** Its limitations section called a full-settings
repeat "an expectation and not a measurement". It is now a measurement, on 48
designs rather than 38: margin +20.81 for binders against +22.28 for
non-binders, Welch p = 0.677, delta-AUC -0.104 with an interval spanning zero.
The boundary stands, and at full settings the point estimate flips sign --
further evidence the subtraction is a constant on designed proteins.

## Caveats

PoseBusters' own summary table is not usable on peptides. Its `bond_lengths`,
angle and flatness checks are vacuous -- RDKit perceives bonds by distance, so a
broken backbone bond is never perceived and cannot fail -- and on the converged
set `all_atoms_connected` reports 0% while RDKit's own `GetMolFrags` returns one
fragment for the same molecule. The fragment counts are the trustworthy numbers.

Chai-1's interface pLDDT sits near 95 where Boltz-1's sits near 90 on the same
complexes. Raw effects are not comparable across the two; only standardised ones.
