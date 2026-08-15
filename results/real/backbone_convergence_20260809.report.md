# At ten sampling steps the predicted coordinates are not folded proteins

Sections 7.4 to 7.10 accumulate readouts that fail for reasons stated
separately. This is one cause for four of them, found by checking something the
earlier work took for granted: whether the predicted structures are structures.

## The audit

A peptide bond fixes consecutive alpha-carbons at 3.80 Å. Every consecutive
CA–CA distance, across every fold on disk:

| | Boltz-2 @ 10 steps | DeCAF @ 10 steps |
| :--- | ---: | ---: |
| structures | 132 | 69 |
| median CA–CA | 5.48 Å | **3.74 Å** |
| physically plausible (3.4–4.2 Å) | **14.0%** | **96.2%** |
| broken (> 5 Å) | 56.2% | **0.0%** |
| implausibly short (< 3.0 Å) | 12.4% | 1.0% |
| chains > 50% non-physical | 99% | — |

Observed distances span 0.33 Å to 63.4 Å. Residue numbering is sequential, so
this is not a file-ordering artefact — the backbone is genuinely not connected.
What a stock model returns at 10 of an intended 200 sampling steps is a
partially denoised point cloud with roughly correct residue identity and
per-residue confidence.

**DeCAF, distilled *for* ten steps, returns folded chains.** Section 7.8
measured that distillation buys a 5–6× larger effect without being able to say
what it bought. It buys geometry.

## Four failures, one cause

**pDockQ (Section 7.6).** Its contact term was found to run backwards —
scrambled peptides making more inter-chain contacts than cognates, 38.5 against
32.9. On converged structures the ordering reverses to the sensible direction:

| class | contacts | PRODIGY ΔG |
| :--- | ---: | ---: |
| cognate | **61.4** | −8.79 |
| decoy | 55.3 | −8.50 |
| scrambled | 51.8 | −8.26 |

So the inversion is a property of unconverged geometry, not of short peptides.
Section 7.6 had the mechanism right and the cause wrong. Every contact-derived
row in that section — contacts, contact density, buried surface area, pDockQ —
describes point clouds.

**pDockQ2.** Substituting a PAE-derived term for the contact term repairs the
metric on the *same* Boltz-2 structures, from p = 0.797 on the scramble control
to **p = 0.00026**. That is what the mechanism predicts.

**Minimum PAE.** Reported in 2026 as the best enrichment metric for AlphaFold3
and Boltz-2 on a ligand benchmark. Here it fails outright on the Boltz-2 panel
(scramble control p = 0.611) and is among the best readouts on the DeCAF
held-out panel (mean rank 1.55 over two draws, 15 of 22 cognates first). PAE is
a prediction *about geometry*; it means little before the geometry converges.
The published result was measured at full sampling.

**Physics rescoring.** PRODIGY returns Kd = 2.4 × 10⁻¹⁴ M — femtomolar — for an
arbitrary peptide on Boltz-2 structures. Garbage in, not method failure.

**The unifying statement:** geometry-dependent readouts require a converged
sampler. At reduced sampling only the non-geometric confidence outputs (ipTM,
per-residue pLDDT) retain meaning, and few-step distillation restores geometry
and the geometric readouts with it.

## Physics rescoring, on structures that support it

69 folds, 12 held-out receptors, DeCAF:

| readout | cognate − scramble | mean rank (chance 2.50) | AUC within |
| :--- | ---: | ---: | ---: |
| interface pLDDT | +7.63 (p = 5e-5) | 1.73 (p = 0.036) | **0.856** |
| pDockQ2 | +0.112 (p = 0.006) | 1.64 (p = 0.025) | 0.807 |
| **PRODIGY ΔG** | +0.671 (p = 0.109) | **2.36 (p = 0.889)** | **0.563** |

And it does not combine: adding ΔG to interface pLDDT under leave-one-receptor-
out takes AUC from 0.856 to **0.823**. The contact signal is real on converged
structures but worth only 0.53 kcal/mol between cognates and scrambles, which
PRODIGY's polarity weighting and non-interacting-surface terms dilute rather
than sharpen.

This bounds one contact-count regression, not physics in general. A force field
computing solvation and hydrogen bonding explicitly may behave differently.

## Caveats

The DeCAF structures are 69 folds across 12 receptors, captured from a running
job before it deleted them; the Boltz-2 set is the full 132. The magnitude of
the difference (96.2% against 14.0%) is far too large for the unequal sample or
the panel difference to explain — backbone convergence is a property of the
sampler, not of which receptors were folded — but the DeCAF figure is measured
on fewer structures.

An earlier version of this report quoted 90.6% from the first 15 structures
captured. The full 69 give 96.2%.

## Reproduce

```
python src/physics_rescore.py --structures <dir of kept predictions>
```

Structures are deleted after scoring by default; capture them from a running
job, or remove the `shutil.rmtree` in the batch loop of `heldout_panel.py`.
