# GPU run, 24 August 2026

Two results, both of which delete a sentence from the preprint's own limitations.
About 460 folds for roughly $1.50 of rental.

| file | what |
| :--- | :--- |
| `settings_confoundsamp_recyc.json` | 200 steps / 3 recycles / MSA 32 |
| `settings_confoundsamp_align.json` | 200 steps / 1 recycle / full MSA |
| `settings_confoundrecyc_align.json` | 10 steps / 3 recycles / full MSA |
| `chai_arm_full.json` | Chai-1, all 131 folds: cognates, scrambles and decoys |

## The factorial changes a recommendation

With these three pairs, all eight cells of the settings 2×2×2 exist. The
single-knob arms said alignment depth carries none of the effect. **They were
misleading.** Alone it contributes −8%; paired with sampling it adds a genuine
+0.44 interaction and reaches *d* = 1.46 — 96% of the full effect from two knobs
rather than three.

  pair                     observed   additive   interaction
  sampling + recycling       +0.77      +0.78         −0.01   additive
  sampling + alignment       +1.18      +0.74         +0.44   synergy
  recycling + alignment      +0.01      −0.17         +0.19   weak

Deep alignments do nothing while the sampler is unconverged and contribute once it
has converged, which is a mechanism rather than a share: coevolutionary signal can
only express itself through a pair representation the sampler has resolved.

Recycling is the genuinely inert knob — interaction −0.01, and sampling+recycling
slightly underperforms sampling alone. The recommendation is now sampling first,
alignment second, drop recycling. The earlier advice said to cut alignment depth
first and was wrong.

Caveat: one draw per cell, and the interactions are of the same order as Section
2.4's run-to-run spread. The sign of the sampling-alignment synergy is better
supported than its magnitude.

## The ranking test transfers between model families; the scramble control transfers less well

                              Chai-1    Boltz-1
  mean rank (chance 2.50)       1.38       1.41
  cognate ranked first        16 / 21    17 / 22
  top-1 accuracy                 76%        77%
  P(cognate > own decoy)       0.873      0.864

On the measure a practitioner uses to pick candidates for synthesis, the choice
between these two models does not matter. The permutation control transfers in
direction and significance but at 60% of the effect size (*d* 0.90 against 1.43).

Those two tests ask different questions — a candidate against its competitors,
versus a candidate against itself — and the second is the more model-dependent.

One fold of 132 is missing: a transient ColabFold error on `pair_000`.

## The boundary test at 720 designs — the conclusion holds, the explanation does not

Every design on every target whose construct fits a 24 GB card: 720 designs, 8
targets, 234 measured binders, 2,160 folds at converged settings. Cas9 (1,368
residues), EGFR and Nipah-G were excluded for size, not biology, as were the four
oligomeric targets with no single-chain construct.

                  n     design   scrambles    margin
  binders       234      83.75       60.08    +23.67
  non-binders   486      79.84       61.61    +18.23

**The primary result stands and tightens fivefold.** Raw interface pLDDT reaches
within-target AUC 0.599, the corrected margin 0.576: dAUC -0.023, 95% CI
[-0.065, +0.006], against [-0.330, +0.071] at 48 designs. The control adds
nothing, and that is now a bound rather than an absence of evidence.

**The mechanism in the paper was wrong.** Both the pre-registered prediction and
Section 2.7 said the permutations are *equally ruined* for binders and
non-binders, so the subtraction is close to a constant. At 720 designs that is
false: the margins differ by 5.4 points at p = 4e-12, and the permutation scores
themselves differ at p = 0.010.

The subtraction carries real signal. It carries no *additional* signal, which is
a different claim: margin correlates with raw at r = 0.813, and a model given
both scores 0.596 against 0.599 for raw alone. A permutation of a designed
protein does not fold, but how far its score falls still tracks how good the
original was.

That distinction changes when to use the control. Against a constant offset the
subtraction is harmless. Against a redundant one it is a coin flip costing two
extra folds per candidate.

Recovered after the pod was stopped mid-transfer; /workspace persisted and the
file was intact.
