# Held-out complexes cost both readouts about half their effect

Section 7.6 recommends ranking peptide binders on interface pLDDT rather than
ipTM, and Sections 7.7 and 7.8 build on it. Every fold behind that
recommendation is a PDB entry, and Boltz-1 — which DeCAF distils — was trained
on entries released before 2021-09-30. So "interface pLDDT is high for cognates"
could mean "the model has seen this complex".

An earlier version of this report, written from a single draw, claimed the two
readouts swap places on held-out complexes. Three draws do not support that and
it has been withdrawn; what survives is that both weaken by roughly half.

A second panel of **22 receptors released after the cutoff** tests it, screened
identically and with decoys drawn from within the held-out set, so no fold in
the comparison involves a training structure. 132 folds on DeCAF at the same 10
sampling steps and 1 recycling step.

## The result, over three independent draws

Folds are unseeded, so re-running the identical panel gives an independent draw.
Three were run. The first two each produced a conclusion the third withdrew, and
that is the main finding of this experiment (see "One draw is not a measurement"
below).

| metric | in-training (16 receptors) | held-out (22 receptors) |
| :--- | ---: | ---: |
| ipTM | +0.265 (p = 1e-5) | +0.137 (p = 0.0001) |
| interface pLDDT | +12.03 (p < 1e-5) | +4.99 (p = 0.0006) |
| receptor side | +7.38 (p = 2e-5) | +2.91 (p = 0.0016) |

Cognate ranked against its own decoys, chance 2.50:

| metric | in-training | held-out |
| :--- | ---: | ---: |
| ipTM | 1.77 (p = 0.0087) | 1.73 (p = 0.0069), 14 of 22 first |
| interface pLDDT | 1.73 (p = 0.0042) | 1.91 (p = 0.020), 11 of 22 first |

**Both readouts lose roughly half their effect on complexes the model was not
trained on** — interface pLDDT retains 41%, ipTM 52%. Neither can be shown to
survive better than the other.

## One draw is not a measurement, and neither are two

**Order sensitivity, cognate minus its own scramble:**

| metric | draw 1 | draw 2 | draw 3 | mean |
| :--- | ---: | ---: | ---: | ---: |
| ipTM | +0.110 (p = 0.0036) | +0.162 (p = 3e-5) | +0.138 (p = 2e-4) | +0.137 |
| **interface pLDDT** | **+2.76 (p = 0.089)** | +6.37 (p = 8e-5) | +5.85 (p = 7e-5) | **+4.99** |
| receptor side | +1.39 (p = 0.201) | +3.75 (p = 0.0019) | +3.59 (p = 5e-4) | +2.91 |

**Receptor specificity, cognate ranked among its own decoys:**

| metric | draw 1 | draw 2 | draw 3 | mean |
| :--- | ---: | ---: | ---: | ---: |
| ipTM | 1.64 (p = 0.0020) | 1.50 (p = 0.0001) | **1.86 (p = 0.017)** | 1.73 |
| interface pLDDT | 2.09 (p = 0.141) | 2.05 (p = 0.086) | **1.73 (p = 0.0029)** | 1.91 |

Draw 1 alone said the interface-pLDDT effect *collapses* held out. Draws 1 and 2
together said the rank test was the stable one and that ipTM specifically
survived while interface pLDDT specifically failed. **Draw 3 reverses the rank
ordering**, and averaged over three draws the two metrics are 1.73 and 1.91,
both significant, differing by less than the spread of a single draw.

Both claims are withdrawn. This is Section 7.5's own finding turned on the
project's headline, twice.

## Testing it properly

A mixed model with receptor as a random effect gives the interaction directly,
fitted on raw scores and on scores z-scored within receptor:

| metric | raw | within-receptor z |
| :--- | ---: | ---: |
| ipTM | −0.128 (p = 0.052) | −0.096 (p = 0.737) |
| interface pLDDT | −7.04 (p = 0.0068) | −0.521 (p = 0.058) |
| receptor side | −4.47 (p = 0.014) | −0.451 (p = 0.124) |

**On the scale-free measure nothing degrades significantly.** Two draws gave
p = 0.0029 for interface pLDDT; three give p = 0.058. The raw column may be a
scale effect, since the held-out panel has smaller within-receptor spread (5.65
against 8.64).

## Confounds that do not explain it

**Peptide length.** Held-out peptides are longer (12.8 against 10.3 aa,
p = 0.10). This cannot matter and is reported as a non-control: the effect is a
within-receptor contrast between a peptide and a *permutation of itself*, so
length and composition are equal by construction. Length is constant within
receptor, so the random intercept absorbs it — adding it as a covariate leaves
beta and p unchanged to five decimals.

**A ceiling compressing the held-out differences.** The opposite holds. Cognate
ipTM is 0.631 in training against 0.419 held out; interface pLDDT 81.57 against
68.42. The model is markedly *less* confident on complexes it has not seen,
which is the signature the experiment was built to detect.

**Fold settings and MSAs.** Both panels: DeCAF, 10 sampling steps, 1 recycling
step. The six receptors common to both reuse the main panel's cached alignments
byte-for-byte.

## The panel is not homology-decontaminated, and cannot be

A temporal split is not the current standard alone; contamination-aware protein
benchmarks add sequence filtering at 30% identity. **Not one of the 22 held-out
receptors passes**: median maximum identity to the pre-cutoff PDB is 1.000, and
17 of 22 have a relative at 90% or above. This is not a selection failure — 22
candidates sampled at random from the entire screened pool give 0 of 22 under
every threshold. Peptide-binding domains are too densely represented in the PDB
for sequence-level decontamination to be achievable.

What the split isolates is therefore **complex-level** novelty with
receptor-level familiarity held constant, which is the operationally relevant
comparison for screening but a weaker claim than "structures never seen".

## What this changes

A screening figure quoted without stating whether the complexes were in the
model's training set is about **a factor of two optimistic**. That holds for
both readouts, and it is the durable result. The metric-specific claims did not
replicate and should not be repeated.

## Reproduce

```
python src/heldout_panel.py --batch-size 20
python src/heldout_panel.py --with-pae --batch-size 20
python src/heldout_panel.py --with-pae --run-tag 2 --batch-size 20
python src/heldout_replicates.py
python src/homology_decontamination.py
```
