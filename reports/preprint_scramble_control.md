# Composition-matched controls change what cofolding confidence metrics appear to measure

**Akik Jana**

Birla Institute of Technology and Science, Pilani — Work Integrated Learning Programmes, Pilani, Rajasthan, India

Correspondence: akik.e.aj@gmail.com

---

## Abstract

Confidence scores from cofolding models — ipTM, interface pLDDT, pDockQ and their
relatives — are widely used to rank candidate binders before synthesis. Such
rankings are usually validated against decoys: unrelated sequences that share
neither composition nor length with the candidate. We show that this comparison
is too weak to identify what the score responds to. On a 22-receptor
peptide–protein panel we scored each cognate peptide against decoys and against
permutations of itself, which hold amino-acid composition and length exactly
fixed and destroy only sequence order. ipTM ranked cognates above decoys
(mean rank 2.00 against a chance value of 2.50, p = 0.034) while failing to
separate a cognate from its own permutation (+0.013, 95% CI [−0.019, +0.044]);
permutations in fact outscored decoys (AUC 0.632, p = 0.0096). Of six interface
readouts tested on identical structures, only interface pLDDT passed the
permutation control, at 8.6 times ipTM's effect-to-noise ratio. Three further
controls each changed a conclusion: replicate folding (single unseeded folds do
not reproduce their own ranking), a training-cutoff split (only 38–40% of the
standardised effect survives on complexes released after the cutoff), and
sampling budget (reducing sampling steps from 200 to 10 suppresses every effect
threefold to sevenfold, and a per-knob decomposition attributes 56–69% of the
gain to sampling steps alone, with alignment depth and recycling contributing
none). Against 1,320 designed miniproteins with binding measured by two contract
research organisations, the same readout reached within-target AUC 0.626, where
the in-training panel gave 0.943. On those designed proteins the permutation
control itself carried no information (ΔAUC −0.092, 95% CI [−0.170, +0.008]),
because permuting a 100-residue protein destroys folding for binders and
non-binders alike. We therefore recommend the control where a permutation of the
candidate remains a plausible candidate, and give the measured boundary of that
condition.

**Keywords:** protein structure prediction, cofolding, confidence metrics,
benchmarking, negative controls, peptide binders, data contamination

---

## 1. Introduction

Cofolding models predict the structure of a protein complex from sequence alone
[1–3], and each emits a confidence score intended to indicate how much of the
prediction should be believed. The open Boltz models [4,5] reproduce much of this
capability under permissive licences, which has made confidence-guided screening
tractable on commodity hardware. In a typical screening workflow, candidate
binders are folded against a target, ranked by ipTM [2] or a related interface
score, and the top-ranked candidates are synthesised.

The validation supporting this workflow is usually a discrimination test:
cognate binders are ranked against decoys, and the score is judged by whether
cognates rise. This is a real test, but it is a weak one, because a cognate
differs from an unrelated decoy in many ways at once — length, amino-acid
composition, hydrophobicity, charge, and sequence order. A score that responded
only to composition would pass it.

Permutation controls that hold composition fixed are standard practice in
sequence analysis, where shuffled-sequence nulls have been used for decades in
motif discovery and alignment statistics. They are, however, rarely reported in
cofolding benchmarks, where decoy sets are the norm. Our contribution is not the
idea of a permutation null but the systematic application of one to cofolding
confidence metrics, together with a measurement of the conditions under which it
is informative and the conditions under which it is not.

We report six controls applied to a peptide-binder screening panel. Each was
capable of overturning a result, and each did. We report the negative results in
full, including two conclusions of our own that the controls forced us to
retract. Finally, we test the readouts and the control itself against a
recently released dataset of designed miniproteins whose binding was measured
experimentally [6], which permits the comparison that a structure-derived panel
cannot make: how a confidence readout performs against binding somebody actually
observed, on designs that postdate every model's training data.

---

## 2. Results

### 2.1 A composition-matched control separates two hypotheses that decoys conflate

We assembled a panel of 22 receptors (132 complexes) from the Protein Data Bank
[10], filtered to remove post-translationally modified residues, expression tags,
and duplicate receptors or peptides. Each cognate peptide was folded against its
receptor, against three unrelated decoy peptides drawn from other receptors in
the panel, and against two permutations of itself. A permutation preserves
amino-acid composition and length exactly and destroys only the order of
residues — and order is what makes a binder a binder. We refer to these
interchangeably as permutations or *scrambles*, the latter being the term used
in our code and figure labels.

Against decoys, ipTM behaved as the literature would predict. The cognate reached
a mean rank of **2.00 against a chance value of 2.50** (first for 8 of 22
receptors, p = 0.034 two-sided, bootstrap 95% CI [1.59, 2.41] excluding chance).
Read alone, this is a positive result.

The permutation control contradicts the binding interpretation of it.

**Table 1. Mean ipTM by class on the 22-receptor panel (10 sampling steps).**

| class | n | mean ipTM |
| :--- | ---: | ---: |
| cognate | 22 | 0.5015 |
| scrambled | 44 | 0.4888 |
| decoy | 66 | 0.4317 |

Cognates did not measurably beat their own permutations (mean difference
+0.0128, 95% CI [−0.019, +0.044], n = 44), and permutations **outscored decoys**
(AUC 0.632, p = 0.0096). The variable separating cognate from decoy is therefore
largely the one a cognate shares exactly with its own permutation. The null on
order is a bound rather than a zero: any order effect is at most 63% of the
composition effect and is plausibly nil.

This is not a length artifact, which we tested directly. ipTM falls with peptide
length (Spearman −0.408, p = 1.2 × 10⁻⁶), but regressing the per-pair
cognate-minus-decoy score difference on the corresponding length difference
leaves an intercept of **+0.0752 (p = 0.0001)**: the advantage survives length
adjustment and is compositional rather than positional.

Reporting only the decoy comparison would have produced a positive headline that
the experiment's own control contradicts.

### 2.2 Of six interface readouts on identical structures, one passes

The finding in §2.1 indicts ipTM, not the predicted structures. We re-scored the
same 132 structures with six interface measures, so that the comparison between
readouts is free of folding variance by construction. Contacts use an 8 Å
CB–CB criterion (CA for glycine); pDockQ follows Bryant et al. [7]; buried area
is Shrake–Rupley.

**Table 2. Interface readouts on the permutation control.**

| Metric | cognate | scrambled | decoy | cognate vs own scramble | mean rank | chance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ipTM | 0.502 | 0.489 | 0.432 | p = 0.416 | 2.00 | 2.50 |
| pDockQ | 0.474 | 0.466 | 0.404 | p = 0.797 | 2.14 | 2.50 |
| **Interface pLDDT** | **49.60** | **46.31** | **45.93** | **p < 0.0001** | **1.91** | 2.50 |
| Inter-chain contacts | 32.9 | 38.5 | 34.0 | p = 0.054 | 2.55 | 2.50 |
| Contact density | 3.24 | 3.69 | 3.62 | p = 0.122 | 2.55 | 2.50 |
| Buried surface area | 1800 | 1861 | 1632 | p = 0.454 | 2.27 | 2.50 |

Only interface pLDDT separates a cognate from its own permutation. The behaviour
of pDockQ is instructive: it multiplies interface pLDDT by a contact term, and
the contact term runs the wrong way in this regime, because permuted peptides
make *more* inter-chain contacts than cognates (38.5 against 32.9). Combining the
two cancels the signal that interface pLDDT carries alone. Replacing pDockQ's
contact term with a PAE-derived one (pDockQ2 [8]) repairs it on the same
structures, from p = 0.797 to p = 0.00026.

![Only interface pLDDT separates a binder from its own scramble](assets/fig_metric_comparison.png)

<p class="figcap"><b>Figure 1.</b> ipTM (left) and interface pLDDT (right) by
class on the 22-receptor panel. Boxes span the interquartile range; the bracket
gives the paired test of a cognate against its own scramble. ipTM places
cognates above decoys while failing to distinguish them from permutations of
themselves; interface pLDDT separates both.</p>

The cause of the inverted contact ordering is unconverged geometry rather than
peptide length: at 10 sampling steps only 14% of backbone bonds in these
structures fall within physically plausible bounds, against 96% for a
few-step-distilled model [9], and on converged structures the contact ordering
reverses to the sensible direction (cognates 61.4, permutations 51.8). Every
contact-derived row of Table 2 should be read as describing point clouds rather
than complexes. The ipTM and interface-pLDDT rows are unaffected, both being read
from the confidence head rather than from coordinates.

### 2.3 Single folds do not reproduce their own ranking

Boltz's sampler seed defaults to unset, so each reported score is one draw from a
distribution whose width is not reported. We folded a subset 96 times to measure
that width, then compared each readout's effect on the permutation control
against its own run-to-run spread.

**Table 3. Permutation-control effect against run-to-run spread.**

| Metric | effect | run-to-run SD | effect / noise |
| :--- | ---: | ---: | ---: |
| ipTM | +0.0128 | 0.0628 | 0.20× |
| pDockQ | +0.008 | 0.1498 | 0.05× |
| **Interface pLDDT** | **+3.30** | **1.917** | **1.72×** |

Interface pLDDT carries roughly **8.6 times** ipTM's effect-to-noise ratio on the
test that matters. Simulating the full benchmark under the measured noise gives a
cognate-minus-permutation effect of +3.30 pLDDT (95% CI [+2.13, +4.47]) that
reproduces at p < 0.05 in **100%** of re-runs, and a within-receptor rank of 1.91
(95% CI [1.64, 2.14]) reproducing in **84%**, against 49% for ipTM.

We emphasise the practical consequence, because it applied to our own work: two
conclusions we had drawn from single-draw comparisons did not survive replication
and were retracted. A benchmark that folds each candidate once is reporting a
sample of size one from a distribution wide enough to reverse its ordering.

### 2.4 A training-cutoff split removes about two thirds of the standardised effect

Panels drawn from the PDB overlap the training data of every model trained on the
PDB. We assembled a second panel of 22 receptors released after the model's
training cutoff, filtered identically, with decoys drawn from within the held-out
set so that no fold in the comparison involves a training structure. Both panels
were then folded at the model's intended settings (200 sampling steps, 3
recycling passes, undiminished alignment depth), so that the only quantity
varying between them is whether the model has seen the complex.

**Table 4. The contamination penalty at full settings.**

| Readout | in-training | held out | *p* held out | effect retained | Cohen's *d* retained |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ipTM | +0.287 | +0.169 | 0.0017 | 58.7% | **40.4%** |
| Interface pLDDT | +11.85 | +6.05 | 0.00047 | 51.1% | **37.6%** |
| Receptor side | +5.13 | +2.95 | 0.00075 | 57.6% | **37.9%** |

All three readouts remain significant on complexes the model was never trained
on, and roughly half the raw effect survives. Standardised, however, retention is
**37.6% to 40.4%** — materially worse than raw retention, because full settings
raise absolute confidence across the board and a raw difference is therefore
partly scale. Any figure quoted for a novel target should use the standardised
retention.

![Sampling budget sets the effect size; training exposure sets how much survives](assets/fig_contamination.png)

<p class="figcap"><b>Figure 2.</b> Cohen's <i>d</i> on the scramble control
across the full 2×2 of training exposure and sampling budget, for three
readouts. Raising the sampling budget multiplies the in-training effect by 1.7
to 1.9 (left to right, blue bars); withholding the complex from training removes
about three fifths of it at full settings (blue to amber), and between a third
and three quarters at reduced settings, where the estimate is itself unstable.
Percentages give the standardised retention held out. Dotted rules mark Cohen's conventional
small/medium/large thresholds.</p>

### 2.5 Sampling budget confounds every comparison above

The measurements in §2.1–2.3 were taken at 10 sampling steps of an intended 200,
a reduction adopted for throughput on consumer hardware. We folded the same panel
on the same model and device at full settings to measure what that reduction had
cost.

**Table 5. Permutation control at reduced and full settings.**

| Metric | reduced | full | Cohen's *d* | within-receptor z |
| :--- | ---: | ---: | :--- | ---: |
| ipTM | +0.039 (p = 0.004) | **+0.287 (p < 1e-5)** | 0.45 → **1.25** | 4.0× |
| Interface pLDDT | +1.54 (p = 0.067) | **+11.85 (p < 1e-5)** | 0.28 → **1.52** | 4.7× |
| Receptor side | +0.71 (p = 0.428) | **+5.13 (p < 1e-5)** | 0.12 → **1.45** | 7.3× |

Raw effects are 7.2 to 7.7 times larger at full settings; standardised, 2.7 to 12
times. The effect is larger at full settings for 21 of 22 receptors on ipTM and
interface pLDDT, with paired p ≤ 0.001 throughout, so it is not driven by
outliers. Two readouts change verdict outright: interface pLDDT from p = 0.067 to
p < 10⁻⁵, and the receptor side from p = 0.428 — no evidence at all — to
p < 10⁻⁵.

This has a direct bearing on §2.1. ipTM's indifference to sequence order is a
property of the reduced sampling regime at least as much as of the metric: at 200
steps the same model on the same panel separates a cognate from its own
permutation at *d* = 1.25. The claim that survives is the narrower one — **ipTM
is indifferent to sequence order when the sampler has not converged** — which
§2.2 identifies as the regime in which the predicted backbone is 14% physically
plausible.

To locate the cost we moved each setting alone from the reduced baseline.

**Table 6. Per-knob decomposition, order-sensitivity test.**

| Arm | interface pLDDT | *p* | Cohen's *d* | share of full gain |
| :--- | ---: | ---: | ---: | ---: |
| reduced | +1.54 | 0.067 | 0.28 | — |
| **sampling** | **+8.82** | **2.5e-09** | **1.13** | **69%** |
| alignment | +0.83 | 0.243 | 0.18 | −8% |
| recycling | +1.40 | 0.157 | 0.22 | −5% |
| full | +11.85 | 7.1e-13 | 1.52 | 100% |

Sampling steps alone carry **56% to 69%** of the standardised gain across the
three readouts and take every one from marginal-or-nothing to p < 10⁻⁶.
Alignment depth and recycling, moved alone, carry none of it: every share is
negative, each arm sitting marginally below the reduced baseline in standardised
terms. The shares sum above 100%, so the three settings are synergistic rather
than independent, and no single-knob budget reproduces the full result.

The practical recommendation is specific: under a compute constraint, spend it on
sampling steps and reduce alignment depth and recycling first.

### 2.6 Against measured binding, the readout reaches 0.626

Every result above scores a cognate pair that was crystallised, not one whose
binding was measured. A recently released dataset [6] permits the direct test:
1,320 designed miniproteins across 15 targets, with binding measured by two
independent contract research organisations, and with designs postdating every
model's training data. 354 (26.8%) bind.

We recomputed interface pLDDT exactly as in §2.2 from the Boltz-2 structures
released with that dataset. One implementation detail mattered: five targets are
oligomeric, so taking the first two chains as receptor and binder would have
measured a target against itself. The binder chain was identified by matching the
released binder length, with every other chain pooled as the target. All 1,320
designs scored.

**Table 7. Readouts against measured binding, and against structure-derived panels.**

| Setting | within-group AUC | macro-AP |
| :--- | ---: | ---: |
| Interface pLDDT, measured binding | 0.626 | 0.436 |
| ipSAE, same structures | 0.628 | 0.436 |
| chance | 0.500 | 0.268 |
| — for scale, same AUC convention — | | |
| this work, in-training panel, full settings | 0.908–0.943 | — |
| this work, held-out panel, full settings | 0.683–0.758 | — |

All AUCs in this table use one convention: scores are z-standardised within
group — receptor for our panels, target for the designs — and pooled into a
single ROC. The held-out row is the mean of two independent full-settings draws.

These are not the same measurement: the panels differ, the positives differ (a
cognate crystal pair against a synthesised binder), and the readouts differ in
part. What they share is the question, and the ordering answers it. A readout
scored on complexes the model was trained on reads far higher than the same
family of readout scored against binding that was actually measured. The
contamination penalty of §2.4, measured internally on a training-cutoff split,
appears again from outside.

For context, nine other predictors scored on the same designs span within-target
AUC 0.621–0.671 and macro-AP 0.453–0.532. No structure-prediction confidence
readout in this comparison, ours included, is close to its own benchmark figures.

![No confidence readout is far above chance on binding that was actually measured](assets/fig_wetlab.png)

<p class="figcap"><b>Figure 3.</b> (a) macro-AP against measured binding for
every predictor scored on the release, with 95% confidence intervals; the dashed
rule is the base rate. Red bars are this work's readouts. Every interval covers
or nearly covers chance. (b) The same readout family across three regimes on one
AUC convention. The descent from a training-adjacent panel to measured binding is
about 0.28–0.32 AUC, and the held-out panel sits between the two. Panels differ
in molecules and in what counts as a positive, so (b) supports an ordering rather
than a numerical correction.</p>

### 2.7 Where the control stops working

The permutation control had itself never been tested against measured binding. We
registered a prediction in source, committed before any fold ran:

> These binders are 60 to 120 residue designed proteins, not the 5 to 25 residue
> peptides of the panel. A permutation of a 100-residue protein does not fold at
> all, so both binders and non-binders should receive an equally ruined scramble,
> and the subtraction may add nothing.

Thirty-eight designs across four targets (RBX1, PD-L1, TrkA, BHRF1) were folded
as delivered and against two permutations of each.

**Table 8. Interface pLDDT of a design and of its own permutations, by measured outcome.**

| | n | design | its permutations | margin |
| :--- | ---: | ---: | ---: | ---: |
| measured binders | 20 | 72.99 | 53.88 | **+19.11** |
| measured non-binders | 18 | 71.15 | 53.22 | **+17.93** |

The permutations lose about nineteen points of interface pLDDT whether the design
binds or not, and the two margins are indistinguishable (Welch p = 0.751). On the
single pre-specified comparison, the raw readout gave within-target AUC 0.672 and
the permutation-corrected margin gave 0.581 — ΔAUC **−0.092**, 95% CI [−0.170,
+0.008].

The interval includes zero, so the honest statement is that the control adds no
information about measured binding here, not that it actively destroys it, though
the point estimate was negative on every subset examined and stable as folds
accumulated (−0.056 at thirty designs, −0.092 at thirty-eight).

The mechanism is the one predicted. A permutation of a fifteen-residue peptide is
still a fifteen-residue peptide, plausibly able to occupy the same groove, so the
comparison isolates order from composition. A permutation of a hundred-residue
designed protein is not a protein: it does not fold, so its low score reflects the
destruction of tertiary structure rather than the loss of a binding-competent
arrangement, and that destruction is equally severe whether the design binds or
not.

---

## 3. Discussion

Each of the six controls reported here changed a conclusion. The permutation
control converted a positive discrimination result into a negative one and
identified composition as the operative variable. Replicate folding showed that
single-draw benchmarks report a sample of size one from a distribution wide
enough to reverse their ordering, and retracted two of our own claims. The
training-cutoff split removed about two thirds of the standardised effect. The
sampling-budget arm reversed the verdict on two readouts and confounded every
measurement taken before it. External validation against measured binding placed
the readout at 0.626 where our in-training panel gave 0.943. And the last control
bounded the first.

We draw three recommendations.

**Report a composition-matched control wherever a permutation of the candidate
remains a candidate.** For peptide screening it does, and there the control
distinguishes a metric that tracks binding from one that tracks composition. For
designed miniproteins it does not, and the raw readout is the better of the two.
The condition is not a technicality: it is the difference between a null that
isolates one variable and a null that destroys the molecule.

**Fold each candidate more than once.** Sampler noise on these readouts is large
enough that a single unseeded fold does not reproduce its own ranking. Where
compute forbids replication, the reported effect should be compared against a
run-to-run standard deviation measured once and quoted.

**State the sampling budget, and prefer it to other economies.** Reduced sampling
suppressed every effect here threefold to sevenfold, and carried 56–69% of the
recoverable gain by itself while alignment depth and recycling carried none.
Benchmarks run at reduced settings for throughput are measuring the sampler as
much as the model, and results obtained under them — including several of ours —
are lower bounds rather than estimates.

The broader point concerns what a confidence score is for. These scores are read
as estimates of binding, but they are trained as estimates of structural
self-consistency, and the two come apart precisely where screening needs them not
to: on candidates that are compositionally plausible but ordered wrongly, and on
targets the model has not seen. The gap between 0.943 on a training-adjacent
panel and 0.626 against measured binding is the size of that divergence in the
one case where we could measure both.

**Limitations.** The panels are small — 22 receptors on each side of the
training-cutoff split, 38 designs in the boundary test — and were folded on a
single consumer machine (Apple M-series, 17 GB unified memory), which constrained
both panel size and replication depth. The boundary test in §2.7 was folded at 10
sampling steps, and §2.5 establishes that a full-settings repeat could change its
magnitudes, though a difference this consistent in sign is unlikely to reverse.
The external comparison in §2.6 differs from our internal panels in panel,
positives and readout simultaneously, so it supports an ordering rather than a
numerical correction. Interface pLDDT is read from the same confidence head as
ipTM, making it a better readout of one model rather than an independent second
opinion. Finally, our results characterise the Boltz family specifically; whether
the same controls behave identically on AlphaFold 3 or its reimplementations is
untested here.

---

## 4. Methods

**Panel construction.** Receptor–peptide complexes were selected programmatically
from the PDB [10]: peptide chains of 5–25 residues, receptor chains of at least
50, no post-translationally modified residues, no expression tags, and
deduplication on both receptor and peptide sequence. The in-training panel used
structures released before the model's training cutoff; the held-out panel used
structures released after it, with decoys drawn from within the held-out set so
that no fold in that comparison involves a training structure.

**Permutation (scramble) generation.** For each peptide, permutations were
generated by a Fisher–Yates shuffle over the residue sequence, preserving
composition and length exactly. The random number generator is seeded per peptide
from a SHA-1 digest of that peptide's sequence, so that a candidate's null set is
a deterministic function of the candidate alone and does not change when other
candidates are added to a run.

**Folding.** Structures were predicted with Boltz-1 and Boltz-2 [4,5] on Apple
Silicon via the Metal Performance Shaders backend. Reduced settings were 10
sampling steps, 1 recycling pass, alignment depth 32; full settings were 200
sampling steps, 3 recycling passes and undiminished alignment depth (up to 12,882
rows). Alignments were built with MMseqs2 through the ColabFold server [11,12].
Each arm of the settings decomposition changed exactly one setting from the
reduced baseline, with all other quantities, the model, and the device held
constant.

**Interface readouts.** Interface residues are those with a representative atom
(CB, or CA for glycine) within 8.0 Å of a representative atom on the opposite
chain. Interface pLDDT is the mean CA pLDDT over the contacting residues of both
sides; the receptor-side variant restricts this to the receptor. ipTM is taken
from the model's confidence head. pDockQ follows Bryant et al. [7] and pDockQ2
substitutes a PAE-derived term [8]. Buried surface area is Shrake–Rupley.
Structures were parsed with gemmi and Biopython.

**Statistics.** Cognate-versus-permutation comparisons are paired within
candidate. Cohen's *d* is the paired form, mean(differences) / SD(differences).
Confidence intervals are bootstrap percentile intervals over receptors
(10,000 resamples). Rank tests compare the cognate's rank among its own decoys
against the chance value, by permutation. For ROC analysis, scores are
z-standardised within receptor before pooling, so that between-receptor offsets
do not contribute to the curve; the positive class is the cognate and the
negative class is every other candidate folded against that receptor, decoys and
permutations alike. The same convention, with target substituted for receptor,
is applied to the designed-miniprotein comparison, so that the AUCs in Table 7
are on one scale. Where a candidate's own permutation set is small,
the null standard deviation is pooled across candidates and the resulting ratio
is treated as a *t* statistic on the pooled degrees of freedom, not as a *z*.
Reproducibility percentages are the fraction of simulated re-runs, under the
measured run-to-run noise, in which the test remains significant at α = 0.05.

**External dataset.** The designed-miniprotein comparison uses the per-design
release of [6] (1,440 designs across 16 targets; 1,320 with a binding call across
15 targets after filtering to designs with two-vendor measurements). Labels are
measured binding from two independent contract research organisations. Interface
pLDDT was recomputed from the released Boltz-2 structures by the same code path
used on our own panels, with the binder chain identified by matching the released
binder length and all other chains pooled as the target. macro-AP is average
precision computed within target and averaged over targets, following that
release's convention.

**Pre-specification.** The boundary test of §2.7 had its prediction and its single
comparison written into source and committed before any fold in that experiment
was run; both are quoted verbatim in §2.7 as they stand in the repository history.

---

## 5. Data and code availability

Analysis code, panel construction scripts, and the screening tool are available
at the repository accompanying this work. The designed-miniprotein dataset is
released by [6] under CC BY 4.0 and is cited as specified in its `CITATION.cff`;
we redistribute none of it. Predicted structures generated in this study, the
replicate-fold measurements, and the per-arm settings decomposition outputs are
available on request.

## 6. Competing interests

The author declares no competing interests.

## 7. Acknowledgements

*[To be completed by the author — supervisory and institutional acknowledgements.]*

---

## References

1. Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583–589 (2021).
2. Evans, R., O'Neill, M., Pritzel, A. et al. Protein complex prediction with AlphaFold-Multimer. *bioRxiv* (2022).
3. Abramson, J., Adler, J., Dunger, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024).
4. Wohlwend, J., Corso, G., Passaro, S. et al. Boltz-1: democratizing biomolecular interaction modeling. *bioRxiv* (2024).
5. Passaro, S., Corso, G., Wohlwend, J. et al. Boltz-2: towards accurate and efficient binding affinity prediction. Technical Report, MIT (2025).
6. Claude Science and Shanehsazzadeh, A. Autonomous de novo protein binder design with Claude. Anthropic (2026). Dataset: huggingface.co/datasets/Anthropic/claude-protein-binder-design (CC BY 4.0).
7. Bryant, P., Pozzati, G., Elofsson, A. Improved prediction of protein–protein interactions using AlphaFold2. *Nature Communications* **13**, 1265 (2022).
8. Zhu, W., Shenoy, A., Kundrotas, P., Elofsson, A. Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes. *Bioinformatics* **39**, btad424 (2023).
9. Scarpellini, G., Shprints, R., Holderrieth, P. et al. Few-step cofolding with all-atom flow maps. *arXiv*:2606.08375 (2026).
10. Berman, H. M., Westbrook, J., Feng, Z. et al. The Protein Data Bank. *Nucleic Acids Research* **28**, 235–242 (2000).
11. Mirdita, M., Schütze, K., Moriwaki, Y. et al. ColabFold: making protein folding accessible to all. *Nature Methods* **19**, 679–682 (2022).
12. Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology* **35**, 1026–1028 (2017).
