# Composition-matched controls change what cofolding confidence metrics appear to measure

**Akik Jana**

Birla Institute of Technology and Science, Pilani — Work Integrated Learning Programmes, Pilani, Rajasthan, India

Correspondence: akik.e.aj@gmail.com

---

## Abstract

Confidence scores from cofolding models — ipTM, interface pLDDT, pDockQ and their
relatives — are widely used to rank candidate binders before synthesis. On 1,320
designed miniproteins whose binding was measured by two independent contract
research organisations, and which postdate every model's training data, nine
published readouts span within-group AUC 0.621–0.671 against a 0.268 base rate:
every confidence interval covers or nearly covers chance. Our own recommended
readout reaches AUC 0.626 there, where our structure-derived panel gives 0.943.
This paper explains that gap with six controls, each of which overturned a result
that had looked solid without it. A composition-matched permutation control —
scoring a cognate peptide against permutations of itself, which fix amino-acid
composition and length and destroy only order — shows that ipTM ranks cognates
above unrelated decoys (mean rank 2.00 against chance 2.50) while failing to
separate a cognate from its own permutation; permutations in fact outscore decoys.
Of six interface readouts on identical structures, only interface pLDDT passes.
Replicate folding shows single unseeded folds do not reproduce their own ranking,
which forced two retractions in our own work. A training-cutoff split leaves
38–40% of the standardised effect. Reducing sampling from 200 steps to 10
suppresses every effect threefold to sevenfold, and at ten steps the predicted
peptide backbone is not a connected chain at all; a full factorial over the three
inference settings shows alignment depth matters only once the sampler has
converged, and recycling not at all. We then test the controls where
they were not developed. On a panel 2.7 times larger the effect survives
(p = 5.3 × 10⁻¹² on 50 of 59 receptors) with every effect size 20–50% smaller, so
our original panel was a favourable draw. Under Chai-1, an independent model
family scored with the same readout code, a cognate beats its own permutation on
20 of 21 receptors, and the ranking test transfers almost exactly (top-1 76%
against 77%): order sensitivity is a property of cofolding confidence, not of one
model. Finally we bound the control itself — on 60–120 residue designed
proteins it carries no information at either sampling budget, because permuting a
folded protein destroys binders and non-binders alike. Use it where a permutation
of the candidate remains a candidate.

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

We begin from the outside. A recently released dataset of designed miniproteins
whose binding was measured experimentally [6] permits the comparison a
structure-derived panel cannot make: how a confidence readout performs against
binding somebody actually observed, on designs that postdate every model's
training data. Every readout scored on it, ours included, lands close to chance
(§2.1) — far below what benchmarks of these same readouts report.

The rest of the paper accounts for that gap. We apply six controls to a
peptide-binder screening panel of our own, and each removes part of it: the
permutation control (§2.2), the choice of readout (§2.3), replicate folding
(§2.4), a training-cutoff split (§2.5), and the sampling budget (§2.6). Each was
capable of overturning a result, and each did. We then test the controls
themselves — on a panel 2.7 times larger (§2.8) and on a second model family
(§2.9) — because a control that only works at the scale and on the model where it
was developed is not a control. We report the negative results in
full, including two conclusions of our own that the controls forced us to retract.
Finally we bound the first control by testing it against measured binding as well
(§2.7), and find a regime where it carries nothing.

---

## 2. Results

### 2.1 Against binding that was actually measured, no readout is far above chance

We begin with the check that does not depend on any panel we built. Benchmarks of
cofolding confidence almost always score a cognate pair that was *crystallised*,
which is evidence that two molecules can be co-ordered in a lattice, not a
measurement of binding. A recently released dataset [6] permits the direct test:
1,320 designed miniproteins across 15 targets, with binding measured by two
independent contract research organisations, and with designs postdating every
model's training data. 354 (26.8%) bind.

We recomputed interface pLDDT exactly as described in §4 from the Boltz-2 structures
released with that dataset. One implementation detail mattered: five targets are
oligomeric, so taking the first two chains as receptor and binder would have
measured a target against itself. The binder chain was identified by matching the
released binder length, with every other chain pooled as the target. All 1,320
designs scored.

**Table 1. Readouts against measured binding, and against structure-derived panels.**

| Setting | within-group AUC | macro-AP |
| :--- | ---: | ---: |
| Interface pLDDT, measured binding | 0.626 | 0.436 |
| ipSAE, same structures | 0.628 | 0.436 |
| chance | 0.500 | 0.268 |
| — for scale, same AUC convention (§2.5, §2.6) — | | |
| this work, in-training panel, full settings | 0.908–0.943 | — |
| this work, held-out panel, full settings | 0.683–0.758 | — |

All AUCs in this table use one convention: scores are z-standardised within
group — receptor for our panels, target for the designs — and pooled into a
single ROC. The held-out row is the mean of two independent full-settings draws.

The two lower rows are established in §2.5 and §2.6 and are quoted here only to
set the scale. These are not the same measurement: the panels differ, the
positives differ (a cognate crystal pair against a synthesised binder), and the
readouts differ in part. What they share is the question, and the ordering answers it. A readout
scored on complexes the model was trained on reads far higher than the same
family of readout scored against binding that was actually measured. The
contamination penalty of §2.5, measured internally on a training-cutoff split,
appears again from outside.

For context, nine other predictors scored on the same designs span within-target
AUC 0.621–0.671 and macro-AP 0.453–0.532. No structure-prediction confidence
readout in this comparison, ours included, is close to its own benchmark figures.

![No confidence readout is far above chance on binding that was actually measured](assets/fig_wetlab.png)

<p class="figcap"><b>Figure 1.</b> (a) macro-AP against measured binding for
every predictor scored on the release, with 95% confidence intervals; the dashed
rule is the base rate. Red bars are this work's readouts. Every interval covers
or nearly covers chance. (b) The same readout family across three regimes on one
AUC convention. The descent from a training-adjacent panel to measured binding is
about 0.28–0.32 AUC, and the held-out panel sits between the two. Panels differ
in molecules and in what counts as a positive, so (b) supports an ordering rather
than a numerical correction.</p>

### 2.2 A composition-matched control separates two hypotheses that decoys conflate

The gap in §2.1 — 0.943 on a structure-derived panel against 0.626 on measured
binding — is the thing the rest of this paper explains. Sections 2.2 to 2.6 apply
six controls to a panel of our own, and each one removes part of that gap.

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

**Table 2. Mean ipTM by class on the 22-receptor panel (10 sampling steps).**

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

### 2.3 Of six interface readouts on identical structures, one passes

The finding in §2.2 indicts ipTM, not the predicted structures. We re-scored the
same 132 structures with six interface measures, so that the comparison between
readouts is free of folding variance by construction. Contacts use an 8 Å
CB–CB criterion (CA for glycine); pDockQ follows Bryant et al. [7]; buried area
is Shrake–Rupley.

**Table 3. Interface readouts on the permutation control.**

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

<p class="figcap"><b>Figure 2.</b> ipTM (left) and interface pLDDT (right) by
class on the 22-receptor panel. Boxes span the interquartile range; the bracket
gives the paired test of a cognate against its own scramble. ipTM places
cognates above decoys while failing to distinguish them from permutations of
themselves; interface pLDDT separates both.</p>

The cause of the inverted contact ordering is unconverged geometry rather than
peptide length: at 10 sampling steps only 14% of backbone bonds in these
structures fall within physically plausible bounds, against 96% for a
few-step-distilled model [9], and on converged structures the contact ordering
reverses to the sensible direction (cognates 61.4, permutations 51.8).

That claim rests on a bespoke distance metric, so we checked it against
PoseBusters [13], the standard structural-validity suite, on both regimes. The
peptide chain is treated as the molecule under test.

**Table 4. Peptide connectivity by sampling budget, PoseBusters / RDKit.**

| | 10 sampling steps | 200 sampling steps |
| :--- | ---: | ---: |
| structures | 144 | 354 |
| peptides that are a single connected fragment | **0 (0%)** | **354 (100%)** |
| median fragments per peptide | **40.5** | **1** |

At ten steps the backbone is not connected at all — a fifteen-residue peptide
comes back as roughly forty disjoint pieces — and at two hundred it is a single
chain in every structure. This is the physical form of the objection, in the
field's own vocabulary rather than ours.

![Ten sampling steps do not return a connected backbone](assets/fig_connectivity.png)

<p class="figcap"><b>Figure 3.</b> Connected fragments per peptide, by sampling
budget, on a shared axis. At two hundred steps every peptide is a single chain;
at ten, none is, and the distribution centres near forty pieces. A fifteen-residue
peptide is not returned as a peptide.</p>

One caution about that suite, because it would otherwise be misread. PoseBusters
is built for small molecules, and three of its checks are **vacuous** on a
peptide read from a PDB file: RDKit perceives bonds by distance, so a backbone
bond stretched past bonding range is never perceived as a bond and cannot fail a
length, angle or planarity test. Those checks report 100% pass on the very
structures that are in forty pieces. Its `all_atoms_connected` check is worse
than uninformative here — it reports 0% pass on the converged set, while RDKit's
own fragment count on the identical molecule returns one. **The fragment counts
in Table 4 are the trustworthy quantity**; we report no other PoseBusters
column. Every
contact-derived row of Table 3 should be read as describing point clouds rather
than complexes. The ipTM and interface-pLDDT rows are unaffected, both being read
from the confidence head rather than from coordinates.

### 2.4 Single folds do not reproduce their own ranking

Boltz's sampler seed defaults to unset, so each reported score is one draw from a
distribution whose width is not reported. We folded a subset 96 times to measure
that width, then compared each readout's effect on the permutation control
against its own run-to-run spread.

**Table 5. Permutation-control effect against run-to-run spread.**

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

### 2.5 A training-cutoff split removes about two thirds of the standardised effect

Panels drawn from the PDB overlap the training data of every model trained on the
PDB. We assembled a second panel of 22 receptors released after the model's
training cutoff, filtered identically, with decoys drawn from within the held-out
set so that no fold in the comparison involves a training structure. Both panels
were then folded at the model's intended settings (200 sampling steps, 3
recycling passes, undiminished alignment depth), so that the only quantity
varying between them is whether the model has seen the complex.

**Table 6. The contamination penalty at full settings.**

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

<p class="figcap"><b>Figure 4.</b> Cohen's <i>d</i> on the scramble control
across the full 2×2 of training exposure and sampling budget, for three
readouts. Raising the sampling budget multiplies the in-training effect by 1.7
to 1.9 (left to right, blue bars); withholding the complex from training removes
about three fifths of it at full settings (blue to amber), and between a third
and three quarters at reduced settings, where the estimate is itself unstable.
Percentages give the standardised retention held out. Dotted rules mark Cohen's conventional
small/medium/large thresholds.</p>

### 2.6 Sampling budget confounds every comparison above

The measurements in §2.2–2.3 were taken at 10 sampling steps of an intended 200,
a reduction adopted for throughput on consumer hardware. We folded the same panel
on the same model and device at full settings to measure what that reduction had
cost.

**Table 7. Permutation control at reduced and full settings.**

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

This has a direct bearing on §2.2. ipTM's indifference to sequence order is a
property of the reduced sampling regime at least as much as of the metric: at 200
steps the same model on the same panel separates a cognate from its own
permutation at *d* = 1.25. The claim that survives is the narrower one — **ipTM
is indifferent to sequence order when the sampler has not converged** — which
§2.3 identifies as the regime in which the predicted backbone is 14% physically
plausible.

To locate the cost we folded the full factorial: each setting alone, each pair,
and both endpoints — eight cells over the same 22 receptors, 1,056 folds.

**Table 8. The full factorial, order-sensitivity test on interface pLDDT.**

| Arm | steps / recycles / MSA | effect | Cohen's *d* | share of full gain |
| :--- | :--- | ---: | ---: | ---: |
| reduced | 10 / 1 / 32 | +1.54 | 0.28 | — |
| **sampling** | 200 / 1 / 32 | **+8.82** | **1.13** | **69%** |
| alignment | 10 / 1 / full | +0.83 | 0.18 | −8% |
| recycling | 10 / 3 / 32 | +1.40 | 0.22 | −5% |
| samp + recyc | 200 / 3 / 32 | +9.32 | 1.05 | 62% |
| **samp + align** | 200 / 1 / full | **+12.43** | **1.46** | **96%** |
| recyc + align | 10 / 3 / full | +1.84 | 0.30 | 1% |
| full | 200 / 3 / full | +11.85 | 1.52 | 100% |

Sampling steps alone carry **69%** of the standardised gain and take the readout
from marginal to p = 2.5 × 10⁻⁹. Moved alone, neither of the others carries any:
both shares are negative, each arm sitting marginally *below* the reduced
baseline.

**The pairs show that reading is incomplete.** An earlier version of this work
stopped at the single-knob arms and concluded that alignment depth does not
matter. It does — but only in company. Combined with sampling it reaches
*d* = 1.46, which is 96% of the full effect from two knobs rather than three, and
the interaction is genuinely positive:

**Table 9. Each pair against the sum of its two single knobs, standardised.**

| pair | observed gain | additive prediction | interaction | |
| :--- | ---: | ---: | ---: | :--- |
| sampling + recycling | +0.77 | +0.78 | **−0.01** | additive |
| **sampling + alignment** | **+1.18** | +0.74 | **+0.44** | **synergy** |
| recycling + alignment | +0.01 | −0.17 | +0.19 | weak synergy |

Deep alignments do nothing while the sampler has not converged and contribute
substantially once it has. That is a coherent mechanism rather than a share: the
coevolutionary signal in a deep alignment can only express itself through a
pair representation the sampler has actually resolved, which §2.3 shows it has
not at ten steps.

Recycling is the knob that is genuinely inert. Its interaction with sampling is
**−0.01** — additive to two decimal places — and `sampling + recycling` slightly
*underperforms* sampling alone (1.05 against 1.13), which is within the
draw-to-draw noise of §2.4 and should be read as "no effect" rather than as harm.

**The practical recommendation, revised.** Under a compute constraint, spend first
on sampling steps and second on alignment depth; recycling passes are the ones to
drop. At 200 sampling steps, one recycling pass and undiminished alignment,
`sampling + alignment` recovers 96% of the full effect while skipping two
recycling passes over every fold. An earlier version of this recommendation said
to cut alignment depth first, on the strength of the single-knob arm alone; the
factorial shows that to be wrong.

### 2.7 Where the control stops working

The permutation control had itself never been tested against measured binding. We
registered a prediction in source, committed before any fold ran:

> These binders are 60 to 120 residue designed proteins, not the 5 to 25 residue
> peptides of the panel. A permutation of a 100-residue protein does not fold at
> all, so both binders and non-binders should receive an equally ruined scramble,
> and the subtraction may add nothing.

Thirty-eight designs across four targets (RBX1, PD-L1, TrkA, BHRF1) were folded
as delivered and against two permutations of each.

**Table 10. Interface pLDDT of a design and of its own permutations, by measured outcome.**

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

**The result is not an artifact of the sampling budget.** Those folds ran at ten
sampling steps, and §2.6 establishes that reductions of that kind suppress every
effect we measure — so a reasonable objection is that the control might work here
at converged settings and we simply could not see it. We therefore repeated the
experiment at 200 sampling steps and 3 recycling passes, on the full 48 designs
rather than the 38 the first attempt reached.

**Table 11. The same experiment at converged settings, 48 designs.**

| | n | design | its permutations | margin |
| :--- | ---: | ---: | ---: | ---: |
| measured binders | 24 | 80.08 | 59.27 | **+20.81** |
| measured non-binders | 24 | 81.30 | 59.02 | **+22.28** |

The conclusion is unchanged: Welch p = 0.677, and on the pre-specified comparison
ΔAUC = **−0.104**, 95% CI [−0.330, +0.071]. Converging the sampler did not make
the control informative.

Two details are worth stating rather than smoothing. At converged settings the
non-binder margin is marginally the *larger* of the two, so the point estimate
changes sign relative to the reduced-settings run — nowhere near significance,
but it reinforces the reading that the subtraction is a constant rather than a
weak signal. And the raw readout itself performs worse on this panel at converged
settings (within-target AUC 0.543 against 0.672), which widens the interval
considerably; the interval, not the point estimate, is the result.

The mechanism is the one predicted. A permutation of a fifteen-residue peptide is
still a fifteen-residue peptide, plausibly able to occupy the same groove, so the
comparison isolates order from composition. A permutation of a hundred-residue
designed protein is not a protein: it does not fold, so its low score reflects the
destruction of tertiary structure rather than the loss of a binding-competent
arrangement, and that destruction is equally severe whether the design binds or
not.

### 2.8 The panel at 2.7 times the size

Every number above rests on 22 receptors, which is the first thing a reader should
distrust. We therefore extended the panel with the identical programmatic screen —
570 further PDB entries examined, 37 accepted, the rest rejected for a required
post-translational modification (43), receptor redundancy (20), or a peptide too
close to one already accepted (20). The combined 59 receptors were folded together
in one run at converged settings, so nothing is merged across hardware or dates.

**Table 12. The permutation control at 22 and 59 receptors, folded identically.**

| readout | n = 22 | n = 59 | *d* at 22 | *d* at 59 | receptors where cognate wins |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ipTM | +0.287 | +0.268 | 1.16 | 0.99 | 48 / 59 |
| Interface pLDDT | +11.89 | +10.65 | 1.43 | **1.04** | **50 / 59** |
| Receptor side | +5.27 | +5.08 | 1.67 | **0.77** | 47 / 59 |

**The effect survives and the effect sizes fall.** Raw margins hold to within 10%,
and interface pLDDT still separates a cognate from its own permutation at
p = 5.3 × 10⁻¹² on 50 of 59 receptors. But Cohen's *d* drops on every readout —
interface pLDDT 1.43 → 1.04, and the receptor side more than halves, 1.67 → 0.77.

The honest reading is that **our original 22 receptors were a favourable draw**,
and the larger panel gives the better estimate. We report the smaller numbers as
the ones to use. The receptor-side readout in particular should not be quoted at
its 22-receptor value; at 59 it is a moderate effect, not a large one.

The settings confound of §2.6 was re-measured on the same 59 receptors, both arms.
Reduced sampling suppresses the ipTM effect **12× in raw terms and 3.6× in
standardised terms** (+0.022, *d* = 0.29 against +0.268, *d* = 0.99), against 7.4×
and 2.8× at 22 receptors. That conclusion strengthens with panel size rather than
weakening.

### 2.9 The control on a second model family

Everything so far measures one family of models. Whether ipTM's indifference to
sequence order is a property of *cofolding confidence* or a property of *Boltz*
cannot be settled from within it.

We folded the same 132-fold panel — same receptors, same cognates, same
permutations — under Chai-1 [14], and scored the output with the identical
interface-pLDDT implementation used throughout this paper. The readout code is
shared, so the model is the only thing that varies.

**Table 13. The permutation control under two independent models.**

| | Chai-1 | Boltz-1 |
| :--- | ---: | ---: |
| receptors | 21 | 22 |
| cognate, interface pLDDT | 94.70 | 90.35 |
| its own permutation | 88.73 | 78.45 |
| margin | **+5.97** | +11.89 |
| paired *t* | **1.8 × 10⁻⁴** | 1.4 × 10⁻⁷ |
| Cohen's *d* | **0.90** | 1.43 |
| cognate beats its own permutation | **20 / 21** | 21 / 22 |

**Interface pLDDT distinguishes a peptide from a permutation of itself in Chai-1
as well**, on 20 of 21 receptors. The order sensitivity is therefore a property of
the readout family rather than of one model's confidence head.

It is weaker in Chai-1 — *d* = 0.90 against 1.43 — and the claim should say so.
What replicates is the direction and the significance, not the magnitude.

**The ranking test, however, replicates almost exactly.** Ranking a cognate
against its own decoys is the question a screen actually asks, and on it the two
models are indistinguishable:

**Table 14. Ranking a cognate against its own decoys, under two model families.**

| | Chai-1 | Boltz-1 |
| :--- | ---: | ---: |
| receptors | 21 | 22 |
| mean rank (chance 2.50) | **1.38** | 1.41 |
| cognate ranked first | 16 / 21 | 17 / 22 |
| top-1 accuracy | **76%** | 77% |
| P(cognate > a decoy of its own receptor) | **0.873** | 0.864 |
| within-group AUC | 0.872 | 0.914 |

Top-1 accuracy differs by one percentage point and mean rank by 0.03. On the
measure a practitioner would actually use to pick candidates for synthesis, the
choice between these two models does not matter.

That asymmetry is worth stating rather than smoothing: **the ranking test
transfers between model families almost perfectly, and the permutation control
transfers in direction but at 60% of the effect size.** The two tests are asking
different questions — one about a candidate against its competitors, the other
about a candidate against itself — and the second is the more model-dependent of
them. Note
also that Chai-1's interface pLDDT sits near 95 where Boltz-1's sits near 90 on
the same complexes, so the two models are differently calibrated and only the
standardised comparison is meaningful; the raw margins are not comparable.

Chai-1 does not run on Apple Silicon — the pair representation's broadcast
outer-product matmul has no Metal implementation, and on CPU a single 66-residue
complex took 2 h 47 min — so this arm required a CUDA device.

![The control holds at 2.7x the panel and on another model](assets/fig_robustness.png)

<p class="figcap"><b>Figure 5.</b> The control tested where it was not developed.
(a) Cohen's <i>d</i> at 22 and 59 receptors: the effect survives on every readout
and shrinks on every readout, most severely on the receptor side. (b) Per-receptor
margins under both model families; each point is one receptor, the bar is the
mean, and the dashed line is the no-order-sensitivity null. Chai-1 gives a smaller
margin than Boltz-1 and clears zero on 20 of 21 receptors.</p>

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

We then applied the same scepticism to the controls themselves, because a control
that only works at the scale and on the model where it was built is not a control.
Both survived, and neither survived unchanged. On 59 receptors the permutation
effect holds at p = 5.3 × 10⁻¹² while every effect size falls 20–50%, which says
our original panel was a favourable draw and the published magnitudes are the
optimistic end. Under a second model family the effect replicates in direction and
significance but at *d* = 0.90 against 1.43. **Both corrections point the same way:
the phenomenon is robust, our estimates of its size were not.** We would rather
report that than the larger numbers we started with.

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
suppressed every effect here threefold to sevenfold, and carried 69% of the
recoverable gain by itself. The full factorial refines what to do with the rest:
alignment depth contributes nothing alone but interacts positively with sampling
(+0.44), so the ordering is sampling first, alignment second, and recycling —
whose interaction is −0.01 — is the pass to drop.
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

**Limitations.** The main panel is now 59 receptors, but the training-cutoff split
of §2.5 still rests on 22 a side and has not been extended to match. The factorial
of §2.6 is complete for interface pLDDT on the in-training panel at one draw per
cell; the interactions it reports are of the same order as §2.4's run-to-run
spread, so the sign of the sampling-alignment synergy is better supported than its
magnitude. The boundary
test covers 48 designs across four targets, and has now been run at both reduced
and converged settings with the same answer, which retires the sampling-budget
caveat that stood in an earlier version of this work. The second-model arm of §2.9
covers 21 receptors and now carries both the permutation control and the ranking
test, but a single model beyond Boltz: whether a third family behaves like either
is untested. Chai-1's interface pLDDT is calibrated differently enough that only
standardised effects compare across the two. The external comparison in §2.1 differs from our internal panels in panel,
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
is applied to the designed-miniprotein comparison, so that the AUCs in Table 1
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

Panels, per-fold scores and the structural-validity runs are published as a
dataset: **`AkikJana/scramble-control-panels`** — 2,456 folds across 16 inference
arms, 75 receptors and two model families, with the sequences as folded. Every
figure in this paper is recomputable from those tables alone.

The control can be run in a browser on any predicted complex at
**`AkikJana/scramble-control-scorer`**, which generates the permutations to fold
and scores a candidate against them using the implementation described in §4.

Analysis code, panel construction scripts and the screening tool are available
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
13. Buttenschoen, M., Morris, G. M., Deane, C. M. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. *Chemical Science* **15**, 3130–3139 (2024).
14. Chai Discovery team. Chai-1: decoding the molecular interactions of life. *bioRxiv* (2024).
10. Berman, H. M., Westbrook, J., Feng, Z. et al. The Protein Data Bank. *Nucleic Acids Research* **28**, 235–242 (2000).
11. Mirdita, M., Schütze, K., Moriwaki, Y. et al. ColabFold: making protein folding accessible to all. *Nature Methods* **19**, 679–682 (2022).
12. Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology* **35**, 1026–1028 (2017).
