# GPU run, 26 August 2026 — widening the noise floor from 4 receptors to 22

528 folds. 22 receptors x 6 complexes x 4 identical unseeded re-runs, the same
panel and settings as Section 7.5, which measured the same thing on 4 receptors.

## The question asked

Section 7.5's pooled SD -- 0.0628 ipTM, 1.917 interface pLDDT -- divides into two
effect-to-noise ratios, including the 8.6x that reaches the preprint's abstract.
It came from 4 receptors chosen to *span* the outcome range (cognate ranked
#1/#2/#3/#4). Selecting the extremes first is a mechanism for inflating a spread,
and nobody had checked whether it did.

## The answer: selection was not the problem

                          iface pLDDT     ipTM
  published 7.5 (4 rec)        1.9172   0.0628
  this run, same 4 rec         2.4787   0.0636
  this run, all 22 rec         2.3762   0.0646

  selection    (22 rec / same 4)     0.96x     1.02x
  reproduction (same 4 / published)  1.29x     1.01x

Widening 4 -> 22 moves the SD by 4% and 2%. The spanning sample was fine. The
hypothesis that motivated the run was wrong, which is what the split between
the two comparisons was built to be able to say.

## What the run found instead

On the same 4 receptors, interface pLDDT's spread is 29% larger than published,
while ipTM's is unchanged at 1%. The two runs differ in one thing: the original
folded on Apple Silicon CPU, this one on a CUDA 4090 with --no_kernels.

The mechanism is visible in the same data. interface_side_split.sides picks
interface residues by a coordinate distance cutoff and then averages their
head pLDDTs. That residue set is not stable:

  mean receptor-side interface            18.5 residues
  run-to-run SD of that count              6.35 residues   = 32% of its own mean

So interface pLDDT averages head values over a membership that reshuffles by a
third between identical re-runs, while ipTM is a single head scalar with no
coordinate dependence. A backend change moves one and not the other, which is
exactly the pattern here.

This refutes a sentence in the preprint (Section 2.3): "The ipTM and
interface-pLDDT rows are unaffected, both being read from the confidence head
rather than from coordinates." The values are from the head. The membership is
from coordinates.

## What was deliberately NOT changed

Table 5 keeps 1.917. Its +3.30 effect is measured on CPU-folded structures, and
an SD must match the regime of the effect it divides. Substituting a CUDA SD into
a CPU-measured ratio would be a worse error than the one it was meant to fix.

Whether the CPU SD on 22 receptors is also 1.917 is open, and needs a CPU run to
answer -- ideally `--accelerator cpu` on this same box, so only the backend
differs.

## Metrics that had no SD of their own until now

  receptor_side   2.3107
  peptide_side    2.9052
  peptide_whole   2.9387

make_presentation_figures.py had been using interface pLDDT's 1.9172 for
receptor_side, a substitution rather than a measurement.
