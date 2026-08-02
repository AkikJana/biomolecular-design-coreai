# Interface pLDDT ranks binders where ipTM does not

**A positive result, from compute already spent.** Sections 7.2–7.5 established
that ipTM tracks peptide composition rather than binding. That is a statement
about ipTM, not about the predicted structures — 132 of which were still on
disk. Re-scoring them with other interface measures cost no folding at all.

## The comparison

All six metrics, same 132 complexes, same tests:

| metric | cognate | scrambled | decoy | **cognate vs own scramble** | rank | chance | rank p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ipTM | 0.502 | 0.489 | 0.432 | p = 0.416 | 2.00 | 2.50 | 0.034 |
| pDockQ | 0.474 | 0.466 | 0.404 | p = 0.797 | 2.14 | 2.50 | 0.108 |
| **interface pLDDT** | **49.60** | **46.31** | **45.93** | **p < 0.0001** | **1.91** | 2.50 | **0.027** |
| contacts | 32.9 | 38.5 | 34.0 | p = 0.054 | 2.55 | 2.50 | 0.854 |
| contact density | 3.24 | 3.69 | 3.62 | p = 0.122 | 2.55 | 2.50 | 0.880 |
| buried SASA | 1800 | 1861 | 1632 | p = 0.454 | 2.27 | 2.50 | 0.407 |

The decisive column is *cognate vs its own scramble*. A scramble holds
composition and length exactly fixed and destroys only order, so a metric that
cannot beat it is not responding to order — and order is what makes a binder a
binder. **Only interface pLDDT passes.**

Note what pDockQ does: it combines interface pLDDT with contact count, and the
contact term is *anti*-correlated with the cognate (scrambles make **more**
contacts, 38.5 against 32.9). Mixing the two destroys the signal that interface
pLDDT carries alone.

## It survives the test that demoted ipTM

The 96 replicate folds from the reproducibility study were re-scored the same
way, giving a run-to-run standard deviation for each new metric:

| metric | effect on the scramble control | run-to-run SD | effect / noise |
| :--- | ---: | ---: | ---: |
| ipTM | +0.0128 | 0.0628 | **0.20x** |
| pDockQ | +0.008 | 0.1498 | 0.05x |
| **interface pLDDT** | **+3.30** | **1.917** | **1.72x** |

Interface pLDDT has roughly **8.6x** the effect-to-noise ratio of ipTM on the
test that matters. Simulating the whole benchmark under the measured noise:

```
cognate - own scramble  +3.30 pLDDT, 95% CI [+2.13, +4.47], n = 44
  reproduces at p < 0.05 in  100%  of simulated re-runs
within-receptor rank    1.91 of 4 (chance 2.50), 95% CI [1.64, 2.14]
  reproduces at p < 0.05 in   84%  of re-runs   (ipTM managed 49%)
```

## What this does and does not establish

**Established: order sensitivity.** Interface pLDDT distinguishes a peptide from
its own scramble, at 100% reproducibility and comfortably past a Bonferroni
correction for the six metrics tested (alpha = 0.0083). This is precisely what
ipTM could not do, and it is the property a screening reference needs.

**Suggestive, not established: receptor specificity.** The within-receptor rank
against real decoys is 1.91 against a chance value of 2.50 with p = 0.027 — which
does *not* clear the Bonferroni threshold, though it reproduces in 84% of
re-runs. More receptors would settle it.

**Caveat.** Interface pLDDT is read from the same confidence head as ipTM, so
this is not an independent model — it is a better *readout* of the same one.
The practical implication stands either way: the information needed to rank
these binders is present in the prediction, and ipTM discards it.

## Consequence

The project's recommendation changes from "folding-model confidence cannot rank
peptide binders" to something more useful:

> **Rank on interface pLDDT, not ipTM, and never on pDockQ for short peptides —
> its contact term cancels the signal.**

## Reproduce

```
python src/rescore_interface_metrics.py
```
