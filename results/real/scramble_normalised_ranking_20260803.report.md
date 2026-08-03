# Scramble-normalised scoring does not work

**The idea.** Section 7.4 showed the ranking signal is contaminated by peptide
composition and length. Every candidate therefore carries its own exact control —
a scramble of itself — so rank on

```
delta(candidate) = score(candidate) - mean score(scrambles of that candidate)
```

Unlike a global correction, this cancels composition *per candidate* with no
assumption that the bias is uniform. It was the most promising methodological
idea the project's own findings suggested.

**It does not work.** It makes ranking worse, for both metrics.

## Result

264 folds — 22 receptors x (1 cognate + 3 decoys) candidates, plus 2 independent
scrambles of each — all on MPS, so the device offset cancels inside the
within-candidate difference.

| ranker | cognate #1 | mean rank | chance | p |
| :--- | ---: | ---: | ---: | ---: |
| ipTM, raw | 9 / 22 | **1.73** | 2.50 | 0.0006 |
| ipTM, scramble-normalised | 9 / 22 | 2.23 | 2.50 | 0.2755 |
| interface pLDDT, raw | 13 / 22 | **1.55** | 2.50 | 0.0003 |
| interface pLDDT, scramble-normalised | 10 / 22 | 1.95 | 2.50 | 0.0331 |

Normalisation costs about half a rank on both metrics, and pushes ipTM from
clearly significant to not significant at all.

## Why — the arithmetic was against it from the start

Delta subtracts two noisy quantities, so with k scrambles its noise is
`sqrt(1 + 1/k)` times the raw score's: **1.22x at k = 2**. Measured directly
here, the scramble mean carries a standard error of 0.033 for ipTM and 1.61 for
interface pLDDT — the same order as the effects being ranked. The bias removed
did not pay for the variance added.

This was stated as the expected failure mode before the run, not after it.

**Could more scrambles rescue it?** The penalty falls as `sqrt(1 + 1/k)` — 1.06x
at k = 8. But that means **9 folds per candidate**, and interface pLDDT already
achieves rank 1.55 raw. Paying 9x the compute to correct a bias that the better
readout largely avoids is not a trade worth making. The idea is not merely
underpowered here; it is the wrong shape for the problem.

## A useful by-product

These are 264 fresh MPS folds, entirely independent of the CPU benchmark, and
they reconfirm the Section 7.6 conclusion from new data:

```
interface pLDDT  rank 1.55, 13/22 first, p = 0.0003
ipTM             rank 1.73,  9/22 first, p = 0.0006
```

Interface pLDDT ranks better than ipTM again, on a different device and a
different set of folds.

**One number here should not be over-read.** Raw ipTM reaches rank 1.73 in this
run against 2.00 in the benchmark, which is at the edge of the [1.77, 2.27] range
the reproducibility simulation predicts. That is a favourable draw, a device
difference, or both — it is not evidence that ipTM is a good ranker. The scramble
control settled that question and this run does not revisit it.

## Conclusion

A negative result on a method the project's own findings motivated. It is
reported because the reasoning that produced it was sound and the failure is
informative: **per-candidate control by scrambling is too noisy to be worth its
cost at any practical k**, and effort is better spent on the readout than on
correcting a bad one.

## Reproduce

```
python src/scramble_normalised_ranking.py --batch-size 12 --device mps
```
