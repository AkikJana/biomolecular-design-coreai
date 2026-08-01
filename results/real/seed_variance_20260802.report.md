# Run-to-run variance of ipTM, and what it does to the binder benchmark

**Why this was run.** Every fold in every benchmark here was unseeded: Boltz's
`--seed` defaults to `None`, and the benchmark's own `--seed` controls only pair
construction, not the diffusion sampler. So every reported ipTM was a single
draw from a distribution whose width had never been measured — while the
load-bearing claim rested on a mean difference of **+0.0128**.

**Design.** 4 receptors spanning every outcome in the n=22 run (cognate ranked
#1, #2, #3, #4), all 6 complexes each, **4 replicates = 96 folds**, at the exact
settings the benchmark used. What is measured is the spread of the numbers
actually reported.

## ipTM is noisy at these settings

```
pooled within-complex SD : 0.0628   (24 complexes x 4 replicates)
median SD                : 0.0584
median range             : 0.1271
```

Against the two effects the conclusions rest on:

| effect | size | vs pooled SD |
| :--- | :---: | :--- |
| cognate − decoy | +0.0698 | 1.11x — comparable to noise |
| cognate − own scramble | +0.0128 | 0.20x — **below noise** |

## Per-receptor rankings are not reproducible

Cognate rank among its own 3 decoys, across 4 identical runs:

```
1YCR: [1, 2, 1, 1]     9F6S: [2, 1, 1, 1]
8KDX: [3, 3, 4, 2]     6YOO: [1, 4, 4, 3]
```

**Four of four receptors flip**, and 6YOO spans the entire range — best to worst
on identical input. The per-receptor rank tables previously published are single
noisy draws and should not be read as properties of those receptors.

## The aggregate effect survives; the significance verdict does not

Parametric bootstrap: re-simulate the whole 132-fold benchmark with the measured
noise (4,000 replications).

| quantity | as reported | on re-running |
| :--- | :---: | :--- |
| mean cognate rank | 2.00 | 2.03, 95% range **[1.77, 2.27]** |
| cognate #1 count | 8/22 | 9.0, 95% range [6, 12] |
| Wilcoxon p | 0.034 | median **0.054**, 95% range [0.004, 0.374] |
| | | **p < 0.05 in only 49% of re-runs** |

The effect is robust — every simulated re-run puts the mean rank below chance
(2.50), and well below. **The p < 0.05 verdict is a coin flip.** Reporting
"p = 0.034" as a stable fact overstates what a single run supports.

This is not a case of noise faking a signal. Measurement error attenuates rank
effects toward chance, so the underlying preference is probably *stronger* than
2.00 suggests. What noise destroys is the precision of any single run's verdict.

## Restating the composition claim

The earlier phrasing — ipTM is "indifferent to sequence order" — was a null
result read as an equivalence, and it was underpowered:

```
cognate - own scramble: +0.0128, 95% CI [-0.0186, +0.0441], n = 44
```

The defensible claim is a **bound**, not a zero: any order effect is at most
+0.044, i.e. **63% of the composition effect (+0.0698)**, and is plausibly zero.
Order contributes less than composition; it is not established to contribute
nothing.

The composition finding itself holds up better, because it aggregates over many
complexes rather than resting on a within-receptor difference:

```
scrambled > decoy: AUC 0.632, p = 0.0096  (44 vs 66)
```

Sequence-order garbage outscoring genuine binders of other receptors is a
110-complex comparison, and survives the noise.

## What this changes for the benchmark

Single-fold ipTM cannot support per-complex or per-receptor claims at these
settings. To stabilise a per-receptor rank the score SE must be small against
the gaps between competitors (~0.05):

| replicates | SE per score |
| :---: | :---: |
| 1 | 0.0628 |
| 4 | 0.0314 |
| 9 | 0.0209 |
| 16 | 0.0157 |

Roughly **9–16 replicate folds per complex** would be needed — 1,200–2,100 folds
for the full panel, which is a GPU-scale job rather than a CPU one. That makes
the deferred CUDA work a prerequisite for the next round rather than a
throughput nicety.

It also generalises beyond this project: ranking binders on a *single*
AlphaFold/Boltz run is common practice, and at reduced sampling settings a single
run does not reproduce its own ranking.

## Reproduce

```
python src/seed_variance_study.py --replicates 4
```
