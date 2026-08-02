# MPS vs CPU: 3x faster, equivalent on ipTM, unresolved on interface pLDDT

**Why.** Every benchmark in this project ran `--accelerator cpu` while the log
reported `GPU available: True (mps), used: False` — on a project whose premise is
Apple Silicon acceleration. If MPS agrees with CPU, the replicate-averaged
folding that Sections 7.5 and 7.6 call for becomes feasible on this machine
instead of requiring cloud GPU.

**Design.** 6 complexes spanning the ipTM range x 4 replicates x 2 devices = 48
folds, identical settings and cached MSAs. Folds are unseeded, so a paired test
that fails to reject is uninformative — equivalence is decided by **TOST**
against a margin set to the run-to-run noise the pipeline already tolerates.

## Speed: real, but less than the single-fold trial suggested

| device | per 6-complex batch | per complex |
| :--- | :--- | :--- |
| CPU | 492, 495, 497, 511 s | ~83 s |
| MPS | 124, 135, 201, 263 s | 21–44 s |

Median speedup **~3.0x**, best case 4.0x. An initial single-fold trial showed
5.2x; sustained batches do not hold that, and MPS timings vary by 2x across
replicates (124 s to 263 s) where CPU is stable within 4%. Thermal or contention
effects are the likely cause. **~3x is the number to plan with.**

## ipTM: a real bias, but inside the tolerance

```
mean MPS - CPU     -0.0266   95% CI [-0.0488, -0.0045]
within-device SD    0.0494
difference test     p = 0.0272   DIFFERENCE DETECTED
TOST (+/-0.0628)    p = 0.0043   EQUIVALENT within margin
```

Both lines are true and not in conflict: MPS scores systematically **lower** by
0.027, and that shift is demonstrably smaller than the +/-0.0628 run-to-run
noise. Five of six complexes moved negative.

**The practical consequence is a constraint, not a green light.** A 0.027
systematic shift is 42% of the noise SD and 38% of the cognate-minus-decoy
effect the benchmark measures (+0.070). Folding half a panel on CPU and half on
MPS would introduce a confound of the same order as the signal.

> **Do not mix devices within an analysis.** Re-fold a panel entirely on one
> device.

## Interface pLDDT: not established, and this is the one that matters

```
mean MPS - CPU     -1.1909   95% CI [-2.9475, +0.5657]
within-device SD    2.1090
difference test     p = 0.1418   no detectable difference
TOST (+/-1.9172)    p = 0.1682   NOT established
```

The confidence interval straddles zero but is *wider than the margin*, so a
meaningful shift cannot be ruled out. This is an underpowered result, not a
negative one: at the observed mean shift, TOST would clear at **n >= 17
complexes**. The current check used 6.

This matters more than the ipTM line, because Section 7.6 recommends ranking on
interface pLDDT rather than ipTM. The metric the recommendation depends on is
the one whose device-equivalence is unresolved.

## Per-complex detail

| complex | CPU ipTM | MPS ipTM | diff | CPU ifaceplDDT | MPS ifaceplDDT | diff |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| pair_035 | 0.5450 | 0.5170 | -0.0280 | 53.57 | 51.31 | -2.25 |
| pair_066 | 0.3933 | 0.3664 | -0.0269 | 46.38 | 42.29 | -4.09 |
| pair_085 | 0.5422 | 0.5498 | +0.0076 | 43.29 | 42.88 | -0.40 |
| pair_108 | 0.5681 | 0.5218 | -0.0463 | 47.95 | 48.01 | +0.06 |
| pair_111 | 0.3670 | 0.3167 | -0.0502 | 43.77 | 43.06 | -0.71 |
| pair_128 | 0.1581 | 0.1422 | -0.0159 | 38.33 | 38.58 | +0.25 |

## Recommendation

1. **Use MPS for new folding**, at ~3x, and re-fold whole panels on it rather
   than mixing with existing CPU results.
2. **Extend this check to >= 17 complexes before trusting MPS interface pLDDT.**
   It costs about an hour and closes the gap on the metric the project now
   recommends.
3. **Revise the claim that cloud GPU is a prerequisite.** At ~3x, the 1,200–2,100
   replicate folds are roughly 10–25 h locally rather than 30–50 h — long, but
   no longer a blocker.

## Reproduce

```
python src/mps_equivalence_check.py --replicates 4 --n-complexes 6
```
