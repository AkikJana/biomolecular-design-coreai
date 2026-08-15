# The reduced settings were suppressing the signal three- to seven-fold

Every fold in Sections 7.2 to 7.12 used 10 sampling steps against a default of
200, 1 recycling pass against 3, and an alignment subsampled to 32 rows.
Section 1.5 stated the confound and called it unresolved. It is now resolved,
and **the confound was real**.

## The comparison

Same panel, same model (stock Boltz-1), same device (MPS), only the settings
changed. The reduced arm already existed from Section 7.8, so model and device
are held constant by construction and settings are the only difference.

72 folds at full settings — every cognate and every scramble, which is what the
decisive test needs.

## Result

| metric | reduced | full | Cohen's *d* | within-receptor z |
| :--- | ---: | ---: | :--- | ---: |
| ipTM | +0.039 (p = 0.004) | **+0.287 (p < 1e-5)** | 0.45 → **1.25** | 4.0× |
| interface pLDDT | +1.54 (p = 0.067) | **+11.85 (p < 1e-5)** | 0.28 → **1.52** | 4.7× |
| receptor side | +0.71 (p = 0.428) | **+5.13 (p < 1e-5)** | 0.12 → **1.45** | 7.3× |

Raw effects are 7.2–7.7× larger. That is partly scale, since the model is far
more confident overall, so the standardised columns are the honest ones — and
they still show **2.7 to 12×**. Cohen's *d* moves from negligible-to-small to
large on all three readouts.

**Not outlier-driven**: larger at full settings for **21 of 22 receptors** on
ipTM and interface pLDDT, 18 of 22 on the receptor side, paired p ≤ 0.001
throughout.

Two readouts change verdict outright — interface pLDDT from p = 0.067 (the
number Section 7.8 called model-dependent) to p < 1e-5, and the receptor side
from p = 0.428, no evidence at all, to p < 1e-5.

## Why: it is Section 7.11's mechanism

| | reduced | full |
| :--- | ---: | ---: |
| cognate ipTM | 0.203 | **0.810** |
| cognate interface pLDDT | 47.79 | **90.70** |

And the backbone becomes physical:

| arm | median CA–CA | physical (3.4–4.2 Å) | broken (> 5 Å) |
| :--- | ---: | ---: | ---: |
| Boltz-2 @ 10 steps | 5.48 Å | 14.0% | 56.2% |
| DeCAF @ 10 steps | 3.74 Å | 96.2% | 0.0% |
| **Boltz-1 @ 200 steps** | **3.80 Å** | **99.7%** | **0.0%** |

3.80 Å is the ideal peptide bond to two decimals. Section 7.11 showed
geometry-dependent readouts need a converged sampler and that distillation
restores convergence; this is the other way of restoring it, and the confidence
readouts recover with it.

## The obstacle was the device, not the settings

A full-settings fold takes **106–109 seconds** on this laptop. The runner's own
comment records that a full alignment was intractable on CPU — a 40-complex run
did not finish one batch in an hour and drove the machine to ~12 GB of swap. On
MPS, full alignment depth is no slower than depth 32.

The reduced regime that shaped every result in Section 7 outlived the constraint
that justified it.

## What it changes

**Section 7.4 needs qualifying.** "ipTM tracks composition, not binding" was
measured at 10 steps. At 200, ipTM separates a cognate from its own scramble at
*d* = 1.25. The finding is a property of the reduced regime at least as much as
of the metric.

**Section 7.8 is reframed, not overturned.** It compared DeCAF against stock
Boltz-1 *at the same reduced settings* and found 5–6×; that comparison stands.
But full-settings Boltz-1 reaches +0.287 and +11.85 where DeCAF at ten steps
reached +0.201 and +9.54, so distillation is not adding something the stock
model lacks — it recovers most of what the stock model has when run properly, at
a twentieth of the sampling budget. For an efficiency thesis that is the better
result.

**Sections 7.6, 7.7 and 7.10 are lower bounds.**

## Limitations

One model, 22 receptors, one draw. Only cognates and scrambles were folded, so
this resolves the scramble control and not the receptor-specificity rank test —
that needs the decoys, another 66 folds at ~2 minutes each. The three settings
were raised together, so which of sampling steps, recycling or MSA depth carries
the effect is unresolved; Section 7.11's geometry result points at sampling
steps, and `settings_confound.py` varies each independently for that test.

Section 7.5's warning applies here too: this is a single draw, and a single draw
has twice misled this dissertation.

## Reproduce

```
python src/settings_confound.py --batch-size 8 --labels cognate,scrambled
python src/settings_confound.py --analyse-only
```
