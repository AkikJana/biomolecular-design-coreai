# Few-step distillation, and what it reveals about the earlier negatives

Every result in Sections 7.2–7.7 was folded at 10 sampling steps from models
whose default is 200. Section 1.5 lists that gap as stated but unresolved, and it
is the obvious objection: the failures might be artefacts of under-sampling
rather than properties of the metrics.

DeCAF-Boltz (arXiv 2606.08375, MIT) is distilled to be accurate *at* few steps.
Running the identical panel through it — and through stock Boltz-1 as a
de-confounding arm — answers the objection, though not in the direction first
expected.

## Design

The same 132 pairs from the n=22 PTM-clean panel (22 cognate, 44 scrambled,
66 decoy), same settings, three models:

| arm | base | trained for 10 steps | device |
| :--- | :--- | :---: | :--- |
| Boltz-2 | Boltz-2 | no | CPU |
| Boltz-1 | Boltz-1 | no | MPS |
| DeCAF-Boltz | Boltz-1 | **yes** | MPS |

**DeCAF and Boltz-1 share a base, a device and a step count**, so that pair
isolates few-step training. The Boltz-2 column ran on CPU and is therefore not
device-matched to the others.

## Order sensitivity — cognate minus its own scramble

| metric | Boltz-2 | Boltz-1 | **DeCAF** |
| :--- | ---: | ---: | ---: |
| ipTM | +0.013 (p = 0.42) | +0.039 (p = 0.0043) | **+0.201 (p = 2e-5)** |
| interface pLDDT | +3.30 (p < 1e-5) | +1.54 (p = 0.067) | **+9.54 (p < 1e-5)** |
| receptor side | +2.38 (p = 6e-5) | +0.71 (p = 0.43) | **+5.95 (p = 1e-5)** |

## Receptor specificity — cognate ranked against its own decoys

| metric | Boltz-2 | Boltz-1 | **DeCAF** |
| :--- | ---: | ---: | ---: |
| ipTM | 2.00 (p = 0.034) | 1.86 (p = 0.017) | **1.77 (p = 0.0087)** |
| interface pLDDT | 1.91 (p = 0.027) | 1.91 (p = 0.010) | **1.73 (p = 0.0042)** |
| receptor side | — | 2.05 (p = 0.054) | **1.77 (p = 0.0032)** |

Chance is 2.50. On DeCAF, interface pLDDT and receptor side clear a Bonferroni
threshold of 0.0083 for six metrics — **the first time receptor specificity has
been established anywhere in this work**, rather than merely suggested. Every
bootstrap CI excludes chance (interface pLDDT [1.32, 2.18]).

## What the de-confounding shows

**Few-step training accounts for the gain.** Against Boltz-1, its own base, on
the same device and step count, DeCAF delivers **5–6× larger effects**: +0.201
against +0.039 on ipTM, +9.54 against +1.54 on interface pLDDT. The base model is
not the explanation.

**But two findings complicate the tidy version.**

*Stock models are not signal-free.* Boltz-1 at 10 steps separates cognate from
scramble on ipTM (p = 0.0043) and ranks above chance (p = 0.017). The claim that
reduced sampling destroys the signal outright is too strong; it attenuates it.
Only Boltz-2's ipTM was genuinely flat.

*Section 7.6 is model-dependent.* Interface pLDDT separates cognate from scramble
strongly on Boltz-2 (+3.30, p < 1e-5) and on DeCAF (+9.54, p < 1e-5), but **not
on Boltz-1** (+1.54, p = 0.067, CI spanning zero; receptor side p = 0.43). The
recommendation to rank on interface pLDDT holds for the models where it was
measured — it does not generalise to every cofolding model.

## Revised position

The earlier framing — that the settings gap explains the negative results — is
too simple and is not adopted. What the three arms support is narrower:

1. **Few-step distillation substantially improves both order sensitivity and
   receptor specificity**, by a factor of 5–6 over its own teacher at the same
   budget.
2. **Stock models at reduced sampling retain weak but real signal**, rather than
   none.
3. **Which readout carries the signal varies by model.** Interface pLDDT is the
   best readout on Boltz-2 and DeCAF; on Boltz-1 the same quantity does not
   reach significance while ipTM does.

## Limitations

**Device confound on one arm.** Boltz-2 ran on CPU, Boltz-1 and DeCAF on MPS. The
MPS equivalence check established ipTM equivalence within the noise margin but
left interface pLDDT unresolved (TOST p = 0.168, underpowered at n = 6). Any
comparison involving the Boltz-2 column carries that caveat; the DeCAF↔Boltz-1
contrast does not.

**Single unseeded folds.** DeCAF's own run-to-run variance is unmeasured. Effects
are 3–5× the Boltz-2 noise floor, but that is a borrowed yardstick.

**n = 22 receptors**, and the bootstrap upper bounds reach 2.18–2.23, close to
chance.

**A silent-fallback hazard was guarded, not assumed.** The DeCAF fork reverts to
the teacher sampler if the head is unrecognised, which would produce plausible
numbers from the wrong model. Every batch asserts the string "Detected Decaf
checkpoint" in its log and refuses to score otherwise.

## Reproduce

```
python src/decaf_scramble_test.py --base decaf  --batch-size 12
python src/decaf_scramble_test.py --base boltz1 --batch-size 12
```
