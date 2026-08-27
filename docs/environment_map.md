# Which environment produced which result

Two environments produced the folds in this project, and the split is not
cosmetic — see preprint §4.1 and §2.4.

| | model build | device |
| :-- | :-- | :-- |
| **local** | the fork in `./boltz` (2.2.1, unpublished) | Apple Silicon, CPU |
| **released** | `boltz` 2.0.3 from PyPI | rented RTX 4090, CUDA |

They differ in 46 of 106 shared source files. `remote/provision.sh` used to run
`pip install boltz`, which silently chose the second; it now installs the fork
and asserts which one loaded.

Everything under `results/gpu_run_*/` is the released build. Everything dated
before 2026-08-23 is local.

## By section

| section | environment | note |
| :-- | :-- | :-- |
| 2.1 external binding | local | single environment |
| 2.2 permutation control | local | single environment |
| 2.3 six readouts | **mixed** | Table 4's two columns; the contrast is 0% vs 100% |
| 2.4 replicate noise | **mixed** | the 29% discrepancy; recorded as unattributed |
| 2.5 held-out split | local | both arms 2026-08-16 |
| 2.6 settings factorial | **mixed** | singles local, pairs released; bounded in the text |
| 2.7 boundary test | **mixed** | 900 designs all released; agrees with the local 48 |
| 2.8 panel 22 → 59 | released | the 22 panel was refolded there for this |
| 2.9 second model family | released | single environment |
| 2.10 recommendations | released | single environment |

## The size of the difference

One arm — the 22-receptor panel at 200/3/full — was folded in both, which bounds
it directly (preprint Table 19):

    ipTM              +0.2873  ->  +0.2868    1.00x
    interface pLDDT   +11.851  ->  +11.894    1.00x
    receptor side      +5.129  ->   +5.273    1.03x
    d(ipTM)               1.25 ->      1.16   0.93x
    d(interface pLDDT)    1.52 ->      1.43   0.94x

Raw effects agree to 3%. Standardised effects run 6-7% lower, which is the same
order as the run-to-run instability of *d* in §2.4 and not separable from it on a
single refold.

So the centres of these measurements are portable and their spreads are not: the
interface-pLDDT *noise* discrepancy in §2.4 is 29%, an order of magnitude larger
than anything visible in the effects themselves.

## Two things this pass found

**Table 14 was already clean.** The 22-receptor panel was deliberately refolded
on the rented device so that the 22-vs-59 comparison would sit in one
environment. An earlier draft of this document claimed otherwise and "corrected"
the table against the 59-receptor run's own 22-receptor subset; that was wrong
and was reverted.

**But refolding is not free.** That subset gives a receptor-side *d* of 0.89
where the dedicated refold gives 1.67 — same environment, same settings, same
receptors, different run. Raw margins agree within 5%. This is §2.4 applying to
§2.8: a single run does not pin a per-receptor *d*, and the 22 → 59 decline
should be read for its direction rather than its magnitude on that readout.

## Regenerating this

    python src/verify_claims.py

The classification itself is a date-and-directory check: anything whose file
appears under `results/gpu_run_*/`, or postdates 2026-08-23, is the released
build.
