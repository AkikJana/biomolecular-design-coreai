# CPU reproduction, 28 August 2026 — the discrepancy was our own imprecision

96 folds. The original 4 receptors, 24 complexes, 4 identical unseeded re-runs,
on Apple Silicon CPU against the development build in ./boltz — the same device
and the same code as Section 7.5, run to settle whether its 1.917 reproduced.

## The question

The widened study on CUDA gave interface pLDDT a run-to-run SD 29% larger than
1.917 while ipTM moved 1.3%. Those runs differed in two ways at once, device and
model build, so the preprint recorded the gap as real but unattributed. Two
outcomes were expected: the original reproduces (gap is environmental), or it
does not (1.917 was a low draw).

## Neither

  1.917   CPU, development build          4 receptors    (original)
  2.192   CPU, development build          4 receptors    (this run)
  2.479   CUDA, released build            same 4
  2.376   CUDA, released build            22 receptors

This run's bootstrap 95% interval is [1.877, 2.514]. It contains every other
estimate, including both values whose difference was being explained.

A pooled SD from 24 complexes folded four times carries about +/-8.3% relative
standard error. The whole 1.917-2.479 spread is a little over two of those.

So there is no environment effect on the noise. There is a noise floor measured
too imprecisely to compare against itself.

The four ipTM estimates -- 0.0628, 0.0616, 0.0636, 0.0646 -- span 4.8% and say
the same thing from the stable side.

## What this cost, and what it is worth

Section 2.4 measured a run-to-run spread and then compared two such spreads,
without ever putting an interval on the spread itself. Its own conclusion -- that
a single draw does not pin a quantity -- applies to the quantity it reports. The
manuscript carried an unexplained 29% discrepancy for two commits because of it.

The effect-to-noise ratios inherit the +/-8.3%, so 1.72x and 8.6x are now written
as 1.7x and 9x. That is the precision the divisor supports.

## Also settled here

pDockQ, which an earlier note called unrecoverable, recomputes at 0.157 against
the published 0.150. It is a function of interface pLDDT and the contact-pair
count, both available from rescore_interface_metrics.interface at the same 8.0 A
cutoff.

## Rate

77 s/fold steady state against the original's 93, with batch 0 at 258 s/fold
because it carries the weight load. The per-batch pattern tracks the original
closely, which is one more reason to think the host had not changed underneath.
