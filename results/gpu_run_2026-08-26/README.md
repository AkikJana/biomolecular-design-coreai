# GPU run, 26 August 2026 — the two targets that were never too large

540 folds, 180 designs, EGFR and Nipah-G, converged settings. Merged with the
720-design run this takes Section 2.7 to **900 designs across 10 of 15 targets**.

## An exclusion that was asserted, not measured

The 8-target run excluded three targets on construct size. Only one of them had
ever been tested.

  target     complex   claimed        reality
  Cas9          1463   too large      OOM observed on a 4090 -- correct
  EGFR           716   too large      folds in 99 s
  Nipah-G        511   too large      folds in 71 s

EGFR and Nipah-G were excluded by a `>500 residues` rule written in the moment
after Cas9 failed, then described in the paper as a size exclusion -- which
implied a measurement that did not exist. Five minutes of smoke-testing avoided
renting a 96 GB card at five times the price, and the section now says only Cas9
is excluded, on evidence.

## The merged boundary test

                  n     design   scrambles    margin
  binders       263      83.29       60.01    +23.28
  non-binders   637      79.39       61.54    +17.85

  raw interface pLDDT             0.592
  design minus its own scrambles  0.583
  margin - raw                   -0.010   95% CI [-0.035, +0.014]

  correlation(raw, margin) = +0.822

The interval has narrowed from [-0.330, +0.071] at 48 designs to [-0.035, +0.014]
at 900 -- an order of magnitude, and now tight enough to exclude any effect worth
acting on in either direction.

The margins still differ between classes at p = 2e-14, so the subtraction is not
a constant. It is redundant: it correlates with the raw score at 0.82, and a model
given both scores 0.589 against 0.592 for raw alone.

## Recovered mid-run

The pod ran out of compute at 468 of 540 folds. /workspace persisted, the store
had every completed design, and the run resumed on a fresh pod with only the venv
needing a rebuild. Nothing was refolded.
