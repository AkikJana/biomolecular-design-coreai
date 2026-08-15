# Requiring the match in both directions raises precision from 56% to 88%

Every readout in this work is something the model reports about its own output,
so all of them sit under the variance ceiling of Section 7.9.2 — which is why
combining readouts, combining models, and physics rescoring each buy nothing.
Escaping it needs information the model did not produce. Three sources were
available. This is the one that worked.

## The idea

The panel has been read one way throughout: *for this receptor, is the cognate
the best of its candidate peptides?* It also contains the transpose, because
each peptide is folded against its own receptor and against the several that
borrowed it as a decoy. That direction is a different and equally reasonable
question, and on its own it works at least as well:

| direction | mean rank | chance | p |
| :--- | ---: | ---: | ---: |
| receptor-centric (Sections 7.4–7.10) | 1.73 | 2.50 | 0.0042 |
| peptide-centric | 1.58 | 2.66 | 0.0014 |

Requiring **both** is the binding analogue of reciprocal best hits in homology
search.

## It replicates

Twelve combinations: two panels × two readouts × up to five draws.

| | one-directional | reciprocal |
| :--- | ---: | ---: |
| precision | 56% (42–72%) | **88% (75–100%)** |
| enrichment over base rate | 2.3× | **3.5×** |
| calls retained | — | 49% |

Pooled across all twelve, the filter discards **87% of wrong calls (84 of 97)**
and **23% of right ones (29 of 125)**. That asymmetry is the result. A criterion
that merely shrank the candidate set would discard both classes in proportion,
and the permutation test asks exactly that: against discarding the same number
of calls at random, ten of twelve cells clear p = 0.05.

**Precision improved in 12 of 12 combinations** — the only intervention in this
project that improved everywhere it was tested. Draws within a panel share
receptors and are not independent, so the defensible unit is panel × readout,
4 of 4, rather than the 12-way sign test (p = 5e-4).

Per-panel detail, interface pLDDT and ipTM:

```
in-training     iface pLDDT  12/20  60%  ->  11/13  85%   perm p 0.0044
in-training     ipTM         11/19  58%  ->   9/11  82%   perm p 0.0209
held-out (5 draws, mean)
                iface pLDDT         50%  ->         85%   perm p 0.040
                ipTM                62%  ->         93%   perm p 0.024
```

## Why this one survived

It is not a better score. It is the same scores read as a two-way competition,
so it draws its power from the structure of the screen rather than from the
model — which is precisely why the Section 7.9.2 ceiling does not constrain it.
It also costs nothing at inference on folds already computed.

## Deployed form, and what it costs

Score each candidate against the target *and* against a small panel of
off-targets; keep only candidates whose best target is yours. Five to ten times
the folds, bounded, buying 56% → 88% precision. When each call is a wet-lab
experiment, thirteen calls at 85% beats twenty at 60%.

## Limitations

Call counts are small — seven to thirteen per combination — so a reported 100%
is eight of eight. The competition is incidental rather than designed: each
peptide meets only the three to six receptors that happened to borrow it, and a
deployed off-target panel would be chosen deliberately. And the filter improves
both regimes without undoing contamination — held-out precision before filtering
is 50% against 60% in training, consistent with Section 7.10's factor of two.

## The two that did not work

**Site correctness against the crystal.** The crystal says which receptor
residues contact the peptide, so one can ask what fraction of a prediction's
contacts fall in that known site — information from entirely outside the model.
On 120 folds across 20 receptors: cognate precision 0.452, decoy 0.445,
scrambled 0.449. Rank 2.50 against chance 2.50, p = 0.906.

The measurement works — random contacts would score 0.148, so the model finds
the true groove three times better than chance. It simply does so for every
peptide equally. Site-finding is receptor-driven.

**Pose convergence across draws.** A real binder should have one favourable pose
and converge; a non-binder should scatter. Cognate peptide RMSD between draws
8.35 Å, decoy 9.67 Å, scrambled 9.99 Å. Direction right, not significant
(+1.64 Å, p = 0.129). Receptor RMSD also varies by class and correlates at
r = +0.36, and regressing it out halves the effect (+1.08 Å, p = 0.350).

**Together these explain the rest of Section 7.** Every peptide lands in the
same correct pocket, then sits 8–10 Å differently within it on each draw. The
pocket is determined by the receptor; the pose is not determined at all. Every
contact-derived readout in this work — contacts, contact density, buried area,
pDockQ, PRODIGY's ΔG, pose agreement — was reading a placement that is correct
in location and arbitrary in detail. Confidence is the only axis that carries
binding information, which is why everything reduces to one ceiling.

## Reproduce

```
python src/reciprocal_match.py
python src/site_correctness.py --tag p1
python src/pose_convergence.py --tags p1,p2
```
