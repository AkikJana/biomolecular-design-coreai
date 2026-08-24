# GPU run, 25 August 2026

Two results, both aimed at limitations the preprint named itself. About 400 folds.

| file | what |
| :--- | :--- |
| `heldout_panel_resultn44.json` | held-out panel doubled: 44 receptors, 264 folds |
| `chai_heldout.json` | Chai-1 on the 22-receptor held-out panel, 131 folds |
| `heldout_expansion.json` | the 22 post-cutoff receptors the screen accepted |
| `heldout_sequences.csv`, `heldout_folds.csv` | the Chai panel, rebuilt from the Boltz fold list |

## The contamination penalty is smaller than 22 receptors suggested

                       d @22   d @44   retained @22   retained @44
  ipTM                  0.59    0.76            51%            65%
  interface pLDDT       0.64    0.75            44%            53%
  receptor side         0.44    0.65            27%            39%

All three remain significant (p 2.0e-06 to 3.0e-05). Section 2.5 reported
37.6-40.4%; at 44 receptors it is 39-65%. The original figures were the
pessimistic end.

This cuts the same way as the 59-receptor panel, where a larger panel made the
scramble control *smaller*. The 22-receptor panel exaggerated whichever direction
a result pointed in.

Caveat: the 22 added receptors are more recent and skew to shorter peptides, so
part of the gain may be composition rather than sample size.

## Boltz-1's in-training advantage does not survive the split

                        Boltz-1   Chai-1
  in-training d            1.43     0.90
  held-out d               0.64     0.76
  held-out p            3.3e-03  1.2e-03
  retention                 44%      84%

In training Boltz leads by a wide margin. On post-cutoff complexes the ordering
reverses and the two are comparable. Some of what looks like a better confidence
head is training exposure.

**The retention difference is not established and the paper does not claim it.**
44% against 84% divides two effect sizes each from ~20 receptors. Bootstrapped
(20k), the intervals are [17%, 79%] and [31%, 224%]; the difference is +48 points
with 95% CI [-25, +183], spanning zero at P = 0.86. The supported claim is the
held-out comparison, which requires no division.
