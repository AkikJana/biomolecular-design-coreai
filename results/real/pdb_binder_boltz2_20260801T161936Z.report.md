# Boltz-2 on the PDB binder benchmark

**Question.** The PDB binder result — ipTM has sensitivity but not specificity —
was measured under Boltz-1 and carried "Boltz-1 rather than Boltz-2" as an
untested caveat. Does a stronger checkpoint recover specificity?

**Design.** Identical 66 pairs (11 cognate, 33 decoy, 22 scrambled), identical
seed, and byte-identical per-receptor MSAs copied from the Boltz-1 run's cache.
The checkpoint is the only variable. 10 sampling steps, 1 recycling, MSA depth
32, CPU. ~3 min per complex, 0 failures.

## Result

Every class shifts up by roughly 2.8x, together:

| | cognate (11) | decoy (33) | scrambled (22) |
| :--- | :---: | :---: | :---: |
| Boltz-1 | 0.1915 | 0.1684 | 0.1758 |
| Boltz-2 | 0.5355 | 0.4556 | 0.4662 |

Within-receptor cognate rank — the screening test — improves in every direction.
Significance does not follow:

| model | competitors | mean rank | chance | 1-sided | 2-sided |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Boltz-1 | decoys only | 2.36 | 2.50 | 0.395 | 0.790 |
| Boltz-1 | decoy + scrambled | 3.27 | 3.50 | 0.373 | 0.746 |
| Boltz-2 | decoys only | 1.91 | 2.50 | 0.062 | 0.124 |
| Boltz-2 | decoy + scrambled | 2.55 | 3.50 | **0.047** | 0.094 |

One of four cells clears 0.05, and it is the least conservative combination
(one-sided, with scrambles counted as competitors). Two-sided is the convention
already used for the Boltz-1 number in the README, and under it Boltz-2 does not
reach significance.

Bootstrap agrees. Boltz-2 mean cognate rank 1.91, 95% CI **[1.36, 2.55]** —
contains chance (2.50). Paired improvement over Boltz-1 +0.45 ranks, 95% CI
**[−0.36, +1.18]** — contains zero.

## The pooled test passes and should not be believed

`pdb_binder_benchmark.py` prints cognate > decoy at AUC 0.689, p = 0.033.

Two reasons not to cite it:

1. **Confounded by receptor identity.** Receptors differ several-fold in
   baseline ipTM, so a pooled test partly measures which receptor a pair
   belongs to rather than whether the peptide is the right one.
2. **Internally incoherent.** Under Boltz-2, decoys and scrambles are
   indistinguishable — AUC 0.501, p = 0.993. A real peptide that binds some
   other receptor scores exactly like sequence-order garbage. If ipTM cannot
   separate a genuine binder from a scramble, its apparent edge over decoys is
   not binder recognition. The p = 0.033 traces to the decoy set being larger
   (33 vs 22), not to a distinction that exists.

The same decoy≈scrambled pattern holds under Boltz-1 (AUC 0.444, p = 0.487), so
it is a property of the signal, not of the newer checkpoint.

## Conclusion

Boltz-2 moves the specificity result in the right direction and by a meaningful
amount, but does not establish specificity at n = 11, and the model-to-model
difference is not distinguishable from noise. The Boltz-1 conclusion stands —
now with the model caveat tested rather than assumed.

## Next step is a power problem with a known answer

For 80% power:

* **21 receptors** to establish Boltz-2 specificity against chance (dz = 0.57)
* **74 receptors** to establish Boltz-2 > Boltz-1 (dz = 0.33)

The first is reachable: roughly double the current panel, ~6 h of CPU folding.
That is the experiment that would settle this, and it is the honest move rather
than reporting p = 0.047 from the one cell that cleared.

## Reproduce

```
python src/pdb_binder_benchmark.py --model boltz2 --work-dir artifacts/pdb_binders_b2
python src/compare_boltz1_boltz2.py
```
