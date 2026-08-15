# Four of the twenty-two cognate pairs were not binders

A cognate pair is supposed to be a receptor and a peptide that bind. The
benchmark folds two sequences, so that holds only if the crystallised peptide
*is* its canonical sequence. For four members of the panel it is not, and the
failure is silent in every direction: the FASTA parses, the fold succeeds, the
confidence scores look ordinary.

## What was wrong

| PDB | peptide as folded | what was actually crystallised |
| :--- | :--- | :--- |
| 1NLO | `XXXXPLPPLPX` | ACE, MN1, MN2, MN7, NH2 — a designed synthetic ligand |
| 2GBQ | `XVPPPVPPRRRX` | acetyl and amide terminal caps |
| 1SEM | `XPPPVPPRRR` | acetyl cap |
| 9GRF | `AASTTTPAPA` | O-glycosylated at Ser3 and Thr4 |

**1NLO is the serious one.** Five of eleven positions are not amino acids —
4-carboxypiperidine, two benzene derivatives, and terminal caps. RCSB FASTA
emits `X` for each. What SH3 binds in that crystal is a small molecule; what the
benchmark folded was the string `XXXXPLPPLPX`.

**9GRF is the subtle one.** StcE is a mucin-selective protease: it recognises
the O-glycan, not the bare peptide backbone. Nine `covale` records attach GalNAc
to the peptide chain. Stripped of its sugars, `AASTTTPAPA` is not a substrate.

This is the defect that removed 1I8H, whose peptide needs phosphothreonine, and
whose "true binder ranked last of six" in the Boltz-1 write-up was correct
behaviour on a non-binder.

## Why the existing PTM filter missed them

`audit_panel_ptms.py` tests each peptide against a fixed list of PTM codes. That
works for the modifications on the list and cannot work for anything else. The
held-out panel turned up **7F3S**, a histone H3 tail carrying `LBZ` —
benzoyl-lysine — read by a bromodomain-containing receptor. It is precisely the
1I8H failure, and the allowlist passed it.

Asking a different question fixes this. Instead of *"is this one of the
modifications I know about?"*, ask *"is anything bonded to this chain?"* — one
`covale` scan over the mmCIF, no list to keep current. It recovers all four main
panel members, plus 7F3S and 7JZQ in the held-out panel.

The same scan clears three false alarms that a bound-ligand-name test flags:
1ELW's nickel, 6YOO's zinc and 7S7J's calcium are not bonded to the peptide.
A metal at a lattice contact does not change whether the peptide binds; a sugar
on its own serine does.

## Does it change the results?

**It should, in a direction fixed in advance.** A cognate that cannot bind
scores like a scramble, so it shrinks the cognate-versus-scramble contrast it
belongs to. Removing these members must therefore *raise* every effect. That is
a prediction, and the exclusions were chosen by whether a sequence can be folded
faithfully — never by looking at a p-value.

| arm | metric | all 22 | minus 1NLO, 9GRF | minus those + capped |
| :--- | :--- | ---: | ---: | ---: |
| DeCAF | ipTM | +0.201 | **+0.225** | +0.202 |
| DeCAF | interface pLDDT | +9.54 | **+10.57** | +10.06 |
| DeCAF | receptor side | +5.95 | **+6.65** | +6.51 |
| Boltz-2 | ipTM | +0.013 | +0.015 | +0.016 |
| Boltz-2 | interface pLDDT | +3.30 | **+3.47** | +3.07 |
| Boltz-2 | receptor side | +2.38 | **+2.48** | +2.19 |
| Boltz-1 | ipTM | +0.039 | +0.044 | +0.049 |
| Boltz-1 | interface pLDDT | +1.54 | **+2.18** | +2.95 |
| Boltz-1 | receptor side | +0.71 | **+1.41** | +2.32 |

**It grew in 9 of 9 arm-metric cells.** Nothing forced that; the flags come from
structural annotation and the effects come from folds run weeks earlier.

### The rank tests move the other way

Quoting only the table above would be selective. The receptor-specificity rank
tests get *worse* on exclusion:

| arm | metric | all 22 | minus unfoldable | minus + capped |
| :--- | :--- | ---: | ---: | ---: |
| DeCAF | interface pLDDT | p = 0.0042 | 0.0084 | 0.0235 |
| DeCAF | ipTM | p = 0.0087 | 0.0173 | 0.0438 |
| Boltz-1 | ipTM | p = 0.0168 | 0.0329 | 0.0828 |

The mean ranks barely move — DeCAF interface pLDDT goes 1.73, 1.75, 1.83 against
a chance value of 2.50. What changes is that the Wilcoxon loses four of
twenty-two samples. This is a power loss, not a weaker effect, and it is the
same limitation Section 7.9 documents: a rank test collapses six folds into one
integer and cannot afford to lose receptors.

So the two families of test answer differently because they are limited by
different things. Removing diluting non-binders helps a paired effect and costs
a rank test. Neither is the panel telling us the signal is fragile.

## The correction this forces

Section 7.7 reports that interface pLDDT is model-dependent, because on Boltz-1
the cognate-versus-scramble effect sits at p = 0.067 — suggestive, not
significant. That was the basis for treating the readout as unreliable across
models.

With the two unfoldable members removed it is **p = 0.013**, and removing the
capped peptides as well takes it to **p = 0.001**:

```
Boltz-1, interface pLDDT
  all 22 receptors            +1.54   p = 0.067     "model-dependent"
  minus 1NLO, 9GRF            +2.18   p = 0.013
  minus those + 1SEM, 2GBQ    +2.95   p = 0.001

Boltz-1, receptor side        (the other half of the same claim)
  all 22 receptors            +0.71   p = 0.428
  minus 1NLO, 9GRF            +1.41   p = 0.125
  minus those + 1SEM, 2GBQ    +2.32   p = 0.012
```

Both halves of the model-dependence claim move together, and the receptor-side
term moves furthest — from nowhere near significance to p = 0.012. Boltz-1 does
show the effect. The claim was substantially an artefact of two non-binders in
the panel, and the correction block appended to Section 7.6, the Section 7.8
tables, and the Section 8 recommendation all need revising rather than
annotating.

The claims that do **not** move: Boltz-2's ipTM stays flat (p = 0.38 to 0.42) on
every variant, so Section 7.4's finding that ipTM tracks composition rather than
binding is unaffected. DeCAF remains the strongest arm throughout.

## Caveat

The panel is 22 receptors and the exclusions cost two to four of them. The
Boltz-1 interface-pLDDT p-value moves across the conventional threshold on a
sample this small, so it is better read as "the effect is present and was being
diluted" than as a precise significance level. The held-out panel now folding
is the independent test.

## Reproduce

```
python src/panel_integrity.py
```
