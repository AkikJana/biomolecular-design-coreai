# Where the interface-pLDDT signal comes from

Two follow-ups to Section 7.6, which found interface pLDDT separates a cognate
peptide from its own scramble where ipTM cannot. Both attack the same question
from different sides: **is that signal about binding, or about something
cheaper?**

## 1. Which side of the interface carries it — no new folding

Interface pLDDT averages over residues on both chains, so the readings separate.
The objection to Section 7.6 is that a scrambled peptide is often more
disordered, which would produce the result with no binding information in it.

| quantity | cognate | scrambled | decoy | cognate − own scramble | p |
| :--- | ---: | ---: | ---: | ---: | ---: |
| pooled interface pLDDT | 49.60 | 46.31 | 45.93 | +3.30 | 0.00000 |
| **receptor side** | **51.94** | **49.56** | **49.41** | **+2.38** | **0.00006** |
| peptide side | 44.37 | 39.12 | 39.16 | +5.25 | 0.00001 |
| peptide whole chain | 44.08 | 39.21 | 39.13 | +4.87 | 0.00003 |

**The objection is half right.** Whole-chain peptide pLDDT (+4.87) is nearly as
large as peptide-side interface pLDDT (+5.25), so most of the peptide-side signal
is not about the interface at all — it is the peptide being placed more
confidently overall. Anyone raising the foldability objection was correct about
that component.

**The receptor side survives it.** The receptor's *own* residues are placed more
confidently when the cognate peptide is present (+2.38, p = 6e-5). A disordered
peptide cannot make the receptor's residues more certain, so peptide foldability
does not account for this term.

Caveat: receptor side alone ranks cognate against decoys at p = 0.086 — it
separates a peptide from its own scramble but does not establish receptor
specificity, matching every other result in this project.

## 2. Is the receptor-side response tied to the binding site?

Four arms, same cognate peptide, all single-sequence so MSA presence is not a
confound. The binding site is the 15 receptor residues nearest the peptide
(mean 13.4); an 8 A cutoff marked ~half the domain as interface, and mutating
that much destroys the fold — reintroducing the confound the control exists to
remove.

| arm | receptor side | vs real | p |
| :--- | ---: | ---: | ---: |
| real | 45.70 | — | — |
| interface → Ala | 36.77 | −8.92 | 0.00001 |
| surface → Ala (control) | 38.62 | −7.08 | 0.00048 |
| scrambled receptor | 34.03 | −11.67 | 0.00000 |

**The decisive comparison is interface-Ala against surface-Ala**, since both
mutate the same number of residues:

```
difference  -1.84   95% CI [-5.05, +1.36]   p = 0.244
paired dz    0.255  ->  80% power needs n = 123 receptors
```

Mutating the binding site costs more than mutating an equal number of exposed
residues elsewhere — but not detectably so. **This is underpowered, not a clean
negative:** the direction is right and the effect would need roughly six times
the current panel to resolve.

Two further caveats on interpretation. Single-sequence folding puts the baseline
at 45.70 against 51.94 with an MSA, leaving less headroom before the floor. And
every perturbation costs 7–12 pLDDT, so both arms may be compressed near that
floor, which would mask a real difference.

The scrambled-receptor arm falls furthest, as expected, but is uninformative on
its own: shuffling destroys the fold as well as the site.

## What the pair establishes

**Strengthened:** the receptor responds to *which* peptide it is given, and that
cannot be explained by peptide foldability. This is the best evidence the project
has that interface pLDDT reads something about the interaction rather than about
the peptide alone.

**Not established:** that the response is localised to the binding site. The
alanine-scanning control points the right way and is far short of significance.

Neither result changes the practical recommendation — rank on interface pLDDT
rather than ipTM — but the mechanism behind it remains only partly characterised,
and the peptide-side component is now known to be largely foldability.

## Reproduce

```
python src/interface_side_split.py
python src/receptor_site_controls.py --device mps
```
