# Worked examples for the screening tool

Paste-ready inputs for `python src/screen_server.py` (http://127.0.0.1:8765).

Every receptor and cognate peptide below comes from this project's own benchmark
panel (`artifacts/pdb_binders_b2_n22/pdb_binder_scores.json`), so these are the
sequences the reported numbers were measured on rather than sequences typed from
memory. **The numbers under each example are what the tool actually returned**,
not what it ought to return.

All four are pre-folded, so they come back in about two seconds instead of the
minutes the interface quotes. Re-running is free; the folds are cached.

A candidate's scrambles are derived from that candidate's own sequence, so you
get the same permutations whichever other peptides are in the box. The folds
themselves are unseeded, so scores move a little between fresh runs — that is
what the replicate SD column is for.

---

## 1. Does it find a known binder? — MDM2

The p53–MDM2 interaction. `SQETFSDLWKLLPEN` is p53's transactivation helix;
`TSFAEYWNLLSP` is PMI, a phage-derived peptide designed against the same pocket.
The other two are cognate ligands **of different targets**, serving as decoys.

**Target**

```
SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV
```

**Candidates**

```
SQETFSDLWKLLPEN
TSFAEYWNLLSP
PPPALPPKKR
KQTSV
PEPTIDEX
```

**Quick**, 2 replicates, 3 scrambles → *null spread 5.36 on df 8*

| # | candidate | score | own scrambles | t | p | |
|---|---|---|---|---|---|---|
| 1 | SQETFSDLWKLLPEN | 87.1 | 68.1 ± 2.7 | +3.88 | 0.002 | **cognate** |
| 2 | TSFAEYWNLLSP | 82.7 | 70.9 ± 7.6 | +2.41 | 0.021 | designed binder |
| 3 | KQTSV | 77.7 | 70.9 ± 6.6 | +1.39 | 0.101 | binds a different target |
| 4 | PPPALPPKKR | 71.5 | 71.5 ± 2.4 | −0.00 | 0.501 | binds a different target |
| — | PEPTIDEX | refused before folding — `X` is not one of the twenty | | | | |

Note `KQTSV` at p = 0.10: not called, but not far off either. That is what a
triage tool looks like when it is being honest.

---

## 2. Does it know *which* target? — PSD-95 PDZ3

**Paste the same three candidates against a different target.** `KQTSV` is a
C-terminal PDZ-binding motif. On MDM2 above it placed third at p = 0.10; here it
is the clear hit and the p53 helix is nothing. This is the example to show — a
tool that merely rewarded "peptide-shaped" sequences would rank the same
candidate top on both.

**Target**

```
GSPEFLGEEDIPREPRRIVIHRGSTGLGFNIIGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEANSRVNSSGRIVTN
```

**Candidates**

```
SQETFSDLWKLLPEN
PPPALPPKKR
KQTSV
```

**Quick**, 2 replicates, 3 scrambles → *null spread 4.18 on df 6*

| # | candidate | score | own scrambles | t | p | |
|---|---|---|---|---|---|---|
| 1 | KQTSV | 91.5 | 62.0 ± 3.9 | +7.72 | <0.001 | **cognate** |
| 2 | PPPALPPKKR | 50.7 | 47.8 ± 5.7 | +0.75 | 0.240 | |
| 3 | SQETFSDLWKLLPEN | 57.2 | 57.0 ± 2.1 | +0.06 | 0.477 | wins on MDM2 |

`KQTSV` is also the *shortest* candidate at five residues, so the ranking is not
being driven by length.

---

## 3. What full sampling buys — MDM2, careful

Same target and candidates as example 1, at the settings the model was meant to
run at: 200 sampling steps, 3 recycling passes, full alignment depth.

**Target**: as example 1.

**Candidates**

```
SQETFSDLWKLLPEN
TSFAEYWNLLSP
PPPALPPKKR
KQTSV
```

**Careful**, 2 replicates, **5 scrambles** → *null spread 2.65 on df 16*

| # | candidate | score | own scrambles | t | p | |
|---|---|---|---|---|---|---|
| 1 | TSFAEYWNLLSP | 92.9 | 74.7 ± 3.6 | +8.24 | <0.001 | designed binder |
| 2 | SQETFSDLWKLLPEN | 87.5 | 70.4 ± 3.4 | +7.72 | <0.001 | **cognate** |
| 3 | PPPALPPKKR | 72.5 | 73.7 ± 1.6 | −0.52 | 0.694 | |
| 4 | KQTSV | 79.1 | 80.3 ± 1.2 | −0.57 | 0.711 | |

Compare with example 1. The scramble null tightens from 5.36 to 2.65, both real
binders move to p < 0.001, and the two foreign ligands go firmly negative —
`KQTSV`'s borderline p = 0.10 becomes p = 0.71. PMI now correctly outranks the
natural p53 helix, which it should: it was designed to bind this pocket harder.

Five scrambles rather than three because the null needs degrees of freedom before
it can call anything. At df = 2 a candidate needs t > 2.92, and a genuine binder
with a large margin still reads as nothing; the tool warns when a job is too
small rather than letting that pass for evidence of inactivity.

---

## 4. Where the control saturates — c-Crk SH3

Worth showing if you want to be straight about the method's limits.

**Target**

```
AEYVRALFDFNGNDEEDLPFKKGDILRIRDKPEEQWWNAEDSEGKRGMIPVPYVEKY
```

**Candidates**: the same three as example 2. **Quick**, 2 replicates, 3 scrambles.

The cognate `PPPALPPKKR` comes back **indistinguishable from its own scrambles**
(t = +0.27, p = 0.40), placed second behind a peptide that binds a different
target. Running it at careful settings does not rescue it — 21 folds, 26 minutes,
and nothing separates:

| # | candidate | score | own scrambles | t | p |
|---|---|---|---|---|---|
| 1 | KQTSV | 93.4 | 86.6 ± 3.1 | +1.71 | 0.056 |
| 2 | PPPALPPKKR | 95.9 | 89.2 ± 5.2 | +1.69 | 0.058 |
| 3 | SQETFSDLWKLLPEN | 76.1 | 75.6 ± 5.6 | +0.12 | 0.454 |

The one thing it gets right at either setting is putting the MDM2 helix last.

A partial explanation: `PPPALPPKKR` is 50% proline, and its permutations —
`APPRLPPKKP`, `APRKPPKPLP` — are still proline-rich PxxP-like sequences. An SH3
domain binds polyproline-II helices substantially through composition, so a
scramble of a proline-rich ligand is a plausible ligand too, and a control that
only destroys order has little left to detect.

That explanation is incomplete, and worth saying so: `KQTSV` has no prolines at
all and behaves the same way, scoring 93.4 against scrambles at 86.6. This target
returns high, flat interface confidence for almost anything put in front of it.

What the tool reports here is *no order-dependence detected*, which is a limit of
the control rather than evidence that nothing binds. Reporting that as
"indistinguishable" instead of "not a binder" is the distinction that matters.

---

## Controls worth pasting

Add these to any screen above.

| Candidate | What it is | What should happen |
|---|---|---|
| `PEPTIDEX` | contains a non-standard residue | refused before folding, not scored badly |
| `GSGSGSGSGSGS` | a flexible Gly–Ser linker | indistinguishable (measured p ≈ 0.14) |
| `AC` | two residues | refused: shorter than 4 |

`GSGSGSGSGSGS` is the one to show if you want to make the statistics honest in
front of an audience. An earlier build called it a hit at 1.06, because it
compared the margin to a pooled spread with a fixed cutoff as though that ratio
were a z. It is a t on the pooled degrees of freedom, and on that scale the
linker is p ≈ 0.14 — the right answer.

---

## Reading the table

| Column | Meaning |
|---|---|
| **score** | mean of the replicates for the chosen readout |
| **own scrambles** | mean ± SD over permutations of *that same candidate* |
| **t** | margin over its own scrambles, in units of the spread pooled across all candidates |
| **p** | one-sided, on the pooled spread's degrees of freedom |
| **replicate SD** | spread between repeat folds of the identical input — the floor on what any difference can mean |

*Beats its own scrambles* is p < 0.01; *suggestive* is p < 0.05.

The ranking is a triage order, not a measurement of affinity. On the 22-target
benchmark at careful settings the true binder ranked first for 17 targets — but
on targets released after the model's training cutoff both readouts lose about
half their effect, so expect roughly half that apparent accuracy on a genuinely
novel target.
