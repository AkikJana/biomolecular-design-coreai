---
title: Scramble-control scorer
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
---

# Scramble-control scorer

A cofolding confidence score that ranks a peptide above an unrelated decoy may be
responding to amino-acid **composition** rather than to binding. Scoring a
candidate against permutations of *itself* holds composition and length fixed and
varies only residue order.

On a 22-receptor panel, ipTM ranked cognate peptides above decoys (mean rank 2.00
against a chance value of 2.50) while failing to separate them from their own
permutations — and permutations outscored decoys. Of six interface readouts on
identical structures, only interface pLDDT passed that test.

## What this does

Two things, both in your browser. No structure is uploaded anywhere.

**Generates the permutations to fold**, seeded from the peptide so the same input
always produces the same null.

**Scores a folded complex** and reports the margin against its own permutations.
The readout is the paper's: 8 Å between representative atoms, CB where present and
CA otherwise, mean CA pLDDT over the contacting residues of both sides. The
JavaScript was checked against the Python implementation on a controlled complex
and returns identical values.

It does **not** fold — that needs a GPU. Bring predicted structures with pLDDT in
the B-factor column; Boltz, Chai-1 and AlphaFold all write it there.

## Two things it tells you that are worth believing

**A margin is not a p-value.** With two or three permutations the null's spread is
barely estimated. The paper pools that null across candidates and treats the ratio
as a *t* on pooled degrees of freedom.

**The control has a measured boundary.** On 60–120 residue designed proteins a
permutation does not fold at all, so binders and non-binders lose the same ~19
points of interface pLDDT and the subtraction carries no information (ΔAUC −0.104,
95% CI [−0.330, +0.071], on 48 designs whose binding two contract research
organisations measured). Use it where a permutation of the candidate remains a
plausible candidate — which for short peptides it is.

## Data

`AkikJana/scramble-control-panels` — 2,456 folds across 16 inference arms and 75
receptors, including a second model family (Chai-1) scored with this same readout.
