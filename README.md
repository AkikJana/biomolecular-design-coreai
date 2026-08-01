# Boltz-Fast: Edge-Accelerated Differentiable Biomolecular Design

**Boltz-Fast** is an edge-accelerated, differentiable surrogate modeling framework for real-time biomolecular structure prediction and binder design on Apple Silicon. By combining **FP8 weight-only quantization** and **Stateful Dynamic Key-Value (KV) Caching**, Boltz-Fast brings structural biology calculations directly onto consumer devices (MacBook Air/Pro) without requiring cloud servers or dedicated Nvidia H100 clusters.

## 🚀 Key Features

* **Dynamic Shapes Support:** Supports variable-length binder sequences (5–100 aa) and target receptor sequences (50–2000 aa) on-the-fly without requiring model recompilation or wasting cycles on zero-padding.
* **Stateful Receptor KV-Caching:** Caches projected key-value representations of the constant target receptor inside Apple Neural Engine (ANE) registers. Screening new mutant binder sequences drops attention complexity from quadratic $O(N^2)$ to linear $O(L_{\text{binder}} \cdot L_{\text{target}})$, reducing evaluation times to milliseconds.
* **Quantized (FP8) Surrogate Architecture:** Uses a single-pass feed-forward network compressed to microscaled `float8_e4m3fn` formats rather than slow, iterative 3D coordinate diffusion processes.
* **Apple Neural Engine (ANE) Native:** Compiled Ahead-of-Time (AOT) to Metal Performance Shaders (`.aimodelc`). Execute predictions directly in unified memory with **zero python runtime, zero interpreter overhead, and zero CPU-GPU copies**.
* **100% Differentiable:** Enables analytical gradients to propagate from output 3D coordinates back to sequence embeddings, allowing automated binder design using gradient descent.

---

## 📊 Performance Benchmarks (1300-Residue Target)

**What is being measured.** Boltz-Fast evaluates a *surrogate*: a single-pass
network (conv → cross-attention → coordinate head) that approximates a binder's
structure and affinity. It is not a reimplementation of Boltz-1's pipeline,
which additionally performs MSA construction, Pairformer trunk recycling,
iterative diffusion sampling, and confidence estimation. Latency numbers below
therefore describe **candidate screening throughput**, not end-to-end structure
prediction, and the two are not interchangeable.

Measured on an Apple M-series MacBook: 200 binder mutants scored against a
constant 1300-residue receptor, with the receptor's projected K/V held in the
ANE state cache so only the binder is re-evaluated per candidate.

| Backend (same surrogate, 1300-residue target) | Avg latency / candidate | 200 candidates |
| :--- | :---: | :---: |
| **CoreAI, Apple Neural Engine** (FP8 + KV cache) | **7.95 ms** | **1.59 s** |
| PyTorch MPS (same model, no KV cache) | run the benchmark | run the benchmark |
| PyTorch CPU (same model, no KV cache) | run the benchmark | run the benchmark |

Reproduce with `python src/benchmark_boltz_coreai.py`, which measures all three
backends on the same architecture. That CPU/MPS-vs-ANE comparison is the
like-for-like result this project can stand behind.

### How this relates to Boltz-1

Full Boltz-1 inference on a target of this size is frequently quoted at
**minutes** per structure on consumer hardware. Earlier revisions of this README
divided such a figure by the 7.95 ms above and reported a five-digit speedup.
That comparison is not valid and has been removed:

* The two systems compute different things (single forward pass vs. full
  diffusion sampling with MSA and confidence heads).
* The reference minutes-per-structure figures were never measured here — they
  were hardcoded constants in the benchmark script, not observations.
* **The accuracy cost is unquantified.** A surrogate is only useful if its
  ranking agrees with the reference; latency alone says nothing.

The meaningful claim has the form *"the surrogate reproduces the reference's
binder ranking with X% top-k recall (Spearman ρ) at Y ms/candidate"*.
`src/run_reference_benchmark.py` produces exactly that, folding each
(target, binder) complex with Boltz and comparing the rankings.

### First measured result

6 single-point mutants of a 15-mer binder against a 51-residue target, folded
with stock Boltz-1 on CPU in single-sequence mode (`msa: empty`), ranked by
interface confidence (ipTM):

| | reference (boltz1 ipTM) | surrogate (edge) |
| :--- | :---: | :---: |
| latency / candidate | 44,022 ms | **0.379 ms** |
| model size | 3.6 GB checkpoint | 0.51 MB |
| Spearman ρ | — | **+0.143** |
| top-3 recall | — | 66.7% |
| top-1 recall | — | 0.0% |

Reproduce with `python src/run_reference_benchmark.py --num-binders 6`.

**Read this as a negative result.** The surrogate is ~116,000x faster and ranks
at close to chance: ρ = +0.143 over N = 6 is not distinguishable from zero, and
it misses the reference's best binder entirely. That is expected — the
`AffinitySurrogate` here is untrained, and the benchmark exists precisely to
stop the latency figure from being quoted on its own.

### Distillation attempt (also negative)

30 complexes folded (batched, 25.5 s/candidate), split 20 train / 10 held-out,
trained with the pairwise rank loss for 400 epochs
(`python src/distill_against_reference.py --num-binders 30`):

| | train | held-out |
| :--- | :---: | :---: |
| Spearman ρ, after training | **+1.000** | **+0.030** |
| Kendall τ, after training | +1.000 | −0.022 |
| Spearman ρ, before training | — | −0.103 |

**Distillation did not work.** Training loss reached 0.0000 and train ρ hit
1.000 — the model memorised the ranking of all 20 training binders exactly — while
held-out ρ stayed at chance. The move from −0.103 to +0.030 is well inside noise:
Spearman over N = 10 has a standard error near 0.33.

Two causes, both structural rather than fixable by more epochs:

* **Too little data for the capacity.** 20 training pairs against a model with
  ~10^5 parameters memorises rather than generalises.
* **A weak reference signal.** Single-sequence-mode Boltz-1 puts every candidate
  in a 0.055–0.088 ipTM band (σ = 0.009). Single-point mutants of one binder
  against one target may not produce a ranking that is mostly signal, so part of
  what the surrogate is being asked to fit could be reference noise.

### MSA-backed rerun (suggestive, still not significant)

Repeated with the target's alignment subsampled to 32 sequences
(`--target-msa ... --max-msa-seqs 32`), 40 candidates, 27 train / 13 held-out:

| | single-sequence | MSA depth 32 |
| :--- | :---: | :---: |
| candidates (train / held-out) | 30 (20 / 10) | 40 (27 / 13) |
| fold cost | 25.5 s/candidate | 42.3 s/candidate |
| reference ipTM spread (σ) | 0.055–0.088 (0.0087) | 0.054–0.092 (0.0088) |
| held-out ρ **before** training | −0.103 | −0.027 |
| **held-out ρ after training** | **+0.030** | **+0.308** |
| train ρ after training | +1.000 | +1.000 |

**+0.308 over n = 13 was noise.** It has since been refuted by a properly
powered run — see below. It is left in the table as a record of why small
held-out sets should not be reported as encouraging.

### Powered rerun: distillation does not work (n = 85 held-out)

258 candidates — a near-exhaustive single-point mutant scan of the 15-mer, 258 of
286 possible — folded MSA-backed at depth 32 (2.0 h, 28.0 s/candidate), split
173 train / 85 held-out:

| held-out n | ρ | 95% CI | |
| :--- | :---: | :---: | :--- |
| 10 (single-sequence) | +0.030 | [−0.611, +0.647] | underpowered |
| 13 (MSA-32) | +0.308 | [−0.293, +0.734] | underpowered |
| **85 (MSA-32)** | **−0.034** | **[−0.245, +0.180]** | **powered** |

This run could detect \|ρ\| ≥ 0.216 at p < 0.05. It found **−0.034**, and its
confidence interval **excludes the +0.308** from the n = 13 run. Training moved
held-out ρ from −0.043 to −0.034 — nothing. The surrogate does not learn to
reproduce Boltz's ranking on binders it has not seen.

**Train ρ is still +1.000 with 173 training pairs.** Eight times more data did
not stop the model fitting the training set exactly. This is not a sample-size
problem that more folding will fix; the model memorises whatever it is given.

### Ridge baseline: the signal exists, the neural model missed it

`src/ridge_baseline.py` fits a regularised linear model over engineered binder
features on the same 258 folded references — no new folding required. It settles
which of the two explanations above is right.

| model | features | train ρ | held-out ρ (single split) |
| :--- | :---: | :---: | :---: |
| ridge, additive main effects | 59 | +0.446 | **+0.180** |
| ridge, position × residue one-hot | 300 | +0.967 | +0.095 |
| neural surrogate | ~10⁵ | +1.000 | −0.034 |

A single split is not trustworthy here — across six seeds the additive model
ranged +0.041 to +0.223. The protocol that holds up is repeated splits against
the *same procedure run on shuffled labels*:

```
real labels     30 splits: mean ρ +0.103 (sd 0.099), 27/30 positive
shuffled labels 30 splits: mean ρ +0.018 (sd 0.093)
difference +0.085, z = 3.44
```

**There is weak but real signal, and the neural surrogate failed to find it.**
Ridge generalises (train ρ 0.446, not 1.000) where the network memorised. This
supersedes the earlier suggestion that the reference was mostly noise — it is
noisy, but not empty.

Three things follow:

* **Feature design decides the outcome.** Additive main effects (+0.103 mean)
  beat a position × residue one-hot (+0.095 on one split, train ρ 0.967 — it
  memorises), because on a single-mutant scan every held-out mutant carries a
  (position, residue) pair unseen in training. Only main effects transfer.
* **The effect is small.** ρ ≈ 0.10 explains ~1% of rank variance — a real
  signal, not a usable ranker. The reference itself is weak: ipTM spans
  0.052–0.103, whereas the experimentally validated SPARK fertilization complex
  scores 0.51–0.57. Values near 0.05 mean no confident interface is predicted at
  all.
* **Single-split numbers on this task are noise.** sd ≈ 0.10 across splits, so
  any one seed over- or under-states by ±0.1. This is why +0.308 at n = 13 looked
  encouraging and why +0.180 at seed 0 should not be quoted alone either.

Reproduce: `python src/ridge_baseline.py --repeats 30`

### Multi-target control: the reference does not measure binding

`src/multi_target_benchmark.py` widens the task to three structurally distinct
targets (haemoglobin α fragment, ubiquitin, protein G B1) × 60 binders each —
wild-type, single mutants, multi-point mutants, and **scrambles**: sequences with
the *identical amino-acid composition* and the order destroyed.

The scramble is the control that matters. If ipTM tracked binding, destroying
sequence order should lower it consistently.

| target | designed | scrambled | Cohen's d | p (Mann-Whitney) |
| :--- | :---: | :---: | :---: | :---: |
| hba | 0.0738 | 0.0736 | +0.02 | 1.000 |
| ubq | 0.1164 | 0.1015 | +0.62 | 0.039 |
| gb1 | 0.0823 | 0.0937 | **−0.53** | 0.116 |

Mean effect across targets **d = +0.035 (t = 0.10, p = 0.93)**, and the signs
disagree — for GB1, scrambling the binder *raised* predicted interface
confidence. **ipTM here is largely insensitive to binder sequence order**, which
is precisely the property binding specificity would require.

Meanwhile the ranking *is* learnable, and far better than on the single-mutant
scan:

| target | ridge (repeated splits) | shuffled-label null | z |
| :--- | :---: | :---: | :---: |
| hba | −0.119 | −0.088 | −0.42 |
| ubq | +0.203 | −0.079 | **+5.24** |
| gb1 | +0.230 | −0.036 | **+3.83** |

**Taken together: the benchmark is learnable but not meaningful.** A surrogate
can be trained to reproduce this ipTM ranking (z > 3 for two of three targets),
but ipTM is not responding to what makes a binder a binder. Simple composition
features correlate only weakly and inconsistently (|ρ| ≤ 0.29, signs vary by
target), so it is not one property either — it is target-specific structure that
does not survive the order control.

The blocking problem is the **reference**, not the surrogate. Distilling harder
against this signal reproduces an artefact. Progress needs binders with measured
affinity — experimental data, or at minimum known peptide–protein complexes with
true positives and negatives — not more compute. For scale, every ipTM here
(0.055–0.212) sits far below the 0.51–0.57 of the experimentally validated SPARK
fertilization complex; Boltz is reporting no confident interface for any of
these peptides.

Reproduce: `python src/multi_target_benchmark.py --binders-per-target 60`

### PDB binders: ipTM has sensitivity but not specificity

`src/pdb_binder_benchmark.py` replaces synthetic peptides with **11
experimentally determined peptide–domain complexes** from the PDB (MDM2/p53,
SH3/proline-rich, PDZ/C-terminal, and others; sequences fetched live from RCSB,
never transcribed). Receptor MSAs fetched per receptor (1,516 homologs for MDM2).
Three classes, folded identically:

* **cognate** (11) — receptor + its own peptide
* **decoy** (33) — receptor + a real peptide from a *different* complex: a
  genuine binder, wrong partner
* **scrambled** (22) — receptor + its own peptide, order destroyed

**The screening test is within-receptor** — rank one receptor's candidates
against each other. That is chance:

```
cognate ranked #1 for 2/11 receptors   (chance = 1.8)
mean rank 3.27 of 6                    (chance = 3.50)
Wilcoxon vs chance:  p = 0.746
```

For two receptors (1D4T, 1I8H) the true binder ranked **last of six**. Pooled
across receptors the direction is right but weak and not significant: cognate
0.1915 vs decoy 0.1684, AUC 0.614, p = 0.13.

**ipTM does discriminate at a coarser level.** Real complexes score far above the
synthetic peptides of the earlier runs — mean 0.1915 vs 0.0905, AUC 0.881,
p = 1e-5 — though that comparison changes both receptor and peptide, so it is not
cleanly attributable to binder quality.

**Conclusion for this project.** ipTM at these settings has *sensitivity*
(real complexes look different from arbitrary peptides) but not *specificity*
(it cannot tell which peptide belongs to which receptor). Screening needs
specificity. That closes the question the benchmark thread opened: the reference
is not usable for binder ranking, so a surrogate distilled from it cannot be
either — and no amount of surrogate engineering changes that.

Caveats: 10 sampling steps, 1 recycling, MSA subsampled to depth 32, and
n = 11 receptors. A stronger configuration may do better; this measures what
this pipeline provides. The "Boltz-1 rather than Boltz-2" caveat is now tested
directly — see below.

Reproduce: `python src/pdb_binder_benchmark.py`

### Boltz-2 on the same benchmark: better, not enough

The obvious objection to the result above is that it indicts Boltz-1, not ipTM.
`--model boltz2` folds the identical 66 pairs with identical seeds and
byte-identical cached MSAs, so the checkpoint is the only variable (~3 min per
complex on CPU, 66 folds, 0 failures).

**Every score shifts up by roughly 2.8x, and the classes shift together:**

| | cognate (11) | decoy (33) | scrambled (22) |
| :--- | :---: | :---: | :---: |
| Boltz-1 | 0.1915 | 0.1684 | 0.1758 |
| Boltz-2 | 0.5355 | 0.4556 | 0.4662 |

Within-receptor rank of the cognate — the screening test — improves in every
direction: **#1 for 3/11 receptors, up from 2/11; mean rank 2.55 of 6, up from
3.27** (chance 3.50). But significance depends entirely on analysis choices that
were not pre-specified:

| model | competitors | mean rank | chance | 1-sided | 2-sided |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Boltz-1 | decoys only | 2.36 | 2.50 | 0.395 | 0.790 |
| Boltz-1 | decoy + scrambled | 3.27 | 3.50 | 0.373 | 0.746 |
| Boltz-2 | decoys only | 1.91 | 2.50 | 0.062 | 0.124 |
| Boltz-2 | decoy + scrambled | 2.55 | 3.50 | **0.047** | 0.094 |

**One of four cells clears 0.05, and it is the least conservative combination.**
The table above uses the two-sided convention, under which Boltz-2 does not reach
significance. Bootstrap agrees: Boltz-2's mean cognate rank is 1.91 with 95% CI
[1.36, 2.55] — the interval contains chance. The paired improvement over Boltz-1
is +0.45 ranks, 95% CI [−0.36, +1.18] — contains zero.

**A pooled test appears to pass and should not be believed.** The benchmark
script's own output reports cognate > decoy at AUC 0.689, p = 0.033. That test is
confounded by receptor identity (receptors differ several-fold in baseline ipTM,
so part of what it measures is *which receptor this is*), and it is internally
incoherent with the rest of the data: **decoys and scrambles are
indistinguishable under Boltz-2 — AUC 0.501, p = 0.993.** Real peptides that bind
some other receptor score exactly like sequence-order garbage. If ipTM cannot
separate a genuine binder from a scramble, its edge over decoys is not
binder recognition. The p = 0.033 comes from the decoy set being larger (33 vs
22), not from a distinction that exists.

**Conclusion.** Boltz-2 moves the specificity result in the right direction and
by a meaningful amount, but does not establish specificity at n = 11, and the
model-to-model difference is not distinguishable from noise. The Boltz-1
conclusion stands, now with the model caveat tested rather than assumed.

**This was a power problem, and the powered run has now been done.** See below.

### The powered run: ipTM tracks composition, not binding

80% power needed 21 receptors (dz = 0.57). A PTM-clean, tag-free,
peptide-deduplicated panel of **22 receptors / 132 complexes** was folded under
Boltz-2. Two of those filters exist because earlier passes got them wrong —
three histone H3 tails and a duplicated PI3K phosphopeptide would have made
other receptors' "decoys" genuine binders, and **7 of 25 candidates were
PTM-dependent**, including 1I8H from the original panel, whose peptide needs
phosphothreonine and therefore never bound as folded.

**The specificity test passes.** Against decoys the cognate reaches mean rank
**2.00 vs chance 2.50** (#1 for 8/22, p = 0.034 two-sided), bootstrap 95% CI
**[1.59, 2.41]** excluding chance. Alone, that reverses the n = 11 result.

**The scramble control refutes the binding reading.**

| class | n | mean ipTM |
| :--- | :---: | :---: |
| cognate | 22 | 0.5015 |
| **scrambled** | 44 | **0.4888** |
| decoy | 66 | 0.4317 |

A scramble keeps composition and length exactly and destroys only order — and
order is what makes a binder a binder.

* Cognates **do not** beat their own scrambles: 11/22, mean +0.0128, p = 0.416.
* Scrambles **outscore decoys**: AUC 0.368, p = 0.019.

So what separates cognate from decoy is exactly what a cognate shares with its
own scramble — composition. Order contributes nothing measurable.

**Not a length artifact**, which was tested rather than assumed: ipTM falls with
peptide length (Spearman −0.408, p = 1.2e-6), but regressing per-pair
cognate-minus-decoy score difference on length difference leaves intercept
**+0.0752, p = 0.0001**. The advantage survives length adjustment. It is
compositional, not positional.

**Conclusion.** The powered run settles the question, and the answer is still
negative — now with a mechanism. This converges with the ridge-regression result
above from an independent direction: additive composition beat the distilled
neural surrogate (+0.103 vs −0.034). Reporting only the decoys-only row
(p = 0.034) would have produced a positive headline this experiment's own
control contradicts.

Reproduce: `python src/pdb_binder_benchmark.py --model boltz2 --work-dir artifacts/pdb_binders_b2_n22`

Reproduce: `python src/pdb_binder_benchmark.py --model boltz2 --work-dir artifacts/pdb_binders_b2`
then `python src/compare_boltz1_boltz2.py`

### The low-rank OPM is not reachable on pretrained weights

The low-rank OuterProductMean saves ~97% of the activation memory the stock layer
materialises, but its parameters do not match stock Boltz checkpoints, so using it
means training from scratch. Three experiments, each stricter than the last, ask
whether the saving can be had on pretrained weights instead. The short answer is
no — and the reason is capacity, established only by the third.

| experiment | what it measures | held-out error @ rank 32 |
| :--- | :--- | :---: |
| CP projection of weights | tensor approximation, random inputs | 0.77 |
| fit on one target's activations | function approximation, one protein | 0.43 |
| **corpus distillation (33 folds)** | **function approximation, at capacity** | **0.378** |

Each step was a genuine improvement, and the first two conclusions were stated
too strongly on the evidence available at the time. The third settles it.

#### 1. Weight-space CP projection — 0.77

`src/opm_cp_projection.py`. Stock computes `out[i,j,e] = mᵢᵀ (Aᵀ Oₑ B) mⱼ`; the
low-rank form computes `mᵢᵀ (Pxᵀ diag(Wₑ) Py) mⱼ`. Matching them requires writing
every `Oₑ` from a **shared** set of rank-1 terms — a CP decomposition of
`O ∈ R^(128×32×32)`. Given one, `Px = UᵀA` and `Py = VᵀB` reproduce the layer
exactly.

Stock computes `out[i,j,e] = mᵢᵀ (Aᵀ Oₑ B) mⱼ`; the low-rank form computes
`mᵢᵀ (Pxᵀ diag(Wₑ) Py) mⱼ`. Matching them requires writing every `Oₑ` from a
**shared** set of rank-1 terms — a CP decomposition of `O ∈ R^(128×32×32)` at
rank R. Given one, `Px = UᵀA` and `Py = VᵀB` reproduce the layer exactly.

Measured on `boltz1_conf.ckpt` (c_in 64, c_hidden 32, c_out 128):

| rank | CP error | output error | params |
| :--- | :---: | :---: | :---: |
| 8 | 0.927 | 0.879 | 2,048 |
| 16 | 0.883 | 0.823 | 4,096 |
| **32** (the rank actually used) | **0.831** | **0.773** | 8,192 |
| 64 | 0.758 | 0.708 | 16,384 |
| 128 | 0.641 | 0.605 | 32,768 |

vs 131,072 params for stock. **It does not work.** At the native rank the
projection discards ~83% of the tensor and the layer output is ~77% wrong.

Two controls confirm this is a property of the weights, not of the solver: an
exactly-rank-32 tensor decomposes with error **0.00000**, and a random dense
tensor gives **0.947** — so the trained tensor (0.831) is only marginally more
compressible than noise.

Reproduce: `python src/opm_cp_projection.py --ranks 8,16,32,64,128`

This measures the **weight tensor**, on `torch.randn` inputs. That is the worst
case and does not by itself establish that the *function* is hard to approximate
— a distinction the next experiment exists to test.

#### 2. Fitting on real activations — 0.43

`src/opm_capture_activations.py` monkeypatches the stock layer and drives the real
Boltz CLI in-process, recording genuine `(m_norm, mask)` and stock outputs;
`src/opm_fit_on_activations.py` then fits Px/Py/W/bias directly against them.
Capture fidelity is verified by recomputing the stock output from stored weights
(rel err ~6e-7). Fitted on MDM2 (1YCR), evaluated on a PDZ domain.

| rank | activations vs stock | layer_0 held-out | layer_1 held-out |
| :--- | :---: | :---: | :---: |
| 16 | 1.6% | 0.478 | 0.732 |
| **32** | 3.1% | **0.426** | 0.676 |
| 64 | 6.3% | 0.363 | 0.589 |
| 128 | 12.5% | 0.284 | 0.477 |

On-distribution matters: 0.77 → 0.43 at the native rank. But this fitted 8k
parameters to a single protein, and train (0.35) sat well below held-out (0.43) —
so the factors were partly target-specific and the ceiling was unclear.

#### 3. Corpus distillation — 0.378, and at capacity

`src/opm_corpus_capture.py` + `src/opm_corpus_distill.py`. 264 activations from
66 folds spanning 11 receptors, split into disjoint 33-fold halves.

| | rank 32 | rank 64 | rank 128 |
| :--- | :---: | :---: | :---: |
| layer_0 train / held-out | 0.375 / **0.378** | 0.309 / 0.312 | 0.228 / 0.231 |
| layer_1 train / held-out | 0.576 / **0.585** | 0.485 / 0.494 | 0.372 / 0.380 |
| activations vs stock | 3.1% | 6.2% | 12.5% |

**The train/held-out gap collapsed** from 0.08 to 0.003. The corpus solved
generalisation — the factors are shared across proteins, not memorised. But train
and held-out now coincide, which is the signature of a model *at capacity*. More
folding data will not lower this floor.

**Rank cannot buy the fidelity back.** Fitting `err = a·rankᵇ` on the held-out
points gives `1.32·rank^−0.356` (layer_0) and `1.75·rank^−0.312` (layer_1). To
reach 10% error needs rank ≈ 1,414 and ≈ 9,750 respectively — against
`c_hidden² = 1024`, the width stock actually materialises. **The low-rank form
costs more activation memory than stock before it becomes accurate enough**;
layer_1 crosses over before even 20%. (That extrapolation runs about an order of
magnitude past the fitted points and is indicative; layer_1's 20% crossover sits
close to measured range.)

**Consequence.** The usable regime is 23–38% per-layer error at 3–13% of stock
activations — a large saving at an error that compounds across the MSA stack, not
a drop-in replacement. No lDDT measurement was run at any stage, because error of
that size per layer makes the structural outcome determined. Obtaining this memory
win requires training the whole model from scratch around the low-rank layer.

Reproduce: `python src/opm_corpus_capture.py --inputs <yamls> --max-per-layer 66`
then `python src/opm_corpus_distill.py --ranks 32,64,128`

Further caveats on the reference itself: ipTM is interface confidence, not
affinity (Boltz-2's affinity head targets protein-*ligand* binding, so it does
not apply to peptide binders); single-sequence mode without an MSA yields low
absolute confidence for every candidate (0.061-0.081), so the reference ranking
is itself weak; and N = 6 is far too small for a stable correlation. Treat the
table as a demonstration that the measurement pipeline works end to end, not as
a characterization of the method.

---

## 🛠️ Codebase Structure

* `src/predict_structure.py`: Main Swift/App wrapper mapping sequence strings to embeddings and calling the CoreAI runtime.
* `src/convert_surrogate_coreai.py`: Quantization, PyTorch Export, and CoreAI compilation toolchain.
* `src/benchmark_boltz_coreai.py`: Comparative latency test comparing CPU, MPS GPU, and CoreAI against public baselines.
* `src/visualize_window.py`: Interactive 3D plotting utility displaying predicted C-alpha backbones using matplotlib.
* `src/benchmark_dynamic_actual_samples.py`: Latency verification on actual protein sequences (Human Insulin, Hemoglobin).

### Target-disjoint prospective ranking

`src/campaign_ranking.py` provides the campaign layer for real binder discovery:

1. Use `TargetedCampaignReward` with a real `predict_fn(target, binder)` to run co-design while recording every proposal's affinity, confidence, clash, and developability features.
2. Attach experimental scores to the resulting `CandidateRecord` objects using a higher-is-better convention (for example `-log10(Kd)`).
3. Create a target-disjoint train/calibration/test split, fit `CalibratedEnsembleRanker`, and select a small, diverse experimental batch with uncertainty- and developability-aware acquisition.

The ranker keeps model scores separate from assay labels and rejects target leakage between training and calibration. Synthetic rewards remain suitable only for software tests, never for a prospective performance claim.

`src/prospective_campaign.py` extends this into a publishable campaign protocol: define intended and off-target `TargetContext` states in a `DesignSpec`, verify length/developability/hotspot constraints with an auditable trace, aggregate desired states conservatively and counter-screens pessimistically, then report fixed-budget top-k success, cluster diversity, and throughput. `TargetConditionalConformalRouter` keeps confident, constraint-satisfying candidates on an edge ranker and routes uncertain candidates to Boltz/reference scoring. The included synthetic tests exercise workflow semantics only; they are not biological validation.

### Risk-controlled cross-context design

`src/certified_selectivity.py` is the research-method layer. It uses one joint split-conformal residual for the weakest desired context and strongest counter-screen, producing a conservative lower bound on robust selectivity. A candidate is selected only when that lower bound exceeds a declared margin and its verifier trace passes. `select_with_familywise_risk_control` applies a Bonferroni campaign-level error budget; `CostAwareReferenceAllocator` reserves expensive Boltz/reference calls for valid candidates near the certification boundary where a narrower reference interval can change the decision. The coverage guarantee is marginal and assumes calibration examples are exchangeable with future target-disjoint examples—report that assumption and assay results explicitly.

`src/empirical_study.py` turns assay-labelled `ContextualAssayRecord` objects into a target-disjoint result table: raw predicted-selectivity, verifier-gated, and joint-conformal risk-controlled baselines; coverage/interval-width/selectivity curves; and fixed-budget discovery replay. `build_prospective_assay_manifest` emits only deduplicated, verifier-passing certified candidates for lab execution. Pre-register the split seed, selectivity threshold, cost model, clustering rule, and assay readout before using this report for a publication.

---

## 💻 Quick Start

### 1. Requirements & Setup
Ensure you have the CoreAI runtime installed in your environment:
```bash
conda activate coreai
pip install matplotlib markdown
```

### 2. Run Latency Benchmarks
Evaluate local CPU, GPU, and Neural Engine execution speeds on a 1300-residue sequence:
```bash
python src/benchmark_boltz_coreai.py
```

### 3. Run Predictions on Actual Protein Sequences
Evaluate structure coordinates dynamically for variable sequence lengths:
```bash
python src/benchmark_dynamic_actual_samples.py
```

### 4. Interactive 3D Visualization
Run structure prediction and spin the resulting 3D coordinate envelope on your desktop:
```bash
python src/visualize_window.py
```

---

## 📦 Native macOS / Swift Application Integration

To package this model into a macOS App, compile the asset folder (`.aimodelc`) using Apple's compiler and drag it into Xcode. Use the following Swift blueprint to manage the Neural Engine state and run predictions:

```swift
import CoreML
import Accelerate

class DynamicStructurePredictor {
    private let model: MLModel
    private var kCache: MLMultiArray
    private var vCache: MLMultiArray
    
    init(compiledModelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: compiledModelURL, configuration: config)
        
        // Pre-allocate Key-Value Cache Buffer [1, 4, 2500, 32]
        self.kCache = try MLMultiArray(shape: [1, 4, 2500, 32], dataType: .float32)
        self.vCache = try MLMultiArray(shape: [1, 4, 2500, 32], dataType: .float32)
    }
    
    func predict(binderEmbeds: MLMultiArray, targetK: MLMultiArray, targetV: MLMultiArray) throws -> MLMultiArray {
        let inputDict: [String: Any] = [
            "binder_seq": binderEmbeds,
            "target_k": targetK,
            "target_v": targetV,
            "cross_attn.k_cache": self.kCache,
            "cross_attn.v_cache": self.vCache
        ]
        let inputs = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let outputs = try model.prediction(from: inputs)
        
        // Update state pointers for next run
        self.kCache = outputs.featureValue(for: "cross_attn.k_cache")!.multiArrayValue!
        self.vCache = outputs.featureValue(for: "cross_attn.v_cache")!.multiArrayValue!
        
        return outputs.featureValue(for: "coords")!.multiArrayValue!
    }
}
```

---

## 🎓 Academic Affiliation
This project was developed in partial fulfillment of the **M.Tech. in Artificial Intelligence and Machine Learning** degree program at the **Birla Institute of Technology and Science (BITS), Pilani**.
