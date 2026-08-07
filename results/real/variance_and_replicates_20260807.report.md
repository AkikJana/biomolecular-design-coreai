# How much signal is there, and how noisy is the few-step model?

Two analyses that between them correct earlier claims in Sections 7.5 and 7.8.

## 1. The rank tests were discarding most of the data

Every test in Sections 7.4 to 7.8 collapses six folds per receptor into one
integer rank. That protects against receptor-level baseline variation but throws
away magnitude, which is why results sat near p = 0.03 at n = 22.

A linear mixed model with receptor as a random effect gives the same protection
without the loss:

    s(R,P) = mu + alpha_R + beta*cognate + gamma*scramble + delta_RP + eps

| arm | metric | rank-test p | **mixed-model p** |
| :--- | :--- | ---: | ---: |
| Boltz-2 | ipTM | 0.034 | **0.0042** |
| Boltz-2 | interface pLDDT | 0.027 | **0.00005** |
| Boltz-1 | ipTM | 0.017 | **0.00009** |
| Boltz-1 | interface pLDDT | 0.010 | 0.0029 |
| DeCAF | ipTM | 0.0087 | **< 1e-5** |
| DeCAF | interface pLDDT | 0.0042 | **< 1e-5** |

Roughly an order of magnitude, from the estimator alone. Several results
described as "suggestive but not significant" in Sections 7.4 to 7.7 were limited
by the analysis rather than by the data.

## 2. Variance decomposition, and a ceiling

Splitting residual variance into the receptor-peptide interaction — the binding
signal — and fold-to-fold sampling noise measured from replicates:

| arm | metric | σ²_receptor | σ²_interaction | σ²_noise | signal/noise |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Boltz-2 | ipTM | 0.0054 | 0.0059 | 0.0039 | 1.48 |
| Boltz-2 | interface pLDDT | 14.46 | 10.01 | 3.68 | 2.72 |
| Boltz-1 | ipTM | 0.0023 | 0.0002 | 0.0039 | **0.05** |
| Boltz-1 | interface pLDDT | 13.48 | 9.90 | 3.68 | 2.69 |
| DeCAF | ipTM | 0.0157 | 0.0366 | 0.0096 | 3.80 |
| DeCAF | interface pLDDT | 85.84 | 55.82 | 9.17 | **6.08** |

σ²_interaction / σ²_noise bounds *any* score computed from that model. Two
readings follow:

**Boltz-1's ipTM is noise-limited (0.05); every other cell is signal-limited.**
Its weak result is a different failure from Boltz-2's — one is drowned, the
other is nearly absent. Section 7.8 treats them as the same phenomenon and
should not.

**Averaging is not the general fix.** Every cell above 1 means noise is not the
binding constraint on aggregate discriminability. Section 8.2 currently leads
with "report replicate-averaged confidence, never a single fold"; on this
evidence that holds for per-receptor claims but not for population-level ones.

## 3. DeCAF is noisier than its teacher — correction

The decomposition above initially borrowed Boltz-2's noise term for the DeCAF
arm, for want of a measurement. The prediction behind that was that a model
distilled to land accurately in ten steps would be *less* stochastic.

**The opposite is true.** Mirroring Section 7.5's design exactly — same 24
complexes, same 4 receptors, 4 identical unseeded re-runs, 96 folds:

| metric | DeCAF SD | Boltz-2 SD | ratio |
| :--- | ---: | ---: | ---: |
| ipTM | 0.0981 | 0.0628 | **1.56** |
| interface pLDDT | 3.029 | 1.917 | **1.58** |

Fewer, larger jumps move further per draw, so run-to-run spread grows. Correcting
the decomposition:

| metric | S/N with borrowed noise | **S/N measured** | DeCAF advantage |
| :--- | ---: | ---: | :--- |
| ipTM | 10.71 | **3.80** | 2.6x over Boltz-2 (not 7.2x) |
| interface pLDDT | 16.68 | **6.08** | 2.2x over Boltz-2 (not 6.1x) |

DeCAF remains the best arm by a clear margin, but by roughly **half** what the
borrowed-noise figure implied. The raw effect sizes in Section 7.8 (+0.201,
+9.54) are unaffected — they are measured, not inferred. What changes is the
discriminability those effects buy once the model's own noise is accounted for.

## 4. Section 7.5's rank instability generalises

Cognate rank among its own decoys, across four identical re-runs:

```
ipTM             stable for 0/4    1YCR [1,2,1,1]  6YOO [2,2,2,1]
                                   8KDX [4,2,3,2]  9F6S [2,3,3,3]
interface pLDDT  stable for 1/4    1YCR [1,3,1,1]  6YOO [2,2,2,2]
                                   8KDX [4,4,3,3]  9F6S [2,3,2,2]
Boltz-2 reference: ipTM stable for 0/4
```

Single unseeded folds do not reproduce their own per-receptor rankings on the
few-step model either. Section 7.5's finding is a property of these folding
models generally, not of the degraded Boltz-2 regime — which strengthens it.

Note this is not in tension with the signal-limited verdict above. Aggregate
discriminability is not noise-bound, but individual within-receptor comparisons
are close enough that a single draw reorders them. **Average for per-receptor
claims; averaging will not rescue a weak metric in aggregate.**

## Reproduce

```
python src/decaf_replicate_study.py --replicates 4
python src/variance_decomposition.py
```
