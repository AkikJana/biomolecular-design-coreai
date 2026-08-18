# BITS PILANI — WORK INTEGRATED LEARNING PROGRAMMES (WILP)
## M. Tech. Final Dissertation Report
**Academic Year: 2025-2026 | Second Semester**

---

# A REPORT

## ON

## EFFICIENCY-OPTIMIZED GENERATIVE PARADIGMS FOR LARGE-SCALE BIOMOLECULAR DESIGN

&nbsp;

### BY

| Name of the Student | ID. No. | Discipline |
| :--- | :--- | :--- |
| Akik Jana | 2024AB05287 | M.Tech. Artificial Intelligence and Machine Learning |

&nbsp;

**Prepared in partial fulfilment of the**
**WILP Dissertation Course (BITS ZG628T)**

### AT

**Mu Sigma, Bangalore**

**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI**

**August 2026**

<div class="page-break"></div>

# CERTIFICATE

This is to certify that the Dissertation entitled **"Efficiency-Optimized
Generative Paradigms for Large-Scale Biomolecular Design"** and submitted by
**Akik Jana**, ID No. **2024AB05287**, in partial fulfilment of the requirements
of the course **BITS ZG628T Dissertation**, embodies the work done by him under
my supervision.

&nbsp;

&nbsp;

<div class="signature-block">
    <div class="sig-col">
        <img class="sig-img" src="assets/sig_supervisor.png" alt="Signature">
        <div class="sig-line"></div>
        <div class="sig-title">Signature of the Supervisor</div>
        <div class="sig-field"><b>Name:</b> Dr. Arnab Bandyopadhyay</div>
        <div class="sig-field"><b>Designation:</b> RnD Division, Dr. Reddy's Laboratories</div>
        <div class="sig-field"><b>Date:</b> August 2, 2026</div>
        <div class="sig-field"><b>Place:</b> Hyderabad</div>
    </div>
    <div class="sig-col">
        <img class="sig-img" src="assets/sig_student.png" alt="Signature">
        <div class="sig-line"></div>
        <div class="sig-title">Signature of the Student</div>
        <div class="sig-field"><b>Name:</b> Akik Jana</div>
        <div class="sig-field"><b>ID No.:</b> 2024AB05287</div>
        <div class="sig-field"><b>Date:</b> August 2, 2026</div>
        <div class="sig-field"><b>Place:</b> Bangalore</div>
    </div>
</div>

<div class="page-break"></div>

# ACKNOWLEDGEMENTS

I am grateful to the management of **Mu Sigma, Bangalore**, for supporting this
dissertation and for providing the environment in which it was carried out.

I thank my supervisor, **Dr. Arnab Bandyopadhyay**, RnD Division, Dr. Reddy's
Laboratories, Hyderabad, for his guidance throughout this work, and in
particular for his insistence that a negative result carefully established is
worth more than a positive result loosely argued. That principle shaped the
second half of this dissertation.

I thank the **faculty mentor** and the **WILP Division, BITS Pilani**, for the
structure and review that kept the work on schedule.

Finally, I acknowledge the maintainers of the open-source **Boltz** project,
whose released model and permissive licence made an independent evaluation of
this kind possible at all.

<div class="page-break"></div>

# ABSTRACT SHEET

### BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI (RAJASTHAN)
### WILP Division

| | |
| :--- | :--- |
| **Organization** | Mu Sigma |
| **Location** | Bangalore, India |
| **Duration** | 05 January 2026 – 02 August 2026 (approximately 30 weeks) |
| **Date of Start** | 05 January 2026 |
| **Date of Submission** | 02 August 2026 |
| **Title of the Project** | Efficiency-Optimized Generative Paradigms for Large-Scale Biomolecular Design |
| **ID No. / Name of the Student** | 2024AB05287 / Akik Jana |
| **Name and Designation of the Supervisor** | Dr. Arnab Bandyopadhyay, RnD Division, Dr. Reddy's Laboratories, Hyderabad |
| **Name of the Faculty Mentor** | As assigned by the WILP Division |
| **Key Words** | Protein structure prediction; binder design; inference efficiency; low-rank factorization; tensor decomposition; knowledge distillation; interface confidence; benchmark reproducibility; Apple Silicon; edge inference |
| **Project Areas** | Machine Learning; Computational Structural Biology; Efficient Deep Learning Inference |

**Abstract:**

This dissertation investigates whether inference-efficiency techniques drawn
from large language models can make all-atom biomolecular structure prediction
and binder screening practical on edge hardware. A framework, Boltz-Fast, was
implemented on the open Boltz model: latent key-value caching, sequence
parallelism, a low-rank pair representation, classifier-free-guidance
distillation, and a neural coordinate refiner, all executing device-agnostically
on Apple Silicon. Component microbenchmarks showed large savings. The work then
tested those components against pretrained checkpoints and experimentally
determined structures rather than synthetic tensors, and two central claims did
not survive. The low-rank pair representation cannot be recovered from released
weights at any rank that saves memory. Interface confidence responds largely to
peptide composition rather than to binding, and a single unseeded fold does not
reproduce its own ranking. Each negative result is supported by controls, held-out splits, power analysis
and replication. Re-scoring the same structures then recovered a positive
result: interface pLDDT separates a binder from its own scramble where ipTM
cannot, at 8.6 times the effect-to-noise ratio. A second panel of receptors
released after the model's training cutoff then bounded that recovery — over
five independent draws both readouts retain only about half their effect on
complexes the model was not trained on, so a benchmark figure quoted without
stating which regime it was measured in is roughly a factor of two optimistic.
Auditing the predicted coordinates themselves explains much of the rest: at the
reduced sampling used throughout, only 14% of backbone bonds are physically
plausible against 96% for a model distilled for that step budget, and every
geometry-derived readout fails on the former and works on the latter. Folding
the panel at the model's intended settings then bounds the negative results
themselves: the cognate-versus-scramble effect is three to seven times larger in
standardised terms, so the conclusions above hold for a model run at a tenth of
its sampling budget and understate what it does when run properly. The efficiency
finding survives that correction and is strengthened by it — a model distilled
for ten steps recovers most of the full-settings signal at a twentieth of the
cost.

&nbsp;

<div class="signature-block">
    <div class="sig-col">
        <img class="sig-img" src="assets/sig_student.png" alt="Signature">
        <img class="sig-img" src="assets/sig_supervisor.png" alt="Signature">
        <div class="sig-line"></div>
        <div class="sig-title">Signature of the Student</div>
        <div class="sig-field"><b>Name:</b> Akik Jana</div>
        <div class="sig-field"><b>Date:</b> August 2, 2026</div>
    </div>
    <div class="sig-col">
        <img class="sig-img" src="assets/sig_supervisor.png" alt="Signature">
        <div class="sig-line"></div>
        <div class="sig-title">Signature of the Supervisor</div>
        <div class="sig-field"><b>Name:</b> Dr. Arnab Bandyopadhyay</div>
        <div class="sig-field"><b>Date:</b> August 2, 2026</div>
    </div>
</div>

<div class="page-break"></div>


# TABLE OF CONTENTS

CERTIFICATE ................................................................. ii  
ACKNOWLEDGEMENTS ........................................................... iii  
ABSTRACT SHEET .............................................................. iv  
1. INTRODUCTION .............................................................. 1  
&nbsp;&nbsp;&nbsp;&nbsp;1.1 Background ............................................................... 1  
&nbsp;&nbsp;&nbsp;&nbsp;1.2 Problem Statement ........................................................ 1  
&nbsp;&nbsp;&nbsp;&nbsp;1.3 Objectives ............................................................... 1  
&nbsp;&nbsp;&nbsp;&nbsp;1.4 Literature Survey ........................................................ 2  
&nbsp;&nbsp;&nbsp;&nbsp;1.5 Scope and Limitations .................................................... 2  
&nbsp;&nbsp;&nbsp;&nbsp;1.6 Organisation of the Report ............................................... 3  
2. MODULES IN BOLTZ-FAST ..................................................... 4  
3. FUNCTIONAL BLOCK DIAGRAM & METHODOLOGY .................................... 7  
4. MAJOR TECHNICAL SPECIFICATIONS & BENCHMARKS ............................... 8  
5. DESIGN CONSIDERATIONS .................................................... 13  
6. VERIFICATION & TESTING (M1 & M6) ......................................... 14  
&nbsp;&nbsp;&nbsp;&nbsp;6.1 Subsequent Hardening of the Suite ....................................... 14  
7. MEASURED RESULTS VS PRETRAINED WEIGHTS ................................... 16  
&nbsp;&nbsp;&nbsp;&nbsp;7.1 Low-Rank Pair Representation on Pretrained Weights ...................... 16  
&nbsp;&nbsp;&nbsp;&nbsp;7.2 Interface Confidence Does Not Rank Peptide Binders ...................... 17  
&nbsp;&nbsp;&nbsp;&nbsp;7.3 Boltz-2 Comparison and a Correction to the Benchmark .................... 17  
&nbsp;&nbsp;&nbsp;&nbsp;7.4 Powered Run: ipTM Tracks Composition, Not Binding — at 10 Sampling Steps ... 18  
&nbsp;&nbsp;&nbsp;&nbsp;7.5 Measurement Reproducibility ............................................. 19  
&nbsp;&nbsp;&nbsp;&nbsp;7.6 Interface pLDDT Recovers What ipTM Discards — On Structures the Model Has Seen ... 21  
&nbsp;&nbsp;&nbsp;&nbsp;7.7 Localising the Interface-pLDDT Signal ................................... 24  
&nbsp;&nbsp;&nbsp;&nbsp;7.8 Few-Step Distillation, and What It Reveals About the Negatives .......... 27  
&nbsp;&nbsp;&nbsp;&nbsp;7.9 Variance Decomposition and the Reproducibility of the Few-Step Model .... 30  
&nbsp;&nbsp;&nbsp;&nbsp;7.10 Is the Panel Measuring Prediction or Retrieval? ........................ 34  
&nbsp;&nbsp;&nbsp;&nbsp;7.11 Why the Readouts Behave As They Do: Backbone Convergence ............... 39  
&nbsp;&nbsp;&nbsp;&nbsp;7.12 Reading the Panel as a Competition ..................................... 42  
&nbsp;&nbsp;&nbsp;&nbsp;7.13 The Settings Confound, Resolved — and It Was Real ...................... 45  
&nbsp;&nbsp;&nbsp;&nbsp;7.14 The Findings as a Tool ................................................. 49  
&nbsp;&nbsp;&nbsp;&nbsp;7.15 An Automated Search Over Readouts, and What Its Controls Refuse ........ 54  
8. CONCLUSIONS AND RECOMMENDATIONS .......................................... 57  
&nbsp;&nbsp;&nbsp;&nbsp;8.1 Conclusions ............................................................. 57  
&nbsp;&nbsp;&nbsp;&nbsp;8.2 Recommendations ......................................................... 61  
9. FUTURE PLAN .............................................................. 64  
10. REFERENCES .............................................................. 68  
APPENDIX A — ABBREVIATIONS AND GLOSSARY ..................................... 70  
APPENDIX B — REPRODUCTION OF RESULTS ........................................ 72  
CHECKLIST OF ITEMS FOR THE FINAL REPORT ..................................... 77  

&nbsp;

## LIST OF FIGURES

Figure 1: Core Modules in Boltz-Fast ......................................... 4  
Figure 2: C-alpha Backbone 3D Coordinate Plot ............................... 10  
Figure 3: Ray-Traced Protein Ribbon Model Rendering ......................... 11  
Figure 4: ANE-Accelerated 3D Insulin Backbone Visualization ................. 12  
Figure 5: Post-Diffusion Neural Coordinate Refinement Comparison ............ 15  
Figure 6: ipTM and Interface pLDDT on the Scramble Control .................. 23  

&nbsp;

## LIST OF TABLES

Table 1: Computational Complexity & Scaling Comparison ....................... 8  
Table 2: Speculative Flow Matching Grid Sweep Data ........................... 8  
Table 3: Low-Rank Pair Representation Performance Scaling (Rank r=16, D_pair=128) ... 9  
Table 4: Dynamic Shape Latency Benchmarks on Apple Silicon .................. 10  

<div class="page-break"></div>

# 1. INTRODUCTION

## 1.1 Background

All-atom biomolecular structure prediction moved from a research problem to an
engineering one with AlphaFold [1], and extended to complexes with
AlphaFold-Multimer [2] and to general biomolecular interactions with AlphaFold 3
[3]. The open Boltz models [4], [5] reproduce much of that capability under a
permissive licence, which makes independent evaluation and modification
possible.

The cost, however, remains high. A single complex prediction involves a deep
trunk executed under several recycling iterations, triangular attention that is
quadratic in sequence length, and a diffusion sampler run for on the order of
200 steps. On a workstation without a datacentre GPU this is a minutes-to-hours
operation. Binder screening compounds the problem: a campaign ranks hundreds or
thousands of candidate sequences against one target, so the per-candidate cost
determines whether the method is usable at all.

## 1.2 Problem Statement

The problem addressed by this dissertation is whether the inference-efficiency
techniques developed for large language models can be transferred to
biomolecular structure prediction so that structure-based binder screening
becomes practical on edge hardware — specifically Apple Silicon — and, equally
importantly, whether the resulting system still measures what it claims to
measure.

## 1.3 Objectives

1. Port the Boltz inference path to Apple Silicon (Metal Performance Shaders)
   with device-agnostic execution.
2. Reduce activation memory in the pair representation and the target-side
   key-value cache.
3. Reduce sampler cost through classifier-free-guidance distillation and
   speculative integration.
4. Build a distilled surrogate capable of ranking candidate binders on-device.
8. **Validate each of the above against pretrained checkpoints and
   experimentally determined structures, rather than against synthetic tensors.**

Objective 5 is the one that distinguishes this work. It was added after the
component-level results proved far stronger than the end-to-end behaviour they
implied.

## 1.4 Literature Survey

**Structure prediction.** AlphaFold [1] established the evolutionary-coupling
plus structure-module architecture that later systems refine. AlphaFold-Multimer
[2] extended it to complexes and introduced the interface predicted TM-score
(ipTM), a confidence measure derived from the TM-score formalism [18]. AlphaFold
3 [3] generalised to nucleic acids and ligands using a diffusion decoder. Boltz-1
[4] is an open reimplementation of that class of model, and Boltz-2 [5] adds an
affinity head aimed at protein–ligand binding. Multiple sequence alignments,
which dominate input preparation cost, are generated in practice by MMseqs2-based
pipelines [17] as popularised by ColabFold [16].

**Binder design.** Generative approaches such as RFdiffusion [21] and
sequence-design methods such as ProteinMPNN [22] produce candidate binders in
volume. The bottleneck then becomes *ranking*: deciding which candidates merit
experimental follow-up. In practice this ranking is frequently performed with
folding-model confidence, most often ipTM. Section 7 of this report examines
that practice directly.

**Efficiency techniques.** Low-rank factorisation of weight matrices, introduced
for adaptation as LoRA [6], underlies the low-rank pair representation used
here; the tensor-decomposition machinery for projecting an existing dense
operator onto a low-rank form is the CANDECOMP/PARAFAC family surveyed by Kolda
and Bader [14]. Latent key-value caching follows the multi-head latent attention
of DeepSeek-V2 [10]. Sequence-parallel attention with exact numerics follows the
ring-attention construction [11]. On the sampler side, classifier-free guidance
[8] costs two network evaluations per step, which distillation [19] can collapse
to one; speculative decoding [7] and consistency models [9] attack the number of
steps itself. Preference alignment for sequence policies follows DPO [13] and its
reference-free variant SimPO [12].

**Gap addressed.** The efficiency literature reports savings on the operator or
the microbenchmark. What is comparatively rarely reported is whether a
substituted operator remains faithful *on pretrained weights*, and whether the
downstream scoring signal is stable enough to support the decisions it is used
for. This dissertation measures both, using the PDB [15] as the source of
experimentally determined positives — including the MDM2–p53 complex [20] among
others.

## 1.5 Scope and Limitations

The scope is inference, not training: no frontier-scale training run was
performed, and the low-rank layer is therefore evaluated as a substitution onto
released weights rather than as a component of a model trained around it.

Three limitations are stated at the outset because they bound every number in
this report:

1. **Reduced inference settings.** Folding throughout uses 10 sampling steps
   (Boltz default 200), 1 recycling iteration (default 3) and MSA depth 32
   (default 8192). These were forced by CPU-only execution on a laptop.
   **Section 7.13 resolves this confound and finds it was real**: at the
   intended settings the cognate-versus-scramble effect is three to seven times
   larger in standardised terms, and Cohen's *d* moves from 0.12–0.45 to
   1.25–1.52. Every effect size measured in Sections 7.2 to 7.12 should be read
   as a lower bound. The constraint that forced the reduced regime was also
   removed part-way through: on MPS a full-settings fold takes 106 seconds and
   is no slower at full alignment depth than at depth 32.
2. **Panel size.** The binder benchmark uses 22 receptors. This was sized by a
   power analysis, but it remains small relative to the generality of the claim.
3. **Measurement noise.** Folds are unseeded, and Section 7.5 quantifies what
   that costs. Per-complex results are not reproducible at these settings.

## 1.6 Organisation of the Report

Section 2 describes the modules implemented. Section 3 gives the pipeline and
methodology. Section 4 reports component-level specifications and benchmarks.
Section 5 covers design considerations arising from the Apple Silicon port.
Section 6 covers verification and testing, including two defects found in the
test infrastructure itself. Section 7 reports the end-to-end measurements
against pretrained weights, which contains the substantive findings of this
work, and closes with the screening tool those findings were built into
(Section 7.14) and an automated search that fails to improve on them
(Section 7.15). Section 8 draws conclusions and recommendations, and Section 9 sets
out the remaining plan.

<div class="page-break"></div>


# 2. MODULES IN BOLTZ-FAST

The Boltz-Fast engine accelerates structures generation and reduces hardware footprint by implementing targeted algorithmic optimizations. The core architecture is shown in Figure 1 below.

<div class="flowchart-container">
    <div class="flowchart-box primary">
        <h3>1. Memory & Parallelism</h3>
        <p>MLA KV target caching (87.5% memory reduction). Fold-CP sharding and Low-Rank Pair autograd (99.9% VRAM savings).</p>
    </div>
    <div class="flowchart-arrow">➔</div>
    <div class="flowchart-box success">
        <h3>2. Accelerated Denoising</h3>
        <p>CFG Distillation (single-pass student) and Speculative Flow solver (3.85x speedup). Post-diffusion ResNet coordinate refinement.</p>
    </div>
    <div class="flowchart-arrow">➔</div>
    <div class="flowchart-box warning">
        <h3>3. Preference Alignment</h3>
        <p>Reference-free SimPO loss tuning saves 50% VRAM; g-DPO best-vs-all clustering scales linearly O(M).</p>
    </div>
</div>

#### Figure 1: Core Modules in Boltz-Fast

### a) Multi-Head Latent Attention (MLA) Cache
Target receptor representations are static during binder screening. Instead of caching raw keys and values of shape `[L_target, D]`, we down-project them into a compressed latent space:
$$c_{\text{target}} = W_{DK} \cdot h_{\text{target}}$$
where $c_{\text{target}} \in \mathbb{R}^{L_{\text{target}} \times D_{\text{latent}}}$. Keys and values are reconstructed dynamically:
$$K = W_{UK} \cdot c_{\text{target}}, \quad V = W_{UV} \cdot c_{\text{target}}$$
This reduces cached state sizes from $2 \times N \times D$ to $N \times D_{\text{latent}}$, obtaining an **87.5% memory saving** at $D_{\text{latent}} = 32, D = 128$.

### b) Fold-CP Sequence Parallelism
We partition the sequence dimension into shards of size $N/P$ across $P$ GPUs. The ranks compute self-attention locally and pass Key/Value shards along a virtual ring using Online Softmax to maintain exact numerical parity:
$$m_{\text{new}} = \max(m_{\text{old}}, S_{ij})$$
$$d_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot d_{\text{old}} + e^{S_{ij} - m_{\text{new}}}$$

### c) 2D Ring Triangular Multiplication Unit (TMU)
Triangular updates $Z_{ij} = \sum_k (A_{ik} \cdot B_{kj})$ are split over a $P_r \times P_c$ grid. Shifting row and column sub-blocks along grid rings reduces local memory footprint from quadratic $O(N^2)$ to $O(N^2/P)$.

### d) Low-Rank Pair Representation (OPM Replacement) [M3]
Outer Product Mean (OPM) layers project sequence matrices to $N \times N \times D_{\text{pair}}$ spaces, creating an $O(N^2 \cdot D_{\text{pair}})$ memory bottleneck. We implement a low-rank decomposition using `LowRankPairUpdater`:
$$U_{b,i,j,c} = \sum_r X_{b,i,r} Y_{b,j,r} W_{c,r}$$
where sequence projections $X$ and $Y$ have a low rank $r \ll D_{\text{pair}}$. A custom autograd backward pass computes gradients on-the-fly, bypassing quadratic activation storage and preventing Metal Out-Of-Memory (OOM) errors.

The implementation is numerically verified against the reference OPM and
mutation-tested. Its parameters do not correspond to those of a released
checkpoint, so it is selected at runtime by the `BOLTZMAC_OPM` environment
variable, defaulting to the stock implementation. Section 7.1 reports the
measured limits of substituting it into a pretrained model.

### e) Classifier-Free Guidance (CFG) Distillation [M4]
Standard CFG requires evaluating the score network twice per denoising step (conditional and unconditional). We implement a distilled student network `CFGDistilledVectorField` that directly accepts a guidance scale $s$:
$$v_{\text{guided}} = v_{\text{cond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$
During training, we minimize the Huber loss between student predictions and teacher outputs across random scales. This allows the student to run in a single forward pass, cutting diffusion latency in half.

### f) Neural Coordinate Refinement [M5]
Diffusion loops can produce structures with biophysical violations, such as incorrect adjacent $C_\alpha$ bond lengths (ideal: 3.80 Å) and steric clashes. We introduce `ResNetCoordinateRefiner`, a 3-block MLP with LayerNorm and residual updates. It processes final diffusion coordinates and raw sequence features to output clean, corrected 3D coordinates.

### g) Speculative Flow Matching Sampler
Speculative sampling uses a lightweight **Draft model** to project $K$ lookahead integration steps:
$$\hat{x}_{t+1}, \dots, \hat{x}_{t+K}$$
The heavy **Target model** evaluates these states in parallel. If the discrepancy is within $\epsilon$:
$$\|v_{\text{target}} - v_{\text{draft}}\|_2 < \epsilon$$
the steps are accepted, skipping expensive target evaluations.

### h) Reference-Free SimPO Loss Tuning
SimPO directly aligns sequence policies without the need for a frozen reference model, optimizing a length-normalized margin loss:
$$\mathcal{L}_{\text{SimPO}} = - \log \sigma \left( \frac{\beta}{L_w} \log \pi_\theta(y_w|x) - \frac{\beta}{L_l} \log \pi_\theta(y_l|x) - \gamma \right)$$
where $y_w$ and $y_l$ are the preferred and dispreferred sequences, saving **50% VRAM** during fine-tuning.

### i) g-DPO Linear Preference Alignment
Compares candidate sequences using a `"best_vs_all"` strategy inside Union Mask Clusters. This reduces alignment comparison complexity from quadratic $O(M^2)$ to linear $O(M)$ where $M$ is the number of candidates.

<div class="page-break"></div>

# 3. FUNCTIONAL BLOCK DIAGRAM & METHODOLOGY

The system integrates context preparation, memory-efficient representation, speculative sampling, and post-diffusion coordinate refinement into a unified pipeline:

```
                  ┌──────────────────────────────────────────────┐
                  │ 1. CONTEXT & MEMORY OPTIMIZATION             │
                  │    - MLA Latent Cache: Target features       │
                  │    - Low-Rank Pair Updater: 99.9% VRAM save  │
                  │    - Fold-CP / Ring Attention sharding       │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 2. ACCELERATED GENERATIVE DIFFUSION          │
                  │    - CFG Distilled Single-Pass Inference     │
                  │    - Speculative Flow ODE: K lookahead steps │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 3. BIOPHYSICAL STRUCTURAL CLEANUP            │
                  │    - ResNet Coordinate Refiner               │
                  │    - CA-CA bond length projection (3.80 Å)   │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 4. PREFERENCE ALIGNMENT                      │
                  │    - Reference-Free SimPO Margin Loss        │
                  │    - g-DPO Linear Candidate Pairing          │
                  └──────────────────────────────────────────────┘
```

The sequence policies derived from preference alignment are passed to the accelerated folding engine. During structure generation, `AtomDiffusion` resolves the trajectory in a single-pass using the CFG-distilled student model, and applies the neural coordinate refiner to ensure physical structural validity.

<div class="page-break"></div>

# 4. MAJOR TECHNICAL SPECIFICATIONS & BENCHMARKS

### Table 1: Computational Complexity & Scaling Comparison
| Component | Standard Baseline | Implemented Optimization | Time Complexity | Memory / VRAM Scaling |
| :--- | :--- | :--- | :---: | :---: |
| **Inference Integration** | Outer ODE Solver | **Speculative Sampler** | O(N) vs. O(N/K) | O(1) vs. O(K) |
| **Pair Representation** | Quadratic OPM | **Low-Rank Pair Updater** | O(N² · D) | O(N² · r) |
| **Guidance Sampling** | Double-Pass CFG | **CFG Distillation** | O(2 · Steps) | O(Steps) |
| **Coordinate Quality** | Raw Diffusion Output | **Neural Refiner** | O(1) | O(1) |
| **Sequence Parallelism** | Monolithic Attention | **Fold-CP Sequence Parallel** | O(N²) vs. O(N²/P) | O(N²) vs. O(N²/P) |

### Table 2: Speculative Flow Matching Grid Sweep Data
| Lookahead (K) | Tolerance (epsilon) | Evals (Target) | Accept Rate (%) | Speedup | Coordinate L2 Err |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 0.001 | 50 | 0.00% | 0.34x | $2.64 \times 10^{-7}$ |
| **2** | 0.030 | 25 | 100.00% | 2.00x | $5.60 \times 10^{-2}$ |
| **4** | 0.030 | 13 | 100.00% | 3.85x | $8.27 \times 10^{-2}$ |
| **8** | 0.030 | 7 | 100.00% | **7.14x** | $9.60 \times 10^{-2}$ |

### Table 3: Low-Rank Pair Representation Performance Scaling (Rank r=16, D_pair=128)
| Sequence Length (N) | Full-Rank VRAM | Low-Rank VRAM | VRAM Saving Factor | Full-Rank Latency (F/B) | Low-Rank Latency (F/B) | Relative MSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 1.25 MB | 0.01 MB | **104x** | 1.0 / 2.2 ms | 0.7 / 2.2 ms | 84.07% |
| 500 | 30.64 MB | 0.06 MB | **510x** | 7.6 / 32.6 ms | 11.1 / 34.0 ms | 89.10% |
| 1000 | 122.31 MB | 0.12 MB | **1002x** | 29.3 / 147.1 ms | 63.5 / 138.6 ms | 92.98% |
| 1500 | 275.02 MB | 0.18 MB | **1502x** | 112.1 / 307.1 ms | 144.3 / 253.6 ms | 89.61% |

**Reading Table 3 correctly.** The final column is not a rounding error: at
rank 16 the low-rank path reproduces roughly a tenth of the full-rank output.
The memory column and the error column describe the same configuration, so the
saving factor is only meaningful for a network *trained* with the low-rank layer
in place, where the layer defines the function rather than approximating an
existing one. Randomly initialised operands make the two paths trivially
separable, which is why this table is reported as a scaling measurement and not
as a drop-in substitution result. Section 7.1 measures what happens when the
substitution is attempted against pretrained weights.

### Dynamic Quantized Attention & Block-Sparsity
*   **Weight-Only Quantization**: Fixed INT8 quantization retains a **99.997% Cosine Similarity** (MSE: $2.2 \times 10^{-5}$). FIXED INT4 achieves a **99.647% Similarity**. Mixed-precision calibration compresses weights to **5.85 bits** average without loss of accuracy.
*   **Gated Block-Sparse Attention**: Straight-Through Estimator (STE) training drives sparsity from 62.9% active blocks (Epoch 1) to **12.5% active blocks** (Epoch 50). The test reconstruction MSE is **0.000639** against dense attention, showing high fidelity.

### CoreAI Hardware Acceleration & Latency
Surrogate models exported to Apple CoreAI runtimes execute dynamically on the macOS Neural Engine (ANE) without recompilation:

### Table 4: Dynamic Shape Latency Benchmarks on Apple Silicon
| Binder Sequence | Binder L | Target Receptor Type | Target L | Latency (ms) | Output Coordinate Shape | Recompilation Required |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Short Mutant** | 8 | Small Target (Hemoglobin) | 153 | 4.34 ms | `(1, 8, 3)` | **No** |
| **Short Mutant** | 8 | Large Target (Receptor) | 1300 | 26.36 ms | `(1, 8, 3)` | **No** |
| **Insulin Monomer** | 50 | Small Target (Hemoglobin) | 153 | 6.31 ms | `(1, 50, 3)` | **No** |
| **Insulin Monomer** | 50 | Large Target (Receptor) | 1300 | 31.29 ms | `(1, 50, 3)` | **No** |
| **Hemoglobin Frag** | 90 | Large Target (Receptor) | 1300 | 36.58 ms | `(1, 90, 3)` | **No** |

![C-alpha Backbone 3D Coordinate Plot](assets/backbone_3d_plot.png)

#### Figure 2: C-alpha Backbone 3D Coordinate Plot

![Ray-Traced Protein Ribbon Rendering](assets/protein_structure_rendering_1781545391670.jpg)

#### Figure 3: Ray-Traced Protein Ribbon Model Rendering

![ANE-Accelerated 3D Insulin Backbone Visualization](assets/backbone_3d_insulin.png)

#### Figure 4: ANE-Accelerated 3D Insulin Backbone Visualization

<div class="page-break"></div>

# 5. DESIGN CONSIDERATIONS

*   **Apple Silicon MPS Acceleration**: pathed 16 CUDA dependencies. Developed utility functions in `boltz/src/boltz/model/modules/utils.py` for device resolution:
    *   `autocast_device_type(device)`: returns `"cpu"` or `"cuda"`. Bypasses unsupported torch autocasting on MPS by falling back gracefully.
    *   `empty_device_cache(device)`: dispatches cache eviction to `torch.mps.empty_cache()` on MPS or `torch.cuda.empty_cache()` on CUDA.
*   **PyTorch 2.6 Serialization Workaround**: Overrode the strict `weights_only=True` checkpoint loading in PyTorch 2.6 to allow OmegaConf metadata to load properly (`weights_only=False` fallback).
*   **Biophysical Constraints**: Added steric clash resolution and adjacent residue distance penalties ($C_\alpha - C_\alpha$ bond length projection) into the coordinate updater and the neural refiner.

<div class="page-break"></div>

# 6. VERIFICATION & TESTING (M1 & M6)

A key focus of the integration phase was implementing a robust testing framework to ensure correctness and prevent regressions.

We created `tests/test_e2e_suite.py`, containing **53 tests** grouped into 4 execution tiers:
*   **Tier 1: Feature-Specific Functional Correctness (24 tests)**: Validates individual components (Low-Rank, CFG Distilled field, Neural Refiner, MLA Cache, Speculative Sampler) against mock variables.
*   **Tier 2: Boundary & Corner Cases (20 tests)**: Assesses edge inputs (zero guidance, negative guidance, single residue sequences, empty sequences, very large batch shapes, extremely high tolerance values).
*   **Tier 3: Cross-Feature Integration (4 tests)**: Tests the coupled execution of the student model, neural refiner, and speculative sampler inside the `AtomDiffusion.sample()` loop.
*   **Tier 4: Real-World Biological Validation (5 tests)**: Verifies predictions on actual biological binders (Insulin, VEGFA) against predefined criteria.

The E2E test suite executes synchronously via `run_e2e_tests.py`:
```
======================================================================
                Executing Biomolecular Design E2E Test Suite          
======================================================================
...
======================== 53 passed, 1 warning in 4.96s =========================

          SUCCESS: All 53 test cases in the E2E suite passed!         
======================================================================
```

### 6.1 Subsequent Hardening of the Suite

The suite has since grown to **99 tests**, and two defects in the testing
infrastructure itself were found and corrected.

**Continuous integration was not running the tests.** The CI workflow invoked
`unittest discover -s tests`, which collected **zero** tests from a `pytest`-style
suite and therefore reported success unconditionally. CI now runs
`python -m pytest -q` together with a `ruff` lint gate, and test discovery is
configured centrally in `pyproject.toml` rather than by `sys.path` manipulation
inside twelve individual test files.

**An audit of the assertions themselves** (`TEST_AUDIT.md`) found tests that
could not fail: assertions comparing a value to itself, RMSD thresholds loose
enough to be satisfied by random noise, and six tests that checked only output
tensor shape while asserting nothing about output values. These were tightened.
The same audit found that the lightweight surrogate predictor emits C-alpha
traces with ~0.56 Å adjacent spacing against a physical value of 3.80 Å,
recorded as a known limitation of the surrogate rather than silently corrected.

The corrections matter for the interpretation of M6: a 100% pass rate against a
suite that was partly unfailable is weaker evidence than the same rate against
the audited suite, and the latter is what the current 99 tests provide.

![Post-Diffusion Neural Coordinate Refinement Comparison](assets/backbone_3d_refinement.png)

#### Figure 5: Post-Diffusion Neural Coordinate Refinement Comparison

The coordinate refiner successfully resolved simulated clashes (Figure 5, right) and adjusted C-alpha bond lengths to $\le 1.0$ Å deviation from the target value.

<div class="page-break"></div>

# 7. MEASURED RESULTS AGAINST PRETRAINED WEIGHTS

Sections 3 and 5 report component behaviour against mock tensors. This section
reports what the components do against pretrained Boltz checkpoints and
experimentally determined structures. Two questions were pursued to a
conclusion, and both answers are negative.

### 7.1 The Low-Rank Pair Representation Is Not Reachable on Pretrained Weights

The low-rank OuterProductMean (OPM) is the project's strongest technical asset:
it is numerically verified against the reference implementation and
mutation-tested. Its parameters (`low_rank_updater.W`, `proj_x`, `proj_y`) do
not correspond to those of a stock checkpoint (`proj_a`, `proj_b`, `proj_o`), so
it had only ever been usable by training from scratch. Whether the ~97%
activation saving could be obtained on released weights was tested in three
experiments, each stricter than the last.

| Experiment | What it measures | Held-out error @ rank 32 |
| :--- | :--- | :---: |
| CP projection of weights | tensor approximation, random inputs | 0.77 |
| Fit on one target's activations | function approximation, one protein | 0.43 |
| **Corpus distillation, 33 folds** | **function approximation, at capacity** | **0.378** |

Each step was a genuine improvement, and the first two were initially stated
more strongly than the evidence supported: the CP result measured the weight
tensor under `torch.randn` inputs, and the single-target fit measured the
function on one protein only.

**What settles the question is that train and held-out error coincide**
(0.375 / 0.378 at rank 32, against a 0.08 gap for the single-target fit). The
factors genuinely are shared across receptors rather than memorised, so
generalisation is solved — but coincident train and held-out error is the
signature of a model at capacity, and more folding data will not lower the
floor.

**Rank cannot recover the fidelity.** Fitting `err = a·rank^b` to held-out
points gives `1.32·rank^−0.356` and `1.75·rank^−0.312` for the two layers.
Reaching 10% error requires rank ≈ 1,414 and ≈ 9,750 respectively, against
`c_hidden² = 1024` — the width the stock implementation actually materialises.
**The low-rank form therefore costs more activation memory than stock before it
becomes accurate enough to substitute.** The usable regime is 23–38% per-layer
error at 3–13% of stock activations: a large saving at an error that compounds
across the MSA stack.

Controls confirm the negative is a property of the weights rather than of the
solver: an exactly-rank-32 tensor decomposes to 0.00000 error, a random dense
tensor to 0.947, and the trained tensor to 0.831 — only marginally more
compressible than noise.

**Consequence for M3.** The 1502x figure in Table 3 stands as a scaling
measurement for a network trained around the low-rank layer. Obtaining it on
pretrained weights would require retraining the full model — a frontier-scale
run outside the scope of this dissertation. The `BOLTZMAC_OPM` environment
toggle keeps both implementations selectable so the layer remains available for
that work.

### 7.2 Interface Confidence Does Not Rank Peptide Binders

The binder design objective requires a scoring signal to rank candidates.
Boltz's interface confidence (ipTM) was evaluated as that reference against
**11 experimentally determined peptide–domain complexes** from the PDB
(MDM2/p53, SH3/proline-rich, PDZ/C-terminal and others; sequences fetched
programmatically from RCSB, never transcribed). Three classes were folded
identically: **cognate** (receptor with its own peptide), **decoy** (receptor
with a genuine peptide from a different complex — the strongest negative, since
screening is exactly this discrimination), and **scrambled** (own peptide,
order destroyed).

ipTM has **sensitivity**: real complexes score far above the synthetic peptides
used in earlier runs (0.1915 vs 0.0905, AUC 0.881, p = 1e-5). It does not have
**specificity**. Ranking each receptor's own candidates against one another —
which is what screening does — is at chance:

```
cognate ranked #1 for 2/11 receptors    (chance = 1.8)
mean rank 3.27 of 6                     (chance = 3.50)
Wilcoxon vs chance:  p = 0.746
```

A surrogate distilled from a reference that cannot rank binders cannot rank them
either, and this was confirmed directly: on n = 85 held-out complexes the
distilled surrogate achieved Spearman ρ = −0.034, while a **ridge regression on
additive amino-acid composition scored +0.103** (z = 3.44 against the neural
model). An earlier ρ = +0.308 at n = 13 did not survive the larger sample and is
recorded here as a sampling artefact rather than as a result.

### 7.3 Boltz-2 Comparison and a Correction to the Benchmark

The natural objection is that Section 7.2 indicts Boltz-1 rather than ipTM. The
identical 66 pairs were therefore re-folded under Boltz-2 with identical seeds
and byte-identical cached alignments, leaving the checkpoint as the only
variable. Every score rises by roughly 2.8x, and the classes rise together:

| | cognate | decoy | scrambled |
| :--- | :---: | :---: | :---: |
| Boltz-1 | 0.1915 | 0.1684 | 0.1758 |
| Boltz-2 | 0.5355 | 0.4556 | 0.4662 |

Boltz-2 improves the screening metric in every direction — cognate first for
3/11 receptors, mean rank 2.55 of 6 — but does not establish specificity at
n = 11 (Wilcoxon p = 0.094 two-sided; bootstrap 95% CI on the mean rank
[1.36, 2.55] contains chance). The paired improvement over Boltz-1 is +0.45
ranks with 95% CI [−0.36, +1.18], which contains zero.

A pooled cognate-versus-decoy test appears to pass (AUC 0.689, p = 0.033) and is
**not** relied upon: it is confounded by receptor identity, and it is
inconsistent with the observation that **decoys and scrambles are
indistinguishable under Boltz-2 (AUC 0.501, p = 0.993)**. A signal that cannot
separate a genuine binder from a sequence-order scramble is not recognising
binders.

**A labelling defect was found and corrected.** RCSB's FASTA endpoint returns
*canonical* sequence, so phosphoserine reads as S, phosphotyrosine as Y and
acetyl-lysine as K. For complexes whose interaction *is* the modification — an
SH2 domain reading phosphotyrosine, a bromodomain reading acetyl-lysine — the
folded "cognate" pair is not binding-competent, making it a mislabelled
positive. An audit against `entity_poly.pdbx_seq_one_letter_code`, which
preserves modified residues, found **7 of 25 candidate receptors PTM-dependent,
including 1I8H from the original panel** — precisely the receptor whose true
binder ranked last of six under Boltz-1. Ranking a non-binder last is correct
behaviour, not a failure of the metric.

Re-analysis without 1I8H does not change any conclusion (Boltz-1 p 0.790 →
0.463; Boltz-2 p 0.094 → 0.131, both still non-significant), so this corrects
the method rather than the result.

### 7.4 Powered Run: ipTM Tracks Composition, Not Binding — at 10 Sampling Steps

**Scope, established after the fact.** Every fold in this section was taken at
10 sampling steps of an intended 200. Section 7.13 folds the same panel on the
same model and device at the intended settings and finds the cognate-versus-
scramble effect three to seven times larger in standardised terms — ipTM's
Cohen's *d* rises from 0.45 to 1.25, and it separates a cognate from its own
scramble at p < 1e-5. **The conclusion below is a property of the reduced
sampling regime at least as much as of the metric**, and should be read with
that qualifier throughout.

The n = 11 result was underpowered rather than decisive, so a PTM-clean,
tag-free, peptide-deduplicated panel of **22 receptors (132 complexes)** was
assembled programmatically and folded under Boltz-2 — 80% power on the
specificity test requires 21 receptors at dz = 0.57.

**The specificity test passes.** Against decoys, the cognate reaches mean rank
**2.00 against a chance value of 2.50** (first for 8 of 22 receptors,
p = 0.034 two-sided), with a bootstrap 95% CI of **[1.59, 2.41]** that excludes
chance. Taken alone this reverses the n = 11 conclusion.

**The scramble control refutes the binding interpretation.** A scramble
preserves amino-acid composition and length exactly and destroys only sequence
order — and order is what makes a binder a binder.

| class | n | mean ipTM |
| :--- | :---: | :---: |
| cognate | 22 | 0.5015 |
| scrambled | 44 | 0.4888 |
| decoy | 66 | 0.4317 |

Cognates do not measurably beat their own scrambles (mean difference +0.0128,
95% CI [−0.019, +0.044], n = 44), and **scrambles outscore decoys** (AUC 0.632,
p = 0.0096). The variable separating cognate from decoy is therefore largely the
one a cognate shares exactly with its own scramble — composition. The null on
order is a *bound* rather than a zero: any order effect is at most 63% of the
composition effect and is plausibly nil.

The effect is not a length artifact, which was tested directly: ipTM falls with
peptide length (Spearman −0.408, p = 1.2e-6), but regressing the per-pair
cognate-minus-decoy score difference on the corresponding length difference
leaves an intercept of **+0.0752 (p = 0.0001)**. The advantage survives length
adjustment; it is compositional rather than positional.

**Conclusion, as measured here.** The powered run resolves the question and at
these settings the answer remains negative, with a mechanism: ipTM responds to
peptide composition and is indifferent to sequence order. Section 7.13 shows the
indifference is not intrinsic — at 200 sampling steps the same model on the same
panel separates a cognate from its own scramble at *d* = 1.25 — so the claim
that survives is narrower: **ipTM is indifferent to sequence order when the
sampler has not converged**, which Section 7.11 shows is the regime in which the
predicted backbone is only 14% physically plausible. A screening reference must prefer a binder to its
own scramble, and this one does not. The finding converges with Section 7.2 from
an independent direction — a ridge regression on additive composition predicted
ipTM better than the distilled neural surrogate — indicating that the signal
available in this pipeline is compositional throughout.

Reporting only the decoys-only comparison would have produced a positive
headline that the experiment's own control contradicts.

### 7.5 Measurement Reproducibility: Single Folds Do Not Reproduce

Every fold in Sections 7.2–7.4 was unseeded. Boltz's `--seed` defaults to
`None`, and the benchmark's own `--seed` governs only pair construction, not the
diffusion sampler — so each reported ipTM is a single draw from a distribution
whose width had not been measured, while the load-bearing comparison rested on a
difference of +0.0128. A replicate study (4 receptors spanning every observed
outcome, all 6 complexes each, **4 identical re-runs = 96 folds**, at the same
settings) measured it.

```
pooled within-complex SD : 0.0628      median range : 0.1271
  cognate - decoy    +0.0698  = 1.11 x SD   comparable to noise
  cognate - scramble +0.0128  = 0.20 x SD   below noise
```

**Per-receptor rankings are not reproducible.** Cognate rank among its own three
decoys across four identical runs: 1YCR [1,2,1,1], 9F6S [2,1,1,1],
8KDX [3,3,4,2], 6YOO [1,4,4,3]. Four of four flip, and 6YOO spans the whole
range — best to worst on identical input.

**The aggregate effect survives; the significance verdict does not.** A
parametric bootstrap re-simulating the full 132-fold benchmark under the measured
noise (4,000 replications) gives a mean cognate rank of 2.03 with 95% range
**[1.77, 2.27]** — always below the chance value of 2.50 — but **p < 0.05 in only
49% of re-runs** (median p = 0.054, range [0.004, 0.374]). The preference is real
and is probably *stronger* than 2.00 indicates, because measurement error
attenuates rank effects toward chance. What a single run cannot support is the
precision of its own verdict, and Section 7.4's p = 0.034 should be read that way.

**Consequence for method.** Supporting per-receptor claims requires roughly 9–16
replicate folds per complex (SE 0.021–0.016 against competitor gaps near 0.05),
i.e. 1,200–2,100 folds for the panel. That is a GPU-scale workload, which makes
the deferred CUDA phase a prerequisite for the next round rather than a
throughput improvement.

**This generalises beyond Boltz-2 (Section 7.9).** Repeating the identical design
on the few-step-distilled model leaves ipTM stable for 0 of 4 receptors and
interface pLDDT for only 1 of 4, so the instability is a property of these
folding models rather than of the reduced-sampling regime — which strengthens
this section rather than qualifying it. Section 7.9 also shows the recommendation
needs splitting: averaging is required for per-receptor claims, but does not
rescue a weak metric in aggregate, because the variance decomposition finds
aggregate discriminability signal-limited rather than noise-limited.

The observation generalises beyond this project: ranking candidate binders from a
*single* AlphaFold or Boltz run is common practice, and at reduced sampling
settings a single run does not reproduce its own ranking.

<div class="page-break"></div>


### 7.6 Interface pLDDT Recovers What ipTM Discards — On Structures the Model Has Seen

Sections 7.2 to 7.5 indict **ipTM**, not the predicted structures — 132 of which
remained on disk. Re-scoring them with other interface measures required no
further folding.

The result below is real and reproduces. Its scope, however, is narrower than
this section originally claimed, in two independent directions:

* **Section 7.10** shows that most of the advantage disappears on complexes
  released after the model's training cutoff. This section characterises the
  readouts on a panel that is largely training data.
* **Section 7.13** shows that every effect here was measured at 10 sampling
  steps of an intended 200, and is three to seven times smaller than the same
  measurement at full settings. Interface pLDDT's Cohen's *d* on the scramble
  control rises from 0.28 to 1.52. **Every effect size in this section is a
  lower bound.**

The comparison *between* readouts is unaffected — all six were scored on the
same folds — so the ranking of metrics stands even though their magnitudes do
not.

Six measures were computed from the same complexes and put through the identical
tests. Contacts use a 8 Å CB–CB criterion (CA for glycine); pDockQ follows
Bryant et al.; buried area is Shrake–Rupley.

| Metric | cognate | scrambled | decoy | cognate vs own scramble | mean rank | chance |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ipTM | 0.502 | 0.489 | 0.432 | p = 0.416 | 2.00 | 2.50 |
| pDockQ | 0.474 | 0.466 | 0.404 | p = 0.797 | 2.14 | 2.50 |
| **Interface pLDDT** | **49.60** | **46.31** | **45.93** | **p < 0.0001** | **1.91** | 2.50 |
| Inter-chain contacts | 32.9 | 38.5 | 34.0 | p = 0.054 | 2.55 | 2.50 |
| Contact density | 3.24 | 3.69 | 3.62 | p = 0.122 | 2.55 | 2.50 |
| Buried surface area | 1800 | 1861 | 1632 | p = 0.454 | 2.27 | 2.50 |

The decisive column is the comparison against a cognate's **own** scramble,
which fixes composition and length and destroys only order. **Only interface
pLDDT passes it.**

The behaviour of pDockQ is instructive rather than incidental. It multiplies
interface pLDDT by a contact term, and the contact term runs the wrong way in
this regime: scrambled peptides make *more* inter-chain contacts than cognates
(38.5 against 32.9). Combining the two cancels the signal that interface pLDDT
carries on its own.

**The cause is not short peptides — it is unconverged geometry.** Section 7.11
shows that only 14% of backbone bonds in these Boltz-2 structures are physically
plausible against 96% for a few-step-distilled model, and that on converged
structures the contact ordering reverses to the sensible direction (cognates
61.4, scrambles 51.8). Every contact-derived entry
in the table above — inter-chain contacts, contact density, buried surface area,
pDockQ — is computed from CB–CB distances on a backbone that is not connected,
and should be read as describing point clouds rather than complexes. Replacing
pDockQ's contact term with a PAE-derived one (pDockQ2) repairs it on the same
structures, from p = 0.797 to p = 0.00026. The ipTM and interface-pLDDT rows are
unaffected: both come from the confidence head rather than the coordinates.

**The result survives the reproducibility test that demoted ipTM.** The 96
replicate folds of Section 7.5 were re-scored the same way to obtain a
run-to-run standard deviation for each new measure:

| Metric | effect on the scramble control | run-to-run SD | effect / noise |
| :--- | ---: | ---: | ---: |
| ipTM | +0.0128 | 0.0628 | 0.20x |
| pDockQ | +0.008 | 0.1498 | 0.05x |
| **Interface pLDDT** | **+3.30** | **1.917** | **1.72x** |

Interface pLDDT therefore carries roughly **8.6 times** the effect-to-noise ratio
of ipTM on the test that matters. Simulating the full benchmark under the
measured noise gives a cognate-minus-scramble effect of **+3.30 pLDDT, 95% CI
[+2.13, +4.47]** that reproduces at p < 0.05 in **100%** of re-runs, and a
within-receptor rank of 1.91 (95% CI [1.64, 2.14]) reproducing in **84%** —
against 49% for ipTM.

![Only interface pLDDT separates a binder from its own scramble](assets/fig_metric_comparison.png)

#### Figure 6: ipTM and Interface pLDDT on the Scramble Control

**What is and is not established.** Order sensitivity is established: interface
pLDDT distinguishes a peptide from its own scramble at full reproducibility, and
clears a Bonferroni correction for the six measures tested (α = 0.0083).
Receptor specificity is suggestive but not established: the within-receptor rank
test gives p = 0.027, which does not clear that threshold, although it
reproduces in 84% of simulated re-runs. More receptors would settle it.

Interface pLDDT is read from the same confidence head as ipTM, so this is a
better *readout* of one model rather than an independent second opinion.

#### 7.6.1 Two corrections, in opposite directions

**Four of the twenty-two cognate pairs were not binders.** The benchmark folds
two sequences, which is faithful only if the crystallised peptide *is* its
canonical sequence. For four members it is not (Section 7.10.1). 1NLO is the
worst: five of eleven positions are synthetic groups rather than amino acids, so
what SH3 binds in that crystal is a small molecule while what was folded is the
string `XXXXPLPPLPX`. 9GRF's peptide is O-glycosylated and its receptor reads
the glycan. A non-binding cognate scores like a scramble and therefore *dilutes*
the contrast it belongs to, so removing these members should raise every effect
— which it does, in 9 of 9 arm-metric cells. Boltz-2's interface pLDDT goes from
+3.30 to +3.47.

**The earlier claim that this result is model-dependent does not survive that
correction.** Section 7.8 reported that interface pLDDT fails on stock Boltz-1
(+1.54, p = 0.067). With the two unfoldable members removed it is p = 0.013, and
removing the capped peptides as well gives p = 0.001; the receptor-side term
moves from p = 0.428 to p = 0.012. Boltz-1 does show the effect. That claim was
substantially an artefact of two non-binders in a 22-receptor panel, and it is
withdrawn rather than annotated.

**The correction that does hold is a larger one.** Both of the above are
in-training comparisons. Section 7.10 folds a second panel of 22 receptors
released *after* the training cutoff and finds the interface-pLDDT effect
weakens from +12.03 to +5.24, and ipTM's from +0.265 to +0.137 — both roughly
halve. The readout recommended here remains the best single one measured, but
the figures in this section describe a panel that is largely training data and
should be read as an upper bound on what a novel target would give.

<div class="page-break"></div>

### 7.7 Localising the Interface-pLDDT Signal

Section 7.6 establishes that interface pLDDT responds to sequence order. It does
not establish *what* it is responding to.

Two scope notes carry into everything below. These folds are at 10 sampling
steps, so the magnitudes are lower bounds (Section 7.13), and the panel is
largely training data, so the receptor-side result in particular does not
transfer (Section 7.10). The receptor side is the component that degrades most
on held-out complexes and gains most at full settings — 0.12 to 1.45 in Cohen's
*d* — which is a warning that it is the most regime-sensitive quantity measured
in this work. Two further experiments address the
most natural objection: a scrambled peptide is often more disordered, so the
result could reflect peptide foldability with no binding information in it.

#### 7.7.1 Which side of the interface carries the signal

Interface pLDDT averages over residues on both chains, so the two readings
separate. This costs no further folding — it re-reads the structures of
Section 7.4.

| Quantity | cognate | scrambled | decoy | cognate − own scramble | p |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Pooled interface pLDDT | 49.60 | 46.31 | 45.93 | +3.30 | 0.00000 |
| **Receptor side** | **51.94** | **49.56** | **49.41** | **+2.38** | **0.00006** |
| Peptide side | 44.37 | 39.12 | 39.16 | +5.25 | 0.00001 |
| Peptide, whole chain | 44.08 | 39.21 | 39.13 | +4.87 | 0.00003 |

**The objection is half correct.** Whole-chain peptide pLDDT moves by +4.87,
almost as much as the peptide-side interface value of +5.25. Most of the
peptide-side component is therefore not about the interface at all: it is the
peptide being placed more confidently along its whole length. Section 7.6 did
not separate these, and to that extent overstated what the pooled figure means.

**The receptor side survives the objection.** The receptor's own residues are
placed more confidently when the cognate peptide is present (+2.38, p = 6e-5).
A more disordered peptide cannot make the receptor's residues more certain, so
peptide foldability does not account for this term.

That argument is sound and the measurement holds on this panel. What it does not
establish is that the effect is about *binding* rather than about recognition of
a complex the model was trained on: on the held-out panel of Section 7.10 the
receptor-side term falls from +7.38 to +2.57, and the interaction is significant
on both scales (p = 0.013 raw, p = 0.0099 within-receptor). The foldability
objection is answered; a retrieval objection takes its place.

Taken alone the receptor side ranks the cognate against decoys at p = 0.086 —
separating a peptide from its own scramble but again not establishing receptor
specificity.

#### 7.7.2 Is the response tied to the binding site?

If the receptor responds to the correct peptide, the response should depend on
the binding site. Four arms were folded with the same cognate peptide, all in
single-sequence mode so that MSA presence is not a confound. The site is defined
as the 15 receptor residues nearest the peptide (mean 13.4): an 8 Å cutoff marks
roughly half of a small domain as interface, and alanine-substituting that many
residues destroys the fold, reintroducing the confound the control exists to
remove.

| Arm | receptor side | vs real | p |
| :--- | ---: | ---: | ---: |
| Real receptor | 45.70 | — | — |
| Interface → alanine | 36.77 | −8.92 | 0.00001 |
| Surface → alanine (control) | 38.62 | −7.08 | 0.00048 |
| Scrambled receptor | 34.03 | −11.67 | 0.00000 |

The decisive comparison is interface-alanine against surface-alanine, since both
mutate the same number of residues:

```
difference   -1.84    95% CI [-5.05, +1.36]    p = 0.244
paired dz     0.255   ->  80% power requires n = 123 receptors
```

Mutating the binding site costs more than mutating an equal number of exposed
residues elsewhere, but not detectably so. **This is an underpowered result
rather than a negative one**: the direction is correct and resolving it would
need roughly six times the present panel.

Two further caveats bound the interpretation. Single-sequence folding lowers the
baseline from 51.94 to 45.70, leaving less headroom; and every perturbation costs
7–12 pLDDT, so both arms may be compressed near a floor that would mask a real
difference. The scrambled-receptor arm falls furthest, as expected, but is
uninformative alone because shuffling destroys the fold as well as the site.

#### 7.7.3 Summary

**Established.** On this panel the receptor responds to which peptide it is
given, and peptide foldability cannot explain that.

**Not established.** That the response is localised to the binding site
(p = 0.244, requiring roughly six times the panel).

**Not established, and newly in doubt.** That the response is about binding at
all rather than about having seen the complex. The receptor-side term is the
component of interface pLDDT that degrades *most* on held-out structures
(Section 7.10) — it is both the part best defended against the foldability
objection and the part least likely to transfer.

One component of the pooled metric is now known to be largely peptide
foldability, and another is now known to be largely retrieval. What remains is
smaller than this section originally implied.

<div class="page-break"></div>

### 7.8 Few-Step Distillation, and What It Reveals About the Negatives

Every result in Sections 7.2 to 7.7 was folded at 10 sampling steps from models
whose default is 200 (Section 1.5). The obvious objection is that the failures
are artefacts of under-sampling rather than properties of the metrics. Testing it
requires a model that is *good* at few steps.

DeCAF-Boltz distils Boltz-1 into a few-step generator [23]. Running the identical
132 pairs through it — and through stock Boltz-1 as a de-confounding arm, since
DeCAF changes both the base model and the sampling regime — answers the
objection, though not in the direction anticipated.

| Arm | Base | Trained for 10 steps | Device |
| :--- | :--- | :---: | :--- |
| Boltz-2 | Boltz-2 | no | CPU |
| Boltz-1 | Boltz-1 | no | MPS |
| DeCAF-Boltz | Boltz-1 | **yes** | MPS |

DeCAF and Boltz-1 share a base, a device and a step count, so that pair isolates
few-step training.

**What Section 7.13 adds.** That isolation is sound and the 5–6× figure stands,
because both arms are folded at the same reduced settings. What changes is the
interpretation. Folding stock Boltz-1 at its intended 200 sampling steps reaches
+0.287 on ipTM and +11.85 on interface pLDDT, against DeCAF's +0.201 and +9.54
at ten steps. Distillation is therefore **not supplying something the stock
model lacks — it recovers most of what the stock model already has when run
properly, at a twentieth of the sampling budget.** For a dissertation about
efficiency that is the stronger claim, and it is the one the evidence supports:
the comparison below measures how much of full-settings behaviour survives a
20× cut in sampling, not a new capability.

**Order sensitivity — cognate minus its own scramble:**

| Metric | Boltz-2 | Boltz-1 | DeCAF |
| :--- | ---: | ---: | ---: |
| ipTM | +0.013 (p = 0.42) | +0.039 (p = 0.0043) | **+0.201 (p = 2e-5)** |
| Interface pLDDT | +3.30 (p < 1e-5) | +1.54 (p = 0.067) | **+9.54 (p < 1e-5)** |
| Receptor side | +2.38 (p = 6e-5) | +0.71 (p = 0.43) | **+5.95 (p = 1e-5)** |

**Receptor specificity — cognate ranked against its own decoys, chance 2.50:**

| Metric | Boltz-2 | Boltz-1 | DeCAF |
| :--- | ---: | ---: | ---: |
| ipTM | 2.00 (p = 0.034) | 1.86 (p = 0.017) | **1.77 (p = 0.0087)** |
| Interface pLDDT | 1.91 (p = 0.027) | 1.91 (p = 0.010) | **1.73 (p = 0.0042)** |
| Receptor side | — | 2.05 (p = 0.054) | **1.77 (p = 0.0032)** |

On DeCAF, interface pLDDT and receptor side clear a Bonferroni threshold of
0.0083 for six metrics. **This is the first point in this dissertation at which
receptor specificity is established rather than merely suggested**, and every
bootstrap interval excludes chance (interface pLDDT [1.32, 2.18]).

#### 7.8.1 What the de-confounding shows

**Few-step training accounts for the gain.** Measured against its own teacher on
the same device at the same step count, DeCAF delivers 5–6× larger effects. The
base model is not the explanation.

**Two findings complicate the simple version, and neither is discarded.**

*Stock models are not signal-free.* Boltz-1 at 10 steps separates cognate from
scramble on ipTM (p = 0.0043) and ranks above chance (p = 0.017). Reduced
sampling attenuates the signal rather than destroying it. Only Boltz-2's ipTM was
genuinely flat, so Section 7.4's conclusion should be read as a statement about
that model rather than about ipTM in general.

*The apparent model-dependence was a panel defect — withdrawn.* This section
originally reported that interface pLDDT reaches significance on Boltz-2 and
DeCAF but not on Boltz-1 (+1.54, p = 0.067), and concluded that which readout
carries the signal varies by model. Section 7.10.1 shows the Boltz-1 column was
being diluted by two cognate pairs that are not binders. Removing them:

| Boltz-1 | all 22 | minus 1NLO, 9GRF | minus those + capped |
| :--- | ---: | ---: | ---: |
| Interface pLDDT | +1.54 (p = 0.067) | +2.18 (p = 0.013) | +2.95 (p = 0.001) |
| Receptor side | +0.71 (p = 0.428) | +1.41 (p = 0.125) | +2.32 (p = 0.012) |

Both halves move together and the receptor-side term moves furthest. Boltz-1
behaves like the other two arms once non-binders are removed, and the claim is
withdrawn.

#### 7.8.2 Revised position

An earlier reading of the DeCAF result — that the settings gap explains the
negatives — was too simple and is not adopted. The three arms support something
narrower:

1. Few-step distillation substantially improves both order sensitivity and
   receptor specificity, by 5–6× over its own teacher at the same budget in raw
   effect size. **Section 7.9 qualifies what that buys**: DeCAF is also ~1.6×
   noisier, so once its own sampling noise is measured rather than assumed, the
   advantage in discriminability is 2.2–2.6×, not 5–6×.
2. Stock models at reduced sampling retain weak but real signal, not none.
3. ~~Which readout carries the signal varies by model.~~ Withdrawn above: the
   three arms agree once the panel is clean. **Section 7.10 replaces this with a
   sharper statement** — which readout carries the signal depends not on the
   model but on whether the complex was in its training set.

#### 7.8.3 Limitations

The Boltz-2 arm ran on CPU while the others ran on MPS. Section 7.5's device
check established ipTM equivalence within the noise margin but left interface
pLDDT unresolved (TOST p = 0.168, underpowered at n = 6), so comparisons
involving the Boltz-2 column carry that caveat; the DeCAF-to-Boltz-1 contrast
does not. Folds are single and unseeded, and DeCAF's own run-to-run variance is
unmeasured. The panel remains 22 receptors, with bootstrap upper bounds reaching
2.18 to 2.23.

The DeCAF fork reverts to the teacher sampler if it does not recognise the
distilled head, which would yield plausible numbers from the wrong model. Each
batch asserts the confirmation string in its log and refuses to score otherwise.

<div class="page-break"></div>

### 7.9 Variance Decomposition and the Reproducibility of the Few-Step Model

Two analyses that correct claims made earlier in this section.

#### 7.9.1 The rank tests were discarding most of the data

Every test in Sections 7.4 to 7.8 collapses six folds per receptor into a single
integer rank. That protects against receptor-level baseline variation — the
nuisance the within-receptor design exists to remove — but discards magnitude,
which is why results sat near p = 0.03 at n = 22.

A linear mixed model with receptor as a random effect,

    s(R,P) = μ + α_R + β·cognate + γ·scramble + δ_RP + ε,

gives the same protection without the loss.

| Arm | Metric | Rank-test p | Mixed-model p |
| :--- | :--- | ---: | ---: |
| Boltz-2 | ipTM | 0.034 | **0.0042** |
| Boltz-2 | Interface pLDDT | 0.027 | **0.00005** |
| Boltz-1 | ipTM | 0.017 | **0.00009** |
| DeCAF | Interface pLDDT | 0.0042 | **< 1e-5** |

Roughly an order of magnitude, from the estimator alone. Several results
described above as suggestive but not significant were limited by the analysis
rather than by the evidence.

#### 7.9.2 A ceiling on any score from these models

Splitting residual variance into the receptor–peptide interaction δ (the binding
signal) and fold-to-fold sampling noise ε measured from replicates:

| Arm | Metric | σ²_receptor | σ²_interaction | σ²_noise | Signal/noise |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Boltz-2 | ipTM | 0.0054 | 0.0059 | 0.0039 | 1.48 |
| Boltz-2 | Interface pLDDT | 14.46 | 10.01 | 3.68 | 2.72 |
| Boltz-1 | ipTM | 0.0023 | 0.0002 | 0.0039 | **0.05** |
| Boltz-1 | Interface pLDDT | 13.48 | 9.90 | 3.68 | 2.69 |
| DeCAF | ipTM | 0.0157 | 0.0366 | 0.0096 | 3.80 |
| DeCAF | Interface pLDDT | 85.84 | 55.82 | 9.17 | **6.08** |

The ratio bounds *any* score computed from that model, which makes two points
the rank tests could not.

**Boltz-1's ipTM is noise-limited (0.05); every other cell is signal-limited.**
Its weak result is a different failure from Boltz-2's — one signal is drowned,
the other is nearly absent — and Section 7.8 treats the two as one phenomenon.

**Averaging is not a general remedy.** Every other cell exceeds 1, so noise is
not the binding constraint on aggregate discriminability.

#### 7.9.3 The few-step model is noisier, not cleaner

The decomposition above initially applied Boltz-2's noise term to the DeCAF arm
for want of a measurement, on the expectation that a model distilled to land
accurately in ten steps would be less stochastic. **The opposite holds.**
Repeating Section 7.5's design exactly — same 24 complexes, same 4 receptors,
4 identical unseeded re-runs:

| Metric | DeCAF SD | Boltz-2 SD | Ratio |
| :--- | ---: | ---: | ---: |
| ipTM | 0.0981 | 0.0628 | **1.56** |
| Interface pLDDT | 3.029 | 1.917 | **1.58** |

Fewer, larger jumps move further per draw, so run-to-run spread grows. Correcting
the decomposition accordingly:

| Metric | S/N, borrowed noise | S/N, measured | Advantage over Boltz-2 |
| :--- | ---: | ---: | :--- |
| ipTM | 10.71 | **3.80** | 2.6× (not 7.2×) |
| Interface pLDDT | 16.68 | **6.08** | 2.2× (not 6.1×) |

DeCAF remains the strongest arm by a clear margin, but by roughly half what the
borrowed figure implied. The raw effect sizes of Section 7.8 are unaffected —
they are measured, not inferred; what changes is the discriminability those
effects purchase once the model's own noise is accounted for.

#### 7.9.4 Rank instability generalises

Cognate rank among its own decoys, across four identical re-runs:

```
ipTM             stable 0/4    1YCR [1,2,1,1]  6YOO [2,2,2,1]
                               8KDX [4,2,3,2]  9F6S [2,3,3,3]
interface pLDDT  stable 1/4    1YCR [1,3,1,1]  6YOO [2,2,2,2]
                               8KDX [4,4,3,3]  9F6S [2,3,2,2]
```

Single unseeded folds do not reproduce their own per-receptor rankings on the
few-step model either, so Section 7.5's finding describes these folding models
generally rather than the reduced-sampling regime.

This does not contradict the signal-limited verdict of 7.9.2. Aggregate
discriminability is not noise-bound, yet individual within-receptor comparisons
are close enough that a single draw reorders them. The recommendation therefore
splits: **average for per-receptor claims; averaging will not rescue a weak
metric in aggregate.**

**Revised on held-out data.** That split was measured where the signal is
strongest. On the held-out panel the effects are roughly halved, so sampling
noise is a proportionally larger share, and averaging helps in aggregate too:
three draws take within-receptor AUC from 0.616 to 0.735, a gain of +0.119 with
a bootstrap interval of [+0.010, +0.226]. It is the only intervention tested in
this work whose interval excludes zero — against +0.017 for a cross-model
ensemble, −0.033 for physics rescoring, and −0.13 for combining thirteen
readouts. Which statistic is used barely matters: mean, median and best-of-three
differ by less than the bootstrap spread, and penalising draw-to-draw
disagreement actively hurts (0.735 to 0.676), because decoys are *more* stable
across draws than cognates.

#### 7.9.5 Limitations

Boltz-1 has no replicate study of its own and borrows the Boltz-2 noise term, so
its rows are an assumption rather than a measurement. The replicate design covers
4 receptors, chosen in Section 7.5 to span the observed outcomes, not the full
panel. The mixed model assumes additive receptor effects and homoscedastic
residuals; neither was tested.

<div class="page-break"></div>

### 7.10 Is the Panel Measuring Prediction or Retrieval?

**Both panels in this section were folded at 10 sampling steps.** Section 7.13
shows that regime suppresses the effect three- to sevenfold, so the absolute
numbers below are lower bounds on both sides of the comparison. The *contrast*
between in-training and held-out is unaffected — settings are identical across
it — which is what this section measures.

Every complex tested so far is a PDB entry, and Boltz-1 — which DeCAF distils —
was trained on entries released before **2021-09-30**, the same cutoff as
AlphaFold3. So "interface pLDDT is high for cognates" may mean "the model has
seen this complex". This section asks two questions the preceding sections
assumed away: whether the cognate pairs are binders at all, and whether the
results survive on structures released after the cutoff.

#### 7.10.1 Four of the twenty-two cognate pairs were not binders

RCSB FASTA returns canonical sequence, and returns `X` for any component that is
not one of the twenty amino acids. Neither is visible downstream: the FASTA
parses, the fold succeeds, the confidence scores look ordinary.

| PDB | folded as | actually crystallised |
| :--- | :--- | :--- |
| 1NLO | `XXXXPLPPLPX` | ACE, MN1, MN2, MN7, NH2 — a designed synthetic ligand |
| 2GBQ | `XVPPPVPPRRRX` | acetyl and amide terminal caps |
| 1SEM | `XPPPVPPRRR` | acetyl cap |
| 9GRF | `AASTTTPAPA` | O-glycosylated at Ser3 and Thr4 |

**1NLO is not a peptide.** Five of eleven positions are 4-carboxypiperidine,
benzene derivatives and terminal caps. **9GRF is the subtle one:** StcE is a
mucin-selective protease that recognises the O-glycan, so the bare backbone is
not a substrate. This is the defect that removed 1I8H (Section 6.3), whose
peptide needs phosphothreonine and whose "true binder ranked last of six" was
correct behaviour on a non-binder.

**Why the PTM audit missed them.** That audit tests each peptide against a fixed
list of PTM codes, which cannot catch a modification not on the list. The
held-out panel turned up **7F3S**, a histone H3 tail carrying benzoyl-lysine read
by a bromodomain-containing receptor — precisely the 1I8H failure, and the
allowlist passed it. Asking a different question fixes this: instead of *"is this
one of the modifications I know about?"*, ask *"is anything bonded to this
chain?"* — one `covale` scan over the mmCIF, with no list to maintain. It
recovers all four, and correctly clears 1ELW, 6YOO and 7S7J, whose nickel, zinc
and calcium are not bonded to the peptide. A metal at a lattice contact does not
change whether a peptide binds; a sugar on its own serine does.

**A prediction, fixed in advance.** A cognate that cannot bind scores like a
scramble and dilutes the contrast it belongs to, so removing these members must
*raise* every effect. The exclusions were chosen by whether a sequence can be
folded faithfully, never by inspecting a p-value. **The effect grew in 9 of 9
arm-metric cells** — the flags come from structural annotation, the effects from
folds run weeks earlier.

The rank tests move the other way, and reporting only the effects would be
selective: DeCAF's interface-pLDDT rank test goes from p = 0.0042 to p = 0.0235.
The mean ranks barely move (1.73 to 1.83 against chance 2.50); what changes is
that the Wilcoxon loses four of twenty-two samples. That is a power loss, not a
weaker effect, and it is the limitation Section 7.9.1 already documents.

#### 7.10.2 A panel of complexes the model was not trained on

Of the main panel only 6 receptors postdate the cutoff. A second panel of **22
receptors released after it** was screened identically — PTM-clean, tag-free,
receptor and peptide deduplicated — with decoys drawn from within the held-out
set, so no fold in the comparison involves a training structure. 132 folds on
DeCAF at the same 10 sampling steps and 1 recycling step.

**These numbers are the mean of five independent draws**, for a reason recorded
in 7.10.3: the first draw alone was misleading, and so was the second.

| Metric | in-training (16 receptors) | held-out (22 receptors) |
| :--- | ---: | ---: |
| ipTM | +0.265 (p = 1e-5) | +0.137 (p = 5e-5) |
| Interface pLDDT | +12.03 (p < 1e-5) | +5.24 (p = 0.0001) |
| Receptor side | +7.38 (p = 2e-5) | +3.16 (p = 0.0002) |

Cognate ranked against its own decoys, chance 2.50:

| Metric | in-training | held-out |
| :--- | ---: | ---: |
| ipTM | 1.77 (p = 0.0087) | 1.59 (p = 0.0004), 13 of 22 first |
| Interface pLDDT | 1.73 (p = 0.0042) | 1.82 (p = 0.0090), 12 of 22 first |

Every effect is smaller held out, by roughly half in every case: interface pLDDT
retains 44% of its in-training effect, ipTM 52% and the receptor side 43%.
Neither readout survives markedly better than the other.

Comparing two p-values is not a test. A mixed model with receptor as a random
effect gives the interaction directly, fitted on raw scores and on scores
z-scored within receptor so the answer does not depend on units:

| Metric | raw | within-receptor z |
| :--- | ---: | ---: |
| ipTM | −0.128 (p = 0.048) | +0.021 (p = 0.941) |
| Interface pLDDT | −6.79 (p = 0.0070) | −0.503 (p = 0.066) |
| Receptor side | −4.22 (p = 0.016) | −0.438 (p = 0.134) |

**On the scale-free measure nothing degrades significantly.** The raw column
says all three weaken held out; the z-scored column says that could be a scale
effect, since the held-out panel has smaller within-receptor spread (5.65
against 8.64) and raw differences shrink with it. ipTM's raw value sits on the
threshold at p = 0.048 and should not be read as a category change from the
p = 0.052 that three draws gave.

Two draws gave p = 0.0029 for interface pLDDT on the z scale, three gave
p = 0.058 and five give p = 0.066 — the earlier version of this section
overstated a result that every subsequent draw declined to support.

**What this panel can and cannot establish.** It can establish that both
readouts weaken substantially on complexes the model was not trained on. It
cannot establish that the weakening is specific to interface pLDDT rather than
general, because that claim is significant on one scale and not the other, and
because the rank ordering of the two metrics reversed between draws (7.10.3).
At 22 receptors with the sampling noise measured here, the panel does not
support a fine distinction between two readouts. That is a statement about
power, and Section 7.10.6 gives what would be needed to settle it.

#### 7.10.3 One draw was not enough, and this section originally used one

The held-out panel was first written up from a single set of folds. Folds are
unseeded, so re-running the identical panel at identical settings gives an
independent draw. Five were run, and the first two each produced a conclusion
that a later draw withdrew.

**Order sensitivity, cognate minus its own scramble:**

| Metric | draw 1 | draw 2 | draw 3 | draw 4 | draw 5 | mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ipTM | +0.110 | +0.162 | +0.138 | +0.152 | +0.120 | **+0.137** |
| **Interface pLDDT** | **+2.76** | +6.37 | +5.85 | +6.46 | +4.76 | **+5.24** |
| Receptor side | +1.39 | +3.75 | +3.59 | +4.21 | +2.89 | **+3.16** |

Draw 1 is the outlier on every row — the only one where interface pLDDT fails to
reach significance (p = 0.089 against 6e-4 to 8e-5 for the other four). Read
alone it said the effect *collapses* held out; five draws say it weakens to 44%
of its in-training value and remains significant at p = 0.0001.

**Receptor specificity, cognate ranked among its own decoys (chance 2.50):**

| Metric | draw 1 | draw 2 | draw 3 | draw 4 | draw 5 | mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ipTM | 1.64 | 1.50 | **1.86** | 1.64 | 1.45 | **1.59** |
| Interface pLDDT | 2.09 | 2.05 | **1.73** | 1.73 | 1.91 | **1.82** |

**Draw 3 reverses the ordering.** After two draws this section claimed the rank
test was the stable one and that interface pLDDT reproducibly failed receptor
specificity while ipTM reproducibly passed. In draw 3 interface pLDDT scores
1.73 and beats ipTM's 1.86 — the one draw in five where that happens.

Averaged over five draws the two are 1.59 and 1.82, both significant, and their
difference is smaller than the range either metric covers across draws (ipTM
1.45–1.86; interface pLDDT 1.73–2.09). ipTM is ahead on average and the panel
cannot resolve whether that is real.

So the honest reading is neither of the first two. **Both readouts weaken held
out by roughly half; neither can be shown to weaken more than the other on this
panel.** The claim that ipTM specifically survives was made on a rank difference
this panel is not powered to resolve.

This is Section 7.5 turned on this dissertation's own headline, twice. Section
8.2 recommends replicate averaging precisely to prevent it, and the
recommendation was written before it was followed. The methodological lesson is
sharper than the scientific one: a single unseeded fold is not a measurement,
and neither, on this evidence, are two.

#### 7.10.4 Confounds that do not explain it

**Peptide length.** Held-out peptides are longer (12.8 against 10.3 aa,
p = 0.10). This cannot matter, and is reported as a non-control rather than a
passed one: the effect is a within-receptor contrast between a peptide and a
*permutation of itself*, so length and composition are equal by construction.
Length is also constant within receptor, so the random intercept absorbs it —
adding it as a covariate leaves β and p unchanged to five decimals.

**A ceiling compressing the held-out differences.** The opposite holds. Cognate
ipTM is 0.631 in training against 0.419 held out; interface pLDDT is 81.57
against 68.42. The model is markedly *less* confident on structures it has not
seen, which is itself the signature the experiment was built to detect.
Within-receptor spread is also smaller on the held-out panel (5.65 against
8.64), which favours the held-out effect after z-scoring — and the collapse
survives it.

**Settings and alignments.** Both panels: DeCAF, 10 sampling steps, 1 recycling
step. The six receptors common to both reuse the main panel's cached alignments
byte-for-byte, so the shared members cannot contribute an MSA difference.

**Panel integrity.** Removing the three held-out members with components bonded
to the peptide *strengthens* the conclusion: ipTM reaches +0.123 (p = 0.0043)
with a rank of 1.26 against chance 2.21 (p = 0.0003, 14 of 19 first), while
interface pLDDT stays marginal at p = 0.057.

#### 7.10.5 What this establishes

**Both readouts lose roughly half their effect on complexes the model was not
trained on**, and the loss is large enough to matter: interface pLDDT falls from
+12.03 to +5.24, ipTM from +0.265 to +0.137. Sections 7.6 to 7.8 characterise
the readouts on a panel that is largely training data, and the numbers there
should be read as an upper bound on what a novel target would give.

**What it does not establish** is that either readout degrades more than the
other. That claim was made twice in earlier versions of this section, in
opposite directions, and neither survived the next draw.

The practical consequence is unchanged by the retraction: a screening figure
quoted without stating whether the complexes were in the model's training set
is roughly a factor of two optimistic.

#### 7.10.6 Limitations

Two panels of 16 and 22 receptors, five draws on the held-out side. The
held-out effects are small in absolute terms (+0.137 and +5.24), and the central
question — whether the degradation is metric-specific — is significant on the
raw scale and not on the scale-free one. Settling it needs more receptors rather
than more draws: the draw-to-draw spread is now well characterised and the
receptor-to-receptor spread is what limits the comparison.

The panels differ in receptor identity as well as in training status; only
folding the same complexes on a model trained with and without them would
isolate contamination completely, which would require retraining. The cutoff is
taken from the published Boltz-1 and AlphaFold3 training description and is
assumed to apply to DeCAF through distillation.

**And the panel is not homology-decontaminated, because it cannot be.** A
temporal split is not the current standard on its own; contamination-aware
protein benchmarks pair it with sequence filtering, admitting only sequences
below 30% identity to anything released before the cutoff. Not one of the 22
held-out receptors passes that: the median maximum identity to the pre-cutoff
PDB is **1.000**, and 17 of 22 have a relative at 90% or above. This is not a
selection failure — 22 candidates sampled at random from the whole screened pool
give the same answer, 0 of 22 under every threshold. Peptide-binding domains are
too densely represented in the PDB for sequence-level decontamination to be
achievable at all.

What the split therefore isolates is **complex-level** novelty with
receptor-level familiarity held constant: both panels sit at ~1.0 receptor
identity to training, and only the receptor-peptide pairing is new. That is the
operationally relevant comparison for screening, where the target is a
characterised protein and the candidate peptide is the novel part, but it is a
weaker claim than "structures the model has never seen", and this section makes
no such claim.

<div class="page-break"></div>

### 7.11 Why the Readouts Behave As They Do: Backbone Convergence

Sections 7.4 to 7.10 accumulate readouts that fail for reasons stated
separately: pDockQ fails because its contact term runs backwards; ipTM tracks
composition; a physics energy function returns nonsense; a PAE-derived metric
reported as best-in-class in the literature does not reproduce here. This
section gives a single cause for all of them, found by auditing something the
earlier sections took for granted — whether the predicted structures are
structures at all.

#### 7.11.1 At ten sampling steps, the coordinates are not a folded protein

A peptide bond fixes consecutive alpha-carbons at 3.80 Å. Measuring every
consecutive CA–CA distance across the 132 Boltz-2 folds of Section 7.4:

| | Boltz-2 at 10 steps | DeCAF at 10 steps |
| :--- | ---: | ---: |
| median CA–CA distance | 5.48 Å | **3.74 Å** |
| physically plausible (3.4–4.2 Å) | **14.0%** | **96.2%** |
| broken (> 5 Å) | 56.2% | **0.0%** |
| implausibly short (< 3.0 Å) | 12.4% | 1.0% |
| chains more than half non-physical | 99% | — |
| structures measured | 132 | 69 |

Observed distances run from 0.33 Å to 63.4 Å. Residue numbering is sequential,
so this is not a file-ordering artefact: the backbone genuinely is not
connected. What Boltz-2 returns at 10 sampling steps is a partially denoised
point cloud carrying approximately correct residue identity and per-residue
confidence, not a folded chain.

This is the physical form of the objection Section 7.8 tested statistically.
Ten steps of an intended two hundred does not converge the coordinates, and
**DeCAF, distilled *for* ten steps, does.** Section 7.8 measured that
distillation buys a 5–6× larger effect without being able to say what it bought.
It buys geometry.

#### 7.11.2 What one cause explains

**pDockQ (Section 7.6).** Its contact term was found to run the wrong way,
scrambled peptides making more inter-chain contacts than cognates, 38.5 against
32.9. On DeCAF's converged structures the ordering reverses to the sensible
direction — cognates 61.4, decoys 55.3, scrambles 51.8 — so the inversion was a
property of unconverged geometry, not of short peptides. Section 7.6's
explanation was right about the mechanism and wrong about the cause.

**pDockQ2.** Replacing that contact term with a PAE-derived one repairs the
metric on the same Boltz-2 structures, from p = 0.797 on the scramble control to
p = 0.00026, which is what the mechanism predicts.

**Minimum PAE.** Reported as the best enrichment metric for AlphaFold3 and
Boltz-2 on a ligand benchmark, it fails outright on the Boltz-2 panel here
(scramble control p = 0.611) and is among the best readouts on the DeCAF held-out
panel (mean rank 1.55 over two draws, 15 of 22 cognates first). PAE is a
prediction *about geometry*; it means little when the geometry has not
converged. The published result was measured at full sampling.

**Physics rescoring.** PRODIGY returns a femtomolar dissociation constant for an
arbitrary peptide on the Boltz-2 structures. That is a garbage-in failure, not a
method failure — an energy function assumes a physical backbone.

The unifying statement: **geometry-dependent readouts require a converged
sampler. At reduced sampling only the non-geometric confidence outputs retain
meaning, and few-step distillation restores geometry and the geometric readouts
with it.**

#### 7.11.3 Physics rescoring on structures that support it

With DeCAF's 96%-physical backbones an energy function has something to read,
which is the direction the recent literature takes to exceed confidence-metric
performance. Scored on 69 folds across 12 held-out receptors:

| Readout | cognate − scramble | mean rank (chance 2.50) | AUC within |
| :--- | ---: | ---: | ---: |
| Interface pLDDT | +7.63 (p = 5e-5) | 1.73 (p = 0.036) | **0.856** |
| pDockQ2 | +0.112 (p = 0.006) | 1.64 (p = 0.025) | 0.807 |
| **PRODIGY ΔG** | +0.671 (p = 0.109) | **2.36 (p = 0.889)** | **0.563** |

And it does not combine: adding ΔG to interface pLDDT under leave-one-receptor-
out takes AUC from 0.856 to 0.823. The contact signal exists on converged
structures but is worth only 0.53 kcal/mol between cognates and scrambles, which
PRODIGY's polarity weighting and non-interacting-surface terms dilute rather
than sharpen.

This is one contact-count regression rather than a force field, so it bounds
what PRODIGY does here and not what physics could do in general.

#### 7.11.4 A sequence model, for scale

Two 2026 results argue that structure-based representations are the wrong
instrument: a fine-tuned Boltz-2 underperforms sequence-based alternatives for
protein-protein affinity at both small and large data scale, and binding
affinity predicted from ESM-2 embeddings gains little from adding structure.
MINT — ESM-2 650M with cross-chain attention, trained on 96 million
protein-protein interactions from STRING — provides a zero-shot interaction
probability with nothing fitted from this panel:

| Score | within-receptor AUC |
| :--- | ---: |
| **DeCAF interface pLDDT (structural)** | **0.807** |
| MINT zero-shot, in-training panel | 0.662 |
| MINT zero-shot, both panels | 0.614 |
| MINT zero-shot, held-out panel | 0.561 |

Its receptor specificity is weak: mean rank 2.05 to 2.37 against chance 2.50,
significant on neither panel. **Those published results do not transfer to this
task** — they concern affinity regression on protein-protein complexes, and this
is binder discrimination for short peptides, where the cofolding readout is
clearly ahead.

One result is worth isolating. MINT **passes the scramble control** (+0.064,
p = 0.003 across both panels), the control that demoted ipTM in Section 7.4. It
reads peptide order from sequence alone, and still cannot rank a cognate against
another receptor's cognate. Order sensitivity and receptor specificity are
therefore separable capabilities, and this dissertation has been treating them as
two tests of one property. MINT has the first without the second.

MINT is somewhat out of domain — trained on full-length STRING pairs, asked here
about peptides of 6 to 23 residues — so this bounds one zero-shot sequence model
rather than sequence models generally.

<div class="page-break"></div>

### 7.12 Reading the Panel as a Competition

Every readout in Sections 7.2 to 7.11 is something the model reports about its
own output — ipTM, pLDDT, PAE and the metrics built from them all come from the
confidence head. That is why they all sit under the ceiling of Section 7.9.2,
and why combining readouts, combining models, and rescoring with a physics
energy function each buy nothing. Escaping it requires information the model did
not produce. Three sources were available and all three were tested.

All folds in this section are at 10 sampling steps, so the precision figures
below are lower bounds in the same way as Sections 7.4 to 7.10 (Section 7.13).
The filter is a re-reading of scores rather than a new measurement, so a larger
underlying effect should if anything sharpen it.

#### 7.12.1 The one that works: reciprocal best match

The panel has been read in one direction throughout — *for this receptor, is the
cognate the best of its candidate peptides?* It also contains the transpose,
because each peptide is folded against its own receptor and against the several
that borrowed it as a decoy. That direction asks *for this peptide, is its own
receptor the best of the targets it was tried against?*, and it works at least
as well:

| direction | mean rank | chance | p |
| :--- | ---: | ---: | ---: |
| receptor-centric (Sections 7.4–7.10) | 1.73 | 2.50 | 0.0042 |
| peptide-centric | 1.58 | 2.66 | 0.0014 |

Requiring **both** is the binding analogue of reciprocal best hits in homology
search. Measured across both panels, both readouts and all five draws — twelve
combinations in total:

| | one-directional | reciprocal |
| :--- | ---: | ---: |
| precision | 56% (range 42–72%) | **88% (range 75–100%)** |
| enrichment over base rate | 2.3× | **3.5×** |
| calls retained | — | 49% |

Pooled over the twelve, the filter discards **87% of wrong calls (84 of 97) and
23% of right ones (29 of 125)**. That asymmetry is the result: a criterion that
merely shrank the candidate set would discard both classes in proportion.
Permutation tests against discarding the same number of calls at random clear
0.05 in ten of twelve cells.

**It replicates, which nothing else in this section does.** Precision improves
in 12 of 12 combinations. Draws within a panel share receptors and are not
independent, so the defensible unit is panel × readout — 4 of 4 — rather than
the 12-way sign test.

This is not a better score. It is the same scores read as a two-way competition,
so it gets its power from the structure of the screen rather than from the
model, which is precisely why the Section 7.9.2 ceiling does not apply to it.

**Deployed form.** Score each candidate against the target *and* against a small
panel of off-targets, and keep only candidates whose best target is yours. The
cost is five to ten times the folds, which is real but bounded, and buys
precision from 56% to 88%.

**Limitations.** Call counts are small — seven to thirteen per combination, so a
reported 100% is eight of eight. The competition is also incidental rather than
designed: each peptide meets only the three to six receptors that happened to
borrow it, and a deployed off-target panel would be chosen deliberately. And the
filter improves both regimes without undoing contamination — held-out precision
before filtering is 50% against 60% in training, consistent with Section 7.10.

#### 7.12.2 Two that do not: external ground truth and sample agreement

**Site correctness.** The crystal says which receptor residues contact the
peptide, so for any prediction one can ask what fraction of its contacts fall
inside that known site. This is information from outside the model entirely.
Measured on 120 folds across 20 held-out receptors:

| class | site precision | site recall |
| :--- | ---: | ---: |
| cognate | 0.452 | 0.562 |
| decoy | 0.445 | 0.580 |
| scrambled | 0.449 | 0.541 |

Rank 2.50 against chance 2.50, p = 0.906. **The measurement works and the
discrimination is absent**: random contacts would give a precision of 0.148, so
the model locates the true groove three times better than chance — and does it
equally for every class. Site-finding is driven by the receptor, not the
peptide. A groove is concave, hydrophobic and accessible whether or not the
peptide belongs in it.

**Pose convergence.** If a peptide genuinely binds it has one favourable pose to
find, so independent draws should agree; a non-binder should scatter. Folding
the panel twice with structures retained and superposing on the receptor:

| class | peptide RMSD between draws | receptor RMSD |
| :--- | ---: | ---: |
| cognate | 8.35 Å | 2.68 Å |
| decoy | 9.67 Å | 3.12 Å |
| scrambled | 9.99 Å | 3.34 Å |

The direction is right and the effect is not significant (+1.64 Å, p = 0.129;
rank 2.41, p = 0.713). Receptor RMSD also varies by class and correlates with
peptide RMSD at r = +0.36, so part of the apparent signal is the whole complex
being more reproducible rather than the peptide being better placed; regressing
it out halves the effect to +1.08 Å (p = 0.350).

**Together these two say something specific.** Every peptide lands in the same
correct pocket, and every peptide is then placed 8 to 10 Å differently within it
on each draw. For a peptide of ten to twenty residues that is a different pose
each time. The pocket is determined by the receptor; the pose is not determined
at all.

That closes the account of why so much of this section failed. Inter-chain
contacts, contact density, buried surface area, pDockQ, PRODIGY's binding energy
and pose agreement are all reading a placement that is correct in location and
arbitrary in detail. **The only axis carrying binding information is
confidence**, which is exactly why every readout reduces to the same ceiling.

<div class="page-break"></div>

### 7.13 The Settings Confound, Resolved — and It Was Real

Every fold in Sections 7.2 to 7.12 was taken far below the model's defaults: 10
sampling steps against 200, 1 recycling pass against 3, and an alignment
subsampled to 32 rows. Section 1.5 states this and Section 7.8 partly addresses
it by running a model *trained* for ten steps, but nothing had been folded at
the intended settings. "The settings confound is stated, not resolved" therefore
stood as the strongest available criticism of every negative result in this
work.

It is now resolved, and **the criticism was correct**.

#### 7.13.1 The comparison

The same panel, the same model (stock Boltz-1), the same device (MPS), with only
the settings changed. The reduced arm already existed from Section 7.8, so model
and device are held constant by construction:

| | sampling steps | recycling | MSA depth |
| :--- | ---: | ---: | ---: |
| reduced (Sections 7.2–7.12) | 10 | 1 | 32 |
| full | 200 | 3 | full (up to 12,882 rows) |

72 folds at full settings — every cognate and every scramble, which is what the
decisive test needs. The scramble control is the one that matters here: a
cognate against a permutation of itself, composition and length held equal.

#### 7.13.2 The signal was suppressed by a factor of three to seven

| Metric | reduced | full | Cohen's *d* | within-receptor z |
| :--- | ---: | ---: | :--- | ---: |
| ipTM | +0.039 (p = 0.004) | **+0.287 (p < 1e-5)** | 0.45 → **1.25** | 4.0× |
| Interface pLDDT | +1.54 (p = 0.067) | **+11.85 (p < 1e-5)** | 0.28 → **1.52** | 4.7× |
| Receptor side | +0.71 (p = 0.428) | **+5.13 (p < 1e-5)** | 0.12 → **1.45** | 7.3× |

The raw effects are 7.2 to 7.7 times larger. That is partly scale — the model is
far more confident overall at full settings — so the standardised columns are
the honest ones, and they still show **2.7 to 12 times**. Cohen's *d* moves from
negligible-to-small (0.12–0.45) to large (1.25–1.52) on all three readouts.

It is not driven by outliers: the effect is larger at full settings for **21 of
22 receptors** on ipTM and interface pLDDT, and 18 of 22 on the receptor side,
with paired p ≤ 0.001 throughout.

Two readouts change verdict outright. Interface pLDDT went from p = 0.067 —
the number Section 7.8 called model-dependent — to p < 1e-5. The receptor side
went from p = 0.428, no evidence at all, to p < 1e-5.

#### 7.13.2b Receptor specificity: the practical result changes entirely

The scramble control asks whether a readout notices sequence order. Ranking a
cognate against its own decoys asks the question a screen actually asks, and it
moves further:

| Metric | reduced rank | full rank | reduced first | **full first** | reduced AUC | **full AUC** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ipTM | 1.86 | **1.41** | 11/22 | **15/22** | 0.684 | **0.908** |
| Interface pLDDT | 1.91 | **1.27** | 9/22 | **17/22** | 0.640 | **0.943** |
| Receptor side | 2.05 | **1.27** | 8/22 | **17/22** | 0.600 | **0.943** |

Chance is a rank of 2.50. At the intended settings **interface pLDDT places the
true binder first for 17 of 22 receptors — 77% top-1 accuracy — at a
within-receptor AUC of 0.943**, with p < 1e-4 on every row.

For comparison, the best figure anywhere else in this dissertation is DeCAF at
ten steps on the in-training panel: rank 1.73, 13 of 22 first, AUC 0.807
(Sections 7.8, 7.12). Full-settings stock Boltz-1 exceeds it on every measure.
It also exceeds the AUC 0.90 that a recent nanobody benchmark reaches with a
model ensemble plus physics-based rescoring — the direction Section 7.12
identified as the only credible route past this work's numbers, and which turned
out to be unnecessary.

**This is the practical conclusion of the dissertation, and it is positive.**
Sections 7.2 to 7.12 measure a model run at a twentieth of its sampling budget
and conclude that cofolding confidence is a weak screening signal. Run at its
intended settings on the same panel, the same readout ranks the true binder
first three times in four. The negative results are properties of the regime,
not of the method.

#### 7.13.3 Why: the mechanism is Section 7.11's

Absolute confidence rises enormously alongside the effect:

| | reduced | full |
| :--- | ---: | ---: |
| cognate ipTM | 0.203 | **0.810** |
| cognate interface pLDDT | 47.79 | **90.70** |

And the backbones become physical. Measuring every consecutive CA–CA distance
against the 3.80 Å a peptide bond fixes:

| Arm | median CA–CA | physically plausible | broken (> 5 Å) |
| :--- | ---: | ---: | ---: |
| Boltz-2 @ 10 steps (Section 7.11) | 5.48 Å | 14.0% | 56.2% |
| DeCAF @ 10 steps (Section 7.11) | 3.74 Å | 96.2% | 0.0% |
| **Boltz-1 @ 200 steps** | **3.80 Å** | **99.7%** | **0.0%** |

A median of exactly 3.80 Å is the ideal bond length to two decimals. Section
7.11 established that geometry-dependent readouts need a converged sampler and
that few-step distillation restores convergence; this shows the other way of
restoring it — simply running the intended number of steps — and the confidence
readouts recover with it.

#### 7.13.4 What this costs, and what it changes

**A full-settings fold takes 106 to 109 seconds on this laptop.** The runner's
own comment records that a full alignment was intractable on CPU — a 40-complex
run did not finish one batch in an hour and drove the machine to ~12 GB of swap.
On MPS a full-depth fold at 200 steps is no slower than the same fold at depth
32. **The obstacle was the device, not the settings**, and the reduced regime
that shaped every result in Section 7 outlived the constraint that justified it.

Three consequences, stated plainly:

**Section 7.4's headline needs qualifying.** "ipTM tracks composition, not
binding" was measured at 10 steps. At 200, ipTM separates a cognate from its own
scramble with *d* = 1.25. The finding is a property of the reduced regime at
least as much as of the metric.

**Section 7.8's DeCAF result is reframed rather than overturned.** That section
compared DeCAF against stock Boltz-1 *at the same reduced settings* and found a
5–6× advantage; that comparison stands. What changes is the interpretation.
Full-settings Boltz-1 reaches +0.287 and +11.85 where DeCAF at ten steps reached
+0.201 and +9.54, so distillation is not adding something the stock model lacks
— it is recovering, at a twentieth of the sampling budget, most of what the
stock model has when run properly. For an efficiency thesis that is a better
result than the original reading, and it is the one the evidence supports.

**Sections 7.6, 7.7 and 7.10 understate their effects.** Every effect size in
those sections was measured in the suppressed regime and should be read as a
lower bound.

#### 7.13.5 Limitations

One model (Boltz-1) and 22 receptors, folded once — all 132 pairs, so both the
scramble control and the rank test are resolved. The three settings were raised
together, so which of sampling steps, recycling or MSA depth carries the effect
is unresolved; Section 7.11's geometry result points at sampling steps, and the
script varies each independently for that test.

Section 7.5's warning applies with force: this is a single draw, and a single
draw has twice misled this dissertation. The effects here are large enough
(*d* = 1.25–1.52, AUC 0.943) that draw noise is unlikely to reverse them, but
the specific figures should be replicated before being quoted as exact values.

Two consequences for earlier sections are untested rather than resolved. The
held-out comparison of Section 7.10 was run entirely at reduced settings, so
whether contamination costs a factor of two at full settings is unknown.
And Section 7.12's reciprocal matching lifted precision from 56% to 88% in the
suppressed regime; at full settings plain top-1 is already 77%, so the headroom
that filter exploits is smaller and its value there is untested.

<div class="page-break"></div>

### 7.14 The Findings as a Tool

Sections 7.2 to 7.13 are negative results with mechanisms attached. A negative
result is only useful to a bench scientist if it changes what they would
otherwise do, and a caveat in a report does not change anything. This section
describes a local screening tool built so that the three findings most likely to
mislead a user are enforced by the software rather than stated beside it, and
reports what it does on targets it was not tuned for.

The tool takes a target sequence and a list of candidate peptides and returns a
ranking. It runs entirely on the machine that folds; nothing is uploaded.

#### 7.14.1 Three findings, three behaviours

**Section 7.4 — a confidence score can be blind to sequence order.** So no
candidate is reported as a raw score. Every candidate is folded against
permutations *of itself*, and reported as a statistic against that null. A
candidate that does not separate from its own scrambles is not a hit, whatever
its absolute confidence. This is the project's central control made compulsory:
a user cannot obtain a ranking without it.

**Section 7.13 — reduced settings suppress the effect three- to sevenfold.** So
the fast mode is offered but declares its cost in the interface, quoting the
suppression factor and the 14% of backbone bonds that are physically plausible
at ten sampling steps (Section 7.11). The user is not told the tool is fast; they
are told what fast costs.

**Section 7.5 — a single fold does not reproduce its own ranking.** So replicates
are a first-class control and the replicate spread is a column in the results
table, next to the score it qualifies, rather than a footnote.

Candidates that cannot be scored honestly are refused rather than scored badly:
a non-standard residue, or a length outside the benchmarked range. Section 7.3
found a non-peptide in the panel because a score was returned for something that
should have been rejected; refusing is the corrected behaviour.

#### 7.14.2 What it costs to run

Almost all of a screen is fixed cost. Timing one fold against a batch of four
separates the two components:

| folds in one call | wall time | implied |
| ---: | ---: | :--- |
| 1 | 53.9 s | — |
| 4 | 74.7 s | fixed ≈ 47 s, marginal ≈ 6.9 s |

**87% of a lone fold is model construction.** Three consequences follow, and each
was measured rather than assumed.

First, a job is folded in a single process. An earlier design split jobs into
chunks of six and paid the 47 s once per chunk.

Second, a persistent worker process does not help, which is worth recording
because it is the obvious optimisation. Calling `predict()` twice inside one
process gave 45.9 s then **60.9 s** — the second call was slower, not cheaper.
Boltz reconstructs its model per call, so keeping the process alive buys nothing
and the idea was dropped.

Third, folds are cached by target, peptide and settings. The cache stores a
*list* of independent folds per key rather than one value: folds are unseeded and
replicates are meant to differ, so collapsing a key to a single value would give
every replicate the same number, drive the replicate spread to exactly zero, and
have the tool claim a reproducibility it does not have.

A further saving is statistical rather than computational. A scramble is folded
once rather than once per replicate, because the null asks what a composition
scores *over orderings*; a fixed budget buys more by sampling more distinct
permutations than by re-folding a few of them. Folds per candidate fall from
`(1 + s) × r` to `r + s`, a third fewer at the defaults, with a better-sampled
null.

| screen | folds | time |
| :--- | ---: | ---: |
| three candidates, cold | 15 | 177 s |
| the same screen again | 15 | 3 s |
| two candidates added to three cached | 25 | 135 s |
| alignment for a target never seen before | — | 2.6 s |

The alignment is fetched from the MSA server directly. The previous route started
a boltz process on CPU purely to reach the MSA step and discarded the structure it
went on to predict; the direct call returns the same 1,628 rows at the same width
in 2.6 s against roughly 60 s.

#### 7.14.3 Does it discriminate between targets?

The question a screen asks is not whether a peptide scores well but whether it
scores well *on this target*. The test is therefore the same candidate list
against two different targets, where the correct answer differs. Three cognate
peptides were taken from the panel of Section 7.4 — the p53 transactivation helix
for MDM2, the C-terminal motif `KQTSV` for the PSD-95 PDZ3 domain, and the
proline-rich `PPPALPPKKR` for the c-Crk SH3 domain — and each list was run
unchanged against MDM2 and against PDZ3, at fast settings, two replicates and
three scrambles.

| candidate | MDM2 *t* | MDM2 *p* | PDZ3 *t* | PDZ3 *p* | cognate of |
| :--- | ---: | ---: | ---: | ---: | :--- |
| SQETFSDLWKLLPEN | **+3.88** | **0.002** | +0.06 | 0.477 | MDM2 |
| KQTSV | +1.39 | 0.101 | **+7.72** | **<0.001** | PSD-95 PDZ3 |
| PPPALPPKKR | −0.00 | 0.501 | +0.75 | 0.240 | c-Crk SH3 |

Each cognate wins on its own target and loses on the other. The result is not an
artefact of peptide length: `KQTSV` is the shortest candidate at five residues
and it wins on one target and places second on the other. Each screen is 15 folds
and about three minutes.

Adding the designed MDM2 binder PMI (`TSFAEYWNLLSP`) and raising the settings
reproduces the ordering more cleanly. At 200 sampling steps, three recycling
passes and full alignment depth, with five scrambles, the pooled scramble spread
falls from 5.36 to 2.65 and the separation widens:

| candidate | *t* | *p* | |
| :--- | ---: | ---: | :--- |
| TSFAEYWNLLSP | **+8.24** | **<0.001** | designed MDM2 binder |
| SQETFSDLWKLLPEN | **+7.72** | **<0.001** | cognate |
| PPPALPPKKR | −0.52 | 0.694 | binds SH3 |
| KQTSV | −0.57 | 0.711 | binds PDZ3 |

Both foreign ligands go firmly negative — `KQTSV`'s borderline p = 0.101 at fast
settings becomes p = 0.711 — and the designed binder correctly outranks the
natural helix. This is Section 7.13's result reproduced by a different route and
on a different question.

#### 7.14.4 Where it does not work

On the c-Crk SH3 domain the tool fails, and the failure is reported by it rather
than discovered afterwards. The cognate `PPPALPPKKR` does not separate from its
own scrambles at fast settings (*t* = +0.27, p = 0.40) and places second behind a
peptide for a different target. Raising the settings does not rescue it:

| candidate | *t* | *p* |
| :--- | ---: | ---: |
| KQTSV | +1.71 | 0.056 |
| PPPALPPKKR | +1.69 | 0.058 |
| SQETFSDLWKLLPEN | +0.12 | 0.454 |

The one thing it gets right at either setting is placing the MDM2 helix last.

A partial mechanism is available. `PPPALPPKKR` is 50% proline, and its
permutations — `APPRLPPKKP`, `APRKPPKPLP` — remain proline-rich PxxP-like
sequences. SH3 domains bind polyproline-II helices substantially through
composition, so a permutation of a proline-rich ligand is a plausible ligand,
and a control that destroys only order has little left to detect. The scramble
control is conservative by construction, and this is the regime where that
conservatism costs the most.

That explanation is incomplete, and is reported as incomplete: `KQTSV` contains
no proline at all and behaves the same way on this target, scoring 93.4 against
scrambles at 86.6. This receptor returns high, flat interface confidence for most
of what is put in front of it. What the tool reports is *no order-dependence
detected*, which is a limit of the control rather than evidence that nothing
binds — and the interface says "indistinguishable from its own scrambles", not
"not a binder".

#### 7.14.5 Two faults found by building it

Turning a result into a tool exposed two statistical faults that the analysis
scripts of Sections 7.2 to 7.13 had not, because those scripts operate on 132
folds and the tool operates on twelve.

**A null standard deviation estimated per candidate collapses.** With two
permutations per candidate, a per-candidate SD occasionally comes out near zero
by chance, and the ratio explodes. A real screen reported **z = +77.56** and
**z = −80.16** — numbers that carry the appearance of overwhelming evidence and
mean nothing. The denominator is now a spread pooled across every candidate in
the job. On the same data the pooled estimate is 3.55 on 3 degrees of freedom,
and the same three candidates report +4.29, −0.05 and −0.71.

**A pooled SD is not a *z* denominator.** Because it is estimated from a handful
of deviations, the ratio is a *t* statistic on those degrees of freedom. Judged
against a fixed cutoff of one, `GSGSGSGSGSGS` — a flexible Gly–Ser linker that
binds nothing — passed as a hit at 1.06. On the *t* scale the same candidate is
p = 0.136. The tool now reports *t* and its one-sided *p*, so a verdict means the
same thing in a three-candidate job as in a ten-candidate one.

The second fault has a consequence the tool now states. With few scrambles the
null has almost no degrees of freedom and the critical value is large: at df = 2
a candidate needs *t* > 2.92 before the job can call anything, so a genuine
binder with a margin of +11.3 still reads as nothing. That is a statement about
the size of the job, not about the candidate, and an underpowered screen now says
so rather than letting a null result pass for evidence of inactivity.

A third fault was an engineering one with the same character: a single scramble
gives `numpy.std(ddof=1) = NaN`, which a guard of the form `sd or 1e-9` does not
catch, because NaN is truthy. The statistic is now withheld rather than invented
when there is no spread to estimate.

#### 7.14.6 What this does and does not establish

The tool is a demonstration that the findings are actionable, not an independent
validation of them. Three limits should be read with the numbers above.

The two-target result rests on three targets and three peptides, one of which
fails. It shows the readout is not simply rewarding peptide-shaped sequences; it
does not establish a hit rate.

All three targets are pre-cutoff structures the model was trained on. Section
7.10 measured that held-out complexes cost roughly half the effect, so the
separations in 7.14.3 should be read as approximately twice what a genuinely
novel target would give.

The ranking remains a triage order, not a measurement of affinity. Section
7.11.3's physics rescoring reached AUC 0.563 against 0.856 for interface pLDDT,
and adding it to the confidence readout made matters worse; nothing in this tool
converts a rank into a binding constant.

<div class="page-break"></div>

### 7.15 An Automated Search Over Readouts, and What Its Controls Refuse

Every readout in Sections 7.6 to 7.12 was chosen by hand, and Section 7.11.3
tried exactly one combination — adding PRODIGY's ΔG to interface pLDDT, which
lowered leave-one-receptor-out AUC from 0.856 to 0.823. That is an anecdote
about one pairing. This section searches the combinations systematically.

The search costs nothing to run: it operates on folds already on disk, needs no
GPU, and finishes in about eleven minutes. What it demonstrates is not a better
readout but how easily a search of this kind appears to find one.

#### 7.15.1 The search, and why it is scored leave-one-receptor-out

Seven per-fold features are available on the full-settings panel: interface
pLDDT, ipTM, the receptor and peptide sides of the interface, the peptide's
whole-chain confidence, and the two interface contact counts. All 63 subsets of
size one to three are scored by within-receptor AUC — the standardised measure
of Section 7.13 — with a logistic combination fitted **leave-one-receptor-out**
for subsets larger than one.

The cross-validation has to be by receptor. A random split leaks, because the
six folds of one receptor share a baseline confidence: a model that has seen
five of them has seen most of what it needs to place the sixth.

#### 7.15.2 What it found

| Readout | within-receptor AUC |
| :--- | ---: |
| **receptor side + interface contacts** (search winner) | **0.961** |
| interface pLDDT alone (Section 7.13's headline) | 0.943 |
| receptor side alone | 0.943 |
| peptide whole-chain | 0.896 |
| peptide side | 0.895 |
| ipTM | 0.908 |
| interface contacts, receptor side | 0.556 |
| interface contacts, peptide side | 0.448 |

The winner beats the dissertation's headline readout by **+0.018 AUC**. Taken at
face value that is a new best result, arrived at automatically, and it is the
kind of number this section exists to disbelieve.

#### 7.15.3 Two controls, and both refuse it

**A searched null.** The entire search is re-run on labels permuted within each
receptor — the cognate label moved to a random fold of the same receptor, which
preserves the design exactly and destroys only the association being measured.
Over 200 permutations:

| | AUC |
| :--- | ---: |
| best-of-search on permuted labels, mean | 0.607 |
| 95th percentile | 0.700 |
| **maximum** | **0.755** |

A search over 63 subsets of pure noise reaches AUC 0.76. Any single number from
an unnulled search of this size is uninterpretable.

The decisive comparison is not the winner against chance — the features
obviously carry signal — but the **gain** the search buys over simply taking the
best readout on its own. On permuted labels that gain averages **+0.032**, which
is *larger* than the +0.018 actually observed. The observed improvement sits at
**p = 0.425**: entirely consistent with what searching noise produces.

**The held-out panel, spent once.** The winning combination was applied a single
time to the full-settings held-out folds of Section 7.10, with nothing selected
on them. It reverses:

| Readout | held-out AUC |
| :--- | ---: |
| interface pLDDT alone | **0.688** |
| receptor side + interface contacts | 0.660 |

The +0.018 gain becomes −0.029.

The winner's own composition says the same thing without any statistics.
`n_rec_iface` reaches AUC 0.556 by itself, barely above chance. Pairing a strong
feature with a nearly useless one and gaining is the signature of a model fitting
22 receptors, not of a discovery.

A length control was run alongside, because decoys are drawn from other
receptors' cognates and therefore differ in length from the true binder: peptide
length alone reaches AUC 0.519. The panel carries no length confound for a search
to exploit, which Section 7.6 could not assume and had to retire a vacuous
control to establish.

#### 7.15.4 What this establishes

The ceiling stands. **Interface pLDDT on its own is not beaten by any
combination of these readouts**, and Section 7.11.3's single negative pairing
generalises to the whole space rather than being a property of PRODIGY.

The more useful contribution is the null distribution. It puts a number on how
readily an automated search over a 22-receptor panel manufactures an apparent
improvement — a mean best of 0.607 and a maximum of 0.755 where no signal
exists, and a mean gain of +0.032 over the best single feature. Any future
search of this panel, by hand or by machine, has to clear that bar before its
result means anything.

This is also the honest boundary of automation for this problem. The search is
cheap because it re-analyses folds already computed; the experiments that would
actually advance this work — resolving which of the three settings carries
Section 7.13's effect, extending the panel past 22 receptors, adding held-out
draws — are folding-bound at roughly 106 seconds a fold, and no search loop
changes that arithmetic.

<div class="page-break"></div>

# 8. CONCLUSIONS AND RECOMMENDATIONS

## 8.1 Conclusions

**On the engineering objectives.** Objectives 1 to 4 of Section 1.3 were met.
The Boltz inference path executes device-agnostically on Apple Silicon, with 16
CUDA dependencies removed and autocast and cache eviction dispatched by device.
Target-side latent key-value caching reduces cached state from `2 × N × D` to
`N × D_latent`, an 87.5% reduction at the operating point used here. The
low-rank pair representation, the classifier-free-guidance student and the
neural coordinate refiner are integrated and verified numerically against their
dense references. A 99-test suite runs under continuous integration.

**On the validation objective.** Objective 5 produced results that qualify two
of the four engineering claims, and both qualifications are properties of the
problem rather than of the implementation.

First, **the low-rank pair representation is not reachable on pretrained
weights** (Section 7.1). Three progressively stricter experiments converge on a
held-out error of 0.378 at rank 32, where training and held-out error coincide —
the signature of a model at capacity rather than one starved of data. The fitted
scaling law places 10% error at rank ≈1,414, against the `c_hidden² = 1024`
width that the stock implementation actually materialises. The low-rank form
therefore costs more activation memory than the operator it replaces before it
becomes accurate enough to substitute for it. The 1502× figure in Table 3
remains valid as a scaling measurement for a network *trained* around the layer;
it is not a drop-in saving.

Second, **interface confidence does not rank peptide binders reliably**
(Sections 7.2 to 7.5). On a PTM-clean panel of 22 receptors and 132 complexes,
scrambled peptides score indistinguishably from cognates and significantly above
genuine binders of other receptors (AUC 0.632, p = 0.0096). Since a scramble
preserves composition and length and destroys only order, the discriminating
variable is composition. Any order effect is bounded at 63% of the composition
effect. A ridge regression on additive amino-acid composition independently
outperformed the distilled neural surrogate (+0.103 against −0.034, z = 3.44),
reaching the same conclusion from an unrelated direction.

Third, and generalising beyond this project, **a single unseeded fold does not
reproduce its own ranking** (Section 7.5). Pooled within-complex ipTM standard
deviation is 0.0628. Across four identical re-runs, all four receptors tested
changed the rank of their cognate, one moving from best to worst. Re-simulating
the benchmark under that noise leaves the aggregate effect intact but reduces
the headline significance verdict to a coin flip: p < 0.05 in 49% of re-runs.
Ranking candidate binders from a single folding run is common practice, and at
reduced sampling settings it is not reproducible.

Fourth, and positively, **the information ipTM discards is recoverable on
structures the model was trained on** (Section 7.6). Re-scoring the same
structures shows interface pLDDT distinguishes a cognate from its own scramble
(p < 0.0001, reproducing in 100% of simulated re-runs) at 8.6 times ipTM's
effect-to-noise ratio.

Section 7.7 then localises that signal, and qualifies it. Splitting the metric by
chain shows most of its peptide-side component is whole-chain peptide confidence
rather than interface confidence — the foldability objection is partly correct.
What survives is the receptor side: the receptor's own residues are placed more
confidently when given the cognate peptide (+2.38, p = 6e-5), which peptide
disorder cannot explain. An alanine scan of the binding site points the same way
but falls short of significance (p = 0.244, requiring roughly six times the
panel), so the response is not yet shown to be site-localised.

**Sixth, and this reframes the fourth: most of that advantage was retrieval**
(Section 7.10). A second panel of 22 receptors released after the training
cutoff — decoys drawn from within it, so no fold involves a training structure —
finds **both** readouts losing roughly half their effect — interface pLDDT from
+12.03 to +5.24, ipTM from +0.265 to +0.137 — over five independent draws. An
earlier reading of one draw, and then of two, claimed the degradation was
specific to one metric; both claims were withdrawn when the next draw reversed
them (Section 7.10.3). What stands is that a screening figure quoted without
saying whether the complexes were in training is about twice as good as it
should be. The interaction is significant on both raw and scale-free
measures for interface pLDDT and undetectable for ipTM. Longer held-out peptides
cannot explain it — the contrast is against a permutation of the same peptide —
and a ceiling cannot either, since the model is markedly *less* confident on
structures it has not seen. The same section finds four of the original
twenty-two cognate pairs were not binders at all, including one that is a
synthetic ligand rather than a peptide; removing them raises every effect, and
withdraws Section 7.8's claim that the readouts are model-dependent.

Fifth, **few-step distillation changes the picture materially** (Section 7.8).
Re-running the panel on a model distilled for 10-step sampling raises both order
sensitivity and receptor specificity by 5–6× over its own teacher at the same
budget, and establishes receptor specificity for the first time in this work
(interface pLDDT rank 1.73 against chance 2.50, p = 0.0042, clearing Bonferroni).
The de-confounding arm attributes this to the distillation rather than to the
base model. It also qualifies two earlier conclusions: stock models at reduced
sampling retain weak but real signal rather than none. (Its further claim that
the readouts are model-dependent is withdrawn in Section 7.10.1.)

**Overall.** The efficiency techniques transfer as engineering. The accuracy
claims are weaker and more conditional than either the ipTM literature or the
early sections of this dissertation suggest: on structures the model has not
seen, a cofolding confidence score enriches for binders but does not select
them, and the readout that works there is not the one that works in training.

**Seventh, and it explains much of the rest: at reduced sampling the predicted
coordinates are not folded proteins** (Section 7.11). Only 14% of backbone bonds
in the Boltz-2 structures are physically plausible, against 96.2% for the
few-step-distilled model, which finally says what Section 7.8's 5–6× advantage
was buying — geometry. One cause then accounts for four separate failures:
pDockQ's inverted contact term, minimum PAE working on one arm and not the
other, a physics energy function returning femtomolar affinities for arbitrary
peptides, and the contact-based rows of Section 7.6 generally. Geometry-dependent
readouts require a converged sampler; at ten steps only the non-geometric
confidence outputs retain meaning.

A sequence-model comparison bounds the alternative. MINT, trained on 96 million
protein-protein interactions, reaches within-receptor AUC 0.614 zero-shot against
0.807 for the best structural readout, so recent results reporting that
sequence-based representations beat structure-based ones for affinity regression
do not transfer to peptide binder discrimination. It does, however, pass the
scramble control that demoted ipTM — showing that order sensitivity and receptor
specificity are separable capabilities rather than two tests of one.

**Eighth, and the one actionable positive of the search: requiring the match to
hold in both directions raises precision from 56% to 88%** (Section 7.12).
Reading the panel as a competition in the peptide direction as well as the
receptor direction, and keeping only reciprocal best matches, discards 87% of
wrong calls and 23% of right ones. It replicates in 12 of 12 panel-readout-draw
combinations, which nothing else in this work does, and it costs nothing at
inference because it re-reads scores already computed. Two other attempts to
escape the ceiling failed: scoring predictions against the crystal's known
binding site does not discriminate at all (p = 0.906 — every peptide finds the
right groove), and pose agreement between draws is not significant once receptor
reproducibility is regressed out. Together those two show that the pocket is
determined by the receptor while the pose is not determined at all, which is why
every contact-derived readout in this work failed.

**Ninth, and it qualifies much of the above: the reduced inference settings were
suppressing the signal by a factor of three to seven** (Section 7.13). Folding
the panel on the same model and device at the intended 200 sampling steps, 3
recycling passes and full alignment depth raises Cohen's *d* on the scramble
control from 0.12–0.45 to 1.25–1.52, larger for 21 of 22 receptors, and takes
interface pLDDT from p = 0.067 to p < 1e-5 and the receptor side from p = 0.428
to p < 1e-5. Backbone geometry goes from 14% physically plausible to **99.7%**,
with a median CA–CA distance of 3.80 Å — the ideal bond length. Section 7.4's
"ipTM tracks composition, not binding" is therefore a property of the reduced
regime at least as much as of the metric, and every effect size in Sections 7.2
to 7.12 is a lower bound. Section 7.8's DeCAF comparison stands but is reframed:
distillation recovers most of what the stock model has when run properly, at a
twentieth of the sampling budget, which is a better result for an efficiency
thesis than the original reading. A full-settings fold costs 106 seconds on this
laptop, so the constraint that justified the reduced regime had lapsed before
most of the measurements were taken.

**And the practical conclusion inverts.** On the receptor-specificity test —
ranking a cognate against its own decoys, which is what a screen does —
interface pLDDT at full settings places the true binder **first for 17 of 22
receptors, 77% top-1, at a within-receptor AUC of 0.943**, against 0.640 at
reduced settings. That exceeds every figure elsewhere in this work, including
DeCAF's 0.807, and exceeds the AUC 0.90 a recent nanobody benchmark reaches with
model ensembling plus physics rescoring — the direction Section 7.12 identified
as the only credible route past these numbers, and which proves unnecessary.
Sections 7.2 to 7.12 conclude that cofolding confidence is a weak screening
signal; at the model's intended settings, on the same panel, it ranks the true
binder first three times in four.

One further correction belongs here rather than in a footnote. Section 7.10 was
first written from a single draw of the held-out panel, and a second independent
draw moved its central p-value four orders of magnitude. The claim that interface
pLDDT *collapses* on held-out complexes was a property of one sample; it weakens.
What replicates is the rank test, and the surviving claim is narrower: retrieval
buys the model the ability to tell receptors apart, not order sensitivity in
general. Section 8.2's own recommendation to average replicates was written
before it was followed.

**Tenth, the findings were made enforceable rather than advisory** (Section
7.14). A caveat in a report changes nothing a bench scientist does, so the three
results most likely to mislead one were built into a local screening tool as
behaviour: no candidate can be ranked without being folded against permutations
of itself, the fast mode declares its measured cost in the interface, and the
replicate spread sits in the results table beside the score it qualifies.
Sequences that cannot be scored honestly are refused rather than scored badly —
the corrected response to Section 7.3's non-peptide. Run unchanged against two
targets, the same three candidate peptides give opposite answers, each cognate
separating from its own scrambles on its own target (p = 0.002 and p < 0.001) and
not on the other (p = 0.477 and p = 0.101); a third target fails, and the tool
reports that failure rather than the author discovering it afterwards. Building
it also exposed two statistical faults that a 132-fold analysis had not: a
per-candidate null SD estimated from two permutations reported z = +77.56, and a
fixed cutoff on that ratio passed a Gly–Ser linker as a hit at 1.06 — p = 0.136
once the statistic is treated as the *t* it is.

**Eleventh, and it bounds what automation can add here: an automated search
over readouts finds an improvement that both of its controls refuse** (Section
7.15). Searching all 63 subsets of seven per-fold features, leave-one-receptor-
out, returns the receptor side combined with interface contacts at AUC 0.961
against interface pLDDT's 0.943 — a result that does not survive being asked
what a search over noise produces. Re-running the identical search on labels
permuted within each receptor gives a best-of-search averaging 0.607 and
reaching 0.755 with no signal present, and a mean gain over the best single
feature of +0.032 — larger than the +0.018 observed, at p = 0.425. Spent once on
the held-out panel the gain reverses, 0.688 for interface pLDDT alone against
0.660 for the pair. Section 7.11.3's single negative pairing therefore
generalises: the confidence readout is not improved by combining it with the
others, and the null distribution puts a number on how readily a 22-receptor
panel manufactures the appearance that it is.

The substantive contribution is that this was measurable at all. Six independent
controls were applied — a composition-matched scramble, replicate folds, a
de-confounding base-model arm, mixed-effects estimation, a peptide-integrity
audit, and a training-cutoff split — and **each one overturned a result that
looked solid without it**. A cofolding benchmark lacking these will overstate
what it measures, and most published ones lack all six.

## 8.2 Recommendations

1. **Rank on interface pLDDT, never on pDockQ, and halve your expectations for
   a novel target.** Interface pLDDT carries 8.6 times ipTM's effect-to-noise
   ratio (Section 7.6) and is the best single readout in every arm tested. On
   complexes released after the training cutoff — the screening case — it and
   ipTM both retain roughly half their effect (Section 7.10); five draws could
   not establish that either degrades more than the other, and two earlier
   attempts to claim one did were withdrawn. Quoting a benchmark figure without
   stating whether the complexes were in training is the single easiest way to
   overstate a cofolding screen, by about a factor of two. pDockQ fails outright:
   its contact term is computed on unconverged geometry and favours scrambled
   peptides (Sections 7.6, 7.11). **pDockQ2 repairs it** by substituting a
   PAE-derived term, and is the second-best readout measured.
2. **Never run a stock cofolding model far below its intended sampling budget,
   and re-measure the budget before assuming it is unaffordable.** At 10 of an
   intended 200 steps the effect on the scramble control is three to seven times
   smaller in standardised terms and the backbone is 14% physically plausible
   rather than 99.7% (Section 7.13). The reduced regime in this work was
   justified by CPU timings that MPS had already made obsolete: a full-settings
   fold takes 106 seconds and is no slower at full alignment depth. The
   practical difference is not marginal: interface pLDDT ranks the true binder
   first for 9 of 22 receptors at reduced settings and **17 of 22** at full,
   with within-receptor AUC rising from 0.640 to **0.943**.
3. **If the budget genuinely is short, use a model distilled for it rather than
   a stock model run short.** At ten sampling steps a stock model returns 14% physically
   plausible backbone bonds; a few-step-distilled model returns 96.2%
   (Section 7.11). Every geometry-derived readout — contacts, buried area,
   pDockQ, PAE-based metrics, any physics energy function — is meaningless on the
   former and works on the latter. This is the largest single quality difference
   measured in this work, and it costs nothing at inference.
4. **Require the match in both directions before calling a hit.** Score each
   candidate against the target *and* against a small panel of off-targets, and
   keep only candidates whose best target is yours. Precision rises from 56% to
   88% and enrichment from 2.3x to 3.5x, discarding 87% of wrong calls against
   23% of right ones (Section 7.12). It costs five to ten times the folds and
   nothing in modelling, and it is the only intervention in this work that
   improved in every panel, readout and draw tested.
5. **Split the benchmark on the model's training cutoff, and report both
   halves.** This was the most consequential control in the project and the last
   one applied. Boltz-1 and AlphaFold3 both cut off at 2021-09-30. Without the
   split, a panel drawn from the PDB measures retrieval and prediction together
   and reports the sum as accuracy.
6. **Report replicate-averaged confidence for per-receptor claims — and for
   any headline number.** Section 7.10.3 is the cautionary case: a single draw
   put the held-out interface-pLDDT effect at p = 0.089 and a second put it at
   p = 0.00008.
   Section 7.9 refines this: rank instability generalises to the few-step model,
   so per-receptor rankings do require averaging — but the variance decomposition
   finds aggregate discriminability signal-limited rather than noise-limited, so
   averaging will not rescue a weak metric at the population level. On this
   evidence, 9 to 16 replicate folds per complex are required before a
   per-receptor ranking is meaningful. Practitioners ranking binders on a single
   AlphaFold or Boltz run should treat those rankings as provisional.
7. **Always include a scramble control.** Composition and length alone reproduce
   most of the apparent discrimination between a cognate and a decoy. Without a
   composition-matched control, a benchmark cannot distinguish binder
   recognition from composition sensitivity.
8. **Audit what is bonded to the peptide, not just its sequence.** Seven of 25
   candidate complexes were PTM-dependent, and folding the canonical sequence
   makes those "positives" non-binding, which silently penalises the metric
   under test. An allowlist of known PTM codes is not sufficient — it passed a
   benzoyl-lysine (7F3S) and cannot catch a modification nobody anticipated.
   Scanning the mmCIF for anything covalently bonded to the peptide chain needs
   no list, and additionally catches glycosylation (9GRF) and peptides that are
   substantially synthetic (1NLO). Reject any peptide whose canonical sequence
   contains `X` (Section 7.10.1).
9. **Do not treat operator-level memory savings as drop-in.** A substituted
   operator should be validated against pretrained weights on real activations
   before its microbenchmark saving is claimed.
10. **Deploy on GPU before extending the biological claims.** The replicate
   folding required by recommendation 6 is a 1,200–2,100 fold workload for this
   panel, which is not a CPU-scale task.

<div class="page-break"></div>


<div class="page-break"></div>


# 9. FUTURE PLAN

| Sl No | Phases | Start Date - End Date | Work to be done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Dissertation Outline | 05 Jan 2026 – 04 Feb 2026 | Literature review and prepare outline | **COMPLETED** |
| 2 | Local Design & Implementation | 05 Feb 2026 – 20 Mar 2026 | Develop MLA, Fold-CP, SpecSampler, and g-DPO | **COMPLETED** |
| 3 | Pre-trained Weight Verification | 21 Mar 2026 – 10 Jun 2026 | Run local predictions with Boltz-1 parameters | **COMPLETED** |
| 4 | Edge Optimization & Integration | 11 Jun 2026 – 22 Jun 2026 | Integrate Low-Rank, CFG student, and Neural Refiner; resolve MPS dependencies | **COMPLETED** |
| 5 | Reference & Surrogate Validation | 23 Jun 2026 – 31 Jul 2026 | Measure ipTM as a ranking reference against PDB complexes; distil and evaluate the surrogate; establish the low-rank OPM reachability result | **COMPLETED** |
| 6 | Powered Specificity Run | 01 Aug 2026 – 02 Aug 2026 | Complete the 22-receptor Boltz-2 benchmark (Section 7.4) | **COMPLETED** |
| 7 | Measurement Reproducibility | 02 Aug 2026 – 02 Aug 2026 | Quantify ipTM run-to-run variance (Section 7.5) | **COMPLETED** |
| 8 | Few-Step Distillation Arms | 05 Aug 2026 – 06 Aug 2026 | Run the panel on Boltz-1 and DeCAF to de-confound model from step count (Sections 7.8, 7.9) | **COMPLETED** |
| 9 | Held-Out Panel & Panel Integrity | 08 Aug 2026 – 09 Aug 2026 | Build a 22-receptor post-cutoff panel, fold it three times, audit the original panel for non-binders (Sections 7.10, 7.10.1) | **COMPLETED** |
| 10 | Backbone Convergence & Search for a Better Readout | 09 Aug 2026 – 10 Aug 2026 | Audit predicted geometry; test physics rescoring, cross-model ensembling, a sequence model, site correctness, pose convergence and reciprocal matching (Sections 7.11, 7.12) | **COMPLETED** |
| 11 | Production GPU Scaling | 12 Aug 2026 – 20 Aug 2026 | Deploy on CUDA and run NCCL scale tests | **PENDING** |
| 12 | Final Thesis | 10 Aug 2026 – 28 Aug 2026 | Consolidate results and write the dissertation | **IN PROGRESS** |

**Change of plan, and why.** The original phase 5 (CUDA/NCCL scaling) was
deferred in favour of validating the scoring reference, because the binder
screen planned for phase 6 depends on it: a screen ranked by a signal that does
not discriminate binders would produce candidates with no meaning regardless of
how fast it ran. That validation (Section 7.2) showed the dependency does not
hold, which is why the plan prioritised measurement over throughput.

**A second change, and the reason GPU scaling has moved to the end.** An earlier
version of this plan listed a replicate-averaged rerun as *pending on CUDA
deployment*, on the assumption that 9 to 16 folds per complex was not a
CPU-scale workload. Two developments removed the dependency. MPS acceleration
gives roughly 3x over CPU (Section 5), and few-step distillation folds the whole
132-complex panel in about twenty minutes (Section 7.8). The held-out panel was
consequently folded **five times** on this laptop rather than once on a cluster,
and it was the third of those draws that overturned the conclusion the first two
supported (Section 7.10.3). Replicate averaging is now measured rather than
deferred, and it is the intervention with the largest effect on held-out
discriminability (+0.119 AUC, Section 7.9.4).

Scaling remains valuable for throughput and for extending the panel past 22
receptors, which Section 7.10.6 identifies as the binding constraint on the
central open question. It no longer sits on the critical path to a defensible
result, and it is scheduled after the analysis rather than before it.

**What the remaining time is for.** Section 7.10.6 shows the panel cannot settle
whether held-out degradation is metric-specific, and Section 7.12 shows the
reciprocal-match result rests on seven to thirteen calls per combination. Both
want more receptors rather than more analysis, and both are throughput problems
— which is what phase 11 addresses.


<div class="page-break"></div>

# 10. REFERENCES

1. Jumper, J., Evans, R., Pritzel, A. et al., "Highly Accurate Protein Structure Prediction with AlphaFold," *Nature*, Vol. 596, 2021, pp. 583–589.
2. Evans, R., O'Neill, M., Pritzel, A. et al., "Protein Complex Prediction with AlphaFold-Multimer," *bioRxiv*, 2022.
3. Abramson, J., Adler, J., Dunger, J. et al., "Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3," *Nature*, Vol. 630, 2024, pp. 493–500.
4. Wohlwend, J., Corso, G., Passaro, S. et al., "Boltz-1: Democratizing Biomolecular Interaction Modeling," *bioRxiv*, 2024.
5. Passaro, S., Corso, G., Wohlwend, J. et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction," Technical Report, MIT, 2025.
6. Hu, E. J., Shen, Y., Wallis, P. et al., "LoRA: Low-Rank Adaptation of Large Language Models," *International Conference on Learning Representations (ICLR)*, 2022.
7. Leviathan, Y., Kalman, M., Matias, Y., "Fast Inference from Transformers via Speculative Decoding," *International Conference on Machine Learning (ICML)*, 2023.
8. Ho, J., Salimans, T., "Classifier-Free Diffusion Guidance," *NeurIPS Workshop on Deep Generative Models and Downstream Applications*, 2021.
9. Song, Y., Dhariwal, P., Chen, M., Sutskever, I., "Consistency Models," *International Conference on Machine Learning (ICML)*, 2023.
10. DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," *arXiv preprint*, 2024.
11. Liu, H., Zaharia, M., Abbeel, P., "Ring Attention with Blockwise Transformers for Near-Infinite Context," *arXiv preprint*, 2023.
12. Meng, Y., Xia, M., Chen, D., "SimPO: Simple Preference Optimization with a Reference-Free Reward," *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
13. Rafailov, R., Sharma, A., Mitchell, E. et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.
14. Kolda, T. G., Bader, B. W., "Tensor Decompositions and Applications," *SIAM Review*, Vol. 51, No. 3, 2009, pp. 455–500.
15. Berman, H. M., Westbrook, J., Feng, Z. et al., "The Protein Data Bank," *Nucleic Acids Research*, Vol. 28, No. 1, 2000, pp. 235–242.
16. Mirdita, M., Schütze, K., Moriwaki, Y. et al., "ColabFold: Making Protein Folding Accessible to All," *Nature Methods*, Vol. 19, 2022, pp. 679–682.
17. Steinegger, M., Söding, J., "MMseqs2 Enables Sensitive Protein Sequence Searching for the Analysis of Massive Data Sets," *Nature Biotechnology*, Vol. 35, 2017, pp. 1026–1028.
18. Zhang, Y., Skolnick, J., "Scoring Function for Automated Assessment of Protein Structure Template Quality," *Proteins: Structure, Function, and Bioinformatics*, Vol. 57, No. 4, 2004, pp. 702–710.
19. Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural Network," *arXiv preprint*, 2015.
20. Kussie, P. H., Gorina, S., Marechal, V. et al., "Structure of the MDM2 Oncoprotein Bound to the p53 Tumor Suppressor Transactivation Domain," *Science*, Vol. 274, No. 5289, 1996, pp. 948–953.
21. Watson, J. L., Juergens, D., Bennett, N. R. et al., "De Novo Design of Protein Structure and Function with RFdiffusion," *Nature*, Vol. 620, 2023, pp. 1089–1100.
22. Dauparas, J., Anishchenko, I., Bennett, N. et al., "Robust Deep Learning-Based Protein Sequence Design Using ProteinMPNN," *Science*, Vol. 378, No. 6615, 2022, pp. 49–56.
23. Scarpellini, G., Shprints, R., Holderrieth, P. et al., "Few-step Cofolding with All-Atom Flow Maps," *arXiv preprint* arXiv:2606.08375, 2026.

<div class="page-break"></div>


# APPENDIX A — ABBREVIATIONS AND GLOSSARY

| Abbreviation | Full Form |
| :--- | :--- |
| MLA | Multi-Head Latent Attention |
| Fold-CP | Folding Context Parallelism |
| TMU | Triangular Multiplication Unit |
| OPM | Outer Product Mean |
| CFG | Classifier-Free Guidance |
| MLP | Multi-Layer Perceptron |
| ANE | Apple Neural Engine |
| MPS | Metal Performance Shaders |
| VRAM | Video Random Access Memory |
| PDB | Protein Data Bank |
| CIF | Crystallographic Information File |
| ODE | Ordinary Differential Equation |
| pLDDT | Predicted Local Distance Difference Test |
| TM-score | Template Modeling Score |
| RMSD | Root Mean Square Deviation |
| OOM | Out Of Memory |
| ipTM | Interface Predicted TM-score |
| MSA | Multiple Sequence Alignment |
| CP | CANDECOMP/PARAFAC tensor decomposition |
| ALS | Alternating Least Squares |
| PTM | Post-Translational Modification |
| AUC | Area Under the ROC Curve |
| CI | Confidence Interval |
| RCSB | Research Collaboratory for Structural Bioinformatics |


<div class="page-break"></div>

# APPENDIX B — REPRODUCTION OF THE REPORTED RESULTS

Every measurement in Section 7 is reproducible from the accompanying
repository. Each experiment writes a provenance manifest under `results/real/`
recording the code revision, seed, device, inference settings, exact command and
input checksums. Source code is not reproduced in this report; the commands
below identify the entry points.

| Result | Section | Command |
| :--- | :---: | :--- |
| Low-rank OPM, weight-space projection | 7.1 | `python src/opm_cp_projection.py` |
| Low-rank OPM, activation capture | 7.1 | `python src/opm_corpus_capture.py --max-per-layer 66` |
| Low-rank OPM, corpus distillation | 7.1 | `python src/opm_corpus_distill.py --ranks 32,64,128` |
| PDB binder benchmark (Boltz-1) | 7.2 | `python src/pdb_binder_benchmark.py` |
| Boltz-1 vs Boltz-2 comparison | 7.3 | `python src/compare_boltz1_boltz2.py` |
| PTM audit of the panel | 7.3 | `python src/audit_panel_ptms.py` |
| Panel construction screen | 7.4 | `python src/discover_pdb_binders.py --want 14` |
| Powered 22-receptor run | 7.4 | `python src/pdb_binder_benchmark.py --model boltz2 --work-dir artifacts/pdb_binders_b2_n22` |
| Run-to-run variance study | 7.5 | `python src/seed_variance_study.py --replicates 4` |
| Interface metrics beyond ipTM | 7.6 | `python src/rescore_interface_metrics.py` |
| Scramble-normalised ranking | 7.6 | `python src/scramble_normalised_ranking.py` |
| Which side of the interface carries the signal | 7.7 | `python src/interface_side_split.py` |
| Binding-site controls for the receptor side | 7.7 | `python src/receptor_site_controls.py` |
| Few-step distillation (DeCAF) scramble test | 7.8 | `python src/decaf_scramble_test.py --batch-size 12` |
| DeCAF replicate study | 7.9 | `python src/decaf_replicate_study.py --replicates 4` |
| Variance decomposition | 7.9 | `python src/variance_decomposition.py` |
| Held-out panel (post-cutoff receptors) | 7.10 | `python src/heldout_panel.py --batch-size 12` |
| Held-out replicate draws | 7.10 | `python src/heldout_replicates.py` |
| Held-out vs in-training comparison | 7.10 | `python src/heldout_vs_training.py` |
| Panel integrity (is every cognate a binder?) | 7.10 | `python src/panel_integrity.py` |
| Homology decontamination | 7.10 | `python src/homology_decontamination.py` |
| Backbone convergence (CA–CA geometry) | 7.11 | `python src/pose_convergence.py` |
| Binding-site correctness against the crystal | 7.11 | `python src/site_correctness.py --tag p1` |
| Physics rescoring (PRODIGY) | 7.11 | `python src/physics_rescore.py --structures DIR` |
| Reciprocal best-match filter | 7.12 | `python src/reciprocal_match.py` |
| Settings confound (full vs reduced) | 7.13 | `python src/settings_confound.py --batch-size 12` |
| Screening tool | 7.14 | `python src/screen_server.py` → `http://127.0.0.1:8765` |
| Held-out panel at full settings | 7.10, 7.15 | `python src/heldout_panel.py --base boltz1 --sampling-steps 200 --recycling-steps 3 --msa-depth 0 --run-tag _full` |
| Held-out vs in-training, both settings | 7.10 | `python src/heldout_at_full.py` |
| Automated readout search | 7.15 | `python src/readout_search.py --n-null 200` |
| Verification suite | 6 | `python -m pytest -q` |

Scripts that only re-analyse folds already on disk — `interface_side_split.py`,
`variance_decomposition.py`, `reciprocal_match.py`, `pose_convergence.py` and the
baselines — run in seconds and require no GPU. The entries that fold structures
(`heldout_panel.py`, `decaf_scramble_test.py`, `settings_confound.py`) are the
expensive ones; `settings_confound.py` additionally accepts `--sampling-steps`,
`--recycling-steps` and `--msa-depth` independently, which is the route to
resolving which of the three raised settings carries the effect of Section 7.13.

Long folding runs should be started through `python src/detach.py <logfile>
<command…>`, which places the run in its own session so it survives the parent
shell exiting. Three runs during this work were lost to session teardown before
that was adopted.

The four screens of Section 7.14 are preset buttons in that interface, and their
inputs are listed in `demo/EXAMPLES.md` for pasting. Because folds are cached by
target, peptide and settings, a screen already run returns in about two seconds;
deleting `artifacts/screen_fold_cache/` forces a genuine refold.

**Inference settings.** Two regimes appear in this report and the distinction is
load-bearing, because Section 7.13 measures the difference between them and finds
it accounts for a factor of three to seven in the standardised effect. No figure
should be read without knowing which regime produced it.

| Setting | Reduced regime | Full regime | Boltz default |
| :--- | :---: | :---: | :---: |
| Sampling steps | 10 | 200 | 200 |
| Recycling steps | 1 | 3 | 3 |
| MSA depth | 32 | full (up to 12,882 rows) | 8192 |
| Accelerator | CPU, later MPS | MPS | GPU |
| Diffusion seed | unseeded | unseeded | unseeded |

A full-settings fold takes 106 to 109 seconds on this laptop and is no slower at
full alignment depth (Section 7.13); the model distilled for ten steps folds the
whole 132-complex panel in about twenty minutes (Section 7.8).

**Sections 7.2 to 7.12 were folded entirely in the reduced regime**, which is why
every effect size in them is a lower bound. Section 7.13 folds the same panel in
the full regime and is the only direct comparison of the two. Section 7.14's fast
mode is the reduced regime with a model distilled for it, and its careful mode is
the full regime.

The reduced regime was adopted when folding ran on CPU and a full-settings fold
was prohibitive. It ceased to be necessary once the MPS path landed; the
measurements that continued in it after that point did so by inertia, which is
the substance of Section 7.13's finding against this work.


<div class="page-break"></div>

# CHECKLIST OF ITEMS FOR THE FINAL DISSERTATION / PROJECT / PROJECT WORK REPORT

<table style="width:100%; border-collapse:collapse; font-size:8.0pt; line-height:1.12;">
  <thead>
    <tr style="background:#0B2B52; color:#fff;">
      <th style="width:5%; padding:2px;">#</th>
      <th style="text-align:left; padding:2px;">Item</th>
      <th style="width:16%; padding:2px;">Yes / No</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:center">1</td><td>Final report neatly formatted with all elements required for a technical report</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">2</td><td>Cover page in proper format as given in Annexure A</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">3</td><td>Title page (inner cover page) in proper format</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">4</td><td>(a) Certificate from the Supervisor in proper format &nbsp; (b) Signed by the Supervisor</td><td style="text-align:center">(a) Yes<br>(b) Yes</td></tr>
    <tr><td style="text-align:center">5</td><td>Abstract within one page &nbsp; / &nbsp; Technical keywords specified</td><td style="text-align:center">Yes / Yes</td></tr>
    <tr><td style="text-align:center">6</td><td>Title appropriate, descriptive and precise; no uncommon abbreviations or acronyms</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">7</td><td>List of abbreviations / acronyms included</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">8</td><td>Report contains a summary of the literature survey</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">9</td><td>Contents include page numbers; pages numbered properly (Ch. 1 on page 1); figures numbered with captions at the bottom; tables numbered with titles at the top; captions proper; appendices numbered with appropriate titles</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">10</td><td>Conclusion based on discussion of the work</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">11</td><td>References given at the end; cited properly inside the text; all references cited in the body</td><td style="text-align:center">Yes</td></tr>
    <tr><td style="text-align:center">12</td><td>Format and content per the guidelines; not a printout of a presentation or a user manual; source code not included</td><td style="text-align:center">Yes</td></tr>
  </tbody>
</table>

<p style="font-size:9pt; margin-top:6px; margin-bottom:4px;"><b>Declaration by Student:</b>
I certify that I have properly verified all the items in this checklist and
ensure that the report is in proper format as specified in the course handout.</p>

<div class="signature-block">
    <div class="sig-col">
        <div class="sig-field"><b>Place:</b> Bangalore</div>
        <div class="sig-field"><b>Date:</b> August 2, 2026</div>
    </div>
    <div class="sig-col">
        <img class="sig-img" src="assets/sig_student.png" alt="Signature">
        <div class="sig-line"></div>
        <div class="sig-title">Signature of the Student</div>
        <div class="sig-field"><b>Name:</b> Akik Jana</div>
        <div class="sig-field"><b>ID No.:</b> 2024AB05287</div>
    </div>
</div>
