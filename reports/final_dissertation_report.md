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
cannot, at 8.6 times the effect-to-noise ratio, yielding a concrete
recommendation for how such candidates should be ranked.

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
&nbsp;&nbsp;&nbsp;&nbsp;7.4 Powered Run: ipTM Tracks Composition, Not Binding ....................... 18  
&nbsp;&nbsp;&nbsp;&nbsp;7.5 Measurement Reproducibility ............................................. 19  
&nbsp;&nbsp;&nbsp;&nbsp;7.6 Interface pLDDT Ranks Binders Where ipTM Does Not ....................... 21  
&nbsp;&nbsp;&nbsp;&nbsp;7.7 Localising the Interface-pLDDT Signal ................................... 24  
&nbsp;&nbsp;&nbsp;&nbsp;7.8 Few-Step Distillation, and What It Reveals About the Negatives .......... 26  
&nbsp;&nbsp;&nbsp;&nbsp;7.9 Variance Decomposition and the Reproducibility of the Few-Step Model .... 28  
8. CONCLUSIONS AND RECOMMENDATIONS .......................................... 31  
&nbsp;&nbsp;&nbsp;&nbsp;8.1 Conclusions ............................................................. 31  
&nbsp;&nbsp;&nbsp;&nbsp;8.2 Recommendations ......................................................... 32  
9. FUTURE PLAN .............................................................. 35  
10. REFERENCES .............................................................. 38  
APPENDIX A — ABBREVIATIONS AND GLOSSARY ..................................... 40  
APPENDIX B — REPRODUCTION OF RESULTS ........................................ 42  
CHECKLIST OF ITEMS FOR THE FINAL REPORT ..................................... 44  

&nbsp;

## LIST OF FIGURES

Figure 1: Core Modules in Boltz-Fast ......................................... 4  
Figure 2: C-alpha Backbone 3D Coordinate Plot ............................... 10  
Figure 3: Ray-Traced Protein Ribbon Model Rendering ......................... 11  
Figure 4: ANE-Accelerated 3D Insulin Backbone Visualization ................. 12  
Figure 5: Post-Diffusion Neural Coordinate Refinement Comparison ............ 15  
Figure 6: ipTM and Interface pLDDT on the Scramble Control .................. 22  

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
5. **Validate each of the above against pretrained checkpoints and
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
   (default 8192). These were forced by CPU-only execution on a laptop. The
   settings confound is stated, not resolved.
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
work. Section 8 draws conclusions and recommendations, and Section 9 sets out
the remaining plan.

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

### 7.4 Powered Run: ipTM Tracks Composition, Not Binding

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

**Conclusion.** The powered run resolves the question and the answer remains
negative, now with a mechanism: ipTM responds to peptide composition and is
indifferent to sequence order. A screening reference must prefer a binder to its
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


### 7.6 Interface pLDDT Ranks Binders Where ipTM Does Not

Sections 7.2 to 7.5 indict **ipTM**, not the predicted structures — 132 of which
remained on disk. Re-scoring them with other interface measures required no
further folding, and changes the practical conclusion of this dissertation.

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
better *readout* of one model rather than an independent second opinion. The
practical implication is unchanged: the information required to rank these
binders is present in the prediction, and ipTM discards it.

**Correction — this result is model-dependent.** Section 7.8 re-runs the
identical pairs on stock Boltz-1 and on a few-step-distilled model. Interface
pLDDT separates a cognate from its own scramble strongly on Boltz-2 (+3.30) and
on DeCAF-Boltz (+9.54), but **not on Boltz-1** (+1.54, p = 0.067, with a
confidence interval spanning zero, and receptor side p = 0.43). The
recommendation to rank on interface pLDDT holds for the models on which it was
measured; it does not generalise to every cofolding model, and this section
should not be read as though it does. Section 7.8 also shows that receptor
specificity, left open above, **is** established on the few-step model.

<div class="page-break"></div>

### 7.7 Localising the Interface-pLDDT Signal

Section 7.6 establishes that interface pLDDT responds to sequence order. It does
not establish *what* it is responding to. Two further experiments address the
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
peptide foldability does not account for this term. It is the strongest evidence
in this dissertation that the metric reads something about the interaction.

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

**Strengthened.** The receptor responds to which peptide it is given, and peptide
foldability cannot explain that.

**Not established.** That the response is localised to the binding site.

Neither result changes the recommendation of Section 7.6 — rank on interface
pLDDT rather than ipTM — but the mechanism behind it is only partly
characterised, and one component of the pooled metric is now known to be largely
peptide foldability.

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

*Section 7.6 is model-dependent.* Interface pLDDT is the best readout on Boltz-2
and on DeCAF, but on Boltz-1 it does not reach significance while ipTM does. The
correction is recorded in Section 7.6 itself.

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
3. Which readout carries the signal varies by model.

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

#### 7.9.5 Limitations

Boltz-1 has no replicate study of its own and borrows the Boltz-2 noise term, so
its rows are an assumption rather than a measurement. The replicate design covers
4 receptors, chosen in Section 7.5 to span the observed outcomes, not the full
panel. The mixed model assumes additive receptor effects and homoscedastic
residuals; neither was tested.

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

Fourth, and positively, **the information ipTM discards is recoverable**
(Section 7.6). Re-scoring the same structures shows interface pLDDT
distinguishes a cognate from its own scramble (p < 0.0001, reproducing in 100%
of simulated re-runs) at 8.6 times ipTM's effect-to-noise ratio. The failure is
specific to ipTM as a readout, not general to the prediction.

Section 7.7 then localises that signal, and qualifies it. Splitting the metric by
chain shows most of its peptide-side component is whole-chain peptide confidence
rather than interface confidence — the foldability objection is partly correct.
What survives is the receptor side: the receptor's own residues are placed more
confidently when given the cognate peptide (+2.38, p = 6e-5), which peptide
disorder cannot explain. An alanine scan of the binding site points the same way
but falls short of significance (p = 0.244, requiring roughly six times the
panel), so the response is not yet shown to be site-localised.

Fifth, **few-step distillation changes the picture materially** (Section 7.8).
Re-running the panel on a model distilled for 10-step sampling raises both order
sensitivity and receptor specificity by 5–6× over its own teacher at the same
budget, and establishes receptor specificity for the first time in this work
(interface pLDDT rank 1.73 against chance 2.50, p = 0.0042, clearing Bonferroni).
The de-confounding arm attributes this to the distillation rather than to the
base model. It also qualifies two earlier conclusions: stock models at reduced
sampling retain weak but real signal rather than none, and the interface-pLDDT
recommendation of Section 7.6 is model-dependent.

**Overall.** The efficiency techniques transfer as engineering. The ipTM-based
accuracy claims do not survive measurement on the models originally tested — but
the structures carry a usable signal once read correctly, and a model built for
the sampling budget recovers substantially more of it. Reporting this is the substantive contribution: each negative
result is established with controls, held-out splits, power analysis and
replication, and each redirects effort away from a direction that would not have
worked.

## 8.2 Recommendations

1. **Rank on interface pLDDT, not ipTM — and not on pDockQ for short
   peptides.** This is the one recommendation here that improves a result rather
   than qualifying it. pDockQ actively cancels the signal in this regime,
   because its contact term favours scrambled peptides.
2. **Report replicate-averaged confidence for per-receptor claims.**
   Section 7.9 refines this: rank instability generalises to the few-step model,
   so per-receptor rankings do require averaging — but the variance decomposition
   finds aggregate discriminability signal-limited rather than noise-limited, so
   averaging will not rescue a weak metric at the population level. On this
   evidence, 9 to 16 replicate folds per complex are required before a
   per-receptor ranking is meaningful. Practitioners ranking binders on a single
   AlphaFold or Boltz run should treat those rankings as provisional.
3. **Always include a scramble control.** Composition and length alone reproduce
   most of the apparent discrimination between a cognate and a decoy. Without a
   composition-matched control, a benchmark cannot distinguish binder
   recognition from composition sensitivity.
4. **Screen benchmark panels for post-translational modifications.** Seven of 25
   candidate complexes were PTM-dependent; folding the canonical sequence makes
   those "positives" non-binding, which silently penalises the metric under test.
5. **Do not treat operator-level memory savings as drop-in.** A substituted
   operator should be validated against pretrained weights on real activations
   before its microbenchmark saving is claimed.
6. **Deploy on GPU before extending the biological claims.** The replicate
   folding required by recommendation 1 is a 1,200–2,100 fold workload for this
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
| 6 | Production GPU Scaling | 01 Aug 2026 – 20 Aug 2026 | Deploy on CUDA and run NCCL scale tests | **PENDING** |
| 7 | Powered Specificity Run | 01 Aug 2026 – 02 Aug 2026 | Complete the 22-receptor Boltz-2 benchmark (Section 7.4) | **COMPLETED** |
| 8 | Measurement Reproducibility | 02 Aug 2026 – 02 Aug 2026 | Quantify ipTM run-to-run variance (Section 7.5) | **COMPLETED** |
| 9 | Replicate-Averaged Rerun (GPU) | 03 Aug 2026 – 20 Aug 2026 | 9-16 replicate folds per complex; requires CUDA deployment | **PENDING** |
| 10 | Final Thesis | 03 Aug 2026 – 28 Aug 2026 | Consolidate results and write the dissertation | **IN PROGRESS** |

**Change of plan, and why.** The original phase 5 (CUDA/NCCL scaling) was
deferred in favour of validating the scoring reference, because the binder
screen planned for phase 6 depends on it: a screen ranked by a signal that does
not discriminate binders would produce candidates with no meaning regardless of
how fast it ran. That validation (Section 7.2) showed the dependency does not
hold, which is why the remaining plan prioritises the powered specificity run
over the TNF-alpha screen. Scaling work remains valuable for throughput but no
longer sits on the critical path to a defensible result.


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
| Verification suite | 6 | `python -m pytest -q` |

**Inference settings used throughout Section 7**, stated here so that no figure
in this report is read as a production-setting result:

| Setting | Value used | Boltz default |
| :--- | :---: | :---: |
| Sampling steps | 10 | 200 |
| Recycling steps | 1 | 3 |
| MSA depth | 32 | 8192 |
| Accelerator | CPU | GPU |
| Diffusion seed | unseeded | unseeded |


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
