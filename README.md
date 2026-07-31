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

Caveats: Boltz-1 rather than Boltz-2, 10 sampling steps, 1 recycling, MSA
subsampled to depth 32, and n = 11 receptors. A stronger configuration may do
better; this measures what this pipeline provides.

Reproduce: `python src/pdb_binder_benchmark.py`

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
