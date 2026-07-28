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
binder ranking with X% top-k recall (Spearman ρ) at 7.95 ms/candidate"*.
`src/benchmark_surrogate_vs_reference.py` exists to produce exactly that number
— plug in a real Boltz-2 scorer and the surrogate, and report both halves
together. Until then, treat the latency figures as throughput characterization
only, not as evidence of equivalence.

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
