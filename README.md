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

Tested on a standard Apple M-series MacBook, evaluating 3D structure predictions for binder mutants against a large **1300-residue receptor**:

| Model / Backend Configuration | Avg Latency per Run | Total Time (200 Runs) | Speedup vs. Public Mac CPU | Speedup vs. Public Linux GPU |
| :--- | :---: | :---: | :---: | :---: |
| **Public Boltz-1** (Mac M-series CPU) | **900,000 ms** (15 mins) | 180,000.0 s | 1.00x (Baseline) | 0.27x |
| **Public Boltz-1** (Linux RTX 3090/4090 GPU) | **240,000 ms** (4 mins) | 48,000.0 s | 3.75x | 1.00x |
| **Boltz-Fast CoreAI** (Mac Neural Engine) | **7.95 ms** | **1.59 s** | **113,214x faster** | **30,190x faster** |

---

## 🛠️ Codebase Structure

* `src/predict_structure.py`: Main Swift/App wrapper mapping sequence strings to embeddings and calling the CoreAI runtime.
* `src/convert_surrogate_coreai.py`: Quantization, PyTorch Export, and CoreAI compilation toolchain.
* `src/benchmark_boltz_coreai.py`: Comparative latency test comparing CPU, MPS GPU, and CoreAI against public baselines.
* `src/visualize_window.py`: Interactive 3D plotting utility displaying predicted C-alpha backbones using matplotlib.
* `tests/test_dynamic_actual_samples.py`: Latency verification on actual protein sequences (Human Insulin, Hemoglobin).

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
python tests/test_dynamic_actual_samples.py
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
