# Boltz Structure Prediction Codebase Exploration Report

## Overview
This report documents the architectural structure of the Boltz structure prediction model and its surrounding ecosystem. It details prediction entrypoints, validation targets, and the implementation details of four planned optimizations: MPS Native Execution, Low-Rank Pair Updates, CFG Distillation, and Neural Refinement.

---

## 1. Directory Structure & CLI Entrypoints

The repository has the following key layout:
```
/Users/akikjana/Documents/BiomolecularDesign/
├── boltz/                                  # Core Boltz model package
│   └── src/boltz/
│       ├── model/
│       │   ├── models/
│       │   │   └── boltz2.py               # Main Boltz2 LightningModule definition
│       │   ├── modules/
│       │   │   ├── diffusion.py            # Coordinate diffusion (v1)
│       │   │   ├── diffusionv2.py          # Coordinate diffusion (v2 / flow matching)
│       │   │   └── utils.py                # Autocast and tensor utilities
│       │   └── layers/
│       │       ├── attention.py            # Custom attention blocks (AttentionPairBias)
│       │       ├── attentionv2.py          # Attention with Pairformer bias
│       │       ├── pairformer.py           # Pairformer module blocks
│       │       └── triangular_mult.py      # Outgoing/Incoming Triangle Multiplication
│       └── data/
│           ├── mol.py                      # Symmetry correction & molecular logic
│           └── module/
│               └── inferencev2.py          # Boltz inference dataloaders
├── src/                                    # Optimization modules & tooling
│   ├── predict_structure.py                # CoreAI wrapper for running structure predictions
│   ├── low_rank_pair_representation.py     # Custom autograd low-rank pair updates
│   ├── cfg_distillation.py                 # Teacher/student vector field models for CFG
│   ├── train_neural_refiner.py             # Coordinate refiner training script
│   ├── speculative_flow_matching.py        # Speculative Euler integration framework
│   ├── dms_generation.py                   # Deep Mutational Scanning sequence library tool
│   └── benchmark_boltz_coreai.py           # Benchmarking tool (CPU vs MPS vs CoreAI)
└── tests/                                  # Test suites
    ├── test_dynamic_actual_samples.py      # E2E test running actual biological sequences
    ├── test_boltz_wrapper.py               # Tests BoltzModelWrapper in surrogate/real mode
    ├── test_boltz_modified_layers.py       # Tests AttentionPairBias & TMU layers
    ├── test_speculative_flow_matching.py   # Verifies speculative sampling performance
    └── test_dms_generation.py              # Verifies DMS library mutation scanning
```

### CLI Entrypoints & Execution Wrappers
1. **`src/predict_structure.py`**:
   - Class: `DynamicStructurePredictor`
   - Purpose: Performs structure prediction via Apple's AOT-compiled CoreAI runtime using `coreai.runtime`.
   - Action: Loads a model from `surrogate_model_dynamic.aimodel`, converts sequence strings to embeddings, projects target keys/values, manages state dictionary cache buffers (`cross_attn.k_cache`, `cross_attn.v_cache`), and runs predictions on the Neural Engine/GPU.
2. **`src/boltz_wrapper.py`**:
   - Classes: `BoltzModelWrapper`, `BoltzDraftModelWrapper`
   - Purpose: Dual-mode wrapper. If the real `boltz` package is installed, it runs all-atom predictions. If not, it falls back to a CPU surrogate coordinate generator.
3. **`tests/test_dynamic_actual_samples.py`**:
   - Running this executes structure predictions for binders of various lengths (e.g. insulin, hemoglobin fragments) against targets up to 1300 residues to demonstrate dynamic shapes without recompilation.

---

## 2. Prediction Invocation & Validation Targets

### Invoking Predictions
Predictions are invoked using the `DynamicStructurePredictor` API:
```python
from src.predict_structure import DynamicStructurePredictor

# Initialize predictor (loads CoreAI model once)
predictor = DynamicStructurePredictor(aimodel_path="path/to/model.aimodel")

# Predict 3D coordinates (returns np.ndarray of shape [1, L_binder, 3])
coords = predictor.predict(binder_seq="GIVEQCCTSICSLYQLENYCNFV", target_seq="GLVLIAFSQYL...")
```

For full-scale model runs, `BoltzModelWrapper` is used:
```python
from boltz_wrapper import BoltzModelWrapper
model = BoltzModelWrapper(use_gpu=True)
coords, info = model.predict_structure(sequence="MATEVLADIGSAKLR", target_pdb_path="target.pdb")
```

### Validation Targets
The codebase tests and validates predictions using several key biological structures:
* **Human Insulin**:
  - Code reference: `tests/test_dynamic_actual_samples.py`
  - Sequence: `"GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT"` (51 residues, monomer)
* **Hemoglobin**:
  - Code reference: `tests/test_dynamic_actual_samples.py`
  - Sequence: Subunit alpha (`"MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"`, 142 residues)
* **TNF-alpha (Tumor Necrosis Factor alpha)**:
  - Code reference: `tests/test_boltz_wrapper.py`, `tests/test_dms_generation.py`
  - RCSB PDB ID: `1TNF`
* **VEGFA (Vascular Endothelial Growth Factor A)**:
  - Code reference: `src/dms_generation.py`
  - RCSB PDB ID: `1FLT`

### Validation Data Locations
* **PDB Complex Files**: Standard structures are dynamically fetched from the RCSB Protein Data Bank (`https://files.rcsb.org/download/{pdb_id}.pdb`) via `urllib.request.urlretrieve` and cached locally inside `/tmp/biomolecular_design/` (e.g. `/tmp/biomolecular_design/TNF-alpha_1TNF.pdb`).
* **Hardcoded Sequences**: Test binder fragments (e.g. 50-residue insulin fragment, 90-residue hemoglobin fragment) and target receptor sequences (153-residue small target, 600-residue medium target, 1300-residue large target) are located directly inside `tests/test_dynamic_actual_samples.py`.

---

## 3. Optimization Features Implementation

### Feature 1: MPS Native Execution
To enable native execution on Apple Silicon (MPS backend) without crashing, the codebase addresses hardcoded CUDA references and autocast assumptions.

* **Current Implementation**:
  - `boltz/src/boltz/model/modules/utils.py` contains `autocast_device_type(device_type: str) -> str` which falls back to `"cpu"` if a device type is not supported by PyTorch autocast.
  - **Issues (Hardcoded CUDA)**:
    1. `boltz/src/boltz/model/models/boltz2.py`:
       - Lines 989, 998: `device="cuda" if torch.cuda.is_available() else "cpu"` (does not account for `"mps"` device).
    2. `boltz/src/boltz/model/modules/diffusion.py`:
       - Lines 694, 820: `with torch.autocast("cuda", enabled=False):`
    3. `boltz/src/boltz/model/modules/diffusionv2.py`:
       - Lines 513, 603: `with torch.autocast("cuda", enabled=False):`
    4. `boltz/src/boltz/model/layers/attention.py`:
       - Line 223: `with torch.autocast("cuda", enabled=False):`
    5. `boltz/src/boltz/model/layers/attentionv2.py`:
       - Line 99: `with torch.autocast("cuda", enabled=False):`
    6. `boltz/src/boltz/model/modules/trunkv2.py`:
       - Lines 311, 462: `with torch.autocast(device_type="cuda", enabled=False):`
  - **Issues (Float64 / Double Precision)**: Apple Silicon GPU architecture natively operates on Float32/Float16 and runs extremely slowly or throws exceptions when executing Float64/Double operations. All inputs and internal variables must be kept in Float32 (using `.float()` or `torch.float32` casts).

### Feature 2: Low-Rank Pair Updates
The Pairformer and Evoformer blocks update the high-dimensional pairwise representation $z \in [B, N, N, C_z]$ from sequence representations $s \in [B, N, C_s]$. The standard Outer Product Mean (OPM) computes a full-rank tensor product $[B, N, N, C_{mid}]$ before projection, requiring $O(N^2 \cdot C_{mid})$ activation storage during backpropagation, which causes out-of-memory errors on large sequences.

* **Implementation in `src/low_rank_pair_representation.py`**:
  - `LowRankTensorProduct(torch.autograd.Function)`: A custom autograd class that performs vectorized projection:
    $$U_{b, i, j, c} = \sum_{r=1}^d X_{b, i, r} Y_{b, j, r} W_{c, r}$$
    Its `backward` pass is highly optimized: instead of storing the $O(N^2 \cdot D_{pair})$ intermediate matrix, it reconstructs gradients by mapping back to the low-rank subspace (rank $d \ll D_{pair}$) via:
    $$M = \text{grad\_output} \times W \quad (\text{shape } [B, N, N, d])$$
    This drops activation VRAM usage from quadratic to linear.
  - `LowRankPairUpdater(nn.Module)`: Projects sequence features $s$ to factors $X$ and $Y$ of rank $d$ (default 16), then applies `LowRankTensorProduct`.
  - `FullRankPairUpdater(nn.Module)`: The baseline OPM implementation for comparison.
* **Testing (`src/test_low_rank_pair.py`)**:
  - Performs gradient checking (`gradcheck`) verifying correctness of the custom backward pass.
  - Benchmarks speed and memory. As $N$ scales from 100 to 1500, the activation memory of the Low-Rank updater is drastically lower than the Full-Rank OPM, maintaining high fidelity (low relative MSE) when trained to approximate the full-rank output.

### Feature 3: CFG Distillation
Classifier-Free Guidance (CFG) improves sample quality by blending conditional and unconditional vector fields during coordinate denoising:
$$v_{\text{guided}} = v_{\text{uncond}} + (1 + s)(v_{\text{cond}} - v_{\text{uncond}})$$
This requires evaluating the large neural network twice per denoising step, doubling prediction latency.

* **Implementation in `src/cfg_distillation.py`**:
  - `TeacherVectorField`: The baseline flow matching model. Supports conditioning feature dropout via a `cond_mask`.
  - `CFGDistilledVectorField`: The student network. Rather than running twice, it takes the guidance scale $s$ as a direct input (encoded via sinusoidal embedding) alongside $t$ and projects it into the joint time-scale MLP. It predicts $v_{\text{guided}}$ in a **single forward pass**.
  - `train_distilled_model`: Trains the student to approximate the teacher's CFG-guided field.
* **Speculative Flow Matching (`src/speculative_flow_matching.py`)**:
  - `SpeculativeFlowMatchingSampler`: Implements speculative execution. A fast draft model (like the student or a pruned Boltz model) draft integrates $K$ speculative steps (lookahead). The target model then verifies all $K$ steps in parallel via a single batched evaluation.
  - The sampler checks if the L2 distance between the draft and target vector fields is below a threshold (`tolerance`). If so, steps are accepted. It also integrates physical manifold constraints (forcing consecutive residues to exactly $3.80\text{ \AA}$) and soft repulsive forces to avoid steric clashes.

### Feature 4: Neural Refinement
Fast/quantized structure generators or distilled diffusion models can sometimes yield structural inaccuracies, including non-physical bond lengths or overlapping atomic coordinates (steric clashes).

* **Implementation in `src/train_neural_refiner.py`**:
  - `ResNetCoordinateRefiner`: A deep residual network that fuses sequence embeddings and noisy, coarse 3D coordinates. It outputs 3D coordinate deltas to correct geometry.
  - **Loss Function**: `compute_supervised_loss` computes:
    1. **Coordinate L2 loss**: Minimizes Euclidean distance between predicted and ground-truth coordinates.
    2. **Pairwise Distance Matrix MSE loss**: Implicitly enforces bond lengths and steric repulsion by forcing the network to match the true physical distance matrix profiles.
* **Results**:
  - Resolves steric overlaps, restoring minimum non-consecutive atom distance to $>3.8\text{ \AA}$.
  - Corrects bond length errors, reducing mean consecutive residue distance error from $\sim0.24\text{ \AA}$ to $\sim0.02\text{ \AA}$.

---

## 4. Architectural Integration & Recommendations

To integrate these four features into the core Boltz library:

1. **MPS Support**:
   - Import `autocast_device_type` inside `boltz/src/boltz/model/modules/diffusion.py`, `diffusionv2.py`, `trunkv2.py`, and layers `attention.py`/`attentionv2.py`.
   - Replace all instances of `torch.autocast("cuda", ...)` with `torch.autocast(autocast_device_type(device.type), ...)`.
   - Update device definitions in `boltz2.py`:
     ```python
     device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
     ```

2. **Low-Rank Pair Integration**:
   - Replace the `OuterProductMean` block inside the `MSAModule` or `PairformerModule` with `LowRankPairUpdater`.
   - Specifically, in `boltz/src/boltz/model/layers/pairformer.py`, swap OPM references with `LowRankPairUpdater`.

3. **CFG Student & Speculative Integration**:
   - Replace standard Euler integration in `AtomDiffusion.sample()` (inside `diffusionv2.py`) with `SpeculativeFlowMatchingSampler`.
   - The student network (`CFGDistilledVectorField`) can be used as the fast draft model, while `Boltz2` acts as the target verification model.

4. **Neural Refiner Hook**:
   - Inject `ResNetCoordinateRefiner` at the very end of structure prediction. After `AtomDiffusion` finishes and returns the coarse C-alpha backbone, pass the predicted coordinates and token embeddings through the trained refiner to yield the final corrected coordinate trace.
