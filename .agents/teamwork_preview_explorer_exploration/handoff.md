# Boltz Model Exploration & Optimization Report

## 1. Observation
Based on a systematic exploration of `/Users/akikjana/Documents/BiomolecularDesign`, the following exact files, lines, and components were identified:

### A. Boltz Model Implementation & Network Architecture
* **Main Model Class**: Located in `/Users/akikjana/Documents/BiomolecularDesign/boltz/src/boltz/model/models/boltz2.py` (`class Boltz2(LightningModule)`).
  * **Trunk Embeddings & Recycling**: The inputs are featurized in `self.input_embedder(feats)` (Line 415), initialized via `self.s_init` and `self.z_init` (Lines 418–424), and passed through a recycling loop for `recycling_steps + 1` rounds (Lines 440–490).
  * **Evoformer-like Attention & Pairformer Blocks**: Inside the recycling stack, pair representation updates are computed via the template module (Line 465), the MSA module (Line 474), and the Pairformer module (`PairformerModule` at Line 484, implemented in `/Users/akikjana/Documents/BiomolecularDesign/boltz/src/boltz/model/layers/pairformer.py`).
  * **Output Modules**: After the recycling stack, the model extracts predictions using `self.distogram_module` (Line 492), `self.bfactor_module` (Line 293), `self.confidence_module` (Line 306), `self.affinity_module` (Line 324), and structural coordinates via `self.structure_module` (Line 276).

### B. Diffusion & Flow-Matching Coordinate Generation
* **Implementation Path**: `/Users/akikjana/Documents/BiomolecularDesign/boltz/src/boltz/model/modules/diffusionv2.py`.
  * **Module Class**: `class AtomDiffusion(Module)` (Line 180).
  * **Coordinate Integration**: Denoising coordinates is performed inside the `sample()` loop (Lines 295–530). At each step, a noisy coord state is updated via the preconditioned network output (`preconditioned_network_forward` at Line 251):
    ```python
    denoised_coords = (
        self.c_skip(padded_sigma) * noised_atom_coords
        + self.c_out(padded_sigma) * r_update
    )
    ```
    where `r_update` represents the predicted vector field computed by the sequence-local atom attention decoder inside `DiffusionModule` (Line 166).

### C. Coordinate Refining Logic
* **Implementation Path**: `/Users/akikjana/Documents/BiomolecularDesign/src/train_neural_refiner.py`.
  * **Refiner Architecture**: `class ResNetCoordinateRefiner(nn.Module)` (Line 13) implements an MLP-based residual coordinate updater:
    ```python
    refined_coords = coarse_coords + deltas
    ```
  * **Loss Function**: `compute_supervised_loss()` (Line 60) enforces physical structures using:
    1. Coordinate L2 loss: Direct alignment of predicted coordinate trace to experimental structures.
    2. Pairwise Distance Matrix MSE loss: Implicitly enforces bond lengths and avoids steric overlaps.

### D. Benchmarks & Validation Datasets
* **Surrogate CoreAI Benchmark**: `/Users/akikjana/Documents/BiomolecularDesign/src/benchmark_boltz_coreai.py` compares Standard PyTorch CPU, PyTorch MPS, and CoreAI AOT Compiled FP8 + KV-Cached models (`/Users/akikjana/Documents/BiomolecularDesign/surrogate_model.aimodel`) over 200 trials.
* **Actual Sequence Verification**: `/Users/akikjana/Documents/BiomolecularDesign/tests/test_dynamic_actual_samples.py` validates inference latencies using dynamic shapes for sequences such as Human Insulin and Hemoglobin.
* **Speculative Flow Sampler Test**: `/Users/akikjana/Documents/BiomolecularDesign/tests/test_speculative_flow_matching.py` verifies the speedups and bond lengths for the speculative flow integration.
* **Low-Rank Memory/Speed Test**: `/Users/akikjana/Documents/BiomolecularDesign/src/test_low_rank_pair.py` profiles O(N^2) memory reduction.
* **CFG Distillation Test**: `/Users/akikjana/Documents/BiomolecularDesign/src/run_distillation.py` benchmarks teacher vs student model CFG passes.
* **Validation Datasets**: Validation is dynamically configured via PyTorch Lightning validators and dataset-specific validator mappers in `boltz2.py` (Lines 117–123, 1009–1020).

### E. Apple Silicon CPU/MPS Compatibility Issues
We identified multiple blockers that cause runtime crashes or force CPU execution fallback:
1. **Device Placement Bug**: In `boltz2.py` (Lines 989, 998), the helper norm methods return a tensor placed on CPU or CUDA:
   ```python
   return torch.tensor(
       0.0, device="cuda" if torch.cuda.is_available() else "cpu"
   )
   ```
   If the model is placed on `mps`, this creates a device mismatch crash during backward pass parameter summation.
2. **Hardcoded CUDA Autocasting**: Standard Boltz code hardcodes `torch.autocast("cuda", enabled=False)` at `/Users/akikjana/Documents/BiomolecularDesign/boltz/src/boltz/model/modules/diffusionv2.py:513` and `:603`, as well as `primitives.py:106`, `attention.py:223`, and other locations. These cause warnings or runtime exceptions on MPS-only systems.
3. **Float64 Operations Blocker**: Sharded ring operations in `fold_cp_sharding.py` and `boltz_fast.py` (Lines 182–185) cast query/key/value/bias tensors to `.double()`. PyTorch MPS does not support Float64 operations natively, causing either a direct runtime error or severe execution slowdown.
4. **CUDA-Specific Kernel Compilation**: `self.use_kernels = False` is hardcoded to trigger if CUDA major version is not >= 8.0, preventing GPU acceleration of the custom kernels on macOS.

---

## 2. Logic Chain
1. The standard Outer Product Mean (OPM) computes intermediate outer-product attention matrices of size $O(N^2 \cdot D_{mid})$ which must be held in VRAM for backward propagation. The custom autograd implementation in `/Users/akikjana/Documents/BiomolecularDesign/src/low_rank_pair_representation.py` avoids saving these quadratic-sized activations by projecting inputs into a low-rank subspace and performing gradient updates on-the-fly, reducing VRAM scaling from $O(N^2 \cdot D_{pair})$ to $O(N \cdot r)$.
2. Classifier-Free Guidance (CFG) requires double the evaluations per flow integration step (one conditional pass, one unconditional pass). Distilling this process into a single-pass student network that takes guidance scale `s` as conditioning (`CFGDistilledVectorField` in `cfg_distillation.py`) yields a 2x inference speedup and halves the peak activation footprint.
3. Rapid unoptimized surrogate models or fast ODE integrations produce coordinates containing physical clashes and bond distance errors. Resolving these via standard physical relaxation (e.g. OpenMM) is highly CPU-intensive and slow. The MLP-based `ResNetCoordinateRefiner` corrects this in a single forward pass by learning coordinate delta updates (`coarse_coords + deltas`) optimized via pairwise distance matrix MSE loss.
4. Running model training/inference on Mac MPS-only devices will fail due to device mismatch bugs in `boltz2.py` (norm tensor creation on `"cpu"` / `"cuda"`), unsupported Float64 casting in simulated parallel attention blocks, and hardcoded `torch.autocast("cuda", ...)` calls. Correcting these hardcoded device strings and converting double precision tensors to float32 will enable native, accelerated MPS execution.

---

## 3. Caveats
* **Kernel Execution**: We assumed Triton/CUDA custom attention kernels cannot be compiled on macOS/MPS. If custom Metal Performance Shaders (MPS) kernels are implemented, this would bypass the kernel fallback.
* **Mock Datasets**: Benchmarks in `run_distillation.py` and `train_neural_refiner.py` are profiled on mock helical coordinates rather than full multi-chain experimental PDB files. Absolute speeds on raw PDB inputs may vary due to sequence featurization and data loading overheads.

---

## 4. Conclusion
The Boltz-2 model can be significantly optimized for edge deployment on Apple Silicon. By resolving the identified MPS-incompatible operations, integrating low-rank tensor updates, adopting distilled single-pass CFG, and appending a neural coordinate refiner, the codebase can run natively on Mac MPS and Neural Engine hardware with a minimal memory footprint.

### Actionable Optimization Recommendations:
1. **Low-Rank Pair Integration**: Replace the Evoformer/Pairformer OPM block with `LowRankPairUpdater` and its custom `LowRankTensorProduct` autograd function to reduce pair representation memory overhead.
2. **Classifier-Free Guidance Distillation**: Implement the single-pass student vector field configuration (`CFGDistilledVectorField` from `cfg_distillation.py`) to bypass the teacher's double-pass CFG requirement.
3. **Neural Coordinate Refinement**: Integrate the `ResNetCoordinateRefiner` block as a post-generation refinement layer to correct fast-surrogate predicted coordinates, removing physical clashes in milliseconds.
4. **Apple Silicon Code Fixes**:
   * Replace `device="cuda" if torch.cuda.is_available() else "cpu"` in `boltz2.py` (Lines 989, 998) with `device=module.device` or `self.device`.
   * Replace all hardcoded `torch.autocast("cuda", ...)` calls with the dynamic utility `torch.autocast(autocast_device_type(device.type), ...)` to support MPS/CPU execution gracefully.
   * Remove `.double()` (Float64) castings in `boltz_fast.py` and `fold_cp_sharding.py`, replacing them with `.float()` or `.half()` to prevent MPS float64 execution crashes.

---

## 5. Verification Method
1. **Norm Placement Bug Verification**:
   Inspect `/Users/akikjana/Documents/BiomolecularDesign/boltz/src/boltz/model/models/boltz2.py` at lines 989 and 998. The line `device="cuda" if torch.cuda.is_available() else "cpu"` confirms the placement discrepancy on MPS.
2. **Autocast Verification**:
   Check `/Users/akikjana/Documents/BiomolecularDesign/boltz/src/boltz/model/modules/diffusionv2.py` at lines 513 and 603. The hardcoded `"cuda"` autocast is visible:
   ```python
   with torch.autocast("cuda", enabled=False):
   ```
3. **Quantized / Low-Rank Functional Test**:
   Execute the low-rank profiling script:
   ```bash
   python src/test_low_rank_pair.py
   ```
   This verifies the gradient check accuracy and reports the physical memory savings between full-rank and low-rank tensors.
4. **Coordinate Refinement Run**:
   Verify coordinate refinement:
   ```bash
   python src/train_neural_refiner.py
   ```
   This generates `backbone_3d_refinement.png` and outputs training statistics showing consecutive bond errors and clash resolution.
