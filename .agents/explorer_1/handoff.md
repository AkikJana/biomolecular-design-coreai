# Handoff Report

## 1. Observation
- The entrypoints for running structure predictions include `src/predict_structure.py`, which implements `DynamicStructurePredictor` (lines 10-95) using `coreai.runtime`.
- The main model definition is in `boltz/src/boltz/model/models/boltz2.py` as `class Boltz2(LightningModule)` (lines 41-723).
- Verification targets and sequences are located in `tests/test_dynamic_actual_samples.py` (lines 22-54), specifying Human Insulin monomer/fragment sequences and Hemoglobin alpha subunit sequences.
- Local PDB structures are downloaded using `urllib.request.urlretrieve` inside `src/dms_generation.py` (lines 20-47), referencing `1TNF` for TNF-alpha and `1FLT` for VEGFA, and cached under `/tmp/biomolecular_design/`.
- Hardcoded `"cuda"` and `device` assumptions are found across several codebase files:
  1. `boltz/src/boltz/model/models/boltz2.py` (Lines 989, 998: `device="cuda" if torch.cuda.is_available() else "cpu"`).
  2. `boltz/src/boltz/model/modules/diffusion.py` (Lines 694, 820: `with torch.autocast("cuda", enabled=False):`).
  3. `boltz/src/boltz/model/modules/diffusionv2.py` (Lines 513, 603: `with torch.autocast("cuda", enabled=False):`).
  4. `boltz/src/boltz/model/layers/attention.py` (Line 223: `with torch.autocast("cuda", enabled=False):`).
  5. `boltz/src/boltz/model/layers/attentionv2.py` (Line 99: `with torch.autocast("cuda", enabled=False):`).
  6. `boltz/src/boltz/model/modules/trunkv2.py` (Lines 311, 462: `with torch.autocast(device_type="cuda", enabled=False):`).
- Low-Rank Pair representation is implemented in `src/low_rank_pair_representation.py` using custom autograd function `LowRankTensorProduct(torch.autograd.Function)` (lines 5-70) and module `LowRankPairUpdater(nn.Module)` (lines 72-128).
- CFG Distillation is implemented in `src/cfg_distillation.py` (lines 174-236) using student vector field network `CFGDistilledVectorField` which accepts `s` (guidance scale) as an input tensor, alongside helper code `SpeculativeFlowMatchingSampler` in `src/speculative_flow_matching.py` (lines 34-245).
- Neural Refinement is implemented in `src/train_neural_refiner.py` using the `ResNetCoordinateRefiner` module (lines 13-58) trained via direct coordinate MSE and pairwise distance MSE (lines 60-86).

## 2. Logic Chain
1. **Entrypoints & Run Mechanics**: Predictions can be invoked via `DynamicStructurePredictor` (using CoreAI runtime, loading `surrogate_model_dynamic.aimodel`) or `BoltzModelWrapper` (Dual-mode, supporting real boltz weights or surrogate coordinates).
2. **Validation Target & Data**: Biological test sequences (Insulin and Hemoglobin) are hardcoded in `tests/test_dynamic_actual_samples.py`. Complex targets (TNF-alpha and VEGFA) are retrieved using `src/dms_generation.py` which retrieves structural data from the RCSB database and caches them under `/tmp/biomolecular_design/`.
3. **MPS Execution Optimization**: Standard PyTorch execution on Apple Silicon requires replacing the hardcoded autocast environments (autocasting `cuda` explicitly) with the `autocast_device_type(device_type)` wrapper defined in `boltz/src/boltz/model/modules/utils.py`. The `device="cuda" if torch.cuda.is_available() else "cpu"` checks in `boltz2.py` must include a check for `torch.backends.mps.is_available()`.
4. **Low-Rank Pair Updates Optimization**: Activation memory savings are achieved by replacing the full Evoformer/Pairformer OPM step with `LowRankPairUpdater`, which computes outer product matrices within a low-rank subspace and bypasses saving quadratic-sized intermediate representations using custom backpropagation.
5. **CFG Distillation Optimization**: Double evaluations of the vector field model are avoided by embedding $s$ directly in the student vector field, enabling $v_{\text{guided}}$ to be calculated in a single forward pass.
6. **Neural Refinement Optimization**: Coarse or clashing outputs are corrected post-diffusion using the residual MLP `ResNetCoordinateRefiner` module trained via data-driven supervised loss functions.

## 3. Caveats
- Direct execution benchmarks on native MPS could not be generated as local terminal command execution prompts timed out waiting for user input.
- Real weights from the full Boltz-1 checkpoint are assumed to be loaded in production, while local CPU execution defaults to surrogate prediction modes when the `boltz` module is not installed.

## 4. Conclusion
The Boltz structure prediction codebase contains all the necessary layout modules and wrappers for integration. Optimizations can proceed by resolving hardcoded autocasts and device strings to support native MPS, replacing Evoformer OPM layers with `LowRankPairUpdater`, implementing `CFGDistilledVectorField` single-pass sampling, and using the `ResNetCoordinateRefiner` post-processing step to correct the coordinate traces.

## 5. Verification Method
- Independent verification of the compiled exploration report can be done by checking:
  - File existence and content at: `/Users/akikjana/Documents/BiomolecularDesign/.agents/sub_orch_e2e_testing/exploration_report.md`
- Verification of optimization components can be done by running their individual test scripts:
  - Speculative Flow Matching: `python tests/test_speculative_flow_matching.py`
  - Low-Rank Pair: `python src/test_low_rank_pair.py`
  - Neural Refiner: `python src/train_neural_refiner.py`
  - Dynamic actual samples test: `python tests/test_dynamic_actual_samples.py`
