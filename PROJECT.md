# Project: Boltz Structure Prediction Architecture Optimization for Apple Silicon

## Architecture
The Boltz structure prediction model comprises:
1. **Trunk and Embedders**: Featurization of inputs and Pairformer-based recycling.
2. **Diffusion / Flow-matching Sampler**: Denoising coordinates via single/double evaluations.
3. **Structure and Confidence Decoders**: Predicting 3D coordinates and validation scores.

We will optimize the model along three axes:
* **MPS Native Execution**: Replacing hardcoded CUDA-only references and Float64 casting with dynamic MPS/CPU checks and Float32.
* **Low-Rank Pair Updates**: Projecting high-dimensional Pairformer O(N^2) representations into a low-rank subspace to drop memory usage.
* **CFG Distillation & Neural Refinement**: Speeding up sampling via distilled single-pass flow matching and repairing structural issues using a fast coordinate refiner.

## Code Layout
* `boltz/src/boltz/model/models/boltz2.py`: Lightning module and global entrypoint.
* `boltz/src/boltz/model/modules/diffusionv2.py`: Diffusion/flow-matching denoising logic.
* `src/low_rank_pair_representation.py`: Custom autograd low-rank pair representation.
* `src/cfg_distillation.py`: CFG student vector field model.
* `src/train_neural_refiner.py`: ResNetCoordinateRefiner implementation.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 1 | E2E Test Suite Development | Create requirement-driven E2E tests (Tiers 1-4) assessing RMSD, pLDDT, latency, and memory | None | ✅ DONE |
| 2 | Apple Silicon MPS Compatibility | Repair device norms, dynamic autocast wrappers, and Float64 casts for native MPS run | M1 | ✅ DONE |
| 3 | Low-Rank Pair Integration | Replace full-rank Evoformer/Pairformer OPM blocks with LowRankPairUpdater | M2 | ✅ DONE |
| 4 | CFG Distillation Integration | Integrate distilled single-pass student vector field into flow-matching step | M2 | ✅ DONE |
| 5 | Neural Coordinate Refinement | Hook up ResNetCoordinateRefiner to correct coordinates post-diffusion | M3, M4 | ✅ DONE |
| 6 | E2E Integration and Adversarial | Run the E2E verification, generate Tier 5 adversarial cases, pass Forensic Audit | M1, M5 | ✅ DONE |

## Interface Contracts
### `LowRankPairUpdater` Integration
* **Input**: Pair representation tensor `z` of shape `[B, N, N, C_z]`
* **Output**: Updated pair representation `z_out` of shape `[B, N, N, C_z]` with custom autograd checkpointing.

### `CFGDistilledVectorField` Integration
* **Input**: Denoising state, time/sigma step, guidance scale parameter `s`
* **Output**: Single-pass vector field prediction, skipping double evaluations of standard CFG.

### `ResNetCoordinateRefiner` Integration
* **Input**: Coarse coordinates `[1, L, 3]` from draft/fast prediction
* **Output**: Refined coordinates `[1, L, 3]` with adjusted bond lengths and minimized clashes.
