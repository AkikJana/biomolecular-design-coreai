# Scope: Implementation Track

## Architecture
The Boltz structure prediction model comprises:
1. **Trunk and Embedders**: Featurization of inputs and Pairformer-based recycling.
2. **Diffusion / Flow-matching Sampler**: Denoising coordinates via single/double evaluations.
3. **Structure and Confidence Decoders**: Predicting 3D coordinates and validation scores.

We optimize:
- **MPS Native Execution**: Replacing hardcoded CUDA-only references and Float64 casting with dynamic MPS/CPU checks and Float32.
- **Low-Rank Pair Updates**: Projecting high-dimensional Pairformer O(N^2) representations into a low-rank subspace to drop memory usage.
- **CFG Distillation & Neural Refinement**: Speeding up sampling via distilled single-pass flow matching and repairing structural issues using a fast coordinate refiner.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 2 | Apple Silicon MPS Compatibility | Repair device norms, dynamic autocast wrappers, and Float64 casts for native MPS run | None | IN_PROGRESS (Consolidated) |
| 3 | Low-Rank Pair Integration | Replace Evoformer/Pairformer OPM blocks with LowRankPairUpdater | M2 | IN_PROGRESS (Consolidated) |
| 4 | CFG Distillation Integration | Integrate distilled single-pass student vector field into flow-matching step | M2 | IN_PROGRESS (Consolidated) |
| 5 | Neural Coordinate Refinement | Hook up ResNetCoordinateRefiner to correct coordinates post-diffusion | M3, M4 | IN_PROGRESS (Consolidated) |
| 6 | E2E Integration and Adversarial | Run the E2E verification, generate Tier 5 adversarial cases, pass Forensic Audit | M5 | PLANNED |

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
