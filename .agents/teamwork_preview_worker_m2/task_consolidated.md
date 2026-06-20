# Consolidated Task for Worker - Milestones 2, 3, 4, and 5 Integration

## Objective
Due to a critical token budget limit, all remaining optimization and integration milestones must be consolidated and implemented in a single pass. You are instructed to implement the following changes in the main `boltz` package:

1. **Milestone 2: Apple Silicon MPS Compatibility**
   - Implement dynamic device-agnostic autocasts (`autocast_device_type(device_type)`).
   - Ensure zero-value tensors returned by device norm calculations in `boltz2.py` are on `self.device` instead of hardcoded CUDA/CPU logic.
   - Implement `empty_device_cache` for MPS/CUDA dynamically in the exception handlers.

2. **Milestone 3: Low-Rank Pair Integration**
   - Replace the Evoformer/Pairformer Outer Product Mean (OPM) blocks with `LowRankPairUpdater` from `src/low_rank_pair_representation.py`.
   - Interface contract:
     - Input: Pair representation tensor `z` of shape `[B, N, N, C_z]`
     - Output: Updated pair representation `z_out` of shape `[B, N, N, C_z]` with custom autograd checkpointing.

3. **Milestone 4: CFG Distillation Integration**
   - Integrate the distilled single-pass student vector field (see reference in `src/cfg_distillation.py`) into the flow-matching/diffusion denoising step inside `boltz/src/boltz/model/modules/diffusionv2.py`.
   - Interface contract:
     - Input: Denoising state, time/sigma step, guidance scale parameter `s`
     - Output: Single-pass vector field prediction, skipping double evaluations of standard CFG.

4. **Milestone 5: Neural Coordinate Refinement**
   - Hook up `ResNetCoordinateRefiner` (see implementation reference in `src/train_neural_refiner.py`) to correctpredicted 3D coordinates post-diffusion.
   - Interface contract:
     - Input: Coarse coordinates `[1, L, 3]` from draft/fast prediction.
     - Output: Refined coordinates `[1, L, 3]` with adjusted bond lengths and minimized clashes.

## Verification
- Run the full test suite (`pytest tests/` or target tests) to verify compilation and correctness.
- Ensure that the E2E verification tests run and compile correctly on the target device.

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
