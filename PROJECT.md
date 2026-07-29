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
* `src/campaign_ranking.py`, `src/prospective_campaign.py`,
  `src/certified_selectivity.py`, `src/empirical_study.py`: workstream C
  (see Milestones).
* `src/benchmark_*.py`: benchmark/demo scripts. These are *not* tests and are
  excluded from pytest collection; `tests/` holds the collected suite.

## Milestones

Three workstreams. IDs are prefixed per workstream because an earlier revision
numbered two of them `1-6` and `M1-M6` while writing dependencies as bare `M1`,
`M2`, ... — which made rows in the first table appear to depend on rows in the
second, and produced DONE items that depended on IN_PROGRESS ones. Dependencies
below are fully qualified; statuses are carried over unchanged.

### A. Apple Silicon inference optimization
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| A1 | E2E Test Suite Development | Requirement-driven E2E tests (Tiers 1-4) assessing RMSD, pLDDT, latency, and memory | None | ✅ DONE |
| A2 | Apple Silicon MPS Compatibility | Repair device norms, dynamic autocast wrappers, and Float64 casts for native MPS run | A1 | ✅ DONE |
| A3 | Low-Rank Pair Integration | Replace full-rank Evoformer/Pairformer OPM blocks with LowRankPairUpdater | A2 | ✅ DONE |
| A4 | CFG Distillation Integration | Integrate distilled single-pass student vector field into flow-matching step | A2 | ✅ DONE |
| A5 | Neural Coordinate Refinement | Hook up ResNetCoordinateRefiner to correct coordinates post-diffusion | A3, A4 | ✅ DONE |
| A6 | E2E Integration and Adversarial | Run the E2E verification, generate Tier 5 adversarial cases, pass Forensic Audit | A1, A5 | ✅ DONE |

### B. RL-driven co-design
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| B1 | Exploration & Design | Design analysis and architecture planning | None | ✅ DONE |
| B2 | Core GRPO | Group Relative Policy Optimization training steps | B1 | 🔄 IN_PROGRESS |
| B3 | Search-Guided Inference | Search-guided speculative sampler with lookahead rollouts | B2 | 🔄 IN_PROGRESS |
| B4 | Closed-Loop Agentic Co-Design | Link sequence design policy, structure sampler and biophysical feedback | B2, B3 | 🔄 IN_PROGRESS |
| B5 | Verification & Testing | Test GRPO advantage calculation, search-guided trajectory, and loop convergence | B4 | 🔄 IN_PROGRESS |
| B6 | Forensic Audit | Validate execution and pass the forensic auditor requirements | B5 | 🔄 IN_PROGRESS |

### C. Prospective campaign layer
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| C1 | Target-Disjoint Ranking | Record per-proposal features during co-design; calibrated ensemble ranker with leakage rejection (`src/campaign_ranking.py`) | B4 | ✅ DONE |
| C2 | Multi-Context Design Spec | Intended/off-target contexts, auditable verifier traces, conformal routing (`src/prospective_campaign.py`) | C1 | ✅ DONE |
| C3 | Certified Selectivity | Joint split-conformal lower bound on robust selectivity with familywise risk control (`src/certified_selectivity.py`) | C2 | ✅ DONE |
| C4 | Empirical Study Report | Target-disjoint result tables, coverage curves, fixed-budget replay, assay manifest (`src/empirical_study.py`) | C3 | ✅ DONE |
| C5 | Wet-Lab Validation | Execute the prospective assay manifest and report measured selectivity | C4 | ⬜ NOT STARTED |

**C1-C4 are validated on synthetic rewards only.** They exercise workflow
semantics, not biological performance; the conformal coverage guarantee is
marginal and assumes calibration examples are exchangeable with future
target-disjoint examples. No prospective performance claim is supported until
C5 produces assay results.

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
