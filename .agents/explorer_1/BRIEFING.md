# BRIEFING — 2026-06-20T21:49:00+05:30

## Mission
Investigate the Boltz structure prediction codebase, run options, validation targets, and implementations of MPS execution, low-rank updates, CFG distillation, and neural refinement.

## 🔒 My Identity
- Archetype: teamwork_preview_explorer
- Roles: Explorer, Investigator, Reporter
- Working directory: /Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_1/
- Original parent: 3b22170c-2360-4307-8490-eadba5d7ed35
- Milestone: Boltz Exploration

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- CODE_ONLY network mode: no external web access

## Current Parent
- Conversation ID: 3b22170c-2360-4307-8490-eadba5d7ed35
- Updated: 2026-06-20T21:49:00+05:30

## Investigation State
- **Explored paths**: `src/predict_structure.py`, `boltz/src/boltz/model/models/boltz2.py`, `boltz/src/boltz/model/modules/utils.py`, `boltz/src/boltz/model/modules/diffusion.py`, `boltz/src/boltz/model/modules/diffusionv2.py`, `boltz/src/boltz/model/layers/attention.py`, `boltz/src/boltz/model/layers/attentionv2.py`, `boltz/src/boltz/model/modules/trunkv2.py`, `src/low_rank_pair_representation.py`, `src/cfg_distillation.py`, `src/train_neural_refiner.py`, `src/speculative_flow_matching.py`, `src/dms_generation.py`, `tests/test_dynamic_actual_samples.py`, `tests/test_boltz_wrapper.py`, `tests/test_speculative_flow_matching.py`.
- **Key findings**:
  1. CLI/App entrypoints exist in `src/predict_structure.py` and `tests/test_dynamic_actual_samples.py`, with model definitions in `boltz/src/boltz/model/models/boltz2.py`.
  2. Validation targets consist of TNF-alpha (1TNF), VEGFA (1FLT), Human Insulin monomer/fragment, and Hemoglobin subunit alpha. PDBs are downloaded from RCSB database.
  3. MPS compatibility requires replacing several hardcoded `torch.autocast("cuda", ...)` calls across 6 files (boltz2.py, diffusion.py, diffusionv2.py, attention.py, attentionv2.py, trunkv2.py) with dynamic device-aware utilities, and ensuring Float32 precision is used since Apple Silicon GPUs lack native Double (Float64) support.
  4. Low-rank pair updates are implemented in `src/low_rank_pair_representation.py` using custom autograd `LowRankTensorProduct` to save $O(N^2)$ activations.
  5. CFG distillation is implemented in `src/cfg_distillation.py` via teacher-student alignment, skipping double evaluations in standard CFG. Speculative flow matching in `src/speculative_flow_matching.py` integrates a fast draft and expensive verification step.
  6. Neural refinement in `src/train_neural_refiner.py` trains `ResNetCoordinateRefiner` to correct coordinates post-diffusion.
- **Unexplored areas**: Real MPS execution benchmarks (terminal access was timed out).

## Key Decisions Made
- Completing the file-based code investigation and compiling the structured report for the E2E testing agent.

## Artifact Index
- /Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_1/ORIGINAL_REQUEST.md — Original request from parent.
- /Users/akikjana/Documents/BiomolecularDesign/.agents/sub_orch_e2e_testing/exploration_report.md — Detailed exploration report.
