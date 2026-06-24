# BRIEFING — 2026-06-21T13:03:00Z

## Mission
Analyze test_e2e_suite.py and the codebase to design a remediation strategy for integrity violations (self-certifying plddt metrics and SimulatedPredictor facade fallback).

## 🔒 My Identity
- Archetype: explorer
- Roles: Teamwork explorer, Read-only investigator
- Working directory: /Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_remediation
- Original parent: b359dd2c-c18d-4f35-bc20-cdf17cbef3eb
- Milestone: explorer_remediation

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Analyze issues in test_e2e_suite.py and propose remediation strategy
- Recommend strategy to replace SimulatedPredictor and mock plddt with actual implementation/module call
- Ensure tests are not self-certifying

## Current Parent
- Conversation ID: b359dd2c-c18d-4f35-bc20-cdf17cbef3eb
- Updated: 2026-06-21T13:03:00Z

## Investigation State
- **Explored paths**:
  - `tests/test_e2e_suite.py`
  - `src/predict_structure.py`
  - `src/boltz_wrapper.py`
  - `src/boltz_fast.py`
  - `src/train_neural_refiner.py`
  - `src/train_preference_alignment.py`
- **Key findings**:
  - `SimulatedPredictor` in `tests/test_e2e_suite.py` is a hardcoded helical sine/cosine formula which bypasses execution of actual machine learning models.
  - The E2E test `test_t4_1_human_insulin_monomer` uses a hardcoded sequence-based mock formula to compute `plddt` directly in the test case rather than invoking the predictor, making the assertion self-certifying.
  - The production class `DynamicStructurePredictor` in `src/predict_structure.py` returns only 3D coordinates, and does not return confidence values like pLDDT.
- **Unexplored areas**:
  - No unexplored areas. Complete understanding of the codebase structure and test mechanics has been achieved.

## Key Decisions Made
- Replace the facade `SimulatedPredictor` with a genuine `LightweightPredictor` using a PyTorch neural network that performs actual structure coordinate prediction.
- Replace the mock `plddt` calculation with a genuine PyTorch-based confidence predictor module that evaluates coordinates and sequence features.
- Update `DynamicStructurePredictor` and `LightweightPredictor` interfaces to return both coords and plddt to remove self-certification in the test suite.

## Artifact Index
- /Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_remediation/analysis.md — Findings and proposed fix strategy
- /Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_remediation/handoff.md — Handoff report
