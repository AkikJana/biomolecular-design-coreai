# E2E Test Infra: Boltz Structure Prediction Apple Silicon Optimization

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + BVA + Pairwise + Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | MPS Execution | ORIGINAL_REQUEST R1/R2 | 5 | 5 | ✓ |
| 2 | Low-rank pair updates | ORIGINAL_REQUEST R1/R2 | 5 | 5 | ✓ |
| 3 | CFG distillation | ORIGINAL_REQUEST R1/R2 | 5 | 5 | ✓ |
| 4 | Neural refinement | ORIGINAL_REQUEST R1/R2 | 5 | 5 | ✓ |

## Test Architecture
- Test runner: pytest command executed via `.venv/bin/pytest -v /Users/akikjana/Documents/BiomolecularDesign/tests/test_e2e_suite.py` or `.venv/bin/python /Users/akikjana/Documents/BiomolecularDesign/run_e2e_tests.py`.
- Test case format: pytest unit/integration test cases verifying exit codes, RMSD/pLDDT correctness, and performance benchmarks.
- Directory layout:
  - `src/`: contains core architecture optimizations.
  - `tests/test_e2e_suite.py`: E2E test cases (49 total).
  - `run_e2e_tests.py`: Python test runner script.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Human Insulin Monomer | F1, F4 | Medium |
| 2 | Hemoglobin alpha subunit | F1, F3 | Medium |
| 3 | TNF-alpha complex | F1, F4 | Medium |
| 4 | VEGFA monomer | F1 | Medium |
| 5 | Large-scale validation (>500 aa) | F1, F2 | High |

## Coverage Thresholds
- Tier 1: 5 per feature (20 total)
- Tier 2: 5 per feature (20 total)
- Tier 3: pairwise coverage of major feature interactions (4 total)
- Tier 4: 5 realistic application scenarios (5 total)
