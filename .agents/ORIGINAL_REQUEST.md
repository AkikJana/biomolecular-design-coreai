# Original User Request

## Initial Request — 2026-06-20T15:34:50Z

Optimize the Boltz structure prediction architecture using modern deep learning techniques and paper literature (e.g., Flow Matching refinements, scaling adjustments, and coordinate refining modules), maximizing edge execution efficiency on Apple Silicon while preserving high structural accuracy comparable to AlphaFold performance.

Working directory: `/Users/akikjana/Documents/BiomolecularDesign`
Integrity mode: benchmark

## Requirements

### R1. Deep Learning Architecture Optimization
Research and implement structural/architectural enhancements to the Boltz model. These can include evolutionary optimizations, flow-matching path refinements, coordinate refiner layers, or state-of-the-art attention techniques, ensuring no shortcuts are taken that compromise model fidelity.

### R2. Latency & Accuracy Trade-Off Evaluation
Verify the performance of the optimized architecture by running comparative benchmarks. The benchmark must measure:
1. Structural accuracy (e.g., RMSD, pLDDT, or GDT) against baseline predictions on validation datasets.
2. Latency and memory usage on target hardware (Apple Silicon MPS/CPU).

## Acceptance Criteria

### Verification & Accuracy
- [ ] Implement a test suite or verification script that runs structure predictions using the optimized architecture on a set of validation targets (e.g., TNF-alpha, Human Insulin monomer, or comparable complexes).
- [ ] Average predicted coordinate discrepancy (RMSD) of the optimized model against the baseline model must be less than 1.0 Å (or maintain >= 98% of baseline pLDDT scores) to prevent accuracy degradation.
- [ ] The implementation must show a measurable reduction in either wall-clock latency (at least 10%) or peak activation memory usage for sequences larger than 500 residues.

## Follow-up — 2026-06-21T12:05:14Z

Integrate M4 (CFG Distillation), M5 (Neural Coordinate Refinement), and M6 (E2E Integration & Verification) into the Boltz structure prediction model.

Working directory: /Users/akikjana/Documents/BiomolecularDesign
Integrity mode: benchmark

## Requirements

### R1. Integrate CFG Distillation into `AtomDiffusion.sample`
- Wire `CFGDistilledVectorField` optional execution path into the `AtomDiffusion.sample()` denoising loop in `diffusionv2.py`.
- When enabled via a constructor or method flag, it must run inference using the student in a single forward pass per step (accepting guidance scale `s`) instead of standard double-pass teacher evaluation.
- The standard path must remain the default and be fully functional.

### R2. Integrate ResNet Coordinate Refinement
- Wire `ResNetCoordinateRefiner` to process the coordinates produced at the end of the diffusion sampling loop.
- It must apply the refinement step to clean up bond length deviations and steric clashes.
- Ensure the refinement step is optional and default-disabled (controllable via configuration/arguments).

### R3. Distillation & Refinement Training Scripts
- Provide clear training scripts or entry points to distill the Boltz teacher into the student and train the coordinate refiner.
- To prevent going over the quota, set low epoch defaults (e.g., 1–3 epochs or small step limits) for validation, and use `empty_device_cache()` to maintain memory stability.

### R4. Complete Verification & Adversarial Test Suite
- Add verification test coverage in `tests/test_e2e_suite.py` (or a dedicated test script) validating:
  1. Functional correctness (standard path outputs do not change).
  2. Latency improvement: Student single-pass CFG path is at least **30% faster** than double-pass CFG.
  3. Accuracy: Student coordinate RMSD vs Teacher is ≤ **1.5 Å** on a synthetic structure benchmark.
  4. Refinement quality: Refined coordinate bond lengths error is ≤ **1.0 Å** (ideal 3.8 Å) and resolves clashes.

## Acceptance Criteria

### Functional Correctness & MPS Compatibility
- Standard path output is identical to baseline (no regressions).
- Student path and Refiner are fully MPS-compatible (no hardcoded `"cuda"`, dynamic autocasting).
- The entire test suite runs and passes.

### Performance & Quality
- Wall-clock speedup for student sampling path is >= 30%.
- Coordinate RMSD (student vs teacher) is <= 1.5 Å.
- Refiner successfully resolves simulated clashes.

## Follow-up — 2026-06-21T13:21:42Z

The E2E verification test suite run failed. There are some test failures in tests/test_e2e_suite.py:
1. test_t2_f1_empty_residue_sequence
2. test_t2_f1_zero_rank_scalar_tensors (IndexError in SinusoidalEmbedding on 0D tensors)
3. test_t2_f3_lookahead_size (assert 10 < 10)
4. test_t4_1_human_insulin_monomer (rmsd check failure)
5. test_t4_2_hemoglobin_subunit_alpha (rmsd check failure)
6. test_t4_3_tnf_alpha_complex (clashes check failure)
7. test_t4_4_vegfa_monomer (assert len(vegf_seq) == 110 failed due to length 112)

## Follow-up — 2026-06-24T05:01:07Z

Implement DeepSeek-style GRPO reinforcement learning and Google-style search-guided inference inside a closed-loop Agentic Co-Design loop for protein binder discovery.

Working directory: /Users/akikjana/Documents/BiomolecularDesign
Integrity mode: development

## Requirements

### R1. DeepSeek-Style GRPO Training Module
- Implement a GRPO (Group Relative Policy Optimization) training step inside `src/train_preference_alignment.py` (or a dedicated script).
- It must generate a group of G sequences, compute the advantage of each sequence relative to the group average and standard deviation, and execute policy updates without using a separate value network or reference model.

### R2. Google-Style Search-Guided Inference
- Implement a search-guided inference mechanism (e.g., Guided Beam Search or lookahead rollouts) inside `src/speculative_flow_matching.py` that utilizes a reward function (pocket affinity, steric overlaps) to steer trajectory steps towards low-energy states.

### R3. Closed-Loop Agentic Co-Design Loop
- Create an orchestration script `src/agentic_design_loop.py` that runs the continuous loop:
  1. Policy generates a group of binder sequences.
  2. Speculative Flow matching folds the sequences into coordinates.
  3. Biophysical scorer computes rewards (H-bonds, steric clashes, pocket affinity).
  4. GRPO updates the policy based on rewards.
- Provide a validation script to test this end-to-end.

## Acceptance Criteria

### Execution & Convergence
- [ ] GRPO loss decreases over training epochs.
- [ ] The agentic loop runs end-to-end for multiple iterations without errors.
- [ ] Test cases verifying GRPO advantage calculation and search-guided trajectory selection are added and pass.
