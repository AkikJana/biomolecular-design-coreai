## 2026-06-20T21:18:51Z

You are teamwork_preview_worker. Your working directory is /Users/akikjana/Documents/BiomolecularDesign/.agents/worker_1/. Your parent conversation ID is 3b22170c-2360-4307-8490-eadba5d7ed35.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your mission is to:
1. Initialize BRIEFING.md and progress.md in your working directory.
2. Create `/Users/akikjana/Documents/BiomolecularDesign/TEST_INFRA.md` following the TEST_INFRA.md Template in the instructions. Detail:
   - Test runner: `pytest` command running `/Users/akikjana/Documents/BiomolecularDesign/tests/test_e2e_suite.py`
   - Case format: pytest test cases checking exit codes, RMSD/pLDDT correctness, and performance benchmarks.
   - Feature inventory: MPS execution (F1), Low-rank pair updates (F2), CFG distillation (F3), Neural refinement (F4).
3. Implement `/Users/akikjana/Documents/BiomolecularDesign/tests/test_e2e_suite.py` containing exactly 49 test cases (20 Tier 1, 20 Tier 2, 4 Tier 3, 5 Tier 4). Let's structure the tests using pytest.
   - Tier 1: 5 tests for each of F1, F2, F3, F4 (total 20).
     - MPS execution tests: check float32/fp32 casting compatibility, dynamic autocast wrappers, device selection CPU/MPS/CUDA, and error-free execution of diffusion modules/attention on MPS if simulated or available.
     - Low-rank pair updates tests: check custom autograd LowRankTensorProduct forward output shapes, gradient consistency check (via gradcheck or manual delta backward), memory scaling verification, weight init verification, and stability with small tensors.
     - CFG distillation: check teacher vector field matching student distilled model output, speculative sampler step lookahead integration (manifold/distance constraints), guidance scale embeddings dimension check, step acceptance logic, and student forward-pass efficiency.
     - Neural refinement: check ResNetCoordinateRefiner shape/continuity, supervised loss calculation (pairwise distance matrix MSE), clash index improvement, bond length error correction, and refiner loading/inference.
   - Tier 2: Boundary & Corner cases (total 20, 5 per feature).
     - F1: Empty/zero-residue sequences, extremely large sequence prediction (1000+ residues), autocast enabled vs disabled, device fallback behavior, and zero-rank/scalar tensors.
     - F2: Minimum rank d=1, full/over-complete rank d >= C_z, sparse/zero input tensors, inf/nan gradients backward safety, and memory-scaling limits.
     - F3: Guidance scale s = 0, negative/extreme guidance scales, lookahead size K=1 vs large K=10, 100% step rejection handling, and time boundaries t=0.0 and 1.0.
     - F4: Coordinates with NaNs/zeros, multi-chain boundaries, perfect coordinates refinement, extreme clashing coordinates, and residue count/sequence length mismatches.
   - Tier 3: Cross-Feature combinations (total 4).
     - Test F1 + F2: Low-rank pair updates under MPS.
     - Test F1 + F3: CFG distilled speculative sampler under MPS.
     - Test F3 + F4: Neural refiner running post-CFG distilled speculative sampler.
     - Test F1 + F2 + F3 + F4: Pipeline integration running low-rank + CFG distilled speculative sampler + Neural refiner under MPS.
   - Tier 4: Real-World scenarios (total 5).
     - Test 4.1: Human Insulin Monomer (51 residues) predicting C-alpha coordinates, verifying RMSD vs baseline, and pLDDT.
     - Test 4.2: Hemoglobin alpha subunit (142 residues) predicting coordinates, verifying RMSD, and latency.
     - Test 4.3: TNF-alpha complex structure (157 residues) predicting coordinates, checking clash penalty and PDB parsing.
     - Test 4.4: VEGFA monomer structure (110 residues) predicting coordinates and checking exit code.
     - Test 4.5: Large-scale validation (>500 residues) testing latency and activation memory reduction compared to baseline.
4. Implement `/Users/akikjana/Documents/BiomolecularDesign/run_e2e_tests.py` which executes the test suite via pytest and outputs results.
5. Create `/Users/akikjana/Documents/BiomolecularDesign/TEST_READY.md` containing the test runner command and the coverage checklist of all 49 tests.
6. Verify and run the test suite to ensure all tests pass (use the .venv/bin/pytest environment if needed).

Write your reports and handoffs in your working directory. Send a message to me (parent) when the task is complete.
