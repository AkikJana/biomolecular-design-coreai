## 2026-06-24T10:34:15Z

You are teamwork_preview_worker. Your mission is to implement DeepSeek-style GRPO reinforcement learning, Google-style search-guided inference, and a closed-loop Agentic Co-Design loop for protein binder discovery.

Please read the design analysis report at `/Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_rl_codesign/analysis.md` for the exact mathematical formulation and PyTorch structures.

Your tasks are:
1. **Update PROJECT.md**: Append the new milestones (M1: Exploration & Design, M2: Core GRPO, M3: Search-Guided Inference, M4: Closed-Loop Agentic Co-Design, M5: Verification & Testing, M6: Forensic Audit) to the milestones table at the root `PROJECT.md`. Mark M1 as DONE, and the others as IN_PROGRESS.
2. **GRPO Implementation**:
   - Implement the GRPO training step inside `src/train_preference_alignment.py`.
   - Ensure it calculates sequence-level log-probabilities using `get_sequence_logps` (length-normalized).
   - Implement the advantage calculation: standardize rewards across a group of G candidates (using group mean and standard deviation).
   - Implement the policy ratio calculation and clipped surrogate loss.
   - Implement the reference-free KL divergence penalty:
     KL = exp(old_logps - policy_logps) - (old_logps - policy_logps) - 1
   - Add a function/option to run training using GRPO.
3. **Search-Guided Speculative Sampling**:
   - Implement a search-guided speculative sampler class (or update/extend the existing `SpeculativeFlowMatchingSampler` in `src/speculative_flow_matching.py`) that implements lookahead rollouts.
   - At each integration step, generate C candidate steps by adding small random perturbations to the draft vector field direction.
   - For each candidate, roll out to t=1.0 using the lightweight draft model (applying manifold projections and clash avoidance).
   - Compute biophysical rewards (pocket affinity score + steric clash penalty) on the folded coordinates at t=1.0.
   - Select the candidate step maximizing reward, and verify it against the target model step within the speculative acceptance threshold.
4. **Closed-Loop Agentic Co-Design**:
   - Create `src/agentic_design_loop.py` that connects sequence generation (via policy model) -> speculative folding (via search-guided sampler) -> biophysical scoring -> GRPO training updates.
   - Run the loop for multiple iterations, outputting convergence metrics (loss, mean reward, KL divergence, etc.) to verify convergence.
5. **E2E test suite**:
   - Create `tests/test_agentic_design_loop.py` containing unit and integration tests:
     - Test GRPO advantage calculation (zero-mean, unit-variance properties).
     - Test search-guided trajectory selection (perturbed structures and biophysical scoring).
     - Run the co-design loop end-to-end for multiple iterations and verify that the loss decreases and the loop finishes successfully.
   - Run the tests using pytest to verify that they pass.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please write a detailed summary of your changes in `.agents/worker_rl_codesign/changes.md` and a handoff report in `.agents/worker_rl_codesign/handoff.md`. Run pytest on the new tests to verify.
