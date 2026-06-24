## 2026-06-24T05:18:06Z
You are the Code Remediation Specialist (teamwork_preview_worker).
Your working directory is `/Users/akikjana/Documents/BiomolecularDesign/.agents/worker_remediation/`.

Please implement the mathematical and logic remediations for the Biomolecular Design project to resolve the two integrity violations identified in the Forensic Audit.

### 1. Speculative Flow Matching Sampler Corrections
In `src/speculative_flow_matching.py` (specifically in the class `SpeculativeFlowMatchingSampler` at line 214):
- Accumulate target-model updates correctly instead of overwriting the simulation state:
  - Initialize `curr_verified_x = x.clone()` before the step-by-step verification loop.
  - In each step k, if accepted or rejected/corrected, update:
    `curr_verified_x = curr_verified_x + v_target * dt`
    If `self.enable_biophysical` is true, apply:
    `curr_verified_x = self.project_manifold(curr_verified_x)`
    `curr_verified_x = self.avoid_steric_clash(curr_verified_x)`
  - Make sure the simulation state updates after the loop: `x = curr_verified_x`.
  - Apply this accumulation logic to both the acceptance branch and the rejection/correction branch.

### 2. GRPO Loss Degeneracy (Batch training)
In `src/train_preference_alignment.py` (lines 473-484):
- Implement multiple inner optimization steps to allow policy log-probabilities to diverge from cached old log-probabilities:
  - Cache `old_logps = policy_logps.detach()` before updating the policy parameters.
  - Run 3 inner epochs/steps over the same batch of sequences.
  - In each step:
    - If it's step 0, you can reuse `policy_logps` if it's already computed; otherwise (steps > 0), run the active policy forward pass to compute the active `policy_logps`.
    - Calculate `loss, kl_mean, advantages = grpo_loss(...)` using active `policy_logps` and the cached `old_logps`.
    - Run `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.

### 3. GRPO Loss Degeneracy (Co-design loop)
In `src/agentic_design_loop.py` (lines 110-134):
- Implement 3 inner optimization steps:
  - Compute and cache `old_logps` once:
    ```python
    with torch.no_grad():
        old_logits = policy(encoded_tokens)
        old_logps = get_sequence_logps(old_logits, encoded_tokens, length_normalize=True).detach()
    ```
  - Loop 3 times (inner steps).
  - Inside the loop, run forward pass to get active `policy_logits` and active `policy_logps`.
  - Calculate `loss, kl, advantages = grpo_loss(...)` with the cached `old_logps`.
  - Run `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.
  - Ensure that `loss` and `kl` values are retrieved from the last inner step to update `metrics_history`.

### 4. Update Tests
In `tests/test_agentic_design_loop.py` (specifically in `test_co_design_loop_e2e` at lines 77-91):
- Assert that the logged loss and KL values in `metrics` are not zero and are positive/non-degenerate:
  `assert entry["loss"] != 0.0`
  `assert entry["kl"] > 0.0`

### 5. Verification
After implementing the changes, run the tests using the virtual environment at `.venv/bin/python`:
- `.venv/bin/python -m pytest tests/test_agentic_design_loop.py -v`
- `.venv/bin/python tests/test_speculative_flow_matching.py`

Write a completion handoff report to `/Users/akikjana/Documents/BiomolecularDesign/.agents/worker_remediation/handoff.md` detailing:
- The changes made in each file.
- The commands used to run the tests and the exact test output/results.

⚠️ MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT
hardcode test results, create dummy/facade implementations, or
circumvent the intended task. A Forensic Auditor will independently
verify your work. Integrity violations WILL be detected and your
work WILL be rejected.
