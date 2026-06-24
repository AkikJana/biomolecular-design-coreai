# Changes Log - GRPO, Speculative Sampling & Closed-Loop Agentic Co-Design

This document summarizes the changes made to implement DeepSeek-style GRPO reinforcement learning, search-guided speculative sampling, and the closed-loop agentic co-design loop.

## 1. Updated Root `PROJECT.md`
- Appended the new milestones (`M1` to `M6`) to the milestones table at the root of `PROJECT.md`.
- Marked `M1` as `DONE` and `M2` - `M6` as `IN_PROGRESS`.

## 2. Implement GRPO Training Step in `src/train_preference_alignment.py`
- Implemented `grpo_loss` function:
  - Standardizes rewards across candidate groups ($G$) using the group mean and standard deviation.
  - Computes policy probability ratios and the clipped PPO surrogate loss.
  - Integrates the reference-free KL divergence estimator:
    $$D_{\text{KL}}(\pi_\theta \mid\mid \pi_{\text{old}}) = \exp(\log \pi_{\text{old}} - \log \pi_\theta) - (\log \pi_{\text{old}} - \log \pi_\theta) - 1$$
- Extended `train_preference_alignment` signature and logic with `use_grpo`, `grpo_beta`, and `grpo_clip_eps` options to enable offline GRPO training on the simulated Teddymer annotations dataset.
- Added command-line parsing to enable training with the `--grpo` flag.

## 3. Implement Search-Guided Speculative Sampler in `src/speculative_flow_matching.py`
- Appended `SearchGuidedSpeculativeSampler` class:
  - Generates $C$ candidate coordinates at each integration step by perturbing the draft model's proposed vector field step direction.
  - Runs parallel draft-model lookahead rollouts to $t=1.0$ for all candidates under biophysical constraints (manifold projection + steric clash avoidance).
  - Evaluates biophysical rewards (pocket affinity score + soft steric clash penalty) on the folded coordinates at $t=1.0$.
  - Selects the candidate maximizing reward and speculative-verifies it against the target model step within the specified tolerance.

## 4. Create Closed-Loop Agentic Co-Design Loop in `src/agentic_design_loop.py`
- Created `src/agentic_design_loop.py` containing:
  - `compute_rewards`: Biophysical scoring combining pocket contact density and steric clash penalties.
  - `sample_sequences`: Generates candidate mutations at specified interface positions.
  - `run_codesign_loop`: Connects sequence generation (policy) -> speculative structure generation (sampler) -> biophysical scoring -> GRPO training updates over multiple iterations. Shows convergence metrics (mean reward, loss, average KL).

## 5. Implement E2E and Unit Test Suite in `tests/test_agentic_design_loop.py`
- Created E2E test file:
  - `test_grpo_advantage_properties`: Asserts that GRPO standardizes rewards to zero-mean and unit-variance.
  - `test_search_guided_trajectory_selection`: Verifies candidate perturbation, lookahead structure folding, biophysical reward calculation, and selection.
  - `test_co_design_loop_e2e`: Runs a multi-iteration co-design loop end-to-end and validates that loss, reward, and KL are tracked and the loop runs successfully.
- Added `if __name__ == "__main__":` block to allow direct script execution.
