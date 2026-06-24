# Handoff Report: GRPO and Lookahead Search-Guided Inference Integration

## 1. Observation
We examined the following files and modules in the `BiomolecularDesign` project:
* **`src/train_preference_alignment.py`**:
  * Tokenizer `AASequenceTokenizer` handles special tokens and amino acid encoding (lines 13–33).
  * `PolicyNetwork` uses a bidirectional GRU that takes tokens and outputs logits of shape `(B, L, vocab_size)` (lines 173–196).
  * `get_sequence_logps` computes token-level log-probs and aggregates them into sequence-level log-probs, optionally length-normalized (lines 200–220).
* **`src/speculative_flow_matching.py`**:
  * `SpeculativeFlowMatchingSampler` accelerates flow matching by using a fast draft model and verifying steps in parallel with a target model (lines 34–269).
  * `project_manifold` enforces C-alpha to C-alpha bond constraints of 3.80 Angstroms (lines 69–88).
  * `avoid_steric_clash` implements a soft repulsion force to prevent overlaps (lines 90–122).
* **`tests/test_speculative_flow_matching.py`**:
  * Contains unit tests evaluating perfect/imperfect draft models and biophysical constraints (lines 34–131).
* **`src/train_g_dpo.py`** & **`src/g_dpo_alignment.py`**:
  * Defines group-based preference training using candidate clustering and preference pairing strategies (like `best_vs_all`).

---

## 2. Logic Chain
1. **GRPO Advantage & Loss Integration**:
   * To implement DeepSeek-style GRPO in `src/train_preference_alignment.py`, we replace preference margin losses (SimPO/DPO) with group relative policy optimization.
   * By standardizing rewards $R_i$ across a sampled group of size $G$, we compute the advantage $A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}$. This removes the need for a value network (critic), optimizing VRAM usage.
   * To avoid the memory footprint of a reference model, we compute the KL regularizer using the detached log-probabilities from the old policy $\pi_{\text{old}}$ (sampled during sequence generation) using Schulman's unbiased estimator.
2. **Search-Guided Inference via Lookahead Rollouts**:
   * To integrate search-guided inference in `src/speculative_flow_matching.py`, we generate $C$ candidate steps at each integration step $t \to t+dt$ by perturbing the draft model's trajectory.
   * We run each candidate step forward to $t=1.0$ using the lightweight draft model.
   * We score the resulting candidate final structures using a biophysical reward combining pocket affinity and steric clash penalties.
   * The candidate step with the highest biophysical reward is selected for integration, and verified against the target model as part of the speculative verification scheme.
3. **Closed-Loop Co-Design**:
   * An agentic design loop (`src/agentic_design_loop.py`) is structured to run iteratively:
     * **Policy Generation**: Sample binder sequences using interface token probabilities from the policy network.
     * **Speculative Folding**: Roll out candidate coordinates and perform lookahead verification.
     * **Biophysical Scoring**: Calculate affinity and steric penalty scores.
     * **GRPO Update**: Feed sequence tokens, rewards, and old log probabilities to the GRPO optimizer to update the policy model.

---

## 3. Caveats
* The draft model's rollout trajectory to $t=1.0$ is assumed to be a reasonable approximation of the target model's trajectory. If the draft model is highly inaccurate, lookahead rollouts may lead to sub-optimal candidate selections.
* Biophysical pocket coordinates are assumed to be static; dynamic target conformational changes are not modeled.

---

## 4. Conclusion
Integrating DeepSeek-style GRPO and search-guided speculative inference provides a robust closed-loop design mechanism. The policy model learns to design binder sequences that physically fit target pockets and satisfy bond-length/clash constraints, all while benefiting from the VRAM savings of a critic-free/reference-free reinforcement learning update.

---

## 5. Verification Method
1. **Code Locations**:
   * Detailed design report is at: `/Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_rl_codesign/analysis.md`
   * Proposed script structures are detailed in the analysis.
2. **Unit Testing Execution**:
   * Execute the existing test suite to ensure no regressions:
     ```bash
     python -m pytest Documents/BiomolecularDesign/tests/test_speculative_flow_matching.py
     python -m pytest Documents/BiomolecularDesign/tests/test_g_dpo.py
     ```
   * The validation scripts for the co-design loop (`tests/test_agentic_design_loop.py`) can be run similarly once implemented.
3. **Invalidation Conditions**:
   * If the training loop loss fails to drop, or if the KL divergence explodes, the learning rate or the KL penalty coefficient $\beta$ must be tuned.
