## 2026-06-24T05:02:43Z
You are teamwork_preview_explorer. Your mission is to explore the codebase and design the integration of:
1. DeepSeek-style GRPO reinforcement learning in `src/train_preference_alignment.py`.
2. Google-style search-guided inference (lookahead rollouts) in `src/speculative_flow_matching.py` with rewards based on pocket affinity and steric clashes.
3. Closed-loop Agentic Co-Design loop in `src/agentic_design_loop.py` that connects Policy generation, Speculative folding, Biophysical scoring, and GRPO policy updates.

Please analyze the following files:
- `src/train_preference_alignment.py` (understand tokenization, policy network, logging probability calculation)
- `src/speculative_flow_matching.py` (understand SpeculativeFlowMatchingSampler, vector field calls, project_manifold, and avoid_steric_clash)
- Other relevant codebase files or existing tests like `tests/test_speculative_flow_matching.py`.

Your analysis must provide:
- Detailed mathematical formulations and PyTorch code structures for the GRPO advantage calculation (group average and standard deviation) and policy update loss, without relying on value/reference models.
- Detailed logic for lookahead rollouts in search-guided inference. Explain how candidates are generated, how the draft model is used for lookahead to t=1.0, and how they are selected based on a biophysical reward.
- Clear specification and script structure for `src/agentic_design_loop.py` and its validation script.
- Suggestions on how to verify convergence and add tests for advantage calculation and search guidance.

Write your findings to `/Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_rl_codesign/analysis.md` and a summary handoff to `/Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_rl_codesign/handoff.md`.
Do NOT write or modify any source code files. You are a read-only explorer.
