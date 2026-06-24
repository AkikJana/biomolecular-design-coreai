# Plan: Biomolecular Design Reinforcement Learning & Speculative Search-Guided Co-Design

This plan outlines the design and implementation of DeepSeek-style GRPO reinforcement learning, Google-style search-guided inference, and a closed-loop Agentic Co-Design loop for protein binder discovery.

## Milestones

| Milestone | Description | Specialist | Status |
|-----------|-------------|------------|--------|
| M1: Exploration & Design | Explore existing codebase, analyze model structures, and design detailed GRPO + search-guided inference math and architecture. | Explorer | ✅ DONE |
| M2: Core GRPO implementation | Implement GRPO training step (without separate value/reference model) inside `src/train_preference_alignment.py` (or new dedicated module). | Worker | ✅ DONE |
| M3: Search-Guided Inference | Implement lookahead rollout search-guided trajectory selection inside `src/speculative_flow_matching.py`. | Worker | ✅ DONE |
| M4: Closed-Loop Agentic Co-Design | Create orchestration script `src/agentic_design_loop.py` to run continuous loop (Policy -> Speculative FM -> Biophysical Scorer -> GRPO update) + validation script. | Worker | ✅ DONE |
| M5: Verification & Testing | Add test cases verifying GRPO advantage calculation and search-guided trajectory selection. Run E2E validation. | Challenger / Reviewer | ✅ DONE |
| M6: Forensic Audit | Run Forensic Auditor to ensure clean verdict, lack of cheating, and authentic implementation. | Auditor | ❌ FAILED |
| M7: Remediation | Address GRPO loss degeneracy (KL divergence check) and speculative sampler intermediate step corrections. | Worker | 🔄 IN_PROGRESS |

## Verification Plan
1. Check that GRPO loss decreases over training epochs.
2. Run the agentic co-design loop for multiple iterations.
3. Run pytest on the new and updated tests to verify all functionality.
4. Forensic auditor validation of integrity.
