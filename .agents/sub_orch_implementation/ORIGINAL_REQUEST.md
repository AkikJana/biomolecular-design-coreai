# Original User Request

## Initial Request — 2026-06-20T15:42:47Z

You are the Implementation Track Orchestrator. Your working directory is /Users/akikjana/Documents/BiomolecularDesign/.agents/sub_orch_implementation/.
Your parent is 28bb360b-18d2-4d24-ad05-eeccd08bc10c.
Your mission is to execute the implementation track of the Boltz optimization project.
Specifically:
1. Read the global /Users/akikjana/Documents/BiomolecularDesign/PROJECT.md and /Users/akikjana/Documents/BiomolecularDesign/.agents/ORIGINAL_REQUEST.md.
2. Initialize your own BRIEFING.md, progress.md, and create your own SCOPE.md covering the implementation milestones:
   - Milestone 2: Apple Silicon MPS Compatibility (repairing device norms, dynamic autocast wrappers, and Float64 casts for native MPS run)
   - Milestone 3: Low-Rank Pair Integration (replacing Evoformer/Pairformer OPM blocks with LowRankPairUpdater)
   - Milestone 4: CFG Distillation Integration (integrating distilled single-pass student vector field into flow-matching step)
   - Milestone 5: Neural Coordinate Refinement (hooking up ResNetCoordinateRefiner to correct coordinates post-diffusion)
   - Milestone 6: E2E Integration and Adversarial (Phase 1: running E2E tests when TEST_READY.md is published; Phase 2: generating Tier 5 adversarial cases, passing Forensic Audit)
3. Decompose each milestone into worker/reviewer/challenger cycles. Use the Explorer -> Worker -> Reviewer -> Challenger -> Auditor flow for each milestone.
4. Ensure the mandatory integrity warning is included in all Worker dispatch prompts: "DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected."
5. Poll for /Users/akikjana/Documents/BiomolecularDesign/TEST_READY.md. Once it is published by the E2E Testing Track, run all implementation milestones through the E2E test suite.
6. Report completion of all implementation milestones back to me (parent 28bb360b-18d2-4d24-ad05-eeccd08bc10c).
