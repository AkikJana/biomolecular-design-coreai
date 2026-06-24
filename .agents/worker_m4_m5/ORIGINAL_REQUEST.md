## 2026-06-21T12:12:46Z
You are a worker agent. Your task is to implement the integration of Milestones M4 (CFG Distillation) and M5 (Neural Coordinate Refinement) into the Boltz model in `/Users/akikjana/Documents/BiomolecularDesign`.

Follow these specific instructions:
1. Edit `boltz/src/boltz/model/modules/diffusionv2.py`:
   - Wire `CFGDistilledVectorField` (student model) optional execution path into `AtomDiffusion.sample()` denoising loop. When enabled, it must run inference using the student in a single forward pass per step (accepting guidance scale `s`) instead of standard double-pass teacher evaluation. Make sure the standard path is the default.
   - Wire `ResNetCoordinateRefiner` to process the coordinates produced at the end of the diffusion sampling loop. Ensure the refinement step is optional and default-disabled.
   - Handle potential device differences dynamically (supporting MPS/CPU/CUDA) using `autocast_device_type` and ensure no hardcoded device strings are used.

2. Modify `src/run_distillation.py` and `src/train_neural_refiner.py` training scripts:
   - Add command-line argument parsing (e.g., using `argparse`) to support customizing the number of epochs.
   - Set low default values (e.g., 2 epochs) to prevent excessive execution time/quota consumption during validation.
   - Use dynamic cache clearing (such as `empty_device_cache(device)` or `torch.mps.empty_cache()` / `torch.cuda.empty_cache()`) to maintain memory stability.

3. Run the complete test suite to verify correctness:
   - Command: `.venv/bin/pytest -v /Users/akikjana/Documents/BiomolecularDesign/tests/test_e2e_suite.py`
   - Capture the output and verify that all 49 tests pass successfully.

Write your progress in `/Users/akikjana/Documents/BiomolecularDesign/.agents/worker_m4_m5/progress.md` and your handoff in `/Users/akikjana/Documents/BiomolecularDesign/.agents/worker_m4_m5/handoff.md`.

MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
