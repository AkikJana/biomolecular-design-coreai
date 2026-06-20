# Task for Explorer 2 - Milestone 2: Apple Silicon MPS Compatibility

## Objective
Investigate the Boltz codebase (especially `boltz/src/boltz/model/models/boltz2.py` and `boltz/src/boltz/model/modules/diffusionv2.py`) to identify CUDA-only operations, device-norm operations, hardcoded device assumptions, Float64 casts, and PyTorch autocast wrappers that prevent native execution on Apple Silicon MPS. Propose a precise strategy/plan for fixing these to run natively on MPS/CPU.

## Constraints & Rules
- Do NOT modify any code. You are a read-only Explorer.
- Create a detailed handoff report (`handoff.md` or `analysis.md`) in your working directory containing your findings and recommendations.
- Include verified evidence chains (code snippets, file paths, line numbers).
