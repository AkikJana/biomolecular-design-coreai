# BRIEFING — 2026-06-20T15:43:34Z

## Mission
Analyze the Boltz codebase for Apple Silicon MPS compatibility issues and formulate a code modification strategy.

## 🔒 My Identity
- Archetype: MPS Compatibility Explorer 1
- Roles: Read-only investigator, analyzer
- Working directory: /Users/akikjana/Documents/BiomolecularDesign/.agents/teamwork_preview_explorer_m2_1/
- Original parent: 223a8feb-d33c-4a64-ab9e-3a0187d84371
- Milestone: Milestone 2: Apple Silicon MPS Compatibility

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- CODE_ONLY network mode: No external network access

## Current Parent
- Conversation ID: 223a8feb-d33c-4a64-ab9e-3a0187d84371
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `boltz/src/boltz/model/models/boltz2.py`
  - `boltz/src/boltz/model/models/boltz1.py`
  - `boltz/src/boltz/model/modules/diffusionv2.py`
  - `boltz/src/boltz/model/modules/diffusion.py`
  - `boltz/src/boltz/model/modules/trunkv2.py`
  - `boltz/src/boltz/model/modules/encodersv2.py`
  - `boltz/src/boltz/model/layers/attentionv2.py`
  - `boltz/src/boltz/model/layers/attention.py`
  - `boltz/src/boltz/model/layers/pairformer.py`
  - `boltz/src/boltz/model/layers/triangular_attention/primitives.py`
  - `boltz/src/boltz/model/loss/confidencev2.py`
  - `boltz/src/boltz/main.py`
  - `src/boltz_wrapper.py`
- **Key findings**:
  - Hardcoded CUDA autocast wrappers (`with torch.autocast("cuda", enabled=False):`) in modules and layers prevent CPU/MPS execution.
  - Hardcoded parameter and gradient norm checks (`device="cuda" if ... else "cpu"`) in `boltz2.py`.
  - Hardcoded empty cache logic (`torch.cuda.empty_cache()`) doesn't clean MPS limits.
  - Device routing in `src/boltz_wrapper.py` only handles CUDA/CPU.
  - Precision setting `bf16-mixed` is unsupported/unstable on MPS and needs dynamic fallback routing.
- **Unexplored areas**: None.

## Key Decisions Made
- Formulated a multi-stage dynamic modification strategy using `autocast_device_type(device_type)` to resolve the autocast bottleneck across the entire model and loss definition layers.

## Artifact Index
- /Users/akikjana/Documents/BiomolecularDesign/.agents/teamwork_preview_explorer_m2_1/ORIGINAL_REQUEST.md — Original request text
- /Users/akikjana/Documents/BiomolecularDesign/.agents/teamwork_preview_explorer_m2_1/analysis.md — MPS Compatibility Analysis report

