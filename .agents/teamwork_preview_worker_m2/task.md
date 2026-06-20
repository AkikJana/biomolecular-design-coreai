# Task for Worker - Milestone 2: Apple Silicon MPS Compatibility

## Objective
Implement the code modifications required to make the Boltz model compatible with Apple Silicon MPS execution, following the findings and proposed strategy in the synthesized analysis.

## Proposed Fix Strategy
1. **Dynamic Autocast**:
   Replace all instances of `with torch.autocast("cuda", ...)` and `with torch.cuda.amp.autocast(...)` in the codebase with a dynamic autocast call that detects the active device. Use the existing `autocast_device_type` utility from `boltz.model.modules.utils` (or write a dynamic helper) and pass the device type of the current context tensors/parameters.
2. **Correct Device Norm Tensors**:
   In `boltz/src/boltz/model/models/boltz2.py`, inside the `gradient_norm` and `parameter_norm` calculations, ensure that when returning 0.0 tensors for empty parameters, they are created on `self.device` instead of hardcoded `"cuda" if torch.cuda.is_available() else "cpu"`.
3. **Empty Cache Helper**:
   Define a helper function that clears cache for the active device type (CUDA or MPS) dynamically. Replace hardcoded `torch.cuda.empty_cache()` calls in the exception handlers of `boltz2.py` (and any other files) with this helper.

## Verification
Run pytest on `boltz/tests/test_regression.py` or the whole test suite to verify that the code compiles, passes tests, and doesn't crash on device-related or autocast runtime errors.

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
