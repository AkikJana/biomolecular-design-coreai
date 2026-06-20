# Synthesized Analysis: Apple Silicon MPS Compatibility (Milestone 2)

## Consensus Findings
1. **Autocast Wrappers**: There are hardcoded `torch.autocast("cuda", enabled=False)` and `torch.cuda.amp.autocast(enabled=False)` wrappers across many files (specifically in `diffusionv2.py`, `diffusion.py`, `trunkv2.py`, `encodersv2.py`, `primitives.py`, `attention.py`, `pairformer.py`, `attentionv2.py`, `confidence_utils.py`, `distogramv2.py`, `bfactor.py`, and `confidencev2.py`). These fail on Apple Silicon (MPS/CPU backends) and prevent disabling autocast on the active MPS device.
2. **Device Norm Mismatch**: In `boltz/src/boltz/model/models/boltz2.py`, the `gradient_norm` and `parameter_norm` calculations return CPU tensors when parameters are empty (returning `"cuda" if torch.cuda.is_available() else "cpu"`). This causes runtime device mismatch errors during model training/evaluation when other parameters are on `"mps"`.
3. **Cache Clearing**: Hardcoded `torch.cuda.empty_cache()` calls exist in exception handlers in `boltz2.py` (and potentially `boltz1.py`). These raise exceptions on non-CUDA setups.
4. **CUDA Kernel Check**: In `boltz2.py` setup, there is a hardcoded CUDA major version check. If CUDA is not available, it correctly falls back to `self.use_kernels = False`, so this does not cause failure but is important for performance expectations.
5. **No Float64 Casting**: There are no precision-based Float64 casts in the model definition. All occurrences of the word `double` refer to chemical double bonds.

## Unique Insights & Resolved Disagreements
- **SVD Driver Option**: `loss/diffusionv2.py` and `loss/diffusion.py` check `.is_cuda` before selecting the `"gesvd"` driver for SVD:
  `cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None`
  Since `is_cuda` is False for MPS, this correctly falls back to `None` (which runs CPU-like algorithm on MPS/CPU, avoiding CUDA-only driver failures). This check is safe.
- **Precision Validation (MPS bfloat16)**: Mixed precision (`bf16-mixed`) is default in `main.py`. Since MPS bfloat16 support is hardware-dependent (M1 Pro/Max or later), running on older Apple Silicon may require fallback to float32 (`16-mixed` or `32` in PyTorch Lightning). The implementation should handle the accelerator and precision configuration dynamically or allow override.

## Proposed Fix Strategy
1. **Dynamic Autocast**: Convert all hardcoded autocast wrappers to resolve the device dynamically. We will import the existing `autocast_device_type` utility from `boltz.model.modules.utils` (which resolves `torch.amp.autocast_mode.is_autocast_available(device_type)` falling back safely to `"cpu"` or `"cuda"`) and call it based on the current tensor's device type:
   ```python
   with torch.autocast(autocast_device_type(device.type), enabled=False):
   ```
2. **Empty Cache Helper**: Define a helper function `empty_device_cache(device: torch.device)` in `boltz/src/boltz/model/modules/utils.py` or similar:
   ```python
   def empty_device_cache(device_type: str = "cuda"):
       if device_type == "cuda":
           torch.cuda.empty_cache()
       elif device_type == "mps":
           torch.mps.empty_cache()
   ```
   Replace `torch.cuda.empty_cache()` inside the model exception handlers with this helper.
3. **Correct Device Norm Tensors**: Modify `gradient_norm` and `parameter_norm` in `boltz2.py` to return the default zero tensor on the actual active device of the module parameters:
   ```python
   return torch.tensor(0.0, device=self.device)
   ```
4. **Test Suite Adaptation**: Update tests to support running on `mps` or fall back correctly if running on macOS.
