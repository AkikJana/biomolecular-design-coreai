# Handoff Report: Apple Silicon MPS Compatibility Investigation

**Date**: 2026-06-20T15:54:30Z  
**Author**: MPS Compatibility Explorer 1  
**Milestone**: Milestone 2: Apple Silicon MPS Compatibility  

---

## 1. Observation

During static analysis of the Boltz codebase, multiple occurrences of hardcoded CUDA checks, device norm calculations, OOM handling, and device routing were identified:

### 1.1 Hardcoded Autocast Wrappers
Precision-sensitive blocks disable autocasting by hardcoding `"cuda"`:
* **`boltz/src/boltz/model/modules/diffusionv2.py:513`**
  ```python
  if self.alignment_reverse_diff:
      with torch.autocast("cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/modules/diffusionv2.py:603`**
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/layers/triangular_attention/primitives.py:106, 119, 139, 167`**
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/layers/attentionv2.py:99`** and **`attention.py:223`**
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/layers/pairformer.py:105`**
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/modules/trunkv2.py:311, 462`**
  ```python
  with torch.autocast(device_type="cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/modules/encodersv2.py:312, 481, 544`**
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **`boltz/src/boltz/model/loss/confidencev2.py:98`** (`torch.autocast("cuda", ...)`) and **`Lines 149, 362, 523`** (`torch.cuda.amp.autocast(...)`)

### 1.2 Device Norm Logic
* **`boltz/src/boltz/model/models/boltz2.py:989, 998`**
  ```python
  return torch.tensor(
      0.0, device="cuda" if torch.cuda.is_available() else "cpu"
  )
  ```

### 1.3 Hardcoded empty_cache
* **`boltz/src/boltz/model/models/boltz2.py:1026, 1046, 1127`**
  ```python
  torch.cuda.empty_cache()
  ```
* **`boltz/src/boltz/model/models/boltz1.py:633, 687, 1201`**
  ```python
  torch.cuda.empty_cache()
  ```

### 1.4 Wrapper Device Setup
* **`src/boltz_wrapper.py:26`**
  ```python
  self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
  ```

---

## 2. Logic Chain

1. **Autocast Check**: In PyTorch, calling `torch.autocast` with `device_type="cuda"` on systems lacking CUDA (like Apple Silicon macOS) will trigger warnings or errors, preventing native execution. Because the model must run on MPS/CPU, hardcoded CUDA autocast references are incompatible.
2. **Device Norms**: When calculating gradient and parameter norms in `boltz2.py`, returning a dummy CPU tensor when the model is on MPS violates PyTorch device placement consistency (where all tensors contributing to loss/logging should be on the same device).
3. **Memory Limits**: `torch.cuda.empty_cache()` does not clean MPS memory allocations, so OOM exception handlers fail to reclaim memory on macOS.
4. **Wrapper Routing**: `src/boltz_wrapper.py` routes strictly to CUDA or CPU, forcing macOS systems to run on CPU (local surrogate or real model evaluation) instead of exploiting the MPS GPU.
5. **Mixed Precision**: `bf16-mixed` is set globally for Boltz-2 in `main.py`, but `bfloat16` operators (such as layer norm or matmul) crash on MPS, so it must be dynamically routed to `16-mixed` or `32`.

---

## 3. Caveats

- **External Libraries**: DeepMind / AlQuraishi JIT-compiled kernels (like those in `triangular_attention`) are strictly CUDA-only. Although `self.use_kernels = False` is correctly handled under CPU/MPS, we assume these standard CPU/MPS PyTorch fallback implementations are functionally equivalent and mathematically correct.
- **Float64 Casts**: No hardcoded `float64` casts were discovered in the model files. However, if any float64 arrays are ingested by data loaders and moved to MPS, PyTorch will crash. The featurizer correctly outputs `float32`, but any new custom code or inputs must respect this constraint.

---

## 4. Conclusion

The Boltz structure prediction model cannot run natively on Apple Silicon MPS due to:
1. Hardcoded `"cuda"` strings in autocasting context managers in 10+ files.
2. Device norm tensors hardcoding CUDA/CPU.
3. Hardcoded `torch.cuda.empty_cache()` in error handlers.
4. Missing MPS routing inside `src/boltz_wrapper.py` and mixed-precision trainer setup.

A dynamic modification strategy has been formulated (detailed in `analysis.md`) that replaces these hardcoded instances with device-agnostic helpers (`autocast_device_type(device_type)`, dynamic device queries, and conditional cache clearing) to safely enable full native MPS execution.

---

## 5. Verification Method

To verify the modifications:
1. Verify the code parses and compiles on a CPU/MPS system:
   ```bash
   python -c "import boltz"
   ```
2. Run the test suite:
   ```bash
   pytest tests/test_boltz_wrapper.py
   pytest tests/test_diffusion_dpo.py
   ```
3. Run structure prediction on MPS:
   ```bash
   boltz predict --data tests/data/input.yaml --out_dir test_out --accelerator mps --devices 1
   ```
   *Verification Success Condition*: The trainer starts, allocates MPS memory, and produces a structure prediction without raising autocast type, device placement, or unsupported operator (`Double`) errors.
