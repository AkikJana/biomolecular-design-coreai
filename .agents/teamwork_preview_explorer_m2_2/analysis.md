# Handoff Analysis: Apple Silicon MPS Compatibility (Milestone 2)

## Summary of Findings
An analysis of the Boltz codebase was performed to identify obstacles preventing native execution on Apple Silicon (MPS). No `Float64` casts or operations were found. However, several critical compatibility barriers were identified:
1. **Hardcoded Autocast Device Types**: Autocast wrappers throughout the model layer, module, and loss definitions are hardcoded to `"cuda"`. When executing on MPS or CPU, these calls can fail or not disable autocast as intended.
2. **Device Norm Hardcoding**: Device norm logic in `gradient_norm` and `parameter_norm` (within `boltz2.py` and `boltz1.py`) defaults to `"cuda" if torch.cuda.is_available() else "cpu"`, which causes device mismatch runtime errors on MPS because the parameters reside on `mps` but the zero tensor is created on `cpu`.
3. **CUDA-Only empty_cache Calls**: Memory clearing operations in exception handlers are hardcoded to `torch.cuda.empty_cache()`, which will not release memory when running on Apple Silicon.

---

## 1. Observations

### 1.1 Hardcoded CUDA and Accelerator Checks
* **Location 1**: `boltz/src/boltz/model/models/boltz2.py:363-366`
  ```python
  if stage == "predict" and not (
      torch.cuda.is_available()
      and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0  # noqa: PLR2004
  ):
      self.use_kernels = False
  ```
* **Location 2**: `boltz/src/boltz/model/models/boltz2.py:1026, 1046, 1127`
  ```python
  torch.cuda.empty_cache()
  ```
* **Location 3**: `boltz/src/boltz/model/loss/diffusionv2.py:51-53`
  ```python
  U, S, V = torch.linalg.svd(
      cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
  )
  ```

### 1.2 Device Norm Logic
* **Location 1**: `boltz/src/boltz/model/models/boltz2.py:988-990`
  ```python
  if len(parameters) == 0:
      return torch.tensor(
          0.0, device="cuda" if torch.cuda.is_available() else "cpu"
      )
  ```
* **Location 2**: `boltz/src/boltz/model/models/boltz2.py:996-999`
  ```python
  if len(parameters) == 0:
      return torch.tensor(
          0.0, device="cuda" if torch.cuda.is_available() else "cpu"
      )
  ```

### 1.3 Autocast Wrapper Calls
* **Location 1**: `boltz/src/boltz/model/modules/diffusionv2.py:513`
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **Location 2**: `boltz/src/boltz/model/modules/diffusionv2.py:603`
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **Location 3**: `boltz/src/boltz/model/loss/confidencev2.py:98`
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **Location 4**: `boltz/src/boltz/model/loss/confidencev2.py:149, 362, 523`
  ```python
  with torch.cuda.amp.autocast(enabled=False):
  ```
* **Location 5**: `boltz/src/boltz/model/modules/trunkv2.py:311, 462`
  ```python
  with torch.autocast(device_type="cuda", enabled=False):
  ```
* **Location 6**: `boltz/src/boltz/model/modules/encodersv2.py:312, 481, 544`
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **Location 7**: Layer files:
  - `boltz/src/boltz/model/layers/triangular_attention/primitives.py:106, 119, 139, 167`
  - `boltz/src/boltz/model/layers/attention.py:223`
  - `boltz/src/boltz/model/layers/pairformer.py:105`
  - `boltz/src/boltz/model/layers/attentionv2.py:99`
  - `boltz/src/boltz/model/layers/confidence_utils.py:26`
  - `boltz/src/boltz/model/loss/distogramv2.py:27`
  - `boltz/src/boltz/model/loss/bfactor.py:24`

### 1.4 Float64 Casts
* **Observation**: Ripgrep/grep searches across all `.py` files inside `boltz/src` confirm there are no Float64 casts (`.to(torch.float64)` or `.double()`) related to float precision operations.

---

## 2. Logic Chain

1. **Autocast Device Validation**: In PyTorch, `torch.autocast` validates the string passed to its `device_type` argument. Hardcoding `"cuda"` will cause an error on environments where CUDA is not available or when using MPS if the backend tries to validate the device compatibility. More importantly, when running on MPS, specifying `"cuda"` does not disable autocasting for the active `"mps"` device.
2. **Device Mismatch in Norm Calculation**: In `boltz2.py`, inside `gradient_norm` and `parameter_norm`, returning a tensor explicitly on `"cpu"` when parameters are on `"mps"` (because `torch.cuda.is_available()` is `False`) causes a device mismatch when these tensors are stacked or sum-reduced together. This is a fatal runtime exception.
3. **Out-of-Memory Handling on MPS**: Hardcoded `torch.cuda.empty_cache()` calls do not clear MPS memory cache. This prevents Apple Silicon GPUs from reclaiming unused memory dynamically when OOM bounds are approached.
4. **Conclusion**: Modifying the autocast context manager arguments, making cache clearing aware of the MPS device, and resolving the norm's tensor device dynamically will enable native, error-free execution on Apple Silicon.

---

## 3. Caveats
- **Hardware Capabilities**: MPS support for `bfloat16` is hardware-dependent (e.g. M1 Pro/Max, M2, M3 support it; older basic M1 does not). Since the CLI currently runs the trainer with precision `"bf16-mixed"` by default for Boltz2, running on an older M1 Mac might throw a bfloat16 exception if the device precision is not overridden or configured.
- **Custom Kernels**: PyTorch kernels or Triton-based operations will be disabled (`self.use_kernels = False`), which degrades performance relative to running on CUDA. Native PyTorch operations are run instead.

---

## 4. Conclusion & Code Modification Strategy

To support Apple Silicon (MPS) native execution, we propose the following changes:

### Proposal A: Use Dynamic Autocast Wrapper
Replace all hardcoded autocast occurrences with the existing utility `autocast_device_type` from `boltz.model.modules.utils`.
* **Example transformation in `diffusionv2.py`**:
  ```python
  # Before
  with torch.autocast("cuda", enabled=False):

  # After
  from boltz.model.modules.utils import autocast_device_type
  with torch.autocast(autocast_device_type(atom_coords_denoised.device.type), enabled=False):
  ```
* **Example transformation in `loss/confidencev2.py`**:
  ```python
  # Before
  with torch.cuda.amp.autocast(enabled=False):

  # After
  from boltz.model.modules.utils import autocast_device_type
  with torch.autocast(autocast_device_type(pred_atom_coords.device.type), enabled=False):
  ```

### Proposal B: Resolve Device Dynamically for Norm Calculations
* **In `boltz2.py` (and `boltz1.py`)**:
  ```python
  # Before
  if len(parameters) == 0:
      return torch.tensor(
          0.0, device="cuda" if torch.cuda.is_available() else "cpu"
      )

  # After
  if len(parameters) == 0:
      first_param = next(module.parameters(), None)
      device = first_param.device if first_param is not None else ("cuda" if torch.cuda.is_available() else "cpu")
      return torch.tensor(0.0, device=device)
  ```

### Proposal C: Implement Unified Cache Clearing Helper
Introduce a helper function to handle cache clearing:
```python
def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        import torch.mps
        torch.mps.empty_cache()
```
Replace all direct calls to `torch.cuda.empty_cache()` with `empty_device_cache()`.

### Proposal D: Make Precision Configurable
In `main.py`, allow the user to control the precision (e.g. `--precision 32`) or default to `32` if the accelerator is not `gpu`/`cuda` (e.g. when `accelerator == "cpu"` or `"mps"`).

---

## 5. Verification Method
1. **Validation Checks**: Inspect each file to confirm that no hardcoded `torch.autocast("cuda")` or `torch.cuda.amp.autocast` references remain.
2. **Execution Test**: Set up a dummy prediction using the MPS accelerator to verify that the forward pass succeeds without device mismatch exceptions:
   ```bash
   boltz predict <input.yaml> --accelerator mps
   ```
3. **Unit Tests**: Run layer-specific tests to verify that no functional regressions were introduced:
   ```bash
   pytest tests/model/layers/
   ```
