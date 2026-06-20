# Handoff Report: Apple Silicon MPS Compatibility Investigation (Milestone 2)

## 1. Observation
- In `boltz/src/boltz/model/models/boltz2.py`:
  - **Lines 364–365**:
    ```python
    if stage == "predict" and not (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0  # noqa: PLR2004
    ):
        self.use_kernels = False
    ```
  - **Lines 989 & 998**:
    ```python
    return torch.tensor(
        0.0, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    ```
  - **Lines 1026, 1046, & 1127**:
    ```python
    torch.cuda.empty_cache()
    ```
- In `boltz/src/boltz/model/modules/diffusionv2.py`:
  - **Line 513**:
    ```python
    with torch.autocast("cuda", enabled=False):
    ```
  - **Line 603**:
    ```python
    with torch.autocast("cuda", enabled=False):
    ```
- A global codebase search returned other instances of hardcoded `"cuda"` autocast wrappers:
  - `boltz/src/boltz/model/layers/triangular_attention/primitives.py:106,119,139,167`
  - `boltz/src/boltz/model/layers/attention.py:223`
  - `boltz/src/boltz/model/layers/pairformer.py:105`
  - `boltz/src/boltz/model/layers/attentionv2.py:99`
  - `boltz/src/boltz/model/layers/confidence_utils.py:26`
  - `boltz/src/boltz/model/loss/distogramv2.py:27`
  - `boltz/src/boltz/model/loss/bfactor.py:24`
  - `boltz/src/boltz/model/loss/confidencev2.py:98`
  - `boltz/src/boltz/model/loss/confidencev2.py:149,362,523` (using `torch.cuda.amp.autocast(enabled=False)`)
  - `boltz/src/boltz/model/modules/diffusion.py:694,820`
  - `boltz/src/boltz/model/modules/trunkv2.py:311,462`
  - `boltz/src/boltz/model/modules/encodersv2.py:312,481,544`
- A search for `double` and `float64` returned references to double bonds (e.g., `planar_double_bond_index` in `boltz/src/boltz/data/mol.py`) and did not find any precision-based casting to double/Float64.

---

## 2. Logic Chain
- **Step 1**: PyTorch `autocast` enforces device type checks at runtime. Calling `torch.autocast("cuda", enabled=False)` or `torch.cuda.amp.autocast(...)` when running on a platform where CUDA is not available (like Apple Silicon) will raise a `RuntimeError` or bypass disabling autocasting for other devices (such as MPS).
- **Step 2**: The utility `autocast_device_type(device_type: str)` in `boltz/src/boltz/model/modules/utils.py` uses `torch.amp.autocast_mode.is_autocast_available(device_type)` to check if autocasting is available for the given device type on the current platform, falling back safely to `"cpu"`.
- **Step 3**: By importing and applying `autocast_device_type(device_type)` dynamically resolved from the device type of input tensors (e.g. `x.device.type`), all hardcoded autocast wrappers can be made fully device-agnostic, preventing execution crashes on MPS/CPU.
- **Step 4**: Calls to `torch.cuda.empty_cache()` inside the exception handling blocks of `boltz2.py` will fail on MPS. Implementing a wrapper `empty_device_cache` that dispatches to `torch.mps.empty_cache()` when the device type is `"mps"` ensures proper memory recovery on Apple Silicon without raising exceptions.
- **Step 5**: Returning a hardcoded `"cuda" if torch.cuda.is_available() else "cpu"` tensor inside the `gradient_norm` and `parameter_norm` calculations in `boltz2.py` places the resulting norm on CPU when executing on MPS. Binding the returned tensor directly to `self.device` guarantees device consistency.

---

## 3. Caveats
- Custom Triton and CUDA kernels (e.g. checked in `boltz/src/boltz/model/models/boltz2.py` setup stage and tested in `test_kernels.py`) are strictly hardware-specific and cannot run on MPS/CPU. The fallback mechanism correctly sets `self.use_kernels = False` when CUDA is unavailable, so they do not impact compatibility, but they will not run on MPS.
- The investigation assumes that the PyTorch version installed on the target machine supports the MPS backend (typically PyTorch >= 2.0).

---

## 4. Conclusion
Native Apple Silicon MPS execution is currently blocked by hardcoded CUDA checks, CUDA cache clearing, and CUDA-only autocast wrappers. By introducing an `empty_device_cache` helper, converting the hardcoded autocast wrappers to use the existing `autocast_device_type` utility, binding norm calculations to `self.device`, and modifying regression tests to support MPS fallbacks, the model can run natively and correctly on Apple Silicon. No Float64 casts exist in the codebase.

---

## 5. Verification Method
- **Implementation**: The proposed plan can be verified by running the regression test suite.
- **Test Command**:
  ```bash
  .venv/bin/pytest boltz/tests/test_regression.py
  ```
- **Expected Outcome**: The tests should compile and execute without raising device-related errors or autocast warnings, successfully predicting the molecular structure on the active MPS or CPU device.
- **Invalidation Condition**: If `boltz/tests/test_regression.py` fails on macOS due to `autocast` invalid device type runtime errors or `torch.cuda` attribute failures, the modifications have not been fully or correctly applied.
