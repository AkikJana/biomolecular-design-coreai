# Handoff Report: Apple Silicon MPS Compatibility Investigation (Milestone 2)

## 1. Observation

Direct observations and file search logs show the following code fragments:
- **`boltz/src/boltz/model/models/boltz2.py`**
  - **Lines 363-366**: Hardcoded check for CUDA properties:
    ```python
    if stage == "predict" and not (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
    ):
    ```
  - **Lines 1026, 1046, 1127**: Hardcoded memory release:
    ```python
    torch.cuda.empty_cache()
    ```
  - **Lines 988-990, 997-999**: Norm calculation hardcoded fallbacks to CUDA/CPU:
    ```python
    return torch.tensor(
        0.0, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    ```

- **`boltz/src/boltz/model/modules/diffusionv2.py`**
  - **Lines 513, 603**: Hardcoded CUDA autocast disabling:
    ```python
    with torch.autocast("cuda", enabled=False):
    ```

- **Other files in `src/boltz/model/`**:
  - Found multiple instances of `with torch.autocast("cuda", enabled=False):` or `with torch.cuda.amp.autocast(enabled=False):` in `loss/confidencev2.py`, `modules/trunkv2.py`, `modules/encodersv2.py`, `layers/triangular_attention/primitives.py`, `layers/attention.py`, `layers/pairformer.py`, `layers/attentionv2.py`, `layers/confidence_utils.py`, `loss/distogramv2.py`, and `loss/bfactor.py`.
  - Found SVD driver condition in `loss/diffusionv2.py:52` and `loss/diffusion.py:64`:
    ```python
    cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    ```
  - No occurrences of `Float64` or `.double()` casting were found inside the model definition.

Detailed findings are recorded in `/Users/akikjana/Documents/BiomolecularDesign/.agents/teamwork_preview_explorer_m2_2/analysis.md`.

---

## 2. Logic Chain

1. **Autocast Device Validation**: PyTorch validates device strings in `torch.autocast`. Passing `"cuda"` to autocast will fail on CPU/MPS machines if CUDA is not available or if the PyTorch build behaves strictly. More importantly, it does not disable autocast on the active MPS device when running on Apple Silicon, leading to precision mismatches.
2. **Device Mismatch in Norms**: When running on MPS, `torch.cuda.is_available()` returns `False`. The norm functions in `boltz2.py` fallback to allocating zero-value tensors on `"cpu"`. When PyTorch tries to perform stack or sum operations between the parameters on `"mps"` and the returned zero-value tensor on `"cpu"`, it throws a device mismatch runtime error.
3. **Memory Management**: MPS memory allocation behaves differently than CUDA, but `torch.mps.empty_cache()` exists for releasing cache. Replacing `torch.cuda.empty_cache()` with a device-agnostic wrapper will ensure cache is cleaned on all acceleration backends.
4. **No Float64 issues**: Since no `Float64` casts or conversions exist, there is no risk of encountering unsupported double precision operators on MPS.

---

## 3. Caveats
- MPS `bfloat16` capability is hardware-dependent (requires Apple Silicon chips starting from M1 Pro/Max or M2/M3). Systems lacking native MPS `bfloat16` support may require running in float32 precision mode.
- In `main.py`, precision is currently hardcoded to `"bf16-mixed"` for Boltz2. This must be handled or configurable to prevent lightning validation failures on systems with older Apple Silicon processors.

---

## 4. Conclusion

To enable native, error-free Apple Silicon (MPS) support, the codebase must be modified to:
- Resolve autocast device type dynamically using the tensor's device type via the existing `autocast_device_type` helper.
- Calculate norms using tensors allocated on the same device as the module's parameters.
- Empty cache on both CUDA and MPS devices dynamically.
- Make precision configurable (or fall back to fp32 on non-CUDA environments).

A detailed strategy is described in the `analysis.md` file.

---

## 5. Verification Method

To verify the proposed changes:
1. Ensure no instances of `torch.autocast("cuda")` or `torch.cuda.amp.autocast` remain in the codebase.
2. Run structure prediction on an Apple Silicon device with MPS acceleration:
   ```bash
   boltz predict <input.yaml> --accelerator mps
   ```
3. Run the pytest suite to ensure no regressions:
   ```bash
   pytest tests/
   ```
