# MPS Compatibility Analysis for Boltz Codebase

This report analyzes the Boltz codebase for compatibility issues with Apple Silicon MPS (Metal Performance Shaders) native execution, focusing on hardcoded CUDA checks, device norm logic, autocast wrappers, and Float64 casts.

---

## 1. Summary of Findings

- **Hardcoded Autocast Wrappers**: The codebase frequently disables autocasting for precision-sensitive operations using `with torch.autocast("cuda", enabled=False):` or `with torch.cuda.amp.autocast(enabled=False):`. This hardcoded `"cuda"` device type raises errors or warnings on Apple Silicon MPS and CPU environments.
- **Hardcoded Device Norm Logic**: Gradient and parameter norm calculations in `boltz2.py` use a hardcoded device selection `"cuda" if torch.cuda.is_available() else "cpu"`, which prevents returning tensors on MPS when the model is running on Apple Silicon.
- **Hardcoded Cache Clearing**: Out-of-memory error recovery blocks call `torch.cuda.empty_cache()` unconditionally, which does not clear memory on MPS.
- **Float64/Double Casts**: While no hardcoded `float64` casts are present in `boltz2.py` or `diffusionv2.py`, MPS does not support `Float64` computations. Any inputs parsed from file formats must be verified to ensure they are strictly in `Float32` before device placement.
- **Wrapper Routing**: The wrapper in `src/boltz_wrapper.py` hardcodes the device setup to CUDA or CPU, missing the MPS device option entirely.
- **Mixed Precision Configuration**: The global inference trainer in `main.py` uses `"bf16-mixed"` for Boltz-2, which is unsupported or highly unstable under MPS and needs fallback routing to `"16-mixed"` or `32` (Float32).

---

## 2. Detailed Findings and Code Locations

### 2.1 Hardcoded Autocast Wrappers

The following files and line numbers use hardcoded CUDA autocast wrappers:

#### A. `boltz/src/boltz/model/modules/diffusionv2.py`
* **Line 513**:
  ```python
  if self.alignment_reverse_diff:
      with torch.autocast("cuda", enabled=False):
          atom_coords_noisy = weighted_rigid_align(...)
  ```
* **Line 603**:
  ```python
  def compute_loss(self, feats, out_dict, ...):
      with torch.autocast("cuda", enabled=False):
          denoised_atom_coords = out_dict["denoised_atom_coords"].float()
  ```

#### B. `boltz/src/boltz/model/modules/diffusion.py`
* **Line 694**: `with torch.autocast("cuda", enabled=False):`
* **Line 820**: `with torch.no_grad(), torch.autocast("cuda", enabled=False):`

#### C. `boltz/src/boltz/model/layers/triangular_attention/primitives.py`
* **Lines 106, 119**:
  ```python
  with torch.autocast("cuda", enabled=False):
      bias = self.bias.to(dtype=self.precision) ...
  ```
* **Line 139** (in `LayerNorm`):
  ```python
  if d is torch.bfloat16:
      with torch.autocast("cuda", enabled=False):
          out = nn.functional.layer_norm(...)
  ```
* **Line 167** (in `softmax_no_cast`):
  ```python
  if d is torch.bfloat16:
      with torch.autocast("cuda", enabled=False):
          s = torch.nn.functional.softmax(t, dim=dim)
  ```

#### D. `boltz/src/boltz/model/layers/attentionv2.py`
* **Line 99**:
  ```python
  with torch.autocast("cuda", enabled=False):
      attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
  ```

#### E. `boltz/src/boltz/model/layers/attention.py`
* **Line 223**:
  ```python
  with torch.autocast("cuda", enabled=False):
      attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
  ```

#### F. `boltz/src/boltz/model/layers/pairformer.py`
* **Line 105**:
  ```python
  with torch.autocast("cuda", enabled=False):
      s_normed = self.pre_norm_s(s.float())
  ```

#### G. `boltz/src/boltz/model/modules/trunkv2.py`
* **Lines 311, 462**:
  ```python
  with torch.autocast(device_type="cuda", enabled=False):
      cb_dists = torch.cdist(cb_coords, cb_coords)
  ```

#### H. `boltz/src/boltz/model/modules/encodersv2.py`
* **Lines 312, 481, 544**:
  ```python
  with torch.autocast("cuda", enabled=False):
      ...
  ```

#### I. `boltz/src/boltz/model/loss/confidencev2.py`
* **Line 98**: `with torch.autocast("cuda", enabled=False):`
* **Lines 149, 362, 523**:
  ```python
  with torch.cuda.amp.autocast(enabled=False):
  ```

---

### 2.2 Device Norm Logic

Calculations of gradient norms and parameter norms in the global Lightning entrypoint contain hardcoded CUDA assumptions:

#### A. `boltz/src/boltz/model/models/boltz2.py`
* **Lines 989, 998**:
  ```python
  def gradient_norm(self, module):
      ...
      if len(parameters) == 0:
          return torch.tensor(
              0.0, device="cuda" if torch.cuda.is_available() else "cpu"
          )
  ```

---

### 2.3 Hardcoded Cache Clearing

When memory limits are exceeded, Boltz attempts to free GPU cache via `torch.cuda.empty_cache()` but fails to clear MPS memory limits:

#### A. `boltz/src/boltz/model/models/boltz2.py`
* **Lines 1026, 1046, 1127**:
  ```python
  except RuntimeError as e:
      if "out of memory" in str(e):
          ...
          torch.cuda.empty_cache()
  ```

#### B. `boltz/src/boltz/model/models/boltz1.py`
* **Lines 633, 687, 1201**:
  ```python
  torch.cuda.empty_cache()
  ```

---

### 2.4 Wrapper Routing and Trainer Configuration

#### A. `src/boltz_wrapper.py`
* **Line 26**:
  ```python
  self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
  ```

#### B. `boltz/src/boltz/main.py`
* **Line 1274**:
  ```python
  precision=32 if model == "boltz1" else "bf16-mixed",
  ```
  Note: `bf16-mixed` is unsupported/unstable under MPS.

---

## 3. Modification Strategy and Action Plan

To enable native execution on Apple Silicon MPS without breaking existing CUDA support, we propose a multi-stage modification strategy:

### Step 1: Utilize `autocast_device_type` for Dynamic Autocasting
Replace all instances of `torch.autocast("cuda", enabled=False)` and `torch.cuda.amp.autocast(enabled=False)` with a device-agnostic call using the helper `autocast_device_type` from `boltz.model.modules.utils`. 

* **Before**:
  ```python
  with torch.autocast("cuda", enabled=False):
  ```
* **After**:
  ```python
  from boltz.model.modules.utils import autocast_device_type
  
  with torch.autocast(autocast_device_type(device_type=some_tensor.device.type), enabled=False):
  ```
  *(Note: Import the helper from `boltz.model.modules.utils`. For files like `triangular_attention/primitives.py`, import dynamically or at module level, as `utils.py` does not depend on layers and will not create a circular dependency.)*

### Step 2: Make Device Norms Self-Referential
For gradient/parameter norm helpers, query the device of the module parameters dynamically instead of hardcoding CUDA.

* **Before**:
  ```python
  if len(parameters) == 0:
      return torch.tensor(
          0.0, device="cuda" if torch.cuda.is_available() else "cpu"
      )
  ```
* **After**:
  ```python
  if len(parameters) == 0:
      try:
          device = next(module.parameters()).device
      except StopIteration:
          device = "cpu"
      return torch.tensor(0.0, device=device)
  ```

### Step 3: Implement Device-Aware Cache Clearing
Create or call a wrapper that handles both CUDA and MPS cache disposal.

* **Implementation**:
  ```python
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  elif hasattr(torch, "mps") and torch.backends.mps.is_available():
      torch.mps.empty_cache()
  ```

### Step 4: Resolve Wrapper Routing
Update `src/boltz_wrapper.py` to route to `"mps"` if specified and available.

* **Implementation**:
  ```python
  self.device = torch.device(
      "cuda" if use_gpu and torch.cuda.is_available() else (
          "mps" if use_gpu and torch.backends.mps.is_available() else "cpu"
      )
  )
  ```

### Step 5: Route Trainer Precision Dynamically
Update `main.py` to select the mixed-precision backend dynamically if MPS is requested.

* **Implementation**:
  ```python
  if accelerator == "mps" and model == "boltz2":
      precision = "16-mixed" # or 32 for Float32 safety
  else:
      precision = 32 if model == "boltz1" else "bf16-mixed"
  ```

---

## 4. Verification and Testing Method

Once these changes are applied, they can be verified on an Apple Silicon machine using the following tests:
1. Run the test suite using `pytest`:
   ```bash
   pytest tests/test_boltz_wrapper.py
   pytest tests/test_diffusion_dpo.py
   ```
2. Validate that the prediction command can run on MPS:
   ```bash
   boltz predict --data <input_path> --out_dir <out_path> --accelerator mps --devices 1
   ```
3. Check the logs to ensure:
   - PyTorch Lightning does not crash with `bf16-mixed` errors.
   - No `RuntimeError: "mps_matmul" not implemented for 'Double'` or autocast errors are raised.
   - The device reported is indeed `mps`.
