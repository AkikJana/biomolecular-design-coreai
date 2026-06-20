# MPS Compatibility Analysis

## Core Summary
The Boltz codebase contains several hardcoded CUDA device constraints, hardcoded CUDA-only empty cache calls, and hardcoded `autocast("cuda", enabled=False)` wrappers that prevent native execution on Apple Silicon MPS. Replacing these with dynamic device checks, MPS-compatible cache clearing, and utilizing the existing `autocast_device_type` utility will allow seamless, native execution on both Apple Silicon (MPS) and CPU backends.

---

## 1. Hardcoded CUDA & Device Norm Checks
These checks assume the hardware is either CUDA-enabled or falls back to CPU, ignoring Apple Silicon's MPS backend, or they call CUDA-only functions that crash on MPS.

| File | Line(s) | Verbatim Code / Context | Issue / Constraint |
|---|---|---|---|
| `boltz/src/boltz/model/models/boltz2.py` | 364–365 | `torch.cuda.is_available() and torch.cuda.get_device_properties(...).major >= 8.0` | Assumes kernels are CUDA-only; safely short-circuits on non-CUDA, but could be cleaner. |
| `boltz/src/boltz/model/models/boltz2.py` | 989, 998 | `device="cuda" if torch.cuda.is_available() else "cpu"` | Used in `gradient_norm` and `parameter_norm`. Hardcoded fallback device when no parameters are present. |
| `boltz/src/boltz/model/models/boltz2.py` | 1026, 1046, 1127 | `torch.cuda.empty_cache()` | Hardcoded CUDA cache clearing inside out-of-memory handlers. Will crash on MPS. |
| `boltz/src/boltz/model/models/boltz1.py` | 267–268 | Similar stage capability checks. | CUDA-only checks. |
| `boltz/src/boltz/model/models/boltz1.py` | 609, 615 | `torch.tensor([p.norm()...])` | Pulls tensors from device back to CPU host to calculate norms. Slow. |
| `boltz/tests/test_regression.py` | 27 | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Falls back to CPU instead of utilizing MPS on Apple Silicon machines. |

---

## 2. Hardcoded Autocast Wrappers
Calling `torch.autocast("cuda", enabled=False)` or `torch.cuda.amp.autocast(enabled=False)` will crash on Apple Silicon machines where CUDA is unavailable. We must use the dynamic `autocast_device_type` utility function to resolve the active device type.

### In boltz2.py and diffusionv2.py:
- **`boltz/src/boltz/model/modules/diffusionv2.py`**
  - **Line 513**: `with torch.autocast("cuda", enabled=False):` inside the `self.alignment_reverse_diff` alignment block.
  - **Line 603**: `with torch.autocast("cuda", enabled=False):` inside `compute_loss`.

### Other codebase locations:
- **`boltz/src/boltz/model/layers/triangular_attention/primitives.py`**:
  - **Lines 106, 119**: `with torch.autocast("cuda", enabled=False):` in `Linear.forward`.
  - **Line 139**: `with torch.autocast("cuda", enabled=False):` in `LayerNorm.forward`.
  - **Line 167**: `with torch.autocast("cuda", enabled=False):` in `softmax_no_cast`.
- **`boltz/src/boltz/model/layers/attention.py`**:
  - **Line 223**: `with torch.autocast("cuda", enabled=False):` in `AttentionPairBias.forward`.
- **`boltz/src/boltz/model/layers/pairformer.py`**:
  - **Line 105**: `with torch.autocast("cuda", enabled=False):` in `PairformerLayer.forward`.
- **`boltz/src/boltz/model/layers/attentionv2.py`**:
  - **Line 99**: `with torch.autocast("cuda", enabled=False):` in `AttentionPairBias.forward`.
- **`boltz/src/boltz/model/layers/confidence_utils.py`**:
  - **Line 26**: `with torch.amp.autocast("cuda", enabled=False):` in `compute_frame_pred`.
- **`boltz/src/boltz/model/loss/distogramv2.py`**:
  - **Line 27**: `with torch.autocast("cuda", enabled=False):` in `distogram_loss`.
- **`boltz/src/boltz/model/loss/bfactor.py`**:
  - **Line 24**: `with torch.autocast("cuda", enabled=False):` in `bfactor_loss_fn`.
- **`boltz/src/boltz/model/loss/confidencev2.py`**:
  - **Line 98**: `with torch.autocast("cuda", enabled=False):` in `resolved_loss`.
  - **Line 149**: `with torch.cuda.amp.autocast(enabled=False):` in `get_target_lddt`.
  - **Line 362**: `with torch.cuda.amp.autocast(enabled=False):` in `get_target_pae`.
  - **Line 523**: `with torch.cuda.amp.autocast(enabled=False):` in `get_target_pde`.
- **`boltz/src/boltz/model/modules/diffusion.py`**:
  - **Line 694**: `with torch.autocast("cuda", enabled=False):` in step loop.
  - **Line 820**: `with torch.no_grad(), torch.autocast("cuda", enabled=False):` in `compute_loss`.
- **`boltz/src/boltz/model/modules/trunkv2.py`**:
  - **Lines 311, 462**: `with torch.autocast(device_type="cuda", enabled=False):` in template feature calculations.
- **`boltz/src/boltz/model/modules/encodersv2.py`**:
  - **Line 312**: `with torch.autocast("cuda", enabled=False):` in `AtomEncoder.forward`.
  - **Line 481**: `with torch.autocast("cuda", enabled=False):` in `AtomAttentionEncoder.forward`.
  - **Line 544**: `with torch.autocast("cuda", enabled=False):` in `AtomAttentionDecoder.forward`.

---

## 3. Float64 Casts
A comprehensive search reveals **no explicit Float64 or double precision casts** in the model source code. All instances of the word `"double"` in the codebase refer to chemical structural entities (e.g., `"planar_double_bond_index"` or `"double_bond_improper_index"`). As such, no precision conversions from Float64 are required.

---

## 4. Code Modification Strategy

We propose a four-part modification strategy to achieve Apple Silicon MPS compatibility.

### Step 1: Create a Device-Agnostic Cache Clearing Wrapper
Add a utility function in `boltz/src/boltz/model/modules/utils.py`:
```python
def empty_device_cache(device: torch.device | str | None) -> None:
    """Clear memory cache for the active hardware accelerator."""
    if device is None:
        return
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if "cuda" in device_type and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif "mps" in device_type and hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()
```

Replace all references to `torch.cuda.empty_cache()` in `boltz2.py` with:
```python
from boltz.model.modules.utils import empty_device_cache
# ...
empty_device_cache(self.device)
```

### Step 2: Correct Norm Computations in Boltz Models
Modify `gradient_norm` and `parameter_norm` in `boltz/src/boltz/model/models/boltz2.py` (and `boltz1.py`) to return tensors directly bound to `self.device`:
```python
    def gradient_norm(self, module):
        parameters = [
            p.grad.norm(p=2) ** 2
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if len(parameters) == 0:
            return torch.tensor(0.0, device=self.device)
        norm = torch.stack(parameters).sum().sqrt()
        return norm
```

### Step 3: Replace Hardcoded Autocast Calls
In all identified codebase locations, import `autocast_device_type` from `boltz.model.modules.utils` and rewrite the wrappers to dynamically resolve device capability.

**Example 1 (in `diffusionv2.py` line 513):**
```python
# Before
with torch.autocast("cuda", enabled=False):
    atom_coords_noisy = weighted_rigid_align(...)

# After
from boltz.model.modules.utils import autocast_device_type
# ...
with torch.autocast(autocast_device_type(atom_coords_noisy.device.type), enabled=False):
    atom_coords_noisy = weighted_rigid_align(...)
```

**Example 2 (in `confidencev2.py` line 149):**
```python
# Before
with torch.cuda.amp.autocast(enabled=False):

# After
from boltz.model.modules.utils import autocast_device_type
# ...
with torch.autocast(autocast_device_type(pred_atom_coords.device.type), enabled=False):
```

### Step 4: Update Regression Test Suite Device Configuration
Modify `boltz/tests/test_regression.py` to leverage MPS if available:
```python
# Before
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# After
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
```
