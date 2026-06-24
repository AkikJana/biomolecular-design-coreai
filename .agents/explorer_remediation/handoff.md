# Handoff Report: E2E Test Suite Integrity Violations Remediation Strategy

This handoff report outlines the direct observations, logical reasoning, and proposed fix strategy to resolve the integrity violations identified in the E2E test suite.

---

## 1. Observation

Direct inspection of the codebase has confirmed the following code structures:

1. **Self-Certifying Metric in `test_t4_1_human_insulin_monomer()`**:
   * File: `/Users/akikjana/Documents/BiomolecularDesign/tests/test_e2e_suite.py`
   * Line Numbers: 741–742
   * Verbatim code snippet:
     ```python
     # Verify simulated pLDDT value exists
     plddt = 80.0 + (sum(1 for c in insulin_seq if c in "LIVAMF") * 1.5)
     assert plddt >= 70.0
     ```
   * Context: The `plddt` score is calculated inside the test function itself based purely on sequence characters, without querying the predictor or confidence module.

2. **Facade Fallback in `SimulatedPredictor`**:
   * File: `/Users/akikjana/Documents/BiomolecularDesign/tests/test_e2e_suite.py`
   * Line Numbers: 67–80
   * Verbatim code snippet:
     ```python
     class SimulatedPredictor:
         def __init__(self):
             self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
         
         def predict(self, binder_seq: str, target_seq: str) -> np.ndarray:
             L = len(binder_seq)
             t = torch.linspace(0, 4 * math.pi, L)
             x = torch.stack([torch.sin(t) * 2.0, torch.cos(t) * 2.0, t], dim=-1)
             # Seed based on sequence values to be deterministic
             seed = sum(ord(c) for c in binder_seq) + sum(ord(c) for c in target_seq)
             torch.manual_seed(seed)
             noise = torch.randn_like(x) * 0.1
             x = x + noise
             return x.unsqueeze(0).numpy()
     ```
   * Context: This class is used when the `coreai` package is not available (i.e. `HAS_COREAI = False`), and generates coordinates using a static trigonometric formula.

3. **Predictor Interface in `predict_structure.py`**:
   * File: `/Users/akikjana/Documents/BiomolecularDesign/src/predict_structure.py`
   * Line Numbers: 93–94
   * Verbatim code snippet:
     ```python
     # Output coords shape: [1, L_binder, 3]
     return outputs["coords"].numpy()
     ```
   * Context: The production `DynamicStructurePredictor.predict()` method returns only coordinates and does not provide pLDDT or confidence scores.

---

## 2. Logic Chain

1. **Observation 1** shows that `test_t4_1_human_insulin_monomer()` calculates the `plddt` assertion targets internally using sequence content. Because it does not query any model or confidence modules, this assertion is **self-certifying** and does not evaluate real model performance.
2. **Observation 2** shows that `SimulatedPredictor` generates coordinates using a deterministic trigonometric formula (`torch.sin`/`torch.cos` sweep) plus noise. Because this bypasses the loading and forward propagation of any actual neural network layers, it acts as a **facade fallback** that does not test actual machine learning execution when the macOS `coreai` library is absent.
3. **Observation 3** shows that the `predict()` interface of the predictor expects to output a single coordinate array. To support confidence assertions without hardcoding them in tests, the `predict()` interface must be updated to return both coordinates and confidence values.
4. **Conclusion**: To resolve the integrity violations, we must replace `SimulatedPredictor` with a genuine lightweight PyTorch network (`LightweightPredictor`) that runs sequence embeddings through functional linear layers to predict structure and confidence. We must also update `test_e2e_suite.py` to assert directly on the returned model confidence rather than using the hardcoded sequence-based formula.

---

## 3. Caveats

* **Weights availability**: A full-sized Boltz-1 or Boltz-2 model checkpoint (typically >1GB) cannot be easily initialized or run in a fast CPU test suite under the strict test latency constraints (<1.0s). The proposed strategy utilizes a lightweight, functional PyTorch model with deterministic random initialization or small saved weights to guarantee immediate execution.
* **CoreAI compatibility**: When running on Apple Silicon with `coreai` present, the test uses `DynamicStructurePredictor`. Since this predictor also needs to output a confidence metric, the `plddt` predictor must be wrapped and run on top of CoreAI coordinate outputs to provide a standard interface.

---

## 4. Conclusion

The remediation strategy consists of:
1. Defining a lightweight structure prediction network (`LightweightStructurePredictorModel`) that runs a real PyTorch forward pass to compute 3D coordinate trajectories using sequence features and a helical baseline.
2. Defining a lightweight confidence estimator network (`LightweightConfidencePredictorModel`) that calculates pLDDT from the predicted coordinate distances and sequence features via PyTorch linear layers.
3. Standardizing the interface of both `DynamicStructurePredictor` and `LightweightPredictor` to return `(coords, plddt)`.
4. Updating `tests/test_e2e_suite.py` to unpack the returned `coords` and `plddt` from `predict()`, removing the mock sequence-based formula, and asserting on the returned model confidence directly.

---

## 5. Verification Method

To verify the remediation:
1. Run the test suite:
   ```bash
   ./.venv/bin/python run_e2e_tests.py
   ```
2. Verify that all 49 tests pass, specifically confirming:
   - `test_t4_1_human_insulin_monomer()` passes using the model-predicted `plddt` instead of the hardcoded formula.
   - `test_t4_2_hemoglobin_subunit_alpha()` executes in `< 1.0s` and verifies that the predicted `plddt` falls within the range `[50.0, 100.0]`.
   - `test_t4_5_large_scale_validation()` executes in `< 1.5s` and verifies structure shape.
3. Check the code of `tests/test_e2e_suite.py` to confirm that the mock mathematical formula `plddt = 80.0 + ...` has been entirely removed from the test file.
