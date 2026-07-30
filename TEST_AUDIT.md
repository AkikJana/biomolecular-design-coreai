# Test Audit: TEST_READY.md claims vs. actual assertions

Audited 2026-07-30 against `tests/test_e2e_suite.py`. Motivation: the suite had
never been executed by CI until recently (the workflow ran `unittest discover`
over a pytest-only suite and collected nothing), so its checklist had never been
checked against what the tests assert.

**Structurally sound.** All 49 tests named in TEST_READY.md exist. Four more are
defined but undocumented: `test_t1_f7_adaptive_lookahead_speculative`,
`test_t1_f8_biophysical_manifold_constraint_speculative`,
`test_t1_f9_bidirectional_codesign`, `test_t1_f10_quantization_aware_training`.
53 tests in this file; 98 in the whole suite.

**All 98 pass.** Nothing here is a failing test. The finding is that several
tests pass while verifying materially less than TEST_READY.md claims.

---

## A. Assertions that cannot fail — FIXED (see commit history)

### A1. `test_t4_1_human_insulin_monomer` — `assert plddt >= 70.0` — **FIXED**
`LightweightPredictor.predict_plddt` returns
`sigmoid(x) * 30.0 + 70.0`, whose infimum is **70.0**. The assertion is
tautological. The `else` branch is `80.0 + count * 1.5`, also always ≥ 70. Both
paths are unfailable.

Measured over 200 random initialisations: min 81.128, max 89.043.

### A2. `test_t4_5_large_scale_validation` — `assert reduction_pct > 30.0` — **FIXED**
```python
reduction_pct = (17640000 - 8912) / 17640000 * 100.0   # = 99.9495
assert reduction_pct > 30.0
```
Both operands are literals written in the test. No model output, tensor, or
measurement participates. This asserts `99.9495 > 30`. TEST_READY.md claims the
test verifies "activation memory reduction compared to baseline"; it measures
nothing. Same defect class as the hardcoded 87.5% and 113,214x figures already
corrected elsewhere in the repo.

## B. Thresholds with no discriminative power

Both compare the surrogate against `generate_mock_ground_truth` — a synthetic
helix, not an experimental structure — with bounds loose enough that random
coordinates also pass.

| test | threshold | actual | random noise | random passes? |
|---|---|---|---|---|
| `test_t4_1_human_insulin_monomer` | `rmsd < 50.0` | 22.16 | 25.21–25.63 | **yes** |
| `test_t4_2_hemoglobin_subunit_alpha` | `rmsd < 120.0` | 65.62 | 69.26 | **yes** |

Neither test can distinguish model output from `torch.randn`.

## C. Claim vs. behavior mismatches

| test | TEST_READY.md claim | what it asserts |
|---|---|---|
| `test_t1_f3_student_forward_efficiency` | "Check student forward-pass efficiency" (docstring: single vs. teacher's double pass) | `pred.shape` only. The teacher is never invoked; the comment "Profile forward passes count" profiles nothing. |
| `test_t1_f4_bond_length_error_correction` | "Verify consecutive residue bond length error correction" | Builds coords at bond length 5.0 vs ideal 3.8, then asserts `dists.shape == (4,)` and `dists > 0`. No movement toward 3.8 is checked. |
| `test_t2_f4_extreme_clashing_coordinates` | "Verify separation of extreme overlapping coordinates" | `refined.shape` and `any(refined != 0.0)`. Separation is not checked. |
| `test_t1_f2_memory_scaling` | "Verify activation memory footprint comparison vs baseline" | Parameter counts, not activations. |
| `test_t1_f4_clash_index_improvement` | "Verify clash index improvement" | Non-collapse only. |
| `test_t1_f4_refiner_loading_inference` | "Verify refiner weight loading and inference" | In-memory `state_dict` round-trip; no disk save/load. (Disk round-trip *is* covered by `test_refiner_checkpoint_load_roundtrip` in `test_boltz_modified_layers.py`.) |

**Root cause for most of C:** these tests instantiate randomly-initialised
modules (`ResNetCoordinateRefiner`, `CFGDistilledVectorField`). Asserting that an
untrained network improves bond lengths or reduces clashes would be asserting
luck, so the assertions were relaxed to shape checks while the names and
checklist entries kept promising the stronger property. Making them real requires
trained checkpoints, not tighter assertions.

## D. Also noted

- `test_t2_f4_residue_length_mismatch` uses `pytest.raises(Exception)`. Valid,
  but broad enough to pass on an unrelated error (e.g. a typo raising
  `AttributeError`). Narrowing to the expected type would strengthen it.
- `test_t2_f3_time_boundaries` claims to verify behavior at t=0.0 and t=1.0 but
  asserts only output shapes — no value or finiteness check.

## Recommendation

A1 and A2 are now fixed:

* A1 asserts finiteness, the valid 0–100 pLDDT range, and determinism for a fixed
  sequence, and skips rather than fabricating a score when the active predictor
  exposes no `predict_plddt`. Mutation-tested: an out-of-range return and a
  nondeterministic return are both caught.
* A2 measures activation retention with `torch.autograd.graph.saved_tensors_hooks`
  instead of arithmetic on literals, asserting >50% reduction at N=128 and N=256
  (measured 86.8% and 93.2%) and that the reduction *widens* with N, which is the
  O(N²)-vs-O(N) claim. Mutation-tested: substituting the full-rank updater for the
  low-rank one is caught.

Remaining, highest value first:

1. **B** — either compare against a real reference structure, or state plainly
   that these are smoke tests over a synthetic baseline. A threshold that random
   noise satisfies should not be described as "verifying RMSD".
2. **C** — leave the assertions as they are (they are honest about an untrained
   model) but correct the names and TEST_READY.md entries so they stop promising
   verification that is not happening. Alternatively, gate the strong assertions
   behind a trained checkpoint fixture and skip without it.
3. **D** — narrow `pytest.raises(Exception)`; add value checks to the time-boundary
   test.

Done: TEST_READY.md and TEST_INFRA.md no longer claim 49/49 and no longer
reference `/Users/akikjana/Documents/BiomolecularDesign/`.

Until B and C are addressed, TEST_READY.md should not be cited as evidence of
verification for the RMSD, clash, bond-length, or parameter-vs-activation memory
claims. The pLDDT and activation-reduction claims (A1, A2) are now genuinely
tested.
