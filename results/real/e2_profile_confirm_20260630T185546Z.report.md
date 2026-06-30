# E2-confirm - profiler-backed confirmation pass

- run_id: `e2_profile_confirm_20260630T185546Z`
- parent_experiment: E2 - MPS op-coverage / device-fallback (e2_mps_opcoverage.py)
- timestamp: 2026-06-30T18:55:47.578936+00:00
- device: **mps** | dtype: **fp16**
- real model ran end-to-end on MPS: **yes**
- weights: `/Users/akikjana/.boltz/boltz1_conf.ckpt`
  - weights_version: **Boltz-1 (boltz1_conf.ckpt)**
  - weights_sha256: `fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5`
- input: **SYNTHETIC feats (n_tokens=24, n_atoms=64, n_msa=8) - real model + real weights, synthetic input (identical methodology to E2)**
- code_sha: `89ead998774ee693758a085617e7eb2a15108aa7` | boltz_commit: `89ead998774ee693758a085617e7eb2a15108aa7`
- opm_mode (BOLTZMAC_OPM): **stock**
- hardware: arm | os: Darwin 25.6.0 (macOS-26.6-arm64-arm-64bit-Mach-O)
- torch: 2.9.1
- seed: 0
- warmup_iterations: 1
- timed_iterations: 4
- command: `python e2_profile_confirm.py --out-dir results/real`

## Notes & caveats

- WEIGHTS CAVEAT (same as E2): the loaded checkpoint is **Boltz-1**, not Boltz-2. Profiling real Boltz-2 needs Boltz-2 weights.
- INPUT CAVEAT (same as E2): synthetic feats are used because op-coverage and timing-shape are architecture driven; default sizes here match E2's --fast config for run-to-run numerical stability across the timed iterations (the default-sized config is known per E2 to occasionally hit synthetic-coordinate SVD non-convergence on CPU).
- torch.profiler has no ProfilerActivity.MPS in this torch build (2.9.1); device_type reports DeviceType.CPU for every op regardless of actual execution device, and self-CPU time is dominated by MPS dispatch/sync overhead. The profiler is therefore used here for op-presence corroboration and explicit scalar-sync separation, NOT for timing-based device classification - see 'Why self-CPU time is not used to classify fallbacks' in the report.
- PyTorch's MPS-unsupported fallback warning is a process-global warn-once (empirically verified: it fires on an op's first invocation in the process and never again, regardless of Python warnings-filter state) - re-harvesting it per iteration cannot confirm the fallback recurs on every iteration. TorchDispatchMode's own device tracking is also no use here: the fallback is transparent at the dispatch boundary it intercepts (aten::linalg_svd's inputs/outputs are still tagged mps there; the host round-trip happens inside the kernel via internal sub-calls like aten::_to_cpu that never reach the tracer). Recurrence was instead confirmed via torch.profiler re-observing aten::_to_cpu on 3 dedicated passes: confirmed_every_iteration=True.
- profiler cross-check verdict: CONFIRMED: sole unsupported-MPS fallback is aten::linalg_svd.

## Profiler cross-check verdict

> **CONFIRMED: sole unsupported-MPS fallback is aten::linalg_svd.**

- fallback ops re-harvested (this run's warmup pass): `['aten::linalg_svd']`
- PyTorch's fallback warning is a **process-global warn-once**: it fired on the warmup pass above and was empirically confirmed to NOT re-fire on any of the later steady-state iterations (re-harvested warnings per timed iteration: `[[], [], [], []]` - all empty, as expected, not evidence of absence). TorchDispatchMode's own device tracking is also no use here: the fallback is transparent at the dispatch boundary it intercepts, so `aten::linalg_svd`'s inputs/outputs are still tagged `mps` there (the host round-trip happens *inside* the kernel, via internal sub-calls like `aten::_to_cpu` that never reach the tracer). Recurrence was instead confirmed via **`torch.profiler` re-observing `aten::_to_cpu`** - an unambiguous host-transfer marker - on 3 separate lightweight passes: residency_confirmed_every_iteration = **True**
- TorchDispatchMode tracer op inventory: 73 distinct ops | torch.profiler op inventory: 126 distinct ops
- ops seen by the dispatch-mode tracer but **missed** by the profiler (should be empty - a non-empty list would mean the tracer over-reports vs. an independent instrumentation point): `[] (none)`
- ops seen by the profiler but not by the tracer (expected: the composite/Python-API SVD-and-rigid-alignment call graph plus generic composite wrappers that decompose to MPS-supported primitives elsewhere): `['aten::__and__', 'aten::_index_put_impl_', 'aten::_linalg_check_errors', 'aten::_linalg_svd', 'aten::_reshape_alias', 'aten::_to_cpu', 'aten::as_strided', 'aten::as_strided_', 'aten::cdist', 'aten::chunk', 'aten::clip', 'aten::contiguous', 'aten::det', 'aten::detach_', 'aten::diagonal', 'aten::einsum', 'aten::empty', 'aten::empty_like', 'aten::empty_strided', 'aten::expand_as', 'aten::fill_', 'aten::flatten', 'aten::fmod_', 'aten::frobenius_norm', 'aten::index_select', 'aten::is_nonzero', 'aten::is_same_size', 'aten::item', 'aten::layer_norm', 'aten::linalg_det', 'aten::linalg_lu_factor_ex', 'aten::mH', 'aten::mT', 'aten::matmul', 'aten::mul_', 'aten::narrow', 'aten::new_empty', 'aten::normal_', 'aten::one_hot', 'aten::pad', 'aten::prod', 'aten::repeat_interleave', 'aten::reshape', 'aten::resize_', 'aten::resolve_conj', 'aten::resolve_neg', 'aten::result_type', 'aten::softmax', 'aten::to', 'aten::transpose_', 'aten::uniform_', 'aten::view_as', 'aten::zero_']`

### Why self-CPU time is not used to classify fallbacks

`torch.profiler`'s `device_type` field reports `DeviceType.CPU` for *every* op on this MPS build (there is no `ProfilerActivity.MPS` in torch 2.9.1), and self-CPU time is dominated by MPS command-buffer dispatch/sync overhead rather than actual execution device. Empirical evidence from this run: the following ops - all classified `mps` by the device-tracking tracer, i.e. **not** fallbacks - show a *higher* average self-CPU time per call than `aten::linalg_svd` itself: `['aten::_cdist_forward', 'aten::_unique2', 'aten::cat', 'aten::clamp', 'aten::constant_pad_nd', 'aten::eq', 'aten::gt', 'aten::index_select', 'aten::le', 'aten::lt', 'aten::max', 'aten::ne', 'aten::norm', 'aten::scatter_']`. A timing-threshold classifier would therefore misclassify ops in both directions; fallback identity is decided solely by PyTorch's own fallback-warning ground truth, exactly as in E2.

### Legitimate scalar syncs vs. true op fallback (not conflated)

| op | calls | self_cpu_us_total | self_cpu_us_avg | classification |
|---|---:|---:|---:|---|
| `aten::item` | 469 | 1218.1 | 2.6 | legitimate host scalar sync (NOT a fallback) |
| `aten::_local_scalar_dense` | 469 | 10836.5 | 23.11 | legitimate host scalar sync (NOT a fallback) |
| `aten::linalg_svd` | 2 | 2500.6 | 1250.3 | **true unsupported-op CPU fallback** (confirmed by PyTorch fallback warning) |
| `aten::_linalg_svd` | 2 | 7370.0 | 3685.01 | **true unsupported-op CPU fallback** (confirmed by PyTorch fallback warning) |

### Top ops by total self-CPU time (informational only)

| op | calls | self_cpu_us_total | self_cpu_us_avg |
|---|---:|---:|---:|
| `aten::mul` | 3041 | 3280818.2 | 1078.86 |
| `aten::copy_` | 14548 | 2429432.3 | 166.99 |
| `aten::slice` | 2243 | 1751014.3 | 780.66 |
| `aten::add` | 1458 | 1653116.3 | 1133.82 |
| `aten::transpose` | 1159 | 1232934.9 | 1063.79 |
| `aten::div` | 625 | 714192.7 | 1142.71 |
| `aten::ge` | 424 | 530769.2 | 1251.81 |
| `aten::add_` | 420 | 457265.0 | 1088.73 |
| `aten::linear` | 8810 | 434380.9 | 49.31 |
| `aten::sub` | 416 | 338324.4 | 813.28 |
| `aten::rsub` | 178 | 304358.4 | 1709.88 |
| `aten::bmm` | 1046 | 301324.3 | 288.07 |
| `aten::split` | 278 | 296948.8 | 1068.16 |
| `aten::_to_copy` | 12627 | 290041.8 | 22.97 |
| `aten::div_` | 208 | 222405.6 | 1069.26 |

## Timing: one-time warmup vs. steady state

Warmup = first forward pass after model load (pays one-time MPS kernel-compile cost; also the profiler-instrumented pass above, so its absolute numbers run slightly high vs. a bare pass and are reported separately, never mixed into the steady-state stats below).

| stage | warmup seconds (1 pass, profiler-instrumented) |
|---|---:|
| input_featurization | 1.1365 |
| trunk_msa_module | 1.0142 |
| trunk_pairformer | 4.5624 |
| distogram | 0.0103 |
| diffusion_sampler | 2.5824 |
| confidence_head | 7.1668 |
| **total** | **16.4725** |

Steady state: mean +/- std over 4 timed iterations (warmup discarded), `torch.mps.synchronize()` around every stage.

| stage | mean (s) | std (s) | n |
|---|---:|---:|---:|
| input_featurization | 0.3476 | 0.3258 | 4 |
| trunk_msa_module | 0.3646 | 0.1911 | 4 |
| trunk_pairformer | 0.8511 | 0.2337 | 4 |
| distogram | 0.0107 | 0.0025 | 4 |
| diffusion_sampler | 1.0924 | 0.3284 | 4 |
| confidence_head | 1.3689 | 0.1861 | 4 |
| **total (sum of stage means)** | **4.0352** | | |

## Relationship to E2

This run reuses E2's model loading, synthetic-feats construction, staged-forward driver and op classification verbatim (`experiments/e2_mps_opcoverage.py`, no changes made to that file) and adds (1) a torch.profiler-based independent cross-check of the device-fallback claim and (2) warmup/steady-state separated timing. It does not change E2's conclusions; see the verdict above and the headline finding in `experiments/README.md`.
