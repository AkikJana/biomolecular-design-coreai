# E2-confirm - profiler-backed confirmation pass

- run_id: `e2_profile_confirm_20260630T191521Z`
- parent_experiment: E2 - MPS op-coverage / device-fallback (e2_mps_opcoverage.py)
- timestamp: 2026-06-30T19:15:22.733207+00:00
- device: **mps** | dtype: **fp16**
- real model ran end-to-end on MPS: **yes**
- weights: `/Users/akikjana/.boltz/boltz1_conf.ckpt`
  - weights_version: **Boltz-1 (boltz1_conf.ckpt)**
  - weights_sha256: `fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5`
- input: **SYNTHETIC feats (n_tokens=24, n_atoms=64, n_msa=8) - real model + real weights, synthetic input (identical methodology to E2)**
- code_sha: `7915a23c69d2673bef5e05c477d41dcac0e70340` | boltz_commit: `7915a23c69d2673bef5e05c477d41dcac0e70340`
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

`torch.profiler`'s `device_type` field reports `DeviceType.CPU` for *every* op on this MPS build (there is no `ProfilerActivity.MPS` in torch 2.9.1), and self-CPU time is dominated by MPS command-buffer dispatch/sync overhead rather than actual execution device. Empirical evidence from this run: the following ops - all classified `mps` by the device-tracking tracer, i.e. **not** fallbacks - show a *higher* average self-CPU time per call than `aten::linalg_svd` itself: `['aten::_cdist_forward', 'aten::_unique2', 'aten::constant_pad_nd', 'aten::eq', 'aten::gt', 'aten::index_select', 'aten::le', 'aten::lt', 'aten::max', 'aten::ne', 'aten::norm', 'aten::prod', 'aten::repeat', 'aten::scatter_']`. A timing-threshold classifier would therefore misclassify ops in both directions; fallback identity is decided solely by PyTorch's own fallback-warning ground truth, exactly as in E2.

### Legitimate scalar syncs vs. true op fallback (not conflated)

| op | calls | self_cpu_us_total | self_cpu_us_avg | classification |
|---|---:|---:|---:|---|
| `aten::item` | 469 | 931.6 | 1.99 | legitimate host scalar sync (NOT a fallback) |
| `aten::_local_scalar_dense` | 469 | 21404.2 | 45.64 | legitimate host scalar sync (NOT a fallback) |
| `aten::linalg_svd` | 2 | 2462.7 | 1231.34 | **true unsupported-op CPU fallback** (confirmed by PyTorch fallback warning) |
| `aten::_linalg_svd` | 2 | 6477.9 | 3238.97 | **true unsupported-op CPU fallback** (confirmed by PyTorch fallback warning) |

### Top ops by total self-CPU time (informational only)

| op | calls | self_cpu_us_total | self_cpu_us_avg |
|---|---:|---:|---:|
| `aten::mul` | 3041 | 2338637.7 | 769.04 |
| `aten::copy_` | 14548 | 1903281.7 | 130.83 |
| `aten::slice` | 2243 | 1257131.3 | 560.47 |
| `aten::add` | 1458 | 1131193.6 | 775.85 |
| `aten::transpose` | 1159 | 873329.8 | 753.52 |
| `aten::div` | 625 | 490796.6 | 785.27 |
| `aten::ge` | 424 | 379031.0 | 893.94 |
| `aten::add_` | 420 | 315334.4 | 750.8 |
| `aten::linear` | 8810 | 300862.9 | 34.15 |
| `aten::sub` | 416 | 253477.6 | 609.32 |
| `aten::bmm` | 1046 | 219032.7 | 209.4 |
| `aten::split` | 278 | 216960.9 | 780.43 |
| `aten::_to_copy` | 12627 | 193289.0 | 15.31 |
| `aten::div_` | 208 | 157532.1 | 757.37 |
| `aten::rsub` | 178 | 132699.5 | 745.5 |

## Timing: one-time warmup vs. steady state

Warmup = first forward pass after model load (pays one-time MPS kernel-compile cost; also the profiler-instrumented pass above, so its absolute numbers run slightly high vs. a bare pass and are reported separately, never mixed into the steady-state stats below). `--warmup-iters=1` -> 1 profiled pass (timed below) + 0 additional plain warmup pass(es) (discarded entirely, not shown in any table).

| stage | warmup seconds (1 pass, profiler-instrumented) |
|---|---:|
| input_featurization | 0.8176 |
| trunk_msa_module | 0.7310 |
| trunk_pairformer | 3.6658 |
| distogram | 0.0079 |
| diffusion_sampler | 2.0597 |
| confidence_head | 4.5912 |
| **total** | **11.8732** |

Steady state: mean +/- std over 4 timed iterations (warmup discarded), `torch.mps.synchronize()` around every stage.

| stage | mean (s) | std (s) | n |
|---|---:|---:|---:|
| input_featurization | 0.1061 | 0.0399 | 4 |
| trunk_msa_module | 0.1851 | 0.0380 | 4 |
| trunk_pairformer | 0.4287 | 0.0088 | 4 |
| distogram | 0.0060 | 0.0019 | 4 |
| diffusion_sampler | 0.6319 | 0.1330 | 4 |
| confidence_head | 0.7303 | 0.0298 | 4 |
| **total (sum of stage means)** | **2.0881** | | |

## Relationship to E2

This run reuses E2's model loading, synthetic-feats construction, staged-forward driver and op classification verbatim (`experiments/e2_mps_opcoverage.py`, no changes made to that file) and adds (1) a torch.profiler-based independent cross-check of the device-fallback claim and (2) warmup/steady-state separated timing. It does not change E2's conclusions; see the verdict above and the headline finding in `experiments/README.md`.
