# experiments/

Reproducible experiment scripts for the Boltz thesis project. Each script is
self-contained, parametrizable (with a fast path for reviewers), and writes artifacts
plus a manifest-style sidecar into `results/{real,synthetic}/`.

## Layout / conventions

```
experiments/                 # experiment scripts (this dir)
results/
  real/                      # artifacts from runs against REAL model weights
  synthetic/                 # artifacts from synthetic-only / dry runs
```

Every run writes two files into its results dir, keyed by a timestamped `run_id`:

- `<run_id>.report.md`      — human-readable report (tables, wall-clock, caveats)
- `<run_id>.manifest.json`  — machine-readable sidecar / manifest

### Manifest sidecar schema

The `*.manifest.json` records exactly what produced the artifact so a run is
reproducible and auditable:

| field | meaning |
|---|---|
| `run_id`, `experiment`, `timestamp_utc` | run identity |
| `code_sha`, `boltz_commit` | repo commit the script + boltz source were at |
| `weights_file`, `weights_version`, `weights_sha256` | exact checkpoint used |
| `seed`, `device`, `dtype`, `torch_version`, `mps_available` | numerics / backend |
| `hardware`, `os`, `python` | machine |
| `input_kind` | real vs synthetic input, with shapes |
| `command`, `config` | how it was invoked |
| `ran_on_device` | `yes` / `partial` / `no` |
| `wall_clock_by_stage_s`, `fallback_ops`, `op_coverage`, `notes` | results |

---

## E2 — MPS op-coverage / device-fallback (`e2_mps_opcoverage.py`)

Answers the project's biggest open question: **does the real Boltz model run
end-to-end on Apple Silicon via the PyTorch `mps` backend, and which ATen ops fall
back to the CPU?**

It loads a real Boltz checkpoint on `device=mps`, drives one forward pass *stage by
stage* (input featurization → trunk recycles {MSA + Pairformer, incl. triangle
multiplication / triangle attention} → distogram → diffusion sampler → confidence
head) by calling the public sub-modules directly — **without modifying any model
source** — and records, per stage:

- a complete ATen-op inventory + the devices each op's tensors touched
  (via a `TorchDispatchMode` tracer), and
- the authoritative set of CPU-fallback ops harvested from PyTorch's own
  *"... not currently supported on the MPS backend and will fall back to run on the
  CPU"* warnings,

then emits an op/stage × {mps | cpu_fallback | cpu_native | unsupported} table plus a
coarse wall-clock-by-stage breakdown.

### Headline finding

On torch 2.9.1 / macOS arm64, the Boltz-1 model **runs end-to-end on MPS**, and the
**only unsupported-MPS fallback _observed_** was **`aten::linalg_svd`**, inside
`weighted_rigid_align` (`boltz.model.loss.diffusion`), which is called on **every**
reverse-diffusion step of the default sampler via `alignment_reverse_diff` — i.e. on
every inference, not just training/steered sampling. Every other op traced (linear,
layer-norm, softmax, bmm, einsum→bmm in triangle mult/attention, sigmoid, silu, cdist,
scatter/gather, randn, …) ran on MPS.

> **Ground truth for fallbacks is PyTorch's own _unsupported-MPS_ fallback warnings.**
> So this is a statement about *op support*, not data movement: silent host↔device
> scalar syncs (e.g. `.item()` / `aten::_local_scalar_dense`) emit no such warning and
> are classified **mps**, so they never show up as fallbacks. The claim is therefore
> "the only unsupported-MPS fallback observed was `aten::linalg_svd`", **not**
> "nothing else touches the CPU".

> `direct_mps_kernel: NO` in the table is **informational** and does *not* mean
> fallback — structural/factory ops (`view`/`expand`/`clone`/`_to_copy`/`arange`/
> `randn`) run on MPS via composite or fall-through kernels. Fallback verdicts come
> *only* from the fallback-warning set.

### Caveats (also flagged in every artifact)

- **WEIGHTS:** the only local checkpoint is `~/.boltz/boltz1_conf.ckpt` =
  **Boltz-1**, not Boltz-2. The op-coverage map is architecture-driven so this is
  still informative, but profiling *real Boltz-2* needs Boltz-2 weights. Re-run with
  `--checkpoint /path/to/boltz2_conf.ckpt`.
- **INPUT:** a **synthetic** feature dict (correct keys/shapes/dtypes) is used because
  op-coverage is architecture-driven and building real inputs needs MSA generation +
  CCD/mol assets. Numeric content is not meaningful — notably the SVD alignment can
  fail to *converge* on CPU for degenerate random coords (seen in the default-config
  run, recorded as `ran_on_device: partial`). That is a synthetic-data numerical
  artifact, **not** an MPS limitation. Swap in a real `BoltzInferenceDataModule`
  `feats` dict for numeric fidelity — no other change needed.

### Reproduce

`boltz` is a src-layout package, so put `boltz/src` on `PYTHONPATH`. The model needs a
few deps not in the base env (`pytorch-lightning`, `torchmetrics`, `einops`, `einx`,
`fairscale`, `dm-tree`, `omegaconf`); install them into a venv that inherits the
MPS-capable torch, e.g.:

```bash
python -m venv --system-site-packages .e2venv
.e2venv/bin/pip install "pytorch-lightning>=2.5.0" torchmetrics einops einx fairscale dm-tree omegaconf

# fast path (tiny, ~10s, completes end-to-end):
PYTHONPATH=boltz/src .e2venv/bin/python experiments/e2_mps_opcoverage.py --fast

# default smoke (larger; SVD may not converge on synthetic coords):
PYTHONPATH=boltz/src .e2venv/bin/python experiments/e2_mps_opcoverage.py

# no/locked weights → emit methodology + static analysis only (never silently fails):
PYTHONPATH=boltz/src .e2venv/bin/python experiments/e2_mps_opcoverage.py --static-only
```

Key flags: `--device {mps,cpu,cuda}`, `--dtype {fp16,fp32,bf16}`, `--checkpoint`,
`--n-tokens/--n-atoms/--n-msa`, `--recycling-steps`, `--sampling-steps`, `--fast`,
`--static-only`, `--out-dir`.

### Committed artifacts

- `results/real/*153317Z*` — primary: `--fast`, `ran_on_device: yes`, all 6 stages.
- `results/real/*153417Z*` — default config, `ran_on_device: partial` (SVD CPU
  fallback failed to converge on synthetic coords; full op table + partial timings).

---

## E2-confirm — profiler-backed confirmation pass (`e2_profile_confirm.py`)

E2's headline finding (op-coverage / fallback table above) rests on a single smoke
run whose wall-clock total mixes in one-time MPS kernel-compile / warmup cost, and on
two signals for the fallback claim (a `TorchDispatchMode` device tracer + PyTorch's
own fallback `UserWarning`s). This script does **not** redo the E2 smoke run — it
imports and reuses E2's model-loading, synthetic-feats, staged-forward-driver and
op-classification code verbatim (`import e2_mps_opcoverage as e2`; `e2_mps_opcoverage.py`
itself is unmodified) and adds two things on top, in the same process / same
execution, so everything below describes literally the same forward passes as E2:

1. **An independent `torch.profiler` cross-check.** `torch.profiler` is wrapped
   around the same staged forward pass as E2's tracer, and the two op inventories
   are diffed: the profiler's (RecordFunction-based, sees composite/Python-API
   entry points before they decompose) is a strict superset of the tracer's
   (post-decomposition ATen primitives) — confirmed empirically: 0 ops were seen by
   the tracer and missed by the profiler. The profiler-only ops are exactly the
   SVD/rigid-alignment linear-algebra call graph (`_linalg_svd`,
   `_linalg_check_errors`, `linalg_lu_factor_ex`, `det`, `linalg_det`,
   `frobenius_norm`, `_to_cpu`, ...) plus generic composite wrappers
   (`matmul`, `cdist`, `einsum`, `softmax`, `layer_norm`, ...) that decompose to
   MPS-supported primitives elsewhere — no new fallback surface.

   Self-CPU timing is reported for transparency but **deliberately not used to
   classify device residency** — fallback identity still comes only from PyTorch's
   own fallback-warning ground truth, exactly as in E2. This is a documented,
   empirically-justified methodology choice, not an oversight: `torch.profiler` has
   no `ProfilerActivity.MPS` on this torch build, so `device_type` reports
   `DeviceType.CPU` for *every* op regardless of actual execution device, and
   several confirmed-MPS ops (`aten::cat`, `aten::eq`, `aten::ne`, ...) show *higher*
   average self-CPU time per call than `aten::linalg_svd` itself (dispatch/sync
   overhead noise dominates the signal). A timing-threshold classifier would
   misclassify in both directions.

   Legitimate host scalar syncs (`aten::item`, `aten::_local_scalar_dense`) are
   reported with their own self-CPU stats and excluded from the fallback
   classification *by identity*, never by timing, so they are never conflated with
   the true `aten::linalg_svd` fallback.

   Recurrence across iterations needed a third approach: PyTorch's fallback warning
   turned out to be a **process-global warn-once** (fires only on an op's very
   first invocation in the process, confirmed empirically — re-harvesting it on
   later iterations always returns empty) and `TorchDispatchMode`'s device tracking
   is transparent to the fallback (`linalg_svd`'s tensors are still tagged `mps` at
   the dispatch boundary it intercepts; the host round-trip happens *inside* the
   kernel via internal sub-calls that never reach the tracer). Recurrence is instead
   confirmed via the profiler re-observing the `aten::_to_cpu` marker call on
   dedicated, separately-run passes every iteration.

2. **Warmup vs. steady-state timing.** One untimed warmup pass (the same pass used
   for the profiler cross-check, since it already pays the one-time MPS
   kernel-compile cost) is discarded from the timing figures and reported
   separately. `--timed-iters` (default 4) plain, uninstrumented forward passes are
   then run back-to-back, each wrapped in `torch.mps.synchronize()` before/after
   every stage, and the per-stage **mean ± sample std** is reported as the
   steady-state figure.

### Headline finding

**CONFIRMED** — re-deriving E2's headline claim with an independent instrumentation
method does not change it: the sole unsupported-MPS-op CPU fallback is
`aten::linalg_svd`, recurring on every reverse-diffusion step, every iteration.

Steady state vs. one-time warmup (default config, n_tokens=24/n_atoms=64/n_msa=8,
0 recycles, 2 sampling steps — same sizes as E2's `--fast`, chosen here for
run-to-run numerical stability across repeated iterations since the default-sized
config is known per E2 to occasionally hit synthetic-coordinate SVD
non-convergence):

| stage | warmup (1 pass, profiler-instrumented) | steady-state mean ± std (4 iters) |
|---|---:|---:|
| input_featurization | 0.8176 s | 0.1061 s ± 0.0399 s |
| trunk_msa_module | 0.7310 s | 0.1851 s ± 0.0380 s |
| trunk_pairformer | 3.6658 s | 0.4287 s ± 0.0088 s |
| distogram | 0.0079 s | 0.0060 s ± 0.0019 s |
| diffusion_sampler | 2.0597 s | 0.6319 s ± 0.1330 s |
| confidence_head | 4.5912 s | 0.7303 s ± 0.0298 s |
| **total** | **11.87 s** | **2.09 s** |

(Numbers from the committed run `results/real/e2_profile_confirm_20260630T191521Z.*`;
expect run-to-run variance on shared/thermally-throttled hardware — see `samples_s`
in the manifest for the raw per-iteration values behind each std.) The one-time
warmup cost is **~5.7×** the steady-state total — confirming E2's own caveat that
its single-smoke-run total was not steady-state. The std on the lighter early stages
is large relative to their mean (first-iteration-of-the-loop noise on top of an
already-small ~0.01-0.2s signal); the heavier, more compute-bound stages
(`trunk_pairformer`, `confidence_head`) are comparatively tighter in relative terms.

### Reproduce

Same env as E2 (`PYTHONPATH=boltz/src` + the `.e2venv` venv above):

```bash
PYTHONPATH=boltz/src .e2venv/bin/python experiments/e2_profile_confirm.py
PYTHONPATH=boltz/src .e2venv/bin/python experiments/e2_profile_confirm.py --timed-iters 5
PYTHONPATH=boltz/src .e2venv/bin/python experiments/e2_profile_confirm.py --static-only
```

Key flags: same as E2 (`--device`, `--dtype`, `--checkpoint`, `--n-tokens/--n-atoms/
--n-msa`, `--recycling-steps`, `--sampling-steps`, `--static-only`, `--out-dir`) plus
`--warmup-iters` (default 1, minimum 1 - the profiler cross-check pass is mandatory
and always counts as the first warmup pass; values above 1 run additional plain,
fully-discarded warmup passes afterwards) and `--timed-iters` (default 4; errors
below 1, warns but proceeds outside the recommended 3-5 range).

### Committed artifacts

- `results/real/e2_profile_confirm_20260630T191521Z*` — default config,
  `ran_on_device: yes`, profiler cross-check verdict `CONFIRMED`, `linalg_svd`
  residency re-confirmed on all 3 dedicated check passes, full warmup +
  steady-state timing table. `code_sha`/`boltz_commit`: `7915a23c69d2673bef5e05c477d41dcac0e70340`
  (the commit that added this script, verified to actually contain
  `experiments/e2_profile_confirm.py` - provenance must be traceable to the exact
  code that produced the artifact, not merely to "some recent commit").
