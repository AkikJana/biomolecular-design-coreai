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
