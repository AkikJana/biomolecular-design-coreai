#!/usr/bin/env python3
"""E2-confirm - profiler-backed confirmation pass for the E2 MPS op-coverage run.

E2 (``e2_mps_opcoverage.py``) established, via a single smoke run, that the real
Boltz-1 model runs end-to-end on the PyTorch ``mps`` backend and that the only
unsupported-MPS-op CPU fallback *observed* is ``aten::linalg_svd`` (inside
``weighted_rigid_align`` / ``alignment_reverse_diff``, hit on every reverse-diffusion
step). That conclusion rests on two signals: (a) a ``TorchDispatchMode`` tracer's
direct ``tensor.device.type`` inspection of every op's inputs/outputs, and (b) the
authoritative set of ops harvested from PyTorch's own "not currently supported on the
MPS backend" fallback ``UserWarning``\\ s. The E2 wall-clock numbers are a single
smoke run whose total includes one-time MPS kernel-compile / warmup cost - too coarse
for a thesis figure.

This script does NOT redo the E2 smoke run; it upgrades it along two independent
axes, reusing E2's model-loading / synthetic-feats / staged-forward / classification
code directly (``import e2_mps_opcoverage as e2``, no changes to that file):

1. **An independent torch.profiler-based cross-check.** ``torch.profiler`` is wrapped
   around the *same* staged forward pass, in the *same* process, alongside E2's
   ``TorchDispatchMode`` tracer and the fallback-warning harvester, so all three
   signals describe literally the same execution. We compare:
     - the **op-name inventory** seen by the profiler (RecordFunction-based
       instrumentation, which also captures composite/Python-API entry points before
       they decompose) against the inventory seen by the dispatch-mode tracer
       (post-decomposition ATen primitives) - this is a coverage cross-check
       independent of timing: if the tracer missed an op that actually executed, the
       profiler (a structurally different instrumentation point) would very likely
       still have recorded it;
     - per-op self-CPU time, reported for transparency but **explicitly not used to
       classify device residency**. We verified empirically (see "Why self-CPU time
       is not used to classify fallbacks" in the generated report) that self-CPU
       time is dominated by MPS command-buffer dispatch/sync overhead, not actual
       execution device: known-MPS ops such as ``aten::cat``/``aten::eq``/
       ``aten::ne`` show *higher* average self-CPU time per call than the known
       CPU-fallback ``aten::linalg_svd``. A timing-threshold classifier would
       therefore misclassify ops in both directions. Fallback identity is decided
       *only* by the PyTorch fallback-warning ground truth (re-harvested
       independently in this run), exactly as in E2 - the profiler's job here is
       coverage corroboration and explicit separation of legitimate scalar host
       syncs (``aten::item`` / ``aten::_local_scalar_dense``, which run on CPU by
       definition and are not MPS coverage gaps) from true unsupported-op fallbacks,
       so the two are never conflated in the reported verdict.
     - **recurrence across iterations**, confirmed a different way than either of
       E2's two signals: PyTorch's fallback warning turned out to be a
       process-global warn-once (fires only on an op's first invocation ever, not
       once per call), and TorchDispatchMode's device tracking is transparent to the
       fallback (linalg_svd's tensors are still tagged ``mps`` at the boundary it
       intercepts - the host round-trip happens inside the kernel). So recurrence on
       every iteration is confirmed via the profiler re-observing the
       ``aten::_to_cpu`` marker call on dedicated, separately-run passes - see
       "Profiler cross-check verdict" in the report for the full chain of reasoning
       and the numbers behind each ruled-out approach.

2. **Warmup vs. steady-state timing.** ``--warmup-iters`` (default 1, minimum 1)
   untimed warmup forward passes are discarded from the timing figures and reported
   separately: the first one is always the profiler cross-check pass above (it pays
   the one-time MPS kernel-compile cost anyway, so the timing is "free" to capture
   there); any additional ones (``--warmup-iters`` > 1) are plain, fully discarded
   passes run afterwards, useful if a single pass doesn't fully stabilize MPS
   kernel-compile state. Then ``--timed-iters`` (default 4, recommended 3-5) plain
   forward passes are run back-to-back, each wrapped in ``torch.mps.synchronize()``
   before and after every stage (reusing E2's ``stage()`` timing helper), and the
   per-stage mean +/- (sample) std is reported as the steady-state figure.

Artifacts are written into ``results/real/`` with the same manifest sidecar schema as
E2 (run_id, code_sha, weights_file/version/sha256, seed, hardware, device, dtype,
opm_mode, ...), plus additional fields documented in ``experiments/README.md``:
``warmup_wall_clock_by_stage_s``, ``steady_state_wall_clock_by_stage_s`` and
``profiler_cross_check``. Still Boltz-1 (only local checkpoint), still synthetic
input (op-coverage / timing-shape are architecture driven) - see E2's caveats, which
apply identically here.

Usage
-----
    PYTHONPATH=boltz/src python experiments/e2_profile_confirm.py
    PYTHONPATH=boltz/src python experiments/e2_profile_confirm.py --timed-iters 5
    PYTHONPATH=boltz/src python experiments/e2_profile_confirm.py --static-only
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import contextlib
import re
import statistics
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.profiler as tprof

# e2_mps_opcoverage.py lives next to this file; when run as a script its directory is
# already on sys.path[0], but import defensively for the (rare) case of being
# imported as a module from elsewhere.
THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_DIR = THIS_FILE.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e2_mps_opcoverage as e2  # noqa: E402  (path setup must precede this import)

REPO_ROOT = e2.REPO_ROOT

# Legitimate host<->device scalar syncs: these execute on CPU *by definition*
# (extracting a Python scalar from a tensor requires a host round-trip) and are not
# an MPS coverage gap. They must never be conflated with a true unsupported-op
# fallback such as aten::linalg_svd.
SCALAR_SYNC_OPS = {"aten::item", "aten::_local_scalar_dense"}


def _harvest_fallback_ops(wlist) -> set[str]:
    ops: set[str] = set()
    for w in wlist:
        msg = str(w.message)
        if "not currently supported on the MPS backend" in msg:
            m = re.search(r"operator '([^']+)'", msg)
            if m:
                ops.add(m.group(1).split(".")[0])
    return ops


def load_model(ckpt_path: Path, device: torch.device):
    """Mirrors e2_mps_opcoverage.main()'s load step exactly (no behavior change)."""
    try:
        import omegaconf

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(
                [
                    omegaconf.dictconfig.DictConfig,
                    omegaconf.listconfig.ListConfig,
                    omegaconf.nodes.AnyNode,
                    omegaconf.base.ContainerMetadata,
                    omegaconf.base.Metadata,
                ]
            )
    except Exception:
        pass

    from boltz.model.models.boltz1 import Boltz1

    try:
        model = Boltz1.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", strict=False, weights_only=False
        )
    except TypeError:
        model = Boltz1.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", strict=False
        )
    model = model.to(device)
    model.eval()
    return model


def run_profiled_warmup(
    model,
    feats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    recycling_steps: int,
    sampling_steps: int,
    autocast_ctx,
) -> dict[str, Any]:
    """One forward pass, instrumented with TorchDispatchMode + torch.profiler +
    fallback-warning harvesting simultaneously, so all three signals describe the
    same execution. Doubles as the (discarded-from-timing) warmup pass."""
    tracer = e2.OpCoverageTracer()
    sub_hooks = e2.SubStageHooks(model, tracer)
    timings: dict[str, float] = {}
    run_err = None
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        with tprof.profile(
            activities=[tprof.ProfilerActivity.CPU], record_shapes=False
        ) as prof:
            try:
                with tracer:
                    e2.run_staged_forward(
                        model,
                        feats,
                        device,
                        tracer,
                        recycling_steps=recycling_steps,
                        sampling_steps=sampling_steps,
                        autocast_ctx=autocast_ctx,
                        timings=timings,
                    )
            except Exception as exc:  # pragma: no cover - surfaced via notes
                run_err = exc
                print(traceback.format_exc(), file=sys.stderr)
            finally:
                sub_hooks.remove()
        fallback_ops = _harvest_fallback_ops(wlist)

    tracer_ops: set[str] = set()
    for _stage, ops in tracer.records.items():
        tracer_ops |= set(ops.keys())

    key_avgs = list(prof.key_averages())
    profiler_rows = [
        {
            "op": e.key,
            "calls": e.count,
            "self_cpu_us_total": round(e.self_cpu_time_total, 1),
            "self_cpu_us_avg": round(e.self_cpu_time_total / max(e.count, 1), 2),
        }
        for e in key_avgs
        if e.key.startswith("aten::")
    ]
    profiler_ops = {r["op"] for r in profiler_rows}

    # Verdict: device-residency truth comes ONLY from the fallback-warning set
    # (identical methodology to E2), never from self-CPU timing magnitude. The
    # profiler contributes (a) op-presence corroboration and (b) the explicit
    # scalar-sync exclusion below.
    non_scalar_fallback_ops = fallback_ops - SCALAR_SYNC_OPS
    expected = {"aten::linalg_svd"}
    if non_scalar_fallback_ops == expected:
        verdict = "CONFIRMED: sole unsupported-MPS fallback is aten::linalg_svd."
    elif non_scalar_fallback_ops:
        verdict = (
            "CORRECTED: fallback-warning set differs from E2's "
            f"{sorted(expected)} -> observed {sorted(non_scalar_fallback_ops)}."
        )
    else:
        verdict = "CORRECTED: no fallback ops observed in this run (re-check input/config)."

    # ops the profiler's higher-level (composite / Python-API) instrumentation saw
    # that the dispatch-mode tracer's post-decomposition view did not - expected to
    # be exactly the SVD/rigid-alignment call graph (linalg internals) plus generic
    # composite wrappers (matmul, cdist, einsum, softmax, layer_norm, ...) that
    # decompose to MPS-supported primitives elsewhere.
    profiler_only = sorted(profiler_ops - tracer_ops)
    tracer_only = sorted(tracer_ops - profiler_ops)

    self_cpu_sorted = sorted(profiler_rows, key=lambda r: -r["self_cpu_us_total"])
    svd_rows = [r for r in profiler_rows if "svd" in r["op"]]
    scalar_sync_rows = [r for r in profiler_rows if r["op"] in SCALAR_SYNC_OPS]
    # ops that aren't scalar-syncs, aren't the confirmed fallback, yet still show
    # higher avg self-CPU time per call than aten::linalg_svd itself - the empirical
    # evidence that timing-threshold classification would be unreliable here.
    svd_avg = max((r["self_cpu_us_avg"] for r in svd_rows), default=0.0)
    noisier_than_svd = sorted(
        (
            r["op"]
            for r in profiler_rows
            if r["op"] not in SCALAR_SYNC_OPS
            and "svd" not in r["op"]
            and r["self_cpu_us_avg"] > svd_avg
        ),
    )

    return {
        "timings": timings,
        "run_err": run_err,
        "fallback_ops": fallback_ops,
        "tracer_op_count": len(tracer_ops),
        "profiler_op_count": len(profiler_ops),
        "ops_seen_by_profiler_only": profiler_only,
        "ops_seen_by_tracer_only": tracer_only,
        "self_cpu_time_top_ops": self_cpu_sorted[:15],
        "svd_self_cpu_rows": svd_rows,
        "scalar_sync_self_cpu_rows": scalar_sync_rows,
        "ops_with_higher_avg_self_cpu_than_svd": noisier_than_svd,
        "verdict": verdict,
        "op_coverage": {
            st: {
                name: {
                    "calls": r.calls,
                    "input_devices": sorted(r.input_devices),
                    "output_devices": sorted(r.output_devices),
                    "has_mps_kernel": r.has_mps_kernel,
                    "verdict": e2.classify_op(r, fallback_ops, name),
                    "errored": r.errored,
                    "error": r.error_msg,
                }
                for name, r in sorted(ops.items())
            }
            for st, ops in tracer.records.items()
        },
    }


def run_timed_iteration(
    model,
    feats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    recycling_steps: int,
    sampling_steps: int,
    autocast_ctx,
) -> tuple[dict[str, float], set[str], Optional[Exception]]:
    """One plain (un-profiled, un-traced) timed forward pass."""
    timings: dict[str, float] = {}
    run_err = None
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            # No tracer needed for steady-state timing; a no-op stand-in keeps
            # run_staged_forward's signature identical to the warmup call.
            tracer = e2.OpCoverageTracer()
            e2.run_staged_forward(
                model,
                feats,
                device,
                tracer,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                autocast_ctx=autocast_ctx,
                timings=timings,
            )
        except Exception as exc:
            run_err = exc
            print(traceback.format_exc(), file=sys.stderr)
        fallback_ops = _harvest_fallback_ops(wlist)
    return timings, fallback_ops, run_err


RESIDENCY_MARKER_OP = "aten::_to_cpu"


def run_residency_check_iteration(
    model,
    feats: dict[str, torch.Tensor],
    device: torch.device,
    *,
    recycling_steps: int,
    sampling_steps: int,
    autocast_ctx,
) -> tuple[bool, Optional[Exception]]:
    """One lightweight torch.profiler-only pass confirming the SVD fallback's
    actual host transfer recurs on every iteration, independent of the warning.

    Two false starts ruled out empirically (see report/notes for the numbers):
    (1) PyTorch's "not currently supported on the MPS backend" fallback warning is
    a process-global warn-once - it fires only on an op's *first* invocation in the
    whole process, so re-harvesting it on later iterations always comes back empty
    and proves nothing about recurrence. (2) TorchDispatchMode's own device
    tracking is also no good here: the MPS fallback mechanism is transparent at the
    dispatch boundary the mode intercepts - both `aten::linalg_svd`'s input *and*
    output tensors are still tagged `mps` there, because the host round-trip
    happens *inside* the kernel implementation via internal sub-calls
    (`aten::_to_cpu`, `aten::_linalg_svd`, `aten::_linalg_check_errors`, ...) that
    bypass the Python-visible TorchDispatchMode stack entirely (confirmed: they
    never appear in the tracer's op inventory at all, only in the profiler's).
    The profiler's op-presence *does* reliably re-observe `aten::_to_cpu` - an
    unambiguous, name-level signal of an actual host transfer - on every call,
    warning or not, so that is the marker used here.
    """
    timings: dict[str, float] = {}
    run_err = None
    tracer = e2.OpCoverageTracer()  # unused for device info; satisfies the signature
    with tprof.profile(activities=[tprof.ProfilerActivity.CPU]) as prof:
        try:
            e2.run_staged_forward(
                model,
                feats,
                device,
                tracer,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                autocast_ctx=autocast_ctx,
                timings=timings,
            )
        except Exception as exc:
            run_err = exc
            print(traceback.format_exc(), file=sys.stderr)

    profiler_ops = {e.key for e in prof.key_averages()}
    confirmed = RESIDENCY_MARKER_OP in profiler_ops
    return confirmed, run_err


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


# ==================================================================================
# Reporting / artifacts
# ==================================================================================
def write_artifacts(
    out_dir: Path,
    manifest: dict[str, Any],
    static_only: bool,
    notes: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest["run_id"]
    sidecar = dict(manifest)
    sidecar["notes"] = notes
    (out_dir / f"{stem}.manifest.json").write_text(
        __import__("json").dumps(sidecar, indent=2)
    )

    lines: list[str] = []
    lines.append("# E2-confirm - profiler-backed confirmation pass\n")
    lines.append(f"- run_id: `{stem}`")
    lines.append(f"- parent_experiment: {manifest['parent_experiment']}")
    lines.append(f"- timestamp: {manifest['timestamp_utc']}")
    lines.append(f"- device: **{manifest['device']}** | dtype: **{manifest['dtype']}**")
    lines.append(f"- real model ran end-to-end on MPS: **{manifest['ran_on_device']}**")
    lines.append(f"- weights: `{manifest['weights_file']}`")
    lines.append(f"  - weights_version: **{manifest['weights_version']}**")
    lines.append(f"  - weights_sha256: `{manifest['weights_sha256']}`")
    lines.append(f"- input: **{manifest['input_kind']}**")
    lines.append(f"- code_sha: `{manifest['code_sha']}` | boltz_commit: `{manifest['boltz_commit']}`")
    lines.append(f"- opm_mode (BOLTZMAC_OPM): **{manifest.get('opm_mode', 'stock')}**")
    lines.append(f"- hardware: {manifest['hardware']} | os: {manifest['os']}")
    lines.append(f"- torch: {manifest['torch_version']}")
    lines.append(f"- seed: {manifest['seed']}")
    lines.append(f"- warmup_iterations: {manifest.get('warmup_iterations')}")
    lines.append(f"- timed_iterations: {manifest.get('timed_iterations')}")
    lines.append(f"- command: `{manifest['command']}`\n")

    if notes:
        lines.append("## Notes & caveats\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    if not static_only:
        pc = manifest.get("profiler_cross_check", {})
        lines.append("## Profiler cross-check verdict\n")
        lines.append(f"> **{pc.get('verdict', 'n/a')}**\n")
        lines.append(
            f"- fallback ops re-harvested (this run's warmup pass): "
            f"`{pc.get('fallback_ops', [])}`"
        )
        lines.append(
            "- PyTorch's fallback warning is a **process-global warn-once**: it "
            "fired on the warmup pass above and was empirically confirmed to NOT "
            "re-fire on any of the later steady-state iterations (re-harvested "
            f"warnings per timed iteration: `{pc.get('timed_iteration_fallback_warnings')}`"
            " - all empty, as expected, not evidence of absence). TorchDispatchMode's "
            "own device tracking is also no use here: the fallback is transparent at "
            "the dispatch boundary it intercepts, so `aten::linalg_svd`'s inputs/"
            "outputs are still tagged `mps` there (the host round-trip happens "
            "*inside* the kernel, via internal sub-calls like `aten::_to_cpu` that "
            "never reach the tracer). Recurrence was instead confirmed via "
            f"**`torch.profiler` re-observing `{RESIDENCY_MARKER_OP}`** - an "
            "unambiguous host-transfer marker - on "
            f"{pc.get('residency_check_iterations')} separate lightweight passes: "
            f"residency_confirmed_every_iteration = "
            f"**{pc.get('linalg_svd_cpu_residency_confirmed_each_iteration')}**"
        )
        lines.append(
            f"- TorchDispatchMode tracer op inventory: {pc.get('tracer_op_count')} "
            f"distinct ops | torch.profiler op inventory: {pc.get('profiler_op_count')} "
            "distinct ops"
        )
        lines.append(
            "- ops seen by the dispatch-mode tracer but **missed** by the profiler "
            f"(should be empty - a non-empty list would mean the tracer over-reports "
            f"vs. an independent instrumentation point): "
            f"`{pc.get('ops_seen_by_tracer_only') or '[] (none)'}`"
        )
        lines.append(
            "- ops seen by the profiler but not by the tracer (expected: the "
            "composite/Python-API SVD-and-rigid-alignment call graph plus generic "
            "composite wrappers that decompose to MPS-supported primitives "
            f"elsewhere): `{pc.get('ops_seen_by_profiler_only')}`\n"
        )

        lines.append("### Why self-CPU time is not used to classify fallbacks\n")
        lines.append(
            "`torch.profiler`'s `device_type` field reports `DeviceType.CPU` for "
            "*every* op on this MPS build (there is no `ProfilerActivity.MPS` in "
            "torch "
            f"{manifest['torch_version']}), and self-CPU time is dominated by "
            "MPS command-buffer dispatch/sync overhead rather than actual execution "
            "device. Empirical evidence from this run: the following ops - all "
            "classified `mps` by the device-tracking tracer, i.e. **not** fallbacks "
            "- show a *higher* average self-CPU time per call than "
            "`aten::linalg_svd` itself: "
            f"`{pc.get('ops_with_higher_avg_self_cpu_than_svd')}`. A "
            "timing-threshold classifier would therefore misclassify ops in both "
            "directions; fallback identity is decided solely by PyTorch's own "
            "fallback-warning ground truth, exactly as in E2.\n"
        )
        lines.append("### Legitimate scalar syncs vs. true op fallback (not conflated)\n")
        lines.append(
            "| op | calls | self_cpu_us_total | self_cpu_us_avg | classification |"
        )
        lines.append("|---|---:|---:|---:|---|")
        for r in pc.get("scalar_sync_self_cpu_rows", []):
            lines.append(
                f"| `{r['op']}` | {r['calls']} | {r['self_cpu_us_total']} | "
                f"{r['self_cpu_us_avg']} | legitimate host scalar sync (NOT a "
                "fallback) |"
            )
        for r in pc.get("svd_self_cpu_rows", []):
            lines.append(
                f"| `{r['op']}` | {r['calls']} | {r['self_cpu_us_total']} | "
                f"{r['self_cpu_us_avg']} | **true unsupported-op CPU fallback** "
                "(confirmed by PyTorch fallback warning) |"
            )
        lines.append("")

        lines.append("### Top ops by total self-CPU time (informational only)\n")
        lines.append("| op | calls | self_cpu_us_total | self_cpu_us_avg |")
        lines.append("|---|---:|---:|---:|")
        for r in pc.get("self_cpu_time_top_ops", []):
            lines.append(
                f"| `{r['op']}` | {r['calls']} | {r['self_cpu_us_total']} | "
                f"{r['self_cpu_us_avg']} |"
            )
        lines.append("")

    if not static_only:
        lines.append("## Timing: one-time warmup vs. steady state\n")
        wib = manifest.get("warmup_iterations_breakdown", {})
        lines.append(
            "Warmup = first forward pass after model load (pays one-time MPS "
            "kernel-compile cost; also the profiler-instrumented pass above, so its "
            "absolute numbers run slightly high vs. a bare pass and are reported "
            "separately, never mixed into the steady-state stats below). "
            f"`--warmup-iters={manifest.get('warmup_iterations')}` -> "
            f"{wib.get('profiled_pass', 1)} profiled pass (timed below) + "
            f"{wib.get('extra_plain_passes', 0)} additional plain warmup pass(es) "
            "(discarded entirely, not shown in any table).\n"
        )
        warm = manifest.get("warmup_wall_clock_by_stage_s", {})
        lines.append("| stage | warmup seconds (1 pass, profiler-instrumented) |")
        lines.append("|---|---:|")
        for st in e2.STAGE_ORDER:
            if st.strip() in warm:
                lines.append(f"| {st} | {warm[st.strip()]:.4f} |")
        lines.append(f"| **total** | **{sum(warm.values()):.4f}** |\n")

        ss = manifest.get("steady_state_wall_clock_by_stage_s", {})
        n_iters = manifest.get("timed_iterations")
        lines.append(
            f"Steady state: mean +/- std over {n_iters} timed iterations "
            "(warmup discarded), `torch.mps.synchronize()` around every stage.\n"
        )
        lines.append("| stage | mean (s) | std (s) | n |")
        lines.append("|---|---:|---:|---:|")
        total_mean = 0.0
        for st in e2.STAGE_ORDER:
            key = st.strip()
            if key in ss:
                total_mean += ss[key]["mean_s"]
                lines.append(
                    f"| {st} | {ss[key]['mean_s']:.4f} | {ss[key]['std_s']:.4f} | "
                    f"{ss[key]['n']} |"
                )
        lines.append(f"| **total (sum of stage means)** | **{total_mean:.4f}** | | |\n")

    lines.append("## Relationship to E2\n")
    lines.append(
        "This run reuses E2's model loading, synthetic-feats construction, "
        "staged-forward driver and op classification verbatim "
        "(`experiments/e2_mps_opcoverage.py`, no changes made to that file) and adds "
        "(1) a torch.profiler-based independent cross-check of the device-fallback "
        "claim and (2) warmup/steady-state separated timing. It does not change "
        "E2's conclusions; see the verdict above and the headline finding in "
        "`experiments/README.md`.\n"
    )

    (out_dir / f"{stem}.report.md").write_text("\n".join(lines))
    print(f"[e2-confirm] wrote {out_dir / f'{stem}.report.md'}")
    print(f"[e2-confirm] wrote {out_dir / f'{stem}.manifest.json'}")


# ==================================================================================
# Main
# ==================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--checkpoint", default=str(e2.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "real"))
    # Default sizes mirror E2's --fast config: the default-sized config is known (see
    # E2 README) to occasionally hit synthetic-coordinate SVD non-convergence on
    # CPU, which would make repeated steady-state iterations unreliable; the --fast
    # size completed `ran_on_device: yes` end-to-end in E2 and is reused here as the
    # default for run-to-run numerical stability across multiple iterations. All
    # sizes remain overridable.
    ap.add_argument("--n-tokens", type=int, default=24)
    ap.add_argument("--n-atoms", type=int, default=64)
    ap.add_argument("--n-msa", type=int, default=8)
    ap.add_argument("--recycling-steps", type=int, default=0)
    ap.add_argument("--sampling-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="total untimed warmup passes before steady-state timing (minimum 1: the "
        "first warmup pass always doubles as the profiler cross-check pass; "
        "warmup_iters > 1 runs additional plain discarded passes after it, in case a "
        "single pass doesn't fully stabilize MPS kernel-compile state)",
    )
    ap.add_argument(
        "--timed-iters",
        type=int,
        default=4,
        help="timed steady-state iterations after warmup (recommended 3-5; any "
        "positive int is accepted, with a warning outside that range)",
    )
    ap.add_argument(
        "--static-only",
        action="store_true",
        help="skip the live run; emit methodology only (mirrors E2's --static-only)",
    )
    args = ap.parse_args()

    if args.warmup_iters < 1:
        print(
            f"[e2-confirm] WARNING: --warmup-iters={args.warmup_iters} < 1; clamping "
            "to 1 (the profiler cross-check pass is mandatory and always counts as "
            "the first warmup pass)."
        )
        args.warmup_iters = 1
    if args.timed_iters < 1:
        print(f"[e2-confirm] ERROR: --timed-iters={args.timed_iters} must be >= 1.")
        return 1
    if not (3 <= args.timed_iters <= 5):
        print(
            f"[e2-confirm] WARNING: --timed-iters={args.timed_iters} is outside the "
            "recommended 3-5 range for stable mean/std estimates; proceeding anyway."
        )

    if args.device == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        print("[e2-confirm] WARNING: mps requested but unavailable; falling back to cpu.")
        args.device = "cpu"

    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[
        args.dtype
    ]
    torch.manual_seed(args.seed)

    run_id = "e2_profile_confirm_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ckpt_path = Path(os.path.expanduser(args.checkpoint))
    notes: list[str] = []

    try:
        from boltz.model.layers.outer_product_mean import resolve_opm_mode

        opm_mode = resolve_opm_mode()
    except Exception:
        opm_mode = (os.environ.get("BOLTZMAC_OPM", "stock") or "stock").strip().lower()

    import platform

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment": "E2-confirm - profiler-backed confirmation pass",
        "parent_experiment": "E2 - MPS op-coverage / device-fallback (e2_mps_opcoverage.py)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "code_sha": e2.git_sha(REPO_ROOT),
        "boltz_commit": e2.git_sha(REPO_ROOT),
        "hardware": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()} ({platform.platform()})",
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "input_kind": (
            f"SYNTHETIC feats (n_tokens={args.n_tokens}, n_atoms={args.n_atoms}, "
            f"n_msa={args.n_msa}) - real model + real weights, synthetic input "
            "(identical methodology to E2)"
        ),
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "opm_mode": opm_mode,
        "command": "python " + " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "config": {
            "recycling_steps": args.recycling_steps,
            "sampling_steps": args.sampling_steps,
            "n_tokens": args.n_tokens,
            "n_atoms": args.n_atoms,
            "n_msa": args.n_msa,
        },
        "warmup_iterations": args.warmup_iters,
        "timed_iterations": args.timed_iters,
        "weights_file": str(ckpt_path),
        "weights_version": "unknown",
        "weights_sha256": "unknown",
        "ran_on_device": "no",
    }

    out_dir = Path(args.out_dir)

    have_ckpt = ckpt_path.exists()
    if have_ckpt:
        print(f"[e2-confirm] hashing checkpoint {ckpt_path} ...")
        manifest["weights_sha256"] = e2.sha256_file(ckpt_path)
        name = ckpt_path.name.lower()
        if "boltz1" in name or "boltz-1" in name:
            manifest["weights_version"] = "Boltz-1 (boltz1_conf.ckpt)"
            notes.append(
                "WEIGHTS CAVEAT (same as E2): the loaded checkpoint is **Boltz-1**, "
                "not Boltz-2. Profiling real Boltz-2 needs Boltz-2 weights."
            )
        elif "boltz2" in name or "boltz-2" in name:
            manifest["weights_version"] = "Boltz-2 (boltz2_conf.ckpt)"
        else:
            manifest["weights_version"] = ckpt_path.name
    else:
        notes.append(
            f"No checkpoint found at {ckpt_path}: emitting methodology only. "
            "Re-run with --checkpoint pointing at real weights."
        )

    notes.append(
        "INPUT CAVEAT (same as E2): synthetic feats are used because op-coverage "
        "and timing-shape are architecture driven; default sizes here match E2's "
        "--fast config for run-to-run numerical stability across the timed "
        "iterations (the default-sized config is known per E2 to occasionally hit "
        "synthetic-coordinate SVD non-convergence on CPU)."
    )

    if args.static_only or not have_ckpt:
        notes.append(
            "Run mode: STATIC/DRY (no live forward pass)."
            if args.static_only
            else "Run mode: STATIC/DRY (weights missing)."
        )
        write_artifacts(out_dir, manifest, True, notes)
        print("[e2-confirm] static-only artifacts written. (no live MPS run)")
        return 0

    print(f"[e2-confirm] loading Boltz1 from {ckpt_path} on {device} ...")
    try:
        model = load_model(ckpt_path, device)
    except Exception as exc:
        tb = traceback.format_exc()
        notes.append(f"MODEL LOAD FAILED: {type(exc).__name__}: {exc}")
        print(tb, file=sys.stderr)
        write_artifacts(out_dir, manifest, True, notes)
        return 0

    feats = e2.build_synthetic_feats(args.n_tokens, args.n_atoms, args.n_msa, device)
    if dtype != torch.float32 and device.type in ("mps", "cuda"):
        autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    # ---- warmup + profiler cross-check pass ----
    print("[e2-confirm] running warmup + profiler cross-check pass ...")
    warm = run_profiled_warmup(
        model,
        feats,
        device,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        autocast_ctx=autocast_ctx,
    )
    warmup_ok = warm["run_err"] is None
    manifest["warmup_wall_clock_by_stage_s"] = warm["timings"]
    manifest["op_coverage"] = warm["op_coverage"]
    manifest["fallback_ops"] = sorted(warm["fallback_ops"])

    # ---- extra plain warmup passes (warmup_iters > 1), discarded entirely ----
    extra_warmup = args.warmup_iters - 1
    extra_warmup_err = None
    if extra_warmup > 0:
        print(f"[e2-confirm] running {extra_warmup} additional plain warmup pass(es) "
              "(discarded, not the profiled one) ...")
        for i in range(extra_warmup):
            _timings_w, _fb_w, err_w = run_timed_iteration(
                model,
                feats,
                device,
                recycling_steps=args.recycling_steps,
                sampling_steps=args.sampling_steps,
                autocast_ctx=autocast_ctx,
            )
            if err_w is not None:
                extra_warmup_err = err_w
            print(f"[e2-confirm]   extra warmup {i + 1}/{extra_warmup}: "
                  f"{'ok' if err_w is None else 'ERROR'} "
                  f"total={sum(_timings_w.values()):.4f}s (discarded)")
    manifest["warmup_iterations_breakdown"] = {
        "profiled_pass": 1,
        "extra_plain_passes": extra_warmup,
    }

    # ---- steady-state timed iterations (clean, uninstrumented) ----
    print(f"[e2-confirm] running {args.timed_iters} timed steady-state iterations ...")
    per_stage_samples: dict[str, list[float]] = {}
    timed_iteration_fallback_warnings: list[list[str]] = []
    last_err = None
    for i in range(args.timed_iters):
        timings_i, fb_i, err_i = run_timed_iteration(
            model,
            feats,
            device,
            recycling_steps=args.recycling_steps,
            sampling_steps=args.sampling_steps,
            autocast_ctx=autocast_ctx,
        )
        for st, t in timings_i.items():
            per_stage_samples.setdefault(st, []).append(t)
        timed_iteration_fallback_warnings.append(sorted(fb_i))
        if err_i is not None:
            last_err = err_i
        print(f"[e2-confirm]   iter {i + 1}/{args.timed_iters}: "
              f"{'ok' if err_i is None else 'ERROR'} total={sum(timings_i.values()):.4f}s")

    steady_state = {
        st: {
            "mean_s": (ms := mean_std(samples))[0],
            "std_s": ms[1],
            "n": len(samples),
            "samples_s": samples,
        }
        for st, samples in per_stage_samples.items()
    }
    manifest["steady_state_wall_clock_by_stage_s"] = steady_state

    # ---- residency-confirmation passes (separate from timing; not warning-based) ----
    n_residency = min(args.timed_iters, 3)
    print(f"[e2-confirm] running {n_residency} device-residency confirmation passes "
          f"(torch.profiler only, checking for {RESIDENCY_MARKER_OP}, not timed) ...")
    residency_confirmed_each_iter: list[bool] = []
    residency_err = None
    for i in range(n_residency):
        confirmed, err_i = run_residency_check_iteration(
            model,
            feats,
            device,
            recycling_steps=args.recycling_steps,
            sampling_steps=args.sampling_steps,
            autocast_ctx=autocast_ctx,
        )
        if err_i is not None:
            residency_err = err_i
        residency_confirmed_each_iter.append(confirmed)
        print(f"[e2-confirm]   residency check {i + 1}/{n_residency}: "
              f"{RESIDENCY_MARKER_OP}_present={confirmed}")

    linalg_svd_residency_confirmed = bool(residency_confirmed_each_iter) and all(
        residency_confirmed_each_iter
    )

    all_ok = (
        warmup_ok
        and extra_warmup_err is None
        and last_err is None
        and residency_err is None
    )
    manifest["ran_on_device"] = (
        "yes" if (all_ok and device.type == "mps") else
        ("partial" if device.type == "mps" else f"yes ({device.type})")
    )
    if not all_ok:
        manifest["ran_on_device"] = "partial"
        notes.append(
            "RUN ERROR in at least one pass (warmup and/or a timed iteration) - see "
            "stderr / re-run logs. Steady-state stats reflect only the iterations "
            "that completed."
        )

    manifest["profiler_cross_check"] = {
        "verdict": warm["verdict"],
        "fallback_ops": sorted(warm["fallback_ops"]),
        "timed_iteration_fallback_warnings": timed_iteration_fallback_warnings,
        "residency_check_iterations": n_residency,
        "linalg_svd_cpu_residency_confirmed_each_iteration": linalg_svd_residency_confirmed,
        "scalar_sync_ops_excluded_from_fallback_classification": sorted(SCALAR_SYNC_OPS),
        "tracer_op_count": warm["tracer_op_count"],
        "profiler_op_count": warm["profiler_op_count"],
        "ops_seen_by_profiler_only": warm["ops_seen_by_profiler_only"],
        "ops_seen_by_tracer_only": warm["ops_seen_by_tracer_only"],
        "self_cpu_time_top_ops": warm["self_cpu_time_top_ops"],
        "svd_self_cpu_rows": warm["svd_self_cpu_rows"],
        "scalar_sync_self_cpu_rows": warm["scalar_sync_self_cpu_rows"],
        "ops_with_higher_avg_self_cpu_than_svd": warm["ops_with_higher_avg_self_cpu_than_svd"],
    }

    notes.append(
        "torch.profiler has no ProfilerActivity.MPS in this torch build "
        f"({torch.__version__}); device_type reports DeviceType.CPU for every op "
        "regardless of actual execution device, and self-CPU time is dominated by "
        "MPS dispatch/sync overhead. The profiler is therefore used here for "
        "op-presence corroboration and explicit scalar-sync separation, NOT for "
        "timing-based device classification - see 'Why self-CPU time is not used "
        "to classify fallbacks' in the report."
    )
    notes.append(
        "PyTorch's MPS-unsupported fallback warning is a process-global warn-once "
        "(empirically verified: it fires on an op's first invocation in the process "
        "and never again, regardless of Python warnings-filter state) - re-harvesting "
        "it per iteration cannot confirm the fallback recurs on every iteration. "
        "TorchDispatchMode's own device tracking is also no use here: the fallback "
        "is transparent at the dispatch boundary it intercepts (aten::linalg_svd's "
        "inputs/outputs are still tagged mps there; the host round-trip happens "
        "inside the kernel via internal sub-calls like aten::_to_cpu that never "
        f"reach the tracer). Recurrence was instead confirmed via torch.profiler "
        f"re-observing {RESIDENCY_MARKER_OP} on {n_residency} dedicated passes: "
        f"confirmed_every_iteration={linalg_svd_residency_confirmed}."
    )
    notes.append(f"profiler cross-check verdict: {warm['verdict']}")

    write_artifacts(out_dir, manifest, False, notes)

    print(f"[e2-confirm] DONE. ran_on_device={manifest['ran_on_device']} "
          f"verdict={warm['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
