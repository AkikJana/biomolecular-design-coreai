#!/usr/bin/env python3
"""E2 - Apple-Silicon (MPS) smoke run + op-coverage / device-fallback table for Boltz.

This experiment answers the single biggest open question in the project: does the
real Boltz model run end-to-end on Apple Silicon via the PyTorch ``mps`` backend,
and *which* ATen ops fall back to the CPU?

What it does
------------
1. Loads a real Boltz checkpoint onto ``device=mps`` (eval mode), optionally with
   autocast / fp16 where applicable.
2. Drives a single forward pass *stage by stage* - replicating the exact sequence in
   ``Boltz1.forward`` (input featurization -> trunk recycles {MSA + Pairformer, which
   contains triangle multiplication / triangle attention} -> distogram -> diffusion
   sampler -> confidence head). Driving the public sub-modules directly (the same way
   the upstream regression tests do) lets us scope timing and op-tracing per stage
   *without modifying any model source*.
3. Captures, per stage, an op-coverage / device-fallback table via two independent,
   cross-referenced signals:
     (a) a ``TorchDispatchMode`` op-tracer that records every ATen op invoked, the
         devices of its tensor inputs, and a static "does this op have an MPS kernel"
         probe (``torch._C._dispatch_has_kernel_for_dispatch_key``);
     (b) the authoritative set of CPU-fallback ops harvested from PyTorch's own
         ``UserWarning`` ("operator ... is not currently supported on the MPS backend
         and will fall back to run on the CPU"). This (b) signal is ground truth for
         fallbacks; (a) provides stage attribution, counts, and a supplementary static
         signal (which can over-report for composite ops that decompose, so we label
         the source of each verdict).
4. Emits an op/stage x {runs on MPS | falls back to CPU | unsupported/fails} table
   plus a coarse wall-clock-by-stage breakdown.
5. Writes artifacts into ``results/real/`` with a manifest-style sidecar (code_sha,
   boltz commit, weights file + sha256, seed, hardware, os, device, dtype, command).

WEIGHTS CAVEAT (read me)
------------------------
The only checkpoint found locally is ``~/.boltz/boltz1_conf.ckpt`` which is
**Boltz-1**, not Boltz-2. The op-coverage / fallback map is largely architecture
driven, so a first MPS smoke run with Boltz-1 weights is still highly informative for
the question "what falls back on MPS". But the run is clearly labelled with the actual
weights version + sha256 used, and properly profiling *Boltz-2* requires Boltz-2
weights. If no checkpoint loads at all, the script still emits the op-coverage
methodology + a static analysis of the Boltz ops known to be problematic on MPS, and
exits non-silently.

INPUT CAVEAT
------------
Building a real featurized input requires MSA generation + CCD/mol assets + RDKit,
which is heavy and network-dependent, and the upstream regression-feature pickle is an
untrusted external download. Because the op-coverage / fallback map is *architecture
driven* (it depends on which ATen ops the layers invoke, not on the numeric content of
the inputs), this script feeds a **synthetic feature dict** with the correct keys,
shapes and dtypes. The run is therefore "real model + real weights, synthetic input"
and is labelled as such in every artifact. Swapping in a real ``feats`` dict (e.g. from
``BoltzInferenceDataModule``) requires no other change.

Usage
-----
    PYTHONPATH=boltz/src python experiments/e2_mps_opcoverage.py            # default smoke
    PYTHONPATH=boltz/src python experiments/e2_mps_opcoverage.py --fast     # tiny + fast
    PYTHONPATH=boltz/src python experiments/e2_mps_opcoverage.py --device cpu --dtype fp32

The heavy knobs (token/atom/MSA counts, recycles, sampling steps) are all
parametrizable so reviewers can run a fast path.
"""

from __future__ import annotations

# NOTE: PYTORCH_ENABLE_MPS_FALLBACK must be set *before* torch is imported so that
# unsupported MPS ops fall back to CPU (and emit the warning we harvest) instead of
# hard-crashing the run. We default it on but respect an explicit override.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

# ----------------------------------------------------------------------------------
# Repo / path setup. boltz is a src-layout package (boltz/src on PYTHONPATH).
# ----------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
BOLTZ_SRC = REPO_ROOT / "boltz" / "src"
if BOLTZ_SRC.exists() and str(BOLTZ_SRC) not in sys.path:
    sys.path.insert(0, str(BOLTZ_SRC))

DEFAULT_CKPT = Path(os.path.expanduser("~/.boltz/boltz1_conf.ckpt"))


# ==================================================================================
# Static knowledge: Boltz ops with a history of being unsupported / slow on MPS.
# Used for the dry-analysis fallback (no weights) and to annotate the live table.
# ==================================================================================
KNOWN_MPS_RISK_OPS: list[dict[str, str]] = [
    {
        "stage": "diffusion / SVD alignment",
        "op": "aten::linalg_svd",
        "where": "boltz.model.loss.diffusion.weighted_rigid_align "
        "(torch.linalg.svd, driver='gesvd' on CUDA only), called from "
        "AtomDiffusion.sample under `alignment_reverse_diff`",
        "risk": "OBSERVED empirically: linalg_svd has no MPS kernel and falls back to "
        "CPU (PyTorch emits its unsupported-MPS fallback warning). It is hit on EVERY "
        "reverse-diffusion step of the DEFAULT (unsteered) sampler via "
        "alignment_reverse_diff - i.e. on every Boltz inference - not only in "
        "training/steered sampling. It was the ONLY unsupported-MPS fallback observed "
        "on this path (silent host<->device scalar syncs are not counted as fallbacks).",
    },
    {
        "stage": "diffusion / SVD alignment",
        "op": "aten::linalg_qr / aten::linalg_eigh",
        "where": "linear-algebra helpers around rigid alignment",
        "risk": "linalg factorizations are commonly CPU-only on MPS.",
    },
    {
        "stage": "trunk / triangle attention",
        "op": "aten::scaled_dot_product_attention / softmax / einsum",
        "where": "TriangleAttentionStartingNode / EndingNode, AttentionPairBias",
        "risk": "Generally supported on MPS, but the fused SDPA kernel coverage varies "
        "by torch version; einsum decomposes to bmm/permute which are supported.",
    },
    {
        "stage": "trunk / triangle multiplication",
        "op": "aten::einsum -> aten::bmm",
        "where": "TriangleMultiplicationOutgoing / Incoming",
        "risk": "Supported on MPS (bmm/mul/sigmoid). Watch for fp16 autocast edge cases.",
    },
    {
        "stage": "diffusion sampler",
        "op": "aten::randn / aten::native_layer_norm / aten::index_add",
        "where": "AtomDiffusion.sample noise + atom-attention scatter/gather",
        "risk": "index/scatter ops occasionally fall back; randn supported on MPS.",
    },
    {
        "stage": "input featurization",
        "op": "aten::one_hot / aten::cdist",
        "where": "encoders / relative position encoding",
        "risk": "cdist and some indexing ops have had MPS gaps across torch versions.",
    },
]


# ==================================================================================
# Op tracer
# ==================================================================================
@dataclass
class OpRecord:
    calls: int = 0
    input_devices: set[str] = field(default_factory=set)
    output_devices: set[str] = field(default_factory=set)
    has_mps_kernel: Optional[bool] = None
    errored: bool = False
    error_msg: str = ""


def _op_base_name(func: Any) -> str:
    """Best-effort qualified ATen op name, e.g. 'aten::add.Tensor' -> 'aten::add'."""
    try:
        name = func.name()  # OpOverload.name() -> 'aten::add.Tensor'
    except Exception:
        name = str(func)
    return name.split(".")[0]


def _has_mps_kernel(func: Any) -> Optional[bool]:
    base = _op_base_name(func)
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(base, "MPS")
        )
    except Exception:
        return None


class OpCoverageTracer(torch.utils._python_dispatch.TorchDispatchMode):
    """Records, per current stage, every ATen op and the devices of its tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.current_stage = "uncategorized"
        # stage -> op_name -> OpRecord
        self.records: dict[str, dict[str, OpRecord]] = defaultdict(
            lambda: defaultdict(OpRecord)
        )

    @staticmethod
    def _tensor_devices(args: Any, devices: set[str]) -> None:
        if isinstance(args, torch.Tensor):
            devices.add(args.device.type)
        elif isinstance(args, (list, tuple)):
            for a in args:
                OpCoverageTracer._tensor_devices(a, devices)
        elif isinstance(args, dict):
            for a in args.values():
                OpCoverageTracer._tensor_devices(a, devices)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = _op_base_name(func)
        rec = self.records[self.current_stage][op_name]
        rec.calls += 1
        if rec.has_mps_kernel is None:
            rec.has_mps_kernel = _has_mps_kernel(func)
        in_devs: set[str] = set()
        self._tensor_devices(args, in_devs)
        self._tensor_devices(kwargs, in_devs)
        rec.input_devices |= in_devs
        try:
            out = func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - records hard failures
            rec.errored = True
            rec.error_msg = f"{type(exc).__name__}: {exc}"
            raise
        out_devs: set[str] = set()
        self._tensor_devices(out, out_devs)
        rec.output_devices |= out_devs
        return out


# ==================================================================================
# Synthetic feature dict (real model, synthetic input).
# ==================================================================================
def build_synthetic_feats(
    n_tokens: int,
    n_atoms: int,
    n_msa: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Construct a feats dict with correct keys / shapes / dtypes for Boltz-1.

    Shapes follow boltz.data.feature.featurizer; a leading batch dim B=1 is added (as
    the inference data module's collate would). atom count is padded to a multiple of
    the atom-attention window (32).
    """
    from boltz.data import const

    g = torch.Generator().manual_seed(0)
    W = 32  # atoms_per_window_queries
    n_atoms = ((n_atoms + W - 1) // W) * W

    def rint(high: int, *shape: int) -> torch.Tensor:
        return torch.randint(0, max(high, 1), shape, generator=g)

    num_tokens = const.num_tokens          # 33
    num_elements = const.num_elements      # 128
    n_pocket = len(const.pocket_contact_info)  # 4

    feats: dict[str, torch.Tensor] = {}

    # --- token-level features (B, N, ...) ---
    feats["token_index"] = torch.arange(n_tokens).long()[None]
    feats["residue_index"] = torch.arange(n_tokens).long()[None]
    feats["asym_id"] = torch.zeros(1, n_tokens).long()
    feats["entity_id"] = torch.zeros(1, n_tokens).long()
    feats["sym_id"] = torch.zeros(1, n_tokens).long()
    # mol_type: protein tokens (0). Keep it polymer so MSA path stays active.
    feats["mol_type"] = torch.zeros(1, n_tokens).long()
    feats["res_type"] = torch.nn.functional.one_hot(
        rint(num_tokens, n_tokens), num_classes=num_tokens
    ).float()[None]
    feats["token_bonds"] = torch.zeros(1, n_tokens, n_tokens, 1).float()
    feats["token_pad_mask"] = torch.ones(1, n_tokens).float()
    feats["token_resolved_mask"] = torch.ones(1, n_tokens).float()
    feats["token_disto_mask"] = torch.ones(1, n_tokens).float()
    feats["pocket_feature"] = torch.nn.functional.one_hot(
        torch.zeros(n_tokens).long(), num_classes=n_pocket
    ).float()[None]
    feats["cyclic_period"] = torch.zeros(1, n_tokens).float()
    feats["disto_center"] = torch.randn(1, n_tokens, 3, generator=g)

    # --- atom-level features (B, A, ...) ---
    feats["ref_pos"] = torch.randn(1, n_atoms, 3, generator=g)
    feats["ref_charge"] = torch.zeros(1, n_atoms).float()
    feats["ref_element"] = torch.nn.functional.one_hot(
        rint(num_elements, n_atoms), num_classes=num_elements
    ).float()[None]
    feats["ref_atom_name_chars"] = torch.nn.functional.one_hot(
        rint(64, n_atoms, 4), num_classes=64
    ).float()[None]
    feats["ref_space_uid"] = torch.zeros(1, n_atoms).long()
    feats["atom_pad_mask"] = torch.ones(1, n_atoms).float()
    feats["atom_resolved_mask"] = torch.ones(1, n_atoms).bool()
    # each atom maps to a token (one-hot over tokens), spread atoms across tokens
    atom_tok = (torch.arange(n_atoms) % n_tokens).long()
    feats["atom_to_token"] = torch.nn.functional.one_hot(
        atom_tok, num_classes=n_tokens
    ).float()[None]
    # representative atom per token (one-hot over atoms)
    rep_atom = (torch.arange(n_tokens) % n_atoms).long()
    feats["token_to_rep_atom"] = torch.nn.functional.one_hot(
        rep_atom, num_classes=n_atoms
    ).float()[None]
    feats["r_set_to_rep_atom"] = torch.nn.functional.one_hot(
        rep_atom, num_classes=n_atoms
    ).float()[None]
    feats["coords"] = torch.randn(1, 1, n_atoms, 3, generator=g)
    feats["frames_idx"] = torch.zeros(1, n_tokens, 3).long()
    feats["frame_resolved_mask"] = torch.ones(1, n_tokens).bool()

    # --- MSA features (B, S, N, ...) ---
    feats["msa"] = torch.nn.functional.one_hot(
        rint(num_tokens, n_msa, n_tokens), num_classes=num_tokens
    ).float()[None]
    feats["msa_mask"] = torch.ones(1, n_msa, n_tokens).float()
    feats["msa_paired"] = torch.zeros(1, n_msa, n_tokens).float()
    feats["deletion_value"] = torch.zeros(1, n_msa, n_tokens).float()
    feats["has_deletion"] = torch.zeros(1, n_msa, n_tokens).bool()
    feats["deletion_mean"] = torch.zeros(1, n_tokens).float()
    feats["profile"] = torch.zeros(1, n_tokens, num_tokens).float()

    return {k: v.to(device) for k, v in feats.items()}


# ==================================================================================
# Stage driver
# ==================================================================================
STAGE_ORDER = [
    "input_featurization",
    "trunk_msa_module",
    "trunk_pairformer",
    "  triangle_multiplication",
    "  triangle_attention",
    "  attention_pair_bias",
    "distogram",
    "diffusion_sampler",
    "confidence_head",
]


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class SubStageHooks:
    """Forward hooks that re-tag the tracer's current stage for nested sub-modules
    (triangle mult / triangle attention / attention-pair-bias) and back again."""

    def __init__(self, model, tracer: OpCoverageTracer) -> None:
        self.tracer = tracer
        self.handles = []
        from boltz.model.layers.triangular_mult import (
            TriangleMultiplicationIncoming,
            TriangleMultiplicationOutgoing,
        )

        tri_attn_names = ("TriangleAttentionStartingNode", "TriangleAttentionEndingNode")
        attn_names = ("AttentionPairBias", "AttentionPairBiasV2")

        for module in model.modules():
            cls = type(module).__name__
            if isinstance(
                module,
                (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming),
            ):
                self._register(module, "  triangle_multiplication")
            elif cls in tri_attn_names:
                self._register(module, "  triangle_attention")
            elif cls in attn_names:
                self._register(module, "  attention_pair_bias")

    def _register(self, module, label: str) -> None:
        def pre(_m, _inp, _label=label):
            self._stack.append(self.tracer.current_stage)
            self.tracer.current_stage = _label

        def post(_m, _inp, _out):
            self.tracer.current_stage = self._stack.pop()

        self._stack: list[str] = getattr(self, "_stack", [])
        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def run_staged_forward(
    model,
    feats: dict[str, torch.Tensor],
    device: torch.device,
    tracer: OpCoverageTracer,
    *,
    recycling_steps: int,
    sampling_steps: int,
    autocast_ctx,
    timings: dict[str, float],
) -> dict[str, float]:
    """Replicate Boltz1.forward stage-by-stage; fill ``timings`` (per-stage seconds).

    ``timings`` is mutated in place so partial wall-clock survives if a later stage
    raises (e.g. SVD non-convergence on degenerate synthetic coords)."""

    @contextlib.contextmanager
    def stage(name: str):
        tracer.current_stage = name
        _sync(device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(device)
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)
            tracer.current_stage = "uncategorized"

    with torch.no_grad(), autocast_ctx:
        # ---- input featurization ----
        with stage("input_featurization"):
            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)
            z_init = (
                model.z_init_1(s_inputs)[:, :, None]
                + model.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = model.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + model.token_bonds(feats["token_bonds"].float())
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

        # ---- trunk recycles ----
        pairformer = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)
        for _ in range(recycling_steps + 1):
            s = s_init + model.s_recycle(model.s_norm(s))
            z = z_init + model.z_recycle(model.z_norm(z))
            if not model.no_msa:
                with stage("trunk_msa_module"):
                    z = z + model.msa_module(
                        z, s_inputs, feats, use_kernels=model.use_kernels
                    )
            with stage("trunk_pairformer"):
                s, z = pairformer(
                    s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels
                )

        # ---- distogram ----
        with stage("distogram"):
            pdistogram = model.distogram_module(z)

        # ---- diffusion sampler ----
        with stage("diffusion_sampler"):
            struct_out = model.structure_module.sample(
                s_trunk=s,
                z_trunk=z,
                s_inputs=s_inputs,
                feats=feats,
                relative_position_encoding=relative_position_encoding,
                num_sampling_steps=sampling_steps,
                atom_mask=feats["atom_pad_mask"],
                multiplicity=1,
                max_parallel_samples=1,
                train_accumulate_token_repr=False,
                steering_args=model.steering_args,
            )

        # ---- confidence head (incl. any affinity-style head) ----
        if getattr(model, "confidence_prediction", False):
            with stage("confidence_head"):
                model.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    s_diffusion=(
                        struct_out["diff_token_repr"]
                        if model.confidence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=struct_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=pdistogram.detach(),
                    multiplicity=1,
                    run_sequentially=False,
                    use_kernels=model.use_kernels,
                )

    return timings  # also mutated in place; returned for convenience


# ==================================================================================
# Reporting / artifacts
# ==================================================================================
def classify_op(rec: OpRecord, fallback_ops: set[str], op_name: str) -> str:
    """Return one of: 'mps', 'cpu_fallback', 'cpu_native', 'unsupported'.

    The *authoritative* fallback signal is ``fallback_ops`` - the set harvested from
    PyTorch's own "not currently supported on the MPS backend ... will fall back to run
    on the CPU" warnings. ``has_mps_kernel`` (a direct-registration probe) is recorded
    for information only and deliberately NOT used to decide fallbacks: structural /
    factory ops (``view``, ``expand``, ``clone``, ``_to_copy``, ``arange``, ``randn``,
    ...) have no *dedicated* MPS kernel yet run on MPS via composite / fall-through
    kernels, so keying off it over-reports fallbacks.
    """
    if rec.errored:
        return "unsupported"
    if op_name in fallback_ops:
        return "cpu_fallback"
    devs = rec.input_devices | rec.output_devices
    if "mps" in devs:
        # touched the MPS backend and did not warn / error -> it ran on MPS
        return "mps"
    return "cpu_native"


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifacts(
    out_dir: Path,
    manifest: dict[str, Any],
    timings: dict[str, float],
    tracer: Optional[OpCoverageTracer],
    fallback_ops: set[str],
    static_only: bool,
    notes: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest["run_id"]

    # ---- JSON sidecar (manifest) ----
    sidecar = dict(manifest)
    sidecar["wall_clock_by_stage_s"] = timings
    sidecar["fallback_ops"] = sorted(fallback_ops)
    sidecar["notes"] = notes
    if tracer is not None:
        op_dump: dict[str, dict[str, Any]] = {}
        for st, ops in tracer.records.items():
            op_dump[st] = {
                name: {
                    "calls": r.calls,
                    "input_devices": sorted(r.input_devices),
                    "output_devices": sorted(r.output_devices),
                    "has_mps_kernel": r.has_mps_kernel,
                    "verdict": classify_op(r, fallback_ops, name),
                    "errored": r.errored,
                    "error": r.error_msg,
                }
                for name, r in sorted(ops.items())
            }
        sidecar["op_coverage"] = op_dump
    (out_dir / f"{stem}.manifest.json").write_text(json.dumps(sidecar, indent=2))

    # ---- Markdown report ----
    lines: list[str] = []
    lines.append(f"# E2 - MPS op-coverage / fallback report\n")
    lines.append(f"- run_id: `{stem}`")
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
    lines.append(f"- command: `{manifest['command']}`\n")

    if notes:
        lines.append("## Notes & caveats\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    if not static_only and timings:
        lines.append("## Wall-clock by stage\n")
        lines.append(
            "> Smoke-run timings, not steady-state: the total includes one-time MPS "
            "kernel compilation / warmup (no separate warmup pass is run).\n"
        )
        total = sum(timings.values()) or 1.0
        lines.append("| stage | seconds | % |")
        lines.append("|---|---:|---:|")
        for st in STAGE_ORDER:
            if st.strip() in timings:
                t = timings[st.strip()]
                lines.append(f"| {st} | {t:.4f} | {100*t/total:.1f}% |")
        lines.append(f"| **total** | **{total:.4f}** | 100% |\n")

    if tracer is not None:
        lines.append("## Op coverage / device-fallback table\n")
        lines.append(
            "Verdicts: **mps** = ran on MPS; **cpu_fallback** = MPS-unsupported, fell "
            "back to CPU (confirmed by PyTorch's own fallback UserWarning); "
            "**cpu_native** = only ever saw CPU tensors; **unsupported** = raised.\n"
        )
        lines.append(
            "> `direct_mps_kernel` is informational only (does the op have a *dedicated* "
            "MPS registration). `NO` does **not** imply a fallback: structural/factory "
            "ops (view/expand/clone/_to_copy/arange/randn) run on MPS via composite or "
            "fall-through kernels. Fallback verdicts come solely from the fallback "
            "warning set.\n"
        )
        lines.append(
            "> **Ground truth for fallbacks is PyTorch's own *unsupported-MPS* fallback "
            "warnings.** Silent host<->device scalar syncs (e.g. `.item()` / "
            "`aten::_local_scalar_dense`) emit no such warning, so they are classified "
            "**mps** and never appear here. This table therefore reports "
            "*unsupported-op* CPU fallbacks, not data-transfer / sync overhead - so the "
            "defensible claim is \"the only unsupported-MPS fallback **observed** was "
            "`aten::linalg_svd`\", not \"everything else runs on MPS\".\n"
        )
        # global per-op rollup
        rollup: dict[str, dict[str, Any]] = {}
        for st, ops in tracer.records.items():
            for name, r in ops.items():
                cur = rollup.setdefault(
                    name,
                    {"calls": 0, "stages": set(), "verdict": set(), "mps_kernel": r.has_mps_kernel},
                )
                cur["calls"] += r.calls
                cur["stages"].add(st.strip())
                cur["verdict"].add(classify_op(r, fallback_ops, name))
        lines.append("### Per-op rollup (all stages)\n")
        lines.append("| op | calls | verdict | direct_mps_kernel | stages |")
        lines.append("|---|---:|---|:--:|---|")

        def vsort(item):
            order = {"cpu_fallback": 0, "unsupported": 0, "cpu_native": 2, "mps": 3}
            v = sorted(item[1]["verdict"], key=lambda x: order.get(x, 9))[0]
            return (order.get(v, 9), -item[1]["calls"])

        for name, info in sorted(rollup.items(), key=vsort):
            verdict = "/".join(sorted(info["verdict"]))
            stages = ", ".join(sorted(info["stages"]))
            mk = {True: "yes", False: "NO", None: "?"}[info["mps_kernel"]]
            lines.append(
                f"| `{name}` | {info['calls']} | {verdict} | {mk} | {stages} |"
            )
        lines.append("")

        # per-stage fallback summary
        lines.append("### Per-stage summary\n")
        lines.append("| stage | total ops | distinct ops | cpu_fallback ops |")
        lines.append("|---|---:|---:|---|")
        for st in STAGE_ORDER:
            ops = tracer.records.get(st.strip())
            if not ops:
                continue
            fb = sorted(
                name for name, r in ops.items()
                if classify_op(r, fallback_ops, name) in ("cpu_fallback", "unsupported")
            )
            total_calls = sum(r.calls for r in ops.values())
            fb_str = ", ".join(f"`{x}`" for x in fb) if fb else "- (none observed)"
            lines.append(f"| {st} | {total_calls} | {len(ops)} | {fb_str} |")
        lines.append("")

    # static analysis section (always include - it's the methodology anchor)
    lines.append("## Static analysis: Boltz ops with known MPS risk\n")
    lines.append("| stage | op | where | risk |")
    lines.append("|---|---|---|---|")
    for r in KNOWN_MPS_RISK_OPS:
        lines.append(f"| {r['stage']} | `{r['op']}` | {r['where']} | {r['risk']} |")
    lines.append("")

    (out_dir / f"{stem}.report.md").write_text("\n".join(lines))
    print(f"[e2] wrote {out_dir / f'{stem}.report.md'}")
    print(f"[e2] wrote {out_dir / f'{stem}.manifest.json'}")


# ==================================================================================
# Main
# ==================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "real"))
    ap.add_argument("--n-tokens", type=int, default=48)
    ap.add_argument("--n-atoms", type=int, default=256)
    ap.add_argument("--n-msa", type=int, default=16)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="tiny + fast: small tokens/atoms/msa, 0 recycles, 2 sampling steps",
    )
    ap.add_argument(
        "--static-only",
        action="store_true",
        help="skip the live run; emit methodology + static analysis only",
    )
    args = ap.parse_args()

    if args.fast:
        args.n_tokens, args.n_atoms, args.n_msa = 24, 64, 8
        args.recycling_steps, args.sampling_steps = 0, 2

    if args.device == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        print("[e2] WARNING: mps requested but unavailable; falling back to cpu.")
        args.device = "cpu"

    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[
        args.dtype
    ]
    torch.manual_seed(args.seed)

    run_id = "e2_mps_opcoverage_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ckpt_path = Path(os.path.expanduser(args.checkpoint))
    notes: list[str] = []

    # Resolve the BOLTZMAC_OPM mode actually in effect for this run (provenance).
    # Prefer boltz's own resolver; fall back to the raw env var / default 'stock'.
    try:
        from boltz.model.layers.outer_product_mean import resolve_opm_mode

        opm_mode = resolve_opm_mode()
    except Exception:
        opm_mode = (os.environ.get("BOLTZMAC_OPM", "stock") or "stock").strip().lower()

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment": "E2 - MPS op-coverage / device-fallback",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "code_sha": git_sha(REPO_ROOT),
        "boltz_commit": git_sha(REPO_ROOT),
        "hardware": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()} ({platform.platform()})",
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "input_kind": (
            f"SYNTHETIC feats (n_tokens={args.n_tokens}, n_atoms={args.n_atoms}, "
            f"n_msa={args.n_msa}) - real model + real weights, synthetic input"
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
        "weights_file": str(ckpt_path),
        "weights_version": "unknown",
        "weights_sha256": "unknown",
        "ran_on_device": "no",
    }

    out_dir = Path(args.out_dir)

    # ---- weights presence / sha ----
    have_ckpt = ckpt_path.exists()
    if have_ckpt:
        print(f"[e2] hashing checkpoint {ckpt_path} ...")
        manifest["weights_sha256"] = sha256_file(ckpt_path)
        name = ckpt_path.name.lower()
        if "boltz1" in name or "boltz-1" in name:
            manifest["weights_version"] = "Boltz-1 (boltz1_conf.ckpt)"
            notes.append(
                "WEIGHTS CAVEAT: the loaded checkpoint is **Boltz-1**, not Boltz-2. "
                "The op-coverage / fallback map is architecture-driven so this is still "
                "informative, but profiling real Boltz-2 requires Boltz-2 weights."
            )
        elif "boltz2" in name or "boltz-2" in name:
            manifest["weights_version"] = "Boltz-2 (boltz2_conf.ckpt)"
        else:
            manifest["weights_version"] = ckpt_path.name
    else:
        notes.append(
            f"No checkpoint found at {ckpt_path}: emitting methodology + static "
            "analysis only. Re-run with --checkpoint pointing at real weights."
        )

    notes.append(
        "INPUT CAVEAT: synthetic feats (correct keys/shapes/dtypes) are used because "
        "op-coverage is architecture-driven; swap in a real BoltzInferenceDataModule "
        "feats dict for numeric fidelity (no other change needed)."
    )
    notes.append(
        "PYTORCH_ENABLE_MPS_FALLBACK="
        f"{os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')} - unsupported MPS ops fall "
        "back to CPU and emit a UserWarning, which we harvest as the authoritative "
        "fallback-op list."
    )

    if args.static_only or not have_ckpt:
        notes.append(
            "Run mode: STATIC/DRY (no live forward pass)."
            if args.static_only
            else "Run mode: STATIC/DRY (weights missing)."
        )
        write_artifacts(out_dir, manifest, {}, None, set(), True, notes)
        print("[e2] static-only artifacts written. (no live MPS run)")
        return 0

    # ---- load model ----
    print(f"[e2] loading Boltz1 from {ckpt_path} on {device} ...")
    # torch>=2.6 defaults torch.load to weights_only=True; Boltz checkpoints embed an
    # omegaconf hyper-parameters blob. Allowlist those globals (the same set
    # boltz.main registers) so the *local, user-owned* checkpoint deserializes.
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
    try:
        from boltz.model.models.boltz1 import Boltz1

        # The local checkpoint is user-owned/trusted (the same file boltz downloads to
        # ~/.boltz and loads itself), and torch>=2.6 weights_only=True cannot
        # deserialize its embedded hyper-params blob, so load with weights_only=False.
        try:
            model = Boltz1.load_from_checkpoint(
                str(ckpt_path), map_location="cpu", strict=False, weights_only=False
            )
        except TypeError:
            # older Lightning without the weights_only kwarg
            model = Boltz1.load_from_checkpoint(
                str(ckpt_path), map_location="cpu", strict=False
            )
        model = model.to(device)
        model.eval()
    except Exception as exc:
        tb = traceback.format_exc()
        notes.append(f"MODEL LOAD FAILED: {type(exc).__name__}: {exc}")
        notes.append("Falling back to static analysis (model did not load).")
        print(tb, file=sys.stderr)
        write_artifacts(out_dir, manifest, {}, None, set(), True, notes)
        return 0

    # ---- build synthetic feats ----
    feats = build_synthetic_feats(args.n_tokens, args.n_atoms, args.n_msa, device)

    # ---- autocast context ----
    if dtype != torch.float32 and device.type in ("mps", "cuda"):
        autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    tracer = OpCoverageTracer()
    sub_hooks = SubStageHooks(model, tracer)

    # ---- run, capturing MPS fallback warnings ----
    fallback_ops: set[str] = set()
    ran_ok = False
    run_err = None
    timings: dict[str, float] = {}
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            with tracer:
                run_staged_forward(
                    model,
                    feats,
                    device,
                    tracer,
                    recycling_steps=args.recycling_steps,
                    sampling_steps=args.sampling_steps,
                    autocast_ctx=autocast_ctx,
                    timings=timings,
                )
            ran_ok = True
        except Exception as exc:
            run_err = exc
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            sub_hooks.remove()
        # harvest fallback op names from PyTorch warnings
        for w in wlist:
            msg = str(w.message)
            if "not currently supported on the MPS backend" in msg:
                # message contains: The operator 'aten::xxx' is not currently ...
                import re

                m = re.search(r"operator '([^']+)'", msg)
                if m:
                    fallback_ops.add(m.group(1).split(".")[0])

    manifest["ran_on_device"] = (
        "yes" if (ran_ok and device.type == "mps") else
        ("partial" if device.type == "mps" else f"yes ({device.type})")
    )
    if run_err is not None:
        manifest["ran_on_device"] = "partial"
        notes.append(f"RUN ERROR (forward did not fully complete): {type(run_err).__name__}: {run_err}")

    notes.append(f"MPS CPU-fallback ops detected via warnings: {sorted(fallback_ops) or 'none'}")
    write_artifacts(out_dir, manifest, timings, tracer, fallback_ops, False, notes)

    print(f"[e2] DONE. ran_on_device={manifest['ran_on_device']} "
          f"fallback_ops={sorted(fallback_ops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
