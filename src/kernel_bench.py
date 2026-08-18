"""A certificate-style bench for MPS operator rewrites.

Section 7.15 measured what happens when a search optimises a noisy statistical
objective: it found a readout combination beating the baseline that both a
permutation null and a held-out panel refused. The lesson was structural --
automated search pays where a candidate can be verified independently of the
data that produced it.

Operator rewrites have exactly that property, and this module supplies it. A
candidate implementation is accepted only if it is numerically equivalent to the
reference on random inputs, and is then scored by measured latency. Equivalence
is checked, not assumed; latency is measured with the device synchronised, not
inferred from wall time.

Profiling a full-settings fold (200 sampling steps, 3 recycling) puts the cost
here:

    AtomDiffusion sampling loop     52.8% of wall
      DiffusionTransformerLayer     47.6%   6,006 calls
        AttentionPairBias           18.9%   6,246 calls   3.7 ms
        ConditionedTransitionBlock  17.8%   6,206 calls   3.5 ms
        AdaLN                       11.4%  12,212 calls   1.1 ms
    TriangleAttention (both arms)    6.3%     520 calls  14.8 ms

At (28, 32, 128) an AdaLN is a handful of microseconds of arithmetic, so 1.1 ms
is kernel launch overhead. That is what a rewrite has to attack, and why fusing
launches matters more here than reducing FLOPs.

Usage:
    python src/kernel_bench.py --op adaln
"""

import argparse
import statistics
import time

import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Captured from a real fold; a benchmark at the wrong shape optimises the wrong
# regime, and these operators are launch-bound precisely because they are small.
SHAPES = {"a": (28, 32, 128), "s": (28, 32, 128), "z": (28, 32, 128, 16)}
DIM = 128


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def latency(fn, args, warmup=20, iters=200):
    """Amortised per-call latency in milliseconds, and the spread over repeats.

    Iterations run back to back with a single synchronise at the end, because
    synchronising around each call measures the wrong thing entirely. Forcing a
    command-buffer flush costs ~0.23 ms on this device while each additional
    queued kernel costs ~0.04 ms, so a per-call sync charges every operator a
    fixed 0.23 ms that it does not pay inside the model, where work is queued and
    batched. Measured that way an AdaLN looked like 0.5 ms and every rewrite of
    it looked slower -- an artefact of the harness, not a property of the code.
    """
    for _ in range(warmup):
        fn(*args)
    sync()
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        sync()
        reps.append((time.perf_counter() - t0) * 1000 / iters)
    reps.sort()
    return reps[len(reps) // 2], reps[-1] - reps[0]


def equivalent(ref, cand, args, atol=1e-5, rtol=1e-4, trials=5):
    """Is the candidate the same function as the reference?

    Several random draws rather than one: a rewrite can agree on a particular
    input by coincidence, and an accepted rewrite is going into every fold.
    Returns (ok, worst_absolute_difference).
    """
    worst = 0.0
    with torch.no_grad():
        for _ in range(trials):
            fresh = [torch.randn_like(a) if torch.is_tensor(a) else a
                     for a in args]
            r, c = ref(*fresh), cand(*fresh)
            if r.shape != c.shape:
                return False, float("inf")
            worst = max(worst, float((r - c).abs().max()))
            if not torch.allclose(r, c, atol=atol, rtol=rtol):
                return False, worst
    return True, worst


def report(name, ref, candidates, args):
    base, base_iqr = latency(ref, args)
    print(f"\n{name}  inputs {[tuple(a.shape) for a in args if torch.is_tensor(a)]}")
    print(f"  {'implementation':32}{'ms':>9}{'IQR':>8}{'speedup':>9}"
          f"{'max|diff|':>12}  verdict")
    print(f"  {'reference':32}{base:>9.3f}{base_iqr:>8.3f}{'1.00x':>9}"
          f"{'—':>12}  —")
    rows = []
    for label, fn in candidates:
        ok, worst = equivalent(ref, fn, args)
        ms, iqr = latency(fn, args)
        verdict = "accepted" if ok else "REJECTED: not equivalent"
        rows.append({"impl": label, "ms": ms, "speedup": base / ms,
                     "max_diff": worst, "equivalent": ok})
        print(f"  {label:32}{ms:>9.3f}{iqr:>8.3f}{base / ms:>8.2f}x"
              f"{worst:>12.2e}  {verdict}")
    return {"reference_ms": base, "candidates": rows}


# ------------------------------------------------------------------ AdaLN

def build_adaln():
    """Reference AdaLN and rewrites of it, all on the same weights.

    A rewrite that trains its own weights would be a different function; every
    candidate below is handed the reference's parameters so that any difference
    in output is a bug rather than a different model.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boltz" / "src"))
    from boltz.model.modules.transformers import AdaLN

    ref = AdaLN(DIM, DIM).to(DEVICE).eval()
    a = torch.randn(*SHAPES["a"], device=DEVICE)
    s = torch.randn(*SHAPES["s"], device=DEVICE)

    def reference(a, s):
        with torch.no_grad():
            return ref(a, s)

    # The two projections read the same normalised `s`, so their weights can be
    # stacked into one matmul and the result split. Exactly equivalent: the bias
    # of the second half is zero because s_bias carries none.
    W = torch.cat([ref.s_scale.weight, ref.s_bias.weight], dim=0).contiguous()
    b = torch.cat([ref.s_scale.bias,
                   torch.zeros_like(ref.s_scale.bias)], dim=0).contiguous()

    def fused_proj(a, s):
        with torch.no_grad():
            a = ref.a_norm(a)
            s = ref.s_norm(s)
            sb = torch.nn.functional.linear(s, W, b)
            scale, bias = sb.chunk(2, dim=-1)
            return torch.sigmoid(scale) * a + bias

    def fused_proj_addcmul(a, s):
        with torch.no_grad():
            a = ref.a_norm(a)
            s = ref.s_norm(s)
            sb = torch.nn.functional.linear(s, W, b)
            scale, bias = sb.chunk(2, dim=-1)
            return torch.addcmul(bias, torch.sigmoid(scale), a)

    # The two changes are separated. Stacking the projections and fusing the
    # scale-multiply-add are independent, and stacking on its own measured
    # slower, so reporting them together would credit the wrong one.
    def addcmul_only(a, s):
        with torch.no_grad():
            a = ref.a_norm(a)
            s = ref.s_norm(s)
            return torch.addcmul(ref.s_bias(s), torch.sigmoid(ref.s_scale(s)), a)

    cands = [("stacked projection only", fused_proj),
             ("addcmul only", addcmul_only),
             ("stacked + addcmul", fused_proj_addcmul)]

    # aot_eager only traces; inductor is the backend that actually fuses
    # elementwise chains into one kernel, which is what a launch-bound operator
    # needs. Both are tried because MPS support differs between them.
    for label, kw in (("torch.compile aot_eager", {"backend": "aot_eager"}),
                      ("torch.compile inductor", {}),
                      ("torch.compile inductor/rf",
                       {"mode": "reduce-overhead"})):
        try:
            fn = torch.compile(reference, **kw)
            fn(a, s)
            cands.append((label, fn))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ({label} unavailable: {str(exc)[:70]})")

    return reference, cands, (a, s)


def build_residual_gate():
    """DiffusionTransformerLayer's `a = a + output_projection(s) * b`.

    The same shape of expression as AdaLN's, at 6,006 calls a fold rather than
    12,212, and the same candidate fusion.
    """
    proj = torch.nn.Sequential(torch.nn.Linear(DIM, DIM),
                               torch.nn.Sigmoid()).to(DEVICE).eval()
    a = torch.randn(*SHAPES["a"], device=DEVICE)
    b = torch.randn(*SHAPES["a"], device=DEVICE)
    s = torch.randn(*SHAPES["s"], device=DEVICE)

    def reference(a, b, s):
        with torch.no_grad():
            return a + proj(s) * b

    def fused(a, b, s):
        with torch.no_grad():
            return torch.addcmul(a, proj(s), b)

    return reference, [("addcmul", fused)], (a, b, s)


def build_cond_transition():
    """ConditionedTransitionBlock's two projections of the same `a`.

    swish_gate's linear and a_to_b both read `a`, so their weights can be
    stacked into one matmul. Stacking measured *slower* for AdaLN, so this is
    tested rather than assumed -- the shapes here are four times larger and the
    tradeoff need not go the same way.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "boltz" / "src"))
    from boltz.model.modules.transformers import ConditionedTransitionBlock

    blk = ConditionedTransitionBlock(DIM, DIM).to(DEVICE).eval()
    a = torch.randn(*SHAPES["a"], device=DEVICE)
    s = torch.randn(*SHAPES["s"], device=DEVICE)

    def reference(a, s):
        with torch.no_grad():
            return blk(a, s)

    Wg = blk.swish_gate[0].weight
    Wb = blk.a_to_b.weight
    W = torch.cat([Wg, Wb], dim=0).contiguous()
    n_gate = Wg.shape[0]

    def stacked(a, s):
        with torch.no_grad():
            x = blk.adaln(a, s)
            both = torch.nn.functional.linear(x, W)
            gate, tob = both[..., :n_gate], both[..., n_gate:]
            from boltz.model.modules.transformers import SwiGLU
            b = SwiGLU()(gate) * tob
            return blk.output_projection(s) * blk.b_to_a(b)

    return reference, [("stacked projections", stacked)], (a, s)


OPS = {"adaln": build_adaln,
       "residual_gate": build_residual_gate,
       "cond_transition": build_cond_transition}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="adaln", choices=sorted(OPS))
    args = ap.parse_args()
    print(f"device {DEVICE} | torch {torch.__version__}")
    ref, cands, inputs = OPS[args.op]()
    report(args.op, ref, cands, inputs)


if __name__ == "__main__":
    main()
