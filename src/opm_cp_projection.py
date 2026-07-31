"""Project stock OuterProductMean weights into the low-rank (S-contracted) form.

The low-rank OPM saves ~93% of activation memory but its parameters do not match
stock Boltz checkpoints, so it currently requires training from scratch. If the
stock weights can be *projected* into the low-rank parameterisation with small
error, the memory win becomes available on pretrained models -- which is the
difference between a research curiosity and a usable result.

The algebra. Stock computes, for output channel e,

    out[i,j,e] = m_i^T (A^T O_e B) m_j ,     A = proj_a, B = proj_b,
                                             O = proj_o reshaped (c_out, c_h, c_h)

and the low-rank form computes

    out[i,j,e] = m_i^T (Px^T diag(W_e) Py) m_j .

Matching them requires writing every O_e with a *shared* set of rank-1 terms:

    O[e,p,q] = sum_r W[e,r] U[p,r] V[q,r]

which is a CP (CANDECOMP/PARAFAC) decomposition of the 3-way tensor O at rank R.
Given that, Px = U^T A and Py = V^T B reproduce the stock layer exactly.

The catch, and the reason this is worth measuring rather than assuming: the
generic CP rank of a 128x32x32 tensor is far above 32, so rank-R=c_hidden cannot
represent an arbitrary O exactly. The low-rank OPM is a genuinely
lower-capacity layer, not a reparameterisation. This script measures how much of
the trained O actually survives the projection.

Usage:
    python src/opm_cp_projection.py --ranks 8,16,32,64,128
"""

import argparse
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = Path.home() / ".boltz" / "boltz1_conf.ckpt"


def khatri_rao(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Column-wise Kronecker product: (I,R),(J,R) -> (I*J,R)."""
    return (a[:, None, :] * b[None, :, :]).reshape(-1, a.shape[1])


def cp_als(tensor: torch.Tensor, rank: int, iters: int = 400, seed: int = 0,
           tol: float = 1e-9):
    """Rank-R CP decomposition of a 3-way tensor by alternating least squares.

    Returns (factors, relative_error). Factors reconstruct as
    ``einsum('ir,jr,kr->ijk', *factors)``.
    """
    g = torch.Generator().manual_seed(seed)
    dims = tensor.shape
    unfold = [tensor.permute(n, *(d for d in range(3) if d != n)).reshape(dims[n], -1)
              for n in range(3)]
    # SVD-based init on each mode: better conditioned than random for real data.
    factors = []
    for n in range(3):
        u, _, _ = torch.linalg.svd(unfold[n].double(), full_matrices=False)
        f = u[:, :rank]
        if f.shape[1] < rank:                      # pad if mode is small
            f = torch.cat([f, torch.randn(dims[n], rank - f.shape[1],
                                          generator=g, dtype=torch.float64)], dim=1)
        factors.append(f.contiguous())

    norm = torch.linalg.norm(tensor.double())
    prev = None
    for _ in range(iters):
        for n in range(3):
            o1, o2 = [d for d in range(3) if d != n]
            kr = khatri_rao(factors[o1], factors[o2])
            gram = (factors[o1].T @ factors[o1]) * (factors[o2].T @ factors[o2])
            factors[n] = unfold[n].double() @ kr @ torch.linalg.pinv(gram)
        approx = torch.einsum("ir,jr,kr->ijk", *factors)
        err = (torch.linalg.norm(tensor.double() - approx) / norm).item()
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err
    return factors, err


def project_layer(state, prefix, rank, iters=400):
    """CP-project one OPM layer. Returns (relative error on O, factors, shapes)."""
    a = state[f"{prefix}.proj_a.weight"].float()          # (c_h, c_in)
    b = state[f"{prefix}.proj_b.weight"].float()
    o = state[f"{prefix}.proj_o.weight"].float()          # (c_out, c_h*c_h)
    c_h, c_in = a.shape
    c_out = o.shape[0]
    o_t = o.reshape(c_out, c_h, c_h)
    (w, u, v), err = cp_als(o_t, rank, iters=iters)
    # Px = U^T A, Py = V^T B  (rank x c_in)
    px = (u.T.float() @ a)
    py = (v.T.float() @ b)
    return err, {"W": w.float(), "Px": px, "Py": py}, (c_in, c_h, c_out)


def bilinear_error(state, prefix, factors):
    """Output-level error on random inputs: stock vs projected low-rank."""
    a = state[f"{prefix}.proj_a.weight"].float()
    b = state[f"{prefix}.proj_b.weight"].float()
    o = state[f"{prefix}.proj_o.weight"].float()
    c_h, c_in = a.shape
    c_out = o.shape[0]
    g = torch.Generator().manual_seed(1)
    mi = torch.randn(256, c_in, generator=g)
    mj = torch.randn(256, c_in, generator=g)
    # stock: sum_{p,q} O[e,p,q] (A m_i)_p (B m_j)_q
    ai, bj = mi @ a.T, mj @ b.T
    stock = torch.einsum("ip,jq,epq->ije", ai, bj, o.reshape(c_out, c_h, c_h))
    xi, yj = mi @ factors["Px"].T, mj @ factors["Py"].T
    low = torch.einsum("ir,jr,er->ije", xi, yj, factors["W"])
    return (torch.linalg.norm(stock - low) / torch.linalg.norm(stock)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--ranks", default="8,16,32,64,128")
    ap.add_argument("--layers", type=int, default=3, help="how many OPM layers to test")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "opm_cp_projection.json"))
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    prefixes = sorted({k.rsplit(".", 2)[0] for k in state
                       if "outer_product_mean" in k and k.endswith("proj_o.weight")})
    prefixes = [p for p in prefixes if "confidence_module" not in p] or prefixes
    prefixes = prefixes[:args.layers]
    ranks = [int(r) for r in args.ranks.split(",")]

    print(f"checkpoint: {args.checkpoint}")
    print(f"testing {len(prefixes)} OPM layers at ranks {ranks}\n")
    results = {}
    for prefix in prefixes:
        print(f"  {prefix}")
        results[prefix] = {}
        for r in ranks:
            err, fac, (c_in, c_h, c_out) = project_layer(state, prefix, r, args.iters)
            out_err = bilinear_error(state, prefix, fac)
            native = c_out * c_h * c_h
            proj = fac["W"].numel() + fac["Px"].numel() + fac["Py"].numel()
            results[prefix][r] = {"cp_rel_error": err, "output_rel_error": out_err,
                                  "params_stock": native, "params_lowrank": proj}
            flag = "  <- rank used by the low-rank OPM" if r == c_h else ""
            print(f"    rank {r:4d}:  CP error {err:6.3f}   output error {out_err:6.3f}"
                  f"   params {proj:6d} vs {native}{flag}")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"summary: {args.out}")
    print("\nRelative error 1.0 means the projection retains nothing; 0.0 is exact.")


if __name__ == "__main__":
    main()
