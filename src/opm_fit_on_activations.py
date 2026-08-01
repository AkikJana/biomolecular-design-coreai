"""Best achievable low-rank OPM error on the real input distribution.

`opm_cp_projection.py` showed the stock OPM weight tensor is not rank-32
compressible: CP error 0.83, output error 0.77 on `torch.randn` inputs. That is a
statement about the *weights*, evaluated off-distribution.

The question that actually decides whether the memory saving is reachable on
pretrained models is different: how well can the low-rank form match the stock
layer **on the activations Boltz really produces**? Post-LayerNorm MSA
representations are strongly correlated and occupy a small part of R^64, so the
function can be far easier to approximate than the weight tensor implies.

This fits Px, Py, W, bias directly against captured stock outputs, which is the
distillation objective itself rather than a tensor-approximation proxy. It is
therefore an upper bound on what distillation could achieve for this layer.

  out_stock[i,j,c] = proj_o( sum_s a_i (x) b_j / num )[c]
  out_low  [i,j,c] = sum_r W[c,r] (sum_s x_i[r] y_j[r]) / num + bias[c]

Usage:
    python src/opm_fit_on_activations.py --rank 32 --steps 1500
"""

import argparse
import json
import pickle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def stock_output(sample):
    """Recompute the stock output from stored weights (sanity check on capture)."""
    m, mask = sample["m_norm"], sample["mask"]
    if mask.dim() == m.dim() - 1:          # (B,S,N) -> (B,S,N,1), as the layer does
        mask = mask.unsqueeze(-1)
    a = m @ sample["proj_a"].T * mask
    b = m @ sample["proj_b"].T * mask
    pm = mask[:, :, None, :] * mask[:, :, :, None]
    num = pm.sum(1).clamp(min=1)
    z = torch.einsum("bsic,bsjd->bijcd", a, b)
    z = z.reshape(*z.shape[:3], -1) / num
    return z @ sample["proj_o_w"].T + sample["proj_o_b"]


def low_rank_output(sample, px, py, w, bias):
    m, mask = sample["m_norm"], sample["mask"]
    if mask.dim() == m.dim() - 1:
        mask = mask.unsqueeze(-1)
    x = m @ px.T * mask
    y = m @ py.T * mask
    pm = mask[:, :, None, :] * mask[:, :, :, None]
    num = pm.sum(1).clamp(min=1)
    g = torch.einsum("bsir,bsjr->bijr", x, y)
    return (g @ w.T) / num + bias


def rel_err(pred, target):
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()


def fit(train, rank, steps, lr, seed=0, holdout=None, log=None):
    c_in = train[0]["m_norm"].shape[-1]
    c_out = train[0]["out"].shape[-1]
    g = torch.Generator().manual_seed(seed)
    px = (torch.randn(rank, c_in, generator=g) * 0.1).requires_grad_(True)
    py = (torch.randn(rank, c_in, generator=g) * 0.1).requires_grad_(True)
    w = (torch.randn(c_out, rank, generator=g) * 0.1).requires_grad_(True)
    bias = torch.zeros(c_out, requires_grad=True)
    opt = torch.optim.Adam([px, py, w, bias], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        opt.zero_grad()
        loss = sum(((low_rank_output(s, px, py, w, bias) - s["out"]) ** 2).mean()
                   for s in train) / len(train)
        loss.backward()
        opt.step()
        sched.step()
        if log and (step + 1) % log == 0:
            with torch.no_grad():
                tr = sum(rel_err(low_rank_output(s, px, py, w, bias), s["out"])
                         for s in train) / len(train)
                msg = f"    step {step+1:5d}  train rel err {tr:.4f}"
                if holdout:
                    ho = sum(rel_err(low_rank_output(s, px, py, w, bias), s["out"])
                             for s in holdout) / len(holdout)
                    msg += f"   held-out {ho:.4f}"
                print(msg, flush=True)
    with torch.no_grad():
        tr = sum(rel_err(low_rank_output(s, px, py, w, bias), s["out"])
                 for s in train) / len(train)
        ho = (sum(rel_err(low_rank_output(s, px, py, w, bias), s["out"])
                  for s in holdout) / len(holdout)) if holdout else None
    return tr, ho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=str(REPO_ROOT / "artifacts" / "opm_activations.pkl"))
    ap.add_argument("--holdout", default=None,
                    help="a second activation pickle from a different target")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--layers", default="layer_0,layer_1")
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "opm_fit_on_activations.json"))
    args = ap.parse_args()

    with open(args.activations, "rb") as fh:
        acts = pickle.load(fh)
    hold = None
    if args.holdout:
        with open(args.holdout, "rb") as fh:
            hold = pickle.load(fh)

    results = {}
    for name in args.layers.split(","):
        if name not in acts:
            print(f"  {name}: not captured, skipping")
            continue
        train = acts[name]
        # Confirm the capture reproduces the stock layer before trusting anything.
        recon = rel_err(stock_output(train[0]), train[0]["out"])
        print(f"\n{name}  ({len(train)} sample(s), capture check err {recon:.2e})")
        ho = hold.get(name) if hold else None
        tr, he = fit(train, args.rank, args.steps, args.lr, holdout=ho,
                     log=max(1, args.steps // 5))
        results[name] = {"train_rel_err": tr, "heldout_rel_err": he,
                         "rank": args.rank, "capture_check": recon}
        print(f"  best-achievable rel err on real activations: {tr:.4f}"
              + (f"   (held-out target: {he:.4f})" if he is not None else ""))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\nFor comparison, the same layer on torch.randn inputs with the")
    print("weight-space CP projection scored ~0.77 relative error.")
    print(f"summary: {args.out}")


if __name__ == "__main__":
    main()
