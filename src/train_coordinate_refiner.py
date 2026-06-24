"""Supervised training entrypoint for the atom-level CoordinateRefiner.

Trains ``boltz.model.layers.coordinate_refiner.CoordinateRefiner`` to map coarse /
noisy atom coordinates to refined coordinates closer to a ground-truth structure,
using SE(3)-alignment-invariant losses (the same weighted rigid alignment the
diffusion model uses to score structures).

Data abstraction
----------------
``RefinerDataset`` yields dicts with:
    seq_embeddings : (N, token_s)  -- token-level conditioning (s_trunk)
    coarse_coords  : (M, 3)        -- noisy/coarse atom coordinates (model input)
    true_coords    : (M, 3)        -- ground-truth atom coordinates (target)
    atom_mask      : (M,)          -- 1 for real atoms, 0 for padding
with N (tokens) != M (atoms), matching the real sampler hook.

``SyntheticRefinerDataset`` provides a runnable, physically-meaningful denoising
task (a fixed helical backbone corrupted with Gaussian noise + induced clashes)
so the entrypoint is testable today. For real training, swap it for a dataset of
(coarse, ground-truth) atom pairs from Boltz predictions vs experimental
structures; the loop, losses and CLI are unchanged.

Note: the current CoordinateRefiner conditions on a *global* (mean-pooled) token
embedding, so it best fits single-/few-structure refinement. Per-atom
conditioning (via an atom->token map) would be required to refine many distinct
structures from token features alone.
"""

import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

# Use the in-repo modified boltz.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "boltz", "src")))
from boltz.model.layers.coordinate_refiner import CoordinateRefiner  # noqa: E402
from boltz.model.loss.diffusionv2 import weighted_rigid_align  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_helix(num_atoms: int, generator: torch.Generator) -> torch.Tensor:
    """A smooth helical atom trace with ~3.8 A spacing, randomly oriented."""
    coords = torch.zeros(num_atoms, 3)
    cur = torch.zeros(3)
    for i in range(1, num_atoms):
        theta = i * 0.5
        step = torch.tensor(
            [3.8 * math.cos(theta) * 0.8, 3.8 * math.sin(theta) * 0.8, 1.5]
        )
        step = step / step.norm() * 3.8
        cur = cur + step
        coords[i] = cur
    # Random rigid orientation so the task is not axis-aligned.
    a = torch.rand(3, generator=generator) * 2 * math.pi
    rx = torch.tensor([[1, 0, 0], [0, math.cos(a[0]), -math.sin(a[0])], [0, math.sin(a[0]), math.cos(a[0])]])
    return coords @ rx.T


class RefinerDataset(Dataset):
    """Base class. Subclass and return the documented dict per item."""


class SyntheticRefinerDataset(RefinerDataset):
    def __init__(
        self,
        num_examples: int = 64,
        num_atoms: int = 32,
        num_tokens: int = 8,
        token_s: int = 64,
        noise_scale: float = 1.0,
        seed: int = 0,
    ):
        self.items = []
        g = torch.Generator().manual_seed(seed)
        # A single shared ground-truth structure + fixed conditioning: a denoising
        # task that is learnable from the global-conditioning refiner.
        base = make_helix(num_atoms, g)
        seq = torch.randn(num_tokens, token_s, generator=g)
        for _ in range(num_examples):
            noise = torch.randn(num_atoms, 3, generator=g) * noise_scale
            coarse = base + noise
            # Induce a few steric clashes by compressing a contiguous stretch.
            lo = int(torch.randint(0, max(1, num_atoms - 5), (1,), generator=g))
            coarse[lo : lo + 5] = coarse[lo : lo + 5] * 0.3
            self.items.append(
                {
                    "seq_embeddings": seq.clone(),
                    "coarse_coords": coarse,
                    "true_coords": base.clone(),
                    "atom_mask": torch.ones(num_atoms),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    return {
        "seq_embeddings": torch.stack([b["seq_embeddings"] for b in batch]),
        "coarse_coords": torch.stack([b["coarse_coords"] for b in batch]),
        "true_coords": torch.stack([b["true_coords"] for b in batch]),
        "atom_mask": torch.stack([b["atom_mask"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# Losses (SE(3)-alignment invariant)
# --------------------------------------------------------------------------- #
def align_target(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rigidly align the ground truth onto the prediction (detached target).

    Runs on CPU float32 for SVD stability/portability (3x3 SVD is cheap), and is
    detached so gradients flow only through the prediction.
    """
    with torch.no_grad():
        p, t, m = pred.detach().cpu().float(), true.detach().cpu().float(), mask.detach().cpu()
        aligned = weighted_rigid_align(t, p, m.float(), m.bool())
    return aligned.to(pred.device).to(pred.dtype)


def coord_loss(pred, true_aligned, mask):
    se = ((pred - true_aligned) ** 2).sum(dim=-1) * mask  # (B, M)
    return se.sum() / (mask.sum() + 1e-8)


def distance_loss(pred, true, mask):
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, M, M)
    dp = torch.cdist(pred, pred)
    dt = torch.cdist(true, true)
    return (((dp - dt) ** 2) * pair_mask).sum() / (pair_mask.sum() + 1e-8)


def aligned_rmsd(pred, true, mask):
    true_aligned = align_target(pred, true, mask)
    se = ((pred - true_aligned) ** 2).sum(dim=-1) * mask
    return torch.sqrt(se.sum() / (mask.sum() + 1e-8)).item()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_refiner(
    epochs: int = 50,
    batch_size: int = 8,
    num_examples: int = 64,
    num_atoms: int = 32,
    num_tokens: int = 8,
    token_s: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 3,
    lr: float = 1e-3,
    dist_weight: float = 1.0,
    device: str = None,
    ckpt_path: str = "coordinate_refiner.pt",
    seed: int = 0,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    dev = pick_device(device)
    if verbose:
        print(f"[refiner-train] device={dev}")

    dataset = SyntheticRefinerDataset(
        num_examples=num_examples, num_atoms=num_atoms,
        num_tokens=num_tokens, token_s=token_s, seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = CoordinateRefiner(token_s=token_s, hidden_dim=hidden_dim, num_layers=num_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Baseline RMSD of the coarse input (identity refiner) vs ground truth.
    b0 = collate([dataset[i] for i in range(len(dataset))])
    initial_rmsd = aligned_rmsd(
        b0["coarse_coords"].to(dev), b0["true_coords"].to(dev), b0["atom_mask"].to(dev)
    )
    if verbose:
        print(f"[refiner-train] initial coarse RMSD: {initial_rmsd:.4f} A")

    final_rmsd = initial_rmsd
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            seq = batch["seq_embeddings"].to(dev)
            coarse = batch["coarse_coords"].to(dev)
            true = batch["true_coords"].to(dev)
            mask = batch["atom_mask"].to(dev)

            pred = model(seq, coarse)
            true_aln = align_target(pred, true, mask)
            loss = coord_loss(pred, true_aln, mask) + dist_weight * distance_loss(pred, true, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            model.eval()
            with torch.no_grad():
                pred_all = model(b0["seq_embeddings"].to(dev), b0["coarse_coords"].to(dev))
            final_rmsd = aligned_rmsd(pred_all, b0["true_coords"].to(dev), b0["atom_mask"].to(dev))
            print(f"  epoch {epoch:3d} | loss {running/len(loader):.4f} | refined RMSD {final_rmsd:.4f} A")

    # Final evaluation + checkpoint.
    model.eval()
    with torch.no_grad():
        pred_all = model(b0["seq_embeddings"].to(dev), b0["coarse_coords"].to(dev))
    final_rmsd = aligned_rmsd(pred_all, b0["true_coords"].to(dev), b0["atom_mask"].to(dev))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"token_s": token_s, "hidden_dim": hidden_dim, "num_layers": num_layers},
        },
        ckpt_path,
    )
    if verbose:
        print(f"[refiner-train] final refined RMSD: {final_rmsd:.4f} A "
              f"(improvement {initial_rmsd - final_rmsd:+.4f} A)")
        print(f"[refiner-train] checkpoint saved -> {ckpt_path}")

    return {"initial_rmsd": initial_rmsd, "final_rmsd": final_rmsd, "ckpt_path": ckpt_path}


def main():
    p = argparse.ArgumentParser(description="Supervised training for the CoordinateRefiner")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-examples", type=int, default=64)
    p.add_argument("--num-atoms", type=int, default=32)
    p.add_argument("--num-tokens", type=int, default=8)
    p.add_argument("--token-s", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dist-weight", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckpt-path", type=str, default="coordinate_refiner.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    train_refiner(
        epochs=args.epochs, batch_size=args.batch_size, num_examples=args.num_examples,
        num_atoms=args.num_atoms, num_tokens=args.num_tokens, token_s=args.token_s,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, lr=args.lr,
        dist_weight=args.dist_weight, device=args.device, ckpt_path=args.ckpt_path,
        seed=args.seed,
    )
    print(f"[refiner-train] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
