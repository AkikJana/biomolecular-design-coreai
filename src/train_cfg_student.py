"""CFG-distillation training entrypoint for the single-pass CFGDistilledStudent.

Classifier-free guidance (CFG) normally needs two teacher passes per step:
    v_guided(x, t, c, s) = v_cond(x, t, c) + s * (v_cond(x, t, c) - v_uncond(x, t))
Distillation trains the student to reproduce ``v_guided`` in ONE pass for a range
of guidance scales ``s``, so inference drops from two evaluations per step (plus
the iterative teacher) to a single student call.

Teacher abstraction
-------------------
``Teacher.guided(x, t, c, s) -> v`` is all the trainer needs. ``SyntheticTeacher``
provides a runnable analytic teacher (a guidance-scaled flow toward a fixed target
structure) so this entrypoint trains and tests today.

For real distillation, wrap the Boltz diffusion model:
    v_cond   = (denoised_cond   - x) / (1 - t)     # from preconditioned_network_forward
    v_uncond = (denoised_uncond - x) / (1 - t)     # same, with conditioning dropped
    v_guided = v_cond + s * (v_cond - v_uncond)
i.e. run AtomDiffusion.preconditioned_network_forward twice (conditioned and
unconditioned) and convert the denoised x0 estimates to velocities. The loop,
losses, checkpoint format and CLI here are unchanged.
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "boltz", "src")))
from boltz.model.layers.cfg_student import CFGDistilledStudent  # noqa: E402


# --------------------------------------------------------------------------- #
# Target structure + teacher
# --------------------------------------------------------------------------- #
def make_helix(num_atoms: int, generator: torch.Generator) -> torch.Tensor:
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
    a = torch.rand(3, generator=generator) * 2 * math.pi
    rx = torch.tensor([[1, 0, 0], [0, math.cos(a[0]), -math.sin(a[0])], [0, math.sin(a[0]), math.cos(a[0])]])
    return coords @ rx.T


class Teacher:
    """Interface: return the CFG-guided vector field at guidance scale s."""

    def guided(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SyntheticTeacher(Teacher):
    """Analytic teacher: conditional flow toward ``target``, unconditional flow
    toward the structure centroid. The guided field is

        v_guided = (target - x) + s * (target - centroid)

    which depends on x, the guidance scale s, and (through target/centroid) the
    conditioning -- a learnable single-pass distillation target.
    """

    def __init__(self, target: torch.Tensor):
        self.target = target  # (M, 3)
        self.centroid = target.mean(dim=0, keepdim=True)  # (1, 3)

    def guided(self, x, t, c, s):
        v_cond = self.target.unsqueeze(0) - x          # (B, M, 3)
        v_uncond = self.centroid.unsqueeze(0) - x      # (B, M, 3)
        return v_cond + s.view(-1, 1, 1) * (v_cond - v_uncond)


# --------------------------------------------------------------------------- #
# Batches, losses, metrics
# --------------------------------------------------------------------------- #
def sample_batch(base, seq, batch_size, s_min, s_max, noise_scale, dev, gen):
    M = base.shape[0]
    N, token_s = seq.shape
    x = base.unsqueeze(0).expand(batch_size, M, 3) + noise_scale * torch.randn(
        batch_size, M, 3, generator=gen
    )
    t = torch.rand(batch_size, generator=gen)
    s = torch.rand(batch_size, generator=gen) * (s_max - s_min) + s_min
    c = seq.unsqueeze(0).expand(batch_size, N, token_s)
    mask = torch.ones(batch_size, M)
    # Order matches the (x, t, c, s) signature of student/teacher; mask last.
    return (x.to(dev), t.to(dev), c.to(dev), s.to(dev), mask.to(dev))


def masked_vfield_mse(v_pred, v_true, mask):
    se = ((v_pred - v_true) ** 2).sum(dim=-1) * mask  # (B, M)
    return se.sum() / (mask.sum() + 1e-8)


def vfield_rmse(v_pred, v_true, mask):
    return math.sqrt(masked_vfield_mse(v_pred, v_true, mask).item())


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


def train_cfg_student(
    epochs: int = 50,
    steps_per_epoch: int = 20,
    batch_size: int = 16,
    num_atoms: int = 32,
    num_tokens: int = 8,
    token_s: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 3,
    time_dim: int = 64,
    lr: float = 1e-3,
    s_min: float = 0.0,
    s_max: float = 4.0,
    noise_scale: float = 1.0,
    device: str = None,
    ckpt_path: str = "cfg_student.pt",
    seed: int = 0,
    verbose: bool = True,
):
    torch.manual_seed(seed)
    dev = pick_device(device)
    gen = torch.Generator().manual_seed(seed)
    if verbose:
        print(f"[cfg-distill] device={dev}")

    base = make_helix(num_atoms, gen).to(dev)
    seq = torch.randn(num_tokens, token_s, generator=gen).to(dev)
    teacher = SyntheticTeacher(base)

    student = CFGDistilledStudent(
        token_s=token_s, hidden_dim=hidden_dim, num_layers=num_layers, time_dim=time_dim
    ).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)

    # Fixed eval batch for a stable distillation-quality metric.
    eval_b = sample_batch(base, seq, 64, s_min, s_max, noise_scale, dev, torch.Generator().manual_seed(123))
    with torch.no_grad():
        v_eval_true = teacher.guided(*eval_b[:4])
        initial_rmse = vfield_rmse(student(*eval_b[:4]), v_eval_true, eval_b[4])
    if verbose:
        print(f"[cfg-distill] initial vector-field RMSE: {initial_rmse:.4f}")

    final_rmse = initial_rmse
    for epoch in range(1, epochs + 1):
        student.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            x, t, c, s, mask = sample_batch(base, seq, batch_size, s_min, s_max, noise_scale, dev, gen)
            with torch.no_grad():
                v_true = teacher.guided(x, t, c, s)
            loss = masked_vfield_mse(student(x, t, c, s), v_true, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            student.eval()
            with torch.no_grad():
                final_rmse = vfield_rmse(student(*eval_b[:4]), v_eval_true, eval_b[4])
            print(f"  epoch {epoch:3d} | loss {running/steps_per_epoch:.4f} | eval vfield RMSE {final_rmse:.4f}")

    student.eval()
    with torch.no_grad():
        final_rmse = vfield_rmse(student(*eval_b[:4]), v_eval_true, eval_b[4])

    torch.save(
        {
            "state_dict": student.state_dict(),
            "config": {
                "token_s": token_s, "hidden_dim": hidden_dim,
                "num_layers": num_layers, "time_dim": time_dim,
            },
        },
        ckpt_path,
    )
    if verbose:
        print(f"[cfg-distill] final vector-field RMSE: {final_rmse:.4f} "
              f"(improvement {initial_rmse - final_rmse:+.4f})")
        print(f"[cfg-distill] checkpoint saved -> {ckpt_path}")

    return {"initial_rmse": initial_rmse, "final_rmse": final_rmse, "ckpt_path": ckpt_path}


def main():
    p = argparse.ArgumentParser(description="CFG distillation training for CFGDistilledStudent")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-atoms", type=int, default=32)
    p.add_argument("--num-tokens", type=int, default=8)
    p.add_argument("--token-s", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--s-min", type=float, default=0.0)
    p.add_argument("--s-max", type=float, default=4.0)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckpt-path", type=str, default="cfg_student.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    train_cfg_student(
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
        num_atoms=args.num_atoms, num_tokens=args.num_tokens, token_s=args.token_s,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, lr=args.lr,
        s_min=args.s_min, s_max=args.s_max, noise_scale=args.noise_scale,
        device=args.device, ckpt_path=args.ckpt_path, seed=args.seed,
    )
    print(f"[cfg-distill] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
