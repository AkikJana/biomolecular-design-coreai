import math

import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for scalar inputs (timestep, guidance scale)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device) / max(half - 1, 1)
        )
        emb = x[:, None] * freqs[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class CFGDistilledStudent(nn.Module):
    """Single-pass distilled student vector field for the diffusion sampler.

    Matches the ``student_model(atom_coords, t, c, s)`` hook in
    ``AtomDiffusion.sample``: predicts a vector field ``v`` that the sampler turns
    into a denoised x0 estimate via ``denoised = atom_coords + (1 - t) * v``. A
    trained student replaces the iterative classifier-free-guided teacher with one
    forward pass at a chosen guidance scale ``s``.

    Interface contract (from the sampler hook):
        atom_coords : (B, M, 3)        -- noisy atom coordinates
        t           : (B,) | (B, 1)    -- normalized timestep in [0, 1]
        c           : (B, N, token_s)  -- token-level conditioning (s_trunk)
        s           : (B,) | (B, 1)    -- guidance scale
        returns     : (B, M, 3)        -- predicted vector field v

    The token/atom cardinality mismatch (N tokens != M atoms) is handled by
    mean-pooling the token conditioning into a global vector broadcast over atoms;
    per-atom geometry enters through ``coord_proj``. A batch mismatch between the
    conditioning and the coordinates (diffusion multiplicity) is handled by
    repeat-interleaving the pooled conditioning up to the coordinate batch.

    Unlike the coordinate refiner, there is no safe identity initialization: a
    student is the denoiser, so it must be trained/distilled before use -- an
    untrained student produces noise.
    """

    def __init__(
        self,
        token_s: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.token_s = token_s
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.time_emb = SinusoidalEmbedding(time_dim)
        self.scale_emb = SinusoidalEmbedding(time_dim)
        self.ts_mlp = nn.Sequential(
            nn.Linear(time_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_proj = nn.Linear(token_s, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)

        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )

        self.out_head = nn.Linear(hidden_dim, 3)

    def forward(self, atom_coords: Tensor, t: Tensor, c: Tensor, s: Tensor) -> Tensor:
        B = atom_coords.shape[0]

        # Timestep + guidance-scale conditioning.
        ts = self.ts_mlp(torch.cat([self.time_emb(t), self.scale_emb(s)], dim=-1))  # (B, H)

        # Global token conditioning (handles N != M).
        cond = self.cond_proj(c.mean(dim=1))  # (B_c, H)
        if cond.shape[0] != B:
            cond = cond.repeat_interleave(B // cond.shape[0], dim=0)
        if ts.shape[0] != B:
            ts = ts.repeat_interleave(B // ts.shape[0], dim=0)

        h = self.coord_proj(atom_coords) + cond.unsqueeze(1) + ts.unsqueeze(1)  # (B, M, H)
        for block in self.blocks:
            h = h + block(h)
        return self.out_head(h)  # (B, M, 3)


def load_cfg_student(checkpoint_path: str, token_s: int, **overrides) -> "CFGDistilledStudent":
    """Build a CFGDistilledStudent and load trained weights from a checkpoint.

    Accepts either ``{"state_dict": ..., "config": {...}}`` or a raw state_dict.
    The checkpoint ``config`` defines the architecture; ``token_s`` must match the
    surrounding model and any ``overrides`` win last.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config = dict(ckpt.get("config", {}))
    else:
        state_dict = ckpt
        config = {}

    ckpt_token_s = config.pop("token_s", token_s)
    if ckpt_token_s != token_s:
        raise ValueError(
            f"CFGDistilledStudent checkpoint token_s ({ckpt_token_s}) does not match "
            f"model token_s ({token_s})."
        )
    config.update(overrides)

    student = CFGDistilledStudent(token_s=token_s, **config)
    student.load_state_dict(state_dict, strict=True)
    return student
