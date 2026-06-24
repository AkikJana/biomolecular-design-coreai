import torch
from torch import Tensor, nn


class CoordinateRefiner(nn.Module):
    """Atom-level coordinate refiner for post-diffusion structure cleanup.

    Matches the ``coordinate_refiner(seq_embeddings, atom_coords)`` hook in
    ``AtomDiffusion.sample`` and predicts a residual delta on the final atom
    coordinates to reduce clashes / bond-length errors.

    Interface contract (from the sampler hook):
        seq_embeddings : (B, N, token_s)  -- token-level trunk single rep
        atom_coords    : (B', M, 3)       -- atom-level coordinates
    where N (tokens) != M (atoms), and B' = B * multiplicity (diffusion samples).

    The token/atom cardinality mismatch is handled by mean-pooling the token
    embeddings into a global conditioning vector and broadcasting it across all
    atoms; per-atom geometry enters through ``coord_proj``. The diffusion
    ``multiplicity`` is handled by repeat-interleaving the pooled conditioning
    up to the atom-batch size.

    The final delta projection is zero-initialized, so an untrained refiner is
    exactly the identity (``return atom_coords``). This makes ``refine_coords=True``
    safe to enable before the refiner has been trained -- it degrades to a no-op
    rather than corrupting the predicted structure.
    """

    def __init__(self, token_s: int, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.token_s = token_s
        self.hidden_dim = hidden_dim

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

        self.delta = nn.Linear(hidden_dim, 3)
        # Zero-init -> identity at initialization (safe no-op before training).
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, seq_embeddings: Tensor, atom_coords: Tensor) -> Tensor:
        """Refine atom coordinates with a residual delta.

        Parameters
        ----------
        seq_embeddings : Tensor
            Token-level conditioning of shape (B, N, token_s).
        atom_coords : Tensor
            Atom coordinates of shape (B', M, 3), B' a multiple of B.

        Returns
        -------
        Tensor
            Refined coordinates of shape (B', M, 3).
        """
        # Global conditioning from token embeddings (handles N != M).
        cond = self.cond_proj(seq_embeddings.mean(dim=1))  # (B, hidden)

        # Handle diffusion multiplicity: atom batch B' may be a multiple of B.
        if cond.shape[0] != atom_coords.shape[0]:
            mult = atom_coords.shape[0] // cond.shape[0]
            cond = cond.repeat_interleave(mult, dim=0)

        h = self.coord_proj(atom_coords) + cond.unsqueeze(1)  # (B', M, hidden)
        for block in self.blocks:
            h = h + block(h)

        delta = self.delta(h)  # (B', M, 3), zero at init
        return atom_coords + delta


def load_coordinate_refiner(
    checkpoint_path: str, token_s: int, **overrides
) -> "CoordinateRefiner":
    """Build a CoordinateRefiner and load trained weights from a checkpoint.

    Accepts either a dict saved as ``{"state_dict": ..., "config": {...}}`` (the
    format written by ``src/train_coordinate_refiner.py``) or a raw state_dict.
    The checkpoint ``config`` defines the architecture (hidden_dim, num_layers);
    ``token_s`` must match the surrounding model and any ``overrides`` win last.
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
            f"CoordinateRefiner checkpoint token_s ({ckpt_token_s}) does not match "
            f"model token_s ({token_s})."
        )
    config.update(overrides)

    refiner = CoordinateRefiner(token_s=token_s, **config)
    refiner.load_state_dict(state_dict, strict=True)
    return refiner
