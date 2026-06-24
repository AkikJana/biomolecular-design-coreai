import math
import torch
from torch import Tensor, nn

from boltz.model.layers import initialize as init


@torch.compiler.disable
def kernel_triangular_mult(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps,
):
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplicationOutgoing."""

    def __init__(self, dim: int = 128, use_fold_cp: bool = False, num_devices: int = 4) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()
        self.use_fold_cp = use_fold_cp
        self.num_devices = num_devices

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)

        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)

        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        use_kernels: bool
            Whether to use the kernel

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return kernel_triangular_mult(
                x,
                direction="outgoing",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        if self.use_fold_cp:
            P = self.num_devices
            P_row = int(math.sqrt(P))
            while P % P_row != 0:
                P_row -= 1
            P_col = P // P_row
            
            B, N, _, D = a.shape
            # Zero-pad token dims to a multiple of P so the P_row x P_col shard
            # grid divides evenly for arbitrary N. a and b are already masked
            # (zeros at pad positions), so padding contributes nothing to the
            # contraction and is sliced off afterwards -> result is unchanged.
            N_pad = ((N + P - 1) // P) * P
            if N_pad != N:
                pad = (0, 0, 0, N_pad - N, 0, N_pad - N)
                a = torch.nn.functional.pad(a, pad)
                b = torch.nn.functional.pad(b, pad)
            R_shard = N_pad // P_row
            C_shard = N_pad // P_col

            a_shards = a.view(B, P_row, R_shard, P_col, C_shard, D).permute(0, 1, 3, 2, 4, 5)
            b_shards = b.view(B, P_row, R_shard, P_col, C_shard, D).permute(0, 1, 3, 2, 4, 5)
            
            out_shards = []
            for batch_idx in range(B):
                a_s = a_shards[batch_idx]
                b_s = b_shards[batch_idx]
                
                c_out = torch.zeros_like(a_s)
                for step in range(P_row):
                    for r in range(P_row):
                        for c in range(P_col):
                            k = (r + c + step) % P_row
                            a_block = a_s[r, k]
                            b_block = b_s[c, k]
                            
                            a_b = a_block.permute(2, 0, 1)
                            b_b = b_block.permute(2, 0, 1)
                            prod = torch.bmm(a_b, b_b.transpose(-1, -2))
                            c_out[r, c] += prod.permute(1, 2, 0)
                out_shards.append(c_out)
                
            c_stacked = torch.stack(out_shards, dim=0)
            c_p = c_stacked.permute(0, 1, 3, 2, 4, 5)
            x = c_p.reshape(B, N_pad, N_pad, D)
            if N_pad != N:
                x = x[:, :N, :N, :]
        else:
            x = torch.einsum("bikd,bjkd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplicationIncoming."""

    def __init__(self, dim: int = 128, use_fold_cp: bool = False, num_devices: int = 4) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()
        self.use_fold_cp = use_fold_cp
        self.num_devices = num_devices

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)

        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)

        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        use_kernels: bool
            Whether to use the kernel

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return kernel_triangular_mult(
                x,
                direction="incoming",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        if self.use_fold_cp:
            P = self.num_devices
            P_row = int(math.sqrt(P))
            while P % P_row != 0:
                P_row -= 1
            P_col = P // P_row
            
            B, N, _, D = a.shape
            # Zero-pad token dims to a multiple of P so the P_row x P_col shard
            # grid divides evenly for arbitrary N. a and b are already masked
            # (zeros at pad positions), so padding contributes nothing to the
            # contraction and is sliced off afterwards -> result is unchanged.
            N_pad = ((N + P - 1) // P) * P
            if N_pad != N:
                pad = (0, 0, 0, N_pad - N, 0, N_pad - N)
                a = torch.nn.functional.pad(a, pad)
                b = torch.nn.functional.pad(b, pad)
            R_shard = N_pad // P_row
            C_shard = N_pad // P_col

            a_shards = a.view(B, P_row, R_shard, P_col, C_shard, D).permute(0, 1, 3, 2, 4, 5)
            b_shards = b.view(B, P_row, R_shard, P_col, C_shard, D).permute(0, 1, 3, 2, 4, 5)
            
            out_shards = []
            for batch_idx in range(B):
                a_s = a_shards[batch_idx]
                b_s = b_shards[batch_idx]
                
                c_out = torch.zeros_like(a_s)
                for step in range(P_row):
                    for r in range(P_row):
                        for c in range(P_col):
                            k = (r + c + step) % P_row
                            a_block = a_s[k, r]
                            b_block = b_s[k, c]
                            
                            a_b = a_block.permute(2, 0, 1)
                            b_b = b_block.permute(2, 0, 1)
                            prod = torch.bmm(a_b.transpose(-1, -2), b_b)
                            c_out[r, c] += prod.permute(1, 2, 0)
                out_shards.append(c_out)
                
            c_stacked = torch.stack(out_shards, dim=0)
            c_p = c_stacked.permute(0, 1, 3, 2, 4, 5)
            x = c_p.reshape(B, N_pad, N_pad, D)
            if N_pad != N:
                x = x[:, :N, :N, :]
        else:
            x = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x
