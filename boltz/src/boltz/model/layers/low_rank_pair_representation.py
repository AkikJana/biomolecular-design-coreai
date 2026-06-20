import math
import torch
import torch.nn as nn

class LowRankTensorProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, W):
        r"""
        Computes the low-rank tensor product update:
        U_{b, i, j, c} = \sum_{r=1}^d X_{b, i, r} Y_{b, j, r} W_{c, r}
        
        Args:
            X: Tensor of shape (B, N, d) - Left sequence project factors
            Y: Tensor of shape (B, N, d) - Right sequence project factors
            W: Tensor of shape (D_pair, d) - Channel factor matrix
            
        Returns:
            U: Tensor of shape (B, N, N, D_pair)
        """
        # Save tensors for backward pass
        ctx.save_for_backward(X, Y, W)
        
        # Vectorized tensor product computation
        # X: (B, N, d), Y: (B, N, d), W: (D_pair, d)
        # Output: (B, N, N, D_pair)
        U = torch.einsum('bix,bjx,cx->bijc', X, Y, W)
        return U

    @staticmethod
    def backward(ctx, grad_output):
        """
        Efficient backward pass that avoids storing O(N^2 * D_pair) intermediate activation tensors.
        
        Args:
            grad_output: Tensor of shape (B, N, N, D_pair) - Gradient of loss w.r.t U
            
        Returns:
            grad_X: Tensor of shape (B, N, d)
            grad_Y: Tensor of shape (B, N, d)
            grad_W: Tensor of shape (D_pair, d)
        """
        X, Y, W = ctx.saved_tensors
        B, N, _, D_pair = grad_output.shape
        rank = W.shape[1]
        
        # Reshape grad_output to perform efficient matrix multiplication
        grad_out_flat = grad_output.reshape(-1, D_pair) # (B*N*N, D_pair)
        
        # Project grad_output back to the low-rank subspace:
        # M = grad_output x W -> shape (B, N, N, rank)
        M_flat = torch.matmul(grad_out_flat, W) # (B*N*N, rank)
        M = M_flat.view(B, N, N, rank)
        
        # Compute grad_X: shape (B, N, rank)
        # grad_X_{b, i, x} = \sum_j M_{b, i, j, x} * Y_{b, j, x}
        grad_X = (M * Y.unsqueeze(1)).sum(dim=2)
        
        # Compute grad_Y: shape (B, N, rank)
        # grad_Y_{b, j, x} = \sum_i M_{b, i, j, x} * X_{b, i, x}
        grad_Y = (M * X.unsqueeze(2)).sum(dim=1)
        
        # Compute grad_W: shape (D_pair, rank)
        # grad_W_{c, x} = \sum_{b, i, j} grad_output_{b, i, j, c} * X_{b, i, x} * Y_{b, j, x}
        # We can construct the low-rank outer product tensor Z_{b, i, j, x} = X_{b, i, x} * Y_{b, j, x}
        # to write this as a matrix product: grad_W = grad_output^T * Z
        Z_flat = (X.unsqueeze(2) * Y.unsqueeze(1)).view(-1, rank) # (B*N*N, rank)
        grad_W = torch.matmul(grad_out_flat.t(), Z_flat) # (D_pair, rank)
        
        return grad_X, grad_Y, grad_W


class LowRankPairUpdater(nn.Module):
    """
    LowRankPairUpdater implements a memory-efficient low-rank tensor factorization update
    for sequence-to-pair representation updates, avoiding O(N^2 * D_pair) activation storage.
    
    Instead of full-rank Outer Product Mean (e.g. projecting sequence embeddings to d_mid
    and computing O = A (x) B of shape (B, N, N, d_mid), then projecting to D_pair),
    this module projects sequence representations to a smaller rank, and uses a custom autograd
    function to perform backpropagation without storing large quadratic-sized intermediate tensors.
    """
    def __init__(self, d_seq, d_pair, rank=16, use_bias=True):
        super().__init__()
        self.d_seq = d_seq
        self.d_pair = d_pair
        self.rank = rank
        
        # Project sequence embeddings to the low-rank latent dimensions
        self.proj_x = nn.Linear(d_seq, rank, bias=use_bias)
        self.proj_y = nn.Linear(d_seq, rank, bias=use_bias)
        
        # Learnable channel projection factor (D_pair, rank)
        self.W = nn.Parameter(torch.empty(d_pair, rank))
        self.reset_parameters()
        
    def reset_parameters(self):
        # Xavier initialization is suitable for projection layers
        nn.init.xavier_uniform_(self.proj_x.weight)
        if self.proj_x.bias is not None:
            nn.init.zeros_(self.proj_x.bias)
            
        nn.init.xavier_uniform_(self.proj_y.weight)
        if self.proj_y.bias is not None:
            nn.init.zeros_(self.proj_y.bias)
            
        # Initialize the channel weight matrix
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, s, pair_rep=None):
        """
        Args:
            s: Sequence embeddings of shape (B, N, D_seq)
            pair_rep: Optional existing pair representations of shape (B, N, N, D_pair)
            
        Returns:
            Updated pair representation of shape (B, N, N, D_pair)
        """
        # Project single representations to the low-rank factors
        X = self.proj_x(s) # (B, N, rank)
        Y = self.proj_y(s) # (B, N, rank)
        
        # Apply low-rank tensor product update using the custom autograd function
        update = LowRankTensorProduct.apply(X, Y, self.W)
        
        if pair_rep is not None:
            return pair_rep + update
        return update


class FullRankPairUpdater(nn.Module):
    """
    FullRankPairUpdater represents the standard Outer Product Mean (OPM) module
    commonly used in AlphaFold-like architectures for baseline comparison.
    """
    def __init__(self, d_seq, d_pair, d_mid=32, use_bias=True):
        super().__init__()
        self.d_seq = d_seq
        self.d_pair = d_pair
        self.d_mid = d_mid
        
        self.proj_a = nn.Linear(d_seq, d_mid, bias=use_bias)
        self.proj_b = nn.Linear(d_seq, d_mid, bias=use_bias)
        self.proj_out = nn.Linear(d_mid, d_pair, bias=use_bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_a.weight)
        if self.proj_a.bias is not None:
            nn.init.zeros_(self.proj_a.bias)
            
        nn.init.xavier_uniform_(self.proj_b.weight)
        if self.proj_b.bias is not None:
            nn.init.zeros_(self.proj_b.bias)
            
        nn.init.xavier_uniform_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
            
    def forward(self, s, pair_rep=None):
        """
        Args:
            s: Sequence embeddings of shape (B, N, D_seq)
            pair_rep: Optional existing pair representations of shape (B, N, N, D_pair)
            
        Returns:
            Updated pair representation of shape (B, N, N, D_pair)
        """
        A = self.proj_a(s) # (B, N, d_mid)
        B = self.proj_b(s) # (B, N, d_mid)
        
        # Outer product: (B, N, N, d_mid)
        O = torch.einsum('bie,bje->bije', A, B)
        
        # Project to D_pair: (B, N, N, D_pair)
        update = self.proj_out(O)
        
        if pair_rep is not None:
            return pair_rep + update
        return update
