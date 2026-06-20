import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicBlockSparseAttention(nn.Module):
    """
    Dynamic Block-Sparse Attention Module.
    This module uses a lightweight neural gating network to predict active attention blocks
    and computes self-attention only on those active blocks, reducing complexity from O(N^2)
    to O(K * B^2) where K is the number of active block pairs and B is the block size.
    """
    def __init__(self, dim, num_heads=8, block_size=64, gate_dim=64, threshold=0.5, sparsity_penalty=0.01):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.gate_dim = gate_dim
        self.threshold = threshold
        self.sparsity_penalty = sparsity_penalty
        
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Projection layers for self-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Gating network
        # Project mean-pooled block representations to gating query and key spaces
        self.gate_q_proj = nn.Linear(dim, num_heads * gate_dim)
        self.gate_k_proj = nn.Linear(dim, num_heads * gate_dim)
        
        # Diagnostics
        self.last_sparsity_ratio = 0.0
        self.last_gate_loss = torch.tensor(0.0)

    def compute_gate_decisions(self, X, force_all_active=False):
        """
        Predicts which block pairs are active.
        """
        batch_size, seq_len, dim = X.shape
        num_blocks = seq_len // self.block_size
        
        # 1. Pool sequence embeddings into block representations
        # Shape: (batch_size, num_blocks, dim)
        block_repr = X.view(batch_size, num_blocks, self.block_size, dim).mean(dim=2)
        
        # 2. Project block representations
        g_q = self.gate_q_proj(block_repr)
        g_k = self.gate_k_proj(block_repr)
        
        # Reshape to (batch_size, num_heads, num_blocks, gate_dim)
        g_q = g_q.view(batch_size, num_blocks, self.num_heads, self.gate_dim).transpose(1, 2)
        g_k = g_k.view(batch_size, num_blocks, self.num_heads, self.gate_dim).transpose(1, 2)
        
        # 3. Compute dot-product gating logits: (batch_size, num_heads, num_blocks, num_blocks)
        gate_logits = torch.matmul(g_q, g_k.transpose(-1, -2)) / math.sqrt(self.gate_dim)
        
        # 4. Probabilities
        gate_prob = torch.sigmoid(gate_logits)
        
        # 5. Apply Straight-Through Estimator (STE) for binarization or force all active
        if force_all_active:
            gate_decision = torch.ones_like(gate_prob)
        else:
            gate_decision = (gate_prob > self.threshold).float() - gate_prob.detach() + gate_prob
        
        # Enforce local context: diagonal blocks are always active
        diag_mask = torch.eye(num_blocks, device=X.device).view(1, 1, num_blocks, num_blocks)
        gate_decision = gate_decision * (1.0 - diag_mask) + diag_mask
        
        return gate_decision, gate_prob

    def forward(self, X, return_gate_info=False, force_all_active=False):
        # X: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = X.shape
        device = X.device
        
        # Handle sequence lengths not divisible by block_size
        pad_len = 0
        if seq_len % self.block_size != 0:
            pad_len = self.block_size - (seq_len % self.block_size)
            X = F.pad(X, (0, 0, 0, pad_len))
            # Updated sequence length
            seq_len = X.shape[1]
            
        num_blocks = seq_len // self.block_size
        
        # 1. Compute queries, keys, values
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # Reshape to (batch_size, num_heads, num_blocks, block_size, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        Q = Q.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        K = K.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        V = V.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        
        # 2. Get gating decisions
        gate_decision, gate_prob = self.compute_gate_decisions(X, force_all_active=force_all_active)
        
        # Calculate sparsity ratio
        num_active = gate_decision.sum()
        total_blocks = batch_size * self.num_heads * num_blocks * num_blocks
        self.last_sparsity_ratio = (num_active / total_blocks).item()
        
        # Sparsity loss (encourages sparsity on off-diagonal blocks)
        diag_mask = torch.eye(num_blocks, device=device).view(1, 1, num_blocks, num_blocks)
        off_diag_probs = gate_prob * (1.0 - diag_mask)
        self.last_gate_loss = off_diag_probs.mean() * self.sparsity_penalty
        
        # 3. Find indices of active block pairs
        active_indices = torch.nonzero(gate_decision) # (K, 4)
        
        batch_idx = active_indices[:, 0]
        head_idx = active_indices[:, 1]
        q_block_idx = active_indices[:, 2]
        k_block_idx = active_indices[:, 3]
        
        # 4. Gather active Q, K, V blocks
        # Shape: (K, block_size, head_dim)
        Q_blocks = Q[batch_idx, head_idx, q_block_idx]
        K_blocks = K[batch_idx, head_idx, k_block_idx]
        V_blocks = V[batch_idx, head_idx, k_block_idx]
        
        # 5. Compute raw attention scores for active blocks
        # Shape: (K, block_size, block_size)
        attn_scores = torch.bmm(Q_blocks, K_blocks.transpose(1, 2)) / math.sqrt(self.head_dim)
        
        # Mask out padded elements if padding was added
        if pad_len > 0:
            k_local_idx = torch.arange(self.block_size, device=device).view(1, 1, self.block_size)
            k_global_idx = k_block_idx.view(-1, 1, 1) * self.block_size + k_local_idx
            invalid_mask = k_global_idx >= (seq_len - pad_len)
            attn_scores = attn_scores.masked_fill(invalid_mask, -1e9)
            
        # 6. Segmented softmax across active key blocks for each query block
        # Unique query block ID: (batch, head, q_block_idx)
        query_block_id = batch_idx * (self.num_heads * num_blocks) + head_idx * num_blocks + q_block_idx
        num_query_blocks = batch_size * self.num_heads * num_blocks
        
        # Subtract max for numerical stability
        block_max = attn_scores.max(dim=-1).values # (K, block_size)
        global_max = torch.full((num_query_blocks, self.block_size), -float('inf'), device=device)
        
        index_expanded = query_block_id.unsqueeze(1).expand(-1, self.block_size)
        global_max = global_max.scatter_reduce(0, index_expanded, block_max, reduce="amax", include_self=False)
        
        max_per_token = global_max[query_block_id] # (K, block_size)
        exp_scores = torch.exp(attn_scores - max_per_token.unsqueeze(-1)) # (K, block_size, block_size)
        
        # Sum of exponentials
        block_sum_exp = exp_scores.sum(dim=-1) # (K, block_size)
        sum_exp = torch.zeros((num_query_blocks, self.block_size), device=device)
        sum_exp = sum_exp.scatter_reduce(0, index_expanded, block_sum_exp, reduce="sum", include_self=False)
        
        sum_exp_per_token = sum_exp[query_block_id] # (K, block_size)
        attn_weights = exp_scores / (sum_exp_per_token.unsqueeze(-1) + 1e-9) # (K, block_size, block_size)
        
        # 7. Compute weighted sum of values
        O_blocks = torch.bmm(attn_weights, V_blocks) # (K, block_size, head_dim)
        
        # 8. Scatter attention outputs back to query blocks
        O_output = torch.zeros((num_query_blocks, self.block_size, self.head_dim), device=device)
        index_expanded_out = query_block_id.unsqueeze(1).unsqueeze(2).expand(-1, self.block_size, self.head_dim)
        O_output = O_output.scatter_reduce(0, index_expanded_out, O_blocks, reduce="sum", include_self=False)
        
        # Reshape and permute back to (batch_size, seq_len_padded, dim)
        O_output = O_output.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        O_output = O_output.permute(0, 2, 3, 1, 4).reshape(batch_size, seq_len, dim)
        
        # Crop back to original sequence length if padded
        if pad_len > 0:
            O_output = O_output[:, :-pad_len, :]
            
        # Project output
        output = self.out_proj(O_output)
        
        if return_gate_info:
            return output, gate_decision, gate_prob
        return output
