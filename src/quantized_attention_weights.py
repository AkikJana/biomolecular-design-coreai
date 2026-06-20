import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicQuantizedLinear(nn.Module):
    """
    A PyTorch linear module that dynamically quantizes its weight matrix to INT8/INT4
    during the forward pass using a neural-network parameterized block scale and offset predictor.
    It supports straight-through estimation (STE) to enable gradient flow during training,
    and supports mixed-precision (INT8 / INT4) selection per block.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        mode: str = 'mixed',  # 'mixed', 'int8', 'int4'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.mode = mode
        
        # Full precision weight parameter (saved for training/updates)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Meta-network to predict block-wise scale correction, offset correction, and bitwidth logit
        # Input features: block mean, standard deviation, minimum, maximum, and L2 norm
        self.meta_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Outputs: [delta_scale, delta_offset, logit_8]
        )
        
        self.reset_parameters()
        self.reset_meta_net()

    def reset_parameters(self):
        # Initialize linear layers weight and bias using Kaiming Uniform / standard initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def reset_meta_net(self):
        # Initialize meta-network weights to zero so that corrections initially start as identity (1.0 scale, 0.0 offset)
        with torch.no_grad():
            for m in self.meta_net.modules():
                if isinstance(m, nn.Linear):
                    m.weight.zero_()
                    m.bias.zero_()

    def get_average_bitwidth(self) -> torch.Tensor:
        """
        Computes the average bitwidth across all blocks. Differentiable.
        """
        if self.mode == 'int8':
            return torch.tensor(8.0, device=self.weight.device)
        elif self.mode == 'int4':
            return torch.tensor(4.0, device=self.weight.device)
            
        W = self.weight
        out_features, in_features = W.shape
        block_size = self.block_size
        
        if in_features % block_size != 0:
            padding = block_size - (in_features % block_size)
            W_padded = F.pad(W, (0, padding))
        else:
            W_padded = W
            
        num_blocks = W_padded.shape[1] // block_size
        W_blocks = W_padded.view(out_features, num_blocks, block_size)
        
        # Calculate statistics
        mean = W_blocks.mean(dim=-1, keepdim=True)
        std = W_blocks.std(dim=-1, keepdim=True)
        min_val = W_blocks.amin(dim=-1, keepdim=True)
        max_val = W_blocks.amax(dim=-1, keepdim=True)
        norm = torch.linalg.vector_norm(W_blocks, dim=-1, keepdim=True)
        stats = torch.cat([mean, std, min_val, max_val, norm], dim=-1)
        
        meta_dtype = self.meta_net[0].weight.dtype
        stats = stats.to(dtype=meta_dtype)
        pred = self.meta_net(stats).to(dtype=W.dtype)
        
        logit_8 = pred[..., 2:3]
        prob_8 = torch.sigmoid(logit_8)
        
        avg_bitwidth = prob_8 * 8.0 + (1.0 - prob_8) * 4.0
        return avg_bitwidth.mean()

    def estimate_inference_weight_size(self) -> float:
        """
        Estimates the weight storage size (in bytes) during inference.
        Assumes scale and offset are stored in float16 (2 bytes each).
        """
        W = self.weight
        out_features, in_features = W.shape
        block_size = self.block_size
        
        if in_features % block_size != 0:
            padding = block_size - (in_features % block_size)
            padded_in_features = in_features + padding
        else:
            padded_in_features = in_features
            
        num_blocks = (out_features * padded_in_features) // block_size
        
        if self.mode == 'mixed':
            avg_bit = self.get_average_bitwidth().item()
        elif self.mode == 'int8':
            avg_bit = 8.0
        else:
            avg_bit = 4.0
            
        # Quantized weights: avg_bit / 8.0 bytes per element
        # Scales and offsets: 4 bytes per block (2 bytes each for float16)
        weight_bytes = padded_in_features * out_features * (avg_bit / 8.0)
        meta_bytes = num_blocks * 4.0
        
        return weight_bytes + meta_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight
        out_features, in_features = W.shape
        block_size = self.block_size
        
        # Pad weight matrix along in_features dimension if necessary
        if in_features % block_size != 0:
            padding = block_size - (in_features % block_size)
            W_padded = F.pad(W, (0, padding))
        else:
            padding = 0
            W_padded = W
            
        num_blocks = W_padded.shape[1] // block_size
        W_blocks = W_padded.view(out_features, num_blocks, block_size)
        
        # Compute block statistics: mean, std, min, max, norm
        mean = W_blocks.mean(dim=-1, keepdim=True)
        std = W_blocks.std(dim=-1, keepdim=True)
        min_val = W_blocks.amin(dim=-1, keepdim=True)
        max_val = W_blocks.amax(dim=-1, keepdim=True)
        norm = torch.linalg.vector_norm(W_blocks, dim=-1, keepdim=True)
        
        stats = torch.cat([mean, std, min_val, max_val, norm], dim=-1)
        
        # Cast to meta_net parameter dtype to avoid mismatch
        meta_dtype = self.meta_net[0].weight.dtype
        stats = stats.to(dtype=meta_dtype)
        
        # Predict corrections and bitwidth logits
        pred = self.meta_net(stats)
        pred = pred.to(dtype=W.dtype)
        
        delta_s = pred[..., 0:1]
        delta_o = pred[..., 1:2]
        logit_8 = pred[..., 2:3]
        
        # Scale and offset corrections
        s_corr = F.softplus(delta_s + 0.5413)  # initialized to 1.0 when delta_s = 0
        
        # INT8 quantization path
        qmin_8, qmax_8 = -128, 127
        range_8 = max_val - min_val
        scale_base_8 = range_8 / 255.0 + 1e-8
        offset_base_8 = min_val - qmin_8 * scale_base_8
        
        scale_8 = scale_base_8 * s_corr
        offset_8 = offset_base_8 + delta_o * scale_8
        
        W_q_precise_8 = (W_blocks - offset_8) / scale_8
        W_q_rounded_8 = torch.clamp(torch.round(W_q_precise_8), qmin_8, qmax_8)
        W_q_8 = W_q_precise_8 + (W_q_rounded_8 - W_q_precise_8).detach()
        W_dq_8 = W_q_8 * scale_8 + offset_8
        
        # INT4 quantization path
        qmin_4, qmax_4 = -8, 7
        range_4 = max_val - min_val
        scale_base_4 = range_4 / 15.0 + 1e-8
        offset_base_4 = min_val - qmin_4 * scale_base_4
        
        scale_4 = scale_base_4 * s_corr
        offset_4 = offset_base_4 + delta_o * scale_4
        
        W_q_precise_4 = (W_blocks - offset_4) / scale_4
        W_q_rounded_4 = torch.clamp(torch.round(W_q_precise_4), qmin_4, qmax_4)
        W_q_4 = W_q_precise_4 + (W_q_rounded_4 - W_q_precise_4).detach()
        W_dq_4 = W_q_4 * scale_4 + offset_4
        
        # Decide selection probability
        if self.mode == 'int8':
            W_dq_blocks = W_dq_8
        elif self.mode == 'int4':
            W_dq_blocks = W_dq_4
        elif self.mode == 'mixed':
            prob_8 = torch.sigmoid(logit_8)
            if self.training:
                # Use Straight-Through Estimator (STE) for hard selection during training
                # to prevent the discretization gap between training and evaluation
                p_8_hard = (prob_8 > 0.5).float()
                p_8 = prob_8 + (p_8_hard - prob_8).detach()
            else:
                p_8 = (prob_8 > 0.5).float()
            W_dq_blocks = p_8 * W_dq_8 + (1.0 - p_8) * W_dq_4
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # Reconstruct full weight matrix
        W_dq_padded = W_dq_blocks.view(out_features, num_blocks * block_size)
        if padding > 0:
            W_dq = W_dq_padded[:, :-padding]
        else:
            W_dq = W_dq_padded
            
        # Compute standard linear forward pass using the reconstructed weights
        return F.linear(x, W_dq, self.bias)
