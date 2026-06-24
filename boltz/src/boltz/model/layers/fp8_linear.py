"""FP8 weight-only quantized linear (portable PyTorch equivalent of the CoreAI
surrogate's FP8 path).

Stores weights as ``float8_e4m3fn`` (1 byte/elem, ~4x smaller than fp32) with a
per-output-channel fp32 scale, and dequantizes on the forward pass. Activations
stay in the input dtype. This is a *lossy*, post-training quantization intended
for inference; wiring it into a model requires accuracy validation (and ideally
a brief fine-tune) -- it is NOT numerically equivalent to the fp32 layer, so it
is provided as an opt-in utility rather than enabled in the live forward path.

Relationship to CoreAI: src/convert_surrogate_coreai.py performs the same FP8
weight-only quantization via Apple's coreai-opt and compiles to the Apple Neural
Engine. That toolchain is not available in every environment; FP8Linear gives the
same compression/accuracy tradeoff in plain PyTorch on CPU/MPS/CUDA.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Max representable magnitude of the e4m3 format.
_FP8_E4M3_MAX = 448.0


class FP8Linear(nn.Module):
    """Weight-only FP8 (e4m3) linear, drop-in for ``nn.Linear`` at inference."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_fp8", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        self.register_buffer("scale", torch.ones(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from(self, weight: Tensor, bias: Tensor = None) -> "FP8Linear":
        # Per-output-channel absmax scaling into the e4m3 range.
        amax = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / _FP8_E4M3_MAX
        self.weight_fp8 = (weight / scale).to(torch.float8_e4m3fn)
        self.scale = scale.to(torch.float32)
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias)
        return self

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        m = cls(linear.in_features, linear.out_features, linear.bias is not None)
        return m.quantize_from(
            linear.weight.data, linear.bias.data if linear.bias is not None else None
        )

    def dequantized_weight(self, dtype: torch.dtype) -> Tensor:
        return self.weight_fp8.to(dtype) * self.scale.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.dequantized_weight(x.dtype), self.bias)

    def weight_bytes(self) -> int:
        """Stored weight footprint (fp8 weights + fp32 per-channel scale)."""
        return self.weight_fp8.numel() * 1 + self.scale.numel() * 4


@torch.no_grad()
def quantize_linears_(module: nn.Module, min_features: int = 0) -> int:
    """Recursively replace ``nn.Linear`` children with ``FP8Linear`` in place.

    Returns the number of layers converted. ``min_features`` skips small layers
    (where quantization rarely pays off). Lossy -- validate accuracy after use.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if child.in_features >= min_features and child.out_features >= min_features:
                setattr(module, name, FP8Linear.from_linear(child))
                count += 1
        else:
            count += quantize_linears_(child, min_features=min_features)
    return count
