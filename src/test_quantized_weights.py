import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.quantized_attention_weights import DynamicQuantizedLinear

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Layer hyperparameters
IN_FEATURES = 1024
OUT_FEATURES = 1024
BATCH_SIZE = 8
BLOCK_SIZE = 32
SEQ_LENGTHS = [100, 500, 1000, 1500, 2000]

def evaluate_model_on_seq(baseline_fp32, baseline_fp16, q_model, seq_len):
    """
    Evaluates the reconstruction fidelity and memory sizes for a specific sequence length.
    """
    baseline_fp32.eval()
    baseline_fp16.eval()
    q_model.eval()
    
    # Generate random input
    x_fp32 = torch.randn(BATCH_SIZE, seq_len, IN_FEATURES, device=device)
    x_fp16 = x_fp32.half()
    
    with torch.no_grad():
        # Baseline FP32 outputs
        y_fp32 = baseline_fp32(x_fp32)
        
        # Baseline FP16 outputs
        y_fp16 = baseline_fp16(x_fp16)
        
        # Quantized outputs (FP32 mode)
        q_model.to(torch.float32)
        y_quant_fp32 = q_model(x_fp32)
        
        # Quantized outputs (FP16 mode)
        q_model_half = DynamicQuantizedLinear(
            IN_FEATURES, OUT_FEATURES, bias=baseline_fp32.bias is not None, block_size=BLOCK_SIZE, mode=q_model.mode
        ).to(device).half()
        q_model_half.weight.copy_(q_model.weight.half())
        if q_model.bias is not None:
            q_model_half.bias.copy_(q_model.bias.half())
        for p_half, p_orig in zip(q_model_half.meta_net.parameters(), q_model.meta_net.parameters()):
            p_half.copy_(p_orig.half())
            
        y_quant_fp16 = q_model_half(x_fp16)
        
        # Compute reconstruction metrics
        mse_fp32 = F.mse_loss(y_fp32, y_quant_fp32).item()
        cos_fp32 = F.cosine_similarity(y_fp32, y_quant_fp32, dim=-1).mean().item()
        
        mse_fp16 = F.mse_loss(y_fp16, y_quant_fp16).item()
        cos_fp16 = F.cosine_similarity(y_fp16, y_quant_fp16, dim=-1).mean().item()
        
        # Compute VRAM sizes of the weights
        fp32_size = IN_FEATURES * OUT_FEATURES * 4.0
        fp16_size = IN_FEATURES * OUT_FEATURES * 2.0
        quant_size = q_model.estimate_inference_weight_size()
        avg_bitwidth = q_model.get_average_bitwidth().item()
        
    return {
        "seq_len": seq_len,
        "mse_fp32": mse_fp32,
        "cos_fp32": cos_fp32,
        "mse_fp16": mse_fp16,
        "cos_fp16": cos_fp16,
        "avg_bitwidth": avg_bitwidth,
        "fp32_size_kb": fp32_size / 1024.0,
        "fp16_size_kb": fp16_size / 1024.0,
        "quant_size_kb": quant_size / 1024.0,
        "fp32_saving_pct": (1.0 - quant_size / fp32_size) * 100.0,
        "fp16_saving_pct": (1.0 - quant_size / fp16_size) * 100.0,
    }

def train_calibration(baseline, q_model, num_steps=150, lr=1e-3, bitwidth_penalty=0.01):
    """
    Calibrates the weights and the meta-network parameters of the quantized model
    to match the baseline FP32 model.
    """
    print(f"\nCalibrating {q_model.mode} mode...")
    q_model.train()
    baseline.eval()
    
    optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)
    
    for step in range(num_steps):
        # Sample random inputs
        x = torch.randn(BATCH_SIZE, 500, IN_FEATURES, device=device)
        
        with torch.no_grad():
            y_baseline = baseline(x)
            
        optimizer.zero_grad()
        y_quant = q_model(x)
        
        mse_loss = F.mse_loss(y_quant, y_baseline)
        bitwidth_loss = q_model.get_average_bitwidth()
        
        # Total loss combines reconstruction fidelity and weight size constraint
        loss = mse_loss + bitwidth_penalty * bitwidth_loss
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 25 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{num_steps} | Loss: {loss.item():.6f} | MSE: {mse_loss.item():.6f} | Avg Bitwidth: {bitwidth_loss.item():.2f}")
            
    q_model.eval()

def main():
    # 1. Initialize baseline linear layers
    baseline_fp32 = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True).to(device)
    baseline_fp16 = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True).to(device).half()
    with torch.no_grad():
        baseline_fp16.weight.copy_(baseline_fp32.weight.half())
        baseline_fp16.bias.copy_(baseline_fp32.bias.half())
    
    results = {}
    
    # 2. Benchmark uncalibrated mixed-precision layer
    print("\n--- 1. Evaluating Uncalibrated Mixed-Precision Layer ---")
    q_mixed_uncal = DynamicQuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True, block_size=BLOCK_SIZE, mode='mixed').to(device)
    with torch.no_grad():
        q_mixed_uncal.weight.copy_(baseline_fp32.weight)
        q_mixed_uncal.bias.copy_(baseline_fp32.bias)
    
    results["mixed_uncalibrated"] = []
    for seq_len in SEQ_LENGTHS:
        res = evaluate_model_on_seq(baseline_fp32, baseline_fp16, q_mixed_uncal, seq_len)
        results["mixed_uncalibrated"].append(res)
        print(f"SeqLen: {seq_len:4d} | MSE FP32: {res['mse_fp32']:.6f} | Cos FP32: {res['cos_fp32']:.6f} | Avg Bits: {res['avg_bitwidth']:.2f}")
        
    # 3. Train/Calibrate mixed-precision layer
    q_mixed_cal = DynamicQuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True, block_size=BLOCK_SIZE, mode='mixed').to(device)
    with torch.no_grad():
        q_mixed_cal.weight.copy_(baseline_fp32.weight)
        q_mixed_cal.bias.copy_(baseline_fp32.bias)
    train_calibration(baseline_fp32, q_mixed_cal, num_steps=150, lr=1e-3, bitwidth_penalty=0.01)
    
    results["mixed_calibrated"] = []
    print("\nEvaluating Calibrated Mixed-Precision Layer:")
    for seq_len in SEQ_LENGTHS:
        res = evaluate_model_on_seq(baseline_fp32, baseline_fp16, q_mixed_cal, seq_len)
        results["mixed_calibrated"].append(res)
        print(f"SeqLen: {seq_len:4d} | MSE FP32: {res['mse_fp32']:.6f} | Cos FP32: {res['cos_fp32']:.6f} | Avg Bits: {res['avg_bitwidth']:.2f}")
        
    # 4. Train/Calibrate Pure INT8 layer
    q_int8 = DynamicQuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True, block_size=BLOCK_SIZE, mode='int8').to(device)
    with torch.no_grad():
        q_int8.weight.copy_(baseline_fp32.weight)
        q_int8.bias.copy_(baseline_fp32.bias)
    train_calibration(baseline_fp32, q_int8, num_steps=100, lr=1e-3, bitwidth_penalty=0.0) # no penalty needed since bits are fixed
    
    results["int8_calibrated"] = []
    print("\nEvaluating Calibrated INT8 Layer:")
    for seq_len in SEQ_LENGTHS:
        res = evaluate_model_on_seq(baseline_fp32, baseline_fp16, q_int8, seq_len)
        results["int8_calibrated"].append(res)
        print(f"SeqLen: {seq_len:4d} | MSE FP32: {res['mse_fp32']:.6f} | Cos FP32: {res['cos_fp32']:.6f} | Avg Bits: {res['avg_bitwidth']:.2f}")

    # 5. Train/Calibrate Pure INT4 layer
    q_int4 = DynamicQuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True, block_size=BLOCK_SIZE, mode='int4').to(device)
    with torch.no_grad():
        q_int4.weight.copy_(baseline_fp32.weight)
        q_int4.bias.copy_(baseline_fp32.bias)
    train_calibration(baseline_fp32, q_int4, num_steps=150, lr=1e-3, bitwidth_penalty=0.0)
    
    results["int4_calibrated"] = []
    print("\nEvaluating Calibrated INT4 Layer:")
    for seq_len in SEQ_LENGTHS:
        res = evaluate_model_on_seq(baseline_fp32, baseline_fp16, q_int4, seq_len)
        results["int4_calibrated"].append(res)
        print(f"SeqLen: {seq_len:4d} | MSE FP32: {res['mse_fp32']:.6f} | Cos FP32: {res['cos_fp32']:.6f} | Avg Bits: {res['avg_bitwidth']:.2f}")

    # Write results to json file in correct path
    output_dir = "/Users/akikjana/.gemini/antigravity-cli/brain/4cdb7261-e55b-4efc-9ffa-c6509d76c9c2"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "summary_quantized.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully wrote benchmark results to {results_path}")

if __name__ == "__main__":
    main()
