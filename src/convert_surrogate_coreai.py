import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import coreai_torch
from coreai_torch import TorchConverter, get_decomp_table
import coreai_opt.quantization as q
from pathlib import Path

class CrossAttentionBlock(nn.Module):
    """
    A stateful Cross-Attention block that implements dynamic KV-Caching.
    Caches the key-value projections of the target receptor to bypass recomputation
    when scanning large libraries of binder candidates.
    """
    def __init__(self, num_heads=4, L_target=150, embed_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.L_target = L_target
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        # State buffers for cached receptor key/value tensors
        self.register_buffer("k_cache", torch.zeros(1, num_heads, L_target, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, num_heads, L_target, self.head_dim))
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, binder_seq: torch.Tensor, target_k: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        # Update cache buffers using an in-place addition difference trick to bypass copy_ compiler segfault:
        # k_cache = k_cache + (target_k - k_cache) => target_k
        self.k_cache.add_(target_k - self.k_cache)
        self.v_cache.add_(target_v - self.v_cache)
        
        # Project Query for the binder candidate
        q_binder = self.q_proj(binder_seq)
        batch_size, seq_len, _ = q_binder.shape
        q_binder = q_binder.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Standard scaled dot-product attention using cached keys/values
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q_binder, self.k_cache.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, self.v_cache)
        
        # Reconstruct heads and project out
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_out)

class SurrogateModel(nn.Module):
    """
    An optimized PyTorch surrogate model representing a fast sequence-to-coordinate
    structure evaluator incorporating dynamic receptor KV-caching.
    """
    def __init__(self, num_heads=4, L_target=150, embed_dim=128):
        super().__init__()
        # Convolutions to refine candidate binder sequence embeddings
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Cross-attention block to query the target receptor
        self.cross_attn = CrossAttentionBlock(num_heads, L_target, embed_dim)
        
        # Linear projection layer mapping embeddings to 3D C-alpha coordinates
        self.proj = nn.Linear(embed_dim, 3)
        
    def forward(self, binder_seq, target_k, target_v):
        # Input shape: [1, L_binder, embed_dim]
        # Conv1d expects [1, embed_dim, L_binder]
        x_t = binder_seq.transpose(1, 2)
        h = self.relu(self.conv1(x_t))
        h_t = h.transpose(1, 2)
        
        # Cross attention to target receptor with KV-cache
        h_attn = self.cross_attn(h_t, target_k, target_v)
        
        # Final coordinate prediction
        coords = self.proj(h_attn)
        return coords

def main():
    print("Initializing PyTorch surrogate model with stateful Cross-Attention...")
    model = SurrogateModel(L_target=1300).eval()
    
    # Define representative inputs (1300 residues target)
    binder_seq = torch.randn(1, 20, 128) # 20 residues
    target_k = torch.randn(1, 4, 1300, 32) # 4 heads, 1300 residues, 32 dim
    target_v = torch.randn(1, 4, 1300, 32)
    
    # Apply FP8 weight-only quantization config using coreai-opt
    print("Applying FP8 weight-only quantization using coreai-opt...")
    q_spec = q.QuantizationSpec(dtype=torch.float8_e4m3fn)
    module_config = q.ModuleQuantizerConfig(op_state_spec={'weight': q_spec})
    quant_config = q.QuantizerConfig(global_config=module_config)
    
    quantizer = q.Quantizer(model, quant_config)
    prepared_model = quantizer.prepare((binder_seq, target_k, target_v))
    quantized_model = quantizer.finalize()
    print("Model quantized successfully.")
    
    print("Exporting model to PyTorch ExportedProgram...")
    ep = torch.export.export(quantized_model, args=(binder_seq, target_k, target_v))
    ep = ep.run_decompositions(get_decomp_table())
    
    print("Converting model to Core AI Intermediate Representation (IR)...")
    converter = TorchConverter().add_exported_program(
        ep,
        state_names=["cross_attn.k_cache", "cross_attn.v_cache"],
        input_names=["binder_seq", "target_k", "target_v"],
        output_names=["coords"]
    )
    coreai_program = converter.to_coreai()
    
    print("Optimizing Core AI graph representation...")
    coreai_program.optimize()
    
    output_path = Path("/Users/akikjana/Documents/BiomolecularDesign/surrogate_model.aimodel")
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    print(f"Saving Core AI model asset (.aimodel) to: {output_path}...")
    coreai_program.save_asset(output_path)
    print("Asset saved successfully.")
    
    # Compile the asset to produced compiled spécializations (.aimodelc)
    compiled_path = Path("/Users/akikjana/Documents/BiomolecularDesign/compiled_surrogate")
    if compiled_path.exists():
        import shutil
        shutil.rmtree(compiled_path)
    compiled_path.mkdir(exist_ok=True)
    
    # Locate coreai-build dynamically. The Metal toolchain cryptex mount path is
    # version-pinned (its UUID changes across OS/toolchain updates), so resolve
    # the binary at runtime rather than hardcoding it.
    build_bin = find_coreai_build()
    if build_bin is None:
        print("coreai-build not found (PATH / xcrun / Metal toolchain cryptex).")
        print("Skipping optional AOT .aimodelc compilation -- the .aimodel asset is")
        print("already runnable via coreai.runtime (see src/predict_structure.py).")
        return

    print(f"Compiling model asset to specialized binaries using {build_bin}...")
    cmd = [
        build_bin, "compile",
        str(output_path),
        "--platform", "macOS",
        "--output", str(compiled_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("Success! Core AI conversion and AOT compilation complete.")
        print(f"Compiled specializations output directory: {compiled_path}")
    else:
        print(f"Compilation failed with exit code: {res.returncode}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)


def find_coreai_build():
    """Resolve the coreai-build CLI across PATH, xcrun, and Metal toolchain cryptex mounts."""
    import glob
    import shutil

    found = shutil.which("coreai-build")
    if found:
        return found
    try:
        r = subprocess.run(["xcrun", "--find", "coreai-build"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    pattern = (
        "/var/run/com.apple.security.cryptexd/mnt/*MetalToolchain*"
        "/Metal.xctoolchain/usr/bin/coreai-build"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


if __name__ == "__main__":
    main()
