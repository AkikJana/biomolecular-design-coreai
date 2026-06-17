import os
import asyncio
import numpy as np
import torch
import torch.nn as nn
import coreai.runtime as rt
from coreai.runtime import NDArray
from pathlib import Path

class DynamicStructurePredictor:
    """
    A dynamic structure predictor wrapper designed for deployment inside macOS Apps.
    
    Accepts raw amino acid sequence strings of varying lengths, converts them to
    embeddings, handles CoreAI target KV-Caching state logic automatically, and
    returns predicted 3D coordinates.
    """
    def __init__(self, aimodel_path: str = "/Users/akikjana/Documents/BiomolecularDesign/surrogate_model_dynamic.aimodel"):
        self.aimodel_path = aimodel_path
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.embed_dim = 128
        self.num_heads = 4
        self.head_dim = 32
        self.L_target_max = 2500
        
        # Load deterministic embeddings for amino acids
        torch.manual_seed(42)
        self.embed_lookup = nn.Embedding(len(self.alphabet), self.embed_dim)
        
        # Deterministic linear projections for target K/V projections
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj.eval()
        self.v_proj.eval()
        
        # Load CoreAI model and function pointers asynchronously
        self._loop = asyncio.get_event_loop()
        print(f"[App] Loading AOT Compiled CoreAI Model from: {self.aimodel_path}...")
        self.model = self._loop.run_until_complete(rt.AIModel.load(self.aimodel_path))
        self.rt_func = self.model.load_function("main")
        print("[App] CoreAI Model and 'main' function loaded successfully.")
        
        # Initialize the state dictionary for Neural Engine cache buffers
        self.state = {
            "cross_attn.k_cache": NDArray(np.zeros((1, self.num_heads, self.L_target_max, self.head_dim), dtype=np.float32)),
            "cross_attn.v_cache": NDArray(np.zeros((1, self.num_heads, self.L_target_max, self.head_dim), dtype=np.float32))
        }
        
    def _seq_to_embeddings(self, seq: str) -> torch.Tensor:
        """Converts an amino acid sequence string to an embedding tensor."""
        indices = [self.alphabet.index(char) if char in self.alphabet else 0 for char in seq]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        with torch.no_grad():
            embeds = self.embed_lookup(idx_tensor) # [L, 128]
        return embeds.unsqueeze(0) # [1, L, 128]
        
    def _project_target_kv(self, target_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Projects target embeddings into multi-head keys and values."""
        batch_size, L_t, _ = target_embeds.shape
        with torch.no_grad():
            k_proj = self.k_proj(target_embeds)
            v_proj = self.v_proj(target_embeds)
            
        # Split into heads: [1, H, L_target, head_dim]
        k_heads = k_proj.view(batch_size, L_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_heads = v_proj.view(batch_size, L_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return k_heads, v_heads

    def predict(self, binder_seq: str, target_seq: str) -> np.ndarray:
        """
        Runs 3D coordinate prediction on the input binder candidate sequence
        against the target receptor sequence.
        """
        # 1. Convert sequences to embeddings
        binder_embeds = self._seq_to_embeddings(binder_seq) # [1, L_binder, 128]
        target_embeds = self._seq_to_embeddings(target_seq) # [1, L_target, 128]
        
        # 2. Project target keys and values
        k_heads, v_heads = self._project_target_kv(target_embeds) # [1, 4, L_target, 32]
        
        # 3. Invoke CoreAI AOT Compiled Model (Neural Engine / GPU)
        # Using run_until_complete to wrap async call synchronously for the host App UI
        inputs = {
            "binder_seq": NDArray(binder_embeds.numpy().astype(np.float32)),
            "target_k": NDArray(k_heads.numpy().astype(np.float32)),
            "target_v": NDArray(v_heads.numpy().astype(np.float32))
        }
        
        outputs = self._loop.run_until_complete(
            self.rt_func(inputs=inputs, state=self.state)
        )
        
        # Output coords shape: [1, L_binder, 3]
        return outputs["coords"].numpy()

def main():
    # Instantiate the predictor
    predictor = DynamicStructurePredictor()
    
    # Define actual target and binder sequences
    target_receptor = "MATEVLADIGSAKLRAVREILAQGEIS" # 28 residues
    binder_candidate_1 = "MATEVLAD" # 8 residues mutant
    binder_candidate_2 = "MATEVLADIGSAKLR" # 15 residues mutant
    
    print("\n--- Predict Mutant 1 (8 residues) ---")
    coords1 = predictor.predict(binder_candidate_1, target_receptor)
    print("Mutant 1 Predicted Coordinates Shape:", coords1.shape)
    
    print("\n--- Predict Mutant 2 (15 residues) ---")
    coords2 = predictor.predict(binder_candidate_2, target_receptor)
    print("Mutant 2 Predicted Coordinates Shape:", coords2.shape)
    print("\nSuccess! Coordinates successfully predicted for actual mutant sequences.")

if __name__ == "__main__":
    main()
