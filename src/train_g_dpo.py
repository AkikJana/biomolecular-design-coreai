import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any

# Resolve local paths
sys.path.append(os.path.dirname(__file__))

from dms_generation import TargetDMSGenerator
from g_dpo_alignment import cluster_by_union_mask, GDPOLoss
from boltz_wrapper import BoltzModelWrapper

# -------------------------------------------------------------
# 1. DEFINE A TRAINABLE SURROGATE NETWORK FOR DEMONSTRATION
# -------------------------------------------------------------

class TrainableSurrogateModel(nn.Module):
    """A simple learnable neural network simulating the structure model's scoring head.
    
    Predicts log-probabilities (likelihoods) for binder sequences.
    Optimizing this via DPO will teach it to favor sequences that fold stably 
    and bind tightly.
    """
    
    def __init__(self, vocab_size: int = 20, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Learnable layers that map sequence features to a single likelihood score
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, sequence: str) -> torch.Tensor:
        # Map amino acids to indices: A=0, C=1, etc.
        # Use standard alphabet mapping
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        indices = []
        for char in sequence:
            idx = alphabet.index(char) if char in alphabet else 0
            indices.append(idx)
            
        x_tensor = torch.tensor(indices, dtype=torch.long)
        embeds = self.embedding(x_tensor) # Shape: [L, embed_dim]
        mean_embed = embeds.mean(dim=0)  # Average pooling: [embed_dim]
        
        logits = self.network(mean_embed) # Shape: [1]
        return logits.squeeze(0)          # Return scalar likelihood score


# -------------------------------------------------------------
# 2. DEFINE INTERFACE SCORER
# -------------------------------------------------------------

def evaluate_simulated_affinity(sequence: str) -> float:
    """Simulates a biophysical interface affinity score based on sequence traits.
    
    In a production setting, this function is replaced by running structure 
    prediction and evaluating physical contacts or energy scores (e.g., ipSAE).
    
    We simulate:
    - Polar residues (D, E, S, K, R, T) forming hydrogen bonds (increases score).
    - Hydrophobic residues (L, I, V, A) stabilizing the core (increases score).
    """
    polar_count = sum(1 for c in sequence if c in "DESKRTQN")
    hydrophobic_count = sum(1 for c in sequence if c in "LIVAMF")
    
    # We define a sweet spot: good core stability + active interface hydrogen bonds
    affinity = (polar_count * 0.4) + (hydrophobic_count * 0.3)
    return affinity


# -------------------------------------------------------------
# 3. TRAINING PIPELINE EXECUTION
# -------------------------------------------------------------

def run_training_pipeline():
    print("=======================================================")
    # 1. Setup target and generate DMS library
    print("[Pipeline] Step 1: Downloading target structure and generating DMS library...")
    target = "TNF-alpha"
    base_sequence = "MATEVLADIGSAKLR"
    
    dms_gen = TargetDMSGenerator(output_dir="/tmp/biomolecular_design")
    pdb_path = dms_gen.download_target_pdb(target)
    
    # Generate single-residue scan library
    dms_library = dms_gen.generate_dms_library(
        base_sequence=base_sequence,
        interface_positions=[2, 4, 8, 12, 15],
        amino_acids="ADEFGHIKLMNPQRSTVWY"
    )
    
    # Calculate simulated affinity scores for all sequences in the library
    for entry in dms_library:
        entry["score"] = evaluate_simulated_affinity(entry["sequence"])
        
    # 2. Cluster the sequences to apply g-DPO local updates
    print("\n[Pipeline] Step 2: Clustering sequences using Union Mask...")
    sequences = [entry["sequence"] for entry in dms_library]
    clusters = cluster_by_union_mask(sequences, max_positions_in_union=3)
    print(f"  Total sequences: {len(sequences)} clustered into {len(clusters)} local groups.")
    
    # 3. Initialize Policy Model (trainable) and Reference Model (frozen)
    print("\n[Pipeline] Step 3: Initializing policy and reference networks...")
    policy_model = TrainableSurrogateModel()
    ref_model = TrainableSurrogateModel()
    
    # Load same starting weights into reference model and freeze it
    ref_model.load_state_dict(policy_model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # Setup optimizer and DPO Loss
    optimizer = optim.Adam(policy_model.parameters(), lr=0.01)
    dpo_loss_module = GDPOLoss(beta=0.1)
    
    # 4. Execute g-DPO Training Loop
    print("\n[Pipeline] Step 4: Starting g-DPO training loop...")
    epochs = 5
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_valid_clusters = 0
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue # Skip clusters with no preference options
                
            # Extract sequence data in the cluster
            cluster_seqs = [dms_library[i]["sequence"] for i in cluster]
            cluster_scores = torch.tensor([dms_library[i]["score"] for i in cluster], dtype=torch.float)
            
            # Predict log-probabilities under Policy Model
            policy_logps = torch.stack([policy_model(seq) for seq in cluster_seqs])
            
            # Predict log-probabilities under Reference Model (no gradients)
            with torch.no_grad():
                ref_logps = torch.stack([ref_model(seq) for seq in cluster_seqs])
                
            # Compute g-DPO loss
            loss, metrics = dpo_loss_module(policy_logps, ref_logps, cluster_scores, pairing_strategy="best_vs_all")
            
            if metrics.get("num_pairs", 0) > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += metrics["accuracy"]
                num_valid_clusters += 1
                
        # Average statistics over epoch
        avg_loss = epoch_loss / max(1, num_valid_clusters)
        avg_acc = epoch_acc / max(1, num_valid_clusters)
        print(f"  Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.6f} | Average Alignment Accuracy: {avg_acc:.2%}")
        
    # 5. Save Model Checkpoint
    checkpoint_path = "/tmp/biomolecular_design/policy_dpo_checkpoint.pth"
    torch.save(policy_model.state_dict(), checkpoint_path)
    print(f"\n[Pipeline] Step 5: Training completed. Policy checkpoint saved to: {checkpoint_path}")
    print("=======================================================")

if __name__ == "__main__":
    run_training_pipeline()
