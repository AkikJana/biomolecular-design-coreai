import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TrainableStructurePredictor(nn.Module):
    """
    A trainable coordinate predictor network.
    Maps a sequence of amino acid embeddings to 3D backbone C-alpha coordinates.
    """
    def __init__(self, sequence_length: int = 50, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.L = sequence_length
        self.embed_dim = embed_dim
        
        # Learnable layers
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim, 3)
        
    def forward(self, seq_embeddings: torch.Tensor) -> torch.Tensor:
        # seq_embeddings shape: [1, L, 128]
        x = seq_embeddings.transpose(1, 2) # [1, 128, L]
        h = self.relu(self.conv1(x))
        h = h.transpose(1, 2) # [1, L, 256]
        
        lstm_out, _ = self.lstm(h) # [1, L, 256]
        coords = self.proj(lstm_out) # [1, L, 3]
        return coords

def compute_physical_loss(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes differentiable physical potential energy losses for a 3D coordinate chain.
    """
    L = coords.shape[1]
    
    # 1. Harmonic Bond Length Potential (Consecutive CA-CA distances should be ~3.8 Angstroms)
    diffs = coords[0, 1:, :] - coords[0, :-1, :] # [L-1, 3]
    dists = torch.norm(diffs, dim=-1) # [L-1]
    bond_loss = torch.mean((dists - 3.8) ** 2)
    
    # 2. Steric Clash Repulsion Potential (Non-consecutive CA-CA distances should be >= 4.0 Angstroms)
    c1 = coords[0].unsqueeze(1) # [L, 1, 3]
    c2 = coords[0].unsqueeze(0) # [1, L, 3]
    pairwise_dists = torch.norm(c1 - c2, dim=-1) # [L, L]
    
    # Mask to select only non-consecutive residue pairs (|i - j| > 1)
    indices = torch.arange(L, device=coords.device)
    i_idx = indices.unsqueeze(1)
    j_idx = indices.unsqueeze(0)
    non_consecutive_mask = torch.abs(i_idx - j_idx) > 1
    
    clash_dists = pairwise_dists[non_consecutive_mask]
    # Penalize only if distance is less than 4.0 Angstroms
    clash_penalties = torch.clamp(4.0 - clash_dists, min=0.0)
    clash_loss = torch.mean(clash_penalties ** 2)
    
    # 3. Chain Smoothness Potential (Prevents self-intersection and extreme hairpins)
    # Angle between consecutive segment vectors v1 and v2
    v1 = diffs[:-1] # [L-2, 3]
    v2 = diffs[1:]  # [L-2, 3]
    dots = torch.sum(v1 * v2, dim=-1)
    norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
    cos_angles = dots / (norms + 1e-6)
    
    # Penalize sharp angles (where cos_angle is negative, meaning hairpin turn)
    angle_loss = torch.mean(torch.clamp(-cos_angles - 0.2, min=0.0) ** 2)
    
    return bond_loss, clash_loss, angle_loss

def main():
    print("==========================================================================")
    print("        PHYSICS-INFORMED NEURAL NETWORK (PINN) TRAINING DEMO")
    print("==========================================================================")
    
    # Sequence details (Human Insulin Fragment - 50 residues)
    insulin_seq = "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPK"
    L = len(insulin_seq)
    
    # Create simple dummy input sequence embeddings
    seq_embeddings = torch.randn(1, L, 128)
    
    # Initialize the predictor model
    model = TrainableStructurePredictor(sequence_length=L)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Check initial coordinates and physical energy
    initial_coords = model(seq_embeddings)
    init_bond, init_clash, init_angle = compute_physical_loss(initial_coords)
    init_total_energy = init_bond * 10.0 + init_clash * 15.0 + init_angle * 5.0
    
    print(f"Initial State Analysis:")
    print(f"  Bond Length Loss (consecutive): {init_bond.item():.4f}")
    print(f"  Steric Clash Loss (overlap):    {init_clash.item():.4f}")
    print(f"  Chain Smoothness Loss:         {init_angle.item():.4f}")
    print(f"  Total Physical Energy:          {init_total_energy.item():.4f}")
    print("==========================================================================")
    
    print("\nRunning PINN optimization loop (Training ONLY on physical potential energy)...")
    print("-" * 90)
    print(f"{'Epoch':<8} | {'Total Energy':<15} | {'Bond Loss':<12} | {'Clash Loss':<12} | {'Angle Loss':<12} | {'Max Clash (A)':<15}")
    print("-" * 90)
    
    epochs = 300
    for epoch in range(epochs + 1):
        # 1. Forward Pass
        coords = model(seq_embeddings)
        
        # 2. Compute Physical Losses
        bond_loss, clash_loss, angle_loss = compute_physical_loss(coords)
        
        # Total loss is a weighted sum of physical potentials
        # No coordinate matching dataset is used! The model is trained purely on physics.
        total_loss = bond_loss * 20.0 + clash_loss * 25.0 + angle_loss * 10.0
        
        # 3. Backpropagation & Parameter Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log progress every 30 epochs
        if epoch % 30 == 0 or epoch == epochs:
            # Calculate max clash metrics for display
            with torch.no_grad():
                c1 = coords[0].unsqueeze(1)
                c2 = coords[0].unsqueeze(0)
                dists = torch.norm(c1 - c2, dim=-1)
                indices = torch.arange(L)
                mask = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)) > 1
                non_consec_dists = dists[mask]
                min_non_consec_dist = torch.min(non_consec_dists).item()
                
            print(f"Epoch {epoch:<3}  | {total_loss.item():<15.4f} | {bond_loss.item():<12.4f} | {clash_loss.item():<12.4f} | {angle_loss.item():<12.4f} | {min_non_consec_dist:<15.2f}")
            
    print("-" * 90)
    print("\nTraining complete!")
    
    # 4. Final verification metrics
    final_coords = model(seq_embeddings)
    final_bond, final_clash, final_angle = compute_physical_loss(final_coords)
    print(f"Final State Analysis:")
    print(f"  Bond Length Loss: {final_bond.item():.6f} (Target: ~0.0000)")
    print(f"  Steric Clash Loss: {final_clash.item():.6f} (Target: ~0.0000)")
    print(f"  Chain Smoothness: {final_angle.item():.6f}")
    
    # Plotting and saving the folded 3D coordinates using matplotlib
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        xs = final_coords[0, :, 0].detach().numpy()
        ys = final_coords[0, :, 1].detach().numpy()
        zs = final_coords[0, :, 2].detach().numpy()
        
        # Plot backbone line
        ax.plot(xs, ys, zs, color='#10b981', linewidth=3, label="PINN-Folded Backbone")
        ax.scatter(xs, ys, zs, color='#3b82f6', s=80, edgecolors='black')
        
        ax.set_title("3D Coordinate Trace optimized via PINN Physical Loss", fontsize=12, fontweight='bold')
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        
        output_png = "/Users/akikjana/Documents/BiomolecularDesign/backbone_3d_pinn.png"
        plt.savefig(output_png, dpi=150)
        print(f"Folded coordinate plot saved to: {output_png}")
    except Exception as e:
        print(f"Plotting skipped due to: {e}")
        
    print("==========================================================================")

if __name__ == "__main__":
    main()
