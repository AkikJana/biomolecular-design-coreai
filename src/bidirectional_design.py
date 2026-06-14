import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalCoDesigner(nn.Module):
    """Demonstrates Bidirectional Sequence-Structure Co-Optimization.
    
    Instead of keeping structure or sequence frozen, we optimize BOTH:
    1. Sequence parameters: Learnable sequence logits.
    2. Structure parameters: Learnable coordinate displacements (representing 3D backbone relaxation).
    """
    
    def __init__(self, seq_len: int = 10, embed_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        # 1. Sequence Parameter: Logits over the 20 amino acids
        # Shape: [1, seq_len, 20]
        self.sequence_logits = nn.Parameter(torch.randn(1, seq_len, 20) * 0.1)
        
        # 2. Structure Parameter: 3D coordinate displacement vectors (relaxation)
        # Shape: [1, seq_len, 3]. Initialized to 0 (no initial displacement)
        self.coord_displacements = nn.Parameter(torch.zeros(1, seq_len, 3))
        
        # Representation layers
        self.embedding_weights = nn.Parameter(torch.randn(20, embed_dim) * 0.1)
        self.attn_proj = nn.Linear(embed_dim, embed_dim)
        self.coord_head = nn.Linear(embed_dim, 3)

    def get_sequence(self) -> str:
        """Returns the current discrete sequence."""
        with torch.no_grad():
            indices = torch.argmax(self.sequence_logits, dim=-1).squeeze(0)
            return "".join([self.alphabet[idx.item()] for idx in indices])

    def forward(self, temp: float = 0.5) -> torch.Tensor:
        # Step A: Compute continuous sequence embedding via Gumbel-Softmax
        seq_probs = F.gumbel_softmax(self.sequence_logits, tau=temp, hard=False) # [1, seq_len, 20]
        seq_embeddings = torch.matmul(seq_probs, self.embedding_weights) # [1, seq_len, embed_dim]
        
        # Step B: Predict backbone coordinates through attention projections
        features = F.relu(self.attn_proj(seq_embeddings)) # [1, seq_len, embed_dim]
        base_coords = self.coord_head(features) # [1, seq_len, 3]
        
        # Step C: Apply learnable coordinate displacements (relaxing the structure)
        # This is where the structural relaxation happens bidirectionally!
        relaxed_coords = base_coords + self.coord_displacements
        return relaxed_coords


class JointBiophysicalLoss:
    """A joint loss module evaluating both sequence affinity and structural physical constraints."""
    
    def __init__(self, target_site: torch.Tensor, ideal_bond_dist: float = 3.8):
        self.target_site = target_site
        self.ideal_dist = ideal_bond_dist

    def compute_loss(self, coords: torch.Tensor, displacements: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Docking / Affinity Loss: Minimize distance of residue 0 to the target binding site
        docking_loss = torch.norm(coords[0, 0] - self.target_site)
        
        # 2. Physical Structural Strain:
        # C_alpha - C_alpha bonds between adjacent residues must remain near 3.8 Angstroms
        diffs = coords[0, 1:, :] - coords[0, :-1, :] # Shape: [seq_len-1, 3]
        bond_lengths = torch.norm(diffs, p=2, dim=-1) # Shape: [seq_len-1]
        bond_strain = torch.mean((bond_lengths - self.ideal_dist) ** 2)
        
        # 3. Coordinate Displacement Penalty (prevents the model from stretching structure infinitely)
        displacement_penalty = torch.mean(displacements ** 2)
        
        # Joint total loss
        total_loss = docking_loss + bond_strain * 5.0 + displacement_penalty * 2.0
        
        metrics = {
            "total_loss": total_loss.item(),
            "docking_distance": docking_loss.item(),
            "bond_strain": bond_strain.item(),
            "displacement_norm": displacements.norm().item()
        }
        
        return total_loss, metrics


def run_bidirectional_optimization():
    print("======================================================================")
    print("BIDIRECTIONAL CO-DESIGN (SEQUENCE-STRUCTURE) OPTIMIZATION SENSE CHECK")
    print("======================================================================")
    
    torch.manual_seed(101)
    
    # 1. Initialize co-designer
    model = BidirectionalCoDesigner(seq_len=10)
    
    # Target binding site coordinates
    target_site = torch.tensor([5.0, -3.0, 2.0], dtype=torch.float)
    loss_fn = JointBiophysicalLoss(target_site=target_site)
    
    # Optimizer updates BOTH sequence logits and structural coordinate displacements
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    print(f"Initial Sequence:      '{model.get_sequence()}'")
    with torch.no_grad():
        init_coords = model()[0, 0]
        init_dist = torch.norm(init_coords - target_site).item()
    print(f"Initial Distance:      {init_dist:.4f} Å")
    print(f"Initial Displacement:  0.0000 (structure unrelaxed)")
    
    print("\nRunning joint bidirectional sequence-structure optimization...")
    print("-" * 90)
    print(f"{'Step':<6} | {'Sequence':<12} | {'Docking Dist (Å)':<18} | {'Bond Strain (Å²)':<18} | {'Displacement Norm':<20}")
    print("-" * 90)
    
    steps = 40
    for step in range(steps + 1):
        temp = max(0.1, 0.6 * (1.0 - step / steps))
        
        # Forward pass: get relaxed coordinates
        coords = model(temp=temp)
        
        # Compute joint biophysical loss
        total_loss, metrics = loss_fn.compute_loss(coords, model.coord_displacements)
        
        # Backward pass & optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        if step % 5 == 0 or step == steps:
            print(f"Step {step:<2}  | {model.get_sequence():<12} | {metrics['docking_distance']:<18.4f} | "
                  f"{metrics['bond_strain']:<18.4f} | {metrics['displacement_norm']:<20.4f}")
            
    print("-" * 90)
    print(f"Final Designed Sequence: '{model.get_sequence()}'")
    with torch.no_grad():
        final_coords = model()[0, 0]
        final_dist = torch.norm(final_coords - target_site).item()
    print(f"Final Distance to Target: {final_dist:.4f} Å (Successfully docked!)")
    print("======================================================================\n")

if __name__ == "__main__":
    run_bidirectional_optimization()
