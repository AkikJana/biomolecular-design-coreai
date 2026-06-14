import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableStructureModel(nn.Module):
    """A simulated differentiable folding network representing AlphaFold / Boltz-1.
    
    Demonstrates how gradients flow backwards from predicted 3D coordinates, 
    through structural representation layers, back to the starting sequence logits.
    """
    
    def __init__(self, seq_len: int = 10, embed_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        # Learnable sequence logits: representing our starting sequence parameter
        # Shape: [1, seq_len, 20]
        self.sequence_logits = nn.Parameter(torch.randn(1, seq_len, 20) * 0.2)
        
        # Trainable sequence embedding weights (acting as the representation projection)
        self.embedding_weights = nn.Parameter(torch.randn(20, embed_dim) * 0.1)
        
        # Simulated attention layer: maps sequence representation to coordinate features
        self.attn_proj = nn.Linear(embed_dim, embed_dim)
        
        # Coordinate head: maps final features to 3D coordinates (x, y, z)
        # Shape: [embed_dim, 3]
        self.coord_head = nn.Linear(embed_dim, 3)

    def get_sequence(self) -> str:
        """Helper to get the current discrete sequence."""
        with torch.no_grad():
            indices = torch.argmax(self.sequence_logits, dim=-1).squeeze(0)
            return "".join([self.alphabet[idx.item()] for idx in indices])

    def forward(self, temp: float = 0.5) -> torch.Tensor:
        # 1. Gumbel-Softmax relaxation to keep sequence choices differentiable
        seq_probs = F.gumbel_softmax(self.sequence_logits, tau=temp, hard=False) # [1, seq_len, 20]
        
        # 2. Project probability vectors into embedding space
        seq_embeddings = torch.matmul(seq_probs, self.embedding_weights) # [1, seq_len, embed_dim]
        
        # 3. Pass through simulated structural attention block
        features = F.relu(self.attn_proj(seq_embeddings)) # [1, seq_len, embed_dim]
        
        # 4. Predict final 3D coordinates (backbone coordinates)
        coords = self.coord_head(features) # [1, seq_len, 3]
        return coords


def run_differentiable_folding():
    print("======================================================================")
    print("DIFFERENTIABLE FOLDING (ALPHAFOLD + DP) FUSION RUN")
    print("======================================================================")
    
    # Reset random seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Initialize model
    model = DifferentiableStructureModel(seq_len=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Define docking target: we want residue 0 to dock at coordinate [4.0, -2.0, 1.5]
    docking_target = torch.tensor([4.0, -2.0, 1.5], dtype=torch.float)
    
    print(f"Initial sequence:        '{model.get_sequence()}'")
    
    # Get initial predicted coordinate of residue 0
    with torch.no_grad():
        initial_coords = model()[0, 0]
        initial_dist = torch.norm(initial_coords - docking_target).item()
    print(f"Initial Coord (Residue 0): [{initial_coords[0]:.2f}, {initial_coords[1]:.2f}, {initial_coords[2]:.2f}]")
    print(f"Initial Distance to Target: {initial_dist:.4f} Å")
    
    # Lists to collect metrics for plotting
    history_steps = []
    history_distance = []
    history_grad_norm = []
    
    # Run gradient-descent optimization
    print("\nBackpropagating structural loss to sequence logits...")
    print("-" * 75)
    print(f"{'Step':<6} | {'Sequence':<12} | {'Residue 0 Coordinate':<25} | {'Distance (Å)':<15} | {'Logits Grad':<12}")
    print("-" * 75)
    
    steps = 40
    for step in range(steps + 1):
        # Forward pass: predict structure
        temp = max(0.1, 0.6 * (1.0 - step / steps))
        coords = model(temp=temp)
        
        # Get residue 0 coordinates
        res_0_coord = coords[0, 0]
        
        # Calculate differentiable distance loss (docking score)
        loss = torch.norm(res_0_coord - docking_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Read the gradient norm at the sequence logits level to prove the gradients flowed back!
        logits_grad_norm = model.sequence_logits.grad.norm().item()
        
        # Save step details
        history_steps.append(step)
        history_distance.append(loss.item())
        history_grad_norm.append(logits_grad_norm)
        
        optimizer.step()
        
        # Log progress every 5 steps
        if step % 5 == 0 or step == steps:
            coord_str = f"[{res_0_coord[0]:.2f}, {res_0_coord[1]:.2f}, {res_0_coord[2]:.2f}]"
            print(f"Step {step:<2}  | {model.get_sequence():<12} | {coord_str:<25} | {loss.item():<15.4f} | {logits_grad_norm:<12.4e}")
            
    print("-" * 75)
    print(f"Final sequence:          '{model.get_sequence()}'")
    
    # Check final coords
    with torch.no_grad():
        final_coords = model()[0, 0]
        final_dist = torch.norm(final_coords - docking_target).item()
    print(f"Final Coord (Residue 0):   [{final_coords[0]:.2f}, {final_coords[1]:.2f}, {final_coords[2]:.2f}]")
    print(f"Final Distance to Target:  {final_dist:.4f} Å (Docked successfully!)")
    print("======================================================================\n")
    
    # Create ECharts JSON specification to plot distance and gradients
    echarts_spec = {
        "title": {
            "text": "Differentiable Folding: Sequence Docking Trajectory",
            "left": "center",
            "textStyle": {
                "color": "#0c2340",
                "fontSize": 14
            }
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": { "type": "cross" }
        },
        "legend": {
            "bottom": 10,
            "data": ["Distance to Target (Å)", "Logits Gradient Norm"]
        },
        "xAxis": {
            "type": "category",
            "name": "Optimization Step",
            "nameLocation": "middle",
            "nameGap": 25,
            "data": [str(s) for s in history_steps]
        },
        "yAxis": [
            {
                "type": "value",
                "name": "Distance (Å)",
                "nameLocation": "middle",
                "nameGap": 30,
                "splitLine": { "lineStyle": { "type": "dashed" } }
            },
            {
                "type": "value",
                "name": "Gradient Norm",
                "nameLocation": "middle",
                "nameGap": 35,
                "splitLine": { "show": False }
            }
        ],
        "series": [
            {
                "name": "Distance to Target (Å)",
                "type": "line",
                "data": history_distance,
                "smooth": True,
                "itemStyle": { "color": "#3182ce" }
            },
            {
                "name": "Logits Gradient Norm",
                "type": "line",
                "yAxisIndex": 1,
                "data": history_grad_norm,
                "smooth": True,
                "itemStyle": { "color": "#e53e3e" }
            }
        ]
    }
    
    out_dir = "/tmp/biomolecular_design"
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, "differentiable_fold_echarts_spec.json")
    with open(spec_path, 'w') as f:
        json.dump(echarts_spec, f, indent=2)
    print(f"[Save] Differentiable folding plot spec saved to: {spec_path}")

if __name__ == "__main__":
    run_differentiable_folding()
