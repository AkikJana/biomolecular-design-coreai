import os
import time
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SirenLayer(nn.Module):
    """
    A single SIREN (Sine Representation Network) layer.
    """
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = omega_0
        
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1 / in_features, 1 / in_features]
                self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
            else:
                # Hidden layers: uniform in [-sqrt(6/in_features) / omega_0, sqrt(6/in_features) / omega_0]
                limit = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)
            # Initialize bias to zero
            self.linear.bias.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """
    Siren MLP for coordinate mapping.
    """
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int, omega_0: float = 30.0):
        super().__init__()
        layers = []
        layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        
        # Output layer is standard linear to support arbitrary ranges
        self.output_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            limit = np.sqrt(6.0 / hidden_features) / omega_0
            self.output_linear.weight.uniform_(-limit, limit)
            self.output_linear.bias.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_linear(self.net(x))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (NeRF-style).
    Maps coordinates to multi-frequency sine and cosine representations.
    """
    def __init__(self, in_features: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Store frequencies as a buffer (non-trainable parameters)
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, in_features]
        out = []
        if self.include_input:
            out.append(x)
        
        # Calculate sin and cos features: x_expanded shape [N, 1, in_features] * frequencies shape [1, L, 1]
        x_expanded = x.unsqueeze(1) * self.frequencies.unsqueeze(0).unsqueeze(-1) * np.pi
        sin_feats = torch.sin(x_expanded).flatten(1)
        cos_feats = torch.cos(x_expanded).flatten(1)
        
        out.append(sin_feats)
        out.append(cos_feats)
        return torch.cat(out, dim=-1)


class StandardMLP(nn.Module):
    """
    A standard MLP with optional positional encoding and GELU activations.
    """
    def __init__(self, in_features: int, hidden_features: int, hidden_layers: int, out_features: int, use_pos_enc: bool = True, num_frequencies: int = 6):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(in_features, num_frequencies)
            input_dim = in_features + 2 * in_features * num_frequencies
        else:
            self.pos_enc = nn.Identity()
            input_dim = in_features
            
        layers = []
        layers.append(nn.Linear(input_dim, hidden_features))
        layers.append(nn.GELU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.output_linear = nn.Linear(hidden_features, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.output_linear(self.net(x))


class ReceptorNeuralField(nn.Module):
    """
    Implicit Neural Representation (INR) module for compressing target receptor spatial embeddings.
    Maps 3D coordinates [N, 3] to target representations [N, D_embed].
    """
    def __init__(self, 
                 in_features: int = 3, 
                 hidden_features: int = 64, 
                 hidden_layers: int = 2, 
                 out_features: int = 128, 
                 mode: str = "siren", 
                 omega_0: float = 30.0,
                 num_frequencies: int = 6):
        super().__init__()
        self.mode = mode.lower()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        
        if self.mode == "siren":
            self.model = SirenMLP(
                in_features=in_features, 
                hidden_features=hidden_features, 
                hidden_layers=hidden_layers, 
                out_features=out_features, 
                omega_0=omega_0
            )
        elif self.mode == "pe_mlp":
            self.model = StandardMLP(
                in_features=in_features, 
                hidden_features=hidden_features, 
                hidden_layers=hidden_layers, 
                out_features=out_features, 
                use_pos_enc=True, 
                num_frequencies=num_frequencies
            )
        elif self.mode == "basic_mlp":
            self.model = StandardMLP(
                in_features=in_features, 
                hidden_features=hidden_features, 
                hidden_layers=hidden_layers, 
                out_features=out_features, 
                use_pos_enc=False
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'siren', 'pe_mlp', 'basic_mlp'.")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def generate_mock_receptor(n_residues: int = 1000, d_embed: int = 128, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates mock coordinates representing a folding protein backbone in 3D space,
    and associated spatial embeddings containing structural information.
    
    Args:
        n_residues: Number of residues (N).
        d_embed: Dimensionality of target embedding representation (D_embed).
        seed: Random seed for reproducibility.
        
    Returns:
        coords: Tensor of normalized 3D coordinates [N, 3].
        target_embeddings: Tensor of target representations [N, D_embed].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Generate 3D coordinates on a complex winding curve
    t = np.linspace(-10, 10, n_residues)
    x = np.sin(t) + 0.2 * np.sin(5 * t)
    y = np.cos(t) + 0.2 * np.cos(5 * t)
    z = t / 5.0 + 0.2 * np.sin(3 * t)
    
    coords = np.stack([x, y, z], axis=1)
    
    # Scale coordinates to [-1, 1] range (critical for INR training stability)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords = 2.0 * (coords - coords_min) / (coords_max - coords_min + 1e-8) - 1.0
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    
    # 2. Generate target spatial embeddings combining coordinates frequencies and distance-based features
    # Coordinate frequency waves
    freqs = torch.randn(d_embed, 3) * 2.0
    phases = torch.rand(d_embed) * 2.0 * np.pi
    target_embeddings = torch.sin(coords_tensor @ freqs.t() + phases.unsqueeze(0))
    
    # Add distance-based "hotspot" pocket features (local spatial correlations)
    hotspot_indices = [int(n_residues * pct) for pct in [0.1, 0.3, 0.5, 0.7, 0.9]]
    hotspots = coords_tensor[hotspot_indices]
    dists = torch.cdist(coords_tensor, hotspots)  # [N, 5]
    proj_matrix = torch.randn(5, d_embed) * 0.5
    target_embeddings = target_embeddings + dists @ proj_matrix
    
    # Normalize targets to mean=0, std=1
    target_embeddings = (target_embeddings - target_embeddings.mean(dim=0, keepdim=True)) / (target_embeddings.std(dim=0, keepdim=True) + 1e-8)
    
    return coords_tensor, target_embeddings


def train_inr(
    model: nn.Module, 
    coords: torch.Tensor, 
    targets: torch.Tensor, 
    epochs: int = 2000, 
    lr: float = 1e-3, 
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, List[float], float]:
    """
    Trains the INR model to overfit the target representations using L2 (MSE) loss.
    """
    model = model.to(device)
    coords = coords.to(device)
    targets = targets.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(coords)
        loss = criterion(preds, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            # Print intermediate progress (can be silenced if called inside a loop)
            pass
            
    training_time = time.time() - start_time
    return model, loss_history, training_time


def evaluate_inr(model: nn.Module, coords: torch.Tensor, targets: torch.Tensor, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """
    Evaluates the trained model on reconstruction fidelity metrics and computes memory statistics.
    """
    model.eval()
    coords = coords.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        preds = model(coords)
        
        # 1. MSE (L2 Loss)
        mse = nn.functional.mse_loss(preds, targets).item()
        
        # 2. Cosine Similarity
        # Compute cosine similarity for each row (residue) and take the average
        cos_sim = nn.functional.cosine_similarity(preds, targets, dim=1).mean().item()
        
        # 3. R2 Score (coefficient of determination)
        target_mean = targets.mean(dim=0, keepdim=True)
        ss_tot = torch.sum((targets - target_mean) ** 2).item()
        ss_res = torch.sum((targets - preds) ** 2).item()
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        
    # 4. Model size in parameters and bytes
    num_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 5. Raw tensor size in bytes
    raw_size_bytes = targets.numel() * targets.element_size()
    
    # 6. VRAM compression factor
    compression_factor = raw_size_bytes / model_size_bytes
    
    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "r2_score": r2,
        "num_params": num_params,
        "model_size_kb": model_size_bytes / 1024.0,
        "raw_size_kb": raw_size_bytes / 1024.0,
        "compression_factor": compression_factor,
    }


def run_experiment(
    n_residues: int = 1000, 
    d_embed: int = 128, 
    epochs: int = 2000, 
    lr: float = 1e-3, 
    device_name: str = "auto"
) -> Dict[str, Dict[str, Any]]:
    """
    Runs compression experiments using multiple models (Siren, PE-MLP, Basic MLP) and various network sizes.
    """
    # Select device
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        
    print(f"Running experiments on device: {device}")
    
    # Generate mock receptor structure
    coords, targets = generate_mock_receptor(n_residues=n_residues, d_embed=d_embed)
    
    # Define configurations to test
    # (name, mode, hidden_features, hidden_layers, lr)
    configs = [
        ("SIREN (Light - H=48, L=2)", "siren", 48, 2, 1e-3),
        ("SIREN (Medium - H=64, L=3)", "siren", 64, 3, 1e-3),
        ("SIREN (Heavy - H=128, L=3)", "siren", 128, 3, 5e-4),
        ("PE-MLP (Medium - H=64, L=3)", "pe_mlp", 64, 3, 1e-3),
        ("Basic MLP (Medium - H=64, L=3)", "basic_mlp", 64, 3, 1e-3),
    ]
    
    results = {}
    
    for name, mode, hidden_features, hidden_layers, config_lr in configs:
        print(f"\n--- Training {name} ---")
        model = ReceptorNeuralField(
            in_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=d_embed,
            mode=mode
        )
        
        # Train
        model, loss_history, train_time = train_inr(
            model=model,
            coords=coords,
            targets=targets,
            epochs=epochs,
            lr=config_lr,
            device=device
        )
        
        # Evaluate
        eval_metrics = evaluate_inr(model, coords, targets, device)
        eval_metrics["training_time"] = train_time
        eval_metrics["final_loss"] = loss_history[-1]
        
        results[name] = eval_metrics
        
        print(f"  Final Loss: {eval_metrics['final_loss']:.5f}")
        print(f"  MSE: {eval_metrics['mse']:.5f}")
        print(f"  Cosine Similarity: {eval_metrics['cosine_similarity']:.4f}")
        print(f"  R2 Score: {eval_metrics['r2_score']:.4f}")
        print(f"  Compression Factor: {eval_metrics['compression_factor']:.2f}x (Model: {eval_metrics['model_size_kb']:.1f} KB vs Raw: {eval_metrics['raw_size_kb']:.1f} KB)")
        print(f"  Training Time: {train_time:.1f}s")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implicit Neural Representation (INR) Compression for Receptors")
    parser.add_argument("--n-residues", type=int, default=1000, help="Number of mock residues")
    parser.add_argument("--d-embed", type=int, default=128, help="Dimensionality of embeddings")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device to run training (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    run_experiment(
        n_residues=args.n_residues,
        d_embed=args.d_embed,
        epochs=args.epochs,
        lr=args.lr,
        device_name=args.device
    )
