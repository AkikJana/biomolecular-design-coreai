import math
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SinusoidalEmbedding(nn.Module):
    """
    Computes sinusoidal positional embeddings for scalar values (e.g., time steps, CFG scales).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: scalar, (B,) or (B, 1)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.squeeze(-1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class StructureAwareBlock(nn.Module):
    """
    A geometry-aware attention layer that updates node features using distance-based bias.
    This simulates structural blocks in networks like Boltz-1/2 or AlphaFold 3.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Distance bias projection
        self.dist_proj = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.SiLU(),
            nn.Linear(num_heads, num_heads)
        )
        
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h: Node features, shape (B, N, dim)
        x: Coordinate tensor, shape (B, N, 3)
        """
        B, N, C = h.shape
        
        # Layer Normalization
        h_norm = self.norm1(h)
        
        # Q, K, V projections
        q = self.q_proj(h_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, head_dim)
        k = self.k_proj(h_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, head_dim)
        v = self.v_proj(h_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, head_dim)
        
        # Self-attention dot products
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, N, N)
        
        # Pairwise distance matrix calculation
        # dists shape: (B, N, N, 1)
        dists = torch.cdist(x, x, p=2).unsqueeze(-1)
        dist_bias = self.dist_proj(dists) # (B, N, N, H)
        dist_bias = dist_bias.permute(0, 3, 1, 2) # (B, H, N, N)
        
        # Modulate attention scores by physical distance
        # Farther residues get penalized attention (subtracted bias)
        scores = scores - dist_bias.abs()
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        h = h + self.out_proj(out)
        h = h + self.ffn(self.norm2(h))
        return h

class TeacherVectorField(nn.Module):
    """
    Teacher network representing the flow matching vector field model.
    Supports Classifier-Free Guidance (CFG) via sequence feature dropout (cond_mask).
    """
    def __init__(self, node_dim: int, seq_dim: int, time_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.seq_dim = seq_dim
        self.time_dim = time_dim
        
        # Learnable null token for unconditional forward pass
        self.null_seq_emb = nn.Parameter(torch.zeros(1, 1, seq_dim))
        nn.init.normal_(self.null_seq_emb, std=0.02)
        
        # Timestep projection
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Input layers
        self.coord_proj = nn.Linear(3, node_dim)
        self.seq_proj = nn.Linear(seq_dim, node_dim)
        
        # Transformer-like structural layers
        self.layers = nn.ModuleList([
            StructureAwareBlock(node_dim) for _ in range(3)
        ])
        
        # Coordinate update head
        self.out_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: Coordinate tensor, shape (B, N, 3)
        t: Timestep tensor, shape (B,) or (B, 1)
        c: Sequence features tensor, shape (B, N, seq_dim)
        cond_mask: Optional mask tensor of 1s and 0s, shape (B,) or (B, 1). 
                   1 = Conditional forward pass, 0 = Unconditional forward pass.
        """
        B, N, _ = x.shape
        
        # Handle conditional vs unconditional masking
        if cond_mask is not None:
            if cond_mask.ndim == 1:
                cond_mask = cond_mask.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
            elif cond_mask.ndim == 2:
                cond_mask = cond_mask.unsqueeze(-1) # (B, 1, 1)
            
            # Interpolate sequence features and null token based on cond_mask
            null_features = self.null_seq_emb.expand(B, N, -1)
            c = torch.where(cond_mask > 0.5, c, null_features)
            
        # Embed timestep
        t_embed = self.time_emb(t).unsqueeze(1) # (B, 1, node_dim)
        
        # Build node features
        h_coords = self.coord_proj(x)
        h_seq = self.seq_proj(c)
        h = h_coords + h_seq + t_embed
        
        # Apply blocks
        for layer in self.layers:
            h = layer(h, x)
            
        # Predict vector field (coordinate updates)
        v = self.out_head(h)
        return v

class CFGDistilledVectorField(nn.Module):
    """
    Distilled student network that accepts coordinate scale `s` as input.
    Learns to output the guided vector field in a single forward pass.
    """
    def __init__(self, node_dim: int, seq_dim: int, time_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.seq_dim = seq_dim
        self.time_dim = time_dim
        
        # Combined embedding for timestep (t) and guidance scale (s)
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.scale_emb = SinusoidalEmbedding(time_dim)
        self.time_scale_mlp = nn.Sequential(
            nn.Linear(time_dim * 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Input layers
        self.coord_proj = nn.Linear(3, node_dim)
        self.seq_proj = nn.Linear(seq_dim, node_dim)
        
        # Structural backbone
        self.layers = nn.ModuleList([
            StructureAwareBlock(node_dim) for _ in range(3)
        ])
        
        # Coordinate update head
        self.out_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        x: Coordinate tensor, shape (B, N, 3)
        t: Timestep tensor, shape (B,) or (B, 1)
        c: Sequence features tensor, shape (B, N, seq_dim)
        s: Conditioning scale tensor, shape (B,) or (B, 1)
        """
        B, N, _ = x.shape
        
        # Embed t and s together
        t_feat = self.time_emb(t)
        s_feat = self.scale_emb(s)
        ts_embed = torch.cat([t_feat, s_feat], dim=-1)
        ts_embed = self.time_scale_mlp(ts_embed).unsqueeze(1) # (B, 1, node_dim)
        
        # Node features
        h_coords = self.coord_proj(x)
        h_seq = self.seq_proj(c)
        h = h_coords + h_seq + ts_embed
        
        # Apply blocks
        for layer in self.layers:
            h = layer(h, x)
            
        # Predict distilled vector field
        v = self.out_head(h)
        return v

def initialize_student_from_teacher(student: CFGDistilledVectorField, teacher: TeacherVectorField) -> None:
    """
    Initialize student backbone and input/output weights using a pre-trained teacher.
    This stabilizes training and speeds up convergence during distillation.
    """
    student.coord_proj.load_state_dict(teacher.coord_proj.state_dict())
    student.seq_proj.load_state_dict(teacher.seq_proj.state_dict())
    student.layers.load_state_dict(teacher.layers.state_dict())
    student.out_head.load_state_dict(teacher.out_head.state_dict())

class SyntheticStructureDataset(Dataset):
    """
    Generates realistic 3D helical coordinates with sequence features.
    Used for training and evaluating structural flow matching models.
    """
    def __init__(self, num_samples: int = 1000, num_residues: int = 32, seq_dim: int = 16):
        self.num_samples = num_samples
        self.num_residues = num_residues
        self.seq_dim = seq_dim
        
        self.coords_list = []
        self.feats_list = []
        
        for _ in range(num_samples):
            # Helix generation parameters
            t_seq = torch.linspace(0, 4 * math.pi, num_residues)
            r = 1.0 + 0.1 * torch.randn(1)
            pitch = 0.2 + 0.05 * torch.randn(1)
            x = r * torch.cos(t_seq)
            y = r * torch.sin(t_seq)
            z = pitch * t_seq
            coords = torch.stack([x, y, z], dim=-1) # (N, 3)
            
            # Apply random rotation and translation to coordinates
            R = torch.randn(3, 3)
            U, S, V = torch.svd(R)
            R = torch.matmul(U, V.t())
            if torch.det(R) < 0:
                R[:, 0] *= -1
            coords = torch.matmul(coords, R.t()) + torch.randn(1, 3)
            
            # Synthetic features
            seq_feats = torch.randn(num_residues, seq_dim)
            
            self.coords_list.append(coords)
            self.feats_list.append(seq_feats)
            
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords_list[idx], self.feats_list[idx]

def train_teacher_model(
    teacher: TeacherVectorField,
    dataloader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    p_uncond: float = 0.2
) -> list:
    """
    Pre-trains the teacher model on the flow matching task.
    Includes random conditioning dropout (p_uncond) to learn conditional and unconditional fields.
    """
    teacher.to(device)
    teacher.train()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    loss_history = []
    
    print(f"Pre-training teacher model for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x1, batch_c in dataloader:
            batch_x1 = batch_x1.to(device)
            batch_c = batch_c.to(device)
            B, N, _ = batch_x1.shape
            
            # Sample flow matching state
            t = torch.rand(B, device=device)
            x0 = torch.randn_like(batch_x1)
            t_expanded = t.view(B, 1, 1)
            xt = t_expanded * batch_x1 + (1.0 - t_expanded) * x0
            
            # Target is the straight flow vector (x1 - x0)
            v_target = batch_x1 - x0
            
            # Random dropout for unconditioned path
            cond_mask = (torch.rand(B, device=device) > p_uncond).float()
            
            v_pred = teacher(xt, t, batch_c, cond_mask=cond_mask)
            
            loss = F.huber_loss(v_pred, v_target, delta=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Dynamic cache clearing
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Teacher Epoch {epoch+1:02d}/{epochs:02d} | Avg Loss: {avg_loss:.6f}")
            
    return loss_history

def train_distilled_model(
    teacher: TeacherVectorField,
    student: CFGDistilledVectorField,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    guidance_range: Tuple[float, float] = (0.0, 4.0),
    use_user_cfg_formula: bool = True
) -> list:
    """
    Performs CFG Distillation. The student learns to approximate the CFG guided vector field.
    
    Guidance formula options:
      - If use_user_cfg_formula: v_guided = v_cond + s * (v_cond - v_uncond)
      - Else (Standard CFG): v_guided = v_uncond + (1 + s) * (v_cond - v_uncond)
    """
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    student.to(device)
    student.train()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    loss_history = []
    
    print(f"Starting distillation training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x1, batch_c in dataloader:
            batch_x1 = batch_x1.to(device)
            batch_c = batch_c.to(device)
            B, N, _ = batch_x1.shape
            
            # 1. Sample flow matching timestep t ~ U(0, 1) and random noise x0 ~ N(0, I)
            t = torch.rand(B, device=device)
            x0 = torch.randn_like(batch_x1)
            
            # 2. Linear flow matching interpolation: xt = t * x1 + (1 - t) * x0
            t_expanded = t.view(B, 1, 1)
            xt = t_expanded * batch_x1 + (1.0 - t_expanded) * x0
            
            # 3. Sample scale s
            s = torch.rand(B, device=device) * (guidance_range[1] - guidance_range[0]) + guidance_range[0]
            s_expanded = s.view(B, 1, 1)
            
            # 4. Compute teacher outputs (conditional and unconditional)
            cond_mask_1 = torch.ones(B, device=device)
            cond_mask_0 = torch.zeros(B, device=device)
            
            v_cond = teacher(xt, t, batch_c, cond_mask=cond_mask_1)
            v_uncond = teacher(xt, t, batch_c, cond_mask=cond_mask_0)
            
            # 5. Compute target guided vector field
            if use_user_cfg_formula:
                # user requested: v_cond + s * (v_cond - v_uncond)
                v_guided = v_cond + s_expanded * (v_cond - v_uncond)
            else:
                # Standard CFG: v_uncond + (1 + s) * (v_cond - v_uncond)
                v_guided = v_uncond + (1.0 + s_expanded) * (v_cond - v_uncond)
                
            # 6. Predict using student network
            v_pred = student(xt, t, batch_c, s)
            
            # Huber loss is more stable for coordinate trajectories
            loss = F.huber_loss(v_pred, v_guided, delta=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Dynamic cache clearing
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Avg Loss: {avg_loss:.6f}")
            
    return loss_history

@torch.no_grad()
def sample_trajectory_euler(
    model: nn.Module,
    x0: torch.Tensor,
    c: torch.Tensor,
    s_val: float,
    steps: int = 20,
    device: torch.device = torch.device("cpu"),
    is_student: bool = True,
    use_user_cfg_formula: bool = True
) -> torch.Tensor:
    """
    Integrates the flow matching ODE using simple Euler steps.
    
    Returns:
        The generated structure at t = 1.0, shape (B, N, 3)
    """
    B, N, _ = x0.shape
    xt = x0.clone().to(device)
    c = c.to(device)
    
    dt = 1.0 / steps
    
    for step in range(steps):
        t_val = step / steps
        t = torch.full((B,), t_val, device=device)
        
        if is_student:
            # Single forward pass for student
            s = torch.full((B,), s_val, device=device)
            vt = model(xt, t, c, s)
        else:
            # Two forward passes for teacher (CFG)
            cond_mask_1 = torch.ones(B, device=device)
            cond_mask_0 = torch.zeros(B, device=device)
            
            v_cond = model(xt, t, c, cond_mask=cond_mask_1)
            v_uncond = model(xt, t, c, cond_mask=cond_mask_0)
            
            if use_user_cfg_formula:
                vt = v_cond + s_val * (v_cond - v_uncond)
            else:
                vt = v_uncond + (1.0 + s_val) * (v_cond - v_uncond)
                
        xt = xt + dt * vt
        
    return xt
