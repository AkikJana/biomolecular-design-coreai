import os
import random
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Vocab and Tokenizer for Amino Acid Sequences
# ==========================================
class AASequenceTokenizer:
    """
    Tokenizer for standard amino acid sequences.
    Maps characters to integer tokens and handles special tokens.
    """
    def __init__(self):
        # 20 standard amino acids + 4 special tokens
        self.vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + list("ACDEFGHIKLMNPQRSTVWY")
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.char_to_id["<pad>"]
        self.unk_token_id = self.char_to_id["<unk>"]
        
    def encode(self, sequence: str) -> torch.Tensor:
        ids = [self.char_to_id.get(char, self.unk_token_id) for char in sequence]
        return torch.tensor(ids, dtype=torch.long)
        
    def decode(self, ids: torch.Tensor) -> str:
        return "".join([self.id_to_char.get(int(idx), "?") for idx in ids])

# ==========================================
# 2. Linear Union Mask Clustering
# ==========================================
def linear_union_mask_clustering(sequences, max_union_size=5):
    """
    Greedy single-pass (linear complexity O(N*K)) Union Mask Clustering.
    Groups candidate sequences by shared mutational positions relative to a reference.
    
    Parameters:
      - sequences: List[str] of sequences (assumed aligned/same length).
      - max_union_size: int, threshold limit for union mask size of a cluster.
      
    Returns:
      - clusters: List[dict] where each dict contains:
            "indices": List[int] indices of sequences in the cluster
            "union_mask": set of mutated sequence positions
    """
    if not sequences:
        return []
        
    # Choose first sequence as reference to define mutation positions
    ref_seq = sequences[0]
    seq_len = len(ref_seq)
    
    # Compute mutation masks relative to reference sequence
    masks = []
    for seq in sequences:
        # Check alignment length matching
        assert len(seq) == seq_len, "All sequences in target must be aligned (same length)."
        mask = {i for i, (c1, c2) in enumerate(zip(seq, ref_seq)) if c1 != c2}
        masks.append(mask)
        
    clusters = []
    for idx, mask in enumerate(masks):
        placed = False
        # Try to fit sequence in an existing cluster without exceeding max_union_size
        for cl in clusters:
            new_union = cl["union_mask"].union(mask)
            if len(new_union) <= max_union_size:
                cl["indices"].append(idx)
                cl["union_mask"] = new_union
                placed = True
                break
        # If no cluster fits, create a new cluster
        if not placed:
            clusters.append({
                "indices": [idx],
                "union_mask": set(mask)
            })
            
    return clusters

# ==========================================
# 3. PyTorch Dataset Classes
# ==========================================
class TeddymerDataset(Dataset):
    """
    Raw Teddymer annotations dataset processing:
    (binder sequence, target name, binding affinity label, model logits)
    """
    def __init__(self, raw_data_list):
        self.data = raw_data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


class ClusteredTeddymerDataset(Dataset):
    """
    Dataset that groups candidates by target using linear Union Mask Clustering,
    returning clustered sequences for preference learning.
    """
    def __init__(self, raw_data_list, tokenizer, max_union_size=5):
        self.raw_data = raw_data_list
        self.tokenizer = tokenizer
        
        # Tokenize sequences in raw data
        for item in self.raw_data:
            item["tokens"] = self.tokenizer.encode(item["binder_sequence"])
            
        # Group candidates by target and then perform Union Mask Clustering
        self.clusters = self.cluster_data(max_union_size)
        
    def cluster_data(self, max_union_size):
        # Group raw data items by target name
        target_groups = {}
        for item in self.raw_data:
            target = item["target_name"]
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(item)
            
        all_clusters = []
        # Cluster within each target group
        for target, group in target_groups.items():
            sequences = [item["binder_sequence"] for item in group]
            clusters_meta = linear_union_mask_clustering(sequences, max_union_size)
            
            for cl in clusters_meta:
                cluster_items = [group[idx] for idx in cl["indices"]]
                # For preference training (like best-vs-all), a cluster needs at least 2 sequences
                if len(cluster_items) >= 2:
                    all_clusters.append(cluster_items)
                    
        return all_clusters
        
    def __len__(self):
        return len(self.clusters)
        
    def __getitem__(self, idx):
        # Return the list of items in the cluster
        return self.clusters[idx]


def collate_cluster_fn(batch):
    """
    Collate function to prepare a clustered batch for the model.
    Since DataLoader batch_size is set to 1, the batch contains a single cluster.
    """
    cluster = batch[0] # Extract the cluster (list of dicts)
    
    # Stack features into batched tensors
    tokens = torch.stack([item["tokens"] for item in cluster], dim=0) # (M, L)
    affinities = torch.tensor([item["binding_affinity"] for item in cluster], dtype=torch.float32) # (M,)
    model_logits = torch.stack([item["model_logits"] for item in cluster], dim=0) # (M, L, vocab_size)
    
    return {
        "tokens": tokens,
        "binding_affinities": affinities,
        "model_logits": model_logits,
        "target_name": cluster[0]["target_name"]
    }

# ==========================================
# 4. Policy Network Model
# ==========================================
class PolicyNetwork(nn.Module):
    """
    Bidirectional GRU Policy Network for Sequence Models.
    Computes sequence token log-probabilities.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        # x shape: (B, L)
        emb = self.embedding(x) # (B, L, embed_dim)
        out, _ = self.gru(emb)  # (B, L, hidden_dim * 2)
        logits = self.fc(out)   # (B, L, vocab_size)
        return logits

# ==========================================
# 5. Log-Probability and Loss Functions
# ==========================================
def get_sequence_logps(logits, tokens, length_normalize=True):
    """
    Computes sequence log-probabilities from token-level logits.
    
    Parameters:
      - logits: torch.Tensor of shape (B, L, vocab_size)
      - tokens: torch.Tensor of shape (B, L)
      - length_normalize: bool, normalize by sequence length (used in SimPO)
      
    Returns:
      - seq_logps: torch.Tensor of shape (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1) # (B, L, vocab_size)
    # Gather logits corresponding to the actual tokens
    seq_logps = torch.gather(log_probs, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1) # (B, L)
    sum_logps = seq_logps.sum(dim=-1) # (B,)
    
    if length_normalize:
        return sum_logps / tokens.shape[1]
    return sum_logps


def simpo_loss(chosen_logps, rejected_logps, beta=2.0, gamma=0.5):
    """
    SimPO (Simple Preference Optimization) margin loss.
    Loss = -log_sigmoid(beta * logpi_chosen - beta * logpi_rejected - gamma)
    """
    logits = beta * chosen_logps - beta * rejected_logps - gamma
    loss = -F.logsigmoid(logits)
    return loss.mean()


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Standard DPO loss function for reference.
    """
    logits = beta * (policy_chosen_logps - ref_chosen_logps) - beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(logits)
    return loss.mean()


def grpo_loss(policy_logps, old_logps, rewards, beta=0.1, clip_eps=0.2):
    """
    DeepSeek-style GRPO training loss.
    
    Parameters:
      - policy_logps: torch.Tensor of shape (G,) - active policy sequence log-probabilities
      - old_logps: torch.Tensor of shape (G,) - detached reference/old policy log-probabilities
      - rewards: torch.Tensor of shape (G,) - biophysical rewards
      - beta: float - KL penalty coefficient
      - clip_eps: float - PPO clip epsilon
      
    Returns:
      - loss: scalar tensor
      - kl: scalar tensor (mean KL divergence)
      - advantages: torch.Tensor of shape (G,) - standardized advantages
    """
    # Advantage calculation: standardize rewards across group
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean_r) / std_r
    
    # Policy ratio calculation
    ratios = torch.exp(policy_logps - old_logps)
    
    # Clipped surrogate loss
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    clip_loss = torch.min(surr1, surr2)
    
    # Reference-free KL divergence penalty:
    # KL = exp(old_logps - policy_logps) - (old_logps - policy_logps) - 1
    log_ratio = old_logps - policy_logps
    kl = torch.exp(log_ratio) - log_ratio - 1
    
    # Total loss (negative to maximize surrogate objective)
    loss = -(clip_loss - beta * kl).mean()
    
    return loss, kl.mean(), advantages


# ==========================================
# 6. Memory and VRAM Monitoring
# ==========================================
def get_memory_usage():
    """
    Monitors RAM and GPU VRAM consumption.
    """
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / (1024 * 1024) # MB
    
    gpu_mem = 0.0
    device_type = "cpu"
    if torch.backends.mps.is_available():
        gpu_mem = torch.mps.current_allocated_memory() / (1024 * 1024) # MB
        device_type = "mps"
    elif torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) # MB
        device_type = "cuda"
        
    return cpu_mem, gpu_mem, device_type

# ==========================================
# 7. Simulated Teddymer Data Generator
# ==========================================
def generate_simulated_teddymer_annotations(num_targets=5, seqs_per_target=20, seq_len=30, vocab_size=24):
    """
    Simulates realistic annotations for Teddymer.
    Each target has a wild-type sequence and candidates derived from it via mutations.
    """
    tokenizer = AASequenceTokenizer()
    # Exclude special tokens from wild-type generation
    aa_tokens = tokenizer.vocab[4:]
    
    data_list = []
    random.seed(42)
    torch.manual_seed(42)
    
    for t_idx in range(num_targets):
        target_name = f"Target_{t_idx:03d}"
        wt_seq = "".join(random.choices(aa_tokens, k=seq_len))
        
        # Define mutation hotspots
        sensitive_positions = random.sample(range(seq_len), k=min(6, seq_len))
        beneficial_mutations = {pos: random.choice(aa_tokens) for pos in sensitive_positions}
        
        for s_idx in range(seqs_per_target):
            # Mutate 1 to 4 positions
            num_mutations = random.randint(1, 4)
            mut_positions = random.sample(range(seq_len), k=num_mutations)
            
            mut_seq_list = list(wt_seq)
            affinity_effect = 0.0
            
            for pos in mut_positions:
                sub_char = random.choice(aa_tokens)
                while sub_char == wt_seq[pos]:
                    sub_char = random.choice(aa_tokens)
                mut_seq_list[pos] = sub_char
                
                # Mutation effects on binding affinity
                if pos in sensitive_positions:
                    if sub_char == beneficial_mutations[pos]:
                        affinity_effect += 2.0 # High improvement
                    else:
                        affinity_effect -= 1.2 # Severe drop
                else:
                    affinity_effect -= 0.1 # Slight baseline penalty
            
            binder_sequence = "".join(mut_seq_list)
            base_affinity = 3.0 + random.normalvariate(0, 0.4)
            binding_affinity = max(0.0, base_affinity + affinity_effect)
            
            # Generate reference model logits
            tokens = tokenizer.encode(binder_sequence)
            model_logits = torch.randn(seq_len, vocab_size) * 0.4
            for i, tok_id in enumerate(tokens):
                model_logits[i, tok_id] += 2.5 # Model bias for true tokens
                
            data_list.append({
                "binder_sequence": binder_sequence,
                "target_name": target_name,
                "binding_affinity": binding_affinity,
                "model_logits": model_logits
            })
            
    return data_list

# ==========================================
# 8. Training Loop
# ==========================================
def train_preference_alignment(
    epochs=5,
    max_union_size=4,
    beta=2.0,
    gamma=0.5,
    lr=1e-3,
    use_simpo=True,
    use_grpo=False,
    grpo_beta=0.1,
    grpo_clip_eps=0.2
):
    print("==================================================")
    print("Initializing Preference Alignment Training Pipeline")
    print("==================================================")
    
    # 1. Setup tokenizer
    tokenizer = AASequenceTokenizer()
    
    # 2. Generate simulated annotations
    print("\n[Step 1] Simulating Teddymer annotations...")
    raw_data = generate_simulated_teddymer_annotations(
        num_targets=10,
        seqs_per_target=25,
        seq_len=30,
        vocab_size=tokenizer.vocab_size
    )
    print(f"Generated {len(raw_data)} sequence annotations across 10 targets.")
    
    # 3. Create clustered dataset
    print("\n[Step 2] Clustering candidates via linear Union Mask Clustering...")
    t_start = time.perf_counter()
    clustered_dataset = ClusteredTeddymerDataset(
        raw_data, 
        tokenizer, 
        max_union_size=max_union_size
    )
    t_end = time.perf_counter()
    
    # Calculate statistics
    num_clusters = len(clustered_dataset)
    total_seqs_in_clusters = sum(len(c) for c in clustered_dataset.clusters)
    print(f"Clustering completed in {(t_end - t_start)*1000:.2f} ms.")
    print(f"Created {num_clusters} training clusters (excluding singletons).")
    print(f"Total sequences preserved in clusters: {total_seqs_in_clusters}/{len(raw_data)}")
    print(f"Average cluster size: {total_seqs_in_clusters / num_clusters:.2f} sequences.")
    
    # 4. DataLoader (batch_size=1 represents one cluster per training iteration)
    dataloader = DataLoader(
        clustered_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_cluster_fn
    )
    
    # 5. Model initialization
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n[Step 3] Initializing policy model on device: {device}")
    policy_model = PolicyNetwork(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2
    ).to(device)
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Measure baseline memory
    cpu_base, gpu_base, dev_name = get_memory_usage()
    print(f"Baseline System Memory: CPU {cpu_base:.2f} MB | {dev_name.upper()} VRAM {gpu_base:.2f} MB")
    
    loss_name = "GRPO" if use_grpo else ("SimPO" if use_simpo else "DPO")
    print(f"\n[Step 4] Starting training loop using {loss_name} loss...")
    
    metrics = []
    
    for epoch in range(epochs):
        policy_model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0 # How often chosen sequence logps > rejected sequence logps
        total_pairs = 0
        
        t_epoch_start = time.perf_counter()
        
        for step, batch in enumerate(dataloader):
            # Retrieve features
            tokens = batch["tokens"].to(device) # (M, L)
            affinities = batch["binding_affinities"].to(device) # (M,)
            ref_logits = batch["model_logits"].to(device) # (M, L, vocab_size)
            
            # Policy forward pass
            policy_logits = policy_model(tokens) # (M, L, vocab_size)
            policy_logps = get_sequence_logps(policy_logits, tokens, length_normalize=(use_simpo or use_grpo)) # (M,)
            
            # Identify the single best candidate (highest binding affinity label)
            best_idx = torch.argmax(affinities).item()
            chosen_logps = policy_logps[best_idx].expand(tokens.shape[0] - 1)
            
            # Extract rejected candidates' logprobs (all candidates except best)
            rejected_indices = [i for i in range(tokens.shape[0]) if i != best_idx]
            rejected_logps = policy_logps[rejected_indices]
            
            # Compute loss and optimize
            if use_grpo:
                old_logps = policy_logps.detach()
                for inner_step in range(3):
                    if inner_step > 0:
                        policy_logits = policy_model(tokens)
                        policy_logps = get_sequence_logps(policy_logits, tokens, length_normalize=True)
                    loss, kl_mean, advantages = grpo_loss(
                        policy_logps=policy_logps,
                        old_logps=old_logps,
                        rewards=affinities,
                        beta=grpo_beta,
                        clip_eps=grpo_clip_eps
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Recompute chosen_logps and rejected_logps for metrics
                chosen_logps = policy_logps[best_idx].expand(tokens.shape[0] - 1)
                rejected_logps = policy_logps[rejected_indices]
            else:
                if use_simpo:
                    # SimPO loss
                    loss = simpo_loss(chosen_logps, rejected_logps, beta=beta, gamma=gamma)
                else:
                    # Standard DPO loss
                    ref_logps = get_sequence_logps(ref_logits, tokens, length_normalize=False)
                    ref_chosen_logps = ref_logps[best_idx].expand(tokens.shape[0] - 1)
                    ref_rejected_logps = ref_logps[rejected_indices]
                    loss = dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Record metrics
            epoch_loss += loss.item()
            # Calculate implicit alignment accuracy (how often chosen_prob > rejected_prob)
            correct_pairs = (chosen_logps > rejected_logps).float().sum().item()
            epoch_accuracy += correct_pairs
            total_pairs += len(rejected_indices)
            
        t_epoch_end = time.perf_counter()
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = (epoch_accuracy / total_pairs) * 100 if total_pairs > 0 else 0.0
        
        # Monitor RAM/VRAM
        cpu_mem, gpu_mem, _ = get_memory_usage()
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss: {avg_loss:.5f} | "
              f"Alignment Acc: {avg_acc:.2f}% | "
              f"CPU: {cpu_mem:.1f} MB | "
              f"{dev_name.upper()} VRAM: {gpu_mem:.2f} MB | "
              f"Time: {t_epoch_end - t_epoch_start:.2f}s")
              
        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "cpu_mem_mb": cpu_mem,
            "vram_mem_mb": gpu_mem
        })
        
    print("\n==================================================")
    print("Preference Alignment Training Completed Successfully!")
    print("==================================================")
    
    return metrics

if __name__ == "__main__":
    import sys
    use_grpo = "--grpo" in sys.argv
    train_preference_alignment(epochs=5, max_union_size=4, use_simpo=not use_grpo, use_grpo=use_grpo)

