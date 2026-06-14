import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSequenceDesigner(nn.Module):
    """Demonstrates Differentiable Programming (DP) for protein sequence design.
    
    Instead of discrete mutational scanning, we parameterize the sequence as continuous 
    logits and use gradient descent to optimize them directly through a differentiable 
    biophysical scorer.
    """
    
    def __init__(self, sequence_length: int = 15, vocab_size: int = 20):
        super().__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        # Initialize sequence logits randomly. These represent the learnable sequence parameters.
        # Shape: [1, sequence_length, vocab_size]
        self.sequence_logits = nn.Parameter(torch.randn(1, sequence_length, vocab_size) * 0.1)
        
        # Amino acid vocabulary mapping
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def get_discrete_sequence(self) -> str:
        """Returns the current discrete sequence by taking the argmax of the logits."""
        with torch.no_grad():
            indices = torch.argmax(self.sequence_logits, dim=-1).squeeze(0)
            return "".join([self.alphabet[idx.item()] for idx in indices])

    def forward(self, temperature: float = 0.5) -> torch.Tensor:
        """Applies Gumbel-Softmax relaxation to obtain a continuous representation.
        
        Returns:
            One-hot-like probability distribution over the 20 amino acids.
            Shape: [1, sequence_length, vocab_size]. Fully differentiable.
        """
        # During training/optimization, we relax the discrete choice to allow gradients to flow
        return F.gumbel_softmax(self.sequence_logits, tau=temperature, hard=False)


class DifferentiableBiophysicalScorer:
    """A differentiable physical scorer representing target binding objectives.
    
    In a full production setting, this is replaced by backpropagating through a 
    differentiable structure model (e.g. Boltz-1) and a coordinate energy function.
    
    We simulate three differentiable physical forces:
    1. Hydrophobic Core Stability: Favors hydrophobic residues in the core (center).
    2. Interface Polar Complementarity: Favors polar residues at the interface ends.
    3. Structural Charge Balance: Minimizes overall sequence charge mismatch.
    """
    
    def __init__(self, sequence_length: int, alphabet: str):
        self.L = sequence_length
        self.alphabet = alphabet
        
        # Map amino acids to physical properties
        # Hydrophobic index: L, I, V, A, M, F, W, Y
        self.hydrophobic_mask = torch.tensor([1.0 if c in "LIVAMFWY" else 0.0 for c in alphabet])
        
        # Polar index: D, E, H, K, R, N, Q, S, T
        self.polar_mask = torch.tensor([1.0 if c in "DEHKRNQST" else 0.0 for c in alphabet])
        
        # Charge index: +1 for K, R, H; -1 for D, E; 0 others
        charges = []
        for c in alphabet:
            if c in "KRH":
                charges.append(1.0)
            elif c in "DE":
                charges.append(-1.0)
            else:
                charges.append(0.0)
        self.charge_values = torch.tensor(charges)

    def compute_energy(self, relaxed_seq: torch.Tensor) -> torch.Tensor:
        """Computes a differentiable biophysical energy score (lower is better).
        
        relaxed_seq shape: [1, L, 20] representing probability distribution.
        """
        energy = torch.tensor(0.0)
        
        # 1. Hydrophobic Core Energy:
        # We want hydrophobic residues in the middle positions (indices 4 to 10)
        core_probs = relaxed_seq[0, 4:11, :] # Shape: [7, 20]
        core_hydrophobicity = torch.matmul(core_probs, self.hydrophobic_mask) # Shape: [7]
        # Energy is minimized (negative) when hydrophobicity is maximized (close to 1.0)
        energy = energy - core_hydrophobicity.mean() * 3.0
        
        # 2. Polar Interface Energy:
        # We want polar/hydrophilic residues at the terminal positions (ends)
        end_probs = torch.cat([relaxed_seq[0, :3, :], relaxed_seq[0, -3:, :]], dim=0) # Shape: [6, 20]
        end_polarity = torch.matmul(end_probs, self.polar_mask) # Shape: [6]
        energy = energy - end_polarity.mean() * 2.5
        
        # 3. Differentiable Charge Balance constraint:
        # Total charge should be close to neutral (0.0) to prevent electrostatic repulsion
        sequence_charges = torch.matmul(relaxed_seq[0, :, :], self.charge_values) # Shape: [L]
        total_charge = sequence_charges.sum()
        charge_penalty = total_charge ** 2 # Quadratic penalty around 0
        energy = energy + charge_penalty * 0.8
        
        return energy


def run_differentiable_optimization():
    print("======================================================================")
    print("DIFFERENTIABLE PROGRAMMING SEQUENCE DESIGN SENSE CHECK")
    print("======================================================================")
    
    designer = DifferentiableSequenceDesigner(sequence_length=15)
    scorer = DifferentiableBiophysicalScorer(sequence_length=15, alphabet=designer.alphabet)
    
    # Setup standard PyTorch optimizer directly over the sequence logits!
    optimizer = torch.optim.Adam(designer.parameters(), lr=0.1)
    
    print(f"Initial discrete sequence: '{designer.get_discrete_sequence()}'")
    initial_relaxed = designer(temperature=0.8)
    initial_energy = scorer.compute_energy(initial_relaxed).item()
    print(f"Initial Continuous Energy:  {initial_energy:.4f}")
    
    print("\nRunning gradient descent optimization loop...")
    print("-" * 65)
    print(f"{'Step':<8} | {'Sequence':<18} | {'Energy (Loss)':<15} | {'Charge':<10}")
    print("-" * 65)
    
    steps = 40
    for step in range(steps + 1):
        # 1. Forward pass with Gumbel-Softmax continuous relaxation
        # Gradually decrease temperature (annealing) to converge to discrete selections
        temp = max(0.1, 0.8 * (1.0 - step / steps))
        relaxed_seq = designer(temperature=temp)
        
        # 2. Compute differentiable energy loss
        loss = scorer.compute_energy(relaxed_seq)
        
        # 3. Backpropagation & optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log progress every 5 steps
        if step % 5 == 0 or step == steps:
            discrete_seq = designer.get_discrete_sequence()
            # Calculate current total charge
            with torch.no_grad():
                probs = designer(temperature=0.1)[0]
                total_charge = torch.matmul(probs, scorer.charge_values).sum().item()
            print(f"Step {step:<3}   | {discrete_seq:<18} | {loss.item():<15.4f} | {total_charge:<10.2f}")
            
    print("-" * 65)
    print(f"\nFinal designed sequence: '{designer.get_discrete_sequence()}'")
    print("Optimization complete.")
    print("======================================================================")


if __name__ == "__main__":
    run_differentiable_optimization()
