import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from boltz_wrapper import BoltzModelWrapper, BoltzDraftModelWrapper

def test_boltz_wrappers():
    print("=== Testing Boltz-1/2 Model Wrappers ===")
    
    # 1. Initialize models
    model = BoltzModelWrapper(use_gpu=False)
    draft_model = BoltzDraftModelWrapper(steps=10, use_gpu=False)
    
    # 2. Setup inputs (using the TNF-alpha structure file downloaded in DMS test)
    sequence = "MATEVLADIGSAKLR"
    target_pdb = "/tmp/biomolecular_design/TNF-alpha_1TNF.pdb"
    
    # Verify target exists
    if not os.path.exists(target_pdb):
        print(f"[Warning] Target PDB '{target_pdb}' not found. Please run test_dms_generation.py first.")
        # Create a dummy target file for testing
        os.makedirs(os.path.dirname(target_pdb), exist_ok=True)
        with open(target_pdb, "w") as f:
            f.write("DUMMY PDB HEADER\n")
            
    # 3. Predict structures
    print(f"\nRunning structure prediction for sequence: '{sequence}'...")
    coords, info = model.predict_structure(sequence, target_pdb)
    print(f"  Target Model Output: Coords shape = {coords.shape}, pLDDT = {info['plddt']}")
    
    print("\nRunning draft structure prediction...")
    coords_draft, info_draft = draft_model.predict_structure_draft(sequence, target_pdb)
    print(f"  Draft Model Output: Coords shape = {coords_draft.shape}, pLDDT = {info_draft['plddt']}")
    
    # Assertions
    assert coords.shape == (1, len(sequence), 3), "Coordinates shape mismatch!"
    assert coords_draft.shape == (1, len(sequence), 3), "Draft coordinates shape mismatch!"
    assert "plddt" in info, "Missing pLDDT score!"
    
    print("\nSuccess: Boltz-1/2 Wrappers initialize and execute successfully!")

if __name__ == "__main__":
    test_boltz_wrappers()
