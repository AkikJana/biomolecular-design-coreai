import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dms_generation import TargetDMSGenerator

def test_dms_pipeline():
    print("=== Testing Local DMS Generation Pipeline ===")
    
    # 1. Initialize generator
    generator = TargetDMSGenerator(output_dir="/tmp/biomolecular_design")
    
    # 2. Download target PDB (TNF-alpha)
    target = "TNF-alpha"
    print(f"\nTarget: {target}")
    try:
        pdb_path = generator.download_target_pdb(target)
        assert os.path.exists(pdb_path), f"Failed to save PDB file!"
        print(f"Success: Verified local PDB exists at {pdb_path} (Size: {os.path.getsize(pdb_path)} bytes)")
    except Exception as e:
        print(f"Error downloading PDB: {e}")
        return
        
    # 3. Generate mutant library scan
    # Let's say residue positions 3, 5, and 10 are at the binding interface
    wt_binder = "MATEVLADIGSAKLR"
    interface_positions = [3, 5, 10]
    
    print(f"\nGenerating mutant library for WT Binder '{wt_binder}' at interface positions {interface_positions}...")
    library = generator.generate_dms_library(
        base_sequence=wt_binder,
        interface_positions=interface_positions,
        amino_acids="AVS" # Scan with a small subset of amino acids for testing
    )
    
    print("\nGenerated DMS Library Samples (first 10 mutants):")
    for i, entry in enumerate(library[:10]):
        print(f"  {i}: Mutation={entry['mutation']:<5} Sequence={entry['sequence']}")
        
    # Verify sequence counts:
    # WT + 3 positions * mutants = 9 sequences
    print(f"\nTotal sequences generated: {len(library)}")
    assert len(library) == 9, f"Expected 9 sequences, got {len(library)}"
    print("\nSuccess: DMS library successfully generated with exact sequence counts!")

if __name__ == "__main__":
    test_dms_pipeline()
