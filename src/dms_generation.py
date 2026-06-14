import os
import urllib.request
from typing import List, Dict, Any, Tuple

class TargetDMSGenerator:
    """Generates a localized Deep Mutational Scanning (DMS) sequence library for a target PDB complex."""
    
    PDB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
    
    # Standard PDB IDs for your dissertation targets
    TARGET_PDB_IDS = {
        "TNF-alpha": "1TNF",  # Human Tumor Necrosis Factor alpha
        "VEGFA": "1FLT"       # VEGF bound to Flt-1 domain
    }

    def __init__(self, output_dir: str = "/tmp/biomolecular_design"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_target_pdb(self, target_name: str) -> str:
        """Downloads the target PDB file from the RCSB Protein Data Bank.
        
        Args:
            target_name: 'TNF-alpha' or 'VEGFA'.
            
        Returns:
            The local file path to the downloaded PDB structure.
        """
        if target_name not in self.TARGET_PDB_IDS:
            raise ValueError(f"Unknown target: {target_name}. Choose from {list(self.TARGET_PDB_IDS.keys())}")
            
        pdb_id = self.TARGET_PDB_IDS[target_name]
        pdb_path = os.path.join(self.output_dir, f"{target_name}_{pdb_id}.pdb")
        
        if os.path.exists(pdb_path):
            print(f"[DMS] PDB file already exists locally: {pdb_path}")
            return pdb_path
            
        url = self.PDB_URL_TEMPLATE.format(pdb_id=pdb_id)
        print(f"[DMS] Downloading {target_name} PDB structure from RCSB ({url})...")
        try:
            urllib.request.urlretrieve(url, pdb_path)
            print(f"[DMS] Successfully saved PDB to: {pdb_path}")
            return pdb_path
        except Exception as e:
            print(f"[DMS] Error downloading PDB: {e}")
            raise e

    def generate_dms_library(
        self, 
        base_sequence: str, 
        interface_positions: List[int], 
        amino_acids: str = "ADEFGHIKLMNPQRSTVWY"
    ) -> List[Dict[str, Any]]:
        """Generates single-point mutations at designated interface positions to create a DMS library.
        
        Args:
            base_sequence: The starting binder sequence (e.g., 'MATEVLADIGSAKLR').
            interface_positions: 1-indexed residue positions at the binding interface to mutate.
            amino_acids: String of candidate amino acids to scan.
            
        Returns:
            A list of dicts containing sequence metadata, mutation labels, and empty slots for rewards.
        """
        dms_library = []
        
        # Add wild-type baseline
        dms_library.append({
            "sequence": base_sequence,
            "mutation": "WT",
            "is_wt": True,
            "mutated_positions": []
        })
        
        # Scan mutational space
        for pos in interface_positions:
            idx = pos - 1 # 0-indexed python position
            if idx < 0 or idx >= len(base_sequence):
                continue
                
            original_aa = base_sequence[idx]
            
            for aa in amino_acids:
                if aa == original_aa:
                    continue
                    
                mutated_seq_list = list(base_sequence)
                mutated_seq_list[idx] = aa
                mutated_seq = "".join(mutated_seq_list)
                
                mutation_label = f"{original_aa}{pos}{aa}"
                
                dms_library.append({
                    "sequence": mutated_seq,
                    "mutation": mutation_label,
                    "is_wt": False,
                    "mutated_positions": [pos]
                })
                
        print(f"[DMS] Generated mutational library of size {len(dms_library)} from base sequence.")
        return dms_library
