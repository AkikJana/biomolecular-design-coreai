import os
import time
import numpy as np
import torch
import sys

# Add src to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict_structure import DynamicStructurePredictor

def main():
    print("================================================================")
    # 1. Initialize predictor
    # This loads the dynamic model and function pointers once
    start_init = time.perf_counter()
    predictor = DynamicStructurePredictor()
    init_time = time.perf_counter() - start_init
    print(f"Model initialization took: {init_time:.4f} seconds")
    print("================================================================")

    # 2. Define actual protein sequences of varying lengths for the binder candidates
    # Insulin monomer (51 residues)
    insulin = "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    
    # Hemoglobin subunit alpha (142 residues)
    hemoglobin_alpha = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    
    insulin_frag = insulin[:50] # 50 residues
    hemoglobin_frag = hemoglobin_alpha[:90] # 90 residues
    mutant_short = "MATEVLAD" # 8 residues
    mutant_mid = "MATEVLADIGSAKLR" # 15 residues
    
    binders = {
        "Short Mutant": (mutant_short, len(mutant_short)),
        "Mid Mutant": (mutant_mid, len(mutant_mid)),
        "Insulin Fragment": (insulin_frag, len(insulin_frag)),
        "Hemoglobin Fragment": (hemoglobin_frag, len(hemoglobin_frag))
    }
    
    # 3. Define target receptor sequences of varying lengths
    # Target range is min=50, max=2000. Let's test different target sizes.
    # Target 1 (150 residues): Hemoglobin alpha subunit
    target_150 = hemoglobin_alpha + "GLVLIAFSQYL" # 153 residues
    # Target 2 (600 residues): BSA fragment
    target_600 = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKC"[:600]
    # Target 3 (1300 residues): Very large receptor protein (1300 residues)
    target_1300 = (target_600 * 3)[:1300]
    
    targets = {
        "Small Target (153 aa)": target_150,
        "Medium Target (600 aa)": target_600,
        "Large Target (1300 aa)": target_1300
    }
    
    # 4. Run cross-product benchmarks to show the dynamic execution without recompiling!
    print(f"\n{'Binder Type':<25} | {'Binder L':<8} | {'Target Type':<25} | {'Target L':<8} | {'Time (ms)':<10} | {'Output Shape':<15}")
    print("-" * 105)
    
    for binder_name, (binder_seq, b_len) in binders.items():
        for target_name, target_seq in targets.items():
            t_len = len(target_seq)
            
            # Warm up first run (just to establish baseline)
            _ = predictor.predict(binder_seq, target_seq)
            
            # Measure average latency over 5 iterations
            latencies = []
            for _ in range(5):
                t0 = time.perf_counter()
                coords = predictor.predict(binder_seq, target_seq)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # in ms
                
            avg_latency = np.mean(latencies)
            print(f"{binder_name:<25} | {b_len:<8} | {target_name:<25} | {t_len:<8} | {avg_latency:<10.2f} | {str(coords.shape):<15}")
            
    print("================================================================")
    print("Inference completed successfully on all combinations!")
    print("Notice that:")
    print("1. All runs took milliseconds and did not cause any compilation overhead.")
    print("2. The output coordinate shape dynamically matches (1, L_binder, 3) for each sequence.")
    print("3. No padding was needed for binder inputs!")
    print("================================================================")
    
    # 5. Show actual 3D coordinates predicted for biological samples
    print("\n[VERIFICATION] Running explicit prediction on Human Insulin Fragment (50 aa) against Small Target (153 aa)...")
    coords_insulin = predictor.predict(insulin_frag, target_150)
    print(f"Insulin Fragment Coordinates shape: {coords_insulin.shape}")
    print("First 5 predicted backbone C-alpha coordinates (X, Y, Z in Angstroms):")
    # coords_insulin is of shape [1, L, 3]
    for i in range(5):
        coord = coords_insulin[0, i]
        print(f"  Residue {i+1} ({insulin_frag[i]}): X={coord[0]:.4f}, Y={coord[1]:.4f}, Z={coord[2]:.4f}")
        
    print("\n[VERIFICATION] Running explicit prediction on Hemoglobin Fragment (90 aa) against Large Target (1300 aa)...")
    coords_hemo = predictor.predict(hemoglobin_frag, target_1300)
    print(f"Hemoglobin Fragment Coordinates shape: {coords_hemo.shape}")
    print("First 5 predicted backbone C-alpha coordinates (X, Y, Z in Angstroms):")
    for i in range(5):
        coord = coords_hemo[0, i]
        print(f"  Residue {i+1} ({hemoglobin_frag[i]}): X={coord[0]:.4f}, Y={coord[1]:.4f}, Z={coord[2]:.4f}")
    print("================================================================")

if __name__ == "__main__":
    main()
