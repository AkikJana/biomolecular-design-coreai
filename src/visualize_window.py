import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict_structure import DynamicStructurePredictor

def main():
    print("================================================================")
    print("DYNAMIC STRUCTURE PREDICTION & 3D VISUALIZATION WINDOW")
    print("================================================================")
    
    # 1. Initialize predictor
    predictor = DynamicStructurePredictor()
    
    # 2. Define human Insulin fragment sequence (50 residues)
    # and a receptor target sequence (153 residues)
    insulin_seq = "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPK"
    target_receptor = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR" + "GLVLIAFSQYL"
    
    print(f"\nPredicting 3D structures for sequence: {insulin_seq}...")
    coords = predictor.predict(insulin_seq, target_receptor) # Shape: (1, 50, 3)
    
    # Extract coordinate vectors
    xs = coords[0, :, 0]
    ys = coords[0, :, 1]
    zs = coords[0, :, 2]
    L = len(insulin_seq)
    
    # 3. Setup the 3D interactive plot
    print("Setting up the 3D plot window...")
    fig = plt.figure(figsize=(12, 10), facecolor='#111827') # Dark premium background
    ax = fig.add_subplot(111, projection='3d', facecolor='#111827')
    
    # Color coordinates sequentially along the chain (N-terminus to C-terminus)
    # Using a beautiful plasma gradient (purple -> orange -> yellow)
    colors = plt.cm.plasma(np.linspace(0, 1, L))
    
    # Plot the backbone trace line
    ax.plot(xs, ys, zs, color='#3b82f6', linewidth=4, alpha=0.7, label="C-alpha Backbone", zorder=1)
    
    # Plot spheres for each residue, color-coded by index
    scatter = ax.scatter(xs, ys, zs, c=np.arange(L), cmap='plasma', s=150, edgecolors='#f3f4f6', linewidth=1.5, depthshade=True, zorder=2)
    
    # Add text labels for some residues to guide the user (every 5 residues)
    for i in range(L):
        if i == 0:
            ax.text(xs[i], ys[i], zs[i], "  N-term (G1)", color='#10b981', fontsize=10, fontweight='bold')
        elif i == L - 1:
            ax.text(xs[i], ys[i], zs[i], f"  C-term ({insulin_seq[i]}{i+1})", color='#ef4444', fontsize=10, fontweight='bold')
        elif i % 5 == 0:
            ax.text(xs[i], ys[i], zs[i], f"  {insulin_seq[i]}{i+1}", color='#9ca3af', fontsize=8)
            
    # Aesthetic styling (premium dark mode)
    ax.set_title(f"Dynamic structure Prediction: Human Insulin ({L} aa)\nApple Neural Engine/GPU Accelerated (FP8)", color='#f3f4f6', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("X (Å)", color='#9ca3af', fontsize=10)
    ax.set_ylabel("Y (Å)", color='#9ca3af', fontsize=10)
    ax.set_zlabel("Z (Å)", color='#9ca3af', fontsize=10)
    
    # Customize grid and panes for dark theme
    ax.xaxis.pane.set_facecolor('#1f2937')
    ax.yaxis.pane.set_facecolor('#1f2937')
    ax.zaxis.pane.set_facecolor('#1f2937')
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    
    # Text colors for ticks
    ax.tick_params(colors='#9ca3af')
    
    # Colorbar to represent sequence progression
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15, pad=0.08)
    cbar.set_label("Sequence Progress (N -> C Terminus)", color='#9ca3af', fontsize=10, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='#9ca3af', labelcolor='#9ca3af')
    
    # View orientation
    ax.view_init(elev=25, azim=35)
    
    # Save a static copy
    output_png = "/Users/akikjana/Documents/BiomolecularDesign/backbone_3d_insulin.png"
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, facecolor='#111827')
    print(f"Static copy saved successfully to: {output_png}")
    
    # Pop open the window
    print("Opening interactive 3D window... (You can rotate and zoom using your mouse!)")
    plt.show()
    print("Window closed. Done!")

if __name__ == "__main__":
    main()
