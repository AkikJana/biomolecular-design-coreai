import os
import sys
import json
from typing import List, Dict, Any

# Resolve local paths
sys.path.append(os.path.dirname(__file__))

from dms_generation import TargetDMSGenerator

def evaluate_metrics(sequence: str) -> Dict[str, float]:
    """Calculates simulated pLDDT and ipSAE for a sequence.
    
    Reflects the actual biophysical properties of the mutants:
    - Hydrophobic content increases pLDDT (folding stability).
    - Polar content increases H-bonds, which together with hydrophobic stability optimizes ipSAE.
    """
    polar_count = sum(1 for c in sequence if c in "DESKRTQN")
    hydrophobic_count = sum(1 for c in sequence if c in "LIVAMF")
    
    # pLDDT (Stability): 60 to 98
    plddt = min(98.0, 60.0 + hydrophobic_count * 4.0)
    
    # ipSAE (Binding Affinity): 0.4 to 0.98
    ipsae = min(0.98, 0.40 + (polar_count * 0.04) + (hydrophobic_count * 0.03))
    
    return {
        "plddt": round(plddt, 1),
        "ipsae": round(ipsae, 2)
    }

def run_metric_comparison():
    print("=======================================================")
    print("[Validation] Running Metric Validation (ipSAE vs pLDDT)...")
    
    # 1. Generate the library of mutants
    dms_gen = TargetDMSGenerator(output_dir="/tmp/biomolecular_design")
    base_sequence = "MATEVLADIGSAKLR"
    library = dms_gen.generate_dms_library(
        base_sequence=base_sequence,
        interface_positions=[2, 4, 8, 12, 15],
        amino_acids="ADEFGHIKLMNPQRSTVWY"
    )
    
    # 2. Evaluate metrics
    scatter_data = []
    designable_count = 0
    total_count = len(library)
    
    for entry in library:
        scores = evaluate_metrics(entry["sequence"])
        plddt = scores["plddt"]
        ipsae = scores["ipsae"]
        
        # Classification
        # Thresholds: pLDDT >= 85.0 (Folded), ipSAE >= 0.80 (Tight Binding)
        is_folded = plddt >= 85.0
        is_binding = ipsae >= 0.80
        
        if is_folded and is_binding:
            category = "Designable Binder"
            designable_count += 1
        elif is_folded and not is_binding:
            category = "Misfolded/Non-binder"
        elif not is_folded and is_binding:
            category = "Unstable Binder"
        else:
            category = "Poor Candidate"
            
        scatter_data.append({
            "name": entry["mutation"],
            "sequence": entry["sequence"],
            "plddt": plddt,
            "ipsae": ipsae,
            "category": category
        })
        
    # 3. Print Statistical Summaries
    print(f"\nMetric Summary for {total_count} Mutants:")
    plddts = [item["plddt"] for item in scatter_data]
    ipsaes = [item["ipsae"] for item in scatter_data]
    
    avg_plddt = sum(plddts) / total_count
    avg_ipsae = sum(ipsaes) / total_count
    
    print(f"  Average pLDDT (Stability): {avg_plddt:.1f} (Range: {min(plddts)} to {max(plddts)})")
    print(f"  Average ipSAE (Affinity): {avg_ipsae:.2f} (Range: {min(ipsaes):.2f} to {max(ipsaes):.2f})")
    print(f"  Designable Binders Identified: {designable_count} / {total_count} ({designable_count/total_count:.1%})")
    
    # 4. Generate Apache ECharts Scatter Plot JSON Specification
    # We group by category
    categories = ["Designable Binder", "Misfolded/Non-binder", "Unstable Binder", "Poor Candidate"]
    series = []
    
    colors = {
        "Designable Binder": "#48bb78",   # Green
        "Misfolded/Non-binder": "#3182ce", # Blue
        "Unstable Binder": "#dd6b20",     # Orange
        "Poor Candidate": "#e53e3e"        # Red
    }
    
    for cat in categories:
        cat_points = [[item["plddt"], item["ipsae"], item["name"]] for item in scatter_data if item["category"] == cat]
        if cat_points:
            series.append({
                "name": cat,
                "type": "scatter",
                "data": cat_points,
                "itemStyle": {
                    "color": colors[cat]
                },
                "symbolSize": 8
            })
            
    echarts_spec = {
        "title": {
            "text": "Metric Validation: Binding Affinity (ipSAE) vs Stability (pLDDT)",
            "left": "center",
            "textStyle": {
                "color": "#0c2340",
                "fontSize": 14
            }
        },
        "legend": {
            "bottom": 10,
            "data": categories
        },
        "tooltip": {
            "trigger": "item",
            "formatter": "Mutation: {c2}<br/>pLDDT: {c0}<br/>ipSAE: {c1}"
        },
        "xAxis": {
            "name": "pLDDT (Folding Confidence)",
            "nameLocation": "middle",
            "nameGap": 25,
            "min": 55,
            "max": 100,
            "splitLine": {
                "lineStyle": {
                    "type": "dashed"
                }
            }
        },
        "yAxis": {
            "name": "ipSAE (Binding Affinity)",
            "nameLocation": "middle",
            "nameGap": 30,
            "min": 0.3,
            "max": 1.0,
            "splitLine": {
                "lineStyle": {
                    "type": "dashed"
                }
            }
        },
        "series": series
    }
    
    # Save the specification to a file for MCP consumption
    spec_path = "/tmp/biomolecular_design/echarts_spec.json"
    with open(spec_path, "w") as f:
        json.dump(echarts_spec, f, indent=2)
        
    print(f"\n[Validation] ECharts specification saved to: {spec_path}")
    print("=======================================================")

if __name__ == "__main__":
    run_metric_comparison()
