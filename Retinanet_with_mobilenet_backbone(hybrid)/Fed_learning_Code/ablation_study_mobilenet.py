"""
ablation_study_mobilenet.py
==========================
Phase 4: Quantifying the impact of technical components on mAP.

Runs 4 configurations for 5 rounds each:
1. Baseline (No CP, No WS)
2. Copy-Paste Only (+CP)
3. Weighted Sampling Only (+WS)
4. Proposed (Full) (+CP, +WS)
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Set path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedprox_train_mobilenet import run_experiment, MobileNetFedConfig

def run_ablation():
    # We use Non-IID alpha=0.1 for ablation as it highlights the improvements best
    configs = [
        # (label, use_cp, use_ws, tag)
        ("Baseline (No Aug/WS)",      False, False, "mb_abl_none"),
        ("Proposed + Copy-Paste",     True,  False, "mb_abl_cp"),
        ("Proposed + Weighted Sampling", False, True,  "mb_abl_ws"),
        ("Full Proposed System",     True,  True,  "mb_abl_full"),
    ]
    
    # Temporarily reduce rounds to 5 for speed
    MobileNetFedConfig.ROUNDS = 5
    results_file = os.path.join(MobileNetFedConfig.OUTPUT_DIR, "ablation_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            ablation_history = json.load(f)
    else:
        ablation_history = {}

    for label, use_cp, use_ws, tag in configs:
        if label in ablation_history:
            print(f"  [Skip] {label} already processed.")
            continue
            
        print(f"\n[Ablation] Running: {label}")
        h = run_experiment(
            partition_mode='non_iid',
            alpha=0.1,
            output_tag=tag,
            algorithm='fedavg', # Fixed algorithm for ablation
            use_copy_paste=use_cp,
            use_weighted_sampling=use_ws
        )
        ablation_history[label] = {
            'map50': h['map50'],
            'best_map50': max(h['map50']),
            'best_map75': max(h['map75'])
        }
        
        # Save after each run
        with open(results_file, 'w') as f:
            json.dump(ablation_history, f, indent=4)

    # Visualization
    plot_ablation(ablation_history)
    print(f"\n[Done] Ablation results saved to {results_file}")

def plot_ablation(history):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})
    
    labels = list(history.keys())
    values = [h['best_map50'] for h in history.values()]
    
    colors = ['#9E9E9E', '#2196F3', '#FFC107', '#4CAF50']
    bars = plt.bar(labels, values, color=colors, width=0.5, zorder=3)
    plt.grid(axis='y', alpha=0.3, zorder=0)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel("Best mAP@50 (Non-IID)")
    plt.title("Ablation Study: Contribution of Proposed Components\n(MobileNetV2-RetinaNet)", 
              fontweight='bold', pad=15)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    
    out_img = os.path.join(MobileNetFedConfig.OUTPUT_DIR, "mobilenet_ablation_study.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    print(f"[Plot] Ablation bar chart saved → {out_img}")

if __name__ == "__main__":
    run_ablation()
