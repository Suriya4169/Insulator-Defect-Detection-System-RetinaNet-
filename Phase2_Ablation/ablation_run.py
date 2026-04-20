"""
ablation_run.py
===============
Phase 2 of the Journal Upgrade — Systematic Ablation Study.

This script quantifies the performance impact of your three major technical
contributions:
1.  Copy-Paste Defect Augmentation
2.  Class-Balanced Focal Loss (alpha tuning)
3.  Weighted Client Sampling

It runs 4 configurations in sequence and generates an "Improvement Table"
suitable for inclusion in your paper's "Ablation Study" section.

Ablation Matrix:
----------------
Config | Name           | Copy-Paste | Bal. Focal | Weighted Sampler | Goal
-------|----------------|------------|------------|------------------|-----
A      | Baseline FL    | No         | No (Equal) | No               | Raw model
B      | + Augmentation | Yes        | No (Equal) | No               | Prove Aug.
C      | + Balancing    | No         | Yes        | Yes              | Prove Loss/Sampling
D      | Full Proposed  | Yes        | Yes        | Yes              | Total System

Notes:
------
- We use 'IID' partitioning for the ablation study to keep the results clean
  and isolated from heterogeneity effects.
- 5 communication rounds per config are usually enough to see the performance delta.
"""

import os
import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

# ── Import training logic from Phase 1 ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Phase1_FedProx'))
from fedprox_train import run_experiment, FedProxConfig

# ── Experiment Definitions ───────────────────────────────────────────────────

# Focal Loss Alpha Weights:
# [0.33, 0.33, 0.33] = Equal/Raw
# [0.25, 0.75, 0.25] = Proposed (Focusing on class 1 - Defect)
EQUAL_ALPHA    = [0.33, 0.33, 0.33]
PROPOSED_ALPHA = [0.25, 0.75, 0.25]

ABLATIONS = [
    {
        'id': 'A',
        'name': 'Baseline FL',
        'tag': 'ablation_A_baseline',
        'copy_paste': False,
        'weighted_sampler': False,
        'alpha': EQUAL_ALPHA
    },
    {
        'id': 'B',
        'name': '+ Augmentation',
        'tag': 'ablation_B_aug',
        'copy_paste': True,
        'weighted_sampler': False,
        'alpha': EQUAL_ALPHA
    },
    {
        'id': 'C',
        'name': '+ Balancing',
        'tag': 'ablation_C_balancing',
        'copy_paste': False,
        'weighted_sampler': True,
        'alpha': PROPOSED_ALPHA
    },
    {
        'id': 'D',
        'name': 'Full Proposed',
        'tag': 'ablation_D_full',
        'copy_paste': True,
        'weighted_sampler': True,
        'alpha': PROPOSED_ALPHA
    }
]

# Shared settings for all ablation runs
ROUNDS = 5 
CLIENTS = 3
OUTPUT_DIR = r'D:\Fed learning project\checkpoints\ablation_study'

def run_ablation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print("\n" + "="*70)
    print("  PHASE 2: SYSTEMATIC ABLATION STUDY")
    print("="*70)

    # We need to temporarily modify FedProxConfig to suit the ablation branch
    # Note: Because FedProxConfig is a class with class-level attributes, 
    # we can modify them globally before calling run_experiment.
    
    original_cp = FedProxConfig.USE_COPY_PASTE
    original_ws = FedProxConfig.USE_WEIGHTED_SAMPLER
    original_out = FedProxConfig.OUTPUT_DIR
    
    FedProxConfig.OUTPUT_DIR = OUTPUT_DIR

    try:
        for cfg in ABLATIONS:
            print(f"\n[Ablation {cfg['id']}] Running: {cfg['name']}...")
            
            # Toggle configuration flags
            FedProxConfig.USE_COPY_PASTE      = cfg['copy_paste']
            FedProxConfig.USE_WEIGHTED_SAMPLER = cfg['weighted_sampler']
            
            # Run the experiment
            # We pass mu=0 since ablation is usually done on plain FedAvg
            history = run_experiment(
                num_clients=CLIENTS,
                partition_mode='iid',
                rounds=ROUNDS,
                output_tag=cfg['tag'],
                algorithm='fedavg',
                mu=0.0
            )
            
            # Collect final results
            best_map50 = max(history['map50'])
            best_map75 = max(history['map75'])
            best_acc   = max(history['accuracy'])
            
            results.append({
                'ID': cfg['id'],
                'Component': cfg['name'],
                'mAP@50': best_map50,
                'mAP@75': best_map75,
                'Accuracy': best_acc
            })

        # ── generate report ───────────────────────────────────────────────────
        df = pd.DataFrame(results)
        
        # Calculate Delta compared to Baseline
        baseline_map = df[df['ID'] == 'A']['mAP@50'].iloc[0]
        df['Delta mAP'] = df['mAP@50'] - baseline_map
        
        print("\n\n" + "="*70)
        print("  ABLATION STUDY SUMMARY")
        print("="*70)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved → {csv_path}")

        # ── generate Figure ───────────────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Component'], df['mAP@50'], color=['#95a5a6', '#3498db', '#e67e22', '#2ecc71'])
        plt.axhline(y=baseline_map, color='r', linestyle='--', alpha=0.5, label='Baseline')
        
        plt.ylabel('Peak mAP@50')
        plt.title('Ablation Study: Contribution of Proposed Components')
        plt.ylim(0, 1.1)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'ablation_comparison.png')
        plt.savefig(plot_path, dpi=200)
        print(f"Plot saved → {plot_path}")

    finally:
        # Restore original config
        FedProxConfig.USE_COPY_PASTE = original_cp
        FedProxConfig.USE_WEIGHTED_SAMPLER = original_ws
        FedProxConfig.OUTPUT_DIR = original_out

if __name__ == '__main__':
    run_ablation()
