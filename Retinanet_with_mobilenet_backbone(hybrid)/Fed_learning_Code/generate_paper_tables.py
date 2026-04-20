"""
generate_paper_tables.py
========================
Final Phase: Results Consolidation.
Aggregates ResNet-50 and MobileNetV2 results into a single LaTeX table
appropriate for a Q1 journal publication.
"""

import os
import json

# Paths
RESNET_DIR    = r"D:\Fed learning project\checkpoints\fedprox"
MOBILENET_DIR = r"D:\Fed learning project\checkpoints\mobilenet_fedprox"
OUTPUT_DIR    = r"D:\Fed learning project\checkpoints\final_tables"

SUBDIRS = [
    ("FedAvg (IID)",              "fedavg_iid",               "mb_fedavg_iid"),
    ("FedAvg (Non-IID a=0.1)",    "fedavg_noniid_a01",        "mb_fedavg_noniid_a01"),
    ("FedProx (Non-IID a=0.1)",   "fedprox_noniid_a01_mu001", "mb_fedprox_noniid_a01_mu001"),
]

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f:
        return json.load(f)

def generate_table():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    rows = []
    for label, res_tag, mb_tag in SUBDIRS:
        res_h = load_json(os.path.join(RESNET_DIR, res_tag, 'history.json'))
        mb_h  = load_json(os.path.join(MOBILENET_DIR, mb_tag, 'history.json'))
        
        res_map = max(res_h['map50']) if res_h else 0.0
        mb_map  = max(mb_h['map50'])  if mb_h  else 0.0
        
        rows.append({
            'label': label,
            'resnet': res_map,
            'mobilenet': mb_map,
        })

    # ── LaTeX Compilation ───────────────────────────────────────────────────
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Backbone Benchmarking: ResNet-50 vs. MobileNetV2-RetinaNet in Federated Learning}",
        r"\label{tab:backbone_comparison}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Federated Setting} & \textbf{ResNet-50 mAP@50} & \textbf{MobileNetV2 mAP@50} & \textbf{$\Delta$ Accuracy} \\",
        r"\hline",
    ]
    
    for r in rows:
        diff = r['mobilenet'] - r['resnet']
        diff_str = f"{diff:+.4f}"
        tex.append(f"{r['label']} & {r['resnet']:.4f} & {r['mobilenet']:.4f} & {diff_str} \\\\")
        
    tex += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}"
    ]
    
    out_file = os.path.join(OUTPUT_DIR, "backbone_benchmarking.tex")
    with open(out_file, 'w') as f:
        f.write("\n".join(tex))
        
    print(f"\n[Success] Final comparison table generated → {out_file}")
    
    # Print a plain text version for the terminal
    print("\n" + "-"*80)
    print(f"{'Method':<30} {'ResNet-50':>15} {'MobileNetV2':>15} {'Delta':>15}")
    print("-"*80)
    for r in rows:
        print(f"{r['label']:<30} {r['resnet']:>15.4f} {r['mobilenet']:>15.4f} {r['mobilenet']-r['resnet']:>15.4f}")
    print("-"*80)

if __name__ == "__main__":
    generate_table()
