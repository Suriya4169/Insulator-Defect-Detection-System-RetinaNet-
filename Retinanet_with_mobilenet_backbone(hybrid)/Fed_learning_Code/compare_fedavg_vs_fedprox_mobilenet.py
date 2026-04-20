"""
compare_fedavg_vs_fedprox_mobilenet.py
=======================================
Phase 3: MobileNetV2 backbone FedAvg vs. FedProx comparison.

Runs THREE configurations (same as ResNet Phase 1) with MobileNetV2:
    Config A  —  FedAvg   (IID)              baseline efficiency
    Config B  —  FedAvg   (Non-IID α=0.1)    worst-case heterogeneity
    Config C  —  FedProx  (Non-IID α=0.1)    proposed fix

The final outputs can be placed side-by-side with the ResNet Phase 1 results
to create the "Backbone Benchmarking" table in the journal paper.

Usage
-----
    # Run all three experiments (takes 1-2 hours)
    python compare_fedavg_vs_fedprox_mobilenet.py

    # Re-generate plots only (skips training)
    python compare_fedavg_vs_fedprox_mobilenet.py --plots-only

Outputs (in checkpoints/mobilenet_fedprox/comparison_plots/)
-------------------------------------------------------------
    mobilenet_fedavg_vs_fedprox_map50.png   — convergence line chart
    mobilenet_fedavg_vs_fedprox_bar.png     — final mAP@50 bar chart
    mobilenet_summary_table.txt             — plain-text table
    mobilenet_summary_latex.tex             — LaTeX tabular block
"""

import os
import sys
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── path resolution ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedprox_train_mobilenet import run_experiment, MobileNetFedConfig

# ─────────────────────────────────────────────────────────────────────────────
# Experiments to run
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # (label,                      algo,      partition,  alpha, mu,   tag)
    ('FedAvg  (IID)',              'fedavg',  'iid',     0.5,  0.00, 'mb_fedavg_iid'),
    ('FedAvg  (Non-IID α=0.1)',   'fedavg',  'non_iid', 0.1,  0.00, 'mb_fedavg_noniid_a01'),
    ('FedProx (Non-IID α=0.1)',   'fedprox', 'non_iid', 0.1,  0.01, 'mb_fedprox_noniid_a01_mu001'),
]

PLOTS_DIR = os.path.join(MobileNetFedConfig.OUTPUT_DIR, 'comparison_plots')
COLORS    = ['#2196F3', '#F44336', '#4CAF50']   # blue, red, green
MARKERS   = ['o', 's', '^']
LINESTYLE = ['-', '--', '-']


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        'figure.dpi':        150,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.35,
        'font.size':         11,
        'font.family':       'DejaVu Sans',
    })


def plot_line(histories: dict, out_dir: str):
    _style()
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, h) in enumerate(histories.items()):
        rounds = range(1, len(h['map50']) + 1)
        ax.plot(rounds, h['map50'],
                marker=MARKERS[i], color=COLORS[i],
                linestyle=LINESTYLE[i], linewidth=2.2, markersize=7,
                label=label)
        ax.annotate(
            f"{h['map50'][-1]:.3f}",
            xy=(len(h['map50']), h['map50'][-1]),
            xytext=(5, -3), textcoords='offset points',
            fontsize=9, color=COLORS[i], fontweight='bold'
        )

    keys = list(histories.keys())
    if len(keys) >= 3:
        h_fedavg  = histories[keys[1]]['map50']
        h_fedprox = histories[keys[2]]['map50']
        min_len = min(len(h_fedavg), len(h_fedprox))
        x = list(range(1, min_len + 1))
        ax.fill_between(
            x, h_fedavg[:min_len], h_fedprox[:min_len],
            where=[h_fedprox[j] >= h_fedavg[j] for j in range(min_len)],
            alpha=0.15, color=COLORS[2], label='FedProx gain'
        )

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Global mAP@50',       fontsize=12)
    ax.set_title(
        'MobileNetV2: FedAvg vs. FedProx on Extreme Non-IID Data\n'
        '(RetinaNet, 3 clients, Dirichlet α=0.1)',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xticks(range(1, MobileNetFedConfig.ROUNDS + 1))

    plt.tight_layout()
    out = os.path.join(out_dir, 'mobilenet_fedavg_vs_fedprox_map50.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Line chart saved → {out}")


def plot_bar(histories: dict, out_dir: str):
    _style()
    fig, ax = plt.subplots(figsize=(9, 6))

    labels     = list(histories.keys())
    final_maps = [max(h['map50']) for h in histories.values()]

    bars = ax.bar(labels, final_maps,
                  color=COLORS[:len(labels)], width=0.45, zorder=3,
                  edgecolor='white', linewidth=1.2)

    for bar, val in zip(bars, final_maps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Best mAP@50', fontsize=12)
    ax.set_title('MobileNetV2: Peak mAP@50 — FedAvg vs. FedProx',
                 fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, 'mobilenet_fedavg_vs_fedprox_bar.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Bar chart saved → {out}")


def save_summary_table(histories: dict, out_dir: str):
    iid_best = max(list(histories.values())[0]['map50'])
    rows = []
    for label, h in histories.items():
        best50 = max(h['map50'])
        best75 = max(h['map75'])
        conv_r = int(np.argmax(h['map50'])) + 1
        drop   = iid_best - best50
        rows.append((label, best50, best75, conv_r, drop))

    # ── Plain text table ───────────────────────────────────────────────────
    header = (f"{'Method':<32} {'mAP@50':>10} {'mAP@75':>10} "
              f"{'Conv.Round':>12} {'Δ vs IID':>10}")
    sep    = '-' * len(header)
    lines  = [sep, header, sep]
    for lbl, b50, b75, cr, drop in rows:
        drop_str = f'-{drop:.4f}' if drop > 0 else '  —   '
        lines.append(
            f"{lbl:<32} {b50:>10.4f} {b75:>10.4f} {cr:>12} {drop_str:>10}"
        )
    lines.append(sep)

    txt_out = os.path.join(out_dir, 'mobilenet_summary_table.txt')
    with open(txt_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Table] Plain-text table → {txt_out}")
    print('\n'.join(lines))

    # ── LaTeX tabular ──────────────────────────────────────────────────────
    tex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{MobileNetV2: FedAvg vs. FedProx on Extreme Non-IID Data '
        r'(Dirichlet $\alpha$=0.1)}',
        r'\label{tab:mobilenet_fedprox_comparison}',
        r'\begin{tabular}{lcccc}',
        r'\hline',
        r'\textbf{Method} & \textbf{mAP@50} & \textbf{mAP@75} '
        r'& \textbf{Conv. Round} & \textbf{$\Delta$ vs IID} \\',
        r'\hline',
    ]
    for lbl, b50, b75, cr, drop in rows:
        lbl_tex  = lbl.replace('α', r'$\alpha$').replace('Non-IID', r'\text{Non-IID}')
        drop_tex = f'-{drop:.4f}' if drop > 0 else '—'
        tex_lines.append(
            f'{lbl_tex} & {b50:.4f} & {b75:.4f} & {cr} & {drop_tex} \\\\'
        )
    tex_lines += [r'\hline', r'\end{tabular}', r'\end{table}']

    tex_out = os.path.join(out_dir, 'mobilenet_summary_latex.tex')
    with open(tex_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"[Table] LaTeX tabular   → {tex_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(plots_only=False):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    histories = {}

    print("\n" + "=" * 72)
    print("  PHASE 3: MobileNetV2 — FedAvg vs. FedProx Comparison")
    print("=" * 72)

    for label, algo, partition, alpha, mu, tag in EXPERIMENTS:
        json_path = os.path.join(MobileNetFedConfig.OUTPUT_DIR, tag, 'history.json')

        if plots_only and os.path.exists(json_path):
            with open(json_path) as f:
                histories[label] = json.load(f)
            print(f"  Loaded saved history for: {label}")
        else:
            print(f"\n  Running: {label}  (algo={algo}, mu={mu})")
            h = run_experiment(
                partition_mode=partition,
                alpha=alpha,
                output_tag=tag,
                algorithm=algo,
                mu=mu,
            )
            histories[label] = h

    plot_line(histories, PLOTS_DIR)
    plot_bar(histories, PLOTS_DIR)
    save_summary_table(histories, PLOTS_DIR)

    print(f"\n  All Phase 3 outputs saved to: {PLOTS_DIR}")
    return histories


def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 3: MobileNetV2 FedAvg vs FedProx comparison'
    )
    p.add_argument('--plots-only', action='store_true',
                   help='Skip training; re-plot from saved JSON files')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_comparison(plots_only=args.plots_only)
