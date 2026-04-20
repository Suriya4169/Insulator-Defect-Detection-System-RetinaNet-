"""
compare_fedavg_vs_fedprox.py
============================
Experiment 3 for the journal paper — Phase 1 output.

Runs THREE configurations on the SAME data split and saves publication-ready
comparison figures + a LaTeX-ready summary table:

    Config A  —  FedAvg   (IID)          baseline, replays original result
    Config B  —  FedAvg   (Non-IID α=0.1) worst-case heterogeneity
    Config C  —  FedProx  (Non-IID α=0.1) proposed fix

These three results together make a compelling journal sub-section:

    "Table II — Impact of FedProx on Extreme Data Heterogeneity"

Usage
-----
    # Run all three experiments (may take 1-3 hours depending on GPU)
    python compare_fedavg_vs_fedprox.py

    # Skip training; re-plot from saved history.json files
    python compare_fedavg_vs_fedprox.py --plots-only

Outputs (in checkpoints/phase1_fedprox/comparison_plots/)
----------------------------------------------------------
    fedavg_vs_fedprox_map50.png     — line plot (mAP@50 per round)
    fedavg_vs_fedprox_bar.png       — bar chart (final mAP@50)
    summary_table.txt               — plain text table for the paper
    summary_latex.tex               — LaTeX tabular block (copy-paste ready)
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

# ── resolve path so we can import fedprox_train ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedprox_train import run_experiment, FedProxConfig

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configurations
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # (label,          algorithm,  partition,  alpha,  mu,    output_tag)
    ('FedAvg  (IID)',        'fedavg',  'iid',     0.5,  0.00, 'fedavg_iid'),
    ('FedAvg  (Non-IID α=0.1)', 'fedavg',  'non_iid', 0.1,  0.00, 'fedavg_noniid_a01'),
    ('FedProx (Non-IID α=0.1)', 'fedprox', 'non_iid', 0.1,  0.01, 'fedprox_noniid_a01_mu001'),
]

PLOTS_DIR = os.path.join(FedProxConfig.OUTPUT_DIR, 'comparison_plots')

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
    """mAP@50 convergence curve — one line per experiment."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, h) in enumerate(histories.items()):
        rounds = range(1, len(h['map50']) + 1)
        ax.plot(rounds, h['map50'],
                marker=MARKERS[i], color=COLORS[i],
                linestyle=LINESTYLE[i], linewidth=2.2, markersize=7,
                label=label)
        # annotate final value
        ax.annotate(
            f"{h['map50'][-1]:.3f}",
            xy=(len(h['map50']), h['map50'][-1]),
            xytext=(5, -3), textcoords='offset points',
            fontsize=9, color=COLORS[i], fontweight='bold'
        )

    # shaded region: FedProx gain over FedAvg Non-IID
    keys = list(histories.keys())
    if len(keys) >= 3:
        h_fedavg_noniid  = histories[keys[1]]['map50']
        h_fedprox_noniid = histories[keys[2]]['map50']
        min_len = min(len(h_fedavg_noniid), len(h_fedprox_noniid))
        x = list(range(1, min_len + 1))
        ax.fill_between(
            x, h_fedavg_noniid[:min_len], h_fedprox_noniid[:min_len],
            where=[h_fedprox_noniid[j] >= h_fedavg_noniid[j]
                   for j in range(min_len)],
            alpha=0.15, color=COLORS[2], label='FedProx gain'
        )

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Global mAP@50',       fontsize=12)
    ax.set_title(
        'FedAvg vs. FedProx on Extreme Non-IID Data\n'
        '(RetinaNet, 3 clients, Dirichlet α=0.1)',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xticks(range(1, FedProxConfig.ROUNDS + 1))

    plt.tight_layout()
    out = os.path.join(out_dir, 'fedavg_vs_fedprox_map50.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Line chart saved → {out}")


def plot_bar(histories: dict, out_dir: str):
    """Final mAP@50 bar chart."""
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
    ax.set_title('Peak mAP@50: FedAvg vs. FedProx', fontsize=13,
                 fontweight='bold', pad=10)
    ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, 'fedavg_vs_fedprox_bar.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Bar chart saved → {out}")


def save_summary_table(histories: dict, out_dir: str):
    """
    Write a plain-text table AND a LaTeX tabular block.

    Example output (summary_table.txt):
    ┌─────────────────────────────┬────────────┬──────────────┬─────────────────┐
    │ Method                      │ Best mAP@50│ mAP@75       │ Rounds to Conv. │
    ├─────────────────────────────┼────────────┼──────────────┼─────────────────┤
    │ FedAvg  (IID)               │   0.9124   │   0.7841     │        5        │
    │ FedAvg  (Non-IID α=0.1)     │   0.8501   │   0.7102     │        7        │
    │ FedProx (Non-IID α=0.1)     │   0.8893   │   0.7489     │        6        │
    └─────────────────────────────┴────────────┴──────────────┴─────────────────┘
    """
    iid_best = max(list(histories.values())[0]['map50'])

    rows = []
    for label, h in histories.items():
        best50  = max(h['map50'])
        best75  = max(h['map75'])
        conv_r  = int(np.argmax(h['map50'])) + 1
        drop    = iid_best - best50
        rows.append((label, best50, best75, conv_r, drop))

    # ── plain text ────────────────────────────────────────────────────────────
    header = f"{'Method':<32} {'mAP@50':>10} {'mAP@75':>10} " \
             f"{'Conv.Round':>12} {'Δ vs IID':>10}"
    sep    = '-' * len(header)
    lines  = [sep, header, sep]
    for lbl, b50, b75, cr, drop in rows:
        drop_str = f'-{drop:.4f}' if drop > 0 else '  —   '
        lines.append(
            f"{lbl:<32} {b50:>10.4f} {b75:>10.4f} {cr:>12} {drop_str:>10}"
        )
    lines.append(sep)
    txt_out = os.path.join(out_dir, 'summary_table.txt')
    with open(txt_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Table] Plain-text table → {txt_out}")
    print('\n'.join(lines))

    # ── LaTeX tabular ─────────────────────────────────────────────────────────
    tex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{FedAvg vs. FedProx on Extreme Non-IID Data (Dirichlet $\alpha$=0.1)}',
        r'\label{tab:fedprox_comparison}',
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
    tex_lines += [
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
    ]
    tex_out = os.path.join(out_dir, 'summary_latex.tex')
    with open(tex_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"[Table] LaTeX tabular   → {tex_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(plots_only: bool = False) -> dict:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    histories = {}

    print("\n" + "=" * 72)
    print("  PHASE 1 COMPARISON: FedAvg vs. FedProx")
    print("=" * 72)

    for label, algo, partition, alpha, mu, tag in EXPERIMENTS:
        json_path = os.path.join(FedProxConfig.OUTPUT_DIR, tag, 'history.json')

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

    # ── generate all outputs ──────────────────────────────────────────────────
    plot_line(histories, PLOTS_DIR)
    plot_bar(histories,  PLOTS_DIR)
    save_summary_table(histories, PLOTS_DIR)

    print(f"\n  All Phase 1 outputs saved to: {PLOTS_DIR}")
    return histories


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1: FedAvg vs FedProx comparison plots'
    )
    p.add_argument('--plots-only', action='store_true',
                   help='Skip training; re-plot from saved JSON files')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_comparison(plots_only=args.plots_only)
