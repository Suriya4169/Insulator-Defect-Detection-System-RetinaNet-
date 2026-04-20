"""
experiments.py
==============
Runs two journal-quality FL experiments and generates publication-ready figures.

Experiment 1 — Client Scalability
    Trains the federated model with 2, 3, 5, and 10 clients (IID partition).
    Produces: scalability_map50.png

Experiment 2 — Non-IID Data Heterogeneity
    Trains with 3 clients under IID, moderate Non-IID (alpha=0.5), and
    extreme Non-IID (alpha=0.1) Dirichlet partitioning.
    Produces: non_iid_map50.png, non_iid_summary_bar.png

Usage
-----
    cd Fed_learning_Code
    python experiments.py --exp all          # run both
    python experiments.py --exp scalability  # only Experiment 1
    python experiments.py --exp noniid       # only Experiment 2
    python experiments.py --plots-only       # skip training; re-plot from saved JSONs
"""

import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── import the reusable training function ─────────────────────────────────────
from federated_train import run_experiment, FederatedConfig

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2']

STYLE = {
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'font.size':         11,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def plot_scalability(histories: dict, out_dir: str):
    """
    Line plot: mAP@50 per round for each client count.

    Parameters
    ----------
    histories : {label: history_dict}  e.g. {'2 Clients': {...}, ...}
    out_dir   : directory to save the PNG
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: mAP@50 per round ────────────────────────────────────────────────
    ax = axes[0]
    for i, (label, h) in enumerate(histories.items()):
        rounds = range(1, len(h['map50']) + 1)
        ax.plot(rounds, h['map50'], marker='o', color=COLORS[i], label=label, linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global mAP@50')
    ax.set_title('Client Scalability — mAP@50 per Round', fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(framealpha=0.9)

    # ── right: final mAP@50 bar chart ─────────────────────────────────────────
    ax2 = axes[1]
    labels       = list(histories.keys())
    final_maps   = [max(h['map50']) for h in histories.values()]
    bars = ax2.bar(labels, final_maps, color=COLORS[:len(labels)], width=0.5, zorder=3)
    ax2.set_xlabel('Number of FL Clients')
    ax2.set_ylabel('Best mAP@50')
    ax2.set_title('Peak mAP@50 vs. Number of Clients', fontweight='bold')
    ax2.set_ylim(0, 1.05)
    # annotate bar values
    for bar, val in zip(bars, final_maps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'scalability_map50.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Scalability] Plot saved → {out_path}")


def plot_non_iid(histories: dict, out_dir: str):
    """
    Line plot + summary bar chart for Non-IID comparison.

    Parameters
    ----------
    histories : {label: history_dict}
                e.g. {'IID': {...}, 'Non-IID α=0.5': {...}, 'Non-IID α=0.1': {...}}
    out_dir   : directory to save the PNGs
    """
    _apply_style()

    # ── (a) Line plot: mAP@50 per round ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, (label, h) in enumerate(histories.items()):
        rounds = range(1, len(h['map50']) + 1)
        ls = '-' if 'IID' == label else ('--' if '0.5' in label else ':')
        ax.plot(rounds, h['map50'], marker='o', linestyle=ls,
                color=COLORS[i], label=label, linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Global mAP@50')
    ax.set_title('Non-IID Robustness — mAP@50 per Round', fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(framealpha=0.9)

    # ── (b) Accuracy degradation bar chart ────────────────────────────────────
    ax2 = axes[1]
    labels  = list(histories.keys())
    final   = [max(h['map50']) for h in histories.values()]
    iid_val = final[0]
    drops   = [iid_val - v for v in final]      # degradation vs. IID

    bars = ax2.bar(labels, drops, color=['#2ca02c', '#ff7f0e', '#d62728'],
                   width=0.5, zorder=3)
    ax2.set_xlabel('Data Distribution')
    ax2.set_ylabel('mAP@50 Drop vs. IID Baseline')
    ax2.set_title('Performance Degradation under Non-IID', fontweight='bold')
    ax2.set_ylim(0, max(drops) * 1.5 + 0.02)
    for bar, drp, val in zip(bars, drops, final):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f'−{drp:.3f}\n({val:.3f})',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'non_iid_map50.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[Non-IID]    Plot saved → {out_path}")

    # ── (c) Summary comparison table as image ─────────────────────────────────
    fig, ax3 = plt.subplots(figsize=(9, 3))
    ax3.axis('off')
    col_labels = ['Setting', 'Best mAP@50', 'Drop vs. IID', 'Rounds to Converge']
    rows = []
    iid_rounds = len(histories[list(histories.keys())[0]]['map50'])
    for label, h in histories.items():
        best   = max(h['map50'])
        drop   = iid_val - best
        conv_r = int(np.argmax(h['map50'])) + 1
        rows.append([label, f'{best:.4f}', f'−{drop:.4f}', str(conv_r)])
    tbl = ax3.table(cellText=rows, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#264653')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4f8')
    plt.title('Non-IID Experiment Summary', fontweight='bold', pad=12)
    plt.tight_layout()
    tbl_path = os.path.join(out_dir, 'non_iid_summary_table.png')
    plt.savefig(tbl_path, bbox_inches='tight')
    plt.close()
    print(f"[Non-IID]    Summary table saved → {tbl_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability_experiment(plots_only: bool = False):
    """
    Experiment 1: Train 4 federated rounds with 2, 3, 5, 10 clients (IID).
    Results are saved as checkpoints/<tag>/history.json for re-plotting.
    """
    CLIENT_COUNTS = [2, 3, 5, 10]
    out_root      = FederatedConfig.OUTPUT_DIR
    histories     = {}

    print("\n" + "=" * 70)
    print("  EXPERIMENT 1 — Client Scalability Analysis")
    print("=" * 70)

    for n in CLIENT_COUNTS:
        tag       = f'scalability_iid_c{n}'
        json_path = os.path.join(out_root, tag, 'history.json')
        label     = f'{n} Clients'

        if plots_only and os.path.exists(json_path):
            # Load saved history — skip training
            with open(json_path) as f:
                histories[label] = json.load(f)
            print(f"  Loaded saved history for {label}")
        else:
            h = run_experiment(
                num_clients=n,
                partition_mode='iid',
                output_tag=tag,
            )
            histories[label] = h

    # ── Generate plots ────────────────────────────────────────────────────────
    plots_dir = os.path.join(out_root, 'experiment_plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_scalability(histories, plots_dir)
    print("\n  Experiment 1 complete.")
    return histories


def run_non_iid_experiment(plots_only: bool = False):
    """
    Experiment 2: Compare IID vs. Non-IID (alpha=0.5 and alpha=0.1)
    with a fixed 3-client setup.
    """
    NUM_CLIENTS = 3
    configs = [
        ('IID',           'iid',     None),
        ('Non-IID α=0.5', 'non_iid', 0.5),
        ('Non-IID α=0.1', 'non_iid', 0.1),
    ]
    out_root  = FederatedConfig.OUTPUT_DIR
    histories = {}

    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 — Non-IID Data Heterogeneity")
    print("=" * 70)

    for label, mode, alpha in configs:
        if alpha is not None:
            tag = f'noniid_alpha{str(alpha).replace(".", "")}_c{NUM_CLIENTS}'
        else:
            tag = f'iid_c{NUM_CLIENTS}'

        json_path = os.path.join(out_root, tag, 'history.json')

        if plots_only and os.path.exists(json_path):
            with open(json_path) as f:
                histories[label] = json.load(f)
            print(f"  Loaded saved history for {label}")
        else:
            kwargs = dict(num_clients=NUM_CLIENTS, partition_mode=mode, output_tag=tag)
            if alpha is not None:
                kwargs['alpha'] = alpha
            h = run_experiment(**kwargs)
            histories[label] = h

    # ── Generate plots ────────────────────────────────────────────────────────
    plots_dir = os.path.join(out_root, 'experiment_plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_non_iid(histories, plots_dir)
    print("\n  Experiment 2 complete.")
    return histories


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='FL Experiment Runner')
    p.add_argument('--exp', choices=['all', 'scalability', 'noniid'],
                   default='all', help='Which experiment to run')
    p.add_argument('--plots-only', action='store_true',
                   help='Skip training; re-plot from saved history.json files')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.exp in ('all', 'scalability'):
        run_scalability_experiment(plots_only=args.plots_only)

    if args.exp in ('all', 'noniid'):
        run_non_iid_experiment(plots_only=args.plots_only)

    print("\nAll requested experiments finished.")
    print(f"Plots saved in: {os.path.join(FederatedConfig.OUTPUT_DIR, 'experiment_plots')}")
