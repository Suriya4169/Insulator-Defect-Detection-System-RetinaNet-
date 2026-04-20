"""
Client Scalability Analysis for Federated Learning
====================================================
Runs FL training with NUM_CLIENTS in [2, 3, 5, 10] and plots mAP@50
curves on a single graph for the paper.

Usage:
    python scalability_analysis.py

Results are saved after each experiment so the script is resume-safe
(re-run after a crash and it will skip already-completed experiments).

Outputs (written to checkpoints/):
    scalability_results.json          -- raw mAP@50 arrays for all runs
    scalability_map50.png             -- comparison plot (paper-ready)
    scalability_c{N}/                 -- per-run checkpoints
"""

import os
import sys
import json
import copy

import matplotlib
matplotlib.use('Agg')           # no display needed
import matplotlib.pyplot as plt
import torch
import torch.utils.data

# ── resolve sibling imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CustomCocoDataset, build_defect_bank, make_weighted_sampler
from transforms import get_transform
from model import get_model
from engine import train_one_epoch, evaluate, print_metrics
from federated_train import FederatedConfig, partition_ids, EarlyStopping, collate_fn

# ── experiment settings ──────────────────────────────────────────────────────
CLIENT_COUNTS    = [2, 3, 5, 10]

# All scalability outputs live inside their own subdirectory so they never
# mix with the original federated_train.py checkpoints.
SCALABILITY_DIR  = os.path.join(FederatedConfig.OUTPUT_DIR, 'scalability_analysis')
RESULTS_FILE     = os.path.join(SCALABILITY_DIR, 'scalability_results.json')
PLOT_FILE        = os.path.join(SCALABILITY_DIR, 'scalability_map50.png')


# ── single experiment ────────────────────────────────────────────────────────
def run_experiment(num_clients: int) -> dict:
    """
    Run a complete FL training cycle with `num_clients` clients.
    Mirrors federated_train.main() but is parameterised by num_clients
    and returns the history dict.
    """
    print(f"\n{'='*60}")
    print(f"  SCALABILITY EXPERIMENT  —  {num_clients} client(s)")
    print(f"{'='*60}")

    device = FederatedConfig.DEVICE
    output_subdir = os.path.join(SCALABILITY_DIR, f'c{num_clients}')
    os.makedirs(output_subdir, exist_ok=True)

    history = {'loss': [], 'accuracy': [], 'map50': [], 'map75': []}
    best_map50 = 0.0
    early_stopping = EarlyStopping(patience=3, min_delta=0.005)

    # ── data ─────────────────────────────────────────────────────────────────
    train_dir = os.path.join(FederatedConfig.DATA_PATH, 'train')
    train_ann = os.path.join(train_dir, '_annotations.coco.json')
    val_dir   = os.path.join(FederatedConfig.DATA_PATH, 'valid')
    val_ann   = os.path.join(val_dir,   '_annotations.coco.json')

    defect_bank = None
    if FederatedConfig.USE_COPY_PASTE:
        defect_bank = build_defect_bank(
            train_dir, train_ann,
            max_crops=FederatedConfig.DEFECT_BANK_SIZE
        )

    _full_train      = CustomCocoDataset(train_dir, train_ann)
    client_id_splits = partition_ids(_full_train.ids, num_clients)

    val_dataset = CustomCocoDataset(
        val_dir, val_ann,
        transforms=get_transform(train=False),
        augment=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    client_datasets = []
    for i in range(num_clients):
        subset = CustomCocoDataset(
            train_dir, train_ann,
            transforms=get_transform(train=True),
            augment=True,
            defect_bank=defect_bank,
            defect_prob=FederatedConfig.COPY_PASTE_PROB,
            subset_ids=client_id_splits[i]
        )
        client_datasets.append(subset)
        print(f"  Client {i+1}: {len(subset)} samples assigned.")

    # ── model ─────────────────────────────────────────────────────────────────
    global_model   = get_model(FederatedConfig.NUM_CLASSES)
    global_model.to(device)
    global_weights = global_model.state_dict()
    class_names    = {0: 'background', 1: 'defect', 2: 'insulator'}

    latest_acc = 0.0
    latest_map = 0.0

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for round_idx in range(FederatedConfig.ROUNDS):
        print(f"\n  --- Round {round_idx+1}/{FederatedConfig.ROUNDS}  "
              f"[clients={num_clients}, mAP@50={latest_map:.4f}] ---")

        local_weights = []
        round_losses  = []

        for client_idx in range(num_clients):
            print(f"    > Client {client_idx+1}...", end=' ', flush=True)

            client_model = get_model(FederatedConfig.NUM_CLASSES)
            client_model.load_state_dict(copy.deepcopy(global_weights))
            client_model.to(device)
            client_model.train()

            params    = [p for p in client_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=FederatedConfig.LEARNING_RATE,
                weight_decay=FederatedConfig.WEIGHT_DECAY
            )

            ds      = client_datasets[client_idx]
            sampler = (make_weighted_sampler(ds, defect_weight=FederatedConfig.DEFECT_SAMPLE_WEIGHT)
                       if FederatedConfig.USE_WEIGHTED_SAMPLER else None)

            client_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=FederatedConfig.BATCH_SIZE,
                sampler=sampler,
                shuffle=(sampler is None),
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True
            )

            # Guard: OneCycleLR requires total_steps >= 1
            total_steps = max(1, len(client_loader) * FederatedConfig.CLIENT_EPOCHS)

            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=FederatedConfig.LEARNING_RATE,
                total_steps=total_steps,
                pct_start=FederatedConfig.PCT_START,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )

            scaler      = torch.amp.GradScaler(device.type)
            client_loss = 0.0

            for epoch in range(FederatedConfig.CLIENT_EPOCHS):
                metrics = train_one_epoch(
                    client_model, optimizer, client_loader,
                    device, epoch,
                    scaler=scaler,
                    scheduler=lr_scheduler,
                    grad_clip=FederatedConfig.GRAD_CLIP,
                    extra_info={'acc': f'{latest_acc:.1f}%'}
                )
                client_loss += metrics['total_loss']

            avg_client_loss = client_loss / FederatedConfig.CLIENT_EPOCHS
            round_losses.append(avg_client_loss)
            local_weights.append(copy.deepcopy(client_model.state_dict()))
            print(f"Done. Loss: {avg_client_loss:.4f}")

            del client_model
            torch.cuda.empty_cache()

        # ── weighted FedAvg ───────────────────────────────────────────────────
        client_sizes = [len(ds) for ds in client_datasets]
        total_size   = sum(client_sizes)
        w            = [s / total_size for s in client_sizes]

        global_weights = copy.deepcopy(local_weights[0])
        for key in global_weights.keys():
            global_weights[key] = sum(
                wi * lw[key].float() for wi, lw in zip(w, local_weights)
            ).to(global_weights[key].dtype)

        global_model.load_state_dict(global_weights)

        # ── record metrics ────────────────────────────────────────────────────
        avg_round_loss = sum(round_losses) / len(round_losses)
        history['loss'].append(avg_round_loss)

        print("  Evaluating global model...")
        res        = evaluate(global_model, val_loader, device=device)
        latest_acc = res['accuracy']
        latest_map = res['map_50']

        history['accuracy'].append(res['accuracy'])
        history['map50'].append(res['map_50'])
        history['map75'].append(res['map_75'])

        print_metrics(res, class_names)

        # ── checkpoint ───────────────────────────────────────────────────────
        ckpt_path = os.path.join(output_subdir, f'global_model_r{round_idx+1}.pth')
        torch.save({
            'round': round_idx,
            'model_state_dict': global_model.state_dict(),
            'history': history
        }, ckpt_path)

        if res['map_50'] > best_map50:
            best_map50 = res['map_50']
            best_path  = os.path.join(output_subdir, 'best_global_model.pth')
            torch.save({
                'round': round_idx,
                'model_state_dict': global_model.state_dict(),
                'best_map50': best_map50,
                'history': history
            }, best_path)
            print(f"  *** New Best mAP@50: {best_map50:.4f} ***")

        # ── early stopping ────────────────────────────────────────────────────
        early_stopping(res['map_50'])
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at round {round_idx+1}")
            break

    return history


# ── plot ─────────────────────────────────────────────────────────────────────
def plot_scalability(all_results: dict, save_path: str):
    """
    Plot mAP@50 vs Communication Round for every client count on one graph.
    Each line ends at the round where that experiment stopped (early stopping
    or ROUNDS), so lines may differ in length — this honestly shows convergence.
    """
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o',       's',       '^',       'D'      ]
    styles  = ['-',       '--',      '-.',      ':'      ]

    fig, ax = plt.subplots(figsize=(10, 6))

    all_map50_vals = []
    for (nc, history), color, marker, style in zip(
        sorted(all_results.items(), key=lambda x: int(x[0])),
        colors, markers, styles
    ):
        map50  = history['map50']
        rounds = list(range(1, len(map50) + 1))
        final  = map50[-1]
        all_map50_vals.extend(map50)
        ax.plot(
            rounds, map50,
            color=color, marker=marker, linestyle=style,
            linewidth=2, markersize=6,
            label=f'{nc} clients  (final mAP@50 = {final:.4f})'
        )
        # Annotate the final value at the end of each line
        ax.annotate(
            f'{final:.3f}',
            xy=(rounds[-1], final),
            xytext=(4, 0), textcoords='offset points',
            fontsize=9, color=color, va='center'
        )

    ax.set_title(
        'Client Scalability Analysis: Global mAP@50 vs Communication Round',
        fontsize=14, fontweight='bold', pad=12
    )
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Global mAP@50',       fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(1, FederatedConfig.ROUNDS + 1))

    # Tight y-axis: just below min, just above max
    if all_map50_vals:
        ymin = max(0.0, min(all_map50_vals) - 0.05)
        ymax = min(1.0, max(all_map50_vals) + 0.05)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlim(0.5, FederatedConfig.ROUNDS + 0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nScalability plot saved → {save_path}")


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(SCALABILITY_DIR, exist_ok=True)

    # Resume support: load any previously completed experiments
    all_results: dict = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        done = sorted(int(k) for k in all_results)
        print(f"Resuming — already completed experiments: {done}")

    for nc in CLIENT_COUNTS:
        key = str(nc)
        if key in all_results:
            print(f"\n  Skipping {nc} clients (already done). "
                  f"Final mAP@50: {all_results[key]['map50'][-1]:.4f}")
            continue

        history = run_experiment(nc)
        all_results[key] = history

        # Persist immediately so a crash doesn't lose this run
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results persisted after {nc}-client run → {RESULTS_FILE}")

    # Final plot
    plot_scalability(all_results, PLOT_FILE)

    # Summary table
    print("\n" + "=" * 52)
    print("  CLIENT SCALABILITY SUMMARY")
    print("=" * 52)
    print(f"{'Clients':>8}  {'Rounds Run':>10}  {'Final mAP@50':>13}")
    print("-" * 40)
    for nc in CLIENT_COUNTS:
        h = all_results[str(nc)]
        print(f"{nc:>8}  {len(h['map50']):>10}  {h['map50'][-1]:>13.4f}")
    print("=" * 52)


if __name__ == '__main__':
    main()
