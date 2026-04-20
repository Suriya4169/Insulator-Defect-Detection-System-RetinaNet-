"""
fedprox_train_mobilenet.py
==========================
Federated Learning training loop for the MobileNetV2-RetinaNet backbone.

This is the MobileNet equivalent of the ResNet fedprox_train.py.
It supports both FedAvg and FedProx, IID and Non-IID data partitioning,
and is designed to produce results that can be directly compared with the
ResNet experiments for the backbone benchmarking section of the paper.

Usage
-----
    # Run a single IID FedAvg experiment (test that everything works)
    python fedprox_train_mobilenet.py --partition iid --algorithm fedavg

    # Run the full FedProx Non-IID comparison
    python compare_fedavg_vs_fedprox_mobilenet.py
"""

import os
import sys
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── resolve path: shared code lives alongside this file ───────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
# Also pull in the ResNet Fed_learning_Code directory for dataset/engine/transforms
# (These are backbone-agnostic and shared between both backbone experiments)
_RESNET_CODE = os.path.join(
    _HERE, '..', '..',
    'Retinanet_resnet_backbone',
    'Fed_learning_model_with_resnet_backbone',
    'Fed_learning_Code'
)
sys.path.insert(0, os.path.normpath(_RESNET_CODE))
sys.path.insert(0, _HERE)

from dataset           import CustomCocoDataset, build_defect_bank, make_weighted_sampler
from transforms        import get_transform
from engine            import evaluate, print_metrics
from model_mobilenet   import get_model
from engine_fedprox_mb import train_one_epoch_fedprox
from torch.amp         import GradScaler


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetFedConfig:
    """All hyperparameters for the MobileNet federated experiments."""

    # ── FL ────────────────────────────────────────────────────────────────────
    NUM_CLIENTS   = 3
    ROUNDS        = 8
    CLIENT_EPOCHS = 2

    # ── Paths  ── UPDATE DATA_PATH to match your system ────────────────────
    DATA_PATH  = r'D:\Fed learning project\Dataset - IDD-CPLID.v3-cplid_new.coco'
    OUTPUT_DIR = r'D:\Fed learning project\checkpoints\mobilenet_fedprox'

    # ── Model ─────────────────────────────────────────────────────────────────
    NUM_CLASSES = 3   # Background, Defect, Insulator

    # ── Optimiser / Scheduler ─────────────────────────────────────────────────
    BATCH_SIZE    = 4    # Balanced for CPU-GPU throughput on laptop hardware
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    PCT_START     = 0.15
    GRAD_CLIP     = 1.0

    # ── Copy-Paste Augmentation ────────────────────────────────────────────────
    USE_COPY_PASTE    = True
    DEFECT_BANK_SIZE  = 300
    COPY_PASTE_PROB   = 0.35

    # ── Weighted Sampler ───────────────────────────────────────────────────────
    USE_WEIGHTED_SAMPLER = True
    DEFECT_SAMPLE_WEIGHT = 4.0

    # ── Early Stopping ─────────────────────────────────────────────────────────
    PATIENCE = 3

    # ── Device ─────────────────────────────────────────────────────────────────
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    NUM_WORKERS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   # Enable for speed boost


def collate_fn(batch):
    return tuple(zip(*batch))


def iid_partition(all_ids, num_clients, seed=42):
    """Randomly split IDs evenly across clients (IID setting)."""
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    chunk = len(ids) // num_clients
    return [ids[i * chunk:(i + 1) * chunk] for i in range(num_clients)]


def noniid_dirichlet_partition(dataset, num_clients, alpha=0.1, seed=42):
    """
    Dirichlet partition for Non-IID simulation.
    Concentrates defect images at specific clients (alpha controls severity).
    alpha=0.1 → extreme heterogeneity, alpha=1.0 → near-IID.
    """
    rng = np.random.default_rng(seed)

    all_ids    = list(dataset.ids)
    defect_ids = []
    clean_ids  = []

    # Identify defect vs clean images using the dataset's own annotation map
    for img_id in all_ids:
        anns    = dataset.img_to_anns.get(img_id, [])
        cat_ids = {a['category_id'] for a in anns}
        names   = {dataset.categories.get(c, '') for c in cat_ids}
        if 'defect' in names:
            defect_ids.append(img_id)
        else:
            clean_ids.append(img_id)

    # Dirichlet allocation of defect images
    props = rng.dirichlet([alpha] * num_clients)
    splits = (props * len(defect_ids)).astype(int)
    splits[-1] = len(defect_ids) - splits[:-1].sum()  # fix rounding

    rng.shuffle(defect_ids)
    client_ids = []
    start = 0
    for n in splits:
        client_ids.append(defect_ids[start:start + n])
        start += n

    # Distribute clean images evenly
    rng.shuffle(clean_ids)
    clean_chunk = len(clean_ids) // num_clients
    for i in range(num_clients):
        client_ids[i] += clean_ids[i * clean_chunk:(i + 1) * clean_chunk]

    for i, ids in enumerate(client_ids):
        d_count = sum(1 for iid in ids if iid in set(defect_ids))
        pct = 100 * d_count / max(len(ids), 1)
        print(f"    [Non-IID α={alpha}] Client {i+1}: {len(ids)} imgs, "
              f"{d_count} defect ({pct:.1f}%)")

    return client_ids


def fedavg_aggregate(global_model, client_models, client_sizes):
    """Weighted FedAvg aggregation based on local dataset sizes."""
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())

    for key in new_state:
        new_state[key] = sum(
            (client_models[i].state_dict()[key].float() * client_sizes[i] / total)
            for i in range(len(client_models))
        )

    global_model.load_state_dict(new_state)
    return global_model


# ─────────────────────────────────────────────────────────────────────────────
# Core Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    partition_mode='iid',
    alpha=0.1,
    output_tag='mobilenet_fedavg_iid',
    algorithm='fedavg',
    mu=0.0,
    cfg=None,
    use_copy_paste=None,
    use_weighted_sampling=None,
):
    """
    Run one full Federated Learning experiment with MobileNetV2 backbone.

    Args:
        partition_mode : 'iid' or 'non_iid'
        alpha          : Dirichlet concentration (only used if non_iid)
        output_tag     : Subfolder name under OUTPUT_DIR
        algorithm      : 'fedavg' or 'fedprox'
        mu             : Proximal coefficient (0.0 = pure FedAvg)
        cfg            : Optional config class override

    Returns:
        history (dict): {'map50': [...], 'map75': [...], 'acc': [...]}
    """
    if cfg is None:
        cfg = MobileNetFedConfig

    set_seed(42)
    device    = cfg.DEVICE
    out_dir   = os.path.join(cfg.OUTPUT_DIR, output_tag)
    os.makedirs(out_dir, exist_ok=True)

    algo_str = algorithm.upper()
    mu_str   = f"_mu{str(mu).replace('.', '')}" if algorithm == 'fedprox' else ""

    print("\n" + "=" * 72)
    print(f"  EXPERIMENT : {output_tag}")
    print(f"  Backbone   : MobileNetV2 (RetinaNet)")
    print(f"  Algorithm  : {algo_str}  |  mu = {mu}")
    print(f"  Clients    : {cfg.NUM_CLIENTS}  |  Partition: {partition_mode}")
    print(f"  Rounds     : {cfg.ROUNDS}  |  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    # ── Load datasets ────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_root = os.path.join(cfg.DATA_PATH, 'train')
    val_root   = os.path.join(cfg.DATA_PATH, 'valid')
    train_ann  = os.path.join(train_root, '_annotations.coco.json')
    val_ann    = os.path.join(val_root,   '_annotations.coco.json')

    # Build defect crop bank for Copy-Paste augmentation
    defect_bank = build_defect_bank(
        root=train_root,
        ann_file=train_ann,
        max_crops=cfg.DEFECT_BANK_SIZE
    )

    # Template dataset (no augmentation) — used only for partitioning IDs
    full_train = CustomCocoDataset(
        root=train_root,
        annotation_file=train_ann,
        transforms=get_transform(train=False),
        augment=False,
    )

    # ── Partition data across clients ─────────────────────────────────────────
    if partition_mode == 'iid':
        print(f"  IID partitioning across {cfg.NUM_CLIENTS} clients...")
        all_ids         = list(full_train.ids)
        client_id_lists = iid_partition(all_ids, cfg.NUM_CLIENTS)
    else:
        print(f"  Non-IID Dirichlet partitioning (alpha={alpha})...")
        client_id_lists = noniid_dirichlet_partition(
            full_train, cfg.NUM_CLIENTS, alpha=alpha
        )

    client_datasets = []
    client_sizes    = []
    for i, ids in enumerate(client_id_lists):
        cp_enabled = use_copy_paste if use_copy_paste is not None else cfg.USE_COPY_PASTE
        ds = CustomCocoDataset(
            root=train_root,
            annotation_file=train_ann,
            transforms=get_transform(train=True),
            augment=cp_enabled,
            defect_bank=defect_bank if cp_enabled else None,
            defect_prob=cfg.COPY_PASTE_PROB,
            subset_ids=ids,
        )
        client_datasets.append(ds)
        client_sizes.append(len(ids))
        print(f"  Client {i+1}: {len(ids)} samples assigned.")

    val_dataset = CustomCocoDataset(
        root=val_root,
        annotation_file=val_ann,
        transforms=get_transform(train=False),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=collate_fn
    )

    # ── Initialise global model ───────────────────────────────────────────────
    global_model = get_model(cfg.NUM_CLASSES).to(device)
    best_map = 0.0
    patience_counter = 0
    history = {'map50': [], 'map75': [], 'acc': []}

    # ── Federated Training Loop ───────────────────────────────────────────────
    for rnd in range(1, cfg.ROUNDS + 1):
        map50_prev = history['map50'][-1] if history['map50'] else 0.0
        acc_prev   = history['acc'][-1]   if history['acc']   else 0.0
        print(f"\n--- Round {rnd}/{cfg.ROUNDS}  "
              f"[{algo_str}{mu_str}  mAP@50={map50_prev:.4f}, "
              f"Acc={acc_prev:.2f}%] ---")

        client_models = []
        global_weights = copy.deepcopy(global_model.state_dict())

        # ── Client Training ───────────────────────────────────────────────────
        for i, ds in enumerate(client_datasets):
            # ── Data Loading with optional Weighted Sampling ────────────────
            ws_enabled = use_weighted_sampling if use_weighted_sampling is not None else cfg.USE_WEIGHTED_SAMPLER
            
            if ws_enabled:
                from torch.utils.data import WeightedRandomSampler
                # Use pre-cached has_defect list from Modified dataset.py
                sample_weights = [5.0 if has_def else 1.0 for has_def in ds.has_defect]
                sampler = WeightedRandomSampler(sample_weights, len(ds))
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                    num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn
                )
            else:
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                    num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn
                )

            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY
            )
            total_steps = cfg.CLIENT_EPOCHS * max(len(loader), 1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.LEARNING_RATE,
                total_steps=total_steps, pct_start=cfg.PCT_START
            )

            scaler = GradScaler('cuda')
            epoch_losses = []
            for epoch in range(cfg.CLIENT_EPOCHS):
                avg_loss, avg_prox = train_one_epoch_fedprox(
                    model=local_model,
                    global_weights=global_weights,
                    optimizer=optimizer,
                    data_loader=loader,
                    device=device,
                    epoch=epoch,
                    mu=mu,
                    scaler=scaler,
                    scheduler=scheduler,
                    grad_clip=cfg.GRAD_CLIP,
                    desc_prefix=f"[MobileNet {algo_str} μ={mu}]"
                )
                epoch_losses.append(avg_loss)
                print(f"    Epoch {epoch} | TotalLoss: {avg_loss:.4f}  "
                      f"ProxLoss: {avg_prox:.5f}")

            avg_client_loss = sum(epoch_losses) / len(epoch_losses)
            avg_prox_last   = avg_prox
            print(f"Done.  Loss={avg_client_loss:.4f}  "
                  f"ProxTerm={avg_prox_last:.5f}")
            client_models.append(local_model)

        # ── Aggregation ───────────────────────────────────────────────────────
        print("  Aggregating weights (Weighted FedAvg)...")
        global_model = fedavg_aggregate(global_model, client_models, client_sizes)

        # ── Global Evaluation ─────────────────────────────────────────────────
        print("  Evaluating global model...")
        metrics = evaluate(global_model, val_loader, device)
        # Fix: Pass categories for the professional report
        print_metrics(metrics, val_dataset.categories)

        map50 = metrics.get('map_50', 0.0)
        map75 = metrics.get('map_75', 0.0)
        acc   = metrics.get('accuracy', 0.0)
        history['map50'].append(map50)
        history['map75'].append(map75)
        history['acc'].append(acc)

        # ── Best model checkpoint ─────────────────────────────────────────────
        if map50 > best_map:
            best_map = map50
            patience_counter = 0
            ckpt = os.path.join(out_dir, 'best_global_model.pt')
            torch.save(global_model.state_dict(), ckpt)
            print(f"  *** New Best mAP@50: {best_map:.4f} – saved ***")
        else:
            patience_counter += 1
            print(f"  EarlyStopping: {patience_counter}/{cfg.PATIENCE}")
            if patience_counter >= cfg.PATIENCE:
                print(f"  Early stopping at round {rnd}.")
                break

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(out_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {hist_path}")
    print(f"  Experiment '{output_tag}' done.  Best mAP@50 = {best_map:.4f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI for standalone use
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='MobileNetV2 FedProx / FedAvg training'
    )
    p.add_argument('--partition',  default='iid',
                   choices=['iid', 'non_iid'])
    p.add_argument('--alpha',      type=float, default=0.1)
    p.add_argument('--algorithm',  default='fedavg',
                   choices=['fedavg', 'fedprox'])
    p.add_argument('--mu',         type=float, default=0.0)
    p.add_argument('--output-tag', default='mobilenet_fedavg_iid')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(
        partition_mode=args.partition,
        alpha=args.alpha,
        output_tag=args.output_tag,
        algorithm=args.algorithm,
        mu=args.mu,
    )
