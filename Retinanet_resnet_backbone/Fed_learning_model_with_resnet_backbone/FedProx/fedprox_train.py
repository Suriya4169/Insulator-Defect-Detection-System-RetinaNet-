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

# ── resolve sibling imports (Fed_learning_Code lives one level up) ─────────────
_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'Retinanet_resnet_backbone',
                         'Fed_learning_model_with_resnet_backbone',
                         'Fed_learning_Code')
sys.path.insert(0, _CODE_DIR)

from dataset    import CustomCocoDataset, build_defect_bank, make_weighted_sampler
from transforms import get_transform
from model      import get_model
from engine     import evaluate, print_metrics          # evaluation is unchanged

# ── FedProx-aware training engine (lives in this folder) ──────────────────────
from engine_fedprox import train_one_epoch_fedprox

# ─────────────────────────────────────────────────────────────────────────────
# Config  (mirrors FederatedConfig in the original federated_train.py)
# ─────────────────────────────────────────────────────────────────────────────

class FedProxConfig:
    # ── FL ────────────────────────────────────────────────────────────────────
    NUM_CLIENTS   = 3
    ROUNDS        = 8
    CLIENT_EPOCHS = 2

    # ── Paths  ── UPDATE THESE to match your system ───────────────────────────
    DATA_PATH  = r'D:\Fed learning project\Dataset - IDD-CPLID.v3-cplid_new.coco'
    OUTPUT_DIR = r'D:\Fed learning project\checkpoints\fedprox'

    # ── Model ─────────────────────────────────────────────────────────────────
    NUM_CLASSES = 3

    # ── Optimiser / Scheduler ─────────────────────────────────────────────────
    BATCH_SIZE    = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    PCT_START     = 0.15
    GRAD_CLIP     = 1.0

    # ── Augmentation ──────────────────────────────────────────────────────────
    USE_COPY_PASTE      = True
    DEFECT_BANK_SIZE    = 300
    COPY_PASTE_PROB     = 0.35
    USE_WEIGHTED_SAMPLER = True
    DEFECT_SAMPLE_WEIGHT = 4.0

    DEVICE = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (identical to the originals)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def partition_ids(all_ids, num_clients, seed=42):
    """IID: random even split."""
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    chunk = len(ids) // num_clients
    return [
        ids[i * chunk:] if i == num_clients - 1
        else ids[i * chunk:(i + 1) * chunk]
        for i in range(num_clients)
    ]


def partition_ids_non_iid(dataset, num_clients, alpha=0.5, seed=42):
    """
    Dirichlet Non-IID label-skew partitioning.

    alpha < 1  → heterogeneous;  alpha = 0.1 = pathological skew (worst case)
    """
    rng = np.random.default_rng(seed)

    defect_ids, normal_ids = [], []
    for img_id in dataset.ids:
        anns   = dataset.img_to_anns.get(img_id, [])
        labels = {a['category_id'] for a in anns}
        (defect_ids if 1 in labels else normal_ids).append(img_id)

    partitions = [[] for _ in range(num_clients)]
    for pool in [defect_ids, normal_ids]:
        if not pool:
            continue
        pool = list(pool);  rng.shuffle(pool)
        props  = rng.dirichlet(np.ones(num_clients) * alpha)
        counts = (props * len(pool)).astype(int)
        counts[-1] += len(pool) - counts.sum()
        counts  = np.maximum(counts, 0)
        idx = 0
        for i, cnt in enumerate(counts):
            partitions[i].extend(pool[idx: idx + cnt])
            idx += cnt
        if idx < len(pool):
            partitions[-1].extend(pool[idx:])

    for i, part in enumerate(partitions):
        n_d = sum(1 for img_id in part
                  if 1 in {a['category_id']
                           for a in dataset.img_to_anns.get(img_id, [])})
        print(f"    [Non-IID α={alpha}] Client {i+1}: "
              f"{len(part)} imgs, {n_d} defect ({100.*n_d/max(len(part),1):.1f}%)")
    return partitions


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter    = 0


def plot_metrics(history, output_dir, tag=''):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['loss']) + 1)
    for key, color, ylabel in [
        ('loss',     'r', 'Loss'),
        ('accuracy', 'b', 'Accuracy (%)'),
        ('map50',    'g', 'mAP@50'),
        ('map75',    'm', 'mAP@75'),
    ]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history[key], f'{color}-o', label=ylabel)
        plt.title(f'{ylabel} per Round  [{tag}]')
        plt.xlabel('Round');  plt.ylabel(ylabel)
        plt.grid(True);       plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{key}_curve_{tag}.png'))
        plt.close()

    # ── prox_loss (FedProx only) ──────────────────────────────────────────────
    if 'prox_loss' in history and any(v > 0 for v in history['prox_loss']):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['prox_loss'], 'c-o', label='Proximal Loss')
        plt.title(f'Proximal Loss per Round  [{tag}]')
        plt.xlabel('Round');  plt.ylabel('Proximal Loss')
        plt.grid(True);       plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prox_loss_curve_{tag}.png'))
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment runner  —  FedAvg OR FedProx
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    num_clients: int    = FedProxConfig.NUM_CLIENTS,
    partition_mode: str = 'iid',      # 'iid' | 'non_iid'
    alpha: float        = 0.5,         # Dirichlet α (ignored for IID)
    output_tag: str     = '',
    rounds: int         = FedProxConfig.ROUNDS,
    algorithm: str      = 'fedavg',    # 'fedavg' | 'fedprox'
    mu: float           = 0.0,         # proximal coefficient (0 = FedAvg)
) -> dict:
    """
    Run one complete FL experiment and return the history dict.

    When algorithm='fedavg' or mu=0, behaviour is identical to the original
    federated_train.py::run_experiment().

    Returns
    -------
    dict with keys: loss, accuracy, map50, map75, prox_loss
    """
    set_seed(42)

    algo_label = f'{algorithm.upper()}' + (f'_mu{mu}' if algorithm == 'fedprox' else '')
    tag_str    = output_tag or f'{partition_mode}_c{num_clients}_{algo_label}'
    out_dir    = os.path.join(FedProxConfig.OUTPUT_DIR, tag_str)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT : {tag_str}")
    print(f"  Algorithm  : {algorithm.upper()}  |  mu = {mu}")
    print(f"  Clients    : {num_clients}  |  Partition: {partition_mode}"
          + (f"  alpha={alpha}" if partition_mode == 'non_iid' else ""))
    print(f"  Rounds     : {rounds}  |  Device: {FedProxConfig.DEVICE}")
    if FedProxConfig.DEVICE.type == 'cuda':
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"{'='*72}\n")

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print("Loading datasets...", flush=True)
    train_dir = os.path.join(FedProxConfig.DATA_PATH, 'train')
    train_ann = os.path.join(train_dir, '_annotations.coco.json')
    val_dir   = os.path.join(FedProxConfig.DATA_PATH, 'valid')
    val_ann   = os.path.join(val_dir,   '_annotations.coco.json')

    defect_bank = None
    if FedProxConfig.USE_COPY_PASTE:
        defect_bank = build_defect_bank(train_dir, train_ann,
                                        max_crops=FedProxConfig.DEFECT_BANK_SIZE)

    _full_train = CustomCocoDataset(train_dir, train_ann)

    # ── 2. Partition data ─────────────────────────────────────────────────────
    if partition_mode == 'non_iid':
        print(f"  Non-IID Dirichlet partitioning (alpha={alpha})...")
        client_splits = partition_ids_non_iid(_full_train, num_clients, alpha=alpha)
    else:
        print(f"  IID partitioning across {num_clients} clients...")
        client_splits = partition_ids(_full_train.ids, num_clients)

    val_dataset = CustomCocoDataset(val_dir, val_ann,
                                    transforms=get_transform(train=False),
                                    augment=False)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    client_datasets = []
    for i in range(num_clients):
        ds = CustomCocoDataset(
            train_dir, train_ann,
            transforms=get_transform(train=True),
            augment=True,
            defect_bank=defect_bank,
            defect_prob=FedProxConfig.COPY_PASTE_PROB,
            subset_ids=client_splits[i]
        )
        client_datasets.append(ds)
        print(f"  Client {i+1}: {len(ds)} samples assigned.")

    # ── 3. Initialise global model ────────────────────────────────────────────
    global_model   = get_model(FedProxConfig.NUM_CLASSES)
    global_model.to(FedProxConfig.DEVICE)
    global_weights = global_model.state_dict()
    class_names    = {0: 'background', 1: 'defect', 2: 'insulator'}

    history    = {'loss': [], 'accuracy': [], 'map50': [], 'map75': [],
                  'prox_loss': []}
    best_map50 = 0.0
    latest_acc = 0.0
    latest_map = 0.0
    early_stop = EarlyStopping(patience=3, min_delta=0.005)

    # ── 4. FL rounds ──────────────────────────────────────────────────────────
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx+1}/{rounds}  "
              f"[{algo_label}  mAP@50={latest_map:.4f}, "
              f"Acc={latest_acc:.2f}%] ---")

        # ── Freeze global params for the proximal term ────────────────────────
        #    (zero cost when mu == 0 because _compute_proximal_loss fast-returns)
        global_params_frozen = [p.detach().clone()
                                 for p in global_model.parameters()]

        local_weights  = []
        round_losses   = []
        round_prox     = []

        for client_idx in range(num_clients):
            print(f"  > Client {client_idx+1}/{num_clients}...",
                  end=' ', flush=True)

            # ── local model initialised from global weights ───────────────────
            client_model = get_model(FedProxConfig.NUM_CLASSES)
            client_model.load_state_dict(copy.deepcopy(global_weights))
            client_model.to(FedProxConfig.DEVICE)
            client_model.train()

            params    = [p for p in client_model.parameters()
                         if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=FedProxConfig.LEARNING_RATE,
                weight_decay=FedProxConfig.WEIGHT_DECAY
            )

            ds      = client_datasets[client_idx]
            sampler = (
                make_weighted_sampler(ds,
                    defect_weight=FedProxConfig.DEFECT_SAMPLE_WEIGHT)
                if FedProxConfig.USE_WEIGHTED_SAMPLER else None
            )
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=FedProxConfig.BATCH_SIZE,
                sampler=sampler, shuffle=(sampler is None),
                collate_fn=collate_fn,
                num_workers=0, pin_memory=True
            )

            total_steps  = max(1, len(loader) * FedProxConfig.CLIENT_EPOCHS)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=FedProxConfig.LEARNING_RATE,
                total_steps=total_steps,
                pct_start=FedProxConfig.PCT_START,
                anneal_strategy='cos',
                div_factor=10.0, final_div_factor=100.0
            )

            scaler      = torch.amp.GradScaler('cuda') \
                          if FedProxConfig.DEVICE.type == 'cuda' else None
            client_loss = 0.0
            client_prox = 0.0

            for epoch in range(FedProxConfig.CLIENT_EPOCHS):
                # ── KEY DIFFERENCE FROM FedAvg ────────────────────────────────
                #    global_params_frozen  : the frozen reference weights
                #    mu                    : proximal strength  (0 ≡ FedAvg)
                metrics = train_one_epoch_fedprox(
                    model=client_model,
                    optimizer=optimizer,
                    data_loader=loader,
                    device=FedProxConfig.DEVICE,
                    epoch=epoch,
                    global_params=global_params_frozen,   # ← FedProx
                    mu=mu,                                 # ← FedProx
                    scaler=scaler,
                    scheduler=lr_scheduler,
                    grad_clip=FedProxConfig.GRAD_CLIP,
                    extra_info={'acc': f'{latest_acc:.1f}%'}
                )
                client_loss += metrics['total_loss']
                client_prox += metrics['prox_loss']

            avg_loss = client_loss / FedProxConfig.CLIENT_EPOCHS
            avg_prox = client_prox / FedProxConfig.CLIENT_EPOCHS
            round_losses.append(avg_loss)
            round_prox.append(avg_prox)
            local_weights.append(copy.deepcopy(client_model.state_dict()))
            print(f"Done.  Loss={avg_loss:.4f}  ProxTerm={avg_prox:.5f}")

            del client_model
            torch.cuda.empty_cache()

        # ── Weighted FedAvg aggregation (UNCHANGED for FedProx) ───────────────
        print("  Aggregating weights (Weighted FedAvg)...")
        sizes  = [len(ds) for ds in client_datasets]
        total  = sum(sizes)
        w_frac = [s / total for s in sizes]

        global_weights = copy.deepcopy(local_weights[0])
        for key in global_weights.keys():
            global_weights[key] = sum(
                wf * lw[key].float()
                for wf, lw in zip(w_frac, local_weights)
            ).to(global_weights[key].dtype)
        global_model.load_state_dict(global_weights)

        # ── Record metrics ────────────────────────────────────────────────────
        history['loss'].append(sum(round_losses) / len(round_losses))
        history['prox_loss'].append(sum(round_prox) / len(round_prox))

        print("  Evaluating global model...", flush=True)
        res        = evaluate(global_model, val_loader,
                              device=FedProxConfig.DEVICE)
        latest_acc = res['accuracy']
        latest_map = res['map_50']
        history['accuracy'].append(res['accuracy'])
        history['map50'].append(res['map_50'])
        history['map75'].append(res['map_75'])
        print_metrics(res, class_names)

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt = {
            'round':             round_idx,
            'algorithm':         algorithm,
            'mu':                mu,
            'model_state_dict':  global_model.state_dict(),
            'history':           history,
        }
        torch.save(ckpt,
                   os.path.join(out_dir, f'global_model_r{round_idx+1}.pth'))

        if res['map_50'] > best_map50:
            best_map50 = res['map_50']
            torch.save(ckpt, os.path.join(out_dir, 'best_global_model.pth'))
            print(f"  *** New Best mAP@50: {best_map50:.4f} – saved ***")

        # ── Early stopping ────────────────────────────────────────────────────
        early_stop(res['map_50'])
        if early_stop.early_stop:
            print(f"  Early stopping at round {round_idx+1}.")
            break

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(out_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {hist_path}")

    plot_metrics(history, out_dir, tag=algo_label)
    print(f"  Experiment '{tag_str}' done.  Best mAP@50 = {best_map50:.4f}\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='FedProx Federated Training (Phase 1 — Journal Upgrade)'
    )
    p.add_argument('--algorithm',  choices=['fedavg', 'fedprox'],
                   default='fedavg',
                   help='fedavg (mu forced to 0) or fedprox')
    p.add_argument('--mu',         type=float, default=0.01,
                   help='FedProx proximal coefficient. Ignored for fedavg.')
    p.add_argument('--partition',  choices=['iid', 'non_iid'],
                   default='iid',
                   help='Data partitioning strategy')
    p.add_argument('--alpha',      type=float, default=0.5,
                   help='Dirichlet alpha for Non-IID (lower = more skewed)')
    p.add_argument('--clients',    type=int, default=FedProxConfig.NUM_CLIENTS)
    p.add_argument('--rounds',     type=int, default=FedProxConfig.ROUNDS)
    p.add_argument('--tag',        type=str, default='',
                   help='Optional custom output tag')
    return p.parse_args()


def main():
    args = parse_args()
    mu   = 0.0 if args.algorithm == 'fedavg' else args.mu
    run_experiment(
        num_clients    = args.clients,
        partition_mode = args.partition,
        alpha          = args.alpha,
        output_tag     = args.tag,
        rounds         = args.rounds,
        algorithm      = args.algorithm,
        mu             = mu,
    )


if __name__ == '__main__':
    main()
