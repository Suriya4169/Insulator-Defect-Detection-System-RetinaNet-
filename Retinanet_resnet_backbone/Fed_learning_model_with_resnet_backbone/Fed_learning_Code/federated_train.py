import os
import torch
import torch.utils.data
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import CustomCocoDataset, build_defect_bank, make_weighted_sampler
from transforms import get_transform
from model import get_model
from engine import train_one_epoch, evaluate, print_metrics
from torch.cuda.amp import GradScaler
import time
import subprocess

def get_gpu_temperature():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return int(result.strip())
    except Exception:
        return -1

def collate_fn(batch):
    return tuple(zip(*batch))

class FederatedConfig:
    # FL Settings
    NUM_CLIENTS = 3
    ROUNDS = 8 # Optimized for speed
    CLIENT_EPOCHS = 2 
    
    # Paths
    # Correcting paths to match current project directory
    DATA_PATH = r'D:\Fed learning project\Insulator-Defect-Detection-System-RetinaNet-\chinesedataset_code\IDD-CPLID.v3-cplid_new.coco'
    PRETRAINED_PATH = r'D:\Fed learning project\Insulator-Defect-Detection-System-RetinaNet-\chinesedataset_code\Fed\Code\best_model.pth' 
    OUTPUT_DIR = r'D:\Fed learning project\checkpoints'
    
    # Model Settings
    NUM_CLASSES = 3
    BATCH_SIZE = 16 
    
    # Optimizer & Scheduler Settings (AdamW)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    PCT_START = 0.15
    GRAD_CLIP = 1.0
    
    # Augmentation Settings
    USE_COPY_PASTE = True
    DEFECT_BANK_SIZE = 300
    COPY_PASTE_PROB = 0.35
    
    # Weighted Sampler Settings
    USE_WEIGHTED_SAMPLER = True
    DEFECT_SAMPLE_WEIGHT = 4.0
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def partition_ids(all_ids, num_clients, seed=42):
    """Randomly split dataset IDs across num_clients evenly."""
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    total = len(ids)
    chunk_size = total // num_clients
    partitions = []
    
    for i in range(num_clients):
        if i == num_clients - 1:
            partitions.append(ids[i*chunk_size:])
        else:
            partitions.append(ids[i*chunk_size:(i+1)*chunk_size])
    return partitions

def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience (int): How many rounds to wait after last time score improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def plot_metrics(history, output_dir):
    """Plot and save training metrics."""
    epochs = range(1, len(history['loss']) + 1)
    
    # 1. Epoch vs Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], 'r-', label='Training Loss')
    plt.title('Training Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # 2. Epoch vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['accuracy'], 'b-', label='Accuracy')
    plt.title('Global Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()
    
    # 3. Epoch vs mAP@50
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['map50'], 'g-', label='mAP@50')
    plt.title('Global mAP@50 per Round')
    plt.xlabel('Round')
    plt.ylabel('mAP@50')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'map50_curve.png'))
    plt.close()

    # 4. Epoch vs mAP@75
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['map75'], 'm-', label='mAP@75')
    plt.title('Global mAP@75 per Round')
    plt.xlabel('Round')
    plt.ylabel('mAP@75')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'map75_curve.png'))
    plt.close()

def main():
    print(f"Starting FINAL Push Federated Learning...")
    print(f"Using Device: {FederatedConfig.DEVICE}")
    if FederatedConfig.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Clients: {FederatedConfig.NUM_CLIENTS}")
    
    os.makedirs(FederatedConfig.OUTPUT_DIR, exist_ok=True)
    
    # History tracking
    history = {
        'loss': [],
        'accuracy': [],
        'map50': [],
        'map75': []
    }
    
    best_map50 = 0.0
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.005)
    
    # 1. Prepare Global Data
    print("Loading Global Datasets...", flush=True)
    train_dir = os.path.join(FederatedConfig.DATA_PATH, 'train')
    train_ann = os.path.join(train_dir, '_annotations.coco.json')
    val_dir = os.path.join(FederatedConfig.DATA_PATH, 'valid')
    val_ann = os.path.join(val_dir, '_annotations.coco.json')
    
    defect_bank = None
    if FederatedConfig.USE_COPY_PASTE:
        defect_bank = build_defect_bank(
            train_dir, train_ann, 
            max_crops=FederatedConfig.DEFECT_BANK_SIZE
        )
    
    # Get all training IDs to split among clients
    _full_train = CustomCocoDataset(train_dir, train_ann)
    client_id_splits = partition_ids(_full_train.ids, FederatedConfig.NUM_CLIENTS)
    
    val_dataset = CustomCocoDataset(
        val_dir, val_ann, 
        transforms=get_transform(train=False), 
        augment=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, 
        num_workers=2, collate_fn=collate_fn,
        pin_memory=True
    )
    
    client_datasets = []
    
    for i in range(FederatedConfig.NUM_CLIENTS):
        subset = CustomCocoDataset(
            train_dir, train_ann,
            transforms=get_transform(train=True),
            augment=True,
            defect_bank=defect_bank,
            defect_prob=FederatedConfig.COPY_PASTE_PROB,
            subset_ids=client_id_splits[i]
        )
        client_datasets.append(subset)
        print(f"Client {i+1}: {len(subset)} augmented samples assigned.")

    # 3. Initialize Global Model with Transfer Learning
    print(f"Initializing Global Model from {FederatedConfig.PRETRAINED_PATH}...", flush=True)
    global_model = get_model(FederatedConfig.NUM_CLASSES)
    global_model.to(FederatedConfig.DEVICE)
    
    # if os.path.exists(FederatedConfig.PRETRAINED_PATH):
    #     checkpoint = torch.load(FederatedConfig.PRETRAINED_PATH, map_location=FederatedConfig.DEVICE)
    #     state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    #     global_model.load_state_dict(state_dict, strict=False)
    #     print("Pretrained weights loaded.", flush=True)

    global_weights = global_model.state_dict()
    class_names = {0: 'background', 1: 'defect', 2: 'insulator'}
    
    # 4. FL Training Loop
    latest_acc = 0.0
    latest_map = 0.0

    for round_idx in range(FederatedConfig.ROUNDS):
        print(f"\n--- Round {round_idx+1}/{FederatedConfig.ROUNDS} ---")
        if round_idx > 0:
            print(f"  [Latest Stats: mAP@50: {latest_map:.4f}, Accuracy: {latest_acc:.2f}%]")
            
        local_weights = []
        round_losses = []
        
        for client_idx in range(FederatedConfig.NUM_CLIENTS):
            print(f"  > Training Client {client_idx+1}...", end=' ', flush=True)
            
            client_model = get_model(FederatedConfig.NUM_CLASSES)
            client_model.load_state_dict(copy.deepcopy(global_weights))
            client_model.to(FederatedConfig.DEVICE)
            client_model.train()
            
            params = [p for p in client_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params, 
                lr=FederatedConfig.LEARNING_RATE, 
                weight_decay=FederatedConfig.WEIGHT_DECAY
            )
            
            ds = client_datasets[client_idx]
            sampler = make_weighted_sampler(ds, defect_weight=FederatedConfig.DEFECT_SAMPLE_WEIGHT) if FederatedConfig.USE_WEIGHTED_SAMPLER else None
            
            client_loader = torch.utils.data.DataLoader(
                ds, 
                batch_size=FederatedConfig.BATCH_SIZE, 
                sampler=sampler, shuffle=(sampler is None), 
                collate_fn=collate_fn,
                num_workers=0, 
                pin_memory=True
            )
            
            total_steps = len(client_loader) * FederatedConfig.CLIENT_EPOCHS
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=FederatedConfig.LEARNING_RATE,
                total_steps=total_steps,
                pct_start=FederatedConfig.PCT_START,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )
            
            scaler = torch.amp.GradScaler('cuda')
            
            client_loss = 0.0
            
            for epoch in range(FederatedConfig.CLIENT_EPOCHS):
                metrics = train_one_epoch(
                    client_model, optimizer, client_loader, 
                    FederatedConfig.DEVICE, epoch, 
                    scaler=scaler, scheduler=lr_scheduler, grad_clip=FederatedConfig.GRAD_CLIP,
                    extra_info={'acc': f'{latest_acc:.1f}%'}
                )
                client_loss += metrics['total_loss']
            
            avg_client_loss = client_loss / FederatedConfig.CLIENT_EPOCHS
            round_losses.append(avg_client_loss)
            
            local_weights.append(copy.deepcopy(client_model.state_dict()))
            print(f"Done. Loss: {avg_client_loss:.4f}")
            
            del client_model
            torch.cuda.empty_cache()
            
        # Aggregation (Weighted FedAvg based on dataset size)
        print("Aggregating weights...")
        client_sizes = [len(ds) for ds in client_datasets]
        global_weights = copy.deepcopy(local_weights[0])
        total_size = sum(client_sizes)
        w = [s / total_size for s in client_sizes]
        
        for key in global_weights.keys():
            global_weights[key] = sum(
                wi * lw[key].float() for wi, lw in zip(w, local_weights)
            ).to(global_weights[key].dtype)
            
        global_model.load_state_dict(global_weights)
        
        # Metrics Recording
        avg_round_loss = sum(round_losses) / len(round_losses)
        history['loss'].append(avg_round_loss)
        
        # Evaluation
        print("Evaluating Global Model...")
        res = evaluate(global_model, val_loader, device=FederatedConfig.DEVICE)
        
        latest_acc = res['accuracy']
        latest_map = res['map_50']
        
        history['accuracy'].append(res['accuracy'])
        history['map50'].append(res['map_50'])
        history['map75'].append(res['map_75'])
        
        print_metrics(res, class_names)
        
        # Save Checkpoint
        save_path = os.path.join(FederatedConfig.OUTPUT_DIR, f'global_model_r{round_idx+1}.pth')
        torch.save({
            'round': round_idx,
            'model_state_dict': global_model.state_dict(),
            'history': history
        }, save_path)
        print(f"Saved checkpoint to {save_path}")
        
        # Save Best Model
        if res['map_50'] > best_map50:
            best_map50 = res['map_50']
            best_path = os.path.join(FederatedConfig.OUTPUT_DIR, 'best_global_model.pth')
            torch.save({
                'round': round_idx,
                'model_state_dict': global_model.state_dict(),
                'best_map50': best_map50,
                'history': history
            }, best_path)
            print(f"*** New Best Model Saved (mAP@50: {best_map50:.4f}) ***")

        # Early Stopping
        early_stopping(res['map_50'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered at round {round_idx+1}")
            break

    # Plotting
    print("\nPlotting metrics...")
    plot_metrics(history, FederatedConfig.OUTPUT_DIR)
    print("Federated Learning Complete.")

if __name__ == "__main__":
    main()
