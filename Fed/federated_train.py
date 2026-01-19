import os
import torch
import torch.utils.data
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import IDIDDataset
from model import get_model
from engine import train_one_epoch, evaluate, print_metrics
from torch.cuda.amp import GradScaler

def collate_fn(batch):
    return tuple(zip(*batch))

class FederatedConfig:
    # FL Settings
    NUM_CLIENTS = 4
    ROUNDS = 20         # Increased rounds for high accuracy
    CLIENT_EPOCHS = 5   # Increased epochs per round
    
    # Paths
    # Updated to user's paths (Relative to Fedlearning/ directory)
    DATA_ROOT = r'../Train_IDID_V1.2/Train/Images'
    ANNOTATION_FILE = r'../Train_IDID_V1.2/Train/labels_v1.2.json'
    OUTPUT_DIR = r'checkpoints'
    
    #Model Settings
    NUM_CLASSES = 2 # Background, Insulator
    BATCH_SIZE = 16  # Adjusted for 800x800 images on standard GPU
    LEARNING_RATE = 0.002 # Standard start for SGD
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def main():
    print(f"Starting Federated Learning for IDID...")
    print(f"Target: >90% Accuracy, >99% mAP@50")
    
    os.makedirs(FederatedConfig.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Dataset...")
    full_dataset = IDIDDataset(
        FederatedConfig.DATA_ROOT, 
        FederatedConfig.ANNOTATION_FILE,
        augment=True,
        target_size=(800, 800) # Resize for speed/memory balance
    )
    
    # 2. Split Train/Val (90/10 split for robust training)
    total_size = len(full_dataset.ids)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    # Create indices
    indices = list(range(total_size))
    random.seed(42) # Reproducibility
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"Total Images: {total_size}")
    print(f"Training Pool: {train_size}")
    print(f"Validation Set: {val_size}")
    
    # Validation Dataset (No augmentation)
    val_dataset = IDIDDataset(
        FederatedConfig.DATA_ROOT,
        FederatedConfig.ANNOTATION_FILE,
        augment=False,
        target_size=(800, 800)
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=4, shuffle=False, 
        num_workers=2, collate_fn=collate_fn
    )
    
    # 3. Distribute Data to Clients
    # We partition the 'train_indices' among clients
    client_datasets = []
    chunk_size = train_size // FederatedConfig.NUM_CLIENTS
    
    for i in range(FederatedConfig.NUM_CLIENTS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < FederatedConfig.NUM_CLIENTS - 1 else train_size
        c_indices = train_indices[start:end]
        
        # Client sees subset of augmented dataset
        subset = torch.utils.data.Subset(full_dataset, c_indices)
        client_datasets.append(subset)
        print(f"Client {i+1} assigned {len(c_indices)} images.")

    # 4. Initialize Global Model
    print("Initializing Global Model (RetinaNet ResNet50 FPN)...")
    global_model = get_model(FederatedConfig.NUM_CLASSES)
    global_model.to(FederatedConfig.DEVICE)
    global_weights = global_model.state_dict()
    
    history = {'map50': [], 'loss': []}
    best_map50 = 0.0
    
    # 5. Training Loop
    scaler = torch.amp.GradScaler('cuda')
    
    for round_idx in range(FederatedConfig.ROUNDS):
        print(f"\n--- Round {round_idx+1}/{FederatedConfig.ROUNDS} ---")
        local_weights = []
        round_losses = []
        
        for client_idx in range(FederatedConfig.NUM_CLIENTS):
            print(f"  Training Client {client_idx+1}...", end=' ')
            
            # Local Model
            client_model = get_model(FederatedConfig.NUM_CLASSES)
            client_model.load_state_dict(copy.deepcopy(global_weights))
            client_model.to(FederatedConfig.DEVICE)
            client_model.train()
            
            optimizer = torch.optim.SGD(
                client_model.parameters(), 
                lr=FederatedConfig.LEARNING_RATE,
                momentum=FederatedConfig.MOMENTUM,
                weight_decay=FederatedConfig.WEIGHT_DECAY
            )
            
            client_loader = torch.utils.data.DataLoader(
                client_datasets[client_idx],
                batch_size=FederatedConfig.BATCH_SIZE,
                shuffle=True, collate_fn=collate_fn,
                num_workers=2
            )
            
            c_loss = 0.0
            for epoch in range(FederatedConfig.CLIENT_EPOCHS):
                metrics = train_one_epoch(
                    client_model, optimizer, client_loader, 
                    FederatedConfig.DEVICE, epoch, scaler=scaler
                )
                c_loss += metrics['total_loss']
            
            round_losses.append(c_loss / FederatedConfig.CLIENT_EPOCHS)
            local_weights.append(client_model.state_dict())
            print(f"Done. Loss: {c_loss/FederatedConfig.CLIENT_EPOCHS:.4f}")
            
            del client_model
            torch.cuda.empty_cache()
            
        # Aggregation
        print("Aggregating weights...")
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        # Evaluation
        print("Evaluating Global Model...")
        res = evaluate(global_model, val_loader, device=FederatedConfig.DEVICE)
        
        print(f"Round {round_idx+1} Results:")
        print(f"  mAP@50: {res['map_50']:.4f}")
        print(f"  Accuracy: {res['accuracy']:.2f}%")
        
        history['map50'].append(res['map_50'])
        
        if res['map_50'] > best_map50:
            best_map50 = res['map_50']
            torch.save(global_model.state_dict(), os.path.join(FederatedConfig.OUTPUT_DIR, 'best_model.pth'))
            print("  *** New Best Model Saved ***")

    print(f"\nFinal Best mAP@50: {best_map50:.4f}")

if __name__ == "__main__":
    main()