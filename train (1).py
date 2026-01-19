import os
import torch
import torch.utils.data
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import BaselineDataset
from model import get_model
from engine import train_one_epoch, evaluate
from transforms import get_transform

def collate_fn(batch):
    return tuple(zip(*batch))

class Config:
    # Path configuration
    DATA_ROOT = r'../Train_IDID_V1.2/Train/Images'
    ANNOTATION_FILE = r'../Train_IDID_V1.2/Train/labels_v1.2.json'
    OUTPUT_DIR = r'checkpoints'
    
    # Model configuration
    NUM_CLASSES = 5 
    BATCH_SIZE = 8
    LEARNING_RATE = 0.002
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    EPOCHS = 25 
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_loss_plot(history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b')
    plt.title('RetinaNet Baseline: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss_plot.png'))
    plt.close()

def main():
    print("-------------------------------------------------------")
    print("Baseline Model Training (Centralized RetinaNet)")
    print(f"Environment: {Config.DEVICE}")
    print("-------------------------------------------------------")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize dataset and loaders
    dataset = BaselineDataset(Config.DATA_ROOT, Config.ANNOTATION_FILE, transforms=get_transform(train=True), target_size=(800, 800))
    
    indices = list(range(len(dataset.ids)))
    random.seed(42)
    random.shuffle(indices)
    
    split = int(0.9 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices), 
        batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices), 
        batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # Model and Optimizer setup
    model = get_model(Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    
    best_map = 0.0
    loss_history = []
    
    print(f"Training Duration: {Config.EPOCHS} Epochs")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("-------------------------------------------------------")
    
    import time
    start_time = time.time()
    MAX_TRAIN_TIME = 2 * 3600 # 2 hours in seconds
    
    for epoch in range(Config.EPOCHS):
        # Training
        avg_loss = train_one_epoch(model, optimizer, train_loader, Config.DEVICE, epoch)
        loss_history.append(avg_loss)
        
        # Validation
        print("Running Evaluation Pass...")
        mAP = evaluate(model, val_loader, device=Config.DEVICE)
        
        # Checkpointing
        if mAP > best_map:
            best_map = mAP
            save_path = os.path.join(Config.OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

        # Time check
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_TRAIN_TIME:
            print(f"\nTime limit reached ({elapsed_time/3600:.2f}h). Saving current state and exiting.")
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, f'model_epoch_{epoch}_timeout.pth'))
            break

    # Finalization
    print("\n-------------------------------------------------------")
    print("Training Cycle Complete")
    save_loss_plot(loss_history, Config.OUTPUT_DIR)
    print(f"Peak mAP@50 recorded: {best_map:.4f}")
    print(f"Loss analytics generated in: {Config.OUTPUT_DIR}")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()