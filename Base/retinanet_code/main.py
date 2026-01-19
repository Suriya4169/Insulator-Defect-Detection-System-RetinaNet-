import os
import torch
import torch.utils.data
from dataset import CustomCocoDataset
from transforms import get_transform
from model import get_model
from engine import train_one_epoch, evaluate, print_metrics
from torch.cuda.amp import GradScaler

def collate_fn(batch):
    return tuple(zip(*batch))

class Config:
    # Paths
    DATA_PATH = r'D:\CLID\IDD-CPLID.v3-cplid_new.coco'
    OUTPUT_DIR = r'D:\CLID\checkpoints'
    
    # Model
    NUM_CLASSES = 3  # background (0), defect (1), insulator (2)
    
    # Training - Precision Tuning for 95%
    BATCH_SIZE = 8  
    NUM_EPOCHS = 100 # Increased to allow for deep fine-tuning
    NUM_WORKERS = 4
    
    # Optimizer - Lowered LR for better precision
    LEARNING_RATE = 0.0005 
    MOMENTUM = 0.93 # Slightly higher momentum
    WEIGHT_DECAY = 0.0005 # Stronger regularization
    
    # Learning Rate Scheduler
    LR_SCHEDULER = 'cosine'
    
    # Mixed Precision
    USE_AMP = True
    
    # Gradient Clipping
    GRAD_CLIP = 1.0
    
    # Early Stopping - More patient for high accuracy
    EARLY_STOPPING_PATIENCE = 20
    
    # Data Augmentation
    USE_AUGMENTATION = True
    
    # Evaluation
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Class names
    CLASS_NAMES = {0: 'background', 1: 'defect', 2: 'insulator'}

def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cudnn.benchmark = True

    # 1. Dataset & Dataloaders
    print("Initializing datasets...", flush=True)
    train_dir = os.path.join(config.DATA_PATH, 'train')
    train_ann = os.path.join(train_dir, '_annotations.coco.json')
    val_dir = os.path.join(config.DATA_PATH, 'valid')
    val_ann = os.path.join(val_dir, '_annotations.coco.json')
    
    dataset_train = CustomCocoDataset(
        train_dir, train_ann, 
        transforms=get_transform(train=True), 
        augment=config.USE_AUGMENTATION
    )
    dataset_val = CustomCocoDataset(
        val_dir, val_ann, 
        transforms=get_transform(train=False), 
        augment=False
    )

    print(f"Training samples: {len(dataset_train)}", flush=True)
    print(f"Validation samples: {len(dataset_val)}", flush=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True
    )

    # 2. Model
    print("Creating model...", flush=True)
    model = get_model(config.NUM_CLASSES)
    model.to(device)
    
    # Print Model Summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*50, flush=True)
    print(f"MODEL SUMMARY: RetinaNet (ResNet50 FPN)", flush=True)
    print(f"Total Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"Classes: {config.CLASS_NAMES}", flush=True)
    print("="*50 + "\n", flush=True)

    # 3. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP and torch.cuda.is_available() else None

    # 4. Training Loop
    best_f1 = 0.0
    patience_counter = 0
    
    print(f"Starting training for {config.NUM_EPOCHS} epochs...", flush=True)
    
    for epoch in range(config.NUM_EPOCHS):
        train_metrics = train_one_epoch(model, optimizer, data_loader_train, device, epoch, scaler=scaler, grad_clip=config.GRAD_CLIP)
        lr_scheduler.step()
        
        # Evaluate
        val_results = evaluate(
            model, data_loader_val, device,
            score_threshold=0.05, # Low threshold for AP calc
            iou_thresholds=[0.5, 0.75]
        )
        
        # Record history
        history['train_loss'].append(train_metrics['total_loss'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_box_loss'].append(train_metrics['box_loss'])
        history['val_f1'].append(val_results['map_50']) # Using mAP@50 as main metric now
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS-1}", flush=True)
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
              f"(cls: {train_metrics['cls_loss']:.4f}, box: {train_metrics['box_loss']:.4f})", flush=True)
        print(f"  Val mAP@50: {val_results['map_50']:.4f} | mAP@75: {val_results['map_75']:.4f}", flush=True)
        print(f"  LR: {current_lr:.6f}", flush=True)
        
        # Print detailed metrics
        print_metrics(val_results, config.CLASS_NAMES)
        
        # Save best model
        val_map = val_results['map_50']
        if val_map > best_f1:
            best_f1 = val_map
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map50': best_f1,
                'config': vars(config)
            }, os.path.join(config.OUTPUT_DIR, 'best_retinanet.pth'))
            print(f"  ✓ Saved new best model (mAP@50: {best_f1:.4f})", flush=True)
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}", flush=True)
            break
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Best mAP@50: {best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()