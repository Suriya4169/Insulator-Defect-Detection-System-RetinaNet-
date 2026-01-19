import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import json
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
import datetime
import seaborn as sns
import time
import sys
import random

# Local project imports
from model import get_model
from transforms import get_transform
from dataset import BaselineDataset

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def collate_fn(batch):
    return tuple(zip(*batch))

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def measure_performance(model, device, input_size=(3, 800, 800)):
    model.eval()
    # Use Half Precision for optimized inference speed if cuda
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model.to(device)
    if device.type == 'cuda':
        model.half()
    
    dummy_input = torch.randn(1, *input_size).to(device)
    if device.type == 'cuda':
        dummy_input = dummy_input.half()
    
    # Warmup
    print("  > Warming up GPU...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # Measure Latency
    print(f"  > Benchmarking with dtype={dtype}...")
    iterations = 50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    # Revert model to float32
    model.float()
    
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000 # in ms
    fps = 1 / (total_time / iterations)
    
    # Measure FLOPs
    macs, params = 0, 0
    if THOP_AVAILABLE:
        try:
            # Profile at float32 for accuracy of parameter count
            dummy_f32 = torch.randn(1, *input_size).to(device)
            macs, params = profile(model, inputs=(dummy_f32,), verbose=False)
        except Exception as e:
            print(f"THOP profiling failed: {e}")
            
    gflops = macs / 1e9
    return avg_latency, fps, gflops, params

def evaluate_metrics(model, dataloader, device, iou_thresh=0.5, num_classes=5):
    model.eval()
    # 1: Broken, 2: Flashover, 3: Good, 4: Insulator
    results = {c: {'preds': [], 'gt_count': 0} for c in range(1, num_classes)}
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Eval IoU={iou_thresh}"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)
                pred_boxes = output['boxes'].to(device)
                pred_scores = output['scores'].to(device)
                pred_labels = output['labels'].to(device)
                
                for label in gt_labels: 
                    if label.item() in results:
                        results[label.item()]['gt_count'] += 1
                
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(gt_boxes, pred_boxes)
                    for i, gt_label in enumerate(gt_labels):
                        max_iou, max_idx = ious[i].max(dim=0)
                        if max_iou > iou_thresh:
                            if pred_labels[max_idx].item() < num_classes:
                                confusion_matrix[gt_label.item(), pred_labels[max_idx].item()] += 1
                        else:
                            confusion_matrix[gt_label.item(), 0] += 1
                    
                    # False Positives check
                    max_iou_per_pred, _ = ious.max(dim=0)
                    for j, pred_label in enumerate(pred_labels):
                        if pred_scores[j] > 0.5:
                            if max_iou_per_pred[j] < iou_thresh:
                                if pred_label.item() < num_classes:
                                    confusion_matrix[0, pred_label.item()] += 1
                elif len(gt_boxes) > 0:
                    for label in gt_labels: 
                        confusion_matrix[label.item(), 0] += 1
                elif len(pred_boxes) > 0:
                    for j, label in enumerate(pred_labels):
                        if pred_scores[j] > 0.5: 
                            if label.item() < num_classes:
                                confusion_matrix[0, label.item()] += 1

                for cls_id in results:
                    mask_pred = pred_labels == cls_id
                    cls_preds = pred_boxes[mask_pred]
                    cls_scores = pred_scores[mask_pred]
                    cls_gts = gt_boxes[gt_labels == cls_id]
                    
                    if len(cls_preds) == 0: continue
                    
                    sorted_idxs = torch.argsort(cls_scores, descending=True)
                    cls_preds = cls_preds[sorted_idxs]
                    cls_scores = cls_scores[sorted_idxs]
                    
                    matched_gt = torch.zeros(len(cls_gts), dtype=torch.bool, device=device)
                    for idx, box in enumerate(cls_preds):
                        iou_max = 0.0
                        match_idx = -1
                        if len(cls_gts) > 0:
                            ious = box_iou(box.unsqueeze(0), cls_gts)[0]
                            iou_max, match_idx = ious.max(dim=0)
                            iou_max = iou_max.item()
                        
                        is_tp = False
                        if iou_max > iou_thresh:
                            if not matched_gt[match_idx]: 
                                is_tp = True
                                matched_gt[match_idx] = True
                        results[cls_id]['preds'].append({'score': cls_scores[idx].item(), 'tp': is_tp})

    final_aps = {}
    mAP = 0
    valid_classes = 0
    for cls_id, data in results.items():
        preds = data['preds']
        gt_count = data['gt_count']
        
        if gt_count == 0: 
            final_aps[cls_id] = 0.0
            continue
            
        preds.sort(key=lambda x: x['score'], reverse=True)
        tps = np.array([p['tp'] for p in preds])
        fps = 1 - tps
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        
        recalls = tp_cumsum / gt_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        ap = compute_ap(recalls, precisions)
        final_aps[cls_id] = ap
        mAP += ap
        valid_classes += 1
        
    avg_mAP = mAP / valid_classes if valid_classes > 0 else 0
    return final_aps, avg_mAP, confusion_matrix

def generate_report():
    # SETTINGS
    CHECKPOINT_PATH = r'checkpoints/best_model.pth'
    DATA_ROOT = r'../Train_IDID_V1.2/Train/Images'
    ANNOTATION_FILE = r'../Train_IDID_V1.2/Train/labels_v1.2.json'
    OUTPUT_REPORT = r'Detailed_Federated_Report.pdf'
    
    NUM_CLASSES = 5
    CLASS_NAMES = {0: 'Background', 1: 'Broken', 2: 'Flashover', 3: 'Good', 4: 'Insulator'}
    # Colors for visualization
    CLASS_COLORS = {1: 'red', 2: 'orange', 3: 'cyan', 4: 'lime'}

    print(f"Generating Report using model: {CHECKPOINT_PATH}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Recreate the validation split
    # Note: ensure get_transform(train=False) is used
    full_dataset = BaselineDataset(DATA_ROOT, ANNOTATION_FILE, transforms=get_transform(train=False), target_size=(800, 800))
    
    # We use the same seed as train.py to ensure we test on the same validation set
    indices = list(range(len(full_dataset.ids)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    val_indices = indices[split:]
    
    dataset = torch.utils.data.Subset(full_dataset, val_indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    print(f"Validation Set Size: {len(dataset)} images")

    # Load Model
    model = get_model(num_classes=NUM_CLASSES)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model.to(device)
    model.eval()
    
    # Benchmarking
    print("Benchmarking performance...")
    latency, fps, gflops, model_params = measure_performance(model, device)
    
    # Metrics
    print("Calculating mAP and Confusion Matrix...")
    aps_50, map_50, cm = evaluate_metrics(model, dataloader, device, 0.5, NUM_CLASSES)
    aps_75, map_75, _ = evaluate_metrics(model, dataloader, device, 0.75, NUM_CLASSES)
    
    # PDF generation
    with PdfPages(OUTPUT_REPORT) as pdf:
        # Page 1: Summary
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.axis('off')
        plt.text(0.5, 0.90, "AI Model Evaluation Report ", ha='center', fontsize=24, weight='bold', color='#2c3e50')
        plt.text(0.5, 0.85, "Insulator Defect Detection System (RetinaNet)", ha='center', fontsize=18, color='#7f8c8d')
        
        details = f"""
        Date:               {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
        Model Architecture: RetinaNet (ResNet50-FPN-V2)
        Evaluation Dataset: IDID V1.2 (Validation Split)
        Total Parameters:   {model_params:,}
        
        Real-World Performance Metrics
        --------------------------------------------------------
        Inference Latency:         {latency:.2f} ms per image
        Throughput (FPS):          {fps:.2f} frames/sec
        Computational Cost:        {gflops:.2f} GFLOPs (approx)
        
        Overall Accuracy Metrics (Validation Set)
        --------------------------------------------------------
        Mean Average Precision (mAP @50):           {map_50:.4f}
        Mean Average Precision (mAP @75):           {map_75:.4f}
        
        Class-wise Breakdown (AP @50)
        --------------------------------------------------------
        """
        for cls_id in range(1, NUM_CLASSES):
            name = CLASS_NAMES.get(cls_id, f'Class {cls_id}')
            details += f"{name:<15}: {aps_50.get(cls_id, 0.0):.4f}\n        "
            
        plt.text(0.1, 0.75, details, fontsize=14, va='top', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [CLASS_NAMES[i] for i in range(1, NUM_CLASSES)]
        scores_50 = [aps_50.get(i, 0.0) for i in range(1, NUM_CLASSES)]
        scores_75 = [aps_75.get(i, 0.0) for i in range(1, NUM_CLASSES)]
        
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, scores_50, width, label='AP @ 0.50', color='#3498db')
        ax.bar(x + width/2, scores_75, width, label='AP @ 0.75', color='#2ecc71')
        ax.set_ylabel('Average Precision')
        ax.set_title('Performance per Class')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Confusion Matrix
        fig = plt.figure(figsize=(10, 8))
        cm_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
        plt.title('Confusion Matrix (Validation)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        pdf.savefig(fig)
        plt.close()
        
        # Page 4: Visual Samples
        print("Generating visual samples...")
        # Select random images from the dataset subset
        # Since dataset is a Subset, we access indices
        sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
        
        for i, idx in enumerate(sample_indices):
            img, target = dataset[idx] # This gets transformed image
            
            # Run inference
            with torch.no_grad():
                out = model([img.to(device)])[0]
            
            # Denormalize/Convert for display
            # img is float tensor [0,1]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img_np)
            ax = plt.gca()
            plt.axis('off')
            
            # Ground Truth (Dashed Green)
            for box, label in zip(target['boxes'], target['labels']):
                x1, y1, x2, y2 = box.cpu().numpy()
                cls_name = CLASS_NAMES.get(label.item(), str(label.item()))
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linestyle='--', linewidth=2))
                # ax.text(x1, y1, f"GT: {cls_name}", color='green', fontsize=8)

            # Predictions (Solid Colors)
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            
            found_pred = False
            for j in range(len(scores)):
                if scores[j] > 0.5:
                    found_pred = True
                    box = boxes[j]
                    label_id = labels[j]
                    color = CLASS_COLORS.get(label_id, 'red')
                    name = CLASS_NAMES.get(label_id, 'Unknown')
                    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=color, linewidth=3))
                    ax.text(box[0], box[1]-10, f'{name} {scores[j]:.2f}', bbox=dict(facecolor=color, alpha=0.5), color='white', weight='bold')
            
            plt.title(f"Validation Sample {idx} (Green=GT, Solid=Pred)")
            pdf.savefig()
            plt.close()

    print(f"Detailed Report generated: {OUTPUT_REPORT}")

if __name__ == "__main__":
    generate_report()
