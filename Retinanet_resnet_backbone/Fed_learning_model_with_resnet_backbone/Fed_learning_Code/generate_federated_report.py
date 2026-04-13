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

# Add Fedlearning path to sys.path
sys.path.append(r'D:\anlog&math\Fedlearning')
from model import get_model
from transforms import get_transform
from dataset import CustomCocoDataset

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def measure_performance(model, device, input_size=(3, 800, 800)):
    model.eval()
    # Use Half Precision for optimized inference speed
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model.to(device).to(dtype)
    
    dummy_input = torch.randn(1, *input_size).to(device).to(dtype)
    
    # Warmup
    print("  > Warming up GPU...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # Measure Latency
    print(f"  > Benchmarking with dtype={dtype}...")
    iterations = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    # Revert model to float32 for metric calculation
    model.to(torch.float32)
    
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
        except:
            pass
            
    gflops = macs / 1e9
    return avg_latency, fps, gflops

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

def evaluate_metrics(model, dataloader, device, iou_thresh=0.5):
    model.eval()
    num_classes = 3 # Background, Defect, Insulator
    class_names = {0: 'Background', 1: 'Defect', 2: 'Insulator'}
    results = {c: {'preds': [], 'gt_count': 0} for c in [1, 2]}
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
                
                for label in gt_labels: results[label.item()]['gt_count'] += 1
                
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(gt_boxes, pred_boxes)
                    for i, gt_label in enumerate(gt_labels):
                        max_iou, max_idx = ious[i].max(dim=0)
                        if max_iou > iou_thresh:
                            confusion_matrix[gt_label.item(), pred_labels[max_idx].item()] += 1
                        else:
                            confusion_matrix[gt_label.item(), 0] += 1
                    max_iou_per_pred, _ = ious.max(dim=0)
                    for j, pred_label in enumerate(pred_labels):
                        if pred_scores[j] > 0.5 and max_iou_per_pred[j] < iou_thresh:
                            confusion_matrix[0, pred_label.item()] += 1
                elif len(gt_boxes) > 0:
                    for label in gt_labels: confusion_matrix[label.item(), 0] += 1
                elif len(pred_boxes) > 0:
                    for j, label in enumerate(pred_labels):
                        if pred_scores[j] > 0.5: confusion_matrix[0, label.item()] += 1

                for cls_id in results:
                    mask_pred = pred_labels == cls_id
                    cls_preds = pred_boxes[mask_pred]
                    cls_scores = pred_scores[mask_pred]
                    cls_gts = gt_boxes[gt_labels == cls_id]
                    if len(cls_preds) == 0: continue
                    sorted_idxs = torch.argsort(cls_scores, descending=True)
                    cls_preds = cls_preds[sorted_idxs]; cls_scores = cls_scores[sorted_idxs]
                    matched_gt = torch.zeros(len(cls_gts), dtype=torch.bool, device=device)
                    for idx, box in enumerate(cls_preds):
                        iou_max = 0.0; match_idx = -1
                        if len(cls_gts) > 0:
                            ious = box_iou(box.unsqueeze(0), cls_gts)[0]
                            iou_max, match_idx = ious.max(dim=0)
                            iou_max = iou_max.item()
                        is_tp = False
                        if iou_max > iou_thresh:
                            if not matched_gt[match_idx]: is_tp = True; matched_gt[match_idx] = True
                        results[cls_id]['preds'].append({'score': cls_scores[idx].item(), 'tp': is_tp})

    final_aps = {}
    mAP = 0
    for cls_id, data in results.items():
        preds = data['preds']; gt_count = data['gt_count']
        if gt_count == 0: final_aps[cls_id] = 0.0; continue
        preds.sort(key=lambda x: x['score'], reverse=True)
        tps = np.array([p['tp'] for p in preds]); fps = 1 - tps
        tp_cumsum = np.cumsum(tps); fp_cumsum = np.cumsum(fps)
        recalls = tp_cumsum / gt_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(recalls, precisions)
        final_aps[cls_id] = ap; mAP += ap
    return final_aps, mAP / len(results), confusion_matrix

def generate_report():
    # SETTINGS
    CHECKPOINT_PATH = r'D:\anlog&math\Fed learning\checkpoints\global_model_r6.pth'
    DATA_PATH = r'D:\anlog&math\CLID\IDD-CPLID.v3-cplid_new.coco'
    OUTPUT_REPORT = r'D:\anlog&math\Fed learning\checkpoints\best_federated_report_r6.pdf'
    NUM_CLASSES = 3
    CLASS_NAMES = {0: 'Background', 1: 'Defect', 2: 'Insulator'}
    CLASS_COLORS = {1: 'red', 2: 'lime'}

    print(f"Generating Report for Federated Model: {CHECKPOINT_PATH}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    val_dir = os.path.join(DATA_PATH, 'valid')
    val_ann = os.path.join(val_dir, '_annotations.coco.json')
    dataset = CustomCocoDataset(val_dir, val_ann, transforms=get_transform(train=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Load Model
    model = get_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device); model.eval()
    
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Benchmarking
    print("Benchmarking performance...")
    latency, fps, gflops = measure_performance(model, device)
    
    # Metrics
    print("Calculating mAP and Confusion Matrix...")
    aps_50, map_50, cm = evaluate_metrics(model, dataloader, device, 0.5)
    aps_75, map_75, _ = evaluate_metrics(model, dataloader, device, 0.75)
    
    # PDF generation
    with PdfPages(OUTPUT_REPORT) as pdf:
        # Page 1: Summary
        fig = plt.figure(figsize=(11.69, 8.27)); plt.axis('off')
        plt.text(0.5, 0.90, "AI Model Evaluation Report (Federated)", ha='center', fontsize=24, weight='bold', color='#2c3e50')
        plt.text(0.5, 0.85, "Insulator Defect Detection System (Federated Learning)", ha='center', fontsize=18, color='#7f8c8d')
        
        details = f"""
        Date:               {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
        Model Architecture: RetinaNet (ResNet50-FPN-V2)
        Evaluation Dataset: IDD-CPLID (Aerial)
        Total Parameters:   {model_params:,}
        
        Real-World Performance Metrics
        --------------------------------------------------------
        Inference Latency:         {latency:.2f} ms per image
        Throughput (FPS):          {fps:.2f} frames/sec
        Computational Cost:        {gflops:.2f} GFLOPs (approx)
        
        Overall Accuracy Metrics (Validation Set)
        --------------------------------------------------------
        Mean Average Precision (mAP@50):           {map_50:.4f}
        Mean Average Precision (mAP@75):           {map_75:.4f}
        
        Class-wise Breakdown (AP@50)
        --------------------------------------------------------
        """
        for cls_id in [1, 2]:
            details += f"{CLASS_NAMES[cls_id]:<15}: {aps_50.get(cls_id, 0.0):.4f}\n        "
        plt.text(0.1, 0.75, details, fontsize=14, va='top', family='monospace')
        pdf.savefig(fig); plt.close()
        
        # Page 2: Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ['Defect', 'Insulator']
        scores_50 = [aps_50.get(1, 0.0), aps_50.get(2, 0.0)]
        scores_75 = [aps_75.get(1, 0.0), aps_75.get(2, 0.0)]
        x = np.arange(len(labels)); width = 0.35
        ax.bar(x - width/2, scores_50, width, label='AP @ 0.50', color='#3498db')
        ax.bar(x + width/2, scores_75, width, label='AP @ 0.75', color='#2ecc71')
        ax.set_ylabel('Average Precision'); ax.set_title('Federated Model Performance per Class')
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)
        pdf.savefig(fig); plt.close()
        
        # Page 3: Confusion Matrix
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values())
        plt.title('Confusion Matrix (Federated Model)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        pdf.savefig(fig); plt.close()
        
        # Page 4: Visual Samples
        print("Generating visual samples...")
        indices = np.random.choice(len(dataset), 5, replace=False)
        for idx in indices:
            img, target = dataset[idx]
            with torch.no_grad():
                out = model([img.to(device)])[0]
            
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img_np)
            ax = plt.gca(); plt.axis('off')
            
            for box in target['boxes']:
                x1, y1, x2, y2 = box.cpu().numpy()
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='lime', linestyle='--', linewidth=2))
            
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            
            for i in range(len(scores)):
                if scores[i] > 0.5:
                    box = boxes[i]; label_id = labels[i]
                    color = CLASS_COLORS.get(label_id, 'red')
                    name = CLASS_NAMES.get(label_id, 'Unknown')
                    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=color, linewidth=3))
                    ax.text(box[0], box[1]-10, f'{name} {scores[i]:.2f}', bbox=dict(facecolor=color, alpha=0.5), color='white', weight='bold')
            
            plt.title(f"Federated Validation Sample {idx}")
            pdf.savefig(); plt.close()

    print(f"Federated Report generated: {OUTPUT_REPORT}")

if __name__ == "__main__":
    generate_report()
