import math
import sys
import torch
import torchvision
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Standard training loop for one epoch with Mixed Precision"""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}', file=sys.stdout)
    for images, targets in pbar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
    
    avg_loss = total_loss / len(data_loader)
    print(f"    Epoch {epoch} | Average Training Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates model performance on validation set.
    Calculates mAP@50 and Overall Accuracy.
    """
    model.eval()
    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(lambda: defaultdict(list))
    
    num_classes = 5 # BG + 4
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    img_counter = 0
    pbar = tqdm(data_loader, desc='Validating', file=sys.stdout)
    
    for images, targets in pbar:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)
        
        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            
            # Tracking GT for mAP
            for box, label in zip(gt_boxes, gt_labels):
                all_ground_truths[label.item()][img_counter].append(box)
            
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            
            # Confusion Matrix / Accuracy calculation (Threshold = 0.5)
            cm_keep = pred_scores > 0.5
            cm_boxes = pred_boxes[cm_keep]
            cm_labels = pred_labels[cm_keep]
            
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            if len(cm_boxes) > 0 and len(gt_boxes) > 0:
                ious_matrix = torchvision.ops.box_iou(torch.from_numpy(cm_boxes), torch.from_numpy(gt_boxes)).numpy()
                for i, pl in enumerate(cm_labels):
                    iou_max = ious_matrix[i].max()
                    match_idx = ious_matrix[i].argmax()
                
                    if iou_max >= 0.5:
                        if not gt_matched[match_idx]:
                            conf_matrix[gt_labels[match_idx], pl] += 1
                            gt_matched[match_idx] = True
                        else: conf_matrix[0, pl] += 1 # False Positive (Double detection)
                    else: conf_matrix[0, pl] += 1 # False Positive
            else:
                for pl in cm_labels:
                    conf_matrix[0, pl] += 1
            
            for i, matched in enumerate(gt_matched):
                if not matched: conf_matrix[gt_labels[i], 0] += 1 # False Negative
            
            # Collecting detections for mAP
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score > 0.05:
                    all_detections[label.item()].append({
                        'score': score.item(),
                        'image_id': img_counter,
                        'bbox': box
                    })
            img_counter += 1
            
    # mAP@50 calculation
    aps = []
    for cls_id in range(1, num_classes):
        dects = all_detections[cls_id]
        dects.sort(key=lambda x: x['score'], reverse=True)
        n_gt = sum(len(boxes) for boxes in all_ground_truths[cls_id].values())
        if n_gt == 0: continue
        
        tp = np.zeros(len(dects))
        fp = np.zeros(len(dects))
        gt_matched_map = defaultdict(set)
        
        for i, det in enumerate(dects):
            img_id = det['image_id']
            gt_boxes_img = all_ground_truths[cls_id][img_id]
            best_iou = 0
            best_gt_idx = -1
            
            if len(gt_boxes_img) > 0:
                ious = torchvision.ops.box_iou(torch.as_tensor(det['bbox']).unsqueeze(0), torch.as_tensor(gt_boxes_img))[0]
                best_iou = ious.max().item()
                best_gt_idx = ious.argmax().item()
            
            if best_iou >= 0.5:
                if best_gt_idx not in gt_matched_map[img_id]:
                    tp[i] = 1
                    gt_matched_map[img_id].add(best_gt_idx)
                else: fp[i] = 1
            else: fp[i] = 1
            
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / n_gt
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        
        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0: p = 0
            else: p = np.max(prec[rec >= t])
            ap += p / 11
        aps.append(ap)
        
    mAP = np.mean(aps) if aps else 0.0
    
    total_gt = sum(sum(len(boxes) for boxes in class_gt.values()) for class_gt in all_ground_truths.values())
    total_tp = np.sum(np.diag(conf_matrix)[1:]) 
    accuracy = (total_tp / total_gt * 100) if total_gt > 0 else 0.0

    print("-" * 30)
    print(f"Validation mAP@50: {mAP:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("-" * 30)
    
    return mAP