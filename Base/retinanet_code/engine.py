import math
import sys
import torch
import torchvision
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, grad_clip=None):
    """Train for one epoch with mixed precision support"""
    model.train()
    
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    
    torch.backends.cudnn.benchmark = True
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}', file=sys.stdout)
    
    for i, (images, targets) in enumerate(pbar):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += losses.item()
        loss_classifier += loss_dict.get('classification', torch.tensor(0)).item()
        loss_box_reg += loss_dict.get('bbox_regression', torch.tensor(0)).item()
        
        if i % 10 == 0:
            pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'cls': f'{loss_dict.get("classification", 0):.4f}',
                'box': f'{loss_dict.get("bbox_regression", 0):.4f}'
            })
    
    n = len(data_loader)
    return {
        'total_loss': total_loss / n,
        'cls_loss': loss_classifier / n,
        'box_loss': loss_box_reg / n
    }

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def calculate_ap(recall, precision):
    """Calculate Average Precision using 11-point interpolation or all-point"""
    # Append sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

@torch.no_grad()
def evaluate(model, data_loader, device, score_threshold=0.05, iou_thresholds=None):
    """
    Evaluate model and compute COCO-style metrics (AP50, AP75, mAP) 
    and Confusion Matrix.
    Note: score_threshold is lower (0.05) for mAP calculation as per COCO standard.
    """
    model.eval()
    
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    # Store all predictions and ground truths for AP calculation
    # predictions[class_id] = list of (score, iou_with_gt, is_tp_at_05, is_tp_at_75)
    # But for standard AP we need to match globally. 
    # Simplified approach: per image accumulation.
    
    # Data structures for AP calculation
    # detections[class_id] -> list of [score, image_id, bbox]
    # ground_truths[class_id][image_id] -> list of bboxes
    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(lambda: defaultdict(list))
    
    # For Confusion Matrix (at score=0.5, iou=0.5)
    num_classes = model.head.classification_head.num_classes
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    pbar = tqdm(data_loader, desc='Evaluating', file=sys.stdout)
    
    img_counter = 0
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            
            # Populate Ground Truths
            for box, label in zip(gt_boxes, gt_labels):
                all_ground_truths[label.item()][img_counter].append(box.cpu().numpy())
            
            # Predictions
            pred_boxes = output['boxes']
            pred_labels = output['labels']
            pred_scores = output['scores']
            
            # Confusion Matrix Logic (Strict 0.5 threshold)
            cm_keep = pred_scores > 0.5
            cm_boxes = pred_boxes[cm_keep]
            cm_labels = pred_labels[cm_keep]
            
            # Match for Confusion Matrix
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            
            for pb, pl in zip(cm_boxes, cm_labels):
                iou_max = 0.0
                match_idx = -1
                
                # Check IoU with ALL GT boxes (agnostic to class for background confusion)
                if len(gt_boxes) > 0:
                    ious = torchvision.ops.box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
                    iou_max, match_idx = torch.max(ious, dim=0)
                    iou_max = iou_max.item()
                    match_idx = match_idx.item()
                
                if iou_max >= 0.5:
                    gt_label = gt_labels[match_idx].item()
                    pred_lbl = pl.item()
                    if not gt_matched[match_idx]:
                        conf_matrix[gt_label, pred_lbl] += 1
                        gt_matched[match_idx] = True
                    else:
                        # Duplicate detection (FP)
                        conf_matrix[0, pred_lbl] += 1 # 0 is background row for Pred
                else:
                    # Background (FP)
                    conf_matrix[0, pl.item()] += 1
            
            # Missed GTs (FN)
            for i, gl in enumerate(gt_labels):
                if not gt_matched[i]:
                    conf_matrix[gl.item(), 0] += 1 # 0 is background col for GT
            
            # AP Calculation Logic (All detections > score_threshold)
            ap_keep = pred_scores > score_threshold
            for box, label, score in zip(pred_boxes[ap_keep], pred_labels[ap_keep], pred_scores[ap_keep]):
                all_detections[label.item()].append({
                    'score': score.item(),
                    'image_id': img_counter,
                    'bbox': box.cpu().numpy()
                })
            
            img_counter += 1
            
    # Calculate APs
    aps = defaultdict(dict)
    
    for cls_id in all_detections.keys():
        dects = all_detections[cls_id]
        # Sort by score desc
        dects.sort(key=lambda x: x['score'], reverse=True)
        
        n_gt = sum(len(boxes) for boxes in all_ground_truths[cls_id].values())
        if n_gt == 0:
            continue
            
        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(dects))
            fp = np.zeros(len(dects))
            
            # Track matched GT per image to avoid double counting
            gt_matched_map = defaultdict(set)
            
            for i, det in enumerate(dects):
                img_id = det['image_id']
                pred_box = det['bbox']
                
                gt_boxes_img = all_ground_truths[cls_id][img_id]
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes_img):
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh:
                    if best_gt_idx not in gt_matched_map[img_id]:
                        tp[i] = 1.0
                        gt_matched_map[img_id].add(best_gt_idx)
                    else:
                        fp[i] = 1.0 # Duplicate
                else:
                    fp[i] = 1.0
            
            # Compute cumulative
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            
            recalls = tp_cum / n_gt
            precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
            
            aps[cls_id][iou_thresh] = calculate_ap(recalls, precisions)

    # Average AP
    map_50 = np.mean([aps[c].get(0.5, 0.0) for c in aps]) if aps else 0.0
    map_75 = np.mean([aps[c].get(0.75, 0.0) for c in aps]) if aps else 0.0
    
    # Calculate simple accuracy: (TP) / (Total GT Objects)
    total_gt = sum(sum(len(boxes) for boxes in class_gt.values()) for class_id, class_gt in all_ground_truths.items())
    total_tp = np.sum(np.diag(conf_matrix)[1:]) # Sum of diagonal excluding background
    accuracy = (total_tp / total_gt * 100) if total_gt > 0 else 0.0

    return {
        'conf_matrix': conf_matrix,
        'map_50': map_50,
        'map_75': map_75,
        'aps': aps,
        'accuracy': accuracy
    }

def print_metrics(results, class_names):
    """Pretty print extended evaluation metrics"""
    cm = results['conf_matrix']
    aps = results['aps']
    
    print("\n" + "="*80, flush=True)
    print("DETECTION METRICS", flush=True)
    print("="*80, flush=True)
    
    print(f"{'Class':<15} {'AP@50':>10} {'AP@75':>10}", flush=True)
    print("-" * 40, flush=True)
    
    for cls_id, ap_dict in aps.items():
        name = class_names.get(cls_id, str(cls_id))
        print(f"{name:<15} {ap_dict.get(0.5, 0.0):>10.4f} {ap_dict.get(0.75, 0.0):>10.4f}", flush=True)
    
    print("-" * 40, flush=True)
    print(f"{'mAP':<15} {results['map_50']:>10.4f} {results['map_75']:>10.4f}", flush=True)
    print(f"{'Accuracy':<15} {results['accuracy']:>10.2f}%", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("CONFUSION MATRIX (IoU=0.5, Conf=0.5)", flush=True)
    print(f"{'Rows: Actual, Cols: Predicted'}", flush=True)
    print("-" * 80, flush=True)
    
    # Headers
    headers = ["Backg"] + [class_names.get(i, str(i)) for i in range(1, cm.shape[0])]
    print(f"{'':<10}", end="")
    for h in headers:
        print(f"{h:>10}", end="")
    print(f"\n{'':<10}" + "-"*(10*len(headers)), flush=True)
    
    # Rows
    for i, row_name in enumerate(headers):
        print(f"{row_name:<10}", end="")
        for val in cm[i]:
            print(f"{val:>10d}", end="")
        print("", flush=True)
    
    print("="*80 + "\n", flush=True)