import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import itertools
from PIL import Image
from dataset import CustomCocoDataset
from transforms import get_transform
from model import get_model
from engine import evaluate, print_metrics
import torchvision.transforms.functional as F

def collate_fn(batch):
    return tuple(zip(*batch))

class Config:
    # Paths
    DATA_PATH = r'D:\CLID\IDD-CPLID.v3-cplid_new.coco'
    CHECKPOINT_PATH = r'D:\CLID\checkpoints\best_model.pth'
    REPORT_DIR = r'D:\CLID\checkpoints'
    SAMPLE_IMAGE_PATH = r'D:\CLID\WhatsApp Image 2025-12-18 at 11.58.37.jpeg'
    
    # Model
    NUM_CLASSES = 3
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    
    # Evaluation
    SCORE_THRESHOLD = 0.5 
    IOU_THRESHOLDS = [0.5, 0.75]
    
    CLASS_NAMES = {0: 'Background', 1: 'Defect', 2: 'Insulator'}
    CLASS_COLORS = {1: 'red', 2: 'lime'}

# --------------------------
# Plotting & Inference Functions
# --------------------------

def plot_confusion_matrix(cm, class_names, output_dir, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Generates and saves a confusion matrix heatmap"""
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14, weight='bold')

    plt.tight_layout()
    plt.ylabel('True Label (Ground Truth)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    return fig

def plot_metrics_chart(aps, class_names, output_dir):
    """Generates and saves a bar chart for AP scores"""
    labels = []
    ap50_scores = []
    ap75_scores = []
    
    for cls_id in [1, 2]: # Defect, Insulator
        if cls_id in aps:
            labels.append(class_names.get(cls_id, str(cls_id)))
            ap50_scores.append(aps[cls_id].get(0.5, 0.0))
            ap75_scores.append(aps[cls_id].get(0.75, 0.0))
        else:
            labels.append(class_names.get(cls_id, str(cls_id)))
            ap50_scores.append(0.0)
            ap75_scores.append(0.0)
            
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, ap50_scores, width, label='AP @ 0.50', color='#3498db')
    rects2 = ax.bar(x + width/2, ap75_scores, width, label='AP @ 0.75', color='#2ecc71')
    
    ax.set_ylabel('Average Precision', fontsize=14)
    ax.set_title('Model Performance by Class', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_chart.png'), dpi=300)
    return fig

def run_inference_on_image(model, image_path, device, config):
    """Runs prediction on a single image and plots the result"""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # NMS is internal to RetinaNet, but we filter by score
    keep = scores >= config.SCORE_THRESHOLD
    count = 0
    
    for i in range(len(boxes)):
        if keep[i]:
            count += 1
            box = boxes[i]
            label_id = labels[i]
            score = scores[i]
            
            color = config.CLASS_COLORS.get(label_id, 'yellow')
            name = config.CLASS_NAMES.get(label_id, str(label_id))
            
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 fill=False, edgecolor=color, linewidth=3)
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 10, f'{name} {score:.2f}', 
                    bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white', fontweight='bold')
    
    ax.axis('off')
    plt.title(f"Inference Result: {os.path.basename(image_path)} ({count} detections)", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(config.REPORT_DIR, 'sample_inference.png')
    plt.savefig(save_path, dpi=300)
    print(f"Sample inference saved to {save_path}", flush=True)
    return fig

def create_pdf_report(results, config, model_params, output_path, inference_fig):
    print(f"\nGenerating PDF report: {output_path}...", flush=True)
    
    cm = results['conf_matrix']
    aps = results['aps']
    map50 = results['map_50']
    map75 = results['map_75']
    cm_class_names = [config.CLASS_NAMES[i] for i in range(3)]
    
    with PdfPages(output_path) as pdf:
        # Page 1: Executive Summary
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.axis('off')
        plt.text(0.5, 0.90, "Insulator Defect Detection: AI Report", ha='center', fontsize=24, weight='bold')
        
        report_text = f"""
        Model Architecture: RetinaNet (ResNet50-FPN-V2)
        Parameters:         {model_params:,}
        
        Overall Accuracy Metrics (Validation Set)
        --------------------------------------------------------
        System Accuracy (Recall-based):            {results['accuracy']:.2f}%
        Mean Average Precision (mAP@50):           {map50:.4f}
        Mean Average Precision (mAP@75):           {map75:.4f}
        
        Target Accuracy Status: {'[ ACHIEVED ]' if map50 >= 0.95 or results['accuracy'] >= 95 else '[ IN PROGRESS ]'}
        """
        plt.text(0.1, 0.75, report_text, fontsize=14, va='top', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Sample Inference Result
        if inference_fig:
            pdf.savefig(inference_fig)
            plt.close(inference_fig)
        
        # Page 3: Class Performance Chart
        fig_charts = plot_metrics_chart(aps, config.CLASS_NAMES, config.REPORT_DIR)
        pdf.savefig(fig_charts)
        plt.close()
        
        # Page 4: Confusion Matrix
        fig_cm = plot_confusion_matrix(cm, cm_class_names, config.REPORT_DIR)
        pdf.savefig(fig_cm)
        plt.close()
        
    print(f"Report fully generated at {output_path}", flush=True)

# --------------------------
# Main Execution
# --------------------------

def main():
    config = Config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}", flush=True)

    # 1. Load Model
    print("Loading Model and Weights...", flush=True)
    model = get_model(config.NUM_CLASSES)
    model.to(device)
    
    if os.path.exists(config.CHECKPOINT_PATH):
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: {config.CHECKPOINT_PATH} not found.")
        return

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Run Sample Inference on the requested image
    inference_fig = None
    if os.path.exists(config.SAMPLE_IMAGE_PATH):
        print(f"Running inference on sample image: {config.SAMPLE_IMAGE_PATH}", flush=True)
        inference_fig = run_inference_on_image(model, config.SAMPLE_IMAGE_PATH, device, config)
    else:
        print(f"Warning: Sample image not found at {config.SAMPLE_IMAGE_PATH}")

    # 3. Load Validation Dataset for metrics
    print("Loading Validation set for final metrics...", flush=True)
    val_dir = os.path.join(config.DATA_PATH, 'valid')
    val_ann = os.path.join(val_dir, '_annotations.coco.json')
    dataset_val = CustomCocoDataset(val_dir, val_ann, transforms=get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config.BATCH_SIZE, 
                                             shuffle=False, num_workers=config.NUM_WORKERS, 
                                             collate_fn=collate_fn)

    # 4. Run Full Evaluation
    val_results = evaluate(model, val_loader, device, score_threshold=0.05)
    print_metrics(val_results, config.CLASS_NAMES)
    
    # 5. Generate PDF Report
    pdf_path = os.path.join(config.REPORT_DIR, 'professional_evaluation_report.pdf')
    create_pdf_report(val_results, config, model_params, pdf_path, inference_fig)

if __name__ == "__main__":
    main()
