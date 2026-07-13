import os
import sys
import json
import shutil
import yaml
from ultralytics import YOLO

DATA_PATH = r'D:\Fed learning project\Dataset - IDD-CPLID.v3-cplid_new.coco'
YOLO_DIR = r'D:\Fed learning project\yolo_dataset'
OUTPUT_DIR = r'D:\Fed learning project\checkpoints\yolov8_baseline'

def coco_to_yolo(coco_ann_path, images_dir, output_images_dir, output_labels_dir):
    with open(coco_ann_path, 'r') as f:
        coco = json.load(f)

    # create a map of image_id to image info
    images_info = {img['id']: img for img in coco['images']}
    
    # create category mapping, ensuring defect is 0 and insulator is 1 if needed
    # (Assuming 1: defect, 2: insulator from previous setup)
    # We'll map them to 0-indexed for YOLO
    cat_mapping = {}
    for i, cat in enumerate(coco['categories']):
        cat_mapping[cat['id']] = i
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # copy images
    for img in coco['images']:
        src = os.path.join(images_dir, img['file_name'])
        dst = os.path.join(output_images_dir, img['file_name'])
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)

    # write labels
    labels = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in labels:
            labels[img_id] = []
            
        img_w = images_info[img_id]['width']
        img_h = images_info[img_id]['height']
        
        cat_id = cat_mapping[ann['category_id']]
        
        x_min, y_min, w, h = ann['bbox']
        
        # YOLO format
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        labels[img_id].append(f"{cat_id} {x_center} {y_center} {w_norm} {h_norm}")

    for img_id, label_list in labels.items():
        file_name = images_info[img_id]['file_name']
        label_file = os.path.splitext(file_name)[0] + '.txt'
        with open(os.path.join(output_labels_dir, label_file), 'w') as f:
            f.write('\n'.join(label_list))

def prepare_yolo_dataset():
    print("Preparing YOLO dataset...")
    for split in ['train', 'valid']:
        coco_ann = os.path.join(DATA_PATH, split, '_annotations.coco.json')
        images_dir = os.path.join(DATA_PATH, split)
        out_images = os.path.join(YOLO_DIR, 'images', split)
        out_labels = os.path.join(YOLO_DIR, 'labels', split)
        
        if os.path.exists(coco_ann):
            coco_to_yolo(coco_ann, images_dir, out_images, out_labels)
    
    # create dataset.yaml
    with open(os.path.join(DATA_PATH, 'train', '_annotations.coco.json'), 'r') as f:
        coco = json.load(f)
        names = [cat['name'] for cat in coco['categories']]

    dataset_yaml = {
        'path': YOLO_DIR,
        'train': 'images/train',
        'val': 'images/valid',
        'nc': len(names),
        'names': names
    }
    
    yaml_path = os.path.join(YOLO_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
        
    return yaml_path

def main():
    yaml_path = prepare_yolo_dataset()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting YOLOv8 training...")
    model = YOLO('yolov8n.pt')  # load a pretrained model
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        project=OUTPUT_DIR,
        name='yolov8_run',
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Training completed. Results saved in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
