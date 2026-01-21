import os
import json
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as F

class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, augment=False, expansion_factor=1):
        self.root = root
        self.transforms = transforms
        self.augment = augment
        self.expansion_factor = expansion_factor
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Map image_id to filename and other info
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # Map image_id to list of annotations
        self.img_to_anns = {img['id']: [] for img in self.coco_data['images']}
        for ann in self.coco_data['annotations']:
            if ann['image_id'] in self.img_to_anns:
                self.img_to_anns[ann['image_id']].append(ann)
        
        # List of image IDs for indexing
        self.ids = list(self.images.keys())
        
        # Categories mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __getitem__(self, index):
        # Handle virtual expansion
        real_index = index % len(self.ids)
        
        img_id = self.ids[real_index]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        
        # Load Image
        img_path = os.path.join(self.root, file_name)
        img = Image.open(img_path).convert("RGB")
        
        # Get Annotations
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO bbox: [x, y, w, h] -> PyTorch: [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Negative example (no objects)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([img_id])
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        # Apply augmentations
        if self.augment:
            img, target = self.apply_augmentations(img, target)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def apply_augmentations(self, img, target):
        """Apply strong data augmentations for 95%+ goal"""
        boxes = target['boxes']
        
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            img = F.hflip(img)
            if len(boxes) > 0:
                w = img.width
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        
        # 2. Random Vertical Flip (New)
        if random.random() > 0.3:
            img = F.vflip(img)
            if len(boxes) > 0:
                h = img.height
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        # 3. Color Jittering (Expensive: Reduced prob from 0.5 to 0.15)
        if random.random() > 0.85:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)

        # 4. Random 90-degree Rotation (Expensive: Reduced prob)
        # Only rotate 30% of the time instead of always checking
        if random.random() > 0.7:
            rot_k = random.randint(1, 3)
            img = F.rotate(img, angle=rot_k * 90, expand=True)
            w_new, h_new = img.size
            
            if len(boxes) > 0:
                if rot_k == 2: # 180 deg
                    boxes[:, [0, 2]] = w_new - boxes[:, [2, 0]]
                    boxes[:, [1, 3]] = h_new - boxes[:, [3, 1]]
                elif rot_k == 1: # 90 deg CCW
                    old_w = h_new
                    new_boxes = boxes.clone()
                    new_boxes[:, 0] = boxes[:, 1]
                    new_boxes[:, 1] = old_w - boxes[:, 2]
                    new_boxes[:, 2] = boxes[:, 3]
                    new_boxes[:, 3] = old_w - boxes[:, 0]
                    boxes = new_boxes
                elif rot_k == 3: # 270 deg
                    old_h = w_new
                    new_boxes = boxes.clone()
                    new_boxes[:, 0] = old_h - boxes[:, 3]
                    new_boxes[:, 1] = boxes[:, 0]
                    new_boxes[:, 2] = old_h - boxes[:, 1]
                    new_boxes[:, 3] = boxes[:, 2]
                    boxes = new_boxes

        target['boxes'] = boxes
        return img, target

    def __len__(self):
        return len(self.ids) * self.expansion_factor