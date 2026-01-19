import os
import json
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as F

class IDIDDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, augment=False, expansion_factor=1, target_size=(800, 800)):
        self.root = root
        self.transforms = transforms
        self.augment = augment
        self.expansion_factor = expansion_factor
        self.target_size = target_size
        
        with open(annotation_file, 'r') as f:
            self.raw_data = json.load(f)
            
        self.images = []
        self.img_to_anns = {}
        
        for idx, entry in enumerate(self.raw_data):
            img_id = idx
            self.images.append({'id': img_id, 'file_name': entry['filename']})
            self.img_to_anns[img_id] = []
            
            if 'Labels' in entry and 'objects' in entry['Labels']:
                for obj in entry['Labels']['objects']:
                    # IEEE Competition Class IDs:
                    # 0: broken, 1: flashed, 2: good, 3: insulator
                    
                    label = -1
                    
                    if obj.get('string') == 1:
                        label = 3 # insulator (string)
                    else:
                        # Check conditions for disks
                        is_broken = False
                        is_flashed = False
                        
                        if 'conditions' in obj:
                            conds = obj['conditions']
                            for k, v in conds.items():
                                val_lower = v.lower()
                                if 'broken' in val_lower:
                                    is_broken = True
                                if 'flashover' in val_lower:
                                    is_flashed = True
                        
                        if is_broken:
                            label = 0 # broken
                        elif is_flashed:
                            label = 1 # flashed
                        else:
                            label = 2 # good
                    
                    # PyTorch needs 1-based indexing for objects (0 is background)
                    # So: 0->1, 1->2, 2->3, 3->4
                    if label != -1:
                        obj_copy = obj.copy()
                        obj_copy['custom_label'] = label + 1 
                        self.img_to_anns[img_id].append(obj_copy)

        self.ids = list(range(len(self.images)))
        
        # Internal mapping
        self.classes = {
            0: 'background',
            1: 'broken',
            2: 'flashed',
            3: 'good',
            4: 'insulator'
        }

    def __getitem__(self, index):
        real_index = index % len(self.ids)
        img_id = self.ids[real_index]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Handle missing images gracefully if needed, or error out
            # Creating a dummy image to avoid crashing dataloader
            img = Image.new('RGB', (800, 800))

        original_w, original_h = img.size
        
        img = F.resize(img, self.target_size)
        new_w, new_h = img.size
        scale_w = new_w / original_w
        scale_h = new_h / original_h
        
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = max(0, x * scale_w), max(0, y * scale_h)
            x2, y2 = min(new_w, (x + w) * scale_w), min(new_h, (y + h) * scale_h)
            
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['custom_label'])
                areas.append((x2 - x1) * (y2 - y1))
                iscrowd.append(0)
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,))
        }

        if self.augment:
            img, target = self.apply_augmentations(img, target)
        
        img = F.to_tensor(img)
        return img, target

    def apply_augmentations(self, img, target):
        boxes = target['boxes']
        if random.random() > 0.5:
            img = F.hflip(img)
            if len(boxes) > 0:
                w = img.width
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if random.random() > 0.3:
            img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
        target['boxes'] = boxes
        return img, target

    def __len__(self):
        return len(self.ids) * self.expansion_factor
