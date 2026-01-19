import os
import json
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as F

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, target_size=(800, 800)):
        self.root = root
        self.transforms = transforms
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
                    label = -1
                    if obj.get('string') == 1:
                        label = 3 # insulator string
                    else:
                        is_broken = False
                        is_flashed = False
                        if 'conditions' in obj:
                            conds = obj['conditions']
                            for k, v in conds.items():
                                val_lower = v.lower()
                                if 'broken' in val_lower: is_broken = True
                                if 'flashover' in val_lower: is_flashed = True
                        
                        if is_broken: label = 0
                        elif is_flashed: label = 1
                        else: label = 2
                    
                    if label != -1:
                        obj_copy = obj.copy()
                        obj_copy['custom_label'] = label + 1 
                        self.img_to_anns[img_id].append(obj_copy)

        self.ids = list(range(len(self.images)))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new('RGB', self.target_size)

        original_w, original_h = img.size
        img = F.resize(img, self.target_size)
        new_w, new_h = img.size
        scale_w = new_w / original_w
        scale_h = new_h / original_h
        
        anns = self.img_to_anns.get(img_id, [])
        boxes, labels = [], []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = max(0, x * scale_w), max(0, y * scale_h)
            x2, y2 = min(new_w, (x + w) * scale_w), min(new_h, (y + h) * scale_h)
            
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['custom_label'])
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)
