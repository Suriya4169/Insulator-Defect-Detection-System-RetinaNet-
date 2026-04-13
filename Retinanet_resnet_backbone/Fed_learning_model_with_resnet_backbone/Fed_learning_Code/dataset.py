import os
import json
import torch
import torch.utils.data
from PIL import Image, ImageFilter
import random
import torchvision.transforms.functional as F

class HeavyAugmentation:
    def __call__(self, image, boxes, labels):
        boxes  = boxes.clone().float()
        labels = labels.clone()

        if random.random() < 0.5:  image, boxes         = self._hflip(image, boxes)
        if random.random() < 0.3:  image, boxes         = self._vflip(image, boxes)
        if random.random() < 0.3:  image, boxes         = self._rot90(image, boxes)
        if random.random() < 0.6:  image, boxes, labels = self._scale_crop(image, boxes, labels)
        if random.random() < 0.7:  image                = self._color_jitter(image)
        if random.random() < 0.25: image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.1:  image = image.convert('L').convert('RGB')
        if random.random() < 0.5:
            from PIL import ImageEnhance
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 1.4))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.4:  image = self._cutout(image)
        if random.random() < 0.3:
            from PIL import ImageEnhance
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 2.0))

        return image, boxes, labels

    def _hflip(self, image, boxes):
        w, _ = image.size
        image = F.hflip(image)
        if len(boxes): boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes

    def _vflip(self, image, boxes):
        _, h = image.size
        image = F.vflip(image)
        if len(boxes): boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return image, boxes

    def _rot90(self, image, boxes):
        w, h = image.size
        # The rotation logic expects PIL Image
        image = image.rotate(-90, expand=True)
        if len(boxes):
            x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
            boxes = torch.stack([h-y2, x1, h-y1, x2], dim=1)
        return image, boxes

    def _scale_crop(self, image, boxes, labels):
        def clip_boxes(boxes, w, h):
            boxes[:, 0].clamp_(0, w); boxes[:, 2].clamp_(0, w)
            boxes[:, 1].clamp_(0, h); boxes[:, 3].clamp_(0, h)
            keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
            return boxes[keep], keep
            
        w, h   = image.size
        scale  = random.uniform(0.7, 1.3)
        nw, nh = int(w*scale), int(h*scale)
        image  = image.resize((nw, nh), Image.BILINEAR)
        if len(boxes):
            boxes[:, [0,2]] *= scale
            boxes[:, [1,3]] *= scale
        if nw > w or nh > h:
            cx = random.randint(0, max(0, nw-w))
            cy = random.randint(0, max(0, nh-h))
            image = image.crop((cx, cy, cx+w, cy+h))
            if len(boxes):
                boxes[:,[0,2]] -= cx; boxes[:,[1,3]] -= cy
                boxes, keep = clip_boxes(boxes, w, h)
                labels = labels[keep]
        else:
            pad_w, pad_h = (w-nw)//2, (h-nh)//2
            canvas = Image.new('RGB', (w, h), (0,0,0))
            canvas.paste(image, (pad_w, pad_h)); image = canvas
            if len(boxes):
                boxes[:,[0,2]] += pad_w; boxes[:,[1,3]] += pad_h
        return image, boxes, labels

    def _color_jitter(self, image):
        image = F.adjust_brightness(image, random.uniform(0.7, 1.3))
        image = F.adjust_contrast(image,   random.uniform(0.8, 1.2))
        image = F.adjust_saturation(image, random.uniform(0.7, 1.3))
        image = F.adjust_hue(image,        random.uniform(-0.08, 0.08))
        return image

    def _cutout(self, image):
        w, h = image.size
        import numpy as np
        arr  = np.array(image)
        for _ in range(random.randint(1, 3)):
            cw = random.randint(max(1, w//10), max(2, w//4))
            ch = random.randint(max(1, h//10), max(2, h//4))
            cx = random.randint(0, max(0, w-cw))
            cy = random.randint(0, max(0, h-ch))
            arr[cy:cy+ch, cx:cx+cw] = 0
        return Image.fromarray(arr)

def build_defect_bank(root, ann_file, max_crops=300, defect_id=1):
    print(f"  Building defect crop bank (max={max_crops})...")
    with open(ann_file) as f:
        data = json.load(f)
    images_map  = {img['id']: img for img in data['images']}
    defect_anns = [a for a in data['annotations'] if a['category_id'] == defect_id]
    random.shuffle(defect_anns)
    bank = []
    for ann in defect_anns:
        if len(bank) >= max_crops: break
        img_info = images_map[ann['image_id']]
        try:
            img = Image.open(os.path.join(root, img_info['file_name'])).convert('RGB')
            x, y, w, h = [int(v) for v in ann['bbox']]
            if w < 4 or h < 4: continue
            bank.append(img.crop((x, y, x+w, y+h)))
        except Exception:
            continue
    print(f"  ✓ Defect bank: {len(bank)} crops")
    return bank

def copy_paste_defects(image, boxes, labels, defect_bank, defect_id=1, n_paste=(1, 3)):
    if not defect_bank: return image, boxes, labels
    import numpy as np
    w, h    = image.size
    img_arr = np.array(image).copy()
    new_boxes, new_labels = [], []
    for _ in range(random.randint(*n_paste)):
        crop   = random.choice(defect_bank)
        cw, ch = crop.size
        scale  = random.uniform(0.05, 0.25) * max(w, h) / max(max(cw, ch), 1)
        scale  = max(0.3, min(scale, 3.0))
        new_cw = max(4, int(cw * scale))
        new_ch = max(4, int(ch * scale))
        if new_cw >= w or new_ch >= h: continue
        crop_r = crop.resize((new_cw, new_ch), Image.BILINEAR)
        px = random.randint(0, w - new_cw)
        py = random.randint(0, h - new_ch)
        img_arr[py:py+new_ch, px:px+new_cw] = np.array(crop_r)
        new_boxes.append([px, py, px+new_cw, py+new_ch])
        new_labels.append(defect_id)
    image = Image.fromarray(img_arr)
    if new_boxes:
        nb = torch.tensor(new_boxes,  dtype=torch.float32)
        nl = torch.tensor(new_labels, dtype=torch.int64)
        boxes  = torch.cat([boxes,  nb], dim=0) if len(boxes) else nb
        labels = torch.cat([labels, nl], dim=0) if len(labels) else nl
    return image, boxes, labels

def make_weighted_sampler(dataset, defect_id=1, defect_weight=4.0):
    weights = []
    for img_id in dataset.ids:
        anns   = dataset.img_to_anns.get(img_id, [])
        labels = {a['category_id'] for a in anns}
        weights.append(defect_weight if defect_id in labels else 1.0)
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True)

class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None, augment=False, defect_bank=None, defect_prob=0.35, subset_ids=None):
        self.root = root
        self.transforms = transforms
        self.augment = augment
        self.aug = HeavyAugmentation() if augment else None
        self.defect_bank = defect_bank
        self.defect_prob = defect_prob
        
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
        all_ids = list(self.images.keys())
        self.ids = subset_ids if subset_ids is not None else all_ids
        
        # Categories mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __getitem__(self, index):
        img_id = self.ids[index]
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
            if self.defect_bank and random.random() < self.defect_prob:
                img, boxes, labels = copy_paste_defects(img, boxes, labels, self.defect_bank)
            if len(boxes) > 0:
                img, boxes, labels = self.aug(img, boxes, labels)
            
            target['boxes'] = boxes
            target['labels'] = labels
            if len(boxes) > 0:
                target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            else:
                target['area'] = torch.zeros((0,), dtype=torch.float32)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)