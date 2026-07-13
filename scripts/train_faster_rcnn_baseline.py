import os
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_HERE, '..', 'Insulator-Defect-Detection-System-RetinaNet-',
                         'Retinanet_resnet_backbone',
                         'Fed_learning_model_with_resnet_backbone',
                         'Fed_learning_Code')
sys.path.insert(0, os.path.normpath(_CODE_DIR))

from dataset import CustomCocoDataset
from transforms import get_transform
from engine import train_one_epoch, evaluate, print_metrics

DATA_PATH  = r'D:\Fed learning project\Dataset - IDD-CPLID.v3-cplid_new.coco'
OUTPUT_DIR = r'D:\Fed learning project\checkpoints\faster_rcnn_baseline'

def get_faster_rcnn_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dir = os.path.join(DATA_PATH, 'train')
    train_ann = os.path.join(train_dir, '_annotations.coco.json')
    val_dir   = os.path.join(DATA_PATH, 'valid')
    val_ann   = os.path.join(val_dir,   '_annotations.coco.json')

    dataset = CustomCocoDataset(train_dir, train_ann, transforms=get_transform(train=True))
    dataset_test = CustomCocoDataset(val_dir, val_ann, transforms=get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn)

    model = get_faster_rcnn_model(num_classes=3)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    best_map50 = 0.0

    class_names = {0: 'background', 1: 'defect', 2: 'insulator'}

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        
        # evaluate
        res = evaluate(model, data_loader_test, device=device)
        print_metrics(res, class_names)
        
        if res['map_50'] > best_map50:
            best_map50 = res['map_50']
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_faster_rcnn.pth'))
            print(f"  *** New Best mAP@50: {best_map50:.4f} – saved ***")

if __name__ == '__main__':
    main()
