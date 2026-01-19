import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetHead

def get_model(num_classes):
    # Load pre-trained weights for ResNet50 FPN V2
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights)
    
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Create new head for 5 classes (Background + Broken, Flashed, Good, Insulator)
    model.head = RetinaNetHead(
        in_channels,
        num_anchors,
        num_classes,
        norm_layer=torch.nn.BatchNorm2d
    )
    
    return model
