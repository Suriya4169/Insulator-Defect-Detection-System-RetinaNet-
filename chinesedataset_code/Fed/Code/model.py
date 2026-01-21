import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetHead

def get_model(num_classes):
    # Load pre-trained weights for ResNet50 FPN V2
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights)
    
    # Replace the classification head
    # The backbone output channels are usually 256 for FPN
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Create new head
    model.head = RetinaNetHead(
        in_channels,
        num_anchors,
        num_classes,
        norm_layer=torch.nn.BatchNorm2d
    )
    
    # Custom Focal Loss Alpha for Class Balancing (Insulator Focus)
    # We want to increase accuracy for Insulator (Class 2).
    # Defect (Class 1) gets default 0.25.
    # Insulator (Class 2) gets boosted 0.75 to increase Recall.
    # Background (Class 0) gets default 0.25.
    # Logic: [Alpha_Back, Alpha_Defect, Alpha_Insulator]
    alpha_weights = torch.tensor([0.25, 0.25, 0.75])
    
    # Register as buffer so it moves to device automatically with model.to(device)
    # This overrides the default scalar 'focal_loss_alpha'
    model.register_buffer('focal_loss_alpha', alpha_weights)
    
    return model
