"""
model_mobilenet.py
==================
MobileNetV2-FPN RetinaNet backbone for the Federated Learning research.

This file is the MobileNet equivalent of the ResNet model.py.
It is used in the FedProx comparison to benchmark a lightweight backbone
suitable for edge-device deployment (e.g., drones, inspection robots).
"""

import torch
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def get_model(num_classes=3):
    """
    Constructs a RetinaNet model with a MobileNetV2 backbone + FPN.

    This is the lightweight (edge-device) version of the architecture.
    MobileNetV2 has ~3.4M parameters vs ~32M for ResNet-50, making it
    ideal for deployment on inspection drones and edge IoT devices.

    Args:
        num_classes (int): Number of detection classes (default: 3).
                           Class 0 = Background, 1 = Defect, 2 = Insulator.

    Returns:
        model (RetinaNet): Initialized RetinaNet with MobileNetV2 backbone.
    """
    # 1. Load pre-trained MobileNetV2 feature extractor
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    backbone_full = mobilenet_v2(weights=weights).features

    # 2. Define FPN return layers:
    #    We extract feature maps from 3 stages of MobileNetV2:
    #    - Layer 6  → early features (stride 8)  → FPN level 0
    #    - Layer 13 → mid features  (stride 16)  → FPN level 1
    #    - Layer 18 → deep features (stride 32)  → FPN level 2
    return_layers = {'6': '0', '13': '1', '18': '2'}

    # Channel widths at the 3 selected layers of MobileNetV2
    in_channels_list = [32, 96, 1280]
    out_channels = 256

    backbone_with_fpn = BackboneWithFPN(
        backbone_full,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelP6P7(out_channels, out_channels)
    )

    # 3. Build the RetinaNet detector
    model = RetinaNet(backbone_with_fpn, num_classes=num_classes)

    # 4. Replace the head to match our num_classes precisely
    num_anchors = model.head.classification_head.num_anchors
    model.head = RetinaNetHead(out_channels, num_anchors, num_classes)

    # 5. Register Focal Loss class-balance alpha weights as a model buffer
    #    so they are automatically moved to GPU with model.to(device)
    #    Logic: [Background, Defect, Insulator]
    #           Defect (Class 1) is rare and hardest → boosted to 0.75
    alpha_weights = torch.tensor([0.25, 0.75, 0.25])
    model.register_buffer('focal_loss_alpha', alpha_weights)

    return model
