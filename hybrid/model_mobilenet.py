import torch
import torch.nn as nn
import math
from torchvision.models.detection import RetinaNet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7

class GhostConv(nn.Module):
    """
    Ghost Convolution block.
    This block splits the convolution into a primary convolution and a cheap
    depthwise convolution to generate more feature maps from intrinsic ones.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super(GhostConv, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # Cheap operation (depthwise convolution)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck block with a residual shortcut connection.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # First GhostConv for channel expansion
        self.ghost1 = GhostConv(in_channels, hidden_channels, kernel_size=1, relu=True)

        # Depthwise conv for downsampling if stride > 1
        self.conv_dw = None
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_channels)

        # Second GhostConv for projection (linear)
        self.ghost2 = GhostConv(hidden_channels, out_channels, kernel_size=1, relu=False)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return nn.ReLU(inplace=True)(x)

class GhostNetBackbone(nn.Module):
    """
    A custom GhostNet-style backbone that outputs feature maps for FPN.
    """
    def __init__(self, cfg):
        super(GhostNetBackbone, self).__init__()
        
        # Initial convolution layer (C1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        in_channels = 16

        # Build stages from the configuration list
        self.stages = nn.ModuleList()
        for out_c, repeat, stride in cfg:
            stage_layers = []
            # Expansion factor is hardcoded to 2 for simplicity
            hidden_c = out_c * 2
            for i in range(repeat):
                s = stride if i == 0 else 1
                stage_layers.append(GhostBottleneck(in_channels, hidden_c, out_c, stride=s))
                in_channels = out_c
            self.stages.append(nn.Sequential(*stage_layers))

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x))) # C1, stride 2
        
        c2 = self.stages[0](x)
        c3 = self.stages[1](c2)
        c4 = self.stages[2](c3)
        c5 = self.stages[3](c4)
        
        # Return C3, C4, C5 feature maps for the FPN
        return {'0': c3, '1': c4, '2': c5}

class GhostNetFPNBackbone(nn.Module):
    """
    Wrapper to connect the GhostNetBackbone with a Feature Pyramid Network.
    This module becomes the final backbone for RetinaNet.
    """
    def __init__(self, backbone, in_channels_list, out_channels):
        super(GhostNetFPNBackbone, self).__init__()
        self.backbone = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels, out_channels),
        )
        self.out_channels = out_channels

    def forward(self, x):
        features = self.backbone(x)
        return self.fpn(features)

def get_mobilenet_retinanet(num_classes):
    """
    Constructs a RetinaNet model with a lightweight GhostNet-style backbone.
    """
    # YAML-style configuration for the GhostNet backbone.
    # Each entry: [output_channels, num_bottlenecks, stride]
    BACKBONE_CFG = [
        # C2
        [24, 1, 2],
        # C3
        [40, 2, 2],
        # C4
        [80, 2, 2],
        # C5
        [160, 2, 2],
    ]
    
    # Base GhostNet backbone
    ghostnet_base = GhostNetBackbone(BACKBONE_CFG)
    
    # The channel counts for the feature maps (C3, C4, C5) that FPN will use
    fpn_in_channels = [
        BACKBONE_CFG[1][0], # 40
        BACKBONE_CFG[2][0], # 80
        BACKBONE_CFG[3][0], # 160
    ]
    fpn_out_channels = 128
    
    # Create the complete FPN-enabled backbone
    backbone = GhostNetFPNBackbone(ghostnet_base, fpn_in_channels, fpn_out_channels)
    
    # Define the anchor generator for RetinaNet
    # The default generator is usually sufficient
    anchor_generator = None 

    # Create the final RetinaNet model
    model = RetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator
    )
    
    return model

if __name__ == "__main__":
    # Create the model to test instantiation
    # The num_classes includes the background class, so 4 classes + 1 BG = 5
    model = get_mobilenet_retinanet(5)
    
    # Print total number of parameters to check model size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GhostNet RetinaNet created successfully.")
    print(f"Total Trainable Parameters: {total_params / 1_000_000:.2f}M")

    # Test with a dummy input
    try:
        dummy_input = torch.randn(2, 3, 256, 256)
        model.eval()
        outputs = model(dummy_input)
        print("Model forward pass successful.")
        # print("Output format (first image):")
        # for k, v in outputs[0].items():
        #     print(f"  - {k}: shape {v.shape}")
    except Exception as e:
        print(f"An error occurred during a test forward pass: {e}")
