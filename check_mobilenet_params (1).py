import torch
from model_mobilenet import get_mobilenet_retinanet

def check_params():
    model = get_mobilenet_retinanet(num_classes=5)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*40)
    print(f" MODEL: RetinaNet + MobileNetV3-Large")
    print("="*40)
    print(f" Total Parameters: {total_params:,}")
    print(f" ResNet-50 Params: 36,414,865")
    print(f" Reduction: {((36414865 - total_params)/36414865)*100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    check_params()

