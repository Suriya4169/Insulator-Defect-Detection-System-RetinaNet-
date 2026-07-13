import os
import sys

# Add MobileNet folder to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOBILENET_CODE = os.path.join(_HERE, '..', 'Insulator-Defect-Detection-System-RetinaNet-',
                               'Retinanet_with_mobilenet_backbone(hybrid)', 'Fed_learning_Code')
sys.path.insert(0, os.path.normpath(_MOBILENET_CODE))

import fedprox_train_mobilenet

def run_ablation():
    print("Starting Ablation Study for Section 7.3...")
    
    configs = [
        {"tag": "base", "cp": False, "ws": False},
        {"tag": "base_cp", "cp": True, "ws": False},
        {"tag": "base_ws", "cp": False, "ws": True},
        {"tag": "proposed", "cp": True, "ws": True},
    ]
    
    results = {}
    
    for c in configs:
        print(f"\n=========================================")
        print(f"Running Configuration: {c['tag']}")
        print(f"Copy-Paste: {c['cp']} | Weighted Sampling: {c['ws']}")
        print(f"=========================================\n")
        
        hist = fedprox_train_mobilenet.run_experiment(
            partition_mode='iid',
            output_tag=f'ablation_{c["tag"]}',
            algorithm='fedavg',
            use_copy_paste=c['cp'],
            use_weighted_sampling=c['ws'],
            num_clients=3
        )
        
        results[c['tag']] = max(hist['map50'])
        
    print("\n--- Ablation Study Results ---")
    for tag, max_map in results.items():
        print(f"{tag}: {max_map:.4f} mAP@50")

if __name__ == '__main__':
    run_ablation()
