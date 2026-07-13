import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOBILENET_CODE = os.path.join(_HERE, '..', 'Insulator-Defect-Detection-System-RetinaNet-',
                               'Retinanet_with_mobilenet_backbone(hybrid)', 'Fed_learning_Code')
sys.path.insert(0, os.path.normpath(_MOBILENET_CODE))

import fedprox_train_mobilenet

def run_statistical_validation():
    print("Starting Statistical Validation (Multiple Seeds)...")
    
    seeds = [42, 100, 2026]
    results_map50 = []
    
    for seed in seeds:
        print(f"\n=========================================")
        print(f"Running Trial with Seed: {seed}")
        print(f"=========================================\n")
        
        hist = fedprox_train_mobilenet.run_experiment(
            partition_mode='iid',
            output_tag=f'stat_val_seed_{seed}',
            algorithm='fedprox',
            mu=0.01,
            num_clients=3,
            seed=seed
        )
        
        best_map = max(hist['map50'])
        results_map50.append(best_map)
        print(f"Seed {seed} -> mAP@50: {best_map:.4f}")
        
    print("\n--- Statistical Validation Results ---")
    mean_map = np.mean(results_map50)
    std_map = np.std(results_map50)
    
    print(f"Trials: {len(seeds)}")
    print(f"Scores: {[f'{x:.4f}' for x in results_map50]}")
    print(f"mAP@50: {mean_map:.4f} ± {std_map:.4f}")

if __name__ == '__main__':
    run_statistical_validation()
