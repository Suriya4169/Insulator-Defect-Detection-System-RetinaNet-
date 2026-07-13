import os
import json
import csv
import matplotlib.pyplot as plt

def plot_comm_efficiency(history_json, comm_csv, output_dir, tag):
    with open(history_json, 'r') as f:
        history = json.load(f)
        
    map50 = history.get('map50', [])
    if not map50:
        print(f"No mAP data found in {history_json}")
        return

    cumulative_mb = []
    with open(comm_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cumulative_mb.append(float(row['Cumulative_MB']))

    # Ensure lengths match
    min_len = min(len(map50), len(cumulative_mb))
    map50 = map50[:min_len]
    cumulative_mb = cumulative_mb[:min_len]

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_mb, map50, marker='o', linestyle='-', linewidth=2, color='b')
    plt.title(f'mAP@50 vs. Communication Cost ({tag})')
    plt.xlabel('Cumulative Communication Cost (MB)')
    plt.ylabel('mAP@50')
    plt.grid(True)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f'{tag}_comm_efficiency.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved communication efficiency plot to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', required=True, help='Path to history.json')
    parser.add_argument('--comm', required=True, help='Path to communication_log.csv')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--tag', default='FedAvg', help='Algorithm tag for the plot title')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    plot_comm_efficiency(args.history, args.comm, args.out, args.tag)
