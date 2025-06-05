#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from zipf import train
from collections import Counter
import math

def main():
    ap = argparse.ArgumentParser(description="Sweep lr and batch for zipf training")
    ap.add_argument('file', help='plain-text corpus')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.1, help='Default LR if --lrs not given')
    ap.add_argument('--batch', type=int, default=4096, help='Default batch if --batches not given')
    ap.add_argument('--lrs', type=float, nargs='+', default=None,
                    help='List of learning rates to sweep')
    ap.add_argument('--batches', type=int, nargs='+', default=None,
                    help='List of batch sizes to sweep')
    ap.add_argument('--diff-thresh', type=float, default=0.05,
                    help='Relative diff threshold (used if zipf.py’s plot_diff is enabled)')
    ap.add_argument('--schedule', choices=['constant','invtime','power','auto'], default='constant')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--switch-patience', type=int, default=3)
    ap.add_argument('--switch-delta', type=float, default=1e-3)
    args = ap.parse_args()

    # Compute empirical entropy from file frequencies
    def empirical_entropy(file_path):
        with open(file_path, 'r') as f:
            data = f.read().split()
        counts = Counter(data)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        return -sum(p * math.log(p) for p in probs)

    H_empirical = empirical_entropy(args.file)

    # Determine sweep ranges
    lrs = args.lrs if args.lrs is not None else [args.lr]
    batches = args.batches if args.batches is not None else [args.batch]
    results = {}

    for lr in lrs:
        for batch in batches:
            print(f"🧪  Running: lr={lr}, batch={batch}")
            kl_div = train(
                args.file, epochs=args.epochs, batch=batch, lr=lr,
                plot=None, plot_diff=None, diff_thresh=args.diff_thresh,
                plot_loss=None, plot_ce=None,
                schedule=args.schedule, alpha=args.alpha,
                switch_patience=args.switch_patience, switch_delta=args.switch_delta
            )
            # Compute KL divergence = cross-entropy - empirical entropy
            results[(lr, batch)] = kl_div

    # Save all final cross-entropy results to CSV
    import csv
    csv_path = 'final_losses.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['learning_rate', 'batch_size', 'kl_divergence'])
        for (lr_val, batch_val), ce_val in results.items():
            writer.writerow([lr_val, batch_val, ce_val])
    print(f"✅ Saved {csv_path} (KL divergence values)")
    # Plot final loss vs LR (one line per batch)
    plt.figure()
    for batch in batches:
        ys = [results[(lr, batch)] for lr in lrs]
        plt.plot(lrs, ys, marker='o', label=f"batch={batch}")
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('KL divergence')
    plt.title('KL divergence vs Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_vs_lr.png')
    print("✅ Saved loss_vs_lr.png")

    # Plot final loss vs Batch (one line per LR)
    plt.figure()
    for lr in lrs:
        ys = [results[(lr, batch)] for batch in batches]
        plt.plot(batches, ys, marker='o', label=f"lr={lr}")
    plt.xlabel('batch size')
    plt.ylabel('KL divergence')
    plt.title('KL divergence vs Batch Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_vs_batch.png')
    print("✅ Saved loss_vs_batch.png")

    # Plot KL divergence vs Temperature (lr / batch size)
    # Compute temperature and corresponding KL divergence values
    temps = []
    kl_vals = []
    for (lr_val, batch_val), kl in results.items():
        temps.append(lr_val / batch_val)
        kl_vals.append(kl)
    # Sort by temperature for a cleaner plot
    sorted_pairs = sorted(zip(temps, kl_vals), key=lambda x: x[0])
    temps_sorted, kl_sorted = zip(*sorted_pairs)
    # Plot
    plt.figure()
    plt.plot(temps_sorted, kl_sorted, marker='o')
    plt.xlabel('Temperature (learning rate / batch size)')
    plt.ylabel('KL divergence')
    plt.title('KL divergence vs Temperature')
    plt.tight_layout()
    plt.savefig('kl_vs_temperature.png')
    print("✅ Saved kl_vs_temperature.png")

if __name__ == '__main__':
    main()