#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # adjust if res/ lives elsewhere
    base_dir  = Path(__file__).parent
    file_path = base_dir / 'res' / 'eigvals_k500.npy'

    # Load eigenvalues
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find {file_path}. "
                                "Run LeNet.py first to generate it.")
    eigvals = np.load(file_path)
    k       = eigvals.size

    # Full-index array
    idx = np.arange(1, k+1)

    # Tail (last 250 eigenvalues) for fit
    tail_idx  = idx[-250:]
    tail_vals = eigvals[-250:]
    # filter out non-positive eigenvalues to avoid log errors
    positive_mask = tail_vals > 0
    tail_idx  = tail_idx[positive_mask]
    tail_vals = tail_vals[positive_mask]
    if tail_vals.size == 0:
        raise ValueError("No positive eigenvalues in the tail to fit.")

    # Fit power-law: log10(λ) = slope·log10(k) + intercept
    log_idx  = np.log10(tail_idx)
    log_tail = np.log10(tail_vals)
    slope, intercept = np.polyfit(log_idx, log_tail, 1)
    fit_vals = 10**intercept * tail_idx**slope

    print(f"Fitted power-law: λ ≃ k^{slope:.4f}  (intercept = {intercept:.4f})")

    # Plot 1: Full spectrum (log–log)
    plt.figure(figsize=(6,4))
    plt.loglog(idx, eigvals, marker='o', linestyle='none', markersize=3)
    plt.xlabel('Index k')
    plt.ylabel('Eigenvalue λ')
    plt.title('Top 500 Hessian Eigenvalues (log–log)')
    plt.tight_layout()
    plt.show()

    # Plot 2: Tail & fit (log–log)
    plt.figure(figsize=(6,4))
    plt.loglog(tail_idx, tail_vals, marker='o', linestyle='none', markersize=4, label='Data')
    plt.loglog(tail_idx, fit_vals, linestyle='--', linewidth=2,
               label=f'Fit slope={slope:.4f}')
    plt.xlabel('Index k (tail)')
    plt.ylabel('Eigenvalue λ')
    plt.title('Power-law Fit to Tail of Spectrum')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()