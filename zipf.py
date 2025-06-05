import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math, json, argparse, pathlib
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import re
import os

# Detect device for Mac M-series (MPS) or fallback
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def ensure_dir(path):
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

class SymbolDataset(Dataset):
    def __init__(self, text, vocab=None):
        # tokenize into words (letters + digits) and lower‑case
        tokens = re.findall(r'\w+', text.lower())
        self.tokens = tokens  # keep for later analysis

        if vocab is None:
            counts = Counter(tokens)
            self.vocab = {tok: i for i, (tok, _) in enumerate(counts.most_common())}
        else:
            self.vocab = vocab

        self.ids = torch.tensor([self.vocab[tok] for tok in tokens],
                                dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]

def train(path, epochs=10, batch=4096, lr=0.1,
          plot=None, plot_diff=None, diff_thresh=0.05,
          plot_loss=None, plot_ce=None,
          plot_kld=None,
          schedule='constant', alpha=1.0,
          switch_patience=3, switch_delta=1e-3,
          kld_thresh=1e-3, kld_patience=2):
    # Create output folder for plots to avoid overwriting
    folder = os.path.join('pic', f'zipf_ep{epochs}_bs{batch}_lr{lr}')
    ensure_dir(folder)

    """
    Train a unigram model to fit Zipf's law via SGD.
    Args:
        path: Path to text file.
        epochs: Number of epochs.
        batch: Batch size.
        lr: Learning rate.
        plot: If not None, path to save log–log rank–frequency plot.
        plot_diff: optional path to save a figure that shows only the ranks
                   where |learned – empirical| / empirical > diff_thresh.
        diff_thresh: relative difference threshold for plot_diff.
        schedule: 'constant', 'invtime', or 'auto'.
                  'auto' keeps lr constant until the epoch loss fails to
                  improve by more than switch_delta for switch_patience epochs,
                  then switches to 1/t decay.
        alpha: exponent for 'power' schedule (lr = lr0 / t^alpha).
        switch_patience: epochs to wait before switching (auto mode).
        switch_delta: minimum loss improvement to reset patience.
        kld_thresh: threshold for KL divergence change for early stopping.
        kld_patience: number of epochs to check KL divergence flattening.
    """
    text = pathlib.Path(path).read_text(encoding='utf‑8')
    ds = SymbolDataset(text)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    V = len(ds.vocab)
    logits = torch.nn.Parameter(torch.zeros(V, device=device))
    opt = torch.optim.SGD([logits], lr=lr)

    ce = torch.nn.CrossEntropyLoss()

    # empirical unigram distribution by model index
    counts = Counter(ds.tokens)
    total = len(ds.tokens)
    emp_dist = np.zeros(V, dtype=float)
    for tok, idx in ds.vocab.items():
        emp_dist[idx] = counts[tok] / total

    # Pre-compute empirical entropy for KL-based early stopping
    switch_epoch = None  # Record epoch when auto schedule switches to invtime
    emp_entropy = -np.sum(emp_dist * np.log(emp_dist + 1e-12))
    kld_list = []

    step = 0                   # global step counter
    initial_lr = lr
    mode = schedule          # may change from 'auto' to 'invtime'
    prev_loss = None
    stale_epochs = 0

    step_list = []
    loss_list = []
    ce_list = []

    for ep in range(epochs):
        for x in loader:                       # x shape: (B,)
            x = x.to(device)
            opt.zero_grad()
            loss = ce(logits.repeat(x.size(0), 1), x)   # broadcast
            loss.backward()
            opt.step()
            step += 1
            step_list.append(step)
            loss_list.append(loss.item())
            # compute true cross-entropy against empirical distribution
            probs_np = torch.softmax(logits, 0).detach().cpu().numpy()
            ce_true = -np.sum(emp_dist * np.log(probs_np + 1e-12))
            ce_list.append(ce_true)
            if mode in ('invtime', 'power'):
                new_lr = initial_lr / ((step + 1) ** (alpha if mode == 'power' else 1.0))
                opt.param_groups[0]['lr'] = new_lr
        current_lr = opt.param_groups[0]['lr']
        print(f'Epoch {ep+1}: loss {loss.item():.4f}  |  lr {current_lr:.3e}')

        # Early stopping based on KL divergence flattening
        # Compute KL divergence for this epoch
        probs_epoch = torch.softmax(logits, 0).detach().cpu().numpy()
        ce_epoch = -np.sum(emp_dist * np.log(probs_epoch + 1e-12))
        kld_epoch = ce_epoch - emp_entropy
        kld_list.append(kld_epoch)
        
        # ---- auto schedule logic using KL divergence patience ----
        if schedule == 'auto':
            if len(kld_list) > kld_patience:
                recent_kld = kld_list[-(kld_patience+1):]
                diffs_kld = [abs(recent_kld[i] - recent_kld[i+1]) for i in range(len(recent_kld)-1)]
                if mode != 'invtime' and all(diff < kld_thresh for diff in diffs_kld):
                    mode = 'invtime'
                    switch_epoch = ep + 1
                    print(f'>>> Switching to 1/t decay after epoch {ep+1} due to KL flattening')
        else:
            if len(kld_list) > kld_patience:
                recent = kld_list[-(kld_patience+1):]
                diffs = [abs(recent[i] - recent[i+1]) for i in range(len(recent)-1)]
                if all(diff < kld_thresh for diff in diffs):
                    print(f'Early stopping at epoch {ep+1} due to KL flattening (ΔKL < {kld_thresh} for {kld_patience} epochs)')
                    break

    probs = torch.softmax(logits, 0).detach().cpu().numpy()
    id2sym = {i: s for s, i in ds.vocab.items()}
    model = {id2sym[i]: float(p) for i, p in enumerate(probs)}
    # compute empirical counts once
    counts = Counter(ds.tokens)
    total = len(ds.tokens)

    # ----- Plot KL divergence vs. epochs -----
    if plot_kld:
        import matplotlib.pyplot as plt
        epochs_range = list(range(1, len(kld_list) + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_range, kld_list, marker='o', label='KL divergence')
        if switch_epoch:
            plt.axvline(x=switch_epoch, color='red', linestyle='--', label=f'Switch at epoch {switch_epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.title('KLD over Epochs')
        plt.legend()
        plt.tight_layout()
        kld_filename = os.path.basename(plot_kld)
        full_kld_path = os.path.join(folder, kld_filename)
        plt.savefig(full_kld_path)
        print(f'KL divergence plot saved to {full_kld_path}')

    # dump counts + learned probabilities side‑by‑side
    stats = {tok: {"count": counts[tok], "prob": model.get(tok, 0.0)}
             for tok in counts}

    with open('zipf_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Wrote counts+probs to zipf_stats.json")

    # pre‑compute empirical and learned rank‑frequency lists
    emp_freq = [c / total for _, c in counts.most_common()]
    learned_freq = sorted(model.values(), reverse=True)

    # ----- Compute and display entropies (in nats) -----
    emp_entropy = -sum(p * math.log(p) for p in emp_freq if p > 0)
    learned_entropy = -sum(p * math.log(p) for p in learned_freq if p > 0)
    print(f'Empirical unigram entropy: {emp_entropy:.4f} nats/token')
    print(f'Learned  unigram entropy: {learned_entropy:.4f} nats/token')
    # Compute and print KL divergence between empirical and learned distributions
    ce_final = -np.sum(emp_dist * np.log(probs + 1e-12))
    kld = ce_final - emp_entropy
    print(f'KL divergence (P_emp || P_model): {kld:.4f} nats/token')

    # ----- Plot learned vs. empirical rank–frequency -----
    if plot:
        ranks = range(1, len(emp_freq) + 1)
        plt.figure(figsize=(6, 4))
        plt.loglog(ranks, emp_freq, label='empirical', alpha=0.8)
        plt.loglog(ranks, learned_freq, label='SGD model', alpha=0.8)
        plt.xlabel('rank')
        plt.ylabel('frequency')
        plt.title('Zipf fit')
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.basename(plot)
        full_plot_path = os.path.join(folder, plot_filename)
        plt.savefig(full_plot_path)
        print(f'Plot saved to {full_plot_path}')

    # ----- Plot only where the two distributions differ more than threshold -----
    if plot_diff:
        emp_arr = np.array(emp_freq)
        learned_arr = np.array(learned_freq)
        rel_diff = np.abs(learned_arr - emp_arr) / np.maximum(emp_arr, 1e-12)
        mask = rel_diff > diff_thresh
        if mask.any():
            ranks_diff = np.where(mask)[0] + 1
            # fit a line in log–log space to the empirical leftovers
            x_log = np.log10(ranks_diff)
            y_log = np.log10(emp_arr[mask])
            slope, intercept = np.polyfit(x_log, y_log, 1)
            fit_vals = 10 ** (intercept + slope * x_log)
            print(f'Linear fit on leftovers:  log10(freq) ≈ {slope:.3f}·log10(rank) + {intercept:.3f}')
            plt.figure(figsize=(6, 4))
            plt.loglog(ranks_diff, emp_arr[mask], 'o', label='empirical')
            plt.loglog(ranks_diff, learned_arr[mask], 'o', label='SGD model')
            plt.loglog(ranks_diff, fit_vals, '--', label=f'fit slope={slope:.2f}')
            plt.xlabel('rank')
            plt.ylabel('frequency')
            plt.title(f'Zipf regions where rel. diff > {diff_thresh}')
            plt.legend()
            plt.tight_layout()
            diff_filename = os.path.basename(plot_diff)
            full_diff_path = os.path.join(folder, diff_filename)
            plt.savefig(full_diff_path)
            print(f'Diff‑only plot saved to {full_diff_path}')
        else:
            print(f'No ranks exceeded diff_thresh={diff_thresh}; no diff plot written.')

    # ----- Plot loss minus empirical entropy vs. global steps (log-log) -----
    if plot_loss:
        # subtract empirical entropy from the recorded loss and plot in log-log
        adjusted_loss = np.array(loss_list) - emp_entropy
        plt.figure(figsize=(6, 4))
        plt.loglog(step_list, adjusted_loss, label='loss − empirical entropy')
        plt.xlabel('step')
        plt.ylabel('loss minus empirical entropy')
        plt.title('Loss minus empirical entropy vs global step (log-log)')
        plt.legend()
        plt.tight_layout()
        loss_filename = os.path.basename(plot_loss)
        full_loss_path = os.path.join(folder, loss_filename)
        plt.savefig(full_loss_path)
        print(f'Log-log loss minus empirical entropy plot saved to {full_loss_path}')

    # ----- Plot true cross-entropy vs. global steps -----
    if plot_ce:
        adjusted_ce = np.array(ce_list) - emp_entropy
        plt.figure(figsize=(6, 4))
        plt.plot(step_list, adjusted_ce, label='true cross-entropy')
        #plt.axhline(learned_entropy, linestyle='--', label='learned entropy')
        plt.xlabel('step')
        plt.ylabel('cross-entropy')
        plt.title('True cross-entropy - Empricial entropy vs global step')
        plt.legend()
        plt.tight_layout()
        ce_filename = os.path.basename(plot_ce)
        full_ce_path = os.path.join(folder, ce_filename)
        plt.savefig(full_ce_path)
        print(f'True cross-entropy with learned entropy plot saved to {full_ce_path}')
        # Print final true cross-entropy value
        print(f'Final true cross-entropy: {ce_list[-1]:.4f} nats/token')
    return kld

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('file', help='plain‑text corpus')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--plot', default=None,
                    help='Path to save log–log rank–frequency plot')
    ap.add_argument('--plot-loss', default=None,
                    help='Path to save loss vs. global step plot')
    ap.add_argument('--plot-ce', default=None,
                    help='Path to save true cross-entropy vs global step plot')
    ap.add_argument('--plot-kld', default=None,
                    help='Path to save KL divergence vs. epoch plot')
    ap.add_argument('--schedule', choices=['constant', 'invtime', 'power', 'auto'],
                    default='constant',
                    help="Learning‑rate schedule: constant or 1/t ('invtime')")
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='exponent for --schedule power (default 1.0)')
    ap.add_argument('--plot-diff', default=None,
                    help='Path to save plot of ranks where learned vs empirical differ')
    ap.add_argument('--diff-thresh', type=float, default=0.1,
                    help='Relative difference threshold for --plot-diff')
    ap.add_argument('--switch-patience', type=int, default=2,
                    help='patience epochs before lr decay in auto schedule')
    ap.add_argument('--switch-delta', type=float, default=0.05,
                    help='min loss improvement to reset patience')
    args = ap.parse_args()
    train(args.file, args.epochs, args.batch, args.lr,
          args.plot, args.plot_diff, args.diff_thresh, args.plot_loss, args.plot_ce,
          args.plot_kld,
          args.schedule, args.alpha, args.switch_patience, args.switch_delta)