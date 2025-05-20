import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math, json, argparse, pathlib
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import re

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
          schedule='constant', alpha=1.0,
          switch_patience=3, switch_delta=1e-3):
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
    """
    text = pathlib.Path(path).read_text(encoding='utf‑8')
    ds = SymbolDataset(text)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    V = len(ds.vocab)
    logits = torch.nn.Parameter(torch.zeros(V))
    opt = torch.optim.SGD([logits], lr=lr)

    ce = torch.nn.CrossEntropyLoss()

    # empirical unigram distribution by model index
    counts = Counter(ds.tokens)
    total = len(ds.tokens)
    emp_dist = np.zeros(V, dtype=float)
    for tok, idx in ds.vocab.items():
        emp_dist[idx] = counts[tok] / total

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

        # ---- auto schedule logic ----
        if schedule == 'auto':
            if prev_loss is not None and (prev_loss - loss.item()) < switch_delta:
                stale_epochs += 1
            else:
                stale_epochs = 0
            prev_loss = loss.item()

            if mode != 'invtime' and stale_epochs >= switch_patience:
                mode = 'invtime'
                print(f'>>> Switching to 1/t decay after epoch {ep+1}')

    probs = torch.softmax(logits, 0).detach().cpu().numpy()
    id2sym = {i: s for s, i in ds.vocab.items()}
    model = {id2sym[i]: float(p) for i, p in enumerate(probs)}
    # compute empirical counts once
    counts = Counter(ds.tokens)
    total = len(ds.tokens)

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
        plt.savefig(plot)
        print(f'Plot saved to {plot}')

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
            plt.savefig(plot_diff)
            print(f'Diff‑only plot saved to {plot_diff}')
        else:
            print(f'No ranks exceeded diff_thresh={diff_thresh}; no diff plot written.')

    # ----- Plot loss vs. global steps -----
    if plot_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(step_list, loss_list)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('Loss vs global step')
        plt.tight_layout()
        plt.savefig(plot_loss)
        print(f'Loss vs steps plot saved to {plot_loss}')

    # ----- Plot true cross-entropy vs. global steps -----
    if plot_ce:
        plt.figure(figsize=(6, 4))
        plt.plot(step_list, ce_list)
        plt.xlabel('step')
        plt.ylabel('true cross-entropy')
        plt.title('Cross-entropy vs global step')
        plt.tight_layout()
        plt.savefig(plot_ce)
        print(f'True cross-entropy plot saved to {plot_ce}')

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
          args.schedule, args.alpha, args.switch_patience, args.switch_delta)