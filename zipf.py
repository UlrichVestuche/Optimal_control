import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math, json, argparse, pathlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

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
          plot_loss=None, plot_ce=None, plot_kld=None,
          schedule='constant', alpha=1.0,
          kld_thresh=0.001, kld_patience=2,
          init_logits=None, start_epoch: int = 0, start_step: int = 0):
    # Create output folder for plots to avoid overwriting
    folder = os.path.join('pic', f'zipf_ep{epochs}_bs{batch}_lr{lr}_sch{schedule}')
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
        alpha: exponent for 'power' schedule (lr = lr0 / t^alpha).
        kld_thresh: threshold for KL divergence change for early stopping.
        kld_patience: number of epochs to check KL divergence flattening.
    """
    text = pathlib.Path(path).read_text(encoding='utf‑8')
    ds = SymbolDataset(text)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    # Compute total number of optimization steps for the remaining-power schedule
    total_steps = epochs * len(loader)

    V = len(ds.vocab)
    if init_logits is not None:
        logits = init_logits.to(device)
        if not logits.requires_grad:
            logits = torch.nn.Parameter(logits)
    else:
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
    switch_step = None  # record the global step when schedule switches
    emp_entropy = -np.sum(emp_dist * np.log(emp_dist + 1e-12))
    kld_list = []
    kld_step_list = []
    kld_step_index = []

    step = start_step          # global step counter
    initial_lr = lr
    # Start with a constant learning‑rate for schedules that need an eventual switch
    # (`auto` should behave like `constant` until the KL plateau, just like
    #  `remaining_power` does.)
    if schedule in ('auto', 'remaining_power'):
        mode = 'constant'
    else:
        mode = schedule

    # If we're resuming (start_epoch > 0) we already *are* at the plateau,
    # so jump straight into the schedule’s post‑plateau phase.
    if start_epoch > 0:
        if schedule == 'remaining_power':
            mode = 'remaining_power'   # skip the constant warm‑up
        elif schedule == 'auto':
            mode = 'invtime'           # auto resumes as 1/t decay

    step_list = []
    loss_list = []
    ce_list = []

    for ep in range(start_epoch, start_epoch + epochs):
        for x in loader:                       # x shape: (B,)
            x = x.to(device)
            opt.zero_grad()
            loss = ce(logits.repeat(x.size(0), 1), x)   # broadcast
            loss.backward()
            opt.step()
            step += 1
            step_list.append(step)
            loss_list.append(loss.item())
            # compute probabilities once for both true CE and KLD
            with torch.no_grad():
                probs_np = torch.softmax(logits, 0).cpu().numpy()
            ce_true = -np.sum(emp_dist * np.log(probs_np + 1e-12))
            ce_list.append(ce_true)

            # compute KLD per step directly from the same probabilities
            kld_step = ce_true - emp_entropy
            kld_step_list.append(kld_step)
            kld_step_index.append(step)

            # Learning-rate updates based on schedule
            if mode == 'invtime':
                new_lr = initial_lr / ((step/(100* epochs) + 1) ** 1.0)
                opt.param_groups[0]['lr'] = new_lr
            elif mode == 'power':
                new_lr = initial_lr / ((step + 1) ** alpha)
                opt.param_groups[0]['lr'] = new_lr
            elif mode == 'remaining_power':
                # Apply u(t) = (1 - t/T)^(alpha/(1-alpha)), where T = total_steps
                frac = 1.0 - (step / float(total_steps))
                if frac < 0:
                    frac = 0.0
                exponent = alpha / (1.0 - alpha)
                new_lr = initial_lr * (frac ** exponent)
                opt.param_groups[0]['lr'] = new_lr
        # ce_true is the cross‑entropy between the *full* empirical distribution and the current model,
        # so it reflects the “true” loss for the whole corpus rather than just the last minibatch.
        print(f'Epoch {ep+1}: true CE {ce_true:.4f}  |  lr {opt.param_groups[0]['lr']:.3e}')

        # Early stopping based on KL divergence flattening (using last step's CE)
        kld_epoch = ce_true - emp_entropy
        kld_list.append(kld_epoch)
        
        # ---- Plateau logic using kld_thresh & kld_patience ----
        if schedule in ('auto', 'remaining_power'):
            target_mode = 'invtime' if schedule == 'auto' else 'remaining_power'
            if len(kld_list) >= kld_patience + 1 and mode != target_mode:
                recent = kld_list[-(kld_patience + 1):]
                diffs  = [abs(recent[i] - recent[i + 1]) for i in range(kld_patience)]
                if all(d < kld_thresh for d in diffs):
                    # --- save a checkpoint only once ---
                    ckpt_path = 'plateau.ckpt'
                    if not os.path.exists(ckpt_path):
                        torch.save({'logits': logits.detach().cpu(),
                                    'epoch' : ep,
                                    'step'  : step,
                                    'path'  : path}, ckpt_path)
                        print(f'Saved plateau checkpoint → {ckpt_path}')
                    mode = target_mode
                    switch_step = step
                    print(f'>>> Switching to {target_mode} after epoch {ep+1} '
                          f'due to plateau (|ΔKLD|<{kld_thresh} for {kld_patience} epochs)')
        else:
            # For constant / power / invtime schedules, early‑stop using the same criterion
            if len(kld_list) >= kld_patience + 1:
                recent = kld_list[-(kld_patience + 1):]
                diffs  = [abs(recent[i] - recent[i + 1]) for i in range(kld_patience)]
                if all(d < kld_thresh for d in diffs):
                    print(f'Early stopping at epoch {ep+1} because KL divergence '
                          f'flattened (|ΔKLD|<{kld_thresh} for {kld_patience} epochs)')
                    break

    probs = torch.softmax(logits, 0).detach().cpu().numpy()
    id2sym = {i: s for s, i in ds.vocab.items()}
    model = {id2sym[i]: float(p) for i, p in enumerate(probs)}

    # --- Persist raw step‑level KLD for later comparison ---
    if plot_kld:
        kld_data_name = os.path.splitext(os.path.basename(plot_kld))[0] + '.npz'
        np.savez(os.path.join(folder, kld_data_name),
                 steps=np.array(kld_step_index),
                 kld=np.array(kld_step_list))
    # ----- Plot KL divergence vs. epochs or steps -----
    if plot_kld:
        # Only plot after schedule switch and KLD plateau
        if switch_step is not None:
            # Identify indices corresponding to steps >= switch_step
            filtered = [i for i, s in enumerate(kld_step_index) if s >= switch_step]
            if filtered:
                steps_to_plot = [kld_step_index[i] for i in filtered]
                kld_to_plot = [kld_step_list[i] for i in filtered]
                plt.figure(figsize=(6, 4))
                plt.plot(steps_to_plot, kld_to_plot, marker='o', label='KLD after schedule switch')
                plt.xlabel('Global Step')
                plt.ylabel('KL Divergence')
                plt.title('KLD over Steps After Schedule Switch')
                plt.tight_layout()
                kld_filename = os.path.basename(plot_kld)
                full_kld_path = os.path.join(folder, kld_filename)
                plt.savefig(full_kld_path)
                print(f'KLD over steps plot saved to {full_kld_path}')
            else:
                print('No KLD values recorded after schedule switch; no KLD-over-steps plot written.')
        else:
            print('Schedule never switched; no KLD-over-steps plot written.')

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


# --- Resume from checkpoint helper ---
def resume_from_ckpt(ckpt_file: str, new_schedule: str,
                     epochs: int = 5, reset_step: bool = False, **kwargs):
    """
    Resume training from a saved plateau checkpoint with a new LR schedule.
    """
    ckpt = torch.load(ckpt_file, map_location=device)
    logits_param = torch.nn.Parameter(ckpt['logits'].to(device))
    return train(ckpt['path'],
                 epochs=epochs,
                 schedule=new_schedule,
                 init_logits=logits_param,
                 start_epoch=ckpt['epoch'] + 1,
                 start_step=0 if reset_step else ckpt['step'],
                 **kwargs)

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
    ap.add_argument('--schedule', choices=['constant', 'invtime', 'power', 'auto', 'remaining_power'],
                    default='constant',
                    help="Learning‑rate schedule: constant or 1/t ('invtime')")
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='exponent for --schedule power (default 1.0)')
    ap.add_argument('--plot-diff', default=None,
                    help='Path to save plot of ranks where learned vs empirical differ')
    ap.add_argument('--diff-thresh', type=float, default=0.1,
                    help='Relative difference threshold for --plot-diff')
    args = ap.parse_args()
    # ----------------------------------------------------------------------------------
    # If plateau.ckpt already exists, skip fresh training and compare post‑plateau
    # schedules (invtime vs. remaining_power) right away.
    # ----------------------------------------------------------------------------------
    if os.path.exists('plateau.ckpt'):
        print('Found plateau.ckpt – running schedule comparison from checkpoint...')
        resume_epochs = args.epochs  # use CLI --epochs for follow‑up phase length
        resume_from_ckpt('plateau.ckpt', 'remaining_power',
                         epochs=resume_epochs,
                         batch=args.batch,
                         lr=args.lr,
                         alpha=args.alpha,
                         plot_kld='rempow_kld.png',
                         reset_step=True)
        
        resume_from_ckpt('plateau.ckpt', 'invtime',
                         epochs=resume_epochs,
                         batch=args.batch,
                         lr=args.lr,
                         alpha=args.alpha,
                         plot_kld='invtime_kld.png',
                         reset_step=True)

        

        # Overlay comparison
        inv_npz = os.path.join('pic', f'zipf_ep{resume_epochs}_bs{args.batch}_lr{args.lr}_schinvtime',  'invtime_kld.npz')
        rem_npz = os.path.join('pic', f'zipf_ep{resume_epochs}_bs{args.batch}_lr{args.lr}_schremaining_power', 'rempow_kld.npz')
        if os.path.exists(inv_npz) and os.path.exists(rem_npz):
            inv_data = np.load(inv_npz)
            rem_data = np.load(rem_npz)
            plt.figure(figsize=(6,4))
            plt.plot(inv_data['steps'], inv_data['kld'], label='invtime')
            plt.plot(rem_data['steps'], rem_data['kld'], label='remaining_power')
            plt.xlabel('Step'); plt.ylabel('KL Divergence')
            plt.title('Schedule comparison (from checkpoint)')
            plt.legend(); plt.tight_layout()
            comp_path = os.path.join('pic', 'kld_compare.png')
            plt.savefig(comp_path)
            print(f'Combined KLD comparison saved to {comp_path}')
        else:
            print('Could not find KLD data for both schedules; comparison plot skipped.')

        sys.exit(0)

    train(args.file, args.epochs, args.batch, args.lr,
          args.plot, args.plot_diff, args.diff_thresh, args.plot_loss, args.plot_ce,
          args.plot_kld,
          args.schedule, args.alpha)