import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# 1. LeNet-5 definition
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # 28→24
        self.pool  = nn.AvgPool2d(2, stride=2)           # 24→12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     # 12→8
        # pool →4
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. Data loaders factory
def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('.', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
            DataLoader(test_ds,  batch_size=1000,        shuffle=False, num_workers=2))

# 3. Single train/eval pass
def run_experiment(batch_size, epochs=5, device='cpu'):
    train_loader, test_loader = get_loaders(batch_size)
    model = LeNet5().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'epoch_time': []}

    for ep in range(epochs):
        t0 = time.time()
        # — train
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            opt.step()
            _, pred = logits.max(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
        history['train_acc'].append(correct/total)

        # — eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, pred = logits.max(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        history['test_acc'].append(correct/total)

        history['epoch_time'].append(time.time() - t0)
        print(f"[bs={batch_size:4d}] Epoch {ep+1}/{epochs} — "
              f"train_acc={history['train_acc'][-1]:.4f}, "
              f"test_acc={history['test_acc'][-1]:.4f}, "
              f"time={history['epoch_time'][-1]:.1f}s")

    return history

# 4. Sweep over batch sizes
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.autograd import grad
    from scipy.sparse.linalg import LinearOperator, eigsh
    import numpy as np
    from tqdm import tqdm

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64  # adjust as needed

    train_loader, test_loader = get_loaders(batch_size)
    model = LeNet5().to(device)
    weight_file = 'lenet_mnist.pth'
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location=device))
        print(f"Loaded model weights from {weight_file}")
    else:
        history = run_experiment(batch_size, epochs=20, device=device)
        torch.save(model.state_dict(), weight_file)
        print(f"Saved model weights to {weight_file}")

    # --- Compute top-k Hessian eigenvalues ---
    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)

    def flat_grad(loss):
        g = grad(loss, params, create_graph=True)
        return torch.cat([gi.view(-1) for gi in g])

    criterion = nn.CrossEntropyLoss()
    def hessian_vector_product(v_np):
        # ensure float32 for MPS compatibility
        v = torch.from_numpy(v_np.astype(np.float32)).to(device)
        # zero old grads
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        # compute full-train loss (retain gradient graph)
        total_loss = 0.0
        model.eval()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y)
        total_loss = total_loss / len(train_loader)
        # gradient and directional derivative
        g = flat_grad(total_loss)
        Hv = grad(g @ v, params, retain_graph=False)
        return torch.cat([h.contiguous().view(-1) for h in Hv]).cpu().numpy()

    k = 500
    # wrap HVP to update progress bar
    bar = tqdm(desc='HVP calls')
    def matvec_progress(v_np):
        bar.update(1)
        return hessian_vector_product(v_np)

    linop = LinearOperator((n_param, n_param), matvec=matvec_progress, dtype=float)
    # set maxiter to control maximum iterations (optional)
    eigvals, _ = eigsh(linop, k=k, which='LM', maxiter=k+10)
    bar.close()
    eigvals = np.sort(eigvals)[::-1]

    # save eigenvalues
    os.makedirs('res', exist_ok=True)
    file_path = os.path.join('res', f'eigvals_k{k}.npy')
    np.save(file_path, eigvals)
    print(f"Saved eigenvalues to {file_path}")

    # fit a power-law to the last 250 eigenvalues
    idx = np.arange(1, k+1)
    tail_idx = idx[-250:]
    tail_vals = eigvals[-250:]
    log_idx = np.log10(tail_idx)
    log_tail = np.log10(tail_vals)
    slope, intercept = np.polyfit(log_idx, log_tail, 1)
    fit_vals = 10**intercept * tail_idx**slope
    print(f"Fitted power-law: λ ~ k^({slope:.4f}) with intercept {intercept:.4f}")

    # plot data and fit on log-log scale
    plt.figure()
    plt.loglog(idx, eigvals, marker='o', linestyle='none', label='Eigenvalues')
    plt.loglog(tail_idx, fit_vals, linestyle='--', label=f'Fit slope={slope:.4f}')
    plt.xlabel('Index (k)')
    plt.ylabel('Eigenvalue (λ)')
    plt.title(f'Top {k} Hessian Eigenvalues & Fit (log-log)')
    plt.legend()
    plt.show()