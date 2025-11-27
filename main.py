import os
import time
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import multiprocessing

# ---------------------------
# Config
# ---------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Flatten / set parameters
# ---------------------------

def get_param_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def get_param_shapes(model: nn.Module):
    shapes = [p.shape for p in model.parameters()]
    sizes = [p.numel() for p in model.parameters()]
    return shapes, sizes

def set_param_vector(model: nn.Module, vec: torch.Tensor):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[pointer:pointer + numel].view_as(p))
        pointer += numel

# ---------------------------
# Hessian Vector Product
# ---------------------------

def hvp(loss: torch.Tensor, model: nn.Module, v: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([g.reshape(-1) for g in grads])
    grad_v = torch.dot(grads_flat, v)
    Hv_grads = torch.autograd.grad(grad_v, model.parameters(), retain_graph=True)
    Hv_flat = torch.cat([h.reshape(-1) for h in Hv_grads]).detach()
    return Hv_flat

def build_loss_on_subset(model: nn.Module, dataloader: DataLoader, device: str, max_samples: int = 1024) -> torch.Tensor:
    model.zero_grad()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    loss_accum = 0.0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        batch_size = xb.size(0)
        loss_accum += loss * batch_size
        total += batch_size
        if total >= max_samples:
            break
    return loss_accum / total

def lanczos(Hv_fn: Callable[[torch.Tensor], torch.Tensor], n: int, m: int, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    alphas = np.zeros(m, dtype=np.float64)
    betas = np.zeros(m - 1, dtype=np.float64)

    q = torch.randn(n, device=device)
    q = q / q.norm()
    q_prev = torch.zeros_like(q)

    for j in range(m):
        z = Hv_fn(q)
        alpha = float(torch.dot(q, z).cpu())
        alphas[j] = alpha
        z = z - alpha * q - (betas[j - 1] * q_prev if j > 0 else 0.0)
        beta = float(z.norm().cpu())
        if j < m - 1:
            betas[j] = beta

        q_prev = q
        if beta == 0:
            break

        q = z / beta

    return alphas, betas

def tridiag_eigvals(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    m = len(alphas)
    T = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    w, _ = np.linalg.eigh(T)
    return np.sort(w)[::-1]

# ---------------------------
# Main execution guard for Windows
# ---------------------------

multiprocessing.freeze_support()

if __name__ == "__main__":

    epochs = 6
    batch_size = 128
    lr = 0.1
    device = "cpu"
    hvp_samples = 1024
    lanczos_steps = 30
    topk = 10
    out_dir = "hessian_out"

    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------
    # Dataset / DataLoaders (Windows safe)
    # ---------------------------

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_subset = Subset(full_train, list(range(5000)))
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    hvp_subset = Subset(full_train, list(range(hvp_samples)))
    hvp_loader = DataLoader(hvp_subset, batch_size=256, shuffle=False, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    # ---------------------------
    # Model / optimizer
    # ---------------------------

    model = SmallCNN(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    shapes, sizes = get_param_shapes(model)
    num_params = sum(sizes)
    print(f"Model has {num_params} params, device: {device}")

    eigs_per_epoch = []

    # ---------------------------
    # Training + spectral probe
    # ---------------------------

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0

        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)

        train_loss = loss_sum / len(trainloader.dataset)
        print(f"Epoch {epoch}/{epochs} | train loss: {train_loss:.4f}")

        model.zero_grad()
        loss_h = build_loss_on_subset(model, hvp_loader, device=device, max_samples=hvp_samples)

        def Hv_fn_torch(vec: torch.Tensor) -> torch.Tensor:
            vec = vec.to(device)
            return hvp(loss_h, model, vec)

        a, b = lanczos(Hv_fn_torch, n=num_params, m=lanczos_steps, device=device)
        eigs = tridiag_eigvals(a, b)
        topk_eigs = eigs[:topk]
        print(f"Lanczos top{topk}: {topk_eigs}")
        eigs_per_epoch.append(topk_eigs)

        np.savez(os.path.join(out_dir, "eigs_epochs.npz"), eigs=np.array(eigs_per_epoch))

    # ---------------------------
    # Plot spectral evolution
    # ---------------------------

    eig_arr = np.array(eigs_per_epoch)
    ep = np.arange(1, eig_arr.shape[0] + 1)

    plt.figure()
    for i in range(min(topk, eig_arr.shape[1])):
        plt.plot(ep, eig_arr[:, i], label=f"eig{i+1}")

    plt.xlabel("Epoch")
    plt.ylabel("Eigenvalue")
    plt.title("Hessian Spectrum Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectrum.png"))
    print("Done.")
