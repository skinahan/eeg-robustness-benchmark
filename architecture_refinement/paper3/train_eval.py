"""
Training and evaluation harness for PAPER 3.

Training: AdamW, lr=1e-3, weight_decay=1e-4, batch=256, epochs=20, early stopping patience=3.
Evaluation: Accuracy, ROC-AUC, AUPC_k, MaxRD_k, RD_k(α).
Dynamics: M1, M2, M3 on test subset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score

from .harmonic_oscillator_dataset import HarmonicOscillatorDataset, create_splits
from .perturbations import apply_perturbation, get_perturbation_grid, PerturbationType
from .dynamics_metrics import compute_all_dynamics_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (accuracy, roc_auc)."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return float(acc), float(auc)


def train_model(
    model: nn.Module,
    train_ds: HarmonicOscillatorDataset,
    val_ds: HarmonicOscillatorDataset,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train with early stopping on val loss.

    Returns dict with best_val_auc, best_epoch, train_history, etc.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if seed is not None:
        torch.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_auc = evaluate(model, val_loader, device)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def evaluate_perturbed(
    model: nn.Module,
    test_ds: HarmonicOscillatorDataset,
    pert_type: PerturbationType,
    alpha_grid: Optional[np.ndarray] = None,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate on perturbed test set across alpha grid.

    Returns: {alpha_values, accuracies, aucs, AUPC, MaxRD, RD_curve}
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if alpha_grid is None:
        alpha_grid = get_perturbation_grid(pert_type)

    accs, aucs = [], []
    for alpha in alpha_grid:
        all_preds, all_probs, all_labels = [], [], []
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x_np = x.numpy()
                sample_seed = seed + i * 10000 + int(alpha * 1000)
                x_pert = np.stack([
                    apply_perturbation(x_np[b], pert_type, float(alpha), seed=sample_seed + b)
                    for b in range(x_np.shape[0])
                ])
                x_pert = torch.from_numpy(x_pert.astype(np.float32))
                x_pert = x_pert.to(device)
                logits = model(x_pert)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.numpy())
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
        accs.append(acc)
        aucs.append(auc)

    score_clean = aucs[0] if alpha_grid[0] == 0 else aucs[0]
    eps = 1e-6
    rd_curve = [(score_clean - s) / max(eps, score_clean) for s in aucs]
    max_rd = max(rd_curve) if rd_curve else 0.0
    aupc = np.mean(aucs)

    return {
        "alpha_values": alpha_grid.tolist(),
        "accuracies": accs,
        "aucs": aucs,
        "AUPC": float(aupc),
        "MaxRD": float(max_rd),
        "RD_curve": rd_curve,
        "score_clean": score_clean,
    }


def evaluate_perturbed_multi(
    model: nn.Module,
    test_ds: HarmonicOscillatorDataset,
    pert_types: List[PerturbationType],
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate on multiple perturbation types. Returns combined AUPC_total (mean over types)
    and MaxRD_total (max over types).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aupcs, max_rds = [], []
    per_type = {}
    for pt in pert_types:
        r = evaluate_perturbed(model, test_ds, pt, batch_size=batch_size, device=device, seed=seed)
        aupcs.append(r["AUPC"])
        max_rds.append(r["MaxRD"])
        per_type[pt] = r
    return {
        "AUPC_total": float(np.mean(aupcs)),
        "MaxRD_total": float(np.max(max_rds)),
        "per_type": per_type,
    }


def evaluate_dynamics(
    model: nn.Module,
    test_ds: HarmonicOscillatorDataset,
    n_samples: int = 512,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute M1, M2, M3 on a subset of test data."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    indices = np.random.default_rng(seed).choice(len(test_ds), size=min(n_samples, len(test_ds)), replace=False)
    samples = [test_ds[i] for i in indices]
    x = torch.stack([s[0] for s in samples])
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    x = x.to(device)
    return compute_all_dynamics_metrics(model, x, seed=seed, device=device)
