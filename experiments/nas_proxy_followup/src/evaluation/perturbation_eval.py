"""Local Gaussian perturbation evaluation on a trained model."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC for binary labels {0,1}."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = np.arange(1, len(y) + 1)
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def evaluate_gaussian_perturbation(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    levels: List[float],
    perturbation_seed: int,
    device: torch.device,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """
    levels: noise std multipliers (0 = clean). RD = (A_clean - A_pert) / A_clean.
    """
    if X.ndim == 3 and X.shape[1] < X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
    model.eval()
    rng = np.random.default_rng(perturbation_seed)
    rows: List[Dict[str, Any]] = []

    def run_auc(noise_scale: float) -> float:
        Xp = X.copy()
        if noise_scale > 0:
            Xp = Xp + rng.standard_normal(Xp.shape).astype(np.float32) * noise_scale
        Xt = torch.from_numpy(np.ascontiguousarray(Xp)).float()
        yt = torch.from_numpy(np.ascontiguousarray(y)).long()
        ds = TensorDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probs = []
        ys = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.append(p)
                ys.append(yb.numpy())
        scores = np.concatenate(probs)
        y_true = np.concatenate(ys)
        return _auc_binary(y_true, scores)

    a_clean = run_auc(0.0)
    for lev in levels:
        if lev == 0.0:
            auc = a_clean
        else:
            auc = run_auc(lev)
        rd = (a_clean - auc) / max(a_clean, 1e-12) if np.isfinite(a_clean) else float("nan")
        rows.append(
            {
                "level": lev,
                "test_auc": auc,
                "relative_degradation": rd,
            }
        )
    return rows
