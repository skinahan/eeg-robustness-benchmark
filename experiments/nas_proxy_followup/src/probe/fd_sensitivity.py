"""Finite-difference output sensitivity (input noise)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def compute_fd_sensitivity(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    noise_std: float,
    num_repeats: int = 1,
) -> Dict[str, Any]:
    """
    S_fd = mean_n ||f(x+ε) - f(x)||_2 / ||ε||_2
    batch_x: [B, T, C]
    """
    model.eval()
    device = batch_x.device
    ratios = []
    with torch.no_grad():
        base = model(batch_x)
        for _ in range(num_repeats):
            eps = torch.randn_like(batch_x) * noise_std
            x2 = batch_x + eps
            out = model(x2)
            num = torch.norm((out - base).reshape(base.shape[0], -1), dim=1)
            den = torch.norm(eps.reshape(eps.shape[0], -1), dim=1) + 1e-12
            ratios.append((num / den).mean().item())
    return {"s_fd_mean": float(sum(ratios) / len(ratios))}
