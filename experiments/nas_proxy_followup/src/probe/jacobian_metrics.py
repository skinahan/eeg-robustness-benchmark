"""Jacobian spectral norms for one-step CfC recurrence ∂h_{t+1}/∂h_t."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.autograd.functional import jacobian


def _wired_cell_step(model: torch.nn.Module, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Single time step: x_t [1, D_in], h [1, state]. Returns h_new [1, state]."""
    cell = model.cfc.rnn_cell
    ts = torch.ones(1, device=x_t.device, dtype=x_t.dtype)
    _, h_new = cell.forward(x_t, h, ts)
    return h_new


def estimate_jacobian_spectral_norm(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    time_step_indices: Optional[List[int]] = None,
    method: str = "power_iteration",
    num_power_iters: int = 20,
) -> Dict[str, Any]:
    """
    batch_x: [B, T, D_in] after encoder would be applied inside model — we use model.encoder first.
    Computes mean log σ_max(J) over selected batch items and time steps.
    """
    device = batch_x.device
    model.eval()
    enc = model.encoder(batch_x)
    B, T, _ = enc.shape
    if time_step_indices is None:
        time_step_indices = [0, T // 2, T - 1]
        time_step_indices = [t for t in time_step_indices if 0 <= t < T]
    sigmas = []
    sigmas_max = []
    expansive = []

    for b in range(min(B, 4)):
        for t in time_step_indices:
            x_t = enc[b : b + 1, t, :]
            h = torch.zeros(1, model.cfc.state_size, device=device, requires_grad=True)
            h0 = torch.zeros(1, model.cfc.state_size, device=device)

            def f(h_flat: torch.Tensor) -> torch.Tensor:
                hh = h_flat.view(1, -1)
                return _wired_cell_step(model, x_t, hh).view(-1)

            J = jacobian(f, h0.view(-1))
            s = torch.linalg.svdvals(J)
            smax = float(s[0].item()) if s.numel() else 0.0
            sigmas.append(smax)
            sigmas_max.append(smax)
            expansive.append(1.0 if smax > 1.0 else 0.0)

    n = max(len(sigmas), 1)
    logs = [torch.log(torch.tensor(s + 1e-12)) for s in sigmas]
    return {
        "s_jac_mean_log_sigma": float(torch.stack(logs).mean().item()),
        "s_jac_max": max(sigmas_max) if sigmas_max else 0.0,
        "s_exp_frac": float(sum(expansive) / n),
        "n_terms": n,
    }


def aggregate_jacobian_metrics(per_batch_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_batch_metrics:
        return {}
    keys = [k for k in per_batch_metrics[0] if isinstance(per_batch_metrics[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [float(m[k]) for m in per_batch_metrics if k in m]
        if vals:
            out[f"{k}_mean"] = float(sum(vals) / len(vals))
    return out
