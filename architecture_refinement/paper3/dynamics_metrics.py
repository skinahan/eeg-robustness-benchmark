"""
Dynamics-level robustness metrics for PAPER 3.

M1: Empirical input sensitivity (approx Lipschitz)
M2: Hidden-state variance under stochastic noise
M3: Empirical contraction/divergence rate (Lyapunov-style)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Callable, Optional, Union

EPS0 = 1e-8


def compute_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    epsilon: float,
    n_directions: int = 1,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    M1: Empirical input sensitivity S(x; ε) = ||h_T(x_ε) - h_T(x)|| / ||x_ε - x||.

    x_ε = x + ε * δ/||δ|| with δ ~ N(0, I).
    Aggregate: mean over samples.

    Args:
        model: Model with forward(x, return_states=True) -> (logits, states).
        x: Input batch [B, T, C].
        epsilon: Perturbation scale (relative to ||x||_2 or fixed).
        n_directions: Number of random directions per sample (default 1).
        seed: RNG seed.
        device: Device for computation.

    Returns:
        Mean sensitivity over batch.
    """
    device = device or x.device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _, states_clean = model(x, return_states=True)
        h_T_clean = states_clean[:, -1, :]

        sensitivities = []
        rng = np.random.default_rng(seed)
        for _ in range(n_directions):
            delta = torch.from_numpy(rng.standard_normal(x.shape).astype(np.float32)).to(device)
            delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
            delta_norm = delta_norm.clamp(min=1e-8)
            delta = delta / delta_norm.view(-1, 1, 1)
            x_eps = x + epsilon * delta

            _, states_pert = model(x_eps, return_states=True)
            h_T_pert = states_pert[:, -1, :]

            diff_h = h_T_pert - h_T_clean
            diff_x = x_eps - x
            norm_h = torch.norm(diff_h, dim=1)
            norm_x = torch.norm(diff_x.view(diff_x.size(0), -1), dim=1).clamp(min=EPS0)
            s = (norm_h / norm_x).cpu().numpy()
            sensitivities.extend(s.tolist())

        return float(np.mean(sensitivities))


def compute_state_variance(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: float,
    R: int = 10,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    M2: Hidden-state variance under stochastic noise.

    For each x, generate R realizations x^(r) via AWGN with σ.
    V_t(x) = Var_r[h_t(x^(r))], then mean over t and x.

    Args:
        model: Model with forward(x, return_states=True).
        x: Input batch [B, T, C].
        sigma: AWGN std.
        R: Number of noise realizations.
        seed: RNG seed.
        device: Device.

    Returns:
        StateVar scalar.
    """
    device = device or x.device
    model = model.to(device)
    model.eval()
    B, T, C = x.shape
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        states_list = []
        for r in range(R):
            noise = torch.from_numpy(rng.normal(0, sigma, x.shape).astype(np.float32)).to(device)
            x_r = x + noise
            _, states = model(x_r, return_states=True)
            states_list.append(states)
        stack = torch.stack(states_list, dim=0)
        var_t = torch.var(stack, dim=0)
        mean_over_dims = var_t.mean(dim=2)
        mean_over_t = mean_over_dims.mean(dim=1)
        return float(mean_over_t.mean().cpu().numpy())


def compute_lambda(
    model: torch.nn.Module,
    x: torch.Tensor,
    epsilon: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    M3: Empirical contraction/divergence rate.

    d_t = ||h_t(x̃) - h_t(x)||_2, x̃ = x + εδ.
    λ(x) = (1/(T-1)) * Σ_t log((d_t + ε0) / (d_{t-1} + ε0))
    Lambda = mean_x λ(x). More negative => more stable (contracting).

    Args:
        model: Model with forward(x, return_states=True).
        x: Input batch [B, T, C].
        epsilon: Perturbation scale.
        seed: RNG seed.
        device: Device.

    Returns:
        Lambda scalar.
    """
    device = device or x.device
    model = model.to(device)
    model.eval()
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        delta = torch.from_numpy(rng.standard_normal(x.shape).astype(np.float32)).to(device)
        delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True).clamp(min=1e-8)
        delta = delta / delta_norm.view(-1, 1, 1)
        x_tilde = x + epsilon * delta

        _, states = model(x, return_states=True)
        _, states_tilde = model(x_tilde, return_states=True)

        d_t = torch.norm(states_tilde - states, dim=2)

        log_ratio = torch.log((d_t[:, 1:] + EPS0) / (d_t[:, :-1] + EPS0))
        lambda_x = log_ratio.mean(dim=1)
        return float(lambda_x.mean().cpu().numpy())


def _x_scale(x: torch.Tensor) -> float:
    """Mean per-sample L2 norm: ε_small = 1e-3*||x||2, ε_med = 1e-1*||x||2 (100x spread for nonlinear regime)."""
    return float(torch.norm(x.view(x.size(0), -1), dim=1).mean().item()) + 1e-8


def compute_all_dynamics_metrics(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma_statevar: float = 0.2,
    R: int = 10,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compute M1, M2, M3 on a batch.
    Reports Sens at ε_small=1e-3*||x|| (linear regime) and ε_med=1e-1*||x|| (nonlinear).
    100x spread probes nonlinear amplification; at 10% of signal norm, saturating RNNs may show Sens(ε_med) != Sens(ε_small).
    Lambda computed at ε_med.

    Returns:
        {"sensitivity": float, "sensitivity_small": float, "sensitivity_med": float,
         "state_var": float, "lambda": float, "lambda_small": float, "lambda_med": float}
    """
    x_scale = _x_scale(x)
    eps_small = 1e-3 * x_scale
    eps_med = 1e-1 * x_scale

    sensitivity_small = compute_sensitivity(model, x, eps_small, seed=seed, device=device)
    sensitivity_med = compute_sensitivity(model, x, eps_med, seed=seed, device=device)
    state_var = compute_state_variance(model, x, sigma_statevar, R=R, seed=seed, device=device)
    lam_small = compute_lambda(model, x, eps_small, seed=seed, device=device)
    lam_med = compute_lambda(model, x, eps_med, seed=seed, device=device)

    return {
        "sensitivity": sensitivity_med,
        "sensitivity_small": sensitivity_small,
        "sensitivity_med": sensitivity_med,
        "state_var": state_var,
        "lambda": lam_med,
        "lambda_small": lam_small,
        "lambda_med": lam_med,
    }
