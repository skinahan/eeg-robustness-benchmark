"""
Perturbation suite for PAPER 3 robustness evaluation.

P1: Additive Gaussian noise (AWGN)
P2: Impulse / spike noise
P3: Low-frequency drift (baseline wander)

Per PAPER 3 spec Section C.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

PerturbationType = Literal["awgn", "impulse", "drift"]

# Intensity grids per spec (TEST 6b stress amplification)
# Extended AWGN to 1.0, 1.2, 1.5 so AUPC degrades (SNR < 1 at high stress)
AWGN_SIGMA_GRID = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
IMPULSE_LAMBDA_GRID = np.array([0, 2, 4, 8, 16])
DRIFT_B_GRID = np.array([0.0, 0.2, 0.4, 0.6, 0.8])


def _get_scale(x: np.ndarray) -> float:
    """Scale perturbations relative to signal std."""
    return float(np.std(x) + 1e-8)


def apply_awgn(
    x: np.ndarray,
    alpha: float,
    seed: Optional[int] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """
    P1: Additive white Gaussian noise.

    x'(t) = x(t) + η, η ~ N(0, σ²)
    σ = alpha * std(x) where alpha maps to grid [0, 0.1, 0.2, 0.4, 0.6, 0.8]

    Args:
        x: Clean signal, shape (T,) or (T, C).
        alpha: Intensity in [0, 1]; maps to sigma = alpha * max_sigma * std(x).
        seed: RNG seed for reproducibility.
        scale: Override scale (default: std(x)).

    Returns:
        Perturbed signal same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = scale if scale is not None else _get_scale(x)
    sigma = alpha * float(AWGN_SIGMA_GRID[-1]) * scale  # max alpha=1 -> sigma=max(AWGN_SIGMA_GRID)*std
    rng = np.random.default_rng(seed)
    eta = rng.normal(0, sigma, size=x.shape)
    return (x + eta).astype(np.float32)


def apply_impulse(
    x: np.ndarray,
    alpha: float,
    seed: Optional[int] = None,
    T: Optional[int] = None,
) -> np.ndarray:
    """
    P2: Impulse / spike noise.

    K_imp ~ Poisson(λ), pick K indices, magnitudes u_k ~ N(0, s²).
    λ = alpha * 16 (max 16 spikes per sequence when alpha=1).
    Spike scale s = 1.5 * std(x).

    Args:
        x: Clean signal, shape (T,) or (T, C).
        alpha: Intensity in [0, 1]; maps to lambda = alpha * 8.
        seed: RNG seed for reproducibility.
        T: Sequence length (inferred from x if None).

    Returns:
        Perturbed signal same shape as x.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    T = T or x.shape[0]
    lambda_val = alpha * 16.0
    rng = np.random.default_rng(seed)
    K = int(rng.poisson(lambda_val))
    if K <= 0:
        return x.astype(np.float32)

    scale = _get_scale(x)
    spike_scale = 1.5 * scale
    indices = rng.integers(0, T, size=min(K, T))
    magnitudes = rng.normal(0, spike_scale, size=len(indices))

    if x.ndim == 1:
        for i, idx in enumerate(indices):
            x[idx] += magnitudes[i]
    else:
        for i, idx in enumerate(indices):
            x[idx, :] += magnitudes[i]

    return x.astype(np.float32)


def apply_drift(
    x: np.ndarray,
    alpha: float,
    seed: Optional[int] = None,
    scale: Optional[float] = None,
    T: Optional[int] = None,
    dt: float = 1.0,
) -> np.ndarray:
    """
    P3: Low-frequency drift (baseline wander).

    d(t) = B * sin(ω_d * t + φ_d), ω_d ∈ [0.002, 0.01].
    B = alpha * 0.8 * std(x) (max when alpha=1).

    Args:
        x: Clean signal, shape (T,) or (T, C).
        alpha: Intensity in [0, 1]; maps to B = alpha * 0.5 * std(x).
        seed: RNG seed for reproducibility.
        scale: Override scale (default: std(x)).
        T: Sequence length (inferred from x if None).
        dt: Timestep for time grid.

    Returns:
        Perturbed signal same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    T = T or x.shape[0]
    scale = scale if scale is not None else _get_scale(x)
    B = alpha * 0.8 * scale
    rng = np.random.default_rng(seed)
    omega_d = float(rng.uniform(0.002, 0.01))
    phi_d = float(rng.uniform(0, 2 * np.pi))
    t = np.arange(T, dtype=np.float64) * dt
    d = B * np.sin(omega_d * t + phi_d)

    if x.ndim == 2:
        d = d[:, np.newaxis]
    return (x + d).astype(np.float32)


def apply_perturbation(
    x: Union[np.ndarray, "torch.Tensor"],
    pert_type: PerturbationType,
    alpha: float,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply perturbation by type.

    Args:
        x: Clean signal.
        pert_type: "awgn", "impulse", or "drift".
        alpha: Intensity in [0, 1].
        seed: RNG seed for identical realizations across models.
        **kwargs: Passed to the specific perturbation function.

    Returns:
        Perturbed signal as numpy array.
    """
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass

    x = np.asarray(x, dtype=np.float64)

    if pert_type == "awgn":
        return apply_awgn(x, alpha, seed=seed, **kwargs)
    elif pert_type == "impulse":
        return apply_impulse(x, alpha, seed=seed, **kwargs)
    elif pert_type == "drift":
        return apply_drift(x, alpha, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")


def get_perturbation_grid(pert_type: PerturbationType) -> np.ndarray:
    """Return the alpha grid for a perturbation type (normalized to [0,1] for max intensity)."""
    if pert_type == "awgn":
        return AWGN_SIGMA_GRID / AWGN_SIGMA_GRID[-1]  # normalize so max=1
    elif pert_type == "impulse":
        return IMPULSE_LAMBDA_GRID / IMPULSE_LAMBDA_GRID[-1]
    elif pert_type == "drift":
        return DRIFT_B_GRID / DRIFT_B_GRID[-1]
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")


