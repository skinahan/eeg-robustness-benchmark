"""
Synthetic dataset: Noisy Harmonic Oscillator Binary Classification.

Task: binary classification of frequency regime (low vs high) given a length-T sequence.
Per PAPER 3 spec Section B.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def generate_clean_signal(
    T: int,
    dt: float,
    omega: float,
    A: float,
    phi: float,
    use_amplitude_modulation: bool = False,
    a_m: float = 0.1,
    omega_m: Optional[float] = None,
    phi_m: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate clean harmonic oscillator signal.

    x_clean(t_i) = A * sin(ω * t_i + phi)
    Optional: m(t) = 1 + a_m * sin(ω_m * t + phi_m), x <- m(t) * x

    Args:
        T: Sequence length.
        dt: Timestep.
        omega: Angular frequency (radians/step).
        A: Amplitude.
        phi: Phase.
        use_amplitude_modulation: Whether to apply mild amplitude modulation.
        a_m: Amplitude modulation strength.
        omega_m: Modulation frequency (sampled if None).
        phi_m: Modulation phase (sampled if None).
        rng: RNG for optional sampling.

    Returns:
        Array of shape (T,) with clean signal.
    """
    t = np.arange(T, dtype=np.float64) * dt
    x = A * np.sin(omega * t + phi)

    if use_amplitude_modulation:
        rng = rng or np.random.default_rng()
        omega_m = omega_m if omega_m is not None else float(rng.uniform(0.01, 0.05))
        phi_m = phi_m if phi_m is not None else float(rng.uniform(0, 2 * np.pi))
        m = 1.0 + a_m * np.sin(omega_m * t + phi_m)
        x = m * x

    return x.astype(np.float32)


def generate_sample(
    T: int,
    dt: float,
    y: int,
    omega_L: Tuple[float, float] = (0.05, 0.15),
    omega_H: Tuple[float, float] = (0.20, 0.35),
    A_range: Tuple[float, float] = (0.5, 1.5),
    use_amplitude_modulation: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """
    Generate a single sample (x, y).

    Args:
        T: Sequence length.
        dt: Timestep.
        y: Class label (0=low freq, 1=high freq).
        omega_L: (min, max) for low-frequency class.
        omega_H: (min, max) for high-frequency class.
        A_range: (min, max) for amplitude.
        use_amplitude_modulation: Whether to add amplitude modulation.
        rng: RNG for reproducibility.

    Returns:
        (x_clean, y) where x_clean has shape (T, 1).
    """
    rng = rng or np.random.default_rng()
    A = float(rng.uniform(A_range[0], A_range[1]))
    phi = float(rng.uniform(0, 2 * np.pi))

    if y == 0:
        omega = float(rng.uniform(omega_L[0], omega_L[1]))
    else:
        omega = float(rng.uniform(omega_H[0], omega_H[1]))

    x = generate_clean_signal(
        T=T,
        dt=dt,
        omega=omega,
        A=A,
        phi=phi,
        use_amplitude_modulation=use_amplitude_modulation,
        rng=rng,
    )
    return x[:, np.newaxis].astype(np.float32), int(y)


class HarmonicOscillatorDataset(Dataset):
    """
    PyTorch Dataset for harmonic oscillator binary classification.

    Returns (x, y) with x shape [T, C] (C=1 by default).
    """

    def __init__(
        self,
        n_samples: int,
        T: int = 256,
        dt: float = 1.0,
        C: int = 1,
        omega_L: Tuple[float, float] = (0.05, 0.15),
        omega_H: Tuple[float, float] = (0.20, 0.35),
        A_range: Tuple[float, float] = (0.5, 1.5),
        use_amplitude_modulation: bool = False,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.T = T
        self.dt = dt
        self.C = C
        self.omega_L = omega_L
        self.omega_H = omega_H
        self.A_range = A_range
        self.use_amplitude_modulation = use_amplitude_modulation
        self.seed = seed

        rng = np.random.default_rng(seed)
        # Stratified: half class 0, half class 1
        n_per_class = n_samples // 2
        remainder = n_samples - 2 * n_per_class

        self.data: list[Tuple[np.ndarray, int]] = []
        for _ in range(n_per_class):
            self.data.append(generate_sample(
                T=T, dt=dt, y=0,
                omega_L=omega_L, omega_H=omega_H, A_range=A_range,
                use_amplitude_modulation=use_amplitude_modulation, rng=rng,
            ))
        for _ in range(n_per_class):
            self.data.append(generate_sample(
                T=T, dt=dt, y=1,
                omega_L=omega_L, omega_H=omega_H, A_range=A_range,
                use_amplitude_modulation=use_amplitude_modulation, rng=rng,
            ))
        for _ in range(remainder):
            y = int(rng.integers(0, 2))
            self.data.append(generate_sample(
                T=T, dt=dt, y=y,
                omega_L=omega_L, omega_H=omega_H, A_range=A_range,
                use_amplitude_modulation=use_amplitude_modulation, rng=rng,
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def create_splits(
    n_train: int = 20_000,
    n_val: int = 2_000,
    n_test: int = 5_000,
    T: int = 256,
    dt: float = 1.0,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[HarmonicOscillatorDataset, HarmonicOscillatorDataset, HarmonicOscillatorDataset]:
    """
    Create train/val/test splits with fixed seeds for reproducibility.

    Uses sequential seeds (seed, seed+1, seed+2) so splits are deterministic.
    """
    train_ds = HarmonicOscillatorDataset(n_samples=n_train, T=T, dt=dt, seed=seed, **dataset_kwargs)
    val_ds = HarmonicOscillatorDataset(n_samples=n_val, T=T, dt=dt, seed=seed + 1, **dataset_kwargs)
    test_ds = HarmonicOscillatorDataset(n_samples=n_test, T=T, dt=dt, seed=seed + 2, **dataset_kwargs)
    return train_ds, val_ds, test_ds
