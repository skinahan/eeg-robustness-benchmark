"""EEG trial layout helpers (numpy only; no MOABB import)."""

from __future__ import annotations

import numpy as np


def infer_eeg_n_channels(X: np.ndarray) -> int:
    """
    MOABB MotorImagery trials are typically (N, n_channels, n_times) with n_times ≫ n_channels.
    Do not use shape[-1] for channel count (that is usually time length).
    """
    if X.ndim != 3:
        return 1
    if X.shape[1] < X.shape[2]:
        return int(X.shape[1])
    return int(X.shape[2])
