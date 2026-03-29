"""
Training-side sliding-window helpers for MotorImagery (Lee2019_MI, Shin2017A, …).

Use only after train/validation trial indices are fixed. Validation uses a single
fixed crop per trial (first ``win_samples``); training uses sliding windows
(1 s window, 0.5 s stride in samples at ``sfreq_hz``).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def lee2019_mi_sliding_params(sfreq_hz: float) -> Tuple[int, int, int]:
    """Return (win_samples, stride_samples, crop_start) for 1s / 0.5s / first-crop protocol."""
    win = int(round(1.0 * sfreq_hz))
    stride = int(round(0.5 * sfreq_hz))
    return win, stride, 0


def shin2017a_sliding_params(sfreq_hz: float) -> Tuple[int, int, int]:
    """Same protocol as Lee2019_MI (1 s / 0.5 s stride); epoch must be at least 1 s long."""
    return lee2019_mi_sliding_params(sfreq_hz)


def fixed_crop_batch(
    X: np.ndarray,
    y: np.ndarray,
    start: int,
    win_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One temporal crop per trial: X[:, :, start:start+win_samples]."""
    end = start + win_samples
    if X.shape[2] < end:
        raise ValueError(
            f"Epoch too short for crop: n_times={X.shape[2]}, need >= {end} (start={start}, win={win_samples})"
        )
    return X[:, :, start:end].astype(np.float32), np.asarray(y)


def sliding_window_tensor(
    X: np.ndarray,
    y: np.ndarray,
    win_samples: int,
    stride_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand each trial into sliding windows; labels repeat per window."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(len(X)):
        trial = X[i]
        nt = trial.shape[1]
        if nt < win_samples:
            raise ValueError(f"Trial {i}: n_times {nt} < win_samples {win_samples}")
        s = 0
        while s + win_samples <= nt:
            xs.append(trial[:, s : s + win_samples])
            ys.append(int(y[i]))
            s += stride_samples
    if not xs:
        raise ValueError("No windows produced; check win_samples, stride_samples, epoch length")
    return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.int64)


def expand_metadata_for_sliding_windows(
    metadata: pd.DataFrame,
    X_train: np.ndarray,
    win_samples: int,
    stride_samples: int,
) -> pd.DataFrame:
    """Repeat metadata rows to align with sliding_window_tensor output order."""
    rows = []
    for i in range(len(X_train)):
        nt = int(X_train.shape[2])
        s = 0
        while s + win_samples <= nt:
            rows.append(metadata.iloc[i])
            s += stride_samples
    return pd.DataFrame(rows).reset_index(drop=True)
