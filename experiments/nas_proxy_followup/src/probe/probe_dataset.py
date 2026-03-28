"""Probe subset indices for BNCI2014_001 cross-session protocol.

Motor imagery uses **two classes** (left_hand, right_hand) only, same as
``config.get_paradigm(..., dataset="BNCI2014_001")`` and
``evaluation.unified_experiment_runner`` — not the dataset's four imagery types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import get_paradigm
from moabb.datasets import BNCI2014_001

from .eeg_layout import infer_eeg_n_channels

_BNCI_MI_TWO_CLASSES = frozenset({"left_hand", "right_hand"})


def encode_moabb_labels(y: np.ndarray) -> np.ndarray:
    """
    MOABB may return string class names or integer codes. Map to contiguous int64 {0..K-1}.
    """
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.floating):
        yi = y.astype(np.int64, copy=False)
        yi = yi - int(yi.min())
        return yi
    from sklearn.preprocessing import LabelEncoder

    return LabelEncoder().fit_transform(y).astype(np.int64)


def _filter_bnci_two_class_mi(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Keep only left_hand / right_hand trials (defensive; paradigm should already restrict)."""
    y = np.asarray(y)
    if y.dtype.kind in ("U", "O", "S"):
        mask = np.array([str(lab) in _BNCI_MI_TWO_CLASSES for lab in y], dtype=bool)
        if not mask.all():
            X = X[mask]
            y = y[mask]
            metadata = metadata.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    elif np.unique(y).size > 2:
        raise ValueError(
            "BNCI2014_001 probe path expects 2 MI classes (left_hand/right_hand via "
            "config.get_paradigm); got more than 2 numeric label codes."
        )
    return X, y, metadata


def load_bnci_cross_session_arrays(
    subject: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Returns X, y (encoded), metadata (DataFrame-like), session array."""
    dataset = BNCI2014_001()
    paradigm = get_paradigm(resample=None, dataset="BNCI2014_001")
    X, y, metadata = paradigm.get_data(dataset, subjects=[subject], return_epochs=False)
    if hasattr(X, "get_data"):
        X = X.get_data()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    X, y, metadata = _filter_bnci_two_class_mi(X, y, metadata)
    if "session" not in metadata.columns:
        raise ValueError("Metadata must contain 'session' for cross_session protocol.")
    sess = metadata["session"].values
    return X, y, metadata, sess


def build_probe_subset_indices(
    dataset_name: str,
    protocol: str,
    max_examples: int,
    stratified: bool,
    seed: int,
    subject: int = 1,
) -> List[int]:
    """
    Validation-session indices for probe (default: session '1test').
    """
    if dataset_name != "BNCI2014_001":
        raise NotImplementedError(f"Only BNCI2014_001 supported; got {dataset_name}")
    X, y, _metadata, sess = load_bnci_cross_session_arrays(subject=subject)
    valid_mask = sess == "1test"
    idx_all = np.where(valid_mask)[0]
    if len(idx_all) == 0:
        raise RuntimeError("No validation session rows (expected session '1test').")
    y_valid = y[idx_all]
    rng = np.random.default_rng(seed)
    n = min(max_examples, len(idx_all))
    if stratified and len(np.unique(y_valid)) > 1:
        # stratified subsample without sklearn dependency on split sizes
        chosen = []
        classes = np.unique(y_valid)
        per = max(1, n // len(classes))
        for c in classes:
            ic = idx_all[y_valid == c]
            rng.shuffle(ic)
            chosen.extend(ic[:per].tolist())
        chosen = chosen[:n]
        if len(chosen) < n:
            rest = [i for i in idx_all.tolist() if i not in chosen]
            rng.shuffle(rest)
            chosen.extend(rest[: n - len(chosen)])
    else:
        perm = rng.permutation(len(idx_all))[:n]
        chosen = idx_all[perm].tolist()
    return [int(i) for i in chosen]


def save_probe_indices(path: Path, indices: List[int], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"indices": indices, **meta}, f, indent=2)
