"""Lightweight tests (no Torch) for EEG trial layout helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_NPF = _REPO / "experiments" / "nas_proxy_followup"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_NPF) not in sys.path:
    sys.path.insert(0, str(_NPF))

from src.probe.eeg_layout import infer_eeg_n_channels


def test_infer_eeg_n_channels_moabb_layout():
    N, C, T = 10, 22, 1001
    X = np.zeros((N, C, T), dtype=np.float32)
    assert infer_eeg_n_channels(X) == C


def test_infer_eeg_n_channels_time_major():
    N, T, C = 5, 500, 22
    X = np.zeros((N, T, C), dtype=np.float32)
    assert infer_eeg_n_channels(X) == C
