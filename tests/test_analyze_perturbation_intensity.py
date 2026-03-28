"""Tests for scripts/analyze_perturbation_intensity.py (dropout semantics, Gaussian edge cases)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_analysis_module():
    path = _REPO / "scripts" / "analyze_perturbation_intensity.py"
    name = "analyze_perturbation_intensity"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_analysis_module()
dropout_channel_drop_count = mod.dropout_channel_drop_count
run_one = mod.run_one


def test_dropout_n_drop_matches_noise_py():
    """Same formula as EEGNoiseAugmentor._apply_channel_dropout."""
    n_ch = 22
    assert dropout_channel_drop_count(0.0, n_ch) == (0, False)
    # 1% of 22 -> 0 raw -> clamp to 1
    n, clamp = dropout_channel_drop_count(1.0, n_ch)
    assert n == 1 and clamp is True
    # 5% -> 1 channel
    n, clamp = dropout_channel_drop_count(5.0, n_ch)
    assert n == 1 and clamp is False
    # 100%
    n, clamp = dropout_channel_drop_count(100.0, n_ch)
    assert n == n_ch and clamp is False


def test_gaussian_intensity_zero_no_delta():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 4, 16)).astype(np.float64)
    template = _REPO / "notebooks" / "eog_mixing_results" / "generic_eog_mixing_template.npz"
    if not template.exists():
        pytest.skip("EOG template npz not present")
    rec = run_one(
        X,
        "gaussian",
        0.0,
        42,
        eog_template_path=str(template),
        artifact_scale_factor=10000.0,
    )
    assert rec.p_noise_or_artifact_mean_sq == 0.0
    assert rec.snr_db is None


def test_gaussian_higher_intensity_higher_noise_power():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((4, 8, 32)).astype(np.float64) * 1e-6
    template = _REPO / "notebooks" / "eog_mixing_results" / "generic_eog_mixing_template.npz"
    if not template.exists():
        pytest.skip("EOG template npz not present")
    low = run_one(X, "gaussian", 10.0, 99, str(template), 10000.0)
    high = run_one(X, "gaussian", 50.0, 99, str(template), 10000.0)
    assert low.p_noise_or_artifact_mean_sq is not None
    assert high.p_noise_or_artifact_mean_sq is not None
    assert high.p_noise_or_artifact_mean_sq > low.p_noise_or_artifact_mean_sq
