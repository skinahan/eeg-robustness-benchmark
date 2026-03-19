#!/usr/bin/env python3
"""
Generate 300 dpi EEG trace panels illustrating perturbation effects.

Creates small figure panels showing clean vs perturbed EEG traces for:
- Gaussian noise
- EOG artifacts
- Channel dropout

Uses a subset of channels to clearly illustrate each perturbation type.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from moabb.datasets import BNCI2014_001
from mne.epochs import BaseEpochs

from config import get_paradigm, get_dataset_sampling_rate
from augmentation.noise import EEGNoiseAugmentor, inject_realistic_eog_artifacts_with_coverage
import mne


def _ensure_eog_template_npz():
    """Create npz from json if npz is missing (for EOG perturbation)."""
    npz_path = project_root / "notebooks" / "eog_mixing_results" / "generic_eog_mixing_template.npz"
    json_path = project_root / "notebooks" / "eog_mixing_results" / "generic_eog_mixing_template.json"
    if npz_path.exists():
        return str(npz_path)
    if json_path.exists():
        import json
        with open(json_path) as f:
            data = json.load(f)
        mm = data.get("mixing_matrix_statistics", {}).get("mean")
        if mm is not None:
            B = np.array(mm)
            reg = data.get("regressor_statistics", {})
            veog = reg.get("veog", {}).get("mean_std", 50e-6)
            heog = reg.get("heog", {}).get("mean_std", 30e-6)
            rms = data.get("target_rms_statistics", {}).get("mean_median", 20e-6)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_path, mixing_matrix=B, veog_std=veog, heog_std=heog, target_rms_median=rms)
            return str(npz_path)
    return None


def load_sample_epoch(dataset_name="BNCI2014_001", subject_id=1, epoch_idx=0):
    """Load one epoch as (n_channels, n_times) and channel names."""
    dataset = BNCI2014_001()
    paradigm = get_paradigm(resample=None, dataset=dataset_name)
    X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id], return_epochs=False)
    if isinstance(X, BaseEpochs):
        X = X.get_data()
    epoch = X[epoch_idx]  # (n_channels, n_times)
    # Paradigm returns 22 EEG channels; use matching names (BNCI2014_001 order)
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]
    assert epoch.shape[0] == len(ch_names), f"Channel count mismatch: {epoch.shape[0]} vs {len(ch_names)}"
    return epoch, ch_names


def select_channels_for_display(ch_names, n_display=4):
    """Pick frontal channels where EOG artifacts are strongest (Fz, FCz, FC3, FC4)."""
    preferred = ["Fz", "FCz", "FC3", "FC4", "FC1", "FC2", "Cz", "C3", "C4"]
    indices = []
    for p in preferred:
        if p in ch_names and len(indices) < n_display:
            indices.append(ch_names.index(p))
    # Fill with first available if needed
    for i, ch in enumerate(ch_names):
        if len(indices) >= n_display:
            break
        if i not in indices:
            indices.append(i)
    indices = indices[:n_display]
    return indices, [ch_names[i] for i in indices]


def plot_trace_panel(ax, data, ch_labels, time_axis, y_offset=50, color="k", alpha=1.0):
    """Plot stacked EEG traces on one axis."""
    for i, (ch_data, label) in enumerate(zip(data, ch_labels)):
        offset = i * y_offset
        ax.plot(time_axis, ch_data + offset, color=color, alpha=alpha, linewidth=0.8)
        ax.text(-0.02, offset, label, transform=ax.get_yaxis_transform(), fontsize=8,
                va="center", ha="right")
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def main():
    output_dir = project_root / "outputs" / "perturbation_trace_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "BNCI2014_001"
    sfreq = get_dataset_sampling_rate(dataset_name)
    seed = 42
    intensity = 25.0  # Moderate intensity for clear visual effect

    print("Loading sample epoch...")
    epoch, ch_names = load_sample_epoch(dataset_name=dataset_name)
    n_chans, n_times = epoch.shape
    ch_indices, ch_labels = select_channels_for_display(ch_names, n_display=4)
    time_axis = np.arange(n_times) / sfreq

    clean_subset = epoch[ch_indices, :]

    # Apply perturbations (need batch shape for augmentor)
    X_batch = epoch[np.newaxis, :, :]  # (1, n_chans, n_times)

    augmentors = {}
    augmentors["gaussian"] = EEGNoiseAugmentor(noise_type="gaussian", intensity=intensity, seed=seed)
    augmentors["dropout"] = EEGNoiseAugmentor(noise_type="dropout", intensity=intensity, seed=seed)

    eog_template = _ensure_eog_template_npz()
    if eog_template:
        augmentors["eog"] = ("direct", eog_template)  # Use direct inject for consistent baseline
    else:
        print("Warning: EOG template not found. Skipping EOG panel.")
        augmentors["eog"] = None

    perturbed = {}
    for name, aug in augmentors.items():
        if aug is not None:
            if name == "eog" and isinstance(aug, tuple) and aug[0] == "direct":
                # EOG: use direct inject with apply_car=False; higher scale for visibility
                info = mne.create_info(
                    ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_chans
                )
                info.set_montage("standard_1020", on_missing="warn")
                temporal_cov = 0.5  # 50% temporal coverage for visible blink artifacts
                eog_epoch = inject_realistic_eog_artifacts_with_coverage(
                    epoch,
                    info,
                    aug[1],
                    montage_name="standard_1020",
                    temporal_coverage=temporal_cov,
                    seed=seed,
                    apply_car=False,  # Match baseline of other panels (no CAR)
                    artifact_scale_factor=500.0,  # Visible but comparable to clean (~20–30 µV)
                    allow_boundary_intersection=True,
                )
                perturbed[name] = eog_epoch[ch_indices, :]
            else:
                out = aug.transform(X_batch.copy())
                perturbed[name] = out[0][ch_indices, :]
        else:
            perturbed[name] = None

    # Build figure: 2x2 grid (Clean, Gaussian, EOG, Dropout)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0), sharex=True)
    axes = axes.flatten()

    panels = [
        ("Clean", clean_subset, "k"),
        ("Gaussian", perturbed.get("gaussian"), "#2e86ab"),
        ("EOG", perturbed.get("eog"), "#a23b72"),
        ("Dropout", perturbed.get("dropout"), "#f18f01"),
    ]

    y_offset = 40
    for ax, (title, data, color) in zip(axes, panels):
        if data is not None:
            plot_trace_panel(ax, data, ch_labels, time_axis, y_offset=y_offset, color=color)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[-1].set_xticks(np.linspace(0, time_axis[-1], 5))
    axes[-1].tick_params(axis="x", labelsize=8)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out_path = output_dir / f"perturbation_traces.{ext}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
