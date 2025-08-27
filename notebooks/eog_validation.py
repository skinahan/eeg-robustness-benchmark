"""
This script implements Step 1 of EOG artifact injection enhancement:
Learning the EOG mixing on the 19-channel semi-simulated dataset.

For each subject in the semi-simulated dataset:
1. Load their pure EEG (clean) and contaminated EEG
2. Convert everything to Volts and apply common average reference (CAR)
3. Subtract pure from contaminated to get ground-truth EOG contribution
4. Regress this contribution against VEOG and HEOG signals to get mixing matrix B
5. Record calibration statistics for later use

This gives us subject-specific mixing matrices and calibration stats for all 54 subjects.
"""

import os
import zipfile
import urllib.request
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Tuple, List
import warnings
from scipy import signal

warnings.filterwarnings('ignore')

# === Step 1: Download and extract dataset ===
extract_dir = "eeg_eog_dataset"

if not os.path.exists(extract_dir):
    url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/wb6yvr725d-4.zip"
    zip_path = "eeg_eog_dataset.zip"
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset extracted successfully!")

print("Extracted files:", os.listdir(extract_dir))
path = os.path.join(extract_dir, os.listdir(extract_dir)[0])

# === Step 2: Load .mat files ===
print("Loading dataset files...")
pure = scipy.io.loadmat(os.path.join(path, "Pure_Data.mat"))
contaminated = scipy.io.loadmat(os.path.join(path, "Contaminated_Data.mat"))
veog = scipy.io.loadmat(os.path.join(path, "VEOG.mat"))
heog = scipy.io.loadmat(os.path.join(path, "HEOG.mat"))

# Get all subject IDs
subject_keys = [k for k in pure.keys() if k.startswith('sim')]
subject_ids = [int(k.replace('sim', '').replace('_resampled', '')) for k in subject_keys]
subject_ids.sort()
print(f"Found {len(subject_ids)} subjects: {subject_ids}")

# 19-channel montage (standard 10-20 system)
channel_names = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

# Target sampling frequency and time length (use the most common length as reference)
target_sfreq = 200  # Hz
target_n_times = 5601  # Most common length in the dataset


# === Utility functions ===
def to_volts(arr: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, str]:
    """
    Ensure an ndarray is in Volts.
    
    Heuristic:
      - If max abs > 1e-3, assume microvolts → convert to V.
      - Else, assume already in V.
    
    Returns:
        arr_V: Data in Volts
        unit: Conversion description
    """
    arr = np.asarray(arr)
    absmax = np.max(np.abs(arr))

    if absmax > 1.0:  # heuristic threshold
        arr_V = arr * 1e-6
        unit = "µV→V"
    else:
        arr_V = arr.copy()
        unit = "V"

    if verbose:
        print(f"[Unit check] Assumed {unit}. "
              f"Input range: [{arr.min():.3e}, {arr.max():.3e}], "
              f"Output range: [{arr_V.min():.3e}, {arr_V.max():.3e}]")

    return arr_V, unit


def car(data_V: np.ndarray) -> np.ndarray:
    """Apply common average reference (CAR) to EEG data."""
    return data_V - np.mean(data_V, axis=0, keepdims=True)


def resample_to_target_length(data: np.ndarray, target_length: int, axis: int = -1) -> np.ndarray:
    """
    Resample data to target length along specified axis.
    
    Args:
        data: Input data array
        target_length: Target number of time points
        axis: Axis along which to resample (default: last axis)
        
    Returns:
        Resampled data with target length
    """
    current_length = data.shape[axis]

    if current_length == target_length:
        return data

    # Create time arrays
    current_time = np.linspace(0, 1, current_length)
    target_time = np.linspace(0, 1, target_length)

    # Resample along the specified axis
    if axis == -1 or axis == data.ndim - 1:
        # Resample last axis (time dimension)
        resampled = np.zeros(data.shape[:-1] + (target_length,))
        for idx in np.ndindex(data.shape[:-1]):
            resampled[idx] = np.interp(target_time, current_time, data[idx])
    else:
        # For other axes, we need to handle differently
        raise ValueError(f"Resampling along axis {axis} not implemented")

    return resampled


def fit_eog_topography_car(pure_V: np.ndarray, cont_V: np.ndarray,
                           veog_V: np.ndarray, heog_V: np.ndarray,
                           alpha: float = 0.01) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit EOG mixing matrix using ridge regression.
    
    Args:
        pure_V: Pure EEG data in Volts, shape (n_channels, n_times)
        cont_V: Contaminated EEG data in Volts, shape (n_channels, n_times)
        veog_V: VEOG signal in Volts, shape (n_times,)
        heog_V: HEOG signal in Volts, shape (n_times,)
        alpha: Ridge regularization parameter
        
    Returns:
        B: Mixing matrix shape (n_channels, 2) mapping [VEOG, HEOG] → EEG channels
        reg_std: Standard deviation of VEOG and HEOG signals
        target_rms_med: Median RMS of EOG artifact across EEG channels
    """
    # Apply CAR to both datasets
    pure_car = car(pure_V)
    cont_car = car(cont_V)

    # EOG contribution (target) - difference between contaminated and pure
    eog_in_eeg = cont_car - pure_car  # (n_ch, n_t)

    # Design matrix: stack VEOG and HEOG
    X = np.vstack([veog_V, heog_V])  # (2, n_t)

    # Ridge regression: B = (X @ X.T + αI)^(-1) @ X @ eog_in_eeg.T
    Xt = X.T  # (n_t, 2)
    XtX = Xt.T @ Xt + alpha * np.eye(2)  # (2, 2)
    XtX_inv = np.linalg.pinv(XtX)
    B = (eog_in_eeg @ Xt) @ XtX_inv  # (n_ch, 2)

    # Regressor amplitude stats (for template calibration)
    reg_std = X.std(axis=1, ddof=1)  # (2,) - std of VEOG and HEOG

    # Target EEG artifact RMS (for end-to-end calibration)
    target_rms = np.sqrt((eog_in_eeg ** 2).mean(axis=1))  # per-channel
    target_rms_med = np.median(target_rms)  # robust scalar

    return B, reg_std, target_rms_med


def compute_eog_statistics(eog_in_eeg: np.ndarray, veog_V: np.ndarray,
                           heog_V: np.ndarray) -> Dict:
    """
    Compute comprehensive statistics about EOG artifacts and signals.
    
    Args:
        eog_in_eeg: EOG contribution in EEG, shape (n_channels, n_times)
        veog_V: VEOG signal in Volts, shape (n_times,)
        heog_V: HEOG signal in Volts, shape (n_times,)
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {}

    # EOG signal statistics
    stats['veog_std'] = float(veog_V.std(ddof=1))
    stats['heog_std'] = float(heog_V.std(ddof=1))
    stats['veog_rms'] = float(np.sqrt((veog_V ** 2).mean()))
    stats['heog_rms'] = float(np.sqrt((heog_V ** 2).mean()))
    stats['veog_range'] = float(veog_V.max() - veog_V.min())
    stats['heog_range'] = float(heog_V.max() - heog_V.min())

    # EOG artifact statistics per channel
    eog_rms_per_channel = np.sqrt((eog_in_eeg ** 2).mean(axis=1))
    stats['eog_rms_per_channel'] = eog_rms_per_channel.tolist()
    stats['eog_rms_median'] = float(np.median(eog_rms_per_channel))
    stats['eog_rms_mean'] = float(np.mean(eog_rms_per_channel))
    stats['eog_rms_std'] = float(np.std(eog_rms_per_channel))
    stats['eog_rms_max'] = float(np.max(eog_rms_per_channel))
    stats['eog_rms_min'] = float(np.min(eog_rms_per_channel))

    # Channel-specific statistics
    stats['most_affected_channels'] = []
    for i, (ch_name, rms) in enumerate(zip(channel_names, eog_rms_per_channel)):
        stats['most_affected_channels'].append({
            'channel': ch_name,
            'rms': float(rms),
            'rank': 0  # Will be filled later
        })

    # Sort channels by RMS and add ranking
    sorted_channels = sorted(stats['most_affected_channels'], key=lambda x: x['rms'], reverse=True)
    for i, ch_info in enumerate(sorted_channels):
        ch_info['rank'] = i + 1

    stats['most_affected_channels'] = sorted_channels

    return stats


# === Step 3: Process all subjects ===
print(f"\nProcessing {len(subject_ids)} subjects...")
print(f"Target dimensions: {len(channel_names)} channels × {target_n_times} time points")

# Storage for results
subject_results = {}
failed_subjects = []

for subject_id in subject_ids:
    print(f"Processing subject {subject_id}...")

    try:
        # Load subject data
        pure_key = f"sim{subject_id}_resampled"
        cont_key = f"sim{subject_id}_con"
        veog_key = f"veog_{subject_id}"
        heog_key = f"heog_{subject_id}"

        # Check if all required data exists
        if not all(key in data.keys() for key, data in [(pure_key, pure), (cont_key, contaminated),
                                                        (veog_key, veog), (heog_key, heog)]):
            print(f"  Warning: Missing data for subject {subject_id}, skipping...")
            failed_subjects.append(subject_id)
            continue

        # Extract data
        pure_eeg = pure[pure_key]  # (19, n_times)
        cont_eeg = contaminated[cont_key]  # (19, n_times)
        veog_sig = veog[veog_key].ravel()  # (n_times,) - flatten to 1D
        heog_sig = heog[heog_key].ravel()  # (n_times,)

        # Check original shapes
        original_shapes = {
            'pure': pure_eeg.shape,
            'contaminated': cont_eeg.shape,
            'veog': veog_sig.shape,
            'heog': heog_sig.shape
        }

        # Resample all data to target length if necessary
        if pure_eeg.shape[1] != target_n_times:
            print(f"  Resampling from {pure_eeg.shape[1]} to {target_n_times} time points...")
            pure_eeg = resample_to_target_length(pure_eeg, target_n_times)
            cont_eeg = resample_to_target_length(cont_eeg, target_n_times)
            veog_sig = resample_to_target_length(veog_sig, target_n_times)
            heog_sig = resample_to_target_length(heog_sig, target_n_times)

        # Convert to Volts
        pure_V, _ = to_volts(pure_eeg, verbose=False)
        cont_V, _ = to_volts(cont_eeg, verbose=False)
        veog_V, _ = to_volts(veog_sig, verbose=False)
        heog_V, _ = to_volts(heog_sig, verbose=False)

        # Verify data consistency after resampling
        assert pure_V.shape == cont_V.shape == (
        19, target_n_times), f"Shape mismatch for subject {subject_id} after resampling"
        assert veog_V.shape == heog_V.shape == (
        target_n_times,), f"EOG shape mismatch for subject {subject_id} after resampling"

        # Fit EOG mixing matrix
        B, reg_std, target_rms_med = fit_eog_topography_car(
            pure_V, cont_V, veog_V, heog_V, alpha=0.01
        )

        # Compute comprehensive statistics
        eog_in_eeg = car(cont_V) - car(pure_V)
        stats = compute_eog_statistics(eog_in_eeg, veog_V, heog_V)

        # Store results
        subject_results[subject_id] = {
            'mixing_matrix': B.tolist(),  # (19, 2) - convert to list for JSON serialization
            'regressor_std': reg_std.tolist(),  # (2,) - [veog_std, heog_std]
            'target_rms_median': float(target_rms_med),
            'statistics': stats,
            'data_shape': {
                'n_channels': 19,
                'n_times': target_n_times,
                'sampling_freq': target_sfreq
            },
            'original_shapes': original_shapes,  # Store original shapes for reference
            'resampled': pure_eeg.shape[1] != target_n_times  # Flag if resampling was needed
        }

        print(f"  ✓ Successfully processed subject {subject_id}")
        print(f"    - Original shapes: {original_shapes}")
        print(f"    - Mixing matrix shape: {B.shape}")
        print(f"    - VEOG std: {reg_std[0]:.2e} V, HEOG std: {reg_std[1]:.2e} V")
        print(f"    - Target RMS median: {target_rms_med:.2e} V")
        print(
            f"    - Most affected channel: {stats['most_affected_channels'][0]['channel']} (RMS: {stats['most_affected_channels'][0]['rms']:.2e} V)")

    except Exception as e:
        print(f"  ✗ Error processing subject {subject_id}: {str(e)}")
        failed_subjects.append(subject_id)
        continue

# === Step 4: Summary and save results ===
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Successfully processed: {len(subject_results)} subjects")
print(f"Failed: {len(failed_subjects)} subjects")
if failed_subjects:
    print(f"Failed subjects: {failed_subjects}")

# Save results
output_dir = Path("eog_mixing_results")
output_dir.mkdir(exist_ok=True)

# Save individual subject results
for subject_id, results in subject_results.items():
    subject_file = output_dir / f"subject_{subject_id:02d}_eog_mixing.json"
    with open(subject_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

# Save summary results
summary_file = output_dir / "all_subjects_eog_mixing_summary.json"
summary_data = {
    'dataset_info': {
        'n_subjects': len(subject_results),
        'n_channels': 19,
        'n_times': target_n_times,
        'sampling_freq': target_sfreq,
        'channel_names': channel_names,
        'target_dimensions': f"{len(channel_names)}×{target_n_times}"
    },
    'subject_ids': list(subject_results.keys()),
    'failed_subjects': failed_subjects,
    'global_statistics': {
        'veog_std_range': [
            min(sub['regressor_std'][0] for sub in subject_results.values()),
            max(sub['regressor_std'][0] for sub in subject_results.values())
        ],
        'heog_std_range': [
            min(sub['regressor_std'][1] for sub in subject_results.values()),
            max(sub['regressor_std'][1] for sub in subject_results.values())
        ],
        'target_rms_median_range': [
            min(sub['target_rms_median'] for sub in subject_results.values()),
            max(sub['target_rms_median'] for sub in subject_results.values())
        ]
    },
    'resampling_info': {
        'target_length': target_n_times,
        'subjects_resampled': [sub_id for sub_id, results in subject_results.items() if
                               results.get('resampled', False)],
        'subjects_not_resampled': [sub_id for sub_id, results in subject_results.items() if
                                   not results.get('resampled', False)]
    }
}

with open(summary_file, 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)

print(f"\nResults saved to: {output_dir}")
print(f"Summary file: {summary_file}")

# === Step 5: Visualize results ===
if len(subject_results) > 0:
    print("\nGenerating visualizations...")

    # Extract key statistics across subjects
    veog_stds = [sub['regressor_std'][0] for sub in subject_results.values()]
    heog_stds = [sub['regressor_std'][1] for sub in subject_results.values()]
    target_rms_medians = [sub['target_rms_median'] for sub in subject_results.values()]

    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # VEOG std distribution
    axes[0, 0].hist(veog_stds, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('VEOG Standard Deviation (V)')
    axes[0, 0].set_ylabel('Number of Subjects')
    axes[0, 0].set_title('Distribution of VEOG Standard Deviations')
    axes[0, 0].grid(True, alpha=0.3)

    # HEOG std distribution
    axes[0, 1].hist(heog_stds, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('HEOG Standard Deviation (V)')
    axes[0, 1].set_ylabel('Number of Subjects')
    axes[0, 1].set_title('Distribution of HEOG Standard Deviations')
    axes[0, 1].grid(True, alpha=0.3)

    # Target RMS median distribution
    axes[1, 0].hist(target_rms_medians, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Target RMS Median (V)')
    axes[1, 0].set_ylabel('Number of Subjects')
    axes[1, 0].set_title('Distribution of Target RMS Medians')
    axes[1, 0].grid(True, alpha=0.3)

    # Scatter plot: VEOG vs HEOG std
    axes[1, 1].scatter(veog_stds, heog_stds, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('VEOG Standard Deviation (V)')
    axes[1, 1].set_ylabel('HEOG Standard Deviation (V)')
    axes[1, 1].set_title('VEOG vs HEOG Standard Deviations')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'eog_mixing_statistics.png', dpi=300, bbox_inches='tight')
    print(f"Statistics plot saved: {output_dir / 'eog_mixing_statistics.png'}")

    # Channel-wise analysis
    channel_rms_data = {ch: [] for ch in channel_names}
    for sub_results in subject_results.values():
        for ch_info in sub_results['statistics']['most_affected_channels']:
            channel_rms_data[ch_info['channel']].append(ch_info['rms'])

    # Plot channel-wise RMS distributions
    fig, ax = plt.subplots(figsize=(15, 8))
    channel_medians = [np.median(channel_rms_data[ch]) for ch in channel_names]
    channel_means = [np.mean(channel_rms_data[ch]) for ch in channel_names]

    x_pos = np.arange(len(channel_names))
    ax.bar(x_pos, channel_medians, alpha=0.7, color='skyblue', edgecolor='black', label='Median')
    ax.bar(x_pos, channel_means, alpha=0.5, color='orange', edgecolor='black', label='Mean')

    ax.set_xlabel('EEG Channels')
    ax.set_ylabel('EOG Artifact RMS (V)')
    ax.set_title('EOG Artifact RMS by Channel (Across All Subjects)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(channel_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'channel_wise_eog_rms.png', dpi=300, bbox_inches='tight')
    print(f"Channel-wise analysis plot saved: {output_dir / 'channel_wise_eog_rms.png'}")

print(f"\n=== STEP 1 COMPLETE ===")
print(f"Successfully learned EOG mixing matrices for {len(subject_results)} subjects")
print(f"Results saved to: {output_dir}")
print(f"Next step: Use these mixing matrices to inject realistic EOG artifacts into new EEG data")
