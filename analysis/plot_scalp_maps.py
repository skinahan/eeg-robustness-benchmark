#!/usr/bin/env python3
"""
Plot EEG montages and EOG artifact spatial distribution.

This script creates scalp maps for:
1. BNCI2014_001 montage
2. BI2015a montage
3. Lee2019_MI montage
4. Lee2019_SSVEP montage
5. EOG artifact spatial distribution from simEOG dataset
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_sensors, plot_topomap, plot_montage
from mne.channels import make_standard_montage
from pathlib import Path
import scipy.io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moabb.datasets import BNCI2014_001, BI2015a, Lee2019_MI, Lee2019_SSVEP
from config import get_paradigm


def get_dataset_info(dataset_name):
    """Load dataset and return info object with montage and raw data."""
    print(f"Loading {dataset_name}...")
    
    if dataset_name == "BNCI2014_001":
        dataset = BNCI2014_001()
        subject_id = 1
    elif dataset_name == "BI2015a":
        dataset = BI2015a()
        subject_id = 1
    elif dataset_name == "Lee2019_MI":
        dataset = Lee2019_MI()
        subject_id = 1
    elif dataset_name == "Lee2019_SSVEP":
        dataset = Lee2019_SSVEP()
        subject_id = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get subject data - this returns a dict with session keys
    subject_data = dataset._get_single_subject_data(subject_id)
    
    # Inspect the structure to find the first available session
    if isinstance(subject_data, dict):
        # Get the first session key (could be 0, 1, 'session_0', etc.)
        session_keys = list(subject_data.keys())
        if not session_keys:
            raise ValueError(f"No sessions found for subject {subject_id} in {dataset_name}")
        
        first_session_key = session_keys[0]
        session_data = subject_data[first_session_key]
        
        # Session data is typically a dict with 'train' and/or 'test' keys
        if isinstance(session_data, dict):
            # Try 'train' first, then 'test', then any available key
            if 'train' in session_data:
                raw_data = session_data['train']
            elif 'test' in session_data:
                raw_data = session_data['test']
            else:
                # Get the first available key
                first_key = list(session_data.keys())[0]
                raw_data = session_data[first_key]
        else:
            # If session_data is directly a Raw object
            raw_data = session_data
    else:
        # If subject_data is directly a Raw object (unlikely but handle it)
        raw_data = subject_data
    
    # Get info from the raw data
    if hasattr(raw_data, 'info'):
        info = raw_data.info.copy()
    else:
        raise ValueError(f"Could not extract info from {dataset_name} data")
    
    # Ensure montage is set
    if info.get_montage() is None:
        # Try to set a standard montage
        try:
            info.set_montage('standard_1020', on_missing='warn')
        except:
            # If that fails, try standard_1005
            try:
                info.set_montage('standard_1005', on_missing='warn')
            except:
                print(f"Warning: Could not set montage for {dataset_name}")
    
    return info, raw_data, dataset_name


def plot_montage_scalp_map(info, raw_data, title, output_path):
    """Plot a scalp map showing the montage using the montage's .plot() method."""
    # Get montage
    montage = info.get_montage()
    if montage is None:
        raise ValueError(f"No montage found for {title}")
    
    # Use the montage's .plot() method with default parameters
    fig = montage.plot(show_names=True, show=False)
    
    # Add title to the figure
    fig.suptitle(f"{title} Montage", fontsize=16, fontweight='bold', y=0.98)
    
    # Save as PDF with 300 DPI
    fig.savefig(output_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def load_simeog_data(dataset_dir):
    """Load simEOG dataset and compute artifact distribution."""
    print(f"Loading simEOG dataset from {dataset_dir}...")
    
    # Find the data directory
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"simEOG dataset directory not found: {dataset_dir}")
    
    # Look for .mat files
    mat_files = {
        'pure': None,
        'contaminated': None
    }
    
    for file in os.listdir(dataset_dir):
        if 'Pure_Data' in file and file.endswith('.mat'):
            mat_files['pure'] = os.path.join(dataset_dir, file)
        elif 'Contaminated_Data' in file and file.endswith('.mat'):
            mat_files['contaminated'] = os.path.join(dataset_dir, file)
    
    if mat_files['pure'] is None or mat_files['contaminated'] is None:
        raise FileNotFoundError(f"Could not find Pure_Data.mat or Contaminated_Data.mat in {dataset_dir}")
    
    # Load the .mat files
    pure = scipy.io.loadmat(mat_files['pure'])
    contaminated = scipy.io.loadmat(mat_files['contaminated'])
    
    # Get subject keys
    subject_keys = [k for k in pure.keys() if k.startswith('sim')]
    if not subject_keys:
        raise ValueError("No subject data found in Pure_Data.mat")
    
    # Use first subject for visualization
    first_key = subject_keys[0]
    pure_key = first_key
    cont_key = first_key.replace('_resampled', '_con')
    
    if cont_key not in contaminated:
        # Try alternative naming
        cont_key = first_key.replace('_resampled', '') + '_con'
        if cont_key not in contaminated:
            cont_key = first_key.replace('sim', 'sim') + '_con'
    
    if cont_key not in contaminated:
        raise ValueError(f"Could not find matching contaminated data for {first_key}")
    
    # Extract data
    pure_eeg = pure[pure_key]  # (n_channels, n_times)
    cont_eeg = contaminated[cont_key]  # (n_channels, n_times)
    
    # Compute EOG artifact as difference (contaminated - pure)
    eog_artifact = cont_eeg - pure_eeg  # (n_channels, n_times)
    
    # Compute RMS per channel across time
    artifact_rms = np.sqrt(np.mean(eog_artifact ** 2, axis=1))  # (n_channels,)
    
    # 19-channel montage (standard 10-20 system) from simEOG
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    return artifact_rms, channel_names


def plot_eog_artifact_distribution(dataset_dir, title, output_path):
    """Plot the spatial distribution of EOG artifacts from simEOG dataset."""
    # Load simEOG data
    artifact_rms, channel_names = load_simeog_data(dataset_dir)
    
    # Create a standard 19-channel montage
    montage = make_standard_montage('standard_1020')
    
    # Get positions for our 19 channels
    ch_pos = montage.get_positions()['ch_pos']
    pos_2d = np.array([ch_pos[ch][:2] for ch in channel_names])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot topomap using MNE defaults
    im, _ = plot_topomap(
        artifact_rms,
        pos=pos_2d,
        names=channel_names,
        axes=ax,
        show=False,
        cmap='Reds',
        vlim=(0, None),  # Auto-scale from 0
        outlines='head'
    )
    
    # Add colorbar with larger font (2-3x increase)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('EOG Artifact RMS (µV)', rotation=270, labelpad=20, fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    
    # Set title with larger font (2-3x increase)
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    
    # Increase font size for electrode labels (2-3x increase)
    for text in ax.texts:
        text.set_fontsize(18)
    
    # Save as PDF with 300 DPI
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to generate all scalp maps."""
    # Create output directory
    output_dir = project_root / "analysis" / "scalp_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to simEOG dataset
    simeog_dir = project_root / "notebooks" / "eog_eog_dataset"
    
    # If the dataset is in a subdirectory, find it
    if not simeog_dir.exists():
        # Try alternative paths
        alt_paths = [
            project_root / "eeg_eog_dataset",
            project_root / "notebooks" / "eeg_eog_dataset"
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                simeog_dir = alt_path
                break
    
    # 1. Plot BNCI2014_001 montage
    print("\n" + "="*60)
    print("Plotting BNCI2014_001 montage")
    print("="*60)
    info_bnci, raw_bnci, _ = get_dataset_info("BNCI2014_001")
    plot_montage_scalp_map(
        info_bnci,
        raw_bnci,
        "BNCI2014_001",
        output_dir / "bnci2014_001_montage.pdf"
    )
    
    # 2. Plot BI2015a montage
    print("\n" + "="*60)
    print("Plotting BI2015a montage")
    print("="*60)
    info_bi, raw_bi, _ = get_dataset_info("BI2015a")
    plot_montage_scalp_map(
        info_bi,
        raw_bi,
        "BI2015a",
        output_dir / "bi2015a_montage.pdf"
    )
    
    # 3. Plot Lee2019_MI montage
    print("\n" + "="*60)
    print("Plotting Lee2019_MI montage")
    print("="*60)
    info_lee_mi, raw_lee_mi, _ = get_dataset_info("Lee2019_MI")
    plot_montage_scalp_map(
        info_lee_mi,
        raw_lee_mi,
        "Lee2019_MI",
        output_dir / "lee2019_mi_montage.pdf"
    )
    
    # 4. Plot Lee2019_SSVEP montage
    print("\n" + "="*60)
    print("Plotting Lee2019_SSVEP montage")
    print("="*60)
    info_lee, raw_lee, _ = get_dataset_info("Lee2019_SSVEP")
    plot_montage_scalp_map(
        info_lee,
        raw_lee,
        "Lee2019_SSVEP",
        output_dir / "lee2019_ssvep_montage.pdf"
    )
    
    # 5. Plot EOG artifact spatial distribution from simEOG
    print("\n" + "="*60)
    print("Plotting EOG artifact spatial distribution")
    print("="*60)
    if simeog_dir.exists():
        # Find the actual data directory (might be in a subdirectory)
        data_dir = simeog_dir
        for item in os.listdir(simeog_dir):
            item_path = os.path.join(simeog_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains .mat files
                if any(f.endswith('.mat') for f in os.listdir(item_path)):
                    data_dir = item_path
                    break
        
        plot_eog_artifact_distribution(
            str(data_dir),
            "EOG Artifact Distribution",
            output_dir / "eog_artifact_distribution.pdf"
        )
    else:
        print(f"Warning: simEOG dataset not found at {simeog_dir}")
        print("Skipping EOG artifact distribution plot.")
    
    print("\n" + "="*60)
    print("All scalp maps generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
