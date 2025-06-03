"""
This script contains exploratory analysis validating the realism of the EOG artifacts simulated
via `mne.simulation.add_eog()` against a semi-simulated EEG/EOG dataset.
"""

import os
import zipfile
import urllib.request
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from mne.io import RawArray
from mne import create_info
from mne.simulation import add_eog
from mne.channels import make_standard_montage
from augmentation.noise import inject_scaled_eog_signal

# === Step 1: Download and extract dataset ===
url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/wb6yvr725d-4.zip"
zip_path = "eeg_eog_dataset.zip"
extract_dir = "eeg_eog_dataset"
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extracted files:", os.listdir(extract_dir))
path = os.path.join(extract_dir, os.listdir(extract_dir)[0])

# === Step 2: Load .mat files ===
pure = scipy.io.loadmat(os.path.join(path, "Pure_Data.mat"))
contaminated = scipy.io.loadmat(os.path.join(path, "Contaminated_Data.mat"))
veog = scipy.io.loadmat(os.path.join(path, "VEOG.mat"))
heog = scipy.io.loadmat(os.path.join(path, "HEOG.mat"))

print("Pure_Data keys:", pure.keys())
print("Contaminated_Data keys:", contaminated.keys())
print("VEOG keys:", veog.keys())
print("HEOG keys:", heog.keys())

# === Step 3: Compare frontal channels visually ===
subject_id = 1
pure_eeg = pure[f"sim{subject_id}_resampled"]
cont_eeg = contaminated[f"sim{subject_id}_con"]

sfreq = 200
n_times = pure_eeg.shape[1]
times = np.arange(n_times) / sfreq

frontal_channels = {
    "Fp1": 0,
    "Fp2": 1
}

plt.figure(figsize=(12, 8))
for i, (ch_name, ch_idx) in enumerate(frontal_channels.items()):
    plt.subplot(2, 1, i+1)
    plt.plot(times, pure_eeg[ch_idx], label=f'Pure EEG ({ch_name})', color='blue', alpha=0.8)
    plt.plot(times, cont_eeg[ch_idx], label=f'Contaminated EEG ({ch_name})', color='red', alpha=0.6)
    plt.title(f"{ch_name} — Pure vs Contaminated")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# === Step 4: Inject MNE-style EOG ===
channel_names = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]
info = create_info(ch_names=channel_names, sfreq=sfreq, ch_types=['eeg'] * len(channel_names))
simulated = inject_scaled_eog_signal(pure_eeg, info, scale_factor=4.0, seed=42)

# === Step 5: RMS analysis ===
def compute_rms_diff(clean, noisy):
    return np.sqrt(np.mean((noisy - clean) ** 2, axis=1))

rms_simulated = compute_rms_diff(pure_eeg, simulated)
rms_real = compute_rms_diff(pure_eeg, cont_eeg)

rms_df = pd.DataFrame({
    'channel': channel_names,
    'rms_mne_eog': rms_simulated,
    'rms_semi_simulated': rms_real
})
rms_df.to_csv("results/rms_comparison_subject1.csv", index=False)

# === Optional: Print top affected channels ===
print("Top channels (real):")
print(rms_df.sort_values("rms_semi_simulated", ascending=False).head())
print("\nTop channels (simulated):")
print(rms_df.sort_values("rms_mne_eog", ascending=False).head())
