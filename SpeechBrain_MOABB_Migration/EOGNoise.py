import numpy as np
import torch
from speechbrain.augment.augmenter import Augmentation
import mne
from mne.simulation import add_eog
from mne.io import RawArray
from mne.channels import make_standard_montage


class SpeechBrainEOGArtifact(Augmentation):
    def __init__(self, scale_factor=4.0, sample_rate=250, seed=42):
        super().__init__()
        self.scale_factor = scale_factor
        self.sample_rate = sample_rate
        self.seed = seed

    def apply(self, wav, lengths=None):
        batch, channels, time = wav.shape
        augmented = []

        for i in range(batch):
            data_np = wav[i].cpu().numpy()
            noisy_data = self._inject_eog_to_sample(data_np, i)
            augmented.append(torch.tensor(noisy_data, dtype=wav.dtype))

        return torch.stack(augmented, dim=0)

    def _inject_eog_to_sample(self, sample, idx_offset=0):
        n_channels, n_times = sample.shape
        ch_names = [f"EEG {i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate, ch_types="eeg")

        scale_back = 1e-6 if np.max(np.abs(sample)) > 1.0 else 1.0
        data_volts = sample * scale_back

        raw_clean = RawArray(data_volts, info, verbose="error")
        raw_clean.set_montage(make_standard_montage("standard_1020"))

        raw_eog = raw_clean.copy()
        add_eog(raw_eog, random_state=self.seed + idx_offset, interp="cos2", head_pos=None)

        eog_signal = raw_eog.get_data() - raw_clean.get_data()
        raw_clean._data += self.scale_factor * eog_signal

        return raw_clean.get_data() / scale_back
