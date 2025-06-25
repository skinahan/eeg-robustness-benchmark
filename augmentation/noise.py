# augmentation/noise.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mne
from mne.simulation import add_eog
from mne.io import RawArray
from mne.channels import make_standard_montage


class EEGNoiseAugmentor(BaseEstimator, TransformerMixin):
    """
    EEG noise augmentation transformer for MOABB pipelines.

    Supports three noise types:
    - 'dropout': Randomly zero out a percentage of EEG channels.
    - 'gaussian': Add Gaussian noise to a subset of channels.
    - 'eog': Add simulated EOG artifacts to epochs.

    Parameters
    ----------
    noise_type : str
        One of ['dropout', 'gaussian', 'eog'].
    intensity : float
        Noise severity/intensity. Meaning depends on noise_type:
        - For 'dropout': percentage of channels to drop (0-100).
        - For 'gaussian': scaling factor (D) for variance.
        - For 'eog': scaling factor for simulated blink strength.
    seed : int
        Random seed for reproducibility.
    """
    def __init__(self, noise_type='dropout', intensity=10.0, seed=42):
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Expected X to be a NumPy array of shape (n_epochs, n_channels, n_times).")

        if self.noise_type == 'dropout':
            return self._apply_channel_dropout(X)
        elif self.noise_type == 'gaussian':
            return self._apply_gaussian_noise(X)
        elif self.noise_type == 'eog':
            return self._apply_eog_noise(X)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def _apply_channel_dropout(self, data):
        np.random.seed(self.seed)
        n_epochs, n_channels, n_times = data.shape
        n_drop = int(n_channels * self.intensity / 100)
        data_aug = data.copy()
        for i in range(n_epochs):
            drop_idxs = np.random.choice(n_channels, size=n_drop, replace=False)
            data_aug[i, drop_idxs, :] = 0
        return data_aug

    def _apply_gaussian_noise(self, data):
        np.random.seed(self.seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()
        for i in range(n_epochs):
            noise = np.random.randn(n_channels, n_times)
            data_aug[i] += self.intensity * noise
        return data_aug

    def _apply_eog_noise(self, data):
        n_epochs, n_channels, n_times = data.shape
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]

        #ch_names = [f"EEG {i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg'] * n_channels)

        noisy_epochs = []
        for i in range(n_epochs):
            contaminated = inject_scaled_eog_signal(
                data[i], info, scale_factor=self.intensity, seed=self.seed + i
            )
            noisy_epochs.append(contaminated)
        return np.stack(noisy_epochs, axis=0)


def inject_scaled_eog_signal(data, info, scale_factor=4.0, seed=42):
    """
    Injects scaled EOG artifacts into a single EEG segment using MNE's add_eog().
    Auto-detects units and returns output in the same scale.
    """
    max_val = np.abs(data).max()
    is_microvolts = max_val > 1.0
    data_volts = data * 1e-6 if is_microvolts else data

    raw_clean = RawArray(data_volts, info, verbose='error')
    raw_clean.set_montage(make_standard_montage('standard_1020'))

    raw_eog = raw_clean.copy()
    add_eog(raw_eog, random_state=seed, interp='cos2', head_pos=None)

    eog_component = raw_eog.get_data() - raw_clean.get_data()
    scaled_eog = scale_factor * eog_component
    raw_scaled = raw_clean.copy()
    raw_scaled._data += scaled_eog

    return raw_scaled.get_data() * 1e6 if is_microvolts else raw_scaled.get_data()


from sklearn.base import BaseEstimator, ClassifierMixin

class TrainOnlyNoiseClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_pipeline, noise_type='dropout', intensity=25.0, seed=42):
        self.base_pipeline = base_pipeline
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed

    def fit(self, X, y):
        augmenter = EEGNoiseAugmentor(
            noise_type=self.noise_type,
            intensity=self.intensity,
            seed=self.seed
        )
        X_aug = augmenter.fit_transform(X)
        self.base_pipeline.fit(X_aug, y)
        self.is_fitted_ = True
        self.base_pipeline.is_fitted_ = True
        # ✅ Expose fitted classes_ attribute from the wrapped classifier
        if hasattr(self.base_pipeline, "classes_"):
            self.classes_ = self.base_pipeline.classes_
        elif hasattr(self.base_pipeline[-1], "classes_"):
            self.classes_ = self.base_pipeline[-1].classes_
        else:
            raise AttributeError("Base pipeline does not expose `classes_` after fit")

        return self

    def predict(self, X):
        return self.base_pipeline.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.base_pipeline.score(X, y, sample_weight)

    def predict_proba(self, X):
        if hasattr(self.base_pipeline, 'predict_proba'):
            return self.base_pipeline.predict_proba(X)
        elif hasattr(self.base_pipeline[-1], 'predict_proba'):
            return self.base_pipeline[-1].predict_proba(X)
        else:
            raise NotImplementedError("Underlying model does not support predict_proba()")
    #
    # def decision_function(self, X):
    #     return self.base_pipeline.decision_function(X)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep and hasattr(self.base_pipeline, 'get_params'):
            base_params = self.base_pipeline.get_params().copy()
            for key in list(base_params.keys()):
                base_params[f'base_pipeline__{key}'] = base_params.pop(key)
            params.update(base_params)
        return params

    def set_params(self, **params):
        base_params = {}
        own_params = {}

        for key, value in params.items():
            if key.startswith('base_pipeline__'):
                base_params[key[len('base_pipeline__'):]] = value
            else:
                own_params[key] = value

        if base_params:
            self.base_pipeline.set_params(**base_params)
        if own_params:
            super().set_params(**own_params)

        return self