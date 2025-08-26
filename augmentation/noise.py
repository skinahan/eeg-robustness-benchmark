# augmentation/noise.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mne
from mne.simulation import add_eog
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.preprocessing import compute_current_source_density
from scipy.interpolate import griddata
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === Enhanced EOG Injection Functions ===

def load_generic_eog_template(template_path):
    """
    Load the generic EOG mixing template created in Step 2.
    
    Parameters
    ----------
    template_path : str or Path
        Path to the generic_eog_mixing_template.npz file
        
    Returns
    -------
    dict
        Dictionary containing:
        - mixing_matrix: (19, 2) array mapping [VEOG, HEOG] → EEG channels
        - veog_std: VEOG standard deviation for calibration
        - heog_std: HEOG standard deviation for calibration
        - target_rms_median: Target EOG artifact RMS for calibration
    """
    try:
        data = np.load(template_path)
        template = {
            'mixing_matrix': data['mixing_matrix'],
            'veog_std': data['veog_std'],
            'heog_std': data['heog_std'],
            'target_rms_median': data['target_rms_median']
        }
        print(f"Loaded EOG template: {template['mixing_matrix'].shape}")
        return template
    except Exception as e:
        raise ValueError(f"Failed to load EOG template from {template_path}: {e}")

def interpolate_eog_topography_to_montage(source_montage, target_montage, source_matrix):
    """
    Interpolate EOG topography from source montage to target montage using spherical splines.
    
    Parameters
    ----------
    source_montage : str
        Source montage name (e.g., 'standard_1005')
    target_montage : str
        Target montage name (e.g., 'biosemi64')
    source_matrix : np.ndarray
        Source mixing matrix (n_source_channels, n_regressors)
        
    Returns
    -------
    np.ndarray
        Interpolated mixing matrix (n_target_channels, n_regressors)
    """
    # Get montage information
    source_montage_obj = make_standard_montage(source_montage)
    target_montage_obj = make_standard_montage(target_montage)
    
    # Extract 3D positions
    source_pos = source_montage_obj.get_positions()['ch_pos']
    target_pos = target_montage_obj.get_positions()['ch_pos']
    
    # Convert to arrays
    source_channels = list(source_pos.keys())
    target_channels = list(target_pos.keys())
    
    source_coords = np.array([source_pos[ch] for ch in source_channels])
    target_coords = np.array([target_pos[ch] for ch in target_channels])
    
    # Normalize to unit sphere for interpolation
    source_coords_norm = source_coords / np.linalg.norm(source_coords, axis=1, keepdims=True)
    target_coords_norm = target_coords / np.linalg.norm(target_coords, axis=1, keepdims=True)
    
    # Interpolate each regressor separately
    n_regressors = source_matrix.shape[1]
    interpolated_matrix = np.zeros((len(target_channels), n_regressors))
    
    for reg_idx in range(n_regressors):
        # Use spherical spline interpolation
        interpolated_values = griddata(
            source_coords_norm, 
            source_matrix[:, reg_idx], 
            target_coords_norm, 
            method='cubic',
            fill_value=0.0  # Fill missing values with 0
        )
        interpolated_matrix[:, reg_idx] = interpolated_values
    
    return interpolated_matrix

def generate_realistic_eog_regressors(n_times, sfreq, template_stats, seed=42):
    """
    Generate realistic VEOG and HEOG time courses using blink templates and calibration.
    
    Parameters
    ----------
    n_times : int
        Number of time points
    sfreq : float
        Sampling frequency in Hz
    template_stats : dict
        Template statistics from load_generic_eog_template()
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (veog_tc, heog_tc) - VEOG and HEOG time courses in Volts
    """
    rng = np.random.RandomState(seed)
    
    # Blink template parameters
    blink_duration_ms = 200
    blink_peak_ms = 80
    blink_frequency = 0.1  # 10% of time contains blinks
    
    # Convert to samples
    blink_duration = int(blink_duration_ms * sfreq / 1000)
    blink_peak = int(blink_peak_ms * sfreq / 1000)
    
    # Generate blink template (smooth gaussian-like)
    t = np.arange(blink_duration)
    blink_template = np.exp(-((t - blink_peak) / (0.35 * blink_peak + 1e-9)) ** 2)
    blink_template = blink_template / np.max(blink_template)
    
    # Place blinks randomly
    n_blinks = int(n_times * blink_frequency)
    blink_starts = rng.choice(n_times - blink_duration, size=n_blinks, replace=False)
    
    # Initialize time courses
    veog_tc = np.zeros(n_times)
    heog_tc = np.zeros(n_times)
    
    # Add blinks
    for start in blink_starts:
        end = min(start + blink_duration, n_times)
        blink_len = end - start
        
        # VEOG: primary blink component
        veog_tc[start:end] += blink_template[:blink_len]
        
        # HEOG: smaller lateral component (random direction)
        lateral_amplitude = 0.2 * rng.uniform(0.5, 1.5)  # 20% of VEOG, with variability
        direction = rng.choice([-1, 1])  # random left/right
        heog_tc[start:end] += direction * lateral_amplitude * blink_template[:blink_len]
    
    # Calibrate to match template statistics
    veog_tc = veog_tc / (np.std(veog_tc) + 1e-12) * template_stats['veog_std']
    heog_tc = heog_tc / (np.std(heog_tc) + 1e-12) * template_stats['heog_std']
    
    return veog_tc, heog_tc

def inject_realistic_eog_artifacts(data, info, template_path, montage_name='standard_1005', 
                                  intensity=1.0, seed=42, apply_car=True):
    """
    Inject realistic EOG artifacts using the learned generic mixing template.
    
    This function:
    1. Loads the generic EOG mixing template
    2. Interpolates to the target montage if different from training
    3. Generates realistic VEOG/HEOG time courses
    4. Projects artifacts to EEG space using the mixing matrix
    5. Calibrates amplitude to match training data statistics
    
    Parameters
    ----------
    data : np.ndarray
        EEG data, shape (n_channels, n_times) in Volts
    info : mne.Info
        MNE info object with channel information
    template_path : str
        Path to generic_eog_mixing_template.npz
    montage_name : str
        Target montage name for interpolation
    intensity : float
        Artifact intensity multiplier (1.0 = realistic, >1.0 = stronger)
    seed : int
        Random seed for reproducibility
    apply_car : bool
        Whether to apply CAR before injection (should match training)
        
    Returns
    -------
    np.ndarray
        Contaminated EEG data in same units as input
    """
    # Load the generic EOG template
    template = load_generic_eog_template(template_path)
    
    # Ensure data is in Volts
    max_val = np.abs(data).max()
    is_microvolts = max_val > 1.0
    data_volts = data * 1e-6 if is_microvolts else data
    
    # Apply CAR if requested (should match training procedure)
    if apply_car:
        data_volts = data_volts - np.mean(data_volts, axis=0, keepdims=True)
    
    # Get current montage
    current_montage = info.get_montage()
    if current_montage is None:
        # Set default montage if none exists
        info.set_montage(montage_name)
        current_montage = info.get_montage()
    
    # Determine if interpolation is needed
    source_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                      'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    current_ch_names = info.ch_names
    needs_interpolation = (len(current_ch_names) != 19 or 
                          not all(ch in current_ch_names for ch in source_channels))
    
    if needs_interpolation:
        print(f"Interpolating EOG topography from 19-channel to {len(current_ch_names)}-channel montage")
        mixing_matrix = interpolate_eog_topography_to_montage(
            'standard_1005', montage_name, template['mixing_matrix']
        )
    else:
        # Use template directly if montage matches
        mixing_matrix = template['mixing_matrix']
    
    # Generate realistic EOG regressors
    n_times = data_volts.shape[1]
    sfreq = info['sfreq']
    veog_tc, heog_tc = generate_realistic_eog_regressors(
        n_times, sfreq, template, seed=seed
    )
    
    # Project EOG artifacts to EEG space
    eog_regressors = np.vstack([veog_tc, heog_tc])  # (2, n_times)
    eog_artifacts = mixing_matrix @ eog_regressors  # (n_channels, n_times)
    
    # Apply intensity scaling
    eog_artifacts *= intensity
    
    # Optional: Calibrate to match target RMS at frontal channels
    if needs_interpolation:
        # Find frontal channels in target montage
        frontal_channels = [i for i, ch in enumerate(current_ch_names) 
                           if any(frontal in ch.upper() for frontal in ['FP', 'FZ', 'F3', 'F4'])]
        
        if frontal_channels:
            # Calculate current RMS at frontal channels
            current_rms = np.sqrt(np.mean(eog_artifacts[frontal_channels, :] ** 2))
            target_rms = template['target_rms_median']
            
            # Scale to match target RMS
            if current_rms > 0:
                scale_factor = target_rms / current_rms
                eog_artifacts *= scale_factor
                print(f"Calibrated EOG artifacts: frontal RMS = {target_rms:.2e} V")
    
    # Add artifacts to clean data
    contaminated_data = data_volts + eog_artifacts
    
    # Convert back to original units
    if is_microvolts:
        contaminated_data *= 1e6
    
    return contaminated_data

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
    add_eog(raw_eog, random_state=seed, interp='cos2', head_pos=None, verbose=0)
    
    eog_component = raw_eog.get_data() - raw_clean.get_data()
    assert(np.sum(np.abs(eog_component)) > 0)
    scaled_eog = scale_factor * eog_component
    # rotation_amount = 2
    # scaled_eog = np.concatenate((scaled_eog[rotation_amount:, :], scaled_eog[:rotation_amount, :]))
    raw_scaled = raw_clean.copy()
    raw_scaled._data += scaled_eog

    assert(np.sum(np.abs(raw_scaled.get_data() - raw_clean.get_data())) > 0)

    return raw_scaled.get_data() * 1e6 if is_microvolts else raw_scaled.get_data()

# === Enhanced EEGNoiseAugmentor Class ===

class EEGNoiseAugmentor(BaseEstimator, TransformerMixin):
    """
    Enhanced EEG noise augmentation transformer for MOABB pipelines.

    Supports four noise types:
    - 'dropout': Randomly zero out a percentage of EEG channels.
    - 'gaussian': Add Gaussian noise to a subset of channels.
    - 'eog': Add realistic EOG artifacts using learned template.
    - 'realistic_eog': Add realistic EOG artifacts using learned template (legacy).

    Parameters
    ----------
    noise_type : str
        One of ['dropout', 'gaussian', 'eog', 'realistic_eog'].
    intensity : float
        Noise severity/intensity. Meaning depends on noise_type:
        - For 'dropout': percentage of channels to drop (0-100).
        - For 'gaussian': scaling factor (D) for variance.
        - For 'eog': scaling factor for realistic artifacts.
    seed : int
        Random seed for reproducibility.
    eog_template_path : str, optional
        Path to generic EOG mixing template (default: 'eog_mixing_results/generic_eog_mixing_template.npz').
    montage_name : str, optional
        Target montage name for EOG interpolation (default: 'standard_1020').
    """

    def __init__(self, noise_type='dropout', intensity=10.0, seed=42, 
                 eog_template_path='eog_mixing_results/generic_eog_mixing_template.npz', montage_name='standard_1020'):
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.eog_template_path = eog_template_path
        self.montage_name = montage_name
        
        # Validate parameters
        if noise_type == 'eog' and eog_template_path is None:
            raise ValueError("eog_template_path is required for 'eog' noise type")
        
        # Check if template file exists when using EOG noise
        if noise_type == 'eog' and not Path(eog_template_path).exists():
            raise FileNotFoundError(f"EOG template file not found: {eog_template_path}")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Expected X to be a NumPy array of shape (n_epochs, n_channels, n_times).")

        if self.noise_type == 'dropout':
            return self._apply_channel_dropout(X)
        elif self.noise_type == 'gaussian':
            return self._apply_gaussian_noise(X)
        # elif self.noise_type == 'eog':
        #     return self._apply_eog_noise(X)
        elif self.noise_type == 'eog':
            return self._apply_realistic_eog_noise(X)
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
        """
        Legacy method: Intensity in [0,1] = fraction of epochs to contaminate.
        self.amp_scale controls artifact amplitude (separate from prevalence).
        Note: This method is kept for backward compatibility but is deprecated.
        """
        n_epochs, n_channels, n_times = data.shape
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg'] * n_channels)

        prevalence = int(n_epochs * (self.intensity / 100))
        contamination_idxs = np.random.choice(n_epochs, size=prevalence, replace=False)        
        data_aug = data.copy()
        for i in contamination_idxs:
            data_aug[i] = inject_scaled_eog_signal(
                data[i], info, scale_factor=50.0, seed=self.seed
            )            
        return data_aug

    def _apply_realistic_eog_noise(self, data):
        """
        Apply realistic EOG artifacts using the learned generic template.
        Intensity controls artifact strength (1.0 = realistic, >1.0 = stronger).
        """
        n_epochs, n_channels, n_times = data.shape
        
        ch_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg'] * n_channels)
        info.set_montage(self.montage_name)
        
        # Determine prevalence (fraction of epochs to contaminate)
        prevalence = int(n_epochs * (self.intensity / 100))
        contamination_idxs = np.random.choice(n_epochs, size=prevalence, replace=False)
        
        data_aug = data.copy()
        for i in contamination_idxs:
            # Inject realistic EOG artifacts
            contaminated_epoch = inject_realistic_eog_artifacts(
                data[i], info, self.eog_template_path, 
                montage_name=self.montage_name,
                intensity=self.intensity, 
                seed=self.seed + i,  # Different seed for each epoch
                apply_car=True
            )
            data_aug[i] = contaminated_epoch
            
        return data_aug


from sklearn.base import BaseEstimator, ClassifierMixin

# Replaces input with noise-augmented version
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

# Creates an augmented sample for every sample in the set X
class ConcatenatedNoiseAugmenter(ClassifierMixin, BaseEstimator):
    def __init__(self, base_pipeline, noise_type='dropout', intensity=25.0, seed=42, return_groups=False):
        self.base_pipeline = base_pipeline
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.return_groups = return_groups  # If True, returns group labels for splitting
        self.augmenter = EEGNoiseAugmentor(
            noise_type=self.noise_type,
            intensity=self.intensity,
            seed=self.seed
        )

    def concat_and_augment(self, X, y, groups=None):
        X_aug = self.augmenter.transform(X)
        X_combined = np.concatenate([X, X_aug], axis=0)
        y_combined = np.concatenate([y, y], axis=0)

        if groups is not None:
            groups_combined = np.concatenate([groups, groups], axis=0)
        else:
            groups_combined = np.concatenate([np.arange(len(X)), np.arange(len(X))], axis=0)

        self._X_train_ = X_combined
        self._y_train_ = y_combined
        self._groups_ = groups_combined
        return self.get_augmented_data()

    def fit(self, X, y, groups=None):
        self.base_pipeline.fit(X, y)
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

    def get_augmented_data(self):
        return self._X_train_, self._y_train_, self._groups_

    def predict(self, X):
        return self.base_pipeline.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.base_pipeline.score(X, y, sample_weight)

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

    def predict_proba(self, X):
        if hasattr(self.base_pipeline, 'predict_proba'):
            return self.base_pipeline.predict_proba(X)
        elif hasattr(self.base_pipeline[-1], 'predict_proba'):
            return self.base_pipeline[-1].predict_proba(X)
        else:
            raise NotImplementedError("Underlying model does not support predict_proba()")

# Test-time corruption classifier for robustness evaluation
class TestOnlyNoiseClassifier(ClassifierMixin, BaseEstimator):
    """
    Wrapper classifier that applies deterministic corruption only at evaluation time.
    
    This class is designed for robustness evaluation where:
    - Training and validation use clean data only
    - Test evaluation applies a grid of corruptions with deterministic seeding
    - Model selection is based on clean validation performance
    
    Parameters
    ----------
    base_pipeline : estimator or callable
        The underlying estimator or a factory function that returns it.
    corruption_plan : dict
        Dictionary defining corruption families and intensity grids.
    seed_base : int
        Base seed for deterministic corruption generation.
    freeze_seeds : bool, default=True
        Whether to use deterministic seeding for corruptions.
    score_clean_first : bool, default=True
        Whether to evaluate clean test set before applying corruptions.
    """
    
    def __init__(self, base_pipeline, corruption_plan, seed_base=42, 
                 freeze_seeds=True, score_clean_first=True):
        self.base_pipeline = base_pipeline
        self.corruption_plan = corruption_plan
        self.seed_base = seed_base
        self.freeze_seeds = freeze_seeds
        self.score_clean_first = score_clean_first
        self.is_fitted_ = False
        
    def fit(self, X, y):
        """Fit the base pipeline on clean data."""
        if callable(self.base_pipeline):
            self.base_pipeline = self.base_pipeline()
        
        self.base_pipeline.fit(X, y)
        self.is_fitted_ = True
        
        # Expose fitted classes_ attribute from the wrapped classifier
        if hasattr(self.base_pipeline, "classes_"):
            self.classes_ = self.base_pipeline.classes_
        elif hasattr(self.base_pipeline[-1], "classes_"):
            self.classes_ = self.base_pipeline[-1].classes_
        else:
            raise AttributeError("Base pipeline does not expose `classes_` after fit")
        
        return self
    
    def predict(self, X):
        """Predict using the base pipeline."""
        return self.base_pipeline.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """Score using the base pipeline."""
        return self.base_pipeline.score(X, y, sample_weight)
    
    def predict_proba(self, X):
        """Predict probabilities using the base pipeline."""
        if hasattr(self.base_pipeline, 'predict_proba'):
            return self.base_pipeline.predict_proba(X)
        elif hasattr(self.base_pipeline[-1], 'predict_proba'):
            return self.base_pipeline[-1].predict_proba(X)
        else:
            raise NotImplementedError("Underlying model does not support predict_proba()")
    
    def evaluate_on_corruptions(self, test_data, test_labels, metadata):
        """
        Evaluate the model on a grid of corruptions.
        
        Parameters
        ----------
        test_data : array-like
            Test data to evaluate on.
        test_labels : array-like
            Test labels.
        metadata : dict
            Metadata containing subject_id, session, fold_id, etc.
        
        Returns
        -------
        dict
            Dictionary containing evaluation results for each corruption family and intensity.
        """
        results = {
            'per_example': [],
            'per_family': [],
            'clean_metrics': {}
        }
        
        # Get test fold ID and subject ID for seed derivation
        test_fold_id = metadata.get('fold_id', 0)
        subject_id = metadata.get('subject_id', metadata.get('subject', 0))
        
        # Evaluate clean test set first if requested
        if self.score_clean_first:
            clean_score = self.score(test_data, test_labels)
            results['clean_metrics']['clean_score'] = clean_score
            
            # Add clean result to per-example results
            clean_result = {
                'family': 'clean',
                'intensity': 0.0,
                'seed_used': self.seed_base,
                'metric_name': 'score',
                'metric_value': clean_score,
                'clean_metric': clean_score,
                'relative_drop': 0.0,
                'test_fold_id': test_fold_id,
                'subject_id': subject_id
            }
            results['per_example'].append(clean_result)
        
        # Build corruption grid from plan
        from augmentation.corruption_utils import build_corruption_grid
        corruption_grid = build_corruption_grid(self.corruption_plan)
        
        # Evaluate on each corruption
        for family, intensity, params in corruption_grid:
            if family == 'clean':  # Skip clean, already handled
                continue
                
            # Derive deterministic seed
            from augmentation.corruption_utils import derive_corruption_seed
            corruption_seed = derive_corruption_seed(
                self.seed_base, family, intensity, test_fold_id, subject_id
            )
            
            # Apply corruption
            from augmentation.corruption_utils import apply_corruption
            corrupted_data = apply_corruption(
                test_data, family, intensity, corruption_seed, params
            )
            
            # Evaluate on corrupted data
            corrupted_score = self.score(corrupted_data, test_labels)
            
            # Compute relative drop
            from augmentation.corruption_utils import compute_relative_drop
            relative_drop = compute_relative_drop(clean_score, corrupted_score)
            
            # Store result
            result = {
                'family': family,
                'intensity': intensity,
                'seed_used': corruption_seed,
                'metric_name': 'score',
                'metric_value': corrupted_score,
                'clean_metric': clean_score,
                'relative_drop': relative_drop,
                'test_fold_id': test_fold_id,
                'subject_id': subject_id
            }
            results['per_example'].append(result)
        
        # Compute per-family summaries
        from augmentation.corruption_utils import compute_aurc
        families = set(r['family'] for r in results['per_example'] if r['family'] != 'clean')
        
        for family in families:
            family_results = [r for r in results['per_example'] if r['family'] == family]
            intensities = [r['intensity'] for r in family_results]
            metrics = [r['metric_value'] for r in family_results]
            
            # Sort by intensity for AURC computation
            sorted_pairs = sorted(zip(intensities, metrics))
            sorted_intensities, sorted_metrics = zip(*sorted_pairs)
            
            aurc = compute_aurc(sorted_intensities, sorted_metrics)
            worst_case = min(sorted_metrics)
            mean_relative_drop = np.mean([r['relative_drop'] for r in family_results])
            
            family_summary = {
                'family': family,
                'aurc': aurc,
                'worst_case_metric': worst_case,
                'mean_relative_drop': mean_relative_drop,
                'n_points': len(family_results)
            }
            results['per_family'].append(family_summary)
        
        return results
    
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
