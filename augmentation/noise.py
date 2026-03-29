# augmentation/noise.py
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import mne
from mne.simulation import add_eog
from mne.io import RawArray
from mne.channels import make_dig_montage, make_standard_montage
from mne.preprocessing import compute_current_source_density
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# MOABB Shin2017A: 30 EEG channels in acquisition order (dataset docstring / BrainAmp cap).
SHIN2017A_EEG_30 = [
    "AFp1", "AFp2", "AFF1h", "AFF2h", "AFF5h", "AFF6h", "F3", "F4", "F7", "F8",
    "FCC3h", "FCC4h", "FCC5h", "FCC6h", "T7", "T8", "Cz", "CCP3h", "CCP4h", "CCP5h",
    "CCP6h", "Pz", "P3", "P4", "P7", "P8", "PPO1h", "PPO2h", "POO1", "POO2",
]

# MOABB Yang2025 (Neuracle 59 EEG; same order as moabb.datasets.yang2025._CH_NAMES_EEG).
YANG2025_EEG_59 = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]

# 19-channel layout matching generic_eog_mixing_template.npz training layout.
_EOG_TEMPLATE_SOURCE_CHANNELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


@lru_cache(maxsize=8)
def _get_standard_montage_cached(montage_name: str):
    """Cache MNE standard montages; avoids thousands of redundant parses in EOG mixing."""
    return make_standard_montage(montage_name)


# === Enhanced EOG Injection Functions ===

@lru_cache(maxsize=8)
def _load_generic_eog_template_cached(resolved_path: str) -> Dict[str, Any]:
    """Load npz once per path; arrays copied so callers cannot corrupt the cache."""
    data = np.load(resolved_path)
    return {
        "mixing_matrix": np.asarray(data["mixing_matrix"], dtype=np.float64).copy(),
        "veog_std": float(np.asarray(data["veog_std"]).squeeze()),
        "heog_std": float(np.asarray(data["heog_std"]).squeeze()),
        "target_rms_median": float(np.asarray(data["target_rms_median"]).squeeze()),
    }


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
        p = str(Path(template_path).resolve())
        return _load_generic_eog_template_cached(p)
    except Exception as e:
        raise ValueError(f"Failed to load EOG template from {template_path}: {e}")
 
def interpolate_eog_topography_to_montage(source_montage, target_montage, source_matrix):
    """
    Interpolate EOG topography from source montage to target montage using spherical splines.
    
    Parameters
    ----------
    source_montage : str
    Source montage name (e.g., 'standard_1020')
    target_montage : str or DigMontage
    Target montage name (e.g., 'biosemi64') or montage object
    source_matrix : np.ndarray
    Source mixing matrix (n_source_channels, n_regressors)
    
    Returns
    -------
    np.ndarray
    Interpolated mixing matrix (n_target_channels, n_regressors)
    """
    source_montage_obj = (
        _get_standard_montage_cached(source_montage)
        if isinstance(source_montage, str)
        else source_montage
    )

    if isinstance(target_montage, str):
        target_montage_obj = _get_standard_montage_cached(target_montage)
    else:
        target_montage_obj = target_montage

    source_pos = source_montage_obj.get_positions()["ch_pos"]
    target_pos = target_montage_obj.get_positions()["ch_pos"]

    expected_source_channels = list(_EOG_TEMPLATE_SOURCE_CHANNELS)

    filtered_source_pos = {ch: source_pos[ch] for ch in expected_source_channels if ch in source_pos}

    target_channels = list(target_pos.keys())

    source_coords = np.array([filtered_source_pos[ch] for ch in filtered_source_pos])
    target_coords = np.array([target_pos[ch] for ch in target_channels])

    if len(source_coords) != source_matrix.shape[0]:
        raise ValueError(
            f"Source matrix has {source_matrix.shape[0]} channels but filtered montage has {len(source_coords)} channels"
        )

    # Nearest-neighbor: same as legacy double loop (closest source row per target channel).
    diff = target_coords[:, None, :] - source_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    closest_source_idx = np.argmin(dists, axis=1)
    return source_matrix[closest_source_idx, :]


# Alternate 10-20 labels used in some datasets vs MNE standard montages (positions align).
_EOG_MONTAGE_POSITION_ALIASES = {
    "T3": ("T7",),
    "T4": ("T8",),
    "T5": ("P7",),
    "T6": ("P8",),
}


def _channel_position_in_montage(ch_name: str, montage_name: str):
    """Return 3D head coords for ``ch_name`` in ``make_standard_montage(montage_name)``, with aliases."""
    full = _get_standard_montage_cached(montage_name)
    pos = full.get_positions()["ch_pos"]
    if ch_name in pos:
        return pos[ch_name]
    for alt in _EOG_MONTAGE_POSITION_ALIASES.get(ch_name, ()):
        if alt in pos:
            return pos[alt]
    return None


def make_ordered_montage_subset(ch_names, montage_name="standard_1005"):
    """
    Build a DigMontage with exactly ``ch_names`` (in order) for EOG topography interpolation.

    Returns None if any channel has no position in the montage (even after aliases).
    """
    ch_pos = {}
    for ch in ch_names:
        xyz = _channel_position_in_montage(ch, montage_name)
        if xyz is None:
            return None
        ch_pos[ch] = xyz
    return make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def build_eog_mixing_matrix(template, current_ch_names, montage_name):
    """
    Build (n_channels, 2) mixing matrix for [VEOG, HEOG] matching ``current_ch_names`` order.

    Uses the 19-channel template directly when the layout matches; otherwise interpolates
    in 3D from standard_1020 to the ordered subset of ``montage_name``.
    """
    if len(current_ch_names) == 19 and all(
        ch in current_ch_names for ch in _EOG_TEMPLATE_SOURCE_CHANNELS
    ):
        return template["mixing_matrix"]
    target_montage = make_ordered_montage_subset(current_ch_names, montage_name)
    if target_montage is None:
        raise ValueError(
            f"EOG mixing: could not place {len(current_ch_names)} channels on montage {montage_name!r}. "
            "Check channel names match the dataset layout."
        )
    return interpolate_eog_topography_to_montage(
        "standard_1020", target_montage, template["mixing_matrix"]
    )


def _eog_needs_interpolation(current_ch_names):
    """True if channel layout differs from the 19-channel template training layout."""
    if len(current_ch_names) != 19:
        return True
    ch_set = set(current_ch_names)
    return not all(ch in ch_set for ch in _EOG_TEMPLATE_SOURCE_CHANNELS)


def compute_scaled_eog_mixing_matrix(template, info, montage_name, artifact_scale_factor):
    """
    Build (n_channels, 2) mixing matrix scaled by ``artifact_scale_factor``.

    Matches the non-preloaded path in ``inject_realistic_eog_artifacts_with_coverage``.
    """
    ch_names = list(info.ch_names)
    if not _eog_needs_interpolation(ch_names):
        mm = template["mixing_matrix"]
    else:
        mm = build_eog_mixing_matrix(template, ch_names, montage_name)
    return np.asarray(mm, dtype=np.float64) * float(artifact_scale_factor)


def generate_realistic_eog_regressors(n_times, sfreq, template_stats, seed=42, allow_boundary_intersection=True):
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
    allow_boundary_intersection : bool
    If True, allows blinks to start before sample start or end after sample end,
    creating partial blinks that intersect with sample boundaries for more variability
    
    Returns
    -------
    tuple
    (veog_tc, heog_tc) - VEOG and HEOG time courses in Volts
    """
    rng = np.random.RandomState(seed)
    
    # Blink template parameters
    blink_duration_ms = 200
    blink_peak_ms = 80
    blink_frequency = 0.1 # 10% of time contains blinks
    
    # Convert to samples
    blink_duration = int(blink_duration_ms * sfreq / 1000)
    blink_peak = int(blink_peak_ms * sfreq / 1000)
    
    # Generate blink template (smooth gaussian-like)
    t = np.arange(blink_duration)
    blink_template = np.exp(-((t - blink_peak) / (0.35 * blink_peak + 1e-9)) ** 2)
    blink_template = blink_template / np.max(blink_template)
    
    # Place blinks randomly with enhanced variability
    n_blinks = int(n_times * blink_frequency)
    
    if allow_boundary_intersection:
        # Allow blinks to start before sample start or extend beyond sample end
        # This creates more realistic variability where blinks can be partial
        blink_starts = rng.choice(n_times + blink_duration, size=n_blinks, replace=False) - blink_duration
    else:
        # Original behavior: blinks must be completely contained within sample
        blink_starts = rng.choice(n_times - blink_duration, size=n_blinks, replace=False)
    
    # Initialize time courses
    veog_tc = np.zeros(n_times)
    heog_tc = np.zeros(n_times)
    
    # Add blinks with boundary intersection support
    for start in blink_starts:
        end = start + blink_duration
        
        # Determine the valid range within the sample
        valid_start = max(0, start)
        valid_end = min(n_times, end)
        
        if valid_start >= valid_end:
            continue  # Skip if blink is completely outside sample bounds
            
        # Calculate template indices for the valid range
        template_start = valid_start - start
        template_end = template_start + (valid_end - valid_start)
        
        # VEOG: primary blink component
        veog_tc[valid_start:valid_end] += blink_template[template_start:template_end]
        
        # HEOG: smaller lateral component (random direction)
        lateral_amplitude = 0.2 * rng.uniform(0.5, 1.5) # 20% of VEOG, with variability
        direction = rng.choice([-1, 1]) # random left/right
        heog_tc[valid_start:valid_end] += direction * lateral_amplitude * blink_template[template_start:template_end]
    
    # Calibrate to match template statistics (realistic EOG artifact strength)
    veog_tc = veog_tc / (np.std(veog_tc) + 1e-12) * template_stats['veog_std']
    heog_tc = heog_tc / (np.std(heog_tc) + 1e-12) * template_stats['heog_std']
    
    return veog_tc, heog_tc

def inject_realistic_eog_artifacts(data, info, template_path, montage_name='standard_1005', 
 intensity=1.0, seed=42, apply_car=True, allow_boundary_intersection=True):
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
        info.set_montage(montage_name, on_missing='warn')
        current_montage = info.get_montage()
    
    current_ch_names = list(info.ch_names)
    if _eog_needs_interpolation(current_ch_names):
        mixing_matrix = build_eog_mixing_matrix(template, current_ch_names, montage_name)
    else:
        mixing_matrix = template['mixing_matrix']
    
    # Generate realistic EOG regressors
    n_times = data_volts.shape[1]
    sfreq = info['sfreq']
    veog_tc, heog_tc = generate_realistic_eog_regressors(
    n_times, sfreq, template, seed=seed, allow_boundary_intersection=allow_boundary_intersection
    )
    
    # Project EOG artifacts to EEG space
    eog_regressors = np.vstack([veog_tc, heog_tc]) # (2, n_times)
    eog_artifacts = mixing_matrix @ eog_regressors # (n_channels, n_times)
    
    # Apply intensity scaling
    eog_artifacts *= intensity
    
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

    if absmax > 1.0: # heuristic threshold
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

def inject_realistic_eog_artifacts_with_coverage(data, info, template_path, montage_name='standard_1005', 
 temporal_coverage=0.1, seed=42, apply_car=True, artifact_scale_factor=15000.0, allow_boundary_intersection=True,
 include_slow_drift=True, include_microsaccades=True, include_blink_clusters=True,
 template_preloaded=None, scaled_mixing_matrix=None):
    """
    Inject realistic EOG artifacts using the learned generic mixing template with controlled temporal coverage.
    
    This function:
    1. Loads the generic EOG mixing template
    2. Interpolates to the target montage if different from training
    3. Generates realistic VEOG/HEOG time courses
    4. Places artifacts to achieve the desired temporal coverage
    5. Projects artifacts to EEG space using the mixing matrix
    
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
    temporal_coverage : float
    Desired fraction of time covered by EOG artifacts (0.0 to 1.0)
    seed : int
    Random seed for reproducibility
    apply_car : bool
    Whether to apply CAR before injection (should match training)
    artifact_scale_factor : float
    Scaling factor to make EOG artifacts more impactful (default: 15000.0)
    This addresses the units mismatch issue where EOG template values are ~1000x too small
    include_slow_drift : bool
    Whether to include slow eye drift components (0.3-2.5 Hz drifts) (default: True)
    include_microsaccades : bool
    Whether to include microsaccades during blinks (default: True)
    include_blink_clusters : bool
    Whether to include blink clusters (rapid successive blinks) (default: True)
    template_preloaded : dict, optional
        If provided, skip loading ``template_path`` from disk (same dict as ``load_generic_eog_template``).
    scaled_mixing_matrix : np.ndarray, optional
        If provided, use this (n_channels, 2) matrix already scaled by ``artifact_scale_factor``;
        skips interpolation/build. Typically passed with ``template_preloaded`` from
        ``compute_scaled_eog_mixing_matrix`` for batch speed.
    
    Returns
    -------
    np.ndarray
    Contaminated EEG data in same units as input
    """
    if template_preloaded is not None:
        template = template_preloaded
    else:
        template = load_generic_eog_template(template_path)
    
    # Ensure data is in Volts
    data_volts, unit = to_volts(data)
    is_microvolts = unit != "V"
    
    # Apply CAR if requested (should match training procedure)
    if apply_car:
        data_volts = data_volts - np.mean(data_volts, axis=0, keepdims=True)

    # Get current montage
    current_montage = info.get_montage()
    if current_montage is None:
    # Set default montage if none exists
        info.set_montage(montage_name, on_missing='warn')
        current_montage = info.get_montage()
    
    if scaled_mixing_matrix is not None:
        mixing_matrix = np.asarray(scaled_mixing_matrix, dtype=np.float64)
    else:
        current_ch_names = list(info.ch_names)
        if _eog_needs_interpolation(current_ch_names):
            mixing_matrix = build_eog_mixing_matrix(template, current_ch_names, montage_name)
        else:
            mixing_matrix = template['mixing_matrix']
        # CRITICAL FIX: Scale up the mixing matrix to make artifacts impactful
        # The original values were too small (10^-4 to 10^-6) compared to EEG signals (10^0)
        mixing_matrix = mixing_matrix * artifact_scale_factor
    
    # Generate realistic EOG regressors with controlled temporal coverage
    n_times = data_volts.shape[1]
    sfreq = info['sfreq']
    veog_tc, heog_tc = generate_realistic_eog_regressors_with_coverage(
        n_times, sfreq, template, temporal_coverage, seed=seed, allow_boundary_intersection=allow_boundary_intersection,
        include_slow_drift=include_slow_drift, include_microsaccades=include_microsaccades, 
        include_blink_clusters=include_blink_clusters
    )
    
    # Project EOG artifacts to EEG space
    eog_regressors = np.vstack([veog_tc, heog_tc]) # (2, n_times)
    eog_artifacts = mixing_matrix @ eog_regressors # (n_channels, n_times)
    
    # Debug assertions to verify artifacts are now substantial
    assert(np.sum(np.abs(eog_artifacts)) > 0)
    assert(np.sum(np.abs(mixing_matrix)) > 0)
    
    # Add artifacts to clean data
    contaminated_data = data_volts + eog_artifacts
    assert(np.sum(np.abs(contaminated_data)) != np.sum(np.abs(data_volts)))
    
    # Convert back to original units
    if is_microvolts:
        contaminated_data *= 1e6
    
    return contaminated_data

def generate_realistic_eog_regressors_with_coverage(n_times, sfreq, template_stats, temporal_coverage, seed=42, allow_boundary_intersection=True,
                                                    include_slow_drift=True, include_microsaccades=True, include_blink_clusters=True):
    """
    Generate realistic VEOG and HEOG time courses with controlled temporal coverage.
    
    Parameters
    ----------
    n_times : int
    Number of time points
    sfreq : float
    Sampling frequency in Hz
    template_stats : dict
    Template statistics from load_generic_eog_template()
    temporal_coverage : float
    Desired fraction of time covered by EOG artifacts (0.0 to 1.0)
    seed : int
    Random seed for reproducibility
    allow_boundary_intersection : bool
    If True, allows blinks to start before sample start or end after sample end,
    creating partial blinks that intersect with sample boundaries for more variability
    include_slow_drift : bool
    Whether to include slow eye drift components (0.3-2.5 Hz drifts) (default: True)
    include_microsaccades : bool
    Whether to include microsaccades during blinks (default: True)
    include_blink_clusters : bool
    Whether to include blink clusters (rapid successive blinks) (default: True)
    
    Returns
    -------
    tuple
    (veog_tc, heog_tc) - VEOG and HEOG time courses in Volts
    """
    rng = np.random.RandomState(seed)
    
    # Blink template parameters - use fixed duration for consistent behavior across intensities
    blink_duration_ms = 200  # Fixed duration for consistent temporal coverage calculation
    blink_peak_ms = 80  # Fixed peak position for consistent blink shape
    # Convert to samples
    blink_duration = int(blink_duration_ms * sfreq / 1000)
    blink_peak = int(blink_peak_ms * sfreq / 1000)
    
    # Generate blink template (smooth gaussian-like)
    t = np.arange(blink_duration)
    blink_template = np.exp(-((t - blink_peak) / (0.35 * blink_peak + 1e-9)) ** 2)
    blink_template = blink_template / np.max(blink_template)
    
    # Calculate how many blinks we need to achieve the desired temporal coverage
    # Each blink covers blink_duration samples
    # We want: (n_blinks * blink_duration) / n_times ≈ temporal_coverage
    
    # Calculate the target number of samples to cover
    target_samples_to_cover = int(n_times * temporal_coverage)
    
    # Calculate how many blinks we need
    n_blinks = max(1, target_samples_to_cover // blink_duration)
    
    # Ensure we don't exceed the maximum possible blinks
    if allow_boundary_intersection:
        # With boundary intersection, we can have more blinks since they can be partial
        max_possible_blinks = (n_times + blink_duration) // blink_duration
    else:
        # Original behavior: blinks must be completely contained
        max_possible_blinks = n_times // blink_duration
    n_blinks = min(n_blinks, max_possible_blinks)
    
    # Calculate actual temporal coverage achieved
    actual_coverage = (n_blinks * blink_duration) / n_times
    # print(f"Desired EOG coverage: {temporal_coverage}, Actual coverage: {actual_coverage}")
    
    # Place blinks to achieve the desired coverage with enhanced variability
    if n_blinks == 1:
        # Single blink in the middle
        if allow_boundary_intersection:
            # Allow the blink to be centered but potentially extend beyond boundaries
            blink_starts = [n_times // 2 - blink_duration // 2]
        else:
            # Original behavior: ensure blink is contained
            blink_starts = [max(0, min(n_times // 2 - blink_duration // 2, n_times - blink_duration))]
    else:
        if allow_boundary_intersection:
            # Distribute blinks with more variability, allowing boundary intersection
            # Use a mix of evenly distributed and random positioning
            n_even = n_blinks // 2
            n_random = n_blinks - n_even
            
            # Evenly distributed blinks
            even_starts = []
            if n_even > 0:
                spacing = n_times // (n_even + 1)
                even_starts = [spacing * (i + 1) - blink_duration // 2 for i in range(n_even)]
            
            # Random blinks that can intersect boundaries
            random_starts = rng.choice(n_times + blink_duration, size=n_random, replace=False) - blink_duration
            
            blink_starts = even_starts + list(random_starts)
        else:
            # Original behavior: distribute blinks evenly and ensure they don't go out of bounds
            spacing = n_times // (n_blinks + 1)
            blink_starts = [spacing * (i + 1) - blink_duration // 2 for i in range(n_blinks)]
            blink_starts = [max(0, min(start, n_times - blink_duration)) for start in blink_starts]
        
    # Initialize time courses
    veog_tc = np.zeros(n_times)
    heog_tc = np.zeros(n_times)
    
    # Add slow eye drift component for more realistic EOG (slow saccades and drifts)
    # These low-frequency components are common and challenging for models
    # FIXED: Make slow drift respect temporal coverage parameter
    if include_slow_drift and temporal_coverage > 0.2:  # Only add drift if significant coverage
        n_drift_components = rng.randint(1, 3)  # 1-2 drift components
        for _ in range(n_drift_components):
            drift_freq = rng.uniform(0.3, 2.5)  # 0.3-2.5 Hz drift
            drift_phase = rng.uniform(0, 2 * np.pi)
            drift_amplitude = rng.uniform(0.1, 0.25) * temporal_coverage  # Scale with temporal coverage
            
            # Only add drift during the covered time periods
            # Handle boundary case when temporal_coverage = 1.0
            if temporal_coverage >= 1.0:
                drift_start = 0
                drift_end = n_times
            else:
                drift_start = rng.randint(0, int(n_times * (1 - temporal_coverage)))
                drift_end = drift_start + int(n_times * temporal_coverage)
            
            t = np.arange(drift_end - drift_start) / sfreq
            drift_veog = drift_amplitude * np.sin(2 * np.pi * drift_freq * t + drift_phase)
            drift_heog = drift_amplitude * 0.7 * np.sin(2 * np.pi * drift_freq * t + drift_phase + rng.uniform(0, np.pi))
            
            veog_tc[drift_start:drift_end] += drift_veog
            heog_tc[drift_start:drift_end] += drift_heog
    
    # Add blinks with boundary intersection support
    for idx, start in enumerate(blink_starts):
        end = start + blink_duration
        
        # Determine the valid range within the sample
        valid_start = max(0, start)
        valid_end = min(n_times, end)
        
        if valid_start >= valid_end:
            continue  # Skip if blink is completely outside sample bounds
            
        # Calculate template indices for the valid range
        template_start = valid_start - start
        template_end = template_start + (valid_end - valid_start)
        
        # Add realistic amplitude variability - real blinks vary significantly in strength
        amplitude_multiplier = rng.uniform(0.6, 1.6)  # 60% to 160% of baseline
    
        # VEOG: primary blink component with variable amplitude
        veog_tc[valid_start:valid_end] += amplitude_multiplier * blink_template[template_start:template_end]
    
        # HEOG: smaller lateral component (random direction)
        lateral_amplitude = 0.25 * rng.uniform(0.5, 1.5) # 25% of VEOG, with variability
        direction = rng.choice([-1, 1]) # random left/right
        heog_tc[valid_start:valid_end] += direction * amplitude_multiplier * lateral_amplitude * blink_template[template_start:template_end]
        
        # Add microsaccades during some blinks (small eye movements that co-occur with blinks)
        if include_microsaccades and rng.random() < 0.4:  # 40% of blinks have microsaccades
            microsaccade_amplitude = rng.uniform(0.08, 0.15)
            microsaccade_direction = rng.choice([-1, 1])
            # Microsaccade is brief and occurs during the blink
            micro_duration = min(int(0.05 * sfreq), valid_end - valid_start)  # 50ms or less
            micro_start = valid_start + (valid_end - valid_start) // 3
            micro_end = min(micro_start + micro_duration, valid_end)
            heog_tc[micro_start:micro_end] += microsaccade_direction * microsaccade_amplitude
        
        # # Add blink clusters (multiple blinks in rapid succession) - common when tired/dry eyes
        if include_blink_clusters and rng.random() < 0.25 and idx < len(blink_starts) - 1:  # 25% chance, not on last blink
            n_cluster_blinks = rng.choice([1, 2])  # 1-2 additional blinks
            cluster_spacing = int(rng.uniform(0.25, 0.5) * sfreq)  # 250-500ms between blinks
            
            for j in range(n_cluster_blinks):
                cluster_start = start + (j + 1) * cluster_spacing
                cluster_end = cluster_start + blink_duration
                
                cluster_valid_start = max(0, cluster_start)
                cluster_valid_end = min(n_times, cluster_end)
                
                if cluster_valid_start >= cluster_valid_end or cluster_valid_start >= n_times:
                    continue
                
                cluster_template_start = cluster_valid_start - cluster_start
                cluster_template_end = cluster_template_start + (cluster_valid_end - cluster_valid_start)
                
                # Cluster blinks are often weaker
                cluster_amplitude = amplitude_multiplier * rng.uniform(0.5, 0.9)
                veog_tc[cluster_valid_start:cluster_valid_end] += cluster_amplitude * blink_template[cluster_template_start:cluster_template_end]
                
                cluster_lateral = 0.25 * rng.uniform(0.5, 1.5)
                cluster_direction = rng.choice([-1, 1])
                heog_tc[cluster_valid_start:cluster_valid_end] += cluster_direction * cluster_amplitude * cluster_lateral * blink_template[cluster_template_start:cluster_template_end]
    
    # Calibrate to match template statistics (realistic EOG artifact strength)
    veog_tc = veog_tc / (np.std(veog_tc) + 1e-12) * template_stats['veog_std']
    heog_tc = heog_tc / (np.std(heog_tc) + 1e-12) * template_stats['heog_std']
    
    # Apply temporal coverage scaling to maintain intensity effect
    # Higher temporal coverage should result in stronger overall contamination
    coverage_amplitude_multiplier = 1.0 + (temporal_coverage - 0.1) * 2.0  # Scale from 0.8 to 2.8 as coverage goes from 0.1 to 1.0
    veog_tc *= coverage_amplitude_multiplier
    heog_tc *= coverage_amplitude_multiplier
    
    return veog_tc, heog_tc

# === Enhanced EEGNoiseAugmentor Class ===

class EEGNoiseAugmentor(BaseEstimator, TransformerMixin):
    """
    Enhanced EEG noise augmentation transformer for MOABB pipelines.

    Supports eight noise types:
    - 'dropout': Randomly zero out a percentage of EEG channels.
    - 'gaussian': Add Gaussian noise with magnitude-aware scaling (i.i.d.).
    - 'eog': Add realistic EOG artifacts using learned template.
    - 'realistic_eog': Add realistic EOG artifacts using learned template (legacy).
    - 'spike': Add transient spike artifacts with configurable intensity and duration.
    - 'spatial_gaussian': Spatially correlated Gaussian (10-20 montage, exp(-d/l) cov).
    - 'ar1_drift': AR(1) temporal drift (rho=0.97), unit variance per channel.
    - 'emg_band': Band-limited [20, 80] Hz noise (EMG-like).

    Parameters
    ----------
    noise_type : str
    One of ['dropout', 'gaussian', 'eog', 'realistic_eog', 'spike', 'spatial_gaussian', 'ar1_drift', 'emg_band'].
    intensity : float
    Noise severity/intensity. Meaning depends on noise_type and mode:
    - For 'dropout': percentage of channels to drop (0-100).
    - For 'gaussian': noise-to-signal ratio as percentage (10.0 = 10% noise relative to signal RMS).
    - For 'eog': temporal coverage of EOG artifacts (10% = 10% of the time covered by artifacts)
    - For 'spike': temporal coverage of spike artifacts (10% = 10% of the time covered by spikes)
    seed : int
    Random seed for reproducibility.
    eog_template_path : str, optional
    Path to generic EOG mixing template (default: 'notebooks/eog_mixing_results/generic_eog_mixing_template.npz').
    montage_name : str, optional
    Target montage name for EOG interpolation (default: 'standard_1020').
    artifact_scale_factor : float, optional
    Scaling factor to make EOG artifacts more impactful (default: 15000.0).
    This addresses the units mismatch issue where EOG template values are ~1000x too small.
    The original EOG signals had RMS ~75 µV, but template shows ~0.07 µV.
    Increase this value to make EOG contamination more severe.
    use_improved_gaussian : bool, optional
    Whether to use the improved magnitude-aware Gaussian noise implementation (default: True).
    If False, uses the original fixed-scaling implementation for backward compatibility.
    allow_boundary_intersection : bool, optional
    For EOG noise: whether to allow blinks to start before sample start or end after sample end,
    creating partial blinks that intersect with sample boundaries for more variability (default: True).
    This significantly increases the variability of EOG artifacts and makes them more challenging for models.
    include_slow_drift : bool, optional
    For EOG noise: whether to include slow eye drift components (0.3-2.5 Hz drifts) (default: True).
    include_microsaccades : bool, optional
    For EOG noise: whether to include microsaccades during blinks (default: True).
    include_blink_clusters : bool, optional
    For EOG noise: whether to include blink clusters (rapid successive blinks) (default: True).
    eog_sfreq : float, optional
    Sampling rate (Hz) used for MNE info and EOG time-course generation. Defaults to 250 when
    omitted (legacy); pass the dataset native rate (e.g. from ``get_dataset_sampling_rate``) for
    correct blink/artifact timing on non-250 Hz data (Shin2017A=200, Yang2025/Lee2019=1000, etc.).
    """

    def __init__(self, noise_type='dropout', intensity=10.0, seed=42,
    eog_template_path='notebooks/eog_mixing_results/generic_eog_mixing_template.npz', montage_name='standard_1020',
    artifact_scale_factor=10000.0, use_improved_gaussian=True, allow_boundary_intersection=True,
    include_slow_drift=True, include_microsaccades=True, include_blink_clusters=True,
    spatial_ell_multiplier=1.0, emg_f_low=20.0, emg_f_high=80.0, emg_use_envelope=False, ar1_rho=0.97,
    gain_drift_rho=0.995, offset_drift_rho=0.995, jitter_sfreq=250.0, jitter_max_ms=None, spatial_dropout_cluster_size=0.25,
    eog_sfreq: Optional[float] = None):
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.eog_template_path = eog_template_path
        self.montage_name = montage_name
        self.eog_sfreq = eog_sfreq
        self.artifact_scale_factor = artifact_scale_factor
        self.use_improved_gaussian = use_improved_gaussian
        self.allow_boundary_intersection = allow_boundary_intersection
        self.include_slow_drift = include_slow_drift
        self.include_microsaccades = include_microsaccades
        self.include_blink_clusters = include_blink_clusters
        self.spatial_ell_multiplier = float(spatial_ell_multiplier)
        self.emg_f_low = float(emg_f_low)
        self.emg_f_high = float(emg_f_high)
        self.emg_use_envelope = bool(emg_use_envelope)
        self.ar1_rho = float(ar1_rho)
        self.gain_drift_rho = float(gain_drift_rho)
        self.offset_drift_rho = float(offset_drift_rho)
        self.jitter_sfreq = float(jitter_sfreq)
        self.jitter_max_ms = float(jitter_max_ms) if jitter_max_ms is not None else None
        self.spatial_dropout_cluster_size = float(spatial_dropout_cluster_size)

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
            # Note: Reverting for a test on 10/1/2025
            # return self._apply_gaussian_noise(X)
            return self._improved_apply_gaussian_noise(X)
        elif self.noise_type == 'eog':
            return self._apply_realistic_eog_noise(X)
        elif self.noise_type == 'spike':
            return self._apply_spike_noise(X)
        elif self.noise_type == 'spatial_gaussian':
            return self._apply_spatial_gaussian_noise(X)
        elif self.noise_type == 'ar1_drift':
            return self._apply_ar1_drift(X)
        elif self.noise_type == 'emg_band':
            return self._apply_emg_band_noise(X)
        elif self.noise_type == 'gain_drift':
            return self._apply_gain_drift(X)
        elif self.noise_type == 'offset_drift':
            return self._apply_offset_drift(X)
        elif self.noise_type == 'temporal_jitter':
            return self._apply_temporal_jitter(X)
        elif self.noise_type == 'spatial_dropout':
            return self._apply_spatial_dropout(X)
        elif self.noise_type == 'ar1_plus_gain_drift':
            return self._apply_ar1_plus_gain_drift(X)
        elif self.noise_type == 'ar1_plus_offset_drift':
            return self._apply_ar1_plus_offset_drift(X)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def _apply_channel_dropout(self, data):
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        n_drop = int(n_channels * self.intensity / 100)
        n_drop = max(1, n_drop) if self.intensity > 0 else 0
        n_drop = min(n_drop, n_channels)
        data_aug = data.copy()
        for i in range(n_epochs):
            drop_idxs = rng.choice(n_channels, size=n_drop, replace=False)
            data_aug[i, drop_idxs, :] = 0.0
        return data_aug

    def _apply_gaussian_noise(self, data):
        """Original implementation: Fixed scaling factor without magnitude awareness."""
        np.random.seed(self.seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()
        for i in range(n_epochs):
            noise = np.random.randn(n_channels, n_times)
            data_aug[i] += self.intensity * noise
        return data_aug

    def _improved_apply_gaussian_noise(self, data):
        """
        Improved Gaussian noise application with magnitude-aware scaling.

        The intensity parameter now represents the percentage of channels to contaminate
        in each epoch (e.g., intensity=10.0 means 10% of channels per epoch will have noise added).
        The noise itself is scaled to the signal RMS (noise_scale = 1.0 * signal_rms).

        Parameters
        ----------
        data : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times)

        Returns
        -------
        np.ndarray
            Data with magnitude-aware Gaussian noise added to a proportion of channels
        """
        np.random.seed(self.seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()

        # Calculate overall signal RMS once (assuming consistent units within dataset)
        signal_rms = np.sqrt(np.mean(data**2))

        # Set noise scale to 10.0 * signal_rms
        noise_scale = 4.0 * signal_rms
        # Use intensity to gradually ramp up the noise scale
        noise_scale *= (self.intensity / 100.0)

        # Determine number of channels to contaminate per epoch
        n_contam = int(np.round(n_channels * self.intensity / 100.0))
        n_contam = max(1, n_contam) if self.intensity > 0 else 0
        n_contam = min(n_contam, n_channels)

        for i in range(n_epochs):
            if n_contam == 0:
                continue
            contam_idxs = np.random.choice(n_channels, size=n_contam, replace=False)
            noise = np.random.randn(n_contam, n_times)
            data_aug[i, contam_idxs, :] += noise_scale * noise

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

    def _generate_channel_names(self, n_channels):
        """
        Generate appropriate channel names based on the number of channels.
        
        Args:
        n_channels (int): Number of channels in the dataset
        
        Returns:
        list: List of channel names
        """
        if n_channels == 22:
            # BNCI2014_001 dataset
            return [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
            ]
        elif n_channels == 32:
            # BI2015a dataset (32 EEG channels following standard_1020 system)
            return [
            'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'PO7', 'O1', 'Oz', 'O2', 'PO8',
            'PO9', 'PO10'
            ]
        elif n_channels == 62:
            # Lee2019_MI / Lee2019_SSVEP (62 EEG, 10-10 / extended layout)
            return [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'Oz',
            'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
            'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6',
            'P1', 'P2', 'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8',
            'F1', 'F2', 'AF3', 'AF4', 'AF7', 'AF8',
            'FT7', 'FT8', 'TP7', 'TP8', 'PO9', 'PO10', 'P9', 'P10',
            'F9', 'F10', 'FT9', 'FT10'
            ]
        elif n_channels == 30:
            # Shin2017A (30 EEG; MOABB BrainAmp 10-5 cap, acquisition order)
            return list(SHIN2017A_EEG_30)
        elif n_channels == 59:
            # Yang2025 (59 EEG; Neuracle 10-10, MOABB order)
            return list(YANG2025_EEG_59)
        else:
            # Generic fallback - generate channel names
            return [f'EEG{i+1:03d}' for i in range(n_channels)]

    def _resolve_eog_montage_name(self, n_channels: int) -> str:
        """Montage used for set_montage + EOG mixing interpolation (must match channel naming)."""
        if n_channels in (30, 59):
            return "standard_1005"
        if n_channels in (22, 32, 62):
            return "standard_1020"
        return self.montage_name

    def _apply_realistic_eog_noise(self, data):
        """
        Apply realistic EOG artifacts using the learned generic template.
        
        For test_perturb mode: intensity controls temporal coverage of EOG artifacts (10% = 10% of time covered by artifacts)
        For other modes: intensity controls prevalence (percentage of epochs to contaminate)
        """
        n_epochs, n_channels, n_times = data.shape
        
        # Generate channel names based on the number of channels
        ch_names = self._generate_channel_names(n_channels)
        resolved_montage = self._resolve_eog_montage_name(n_channels)
        sfreq_eog = float(self.eog_sfreq) if self.eog_sfreq is not None else 250.0
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq_eog, ch_types=['eeg'] * n_channels)
        info.set_montage(resolved_montage, on_missing='warn')
        
        # FIXED: Always use intensity to control temporal coverage for consistent behavior
        # This makes EOG behave like true transient artifacts regardless of mode
        prevalence = n_epochs  # Always contaminate all epochs
        # Convert intensity to temporal coverage (1% -> 0.01, 100% -> 1.0)
        # Use a more gradual scaling for better control
        temporal_coverage = min(1.0, self.intensity / 100.0)
        contamination_idxs = np.random.choice(n_epochs, size=prevalence, replace=False)

        template = load_generic_eog_template(self.eog_template_path)
        scaled_mixing_matrix = compute_scaled_eog_mixing_matrix(
            template, info, resolved_montage, self.artifact_scale_factor
        )

        data_aug = data.copy()
        for i in contamination_idxs:
            # temporal_coverage = rng.uniform(0.1, 0.9)
            # Inject realistic EOG artifacts with controlled temporal coverage
            contaminated_epoch = inject_realistic_eog_artifacts_with_coverage(
                data[i], info, self.eog_template_path, 
                montage_name=resolved_montage,
                temporal_coverage=temporal_coverage, # Control temporal coverage
                seed=self.seed + i, # Different seed for each epoch
                apply_car=True,
                artifact_scale_factor=self.artifact_scale_factor, # Use the scaling factor
                allow_boundary_intersection=self.allow_boundary_intersection, # Pass boundary intersection setting
                include_slow_drift=self.include_slow_drift, # Control slow drift
                include_microsaccades=self.include_microsaccades, # Control microsaccades
                include_blink_clusters=self.include_blink_clusters, # Control blink clusters
                template_preloaded=template,
                scaled_mixing_matrix=scaled_mixing_matrix,
            )
            data_aug[i] = contaminated_epoch
            
        return data_aug

    def _apply_spike_noise(self, data):
        """
        Apply transient spike artifacts with configurable intensity and duration.
        
        intensity controls temporal coverage of spike artifacts (10% = 10% of time covered by spikes)
        """
        rng = np.random.RandomState(self.seed)
        n_epochs, n_channels, n_times = data.shape
        
        # Convert intensity to temporal coverage (1% -> 0.01, 100% -> 1.0)
        temporal_coverage = min(1.0, self.intensity / 100.0)
        
        data_aug = data.copy()
        for i in range(n_epochs):
            # Calculate how many spikes we need to achieve the desired temporal coverage
            spike_duration = rng.choice([5, 10, 15, 20])  # 5-20 samples per spike
            target_samples_to_cover = int(n_times * temporal_coverage)
            n_spikes = max(1, target_samples_to_cover // spike_duration)
            n_spikes = min(n_spikes, n_times // spike_duration)  # Don't exceed bounds
            
            # Generate spike locations
            spike_starts = rng.choice(n_times - spike_duration, size=n_spikes, replace=False)
            
            # Apply spikes to random channels
            for spike_start in spike_starts:
                spike_end = spike_start + spike_duration
                # Random channel selection (could be all channels or subset)
                affected_channels = rng.choice(n_channels, size=rng.randint(1, n_channels//2 + 1), replace=False)
                
                # Generate spike amplitude relative to signal
                signal_rms = np.sqrt(np.mean(data[i, affected_channels, spike_start:spike_end] ** 2))
                spike_amplitude = rng.uniform(2.0, 8.0) * signal_rms  # 2-8x signal RMS
                
                # Create spike shape (sharp rise, exponential decay)
                spike_shape = np.zeros(spike_duration)
                peak_idx = spike_duration // 4  # Peak at 1/4 through spike
                spike_shape[:peak_idx] = np.linspace(0, 1, peak_idx)  # Sharp rise
                spike_shape[peak_idx:] = np.exp(-np.linspace(0, 3, spike_duration - peak_idx))  # Exponential decay
                
                # Apply spike to affected channels
                for ch in affected_channels:
                    data_aug[i, ch, spike_start:spike_end] += spike_amplitude * spike_shape
        
        return data_aug

    def _get_spatial_covariance_cholesky(self, n_channels, ch_names):
        """
        Build spatial covariance Sigma_s from 10-20 montage distances and return
        Cholesky L such that Sigma_s = L L^T. Cached per (n_channels, tuple(ch_names)).
        Falls back to identity (i.i.d.) when montage positions are unavailable.
        """
        cache = getattr(self, "_spatial_gaussian_chol_cache", None)
        if cache is None:
            self._spatial_gaussian_chol_cache = {}
            cache = self._spatial_gaussian_chol_cache
        ell_mult = getattr(self, "spatial_ell_multiplier", 1.0)
        key = (n_channels, tuple(ch_names), ell_mult)
        if key in cache:
            return cache[key]

        montage = make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        positions = []
        for name in ch_names:
            if name in ch_pos:
                positions.append(ch_pos[name])
            else:
                # Fallback: cannot build spatial cov; return None to use i.i.d.
                self._spatial_gaussian_chol_cache[key] = None
                return None
        positions = np.array(positions)
        if len(positions) != n_channels:
            self._spatial_gaussian_chol_cache[key] = None
            return None

        # Pairwise Euclidean distances
        d = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                d[i, j] = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
        off_diag = d[np.triu_indices(n_channels, k=1)]
        ell = float(np.median(off_diag)) if off_diag.size > 0 else 1.0
        if ell <= 0:
            ell = 1.0
        ell = ell * float(getattr(self, "spatial_ell_multiplier", 1.0))
        Sigma_s = np.exp(-d / ell)
        # Ensure positive definite
        Sigma_s += 1e-8 * np.eye(n_channels)
        try:
            L = np.linalg.cholesky(Sigma_s)
        except np.linalg.LinAlgError:
            self._spatial_gaussian_chol_cache[key] = None
            return None
        self._spatial_gaussian_chol_cache[key] = L
        return L

    def _apply_spatial_gaussian_noise(self, data):
        """
        Spatially correlated Gaussian noise: epsilon ~ N(0, Sigma_s) with
        (Sigma_s)_{ij} = exp(-d_ij / ell), ell = median inter-electrode distance.
        intensity = alpha * alpha_max (scale factor). Falls back to i.i.d. Gaussian
        when montage positions are unavailable.
        """
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        ch_names = self._generate_channel_names(n_channels)
        L = self._get_spatial_covariance_cholesky(n_channels, ch_names)
        data_aug = data.copy()
        for i in range(n_epochs):
            Z = rng.standard_normal((n_channels, n_times))
            if L is not None:
                # epsilon = L @ Z so cov(epsilon) = L I L^T = Sigma_s
                epsilon = (L @ Z).astype(data.dtype, copy=False)
            else:
                epsilon = Z.astype(data.dtype, copy=False)
            data_aug[i] += self.intensity * epsilon
        return data_aug

    def _apply_ar1_drift(self, data):
        """
        Temporally correlated drift: epsilon_t = rho * epsilon_{t-1} + eta_t,
        eta_t ~ N(0, sigma_eta^2), rho from self.ar1_rho (default 0.97), sigma_eta chosen so var(epsilon)=1.
        Applied per channel. intensity = alpha * alpha_max.
        """
        rho = float(getattr(self, "ar1_rho", 0.97))
        sigma_eta = np.sqrt(1.0 - rho ** 2)
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()
        for i in range(n_epochs):
            eps = np.zeros((n_channels, n_times), dtype=data.dtype)
            Z = rng.standard_normal((n_channels, n_times))
            eps[:, 0] = sigma_eta * Z[:, 0]
            for t in range(1, n_times):
                eps[:, t] = rho * eps[:, t - 1] + sigma_eta * Z[:, t]
            data_aug[i] += self.intensity * eps
        return data_aug

    def _apply_emg_band_noise(self, data):
        """
        Band-limited [20, f_high] Hz noise (EMG-like). White noise filtered with
        bandpass. Optional low-frequency Gaussian-smoothed envelope (bursty EMG).
        intensity = alpha * alpha_max. Assumes sfreq=250 Hz if not provided.
        """
        sfreq = 250.0
        f_low = float(getattr(self, "emg_f_low", 20.0))
        f_high = float(getattr(self, "emg_f_high", 80.0))
        use_envelope = bool(getattr(self, "emg_use_envelope", False))
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        nyq = sfreq / 2.0
        low = max(0.1, f_low / nyq)
        high = min(0.99, f_high / nyq)
        if low >= high:
            b, a = butter(4, high, btype="low")
        else:
            b, a = butter(4, [low, high], btype="band")
        data_aug = data.copy()
        for i in range(n_epochs):
            eta = rng.standard_normal((n_channels, n_times))
            eps = filtfilt(b, a, eta, axis=1)
            # Optional: scale so typical variance is ~1 (bandpass changes variance)
            var_eps = np.var(eps)
            if var_eps > 0:
                eps = eps / np.sqrt(var_eps)
            if use_envelope:
                # Low-frequency Gaussian-smoothed envelope (bursty EMG)
                env_raw = rng.standard_normal(n_times)
                sigma_samp = max(1, int(0.05 * n_times))
                from scipy.ndimage import gaussian_filter1d
                env = gaussian_filter1d(env_raw, sigma=sigma_samp, mode="nearest")
                env = env - np.mean(env)
                std_env = np.std(env)
                if std_env > 0:
                    env = env / std_env
                env = np.clip(0.3 + 0.7 * (env + 1) / 2, 0.2, 1.0)
                eps = eps * env[np.newaxis, :]
            data_aug[i] += self.intensity * eps.astype(data.dtype, copy=False)
        return data_aug

    def _apply_gain_drift(self, data):
        """Per-channel multiplicative slow drift. intensity scales drift magnitude (0.01-0.5 typical)."""
        rho = float(getattr(self, "gain_drift_rho", 0.995))
        sigma_eta = np.sqrt(1.0 - rho ** 2)
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()
        for i in range(n_epochs):
            log_gain = np.zeros((n_channels, n_times), dtype=np.float64)
            Z = rng.standard_normal((n_channels, n_times))
            log_gain[:, 0] = sigma_eta * Z[:, 0]
            for t in range(1, n_times):
                log_gain[:, t] = rho * log_gain[:, t - 1] + sigma_eta * Z[:, t]
            gain = 1.0 + self.intensity * log_gain
            data_aug[i] = data_aug[i] * gain.astype(data.dtype, copy=False)
        return data_aug

    def _apply_offset_drift(self, data):
        """Per-channel additive slow drift. intensity scales drift magnitude."""
        rho = float(getattr(self, "offset_drift_rho", 0.995))
        sigma_eta = np.sqrt(1.0 - rho ** 2)
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()
        std_per_ch = np.std(data_aug, axis=2) + 1e-10
        for i in range(n_epochs):
            offset = np.zeros((n_channels, n_times), dtype=data.dtype)
            Z = rng.standard_normal((n_channels, n_times))
            offset[:, 0] = sigma_eta * Z[:, 0]
            for t in range(1, n_times):
                offset[:, t] = rho * offset[:, t - 1] + sigma_eta * Z[:, t]
            data_aug[i] += self.intensity * std_per_ch[i:i + 1, :] * offset
        return data_aug

    def _apply_temporal_jitter(self, data):
        """Circular shift time axis by +/- intensity. intensity in samples, or ms if jitter_max_ms set."""
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        sfreq = float(getattr(self, "jitter_sfreq", 250.0))
        if getattr(self, "jitter_max_ms", None) is not None:
            max_shift_samp = int(self.jitter_max_ms * sfreq / 1000.0)
            max_shift = min(max_shift_samp, n_times // 2)
        else:
            max_shift = int(min(abs(self.intensity), n_times // 2))
        if max_shift < 1:
            return data.copy()
        data_aug = data.copy()
        for i in range(n_epochs):
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift != 0:
                data_aug[i] = np.roll(data_aug[i], shift, axis=1)
        return data_aug

    def _apply_spatial_dropout(self, data):
        """Contiguous channel-region dropout (electrode cluster failure). intensity = fraction of channels to drop."""
        rng = np.random.default_rng(self.seed)
        n_epochs, n_channels, n_times = data.shape
        cluster_frac = float(getattr(self, "spatial_dropout_cluster_size", 0.25))
        n_drop = max(1, int(n_channels * self.intensity))
        n_drop = min(n_drop, n_channels)
        cluster_size = max(1, int(n_drop * cluster_frac))
        data_aug = data.copy()
        for i in range(n_epochs):
            start = int(rng.integers(0, n_channels - cluster_size + 1)) if n_channels > cluster_size else 0
            end = min(start + cluster_size, n_channels)
            data_aug[i, start:end, :] = 0.0
        return data_aug

    def _apply_ar1_plus_gain_drift(self, data):
        """Apply AR(1) drift then gain drift (combined for Plot2 diagnostic). intensity scales both."""
        data_aug = self._apply_ar1_drift(data)
        gain_int = float(getattr(self, "gain_drift_intensity", self.intensity * 0.5))
        orig = self.intensity
        self.intensity = gain_int
        data_aug = self._apply_gain_drift(data_aug)
        self.intensity = orig
        return data_aug

    def _apply_ar1_plus_offset_drift(self, data):
        """Apply AR(1) drift then offset drift (combined for Plot2 diagnostic)."""
        data_aug = self._apply_ar1_drift(data)
        offset_int = float(getattr(self, "offset_drift_intensity", self.intensity * 0.5))
        orig = self.intensity
        self.intensity = offset_int
        data_aug = self._apply_offset_drift(data_aug)
        self.intensity = orig
        return data_aug

# Creates an augmented sample for every sample in the set X
class ConcatenatedNoiseAugmenter(ClassifierMixin, BaseEstimator):
    def __init__(self, base_pipeline, noise_type='dropout', intensity=25.0, seed=42, return_groups=False,
                 noise_augmentor_kwargs=None):
        self.base_pipeline = base_pipeline
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.return_groups = return_groups # If True, returns group labels for splitting
        _kw = dict(noise_type=self.noise_type, intensity=self.intensity, seed=self.seed)
        _kw.update(noise_augmentor_kwargs or {})
        self.augmenter = EEGNoiseAugmentor(**_kw)

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
        # Expose fitted classes_ attribute from the wrapped classifier
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


    # Replaces input with noise-augmented version

class TrainOnlyNoiseClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_pipeline, noise_type='dropout', intensity=25.0, seed=42, noise_augmentor_kwargs=None):
        self.base_pipeline = base_pipeline
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.noise_augmentor_kwargs = noise_augmentor_kwargs or {}

    def fit(self, X, y):
        _kw = dict(noise_type=self.noise_type, intensity=self.intensity, seed=self.seed)
        _kw.update(self.noise_augmentor_kwargs)
        augmenter = EEGNoiseAugmentor(**_kw)
        X_aug = augmenter.fit_transform(X)
        self.base_pipeline.fit(X_aug, y)
        self.is_fitted_ = True
        self.base_pipeline.is_fitted_ = True
        # Expose fitted classes_ attribute from the wrapped classifier
        if hasattr(self.base_pipeline, "classes_"):
            self.classes_ = self.base_pipeline.classes_
        elif hasattr(self.base_pipeline[-1], "classes_"):
            self.classes_ = self.base_pipeline[-1].classes_
        else:
            raise AttributeError("Base pipeline does not expose `classes_` attribute")
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("This TrainOnlyNoiseClassifier instance is not fitted yet.")
        return self.base_pipeline.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("This TrainOnlyNoiseClassifier instance is not fitted yet.")
        return self.base_pipeline.predict_proba(X)