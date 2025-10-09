# augmentation/noise.py
from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
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
        return template
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
    # Get montage information
    source_montage_obj = make_standard_montage(source_montage)
    
    # Handle target montage - could be string or montage object
    if isinstance(target_montage, str):
        target_montage_obj = make_standard_montage(target_montage)
    else:
        target_montage_obj = target_montage # Already a montage object
    
    # Extract 3D positions
    source_pos = source_montage_obj.get_positions()['ch_pos']
    target_pos = target_montage_obj.get_positions()['ch_pos']
    
    # For the source montage, we need to filter to only the channels that correspond to our source matrix
    # The source matrix has 19 channels, so we need to find the 19 channels from standard_1020
    # that correspond to our training data channels
    expected_source_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    # Filter source positions to only include our expected channels
    filtered_source_pos = {ch: source_pos[ch] for ch in expected_source_channels if ch in source_pos}
    
    # Convert to arrays
    source_channels = list(filtered_source_pos.keys())
    target_channels = list(target_pos.keys())
    
    source_coords = np.array([filtered_source_pos[ch] for ch in source_channels])
    target_coords = np.array([target_pos[ch] for ch in target_channels])
    
    # Check if we have the right number of source coordinates
    if len(source_coords) != source_matrix.shape[0]:
        raise ValueError(f"Source matrix has {source_matrix.shape[0]} channels but filtered montage has {len(source_coords)} channels")
    
    # CRITICAL FIX: Use nearest neighbor interpolation instead of linear to avoid zero-filling
    # This ensures that each target channel gets the value from the closest source channel
    n_regressors = source_matrix.shape[1]
    interpolated_matrix = np.zeros((len(target_channels), n_regressors))
    
    for reg_idx in range(n_regressors):
    # For each target channel, find the closest source channel and use its value
        for target_idx in range(len(target_channels)):
            target_coord = target_coords[target_idx]
    
    # Find the closest source channel
    distances = np.linalg.norm(source_coords - target_coord, axis=1)
    closest_source_idx = np.argmin(distances)
    
    # Use the value from the closest source channel
    interpolated_matrix[target_idx, reg_idx] = source_matrix[closest_source_idx, reg_idx]
    
    return interpolated_matrix

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
        info.set_montage(montage_name)
        current_montage = info.get_montage()
    
    # Determine if interpolation is needed
    source_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    current_ch_names = info.ch_names
    needs_interpolation = (len(current_ch_names) != 19 or 
    not all(ch in current_ch_names for ch in source_channels))
    
    if needs_interpolation:
        # print(f"Interpolating EOG topography from 19-channel to {len(current_ch_names)}-channel montage")
        # Use standard_1020 montage for source, but pass the actual target montage object
        # so we only interpolate to the channels that actually exist in our data
        mixing_matrix = interpolate_eog_topography_to_montage(
        'standard_1020', current_montage, template['mixing_matrix']
        )
    else:
        # Use template directly if montage matches
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
 temporal_coverage=0.1, seed=42, apply_car=True, artifact_scale_factor=15000.0, allow_boundary_intersection=True):
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
    
    Returns
    -------
    np.ndarray
    Contaminated EEG data in same units as input
    """
    # Load the generic EOG template
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
        info.set_montage(montage_name)
        current_montage = info.get_montage()
    
    # Determine if interpolation is needed
    source_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    current_ch_names = info.ch_names
    needs_interpolation = (len(current_ch_names) != 19 or 
    not all(ch in current_ch_names for ch in source_channels))
    
    if needs_interpolation:
        # Use standard_1020 montage for source, but pass the actual target montage object
        mixing_matrix = interpolate_eog_topography_to_montage(
        'standard_1020', current_montage, template['mixing_matrix']
        )
    else:
        # Use template directly if montage matches
        mixing_matrix = template['mixing_matrix']
        
    # CRITICAL FIX: Scale up the mixing matrix to make artifacts impactful
    # The original values were too small (10^-4 to 10^-6) compared to EEG signals (10^0)
    mixing_matrix = mixing_matrix * artifact_scale_factor
    
    # Generate realistic EOG regressors with controlled temporal coverage
    n_times = data_volts.shape[1]
    sfreq = info['sfreq']
    veog_tc, heog_tc = generate_realistic_eog_regressors_with_coverage(
        n_times, sfreq, template, temporal_coverage, seed=seed, allow_boundary_intersection=allow_boundary_intersection
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

def generate_realistic_eog_regressors_with_coverage(n_times, sfreq, template_stats, temporal_coverage, seed=42, allow_boundary_intersection=True):
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
    
    Returns
    -------
    tuple
    (veog_tc, heog_tc) - VEOG and HEOG time courses in Volts
    """
    rng = np.random.RandomState(seed)
    
    # Blink template parameters - use shorter duration to allow more blinks
    blink_duration_ms = np.random.choice(np.arange(100, 300)) # Reduced from 200ms to allow more blinks
    blink_peak_ms = np.random.choice([50, 100, 150, 200]) # Reduced from 80ms proportionally
    blink_peak_ms = np.max([blink_peak_ms, int(blink_duration_ms / 2)])
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
    n_drift_components = rng.randint(1, 4)  # 1-3 drift components
    for _ in range(n_drift_components):
        drift_freq = rng.uniform(0.3, 2.5)  # 0.3-2.5 Hz drift
        drift_phase = rng.uniform(0, 2 * np.pi)
        drift_amplitude = rng.uniform(0.1, 0.25)  # 10-25% of blink amplitude
        t = np.arange(n_times) / sfreq
        veog_tc += drift_amplitude * np.sin(2 * np.pi * drift_freq * t + drift_phase)
        heog_tc += drift_amplitude * 0.7 * np.sin(2 * np.pi * drift_freq * t + drift_phase + rng.uniform(0, np.pi))
    
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
        if rng.random() < 0.4:  # 40% of blinks have microsaccades
            microsaccade_amplitude = rng.uniform(0.08, 0.15)
            microsaccade_direction = rng.choice([-1, 1])
            # Microsaccade is brief and occurs during the blink
            micro_duration = min(int(0.05 * sfreq), valid_end - valid_start)  # 50ms or less
            micro_start = valid_start + (valid_end - valid_start) // 3
            micro_end = min(micro_start + micro_duration, valid_end)
            heog_tc[micro_start:micro_end] += microsaccade_direction * microsaccade_amplitude
        
        # # Add blink clusters (multiple blinks in rapid succession) - common when tired/dry eyes
        if rng.random() < 0.25 and idx < len(blink_starts) - 1:  # 25% chance, not on last blink
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
    
    return veog_tc, heog_tc

# === Enhanced EEGNoiseAugmentor Class ===

class EEGNoiseAugmentor(BaseEstimator, TransformerMixin):
    """
    Enhanced EEG noise augmentation transformer for MOABB pipelines.

    Supports four noise types:
    - 'dropout': Randomly zero out a percentage of EEG channels.
    - 'gaussian': Add Gaussian noise with magnitude-aware scaling.
    - 'eog': Add realistic EOG artifacts using learned template.
    - 'realistic_eog': Add realistic EOG artifacts using learned template (legacy).

    Parameters
    ----------
    noise_type : str
    One of ['dropout', 'gaussian', 'eog', 'realistic_eog'].
    intensity : float
    Noise severity/intensity. Meaning depends on noise_type and mode:
    - For 'dropout': percentage of channels to drop (0-100).
    - For 'gaussian': noise-to-signal ratio as percentage (10.0 = 10% noise relative to signal RMS).
    - For 'eog': 
    * In test_perturb mode: temporal coverage of EOG artifacts (10% = 10% of the time covered by artifacts)
    * In other modes: percentage of epochs to contaminate (0-100)
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
    """

    def __init__(self, noise_type='dropout', intensity=10.0, seed=42, 
    eog_template_path='notebooks/eog_mixing_results/generic_eog_mixing_template.npz', montage_name='standard_1020',
    artifact_scale_factor=10000.0, use_improved_gaussian=True, allow_boundary_intersection=True):
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.eog_template_path = eog_template_path
        self.montage_name = montage_name
        self.artifact_scale_factor = artifact_scale_factor
        self.use_improved_gaussian = use_improved_gaussian
        self.allow_boundary_intersection = allow_boundary_intersection
        
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
        elif n_channels == 62:
            # Lee2019_SSVEP dataset (62 EEG channels following 10-20 system)
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
        else:
            # Generic fallback - generate channel names
            return [f'EEG{i+1:03d}' for i in range(n_channels)]

    def _apply_realistic_eog_noise(self, data):
        """
        Apply realistic EOG artifacts using the learned generic template.
        
        For test_perturb mode: intensity controls temporal coverage of EOG artifacts (10% = 10% of time covered by artifacts)
        For other modes: intensity controls prevalence (percentage of epochs to contaminate)
        """
        rng = np.random.RandomState(self.seed)
        n_epochs, n_channels, n_times = data.shape
        
        # Generate channel names based on the number of channels
        ch_names = self._generate_channel_names(n_channels)
        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg'] * n_channels)
        info.set_montage(self.montage_name)
        
        # Check if we're in test_perturb mode by looking at the intensity value
        # In test_perturb, intensity should control temporal coverage, not prevalence
        is_test_perturb_mode = False
        
        if is_test_perturb_mode:
            # For test_perturb: use 100% prevalence, intensity controls temporal coverage
            prevalence = n_epochs # Contaminate all epochs
            # Convert intensity to temporal coverage (10% -> 0.1, 90% -> 0.9)
            temporal_coverage = self.intensity / 100.0
        else:
            # For other modes: intensity controls prevalence, use fixed temporal coverage
            prevalence = int(n_epochs * (self.intensity / 100))
            temporal_coverage = rng.uniform(0.3, 0.9) # Use 10% temporal coverage for non-test_perturb modes
        contamination_idxs = np.random.choice(n_epochs, size=prevalence, replace=False)
        
        data_aug = data.copy()
        for i in contamination_idxs:
            temporal_coverage = rng.uniform(0.1, 0.9)
            # Inject realistic EOG artifacts with controlled temporal coverage
            contaminated_epoch = inject_realistic_eog_artifacts_with_coverage(
                data[i], info, self.eog_template_path, 
                montage_name=self.montage_name,
                temporal_coverage=temporal_coverage, # Control temporal coverage
                seed=self.seed + i, # Different seed for each epoch
                apply_car=True,
                artifact_scale_factor=self.artifact_scale_factor, # Use the scaling factor
                allow_boundary_intersection=self.allow_boundary_intersection # Pass boundary intersection setting
            )
            data_aug[i] = contaminated_epoch
            
        return data_aug

# Creates an augmented sample for every sample in the set X
class ConcatenatedNoiseAugmenter(ClassifierMixin, BaseEstimator):
    def __init__(self, base_pipeline, noise_type='dropout', intensity=25.0, seed=42, return_groups=False):
        self.base_pipeline = base_pipeline
        self.noise_type = noise_type
        self.intensity = intensity
        self.seed = seed
        self.return_groups = return_groups # If True, returns group labels for splitting
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