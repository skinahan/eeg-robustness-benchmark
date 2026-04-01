import os
import re
import hashlib
from typing import Optional

import pandas as pd
import numpy as np


def get_short_session_id(session: str, eval_mode: str = 'CrossSession') -> str:
    """
    Create a short session identifier for CrossSubject evaluation to avoid long file paths.
    
    For CrossSubject evaluation, converts long session strings like:
    "fold_0_eval_subjects_1,2,3,4,5,6,7,8,9,10,11,12,13,14" 
    to short identifiers like: "fold_0"
    
    The full eval_subjects information is preserved in the CSV file data itself,
    so we don't lose any information by shortening the path.
    
    For other eval modes, returns the session unchanged.
    
    Args:
        session: Full session identifier string
        eval_mode: Evaluation mode (CrossSubject, CrossSession, WithinSession, etc.)
        
    Returns:
        Short session identifier for CrossSubject, original session for others
    """
    # Only shorten for CrossSubject evaluation
    if eval_mode == 'CrossSubject' or eval_mode == 'CrossSubjectEvaluation':
        # Extract fold number from session string (e.g., "fold_0_eval_subjects_..." -> "fold_0")
        match = re.match(r'fold_(\d+)', session)
        if match:
            return f"fold_{match.group(1)}"
        # Fallback: if pattern doesn't match, return original (shouldn't happen in normal operation)
        return session
    return session


def create_hdf5_model_path(model, seed, session, mode, session_type='WithinSessionEvaluation', paradigm='MotorImagery', dataset='BNCI2014_001', others=[]):
    if not session_type.endswith("Evaluation"):
        session_type = f"{session_type}Evaluation"
    full_list = [
        "results",
        paradigm,
        dataset,
        model,
        session_type,
        str(seed),
        f"checkpoints",
        session,
        mode
    ]
    if len(others) > 0:
        full_list.extend(others)

    return os.path.join(
        "//".join(full_list)
    )

def short_run_id(model_name: str, length: int = 12) -> str:
    """
    Deterministic short id for a model/run name to keep Windows paths under 260 chars.
    Use this when building result paths; full model name is stored in CSV and manifest.
    """
    if not model_name or not isinstance(model_name, str):
        return "default"
    h = hashlib.sha256(model_name.encode("utf-8")).hexdigest()
    return h[:length]


def results_paradigm_folder(dataset: str) -> str:
    """
    Top-level directory name under results/ for unified runner outputs:
    results/<ParadigmFolder>/<dataset>/...

    Uses exact MOABB dataset codes only — do not infer via substrings like 'Lee2019',
    because Lee2019_MI and Lee2019_SSVEP share a prefix but use different folders
    (MotorImagery vs SSVEP).
    """
    if dataset == "Lee2019_SSVEP":
        return "SSVEP"
    if dataset == "BI2015a":
        return "ERP"
    return "MotorImagery"


def create_output_path(model, seed, subject, session, mode, session_type='WithinSessionEvaluation', paradigm='MotorImagery', dataset='BNCI2014_001', others=[], use_short_run_id=True):
    if not session_type.endswith("Evaluation"):
        session_type = f"{session_type}Evaluation"

    # Use short run id for model segment to avoid Windows path length limits
    model_segment = short_run_id(model, length=12) if use_short_run_id else model

    # Use short session identifier for CrossSubject to avoid long paths
    eval_mode = session_type.replace("Evaluation", "")
    short_session = get_short_session_id(session, eval_mode)

    subject_str = f"sub-{int(subject):03d}"
    full_list = [
        "results",
        paradigm,
        dataset,
        model_segment,
        session_type,
        str(seed),
        subject_str,
        short_session,  # Use shortened session for path
        mode
    ]
   
    if len(others) > 0:
        full_list.extend(others)
    full_list = [str(item) for item in full_list]
    return os.path.join(
        "//".join(full_list)
    )


# Correlated perturbation types (Plot 2): alpha_max may be computed from data in the runner
# Plot2_revision3: added gain_drift, offset_drift, temporal_jitter, spatial_dropout
_CORRELATED_NOISE_TYPES = (
    "spatial_gaussian", "ar1_drift", "emg_band",
    "gain_drift", "offset_drift", "temporal_jitter", "spatial_dropout",
    "ar1_plus_gain_drift", "ar1_plus_offset_drift",
)

# Cache: (dataset, noise_type) -> min physical intensity from sol_results or None
_SOL_PHYSICAL_MIN_CACHE: dict[tuple[str, str], Optional[float]] = {}


def _physical_min_intensity_from_sol_results_motor_imagery(
    dataset: str, noise_type: str
) -> Optional[float]:
    """
    Minimum physical intensity (> 1) in ``sol_results/MotorImagery/{dataset}/all_results.csv``
    for this noise_type. When historical sweeps did not use 1.0 as the lower bound (merged CSVs),
    aligning linspace(min, saturation_max, n) with this value matches stored rows and the runner.
    """
    key = (str(dataset), str(noise_type))
    if key in _SOL_PHYSICAL_MIN_CACHE:
        return _SOL_PHYSICAL_MIN_CACHE[key]
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "sol_results", "MotorImagery", str(dataset), "all_results.csv")
    if not os.path.isfile(path):
        _SOL_PHYSICAL_MIN_CACHE[key] = None
        return None
    try:
        df = pd.read_csv(path, usecols=["noise_type", "intensity"], low_memory=False)
    except (ValueError, FileNotFoundError, OSError, KeyError):
        try:
            df = pd.read_csv(path, low_memory=False)
        except (FileNotFoundError, OSError):
            _SOL_PHYSICAL_MIN_CACHE[key] = None
            return None
    if "noise_type" not in df.columns or "intensity" not in df.columns:
        _SOL_PHYSICAL_MIN_CACHE[key] = None
        return None
    sub = df[df["noise_type"].astype(str) == str(noise_type)]
    if sub.empty:
        _SOL_PHYSICAL_MIN_CACHE[key] = None
        return None
    raw = pd.to_numeric(sub["intensity"], errors="coerce").dropna()
    vals = raw[raw > 1.0].to_numpy(dtype=float)
    if vals.size == 0:
        _SOL_PHYSICAL_MIN_CACHE[key] = None
        return None
    gmin = float(np.min(vals))
    _SOL_PHYSICAL_MIN_CACHE[key] = gmin
    return gmin


def get_noise_perturbation_bounds(dataset: str, noise_type: str, saturation_file: str = "saturation_results/saturation_points_summary.csv") -> tuple[float, float]:
    """
    Get noise perturbation bounds based on dataset and noise type from saturation results.
    
    Args:
        dataset: Dataset name (e.g., 'BNCI2014_001', 'Lee2019_SSVEP')
        noise_type: Type of noise ('gaussian', 'dropout', 'eog', 'spike', 'spatial_gaussian', 'ar1_drift', 'emg_band')
        saturation_file: Path to saturation results CSV file
        
    Returns:
        Tuple of (min_intensity, max_intensity). For correlated types (spatial_gaussian, ar1_drift, emg_band),
        if no saturation row exists, returns (0.0, 1.0) as nominal range; the runner may compute alpha_max from data.
        
    Example:
        >>> get_noise_perturbation_bounds('BNCI2014_001', 'dropout')
        (1.0, 50.0)  # Based on saturation point of 50.0
        
        >>> get_noise_perturbation_bounds('Lee2019_SSVEP', 'dropout')
        (1.0, 100.0)  # Based on saturation point of 100.0
    """
    try:
        # Correlated types: default nominal bounds when no saturation row; runner may compute alpha_max from data
        if noise_type in _CORRELATED_NOISE_TYPES:
            try:
                saturation_df = pd.read_csv(saturation_file)
                filtered_df = saturation_df[
                    (saturation_df["dataset"] == dataset)
                    & (saturation_df["noise_type"] == noise_type)
                ]
                if not filtered_df.empty:
                    sat = float(filtered_df.iloc[0]["saturation_point"])
                    return 0.0, sat
            except (FileNotFoundError, Exception):
                pass
            return 0.0, 1.0

        # Load saturation results
        saturation_df = pd.read_csv(saturation_file)
        
        # Filter for specific dataset and noise type
        filtered_df = saturation_df[
            (saturation_df['dataset'] == dataset) & 
            (saturation_df['noise_type'] == noise_type)
        ]
        
        if filtered_df.empty:
            print(f"[WARNING] No saturation data found for {dataset} + {noise_type}, using default bounds (1.0, 50.0)")
            return 1.0, 50.0
        
        # Get the saturation point
        saturation_point = float(filtered_df.iloc[0]["saturation_point"])

        # Default: [1, saturation_point]. If sol_results shows physical intensities strictly above 1.0,
        # use that minimum so np.linspace matches merged CSVs and multirun sweeps (Lee2019_MI, Shin2017A).
        min_intensity = 1.0
        max_intensity = saturation_point
        gmin = _physical_min_intensity_from_sol_results_motor_imagery(dataset, noise_type)
        if (
            gmin is not None
            and gmin > 1.0 + 1e-9
            and gmin < max_intensity - 1e-9
        ):
            min_intensity = gmin

        return float(min_intensity), float(max_intensity)
        
    except FileNotFoundError:
        print(f"[WARNING] Saturation file {saturation_file} not found, using default bounds (1.0, 50.0)")
        return 1.0, 50.0
    except Exception as e:
        print(f"[WARNING] Error loading saturation data: {e}, using default bounds (1.0, 50.0)")
        return 1.0, 50.0


def get_noise_intensities(dataset: str, noise_type: str, num_steps: int = 20, saturation_file: str = "saturation_results/saturation_points_summary.csv") -> np.ndarray:
    """
    Get noise intensity array based on dataset and noise type from saturation results.
    
    Args:
        dataset: Dataset name (e.g., 'BNCI2014_001', 'Lee2019_SSVEP')
        noise_type: Type of noise ('gaussian', 'dropout', 'eog', 'spike', 'spatial_gaussian', 'ar1_drift', 'emg_band')
        num_steps: Number of intensity steps
        saturation_file: Path to saturation results CSV file
        
    Returns:
        Array of noise intensities. For correlated types with no saturation row, returns linspace(0, 1, num_steps).
    """
    min_intensity, max_intensity = get_noise_perturbation_bounds(dataset, noise_type, saturation_file)
    return np.linspace(start=min_intensity, stop=max_intensity, num=num_steps)