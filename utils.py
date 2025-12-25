import os
import pandas as pd
import numpy as np
import re


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

def create_output_path(model, seed, subject, session, mode, session_type='WithinSessionEvaluation', paradigm='MotorImagery', dataset='BNCI2014_001', others=[]):
    if not session_type.endswith("Evaluation"):
        session_type = f"{session_type}Evaluation"

    # Use short session identifier for CrossSubject to avoid long paths
    eval_mode = session_type.replace("Evaluation", "")
    short_session = get_short_session_id(session, eval_mode)

    subject_str = f"sub-{int(subject):03d}"
    full_list = [
        "results",
        paradigm,
        dataset,
        model,
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


def get_noise_perturbation_bounds(dataset: str, noise_type: str, saturation_file: str = "saturation_results/saturation_points_summary.csv") -> tuple[float, float]:
    """
    Get noise perturbation bounds based on dataset and noise type from saturation results.
    
    Args:
        dataset: Dataset name (e.g., 'BNCI2014_001', 'Lee2019_SSVEP')
        noise_type: Type of noise ('gaussian', 'dropout', 'eog', 'spike')
        saturation_file: Path to saturation results CSV file
        
    Returns:
        Tuple of (min_intensity, max_intensity)
        
    Example:
        >>> get_noise_perturbation_bounds('BNCI2014_001', 'dropout')
        (1.0, 50.0)  # Based on saturation point of 50.0
        
        >>> get_noise_perturbation_bounds('Lee2019_SSVEP', 'dropout')
        (1.0, 100.0)  # Based on saturation point of 100.0
    """
    try:
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
        saturation_point = filtered_df.iloc[0]['saturation_point']
        
        # Set bounds based on saturation point
        # Use 1.0 as minimum intensity and saturation point as maximum
        min_intensity = 1.0
        max_intensity = saturation_point
        
        # print(f"[INFO] Using dynamic bounds for {dataset} + {noise_type}: {min_intensity} to {max_intensity} (saturation point: {saturation_point})")
        
        return min_intensity, max_intensity
        
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
        noise_type: Type of noise ('gaussian', 'dropout', 'eog', 'spike')
        num_steps: Number of intensity steps
        saturation_file: Path to saturation results CSV file
        
    Returns:
        Array of noise intensities
    """
    min_intensity, max_intensity = get_noise_perturbation_bounds(dataset, noise_type, saturation_file)
    return np.linspace(start=min_intensity, stop=max_intensity, num=num_steps)