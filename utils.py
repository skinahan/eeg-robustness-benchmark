import os
import pandas as pd
import numpy as np


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
    full_list = [
        "results",
        paradigm,
        dataset,
        model,
        session_type,
        str(seed),
        f"sub-{int(subject):03d}",
        session,
        mode
    ]
    if len(others) > 0:
        full_list.extend(others)

    return os.path.join(
        "//".join(full_list)
    )


def get_noise_perturbation_bounds(dataset: str, noise_type: str, saturation_file: str = "saturation_results/saturation_points_summary.csv") -> tuple[float, float]:
    """
    Get noise perturbation bounds based on dataset and noise type from saturation results.
    
    Args:
        dataset: Dataset name (e.g., 'BNCI2014_001', 'Lee2019_SSVEP')
        noise_type: Type of noise ('gaussian', 'dropout', 'eog')
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
        
        print(f"[INFO] Using dynamic bounds for {dataset} + {noise_type}: {min_intensity} to {max_intensity} (saturation point: {saturation_point})")
        
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
        noise_type: Type of noise ('gaussian', 'dropout', 'eog')
        num_steps: Number of intensity steps
        saturation_file: Path to saturation results CSV file
        
    Returns:
        Array of noise intensities
    """
    min_intensity, max_intensity = get_noise_perturbation_bounds(dataset, noise_type, saturation_file)
    return np.linspace(start=min_intensity, stop=max_intensity, num=num_steps)