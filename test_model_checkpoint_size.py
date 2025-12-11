#!/usr/bin/env python3
"""
Test script to estimate the size of saved model checkpoints.
"""
import os
import sys
import torch
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import MODEL_REGISTRY
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

def get_model_checkpoint_size(model_name, n_chans, n_times, n_outputs):
    """
    Create a model, save it, and return the checkpoint size.
    
    Args:
        model_name: Name of the model
        n_chans: Number of EEG channels
        n_times: Number of time points
        n_outputs: Number of output classes
        
    Returns:
        Size in bytes
    """
    # Get model function
    model_fn = MODEL_REGISTRY[model_name]
    
    # Create model instance
    model = model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save the model's module (the neural network)
        torch.save(model.module_.state_dict(), tmp_path)
        
        # Get file size
        size_bytes = os.path.getsize(tmp_path)
        
        return size_bytes
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def format_size(size_bytes):
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    print("="*70)
    print("MODEL CHECKPOINT SIZE ANALYSIS")
    print("="*70)
    
    # Get actual dataset dimensions for BNCI2014_001
    print("\n[1] Loading BNCI2014_001 to get actual dimensions...")
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8, fmax=35,
        tmin=0.0, tmax=None,
        baseline=None,
        resample=250.0
    )
    
    X, y, metadata = paradigm.get_data(dataset, subjects=[1])
    n_chans_bnci = X.shape[1]
    n_times_bnci = X.shape[2]
    n_outputs_bnci = 2
    
    print(f"   BNCI2014_001 dimensions: {n_chans_bnci} channels × {n_times_bnci} timepoints")
    
    # Test each model
    print("\n[2] Estimating model checkpoint sizes...")
    models = ['eegnet', 'reegnet', 'cnn_ncp']
    sizes = {}
    
    for model_name in models:
        try:
            size = get_model_checkpoint_size(
                model_name, 
                n_chans_bnci, 
                n_times_bnci, 
                n_outputs_bnci
            )
            sizes[model_name] = size
            print(f"   {model_name:15s}: {format_size(size):>12s}")
        except Exception as e:
            print(f"   {model_name:15s}: ERROR - {e}")
            sizes[model_name] = 0
    
    # Calculate average
    valid_sizes = [s for s in sizes.values() if s > 0]
    if valid_sizes:
        avg_size = np.mean(valid_sizes)
        print(f"\n   Average size:    {format_size(avg_size):>12s}")
    else:
        avg_size = 0
        print("\n   Could not calculate average size")
    
    # Now calculate storage requirements for different scenarios
    print("\n" + "="*70)
    print("STORAGE REQUIREMENTS ANALYSIS")
    print("="*70)
    
    # Configuration from experiment_config.yaml
    n_subjects_bnci = 9
    n_sessions_per_subject = 2  # BNCI2014_001 has 2 sessions
    n_seeds = 5  # [100, 200, 300, 400, 500]
    n_models = 3  # [eegnet, reegnet, cnn_ncp]
    n_noise_types = 3  # [gaussian, dropout, eog]
    n_intensity_steps = 20  # From config: num_steps: 20
    
    print(f"\n[3] Experiment Configuration:")
    print(f"   Subjects:        {n_subjects_bnci}")
    print(f"   Sessions/subject: {n_sessions_per_subject}")
    print(f"   Seeds:           {n_seeds}")
    print(f"   Models:          {n_models}")
    print(f"   Noise types:     {n_noise_types}")
    print(f"   Intensity steps: {n_intensity_steps}")
    
    # Scenario 1: CrossSession evaluation
    print(f"\n[4] Scenario 1: CrossSession Evaluation")
    print(f"   CV Strategy: LeaveOneGroupOut (leave one session out)")
    print(f"   Folds per subject: {n_sessions_per_subject}")
    
    # For CrossSession: each subject × each session × each seed × each model
    n_models_crosssession = n_subjects_bnci * n_sessions_per_subject * n_seeds * n_models
    storage_crosssession = n_models_crosssession * avg_size
    
    print(f"\n   Calculation:")
    print(f"   {n_subjects_bnci} subjects × {n_sessions_per_subject} sessions × {n_seeds} seeds × {n_models} models")
    print(f"   = {n_models_crosssession} trained models needed")
    print(f"\n   Storage required: {format_size(storage_crosssession)}")
    
    # Calculate how many evaluations this saves
    n_evaluations_per_model = n_noise_types * n_intensity_steps
    total_evaluations = n_models_crosssession * n_evaluations_per_model
    
    print(f"\n   Each model can be reused for:")
    print(f"   {n_noise_types} noise types × {n_intensity_steps} intensities = {n_evaluations_per_model} evaluations")
    print(f"\n   Total evaluations: {total_evaluations:,}")
    print(f"   Reuse factor: {n_evaluations_per_model}×")
    
    # Scenario 2: WithinSession evaluation
    print(f"\n[5] Scenario 2: WithinSession Evaluation")
    print(f"   CV Strategy: StratifiedKFold (5 splits)")
    n_folds_withinsession = 5
    
    # For WithinSession: each subject × each session × each fold × each seed × each model
    n_models_withinsession = (n_subjects_bnci * n_sessions_per_subject * 
                              n_folds_withinsession * n_seeds * n_models)
    storage_withinsession = n_models_withinsession * avg_size
    
    print(f"\n   Calculation:")
    print(f"   {n_subjects_bnci} subjects × {n_sessions_per_subject} sessions × {n_folds_withinsession} folds × {n_seeds} seeds × {n_models} models")
    print(f"   = {n_models_withinsession} trained models needed")
    print(f"\n   Storage required: {format_size(storage_withinsession)}")
    
    # Scenario 3: CrossSubject evaluation
    print(f"\n[6] Scenario 3: CrossSubject Evaluation")
    print(f"   CV Strategy: LeaveOneGroupOut (leave one subject out)")
    
    # For CrossSubject: each subject × each seed × each model
    n_models_crosssubject = n_subjects_bnci * n_seeds * n_models
    storage_crosssubject = n_models_crosssubject * avg_size
    
    print(f"\n   Calculation:")
    print(f"   {n_subjects_bnci} subjects × {n_seeds} seeds × {n_models} models")
    print(f"   = {n_models_crosssubject} trained models needed")
    print(f"\n   Storage required: {format_size(storage_crosssubject)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print(f"\n[7] Storage Requirements Summary:")
    print(f"   CrossSession:   {format_size(storage_crosssession):>12s} ({n_models_crosssession} models)")
    print(f"   WithinSession:  {format_size(storage_withinsession):>12s} ({n_models_withinsession} models)")
    print(f"   CrossSubject:   {format_size(storage_crosssubject):>12s} ({n_models_crosssubject} models)")
    
    print(f"\n[8] Feasibility Analysis:")
    
    # Check if storage is reasonable (< 10 GB is very feasible, < 50 GB is feasible)
    if storage_crosssession < 10 * 1024**3:
        feasibility = "HIGHLY FEASIBLE ✓"
    elif storage_crosssession < 50 * 1024**3:
        feasibility = "FEASIBLE ✓"
    elif storage_crosssession < 200 * 1024**3:
        feasibility = "SOMEWHAT FEASIBLE"
    else:
        feasibility = "CHALLENGING"
    
    print(f"   CrossSession: {feasibility}")
    print(f"   - Modern systems can easily handle this storage")
    print(f"   - Each model can be reused {n_evaluations_per_model}× for different perturbations")
    print(f"   - Saves {n_evaluations_per_model-1}× training time per cached model")
    
    print(f"\n[9] Time Savings Estimate:")
    print(f"   Assuming ~2 minutes avg training time per model:")
    training_time_mins = 2
    time_without_caching = n_models_crosssession * n_evaluations_per_model * training_time_mins
    time_with_caching = n_models_crosssession * training_time_mins
    time_saved = time_without_caching - time_with_caching
    
    print(f"   Without caching: {time_without_caching:,} minutes ({time_without_caching/60:.1f} hours)")
    print(f"   With caching:    {time_with_caching:,} minutes ({time_with_caching/60:.1f} hours)")
    print(f"   Time saved:      {time_saved:,} minutes ({time_saved/60:.1f} hours)")
    print(f"   Efficiency gain: {(time_saved/time_without_caching)*100:.1f}%")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()


