#!/usr/bin/env python3
"""
Aggregate results from fold-by-fold CrossSubject experiments.

This script collects results from individual fold runs and combines them
into a single results DataFrame, maintaining compatibility with existing
analysis pipelines.

Usage:
    python aggregate_crosssubject_results.py \
        --model cnn_ncp \
        --dataset BNCI2014_001 \
        --subjects 1 2 3 4 5 6 7 8 9 \
        --mode test_perturb \
        --seed 42
"""

import sys
import os
import argparse
import pandas as pd
import glob
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.unified_experiment_runner import ThreeFoldSubjectSplit
from evaluation.experiment_utils import log_all_subjects
from config import get_paradigm


def find_result_files(base_dir, model, dataset, subjects, mode, seed, eval_mode="CrossSubject"):
    """
    Find all result files from fold-by-fold runs.
    
    Returns a list of (fold_idx, filepath) tuples.
    """
    # Determine output directory structure
    # Results are saved in: outputs/{model}/{seed}/subject_{subject}/session_{session}/results.csv
    # For CrossSubject fold-by-fold, session is like "fold_0_eval_subjects_1,2,3"
    
    result_files = []
    
    # Get all subject directories
    subjects_str = '_'.join(map(str, sorted(subjects)))
    pattern = os.path.join(
        base_dir, "outputs", model, str(seed),
        f"subject_*", "session_fold_*", "results.csv"
    )
    
    files = glob.glob(pattern)
    
    # Filter to only CrossSubject fold results
    for filepath in files:
        # Extract fold_idx from path
        # Path format: .../subject_X/session_fold_Y_eval_subjects_Z/results.csv
        parts = filepath.split(os.sep)
        for part in parts:
            if part.startswith("session_fold_"):
                try:
                    fold_idx = int(part.split("_")[1])
                    result_files.append((fold_idx, filepath))
                    break
                except (ValueError, IndexError):
                    continue
    
    # Sort by fold_idx
    result_files.sort(key=lambda x: x[0])
    
    return result_files


def aggregate_results(model, dataset, subjects, mode, seed, eval_mode="CrossSubject", 
                     base_dir=".", tune=False, noise_type=None, intensity=None):
    """
    Aggregate results from fold-by-fold CrossSubject runs.
    """
    print(f"Aggregating results for:")
    print(f"  Model: {model}")
    print(f"  Dataset: {dataset}")
    print(f"  Subjects: {subjects}")
    print(f"  Mode: {mode}")
    print(f"  Seed: {seed}")
    print(f"  Eval Mode: {eval_mode}")
    print()
    
    # Find all result files
    result_files = find_result_files(base_dir, model, dataset, subjects, mode, seed, eval_mode)
    
    if not result_files:
        print("ERROR: No result files found. Make sure all folds have completed.")
        print(f"  Searched pattern: outputs/{model}/{seed}/subject_*/session_fold_*/results.csv")
        return None
    
    print(f"Found {len(result_files)} fold result file(s):")
    for fold_idx, filepath in result_files:
        print(f"  Fold {fold_idx}: {filepath}")
    print()
    
    # Load and combine results
    all_results = []
    for fold_idx, filepath in result_files:
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded fold {fold_idx}: {len(df)} rows")
            all_results.append(df)
        except Exception as e:
            print(f"WARNING: Failed to load {filepath}: {e}")
            continue
    
    if not all_results:
        print("ERROR: No results could be loaded.")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\nCombined results: {len(combined_df)} total rows")
    
    # Verify we have all folds
    unique_folds = sorted(combined_df['fold_idx'].unique())
    expected_folds = [0, 1, 2]
    missing_folds = set(expected_folds) - set(unique_folds)
    
    if missing_folds:
        print(f"WARNING: Missing results for fold(s): {sorted(missing_folds)}")
    else:
        print(f"[OK] All {len(expected_folds)} folds present")
    
    # Determine paradigm for logging
    if dataset == "Lee2019_SSVEP":
        paradigm_name = "SSVEP"
    elif dataset == "BI2015a":
        paradigm_name = "ERP"
    else:
        paradigm_name = "MotorImagery"
    
    # Log aggregated results (same as original flow)
    mode_str = mode
    if tune:
        mode_str = f"{mode}_tune"
    
    # Get paradigm for log_all_subjects
    paradigm = get_paradigm(resample=None, dataset=dataset)
    
    log_all_subjects(
        results=combined_df,
        subject_list=subjects,
        model_name=model,
        mode=mode_str,
        noise_type=noise_type,
        intensity=intensity,
        seed=seed,
        eval_mode=eval_mode,
        paradigm=paradigm_name,
        dataset=dataset
    )
    
    print(f"\n[OK] Results aggregated and logged successfully")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Folds: {unique_folds}")
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from fold-by-fold CrossSubject experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., cnn_ncp, eegnet)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["BNCI2014_001", "Lee2019_SSVEP", "BI2015a"],
                        help="Dataset name")
    parser.add_argument("--subjects", type=int, nargs="+", required=True,
                        help="List of all subject IDs")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["test_perturb", "multirun", "aggregate_only"],
                        help="Experiment mode")
    parser.add_argument("--eval_mode", type=str, default="CrossSubject",
                        help="Evaluation mode (default: CrossSubject)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tune", action="store_true",
                        help="Results include hyperparameter tuning")
    parser.add_argument("--noise_type", type=str, 
                        choices=["dropout", "gaussian", "eog", "spike"], 
                        default=None,
                        help="Noise type (if applicable)")
    parser.add_argument("--intensity", type=float, default=None,
                        help="Noise intensity (if applicable)")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory for searching results")
    
    args = parser.parse_args()
    
    aggregate_results(
        model=args.model,
        dataset=args.dataset,
        subjects=args.subjects,
        mode=args.mode,
        seed=args.seed,
        eval_mode=args.eval_mode,
        base_dir=args.base_dir,
        tune=args.tune,
        noise_type=args.noise_type,
        intensity=args.intensity
    )


if __name__ == "__main__":
    main()
