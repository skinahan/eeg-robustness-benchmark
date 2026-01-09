#!/usr/bin/env python3
"""
Wrapper script for running CrossSubject experiments fold-by-fold.

This script splits a CrossSubject experiment into separate runs, one per fold,
to reduce memory requirements. Each fold is processed independently, loading
only the subjects needed for that fold.

Usage:
    python run_crosssubject_folds.py \
        --model cnn_ncp \
        --dataset BNCI2014_001 \
        --subjects 1 2 3 4 5 6 7 8 9 \
        --mode test_perturb \
        --seed 42 \
        [--tune] \
        [--overwrite] \
        [--noise_type gaussian] \
        [--intensity 10.0] \
        [--subject_chunk_size 3]

After all folds complete, run aggregate_crosssubject_results.py to combine results.

Memory Optimization:
    --subject_chunk_size controls memory-efficient chunked training for CrossSubject mode.
    Default: 3 (loads 3 subjects at a time to reduce peak memory usage).
    This is especially useful for large datasets with many subjects.
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.unified_experiment_runner import ThreeFoldSubjectSplit


def main():
    parser = argparse.ArgumentParser(
        description="Run CrossSubject experiments fold-by-fold to reduce memory usage",
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
                        help="Enable hyperparameter tuning")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--noise_type", type=str, 
                        choices=["dropout", "gaussian", "eog", "spike"], 
                        default=None,
                        help="Noise type (for test_perturb mode)")
    parser.add_argument("--intensity", type=float, default=None,
                        help="Noise intensity (for test_perturb mode)")
    parser.add_argument("--script_path", type=str, 
                        default="evaluation/unified_experiment_runner.py",
                        help="Path to unified_experiment_runner.py")
    parser.add_argument("--save_config", type=str, default=None,
                        help="Save fold configuration to JSON file")
    parser.add_argument("--load_config", type=str, default=None,
                        help="Load fold configuration from JSON file (skip planning)")
    parser.add_argument("--fold_idx", type=int, default=None,
                        help="Run only a specific fold (0-2). If not provided, runs all folds.")
    parser.add_argument("--subject_chunk_size", type=int, default=3,
                        help="Number of subjects to load per chunk for memory-efficient training (CrossSubject mode). Default: 3")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.eval_mode != "CrossSubject":
        print(f"WARNING: This script is designed for CrossSubject mode, but eval_mode is {args.eval_mode}")
    
    if args.mode == "test_perturb" and (args.noise_type is None or args.intensity is None):
        # Use defaults for test_perturb
        args.noise_type = args.noise_type or "gaussian"
        args.intensity = args.intensity or 10.0
        print(f"Using default noise: {args.noise_type} with intensity {args.intensity}")
    elif args.mode == "multirun":
        # For multirun mode, noise_type and intensity are not needed
        # multirun mode handles all noise types and intensities internally
        args.noise_type = None
        args.intensity = None
    
    # Determine fold configuration
    if args.load_config:
        print(f"Loading fold configuration from {args.load_config}...")
        with open(args.load_config, 'r') as f:
            fold_configs = json.load(f)
    else:
        print("Determining fold configuration...")
        splitter = ThreeFoldSubjectSplit()
        fold_configs = splitter.get_fold_subjects(args.subjects)
        
        if args.save_config:
            print(f"Saving fold configuration to {args.save_config}...")
            with open(args.save_config, 'w') as f:
                json.dump(fold_configs, f, indent=2)
    
    # Filter to specific fold if requested
    if args.fold_idx is not None:
        fold_configs = [fc for fc in fold_configs if fc['fold_idx'] == args.fold_idx]
        if not fold_configs:
            print(f"ERROR: Fold {args.fold_idx} not found in configuration")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Running {len(fold_configs)} fold(s) for CrossSubject experiment")
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Subjects: {args.subjects}")
    print(f"Memory optimization: Chunked training enabled (subject_chunk_size={args.subject_chunk_size})")
    print(f"{'='*80}\n")
    
    # Build base command
    base_cmd = [
        sys.executable,
        args.script_path,
        "--model", args.model,
        "--dataset", args.dataset,
        "--subjects"] + [str(s) for s in args.subjects] + [
        "--mode", args.mode,
        "--eval_mode", args.eval_mode,
        "--seed", str(args.seed)
    ]
    
    if args.tune:
        base_cmd.append("--tune")
    if args.overwrite:
        base_cmd.append("--overwrite")
    # Only add noise_type and intensity for test_perturb mode
    # multirun mode handles all noise types internally
    if args.mode == "test_perturb":
        if args.noise_type:
            base_cmd.extend(["--noise_type", args.noise_type])
        if args.intensity:
            base_cmd.extend(["--intensity", str(args.intensity)])
    
    # Run each fold
    successful_folds = []
    failed_folds = []
    
    for fold_config in fold_configs:
        fold_idx = fold_config['fold_idx']
        train_subjects = fold_config['train_subjects']
        eval_subjects = fold_config['eval_subjects']
        
        print(f"\n{'='*80}")
        print(f"Running Fold {fold_idx}")
        print(f"  Training subjects: {train_subjects}")
        print(f"  Evaluation subjects: {eval_subjects}")
        print(f"{'='*80}\n")
        
        # Build command for this fold
        cmd = base_cmd + [
            "--fold_idx", str(fold_idx),
            "--train_subjects"] + [str(s) for s in train_subjects] + [
            "--eval_subjects"] + [str(s) for s in eval_subjects] + [
            "--subject_chunk_size", str(args.subject_chunk_size)
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        print(f"[MEMORY] Using chunked training with chunk_size={args.subject_chunk_size}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n✓ Fold {fold_idx} completed successfully")
            successful_folds.append(fold_idx)
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Fold {fold_idx} failed with exit code {e.returncode}")
            failed_folds.append(fold_idx)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user. Completed {len(successful_folds)} fold(s).")
            sys.exit(1)
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total folds: {len(fold_configs)}")
    print(f"Successful: {len(successful_folds)} - {successful_folds}")
    print(f"Failed: {len(failed_folds)} - {failed_folds}")
    
    if failed_folds:
        print(f"\nWARNING: {len(failed_folds)} fold(s) failed. Results may be incomplete.")
        sys.exit(1)
    else:
        print(f"\n✓ All folds completed successfully!")
        if args.mode != "multirun":
            # For non-multirun modes, user needs to manually aggregate
            print(f"\nNext step: Run aggregate_crosssubject_results.py to combine results:")
            print(f"  python aggregate_crosssubject_results.py \\")
            print(f"      --model {args.model} \\")
            print(f"      --dataset {args.dataset} \\")
            print(f"      --subjects {' '.join(map(str, args.subjects))} \\")
            print(f"      --mode {args.mode} \\")
            print(f"      --seed {args.seed}")
            if args.tune:
                print(f"      --tune")
            if args.noise_type:
                print(f"      --noise_type {args.noise_type}")
            if args.intensity:
                print(f"      --intensity {args.intensity}")
        # For multirun mode, aggregation is handled by the shell script


if __name__ == "__main__":
    main()
