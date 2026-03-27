#!/usr/bin/env python3
"""
Local Ablation Experiments Runner

This script runs all ablation experiments locally (without SLURM/sbatch).
It iterates through all combinations of datasets, eval_modes, ablations, and seeds,
and invokes run_ablations.py for each experiment.

This is equivalent to run_ablation_jobs.sh but runs experiments sequentially
on the local machine instead of submitting SLURM jobs.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import time

# Configuration - matches run_ablation_jobs.sh
DATASETS = ["BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP", "BI2015a"]
EVAL_MODES = ["CrossSubject", "CrossSession", "WithinSession"]
ABLATIONS = ["baseline", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
SEEDS = [100, 200, 300, 400, 500]

# Ablation name mapping - matches get_ablation_config() in run_ablations.py
ABLATION_NAME_MAP: Dict[str, str] = {
    "baseline": "baseline",
    "1": "ablation1_no_carry_gate",
    "2": "ablation2_no_branching",
    "3": "ablation3_lstm_replacement",
    "4": "ablation4_no_snr_gate",
    "5": "ablation5_no_carry_gate_no_branching",
    "6": "ablation6_no_carry_gate_no_snr_gate",
    "7": "ablation7_no_branching_no_snr_gate",
    "8": "ablation8_no_carry_gate_no_branching_no_snr_gate",
    "9": "ablation9_lstm_no_carry_gate",
    "10": "ablation10_lstm_no_branching",
    "11": "ablation11_lstm_no_snr_gate",
    "12": "ablation12_lstm_no_carry_gate_no_branching",
    "13": "ablation13_lstm_no_carry_gate_no_snr_gate",
    "14": "ablation14_lstm_no_branching_no_snr_gate",
    "15": "ablation15_lstm_no_carry_gate_no_branching_no_snr_gate",
}

# Get script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent  # ablations/ directory
PROJECT_ROOT = SCRIPT_DIR.parent  # project root directory


def calculate_total_experiments(
    datasets: List[str],
    eval_modes: List[str],
    ablations: List[str],
    seeds: List[int]
) -> int:
    """Calculate total number of experiments."""
    return len(datasets) * len(eval_modes) * len(ablations) * len(seeds)


def run_single_experiment(
    ablation: str,
    seed: int,
    dataset: str,
    eval_mode: str,
    python_executable: str = "python",
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Run a single ablation experiment.
    
    Args:
        ablation: Ablation number (e.g., "baseline", "1", "2", etc.)
        seed: Random seed (e.g., 100, 200, etc.)
        dataset: Dataset name (e.g., "BNCI2014_001")
        eval_mode: Evaluation mode (e.g., "CrossSubject")
        python_executable: Python executable to use (default: "python")
        dry_run: If True, only print what would be run without executing
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Build command
    run_ablations_script = SCRIPT_DIR / "run_ablations.py"
    
    if not run_ablations_script.exists():
        return False, f"Error: run_ablations.py not found at {run_ablations_script}"
    
    cmd = [
        python_executable,
        str(run_ablations_script),
        "--ablation", ablation,
        "--seed", str(seed),
        "--dataset", dataset,
        "--eval_mode", eval_mode
    ]
    
    if dry_run:
        cmd_str = " ".join(cmd)
        return True, f"[DRY RUN] Would run: {cmd_str}"
    
    # Run the experiment
    try:
        # Change to project root directory (like the sbatch script does)
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # Let output go to stdout/stderr in real-time
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode == 0:
            return True, f"Successfully completed"
        else:
            return False, f"Failed with exit code {result.returncode}"
            
    except KeyboardInterrupt:
        return False, "Interrupted by user"
    except Exception as e:
        return False, f"Exception occurred: {str(e)}"


def run_all_experiments(
    datasets: Optional[List[str]] = None,
    eval_modes: Optional[List[str]] = None,
    ablations: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    python_executable: str = "python",
    dry_run: bool = False,
    continue_on_error: bool = False,
    skip_completed: bool = False
):
    """
    Run all ablation experiments.
    
    Args:
        datasets: List of datasets to run (default: all)
        eval_modes: List of eval modes to run (default: all)
        ablations: List of ablations to run (default: all)
        seeds: List of seeds to run (default: all)
        python_executable: Python executable to use
        dry_run: If True, only print what would be run
        continue_on_error: If True, continue to next experiment on error
        skip_completed: If True, skip experiments that already have results files
    """
    # Use defaults if not provided
    datasets = datasets or DATASETS
    eval_modes = eval_modes or EVAL_MODES
    ablations = ablations or ABLATIONS
    seeds = seeds or SEEDS
    
    # Calculate totals
    total_experiments = calculate_total_experiments(datasets, eval_modes, ablations, seeds)
    
    print("=" * 80)
    print("Local Ablation Experiments Runner")
    print("=" * 80)
    print(f"Total experiments to run: {total_experiments}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Eval Modes: {', '.join(eval_modes)}")
    print(f"Ablations: {', '.join(ablations)}")
    print(f"Seeds: {', '.join(map(str, seeds))}")
    print(f"Python executable: {python_executable}")
    print(f"Dry run: {dry_run}")
    print(f"Continue on error: {continue_on_error}")
    print(f"Skip completed: {skip_completed}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {SCRIPT_DIR / 'results'}")
    print("=" * 80)
    print()
    
    if dry_run:
        print("[DRY RUN MODE] - No experiments will actually be executed")
        print()
    
    # Statistics
    experiment_num = 0
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    # Results directory for checking completed experiments
    results_dir = SCRIPT_DIR / "results"
    
    # Loop through all combinations
    for dataset in datasets:
        for eval_mode in eval_modes:
            for ablation in ablations:
                for seed in seeds:
                    experiment_num += 1
                    
                    # Check if already completed
                    if skip_completed:
                        # Get the ablation name (matches run_ablations.py naming)
                        ablation_name = ABLATION_NAME_MAP.get(ablation.lower(), f"ablation{ablation}")
                        results_file = results_dir / f"{dataset}_{eval_mode}_{ablation_name}_seed{seed}.csv"
                        
                        if results_file.exists():
                            skipped += 1
                            print(f"[{experiment_num}/{total_experiments}] SKIPPED (already completed): "
                                  f"{dataset}/{eval_mode}/ablation{ablation}/seed{seed}")
                            print(f"  Results file exists: {results_file.name}")
                            continue
                    
                    # Print experiment info
                    print(f"\n[{experiment_num}/{total_experiments}] Running experiment:")
                    print(f"  Dataset: {dataset}")
                    print(f"  Eval Mode: {eval_mode}")
                    print(f"  Ablation: {ablation}")
                    print(f"  Seed: {seed}")
                    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("-" * 80)
                    
                    # Run the experiment
                    success, message = run_single_experiment(
                        ablation=ablation,
                        seed=seed,
                        dataset=dataset,
                        eval_mode=eval_mode,
                        python_executable=python_executable,
                        dry_run=dry_run
                    )
                    
                    # Update statistics
                    if success:
                        successful += 1
                        status = "SUCCESS"
                    else:
                        failed += 1
                        status = "FAILED"
                    
                    elapsed = time.time() - start_time
                    avg_time = elapsed / experiment_num if experiment_num > 0 else 0
                    remaining = (total_experiments - experiment_num) * avg_time if avg_time > 0 else 0
                    
                    print("-" * 80)
                    print(f"[{status}] {message}")
                    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Elapsed time: {elapsed / 3600:.2f} hours")
                    print(f"  Estimated remaining: {remaining / 3600:.2f} hours")
                    print(f"  Progress: {successful} successful, {failed} failed, {skipped} skipped")
                    print()
                    
                    # Handle errors
                    if not success and not continue_on_error:
                        print("=" * 80)
                        print("ERROR: Experiment failed and continue_on_error is False")
                        print("=" * 80)
                        print(f"Failed experiment: {dataset}/{eval_mode}/ablation{ablation}/seed{seed}")
                        print(f"Total completed: {experiment_num}/{total_experiments}")
                        print(f"Successful: {successful}, Failed: {failed}, Skipped: {skipped}")
                        sys.exit(1)
    
    # Final summary
    total_time = time.time() - start_time
    print("=" * 80)
    print("All Experiments Complete!")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total time: {total_time / 3600:.2f} hours ({total_time:.0f} seconds)")
    if successful > 0:
        print(f"Average time per experiment: {total_time / successful:.0f} seconds")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all ablation experiments locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_local_ablations.py
  
  # Dry run (see what would be executed)
  python run_local_ablations.py --dry-run
  
  # Run only specific dataset and eval mode
  python run_local_ablations.py --dataset BNCI2014_001 --eval-mode CrossSubject
  
  # Run only baseline and first 3 ablations
  python run_local_ablations.py --ablation baseline --ablation 1 --ablation 2 --ablation 3
  
  # Run with custom Python executable
  python run_local_ablations.py --python python3
  
  # Skip already completed experiments
  python run_local_ablations.py --skip-completed
  
  # Continue even if some experiments fail
  python run_local_ablations.py --continue-on-error
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        choices=DATASETS,
        help="Dataset to run (can be specified multiple times). Default: all datasets"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        dest="eval_mode",
        action="append",
        choices=EVAL_MODES,
        help="Evaluation mode to run (can be specified multiple times). Default: all eval modes"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        action="append",
        choices=ABLATIONS,
        help="Ablation to run (can be specified multiple times). Default: all ablations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed to run (can be specified multiple times). Default: all seeds [100, 200, 300, 400, 500]"
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable to use (default: 'python')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually executing experiments"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next experiment even if current one fails"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments that already have results files"
    )
    
    args = parser.parse_args()
    
    # Run all experiments
    run_all_experiments(
        datasets=args.dataset,
        eval_modes=args.eval_mode,
        ablations=args.ablation,
        seeds=args.seed,
        python_executable=args.python,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        skip_completed=args.skip_completed
    )


if __name__ == "__main__":
    main()
