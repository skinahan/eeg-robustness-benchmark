#!/usr/bin/env python3
"""
Ablation Studies Experiment Runner

This script runs ablation studies on the HYDRA model (branched_wiredcfc architecture #4)
to evaluate the contribution of individual model mechanisms.

Ablations:
1. No Carry Gate - Disables weighted residual connection
2. No Branching - Processes entire sequence in single temporal bin
3. Replace CfC with LSTM - Uses LSTM instead of CfC recurrent cells
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

# Add project root to path - use Path.resolve() for robust absolute paths
# __file__ is the path to this script (run_ablations.py)
script_file = Path(__file__).resolve()  # Get absolute, normalized path
current_dir = script_file.parent  # ablations/ directory
project_root = current_dir.parent  # project root directory

# Convert to string for sys.path (needs string, not Path)
project_root_str = str(project_root)
sys.path.insert(0, project_root_str)

# Debug: Print paths
print(f"[DEBUG] Script file: {script_file}", file=sys.stderr)
print(f"[DEBUG] Current directory (ablations/): {current_dir}", file=sys.stderr)
print(f"[DEBUG] Project root: {project_root}", file=sys.stderr)
print(f"[DEBUG] Project root (string): {project_root_str}", file=sys.stderr)
print(f"[DEBUG] Python path: {sys.path[:3]}", file=sys.stderr)

from config import (
    MODEL_REGISTRY, 
    get_paradigm, 
    get_dataset_sampling_rate,
    add_branched_wiredcfc_architecture,
    get_branched_wiredcfc_architecture_registry
)
from globals import set_seeds, get_seed
from models.branched_lstm import create_branched_lstm_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from evaluation.unified_experiment_runner import UnifiedExperimentRunner
from moabb.datasets import BNCI2014_001

# Import ablation model variants (from same directory)
# Use absolute import since project_root is added to sys.path
try:
    from ablations.ablation_models import (
        create_branched_wiredcfc_no_carry_gate_classifier,
        create_branched_wiredcfc_no_branching_classifier
    )
except ImportError:
    # Fallback: import from same directory if running from ablations/
    import importlib.util
    ablation_models_path = current_dir / "ablation_models.py"
    if not ablation_models_path.exists():
        raise ImportError(
            f"Could not import ablation_models and file not found: {ablation_models_path}\n"
            f"Current directory: {current_dir}\n"
            f"Project root: {project_root}"
        )
    spec = importlib.util.spec_from_file_location("ablation_models", str(ablation_models_path))
    ablation_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation_models)
    create_branched_wiredcfc_no_carry_gate_classifier = ablation_models.create_branched_wiredcfc_no_carry_gate_classifier
    create_branched_wiredcfc_no_branching_classifier = ablation_models.create_branched_wiredcfc_no_branching_classifier
    print(f"[DEBUG] Loaded ablation_models from fallback location: {ablation_models_path}", file=sys.stderr)

# Configuration
ARCHITECTURE_FILE_RELATIVE = "outputs/architectures/best_architecture_4_trial_178.json"
DATASET = "BNCI2014_001"
EVAL_MODE = "CrossSubject"
SEEDS = [100, 200, 300, 400, 500]  # 5 experimental runs
NUM_RUNS = len(SEEDS)

# Resolve all paths relative to project root using Path objects
OUTPUT_DIR = current_dir  # ablations/ directory
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

# Resolve architecture file path (relative to project root)
ARCHITECTURE_FILE_PATH = project_root / ARCHITECTURE_FILE_RELATIVE

# Sanity check: Verify project root exists and is a directory
if not project_root.exists():
    raise RuntimeError(
        f"Project root does not exist: {project_root}\n"
        f"Resolved from script: {script_file}"
    )
if not project_root.is_dir():
    raise RuntimeError(
        f"Project root is not a directory: {project_root}\n"
        f"Resolved from script: {script_file}"
    )

# Sanity check: Verify architecture file exists
# Try to resolve the path (handles symlinks, relative paths, etc.)
try:
    ARCHITECTURE_FILE_PATH = ARCHITECTURE_FILE_PATH.resolve()
except (OSError, RuntimeError) as e:
    raise RuntimeError(
        f"Failed to resolve architecture file path: {ARCHITECTURE_FILE_PATH}\n"
        f"Error: {e}"
    )

if not ARCHITECTURE_FILE_PATH.exists():
    # Provide helpful error message with alternative locations to check
    alt_locations = [
        project_root / "outputs" / "architectures" / "best_architecture_4_trial_178.json",
        current_dir / "best_architecture_4_trial_178.json",
        Path.cwd() / ARCHITECTURE_FILE_RELATIVE,
    ]
    
    error_msg = (
        f"Architecture file not found: {ARCHITECTURE_FILE_PATH}\n"
        f"Expected relative to project root: {ARCHITECTURE_FILE_RELATIVE}\n"
        f"Project root: {project_root}\n"
        f"Current working directory: {Path.cwd()}\n"
        f"\nChecked locations:\n"
    )
    for alt in alt_locations:
        exists = "✓" if alt.exists() else "✗"
        error_msg += f"  {exists} {alt}\n"
    
    raise FileNotFoundError(error_msg)

if not ARCHITECTURE_FILE_PATH.is_file():
    raise RuntimeError(
        f"Architecture path exists but is not a file: {ARCHITECTURE_FILE_PATH}"
    )

print(f"[DEBUG] Architecture file found: {ARCHITECTURE_FILE_PATH}", file=sys.stderr)
print(f"[DEBUG] Architecture file size: {ARCHITECTURE_FILE_PATH.stat().st_size} bytes", file=sys.stderr)

# Create output directories with error handling
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directories created/exist:", file=sys.stderr)
    print(f"[DEBUG]   - Results: {RESULTS_DIR.absolute()}", file=sys.stderr)
    print(f"[DEBUG]   - Plots: {PLOTS_DIR.absolute()}", file=sys.stderr)
    print(f"[DEBUG]   - Models: {MODELS_DIR.absolute()}", file=sys.stderr)
except Exception as e:
    raise RuntimeError(f"Failed to create output directories: {e}")

# Sanity check: Verify we can write to results directory
test_file = RESULTS_DIR / ".write_test"
try:
    test_file.touch()
    test_file.unlink()
except Exception as e:
    raise RuntimeError(f"Cannot write to results directory {RESULTS_DIR}: {e}")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_baseline_results() -> Optional[pd.DataFrame]:
    """Load baseline results if they exist."""
    baseline_file = RESULTS_DIR / "baseline_results.csv"
    if baseline_file.exists():
        return pd.read_csv(baseline_file)
    return None


def run_ablation_experiment(
    model_name: str,
    model_factory,
    ablation_name: str,
    seed: int
) -> pd.DataFrame:
    """
    Run ablation experiment for a given model variant with a single seed.
    
    Args:
        model_name: Name of the model in registry
        model_factory: Function to create the model
        ablation_name: Name of the ablation (for file naming)
        seed: Random seed to use for this run
        
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*60}")
    print(f"Running ablation: {ablation_name}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Sanity check: Validate inputs
    if not isinstance(seed, int) or seed <= 0:
        raise ValueError(f"Invalid seed: {seed}. Must be a positive integer.")
    
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"Invalid model_name: {model_name}")
    
    print(f"[DEBUG] Model name: {model_name}")
    print(f"[DEBUG] Ablation name: {ablation_name}")
    print(f"[DEBUG] Seed: {seed}")
    
    # Sanity check: Verify model is registered
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_name}' not found in MODEL_REGISTRY.\n"
            f"Available models: {available_models}"
        )
    print(f"[DEBUG] Model '{model_name}' found in registry")
    
    # Set random seeds
    print(f"[DEBUG] Setting random seeds to {seed}")
    try:
        set_seeds(seed)
        print(f"[DEBUG] Seeds set successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to set seeds: {e}")
    
    try:
        # Get subjects for the dataset
        print(f"[DEBUG] Loading dataset: {DATASET}")
        try:
            dataset_obj = BNCI2014_001()
            print(f"[DEBUG] Dataset loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {DATASET}: {e}")
        
        # Get subjects list
        if hasattr(dataset_obj, 'subject_list') and dataset_obj.subject_list:
            subjects = dataset_obj.subject_list
            print(f"[DEBUG] Using dataset.subject_list: {subjects}")
        else:
            subjects = list(range(1, 10))
            print(f"[DEBUG] Using default subjects (1-9): {subjects}")
        
        if not subjects or len(subjects) == 0:
            raise ValueError(f"No subjects available for dataset {DATASET}")
        
        print(f"[DEBUG] Number of subjects: {len(subjects)}")
        
        # Sanity check: Verify CUDA availability if using GPU
        if torch.cuda.is_available():
            print(f"[DEBUG] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[DEBUG] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"[DEBUG] CUDA not available, using CPU")
        
        # Create experiment runner
        print(f"[DEBUG] Creating UnifiedExperimentRunner...")
        print(f"[DEBUG]   - Model: {model_name}")
        print(f"[DEBUG]   - Dataset: {DATASET}")
        print(f"[DEBUG]   - Subjects: {subjects}")
        print(f"[DEBUG]   - Mode: test_perturb")
        print(f"[DEBUG]   - Eval mode: {EVAL_MODE}")
        print(f"[DEBUG]   - Seed: {seed}")
        
        try:
            runner = UnifiedExperimentRunner(
                model=model_name,
                dataset=DATASET,
                subjects=subjects,
                mode="test_perturb",  # test_perturb mode evaluates on multiple noise types/intensities
                eval_mode=EVAL_MODE,
                seed=seed,
                noise_type="gaussian",  # Placeholder, test_perturb handles all noise types
                intensity=10.0,  # Placeholder, test_perturb handles all intensities
                tune=False,  # No hyperparameter tuning
                overwrite=False
            )
            print(f"[DEBUG] UnifiedExperimentRunner created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create UnifiedExperimentRunner: {e}\n{traceback.format_exc()}")
        
        # Run experiment
        print(f"[DEBUG] Starting experiment run...")
        try:
            results_df = runner.run_experiment()
            print(f"[DEBUG] Experiment run completed")
        except Exception as e:
            raise RuntimeError(f"Experiment run failed: {e}\n{traceback.format_exc()}")
        
        # Sanity check: Verify results
        if results_df is None:
            raise RuntimeError("Experiment returned None results")
        
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(results_df)}")
        
        if results_df.empty:
            raise RuntimeError("Experiment returned empty DataFrame")
        
        print(f"[DEBUG] Results DataFrame shape: {results_df.shape}")
        print(f"[DEBUG] Results columns: {list(results_df.columns)}")
        
        # Add metadata
        results_df['ablation'] = ablation_name
        results_df['run'] = 1
        results_df['seed'] = seed
        
        print(f"[DEBUG] Added metadata columns: ablation, run, seed")
        
        # Save results
        results_file = RESULTS_DIR / f"{ablation_name}_seed{seed}.csv"
        print(f"[DEBUG] Saving results to: {results_file.absolute()}")
        
        try:
            results_df.to_csv(results_file, index=False)
            print(f"[DEBUG] Results saved successfully ({len(results_df)} rows)")
            
            # Sanity check: Verify file was created and readable
            if not results_file.exists():
                raise RuntimeError(f"Results file was not created: {results_file}")
            
            # Verify we can read it back
            verify_df = pd.read_csv(results_file)
            if len(verify_df) != len(results_df):
                raise RuntimeError(
                    f"File verification failed: saved {len(results_df)} rows, "
                    f"read back {len(verify_df)} rows"
                )
            print(f"[DEBUG] File verification passed")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save results to {results_file}: {e}")
        
        print(f"\nSaved results to {results_file}")
        
        return results_df
        
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR in ablation {ablation_name} (seed={seed})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        print(f"{'='*60}\n", file=sys.stderr)
        return pd.DataFrame()


def perform_statistical_tests(
    baseline_df: pd.DataFrame,
    ablation_dfs: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Perform statistical tests comparing each ablation to baseline.
    
    Args:
        baseline_df: Baseline results DataFrame
        ablation_dfs: Dictionary mapping ablation names to their results DataFrames
        
    Returns:
        DataFrame with statistical test results
    """
    print("\n" + "="*60)
    print("Performing Statistical Tests")
    print("="*60 + "\n")
    
    # Extract clean ROC-AUC scores for comparison
    metric = 'validation_roc_auc'
    
    # Get baseline scores (aggregate across subjects/folds if needed)
    if 'subject' in baseline_df.columns:
        baseline_scores = baseline_df.groupby(['subject', 'run'])[metric].mean().values
    else:
        baseline_scores = baseline_df[metric].values
    
    test_results = []
    
    for ablation_name, ablation_df in ablation_dfs.items():
        if ablation_df.empty:
            continue
            
        # Get ablation scores
        if 'subject' in ablation_df.columns:
            ablation_scores = ablation_df.groupby(['subject', 'run'])[metric].mean().values
        else:
            ablation_scores = ablation_df[metric].values
        
        # Ensure same length (take minimum)
        min_len = min(len(baseline_scores), len(ablation_scores))
        baseline_subset = baseline_scores[:min_len]
        ablation_subset = ablation_scores[:min_len]
        
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(baseline_subset, ablation_subset)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = stats.wilcoxon(baseline_subset, ablation_subset)
        
        # Effect size (Cohen's d)
        diff = baseline_subset - ablation_subset
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        # Means and standard deviations
        baseline_mean = np.mean(baseline_subset)
        baseline_std = np.std(baseline_subset)
        ablation_mean = np.mean(ablation_subset)
        ablation_std = np.std(ablation_subset)
        
        test_results.append({
            'ablation': ablation_name,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'ablation_mean': ablation_mean,
            'ablation_std': ablation_std,
            'mean_difference': baseline_mean - ablation_mean,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'cohens_d': cohens_d,
            'significant_t_test': t_pvalue < 0.05,
            'significant_wilcoxon': w_pvalue < 0.05,
        })
        
        print(f"{ablation_name}:")
        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Ablation: {ablation_mean:.4f} ± {ablation_std:.4f}")
        print(f"  Difference: {baseline_mean - ablation_mean:.4f}")
        print(f"  T-test p-value: {t_pvalue:.4f} {'*' if t_pvalue < 0.05 else ''}")
        print(f"  Wilcoxon p-value: {w_pvalue:.4f} {'*' if w_pvalue < 0.05 else ''}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print()
    
    # Apply Bonferroni correction
    num_tests = len(test_results)
    for result in test_results:
        result['t_pvalue_corrected'] = min(result['t_pvalue'] * num_tests, 1.0)
        result['wilcoxon_pvalue_corrected'] = min(result['wilcoxon_pvalue'] * num_tests, 1.0)
        result['significant_t_test_corrected'] = result['t_pvalue_corrected'] < 0.05
        result['significant_wilcoxon_corrected'] = result['wilcoxon_pvalue_corrected'] < 0.05
    
    stats_df = pd.DataFrame(test_results)
    stats_file = RESULTS_DIR / "statistical_tests.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistical test results to {stats_file}")
    
    return stats_df


def create_plots(
    baseline_df: pd.DataFrame,
    ablation_dfs: Dict[str, pd.DataFrame],
    stats_df: pd.DataFrame
):
    """Create visualization plots for the ablation studies."""
    print("\n" + "="*60)
    print("Creating Plots")
    print("="*60 + "\n")
    
    metric = 'validation_roc_auc'
    
    # Prepare data for plotting
    plot_data = []
    
    # Add baseline
    if 'subject' in baseline_df.columns:
        baseline_scores = baseline_df.groupby(['subject', 'run'])[metric].mean().reset_index()
        for _, row in baseline_scores.iterrows():
            plot_data.append({
                'Model': 'Baseline (Full HYDRA)',
                'ROC-AUC': row[metric],
                'Run': row['run']
            })
    else:
        for _, row in baseline_df.iterrows():
            plot_data.append({
                'Model': 'Baseline (Full HYDRA)',
                'ROC-AUC': row[metric],
                'Run': row.get('run', 1)
            })
    
    # Add ablations
    for ablation_name, ablation_df in ablation_dfs.items():
        if ablation_df.empty:
            continue
        
        display_name = {
            'ablation1_no_carry_gate': 'No Carry Gate',
            'ablation2_no_branching': 'No Branching',
            'ablation3_lstm_replacement': 'LSTM (vs CfC)'
        }.get(ablation_name, ablation_name)
        
        if 'subject' in ablation_df.columns:
            ablation_scores = ablation_df.groupby(['subject', 'run'])[metric].mean().reset_index()
            for _, row in ablation_scores.iterrows():
                plot_data.append({
                    'Model': display_name,
                    'ROC-AUC': row[metric],
                    'Run': row['run']
                })
        else:
            for _, row in ablation_df.iterrows():
                plot_data.append({
                    'Model': display_name,
                    'ROC-AUC': row[metric],
                    'Run': row.get('run', 1)
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Box plot comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, x='Model', y='ROC-AUC', palette='Set2')
    plt.title('Ablation Studies: Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Model Variant', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    boxplot_file = PLOTS_DIR / "ablation_comparison_boxplot.png"
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    print(f"Saved box plot to {boxplot_file}")
    plt.close()
    
    # 2. Violin plot with individual points
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x='Model', y='ROC-AUC', palette='Set2', inner='box')
    sns.stripplot(data=plot_df, x='Model', y='ROC-AUC', color='black', alpha=0.3, size=3)
    plt.title('Ablation Studies: Performance Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Model Variant', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    violin_file = PLOTS_DIR / "ablation_comparison_violin.png"
    plt.savefig(violin_file, dpi=300, bbox_inches='tight')
    print(f"Saved violin plot to {violin_file}")
    plt.close()
    
    # 3. Statistical summary table plot
    if not stats_df.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in stats_df.iterrows():
            table_data.append([
                row['ablation'].replace('ablation', 'Ablation ').replace('_', ' ').title(),
                f"{row['baseline_mean']:.4f} ± {row['baseline_std']:.4f}",
                f"{row['ablation_mean']:.4f} ± {row['ablation_std']:.4f}",
                f"{row['mean_difference']:.4f}",
                f"{row['t_pvalue_corrected']:.4f}",
                f"{row['cohens_d']:.4f}",
                '*' if row['significant_t_test_corrected'] else ''
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Ablation', 'Baseline (Mean ± Std)', 'Ablation (Mean ± Std)', 
                      'Difference', 'P-value (corrected)', "Cohen's d", 'Significant'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        plt.title('Statistical Test Results Summary', fontsize=14, fontweight='bold', pad=20)
        stats_table_file = PLOTS_DIR / "statistical_summary.png"
        plt.savefig(stats_table_file, dpi=300, bbox_inches='tight')
        print(f"Saved statistical summary table to {stats_table_file}")
        plt.close()


def register_model_variants(wiring):
    """Register all model variants for ablation studies."""
    print(f"[DEBUG] Registering model variants...")
    
    # Sanity check: Verify wiring is valid
    if wiring is None:
        raise ValueError("Wiring cannot be None")
    print(f"[DEBUG] Wiring validated: {type(wiring)}")
    
    # 1. Baseline (full model) - should already be registered, but ensure it's there
    try:
        add_branched_wiredcfc_architecture("branched_wiredcfc_arch4", wiring)
        print(f"[DEBUG] Registered: branched_wiredcfc_arch4")
    except Exception as e:
        raise RuntimeError(f"Failed to register baseline model: {e}")
    
    # 2. No Carry Gate - create factory with proper closure
    try:
        def create_no_carry_gate_factory(wiring_ref):
            def factory(n_chans, n_times, n_outputs, **kwargs):
                return create_branched_wiredcfc_no_carry_gate_classifier(n_chans, n_times, n_outputs, wiring_ref, **kwargs)
            return factory
        MODEL_REGISTRY["branched_wiredcfc_arch4_no_carry_gate"] = create_no_carry_gate_factory(wiring)
        print(f"[DEBUG] Registered: branched_wiredcfc_arch4_no_carry_gate")
    except Exception as e:
        raise RuntimeError(f"Failed to register no_carry_gate model: {e}")
    
    # 3. No Branching - create factory with proper closure
    try:
        def create_no_branching_factory(wiring_ref):
            def factory(n_chans, n_times, n_outputs, **kwargs):
                return create_branched_wiredcfc_no_branching_classifier(n_chans, n_times, n_outputs, wiring_ref, **kwargs)
            return factory
        MODEL_REGISTRY["branched_wiredcfc_arch4_no_branching"] = create_no_branching_factory(wiring)
        print(f"[DEBUG] Registered: branched_wiredcfc_arch4_no_branching")
    except Exception as e:
        raise RuntimeError(f"Failed to register no_branching model: {e}")
    
    # 4. LSTM Replacement (use BranchedLSTM with equivalent parameters)
    # Note: This uses BranchedLSTM which doesn't use wiring, so we'll use the standard factory
    try:
        MODEL_REGISTRY["branched_lstm_arch4_equivalent"] = create_branched_lstm_classifier
        print(f"[DEBUG] Registered: branched_lstm_arch4_equivalent")
    except Exception as e:
        raise RuntimeError(f"Failed to register LSTM model: {e}")
    
    # Sanity check: Verify all models are registered
    expected_models = [
        "branched_wiredcfc_arch4",
        "branched_wiredcfc_arch4_no_carry_gate",
        "branched_wiredcfc_arch4_no_branching",
        "branched_lstm_arch4_equivalent"
    ]
    missing_models = [m for m in expected_models if m not in MODEL_REGISTRY]
    if missing_models:
        raise RuntimeError(f"Failed to register models: {missing_models}")
    
    print(f"[DEBUG] All models registered successfully")


def get_ablation_config(ablation_num: str):
    """
    Get model name and ablation name for a given ablation number.
    
    Args:
        ablation_num: Ablation number as string ("baseline", "1", "2", or "3")
        
    Returns:
        Tuple of (model_name, ablation_name)
    """
    ablation_map = {
        "baseline": ("branched_wiredcfc_arch4", "baseline"),
        "1": ("branched_wiredcfc_arch4_no_carry_gate", "ablation1_no_carry_gate"),
        "2": ("branched_wiredcfc_arch4_no_branching", "ablation2_no_branching"),
        "3": ("branched_lstm_arch4_equivalent", "ablation3_lstm_replacement"),
    }
    
    if ablation_num.lower() not in ablation_map:
        raise ValueError(f"Invalid ablation number: {ablation_num}. Must be 'baseline', '1', '2', or '3'")
    
    return ablation_map[ablation_num.lower()]


def main():
    """Main function to run a single ablation experiment with specified seed."""
    parser = argparse.ArgumentParser(
        description="Run ablation study experiment for HYDRA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ablations.py --ablation baseline --seed 100
  python run_ablations.py --ablation 1 --seed 200
  python run_ablations.py --ablation 2 --seed 300
  python run_ablations.py --ablation 3 --seed 400
        """
    )
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["baseline", "1", "2", "3"],
        help="Ablation number: 'baseline', '1' (No Carry Gate), '2' (No Branching), or '3' (LSTM Replacement)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for the experiment (e.g., 100, 200, 300, 400, 500)"
    )
    
    args = parser.parse_args()
    
    # Sanity check: Validate seed
    if args.seed <= 0:
        raise ValueError(f"Invalid seed: {args.seed}. Must be a positive integer.")
    
    print("="*60)
    print("HYDRA Model Ablation Studies")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Evaluation Mode: {EVAL_MODE}")
    print(f"Ablation: {args.ablation}")
    print(f"Seed: {args.seed}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    try:
        # Load architecture 4
        print(f"\n[DEBUG] Loading architecture from {ARCHITECTURE_FILE_PATH}...")
        
        # Convert Path to string for load_architecture_from_file (it expects a string)
        architecture_file_str = str(ARCHITECTURE_FILE_PATH)
        print(f"[DEBUG] Architecture file path (string): {architecture_file_str}")
        
        # Verify file is readable before attempting to load
        if not os.access(ARCHITECTURE_FILE_PATH, os.R_OK):
            raise PermissionError(f"Cannot read architecture file: {ARCHITECTURE_FILE_PATH}")
        
        wiring = load_architecture_from_file(architecture_file_str)
        
        if wiring is None:
            raise RuntimeError("Failed to load wiring from architecture file")
        print(f"[DEBUG] Wiring loaded successfully: {type(wiring)}")
        
        # Register model variants
        print("\n[DEBUG] Registering model variants...")
        try:
            register_model_variants(wiring)
            print("[DEBUG] Model variants registered successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to register model variants: {e}\n{traceback.format_exc()}")
        
        # Get model configuration for the specified ablation
        print(f"\n[DEBUG] Getting configuration for ablation: {args.ablation}")
        try:
            model_name, ablation_name = get_ablation_config(args.ablation)
            print(f"[DEBUG] Model name: {model_name}")
            print(f"[DEBUG] Ablation name: {ablation_name}")
        except Exception as e:
            raise ValueError(f"Invalid ablation configuration: {e}")
        
        # Sanity check: Verify model is in registry
        if model_name not in MODEL_REGISTRY:
            raise KeyError(
                f"Model '{model_name}' not found in registry after registration.\n"
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )
        
        # Run the experiment
        print(f"\n{'='*60}")
        print(f"Running: {ablation_name}")
        print(f"{'='*60}")
        
        results_df = run_ablation_experiment(
            model_name=model_name,
            model_factory=None,
            ablation_name=ablation_name,
            seed=args.seed
        )
        
        if not results_df.empty:
            print(f"\n{'='*60}")
            print("Experiment Complete!")
            print("="*60)
            print(f"Results saved to: {RESULTS_DIR.absolute()}")
            print(f"Total rows: {len(results_df)}")
            print(f"Columns: {list(results_df.columns)}")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60, file=sys.stderr)
            print("ERROR: Experiment completed but no results were collected.", file=sys.stderr)
            print("="*60, file=sys.stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[ERROR] Experiment interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print("\n" + "="*60, file=sys.stderr)
        print("FATAL ERROR in main()", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        print("="*60, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

