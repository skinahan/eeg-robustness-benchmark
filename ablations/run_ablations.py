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
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

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
    ablation_models_path = os.path.join(current_dir, "ablation_models.py")
    spec = importlib.util.spec_from_file_location("ablation_models", ablation_models_path)
    ablation_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation_models)
    create_branched_wiredcfc_no_carry_gate_classifier = ablation_models.create_branched_wiredcfc_no_carry_gate_classifier
    create_branched_wiredcfc_no_branching_classifier = ablation_models.create_branched_wiredcfc_no_branching_classifier

# Configuration
ARCHITECTURE_FILE = "outputs/architectures/best_architecture_4_trial_178.json"
DATASET = "BNCI2014_001"
EVAL_MODE = "CrossSubject"
SEEDS = [100, 200, 300, 400, 500]  # 5 experimental runs
NUM_RUNS = len(SEEDS)
OUTPUT_DIR = Path(current_dir)
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

# Create output directories
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

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
    seeds: List[int]
) -> pd.DataFrame:
    """
    Run ablation experiment for a given model variant.
    
    Args:
        model_name: Name of the model in registry
        model_factory: Function to create the model
        ablation_name: Name of the ablation (for file naming)
        seeds: List of seeds to run
        
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*60}")
    print(f"Running ablation: {ablation_name}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for run_idx, seed in enumerate(seeds):
        print(f"\nRun {run_idx + 1}/{len(seeds)} (seed={seed})")
        set_seeds(seed)
        
        try:
            # Get subjects for the dataset
            dataset_obj = BNCI2014_001()
            subjects = dataset_obj.subject_list if hasattr(dataset_obj, 'subject_list') else list(range(1, 10))
            
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
            
            results_df = runner.run_experiment()
            
            # Add metadata
            results_df['ablation'] = ablation_name
            results_df['run'] = run_idx + 1
            results_df['seed'] = seed
            
            all_results.append(results_df)
            
            # Save partial results after each run
            partial_file = RESULTS_DIR / f"{ablation_name}_partial_run{run_idx+1}.csv"
            results_df.to_csv(partial_file, index=False)
            print(f"Saved partial results to {partial_file}")
            
        except Exception as e:
            print(f"Error in run {run_idx + 1} (seed={seed}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print(f"Warning: No results collected for {ablation_name}")
        return pd.DataFrame()
    
    # Combine all runs
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    results_file = RESULTS_DIR / f"{ablation_name}_results.csv"
    combined_results.to_csv(results_file, index=False)
    print(f"\nSaved combined results to {results_file}")
    
    return combined_results


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


def main():
    """Main function to run all ablation experiments."""
    print("="*60)
    print("HYDRA Model Ablation Studies")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Evaluation Mode: {EVAL_MODE}")
    print(f"Number of Runs per Ablation: {NUM_RUNS}")
    print(f"Seeds: {SEEDS}")
    print("="*60)
    
    # Load architecture 4
    print(f"\nLoading architecture from {ARCHITECTURE_FILE}...")
    wiring = load_architecture_from_file(ARCHITECTURE_FILE)
    
    # Register model variants
    print("\nRegistering model variants...")
    
    # 1. Baseline (full model) - should already be registered, but ensure it's there
    add_branched_wiredcfc_architecture("branched_wiredcfc_arch4", wiring)
    
    # 2. No Carry Gate - create factory with proper closure
    def create_no_carry_gate_factory(wiring_ref):
        def factory(n_chans, n_times, n_outputs, **kwargs):
            return create_branched_wiredcfc_no_carry_gate_classifier(n_chans, n_times, n_outputs, wiring_ref, **kwargs)
        return factory
    MODEL_REGISTRY["branched_wiredcfc_arch4_no_carry_gate"] = create_no_carry_gate_factory(wiring)
    
    # 3. No Branching - create factory with proper closure
    def create_no_branching_factory(wiring_ref):
        def factory(n_chans, n_times, n_outputs, **kwargs):
            return create_branched_wiredcfc_no_branching_classifier(n_chans, n_times, n_outputs, wiring_ref, **kwargs)
        return factory
    MODEL_REGISTRY["branched_wiredcfc_arch4_no_branching"] = create_no_branching_factory(wiring)
    
    # 4. LSTM Replacement (use BranchedLSTM with equivalent parameters)
    # Note: This uses BranchedLSTM which doesn't use wiring, so we'll use the standard factory
    MODEL_REGISTRY["branched_lstm_arch4_equivalent"] = create_branched_lstm_classifier
    
    print("Model variants registered successfully.")
    
    # ========================================================================
    # PHASE 1: RUN ALL EXPERIMENTS (5 runs per ablation)
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Running All Experiments")
    print("="*60)
    print("This phase will run 5 experimental runs for each ablation.")
    print("Statistical tests and plotting will be performed after all experiments complete.\n")
    
    # Load baseline results if available
    print("Checking for existing baseline results...")
    baseline_df = load_baseline_results()
    
    if baseline_df is None:
        print("Baseline results not found. Running baseline experiment...")
        print(f"Running {NUM_RUNS} runs with seeds {SEEDS}...")
        baseline_df = run_ablation_experiment(
            "branched_wiredcfc_arch4",
            None,
            "baseline",
            SEEDS
        )
        if not baseline_df.empty:
            baseline_file = RESULTS_DIR / "baseline_results.csv"
            baseline_df.to_csv(baseline_file, index=False)
            print(f"Baseline experiment complete. Results saved to {baseline_file}")
    else:
        print(f"Loaded baseline results from {RESULTS_DIR / 'baseline_results.csv'}")
        print(f"Baseline has {len(baseline_df)} rows from {baseline_df['run'].nunique()} runs")
    
    # Run ablation experiments (5 runs each)
    ablation_dfs = {}
    
    # Ablation 1: No Carry Gate
    print(f"\n{'='*60}")
    print("Ablation 1: No Carry Gate")
    print(f"{'='*60}")
    ablation1_df = run_ablation_experiment(
        "branched_wiredcfc_arch4_no_carry_gate",
        None,
        "ablation1_no_carry_gate",
        SEEDS
    )
    if not ablation1_df.empty:
        ablation_dfs["ablation1_no_carry_gate"] = ablation1_df
        print(f"Ablation 1 complete. Results saved.")
    
    # Ablation 2: No Branching
    print(f"\n{'='*60}")
    print("Ablation 2: No Branching")
    print(f"{'='*60}")
    ablation2_df = run_ablation_experiment(
        "branched_wiredcfc_arch4_no_branching",
        None,
        "ablation2_no_branching",
        SEEDS
    )
    if not ablation2_df.empty:
        ablation_dfs["ablation2_no_branching"] = ablation2_df
        print(f"Ablation 2 complete. Results saved.")
    
    # Ablation 3: LSTM Replacement
    print(f"\n{'='*60}")
    print("Ablation 3: LSTM Replacement")
    print(f"{'='*60}")
    ablation3_df = run_ablation_experiment(
        "branched_lstm_arch4_equivalent",
        None,
        "ablation3_lstm_replacement",
        SEEDS
    )
    if not ablation3_df.empty:
        ablation_dfs["ablation3_lstm_replacement"] = ablation3_df
        print(f"Ablation 3 complete. Results saved.")
    
    # ========================================================================
    # PHASE 2: STATISTICAL ANALYSIS AND PLOTTING (only after all experiments)
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Statistical Analysis and Plotting")
    print("="*60)
    print("All experiments complete. Now performing statistical tests and creating plots...\n")
    
    # Verify we have all required data
    if baseline_df is None or baseline_df.empty:
        print("WARNING: No baseline results available. Skipping statistical analysis.")
    elif not ablation_dfs:
        print("WARNING: No ablation results available. Skipping statistical analysis.")
    else:
        # Perform statistical tests
        stats_df = perform_statistical_tests(baseline_df, ablation_dfs)
        
        # Create plots
        create_plots(baseline_df, ablation_dfs, stats_df)
        
        # Combine all results
        all_results = [baseline_df]
        all_results.extend(ablation_dfs.values())
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = RESULTS_DIR / "combined_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\nSaved combined results to {combined_file}")
    
    print("\n" + "="*60)
    print("Ablation Studies Complete!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"\nSummary:")
    print(f"  - Baseline: {len(baseline_df) if baseline_df is not None and not baseline_df.empty else 0} rows")
    for name, df in ablation_dfs.items():
        print(f"  - {name}: {len(df)} rows")


if __name__ == "__main__":
    main()

