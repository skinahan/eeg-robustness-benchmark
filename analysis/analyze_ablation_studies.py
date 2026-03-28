"""
Analyze Ablation Studies Results

This script analyzes the results from ablation studies on the HYDRA model.
It computes clean scores, statistical measures, and robustness metrics comparing
each ablation variant to the baseline (full HYDRA model).

Based on the ablation experiment specification in ablations/experiment_specification.txt:
- Baseline: Full HYDRA Model (branched_wiredcfc_arch4)
- Ablation 1: No Carry Gate (branched_wiredcfc_arch4_no_carry_gate)
- Ablation 2: No Branching (branched_wiredcfc_arch4_no_branching)
- Ablation 3: LSTM Replacement (branched_lstm_arch4_equivalent)

Usage:
    python analysis/analyze_ablation_studies.py [--results-dir PATH] [--output-dir PATH]
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import analysis utilities
from analysis.calculate_clean_scores import (
    canonicalize_columns,
    compute_clean_scores_summary,
    format_clean_scores_table,
)
from analysis.statistical_analysis import (
    AnalysisConfig,
    aggregate_seeds,
    compute_aupc_per_subject,
    compute_clean_scores_per_subject,
    compute_rd_per_subject,
    build_inference_dataset,
    check_normality,
    compute_cohens_dz,
    bootstrap_ci_cohens_dz,
)
from analysis.robustness_metrics import (
    MetricConfig,
    add_normalized_p,
    find_subject_col,
    replace_hydra_model_name,
)


# ----------------------------
# Configuration
# ----------------------------

ABLATION_NAMES = {
    'baseline': 'Baseline (Full HYDRA)',
    'ablation1_no_carry_gate': 'Ablation 1: No Carry Gate',
    'ablation2_no_branching': 'Ablation 2: No Branching',
    'ablation3_lstm_replacement': 'Ablation 3: LSTM Replacement',
}

ABLATION_MODEL_NAMES = {
    'baseline': 'branched_wiredcfc_arch4',
    'ablation1_no_carry_gate': 'branched_wiredcfc_arch4_no_carry_gate',
    'ablation2_no_branching': 'branched_wiredcfc_arch4_no_branching',
    'ablation3_lstm_replacement': 'branched_lstm_arch4_equivalent',
}


# ----------------------------
# Data Loading
# ----------------------------

def load_baseline_from_unified(unified_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load baseline (branched_wiredcfc_arch4) results from unified_all_results.csv.
    
    Parameters:
    -----------
    unified_file : str, optional
        Path to unified_all_results.csv. If None, uses default location.
        
    Returns:
    --------
    pd.DataFrame
        Baseline results DataFrame (may be empty if not found)
    """
    if unified_file is None:
        unified_file = os.path.join(_project_root, "evaluation", "results", "unified_all_results.csv")
    
    if not os.path.exists(unified_file):
        print(f"  [WARNING] Unified results file not found: {unified_file}")
        return pd.DataFrame()
    
    print(f"[INFO] Loading baseline from unified results: {unified_file}")
    
    try:
        df = pd.read_csv(unified_file, low_memory=False)
        print(f"  [INFO] Loaded {len(df)} total rows from unified file")
        
        # Canonicalize column names
        df = canonicalize_columns(df)
        
        # Filter to branched_wiredcfc_arch4 model
        if 'model' not in df.columns:
            print("  [WARNING] No 'model' column found in unified results")
            return pd.DataFrame()
        
        # Handle model name variations
        baseline_model_patterns = ['branched_wiredcfc_arch4', 'branched-wiredcfc-arch4']
        model_col = df['model'].astype(str).str.strip().str.lower().str.replace('-', '_')
        
        baseline_mask = False
        for pattern in baseline_model_patterns:
            pattern_normalized = pattern.lower().replace('-', '_')
            baseline_mask = baseline_mask | (model_col == pattern_normalized)
        
        baseline_df = df[baseline_mask].copy()
        
        if baseline_df.empty:
            print(f"  [WARNING] No baseline (branched_wiredcfc_arch4) results found in unified file")
            print(f"  [INFO] Available models: {df['model'].unique()[:10]}")
            return pd.DataFrame()
        
        # Normalize model name to standard format
        baseline_df['model'] = 'branched_wiredcfc_arch4'
        
        # Filter to match ablation study conditions exactly:
        # - BNCI2014_001 dataset (ablation studies use this dataset)
        # - CrossSubject eval_mode (ablation studies use CrossSubject)
        # - test_perturb mode (ablation studies use test_perturb, not test_perturb_tune)
        # - tune == False (ablation studies don't use hyperparameter tuning)
        # - Valid seeds: [100, 200, 300, 400, 500] (ablation studies use these seeds)
        
        if 'dataset' in baseline_df.columns:
            baseline_df = baseline_df[baseline_df['dataset'] == 'BNCI2014_001'].copy()
            print(f"  [INFO] Filtered to BNCI2014_001 dataset: {len(baseline_df)} rows")
        
        if 'tune' in baseline_df.columns:
            baseline_df = baseline_df[baseline_df['tune'] == False].copy()
            print(f"  [INFO] Filtered to tune=False: {len(baseline_df)} rows")
        elif 'mode' in baseline_df.columns:
            # If tune column doesn't exist, infer from mode column
            # test_perturb_tune means tune=True, test_perturb means tune=False
            baseline_df = baseline_df[~baseline_df['mode'].astype(str).str.contains('_tune', na=False)].copy()
            print(f"  [INFO] Filtered to non-tuned mode: {len(baseline_df)} rows")
        
        if 'eval_mode' in baseline_df.columns:
            # Handle variations like 'CrossSubjectEvaluation' -> 'CrossSubject'
            eval_mode_normalized = baseline_df['eval_mode'].astype(str).str.replace('Evaluation', '', regex=False).str.strip()
            baseline_df = baseline_df[eval_mode_normalized.str.contains('CrossSubject', case=False, na=False)].copy()
            print(f"  [INFO] Filtered to CrossSubject eval_mode: {len(baseline_df)} rows")
        
        if 'mode' in baseline_df.columns:
            # Filter to test_perturb (ablation studies use test_perturb, not test_perturb_tune)
            mode_normalized = baseline_df['mode'].astype(str).str.replace('_tune', '', regex=False).str.strip()
            baseline_df = baseline_df[mode_normalized == 'test_perturb'].copy()
            print(f"  [INFO] Filtered to test_perturb mode: {len(baseline_df)} rows")
        
        # Filter to valid seeds: [100, 200, 300, 400, 500]
        valid_seeds = [100, 200, 300, 400, 500]
        if 'seed' in baseline_df.columns:
            # Convert seed to numeric, handling any string representations
            baseline_df['seed'] = pd.to_numeric(baseline_df['seed'], errors='coerce')
            initial_len = len(baseline_df)
            baseline_df = baseline_df[baseline_df['seed'].isin(valid_seeds)].copy()
            filtered_count = initial_len - len(baseline_df)
            if filtered_count > 0:
                print(f"  [INFO] Filtered to valid seeds {valid_seeds}: removed {filtered_count} rows, kept {len(baseline_df)} rows")
            else:
                print(f"  [INFO] All rows already have valid seeds {valid_seeds}: {len(baseline_df)} rows")
        else:
            print(f"  [WARNING] No 'seed' column found - cannot filter by seed values")
        
        print(f"  [OK] Loaded baseline: {len(baseline_df)} rows")
        return baseline_df
        
    except Exception as e:
        print(f"  [ERROR] Failed to load baseline from unified file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_ablation_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all ablation results from the results directory.
    Baseline is loaded from unified_all_results.csv instead of ablation results directory.
    
    Parameters:
    -----------
    results_dir : str
        Path to ablations/results/ directory
        
    Returns:
    --------
    dict
        Dictionary mapping ablation names to DataFrames
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"[INFO] Loading ablation results from: {results_dir}")
    
    ablation_results = {}
    
    # Load baseline results from unified_all_results.csv
    baseline_df = load_baseline_from_unified()
    if not baseline_df.empty:
        ablation_results['baseline'] = baseline_df
    
    # Load ablation results
    for ablation_key, ablation_name in ABLATION_NAMES.items():
        if ablation_key == 'baseline':
            continue
        
        # Look for files matching the ablation pattern
        pattern = f"{ablation_key}*.csv"
        ablation_files = list(results_dir.glob(pattern))
        
        if ablation_files:
            ablation_dfs = []
            for f in ablation_files:
                df = pd.read_csv(f)
                ablation_dfs.append(df)
            if ablation_dfs:
                ablation_results[ablation_key] = pd.concat(ablation_dfs, ignore_index=True)
                print(f"  [OK] Loaded {ablation_key}: {len(ablation_results[ablation_key])} rows from {len(ablation_files)} files")
    
    if not ablation_results:
        raise ValueError(f"No ablation results found in {results_dir}")
    
    print(f"[OK] Loaded {len(ablation_results)} ablation result sets")
    return ablation_results


def prepare_ablation_dataframe(ablation_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all ablation results into a single DataFrame with proper metadata.
    
    Parameters:
    -----------
    ablation_results : dict
        Dictionary mapping ablation names to DataFrames
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with ablation metadata
    """
    print("[INFO] Preparing combined ablation DataFrame...")
    
    combined_dfs = []
    
    for ablation_key, df in ablation_results.items():
        df = df.copy()
        
        # Add ablation identifier
        df['ablation'] = ablation_key
        df['ablation_name'] = ABLATION_NAMES.get(ablation_key, ablation_key)
        
        # Ensure model name is set correctly
        if 'model' not in df.columns or df['model'].isna().all():
            model_name = ABLATION_MODEL_NAMES.get(ablation_key)
            if model_name:
                df['model'] = model_name
        
        combined_dfs.append(df)
    
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Canonicalize column names
    combined_df = canonicalize_columns(combined_df)
    
    print(f"  [OK] Combined DataFrame: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    return combined_df


# ----------------------------
# Clean Scores Computation
# ----------------------------

def compute_ablation_clean_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute clean scores for each ablation variant.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined ablation results DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Clean scores summary by ablation
    """
    print("\n[STEP 1] Computing clean scores for ablation studies...")
    
    # Detect clean metric column
    clean_metric_candidates = ['clean_roc_auc', 'clean_score', 'validation_roc_auc']
    clean_metric_col = None
    for candidate in clean_metric_candidates:
        if candidate in df.columns:
            clean_metric_col = candidate
            break
    
    if not clean_metric_col:
        # Try to infer from metric at intensity=0
        metric_cols = ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']
        if 'intensity' in df.columns:
            for metric_candidate in metric_cols:
                if metric_candidate in df.columns:
                    clean_df = df[df['intensity'] == 0.0].copy()
                    if not clean_df.empty:
                        clean_metric_col = metric_candidate
                        print(f"  [INFO] Using {metric_candidate} at intensity=0.0 as clean metric")
                        break
        
        if not clean_metric_col:
            print(f"  [WARNING] No clean metric column found. Tried: {clean_metric_candidates}")
            print(f"  [INFO] Available columns: {list(df.columns)}")
            print(f"  [INFO] Will try to infer from intensity=0 data if available")
            # Try one more time with any available metric column
            for metric_candidate in ['corrupted_score', 'score', 'roc_auc']:
                if metric_candidate in df.columns:
                    clean_metric_col = metric_candidate
                    print(f"  [INFO] Using {metric_candidate} as fallback clean metric")
                    break
    
    if not clean_metric_col:
        print("  [WARNING] Could not determine clean metric column - skipping clean scores computation")
        return pd.DataFrame()
    
    # For clean scores, we use the clean_score/clean_roc_auc column directly
    # Clean scores are the same across all noise types for a given model/subject/seed
    # We don't need to filter by intensity=0.0 - the clean_score column already contains
    # the baseline performance for each unique evaluation
    clean_data = df.dropna(subset=[clean_metric_col]).copy()
    
    if clean_data.empty:
        print(f"  [WARNING] No clean data found (no valid {clean_metric_col} values)")
        print(f"  [INFO] Checking intensity column...")
        # Fallback: try intensity=0.0 if clean_score column doesn't work
        if 'intensity' in df.columns:
            intensity_zero = df[df['intensity'] == 0.0].copy()
            if not intensity_zero.empty:
                # Use corrupted_score at intensity=0 as clean score
                if 'corrupted_score' in intensity_zero.columns:
                    clean_metric_col = 'corrupted_score'
                    clean_data = intensity_zero.copy()
                    print(f"  [INFO] Using corrupted_score at intensity=0.0 as clean metric")
                elif 'corrupted_roc_auc' in intensity_zero.columns:
                    clean_metric_col = 'corrupted_roc_auc'
                    clean_data = intensity_zero.copy()
                    print(f"  [INFO] Using corrupted_roc_auc at intensity=0.0 as clean metric")
        
        if clean_data.empty:
            print("  [WARNING] No clean data found even after fallback")
            return pd.DataFrame()
    
    # Group by ablation and compute statistics
    group_cols = ['ablation', 'ablation_name']
    if 'seed' in clean_data.columns:
        group_cols.append('seed')
    if 'subject' in clean_data.columns:
        group_cols.append('subject')
    
    # Get unique clean scores per group
    clean_scores_list = []
    for keys, group_df in clean_data.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        row_dict = dict(zip(group_cols, keys))
        
        # Get clean score values
        clean_values = group_df[clean_metric_col].dropna().unique()
        if len(clean_values) > 0:
            row_dict['clean_score'] = float(np.median(clean_values))
            clean_scores_list.append(row_dict)
    
    if not clean_scores_list:
        return pd.DataFrame()
    
    clean_scores_df = pd.DataFrame(clean_scores_list)
    
    # Aggregate across seeds/subjects to get mean and std per ablation
    summary_rows = []
    for ablation_key in clean_scores_df['ablation'].unique():
        ablation_scores = clean_scores_df[clean_scores_df['ablation'] == ablation_key]['clean_score'].values
        
        if len(ablation_scores) > 0:
            summary_rows.append({
                'ablation': ablation_key,
                'ablation_name': ABLATION_NAMES.get(ablation_key, ablation_key),
                'clean_score_mean': float(np.mean(ablation_scores)),
                'clean_score_std': float(np.std(ablation_scores, ddof=1)) if len(ablation_scores) > 1 else 0.0,
                'n_samples': len(ablation_scores),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Format with mean ± std
    summary_df['clean_score_mean_std'] = summary_df.apply(
        lambda row: f"{row['clean_score_mean']:.4f} ± {row['clean_score_std']:.4f}",
        axis=1
    )
    
    print(f"  [OK] Computed clean scores for {len(summary_df)} ablations")
    return summary_df


# ----------------------------
# Statistical Analysis (Subject-Level)
# ----------------------------

def prepare_subject_level_data(
    combined_df: pd.DataFrame,
    config: AnalysisConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare subject-level data for statistical comparisons.
    
    This follows the methodology from statistical_analysis.py:
    1. Aggregate across seeds to get subject-level means
    2. Compute AUPC per subject
    3. Compute RD per subject
    4. Build collapsed inference dataset
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined ablation results DataFrame
    config : AnalysisConfig
        Analysis configuration
        
    Returns:
    --------
    tuple
        (df_points, df_collapsed, df_resolved)
        - df_points: seed-aggregated curve points
        - df_collapsed: subject-level collapsed metrics (primary for inference)
        - df_resolved: subject-level resolved metrics (by noise type)
    """
    print("\n[STEP 2a] Preparing subject-level data...")
    
    # Work on a copy to avoid modifying the original
    df = combined_df.copy()
    
    # Ensure required columns exist
    required_cols = ['ablation', 'subject', 'model']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  [WARNING] Missing required columns: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Add default columns if missing (for compatibility with statistical_analysis functions)
    if 'dataset' not in df.columns:
        df['dataset'] = 'BNCI2014_001'  # Default for ablation studies
    if 'eval_mode' not in df.columns:
        df['eval_mode'] = 'CrossSubject'  # Default for ablation studies
    if 'tune' not in df.columns:
        df['tune'] = False  # Ablation studies don't use tuning
    
    # Ensure we have clean_roc_auc and corrupted_roc_auc
    if 'clean_roc_auc' not in df.columns:
        # Try to infer from clean_score or intensity=0
        if 'clean_score' in df.columns:
            df['clean_roc_auc'] = df['clean_score']
        elif 'intensity' in df.columns:
            clean_data = df[df['intensity'] == 0.0].copy()
            if not clean_data.empty:
                # Map clean scores back to all rows
                metric_col = None
                for col in ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']:
                    if col in clean_data.columns:
                        metric_col = col
                        break
                
                if metric_col:
                    # Group by subject/seed to get clean scores
                    group_cols = ['ablation', 'subject']
                    if 'seed' in df.columns:
                        group_cols.append('seed')
                    
                    for keys, group_df in clean_data.groupby(group_cols):
                        if not isinstance(keys, tuple):
                            keys = (keys,)
                        clean_val = group_df[metric_col].mean()
                        # Map to all rows with matching keys
                        mask = True
                        for i, col in enumerate(group_cols):
                            mask = mask & (df[col] == keys[i])
                        df.loc[mask, 'clean_roc_auc'] = clean_val
    
    if 'corrupted_roc_auc' not in df.columns:
        # Try to infer from available metric columns
        for col in ['corrupted_score', 'score', 'roc_auc']:
            if col in df.columns:
                df['corrupted_roc_auc'] = df[col]
                break
    
    # Ensure relative_drop exists for RD computation
    if 'relative_drop' not in df.columns:
        if 'clean_roc_auc' in df.columns and 'corrupted_roc_auc' in df.columns:
            df['relative_drop'] = (
                (df['clean_roc_auc'] - df['corrupted_roc_auc']) / 
                df['clean_roc_auc']
            )
            df['relative_drop'] = df['relative_drop'].replace([np.inf, -np.inf], np.nan)
    
    # Step 1: Aggregate across seeds
    group_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type', 'intensity', 'ablation']
    group_cols = [c for c in group_cols if c in df.columns]
    
    if 'seed' in df.columns:
        df_points = aggregate_seeds(df, group_cols)
    else:
        print("  [INFO] No seed column found, skipping seed aggregation")
        df_points = df.copy()
    
    # Step 2: Compute AUPC per subject
    df_aupc = compute_aupc_per_subject(df_points, config)
    
    # Step 3: Compute RD per subject
    df_rd = compute_rd_per_subject(df_points, config)

    df_clean = compute_clean_scores_per_subject(df_points, config)

    # Step 4: Build inference dataset (collapsed and resolved)
    # Note: We'll merge 'ablation' back after this step to avoid duplicate column issues
    df_resolved, df_collapsed = build_inference_dataset(df_aupc, df_rd, df_clean, config)
    
    # Preserve 'ablation' column: merge it back from df_points
    # The AUPC/RD functions and build_inference_dataset don't include 'ablation' in grouping,
    # so we need to merge it back using the common grouping columns
    if 'ablation' in df_points.columns:
        # For resolved dataset: merge on ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type']
        merge_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type']
        merge_cols = [c for c in merge_cols if c in df_points.columns]
        
        if not df_resolved.empty and all(c in df_resolved.columns for c in merge_cols):
            # Get unique ablation values per group from df_points
            ablation_map = df_points.groupby(merge_cols)['ablation'].first().reset_index()
            df_resolved = df_resolved.merge(ablation_map, on=merge_cols, how='left')
            ablation_count = df_resolved['ablation'].notna().sum() if 'ablation' in df_resolved.columns else 0
            print(f"  [INFO] Merged 'ablation' to resolved: {ablation_count}/{len(df_resolved)} rows")
        
        # For collapsed dataset: merge on ['dataset', 'eval_mode', 'tune', 'subject', 'model']
        collapse_merge_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model']
        collapse_merge_cols = [c for c in collapse_merge_cols if c in df_points.columns]
        
        if not df_collapsed.empty and all(c in df_collapsed.columns for c in collapse_merge_cols):
            ablation_map_collapsed = df_points.groupby(collapse_merge_cols)['ablation'].first().reset_index()
            df_collapsed = df_collapsed.merge(ablation_map_collapsed, on=collapse_merge_cols, how='left')
            ablation_count = df_collapsed['ablation'].notna().sum() if 'ablation' in df_collapsed.columns else 0
            print(f"  [INFO] Merged 'ablation' to collapsed: {ablation_count}/{len(df_collapsed)} rows")
    
    print(f"  [OK] Prepared subject-level data:")
    print(f"    - Curve points: {len(df_points)} rows")
    print(f"    - Resolved (by noise type): {len(df_resolved)} rows")
    print(f"    - Collapsed (over noise types): {len(df_collapsed)} rows")
    
    return df_points, df_collapsed, df_resolved


def compute_subject_level_statistics(
    df_collapsed: pd.DataFrame,
    primary_metrics: List[str] = ['aupc_collapsed', 'rd_collapsed'],
    secondary_metrics: List[str] = ['clean_roc_auc'],
) -> pd.DataFrame:
    """
    Compute subject-level statistical comparisons between baseline and each ablation.
    
    This uses the correct methodology: subjects as the inferential unit, with paired
    comparisons at the subject level (not n=15 summary statistics).
    
    Parameters:
    -----------
    df_collapsed : pd.DataFrame
        Subject-level collapsed metrics (from prepare_subject_level_data)
    primary_metrics : list
        Primary metrics to compare (RD/AUPC)
    secondary_metrics : list
        Secondary metrics to compare (clean ROC-AUC)
        
    Returns:
    --------
    pd.DataFrame
        Statistical test results for each ablation and metric
    """
    print("\n[STEP 2b] Computing subject-level statistical comparisons...")
    
    if df_collapsed.empty:
        print("  [WARNING] No collapsed data available")
        return pd.DataFrame()
    
    # Check if we have baseline
    if 'ablation' not in df_collapsed.columns:
        print("  [WARNING] No 'ablation' column found")
        return pd.DataFrame()
    
    baseline_mask = df_collapsed['ablation'] == 'baseline'
    if not baseline_mask.any():
        print("  [WARNING] No baseline data found")
        return pd.DataFrame()
    
    # Get all metrics to test
    all_metrics = []
    for metric in primary_metrics + secondary_metrics:
        if metric in df_collapsed.columns:
            all_metrics.append(metric)
    
    if not all_metrics:
        print("  [WARNING] No valid metrics found for comparison")
        print(f"  [INFO] Available columns: {list(df_collapsed.columns)}")
        return pd.DataFrame()
    
    print(f"  [INFO] Primary metrics: {[m for m in primary_metrics if m in all_metrics]}")
    print(f"  [INFO] Secondary metrics: {[m for m in secondary_metrics if m in all_metrics]}")
    
    test_results = []
    
    # Get unique ablations (excluding baseline)
    ablation_keys = [k for k in df_collapsed['ablation'].unique() if k != 'baseline']
    
    for ablation_key in ablation_keys:
        ablation_name = ABLATION_NAMES.get(ablation_key, ablation_key)
        
        # Filter to baseline and this ablation
        comparison_mask = (df_collapsed['ablation'] == 'baseline') | (df_collapsed['ablation'] == ablation_key)
        comparison_df = df_collapsed[comparison_mask].copy()
        
        if comparison_df.empty:
            print(f"  [WARNING] No data for {ablation_name}")
            continue
        
        # For each metric, do subject-level comparison
        for metric in all_metrics:
            if metric not in comparison_df.columns:
                continue
            
            # Pivot to wide format: subjects as rows, models (baseline/ablation) as columns
            # Group by subject to ensure we have paired data
            group_cols = ['dataset', 'eval_mode', 'tune', 'subject']
            group_cols = [c for c in group_cols if c in comparison_df.columns]
            
            # Create pivot table - use group_cols as index (already includes 'subject')
            pivot_df = comparison_df.pivot_table(
                index=group_cols,
                columns='ablation',
                values=metric,
                aggfunc='first'
            ).reset_index()
            
            # Check if we have both baseline and ablation columns
            if 'baseline' not in pivot_df.columns or ablation_key not in pivot_df.columns:
                print(f"  [WARNING] Missing paired data for {ablation_name} - {metric}")
                continue
            
            # Extract paired data
            baseline_values = pivot_df['baseline'].values
            ablation_values = pivot_df[ablation_key].values
            
            # Remove pairs with any NaN
            valid_mask = ~(np.isnan(baseline_values) | np.isnan(ablation_values))
            baseline_paired = baseline_values[valid_mask]
            ablation_paired = ablation_values[valid_mask]
            
            if len(baseline_paired) < 2:
                print(f"  [WARNING] Insufficient paired data for {ablation_name} - {metric} (n={len(baseline_paired)})")
                continue
            
            # Check normality of differences
            diffs = baseline_paired - ablation_paired
            is_normal = check_normality(diffs)
            parametric = is_normal
            
            # Run paired test
            if parametric:
                try:
                    statistic, p_value = ttest_rel(baseline_paired, ablation_paired)
                    test_type = 'paired_ttest'
                except Exception as e:
                    print(f"  [WARNING] T-test failed for {ablation_name} - {metric}: {e}")
                    statistic, p_value = np.nan, np.nan
                    test_type = 'failed'
            else:
                try:
                    statistic, p_value = wilcoxon(baseline_paired, ablation_paired, alternative='two-sided')
                    test_type = 'wilcoxon'
                except Exception as e:
                    print(f"  [WARNING] Wilcoxon test failed for {ablation_name} - {metric}: {e}")
                    statistic, p_value = np.nan, np.nan
                    test_type = 'failed'
            
            # Compute effect size
            cohens_dz = compute_cohens_dz(baseline_paired, ablation_paired)
            
            # Bootstrap CI for Cohen's dz
            try:
                data_hash = hash((tuple(baseline_paired), tuple(ablation_paired))) % (2**31)
                ci_low, ci_high = bootstrap_ci_cohens_dz(
                    baseline_paired, ablation_paired, 
                    n_reps=10000, 
                    random_seed=int(data_hash)
                )
            except Exception as e:
                print(f"  [WARNING] Bootstrap CI failed for {ablation_name} - {metric}: {e}")
                ci_low, ci_high = np.nan, np.nan
            
            # Summary statistics
            baseline_mean = float(np.mean(baseline_paired))
            baseline_std = float(np.std(baseline_paired, ddof=1))
            ablation_mean = float(np.mean(ablation_paired))
            ablation_std = float(np.std(ablation_paired, ddof=1))
            mean_diff = baseline_mean - ablation_mean
            
            # Determine if primary or secondary
            is_primary = metric in primary_metrics
            
            test_results.append({
                'ablation': ablation_key,
                'ablation_name': ablation_name,
                'metric': metric,
                'metric_type': 'primary' if is_primary else 'secondary',
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'ablation_mean': ablation_mean,
                'ablation_std': ablation_std,
                'mean_difference': mean_diff,
                'test_type': test_type,
                'parametric': parametric,
                'statistic': float(statistic) if np.isfinite(statistic) else np.nan,
                'p_value': float(p_value) if np.isfinite(p_value) else np.nan,
                'cohens_dz': float(cohens_dz) if np.isfinite(cohens_dz) else np.nan,
                'cohens_dz_ci_low': float(ci_low) if np.isfinite(ci_low) else np.nan,
                'cohens_dz_ci_high': float(ci_high) if np.isfinite(ci_high) else np.nan,
                'n_subjects': len(baseline_paired),
            })
            
            print(f"  {ablation_name} - {metric} ({'primary' if is_primary else 'secondary'}):")
            print(f"    Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
            print(f"    Ablation: {ablation_mean:.4f} ± {ablation_std:.4f}")
            print(f"    Difference: {mean_diff:.4f}")
            print(f"    n={len(baseline_paired)} subjects, {test_type}")
            if np.isfinite(p_value):
                print(f"    p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
            if np.isfinite(cohens_dz):
                print(f"    Cohen's dz: {cohens_dz:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    if not test_results:
        return pd.DataFrame()
    
    stats_df = pd.DataFrame(test_results)
    
    # Apply Bonferroni correction separately for primary and secondary metrics
    for metric_type in ['primary', 'secondary']:
        mask = stats_df['metric_type'] == metric_type
        num_tests = mask.sum()
        if num_tests > 0:
            stats_df.loc[mask, 'p_value_corrected'] = stats_df.loc[mask, 'p_value'].apply(
                lambda p: min(p * num_tests, 1.0) if np.isfinite(p) else np.nan
            )
            stats_df.loc[mask, 'significant'] = stats_df.loc[mask, 'p_value_corrected'] < 0.05
    
    print(f"  [OK] Computed statistics for {len(stats_df)} ablation/metric combinations")
    return stats_df


# ----------------------------
# Robustness Metrics
# ----------------------------

def compute_ablation_robustness_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute robustness metrics (AUPC, RD) for each ablation.
    
    Note: This is a simplified version adapted for ablation study data structure.
    Full robustness metrics may require additional data preprocessing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined ablation results DataFrame
        
    Returns:
    --------
    dict
        Dictionary of robustness metric DataFrames
    """
    print("\n[STEP 3] Computing robustness metrics...")
    
    # Check if we have the required columns for robustness metrics
    required_cols = ['ablation', 'noise_type', 'intensity']
    metric_cols = ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']
    
    has_metric = any(col in df.columns for col in metric_cols)
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols or not has_metric:
        print(f"  [INFO] Skipping robustness metrics computation:")
        if missing_cols:
            print(f"    Missing columns: {missing_cols}")
        if not has_metric:
            print(f"    No metric column found (tried: {metric_cols})")
        print(f"  [INFO] Robustness metrics require intensity and metric columns")
        return {}
    
    # Detect metric column
    metric_col = None
    for candidate in metric_cols:
        if candidate in df.columns:
            metric_col = candidate
            break
    
    # Configure metric
    cfg = MetricConfig(metric_col=metric_col, intensity_col='intensity')
    
    # Ensure we have clean_roc_auc for RD computation
    if 'clean_roc_auc' not in df.columns:
        # Try to infer from intensity=0 or clean_score
        if 'intensity' in df.columns:
            clean_data = df[df['intensity'] == 0.0].copy()
            if not clean_data.empty and metric_col in clean_data.columns:
                # Create clean_roc_auc from intensity=0 data
                for ablation_key in df['ablation'].unique():
                    ablation_df = df[df['ablation'] == ablation_key].copy()
                    clean_ablation = ablation_df[ablation_df['intensity'] == 0.0]
                    if not clean_ablation.empty:
                        # Get clean score per subject/seed
                        group_cols = []
                        if 'subject' in ablation_df.columns:
                            group_cols.append('subject')
                        if 'seed' in ablation_df.columns:
                            group_cols.append('seed')
                        
                        if group_cols:
                            clean_scores = clean_ablation.groupby(group_cols)[metric_col].first()
                            # Map back to all rows for this ablation
                            for keys, group_df in ablation_df.groupby(group_cols):
                                if not isinstance(keys, tuple):
                                    keys = (keys,)
                                if keys in clean_scores.index:
                                    mask = True
                                    for i, col in enumerate(group_cols):
                                        mask = mask & (ablation_df[col] == keys[i])
                                    df.loc[df['ablation'] == ablation_key & mask, 'clean_roc_auc'] = clean_scores[keys]
                        else:
                            # No grouping, use mean
                            clean_val = clean_ablation[metric_col].mean()
                            df.loc[df['ablation'] == ablation_key, 'clean_roc_auc'] = clean_val
    
    # Add normalized p
    try:
        df = add_normalized_p(df, cfg, normalize_within=['ablation', 'noise_type'], clip=True)
    except Exception as e:
        print(f"  [WARNING] Failed to add normalized p: {e}")
        return {}
    
    # Ensure we have required columns for statistical analysis functions
    # These functions expect: dataset, eval_mode, tune, subject, model, noise_type
    if 'dataset' not in df.columns:
        # Try to infer from data or set default
        df['dataset'] = 'BNCI2014_001'  # Default for ablation studies
    if 'eval_mode' not in df.columns:
        df['eval_mode'] = 'CrossSubject'  # Default for ablation studies
    if 'tune' not in df.columns:
        df['tune'] = False  # Ablation studies don't use tuning
    
    # Compute relative_drop for RD computation
    if 'relative_drop' not in df.columns and 'clean_roc_auc' in df.columns and metric_col in df.columns:
        df['relative_drop'] = (df['clean_roc_auc'] - df[metric_col]) / df['clean_roc_auc']
        df['relative_drop'] = df['relative_drop'].replace([np.inf, -np.inf], np.nan)
    
    # Group by ablation for separate analysis
    results = {}
    config = AnalysisConfig(normalize_aupc=True, rd_summary='mean')
    
    for ablation_key in df['ablation'].unique():
        ablation_df = df[df['ablation'] == ablation_key].copy()
        
        if ablation_df.empty:
            continue
        
        ablation_name = ABLATION_NAMES.get(ablation_key, ablation_key)
        print(f"  Computing metrics for {ablation_name}...")
        
        # Aggregate across seeds first (if needed)
        if 'seed' in ablation_df.columns:
            group_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type', 'intensity']
            group_cols = [c for c in group_cols if c in ablation_df.columns]
            try:
                ablation_df = aggregate_seeds(ablation_df, group_cols)
            except Exception as e:
                print(f"    [WARNING] Seed aggregation failed: {e}")
        
        # Compute AUPC
        try:
            aupc_df = compute_aupc_per_subject(ablation_df, config)
            if not aupc_df.empty:
                aupc_df['ablation'] = ablation_key
                aupc_df['ablation_name'] = ablation_name
                results[f'aupc_{ablation_key}'] = aupc_df
                print(f"    [OK] AUPC: {len(aupc_df)} rows")
        except Exception as e:
            print(f"    [WARNING] AUPC computation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute RD
        try:
            rd_df = compute_rd_per_subject(ablation_df, config)
            if not rd_df.empty:
                rd_df['ablation'] = ablation_key
                rd_df['ablation_name'] = ablation_name
                results[f'rd_{ablation_key}'] = rd_df
                print(f"    [OK] RD: {len(rd_df)} rows")
        except Exception as e:
            print(f"    [WARNING] RD computation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"  [OK] Computed robustness metrics for {len(results)} ablation/metric combinations")
    return results


# ----------------------------
# Main Analysis Pipeline
# ----------------------------

def analyze_ablation_studies(
    results_dir: str = None,
    output_dir: str = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main function to analyze ablation studies.
    
    Parameters:
    -----------
    results_dir : str, optional
        Path to ablations/results/ directory. If None, uses default location.
    output_dir : str, optional
        Output directory. If None, uses analysis/ablation_study_results/
        
    Returns:
    --------
    dict
        Dictionary of result DataFrames
    """
    print("=" * 80)
    print("ABLATION STUDIES ANALYSIS")
    print("=" * 80)
    
    # Set default paths
    if results_dir is None:
        results_dir = os.path.join(_project_root, "ablations", "results")
    if output_dir is None:
        output_dir = os.path.join(_project_root, "analysis", "ablation_study_results")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Results will be saved to: {output_dir}")
    
    # Load ablation results
    print("\n[STEP 0] Loading ablation results...")
    ablation_results = load_ablation_results(results_dir)
    
    # Prepare combined DataFrame
    combined_df = prepare_ablation_dataframe(ablation_results)
    
    # Print available columns for debugging
    print(f"\n[INFO] Available columns in combined DataFrame: {list(combined_df.columns)}")
    print(f"[INFO] DataFrame shape: {combined_df.shape}")
    
    # Compute clean scores (secondary outcome)
    clean_scores_df = compute_ablation_clean_scores(combined_df)
    
    # Prepare subject-level data and compute robustness metrics (primary outcomes)
    config = AnalysisConfig(normalize_aupc=True, rd_summary='mean')
    df_points, df_collapsed, df_resolved = prepare_subject_level_data(combined_df, config)
    
    # Compute subject-level statistical comparisons
    # Primary metrics: RD and AUPC (robustness endpoints)
    # Secondary metrics: clean ROC-AUC
    primary_metrics = ['aupc_collapsed', 'rd_collapsed']
    secondary_metrics = []
    
    # Add clean_roc_auc as secondary if available in collapsed data
    if 'clean_roc_auc' in df_collapsed.columns:
        # Need to aggregate clean_roc_auc to subject level
        # Clean ROC-AUC should be constant per subject/model, so we can take first value
        if 'subject' in df_collapsed.columns and 'ablation' in df_collapsed.columns:
            # Get clean ROC-AUC from the original data (it's constant per subject/model)
            clean_by_subject = combined_df.groupby(['ablation', 'subject'])['clean_roc_auc'].first().reset_index()
            clean_by_subject = clean_by_subject.rename(columns={'clean_roc_auc': 'clean_roc_auc_subject'})
            df_collapsed = df_collapsed.merge(clean_by_subject, on=['ablation', 'subject'], how='left')
            if 'clean_roc_auc_subject' in df_collapsed.columns:
                df_collapsed['clean_roc_auc'] = df_collapsed['clean_roc_auc_subject']
                secondary_metrics.append('clean_roc_auc')
    
    stats_df = compute_subject_level_statistics(
        df_collapsed, 
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics
    )
    
    # Save results
    print("\n[STEP 4] Saving results...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    # Save clean scores
    if not clean_scores_df.empty:
        clean_scores_path = os.path.join(output_dir, f"clean_scores_{timestamp}.csv")
        clean_scores_df.to_csv(clean_scores_path, index=False)
        saved_files['clean_scores'] = clean_scores_path
        print(f"  [OK] Saved clean scores: {clean_scores_path}")
    
    # Save statistical tests
    if not stats_df.empty:
        stats_path = os.path.join(output_dir, f"statistical_tests_{timestamp}.csv")
        stats_df.to_csv(stats_path, index=False)
        saved_files['statistical_tests'] = stats_path
        print(f"  [OK] Saved statistical tests: {stats_path}")
    
    # Save subject-level data
    if not df_points.empty:
        points_path = os.path.join(output_dir, f"subject_level_points_{timestamp}.csv")
        df_points.to_csv(points_path, index=False)
        saved_files['subject_level_points'] = points_path
        print(f"  [OK] Saved subject-level curve points: {points_path}")
    
    if not df_collapsed.empty:
        collapsed_path = os.path.join(output_dir, f"subject_level_collapsed_{timestamp}.csv")
        df_collapsed.to_csv(collapsed_path, index=False)
        saved_files['subject_level_collapsed'] = collapsed_path
        print(f"  [OK] Saved subject-level collapsed metrics: {collapsed_path}")
    
    if not df_resolved.empty:
        resolved_path = os.path.join(output_dir, f"subject_level_resolved_{timestamp}.csv")
        df_resolved.to_csv(resolved_path, index=False)
        saved_files['subject_level_resolved'] = resolved_path
        print(f"  [OK] Saved subject-level resolved metrics: {resolved_path}")
    
    # Save combined results
    combined_path = os.path.join(output_dir, f"combined_results_{timestamp}.csv")
    combined_df.to_csv(combined_path, index=False)
    saved_files['combined_results'] = combined_path
    print(f"  [OK] Saved combined results: {combined_path}")
    
    # Create summary report
    summary_path = os.path.join(output_dir, f"summary_report_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDIES ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        # Primary outcomes (RD/AUPC) - robustness endpoints
        if not stats_df.empty:
            primary_stats = stats_df[stats_df['metric_type'] == 'primary'].copy()
            if not primary_stats.empty:
                f.write("-" * 80 + "\n")
                f.write("PRIMARY OUTCOMES: ROBUSTNESS METRICS (RD/AUPC)\n")
                f.write("Subject-level comparisons (n=subjects, not n=15 summary statistics)\n")
                f.write("-" * 80 + "\n")
                f.write(primary_stats.to_string(index=False))
                f.write("\n\n")
        
        # Secondary outcomes (clean ROC-AUC)
        if not stats_df.empty:
            secondary_stats = stats_df[stats_df['metric_type'] == 'secondary'].copy()
            if not secondary_stats.empty:
                f.write("-" * 80 + "\n")
                f.write("SECONDARY OUTCOMES: CLEAN ROC-AUC\n")
                f.write("Subject-level comparisons\n")
                f.write("-" * 80 + "\n")
                f.write(secondary_stats.to_string(index=False))
                f.write("\n\n")
        
        # Clean scores summary (descriptive only)
        if not clean_scores_df.empty:
            f.write("-" * 80 + "\n")
            f.write("CLEAN SCORES SUMMARY (Descriptive Statistics)\n")
            f.write("-" * 80 + "\n")
            f.write(clean_scores_df.to_string(index=False))
            f.write("\n\n")
        
        # File locations
        f.write("-" * 80 + "\n")
        f.write("SAVED FILES\n")
        f.write("-" * 80 + "\n")
        for key, filepath in saved_files.items():
            f.write(f"{key:30s}: {os.path.basename(filepath)}\n")
        f.write("\n")
    
    saved_files['summary'] = summary_path
    print(f"  [OK] Saved summary report: {summary_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    
    return {
        'clean_scores': clean_scores_df,
        'statistical_tests': stats_df,
        'subject_level_points': df_points,
        'subject_level_collapsed': df_collapsed,
        'subject_level_resolved': df_resolved,
        'combined_results': combined_df,
    }


# ----------------------------
# CLI Interface
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation studies results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python analyze_ablation_studies.py
  
  # Specify custom directories
  python analyze_ablation_studies.py --results-dir ./ablations/results --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to ablations/results/ directory (default: ablations/results/)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis/ablation_study_results/)"
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_ablation_studies(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
        )
        print("\n[OK] Ablation studies analysis completed successfully!")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
