"""
statistical_analysis.py

Implements the statistical analysis pipeline specified in analysis_schema.md.

This script:
1. Loads and filters results data
2. Aggregates across seeds to get subject-level metrics
3. Computes clean ROC-AUC, AUPC, and RD summaries per subject
4. Performs omnibus tests (RM-ANOVA or Friedman) across models for each metric
5. Performs planned pairwise tests (conditional on omnibus significance)
6. Computes CSV under perturbation
7. Exports all results to CSV and JSON

Reuses functions from robustness_metrics.py where possible.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests

# Try to import pingouin for RM-ANOVA (preferred), fallback to statsmodels
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    try:
        from statsmodels.stats.anova import AnovaRM
        HAS_STATSMODELS_ANOVA = True
    except ImportError:
        HAS_STATSMODELS_ANOVA = False

# Add project root to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import from robustness_metrics
from analysis.robustness_metrics import (
    load_results_dataframe,
    canonicalize_columns,
    MetricConfig,
    add_normalized_p,
    compute_aupc,
    compute_rd_curve,
    compute_csv_p_curve,
    find_subject_col,
)


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""
    primary_metric: str = "roc_auc"
    alpha: float = 0.05
    collapse_noise_types: str = "mean"  # "mean" or "median"
    rd_summary: str = "mean"  # "mean" or "worst"
    normalize_aupc: bool = True
    parametric: str = "auto"  # "auto", "true", "false"
    rd_sign: str = "auto"  # "auto", "negate", "identity"
    mode_regex: str = "test_perturb"
    eval_modes: Optional[List[str]] = None
    tune_values: Optional[List[bool]] = None
    models: Optional[List[str]] = None
    noise_types: Optional[List[str]] = None
    bootstrap_reps: int = 10000
    hydra: bool = False


# ----------------------------
# Model canonicalization
# ----------------------------

MODEL_CANONICAL_MAP = {
    "cnn_ncp": "CNN-NCP",
    "eegnet": "EEGNet",
    "reegnet": "REEGNet",
}


def replace_hydra_model_name(df, model_col='model'):
    """
    Replace 'branched_wiredcfc_arch4' (and variations) with 'HYDRA' in the model column.
    Handles various naming formats (with/without hyphens, different cases).
    
    Parameters:
    - df: pd.DataFrame with a model column
    - model_col: str, name of the model column (default: 'model')
    
    Returns:
    - pd.DataFrame with model names replaced
    """
    if model_col not in df.columns:
        return df
    
    df = df.copy()
    # Normalize model names for comparison (lowercase, hyphens/spaces to underscores)
    # The canonical form after canonicalize_columns is 'branched_wiredcfc_arch4'
    df_model_normalized = df[model_col].astype(str).str.lower().str.replace('-', '_').str.replace(' ', '_')
    
    # Replace any variant of branched_wiredcfc_arch4 with HYDRA
    mask = df_model_normalized == 'branched_wiredcfc_arch4'
    df.loc[mask, model_col] = 'HYDRA'
    
    return df


def canonicalize_model(model_name: str) -> str:
    """Map model name to canonical form."""
    model_lower = str(model_name).lower().strip()
    return MODEL_CANONICAL_MAP.get(model_lower, model_name)


# ----------------------------
# Data loading and filtering
# ----------------------------

def load_and_filter_data(
    input_csv: Optional[str],
    config: AnalysisConfig,
    aggregate_from_directories: bool = True,
    hydra: bool = False,
) -> pd.DataFrame:
    """
    Load and filter data according to config.
    
    Returns:
        Filtered DataFrame ready for analysis
    """
    print("[STEP 1] Loading and filtering data...")
    
    # Load using robustness_metrics function
    df = load_results_dataframe(
        results_file=input_csv,
        aggregate_from_directories=aggregate_from_directories,
        hydra=hydra
    )
    
    # Canonicalize columns
    df = canonicalize_columns(df)
    
    # Filter by mode_regex
    if 'mode' in df.columns:
        mode_mask = df['mode'].astype(str).str.contains(config.mode_regex, na=False, regex=False)
        df = df[mode_mask].copy()
        print(f"  Filtered to mode containing '{config.mode_regex}': {len(df)} rows")
    
    # Filter by eval_mode
    if config.eval_modes is not None and 'eval_mode' in df.columns:
        df = df[df['eval_mode'].isin(config.eval_modes)].copy()
        print(f"  Filtered to eval_modes {config.eval_modes}: {len(df)} rows")
    
    # Filter by tune
    if config.tune_values is not None and 'tune' in df.columns:
        df = df[df['tune'].isin(config.tune_values)].copy()
        print(f"  Filtered to tune values {config.tune_values}: {len(df)} rows")
    
    # Filter by models
    if config.models is not None and 'model' in df.columns:
        df = df[df['model'].isin(config.models)].copy()
        print(f"  Filtered to models {config.models}: {len(df)} rows")
    
    # Filter by noise_types (exclude 'clean' by default if not specified)
    if config.noise_types is not None and 'noise_type' in df.columns:
        df = df[df['noise_type'].isin(config.noise_types)].copy()
        print(f"  Filtered to noise_types {config.noise_types}: {len(df)} rows")
    elif 'noise_type' in df.columns:
        # Exclude 'clean' noise type by default
        df = df[df['noise_type'] != 'clean'].copy()
        print(f"  Excluded 'clean' noise_type: {len(df)} rows")
    
    # Drop rows with missing critical columns
    required_cols = ['clean_roc_auc', 'corrupted_roc_auc', 'intensity']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        # Try alternative names
        if 'clean_roc_auc' not in df.columns:
            if 'clean_score' in df.columns:
                df['clean_roc_auc'] = df['clean_score']
            else:
                raise KeyError(f"Missing clean_roc_auc or clean_score column")
        if 'corrupted_roc_auc' not in df.columns:
            if 'corrupted_score' in df.columns:
                df['corrupted_roc_auc'] = df['corrupted_score']
            elif 'score' in df.columns:
                df['corrupted_roc_auc'] = df['score']
            else:
                raise KeyError(f"Missing corrupted_roc_auc, corrupted_score, or score column")
    
    # Canonicalize model names
    if 'model' in df.columns:
        df['model'] = df['model'].apply(canonicalize_model)
        # Replace branched_wiredcfc_arch4 with HYDRA early in the pipeline
        # This ensures consistent naming throughout the analysis
        if config.hydra:
            df = replace_hydra_model_name(df, model_col='model')
    
    # Ensure intensity is numeric (do this BEFORE dropping NaN values)
    df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
    
    # Drop rows with missing values
    # For rows with intensity > 0, we need corrupted_roc_auc
    # For rows with intensity = 0, we only need clean_roc_auc (corrupted_roc_auc may be NaN or equal to clean)
    initial_len = len(df)
    
    # First, drop rows with missing intensity (NaN from coercion) or clean_roc_auc (always required)
    df = df.dropna(subset=['intensity', 'clean_roc_auc']).copy()
    
    # For perturbation rows (intensity > 0), also require corrupted_roc_auc
    # Keep rows where intensity is 0 OR (intensity > 0 AND corrupted_roc_auc is not NaN)
    mask = (df['intensity'] == 0) | (~df['corrupted_roc_auc'].isna())
    df = df[mask].copy()
    
    if len(df) < initial_len:
        print(f"  Dropped {initial_len - len(df)} rows with missing values")
        # Diagnostic: show what was dropped
        dropped_count_by_intensity = initial_len - len(df)
        if dropped_count_by_intensity > 0:
            # Check how many had intensity > 0 but missing corrupted_roc_auc
            # (We can't check this directly since we already dropped them, but we can log)
            print(f"    Note: Rows with intensity > 0 require non-null corrupted_roc_auc")
    
    # Diagnostic: Report intensity distribution
    if 'intensity' in df.columns:
        intensity_0_count = len(df[df['intensity'] == 0])
        intensity_gt0_count = len(df[df['intensity'] > 0])
        intensity_nan_count = len(df[df['intensity'].isna()])
        print(f"  Intensity distribution: {intensity_0_count} rows with intensity=0, "
              f"{intensity_gt0_count} rows with intensity>0, {intensity_nan_count} rows with NaN intensity")
        if intensity_nan_count > 0:
            print(f"    [WARNING] {intensity_nan_count} rows have NaN intensity (should be dropped)")
    
    print(f"  [OK] Final data shape: {df.shape}")
    return df


# ----------------------------
# Seed aggregation
# ----------------------------

def aggregate_seeds(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Aggregate across seeds to get mean and SD at curve-point level.
    
    Groups by: (dataset, eval_mode, tune, subject, model, noise_type, intensity)
    Aggregates: clean_roc_auc, corrupted_roc_auc, relative_drop
    
    Returns:
        DataFrame with seed-aggregated values
    """
    print("[STEP 2] Aggregating across seeds...")
    
    # Ensure we have seed column
    if 'seed' not in df.columns:
        print("  [WARNING] No 'seed' column found, skipping seed aggregation")
        return df
    
    # Group columns for aggregation
    agg_cols = group_cols + ['seed']
    
    # Aggregate functions
    agg_dict = {
        'clean_roc_auc': ['mean', 'std', 'count'],
        'corrupted_roc_auc': ['mean', 'std', 'count'],
    }
    
    # Add relative_drop if it exists
    if 'relative_drop' in df.columns:
        agg_dict['relative_drop'] = ['mean', 'std']
    
    # Use pandas groupby.agg for efficiency (but handle missing columns gracefully)
    agg_dict = {}
    for metric in ['clean_roc_auc', 'corrupted_roc_auc']:
        if metric in df.columns:
            agg_dict[metric] = ['mean', 'std', 'count']
    
    if 'relative_drop' in df.columns:
        agg_dict['relative_drop'] = ['mean', 'std']
    
    if not agg_dict:
        print("  [WARNING] No metrics to aggregate")
        return df
    
    # Aggregate
    try:
        df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Flatten column names (handle MultiIndex)
        if isinstance(df_agg.columns, pd.MultiIndex):
            df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_agg.columns.values]
        
        # Rename count columns to n_seeds
        for metric in ['clean_roc_auc', 'corrupted_roc_auc']:
            count_col = f'{metric}_count'
            if count_col in df_agg.columns:
                df_agg = df_agg.rename(columns={count_col: f'{metric}_n_seeds'})
    except Exception as e:
        # Fallback to manual aggregation if groupby.agg fails
        print(f"  [WARNING] groupby.agg failed ({e}), using manual aggregation")
        result_rows = []
        for keys, group_df in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            
            row_dict = {}
            for i, col in enumerate(group_cols):
                row_dict[col] = keys[i]
            
            for metric in ['clean_roc_auc', 'corrupted_roc_auc']:
                if metric in group_df.columns:
                    values = group_df[metric].dropna()
                    if len(values) > 0:
                        row_dict[f'{metric}_mean'] = float(np.mean(values))
                        row_dict[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        row_dict[f'{metric}_n_seeds'] = int(len(values))
                    else:
                        row_dict[f'{metric}_mean'] = np.nan
                        row_dict[f'{metric}_std'] = np.nan
                        row_dict[f'{metric}_n_seeds'] = 0
            
            if 'relative_drop' in group_df.columns:
                values = group_df['relative_drop'].dropna()
                if len(values) > 0:
                    row_dict['relative_drop_mean'] = float(np.mean(values))
                    row_dict['relative_drop_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                else:
                    row_dict['relative_drop_mean'] = np.nan
                    row_dict['relative_drop_std'] = np.nan
            
            result_rows.append(row_dict)
        
        df_agg = pd.DataFrame(result_rows)
    
    # Use mean columns as primary columns for downstream analysis
    if 'clean_roc_auc_mean' in df_agg.columns:
        df_agg['clean_roc_auc'] = df_agg['clean_roc_auc_mean']
    if 'corrupted_roc_auc_mean' in df_agg.columns:
        df_agg['corrupted_roc_auc'] = df_agg['corrupted_roc_auc_mean']
    if 'relative_drop_mean' in df_agg.columns:
        df_agg['relative_drop'] = df_agg['relative_drop_mean']
    
    print(f"  [OK] Aggregated to {len(df_agg)} curve points")
    return df_agg


# ----------------------------
# Completeness checks
# ----------------------------

def check_completeness(
    df: pd.DataFrame,
    out_dir: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Check data completeness and log dropped units.
    
    Returns:
        (filtered_df, dropped_log)
    """
    print("[STEP 3] Checking data completeness...")
    
    dropped_log = []
    
    # Check for required columns
    required_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type', 'intensity']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Check 1: For each (dataset, eval_mode, tune, subject, noise_type), verify all models exist
    model_check_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'noise_type']
    expected_models = df['model'].unique()
    
    for keys, group_df in df.groupby(model_check_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        present_models = group_df['model'].unique()
        missing_models = set(expected_models) - set(present_models)
        if missing_models:
            dropped_log.append(
                f"MISSING_MODELS: {dict(zip(model_check_cols, keys))} - missing models: {missing_models}"
            )
    
    # Check 2: For each (dataset, eval_mode, tune, subject, model, noise_type),
    # verify there are >= 2 intensity points (>0) for AUPC
    # Note: 'spike' noise type is excluded from AUPC calculations
    aupc_check_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type']
    incomplete_units = []
    
    # Exclude 'spike' noise type from AUPC completeness check
    df_aupc_check = df[df['noise_type'] != 'spike'].copy() if 'noise_type' in df.columns else df.copy()
    
    for keys, group_df in df_aupc_check.groupby(aupc_check_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        # Count intensity points > 0
        intensities = group_df[group_df['intensity'] > 0]['intensity'].unique()
        intensity_0_count = len(group_df[group_df['intensity'] == 0])
        total_rows = len(group_df)
        
        if len(intensities) < 2:
            incomplete_units.append(keys)
            # Add diagnostic info
            diagnostic_info = f"only {len(intensities)} intensity points > 0"
            if intensity_0_count > 0:
                diagnostic_info += f" (has {intensity_0_count} intensity=0 rows, {total_rows} total rows)"
            else:
                diagnostic_info += f" (no intensity=0 rows, {total_rows} total rows)"
            dropped_log.append(
                f"INCOMPLETE_AUPC: {dict(zip(aupc_check_cols, keys))} - {diagnostic_info}"
            )
    
    # Filter out incomplete units (only for AUPC analysis, keep for other analyses)
    if incomplete_units:
        print(f"  Found {len(incomplete_units)} incomplete units for AUPC analysis")
        # We'll mark these but not drop globally - handle per analysis
    
    # Write dropped units log
    log_path = os.path.join(out_dir, "dropped_units.log")
    with open(log_path, 'w') as f:
        f.write("DROPPED/INCOMPLETE UNITS LOG\n")
        f.write("=" * 80 + "\n\n")
        if dropped_log:
            for entry in dropped_log:
                f.write(entry + "\n")
        else:
            f.write("No units dropped.\n")
    
    print(f"  [OK] Completeness check complete. Log written to {log_path}")
    return df, dropped_log


# ----------------------------
# Compute AUPC per subject/model/noise_type
# ----------------------------

def compute_aupc_per_subject(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """
    Compute AUPC for each (dataset, eval_mode, tune, subject, model, noise_type).
    
    Uses intensity directly (not normalized p) as specified in schema.
    Excludes 'spike' noise type from AUPC calculations.
    """
    print("[STEP 4] Computing AUPC per subject/model/noise_type...")
    
    # Exclude 'spike' noise type from AUPC calculations
    df = df.copy()
    if 'noise_type' in df.columns:
        initial_count = len(df)
        df = df[df['noise_type'] != 'spike'].copy()
        excluded_count = initial_count - len(df)
        if excluded_count > 0:
            print(f"  Excluding 'spike' noise type from AUPC: removed {excluded_count} rows")
    
    # Create MetricConfig for AUPC computation
    cfg = MetricConfig(metric_col='corrupted_roc_auc', intensity_col='intensity')
    
    # Add normalized p column (using intensity directly, normalized by max per group)
    normalize_within = ['dataset', 'noise_type']
    df = add_normalized_p(df, cfg, normalize_within=normalize_within, clip=True)
    
    # Group columns for AUPC computation
    group_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type']
    
    # For each group, build curve and compute AUPC
    result_rows = []
    
    for keys, group_df in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        row_dict = dict(zip(group_cols, keys))
        
        # Get clean_roc_auc (should be constant, use mean to be safe)
        clean_values = group_df['clean_roc_auc'].dropna().unique()
        if len(clean_values) > 0:
            clean_roc_auc = float(np.mean(clean_values))
        else:
            clean_roc_auc = np.nan
        
        # Build intensity and corrupted_roc_auc arrays
        # Include intensity=0 with clean_roc_auc
        intensities = [0.0]
        roc_aucs = [clean_roc_auc]
        
        # Add intensity > 0 points
        perturbed = group_df[group_df['intensity'] > 0].sort_values('intensity')
        if len(perturbed) > 0:
            # Get unique intensities (in case of duplicates, take mean)
            for intensity, sub_df in perturbed.groupby('intensity'):
                intensities.append(float(intensity))
                roc_aucs.append(float(sub_df['corrupted_roc_auc'].mean()))
        
        # Remove any NaN values
        valid_mask = ~np.isnan(roc_aucs)
        intensities = np.array(intensities)[valid_mask]
        roc_aucs = np.array(roc_aucs)[valid_mask]
        
        # Need at least 2 points for AUPC
        if len(intensities) >= 2:
            # Compute AUPC using trapezoidal rule
            aupc = float(np.trapezoid(y=roc_aucs, x=intensities))
            
            # Normalize if requested
            if config.normalize_aupc:
                max_intensity = float(np.max(intensities))
                if max_intensity > 0:
                    aupc = aupc / max_intensity
            
            row_dict['aupc_roc_auc'] = aupc
        else:
            row_dict['aupc_roc_auc'] = np.nan
        
        result_rows.append(row_dict)
    
    df_aupc = pd.DataFrame(result_rows)
    
    # Drop rows with NaN AUPC (incomplete curves)
    initial_len = len(df_aupc)
    df_aupc = df_aupc.dropna(subset=['aupc_roc_auc']).copy()
    if len(df_aupc) < initial_len:
        print(f"  Dropped {initial_len - len(df_aupc)} rows with incomplete AUPC curves")
    
    print(f"  [OK] Computed AUPC for {len(df_aupc)} subject/model/noise_type combinations")
    return df_aupc


# ----------------------------
# Compute clean ROC-AUC per subject/model
# ----------------------------

def compute_clean_scores_per_subject(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """
    Extract clean ROC-AUC scores for each subject/model.
    
    Clean scores are duplicated across noise types and intensities, so we extract
    one value per (dataset, eval_mode, tune, subject, model) combination.
    After seed aggregation, clean_roc_auc should be constant across noise types
    and intensities for each subject/model. We use the mean (standard practice
    for statistical analysis, consistent with other metrics) and check for
    unexpected variation as a data quality indicator.
    """
    print("[STEP 5] Computing clean ROC-AUC per subject/model...")
    
    # Group by subject-level identifiers (no noise_type needed since clean is same across noise types)
    group_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model']
    
    result_rows = []
    
    for keys, group_df in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        row_dict = dict(zip(group_cols, keys))
        
        # Extract clean_roc_auc values
        # Since clean scores are duplicated across noise types and intensities,
        # and already aggregated across seeds, we just need to get one value per subject/model
        clean_scores = group_df['clean_roc_auc'].dropna().values
        
        if len(clean_scores) > 0:
            # Use mean (standard practice for statistical analysis, consistent with other metrics)
            # Clean scores should be identical across noise types/intensities, so mean = median
            # But we check for variation as a data quality check
            clean_val = float(np.mean(clean_scores))
            clean_std = float(np.std(clean_scores)) if len(clean_scores) > 1 else 0.0
            
            # Warn if there's unexpected variation (indicates potential data issue)
            if clean_std > 1e-6:  # Threshold for floating point precision
                print(f"  [WARNING] Clean score variation detected for {dict(zip(group_cols, keys))}: "
                      f"std={clean_std:.6f} (expected ~0)")
            
            if np.isfinite(clean_val):
                row_dict['clean_roc_auc'] = clean_val
            else:
                row_dict['clean_roc_auc'] = np.nan
        else:
            row_dict['clean_roc_auc'] = np.nan
        
        result_rows.append(row_dict)
    
    df_clean = pd.DataFrame(result_rows)
    
    # Drop rows with NaN clean scores
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=['clean_roc_auc']).copy()
    if len(df_clean) < initial_len:
        print(f"  Dropped {initial_len - len(df_clean)} rows with missing clean scores")
    
    print(f"  [OK] Computed clean ROC-AUC for {len(df_clean)} subject/model combinations")
    return df_clean


# ----------------------------
# Compute RD per subject/model/noise_type
# ----------------------------

def compute_rd_per_subject(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """
    Compute RD (relative degradation) for each subject/model/noise_type.
    
    Summarizes RD across intensities into rd_mean and rd_worst.
    Excludes 'spike' noise type from RD calculations.
    """
    print("[STEP 6] Computing RD per subject/model/noise_type...")
    
    # Exclude 'spike' noise type from RD calculations
    df = df.copy()
    if 'noise_type' in df.columns:
        initial_count = len(df)
        df = df[df['noise_type'] != 'spike'].copy()
        excluded_count = initial_count - len(df)
        if excluded_count > 0:
            print(f"  Excluding 'spike' noise type from RD: removed {excluded_count} rows")
    
    group_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type']
    
    result_rows = []
    
    for keys, group_df in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        row_dict = dict(zip(group_cols, keys))
        
        # Get relative_drop values for intensity > 0
        perturbed = group_df[group_df['intensity'] > 0].copy()
        
        if len(perturbed) == 0:
            row_dict['rd_mean'] = np.nan
            row_dict['rd_worst'] = np.nan
            result_rows.append(row_dict)
            continue
        
        # Determine sign convention
        relative_drops = perturbed['relative_drop'].dropna().values
        
        if len(relative_drops) == 0:
            row_dict['rd_mean'] = np.nan
            row_dict['rd_worst'] = np.nan
            result_rows.append(row_dict)
            continue
        
        # Determine sign convention
        # Note: relative_drop = (clean - corrupted)/clean
        # Positive relative_drop means performance dropped (degradation)
        # Negative relative_drop means performance improved (rare, but possible)
        # For RD, we want positive = degradation, so we typically keep relative_drop as-is
        if config.rd_sign == "auto":
            # If most values are negative (performance improved), negate to make degradation positive
            # Otherwise, keep as-is (positive = degradation is correct)
            if np.median(relative_drops) < 0:
                rd_values = -relative_drops
            else:
                rd_values = relative_drops
        elif config.rd_sign == "negate":
            rd_values = -relative_drops
        else:  # identity
            rd_values = relative_drops
        
        # Summarize RD
        row_dict['rd_mean'] = float(np.mean(rd_values))
        if config.rd_summary == "worst":
            row_dict['rd_worst'] = float(np.max(rd_values))
        
        result_rows.append(row_dict)
    
    df_rd = pd.DataFrame(result_rows)
    
    print(f"  [OK] Computed RD for {len(df_rd)} subject/model/noise_type combinations")
    return df_rd


# ----------------------------
# Build inference dataset (collapsed + resolved)
# ----------------------------

def build_inference_dataset(
    df_aupc: pd.DataFrame,
    df_rd: pd.DataFrame,
    df_clean: pd.DataFrame,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge AUPC, RD, and clean scores, create collapsed versions over noise types.
    
    Returns:
        (df_resolved, df_collapsed)
    """
    print("[STEP 7] Building inference dataset...")
    
    # Merge AUPC and RD (both have noise_type)
    merge_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type']
    df_resolved = pd.merge(df_aupc, df_rd, on=merge_cols, how='outer')
    
    # Merge clean scores (no noise_type, so merge on subject-level cols)
    clean_merge_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model']
    df_resolved = pd.merge(df_resolved, df_clean, on=clean_merge_cols, how='left')
    
    # Create collapsed version (mean/median over noise types)
    collapse_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model']
    
    if config.collapse_noise_types == "mean":
        agg_func = 'mean'
    else:  # median
        agg_func = 'median'
    
    collapsed_rows = []
    for keys, group_df in df_resolved.groupby(collapse_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        row_dict = dict(zip(collapse_cols, keys))
        
        # Collapse AUPC
        aupc_values = group_df['aupc_roc_auc'].dropna()
        if len(aupc_values) > 0:
            row_dict['aupc_collapsed'] = float(getattr(aupc_values, agg_func)())
        else:
            row_dict['aupc_collapsed'] = np.nan
        
        # Collapse RD
        rd_values = group_df['rd_mean'].dropna()
        if len(rd_values) > 0:
            row_dict['rd_collapsed'] = float(getattr(rd_values, agg_func)())
        else:
            row_dict['rd_collapsed'] = np.nan
        
        # Clean ROC-AUC (should be constant across noise types, so just take first non-nan)
        clean_values = group_df['clean_roc_auc'].dropna()
        if len(clean_values) > 0:
            row_dict['clean_roc_auc'] = float(clean_values.iloc[0])  # Should be same across noise types
        else:
            row_dict['clean_roc_auc'] = np.nan
        
        collapsed_rows.append(row_dict)
    
    df_collapsed = pd.DataFrame(collapsed_rows)
    
    print(f"  [OK] Resolved: {len(df_resolved)} rows, Collapsed: {len(df_collapsed)} rows")
    return df_resolved, df_collapsed


# ----------------------------
# Statistical testing helpers
# ----------------------------

def check_normality(
    diffs: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """Check if differences are approximately normal using Shapiro-Wilk."""
    if len(diffs) < 3:
        return False  # Need at least 3 samples
    
    # Remove NaN and infinite values
    diffs_clean = diffs[np.isfinite(diffs)]
    if len(diffs_clean) < 3:
        return False
    
    try:
        stat, p_value = shapiro(diffs_clean)
        return p_value > alpha
    except:
        return False


def compute_cohens_dz(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute Cohen's dz for paired samples."""
    diff = x - y
    if len(diff) == 0:
        return np.nan
    
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff == 0:
        return np.nan
    
    return mean_diff / std_diff


def bootstrap_ci_cohens_dz(
    x: np.ndarray,
    y: np.ndarray,
    n_reps: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for Cohen's dz."""
    if len(x) != len(y):
        return (np.nan, np.nan)
    
    n = len(x)
    dz_bootstrap = []
    
    # Set random seed for reproducibility if provided
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    for _ in range(n_reps):
        indices = rng.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        dz = compute_cohens_dz(x_boot, y_boot)
        if np.isfinite(dz):
            dz_bootstrap.append(dz)
    
    if len(dz_bootstrap) == 0:
        return (np.nan, np.nan)
    
    alpha = 1 - ci_level
    ci_low = np.percentile(dz_bootstrap, 100 * alpha / 2)
    ci_high = np.percentile(dz_bootstrap, 100 * (1 - alpha / 2))
    
    return (ci_low, ci_high)


def prepare_wide_format(
    df: pd.DataFrame,
    metric_col: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Pivot to wide format: index=subject, columns=model, values=metric.
    
    Drops subjects missing any model (paired requirement).
    """
    # Ensure we have the metric column
    if metric_col not in df.columns:
        return pd.DataFrame()
    
    # Remove rows with missing metric values
    df_clean = df.dropna(subset=[metric_col]).copy()
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Pivot to wide format
    pivot_df = df_clean.pivot_table(
        index=group_cols + ['subject'],
        columns='model',
        values=metric_col,
        aggfunc='first'
    ).reset_index()
    
    # Drop rows with any missing model
    model_cols = [col for col in pivot_df.columns if col not in group_cols + ['subject']]
    if model_cols:
        pivot_df = pivot_df.dropna(subset=model_cols)
    
    return pivot_df


# ----------------------------
# Omnibus tests
# ----------------------------

def run_omnibus_test(
    pivot_df: pd.DataFrame,
    model_cols: List[str],
    parametric: bool,
) -> Dict:
    """
    Run omnibus test (RM-ANOVA or Friedman).
    
    Returns dict with test results.
    """
    # Validate inputs
    if len(model_cols) < 2:
        return {
            'test_type': 'insufficient_groups',
            'statistic': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
        }
    
    # Extract model columns (values only)
    data_matrix = pivot_df[model_cols].values
    
    # Remove rows with any NaN
    valid_mask = ~np.isnan(data_matrix).any(axis=1)
    data_matrix = data_matrix[valid_mask]
    
    # Need at least 3 subjects and at least 2 groups for Friedman
    # Need at least 2 subjects for RM-ANOVA (though 3+ is better)
    if len(data_matrix) < 2:
        return {
            'test_type': 'insufficient_subjects',
            'statistic': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
        }
    
    # For Friedman, need at least 3 subjects
    if not parametric and len(data_matrix) < 3:
        return {
            'test_type': 'insufficient_subjects_friedman',
            'statistic': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
        }
    
    if parametric:
        # Repeated-measures ANOVA
        if HAS_PINGOUIN:
            # Use pingouin
            try:
                # Reshape to long format for pingouin
                n_subjects = len(data_matrix)
                long_data = []
                for i in range(n_subjects):
                    for j, model in enumerate(model_cols):
                        long_data.append({
                            'subject': i,
                            'model': model,
                            'value': data_matrix[i, j]
                        })
                long_df = pd.DataFrame(long_data)
                
                aov = pg.rm_anova(data=long_df, dv='value', within='model', subject='subject')
                
                return {
                    'test_type': 'rm_anova',
                    'statistic': float(aov.loc[0, 'F']),
                    'p_value': float(aov.loc[0, 'p-unc']),
                    'effect_size': float(aov.loc[0, 'np2']) if 'np2' in aov.columns else np.nan,  # partial eta squared
                }
            except Exception as e:
                print(f"  [WARNING] Pingouin RM-ANOVA failed: {e}, falling back to Friedman")
                parametric = False
        
        if not parametric or not HAS_PINGOUIN:
            # Fallback to Friedman
            statistic, p_value = friedmanchisquare(*[data_matrix[:, i] for i in range(data_matrix.shape[1])])
            n = len(data_matrix)
            k = len(model_cols)
            kendall_w = statistic / (n * (k - 1))
            
            return {
                'test_type': 'friedman',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(kendall_w),  # Kendall's W
            }
    else:
        # Friedman test
        statistic, p_value = friedmanchisquare(*[data_matrix[:, i] for i in range(data_matrix.shape[1])])
        n = len(data_matrix)
        k = len(model_cols)
        kendall_w = statistic / (n * (k - 1))
        
        return {
            'test_type': 'friedman',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(kendall_w),
        }


# ----------------------------
# Pairwise tests
# ----------------------------

def run_pairwise_tests(
    pivot_df: pd.DataFrame,
    model1: str,
    model2: str,
    parametric: bool,
    config: AnalysisConfig,
) -> Dict:
    """
    Run pairwise test between two models.
    
    Returns dict with test results including effect size and CI.
    """
    if model1 not in pivot_df.columns or model2 not in pivot_df.columns:
        return {
            'model1': model1,
            'model2': model2,
            'test_type': 'missing_data',
            'statistic': np.nan,
            'p_value': np.nan,
            'cohens_dz': np.nan,
            'cohens_dz_ci_low': np.nan,
            'cohens_dz_ci_high': np.nan,
        }
    
    # Extract paired data
    x = pivot_df[model1].values
    y = pivot_df[model2].values
    
    # Remove pairs with any NaN
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 2:
        return {
            'model1': model1,
            'model2': model2,
            'test_type': 'insufficient_data',
            'statistic': np.nan,
            'p_value': np.nan,
            'cohens_dz': np.nan,
            'cohens_dz_ci_low': np.nan,
            'cohens_dz_ci_high': np.nan,
        }
    
    # Run test
    if parametric:
        statistic, p_value = ttest_rel(x, y)
        test_type = 'paired_ttest'
    else:
        statistic, p_value = wilcoxon(x, y, alternative='two-sided')
        test_type = 'wilcoxon'
    
    # Compute effect size
    cohens_dz = compute_cohens_dz(x, y)
    # Use a deterministic seed based on data hash for reproducibility
    # (This ensures same data gives same CI, but different data gives different CI)
    data_hash = hash((tuple(x), tuple(y))) % (2**31)
    ci_low, ci_high = bootstrap_ci_cohens_dz(x, y, n_reps=config.bootstrap_reps, random_seed=int(data_hash))
    
    return {
        'model1': model1,
        'model2': model2,
        'test_type': test_type,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'cohens_dz': float(cohens_dz) if np.isfinite(cohens_dz) else np.nan,
        'cohens_dz_ci_low': float(ci_low) if np.isfinite(ci_low) else np.nan,
        'cohens_dz_ci_high': float(ci_high) if np.isfinite(ci_high) else np.nan,
    }


# ----------------------------
# Main statistical analysis pipeline
# ----------------------------

def run_statistical_analysis(
    input_csv: Optional[str] = None,
    out_dir: str = "./analysis/statistical_results",
    config: Optional[AnalysisConfig] = None,
    aggregate_from_directories: bool = True,
    hydra: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete statistical analysis pipeline.
    
    Returns:
        Dictionary of result DataFrames
    """
    if config is None:
        config = AnalysisConfig()
    
    # Set hydra flag in config
    config.hydra = hydra
    
    # Adjust output directory for hydra mode
    if hydra:
        out_dir = os.path.join(out_dir, 'hydra')
        print(f"[INFO] Hydra mode enabled: Including 'branched_wiredcfc_arch4' with core models")
        print(f"[INFO] Results will be saved to: {out_dir}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and filter
    df = load_and_filter_data(input_csv, config, aggregate_from_directories, hydra=hydra)
    
    # Step 2: Aggregate across seeds
    group_cols = ['dataset', 'eval_mode', 'tune', 'subject', 'model', 'noise_type', 'intensity']
    df_points = aggregate_seeds(df, group_cols)
    
    # Step 3: Completeness checks
    df_points, dropped_log = check_completeness(df_points, out_dir)
    
    # Step 4: Compute AUPC
    df_aupc = compute_aupc_per_subject(df_points, config)
    
    # Step 5: Compute clean ROC-AUC
    df_clean = compute_clean_scores_per_subject(df_points, config)
    
    # Step 6: Compute RD
    df_rd = compute_rd_per_subject(df_points, config)
    
    # Step 7: Build inference dataset
    df_resolved, df_collapsed = build_inference_dataset(df_aupc, df_rd, df_clean, config)
    
    # Save subject-level tables
    resolved_path = os.path.join(out_dir, "analysis_subject_level_resolved.csv")
    collapsed_path = os.path.join(out_dir, "analysis_subject_level_collapsed.csv")
    # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
    df_resolved = replace_hydra_model_name(df_resolved, model_col='model')
    df_collapsed = replace_hydra_model_name(df_collapsed, model_col='model')
    df_resolved.to_csv(resolved_path, index=False)
    df_collapsed.to_csv(collapsed_path, index=False)
    print(f"\n[OK] Saved subject-level tables:")
    print(f"  - {resolved_path}")
    print(f"  - {collapsed_path}")
    
    # Step 8-10: Statistical tests
    print("\n[STEP 8-10] Running statistical tests...")
    
    # Get unique combinations for testing
    test_group_cols = ['dataset', 'eval_mode', 'tune']
    unique_combos = df_collapsed.groupby(test_group_cols).size().reset_index().drop(columns=0)
    
    omnibus_results = []
    pairwise_results = []
    
    # Get model names
    models = sorted(df_collapsed['model'].unique())
    if len(models) < 2:
        print("  [WARNING] Need at least 2 models for statistical tests")
        return {
            'resolved': df_resolved,
            'collapsed': df_collapsed,
            'omnibus': pd.DataFrame(),
            'pairwise': pd.DataFrame(),
        }
    
    # Primary model for comparisons (CNN-NCP)
    primary_model = "CNN-NCP"
    if primary_model not in models:
        primary_model = models[0]  # Fallback to first model
    
    for _, combo in unique_combos.iterrows():
        dataset = combo['dataset']
        eval_mode = combo['eval_mode']
        tune = combo['tune']
        
        # Filter to this combination
        combo_df = df_collapsed[
            (df_collapsed['dataset'] == dataset) &
            (df_collapsed['eval_mode'] == eval_mode) &
            (df_collapsed['tune'] == tune)
        ].copy()
        
        if combo_df.empty:
            continue
        
        # Test each metric
        for metric in ['clean_roc_auc', 'aupc_collapsed', 'rd_collapsed']:
            if metric not in combo_df.columns:
                continue
            
            # Prepare wide format
            pivot_df = prepare_wide_format(combo_df, metric, test_group_cols)
            
            if len(pivot_df) < 3:
                continue  # Need at least 3 subjects
            
            # Get all model columns
            all_model_cols = [col for col in pivot_df.columns if col not in test_group_cols + ['subject']]
            
            # Filter to core models (or hydra models if hydra mode is enabled)
            # This ensures omnibus and pairwise tests only compare the specified models
            # Note: branched_wiredcfc_arch4 has already been converted to HYDRA earlier in the pipeline
            if config.hydra:
                core_models = ['CNN-NCP', 'EEGNet', 'REEGNet', 'HYDRA']
            else:
                core_models = ['CNN-NCP', 'EEGNet', 'REEGNet']
            model_cols = [col for col in all_model_cols if col in core_models]
            
            # Remove non-core models from pivot_df if they exist
            cols_to_drop = [col for col in all_model_cols if col not in core_models]
            if cols_to_drop:
                pivot_df = pivot_df.drop(columns=cols_to_drop)
                print(f"  [INFO] Excluding models {cols_to_drop} from {dataset}/{eval_mode}/tune={tune}/{metric}")
            
            # Validate we have enough models
            if len(model_cols) < 2:
                print(f"  [WARNING] Skipping {metric} for {dataset}/{eval_mode}/tune={tune}: need at least 2 core models, found {len(model_cols)}")
                continue
            
            # Validate primary_model exists (should always be in core_models, but check anyway)
            if primary_model not in model_cols:
                # Use first available model as primary
                primary_model_actual = model_cols[0] if len(model_cols) > 0 else None
                if primary_model_actual:
                    print(f"  [WARNING] Primary model {primary_model} not found in filtered models, using {primary_model_actual}")
                else:
                    print(f"  [WARNING] No models available after filtering for {dataset}/{eval_mode}/tune={tune}/{metric}")
                    continue
            else:
                primary_model_actual = primary_model
            
            # Determine if parametric
            if config.parametric == "auto":
                # Check normality of paired differences
                if len(model_cols) >= 2 and primary_model_actual in pivot_df.columns:
                    # Compare primary to first other model
                    other_models = [m for m in model_cols if m != primary_model_actual]
                    if len(other_models) > 0:
                        diffs1 = (pivot_df[primary_model_actual] - pivot_df[other_models[0]]).dropna().values
                        is_normal1 = check_normality(diffs1) if len(diffs1) > 0 else False
                        
                        # If we have a third model, check that too
                        if len(other_models) >= 2:
                            diffs2 = (pivot_df[primary_model_actual] - pivot_df[other_models[1]]).dropna().values
                            is_normal2 = check_normality(diffs2) if len(diffs2) > 0 else True
                            parametric = is_normal1 and is_normal2
                        else:
                            parametric = is_normal1
                    else:
                        parametric = False
                else:
                    parametric = False
            elif config.parametric == "true":
                parametric = True
            else:
                parametric = False
            
            # Run omnibus test
            omnibus_result = run_omnibus_test(pivot_df, model_cols, parametric)
            omnibus_result.update({
                'dataset': dataset,
                'eval_mode': eval_mode,
                'tune': tune,
                'metric': metric,
            })
            omnibus_results.append(omnibus_result)
            
            # Run pairwise tests if omnibus significant
            if omnibus_result['p_value'] < config.alpha:
                # Compare primary model to others
                for other_model in model_cols:
                    if other_model != primary_model_actual:
                        pairwise_result = run_pairwise_tests(
                            pivot_df, primary_model_actual, other_model, parametric, config
                        )
                        pairwise_result.update({
                            'dataset': dataset,
                            'eval_mode': eval_mode,
                            'tune': tune,
                            'metric': metric,
                            'omnibus_p': omnibus_result['p_value'],
                        })
                        pairwise_results.append(pairwise_result)
    
    # Apply Holm correction to pairwise p-values within each family
    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        
        # Group by (dataset, eval_mode, tune, metric) and apply Holm
        for keys, group_df in pairwise_df.groupby(['dataset', 'eval_mode', 'tune', 'metric']):
            p_values = group_df['p_value'].values
            _, p_adjusted, _, _ = multipletests(p_values, method='holm')
            pairwise_df.loc[group_df.index, 'p_adj'] = p_adjusted
        
        pairwise_df['significant'] = pairwise_df['p_adj'] < config.alpha
    else:
        pairwise_df = pd.DataFrame()
    
    # Create omnibus DataFrame
    omnibus_df = pd.DataFrame(omnibus_results)
    if not omnibus_df.empty:
        omnibus_df['significant'] = omnibus_df['p_value'] < config.alpha
    
    # Save statistical test results
    if not omnibus_df.empty:
        # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
        omnibus_df = replace_hydra_model_name(omnibus_df, model_col='model')
        omnibus_path = os.path.join(out_dir, "stats_omnibus.csv")
        omnibus_df.to_csv(omnibus_path, index=False)
        print(f"  [OK] Saved omnibus results: {omnibus_path}")
    
    if not pairwise_df.empty:
        # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
        pairwise_df = replace_hydra_model_name(pairwise_df, model_col='model')
        pairwise_path = os.path.join(out_dir, "stats_pairwise.csv")
        pairwise_df.to_csv(pairwise_path, index=False)
        print(f"  [OK] Saved pairwise results: {pairwise_path}")
    
    # Step 10: CSV under perturbation
    print("\n[STEP 10] Computing CSV under perturbation...")
    csv_curve = pd.DataFrame()
    try:
        cfg = MetricConfig(metric_col='corrupted_roc_auc', intensity_col='intensity')
        
        # Exclude 'spike' noise type from CSV calculations
        df_points_csv = df_points.copy()
        if 'noise_type' in df_points_csv.columns:
            initial_count = len(df_points_csv)
            df_points_csv = df_points_csv[df_points_csv['noise_type'] != 'spike'].copy()
            excluded_count = initial_count - len(df_points_csv)
            if excluded_count > 0:
                print(f"  Excluding 'spike' noise type from CSV: removed {excluded_count} rows")
        
        subject_col = find_subject_col(df_points_csv, cfg)
        
        if subject_col:
            # Add normalized p
            df_points_p = add_normalized_p(df_points_csv, cfg, normalize_within=['dataset', 'noise_type'], clip=True)
            
            csv_group_cols = ['dataset', 'eval_mode', 'tune', 'model', 'noise_type']
            csv_curve = compute_csv_p_curve(df_points_p, cfg, group_cols=csv_group_cols, subject_col=subject_col)
            
            # Add intensity column from p (for backward compatibility)
            if 'p' in csv_curve.columns and 'intensity' not in csv_curve.columns:
                # Map p back to intensity
                # Use a tolerance-based merge to handle floating point precision
                merge_cols = csv_group_cols + ['p']
                
                # Create a mapping from (group_cols, p) to intensity
                # Round p to avoid floating point issues
                df_points_p_rounded = df_points_p.copy()
                df_points_p_rounded['p_rounded'] = df_points_p_rounded['p'].round(decimals=6)
                csv_curve_rounded = csv_curve.copy()
                csv_curve_rounded['p_rounded'] = csv_curve_rounded['p'].round(decimals=6)
                
                # Merge on rounded p
                merge_cols_rounded = csv_group_cols + ['p_rounded']
                intensity_map = df_points_p_rounded[merge_cols_rounded + ['intensity']].drop_duplicates(subset=merge_cols_rounded)
                csv_curve = csv_curve_rounded.merge(
                    intensity_map,
                    on=merge_cols_rounded,
                    how='left'
                )
                # Drop the rounded column
                csv_curve = csv_curve.drop(columns=['p_rounded'], errors='ignore')
            
            # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
            csv_curve = replace_hydra_model_name(csv_curve, model_col='model')
            csv_path = os.path.join(out_dir, "csv_by_level.csv")
            csv_curve.to_csv(csv_path, index=False)
            print(f"  [OK] Saved CSV results: {csv_path}")
        else:
            print("  [WARNING] No subject column found, skipping CSV computation")
    except Exception as e:
        print(f"  [WARNING] CSV computation failed: {e}")
        traceback.print_exc()
    
    # Step 11: JSON summary
    print("\n[STEP 11] Creating JSON summary...")
    summary_dict = {}
    
    for _, row in omnibus_df.iterrows():
        key = (row['dataset'], row['eval_mode'], str(row['tune']), row['metric'])
        if key not in summary_dict:
            summary_dict[key] = {}
        
        summary_dict[key]['omnibus'] = {
            'test_type': row['test_type'],
            'statistic': float(row['statistic']) if pd.notna(row['statistic']) else None,
            'p_value': float(row['p_value']) if pd.notna(row['p_value']) else None,
            'effect_size': float(row['effect_size']) if pd.notna(row['effect_size']) else None,
            'significant': bool(row['significant']) if 'significant' in row else False,
        }
    
    # Add pairwise results
    if not pairwise_df.empty:
        for _, row in pairwise_df.iterrows():
            key = (row['dataset'], row['eval_mode'], str(row['tune']), row['metric'])
            if key not in summary_dict:
                summary_dict[key] = {}
            
            if 'pairwise' not in summary_dict[key]:
                summary_dict[key]['pairwise'] = []
            
            summary_dict[key]['pairwise'].append({
                'comparison': f"{row['model1']} vs {row['model2']}",
                'test_type': row['test_type'],
                'p_value': float(row['p_value']) if pd.notna(row['p_value']) else None,
                'p_adj': float(row['p_adj']) if 'p_adj' in row and pd.notna(row['p_adj']) else None,
                'cohens_dz': float(row['cohens_dz']) if pd.notna(row['cohens_dz']) else None,
                'cohens_dz_ci': [float(row['cohens_dz_ci_low']), float(row['cohens_dz_ci_high'])] if pd.notna(row['cohens_dz_ci_low']) else None,
                'significant': bool(row['significant']) if 'significant' in row else False,
            })
    
    # Convert to nested dict format: dataset -> eval_mode -> tune -> metric
    nested_summary = {}
    for (dataset, eval_mode, tune, metric), value in summary_dict.items():
        if dataset not in nested_summary:
            nested_summary[dataset] = {}
        if eval_mode not in nested_summary[dataset]:
            nested_summary[dataset][eval_mode] = {}
        if tune not in nested_summary[dataset][eval_mode]:
            nested_summary[dataset][eval_mode][tune] = {}
        nested_summary[dataset][eval_mode][tune][metric] = value
    
    json_path = os.path.join(out_dir, "stats_summary.json")
    with open(json_path, 'w') as f:
        json.dump(nested_summary, f, indent=2, default=str)
    print(f"  [OK] Saved JSON summary: {json_path}")
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'resolved': df_resolved,
        'collapsed': df_collapsed,
        'omnibus': omnibus_df,
        'pairwise': pairwise_df,
        'csv': csv_curve,
    }


# ----------------------------
# CLI interface
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis pipeline for EEG perturbation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python statistical_analysis.py
  
  # Specify input file and output directory
  python statistical_analysis.py --input-csv results/all_results.csv --out-dir ./stats_results
  
  # Customize analysis parameters
  python statistical_analysis.py --alpha 0.01 --normalize-aupc false --parametric false
        """
    )
    
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to aggregated results CSV file (optional)"
    )
    
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./analysis/statistical_results",
        help="Output directory for results (default: ./analysis/statistical_results)"
    )
    
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="roc_auc",
        help="Primary metric name (default: roc_auc)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)"
    )
    
    parser.add_argument(
        "--collapse-noise-types",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="How to collapse across noise types (default: mean)"
    )
    
    parser.add_argument(
        "--rd-summary",
        type=str,
        choices=["mean", "worst"],
        default="mean",
        help="How to summarize RD across intensities (default: mean)"
    )
    
    parser.add_argument(
        "--normalize-aupc",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Normalize AUPC by max intensity (default: true)"
    )
    
    parser.add_argument(
        "--parametric",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Use parametric tests (default: auto)"
    )
    
    parser.add_argument(
        "--rd-sign",
        type=str,
        choices=["auto", "negate", "identity"],
        default="auto",
        help="RD sign convention (default: auto)"
    )
    
    parser.add_argument(
        "--mode-regex",
        type=str,
        default="test_perturb",
        help="Mode filter regex (default: test_perturb)"
    )
    
    parser.add_argument(
        "--eval-modes",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific eval modes (e.g., WithinSession CrossSession)"
    )
    
    parser.add_argument(
        "--tune-values",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific tune values (e.g., True False)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific models (e.g., cnn_ncp eegnet reegnet)"
    )
    
    parser.add_argument(
        "--noise-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific noise types (e.g., eog emg gaussian dropout)"
    )
    
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Don't aggregate from directories if input file not found"
    )
    
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
        help="Number of bootstrap repetitions for CI (default: 10000)"
    )
    
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Include 'branched_wiredcfc_arch4' model along with core models (eegnet, reegnet, cnn_ncp) and save to 'hydra' subdirectory"
    )
    
    args = parser.parse_args()
    
    # Parse tune values
    tune_values = None
    if args.tune_values:
        tune_values = [v.lower() == "true" for v in args.tune_values]
    
    # Create config
    config = AnalysisConfig(
        primary_metric=args.primary_metric,
        alpha=args.alpha,
        collapse_noise_types=args.collapse_noise_types,
        rd_summary=args.rd_summary,
        normalize_aupc=args.normalize_aupc,
        parametric=args.parametric,
        rd_sign=args.rd_sign,
        mode_regex=args.mode_regex,
        eval_modes=args.eval_modes,
        tune_values=tune_values,
        models=args.models,
        noise_types=args.noise_types,
        bootstrap_reps=args.bootstrap_reps,
    )
    
    try:
        results = run_statistical_analysis(
            input_csv=args.input_csv,
            out_dir=args.out_dir,
            config=config,
            aggregate_from_directories=not args.no_aggregate,
            hydra=args.hydra,
        )
        print("\n[OK] Statistical analysis completed successfully!")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Statistical analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

