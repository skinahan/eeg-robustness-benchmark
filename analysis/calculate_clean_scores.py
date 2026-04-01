"""
Calculate and organize clean scores (baseline performance) for benchmark results.

Clean scores represent the baseline ROC-AUC performance without any noise perturbation.
Results are aggregated across seeds/subjects to get mean +/- std dev for each
unique combination of dataset, model, eval_mode, and tune flag.

Note: Clean scores are the same across all noise types since a unique trained model
is evaluated once to get the clean score, then repeatedly evaluated for corrupted scores.

Usage:
    python analysis/calculate_clean_scores.py [--results-file PATH] [--output-dir PATH]
    
    Or import and use:
    from analysis.calculate_clean_scores import compute_clean_scores
    results = compute_clean_scores()
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings

# Add project root to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.experiment_utils import apply_perturb_sweep_mode_canonicalization


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy with canonical column names:
    - lowercase
    - spaces/hyphens -> underscores
    """
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    return out


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


def load_results_dataframe(
    results_file: Optional[str] = None,
    aggregate_from_directories: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Load and aggregate results using the same logic as robustness_metrics.py.
    
    This function:
    1. First tries to load from a pre-aggregated CSV file if provided
    2. Otherwise, uses collect_all_results_unified() to aggregate from directories
    3. Normalizes column names and maps to expected format
    4. Handles clean score column detection
    
    Parameters:
    -----------
    results_file : str, optional
        Path to pre-aggregated results CSV file. If provided and exists, loads from this file.
        Otherwise, aggregates from directories.
    aggregate_from_directories : bool, default=True
        If True and results_file is not provided or doesn't exist, aggregate from directories.
        
    Returns:
    --------
    pd.DataFrame
        Aggregated results with canonicalized column names.
    """
    # Try to load from pre-aggregated file first
    if results_file and os.path.exists(results_file):
        print(f"[INFO] Loading pre-aggregated results from: {results_file}")
        df = pd.read_csv(results_file)
        print(f"[OK] Loaded {len(df)} rows from {results_file}")
    else:
        # Check for unified results file
        unified_file = os.path.join(_project_root, "evaluation", "results", "unified_all_results.csv")
        if os.path.exists(unified_file):
            print(f"[INFO] Loading unified results from: {unified_file}")
            df = pd.read_csv(unified_file)
            print(f"[OK] Loaded {len(df)} rows from unified results file")
        elif aggregate_from_directories:
            # Use collect_all_results_unified to aggregate from directories
            print("[INFO] Aggregating results from directories...")
            try:
                from evaluation.experiment_utils import collect_all_results_unified
                df = collect_all_results_unified()
                if df is None:
                    raise ValueError("No results found to aggregate")
                print(f"[OK] Aggregated {len(df)} rows from directories")
            except ImportError as e:
                raise ImportError(
                    f"Could not import collect_all_results_unified from evaluation.experiment_utils: {e}\n"
                    "Make sure you're running from the project root directory."
                )
        else:
            raise FileNotFoundError(
                f"Results file not found: {results_file}\n"
                "Set aggregate_from_directories=True to aggregate from directories, "
                "or provide a valid results_file path."
            )
    
    if df is None or df.empty:
        raise ValueError("No results loaded - DataFrame is None or empty")
    
    # Canonicalize column names (handles spaces, hyphens, case)
    df = canonicalize_columns(df)
    df = apply_perturb_sweep_mode_canonicalization(
        df, log_label="calculate_clean_scores.load_results_for_clean_scores"
    )

    # Map column names to expected format
    # Handle 'tuned' -> 'tune' mapping
    if 'tuned' in df.columns and 'tune' not in df.columns:
        df['tune'] = df['tuned'].astype(bool)
    
    # Normalize eval_mode (remove 'Evaluation' suffix if present)
    if 'eval_mode' in df.columns:
        df['eval_mode'] = df['eval_mode'].astype(str).str.replace('Evaluation', '', regex=False)
    
    # Normalize mode column (extract tune flag from mode if not already present)
    if 'mode' in df.columns:
        if 'tune' not in df.columns:
            df['tune'] = df['mode'].astype(str).str.contains('_tune', na=False)
    
    # Detect clean metric column (priority: clean_roc_auc > clean_score)
    clean_metric_candidates = ['clean_roc_auc', 'clean_score']
    clean_metric_col = None
    for candidate in clean_metric_candidates:
        if candidate in df.columns:
            clean_metric_col = candidate
            break
    
    if clean_metric_col:
        # Ensure clean metric column is numeric
        df[clean_metric_col] = pd.to_numeric(df[clean_metric_col], errors='coerce')
        print(f"[INFO] Using clean metric column: {clean_metric_col}")
    else:
        raise KeyError(
            f"No clean metric column found. Tried: {clean_metric_candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Ensure required columns exist
    required_cols = ['dataset', 'model', 'eval_mode', 'tune']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns after aggregation: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Filter to test_perturb mode if available
    if 'mode' in df.columns:
        test_perturb_mask = df['mode'].astype(str).str.replace('_tune', '', regex=False) == 'test_perturb'
        if test_perturb_mask.any():
            print(f"[INFO] Filtering to test_perturb mode: {test_perturb_mask.sum()} rows")
            df = df[test_perturb_mask].copy()
        else:
            print("[WARNING] No test_perturb results found in data")
    
    # Filter to only include intended experimental seeds: [100, 200, 300, 400, 500]
    valid_seeds = [100, 200, 300, 400, 500]
    if 'seed' in df.columns:
        initial_len = len(df)
        # Convert seed to numeric, handling any string representations
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
        # Filter to valid seeds (drop rows with NaN seeds or seeds not in valid list)
        df = df[df['seed'].isin(valid_seeds)].copy()
        filtered_count = initial_len - len(df)
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} rows with seeds not in {valid_seeds}")
        print(f"[INFO] Remaining rows with valid seeds: {len(df)}")
    else:
        print("[WARNING] No 'seed' column found - cannot filter by seed values")
    
    print(f"[OK] Final DataFrame shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    return df, clean_metric_col


def compute_clean_scores_summary(
    df: pd.DataFrame,
    clean_metric_col: str = 'clean_score',
    group_by_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate clean scores summary statistics (mean and std dev) for each unique
    combination of dataset, model, eval_mode, and tune.
    
    Clean scores are aggregated across seeds/subjects/sessions to get statistics.
    Note: Clean scores should be the same across noise types, so we don't group by noise_type.
    
    CRITICAL: n_samples is computed from a reference set of unique seed/subject/session
    combinations across ALL models for each dataset/eval_mode/tune combination. This ensures
    consistent n_samples across all models for the same experimental condition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe with test_perturb results (must include clean_score or clean_roc_auc)
    clean_metric_col : str
        Column name containing clean scores (default: 'clean_score')
    group_by_cols : list, optional
        Columns to group by when extracting unique clean scores (e.g., ['seed', 'subject', 'session'])
        If None, uses ['seed', 'subject', 'session'] as default
        These columns identify unique model evaluations
    
    Returns:
    --------
    pd.DataFrame
        Summary table with mean and std dev of clean scores for each combination
        Columns: dataset, model, eval_mode, tune, clean_score_mean, clean_score_std, n_samples
    """
    if group_by_cols is None:
        group_by_cols = ['seed', 'subject', 'session']
    
    # Filter to only rows with valid clean scores
    df_filtered = df.dropna(subset=[clean_metric_col]).copy()
    
    if df_filtered.empty:
        print("[WARNING] No rows with valid clean scores found")
        return pd.DataFrame()
    
    # CRITICAL FIX: First, determine reference sets of unique seed/subject/session combinations
    # for each dataset/eval_mode/tune combination (across ALL models)
    # This ensures n_samples is consistent across models
    group_cols = [col for col in group_by_cols if col in df_filtered.columns]
    
    reference_sets = {}
    for (dataset, eval_mode, tune), group_df in df_filtered.groupby(['dataset', 'eval_mode', 'tune']):
        # Get all unique seed/subject/session combinations across all models for this condition
        if group_cols:
            unique_combos = set(group_df.groupby(group_cols).groups.keys())
        else:
            # Fallback: use all rows (shouldn't happen with proper data)
            unique_combos = set(range(len(group_df)))
        reference_sets[(dataset, eval_mode, tune)] = unique_combos
    
    # Get unique combinations of dataset, model, eval_mode, tune
    unique_combos = df_filtered.groupby([
        'dataset', 'model', 'eval_mode', 'tune'
    ]).size().reset_index().drop(columns=0)
    
    results = []
    
    for _, combo in unique_combos.iterrows():
        dataset = combo['dataset']
        model = combo['model']
        eval_mode = combo['eval_mode']
        tune = combo['tune']
        
        # Filter for this combination
        combo_df = df_filtered[
            (df_filtered['dataset'] == dataset) &
            (df_filtered['model'] == model) &
            (df_filtered['eval_mode'] == eval_mode) &
            (df_filtered['tune'] == tune)
        ].copy()
        
        if combo_df.empty:
            continue
        
        # Extract clean scores - group by seed/subject/session to get unique evaluations
        clean_score_values = []
        
        if group_cols:
            # Group by seed/subject/session to get one clean score per unique evaluation
            for group_key, group_df in combo_df.groupby(group_cols):
                # Get clean score from this group (should be same for all rows in group)
                clean_scores = group_df[clean_metric_col].dropna().unique()
                if len(clean_scores) > 0:
                    # Use median in case there are slight variations
                    clean_score_val = np.median(clean_scores)
                    if np.isfinite(clean_score_val) and clean_score_val > 0:
                        clean_score_values.append(clean_score_val)
        else:
            # If no grouping columns, use all clean scores (deduplicate)
            clean_scores = combo_df[clean_metric_col].dropna().unique()
            for score in clean_scores:
                if np.isfinite(score) and score > 0:
                    clean_score_values.append(score)
        
        if len(clean_score_values) == 0:
            continue
        
        # Calculate mean and std
        mean_clean_score = np.mean(clean_score_values)
        std_clean_score = np.std(clean_score_values, ddof=1) if len(clean_score_values) > 1 else 0.0
        
        # CRITICAL FIX: Use reference set size for n_samples to ensure consistency across models
        ref_key = (dataset, eval_mode, tune)
        if ref_key in reference_sets:
            n_samples = len(reference_sets[ref_key])
        else:
            # Fallback to model-specific count if reference set not found
            n_samples = len(clean_score_values)
            print(f"[WARNING] Reference set not found for {ref_key}, using model-specific count: {n_samples}")
        
        results.append({
            'dataset': dataset,
            'model': model,
            'eval_mode': eval_mode,
            'tune': tune,
            'clean_score_mean': mean_clean_score,
            'clean_score_std': std_clean_score,
            'n_samples': n_samples
        })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    return results_df


def format_clean_scores_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format clean scores results table with mean +/- std dev in readable format.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from compute_clean_scores_summary
    
    Returns:
    --------
    pd.DataFrame
        Formatted table with 'clean_score_mean_std' column
    """
    formatted_df = results_df.copy()
    
    # Create formatted string: mean +/- std
    formatted_df['clean_score_mean_std'] = formatted_df.apply(
        lambda row: f"{row['clean_score_mean']:.4f} ± {row['clean_score_std']:.4f}",
        axis=1
    )
    
    # Format eval_mode for better readability
    if 'eval_mode' in formatted_df.columns:
        formatted_df['eval_mode'] = formatted_df['eval_mode'].astype(str).str.replace('Evaluation', '', regex=False)
    
    # Convert tune boolean to string
    if 'tune' in formatted_df.columns:
        formatted_df['tune'] = formatted_df['tune'].map({True: 'Tuned', False: 'Baseline'})
    
    # Reorder columns for better readability
    col_order = [
        'dataset', 'model', 'eval_mode', 'tune',
        'clean_score_mean', 'clean_score_std', 'clean_score_mean_std', 'n_samples'
    ]
    formatted_df = formatted_df[[col for col in col_order if col in formatted_df.columns]]
    
    return formatted_df


def create_clean_scores_pivot_table(
    results_df: pd.DataFrame,
    index_cols: List[str] = ['dataset', 'model'],
    value_col: str = 'clean_score_mean_std',
    columns_col: str = 'eval_mode'
) -> pd.DataFrame:
    """
    Create a pivot table view of clean scores results for easier comparison.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Formatted results dataframe from format_clean_scores_table
    index_cols : list
        Columns to use as index (rows)
    value_col : str
        Column to use as values
    columns_col : str
        Column to use as columns
    
    Returns:
    --------
    pd.DataFrame
        Pivot table
    """
    pivot_df = results_df.pivot_table(
        index=index_cols,
        columns=columns_col,
        values=value_col,
        aggfunc='first'
    )
    return pivot_df


def compute_clean_scores(
    results_file: Optional[str] = None,
    output_dir: str = "./analysis/clean_scores_results",
    aggregate_from_directories: bool = True,
    save_csv: bool = True,
    save_excel: bool = True,
    save_json: bool = False,
    hydra: bool = False,
) -> pd.DataFrame:
    """
    Compute and save clean scores summary for all datasets.
    
    Clean scores represent the baseline ROC-AUC performance without any noise perturbation.
    Results are aggregated across seeds/subjects to get mean +/- std dev for each
    unique combination of dataset, model, eval_mode, and tune flag.
    
    Parameters:
    -----------
    results_file : str, optional
        Path to pre-aggregated results CSV file. If None, will aggregate from directories.
    output_dir : str
        Directory to save output files
    aggregate_from_directories : bool
        If True and results_file is not provided, aggregate from directories
    save_csv : bool
        Whether to save CSV file
    save_excel : bool
        Whether to save Excel file
    save_json : bool
        Whether to save JSON file
    
    Returns:
    --------
    pd.DataFrame
        Formatted clean scores summary
    """
    print("=" * 80)
    print("CLEAN SCORES COMPUTATION")
    print("=" * 80)
    
    # Adjust output directory for hydra mode
    if hydra:
        output_dir = os.path.join(output_dir, 'hydra')
        print(f"[INFO] Hydra mode enabled: Including 'branched_wiredcfc_arch4' with core models")
        print(f"[INFO] Results will be saved to: {output_dir}")
    
    # Load results using the same approach as robustness_metrics.py
    print("\n[STEP 1] Loading/aggregating results...")
    df, clean_metric_col = load_results_dataframe(
        results_file=results_file,
        aggregate_from_directories=aggregate_from_directories
    )
    
    # Filter to hydra models if hydra mode is enabled
    if hydra:
        # Core models and hydra model (handle case variations)
        # Common formats: 'EEGNet', 'eegnet', 'REEGNet', 'reegnet', 'CNN-NCP', 'cnn_ncp', etc.
        hydra_model_patterns = [
            'eegnet', 'reegnet', 'cnn_ncp', 'cnn-ncp',
            'branched_wiredcfc_arch4', 'branched-wiredcfc-arch4'
        ]
        
        if 'model' in df.columns:
            initial_count = len(df)
            # Normalize model names for comparison (case-insensitive, handle underscores/hyphens)
            df['model_normalized'] = df['model'].astype(str).str.lower().str.strip().str.replace('-', '_')
            
            # CRITICAL: Explicitly exclude hydra_v2 and other hydra variants
            # Exclude any model containing 'hydra_v' (which would catch hydra_v2, hydra_v3, etc.)
            exclude_mask = df['model_normalized'].str.contains('hydra_v', na=False, regex=False)
            excluded_count = exclude_mask.sum()
            if excluded_count > 0:
                excluded_models = df.loc[exclude_mask, 'model'].unique()
                print(f"[INFO] Excluding {excluded_count} rows with hydra variants (e.g., {list(excluded_models[:3])}): not part of core experiment")
            df = df[~exclude_mask].copy()
            
            # Filter to only allowed models
            hydra_patterns_normalized = [p.lower().strip().replace('-', '_') for p in hydra_model_patterns]
            df = df[df['model_normalized'].isin(hydra_patterns_normalized)].copy()
            df = df.drop(columns=['model_normalized'])
            filtered_count = len(df)
            excluded = initial_count - filtered_count
            if excluded > 0:
                print(f"[INFO] Filtered to hydra models (eegnet, reegnet, cnn_ncp, branched_wiredcfc_arch4): removed {excluded} rows, kept {filtered_count} rows")
            # Replace branched_wiredcfc_arch4 with HYDRA early in the pipeline
            # This ensures consistent naming throughout the analysis
            df = replace_hydra_model_name(df, model_col='model')
    
    # Compute clean scores summary
    print("\n[STEP 2] Computing clean scores summary...")
    clean_scores_summary = compute_clean_scores_summary(df, clean_metric_col=clean_metric_col)
    
    if clean_scores_summary.empty:
        print("[WARNING] No clean scores found!")
        return pd.DataFrame()
    
    print(f"[OK] Computed clean scores for {len(clean_scores_summary)} combinations")
    
    # Format table
    print("\n[STEP 3] Formatting results...")
    formatted_results = format_clean_scores_table(clean_scores_summary)
    
    # Sort for better organization
    sort_cols = ['dataset', 'model', 'eval_mode', 'tune']
    formatted_results = formatted_results.sort_values([col for col in sort_cols if col in formatted_results.columns])
    
    # Save results
    print("\n[STEP 4] Saving results to files...")
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    if save_csv:
        csv_path = os.path.join(output_dir, f'clean_scores_summary_{timestamp}.csv')
        try:
            # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
            formatted_results = replace_hydra_model_name(formatted_results, model_col='model')
            formatted_results.to_csv(csv_path, index=False)
            saved_files['csv'] = csv_path
            print(f"  [OK] Saved CSV to: {csv_path}")
        except Exception as e:
            print(f"  [ERROR] Could not save CSV file: {e}")
    
    # Save Excel
    if save_excel:
        excel_path = os.path.join(output_dir, f'clean_scores_summary_{timestamp}.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                formatted_results.to_excel(writer, index=False, sheet_name='Clean Scores Summary')
                
                # Create pivot tables for different views
                try:
                    # Pivot by eval_mode
                    pivot_eval = create_clean_scores_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'tune'],
                        value_col='clean_score_mean_std',
                        columns_col='eval_mode'
                    )
                    pivot_eval.to_excel(writer, sheet_name='Pivot by Eval Mode')
                    
                    # Pivot by tune setting
                    pivot_tune = create_clean_scores_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'eval_mode'],
                        value_col='clean_score_mean_std',
                        columns_col='tune'
                    )
                    pivot_tune.to_excel(writer, sheet_name='Pivot by Tune Setting')
                except Exception as e:
                    print(f"  [WARNING] Could not create pivot tables: {e}")
            
            saved_files['excel'] = excel_path
            print(f"  [OK] Saved Excel to: {excel_path}")
        except ImportError:
            print("  [WARNING] openpyxl not installed. Skipping Excel export. Install with: pip install openpyxl")
        except Exception as e:
            print(f"  [WARNING] Could not save Excel file: {e}")
    
    # Save JSON
    if save_json:
        json_path = os.path.join(output_dir, f'clean_scores_summary_{timestamp}.json')
        try:
            # Convert DataFrame to records format for JSON
            json_data = formatted_results.to_dict(orient='records')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            saved_files['json'] = json_path
            print(f"  [OK] Saved JSON to: {json_path}")
        except Exception as e:
            print(f"  [WARNING] Could not save JSON file: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal combinations: {len(formatted_results)}")
    print(f"\nSample of results:")
    print(formatted_results.head(20).to_string())
    
    # Print pivot table view
    try:
        print(f"\n=== Pivot Table View (by Eval Mode) ===")
        pivot_eval = create_clean_scores_pivot_table(
            formatted_results,
            index_cols=['dataset', 'model', 'tune'],
            value_col='clean_score_mean_std',
            columns_col='eval_mode'
        )
        print(pivot_eval.head(30).to_string())
    except Exception as e:
        print(f"Could not display pivot table: {e}")
    
    if saved_files:
        print(f"\nSaved files:")
        for key, filepath in saved_files.items():
            file_size = os.path.getsize(filepath) / 1024  # Size in KB
            print(f"  - {key:10s}: {os.path.basename(filepath)} ({file_size:.2f} KB)")
    
    return formatted_results


# Compatibility alias for old function name
def generate_clean_scores_report(
    results_dir: str = '../sol_results/',
    output_dir: str = './analysis/clean_scores_results/',
    output_format: str = 'both'
) -> pd.DataFrame:
    """
    Legacy compatibility function. Use compute_clean_scores() instead.
    
    This function is kept for backward compatibility but is deprecated.
    """
    import warnings
    warnings.warn(
        "generate_clean_scores_report() is deprecated. Use compute_clean_scores() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Try to find results file in the old location
    results_file = None
    if results_dir and os.path.exists(results_dir):
        # Try to find all_results.csv in subdirectories
        for root, dirs, files in os.walk(results_dir):
            if 'all_results.csv' in files:
                results_file = os.path.join(root, 'all_results.csv')
                break
    
    save_excel = output_format in ['excel', 'both', 'all']
    save_json = output_format in ['json', 'all']
    
    return compute_clean_scores(
        results_file=results_file,
        output_dir=output_dir,
        aggregate_from_directories=True,
        save_csv=True,
        save_excel=save_excel,
        save_json=save_json,
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute and save clean scores (baseline ROC-AUC) for EEG benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (aggregate from directories)
  python calculate_clean_scores.py
  
  # Load from pre-aggregated file
  python calculate_clean_scores.py --results-file results/all_results.csv
  
  # Specify custom output directory
  python calculate_clean_scores.py --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to pre-aggregated results CSV file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis/clean_scores_results",
        help="Directory to save output files (default: ./analysis/clean_scores_results)"
    )
    
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Don't aggregate from directories if results file not found"
    )
    
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Don't save Excel file"
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save JSON file"
    )
    
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Include 'branched_wiredcfc_arch4' model along with core models (eegnet, reegnet, cnn_ncp) and save to 'hydra' subdirectory"
    )
    
    args = parser.parse_args()
    
    try:
        results = compute_clean_scores(
            results_file=args.results_file,
            output_dir=args.output_dir,
            aggregate_from_directories=not args.no_aggregate,
            save_csv=True,
            save_excel=not args.no_excel,
            save_json=args.save_json,
            hydra=args.hydra,
        )
        print("\n[OK] Clean scores computation completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Failed to compute clean scores: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

