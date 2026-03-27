"""
Calculate Robustness Area Under the Curve (RAUC) for benchmark results.

RAUC is calculated as:
1. For each intensity i: Rscore_i = (Scoreclean - Scorecorrupted_i) / Scoreclean
2. RAUC = Area under the curve of Rscore vs intensity

Results are aggregated across seeds/subjects to get mean +/- std dev for each
combination of dataset, eval_mode, noise_type, tune setting, and model.

Usage:
    python analysis/calculate_rauc.py [results_dir] [output_dir]
    
    Or import and use:
    from analysis.calculate_rauc import generate_rauc_report
    results = generate_rauc_report(results_dir='../sol_results/', output_dir='./rauc_results/')
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings

# Import helper functions from analyze_results
from analyze_results import load_saturation_points, get_correct_intensities


def calculate_rscore(clean_score: float, corrupted_score: float) -> float:
    """
    Calculate robustness score at a given intensity.
    
    Parameters:
    -----------
    clean_score : float
        Baseline ROC-AUC score (no perturbation, intensity 0)
    corrupted_score : float
        ROC-AUC score at a given intensity
    
    Returns:
    --------
    float
        Rscore = (Scoreclean - Scorecorrupted) / Scoreclean
    """
    if clean_score <= 0:
        return np.nan
    return (clean_score - corrupted_score) / clean_score


def calculate_rauc_for_group(
    group_df: pd.DataFrame,
    intensity_col: str = 'intensity',
    clean_score_col: str = 'clean_score',
    corrupted_score_col: str = 'corrupted_score'
) -> float:
    """
    Calculate RAUC for a single group (e.g., one seed/subject combination).
    
    Parameters:
    -----------
    group_df : pd.DataFrame
        DataFrame containing data for one group, must have intensity, clean_score, and corrupted_score
    intensity_col : str
        Column name for intensity values
    clean_score_col : str
        Column name for clean (baseline) scores
    corrupted_score_col : str
        Column name for corrupted scores
    
    Returns:
    --------
    float
        RAUC value (area under Rscore vs intensity curve)
    """
    # Get clean score (should be the same for all rows in the group)
    clean_scores = group_df[clean_score_col].dropna().unique()
    if len(clean_scores) == 0:
        return np.nan
    
    # Use the median clean score (in case there are slight variations, median is more robust)
    clean_score = np.median(clean_scores)
    
    # If clean_score is 0 or negative, return NaN
    if clean_score <= 0:
        return np.nan
    
    # Filter to rows with valid corrupted scores and intensities
    valid_data = group_df[
        group_df[corrupted_score_col].notna() & 
        group_df[intensity_col].notna()
    ].copy()
    
    if len(valid_data) == 0:
        return np.nan
    
    # Calculate Rscore for each intensity
    valid_data['rscore'] = valid_data.apply(
        lambda row: calculate_rscore(clean_score, row[corrupted_score_col]),
        axis=1
    )
    
    # Remove rows with invalid rscore
    valid_data = valid_data[valid_data['rscore'].notna()]
    
    if len(valid_data) < 2:
        return np.nan
    
    # Sort by intensity
    valid_data = valid_data.sort_values(intensity_col)
    
    # Calculate area under curve using trapezoidal integration
    intensities = valid_data[intensity_col].values
    rscores = valid_data['rscore'].values
    
    # Ensure intensities start at 0 (add clean baseline if not present)
    if intensities[0] > 0:
        # Add baseline point at intensity 0 (rscore = 0)
        intensities = np.concatenate([[0.0], intensities])
        rscores = np.concatenate([[0.0], rscores])
    
    # Calculate RAUC using numpy's trapezoidal integration
    rauc = np.trapz(rscores, intensities)
    
    return rauc


def calculate_rauc_summary(
    df: pd.DataFrame,
    dataset: str,
    saturation_dict: Optional[Dict] = None,
    group_by_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate RAUC summary statistics for all combinations in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full results dataframe with test_perturb results
    dataset : str
        Dataset name (BNCI2014_001, Lee2019_MI, Lee2019_SSVEP, BI2015a)
    saturation_dict : dict, optional
        Dictionary of saturation points for filtering intensities
    group_by_cols : list, optional
        Additional columns to group by when calculating RAUC (e.g., ['subject', 'seed'])
        If None, uses ['seed', 'subject', 'session'] as default
    
    Returns:
    --------
    pd.DataFrame
        Summary table with mean and std dev of RAUC for each combination
    """
    if group_by_cols is None:
        group_by_cols = ['seed', 'subject', 'session']
    
    # Filter for test_perturb mode (can be 'test_perturb' or 'test_perturb_tune')
    df_filtered = df[df['mode'].str.contains('test_perturb', na=False)].copy()
    
    if df_filtered.empty:
        print(f"Warning: No test_perturb data found for {dataset}")
        return pd.DataFrame()
    
    # Get correct intensity ranges if saturation_dict is provided
    if saturation_dict is None:
        saturation_dict = load_saturation_points()
    
    # Get unique combinations
    unique_combos = df_filtered.groupby([
        'model', 'noise_type', 'eval_mode', 'tune'
    ]).size().reset_index()
    
    results = []
    
    for _, combo in unique_combos.iterrows():
        model = combo['model']
        noise_type = combo['noise_type']
        eval_mode = combo['eval_mode']
        tune = combo['tune']
        
        # Filter for this combination
        combo_df = df_filtered[
            (df_filtered['model'] == model) &
            (df_filtered['noise_type'] == noise_type) &
            (df_filtered['eval_mode'] == eval_mode) &
            (df_filtered['tune'] == tune)
        ].copy()
        
        if combo_df.empty:
            continue
        
        # Filter by correct intensities if saturation_dict available
        # Convert intensity to float for comparison
        if 'intensity' in combo_df.columns:
            combo_df['intensity'] = pd.to_numeric(combo_df['intensity'], errors='coerce')
        
        if saturation_dict and dataset in saturation_dict and noise_type in saturation_dict[dataset]:
            correct_intensities = get_correct_intensities(
                dataset=dataset,
                noise_type=noise_type,
                saturation_dict=saturation_dict
            )
            # Include intensity 0.0 (baseline) and any intensities in the correct range
            combo_df = combo_df[
                (combo_df['intensity'] == 0.0) |
                combo_df['intensity'].isin(correct_intensities) |
                combo_df['intensity'].isna()
            ]
        
        # Ensure we have clean scores - they should already be in each row
        # But we need to add intensity=0 baseline points if they don't exist
        # Clean score should be the same for all rows in a group (seed, subject, session)
        group_cols = [col for col in group_by_cols if col in combo_df.columns]
        
        # For each group, ensure we have a baseline point at intensity 0
        if group_cols:
            for group_key, group_df in combo_df.groupby(group_cols):
                # Check if we have intensity 0 in this group
                has_intensity_0 = (group_df['intensity'] == 0.0).any() if 'intensity' in group_df.columns else False
                
                if not has_intensity_0:
                    # Get clean score from this group (should be same for all rows)
                    clean_scores = group_df['clean_score'].dropna().unique()
                    if len(clean_scores) > 0:
                        clean_score_val = clean_scores[0]
                        # Create baseline row
                        baseline_row = group_df.iloc[0].copy()
                        baseline_row['intensity'] = 0.0
                        baseline_row['corrupted_score'] = clean_score_val
                        # Add to combo_df
                        combo_df = pd.concat([combo_df, baseline_row.to_frame().T], ignore_index=True)
        
        # Calculate RAUC for each group (seed, subject, session combination)
        rauc_values = []
        
        # Group by the specified columns
        group_cols = [col for col in group_by_cols if col in combo_df.columns]
        if not group_cols:
            # If no grouping columns, calculate single RAUC
            rauc = calculate_rauc_for_group(combo_df)
            if not np.isnan(rauc):
                rauc_values.append(rauc)
        else:
            for group_key, group_df in combo_df.groupby(group_cols):
                rauc = calculate_rauc_for_group(group_df)
                if not np.isnan(rauc):
                    rauc_values.append(rauc)
        
        if len(rauc_values) == 0:
            continue
        
        # Calculate mean and std
        mean_rauc = np.mean(rauc_values)
        std_rauc = np.std(rauc_values, ddof=1) if len(rauc_values) > 1 else 0.0
        n_samples = len(rauc_values)
        
        results.append({
            'dataset': dataset,
            'model': model,
            'noise_type': noise_type,
            'eval_mode': eval_mode,
            'tune': tune,
            'rauc_mean': mean_rauc,
            'rauc_std': std_rauc,
            'n_samples': n_samples
        })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    return results_df


def calculate_normalized_rauc(
    df: pd.DataFrame,
    dataset: str,
    saturation_dict: Optional[Dict] = None,
    group_by_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate normalized RAUC (RAUC divided by max intensity) for easier interpretation.
    
    Normalized RAUC represents the average relative performance drop across the intensity range.
    This makes it easier to compare across different intensity ranges.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full results dataframe with test_perturb results
    dataset : str
        Dataset name
    saturation_dict : dict, optional
        Dictionary of saturation points for determining max intensity
    group_by_cols : list, optional
        Additional columns to group by
    
    Returns:
    --------
    pd.DataFrame
        Summary table with normalized RAUC values
    """
    if saturation_dict is None:
        saturation_dict = load_saturation_points()
    
    # Calculate regular RAUC first
    rauc_df = calculate_rauc_summary(df, dataset, saturation_dict, group_by_cols)
    
    if rauc_df.empty:
        return pd.DataFrame()
    
    # Get max intensity for each noise type
    normalized_results = []
    
    for _, row in rauc_df.iterrows():
        noise_type = row['noise_type']
        
        # Get max intensity from saturation point
        if dataset in saturation_dict and noise_type in saturation_dict[dataset]:
            max_intensity = saturation_dict[dataset][noise_type]
        else:
            # Try to infer from data
            df_filtered = df[
                (df['dataset'] == dataset) &
                (df['noise_type'] == noise_type) &
                (df['mode'].str.contains('test_perturb', na=False))
            ]
            if 'intensity' in df_filtered.columns:
                max_intensity = df_filtered['intensity'].max()
            else:
                max_intensity = 50.0  # Default
        
        # Normalize RAUC by max intensity
        # This gives average relative drop (interpretable as percentage)
        normalized_rauc = row['rauc_mean'] / max_intensity if max_intensity > 0 else np.nan
        normalized_std = row['rauc_std'] / max_intensity if max_intensity > 0 else np.nan
        
        result_row = row.to_dict()
        result_row['normalized_rauc_mean'] = normalized_rauc
        result_row['normalized_rauc_std'] = normalized_std
        result_row['max_intensity'] = max_intensity
        normalized_results.append(result_row)
    
    return pd.DataFrame(normalized_results)


def format_rauc_table(results_df: pd.DataFrame, include_normalized: bool = True) -> pd.DataFrame:
    """
    Format RAUC results table with mean +/- std dev in readable format.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from calculate_rauc_summary
    include_normalized : bool
        If True and normalized columns exist, include normalized RAUC interpretation
    
    Returns:
    --------
    pd.DataFrame
        Formatted table with 'rauc_mean_std' column and optionally normalized metrics
    """
    formatted_df = results_df.copy()
    
    # Create formatted string: mean +/- std
    formatted_df['rauc_mean_std'] = formatted_df.apply(
        lambda row: f"{row['rauc_mean']:.4f} ± {row['rauc_std']:.4f}",
        axis=1
    )
    
    # Add normalized RAUC if available
    if include_normalized and 'normalized_rauc_mean' in formatted_df.columns:
        formatted_df['normalized_rauc_mean_std'] = formatted_df.apply(
            lambda row: f"{row['normalized_rauc_mean']:.4f} ± {row['normalized_rauc_std']:.4f}",
            axis=1
        )
        # Add interpretation: normalized RAUC as percentage
        formatted_df['avg_relative_drop_pct'] = formatted_df['normalized_rauc_mean'] * 100
    
    # Capitalize noise_type and eval_mode for better readability
    if 'noise_type' in formatted_df.columns:
        formatted_df['noise_type'] = formatted_df['noise_type'].str.capitalize()
    if 'eval_mode' in formatted_df.columns:
        formatted_df['eval_mode'] = formatted_df['eval_mode'].str.replace('Evaluation', '').str.replace('Session', ' Session')
    
    # Convert tune boolean to string
    if 'tune' in formatted_df.columns:
        formatted_df['tune'] = formatted_df['tune'].map({True: 'Tuned', False: 'Baseline'})
    
    # Reorder columns for better readability
    col_order = [
        'dataset', 'model', 'noise_type', 'eval_mode', 'tune',
        'rauc_mean', 'rauc_std', 'rauc_mean_std'
    ]
    
    if include_normalized and 'normalized_rauc_mean' in formatted_df.columns:
        col_order.extend(['normalized_rauc_mean', 'normalized_rauc_std', 'normalized_rauc_mean_std', 'avg_relative_drop_pct', 'max_intensity'])
    
    col_order.append('n_samples')
    formatted_df = formatted_df[[col for col in col_order if col in formatted_df.columns]]
    
    return formatted_df


def create_rauc_pivot_table(
    results_df: pd.DataFrame,
    index_cols: List[str] = ['dataset', 'model', 'noise_type'],
    value_col: str = 'rauc_mean_std',
    columns_col: str = 'eval_mode'
) -> pd.DataFrame:
    """
    Create a pivot table view of RAUC results for easier comparison.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Formatted results dataframe from format_rauc_table
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


def generate_rauc_report(
    results_dir: str = '../sol_results/',
    output_dir: str = './analysis/rauc_results/',
    output_format: str = 'both'  # 'csv', 'excel', 'json', or 'all'
) -> pd.DataFrame:
    """
    Generate RAUC report for all datasets.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing results (should have subdirectories for each dataset)
    output_dir : str
        Directory to save output files
    output_format : str
        Output format: 'csv', 'excel', 'json', 'both', or 'all'
        - 'csv': Save only CSV file (always saved regardless of format)
        - 'excel': Save CSV and Excel files
        - 'json': Save CSV and JSON files
        - 'both': Save CSV and Excel files (default)
        - 'all': Save CSV, Excel, and JSON files
    
    Returns:
    --------
    pd.DataFrame
        Combined results for all datasets
    """
    dataset_configs = [
        {
            'label': 'MotorImagery/BNCI2014_001',
            'input_dir': os.path.join(results_dir, 'MotorImagery/BNCI2014_001/'),
            'csv_path': os.path.join(results_dir, 'MotorImagery/BNCI2014_001/all_results.csv'),
            'dataset': 'BNCI2014_001'
        },
        {
            'label': 'MotorImagery/Lee2019_MI',
            'input_dir': os.path.join(results_dir, 'MotorImagery/Lee2019_MI/'),
            'csv_path': os.path.join(results_dir, 'MotorImagery/Lee2019_MI/all_results.csv'),
            'dataset': 'Lee2019_MI'
        },
        {
            'label': 'SSVEP/Lee2019_SSVEP',
            'input_dir': os.path.join(results_dir, 'SSVEP/Lee2019_SSVEP/'),
            'csv_path': os.path.join(results_dir, 'SSVEP/Lee2019_SSVEP/all_results.csv'),
            'dataset': 'Lee2019_SSVEP'
        },
        {
            'label': 'ERP/BI2015a',
            'input_dir': os.path.join(results_dir, 'ERP/BI2015a/'),
            'csv_path': os.path.join(results_dir, 'ERP/BI2015a/all_results.csv'),
            'dataset': 'BI2015a'
        }
    ]
    
    all_results = []
    saturation_dict = load_saturation_points()
    
    for config in dataset_configs:
        csv_path = config['csv_path']
        dataset = config['dataset']
        
        if not os.path.exists(csv_path):
            print(f"Warning: Results file not found: {csv_path}")
            continue
        
        print(f"\n=== Processing {dataset} ===")
        df = pd.read_csv(csv_path)
        
        # Ensure dataset column is set
        if 'dataset' not in df.columns:
            df['dataset'] = dataset
        
        # Calculate RAUC summary
        rauc_summary = calculate_rauc_summary(df, dataset, saturation_dict)
        
        if not rauc_summary.empty:
            # Add normalized RAUC for easier interpretation
            try:
                normalized_rauc = calculate_normalized_rauc(df, dataset, saturation_dict)
                if not normalized_rauc.empty:
                    # Merge normalized metrics
                    rauc_summary = rauc_summary.merge(
                        normalized_rauc[['dataset', 'model', 'noise_type', 'eval_mode', 'tune', 
                                         'normalized_rauc_mean', 'normalized_rauc_std', 'max_intensity']],
                        on=['dataset', 'model', 'noise_type', 'eval_mode', 'tune'],
                        how='left'
                    )
            except Exception as e:
                print(f"Warning: Could not calculate normalized RAUC: {e}")
            
            all_results.append(rauc_summary)
            print(f"Calculated RAUC for {len(rauc_summary)} combinations")
        else:
            print(f"No valid RAUC calculations for {dataset}")
    
    if not all_results:
        print("No results to aggregate!")
        return pd.DataFrame()
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Format table
    formatted_results = format_rauc_table(combined_results)
    
    # Sort for better organization
    sort_cols = ['dataset', 'model', 'noise_type', 'eval_mode', 'tune']
    formatted_results = formatted_results.sort_values([col for col in sort_cols if col in formatted_results.columns])
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # Always save CSV as the base format
    csv_path = os.path.join(output_dir, 'rauc_summary.csv')
    try:
        formatted_results.to_csv(csv_path, index=False)
        saved_files.append(csv_path)
        print(f"\nSaved CSV to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
    
    # Save Excel if requested
    if output_format in ['excel', 'both', 'all']:
        try:
            excel_path = os.path.join(output_dir, 'rauc_summary.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                formatted_results.to_excel(writer, index=False, sheet_name='RAUC Summary')
                
                # Also create pivot tables for different views
                try:
                    # Pivot by eval_mode
                    pivot_eval = create_rauc_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'noise_type', 'tune'],
                        value_col='rauc_mean_std',
                        columns_col='eval_mode'
                    )
                    pivot_eval.to_excel(writer, sheet_name='Pivot by Eval Mode')
                    
                    # Pivot by noise_type
                    pivot_noise = create_rauc_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'eval_mode', 'tune'],
                        value_col='rauc_mean_std',
                        columns_col='noise_type'
                    )
                    pivot_noise.to_excel(writer, sheet_name='Pivot by Noise Type')
                except Exception as e:
                    print(f"Warning: Could not create pivot tables: {e}")
            
            saved_files.append(excel_path)
            print(f"Saved Excel to: {excel_path}")
        except ImportError:
            print("Warning: openpyxl not installed. Skipping Excel export. Install with: pip install openpyxl")
        except Exception as e:
            print(f"Warning: Could not save Excel file: {e}")
    
    # Save JSON if requested
    if output_format in ['json', 'all']:
        try:
            json_path = os.path.join(output_dir, 'rauc_summary.json')
            # Convert DataFrame to records format for JSON
            json_data = formatted_results.to_dict(orient='records')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            saved_files.append(json_path)
            print(f"Saved JSON to: {json_path}")
        except Exception as e:
            print(f"Warning: Could not save JSON file: {e}")
    
    # Print summary of saved files
    if saved_files:
        print(f"\n=== Files Saved ({len(saved_files)}) ===")
        for file_path in saved_files:
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  - {os.path.basename(file_path)} ({file_size:.2f} KB)")
    else:
        print("\nWarning: No files were saved successfully!")
    
    # Print summary
    print(f"\n=== RAUC Summary ===")
    print(f"Total combinations: {len(formatted_results)}")
    print(f"\nSample of results:")
    print(formatted_results.head(20).to_string())
    
    # Print pivot table view
    try:
        print(f"\n=== Pivot Table View (by Eval Mode) ===")
        pivot_eval = create_rauc_pivot_table(
            formatted_results,
            index_cols=['dataset', 'model', 'noise_type', 'tune'],
            value_col='rauc_mean_std',
            columns_col='eval_mode'
        )
        print(pivot_eval.head(30).to_string())
    except Exception as e:
        print(f"Could not display pivot table: {e}")
    
    return formatted_results


if __name__ == '__main__':
    import sys
    
    # Default paths
    results_dir = '../sol_results/'
    output_dir = './rauc_results/'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Generate report
    results = generate_rauc_report(results_dir, output_dir, output_format='both')

