"""
Calculate and organize clean scores (baseline performance) for benchmark results.

Clean scores represent the baseline ROC-AUC performance without any noise perturbation.
Results are aggregated across seeds/subjects to get mean +/- std dev for each
combination of dataset, eval_mode, noise_type, tune setting, and model.

Usage:
    python analysis/calculate_clean_scores.py [results_dir] [output_dir]
    
    Or import and use:
    from analysis.calculate_clean_scores import generate_clean_scores_report
    results = generate_clean_scores_report(results_dir='../sol_results/', output_dir='./clean_scores_results/')
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings


def calculate_clean_scores_summary(
    df: pd.DataFrame,
    dataset: str,
    group_by_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate clean scores summary statistics for all combinations in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full results dataframe with test_perturb results
    dataset : str
        Dataset name (BNCI2014_001, Lee2019_SSVEP, BI2015a)
    group_by_cols : list, optional
        Additional columns to group by when calculating statistics (e.g., ['subject', 'seed'])
        If None, uses ['seed', 'subject', 'session'] as default
    
    Returns:
    --------
    pd.DataFrame
        Summary table with mean and std dev of clean scores for each combination
    """
    if group_by_cols is None:
        group_by_cols = ['seed', 'subject', 'session']
    
    # Filter for test_perturb mode (can be 'test_perturb' or 'test_perturb_tune')
    df_filtered = df[df['mode'].str.contains('test_perturb', na=False)].copy()
    
    if df_filtered.empty:
        print(f"Warning: No test_perturb data found for {dataset}")
        return pd.DataFrame()
    
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
        
        # Extract clean scores - use clean_score column if available
        # Clean score should be the same for all rows in a group (seed, subject, session)
        group_cols = [col for col in group_by_cols if col in combo_df.columns]
        
        # Get clean scores for each group
        clean_score_values = []
        
        if group_cols:
            for group_key, group_df in combo_df.groupby(group_cols):
                # Get clean score from this group (should be same for all rows)
                clean_scores = group_df['clean_score'].dropna().unique()
                if len(clean_scores) > 0:
                    # Use median in case there are slight variations
                    clean_score_val = np.median(clean_scores)
                    if clean_score_val > 0:  # Only include valid scores
                        clean_score_values.append(clean_score_val)
        else:
            # If no grouping columns, use all clean scores
            clean_scores = combo_df['clean_score'].dropna().unique()
            if len(clean_scores) > 0:
                clean_score_val = np.median(clean_scores)
                if clean_score_val > 0:
                    clean_score_values.append(clean_score_val)
        
        if len(clean_score_values) == 0:
            continue
        
        # Calculate mean and std
        mean_clean_score = np.mean(clean_score_values)
        std_clean_score = np.std(clean_score_values, ddof=1) if len(clean_score_values) > 1 else 0.0
        n_samples = len(clean_score_values)
        
        results.append({
            'dataset': dataset,
            'model': model,
            'noise_type': noise_type,
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
        Results dataframe from calculate_clean_scores_summary
    
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
        'clean_score_mean', 'clean_score_std', 'clean_score_mean_std', 'n_samples'
    ]
    formatted_df = formatted_df[[col for col in col_order if col in formatted_df.columns]]
    
    return formatted_df


def create_clean_scores_pivot_table(
    results_df: pd.DataFrame,
    index_cols: List[str] = ['dataset', 'model', 'noise_type'],
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


def generate_clean_scores_report(
    results_dir: str = '../sol_results/',
    output_dir: str = './analysis/clean_scores_results/',
    output_format: str = 'both'  # 'csv', 'excel', 'json', or 'all'
) -> pd.DataFrame:
    """
    Generate clean scores report for all datasets.
    
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
        
        # Calculate clean scores summary
        clean_scores_summary = calculate_clean_scores_summary(df, dataset)
        
        if not clean_scores_summary.empty:
            all_results.append(clean_scores_summary)
            print(f"Calculated clean scores for {len(clean_scores_summary)} combinations")
        else:
            print(f"No valid clean scores for {dataset}")
    
    if not all_results:
        print("No results to aggregate!")
        return pd.DataFrame()
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Format table
    formatted_results = format_clean_scores_table(combined_results)
    
    # Sort for better organization
    sort_cols = ['dataset', 'model', 'noise_type', 'eval_mode', 'tune']
    formatted_results = formatted_results.sort_values([col for col in sort_cols if col in formatted_results.columns])
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # Always save CSV as the base format
    csv_path = os.path.join(output_dir, 'clean_scores_summary.csv')
    try:
        formatted_results.to_csv(csv_path, index=False)
        saved_files.append(csv_path)
        print(f"\nSaved CSV to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
    
    # Save Excel if requested
    if output_format in ['excel', 'both', 'all']:
        try:
            excel_path = os.path.join(output_dir, 'clean_scores_summary.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                formatted_results.to_excel(writer, index=False, sheet_name='Clean Scores Summary')
                
                # Also create pivot tables for different views
                try:
                    # Pivot by eval_mode
                    pivot_eval = create_clean_scores_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'noise_type', 'tune'],
                        value_col='clean_score_mean_std',
                        columns_col='eval_mode'
                    )
                    pivot_eval.to_excel(writer, sheet_name='Pivot by Eval Mode')
                    
                    # Pivot by noise_type
                    pivot_noise = create_clean_scores_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'eval_mode', 'tune'],
                        value_col='clean_score_mean_std',
                        columns_col='noise_type'
                    )
                    pivot_noise.to_excel(writer, sheet_name='Pivot by Noise Type')
                    
                    # Pivot by tune setting
                    pivot_tune = create_clean_scores_pivot_table(
                        formatted_results,
                        index_cols=['dataset', 'model', 'noise_type', 'eval_mode'],
                        value_col='clean_score_mean_std',
                        columns_col='tune'
                    )
                    pivot_tune.to_excel(writer, sheet_name='Pivot by Tune Setting')
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
            json_path = os.path.join(output_dir, 'clean_scores_summary.json')
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
    print(f"\n=== Clean Scores Summary ===")
    print(f"Total combinations: {len(formatted_results)}")
    print(f"\nSample of results:")
    print(formatted_results.head(20).to_string())
    
    # Print pivot table view
    try:
        print(f"\n=== Pivot Table View (by Eval Mode) ===")
        pivot_eval = create_clean_scores_pivot_table(
            formatted_results,
            index_cols=['dataset', 'model', 'noise_type', 'tune'],
            value_col='clean_score_mean_std',
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
    output_dir = './clean_scores_results/'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Generate report
    results = generate_clean_scores_report(results_dir, output_dir, output_format='both')

