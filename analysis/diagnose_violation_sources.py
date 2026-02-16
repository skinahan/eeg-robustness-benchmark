"""
Diagnostic script to investigate the root cause of sanity check violations.

For each violation, this script examines the original data rows to identify
which columns vary and might explain why clean scores differ.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def diagnose_violation_sources(violation_csv_path, results_csv_path=None, dataset_name=None):
    """
    Examine the original data rows for each violation to identify varying columns.
    
    Parameters:
    -----------
    violation_csv_path : str
        Path to the violation report CSV
    results_csv_path : str, optional
        Path to the original results CSV. If None, will try to find it from dataset name.
    dataset_name : str, optional
        Dataset name to filter results (if None, will try to infer from violation CSV)
    """
    print("=" * 80)
    print("DIAGNOSING VIOLATION SOURCES")
    print("=" * 80)
    
    # Load violation report
    print(f"\n[1] Loading violation report: {violation_csv_path}")
    violations_df = pd.read_csv(violation_csv_path)
    print(f"    Found {len(violations_df)} violations")
    
    # If dataset_name not provided, try to infer from violation CSV
    if dataset_name is None and 'dataset' in violations_df.columns:
        dataset_name = violations_df['dataset'].iloc[0]
        print(f"    Inferred dataset name: {dataset_name}")
    
    # Use unified results CSV if not provided
    if results_csv_path is None or not os.path.exists(results_csv_path):
        # Default to unified_all_results.csv
        unified_paths = [
            'evaluation/results/unified_all_results.csv',
            '../evaluation/results/unified_all_results.csv',
            './evaluation/results/unified_all_results.csv'
        ]
        
        for path in unified_paths:
            if os.path.exists(path):
                results_csv_path = path
                print(f"    Found unified results CSV: {results_csv_path}")
                break
        
        if results_csv_path is None or not os.path.exists(results_csv_path):
            print(f"[ERROR] Could not find unified results CSV")
            print(f"    Searched in: {unified_paths}")
            print(f"    Please provide the path to evaluation/results/unified_all_results.csv")
            return
    
    # Load original results
    print(f"\n[2] Loading original results: {results_csv_path}")
    results_df = pd.read_csv(results_csv_path, low_memory=False)
    print(f"    Loaded {len(results_df)} rows")
    print(f"    Columns ({len(results_df.columns)}): {list(results_df.columns)[:20]}...")
    
    # Filter to test_perturb mode and valid clean scores (same as sanity check)
    if 'mode' in results_df.columns:
        results_df = results_df[results_df['mode'] == 'test_perturb'].copy()
        print(f"    After filtering to test_perturb: {len(results_df)} rows")
    
    if 'clean_score' not in results_df.columns:
        print("[ERROR] 'clean_score' column not found in results")
        return
    
    results_df = results_df[results_df['clean_score'].notna()].copy()
    print(f"    After filtering to valid clean scores: {len(results_df)} rows")
    
    # Filter to valid seeds (same as sanity check)
    valid_seeds = [100, 200, 300, 400, 500]
    if 'seed' in results_df.columns:
        initial_len = len(results_df)
        # Handle both int and numpy int64 types
        results_df = results_df[results_df['seed'].isin(valid_seeds)].copy()
        print(f"    After filtering to valid seeds {valid_seeds}: {len(results_df)} rows (removed {initial_len - len(results_df)})")
        
        # Show what seeds are actually in the data
        actual_seeds = sorted(results_df['seed'].dropna().unique())
        print(f"    Actual seeds in filtered data: {actual_seeds}")
    
    if dataset_name and 'dataset' in results_df.columns:
        initial_len = len(results_df)
        results_df = results_df[results_df['dataset'] == dataset_name].copy()
        print(f"    After filtering to dataset '{dataset_name}': {len(results_df)} rows (removed {initial_len - len(results_df)})")
        
        # Show what values are in key columns
        if 'tune' in results_df.columns:
            tune_vals = sorted(results_df['tune'].dropna().unique())
            print(f"    Tune values in filtered data: {tune_vals}")
            print(f"    Tune value counts: {results_df['tune'].value_counts().to_dict()}")
        
        if 'eval_mode' in results_df.columns:
            eval_modes = sorted(results_df['eval_mode'].dropna().unique())
            print(f"    Eval modes in filtered data: {eval_modes}")
        
        if 'session' in results_df.columns:
            session_sample = results_df['session'].dropna().unique()[:3]
            print(f"    Session sample (showing format): {list(session_sample)}")
            print(f"    Session dtype: {results_df['session'].dtype}")
    
    # Grouping columns used in sanity check
    grouping_cols = ['model', 'dataset', 'seed', 'subject', 'tune', 'session', 'eval_mode']
    grouping_cols = [col for col in grouping_cols if col in violations_df.columns and col in results_df.columns]
    
    print(f"\n[3] Analyzing violations...")
    print(f"    Grouping columns: {grouping_cols}")
    print(f"    Available in violations: {[col for col in grouping_cols if col in violations_df.columns]}")
    print(f"    Available in results: {[col for col in grouping_cols if col in results_df.columns]}")
    
    # Get all columns in results that are not grouping columns
    all_cols = list(results_df.columns)
    non_grouping_cols = [col for col in all_cols if col not in grouping_cols + ['clean_score', 'noise_type', 'intensity']]
    
    print(f"    Non-grouping columns to examine: {len(non_grouping_cols)}")
    if len(non_grouping_cols) > 0:
        print(f"      Sample: {non_grouping_cols[:10]}")
    
    # Analyze each violation
    violation_analyses = []
    no_match_count = 0
    
    # Debug: Show what values are actually in the data for first violation
    if len(violations_df) > 0:
        print(f"\n[DEBUG] First violation sample:")
        first_violation = violations_df.iloc[0]
        for col in grouping_cols:
            if col in first_violation.index:
                print(f"  {col}: {first_violation[col]} (type: {type(first_violation[col]).__name__})")
        
        print(f"\n[DEBUG] Sample values in results data:")
        for col in grouping_cols:
            if col in results_df.columns:
                unique_vals = results_df[col].dropna().unique()[:5]
                print(f"  {col}: {list(unique_vals)} (dtype: {results_df[col].dtype})")
    
    for idx, violation in violations_df.iterrows():
        # Build filter mask for this violation - handle type mismatches
        # Initialize mask with same index as results_df to avoid reindexing warnings
        mask = pd.Series(True, index=results_df.index)
        mismatch_info = []
        match_details = {}
        
        for col in grouping_cols:
            if col in results_df.columns and col in violation.index:
                # Handle type mismatches (e.g., string vs int, numpy bool vs Python bool)
                violation_val = violation[col]
                
                if pd.isna(violation_val):
                    col_mask = results_df[col].isna()
                else:
                    # Try exact match first
                    col_mask = (results_df[col] == violation_val)
                    
                    # If no match, try type conversions
                    if not col_mask.any():
                        try:
                            # Handle numpy bool vs Python bool
                            if isinstance(violation_val, (bool, np.bool_)):
                                bool_val = bool(violation_val)
                                if results_df[col].dtype == 'bool':
                                    col_mask = (results_df[col] == bool_val)
                                elif results_df[col].dtype == 'object':
                                    # Try both 'True'/'False' and '1'/'0'
                                    col_mask = (
                                        (results_df[col].astype(str).str.lower() == str(bool_val).lower()) |
                                        (results_df[col].astype(str) == str(int(bool_val)))
                                    )
                                else:
                                    # Try converting to int (True=1, False=0)
                                    col_mask = (results_df[col] == int(bool_val))
                            # Handle numpy int64 vs Python int
                            elif isinstance(violation_val, (int, np.integer)):
                                int_val = int(violation_val)
                                if results_df[col].dtype in ['int64', 'int32', 'int', 'float64', 'float32']:
                                    col_mask = (results_df[col] == int_val)
                                elif results_df[col].dtype == 'object':
                                    col_mask = (results_df[col].astype(str) == str(int_val))
                            # Handle string comparisons
                            elif isinstance(violation_val, str):
                                if results_df[col].dtype == 'object':
                                    col_mask = (results_df[col].astype(str) == violation_val)
                        except Exception as e:
                            pass
                    
                    if not col_mask.any():
                        # Get sample values from results to help debug
                        sample_vals = results_df[col].dropna().unique()[:5]
                        mismatch_info.append({
                            'col': col,
                            'violation_val': violation_val,
                            'violation_type': type(violation_val).__name__,
                            'results_dtype': str(results_df[col].dtype),
                            'sample_results_vals': list(sample_vals)
                        })
                    else:
                        match_details[col] = f"matched ({col_mask.sum()} rows)"
                
                mask = mask & col_mask
        
        # Get all rows matching this violation
        # Use .loc to avoid reindexing warning
        violation_rows = results_df.loc[mask].copy()
        
        if len(violation_rows) == 0:
            no_match_count += 1
            if no_match_count <= 3:  # Only print first 3 with full details
                print(f"\n    [DEBUG] Violation {idx}: No matching rows found")
                print(f"      Violation values: {dict([(col, violation[col]) for col in grouping_cols if col in violation.index])}")
                print(f"      Match details: {match_details}")
                if mismatch_info:
                    print(f"      Mismatches:")
                    for mm in mismatch_info:
                        print(f"        {mm['col']}: violation has {mm['violation_val']} ({mm['violation_type']}), "
                              f"results dtype={mm['results_dtype']}, sample values={mm['sample_results_vals']}")
                # Show how many rows match each individual column
                print(f"      Individual column match counts:")
                for col in grouping_cols:
                    if col in results_df.columns and col in violation.index:
                        violation_val = violation[col]
                        if pd.isna(violation_val):
                            match_count = results_df[col].isna().sum()
                        else:
                            match_count = (results_df[col] == violation_val).sum()
                        print(f"        {col}={violation_val}: {match_count} rows match")
            continue
        
        # Analyze which columns vary
        varying_cols = {}
        constant_cols = {}
        
        for col in all_cols:
            if col in ['clean_score']:  # Skip clean_score itself
                continue
            
            unique_vals = violation_rows[col].dropna().unique()
            if len(unique_vals) > 1:
                # Column varies
                varying_cols[col] = {
                    'num_unique': len(unique_vals),
                    'unique_values': list(unique_vals)[:10],  # Limit to first 10
                    'has_nan': violation_rows[col].isna().any()
                }
            elif len(unique_vals) == 1:
                # Column is constant
                constant_cols[col] = unique_vals[0]
        
        # Analyze clean scores by noise type
        clean_scores_by_noise = {}
        if 'noise_type' in violation_rows.columns:
            for noise_type in violation_rows['noise_type'].dropna().unique():
                noise_rows = violation_rows[violation_rows['noise_type'] == noise_type]
                scores = noise_rows['clean_score'].dropna().unique()
                if len(scores) > 0:
                    clean_scores_by_noise[noise_type] = {
                        'num_unique': len(scores),
                        'values': sorted(scores)[:10],  # Limit to first 10
                        'range': (float(np.min(scores)), float(np.max(scores))),
                        'num_rows': len(noise_rows)
                    }
        
        violation_analyses.append({
            'violation_idx': idx,
            'num_matching_rows': len(violation_rows),
            'num_varying_cols': len(varying_cols),
            'varying_columns': varying_cols,
            'clean_scores_by_noise': clean_scores_by_noise,
            'grouping_values': {col: violation[col] for col in grouping_cols}
        })
    
    # Print summary
    print(f"\n[4] SUMMARY")
    print("-" * 80)
    print(f"Total violations analyzed: {len(violations_df)}")
    print(f"Violations with matching rows: {len(violation_analyses)}")
    print(f"Violations with no matching rows: {no_match_count}")
    
    if len(violation_analyses) == 0:
        print("\n[WARNING] No violations had matching rows in the unified CSV.")
        print("This suggests the violation CSV and unified CSV are from different data sources.")
        print("The violations may have been generated from individual dataset CSVs that contain")
        print("different eval_modes (e.g., CrossSession, WithinSession) than what's in the unified CSV.")
        print("\nAnalyzing violations directly (without matching to unified CSV)...")
        
        # Analyze violations directly to see what columns vary
        print("\n[ALTERNATIVE ANALYSIS] Examining violations directly:")
        print("-" * 80)
        
        # For each violation, we can't match to rows, but we can analyze the violation data itself
        # Group violations by their grouping columns and see what other columns vary
        violation_varying_cols = {}
        
        # Check if there are any non-grouping columns in violations that might explain differences
        non_grouping_violation_cols = [col for col in violations_df.columns 
                                       if col not in grouping_cols + ['num_noise_types', 'unique_clean_scores', 
                                                                      'score_range', 'score_diff', 'median_diff', 
                                                                      'max_intra_noise_variation']]
        
        if non_grouping_violation_cols:
            print(f"\nNon-grouping columns in violations: {non_grouping_violation_cols}")
            for col in non_grouping_violation_cols:
                if col.endswith('_representative') or col.endswith('_num_unique') or col.endswith('_intra_range'):
                    continue
                unique_vals = violations_df[col].dropna().unique()
                if len(unique_vals) > 1:
                    violation_varying_cols[col] = {
                        'num_unique': len(unique_vals),
                        'sample_values': list(unique_vals)[:10]
                    }
        
        if violation_varying_cols:
            print(f"\nColumns that vary across violations:")
            for col, info in sorted(violation_varying_cols.items(), key=lambda x: x[1]['num_unique'], reverse=True):
                print(f"  {col}: {info['num_unique']} unique values, sample: {info['sample_values'][:5]}")
        
        # Analyze violations by grouping to see patterns
        print(f"\nViolation patterns by grouping columns:")
        for col in grouping_cols:
            if col in violations_df.columns:
                unique_vals = violations_df[col].dropna().unique()
                print(f"  {col}: {len(unique_vals)} unique values")
                if len(unique_vals) <= 10:
                    print(f"    Values: {sorted(unique_vals)}")
                else:
                    print(f"    Sample values: {sorted(unique_vals)[:10]}")
        
        print(f"\n[NOTE] To properly diagnose these violations, you may need to:")
        print(f"  1. Use the same data source that generated the violations (individual dataset CSVs)")
        print(f"  2. Or regenerate the unified CSV to include all eval_modes")
        print(f"  3. Or examine the violation CSV directly to see what columns vary")
        
        return
    
    # Count how many violations have varying columns
    violations_with_varying_cols = [v for v in violation_analyses if v['num_varying_cols'] > 0]
    print(f"Violations with varying columns: {len(violations_with_varying_cols)}/{len(violation_analyses)}")
    
    # Find most common varying columns
    all_varying_cols = {}
    for analysis in violation_analyses:
        for col_name, col_info in analysis['varying_columns'].items():
            if col_name not in all_varying_cols:
                all_varying_cols[col_name] = {
                    'count': 0,
                    'total_unique_vals': set(),
                    'sample_values': []
                }
            all_varying_cols[col_name]['count'] += 1
            all_varying_cols[col_name]['total_unique_vals'].update(col_info['unique_values'])
            if len(all_varying_cols[col_name]['sample_values']) < 20:
                all_varying_cols[col_name]['sample_values'].extend(col_info['unique_values'][:5])
    
    if all_varying_cols:
        print(f"\nMost common varying columns (across all violations):")
        sorted_cols = sorted(all_varying_cols.items(), key=lambda x: x[1]['count'], reverse=True)
        for col_name, col_info in sorted_cols[:20]:  # Top 20
            print(f"  {col_name}:")
            print(f"    - Varies in {col_info['count']}/{len(violation_analyses)} violations")
            print(f"    - Total unique values seen: {len(col_info['total_unique_vals'])}")
            sample_vals = list(col_info['sample_values'])[:5]
            print(f"    - Sample values: {sample_vals}")
    
    # Show detailed analysis for first few violations
    print(f"\n[5] DETAILED ANALYSIS (first 5 violations)")
    print("-" * 80)
    
    for i, analysis in enumerate(violation_analyses[:5]):
        print(f"\nViolation {i+1} (index {analysis['violation_idx']}):")
        print(f"  Matching rows: {analysis['num_matching_rows']}")
        print(f"  Varying columns: {analysis['num_varying_cols']}")
        
        if analysis['num_varying_cols'] > 0:
            print(f"  Varying columns:")
            for col_name, col_info in sorted(analysis['varying_columns'].items()):
                print(f"    - {col_name}:")
                print(f"        Unique values: {col_info['num_unique']}")
                print(f"        Sample: {col_info['unique_values'][:5]}")
        
        if analysis['clean_scores_by_noise']:
            print(f"  Clean scores by noise type:")
            for noise_type, noise_info in analysis['clean_scores_by_noise'].items():
                print(f"    - {noise_type}:")
                print(f"        Unique scores: {noise_info['num_unique']}")
                print(f"        Range: {noise_info['range'][0]:.6f} - {noise_info['range'][1]:.6f}")
                print(f"        Rows: {noise_info['num_rows']}")
                if len(noise_info['values']) <= 5:
                    print(f"        All values: {noise_info['values']}")
                else:
                    print(f"        Sample values: {noise_info['values'][:5]}...")
    
    # Save detailed report
    output_file = violation_csv_path.replace('.csv', '_diagnosis.txt')
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VIOLATION DIAGNOSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total violations analyzed: {len(violation_analyses)}\n")
        f.write(f"Violations with varying columns: {len(violations_with_varying_cols)}\n\n")
        
        f.write("MOST COMMON VARYING COLUMNS:\n")
        f.write("-" * 80 + "\n")
        for col_name, col_info in sorted(all_varying_cols.items(), key=lambda x: x[1]['count'], reverse=True):
            f.write(f"\n{col_name}:\n")
            f.write(f"  - Varies in {col_info['count']}/{len(violation_analyses)} violations\n")
            f.write(f"  - Total unique values: {len(col_info['total_unique_vals'])}\n")
            f.write(f"  - Sample values: {list(col_info['total_unique_vals'])[:20]}\n")
        
        f.write("\n\nDETAILED ANALYSIS FOR ALL VIOLATIONS:\n")
        f.write("=" * 80 + "\n")
        
        for i, analysis in enumerate(violation_analyses):
            f.write(f"\nViolation {i+1} (index {analysis['violation_idx']}):\n")
            f.write(f"  Grouping values: {analysis['grouping_values']}\n")
            f.write(f"  Matching rows: {analysis['num_matching_rows']}\n")
            f.write(f"  Varying columns: {analysis['num_varying_cols']}\n")
            
            if analysis['num_varying_cols'] > 0:
                f.write(f"  Varying columns:\n")
                for col_name, col_info in sorted(analysis['varying_columns'].items()):
                    f.write(f"    - {col_name}:\n")
                    f.write(f"        Unique values: {col_info['num_unique']}\n")
                    f.write(f"        Values: {col_info['unique_values']}\n")
                    if col_info['has_nan']:
                        f.write(f"        (Also contains NaN)\n")
            
            if analysis['clean_scores_by_noise']:
                f.write(f"  Clean scores by noise type:\n")
                for noise_type, noise_info in analysis['clean_scores_by_noise'].items():
                    f.write(f"    - {noise_type}:\n")
                    f.write(f"        Unique scores: {noise_info['num_unique']}\n")
                    f.write(f"        Range: {noise_info['range'][0]:.6f} - {noise_info['range'][1]:.6f}\n")
                    f.write(f"        Rows: {noise_info['num_rows']}\n")
                    f.write(f"        Values: {noise_info['values']}\n")
    
    print(f"\n[6] Detailed report saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_violation_sources.py <violation_csv> [results_csv] [dataset_name]")
        print("\nExample:")
        print("  python diagnose_violation_sources.py analysis/analysis/sanity_check_violations_BNCI2014_001.csv")
        print("  python diagnose_violation_sources.py analysis/analysis/sanity_check_violations_BNCI2014_001.csv evaluation/results/unified_all_results.csv")
        sys.exit(1)
    
    violation_csv = sys.argv[1]
    results_csv = sys.argv[2] if len(sys.argv) > 2 else None
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(violation_csv):
        print(f"[ERROR] Violation CSV not found: {violation_csv}")
        sys.exit(1)
    
    # If results_csv provided, check it exists
    if results_csv and not os.path.exists(results_csv):
        print(f"[ERROR] Results CSV not found: {results_csv}")
        sys.exit(1)
    
    diagnose_violation_sources(violation_csv, results_csv, dataset_name)

