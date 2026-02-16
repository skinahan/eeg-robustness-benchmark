

"""
Analyze which columns vary in violations to identify the root cause of clean score differences.
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_violation_varying_cols(violation_csv_path):
    """
    Analyze the violation CSV to identify which columns vary and might explain clean score differences.
    """
    print("=" * 80)
    print("ANALYZING VARYING COLUMNS IN VIOLATIONS")
    print("=" * 80)
    
    # Load violation report
    print(f"\n[1] Loading violation report: {violation_csv_path}")
    violations_df = pd.read_csv(violation_csv_path)
    print(f"    Found {len(violations_df)} violations")
    
    # Find all columns that start with "varying_"
    varying_cols = [col for col in violations_df.columns if col.startswith('varying_')]
    print(f"\n[2] Found {len(varying_cols)} varying column indicators")
    
    # Group by base column name (remove _num_unique and _sample suffixes)
    varying_base_cols = {}
    for col in varying_cols:
        if col.endswith('_num_unique'):
            base_col = col.replace('_num_unique', '')
            if base_col not in varying_base_cols:
                varying_base_cols[base_col] = {}
            varying_base_cols[base_col]['num_unique_col'] = col
        elif col.endswith('_sample'):
            base_col = col.replace('_sample', '')
            if base_col not in varying_base_cols:
                varying_base_cols[base_col] = {}
            varying_base_cols[base_col]['sample_col'] = col
    
    print(f"\n[3] Varying columns summary:")
    print("-" * 80)
    
    # Analyze each varying column
    varying_summary = []
    for base_col, info in sorted(varying_base_cols.items()):
        num_unique_col = info.get('num_unique_col')
        sample_col = info.get('sample_col')
        
        if num_unique_col and num_unique_col in violations_df.columns:
            # Count how many violations have this column varying
            non_null = violations_df[num_unique_col].notna()
            if non_null.any():
                num_violations_with_variation = non_null.sum()
                avg_unique_vals = violations_df[num_unique_col].mean()
                max_unique_vals = violations_df[num_unique_col].max()
                
                varying_summary.append({
                    'column': base_col,
                    'num_violations': num_violations_with_variation,
                    'pct_violations': 100 * num_violations_with_variation / len(violations_df),
                    'avg_unique_values': avg_unique_vals,
                    'max_unique_values': max_unique_vals
                })
    
    # Sort by number of violations affected
    varying_summary.sort(key=lambda x: x['num_violations'], reverse=True)
    
    print(f"\nColumns that vary across violations (sorted by frequency):")
    print(f"{'Column':<30} {'Violations':<12} {'%':<8} {'Avg Unique':<12} {'Max Unique':<12}")
    print("-" * 80)
    
    # Categorize columns
    metadata_cols = []  # Columns that shouldn't vary (like fold_idx, eval_subjects)
    performance_cols = []  # Columns that are expected to vary (like corrupted_score, score)
    other_cols = []
    
    for item in varying_summary:
        col = item['column']
        if col in ['fold_idx', 'eval_subjects', 'n_eval_subjects', 'cv_type', 'split_level']:
            metadata_cols.append(item)
        elif col in ['corrupted_score', 'score', 'corrupted_roc_auc', 'corrupted_accuracy', 
                     'corrupted_precision', 'corrupted_recall', 'corrupted_f1', 'relative_drop',
                     'clean_roc_auc', 'clean_accuracy', 'clean_precision', 'clean_recall', 'clean_f1']:
            performance_cols.append(item)
        else:
            other_cols.append(item)
    
    # Print metadata columns first (most interesting - these shouldn't vary!)
    if metadata_cols:
        print("\n[METADATA COLUMNS - These shouldn't vary and may explain clean score differences]:")
        for item in metadata_cols:
            print(f"  {item['column']:<30} {item['num_violations']:<12} {item['pct_violations']:>6.1f}% "
                  f"{item['avg_unique_values']:>10.1f} {item['max_unique_values']:>10.0f}")
    
    # Print other interesting columns
    if other_cols:
        print("\n[OTHER COLUMNS - May be relevant]:")
        for item in other_cols[:10]:  # Top 10
            print(f"  {item['column']:<30} {item['num_violations']:<12} {item['pct_violations']:>6.1f}% "
                  f"{item['avg_unique_values']:>10.1f} {item['max_unique_values']:>10.0f}")
    
    # Print performance columns (expected to vary, less interesting)
    if performance_cols:
        print("\n[PERFORMANCE COLUMNS - Expected to vary by intensity, less relevant]:")
        for item in performance_cols[:5]:  # Top 5
            print(f"  {item['column']:<30} {item['num_violations']:<12} {item['pct_violations']:>6.1f}% "
                  f"{item['avg_unique_values']:>10.1f} {item['max_unique_values']:>10.0f}")
    
    # Show detailed examples for metadata columns
    if metadata_cols:
        print(f"\n[4] DETAILED ANALYSIS OF METADATA COLUMNS:")
        print("-" * 80)
        
        for item in metadata_cols:
            col = item['column']
            num_unique_col = f'varying_{col}_num_unique'
            sample_col = f'varying_{col}_sample'
            
            if num_unique_col in violations_df.columns:
                # Get violations where this column varies
                varying_violations = violations_df[violations_df[num_unique_col].notna()].copy()
                
                print(f"\n{col}:")
                print(f"  Varies in {len(varying_violations)}/{len(violations_df)} violations")
                
                # Show sample values
                if sample_col in violations_df.columns:
                    # Get unique sample values (first 100 chars of each)
                    samples = varying_violations[sample_col].dropna().unique()[:5]
                    print(f"  Sample values (first 5):")
                    for i, sample in enumerate(samples, 1):
                        sample_str = str(sample)[:150]
                        print(f"    {i}. {sample_str}...")
                
                # Show distribution of unique value counts
                if num_unique_col in varying_violations.columns:
                    unique_counts = varying_violations[num_unique_col].value_counts().head(5)
                    print(f"  Distribution of unique value counts:")
                    for count, freq in unique_counts.items():
                        print(f"    {count} unique values: {freq} violations")
    
    # Summary
    print(f"\n[5] SUMMARY")
    print("-" * 80)
    print(f"Total violations: {len(violations_df)}")
    if metadata_cols:
        print(f"\n⚠️  CRITICAL FINDING: {len(metadata_cols)} metadata columns vary across violations!")
        print(f"   These columns should be constant for a given (model, dataset, seed, subject, tune, session, eval_mode) combination.")
        print(f"   If they vary, it suggests rows from different experimental runs are being grouped together.")
        print(f"\n   Varying metadata columns: {[item['column'] for item in metadata_cols]}")
    else:
        print(f"\n[OK] No metadata columns vary - violations are likely due to other factors.")
    
    return varying_summary, metadata_cols


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_violation_varying_cols.py <violation_csv>")
        print("\nExample:")
        print("  python analyze_violation_varying_cols.py analysis/analysis/sanity_check_violations_BNCI2014_001.csv")
        sys.exit(1)
    
    violation_csv = sys.argv[1]
    
    if not os.path.exists(violation_csv):
        print(f"[ERROR] Violation CSV not found: {violation_csv}")
        sys.exit(1)
    
    analyze_violation_varying_cols(violation_csv)

