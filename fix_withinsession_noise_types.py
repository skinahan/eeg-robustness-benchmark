#!/usr/bin/env python3
"""
Utility script to fix noise_type labels in old WithinSession results.

The old buggy code only saved results with noise_type='eog', but the actual results
for all 4 noise types were present. This script relabels them based on row position:

- Rows 0-19:   eog
- Rows 20-39:  gaussian  
- Rows 40-59:  dropout
- Rows 60-79:  spike

Usage:
    python fix_withinsession_noise_types.py [--dataset DATASET] [--dry-run]
"""

import os
import sys
import argparse
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

# Expected noise type order (from code)
NOISE_TYPE_ORDER = ['eog', 'gaussian', 'dropout', 'spike']
INTENSITIES_PER_NOISE_TYPE = 20

def create_backup(file_path: str) -> str:
    """Create a backup copy of the file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    return backup_path

def fix_noise_types_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix noise_type labels in a DataFrame based on row position.
    
    Assumes that results are grouped by (subject, session) and that within each group,
    rows are ordered: 20×eog, 20×gaussian, 20×dropout, 20×spike
    
    Args:
        df: DataFrame with WithinSession results
        
    Returns:
        DataFrame with corrected noise_type labels
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Group by subject and session to handle multiple groups in one file
    # For WithinSession, results should be grouped by (subject, session)
    if 'subject' in df.columns and 'session' in df.columns:
        groups = df.groupby(['subject', 'session'])
    elif 'subject' in df.columns:
        groups = df.groupby(['subject'])
    else:
        # No grouping - assume all rows are one group
        groups = [(None, df)]
    
    fixed_dfs = []
    
    for group_key, group_df in groups:
        group_df = group_df.copy()
        n_rows = len(group_df)
        
        # Check if this group has all rows labeled as 'eog' (indicates buggy results)
        unique_noise_types = group_df['noise_type'].unique()
        if len(unique_noise_types) == 1 and unique_noise_types[0] == 'eog':
            # This group needs fixing
            if n_rows % INTENSITIES_PER_NOISE_TYPE != 0:
                print(f"  WARNING: Group {group_key} has {n_rows} rows, not a multiple of {INTENSITIES_PER_NOISE_TYPE}")
            
            # Calculate how many complete noise type blocks we have
            n_blocks = n_rows // INTENSITIES_PER_NOISE_TYPE
            remaining_rows = n_rows % INTENSITIES_PER_NOISE_TYPE
            
            if remaining_rows > 0:
                print(f"  WARNING: Group {group_key} has {remaining_rows} extra rows that won't be relabeled")
            
            # Relabel noise types based on position
            fixed_noise_types = []
            for block_idx in range(n_blocks):
                noise_type = NOISE_TYPE_ORDER[block_idx % len(NOISE_TYPE_ORDER)]
                fixed_noise_types.extend([noise_type] * INTENSITIES_PER_NOISE_TYPE)
            
            # Keep original label for any remaining rows
            if remaining_rows > 0:
                fixed_noise_types.extend([group_df['noise_type'].iloc[-1]] * remaining_rows)
            
            group_df['noise_type'] = fixed_noise_types[:n_rows]
            print(f"  Fixed {n_blocks} noise type blocks ({n_blocks * INTENSITIES_PER_NOISE_TYPE} rows) for group {group_key}")
        else:
            # Already has multiple noise types or different label - skip
            print(f"  Skipping group {group_key} - already has multiple noise types or not 'eog': {unique_noise_types}")
        
        fixed_dfs.append(group_df)
    
    result_df = pd.concat(fixed_dfs, ignore_index=True) if len(fixed_dfs) > 1 else fixed_dfs[0]
    return result_df

def fix_csv_file(csv_path: str, dry_run: bool = False) -> bool:
    """
    Fix noise_type labels in a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        dry_run: If True, don't modify files, just report what would be changed
        
    Returns:
        True if file was fixed (or would be fixed), False if skipped
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"  Skipping {os.path.basename(csv_path)}: empty file")
            return False
        
        # Check if this is a WithinSession result file
        if 'eval_mode' in df.columns:
            eval_modes = df['eval_mode'].unique()
            if 'WithinSession' not in eval_modes and 'WithinSessionEvaluation' not in [str(x) for x in eval_modes]:
                # print(f"  Skipping {os.path.basename(csv_path)}: not WithinSession (eval_modes: {eval_modes})")
                return False
        
        # Check if file has all rows labeled as 'eog'
        if 'noise_type' not in df.columns:
            print(f"  Skipping {os.path.basename(csv_path)}: no 'noise_type' column")
            return False
        
        unique_noise_types = df['noise_type'].unique()
        if len(unique_noise_types) > 1 or (len(unique_noise_types) == 1 and unique_noise_types[0] != 'eog'):
            print(f"  Skipping {os.path.basename(csv_path)}: already has multiple noise types or not all 'eog'")
            return False
        
        # Check if file looks like it needs fixing (all 'eog' and multiple of 20 rows per group)
        # For WithinSession, we expect 80 rows per (subject, session) combination
        if 'subject' in df.columns and 'session' in df.columns:
            groups = df.groupby(['subject', 'session'])
            needs_fixing = False
            for group_key, group_df in groups:
                if len(group_df) >= 80 and group_df['noise_type'].nunique() == 1:
                    needs_fixing = True
                    break
            if not needs_fixing:
                # print(f"  Skipping {os.path.basename(csv_path)}: doesn't match expected pattern")
                return False
        
        print(f"\nProcessing: {os.path.basename(csv_path)}")
        print(f"  Original rows: {len(df)}")
        print(f"  Original noise types: {unique_noise_types}")
        
        # Fix noise types
        fixed_df = fix_noise_types_in_dataframe(df)
        
        # Verify the fix
        new_noise_types = fixed_df['noise_type'].unique()
        print(f"  Fixed noise types: {sorted(new_noise_types)}")
        print(f"  Noise type counts: {dict(fixed_df['noise_type'].value_counts())}")
        
        if dry_run:
            print(f"  [DRY RUN] Would fix and save this file")
            return True
        
        # Create backup
        backup_path = create_backup(csv_path)
        print(f"  Created backup: {os.path.basename(backup_path)}")
        
        # Save fixed file
        fixed_df.to_csv(csv_path, index=False)
        print(f"  Saved fixed file: {os.path.basename(csv_path)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_withinsession_csv_files(base_dir: str, dataset: str = None, model: str = None) -> list:
    """
    Find all CSV files with WithinSession results.
    
    Searches for files matching pattern: <base_dir>/<paradigm>/<dataset>/<model>/WithinSessionEvaluation/...
    
    Args:
        base_dir: Base directory to search (e.g., 'results' or 'sol_results')
        dataset: Optional dataset name to filter (e.g., 'Lee2019_SSVEP')
        model: Optional model name to filter (e.g., 'cnn_ncp')
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    # Paradigms to search
    paradigms = ["SSVEP", "MotorImagery", "ERP"]
    
    for paradigm in paradigms:
        paradigm_dir = os.path.join(base_dir, paradigm)
        if not os.path.exists(paradigm_dir):
            continue
        
        # Get datasets in this paradigm
        if dataset:
            dataset_dirs = [os.path.join(paradigm_dir, dataset)] if os.path.exists(os.path.join(paradigm_dir, dataset)) else []
        else:
            dataset_dirs = [os.path.join(paradigm_dir, d) for d in os.listdir(paradigm_dir) 
                           if os.path.isdir(os.path.join(paradigm_dir, d))]
        
        for dataset_dir in dataset_dirs:
            if not os.path.exists(dataset_dir):
                continue
            
            # Get models in this dataset
            if model:
                model_dirs = [os.path.join(dataset_dir, model)] if os.path.exists(os.path.join(dataset_dir, model)) else []
            else:
                model_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
                             if os.path.isdir(os.path.join(dataset_dir, d))]
            
            for model_dir in model_dirs:
                if not os.path.exists(model_dir):
                    continue
                
                # Look for WithinSessionEvaluation directory
                wseval_dir = os.path.join(model_dir, "WithinSessionEvaluation")
                if os.path.exists(wseval_dir):
                    # Walk through subdirectories and find all CSV files
                    for root, dirs, files in os.walk(wseval_dir):
                        for file in files:
                            if file.endswith('.csv') and not file.endswith('all_results.csv'):
                                csv_path = os.path.join(root, file)
                                csv_files.append(csv_path)
    
    return csv_files

def main():
    parser = argparse.ArgumentParser(description="Fix noise_type labels in old WithinSession results")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to process (e.g., 'Lee2019_SSVEP'). If not specified, processes all datasets.")
    parser.add_argument("--base-dir", type=str, default="results",
                       help="Base directory to search for CSV files (default: 'results')")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to process (e.g., 'cnn_ncp'). If not specified, processes all models.")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't modify files, just report what would be changed")
    parser.add_argument("--all-results", type=str, default=None,
                       help="Path to all_results.csv file to fix directly (instead of searching for individual files)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("FIXING WITHINSESSION NOISE_TYPE LABELS")
    print("="*80)
    print(f"Base directory: {args.base_dir}")
    if args.dataset:
        print(f"Dataset filter: {args.dataset}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()
    
    if args.all_results:
        # Fix the aggregated all_results.csv file directly
        print(f"Processing aggregated file: {args.all_results}")
        csv_files = [args.all_results]
    else:
        # Find individual CSV files
        print("Searching for WithinSession CSV files...")
        print(f"Pattern: {args.base_dir}/<paradigm>/<dataset>/<model>/WithinSessionEvaluation/...")
        csv_files = find_withinsession_csv_files(args.base_dir, args.dataset, args.model)
        print(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        print("No CSV files found to process.")
        return
    
    print(f"\nProcessing {len(csv_files)} files...")
    print("-"*80)
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        if fix_csv_file(csv_file, dry_run=args.dry_run):
            fixed_count += 1
        else:
            skipped_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files processed: {len(csv_files)}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"\nBackup files created with timestamp suffix.")
        print("Original files have been updated with corrected noise_type labels.")

if __name__ == "__main__":
    main()
