#!/usr/bin/env python3
"""
Analyze EEGNet results on Lee2019_SSVEP dataset for suspicious patterns.

This script reads the result CSV files and identifies:
1. Suspiciously high and consistent accuracy values
2. Patterns that might indicate data leakage or evaluation bugs
3. Inconsistencies between clean_score and clean_accuracy
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def analyze_results():
    """Analyze EEGNet results for suspicious patterns."""
    
    # Find all result CSV files for EEGNet on Lee2019_SSVEP
    results_dir = Path("results/SSVEP/Lee2019_SSVEP/eegnet/CrossSessionEvaluation/42")
    
    csv_files = list(results_dir.glob("**/eegnet_test_perturb_subject_*_seed42.csv"))
    
    if not csv_files:
        print("No result files found!")
        return
    
    print(f"Found {len(csv_files)} result files")
    print("=" * 80)
    
    # Collect all results
    all_results = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print("\n" + "=" * 80)
    print("ANALYSIS OF EEGNET RESULTS ON LEE2019_SSVEP")
    print("=" * 80)
    
    # Group by subject and session
    grouped = combined_df.groupby(['subject', 'session'])
    
    print("\n1. CLEAN ACCURACY BY SUBJECT AND SESSION:")
    print("-" * 80)
    clean_acc_summary = []
    for (subject, session), group in grouped:
        # Get clean data (intensity=1.0, noise_type='gaussian' is typically the baseline)
        clean_data = group[(group['intensity'] == 1.0) & (group['noise_type'] == 'gaussian')]
        if len(clean_data) > 0:
            clean_acc = clean_data['clean_accuracy'].iloc[0]
            clean_score = clean_data['clean_score'].iloc[0]
            clean_acc_summary.append({
                'subject': subject,
                'session': session,
                'clean_accuracy': clean_acc,
                'clean_score': clean_score
            })
            print(f"Subject {subject}, Session {session}: clean_accuracy={clean_acc:.4f}, clean_score={clean_score:.4f}")
    
    clean_acc_df = pd.DataFrame(clean_acc_summary)
    
    print("\n2. STATISTICAL SUMMARY OF CLEAN ACCURACY:")
    print("-" * 80)
    print(f"Mean: {clean_acc_df['clean_accuracy'].mean():.4f}")
    print(f"Std: {clean_acc_df['clean_accuracy'].std():.4f}")
    print(f"Min: {clean_acc_df['clean_accuracy'].min():.4f}")
    print(f"Max: {clean_acc_df['clean_accuracy'].max():.4f}")
    print(f"Median: {clean_acc_df['clean_accuracy'].median():.4f}")
    
    print("\n3. SUSPICIOUS PATTERNS:")
    print("-" * 80)
    
    # Check for suspiciously high accuracy (>0.95)
    high_acc = clean_acc_df[clean_acc_df['clean_accuracy'] > 0.95]
    if len(high_acc) > 0:
        print(f"\n⚠️  Found {len(high_acc)} cases with clean_accuracy > 0.95:")
        for _, row in high_acc.iterrows():
            print(f"   Subject {row['subject']}, Session {row['session']}: {row['clean_accuracy']:.4f}")
    
    # Check for suspiciously consistent accuracy across subjects
    if clean_acc_df['clean_accuracy'].std() < 0.05:
        print(f"\n⚠️  Very low standard deviation ({clean_acc_df['clean_accuracy'].std():.4f}) suggests suspiciously consistent accuracy")
    
    # Check for perfect or near-perfect clean_score
    perfect_score = clean_acc_df[clean_acc_df['clean_score'] > 0.99]
    if len(perfect_score) > 0:
        print(f"\n⚠️  Found {len(perfect_score)} cases with clean_score > 0.99:")
        for _, row in perfect_score.iterrows():
            print(f"   Subject {row['subject']}, Session {row['session']}: clean_score={row['clean_score']:.4f}, clean_accuracy={row['clean_accuracy']:.4f}")
    
    # Check for cases where corrupted_score equals clean_score (suggests evaluation bug)
    print("\n4. CHECKING FOR EVALUATION BUGS:")
    print("-" * 80)
    for (subject, session), group in grouped:
        # Check gaussian noise results
        gaussian_data = group[group['noise_type'] == 'gaussian']
        if len(gaussian_data) > 0:
            # Check if corrupted_score equals clean_score for multiple intensities
            matches = gaussian_data[gaussian_data['corrupted_score'] == gaussian_data['clean_score']]
            if len(matches) > 3:  # More than 3 matches is suspicious
                print(f"\n⚠️  Subject {subject}, Session {session}: {len(matches)} cases where corrupted_score == clean_score for gaussian noise")
                print(f"   This suggests the model might not be properly evaluating on corrupted data")
                print(f"   Example: intensity={matches.iloc[0]['intensity']:.2f}, clean_score={matches.iloc[0]['clean_score']:.4f}")
    
    # Check for cases where accuracy doesn't change with noise
    print("\n5. CHECKING NOISE ROBUSTNESS:")
    print("-" * 80)
    for (subject, session), group in grouped:
        # Get clean accuracy
        clean_data = group[(group['intensity'] == 1.0) & (group['noise_type'] == 'gaussian')]
        if len(clean_data) == 0:
            continue
        clean_acc = clean_data['corrupted_accuracy'].iloc[0]
        
        # Check high intensity noise
        high_noise = group[(group['intensity'] >= 50.0) & (group['noise_type'] == 'gaussian')]
        if len(high_noise) > 0:
            high_noise_acc = high_noise['corrupted_accuracy'].mean()
            if abs(clean_acc - high_noise_acc) < 0.01:  # Less than 1% change
                print(f"\n⚠️  Subject {subject}, Session {session}: Accuracy barely changes with high noise")
                print(f"   Clean accuracy: {clean_acc:.4f}, High noise accuracy: {high_noise_acc:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_results()
