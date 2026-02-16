#!/usr/bin/env python3
"""Test script to investigate aggregation issue."""

import pandas as pd
import os

# Test reading a correct individual file
test_file = r'results\SSVEP\Lee2019_SSVEP\cnn_ncp\WithinSessionEvaluation\100\sub-001\1\test_perturb\cnn_ncp_test_perturb_subject_001_seed100.csv'

if os.path.exists(test_file):
    print(f"Reading: {test_file}")
    df = pd.read_csv(test_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if 'noise_type' in df.columns:
        print(f"Unique noise types: {df['noise_type'].unique()}")
        print(f"Noise type counts:\n{df['noise_type'].value_counts()}")
else:
    print(f"File not found: {test_file}")

# Check if sol_results has all_results.csv
sol_results_file = r'sol_results\SSVEP\Lee2019_SSVEP\all_results.csv'
if os.path.exists(sol_results_file):
    print(f"\nReading: {sol_results_file}")
    df_sol = pd.read_csv(sol_results_file, nrows=1000, low_memory=False)
    print(f"Shape: {df_sol.shape}")
    
    # Filter for WithinSession
    if 'eval_mode' in df_sol.columns:
        ws = df_sol[df_sol['eval_mode'].str.contains('WithinSession', na=False)]
        print(f"WithinSession rows in sample: {len(ws)}")
        if len(ws) > 0 and 'noise_type' in ws.columns:
            print(f"Unique noise types in WithinSession: {ws['noise_type'].unique()}")
            print(f"Noise type counts:\n{ws['noise_type'].value_counts()}")
