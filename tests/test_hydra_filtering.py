"""Test script to diagnose HYDRA filtering issue in robustness_metrics.py"""

import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.robustness_metrics import load_results_dataframe, compute_results_metrics, MetricConfig

# Test loading with hydra=True
print("=" * 80)
print("TESTING HYDRA FILTERING")
print("=" * 80)

# Load data with hydra=True
print("\n[TEST 1] Loading results with hydra=True...")
df = load_results_dataframe(hydra=True, aggregate_from_directories=True)

print(f"\nLoaded DataFrame shape: {df.shape}")
print(f"Models in loaded data: {df['model'].unique() if 'model' in df.columns else 'N/A'}")
print(f"Number of rows per model:")
if 'model' in df.columns:
    print(df['model'].value_counts())

# Test computing metrics with hydra=True
print("\n[TEST 2] Computing metrics with hydra=True...")
cfg = MetricConfig(metric_col="corrupted_score")
results = compute_results_metrics(df, cfg=cfg, hydra=True)

print(f"\nResults keys: {list(results.keys())}")
for key, result_df in results.items():
    if result_df is not None and not result_df.empty:
        print(f"\n{key}:")
        print(f"  Shape: {result_df.shape}")
        if 'model' in result_df.columns:
            print(f"  Models: {result_df['model'].unique()}")
            print(f"  Rows per model:")
            print(result_df['model'].value_counts())
    else:
        print(f"\n{key}: Empty or None")
