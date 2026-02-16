import pandas as pd
import numpy as np

# Load violation CSV
df = pd.read_csv('analysis/sanity_check_violations_BNCI2014_001.csv')

print("=" * 80)
print("ANALYZING VIOLATION DETAILS")
print("=" * 80)

print(f"\nTotal violations: {len(df)}")

# Check first violation in detail
print("\n" + "=" * 80)
print("FIRST VIOLATION DETAILS")
print("=" * 80)
first = df.iloc[0]
print(f"Grouping: model={first['model']}, dataset={first['dataset']}, seed={first['seed']}, subject={first['subject']}, tune={first['tune']}, session={first['session']}, eval_mode={first['eval_mode']}")
print(f"\nIntra-noise variation:")
print(f"  EOG: {first['eog_num_unique']} unique clean scores, range={first['eog_intra_range']:.6f}")
print(f"  Gaussian: {first['gaussian_num_unique']} unique clean scores, range={first['gaussian_intra_range']:.6f}")
print(f"  Dropout: {first['dropout_num_unique']} unique clean scores, range={first['dropout_intra_range']:.6f}")
print(f"  Max intra-noise variation: {first['max_intra_noise_variation']:.6f}")

# Check which columns vary
print("\n" + "=" * 80)
print("VARYING COLUMNS IN FIRST VIOLATION")
print("=" * 80)
varying_cols = [c for c in df.columns if c.startswith('varying_') and c.endswith('_num_unique')]
for col in varying_cols:
    val = first[col]
    if pd.notna(val) and val > 1:
        base_col = col.replace('varying_', '').replace('_num_unique', '')
        sample_col = f'varying_{base_col}_sample'
        sample_val = first[sample_col] if sample_col in df.columns else 'N/A'
        print(f"  {base_col}: {int(val)} unique values")
        if pd.notna(sample_val):
            print(f"    Sample values: {str(sample_val)[:200]}")

# Check if there are any metadata columns that vary
print("\n" + "=" * 80)
print("METADATA COLUMNS THAT VARY (across all violations)")
print("=" * 80)
metadata_cols = ['fold_idx', 'run_id', 'experiment_id', 'trial', 'replicate', 'iteration', 
                 'cv_type', 'split_level', 'verbose']  # Add cv_type and split_level which are in the data
found_metadata_variation = False
for col in metadata_cols:
    varying_col = f'varying_{col}_num_unique'
    if varying_col in df.columns:
        violations_with_variation = df[df[varying_col] > 1]
        if len(violations_with_variation) > 0:
            found_metadata_variation = True
            print(f"  {col}: {len(violations_with_variation)} violations have varying {col}")
            print(f"    Average unique values: {violations_with_variation[varying_col].mean():.1f}")
            print(f"    Max unique values: {violations_with_variation[varying_col].max()}")
            # Show sample values for the first violation
            if len(violations_with_variation) > 0:
                sample_col = f'varying_{col}_sample'
                if sample_col in df.columns:
                    sample_val = violations_with_variation.iloc[0][sample_col]
                    if pd.notna(sample_val):
                        print(f"    Sample values: {str(sample_val)[:150]}")
if not found_metadata_variation:
    print("  No metadata columns (fold_idx, run_id, cv_type, split_level, etc.) show variation")
    print("  This suggests fold_idx might be missing from the data, or the variation")
    print("  is due to unaggregated fold results within the same session.")
    
# Check ALL varying columns to see what might explain the clean score differences
print("\n" + "=" * 80)
print("ALL VARYING COLUMNS (excluding performance metrics)")
print("=" * 80)
# Performance metrics that are expected to vary
performance_metrics = ['corrupted_score', 'corrupted_roc_auc', 'corrupted_accuracy', 'corrupted_f1',
                      'corrupted_precision', 'corrupted_recall', 'score', 'relative_drop',
                      'evaluation_time', 'total_time', 'training_time']
all_varying_cols = [c for c in df.columns if c.startswith('varying_') and c.endswith('_num_unique')]
non_perf_varying = []
for col in all_varying_cols:
    base_col = col.replace('varying_', '').replace('_num_unique', '')
    if base_col not in performance_metrics:
        non_perf_varying.append(col)

if non_perf_varying:
    print("Columns that vary (excluding performance metrics):")
    for col in sorted(non_perf_varying):
        base_col = col.replace('varying_', '').replace('_num_unique', '')
        violations_with_variation = df[df[col] > 1]
        if len(violations_with_variation) > 0:
            print(f"  {base_col}: {len(violations_with_variation)} violations ({len(violations_with_variation)/len(df)*100:.1f}%)")
            print(f"    Avg unique values: {violations_with_variation[col].mean():.1f}, Max: {violations_with_variation[col].max()}")
            # Show sample values for this column
            sample_col = f'varying_{base_col}_sample'
            if sample_col in df.columns:
                sample_vals = violations_with_variation[sample_col].dropna().iloc[0] if len(violations_with_variation) > 0 else None
                if sample_vals is not None:
                    print(f"    Sample values: {str(sample_vals)[:200]}")
else:
    print("  No non-performance columns vary (this is the root cause of the issue!)")
    
# Check if mode column variation explains the clean score differences
print("\n" + "=" * 80)
print("INVESTIGATING MODE COLUMN VARIATION")
print("=" * 80)
if 'varying_mode_num_unique' in df.columns:
    mode_violations = df[df['varying_mode_num_unique'] > 1]
    if len(mode_violations) > 0:
        print(f"Found {len(mode_violations)} violations where 'mode' column varies")
        print("This suggests different experimental modes are mixed together.")
        print("\nSample violation with mode variation:")
        sample = mode_violations.iloc[0]
        print(f"  Grouping: model={sample['model']}, dataset={sample['dataset']}, seed={sample['seed']}, subject={sample['subject']}, tune={sample['tune']}, session={sample['session']}, eval_mode={sample['eval_mode']}")
        if 'varying_mode_sample' in df.columns:
            mode_samples = sample['varying_mode_sample']
            if pd.notna(mode_samples):
                print(f"  Mode values: {mode_samples}")
        print(f"  Clean score range: {sample['score_range']}")
        print(f"  Max intra-noise variation: {sample['max_intra_noise_variation']:.6f}")
    else:
        print("Mode column does not vary in any violations.")
else:
    print("'varying_mode_num_unique' column not found in violation CSV.")
    
# Check the pattern of unique clean scores
print("\n" + "=" * 80)
print("PATTERN ANALYSIS: Unique clean scores per noise type")
print("=" * 80)
print("Checking if there's a consistent pattern (e.g., always 3 unique scores):")
for noise_type in ['eog', 'gaussian', 'dropout', 'spike']:
    num_unique_col = f'{noise_type}_num_unique'
    if num_unique_col in df.columns:
        unique_counts = df[num_unique_col].dropna().value_counts().sort_index()
        print(f"\n  {noise_type.upper()}:")
        for count, freq in unique_counts.items():
            print(f"    {int(count)} unique clean scores: {freq} violations ({freq/len(df)*100:.1f}%)")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Average max_intra_noise_variation: {df['max_intra_noise_variation'].mean():.6f}")
print(f"Max max_intra_noise_variation: {df['max_intra_noise_variation'].max():.6f}")
print(f"Min max_intra_noise_variation: {df['max_intra_noise_variation'].min():.6f}")
print(f"\nViolations with max_intra_noise_variation > 0.01: {len(df[df['max_intra_noise_variation'] > 0.01])}")
print(f"Violations with max_intra_noise_variation > 0.001: {len(df[df['max_intra_noise_variation'] > 0.001])}")

