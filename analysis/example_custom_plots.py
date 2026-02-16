"""
Example usage of custom plotting and data extraction functions.

This script demonstrates how to use the flexible plot_custom_comparison() and
extract_custom_data() functions to create specific comparisons from the results data.
"""

import pandas as pd
from analyze_results import plot_custom_comparison, extract_custom_data

# Load the results data
df = pd.read_csv('../sol_results/all_results.csv')

print("Available columns:", df.columns.tolist())
print("\nUnique models:", df['model'].unique())
print("Unique noise types:", df['noise_type'].unique())
print("Unique eval modes:", df['eval_mode'].unique())

# Example 1: Compare eegnet and cnn_ncp under EOG noise with CrossSession evaluation
print("\n=== Example 1: EEGNet vs CNN_NCP under EOG noise (CrossSession) ===")
filtered_data, plot_path = plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True
    },
    hue_var='model',
    style_var='session',
    title='EEGNet vs CNN_NCP: EOG Noise (CrossSession, Tuned)',
    output_filename='example1_eegnet_vs_cnn_ncp_eog.png'
)
print(f"Data shape: {filtered_data.shape if filtered_data is not None else 'No data'}")

# Example 2: Compare all noise types for a single model
print("\n=== Example 2: All noise types for EEGNet (CrossSession) ===")
plot_custom_comparison(
    df,
    filters={
        'model': 'eegnet',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True
    },
    hue_var='noise_type',
    style_var='session',
    title='EEGNet: Performance across Noise Types (CrossSession, Tuned)',
    output_filename='example2_eegnet_all_noise_types.png'
)

# Example 3: Compare tuned vs baseline for a specific model and noise
print("\n=== Example 3: Tuned vs Baseline for CNN_NCP under Gaussian noise ===")
plot_custom_comparison(
    df,
    filters={
        'model': 'cnn_ncp',
        'noise_type': 'gaussian',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'session': '1test'  # Only test session
    },
    hue_var='tune',
    title='CNN_NCP: Tuned vs Baseline (Gaussian Noise, CrossSession, Test)',
    output_filename='example3_cnn_ncp_tuned_vs_baseline.png'
)

# Example 4: Low intensity only (custom filter with lambda)
print("\n=== Example 4: Low intensity comparison (intensity <= 10) ===")
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'reegnet', 'cnn_ncp'],
        'noise_type': 'dropout',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True,
        'intensity': lambda x: x <= 10.0  # Custom filter
    },
    hue_var='model',
    title='Low Intensity Dropout: Model Comparison (intensity <= 10%)',
    output_filename='example4_low_intensity_dropout.png'
)

# Example 5: Faceted plot by noise type
print("\n=== Example 5: Faceted plot by noise type ===")
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True,
        'session': '1test'
    },
    hue_var='model',
    col_var='noise_type',  # Create separate plots for each noise type
    title='Model Comparison Across Noise Types',
    output_filename='example5_faceted_by_noise.png'
)

# Example 6: Box plot for variability analysis
print("\n=== Example 6: Box plot showing variability ===")
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True,
        'intensity': lambda x: x in [1.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # Selected intensities
    },
    x_var='intensity',
    y_var='corrupted_score',
    hue_var='model',
    plot_type='box',
    title='Performance Variability: Box Plot',
    output_filename='example6_box_plot_variability.png'
)

# Example 7: Extract data without plotting
print("\n=== Example 7: Extract data without plotting ===")
data = extract_custom_data(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb'
    },
    columns=['model', 'noise_type', 'intensity', 'corrupted_score', 'session', 'subject']
)
print(f"Extracted data shape: {data.shape}")
print("\nFirst few rows:")
print(data.head(10))

# Example 8: Extract aggregated statistics
print("\n=== Example 8: Extract mean scores by model and intensity ===")
agg_data = extract_custom_data(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True
    },
    aggregate='mean',
    group_by=['model', 'intensity']
)
print(agg_data.head(20))

# Example 9: Compare specific subjects
print("\n=== Example 9: Compare specific subjects ===")
plot_custom_comparison(
    df,
    filters={
        'model': 'eegnet',
        'noise_type': 'gaussian',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True,
        'subject': [1, 2, 3]  # First three subjects only
    },
    hue_var='subject',
    style_var='session',
    title='EEGNet: Per-Subject Comparison (Subjects 1-3)',
    output_filename='example9_per_subject_comparison.png'
)

# Example 10: Multiple metrics comparison
print("\n=== Example 10: Compare different metrics ===")
for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
    print(f"\nPlotting {metric}...")
    plot_custom_comparison(
        df,
        filters={
            'model': ['eegnet', 'cnn_ncp'],
            'noise_type': 'gaussian',
            'eval_mode': 'CrossSession',
            'mode': 'test_perturb',
            'tune': True,
            'session': '1test'
        },
        hue_var='model',
        metric=metric,
        title=f'Model Comparison: {metric.upper().replace("_", " ")}',
        output_filename=f'example10_comparison_{metric}.png'
    )

print("\n=== All examples completed! Check the plots directory for output. ===")


