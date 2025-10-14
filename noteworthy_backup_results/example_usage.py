#!/usr/bin/env python3
"""
Example usage of the analysis script as a module.

This demonstrates how to use the analysis functions programmatically
rather than from the command line.
"""

from analyze_models import (
    find_result_files,
    load_and_prepare_data,
    calculate_statistics,
    generate_markdown_report,
    plot_results
)
import pandas as pd


def main():
    print("Example: Using analyze_models as a module\n")
    
    # 1. Find result files (seed 42 only)
    print("Step 1: Finding result files...")
    result_files = find_result_files('.', seed=42)
    print(f"Found {len(result_files)} models: {list(result_files.keys())}\n")
    
    # 2. Load and combine data
    print("Step 2: Loading data...")
    df = load_and_prepare_data(result_files)
    print(f"Total rows: {len(df)}")
    print(f"Noise types: {df['noise_type'].unique().tolist()}\n")
    
    # 3. Calculate statistics
    print("Step 3: Calculating statistics...")
    stats = calculate_statistics(df)
    
    # 4. Print some specific findings
    print("\nSpecific Findings:")
    print("-" * 60)
    
    # Find best model for each noise type at 50% intensity
    for noise_type in ['eog', 'gaussian', 'dropout']:
        best_model = None
        best_retention = 0
        
        for model, model_stats in stats.items():
            if noise_type in model_stats['noise_types']:
                retention = model_stats['noise_types'][noise_type].get('retention_at_50', 0)
                if retention > best_retention:
                    best_retention = retention
                    best_model = model
        
        print(f"\nBest at {noise_type.upper()} (50% intensity):")
        print(f"  Model: {best_model}")
        print(f"  Retention: {best_retention:.1f}%")
    
    # Find model with highest clean performance
    print("\n" + "-" * 60)
    best_clean_model = max(stats.items(), key=lambda x: x[1]['clean_performance'])
    print(f"Highest Clean Performance:")
    print(f"  Model: {best_clean_model[0]}")
    print(f"  ROC-AUC: {best_clean_model[1]['clean_performance']:.4f}")
    
    # 5. Custom analysis: Calculate average retention across all noise types
    print("\n" + "-" * 60)
    print("Average Retention at 50% Intensity (all noise types):")
    for model, model_stats in stats.items():
        retentions = []
        for noise_type in ['eog', 'gaussian', 'dropout']:
            if noise_type in model_stats['noise_types']:
                ret = model_stats['noise_types'][noise_type].get('retention_at_50', None)
                if ret is not None:
                    retentions.append(ret)
        
        avg_retention = sum(retentions) / len(retentions) if retentions else 0
        print(f"  {model}: {avg_retention:.1f}%")
    
    # 6. Generate markdown report
    print("\n" + "=" * 60)
    print("Generating report...")
    generate_markdown_report(df, stats, 'example_analysis.md')
    
    # 7. Optional: Generate plots
    try:
        import matplotlib
        print("Generating plots...")
        plot_results(df, 'example_plots')
    except ImportError:
        print("Skipping plots (matplotlib not available)")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("Check 'example_analysis.md' and 'example_plots/' for outputs")
    

def custom_comparison():
    """
    Example of custom comparison: Find crossover points.
    
    This finds the intensity at which one model surpasses another.
    """
    print("\n\nCustom Analysis: Performance Crossover Points")
    print("=" * 60)
    
    result_files = find_result_files('.', seed=42)
    df = load_and_prepare_data(result_files)
    
    models = sorted(df['model_name'].unique())
    
    for noise_type in ['eog', 'gaussian', 'dropout']:
        print(f"\n{noise_type.upper()} Noise Crossovers:")
        df_noise = df[df['noise_type'] == noise_type]
        
        # Get data for each model
        model_data = {}
        for model in models:
            df_model = df_noise[df_noise['model_name'] == model].sort_values('intensity')
            model_data[model] = df_model
        
        # Compare pairs
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Find where model2 performance exceeds model1
                data1 = model_data[model1]
                data2 = model_data[model2]
                
                # Merge on intensity
                merged = pd.merge(
                    data1[['intensity', 'corrupted_roc_auc']], 
                    data2[['intensity', 'corrupted_roc_auc']],
                    on='intensity', 
                    suffixes=('_1', '_2')
                )
                
                # Find crossover
                merged['diff'] = merged['corrupted_roc_auc_2'] - merged['corrupted_roc_auc_1']
                sign_changes = merged['diff'].diff().fillna(0)
                crossovers = merged[sign_changes != 0]
                
                if len(crossovers) > 0:
                    for _, row in crossovers.iterrows():
                        if row['diff'] > 0:
                            print(f"  ~{row['intensity']:.0f}%: {model2} overtakes {model1}")


if __name__ == '__main__':
    main()
    custom_comparison()

