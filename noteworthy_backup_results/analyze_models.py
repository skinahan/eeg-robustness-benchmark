#!/usr/bin/env python3
"""
Lightweight analysis script for comparing model performance under noise.

This script analyzes the comparative performance of multiple EEG classification models
under different noise conditions (EOG, Gaussian, Dropout).

Usage:
    python analyze_models.py [--plot] [--output OUTPUT_FILE]

Options:
    --plot              Generate visualization plots (requires matplotlib)
    --output FILE       Output markdown file path (default: analysis_summary.md)
    --seed SEED         Filter results by specific seed (default: all seeds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime


def find_result_files(base_dir, seed=None):
    """
    Recursively find all test_perturb CSV files in the directory structure.
    
    Args:
        base_dir: Base directory to search
        seed: Optional seed to filter results
        
    Returns:
        Dictionary mapping model names to their result file paths
    """
    base_path = Path(base_dir)
    result_files = {}
    
    # Search for CSV files in the expected structure
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Look for CrossSessionEvaluation results
        csv_files = list(model_dir.rglob("*test_perturb*.csv"))
        
        # Filter by seed if specified
        if seed is not None:
            csv_files = [f for f in csv_files if f"seed{seed}" in f.name]
        
        # Prefer 1test session over 0train
        test_files = [f for f in csv_files if "1test" in str(f)]
        if test_files:
            result_files[model_name] = test_files[0]
        elif csv_files:
            result_files[model_name] = csv_files[0]
    
    return result_files


def load_and_prepare_data(result_files):
    """
    Load CSV files and prepare combined dataframe.
    
    Args:
        result_files: Dictionary of model names to file paths
        
    Returns:
        Combined pandas DataFrame with all results
    """
    dfs = []
    
    for model_name, file_path in result_files.items():
        try:
            df = pd.read_csv(file_path)
            df['model_name'] = model_name
            dfs.append(df)
            print(f"✓ Loaded: {model_name} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}", file=sys.stderr)
    
    if not dfs:
        raise ValueError("No data loaded. Check directory structure.")
    
    return pd.concat(dfs, ignore_index=True)


def calculate_statistics(df):
    """
    Calculate key performance statistics for each model and noise type.
    
    Args:
        df: Combined dataframe with all results
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        stats[model] = {
            'clean_performance': model_data['clean_roc_auc'].iloc[0],
            'noise_types': {}
        }
        
        for noise_type in ['eog', 'gaussian', 'dropout']:
            noise_data = model_data[model_data['noise_type'] == noise_type].copy()
            if len(noise_data) == 0:
                continue
                
            noise_stats = {
                'mean_performance': noise_data['corrupted_roc_auc'].mean(),
                'std_performance': noise_data['corrupted_roc_auc'].std(),
                'min_performance': noise_data['corrupted_roc_auc'].min(),
                'max_performance': noise_data['corrupted_roc_auc'].max(),
            }
            
            # Performance at key intensities
            for intensity in [25, 50, 75, 100]:
                closest_idx = (noise_data['intensity'] - intensity).abs().idxmin()
                if abs(noise_data.loc[closest_idx, 'intensity'] - intensity) < 10:
                    clean = noise_data.loc[closest_idx, 'clean_roc_auc']
                    corrupted = noise_data.loc[closest_idx, 'corrupted_roc_auc']
                    retention = (corrupted / clean) * 100
                    noise_stats[f'performance_at_{intensity}'] = corrupted
                    noise_stats[f'retention_at_{intensity}'] = retention
            
            stats[model]['noise_types'][noise_type] = noise_stats
    
    return stats


def generate_markdown_report(df, stats, output_file):
    """
    Generate a markdown report with analysis results.
    
    Args:
        df: Combined dataframe
        stats: Statistics dictionary
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write(f"# Model Performance Analysis\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Model overview
        f.write("## Models Analyzed\n\n")
        models = sorted(df['model_name'].unique())
        for model in models:
            clean_perf = stats[model]['clean_performance']
            f.write(f"- **{model}**: Clean ROC-AUC = {clean_perf:.4f}\n")
        f.write("\n---\n\n")
        
        # Performance by noise type
        for noise_type in ['eog', 'gaussian', 'dropout']:
            f.write(f"## {noise_type.upper()} Noise Performance\n\n")
            
            # Create comparison table
            f.write("| Model | Clean | 25% | 50% | 75% | 100% | Avg Retention |\n")
            f.write("|-------|-------|-----|-----|-----|------|---------------|\n")
            
            for model in models:
                if noise_type not in stats[model]['noise_types']:
                    continue
                    
                noise_stats = stats[model]['noise_types'][noise_type]
                clean = stats[model]['clean_performance']
                
                row = [model, f"{clean:.3f}"]
                retentions = []
                
                for intensity in [25, 50, 75, 100]:
                    perf_key = f'performance_at_{intensity}'
                    ret_key = f'retention_at_{intensity}'
                    if perf_key in noise_stats:
                        perf = noise_stats[perf_key]
                        ret = noise_stats[ret_key]
                        row.append(f"{perf:.3f}")
                        retentions.append(ret)
                    else:
                        row.append("N/A")
                
                avg_retention = np.mean(retentions) if retentions else 0
                row.append(f"{avg_retention:.1f}%")
                
                f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n")
            
            # Best performer
            best_model = max(models, 
                           key=lambda m: stats[m]['noise_types'].get(noise_type, {}).get('mean_performance', 0))
            best_perf = stats[best_model]['noise_types'].get(noise_type, {}).get('mean_performance', 0)
            f.write(f"**Best Performer**: {best_model} (avg: {best_perf:.4f})\n\n")
            f.write("---\n\n")
        
        # Overall rankings
        f.write("## Overall Rankings\n\n")
        
        # By clean performance
        f.write("### By Clean Performance\n\n")
        ranked_by_clean = sorted(models, key=lambda m: stats[m]['clean_performance'], reverse=True)
        for i, model in enumerate(ranked_by_clean, 1):
            perf = stats[model]['clean_performance']
            f.write(f"{i}. **{model}**: {perf:.4f}\n")
        f.write("\n")
        
        # By average noise robustness
        f.write("### By Average Noise Robustness (50% intensity)\n\n")
        noise_robustness = {}
        for model in models:
            retentions = []
            for noise_type in ['eog', 'gaussian', 'dropout']:
                if noise_type in stats[model]['noise_types']:
                    ret_key = 'retention_at_50'
                    if ret_key in stats[model]['noise_types'][noise_type]:
                        retentions.append(stats[model]['noise_types'][noise_type][ret_key])
            noise_robustness[model] = np.mean(retentions) if retentions else 0
        
        ranked_by_robustness = sorted(models, key=lambda m: noise_robustness[m], reverse=True)
        for i, model in enumerate(ranked_by_robustness, 1):
            robustness = noise_robustness[model]
            f.write(f"{i}. **{model}**: {robustness:.1f}% average retention\n")
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        best_overall = ranked_by_clean[0]
        most_robust = ranked_by_robustness[0]
        
        f.write(f"- **Best Overall Performance**: {best_overall}\n")
        f.write(f"- **Most Noise Robust**: {most_robust}\n")
        
        if best_overall == most_robust:
            f.write(f"\n**{best_overall}** achieves both highest clean performance and best noise robustness. ")
            f.write("This is the recommended model for most applications.\n")
        else:
            f.write(f"\nChoose **{best_overall}** for clean or low-noise data.\n")
            f.write(f"Choose **{most_robust}** for noisy or challenging conditions.\n")
    
    print(f"\n✓ Report saved to: {output_file}")


def plot_results(df, output_dir):
    """
    Generate visualization plots (requires matplotlib).
    
    Args:
        df: Combined dataframe
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping plots.", file=sys.stderr)
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    models = sorted(df['model_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance Comparison Across Noise Types', fontsize=14, fontweight='bold')
    
    for idx, noise_type in enumerate(['eog', 'gaussian', 'dropout']):
        ax = axes[idx]
        df_noise = df[df['noise_type'] == noise_type]
        
        for model in models:
            df_model = df_noise[df_noise['model_name'] == model].sort_values('intensity')
            if len(df_model) > 0:
                ax.plot(df_model['intensity'], df_model['corrupted_roc_auc'],
                       label=model, linewidth=2.5, alpha=0.8,
                       color=model_colors[model], marker='o', markersize=3, markevery=5)
        
        ax.set_title(f'{noise_type.upper()} Noise', fontsize=12, fontweight='bold')
        ax.set_xlabel('Noise Intensity (%)', fontsize=10)
        ax.set_ylabel('ROC-AUC Score', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plot_file = output_dir / 'model_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved to: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance under noise')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    parser.add_argument('--output', default='analysis_summary.md', help='Output markdown file')
    parser.add_argument('--seed', type=int, help='Filter by specific seed')
    parser.add_argument('--dir', default='.', help='Base directory (default: current)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Find result files
    print("Searching for result files...")
    result_files = find_result_files(args.dir, seed=args.seed)
    
    if not result_files:
        print("Error: No result files found!", file=sys.stderr)
        print("Expected directory structure: model_name/CrossSessionEvaluation/.../test_perturb/*.csv")
        return 1
    
    print(f"Found {len(result_files)} model(s)\n")
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(result_files)
    print()
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(df)
    print()
    
    # Print summary to console
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        print(f"  Clean ROC-AUC: {model_stats['clean_performance']:.4f}")
        for noise_type, noise_stats in model_stats['noise_types'].items():
            print(f"  {noise_type.upper()}:")
            print(f"    Mean: {noise_stats['mean_performance']:.4f}")
            print(f"    Range: {noise_stats['min_performance']:.4f} - {noise_stats['max_performance']:.4f}")
    print()
    
    # Generate markdown report
    print("Generating report...")
    generate_markdown_report(df, stats, args.output)
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        plot_results(df, 'plots')
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

