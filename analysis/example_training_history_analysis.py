#!/usr/bin/env python3
"""
Example: Training History Analysis

This script demonstrates how to quickly analyze training history
from your experiments to detect overfitting and underfitting.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_training_history import (
    load_history,
    plot_loss_curves,
    plot_accuracy_curves,
    detect_overfitting,
    detect_underfitting,
    analyze_early_stopping,
    compare_histories
)


def example_single_file_analysis():
    """Example: Analyze a single training history file."""
    print("="*80)
    print("Example 1: Single File Analysis")
    print("="*80)
    
    # Path to a training history file
    # Replace with actual path from your experiments
    history_file = "results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/sub-001/0train/training_history/history_sub001_sess0train_fold0_test_perturb.json"
    
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        print("Please run an experiment first to generate history files.")
        return
    
    # Load history
    history = load_history(history_file)
    print(f"\nLoaded history with {len(history)} epochs")
    
    # Create plots
    plot_loss_curves(history, title="Training Loss Curves", save_path="outputs/plots/example_loss.png")
    plot_accuracy_curves(history, title="Training Accuracy Curves", save_path="outputs/plots/example_accuracy.png")
    
    # Detect overfitting
    overfit_result = detect_overfitting(history, threshold=0.1)
    print(f"\nOverfitting Detection:")
    print(f"  {overfit_result['message']}")
    if overfit_result['detected']:
        print(f"  Train Loss: {overfit_result['train_loss']:.4f}")
        print(f"  Valid Loss: {overfit_result['valid_loss']:.4f}")
        print(f"  Gap: {overfit_result['gap']:.4f}")
    
    # Detect underfitting
    underfit_result = detect_underfitting(history, window=5)
    print(f"\nUnderfitting Detection:")
    print(f"  {underfit_result['message']}")
    if underfit_result.get('detected'):
        print(f"  Slope: {underfit_result.get('slope', 0):.6f}")
    
    # Analyze early stopping
    early_stop_result = analyze_early_stopping(history)
    if early_stop_result['analyzed']:
        print(f"\nEarly Stopping Analysis:")
        print(f"  {early_stop_result['message']}")
        print(f"  Stopped at epoch: {early_stop_result['final_epoch']}")
        print(f"  Epochs after best: {early_stop_result['epochs_after_best']}")


def example_compare_multiple_runs():
    """Example: Compare training histories from multiple runs."""
    print("\n" + "="*80)
    print("Example 2: Compare Multiple Training Runs")
    print("="*80)
    
    # Paths to multiple history files (e.g., different folds or configurations)
    history_dir = "results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/sub-001/0train/training_history/"
    
    if not os.path.exists(history_dir):
        print(f"History directory not found: {history_dir}")
        print("Please run experiments first to generate history files.")
        return
    
    # Get all history files
    import glob
    history_files = glob.glob(os.path.join(history_dir, "*.json"))
    
    if not history_files:
        print(f"No history files found in {history_dir}")
        return
    
    print(f"\nFound {len(history_files)} history files")
    
    # Compare all histories
    compare_histories(history_files, save_dir="outputs/plots")
    
    # Analyze each file
    for filepath in history_files[:3]:  # Limit to first 3 for brevity
        print(f"\n{os.path.basename(filepath)}:")
        history = load_history(filepath)
        
        overfit = detect_overfitting(history)
        print(f"  {overfit['message']}")
        
        underfit = detect_underfitting(history)
        print(f"  {underfit['message']}")


def example_batch_analysis():
    """Example: Batch analyze all histories for a subject."""
    print("\n" + "="*80)
    print("Example 3: Batch Analysis for Subject")
    print("="*80)
    
    # Path to subject's results
    subject_dir = "results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/sub-001/"
    
    if not os.path.exists(subject_dir):
        print(f"Subject directory not found: {subject_dir}")
        return
    
    # Find all history files recursively
    import glob
    history_files = glob.glob(os.path.join(subject_dir, "**/training_history/*.json"), recursive=True)
    
    if not history_files:
        print(f"No history files found in {subject_dir}")
        return
    
    print(f"\nFound {len(history_files)} history files for subject")
    
    # Summary statistics
    overfitting_count = 0
    underfitting_count = 0
    
    for filepath in history_files:
        history = load_history(filepath)
        
        overfit = detect_overfitting(history)
        underfit = detect_underfitting(history)
        
        if overfit.get('detected'):
            overfitting_count += 1
        if underfit.get('detected'):
            underfitting_count += 1
    
    print(f"\nSummary:")
    print(f"  Total runs: {len(history_files)}")
    print(f"  Overfitting detected: {overfitting_count} ({100*overfitting_count/len(history_files):.1f}%)")
    print(f"  Underfitting detected: {underfitting_count} ({100*underfitting_count/len(history_files):.1f}%)")


def example_custom_analysis():
    """Example: Custom analysis of training dynamics."""
    print("\n" + "="*80)
    print("Example 4: Custom Training Dynamics Analysis")
    print("="*80)
    
    history_file = "results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/sub-001/0train/training_history/history_sub001_sess0train_fold0_test_perturb.json"
    
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return
    
    history = load_history(history_file)
    
    # Custom analysis: Find learning rate schedule effects
    print(f"\nCustom Analysis: Learning Dynamics")
    
    # Calculate improvement rate per epoch
    if len(history) > 1:
        valid_loss_key = None
        for key in history[0].keys():
            if 'valid' in key.lower() and 'loss' in key.lower():
                valid_loss_key = key
                break
        
        if valid_loss_key:
            improvements = []
            for i in range(1, len(history)):
                prev_loss = history[i-1][valid_loss_key]
                curr_loss = history[i][valid_loss_key]
                improvement = prev_loss - curr_loss
                improvements.append(improvement)
            
            print(f"  Total epochs: {len(history)}")
            print(f"  Initial loss: {history[0][valid_loss_key]:.4f}")
            print(f"  Final loss: {history[-1][valid_loss_key]:.4f}")
            print(f"  Total improvement: {history[0][valid_loss_key] - history[-1][valid_loss_key]:.4f}")
            print(f"  Avg improvement per epoch: {sum(improvements)/len(improvements):.6f}")
            print(f"  Best single-epoch improvement: {max(improvements):.6f}")
            print(f"  Worst single-epoch change: {min(improvements):.6f}")
            
            # Count epochs with negative progress (loss increased)
            negative_epochs = sum(1 for imp in improvements if imp < 0)
            print(f"  Epochs with negative progress: {negative_epochs} ({100*negative_epochs/len(improvements):.1f}%)")


if __name__ == "__main__":
    print("Training History Analysis Examples")
    print("This script demonstrates various ways to analyze training history.")
    print("\nNote: Make sure you have run experiments first to generate history files.")
    print("=" * 80)
    
    # Run examples
    try:
        example_single_file_analysis()
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    try:
        example_compare_multiple_runs()
    except Exception as e:
        print(f"Error in Example 2: {e}")
    
    try:
        example_batch_analysis()
    except Exception as e:
        print(f"Error in Example 3: {e}")
    
    try:
        example_custom_analysis()
    except Exception as e:
        print(f"Error in Example 4: {e}")
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)

