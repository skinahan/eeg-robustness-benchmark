#!/usr/bin/env python3
"""
Analyze Training History

This script demonstrates how to load and analyze training history logs
to detect overfitting, underfitting, and training dynamics.
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_history(filepath: str) -> List[Dict[str, Any]]:
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def plot_loss_curves(history: List[Dict[str, Any]], title: str = "Training History", save_path: str = None):
    """
    Plot training and validation loss curves.
    
    Args:
        history: List of epoch dictionaries
        title: Plot title
        save_path: Optional path to save figure
    """
    epochs = [h['epoch'] for h in history]
    
    # Extract available metrics
    metrics = {}
    for key in history[0].keys():
        if key != 'epoch' and 'loss' in key.lower():
            metrics[key] = [h.get(key, np.nan) for h in history]
    
    if not metrics:
        print("No loss metrics found in history")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        # Clean up metric name for legend
        label = metric_name.replace('_', ' ').title()
        ax.plot(epochs, values, marker='o', label=label, linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_accuracy_curves(history: List[Dict[str, Any]], title: str = "Accuracy History", save_path: str = None):
    """
    Plot training and validation accuracy curves.
    
    Args:
        history: List of epoch dictionaries
        title: Plot title
        save_path: Optional path to save figure
    """
    epochs = [h['epoch'] for h in history]
    
    # Extract available metrics
    metrics = {}
    for key in history[0].keys():
        if key != 'epoch' and 'acc' in key.lower():
            metrics[key] = [h.get(key, np.nan) for h in history]
    
    if not metrics:
        print("No accuracy metrics found in history")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        # Clean up metric name for legend
        label = metric_name.replace('_', ' ').title()
        ax.plot(epochs, values, marker='o', label=label, linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def detect_overfitting(history: List[Dict[str, Any]], threshold: float = 0.1) -> Dict[str, Any]:
    """
    Detect potential overfitting by analyzing train/validation gap.
    
    Args:
        history: List of epoch dictionaries
        threshold: Loss gap threshold for overfitting detection
        
    Returns:
        Dictionary with overfitting analysis results
    """
    final_epoch = history[-1]
    
    # Try to find train and validation losses
    train_loss_key = None
    valid_loss_key = None
    
    for key in final_epoch.keys():
        if 'train' in key.lower() and 'loss' in key.lower():
            train_loss_key = key
        elif 'valid' in key.lower() and 'loss' in key.lower():
            valid_loss_key = key
    
    if not train_loss_key or not valid_loss_key:
        return {'detected': False, 'reason': 'Missing train or validation loss'}
    
    train_loss = final_epoch[train_loss_key]
    valid_loss = final_epoch[valid_loss_key]
    loss_gap = valid_loss - train_loss
    
    detected = loss_gap > threshold
    
    return {
        'detected': detected,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'gap': loss_gap,
        'threshold': threshold,
        'message': f"Overfitting detected (gap: {loss_gap:.4f})" if detected else "No overfitting detected"
    }


def detect_underfitting(history: List[Dict[str, Any]], window: int = 5) -> Dict[str, Any]:
    """
    Detect potential underfitting by checking if validation loss is still decreasing.
    
    Args:
        history: List of epoch dictionaries
        window: Number of recent epochs to analyze
        
    Returns:
        Dictionary with underfitting analysis results
    """
    if len(history) < window:
        return {'detected': False, 'reason': 'Not enough epochs to analyze'}
    
    # Find validation loss key
    valid_loss_key = None
    for key in history[0].keys():
        if 'valid' in key.lower() and 'loss' in key.lower():
            valid_loss_key = key
            break
    
    if not valid_loss_key:
        return {'detected': False, 'reason': 'Missing validation loss'}
    
    last_epochs = history[-window:]
    valid_losses = [h[valid_loss_key] for h in last_epochs]
    
    # Check if there's a decreasing trend
    # Simple approach: compare first and last
    is_decreasing = valid_losses[-1] < valid_losses[0]
    
    # More sophisticated: check slope
    epochs_idx = list(range(len(valid_losses)))
    slope = np.polyfit(epochs_idx, valid_losses, 1)[0]
    
    detected = is_decreasing or slope < -0.001
    
    return {
        'detected': detected,
        'initial_loss': valid_losses[0],
        'final_loss': valid_losses[-1],
        'slope': slope,
        'message': f"Underfitting detected (still improving, slope: {slope:.6f})" if detected else "No underfitting detected"
    }


def analyze_early_stopping(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze if early stopping was triggered and find best epoch.
    
    Args:
        history: List of epoch dictionaries
        
    Returns:
        Dictionary with early stopping analysis
    """
    # Find validation loss key
    valid_loss_key = None
    for key in history[0].keys():
        if 'valid' in key.lower() and 'loss' in key.lower():
            valid_loss_key = key
            break
    
    if not valid_loss_key:
        return {'analyzed': False, 'reason': 'Missing validation loss'}
    
    valid_losses = [h[valid_loss_key] for h in history]
    best_epoch = np.argmin(valid_losses) + 1
    best_loss = min(valid_losses)
    final_epoch = len(history)
    
    early_stopped = best_epoch < final_epoch - 1
    
    return {
        'analyzed': True,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'final_epoch': final_epoch,
        'early_stopped': early_stopped,
        'epochs_after_best': final_epoch - best_epoch,
        'message': f"Best epoch: {best_epoch}/{final_epoch} (loss: {best_loss:.4f})"
    }


def compare_histories(history_files: List[str], save_dir: str = None):
    """
    Compare multiple training histories side-by-side.
    
    Args:
        history_files: List of history file paths
        save_dir: Optional directory to save comparison plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for filepath in history_files:
        history = load_history(filepath)
        filename = os.path.basename(filepath)
        
        epochs = [h['epoch'] for h in history]
        
        # Find loss and accuracy keys
        train_loss_key = valid_loss_key = None
        train_acc_key = valid_acc_key = None
        
        for key in history[0].keys():
            if 'train' in key.lower() and 'loss' in key.lower():
                train_loss_key = key
            elif 'valid' in key.lower() and 'loss' in key.lower():
                valid_loss_key = key
            elif 'train' in key.lower() and 'acc' in key.lower():
                train_acc_key = key
            elif 'valid' in key.lower() and 'acc' in key.lower():
                valid_acc_key = key
        
        # Plot losses
        if valid_loss_key:
            valid_losses = [h[valid_loss_key] for h in history]
            axes[0].plot(epochs, valid_losses, marker='o', label=filename, linewidth=2, markersize=4)
        
        # Plot accuracies
        if valid_acc_key:
            valid_accs = [h[valid_acc_key] for h in history]
            axes[1].plot(epochs, valid_accs, marker='o', label=filename, linewidth=2, markersize=4)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'history_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze training history logs")
    parser.add_argument("--history_file", type=str, help="Path to a single history JSON file")
    parser.add_argument("--history_dir", type=str, help="Directory containing multiple history files")
    parser.add_argument("--compare", action="store_true", help="Compare multiple histories")
    parser.add_argument("--output_dir", type=str, default="outputs/plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.history_file:
        # Analyze single file
        print(f"Analyzing: {args.history_file}")
        history = load_history(args.history_file)
        
        print(f"\nTraining Duration: {len(history)} epochs")
        
        # Plot curves
        base_name = os.path.splitext(os.path.basename(args.history_file))[0]
        plot_loss_curves(
            history, 
            title=f"Loss Curves - {base_name}",
            save_path=os.path.join(args.output_dir, f"{base_name}_loss.png")
        )
        plot_accuracy_curves(
            history,
            title=f"Accuracy Curves - {base_name}",
            save_path=os.path.join(args.output_dir, f"{base_name}_accuracy.png")
        )
        
        # Detect issues
        overfit_result = detect_overfitting(history)
        print(f"\n{overfit_result['message']}")
        if overfit_result['detected']:
            print(f"  Train Loss: {overfit_result['train_loss']:.4f}")
            print(f"  Valid Loss: {overfit_result['valid_loss']:.4f}")
            print(f"  Gap: {overfit_result['gap']:.4f}")
        
        underfit_result = detect_underfitting(history)
        print(f"\n{underfit_result['message']}")
        if underfit_result['detected']:
            print(f"  Initial Loss: {underfit_result['initial_loss']:.4f}")
            print(f"  Final Loss: {underfit_result['final_loss']:.4f}")
            print(f"  Slope: {underfit_result['slope']:.6f}")
        
        early_stop_result = analyze_early_stopping(history)
        if early_stop_result['analyzed']:
            print(f"\n{early_stop_result['message']}")
            if early_stop_result['early_stopped']:
                print(f"  Early stopping may have been triggered")
                print(f"  Epochs after best: {early_stop_result['epochs_after_best']}")
    
    elif args.history_dir:
        # Find all history files
        history_files = glob.glob(os.path.join(args.history_dir, "*.json"))
        
        if not history_files:
            print(f"No history files found in {args.history_dir}")
            return
        
        print(f"Found {len(history_files)} history files")
        
        if args.compare:
            # Compare all files
            print("Comparing histories...")
            compare_histories(history_files, save_dir=args.output_dir)
        else:
            # Analyze each file individually
            for filepath in history_files:
                print(f"\n{'='*80}")
                print(f"Analyzing: {os.path.basename(filepath)}")
                print('='*80)
                
                history = load_history(filepath)
                print(f"Training Duration: {len(history)} epochs")
                
                # Quick analysis
                overfit_result = detect_overfitting(history)
                print(f"  {overfit_result['message']}")
                
                underfit_result = detect_underfitting(history)
                print(f"  {underfit_result['message']}")
                
                early_stop_result = analyze_early_stopping(history)
                if early_stop_result['analyzed']:
                    print(f"  {early_stop_result['message']}")
    else:
        print("Please specify either --history_file or --history_dir")
        parser.print_help()


if __name__ == "__main__":
    main()

