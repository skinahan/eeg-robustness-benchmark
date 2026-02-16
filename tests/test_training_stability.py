#!/usr/bin/env python3
"""
Test Training Stability Comparison

This script compares training stability between two models by manually training
them with fixed epochs (100) and no early stopping, then plotting training
loss curves side-by-side.

Models compared: cnn_ncp and branched_wiredcfc_arch4
Dataset: BNCI2014_001
Subject: 1
Evaluation Mode: CrossSession
Seed: 12217
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from config import get_model_registry, get_paradigm, get_dataset_sampling_rate
from globals import set_seeds
from moabb.datasets import BNCI2014_001
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import Callback


class ValidationLossTracker(Callback):
    """Callback to track validation loss during training."""
    
    def __init__(self, X_valid, y_valid):
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.valid_losses = []
        self.train_losses = []
        
    def on_epoch_end(self, net, **kwargs):
        """Record losses at end of each epoch."""
        # Get losses from history (skorch logs these automatically)
        if len(net.history) > 0:
            last_epoch = net.history[-1]
            
            # Get training loss
            train_loss = last_epoch.get('train_loss', None)
            if train_loss is not None:
                self.train_losses.append(train_loss)
            
            # Get validation loss from history (skorch logs this if train_split is used)
            valid_loss = last_epoch.get('valid_loss', None)
            if valid_loss is not None:
                self.valid_losses.append(valid_loss)
            else:
                # If not in history, compute it manually
                try:
                    net.module_.eval()
                    with torch.no_grad():
                        # Use predict_proba which handles data format correctly
                        y_pred_proba = net.predict_proba(self.X_valid)
                        
                        # Compute cross-entropy loss manually
                        criterion = torch.nn.CrossEntropyLoss()
                        # Convert probabilities to logits
                        eps = 1e-8
                        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
                        y_pred_logits = np.log(y_pred_proba + eps)
                        
                        y_pred_tensor = torch.from_numpy(y_pred_logits).float()
                        y_true_tensor = torch.from_numpy(self.y_valid).long()
                        valid_loss = criterion(y_pred_tensor, y_true_tensor).item()
                        self.valid_losses.append(valid_loss)
                    net.module_.train()
                except Exception as e:
                    # If computation fails, skip this epoch
                    print(f"Warning: Could not compute validation loss for epoch {len(net.history)}: {e}")


def train_model_manual(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                      X_valid: np.ndarray, y_valid: np.ndarray,
                      n_chans: int, n_times: int, n_outputs: int,
                      max_epochs: int = 100, seed: int = 12217) -> Dict[str, Any]:
    """
    Manually train a model with fixed epochs and no early stopping.
    
    Args:
        model_name: Name of the model to train
        X_train: Training data
        y_train: Training labels
        X_valid: Validation data
        y_valid: Validation labels
        n_chans: Number of channels
        n_times: Number of time points
        n_outputs: Number of output classes
        max_epochs: Maximum number of epochs (default: 100)
        seed: Random seed
        
    Returns:
        Dictionary with training history (epochs, train_losses, valid_losses)
    """
    # Set seed for reproducibility
    set_seeds(seed)
    
    # Get model factory from registry
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Model {model_name} not found in registry. Available models: {list(registry.keys())}")
    
    model_fn = registry[model_name]
    
    # Create model
    print(f"Creating {model_name} model...")
    model = model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
    
    # Set training parameters
    model.max_epochs = max_epochs
    model.optimizer.lr = 1e-2
    
    # Remove early stopping callbacks if any
    from skorch.callbacks import EarlyStopping
    # Remove from callbacks list
    if hasattr(model, 'callbacks') and model.callbacks:
        model.callbacks = [cb for cb in model.callbacks if not isinstance(cb, EarlyStopping)]
    # Remove from callbacks_ registry (list of tuples)
    if hasattr(model, 'callbacks_') and model.callbacks_:
        model.callbacks_ = [(name, cb) for name, cb in model.callbacks_ 
                           if not isinstance(cb, EarlyStopping)]
    
    # Add validation loss tracker
    valid_tracker = ValidationLossTracker(X_valid, y_valid)
    if not hasattr(model, 'callbacks_'):
        model.callbacks_ = []
    model.callbacks_.append(('valid_loss_tracker', valid_tracker))
    # if model_name == "branched_wiredcfc_arch4":
    #     model.module_.residual_init_strategy = "correct_rezero"
    model.initialize()
    # Train model
    print(f"Training {model_name} for {max_epochs} epochs (no early stopping)...")
    model.fit(X_train, y_train)
    
    # Extract training history from model history (most reliable)
    epochs = []
    train_losses = []
    valid_losses = []
    
    if hasattr(model, 'history') and model.history:
        for i, epoch_data in enumerate(model.history):
            epochs.append(i + 1)
            
            # Get training loss
            train_loss = epoch_data.get('train_loss', None)
            if train_loss is not None:
                train_losses.append(train_loss)
            
            # Get validation loss from history (preferred)
            valid_loss = epoch_data.get('valid_loss', None)
            if valid_loss is not None:
                valid_losses.append(valid_loss)
    
    # Fallback: Use callback tracker if history doesn't have valid_loss
    if not valid_losses and valid_tracker.valid_losses:
        valid_losses = valid_tracker.valid_losses
        # Ensure epochs match
        if not epochs:
            epochs = list(range(1, len(valid_losses) + 1))
    
    # Fallback: Use callback tracker for train losses if needed
    if not train_losses and valid_tracker.train_losses:
        train_losses = valid_tracker.train_losses
    
    # Ensure we have the same number of epochs for all
    if epochs:
        min_len = len(epochs)
        if valid_losses:
            min_len = min(min_len, len(valid_losses))
        if train_losses:
            min_len = min(min_len, len(train_losses))
        
        epochs = epochs[:min_len]
        if valid_losses:
            valid_losses = valid_losses[:min_len]
        if train_losses:
            train_losses = train_losses[:min_len]
    elif valid_losses:
        # If no epochs but we have valid_losses, create epochs
        epochs = list(range(1, len(valid_losses) + 1))
    
    # Final check: ensure we have data
    if not train_losses:
        raise ValueError(f"Could not extract training losses for {model_name}. "
                        f"History length: {len(model.history) if hasattr(model, 'history') else 0}, "
                        f"Tracker length: {len(valid_tracker.train_losses)}")
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'model_name': model_name
    }


def plot_training_loss_side_by_side(histories: Dict[str, Dict[str, Any]], 
                                      output_path: str, 
                                      title: str = "Training Stability Comparison"):
    """
    Plot training loss curves for multiple models side-by-side.
    
    Args:
        histories: Dictionary mapping model names to their training histories
        output_path: Path to save the PDF plot
        title: Plot title
    """
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(12, 5))
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # First pass: determine global y-axis limits for all models
    all_train_losses = []
    for history in histories.values():
        train_losses = history['train_losses']
        if train_losses:
            all_train_losses.extend(train_losses)
    
    if all_train_losses:
        y_min = min(all_train_losses)
        y_max = max(all_train_losses)
        # Add some padding (5% on each side)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        y_min = max(0, y_min - y_padding)  # Don't go below 0 for loss
        y_max = y_max + y_padding
    else:
        y_min, y_max = 0, 1  # Default fallback
    
    # Second pass: plot all models with same y-axis limits
    for idx, (model_name, history) in enumerate(histories.items()):
        ax = axes[idx]
        
        epochs = history['epochs']
        train_losses = history['train_losses']
        
        # Plot training loss
        ax.plot(epochs, train_losses, 
               label='Training Loss', 
               linewidth=2, 
               color=colors[idx % len(colors)],
               marker='o', 
               markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.set_ylim(y_min, y_max)  # Set same y-axis limits for all subplots
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        # Add metadata
        d = pdf.infodict()
        d['Title'] = title
        d['Author'] = 'Training Stability Comparison Script'
    
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare training stability between models")
    parser.add_argument("--output", type=str, default="training_stability_comparison.pdf",
                       help="Output PDF file path (default: training_stability_comparison.pdf)")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--seed", type=int, default=12217,
                       help="Random seed (default: 12217)")
    
    args = parser.parse_args()
    
    # Experiment parameters
    models = ["cnn_ncp", "branched_wiredcfc_arch4"]
    dataset_name = "BNCI2014_001"
    subject = 1
    eval_mode = "CrossSession"
    seed = args.seed
    max_epochs = args.max_epochs
    
    print("=" * 80)
    print("Training Stability Comparison")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Dataset: {dataset_name}")
    print(f"Subject: {subject}")
    print(f"Evaluation Mode: {eval_mode}")
    print(f"Seed: {seed}")
    print(f"Max Epochs: {max_epochs} (no early stopping)")
    print("=" * 80)
    
    # Set seed
    set_seeds(seed)
    
    # Initialize dataset and paradigm
    print("\nLoading dataset...")
    dataset_obj = BNCI2014_001()
    paradigm = get_paradigm(resample=None, dataset=dataset_name)
    
    # Get data for subject
    print(f"Loading data for subject {subject}...")
    X, y, metadata = paradigm.get_data(dataset_obj, subjects=[subject])
    y_encoded = LabelEncoder().fit_transform(y)
    
    # For CrossSession evaluation, split by session
    # Train on '0train', validate on '1test'
    if 'session' not in metadata.columns:
        raise ValueError("Metadata does not contain 'session' column")
    
    train_mask = metadata['session'] == '0train'
    valid_mask = metadata['session'] == '1test'
    
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError(f"Could not find both training and validation sessions. "
                        f"Found sessions: {metadata['session'].unique()}")
    
    X_train = X[train_mask]
    y_train = y_encoded[train_mask]
    X_valid = X[valid_mask]
    y_valid = y_encoded[valid_mask]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    
    # Determine data dimensions
    if len(X_train.shape) == 3:
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]
    else:
        raise ValueError(f"Unexpected data shape: {X_train.shape}")
    
    # Determine number of output classes
    n_outputs = len(np.unique(y_encoded))
    
    print(f"Data dimensions: {n_chans} channels, {n_times} time points")
    print(f"Number of classes: {n_outputs}")
    
    # Train each model
    print("\n" + "=" * 80)
    print("Training Models")
    print("=" * 80)
    
    histories = {}
    
    for model_name in models:
        print(f"\n{'=' * 80}")
        print(f"Training {model_name}")
        print(f"{'=' * 80}")
        
        try:

            history = train_model_manual(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
                max_epochs=max_epochs,
                seed=seed
            )
            
            histories[model_name] = history
            print(f"[OK] {model_name} training completed")
            print(f"  Trained for {len(history['epochs'])} epochs")
            if history['train_losses']:
                print(f"  Final training loss: {history['train_losses'][-1]:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Create comparison plot
    print("\n" + "=" * 80)
    print("Creating Comparison Plot")
    print("=" * 80)
    
    title = f"Training Loss Comparison\n" \
            f"Dataset: {dataset_name}, Subject: {subject}, Seed: {seed}, Epochs: {max_epochs}"
    
    plot_training_loss_side_by_side(histories, args.output, title=title)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print(f"Plot saved to: {args.output}")
    print("=" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    for model_name, history in histories.items():
        train_losses = history['train_losses']
        print(f"\n{model_name}:")
        print(f"  Initial training loss: {train_losses[0]:.4f}")
        print(f"  Final training loss: {train_losses[-1]:.4f}")
        print(f"  Minimum training loss: {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses)) + 1})")
        print(f"  Loss reduction: {train_losses[0] - train_losses[-1]:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
