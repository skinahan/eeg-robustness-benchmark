"""
Chunked Subject Trainer for Memory-Efficient Training

This module implements memory-efficient training and evaluation by loading
subjects in small chunks, avoiding the need to load all subjects into memory
simultaneously. This is particularly useful for CrossSubject evaluation with
large numbers of subjects.
"""

import numpy as np
import torch
import gc
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder
from braindecode import EEGClassifier


def train_with_subject_chunks(
    model: EEGClassifier,
    paradigm,
    dataset_obj,
    train_subjects: List[int],
    chunk_size: int = 3,
    max_epochs_per_chunk: Optional[int] = None,
    verbose: bool = True
) -> EEGClassifier:
    """
    Train a model incrementally by loading training subjects in chunks.
    
    This function loads subjects in small groups, trains the model on each
    chunk, then continues training on the next chunk. This avoids loading
    all training subjects into memory at once.
    
    Args:
        model: Initialized EEGClassifier model
        paradigm: MOABB paradigm object (e.g., MotorImagery)
        dataset_obj: MOABB dataset object (e.g., BNCI2014_001())
        train_subjects: List of subject IDs to use for training
        chunk_size: Number of subjects to load per chunk (default: 3)
        max_epochs_per_chunk: Maximum epochs to train on each chunk.
                             If None, uses model's max_epochs setting.
        verbose: Whether to print progress information
    
    Returns:
        Trained model (same object, modified in place)
    """
    if max_epochs_per_chunk is None:
        max_epochs_per_chunk = model.max_epochs
    
    # Split subjects into chunks
    subject_chunks = [
        train_subjects[i:i + chunk_size]
        for i in range(0, len(train_subjects), chunk_size)
    ]
    
    if verbose:
        print(f"[CHUNKED_TRAINING] Training on {len(train_subjects)} subjects in {len(subject_chunks)} chunks (chunk_size={chunk_size})")
    
    # Store original max_epochs and warm_start to restore later
    original_max_epochs = model.max_epochs
    original_warm_start = getattr(model, 'warm_start', False)
    
    # CRITICAL: Ensure model is initialized before first fit call
    # This sets up the module, optimizer, and other internal state
    # In skorch, initialized state is checked via initialized_ attribute
    # But fit() will initialize automatically if not already done, so we can optionally
    # initialize here explicitly for clarity, or let fit() handle it
    # We'll check if module_ exists (only created after initialization)
    if not hasattr(model, 'module_') or model.module_ is None:
        if verbose:
            print(f"[CHUNKED_TRAINING] Model not yet initialized, fit() will initialize automatically")
    
    # Track if this is the first chunk (for initialization)
    first_chunk = True
    
    # Set warm_start=True after first chunk to preserve model state
    # Note: We'll set this after the first chunk is processed
    
    for chunk_idx, subject_chunk in enumerate(subject_chunks):
        if verbose:
            print(f"[CHUNKED_TRAINING] Loading chunk {chunk_idx + 1}/{len(subject_chunks)}: subjects {subject_chunk}")
        
        # Load chunk of subjects
        gc.collect()
        X_chunk, y_chunk, _ = paradigm.get_data(
            dataset_obj, subjects=subject_chunk
        )
        
        # Convert to float32 immediately to save memory
        if X_chunk.dtype == np.float64:
            if verbose:
                data_size_mb_before = X_chunk.nbytes / 1024 / 1024
                X_chunk = X_chunk.astype(np.float32)
                data_size_mb_after = X_chunk.nbytes / 1024 / 1024
                print(f"[CHUNKED_TRAINING] Converted chunk from float64 to float32: {data_size_mb_before:.2f} MB -> {data_size_mb_after:.2f} MB")
            else:
                X_chunk = X_chunk.astype(np.float32)
        
        # Encode labels if needed
        if isinstance(y_chunk[0], str):
            if first_chunk:
                label_encoder = LabelEncoder()
                y_chunk = label_encoder.fit_transform(y_chunk)
                # Store encoder for subsequent chunks
                model._label_encoder = label_encoder
            else:
                # Use the encoder from first chunk
                y_chunk = model._label_encoder.transform(y_chunk)
        
        if verbose:
            chunk_size_mb = X_chunk.nbytes / 1024 / 1024
            print(f"[CHUNKED_TRAINING] Chunk loaded: shape={X_chunk.shape}, size={chunk_size_mb:.2f} MB")
        
        # CRITICAL: For subsequent chunks, we MUST use warm_start=True to preserve
        # model weights and optimizer state. Without this, skorch will reinitialize
        # the model parameters on each fit() call.
        if first_chunk:
            # First chunk: train with warm_start=False (default, model is fresh)
            # Note: warm_start can stay False for first chunk since model is newly initialized
            model.module_.train()
            if verbose:
                print(f"[CHUNKED_TRAINING] Training on chunk {chunk_idx + 1} for up to {max_epochs_per_chunk} epochs...")
            
            # Temporarily adjust max_epochs for this chunk
            model.max_epochs = max_epochs_per_chunk
            model.fit(X_chunk, y_chunk)
            
            # Enable warm_start for subsequent chunks
            model.warm_start = True
            first_chunk = False
            
            if verbose:
                print(f"[CHUNKED_TRAINING] Enabled warm_start=True for subsequent chunks")
        else:
            # Subsequent chunks: continue training with warm_start=True
            # This preserves:
            # 1. Model weights (parameters)
            # 2. Optimizer state (momentum, Adam statistics, etc.)
            # 3. Callback states (if applicable)
            
            if verbose:
                print(f"[CHUNKED_TRAINING] Continuing training on chunk {chunk_idx + 1} with warm_start=True...")
                print(f"[CHUNKED_TRAINING] Model state preserved from previous chunks")
            
            # Verify warm_start is enabled
            if not getattr(model, 'warm_start', False):
                raise RuntimeError(
                    "warm_start must be True for subsequent chunks. "
                    "This ensures model weights and optimizer state are preserved."
                )
            
            # For subsequent chunks, we can train with fewer epochs (fine-tuning approach)
            # or same epochs depending on strategy
            epochs_for_chunk = max_epochs_per_chunk // 2 if max_epochs_per_chunk > 1 else 1
            model.max_epochs = epochs_for_chunk
            
            if verbose:
                print(f"[CHUNKED_TRAINING] Training for up to {epochs_for_chunk} epochs on this chunk...")
            
            # Optionally reduce learning rate for fine-tuning (helps avoid catastrophic forgetting)
            # Access optimizer via optimizer_ (with underscore) after initialization in skorch
            if hasattr(model, 'optimizer_') and model.optimizer_ is not None:
                original_lr = model.optimizer_.param_groups[0]['lr']
                model.optimizer_.param_groups[0]['lr'] = original_lr * 0.5
                
                if verbose:
                    print(f"[CHUNKED_TRAINING] Temporarily reduced learning rate: {original_lr:.6f} -> {original_lr * 0.5:.6f}")
                
                model.fit(X_chunk, y_chunk)
                
                # Restore learning rate
                model.optimizer_.param_groups[0]['lr'] = original_lr
                
                if verbose:
                    print(f"[CHUNKED_TRAINING] Restored learning rate: {original_lr:.6f}")
            else:
                # Optimizer not yet initialized (shouldn't happen after first chunk, but handle gracefully)
                if verbose:
                    print(f"[CHUNKED_TRAINING] Warning: optimizer not found, training without LR adjustment")
                model.fit(X_chunk, y_chunk)
        
        # Clean up chunk data
        del X_chunk, y_chunk
        gc.collect()
        
        if verbose:
            print(f"[CHUNKED_TRAINING] Completed chunk {chunk_idx + 1}/{len(subject_chunks)}")
    
    # Restore original max_epochs and warm_start
    model.max_epochs = original_max_epochs
    model.warm_start = original_warm_start
    
    if verbose:
        print(f"[CHUNKED_TRAINING] Training completed on all {len(train_subjects)} subjects")
        print(f"[CHUNKED_TRAINING] Restored original max_epochs={original_max_epochs}, warm_start={original_warm_start}")
    
    return model


def evaluate_with_subject_chunks(
    model: EEGClassifier,
    paradigm,
    dataset_obj,
    eval_subjects: List[int],
    chunk_size: int = 3,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a model by loading validation subjects in chunks.
    
    This function loads validation subjects in small groups, evaluates the
    model on each chunk, and aggregates the results. This avoids loading
    all validation subjects into memory at once.
    
    Args:
        model: Trained EEGClassifier model
        paradigm: MOABB paradigm object
        dataset_obj: MOABB dataset object
        eval_subjects: List of subject IDs to use for evaluation
        chunk_size: Number of subjects to load per chunk (default: 3)
        verbose: Whether to print progress information
    
    Returns:
        Tuple of (y_true_all, y_pred_proba_all) - aggregated predictions
    """
    # Split subjects into chunks
    subject_chunks = [
        eval_subjects[i:i + chunk_size]
        for i in range(0, len(eval_subjects), chunk_size)
    ]
    
    if verbose:
        print(f"[CHUNKED_EVAL] Evaluating on {len(eval_subjects)} subjects in {len(subject_chunks)} chunks (chunk_size={chunk_size})")
    
    all_y_true = []
    all_y_pred_proba = []
    
    # Get label encoder if it exists (from training)
    label_encoder = getattr(model, '_label_encoder', None)
    
    model.module_.eval()
    
    for chunk_idx, subject_chunk in enumerate(subject_chunks):
        if verbose:
            print(f"[CHUNKED_EVAL] Loading chunk {chunk_idx + 1}/{len(subject_chunks)}: subjects {subject_chunk}")
        
        # Load chunk of subjects
        gc.collect()
        X_chunk, y_chunk, _ = paradigm.get_data(
            dataset_obj, subjects=subject_chunk
        )
        
        # Convert to float32
        if X_chunk.dtype == np.float64:
            X_chunk = X_chunk.astype(np.float32)
        
        # Encode labels if needed
        if isinstance(y_chunk[0], str):
            if label_encoder is not None:
                y_chunk = label_encoder.transform(y_chunk)
            else:
                # Shouldn't happen if model was trained with chunked trainer
                label_encoder = LabelEncoder()
                y_chunk = label_encoder.fit_transform(y_chunk)
        
        if verbose:
            chunk_size_mb = X_chunk.nbytes / 1024 / 1024
            print(f"[CHUNKED_EVAL] Chunk loaded: shape={X_chunk.shape}, size={chunk_size_mb:.2f} MB")
        
        # Evaluate on chunk
        with torch.no_grad():
            y_pred_proba_chunk = model.predict_proba(X_chunk)
        
        # Collect results
        all_y_true.append(y_chunk)
        all_y_pred_proba.append(y_pred_proba_chunk)
        
        # Clean up chunk data
        del X_chunk, y_chunk
        gc.collect()
        
        if verbose:
            print(f"[CHUNKED_EVAL] Completed chunk {chunk_idx + 1}/{len(subject_chunks)}")
    
    # Concatenate all results
    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_proba_all = np.concatenate(all_y_pred_proba, axis=0)
    
    if verbose:
        print(f"[CHUNKED_EVAL] Evaluation completed on all {len(eval_subjects)} subjects")
        print(f"[CHUNKED_EVAL] Total predictions: {len(y_true_all)}")
    
    return y_true_all, y_pred_proba_all


def train_and_evaluate_with_chunks(
    model: EEGClassifier,
    paradigm,
    dataset_obj,
    train_subjects: List[int],
    eval_subjects: List[int],
    train_chunk_size: int = 3,
    eval_chunk_size: int = 3,
    max_epochs_per_chunk: Optional[int] = None,
    verbose: bool = True
) -> Tuple[EEGClassifier, np.ndarray, np.ndarray]:
    """
    Train and evaluate a model using chunked subject loading.
    
    Convenience function that combines training and evaluation with chunked
    subject loading.
    
    Args:
        model: Initialized EEGClassifier model
        paradigm: MOABB paradigm object
        dataset_obj: MOABB dataset object
        train_subjects: List of subject IDs for training
        eval_subjects: List of subject IDs for evaluation
        train_chunk_size: Number of training subjects per chunk
        eval_chunk_size: Number of evaluation subjects per chunk
        max_epochs_per_chunk: Maximum epochs per training chunk
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained_model, y_true, y_pred_proba)
    """
    # Train with chunks
    model = train_with_subject_chunks(
        model=model,
        paradigm=paradigm,
        dataset_obj=dataset_obj,
        train_subjects=train_subjects,
        chunk_size=train_chunk_size,
        max_epochs_per_chunk=max_epochs_per_chunk,
        verbose=verbose
    )
    
    # Evaluate with chunks
    y_true, y_pred_proba = evaluate_with_subject_chunks(
        model=model,
        paradigm=paradigm,
        dataset_obj=dataset_obj,
        eval_subjects=eval_subjects,
        chunk_size=eval_chunk_size,
        verbose=verbose
    )
    
    return model, y_true, y_pred_proba
