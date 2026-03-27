import math
import os
import json
import time

import optuna
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import TensorDataset
import joblib
from moabb.evaluations import WithinSessionSplitter
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import StratifiedKFold, GroupKFold
from optuna.integration import SkorchPruningCallback
from skorch.dataset import ValidSplit, StratifiedShuffleSplit
from skorch.helper import SliceDataset
from sklearn.metrics import roc_auc_score

from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from config import get_dataset_sampling_rate


def format_params(param_block, prefix):
    module_params = ['F1', 'D', 'kernel_length', 'lstm_hidden_size', 'lstm_hidden_dim', 'lstm_num_layers', 'lstm_dropout', 'ncp_hidden_dim', 'sparsity', 'temporal_kernel_size', 'temporal_stride', 'drop_prob', 'n_modules', 'rewiring_prob', 'max_seq_length', 'bin_len', 'bin_stride', 'fusion']
    optimizer_params = ['lr', 'weight_decay']
    module_prefix = f"{prefix}module__"
    optim_prefix = f"{prefix}optimizer__"
    final_params = {}
    for k, v in param_block.items():
        key = k
        if key in module_params:
            key = module_prefix + key
        if key in optimizer_params:
            key = optim_prefix + k
        if not key.startswith(prefix):
            key = prefix + key
        final_params[key] = v

    return final_params

def create_optuna_study_with_pruning(output_dir, stage_name, direction="maximize"):
    """Create an Optuna study with advanced pruning strategies."""
    import optuna.pruners
    
    # Create study with pruning
    study = optuna.create_study(
        direction=direction,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    return study

def enhanced_cv_training_loop(
    model, cv, X_train, y_train, trial=None, groups=None, 
    early_stopping_patience=5, min_epochs=10
):
    """
    Enhanced CV training loop with early stopping and better pruning.
    """
    # Memory optimization: Convert to float32 to reduce memory usage by 50%
    # This is especially important for large arrays when using fancy indexing (which creates copies)
    # Note: This conversion happens here as this function may be called from other contexts
    if X_train.dtype == np.float64:
        X_train = X_train.astype(np.float32)
    
    fold_scores = []
    fold_times = []
    
    for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups)):
        # Fancy indexing creates copies - unavoidable with NumPy integer array indexing
        # But since X_train is already float32 (if converted), these copies use 50% less memory than float64
        X_train_part, y_train_part = X_train[train_idx], y_train[train_idx]
        X_valid_part, y_valid_part = X_train[valid_idx], y_train[valid_idx]
        
        # Fit with early stopping
        start_time = time.time()
        
        # Add early stopping callback if supported
        if hasattr(model, 'callbacks'):
            from globals import get_early_stopping_callback
            model.callbacks = [get_early_stopping_callback()]
        
        model.fit(X_train_part, y_train_part)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_valid_part)
        
        # Handle both binary and multi-class classification
        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            # Binary classification - use positive class probabilities
            y_pred = y_pred_proba[:, 1]
            auc = roc_auc_score(y_valid_part, y_pred)
        else:
            # Multi-class classification - use all probabilities with OvR strategy
            auc = roc_auc_score(y_valid_part, y_pred_proba, multi_class='ovr')
        fold_scores.append(auc)
        fold_times.append(time.time() - start_time)
        
        # if trial is not None:
        #     # Report intermediate value for pruning
        #     trial.report(auc, i)
            
        #     # Enhanced pruning logic
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    mean_time = np.mean(fold_times)
    
    return mean_score, std_score, mean_time

# Returns the mean roc-auc over the passed CV folds
def unified_cv_training_loop_method(
    model, cv, X_train, y_train, trial=None, groups=None, 
    use_slice_dataset=False, 
    use_chunked_training=False,
    paradigm=None,
    dataset_obj=None,
    train_subjects=None,
    subject_chunk_size=None
):
    """
    Root Issue: NumPy fancy indexing (integer array indexing like X_train[valid_idx]) 
    ALWAYS creates copies, not views. This is a fundamental NumPy limitation.
    
    Solution for CrossSubject mode: Use skorch's SliceDataset which wraps PyTorch Datasets
    and allows slicing without copying. SliceDataset behaves like numpy arrays for sklearn
    compatibility, but uses PyTorch's efficient indexing under the hood.
    
    Reference: https://skorch.readthedocs.io/en/stable/user/helper.html#slicedataset
    
    For chunked training: If use_chunked_training=True, loads subjects in chunks to avoid
    loading all data into memory. Requires paradigm, dataset_obj, and train_subjects.
    """
    # Check if we should use chunked training (for HPO with large datasets)
    if use_chunked_training:
        if paradigm is None or dataset_obj is None or train_subjects is None:
            raise ValueError(
                "For chunked training, paradigm, dataset_obj, and train_subjects must be provided"
            )
        
        # Import chunked trainer
        from evaluation.chunked_subject_trainer import train_with_subject_chunks
        
        # For HPO, we train on all training subjects using chunked loading
        # Each trial will train incrementally on chunks to save memory
        model.module_.train()
        
        # Train model with chunked subject loading
        # Note: warm_start is handled inside train_with_subject_chunks
        model = train_with_subject_chunks(
            model=model,
            paradigm=paradigm,
            dataset_obj=dataset_obj,
            train_subjects=train_subjects,
            chunk_size=subject_chunk_size if subject_chunk_size else 3,
            max_epochs_per_chunk=model.max_epochs,  # Use model's max_epochs
            verbose=False  # Reduce verbosity during HPO
        )
        
        # For evaluation during HPO, we need validation data
        # Since we're in HPO and don't have X_valid loaded, we'll use a simple approach:
        # Evaluate on a small held-out subset of training data
        # This is acceptable for HPO where we just need relative performance between trials
        from sklearn.model_selection import train_test_split
        _, X_val_small, _, y_val_small = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Convert to float32 if needed
        if X_val_small.dtype == np.float64:
            X_val_small = X_val_small.astype(np.float32)
        
        # Evaluate on validation subset
        model.module_.eval()
        with torch.no_grad():
            y_pred_proba = model.predict_proba(X_val_small)
        
        # Compute ROC-AUC
        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            y_pred = y_pred_proba[:, 1]
            auc = roc_auc_score(y_val_small, y_pred)
        else:
            auc = roc_auc_score(y_val_small, y_pred_proba, multi_class='ovr')
        
        return auc
    
    if use_slice_dataset:
        # Memory-efficient approach: Use SliceDataset to avoid NumPy fancy indexing copies
        # Reference: https://skorch.readthedocs.io/en/stable/user/helper.html#slicedataset
        # 
        # CRITICAL: We keep the data as float64 in the tensor and let PyTorch/skorch handle
        # the conversion to float32 during training (which happens in batches, so it's memory efficient).
        # This avoids allocating a full 5.32 GiB float32 tensor upfront.
        if isinstance(X_train, np.ndarray):
            # Create tensor views (no copy) - keep original dtype
            # PyTorch will handle dtype conversion during training in batches
            X_train_tensor = torch.from_numpy(X_train)
            y_train_tensor = torch.from_numpy(y_train)
        else:
            # Already tensors
            X_train_tensor = X_train
            y_train_tensor = y_train
        
        # Create a TensorDataset that holds references (no copy)
        full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        # Wrap with SliceDataset - this allows slicing without copying
        # idx=0 for X, idx=1 for y (as per skorch documentation)
        X_slice = SliceDataset(full_dataset, idx=0)
        y_slice = SliceDataset(full_dataset, idx=1)
        
        # cv.split() needs numpy arrays to work, but we'll use SliceDataset for slicing
        # Get the underlying numpy arrays for cv.split() - use original arrays to avoid copies
        # If X_train is already a numpy array, use it directly (no copy)
        # If it's a tensor, we need to convert, but this should be rare
        if isinstance(X_train, np.ndarray):
            X_train_np = X_train
            y_train_np = y_train
        else:
            # If somehow we got tensors, convert back (creates copy but should be rare)
            X_train_np = X_train_tensor.numpy()
            y_train_np = y_train_tensor.numpy()
        
        fold_scores = []
        for i, (train_idx, valid_idx) in enumerate(cv.split(X_train_np, y_train_np, groups)):
            # SliceDataset slicing doesn't create copies - it returns views!
            # However, braindecode's EEGClassifier doesn't fully support SliceDataset for shape inference,
            # so we need to convert back to numpy arrays before passing to model.fit()
            # This still saves memory because:
            # 1. We only copy the fold's data (not the full dataset)
            # 2. Data is already float32, so copies are 50% smaller
            # 3. We avoid the initial large float64 allocation
            X_train_part_slice = X_slice[train_idx]
            y_train_part_slice = y_slice[train_idx]
            X_valid_part_slice = X_slice[valid_idx]
            y_valid_part_slice = y_slice[valid_idx]
            
            # Convert SliceDataset to numpy arrays for braindecode compatibility
            # SliceDataset returns tensors, so convert to numpy
            if isinstance(X_train_part_slice, torch.Tensor):
                X_train_part = X_train_part_slice.numpy()
                y_train_part = y_train_part_slice.numpy() if isinstance(y_train_part_slice, torch.Tensor) else y_train_part_slice
                X_valid_part = X_valid_part_slice.numpy()
                y_valid_part = y_valid_part_slice.numpy() if isinstance(y_valid_part_slice, torch.Tensor) else y_valid_part_slice
            else:
                # Already numpy arrays (shouldn't happen with SliceDataset, but handle it)
                X_train_part = np.asarray(X_train_part_slice)
                y_train_part = np.asarray(y_train_part_slice)
                X_valid_part = np.asarray(X_valid_part_slice)
                y_valid_part = np.asarray(y_valid_part_slice)
            
            # Fit on training fold
            model.module_.train()
            model.fit(X_train_part, y_train_part)

            # Evaluate on held-out validation set
            model.module_.eval()
            with torch.no_grad():
                y_pred_proba = model.predict_proba(X_valid_part)
            
            # Handle both binary and multi-class classification
            # y_valid_part is already numpy from conversion above
            
            n_classes = y_pred_proba.shape[1]
            if n_classes == 2:
                # Binary classification - use positive class probabilities
                y_pred = y_pred_proba[:, 1]
                auc = roc_auc_score(y_valid_part, y_pred)
            else:
                # Multi-class classification - use all probabilities with OvR strategy
                auc = roc_auc_score(y_valid_part, y_pred_proba, multi_class='ovr')
            fold_scores.append(auc)
    else:
        # Fallback: Traditional NumPy approach (creates copies but works for smaller datasets)
        # Note: X_train should already be converted to float32 in run_optuna_stage to save memory
        fold_scores = []
        for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups)):
            # Fancy indexing creates copies - unavoidable with NumPy integer array indexing
            # But since X_train is already float32, these copies use 50% less memory than float64
            X_train_part, y_train_part = X_train[train_idx], y_train[train_idx]
            X_valid_part, y_valid_part = X_train[valid_idx], y_train[valid_idx]
            
            # Fit on training fold
            model.module_.train()
            model.fit(X_train_part, y_train_part)

            # Evaluate on held-out validation set
            model.module_.eval()
            with torch.no_grad():
                y_pred_proba = model.predict_proba(X_valid_part)
            
            # Handle both binary and multi-class classification
            n_classes = y_pred_proba.shape[1]
            if n_classes == 2:
                # Binary classification - use positive class probabilities
                y_pred = y_pred_proba[:, 1]
                auc = roc_auc_score(y_valid_part, y_pred)
            else:
                # Multi-class classification - use all probabilities with OvR strategy
                auc = roc_auc_score(y_valid_part, y_pred_proba, multi_class='ovr')
            fold_scores.append(auc)
    
    return np.mean(fold_scores)


def run_optuna_stage(
        model_fn,
        model_name,
        stage_name,
        X,
        y,
        metadata,
        param_space_fn,
        resample=250.0,
        n_trials=25,
        seed=42,
        perturbed=False,
        output_root="optuna_results",
        eval_mode=None,
        paradigm=None,
        dataset_obj=None,
        train_subjects=None,
        subject_chunk_size=None,
        n_outputs=None
):
    # In the old version we explicitly wanted to only use 0train for hyperparameter optimization. That is no longer the case.
    # In the current code version, we can expect X and y to be split before run_optuna_stage is called.
    X_train = X
    y_train = y
    metadata_train = metadata

    # Memory optimization: For CrossSubject mode, we use SliceDataset which avoids copies entirely
    # For other modes, convert to float32 to reduce memory usage by 50%
    # This is especially important for large arrays when using fancy indexing (which creates copies)
    if eval_mode == "CrossSubject":
        print(f"[MEMORY] Using SliceDataset for CrossSubject mode to avoid NumPy fancy indexing copies. Shape: {X_train.shape}")
        print(f"[MEMORY] Will convert to float32 only when creating tensors (no intermediate copy).")
    else:
        # For non-CrossSubject modes, convert to float32 here to reduce memory usage
        if X_train.dtype == np.float64:
            X_train = X_train.astype(np.float32)
            print(f"[MEMORY] Converted X_train from float64 to float32 to reduce memory usage. Shape: {X_train.shape}")

    if len(X_train) < 10:
        print(f"Too few training samples: {len(X_train)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"metadata shape: {metadata.shape}")
        print(f"metadata: {metadata}")
        raise ValueError("Too few training samples for session 0.")

    param_prefix = "base_pipeline__" if perturbed else ""
    if resample is None:
        # Default to 250 Hz (common for MotorImagery and ERP datasets)
        # Note: Lee2019_SSVEP uses 1000 Hz, so resample should be provided explicitly
        resample = 250.0

    sfreq = float(resample)

    check_time = False
    
    # Determine n_chans and n_times from input data
    # X should have shape (n_samples, n_chans, n_times)
    n_chans = X_train.shape[1]
    n_times = X_train.shape[2] if len(X_train.shape) > 2 else int(resample * 4)
    
    # Determine n_outputs from y if not provided
    if n_outputs is None:
        n_outputs = len(np.unique(y_train))
        print(f"[INFO] Auto-detected n_outputs={n_outputs} from number of unique classes in y")
    
    # Use a mutable container to allow recreating the model inside objective() when wiring_arch_index changes
    model_container = [model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=sfreq)]
    # model_container[0].verbose = 0
    # model_container[0].callbacks = []
    # model_container[0].train_split = None
    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        # params[f"{param_prefix}train_split"] = None
        params[f"{param_prefix}verbose"] = 0
        # params[f"{param_prefix}callbacks"] = []
        
        # For CrossSubject evaluation, use only 5 epochs during optimization to speed up trials
        from globals import get_max_epochs_for_dataset, DEFAULT_MAX_EPOCHS
        if eval_mode == "CrossSubject":
            params[f"{param_prefix}max_epochs"] = 5
        else:
            # Use normal max_epochs for other evaluation modes
            params[f"{param_prefix}max_epochs"] = get_max_epochs_for_dataset(None, eval_mode=eval_mode) if eval_mode else DEFAULT_MAX_EPOCHS
        
        # Extract wiring_arch_index if present - it's handled at factory level, not via set_params
        wiring_arch_index = params.pop(f"{param_prefix}wiring_arch_index", None)
        # Also check without prefix for compatibility
        if wiring_arch_index is None:
            wiring_arch_index = params.pop("wiring_arch_index", None)
        
        # If wiring_arch_index is present, recreate the model with the new wiring
        # This is necessary because wiring_arch_index is a factory-level parameter
        if wiring_arch_index is not None:
            # Recreate model with wiring_arch_index passed to factory function
            model_container[0] = model_fn(
                n_chans=n_chans, 
                n_times=n_times, 
                n_outputs=n_outputs,
                wiring_arch_index=wiring_arch_index,
                sfreq=sfreq,
            )
        
        model = model_container[0]
        
        # Define model
        model.set_params(**params)
        model.initialize()
        if model.max_epochs > 200:
            print("ERROR: Max epochs is greater than 200")
            raise ValueError("Max epochs is greater than 200")
            sys.exit(1)
        

        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        if check_time:
            start_time = time.time()
        # Use SliceDataset for CrossSubject mode to avoid memory copies
        # OR use chunked training if subject_chunk_size is provided
        use_slice_dataset = (eval_mode == "CrossSubject") and (subject_chunk_size is None)
        use_chunked_training = (eval_mode == "CrossSubject") and (subject_chunk_size is not None)
        
        roc_auc_score = unified_cv_training_loop_method(
            model, cv, X_train, y_train, trial=trial, 
            use_slice_dataset=use_slice_dataset,
            use_chunked_training=use_chunked_training,
            paradigm=paradigm if use_chunked_training else None,
            dataset_obj=dataset_obj if use_chunked_training else None,
            train_subjects=train_subjects if use_chunked_training else None,
            subject_chunk_size=subject_chunk_size if use_chunked_training else None
        )
        if check_time:
            elapsed_time = time.time() - start_time
            target_time = 90.0
            alpha = 1.0  # accuracy weight
            beta = 0.2  # penalty for slowness
            normalized_time = min(elapsed_time / target_time, 5.0)  # cap to prevent explosion
            composite_score = (alpha * roc_auc_score) - (beta * normalized_time)
            return composite_score
        else:
            composite_score = roc_auc_score
        return composite_score


    output_dir = os.path.join(output_root, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    # Set Optuna's logging level to WARNING (only show warnings and errors)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    study_path = os.path.join(output_dir, "optuna_study.pkl")
    
    joblib.dump(study, study_path)

    # with open(os.path.join(output_dir, "best_params.json"), "w") as f:
    #     json.dump({"best_score": study.best_value, "best_params": study.best_params}, f, indent=2)

    # try:
    #     plot_optimization_history(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_history.html"))
    #     plot_param_importances(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_importances.html"))
    # except Exception as e:
    #     print(f"Plotting failed: {e}")

    return study.best_params, study.best_value


def alternate_optuna_stage(
        model_fn,
        model_name,
        stage_name,
        X,
        y,
        metadata,
        param_space_fn,
        mode,
        noise_dict,
        resample=250.0,
        n_trials=25,
        seed=42,
        output_root="optuna_results",
        dataset=None,
        eval_mode=None,
        paradigm=None,
        dataset_obj=None,
        train_subjects=None,
        subject_chunk_size=None
):
    noise_type = noise_dict["noise_type"]
    intensity = noise_dict["intensity"]
    X_train = X
    y_train = y
    # train_mask = metadata["session"] == "0train"
    # X_train = X[train_mask]
    # y_train = y[train_mask]

    if len(X_train) < 10:
        raise ValueError("Too few training samples for session 0.")
    param_prefix = "base_pipeline__" if (mode == "perturb") or (mode == "augment") else ""
    if resample is None:
        # Default to 250 Hz (common for MotorImagery and ERP datasets)
        # Note: Lee2019_SSVEP uses 1000 Hz, so resample should be provided explicitly
        resample = 250.0

    sfreq = float(get_dataset_sampling_rate(dataset)) if dataset else float(resample)
    
    # Determine n_chans and n_times from input data
    n_chans = X_train.shape[1]
    n_times = X_train.shape[2] if len(X_train.shape) > 2 else int(resample * 4)
    
    # Determine n_outputs from y if not provided
    n_outputs = len(np.unique(y_train))
    model = model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=sfreq)
    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        params[f"{param_prefix}train_split"] = None
        from globals import get_max_epochs_for_dataset, DEFAULT_MAX_EPOCHS
        # For CrossSubject evaluation, use only 5 epochs during optimization to speed up trials
        if eval_mode == "CrossSubject":
            params[f"{param_prefix}max_epochs"] = 5
        else:
            # Use dataset-specific max_epochs if dataset is provided, otherwise use default
            max_epochs = get_max_epochs_for_dataset(dataset, eval_mode=eval_mode) if dataset else DEFAULT_MAX_EPOCHS
            params[f"{param_prefix}max_epochs"] = max_epochs
        params[f"{param_prefix}verbose"] = 0
        params[f"{param_prefix}callbacks"] = []

        # Define model
        model.set_params(**params)
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        # Augmentation mode: Supplement clean input data with contaminated samples
        groups = None
        X_obj = X_train
        y_obj = y_train
        # If doing concatenated data augmentation, we need to track what set (training/validation) each sample belonged to originally
        if mode == 'augment':
            cv = GroupKFold(n_splits=3)
            X_obj, y_obj, groups = model.concat_and_augment(X_train, y_train)
        
        # Use chunked training if enabled
        use_slice_dataset = (eval_mode == "CrossSubject") and (subject_chunk_size is None)
        use_chunked_training = (eval_mode == "CrossSubject") and (subject_chunk_size is not None)
        
        return unified_cv_training_loop_method(
            model, cv, X_obj, y_obj, trial=trial, groups=groups,
            use_slice_dataset=use_slice_dataset,
            use_chunked_training=use_chunked_training,
            paradigm=paradigm if use_chunked_training else None,
            dataset_obj=dataset_obj if use_chunked_training else None,
            train_subjects=train_subjects if use_chunked_training else None,
            subject_chunk_size=subject_chunk_size if use_chunked_training else None
        )

    output_dir = os.path.join(output_root, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    study_path = os.path.join(output_dir, "optuna_study.pkl")
    import joblib
    joblib.dump(study, study_path)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump({"best_score": study.best_value, "best_params": study.best_params}, f, indent=2)

    # try:
    #     plot_optimization_history(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_history.html"))
    #     plot_param_importances(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_importances.html"))
    # except Exception as e:
    #     print(f"Plotting failed: {e}")

    best_params = study.best_params
    best_params[f"{param_prefix}train_split"] = None
    # best_params[f"{param_prefix}max_epochs"] = 100
    best_params[f"{param_prefix}verbose"] = 1
    best_params[f"{param_prefix}callbacks"] = []

    return best_params, study.best_value


def cnncfc_compact_architecture_space(trial, prefix):
    return {
        # CfC core parameters
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(
            f"{prefix}module__ncp_hidden_dim", 8, 64
        ),
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        f"{prefix}module__use_stochastic_depth": trial.suggest_categorical(
            f"{prefix}module__use_stochastic_depth", [True, False]
        ),
        # Sequence length control
        f"{prefix}module__max_seq_length": trial.suggest_int(
            f"{prefix}module__max_seq_length", 150, 300
        ),
        # NEW: CfC-specific parameters that were previously hardcoded
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
        f"{prefix}module__mode": trial.suggest_categorical(
            f"{prefix}module__mode", ["default", "pure", "no_gate"]
        ),
        f"{prefix}module__activation": trial.suggest_categorical(
            f"{prefix}module__activation", ["lecun_tanh", "silu", "relu", "tanh", "gelu"]
        ),
        f"{prefix}module__backbone_units": trial.suggest_int(
            f"{prefix}module__backbone_units", 16, 256
        ),
        f"{prefix}module__backbone_layers": trial.suggest_int(
            f"{prefix}module__backbone_layers", 1, 3
        ),
        f"{prefix}module__backbone_dropout": trial.suggest_float(
            f"{prefix}module__backbone_dropout", 0.0, 0.5
        )
    }


def cnncfc_compact_training_space(trial, prefix):
    return {
        # Optimizer parameters
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        
        # Training parameters
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }

def diva_ncp_architecture_space(trial, prefix):
    return {
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(
            f"{prefix}module__ncp_hidden_dim", 16, 128
        ),
    }

def diva_ncp_training_space(trial, prefix):
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
    }

def branched_diva_ncp_architecture_space(trial, prefix):
    """Architecture parameter space for BranchedDIVANCP model."""
    return {
        # NCP parameters
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(
            f"{prefix}module__ncp_hidden_dim", 16, 128
        ),
        f"{prefix}module__sparsity": trial.suggest_float(
            f"{prefix}module__sparsity", 0.2, 0.9
        ),
        # CNN parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=32
        ),
        # Temporal processing
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        # Binning parameters
        f"{prefix}module__bin_len": trial.suggest_int(
            f"{prefix}module__bin_len", 32, 128, step=16
        ),
        f"{prefix}module__bin_stride": trial.suggest_int(
            f"{prefix}module__bin_stride", 24, 96, step=12
        ),
        f"{prefix}module__fusion": trial.suggest_categorical(
            f"{prefix}module__fusion", ["attn", "mean"]
        ),
    }

def branched_diva_ncp_training_space(trial, prefix):
    """Training parameter space for BranchedDIVANCP model."""
    return {
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def branched_lstm_architecture_space(trial, prefix):
    """Architecture parameter space for BranchedLSTM model."""
    return {
        # LSTM parameters (lstm_hidden_dim is now used for recurrent_output_size)
        f"{prefix}module__lstm_hidden_dim": trial.suggest_int(
            f"{prefix}module__lstm_hidden_dim", 16, 128
        ),
        f"{prefix}module__lstm_num_layers": trial.suggest_int(
            f"{prefix}module__lstm_num_layers", 1, 3
        ),
        f"{prefix}module__lstm_dropout": trial.suggest_float(
            f"{prefix}module__lstm_dropout", 0.0, 0.3
        ),
        # CNN parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=32
        ),
        # Temporal processing
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        # Binning parameters
        f"{prefix}module__bin_len": trial.suggest_int(
            f"{prefix}module__bin_len", 32, 128, step=16
        ),
        f"{prefix}module__bin_stride": trial.suggest_int(
            f"{prefix}module__bin_stride", 24, 96, step=12
        ),
        f"{prefix}module__fusion": trial.suggest_categorical(
            f"{prefix}module__fusion", ["attn", "mean"]
        ),
    }


def branched_lstm_training_space(trial, prefix):
    """Training parameter space for BranchedLSTM model."""
    return {
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def branched_wiredcfc_architecture_space(trial, prefix):
    """Architecture parameter space for BranchedWiredCfC model."""
    return {
        # NOTE:
        # - We intentionally DO NOT tune `recurrent_output_size` here.
        #   The BranchedWiredCfC base class defaults `recurrent_output_size` to F2
        #   (the CNN feature dimension) to keep the residual connection valid.
        #   Tuning it independently caused shape mismatches (H != F2) in the
        #   residual add. By omitting it from the search space, we keep the
        #   safe default behaviour for all tuned runs.

        # CfC / regularization parameters
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        
        # CNN feature extraction parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=32
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        # NOTE:
        # - We also intentionally DO NOT tune `bin_len` or `bin_stride` here.
        #   The BranchedDIVABase uses defaults bin_len=48 and bin_stride=44,
        #   which are known to work for BNCI2014_001 CrossSession runs.
        #   Allowing Optuna to sample larger window sizes led to runtime errors
        #   in `_chunk_time` when the post-downsampled length T2 was shorter
        #   than the suggested bin length. By omitting these parameters from
        #   the search space, tuned runs re-use the stable defaults.

        # Fusion type over bins
        f"{prefix}module__fusion": trial.suggest_categorical(
            f"{prefix}module__fusion", ["attn", "mean"]
        ),
        
        # CfC-specific parameters
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
        f"{prefix}module__mode": trial.suggest_categorical(
            f"{prefix}module__mode", ["default", "pure", "no_gate"]
        ),
        f"{prefix}module__activation": trial.suggest_categorical(
            f"{prefix}module__activation", ["lecun_tanh", "silu", "relu", "tanh", "gelu"]
        ),
        f"{prefix}module__backbone_units": trial.suggest_int(
            f"{prefix}module__backbone_units", 64, 256
        ),
        f"{prefix}module__backbone_layers": trial.suggest_int(
            f"{prefix}module__backbone_layers", 1, 3
        ),
        f"{prefix}module__backbone_dropout": trial.suggest_float(
            f"{prefix}module__backbone_dropout", 0.0, 0.5
        ),
    }


def branched_wiredcfc_training_space(trial, prefix):
    """Training parameter space for BranchedWiredCfC model."""
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def hydra_v2_architecture_space(trial, prefix):
    """
    Architecture parameter space for HYDRAv2 model.
    
    Includes wiring selection from the 10 available architectures in outputs/architectures.
    The wiring's input/output dimensions will be automatically reconfigured based on F1 and D.
    """
    import os
    from pathlib import Path
    
    # Find available architecture files
    architectures_dir = Path("outputs/architectures")
    architecture_files = sorted(architectures_dir.glob("best_architecture_*.json"))
    
    # Create list of architecture indices (1-10)
    architecture_choices = [i for i in range(1, len(architecture_files) + 1)]
    
    if not architecture_choices:
        raise ValueError("No architecture files found in outputs/architectures")
    
    return {
        # NOTE:
        # - We intentionally DO NOT tune `recurrent_output_size` here.
        #   The BranchedWiredCfC base class defaults `recurrent_output_size` to F2
        #   (the CNN feature dimension) to keep the residual connection valid.
        #   Tuning it independently caused shape mismatches (H != F2) in the
        #   residual add. By omitting it from the search space, we keep the
        #   safe default behaviour for all tuned runs.
        
        # Wiring selection: Choose from available architectures (1-10)
        # This is handled at the model factory level, not as a module parameter
        # Note: Not using module__ prefix since wiring is selected before module creation
        f"{prefix}wiring_arch_index": trial.suggest_categorical(
            f"{prefix}wiring_arch_index", architecture_choices
        ),
        
        # CfC / regularization parameters
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        
        # CNN feature extraction parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=32
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        
        # Fusion type over bins
        f"{prefix}module__fusion": trial.suggest_categorical(
            f"{prefix}module__fusion", ["attn", "mean"]
        ),
        
        # HYDRAv2-specific parameters
        f"{prefix}module__num_attn_queries": trial.suggest_categorical(
            f"{prefix}module__num_attn_queries", [2, 4, 6, 8]
        ),
        f"{prefix}module__attn_dropout": trial.suggest_float(
            f"{prefix}module__attn_dropout", 0.0, 0.3
        ),
        f"{prefix}module__use_adaptive_residual": trial.suggest_categorical(
            f"{prefix}module__use_adaptive_residual", [True, False]
        ),
        f"{prefix}module__use_erp_head": trial.suggest_categorical(
            f"{prefix}module__use_erp_head", [True, False]
        ),
        f"{prefix}module__use_ssvep_head": trial.suggest_categorical(
            f"{prefix}module__use_ssvep_head", [True, False]
        ),
        f"{prefix}module__erp_num_queries": trial.suggest_categorical(
            f"{prefix}module__erp_num_queries", [2, 4, 6]
        ),
        f"{prefix}module__ssvep_num_filters": trial.suggest_categorical(
            f"{prefix}module__ssvep_num_filters", [2, 4, 6]
        ),
        f"{prefix}module__use_cross_bin_context": trial.suggest_categorical(
            f"{prefix}module__use_cross_bin_context", [True, False]
        ),
        f"{prefix}module__context_type": trial.suggest_categorical(
            f"{prefix}module__context_type", ["transformer", "gru"]
        ),
        f"{prefix}module__use_global_skip": trial.suggest_categorical(
            f"{prefix}module__use_global_skip", [True, False]
        ),
        
        # CfC-specific parameters
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
        f"{prefix}module__mode": trial.suggest_categorical(
            f"{prefix}module__mode", ["default", "pure", "no_gate"]
        ),
        f"{prefix}module__activation": trial.suggest_categorical(
            f"{prefix}module__activation", ["lecun_tanh", "silu", "relu", "tanh", "gelu"]
        ),
        f"{prefix}module__backbone_units": trial.suggest_int(
            f"{prefix}module__backbone_units", 64, 256
        ),
        f"{prefix}module__backbone_layers": trial.suggest_int(
            f"{prefix}module__backbone_layers", 1, 3
        ),
        f"{prefix}module__backbone_dropout": trial.suggest_float(
            f"{prefix}module__backbone_dropout", 0.0, 0.5
        ),
    }


def hydra_v2_training_space(trial, prefix):
    """Training parameter space for HYDRAv2 model."""
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def hydra_v3_architecture_space(trial, prefix):
    """
    Architecture parameter space for HYDRAv3 model.
    
    HYDRAv3 uses CfC-based carry controller instead of explicit adaptive gates.
    Simplified architecture based on analysis recommendations.
    """
    import os
    from pathlib import Path
    
    # Find available architecture files
    architectures_dir = Path("outputs/architectures")
    architecture_files = sorted(architectures_dir.glob("best_architecture_*.json"))
    
    # Create list of architecture indices (1-10)
    architecture_choices = [i for i in range(1, len(architecture_files) + 1)]
    
    if not architecture_choices:
        raise ValueError("No architecture files found in outputs/architectures")
    
    return {
        # NOTE:
        # - We intentionally DO NOT tune `recurrent_output_size` here.
        #   The BranchedWiredCfC base class defaults `recurrent_output_size` to F2
        #   (the CNN feature dimension) to keep the residual connection valid.
        
        # Wiring selection: Choose from available architectures (1-10)
        f"{prefix}wiring_arch_index": trial.suggest_categorical(
            f"{prefix}wiring_arch_index", architecture_choices
        ),
        
        # CfC / regularization parameters
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        
        # CNN feature extraction parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=32
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        
        # Fusion type over bins
        f"{prefix}module__fusion": trial.suggest_categorical(
            f"{prefix}module__fusion", ["attn", "mean"]
        ),
        
        # HYDRAv3-specific parameters
        f"{prefix}module__use_cfc_carry_controller": trial.suggest_categorical(
            f"{prefix}module__use_cfc_carry_controller", [True, False]
        ),
        f"{prefix}module__controller_dim": trial.suggest_categorical(
            f"{prefix}module__controller_dim", [1, 2]  # d_c ∈ {1, 2} per spec
        ),
        f"{prefix}module__use_ssvep_head": trial.suggest_categorical(
            f"{prefix}module__use_ssvep_head", [True, False]
        ),
        f"{prefix}module__ssvep_num_filters": trial.suggest_categorical(
            f"{prefix}module__ssvep_num_filters", [2, 4, 6]
        ),
        
        # CfC-specific parameters
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
        f"{prefix}module__mode": trial.suggest_categorical(
            f"{prefix}module__mode", ["default", "pure", "no_gate"]
        ),
        f"{prefix}module__activation": trial.suggest_categorical(
            f"{prefix}module__activation", ["lecun_tanh", "silu", "relu", "tanh", "gelu"]
        ),
        f"{prefix}module__backbone_units": trial.suggest_int(
            f"{prefix}module__backbone_units", 64, 256
        ),
        f"{prefix}module__backbone_layers": trial.suggest_int(
            f"{prefix}module__backbone_layers", 1, 3
        ),
        f"{prefix}module__backbone_dropout": trial.suggest_float(
            f"{prefix}module__backbone_dropout", 0.0, 0.5
        ),
    }


def hydra_v3_training_space(trial, prefix):
    """Training parameter space for HYDRAv3 model."""
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def diva_full_architecture_space(trial, prefix):
    """Architecture parameter space for DIVAInspiredEEG (diva_full) model."""
    return {
        # CNN front-end parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=32
        ),
        
        # NCP/CfC parameters
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(
            f"{prefix}module__ncp_hidden_dim", 16, 128
        ),
        f"{prefix}module__sparsity": trial.suggest_float(
            f"{prefix}module__sparsity", 0.2, 0.9
        ),
        
        # Temporal processing
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        
        # DIVA-specific switches
        f"{prefix}module__use_ms_block": trial.suggest_categorical(
            f"{prefix}module__use_ms_block", [True, False]
        ),
        f"{prefix}module__use_snr_gate": trial.suggest_categorical(
            f"{prefix}module__use_snr_gate", [True, False]
        ),
        f"{prefix}module__use_forward_model": trial.suggest_categorical(
            f"{prefix}module__use_forward_model", [True, False]
        ),
        f"{prefix}module__use_feedback_controller": trial.suggest_categorical(
            f"{prefix}module__use_feedback_controller", [True, False]
        ),
        f"{prefix}module__use_delay": trial.suggest_categorical(
            f"{prefix}module__use_delay", [True, False]
        ),
        f"{prefix}module__use_uncertainty_mixer": trial.suggest_categorical(
            f"{prefix}module__use_uncertainty_mixer", [True, False]
        ),
        
        # Feedback controller parameters
        f"{prefix}module__feedback_hidden": trial.suggest_int(
            f"{prefix}module__feedback_hidden", 32, 128, step=16
        ),
    }


def diva_full_training_space(trial, prefix):
    """Training parameter space for DIVAInspiredEEG (diva_full) model."""
    return {
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }
    


def cnn_smallworld_architecture_space(trial, prefix):
    """Architecture parameter space for CNNSmallWorld model."""
    return {
        # CfC core parameters
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(
            f"{prefix}module__ncp_hidden_dim", 16, 128
        ),
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        
        # CNN feature extraction parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16, 20]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4, 8]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 512, step=32
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7, 9]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        
        # Sequence length control
        f"{prefix}module__max_seq_length": trial.suggest_int(
            f"{prefix}module__max_seq_length", 150, 500
        ),
        
        # Small World wiring parameters
        f"{prefix}module__n_modules": trial.suggest_categorical(
            f"{prefix}module__n_modules", [1, 2, 4, 6, 8]
        ),
        f"{prefix}module__rewiring_prob": trial.suggest_float(
            f"{prefix}module__rewiring_prob", 0.1, 0.5
        )
    }


def cnn_smallworld_training_space(trial, prefix):
    """Training parameter space for CNNSmallWorld model."""
    return {
        # Optimizer parameters
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        
        # Training parameters
        # f"{prefix}max_epochs": trial.suggest_int(
        #     f"{prefix}max_epochs", 20, 100
        # ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def cnn_wiredcfc_architecture_space(trial, prefix):
    """Architecture parameter space for CNNWiredCfC models."""
    return {
         # CNN feature extraction parameters - PROPERLY UTILIZED
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=16
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        # Sequence length control
        f"{prefix}module__max_seq_length": trial.suggest_int(
            f"{prefix}module__max_seq_length", 150, 1000
        ),
        # NEW: CfC-specific parameters that were previously hardcoded
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
    }


def cnn_wiredcfc_training_space(trial, prefix):
    """Training parameter space for CNNWiredCfC models."""
    return {
        # Optimizer parameters
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        
        # Training parameters
        # f"{prefix}max_epochs": trial.suggest_int(
        #     f"{prefix}max_epochs", 20, 100
        # ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }

# Iterate over model parameters and return a list of all parameters that we may have tuned
def get_all_model_params(model_name):    
    architecture_space = get_model_architecture_space(model_name)
    training_space = get_model_training_space(model_name)
    # Architecture space expects an optuna trial object, so we need to create one.    
    dummy_study = optuna.create_study(direction="maximize")    
    study_trial = dummy_study.ask()
    architecture_params = architecture_space(study_trial, "")
    training_params = training_space(study_trial, "")
    all_params = []
    for k in architecture_params.keys():
        all_params.append(k)
    for k in training_params.keys():
        all_params.append(k)
    return all_params


def get_model_architecture_space(model_name):
    # NAS pilot study runtime-registered models
    if model_name.startswith("nas_pilot_"):
        return cnn_wiredcfc_architecture_space
    # Plot 2 topology study runtime-registered models
    if model_name.startswith("plot2_"):
        return cnn_wiredcfc_architecture_space
    # Paper 3 experiments (sanity check, mini selection, exp2; ws_flex/dense from nas_pilot_dir)
    if model_name.startswith("paper3_"):
        return cnn_wiredcfc_architecture_space
    # Wiring robustness experiment (CNNWiredCfCMin base)
    if model_name.startswith("wiring_graph_"):
        return cnn_wiredcfc_architecture_space
    # Orientation sensitivity experiment (random_oriented vs symmetric)
    if model_name.startswith("orient_"):
        return cnn_wiredcfc_architecture_space
    if model_name == "ncp_baseline_32":
        return cnn_wiredcfc_architecture_space
    if model_name == "cnn_wiredcfc_min":
        return cnn_wiredcfc_architecture_space

    # Check if this is a wiredcfc architecture model
    if model_name.startswith("wiredcfc_arch"):
        return cnn_wiredcfc_architecture_space
    
    # Check if this is a branched_wiredcfc architecture model
    if model_name.startswith("branched_wiredcfc_arch"):
        return branched_wiredcfc_architecture_space
    
    # Check if this is a branched_lstm architecture model
    if model_name.startswith("branched_lstm_arch"):
        return branched_lstm_architecture_space
    
    # Check if this is a hydra_v2 architecture model
    if model_name.startswith("hydra_v2"):
        return hydra_v2_architecture_space
    
    # Check if this is a hydra_v3 architecture model
    if model_name.startswith("hydra_v3"):
        return hydra_v3_architecture_space
    
    architecture_registry = {
        "eegnet": eegnet_architecture_space,
        "ctnet": ctnet_architecture_space,
        "reegnet": reegnet_architecture_space,
        "cnn_ncp": cnn_ncp_architecture_space,
        "cnn_ncp_v2": cnn_ncp_architecture_space,
        "cnncfc_v2": improved_cnncfc_architecture_space,  # CNNCfCv2 uses the same space as improved_cnncfc
        "cnncfc_compact": cnncfc_compact_architecture_space,
        "spp_ncp": spp_ncp_architecture_space,
        "cnn_smallworld": cnn_smallworld_architecture_space,
        "cnn_ncp_branch": cnn_ncp_branched_bins_architecture_space,
        "diva_ncp": diva_ncp_architecture_space,
        "branched_diva_ncp": branched_diva_ncp_architecture_space,
        "branched_lstm": branched_lstm_architecture_space,
        "branched_wiredcfc": branched_wiredcfc_architecture_space,
        "hydra_v2": hydra_v2_architecture_space,
        "hydra_v3": hydra_v3_architecture_space,
        "diva_full": diva_full_architecture_space,
        "cnn_wiredcfc_min": cnn_wiredcfc_architecture_space,
    }
    if model_name not in architecture_registry:
        raise KeyError(
            f"Unknown model_name for architecture space: {model_name}. "
            f"Known models: {sorted(architecture_registry.keys())} "
            f"(plus dynamic prefixes: wiredcfc_arch*, branched_wiredcfc_arch*, branched_lstm_arch*, hydra_v2*, hydra_v3*, nas_pilot_*, plot2_*, paper3_*, wiring_graph_*, orient_*)"
        )
    return architecture_registry[model_name]


def get_model_training_space(model_name):
    # NAS pilot study runtime-registered models
    if model_name.startswith("nas_pilot_"):
        return cnn_wiredcfc_training_space
    # Plot 2 topology study runtime-registered models
    if model_name.startswith("plot2_"):
        return cnn_wiredcfc_training_space
    # Paper 3 experiments (sanity check, mini selection, exp2; ws_flex/dense from nas_pilot_dir)
    if model_name.startswith("paper3_"):
        return cnn_wiredcfc_training_space
    # Wiring robustness experiment (CNNWiredCfCMin base)
    if model_name.startswith("wiring_graph_"):
        return cnn_wiredcfc_training_space
    # Orientation sensitivity experiment (random_oriented vs symmetric)
    if model_name.startswith("orient_"):
        return cnn_wiredcfc_training_space
    if model_name == "ncp_baseline_32":
        return cnn_wiredcfc_training_space
    if model_name == "cnn_wiredcfc_min":
        return cnn_wiredcfc_training_space

    # Check if this is a wiredcfc architecture model
    if model_name.startswith("wiredcfc_arch"):
        return cnn_wiredcfc_training_space
    
    # Check if this is a branched_wiredcfc architecture model
    if model_name.startswith("branched_wiredcfc_arch"):
        return branched_wiredcfc_training_space
    
    # Check if this is a branched_lstm architecture model
    if model_name.startswith("branched_lstm_arch"):
        return branched_lstm_training_space
    
    # Check if this is a hydra_v2 architecture model
    if model_name.startswith("hydra_v2"):
        return hydra_v2_training_space
    
    # Check if this is a hydra_v3 architecture model
    if model_name.startswith("hydra_v3"):
        return hydra_v3_training_space
    
    training_registry = {
        "eegnet": eegnet_training_space,
        "ctnet": ctnet_training_space,
        "reegnet": reegnet_training_space,
        "cnn_ncp": cnn_ncp_training_space,
        "cnn_ncp_v2": cnn_ncp_training_space,
        "cnncfc_v2": improved_cnncfc_training_space,  # CNNCfCv2 uses the same space as improved_cnncfc
        "cnncfc_compact": cnncfc_compact_training_space,
        "spp_ncp": spp_ncp_training_space,
        "cnn_smallworld": cnn_smallworld_training_space,
        "cnn_ncp_branch": cnn_ncp_branched_bins_training_space,
        "diva_ncp": diva_ncp_training_space,
        "branched_diva_ncp": branched_diva_ncp_training_space,
        "branched_lstm": branched_lstm_training_space,
        "branched_wiredcfc": branched_wiredcfc_training_space,
        "hydra_v2": hydra_v2_training_space,
        "hydra_v3": hydra_v3_training_space,
        "diva_full": diva_full_training_space,
        "cnn_wiredcfc_min": cnn_wiredcfc_training_space,
    }
    if model_name not in training_registry:
        raise KeyError(
            f"Unknown model_name for training space: {model_name}. "
            f"Known models: {sorted(training_registry.keys())} "
            f"(plus dynamic prefixes: wiredcfc_arch*, branched_wiredcfc_arch*, branched_lstm_arch*, hydra_v2*, hydra_v3*, nas_pilot_*, plot2_*, paper3_*, wiring_graph_*, orient_*)"
        )
    return training_registry[model_name]


def improved_cnncfc_architecture_space(trial, prefix):
    """Enhanced architecture parameter space for improved CNNCfC."""
    return {
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(
            f"{prefix}module__ncp_hidden_dim", 8, 64
        ),
        
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
        
        # CNN feature extraction parameters - PROPERLY UTILIZED
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 64, 256, step=16
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [2, 4, 6, 8]
        ),
        # Sequence length control
        f"{prefix}module__max_seq_length": trial.suggest_int(
            f"{prefix}module__max_seq_length", 150, 1000
        ),
        # NEW: CfC-specific parameters that were previously hardcoded
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
        f"{prefix}module__mode": trial.suggest_categorical(
            f"{prefix}module__mode", ["default", "pure", "no_gate"]
        ),
        f"{prefix}module__activation": trial.suggest_categorical(
            f"{prefix}module__activation", ["lecun_tanh", "silu", "relu", "tanh", "gelu"]
        ),
        f"{prefix}module__backbone_units": trial.suggest_int(
            f"{prefix}module__backbone_units", 16, 256
        ),
        f"{prefix}module__backbone_layers": trial.suggest_int(
            f"{prefix}module__backbone_layers", 1, 3
        ),
        f"{prefix}module__backbone_dropout": trial.suggest_float(
            f"{prefix}module__backbone_dropout", 0.0, 0.5
        )
    }


def improved_cnncfc_training_space(trial, prefix):
    """Enhanced training parameter space for improved CNNCfC."""
    return {
        # Optimizer parameters
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        
        # Training parameters
        # f"{prefix}max_epochs": trial.suggest_int(
        #     f"{prefix}max_epochs", 20, 100
        # ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
    }


def cnn_ncp_branched_bins_architecture_space(trial, prefix):
    return {
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(f"{prefix}module__ncp_hidden_dim", 19, 128),
        f"{prefix}module__sparsity": trial.suggest_float(f"{prefix}module__sparsity", 0.2, 0.9),
    }

def cnn_ncp_branched_bins_training_space(trial, prefix):
    return {
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(f"{prefix}optimizer__weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
    }

def cnn_ncp_architecture_space(trial, prefix):
    return {
        # V4 parameters:
        f"{prefix}module__F1": trial.suggest_categorical(f"{prefix}module__F1", [4, 8, 16]),
        f"{prefix}module__D": trial.suggest_categorical(f"{prefix}module__D", [1, 2, 4]),
        f"{prefix}module__kernel_length": trial.suggest_int(f"{prefix}module__kernel_length", 64, 256, step=32),
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(f"{prefix}module__ncp_hidden_dim", 19, 128),
        f"{prefix}module__sparsity": trial.suggest_float(f"{prefix}module__sparsity", 0.2, 0.9),
    }


def cnn_ncp_training_space(trial, prefix):
    return {
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(f"{prefix}optimizer__weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
    }


def spp_ncp_architecture_space(trial, prefix):
    return {
        # f"{prefix}module__F1": trial.suggest_categorical(f"{prefix}module__F1", [4, 8, 16]),
        # f"{prefix}module__D": trial.suggest_categorical(f"{prefix}module__D", [1, 2, 4]),
        # f"{prefix}module__kernel_length": trial.suggest_int(f"{prefix}module__kernel_length", 64, 256, step=32),
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(f"{prefix}module__ncp_hidden_dim", 11, 16),
        f"{prefix}module__sparsity": trial.suggest_float(f"{prefix}module__sparsity", 0.4, 0.9),
        # f"{prefix}module__temporal_kernel_size": trial.suggest_int(f"{prefix}module__temporal_kernel_size", 3, 9,
        #                                                            step=2),
        # f"{prefix}module__temporal_stride": trial.suggest_int(f"{prefix}module__temporal_stride", 1, 4)
    }


def spp_ncp_training_space(trial, prefix):
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(f"{prefix}optimizer__weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5)
    }


def reegnet_architecture_space(trial, prefix):
    """
    Architecture parameter space for REEGNet with sanity-checked parameter ranges.
    
    Constraints:
    - F1 * D determines LSTM input size, so we keep these reasonable
    - pool1_kernel_size should not be too large to avoid very short sequences
    - kernel_length should be reasonable for temporal filtering
    - depthwise_kernel_length should be odd for proper padding
    """
    return {
        # Temporal convolution parameters (similar to EEGNet)
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 16]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 8, 64, step=8
        ),
        
        # Pooling parameter (keep reasonable to avoid very short sequences)
        f"{prefix}module__pool1_kernel_size": trial.suggest_categorical(
            f"{prefix}module__pool1_kernel_size", [2, 4, 8]
        ),
        
        # LSTM parameters
        f"{prefix}module__lstm_hidden_size": trial.suggest_categorical(
            f"{prefix}module__lstm_hidden_size", [8, 16, 32, 64]
        ),
        f"{prefix}module__lstm_num_layers": trial.suggest_int(
            f"{prefix}module__lstm_num_layers", 1, 3
        ),
        
        # Separable convolution parameter (must be odd for proper padding)
        f"{prefix}module__depthwise_kernel_length": trial.suggest_categorical(
            f"{prefix}module__depthwise_kernel_length", [3, 5, 7]
        ),
        
        # Dropout (architecture-level regularization)
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.1, 0.5
        ),
    }


def reegnet_training_space(trial, prefix):
    """
    Training parameter space for REEGNet.
    Note: drop_prob is optimized in architecture space, not here.
    """
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(f"{prefix}optimizer__weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
    }


def eegnet_architecture_space(trial, prefix):
    return {
        f"{prefix}module__F1": trial.suggest_categorical(f"{prefix}module__F1", [4, 8, 16]),
        f"{prefix}module__D": trial.suggest_categorical(f"{prefix}module__D", [1, 2, 4]),
        f"{prefix}module__kernel_length": trial.suggest_int(f"{prefix}module__kernel_length", 64, 256, step=32),
    }


def eegnet_training_space(trial, prefix):
    return {
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(f"{prefix}optimizer__weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
    }


def ctnet_architecture_space(trial, prefix):
    """Braindecode CTNet: https://braindecode.org/stable/generated/braindecode.models.CTNet.html"""
    return {
        f"{prefix}module__num_heads": trial.suggest_categorical(
            f"{prefix}module__num_heads", [2, 4, 8]
        ),
        f"{prefix}module__embed_dim": trial.suggest_categorical(
            f"{prefix}module__embed_dim", [32, 40, 48, 64]
        ),
        f"{prefix}module__num_layers": trial.suggest_int(f"{prefix}module__num_layers", 4, 8),
        f"{prefix}module__kernel_size": trial.suggest_int(
            f"{prefix}module__kernel_size", 32, 128, step=16
        ),
    }


def ctnet_training_space(trial, prefix):
    return {
        f"{prefix}module__cnn_drop_prob": trial.suggest_float(
            f"{prefix}module__cnn_drop_prob", 0.1, 0.5
        ),
        f"{prefix}module__final_drop_prob": trial.suggest_float(
            f"{prefix}module__final_drop_prob", 0.2, 0.6
        ),
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
    }


# Perform two-stage hp opt. with augmentation or perturbation
def alternate_two_stage_optuna(
        model_fn,
        model_name,
        X,
        y,
        metadata,
        resample,
        seed,
        mode,
        noise_dict,
        output_root="optuna_results",
        arch_trials=10,
        train_trials=10,
        dataset=None,
        eval_mode=None,
        paradigm=None,
        dataset_obj=None,
        train_subjects=None,
        subject_chunk_size=None
):
    noise_type = noise_dict["noise_type"]
    intensity = noise_dict["intensity"]
    print("\n[Stage 1] Architecture Search")
    arch_param_space_fn = get_model_architecture_space(model_name)

    arch_params, _ = alternate_optuna_stage(
        model_fn=model_fn,
        model_name=model_name,
        stage_name="architecture",
        X=X,
        y=y,
        metadata=metadata,
        param_space_fn=arch_param_space_fn,
        mode=mode,
        noise_dict=noise_dict,
        resample=resample,
        n_trials=arch_trials,
        seed=seed,
        output_root=output_root,
        dataset=dataset,
        eval_mode=eval_mode,
        paradigm=paradigm,
        dataset_obj=dataset_obj,
        train_subjects=train_subjects,
        subject_chunk_size=subject_chunk_size
    )

    print("\n[Stage 2] Training Optimization")
    training_param_space_fn = get_model_training_space(model_name)

    def training_space_with_arch(trial, prefix):
        # Freeze architecture params
        arch_prefixed = {k: v for k, v in arch_params.items()}
        tuning_params = training_param_space_fn(trial, prefix)
        arch_prefixed.update(tuning_params)
        return arch_prefixed

    final_params, final_score = alternate_optuna_stage(
        model_fn=model_fn,
        model_name=model_name,
        stage_name="training",
        X=X,
        y=y,
        metadata=metadata,
        param_space_fn=training_space_with_arch,
        mode=mode,
        noise_dict=noise_dict,
        resample=resample,
        n_trials=train_trials,
        seed=seed,
        output_root=output_root,
        dataset=dataset,
        eval_mode=eval_mode,
        paradigm=paradigm,
        dataset_obj=dataset_obj,
        train_subjects=train_subjects,
        subject_chunk_size=subject_chunk_size
    )
    for k, v in arch_params.items():
        final_params[k] = v

    return final_params, final_score


def run_two_stage_optuna(
        model_fn,
        model_name,
        X,
        y,
        metadata,
        resample=250.0,
        seed=42,
        output_root="optuna_results",
        arch_trials=10,
        train_trials=10,
        perturbed=False,
        dataset=None,
        eval_mode=None,
        paradigm=None,
        dataset_obj=None,
        train_subjects=None,
        subject_chunk_size=None
):
    print("\n[Stage 1] Architecture Search")
    arch_param_space_fn = get_model_architecture_space(model_name)
    arch_params, _ = run_optuna_stage(
        model_fn=model_fn,
        model_name=model_name,
        stage_name="architecture",
        X=X,
        y=y,
        metadata=metadata,
        param_space_fn=arch_param_space_fn,
        resample=resample,
        n_trials=arch_trials,
        seed=seed,
        perturbed=perturbed,
        output_root=output_root,
        eval_mode=eval_mode,
        paradigm=paradigm,
        dataset_obj=dataset_obj,
        train_subjects=train_subjects,
        subject_chunk_size=subject_chunk_size
    )

    print("\n[Stage 2] Training Optimization")
    training_param_space_fn = get_model_training_space(model_name)

    def training_space_with_arch(trial, prefix):
        # Freeze architecture params
        arch_prefixed = {k: v for k, v in arch_params.items()}
        tuning_params = training_param_space_fn(trial, prefix)
        arch_prefixed.update(tuning_params)
        return arch_prefixed

    final_params, final_score = run_optuna_stage(
        model_fn=model_fn,
        model_name=model_name,
        stage_name="training",
        X=X,
        y=y,
        metadata=metadata,
        param_space_fn=training_space_with_arch,
        resample=resample,
        n_trials=train_trials,
        seed=seed,
        perturbed=perturbed,
        output_root=output_root,
        eval_mode=eval_mode,
        paradigm=paradigm,
        dataset_obj=dataset_obj,
        train_subjects=train_subjects,
        subject_chunk_size=subject_chunk_size
    )
    for k, v in arch_params.items():
        final_params[k] = v

    return final_params, final_score


def improved_two_stage_optuna(
        model_fn,
        model_name,
        X,
        y,
        metadata,
        resample=250.0,
        seed=42,
        output_root="optuna_results",
        arch_trials=15,
        train_trials=15,
        joint_trials=10,
        use_joint_optimization=True
):
    """
    Improved two-stage hyperparameter optimization with optional joint optimization.
    
    Args:
        use_joint_optimization: If True, adds a third stage for joint optimization
        joint_trials: Number of trials for joint optimization stage
    """
    print("\n[Stage 1] Architecture Search")
    arch_param_space_fn = get_model_architecture_space(model_name)
    arch_params, arch_score = run_optuna_stage(
        model_fn=model_fn,
        model_name=model_name,
        stage_name="architecture",
        X=X,
        y=y,
        metadata=metadata,
        param_space_fn=arch_param_space_fn,
        resample=resample,
        n_trials=arch_trials,
        seed=seed,
        output_root=output_root
    )

    print(f"\n[Stage 2] Training Optimization")
    training_param_space_fn = get_model_training_space(model_name)

    def training_space_with_arch(trial, prefix):
        # Freeze architecture params
        arch_prefixed = {k: v for k, v in arch_params.items()}
        tuning_params = training_param_space_fn(trial, prefix)
        arch_prefixed.update(tuning_params)
        return arch_prefixed

    train_params, train_score = run_optuna_stage(
        model_fn=model_fn,
        model_name=model_name,
        stage_name="training",
        X=X,
        y=y,
        metadata=metadata,
        param_space_fn=training_space_with_arch,
        resample=resample,
        n_trials=train_trials,
        seed=seed,
        output_root=output_root
    )

    # Combine parameters
    final_params = {**arch_params, **train_params}
    
    if use_joint_optimization:
        print(f"\n[Stage 3] Joint Optimization")
        
        def joint_space(trial, prefix):
            # Allow both architecture and training parameters to be optimized together
            arch_params_joint = arch_param_space_fn(trial, prefix)
            train_params_joint = training_param_space_fn(trial, prefix)
            return {**arch_params_joint, **train_params_joint}
        
        joint_params, joint_score = run_optuna_stage(
            model_fn=model_fn,
            model_name=model_name,
            stage_name="joint",
            X=X,
            y=y,
            metadata=metadata,
            param_space_fn=joint_space,
            resample=resample,
            n_trials=joint_trials,
            seed=seed,
            output_root=output_root
        )
        
        # Use the best of the three stages
        scores = {
            "architecture_only": arch_score,
            "training_only": train_score, 
            "joint": joint_score
        }
        
        best_stage = max(scores, key=scores.get)
        print(f"\nBest stage: {best_stage} (score: {scores[best_stage]:.4f})")
        
        if best_stage == "joint":
            final_params = joint_params
            final_score = joint_score
        elif best_stage == "training_only":
            final_params = {**arch_params, **train_params}
            final_score = train_score
        else:
            final_params = arch_params
            final_score = arch_score
    else:
        final_score = train_score

    return final_params, final_score

def multi_objective_optuna_stage(
        model_fn,
        model_name,
        stage_name,
        X,
        y,
        metadata,
        param_space_fn,
        resample=250.0,
        n_trials=25,
        seed=42,
        output_root="optuna_results",
        objectives=["accuracy", "efficiency"]
):
    """
    Multi-objective optimization stage that considers both accuracy and efficiency.
    """
    train_mask = metadata["session"] == "0train"
    X_train = X[train_mask]
    y_train = y[train_mask]

    if len(X_train) < 10:
        raise ValueError("Too few training samples for session 0.")

    param_prefix = ""
    if resample is None:
        resample = 250.0

    sfreq = float(resample)

    # Determine n_chans and n_times from input data
    n_chans = X_train.shape[1]
    n_times = X_train.shape[2] if len(X_train.shape) > 2 else int(resample * 4)
    
    # Determine n_outputs from y if not provided
    n_outputs = len(np.unique(y_train))
    model = model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, sfreq=sfreq)
    model.verbose = 1
    model.callbacks = []
    model.train_split = None

    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        params[f"{param_prefix}train_split"] = None
        params[f"{param_prefix}verbose"] = 1
        params[f"{param_prefix}callbacks"] = []

        # Define model
        model.set_params(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        
        # Measure both accuracy and efficiency
        start_time = time.time()
        accuracy_score, std_score, mean_time = enhanced_cv_training_loop(
            model, cv, X_train, y_train, trial=trial
        )
        total_time = time.time() - start_time
        
        # Calculate efficiency metric (inverse of time)
        efficiency_score = 1.0 / (total_time + 1e-6)  # Avoid division by zero
        
        # Return multiple objectives
        if len(objectives) == 1:
            return accuracy_score
        else:
            return accuracy_score, efficiency_score

    output_dir = os.path.join(output_root, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create multi-objective study
    if len(objectives) > 1:
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            pruner=optuna.pruners.MedianPruner()
        )
    else:
        study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    # Save study
    study_path = os.path.join(output_dir, "optuna_study.pkl")
    import joblib
    joblib.dump(study, study_path)

    # Save best parameters
    if len(objectives) > 1:
        # For multi-objective, save Pareto front
        pareto_front = study.best_trials
        best_params = []
        for trial in pareto_front:
            best_params.append({
                "params": trial.params,
                "values": trial.values
            })
    else:
        best_params = study.best_params
        best_values = study.best_value

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        if len(objectives) > 1:
            json.dump({
                "pareto_front": best_params,
                "objectives": objectives
            }, f, indent=2)
        else:
            json.dump({
                "best_score": best_values,
                "best_params": best_params
            }, f, indent=2)

    return best_params, study

def adaptive_improved_cnncfc_architecture_space(trial, prefix, previous_best=None):
    """
    Adaptive architecture parameter space that can use previous best results.
    """
    # Base parameter space
    params = {
        # CfC core parameters
        f"{prefix}module__ncp_hidden_dim": trial.suggest_categorical(
            f"{prefix}module__ncp_hidden_dim", [8, 16, 32, 48, 64, 96]
        ),
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.05, 0.6
        ),
        
        # CNN feature extraction parameters
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16, 20]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4, 8]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 32, 512, step=32
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7, 9]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [1, 2, 4, 6, 8]
        ),
        
        # Sequence length control
        f"{prefix}module__max_seq_length": trial.suggest_categorical(
            f"{prefix}module__max_seq_length", [100, 150, 200, 250, 300, 400]
        ),
        
        # Sparsity for CfC
        f"{prefix}module__sparsity": trial.suggest_float(
            f"{prefix}module__sparsity", 0.3, 0.95
        ),
        
        # NEW: CfC-specific parameters that were previously hardcoded
        f"{prefix}module__mixed_memory": trial.suggest_categorical(
            f"{prefix}module__mixed_memory", [True, False]
        ),
        f"{prefix}module__mode": trial.suggest_categorical(
            f"{prefix}module__mode", ["default", "pure", "no_gate"]
        ),
        f"{prefix}module__activation": trial.suggest_categorical(
            f"{prefix}module__activation", ["lecun_tanh", "tanh", "relu", "sigmoid", "gelu"]
        ),
        f"{prefix}module__backbone_units": trial.suggest_categorical(
            f"{prefix}module__backbone_units", [64, 128, 256, 512, 1024]
        ),
        f"{prefix}module__backbone_layers": trial.suggest_categorical(
            f"{prefix}module__backbone_layers", [1, 2, 3, 4, 5]
        ),
        f"{prefix}module__backbone_dropout": trial.suggest_float(
            f"{prefix}module__backbone_dropout", 0.0, 0.7
        )
    }
    
    # Add conditional parameters based on previous best
    if previous_best and trial.number > 10:
        # Focus search around previous best values
        best_ncp = previous_best.get(f"{prefix}module__ncp_hidden_dim", 32)
        best_f1 = previous_best.get(f"{prefix}module__F1", 8)
        
        # Adjust search space based on previous performance
        if best_ncp > 48:
            params[f"{prefix}module__ncp_hidden_dim"] = trial.suggest_categorical(
                f"{prefix}module__ncp_hidden_dim", [32, 48, 64, 96]
            )
        if best_f1 > 12:
            params[f"{prefix}module__F1"] = trial.suggest_categorical(
                f"{prefix}module__F1", [8, 12, 16, 20]
            )
    
    return params

def adaptive_improved_cnncfc_training_space(trial, prefix, previous_best=None):
    """
    Adaptive training parameter space with learning rate scheduling.
    """
    params = {
        # Optimizer parameters with wider ranges
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        
        # Training parameters
        # f"{prefix}max_epochs": trial.suggest_int(
        #     f"{prefix}max_epochs", 15, 200
        # ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
        
        # Learning rate scheduling
        f"{prefix}lr_scheduler": trial.suggest_categorical(
            f"{prefix}lr_scheduler", ["none", "step", "cosine", "plateau"]
        ),
        f"{prefix}lr_step_size": trial.suggest_int(
            f"{prefix}lr_step_size", 10, 100
        ),
        f"{prefix}lr_gamma": trial.suggest_float(
            f"{prefix}lr_gamma", 0.1, 0.9
        ),
    }
    
    # Conditional parameters
    if params[f"{prefix}lr_scheduler"] == "none":
        params.pop(f"{prefix}lr_step_size")
        params.pop(f"{prefix}lr_gamma")
    
    return params


def adaptive_cnn_wiredcfc_architecture_space(trial, prefix, previous_best=None):
    """
    Adaptive architecture parameter space for CNNWiredCfC models.
    """
    params = {
        # CNN feature extraction parameters with wider ranges
        f"{prefix}module__F1": trial.suggest_categorical(
            f"{prefix}module__F1", [4, 8, 12, 16, 20, 24, 32]
        ),
        f"{prefix}module__D": trial.suggest_categorical(
            f"{prefix}module__D", [1, 2, 4, 8, 16]
        ),
        f"{prefix}module__kernel_length": trial.suggest_int(
            f"{prefix}module__kernel_length", 32, 1024, step=32
        ),
        
        # Temporal processing parameters
        f"{prefix}module__temporal_kernel_size": trial.suggest_categorical(
            f"{prefix}module__temporal_kernel_size", [3, 5, 7, 9, 11]
        ),
        f"{prefix}module__temporal_stride": trial.suggest_categorical(
            f"{prefix}module__temporal_stride", [1, 2, 4, 6, 8, 10]
        ),
        
        # Sequence length control with wider range
        f"{prefix}module__max_seq_length": trial.suggest_categorical(
            f"{prefix}module__max_seq_length", [100, 150, 200, 250, 300, 400, 500, 750]
        ),
        
        # Dropout for regularization
        f"{prefix}module__drop_prob": trial.suggest_float(
            f"{prefix}module__drop_prob", 0.05, 0.7
        ),
    }
    
    # Add conditional parameters based on previous best
    if previous_best and trial.number > 10:
        # Focus search around previous best values
        best_f1 = previous_best.get(f"{prefix}module__F1", 8)
        best_kernel = previous_best.get(f"{prefix}module__kernel_length", 128)
        
        # Adjust search space based on previous performance
        if best_f1 > 16:
            params[f"{prefix}module__F1"] = trial.suggest_categorical(
                f"{prefix}module__F1", [12, 16, 20, 24, 32]
            )
        if best_kernel > 256:
            params[f"{prefix}module__kernel_length"] = trial.suggest_int(
                f"{prefix}module__kernel_length", 128, 1024, step=64
            )
    
    return params


def adaptive_cnn_wiredcfc_training_space(trial, prefix, previous_best=None):
    """
    Adaptive training parameter space for CNNWiredCfC models.
    """
    params = {
        # Optimizer parameters with wider ranges
        f"{prefix}optimizer__lr": trial.suggest_loguniform(
            f"{prefix}optimizer__lr", 1e-6, 1e-2
        ),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(
            f"{prefix}optimizer__weight_decay", 1e-6, 1e-2
        ),
        
        # Training parameters
        # f"{prefix}max_epochs": trial.suggest_int(
        #     f"{prefix}max_epochs", 15, 200
        # ),
        f"{prefix}batch_size": trial.suggest_categorical(
            f"{prefix}batch_size", [4, 8, 16, 32, 64]
        ),
        
        # Learning rate scheduling
        f"{prefix}lr_scheduler": trial.suggest_categorical(
            f"{prefix}lr_scheduler", ["none", "step", "cosine", "plateau"]
        ),
        f"{prefix}lr_step_size": trial.suggest_int(
            f"{prefix}lr_step_size", 10, 100
        ),
        f"{prefix}lr_gamma": trial.suggest_float(
            f"{prefix}lr_gamma", 0.1, 0.9
        ),
    }
    
    # Conditional parameters
    if params[f"{prefix}lr_scheduler"] == "none":
        params.pop(f"{prefix}lr_step_size")
        params.pop(f"{prefix}lr_gamma")
    
    return params


def get_adaptive_model_architecture_space(model_name):
    """Get adaptive architecture parameter spaces."""
    # Check if this is a wiredcfc architecture model
    if model_name.startswith("wiredcfc_arch"):
        return adaptive_cnn_wiredcfc_architecture_space
    
    # Check if this is a branched_wiredcfc architecture model
    if model_name.startswith("branched_wiredcfc_arch"):
        return branched_wiredcfc_architecture_space  # Use standard space for now
    
    adaptive_registry = {
        "improved_cnncfc": adaptive_improved_cnncfc_architecture_space,
        # Add other models as needed
    }
    return adaptive_registry.get(model_name, get_model_architecture_space(model_name))


def get_adaptive_model_training_space(model_name):
    """Get adaptive training parameter spaces."""
    # Check if this is a wiredcfc architecture model
    if model_name.startswith("wiredcfc_arch"):
        return adaptive_cnn_wiredcfc_training_space
    
    # Check if this is a branched_wiredcfc architecture model
    if model_name.startswith("branched_wiredcfc_arch"):
        return branched_wiredcfc_training_space  # Use standard space for now
    
    adaptive_registry = {
        "improved_cnncfc": adaptive_improved_cnncfc_training_space,
        # Add other models as needed
    }
    return adaptive_registry.get(model_name, get_model_training_space(model_name))

def analyze_optimization_results(output_root, model_name, stage_name):
    """
    Comprehensive analysis of optimization results with visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from optuna.visualization import (
        plot_optimization_history, plot_param_importances,
        plot_parallel_coordinate, plot_contour
    )
    
    study_path = os.path.join(output_root, stage_name, "optuna_study.pkl")
    if not os.path.exists(study_path):
        print(f"Study not found at {study_path}")
        return
    
    import joblib
    study = joblib.load(study_path)
    
    # Create analysis directory
    analysis_dir = os.path.join(output_root, stage_name, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate visualizations
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(analysis_dir, "optimization_history.html"))
        
        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_html(os.path.join(analysis_dir, "param_importances.html"))
        
        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_html(os.path.join(analysis_dir, "parallel_coordinate.html"))
        
        # Contour plots for top parameters
        if len(study.best_params) >= 2:
            top_params = list(study.best_params.keys())[:2]
            fig = plot_contour(study, params=top_params)
            fig.write_html(os.path.join(analysis_dir, "contour_plot.html"))
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Statistical analysis
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(analysis_dir, "trials_data.csv"), index=False)
    
    # Summary statistics
    summary = {
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "mean_value": trials_df["value"].mean(),
        "std_value": trials_df["value"].std(),
        "min_value": trials_df["value"].min(),
        "max_value": trials_df["value"].max(),
    }
    
    with open(os.path.join(analysis_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis saved to: {analysis_dir}")
    return summary

def compare_optimization_stages(output_root, model_name):
    """
    Compare results across different optimization stages.
    """
    stages = ["architecture", "training", "joint"]
    results = {}
    
    for stage in stages:
        stage_dir = os.path.join(output_root, stage)
        if os.path.exists(stage_dir):
            best_params_file = os.path.join(stage_dir, "best_params.json")
            if os.path.exists(best_params_file):
                with open(best_params_file, "r") as f:
                    results[stage] = json.load(f)
    
    # Create comparison plot
    if len(results) > 1:
        import matplotlib.pyplot as plt
        
        stages_list = list(results.keys())
        scores = [results[stage]["best_score"] for stage in stages_list]
        
        plt.figure(figsize=(10, 6))
        plt.bar(stages_list, scores)
        plt.title(f"Optimization Stage Comparison - {model_name}")
        plt.ylabel("Best Score")
        plt.xlabel("Optimization Stage")
        plt.ylim(0, 1)
        
        # Add score labels on bars
        for i, score in enumerate(scores):
            plt.text(i, score + 0.01, f"{score:.4f}", 
                    ha='center', va='bottom')
        
        comparison_path = os.path.join(output_root, "stage_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Stage comparison saved to: {comparison_path}")
    
    return results

def generate_optimization_report(output_root, model_name):
    """
    Generate a comprehensive optimization report.
    """
    report_path = os.path.join(output_root, "optimization_report.md")
    
    with open(report_path, "w") as f:
        f.write(f"# Hyperparameter Optimization Report\n\n")
        f.write(f"**Model**: {model_name}\n")
        f.write(f"**Output Directory**: {output_root}\n\n")
        
        # Analyze each stage
        stages = ["architecture", "training", "joint"]
        for stage in stages:
            stage_dir = os.path.join(output_root, stage)
            if os.path.exists(stage_dir):
                f.write(f"## {stage.title()} Stage\n\n")
                
                # Get best parameters
                best_params_file = os.path.join(stage_dir, "best_params.json")
                if os.path.exists(best_params_file):
                    with open(best_params_file, "r") as bp:
                        best_data = json.load(bp)
                    
                    f.write(f"**Best Score**: {best_data.get('best_score', 'N/A'):.4f}\n\n")
                    
                    if 'best_params' in best_data:
                        f.write("**Best Parameters**:\n")
                        for param, value in best_data['best_params'].items():
                            f.write(f"- {param}: {value}\n")
                        f.write("\n")
                
                # Run analysis
                try:
                    summary = analyze_optimization_results(output_root, model_name, stage)
                    if summary:
                        f.write(f"**Trials**: {summary['n_trials']}\n")
                        f.write(f"**Mean Score**: {summary['mean_value']:.4f}\n")
                        f.write(f"**Std Score**: {summary['std_value']:.4f}\n\n")
                except Exception as e:
                    f.write(f"Analysis failed: {e}\n\n")
        
        # Stage comparison
        f.write("## Stage Comparison\n\n")
        comparison_results = compare_optimization_stages(output_root, model_name)
        if comparison_results:
            f.write("| Stage | Best Score |\n")
            f.write("|------|------------|\n")
            for stage, data in comparison_results.items():
                score = data.get('best_score', 'N/A')
                f.write(f"| {stage} | {score:.4f} |\n")
    
    print(f"Optimization report saved to: {report_path}")
