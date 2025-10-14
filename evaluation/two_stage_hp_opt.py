import math
import os
import json
import time

import optuna
import numpy as np
import pandas as pd
import sklearn
import torch
import joblib
from moabb.evaluations import WithinSessionSplitter
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import StratifiedKFold, GroupKFold
from optuna.integration import SkorchPruningCallback
from skorch.dataset import ValidSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter


def format_params(param_block, prefix):
    module_params = ['F1', 'D', 'kernel_length', 'lstm_hidden_size','ncp_hidden_dim', 'sparsity', 'temporal_kernel_size', 'temporal_stride', 'drop_prob', 'n_modules', 'rewiring_prob', 'max_seq_length']
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
    fold_scores = []
    fold_times = []
    
    for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups)):
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
def unified_cv_training_loop_method(model, cv, X_train, y_train, trial=None, groups=None):
    fold_scores = []
    for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups)):
        # print(f"Fold {i}:")
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
        # print(f"Fold {i} auc: {auc}")
        fold_scores.append(auc)
        # if trial is not None:
        #     trial.report(auc, i)

        #     # Handle pruning based on the intermediate value.
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()
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
        output_root="optuna_results"
):
    # In the old version we explicitly wanted to only use 0train for hyperparameter optimization. That is no longer the case.
    # In the current code version, we can expect X and y to be split before run_optuna_stage is called.
    X_train = X
    y_train = y
    metadata_train = metadata

    if len(X_train) < 10:
        print(f"Too few training samples: {len(X_train)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"metadata shape: {metadata.shape}")
        print(f"metadata: {metadata}")
        raise ValueError("Too few training samples for session 0.")

    param_prefix = "base_pipeline__" if perturbed else ""
    if resample is None:
        resample = 250.0

    check_time = False
    model = model_fn(n_chans=22, n_times=int(resample * 4), n_outputs=2)
    # model.verbose = 0
    # model.callbacks = []
    # model.train_split = None
    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        # params[f"{param_prefix}train_split"] = None
        params[f"{param_prefix}verbose"] = 0
        # params[f"{param_prefix}callbacks"] = []
        
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
        roc_auc_score = unified_cv_training_loop_method(model, cv, X_train, y_train, trial=trial)
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
        output_root="optuna_results"
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
        resample = 250.0
    model = model_fn(n_chans=22, n_times=int(resample * 4), n_outputs=2)
    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        params[f"{param_prefix}train_split"] = None
        from globals import DEFAULT_MAX_EPOCHS
        params[f"{param_prefix}max_epochs"] = DEFAULT_MAX_EPOCHS  # Fixed reasonable value
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
        return unified_cv_training_loop_method(model, cv, X_obj, y_obj, trial=trial, groups=groups)

    output_dir = os.path.join(output_root, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")

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
    # Check if this is a wiredcfc architecture model
    if model_name.startswith("wiredcfc_arch"):
        return cnn_wiredcfc_architecture_space
    
    architecture_registry = {
        "eegnet": eegnet_architecture_space,
        "reegnet": reegnet_architecture_space,
        "cnn_ncp": cnn_ncp_architecture_space,
        "cnn_ncp_v2": cnn_ncp_architecture_space,
        "cnncfc_v2": improved_cnncfc_architecture_space,  # CNNCfCv2 uses the same space as improved_cnncfc
        "cnncfc_compact": cnncfc_compact_architecture_space,
        "spp_ncp": spp_ncp_architecture_space,
        "cnn_smallworld": cnn_smallworld_architecture_space,
        "cnn_ncp_branched_bins": cnn_ncp_branched_bins_architecture_space,
    }
    return architecture_registry[model_name]


def get_model_training_space(model_name):
    # Check if this is a wiredcfc architecture model
    if model_name.startswith("wiredcfc_arch"):
        return cnn_wiredcfc_training_space
    
    training_registry = {
        "eegnet": eegnet_training_space,
        "reegnet": reegnet_training_space,
        "cnn_ncp": cnn_ncp_training_space,
        "cnn_ncp_v2": cnn_ncp_training_space,
        "cnncfc_v2": improved_cnncfc_training_space,  # CNNCfCv2 uses the same space as improved_cnncfc
        "cnncfc_compact": cnncfc_compact_training_space,
        "spp_ncp": spp_ncp_training_space,
        "cnn_smallworld": cnn_smallworld_training_space,
        "cnn_ncp_branched_bins": cnn_ncp_branched_bins_training_space,
    }
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
    return {
        f"{prefix}module__lstm_hidden_size": trial.suggest_categorical(f"{prefix}module__lstm_hidden_size",
                                                                       [8, 16, 32, 64]),
    }


def reegnet_training_space(trial, prefix):
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform(f"{prefix}optimizer__lr", 1e-6, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform(f"{prefix}optimizer__weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical(f"{prefix}batch_size", [4, 8, 16, 32, 64]),
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5),
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
        train_trials=10
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
        output_root=output_root
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
        output_root=output_root
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
        perturbed=False
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
        output_root=output_root
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
        output_root=output_root
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

    model = model_fn(n_chans=22, n_times=int(resample * 4), n_outputs=2)
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
