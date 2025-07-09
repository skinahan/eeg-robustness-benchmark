import math
import os
import json
import optuna
import numpy as np
import pandas as pd
import sklearn
import torch
from moabb.evaluations import WithinSessionSplitter
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import StratifiedKFold, GroupKFold
from optuna.integration import SkorchPruningCallback
from skorch.dataset import ValidSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter


def format_params(param_block, prefix):
    module_params = ['ncp_hidden_dim', 'sparsity', 'temporal_kernel_size', 'temporal_stride', 'drop_prob']
    optimizer_params = ['lr', 'weight_decay']
    module_prefix = f"{prefix}module__"
    optim_prefix = f"{prefix}optimizer__"
    final_params = {}
    for k, v in param_block.items():
        if k in module_params:
            final_params[f"{module_prefix}{k}"] = v
        elif k in optimizer_params:
            final_params[f"{optim_prefix}{k}"] = v
        else:
            final_params[k] = v
    return final_params


def unified_cv_training_loop_method(model, cv, X_train, y_train, trial=None, groups=None):
    fold_scores = []
    for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups)):
        print(f"Fold {i}:")
        X_train_part, y_train_part = X_train[train_idx], y_train[train_idx]
        X_valid_part, y_valid_part = X_train[valid_idx], y_train[valid_idx]
        # Fit on training fold
        torch.use_deterministic_algorithms(False)
        model.fit(X_train_part, y_train_part)

        # Evaluate on held-out validation set
        y_pred = model.predict_proba(X_valid_part)[:, 1]
        auc = roc_auc_score(y_valid_part, y_pred)
        print(f"Fold {i} auc: {auc}")
        fold_scores.append(auc)
        if trial is not None:
            trial.report(auc, i)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
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
    train_mask = metadata["session"] == "0train"
    X_train = X[train_mask]
    y_train = y[train_mask]
    # meta_train = metadata[train_mask].reset_index(drop=True)

    if len(X_train) < 10:
        raise ValueError("Too few training samples for session 0.")

    param_prefix = "base_pipeline__" if perturbed else ""
    if resample is None:
        resample = 250.0

    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        # Define model
        model = model_fn(n_chans=22, n_times=int(resample * 4), n_outputs=2)
        model.set_params(**params)
        model.max_epochs = 200
        model.verbose = 1
        model.train_split = None
        model.callbacks = []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return unified_cv_training_loop_method(model, cv, X_train, y_train, trial=trial)

    output_dir = os.path.join(output_root, model_name, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials)

    study_path = os.path.join(output_dir, "optuna_study.pkl")
    import joblib
    joblib.dump(study, study_path)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump({"best_score": study.best_value, "best_params": study.best_params}, f, indent=2)

    try:
        plot_optimization_history(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_history.html"))
        plot_param_importances(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_importances.html"))
    except Exception as e:
        print(f"Plotting failed: {e}")

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
    train_mask = metadata["session"] == "0train"
    X_train = X[train_mask]
    y_train = y[train_mask]

    if len(X_train) < 10:
        raise ValueError("Too few training samples for session 0.")
    add_prefix = (mode == "perturb")
    param_prefix = "base_pipeline__" if add_prefix else ""
    if resample is None:
        resample = 250.0

    def objective(trial):
        # Sample hyperparameters
        params = param_space_fn(trial, param_prefix)
        # Define model
        model = model_fn(n_chans=22, n_times=int(resample * 4), n_outputs=2)
        model.set_params(**params)
        model.max_epochs = 100
        model.verbose = 1
        model.train_split = None
        model.callbacks = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        # Augmentation mode: Supplement clean input data with contaminated samples
        groups = None
        if mode == 'augment':
            augmenter = EEGNoiseAugmentor(noise_type=noise_type, intensity=intensity, seed=seed)
            concat_aug = ConcatenatedNoiseAugmenter(augmenter)
            cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
            X_aug, y_aug, groups = concat_aug.transform_with_groups(X_train, y_train)
            return unified_cv_training_loop_method(model, cv, X_aug, y_aug, trial=trial, groups=groups)
        return unified_cv_training_loop_method(model, cv, X_train, y_train, trial=trial)

    output_dir = os.path.join(output_root, model_name, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials)

    study_path = os.path.join(output_dir, "optuna_study.pkl")
    import joblib
    joblib.dump(study, study_path)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump({"best_score": study.best_value, "best_params": study.best_params}, f, indent=2)

    try:
        plot_optimization_history(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_history.html"))
        plot_param_importances(study).write_html(os.path.join(output_dir, f"optuna_{stage_name}_importances.html"))
    except Exception as e:
        print(f"Plotting failed: {e}")

    return study.best_params, study.best_value


def get_model_architecture_space(model_name):
    architecture_registry = {
        "eegnet": eegnet_architecture_space,
        "reegnet": reegnet_architecture_space,
        "cnn_ncp": cnn_ncp_architecture_space,
        "spp_ncp": spp_ncp_architecture_space
    }
    return architecture_registry[model_name]


def get_model_training_space(model_name):
    training_registry = {
        "eegnet": eegnet_training_space,
        "reegnet": reegnet_training_space,
        "cnn_ncp": cnn_ncp_training_space,
        "spp_ncp": spp_ncp_training_space
    }
    return training_registry[model_name]


def cnn_ncp_architecture_space(trial, prefix):
    return {
        # f"{prefix}module__F1": trial.suggest_categorical(f"{prefix}module__F1", [4, 8, 16]),
        # f"{prefix}module__D": trial.suggest_categorical(f"{prefix}module__D", [1, 2, 4]),
        f"{prefix}module__kernel_length": trial.suggest_int(f"{prefix}module__kernel_length", 64, 256, step=32),
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(f"{prefix}module__ncp_hidden_dim", 11, 16),
        # f"{prefix}module__ncp_hidden_dim": trial.suggest_int(f"{prefix}module__ncp_hidden_dim", 7, 12),
        f"{prefix}module__sparsity": trial.suggest_float(f"{prefix}module__sparsity", 0.4, 0.9),
        f"{prefix}module__temporal_kernel_size": trial.suggest_int(f"{prefix}module__temporal_kernel_size", 3, 9,
                                                                   step=2),
        f"{prefix}module__temporal_stride": trial.suggest_int(f"{prefix}module__temporal_stride", 1, 4)
    }


def cnn_ncp_training_space(trial, prefix):
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5)
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
        f"{prefix}optimizer__lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5)
    }


def reegnet_architecture_space(trial, prefix):
    return {
        f"{prefix}module__lstm_hidden_size": trial.suggest_categorical(f"{prefix}module__lstm_hidden_size",
                                                                       [8, 16, 32, 64]),
    }


def reegnet_training_space(trial, prefix):
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
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
        f"{prefix}optimizer__lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
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
    # Perturbation mode: Replace clean input data with contaminated samples
    if mode == 'perturb':
        def wrapped_model_fn(model_generator):
            base_model = model_generator(n_chans=22, n_times=int(resample * 4), n_outputs=2)
            return TrainOnlyNoiseClassifier(
                base_pipeline=base_model,
                noise_type=noise_type,
                intensity=intensity,
                seed=seed
            )

        og_model_fn = model_fn
        model_fn = wrapped_model_fn(og_model_fn)

    arch_params, _ = alternate_optuna_stage(
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
        perturbed=True,
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
        perturbed=True,
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
