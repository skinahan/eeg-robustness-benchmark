import os
import json
import optuna
import numpy as np
import pandas as pd
import sklearn
from moabb.evaluations import WithinSessionSplitter
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import StratifiedKFold


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
        augmented=False,
        output_root="optuna_results"
):
    train_mask = metadata["session"] == "0train"
    X_train = X[train_mask]
    y_train = y[train_mask]
    # meta_train = metadata[train_mask].reset_index(drop=True)

    if len(X_train) < 10:
        raise ValueError("Too few training samples for session 0.")

    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    param_prefix = "base_pipeline__" if augmented else ""
    if resample is None:
        resample = 250.0

    # TODO:
    #  - optuna multi-objective study?
    #  - partial fit / intermediate results for optuna pruning?
    def objective(trial):
        params = param_space_fn(trial, param_prefix)

        model = model_fn(n_chans=22, n_times=int(resample * 4), n_outputs=2)
        model.set_params(**params)
        model.max_epochs = 100
        model.train_split = None
        model.callbacks = []
        model.verbose = 0
        results = sklearn.model_selection.cross_validate(model, X_train, y_train, scoring="roc_auc", cv=splitter, n_jobs=3, verbose=0, return_train_score=True,error_score='raise')
        return results["test_score"].mean()

    output_dir = os.path.join(output_root, model_name, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    # if stage_name == 'architecture':
    #     start_params = {
    #         f"{param_prefix}module__ncp_hidden_dim": 11,
    #         f"{param_prefix}module__sparsity": 0.6,
    #         f"{param_prefix}module__temporal_kernel_size": 3,
    #         f"{param_prefix}module__temporal_stride": 2,
    #     }
    #     study.enqueue_trial(start_params)
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
        "cnn_ncp": cnn_ncp_architecture_space
    }
    return architecture_registry[model_name]

def get_model_training_space(model_name):
    training_registry = {
        "eegnet": eegnet_training_space,
        "reegnet": reegnet_training_space,
        "cnn_ncp": cnn_ncp_training_space
    }
    return training_registry[model_name]

def cnn_ncp_architecture_space(trial, prefix):
    return {
        f"{prefix}module__F1": trial.suggest_categorical(f"{prefix}module__F1", [4, 8, 16]),
        f"{prefix}module__D": trial.suggest_categorical(f"{prefix}module__D", [1, 2, 4]),
        f"{prefix}module__kernel_length": trial.suggest_int(f"{prefix}module__kernel_length", 64, 256, step=32),
        f"{prefix}module__ncp_hidden_dim": trial.suggest_int(f"{prefix}module__ncp_hidden_dim", 11, 24),
        f"{prefix}module__sparsity": trial.suggest_float(f"{prefix}module__sparsity", 0.4, 0.9),
        # f"{prefix}module__temporal_kernel_size": trial.suggest_int(f"{prefix}module__temporal_kernel_size", 3, 9,
        #                                                            step=2),
        # f"{prefix}module__temporal_stride": trial.suggest_int(f"{prefix}module__temporal_stride", 1, 4)
    }

def cnn_ncp_training_space(trial, prefix):
    return {
        f"{prefix}optimizer__lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        f"{prefix}optimizer__weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        f"{prefix}batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        f"{prefix}module__drop_prob": trial.suggest_float(f"{prefix}module__drop_prob", 0.1, 0.5)
    }

def reegnet_architecture_space(trial, prefix):
    return {
        f"{prefix}module__lstm_hidden_size": trial.suggest_categorical(f"{prefix}module__lstm_hidden_size", [8, 16, 32, 64]),
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
        augmented=False
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
        augmented=augmented,
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
        augmented=augmented,
        output_root=output_root
    )
    for k, v in arch_params.items():
        final_params[k] = v

    return final_params, final_score