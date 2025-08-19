import os
import shutil
import sys
import argparse
import uuid
import warnings
import time
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn import clone
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY, get_paradigm
from globals import set_seeds, get_seed
from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from evaluation.session_evaluator import NoiseWithinSessionEvaluation
from utils import create_output_path, create_hdf5_model_path
from evaluation.experiment_utils import check_skip_eval, log_all_subjects

from moabb.evaluations import WithinSessionSplitter
from sklearn.model_selection import cross_val_score
import optuna

from evaluation.two_stage_hp_opt import run_two_stage_optuna


def get_param_grid(model_name: str, noise_augmented: bool = False) -> Dict[str, Any]:
    base_prefix = "base_pipeline__" if noise_augmented else ""
    return {
        "eegnet": {
            f"{base_prefix}module__drop_prob": [0.1, 0.15],
            f"{base_prefix}optimizer__lr": [1e-4, 5e-4],
            f"{base_prefix}batch_size": [16, 32]
        },
        "reegnet": {
            f"{base_prefix}module__drop_prob": [0.1, 0.15],
            f"{base_prefix}module__lstm_hidden_size": [16, 32],
            f"{base_prefix}optimizer__lr": [1e-3, 1e-4],
            f"{base_prefix}batch_size": [16, 32]
        },
        "cnn_ncp": {
            f"{base_prefix}module__ncp_hidden_dim": [11, 16],
            f"{base_prefix}module__sparsity": [0.6, 0.8],
        },
        "cnncfc_v2": {
            f"{base_prefix}module__ncp_hidden_dim": [16, 32, 64],
            f"{base_prefix}module__drop_prob": [0.1, 0.2, 0.3],
            f"{base_prefix}module__F1": [4, 8, 12],
            f"{base_prefix}module__D": [1, 2],
            f"{base_prefix}module__temporal_kernel_size": [3, 5],
            f"{base_prefix}module__temporal_stride": [2, 4],
            f"{base_prefix}module__max_seq_length": [200, 250],
            f"{base_prefix}optimizer__lr": [1e-4, 5e-4, 1e-3],
            f"{base_prefix}optimizer__weight_decay": [1e-4, 1e-3],
            f"{base_prefix}batch_size": [16, 32],
            f"{base_prefix}max_epochs": [50, 100],
        },
        "spp_ncp": {
            f"{base_prefix}module__ncp_hidden_dim": [11, 16],
            f"{base_prefix}module__sparsity": [0.6, 0.8],
        }
    }.get(model_name, {
        f"{base_prefix}module__drop_prob": [0.1, 0.2],
        f"{base_prefix}optimizer__lr": [1e-4],
        f"{base_prefix}batch_size": [32]
    })


def extract_model_params(model) -> Dict[str, Any]:
    if hasattr(model, 'get_params'):
        return model.get_params()
    return {}


def collect_all_results(paradigm: str, dataset: str = "BNCI2014_001"):
    root = os.path.join("results", paradigm, dataset)
    all_dfs = []
    noise_types = ['gaussian', 'eog', 'dropout']
    intensities = [str(x*10.0) for x in range(1, 10)]
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".csv") and not file.startswith("all_results"):
                full_path = os.path.join(dirpath, file)
                print(full_path)
                try:
                    df = pd.read_csv(full_path)
                    selected_type = None
                    intensity = None
                    for type in noise_types:
                        if type in file:
                            selected_type = type
                            for strength in intensities:
                                if strength in file:
                                    intensity = strength
                                    break
                            break
                    if selected_type is not None and intensity is not None:
                        df['noise_type'] = selected_type
                        df['intensity'] = intensity


                    all_dfs.append(df)
                except Exception as e:
                    print(f"Failed to read {full_path}: {e}")
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_file = os.path.join(root, "all_results.csv")
        full_df.to_csv(out_file, index=False)
        print(f"Aggregated results saved to: {out_file}")
    else:
        print("No CSV files found to aggregate.")


def run_optuna_tuning_within_session(model_fn, model_name, X, y, metadata, resample=250.0, n_trials=25, seed=42,
                                     augmented=False):
    # Restrict to session 0 trials (typically '0train')
    train_mask = metadata["session"] == "0train"
    X_train = X[train_mask]
    y_train = y[train_mask]
    meta_train = metadata[train_mask].reset_index(drop=True)

    if len(X_train) < 10:
        raise ValueError("Too few training samples for subject/session.")

    # Create a session-aware splitter for training session only
    splitter = WithinSessionSplitter(n_folds=3, shuffle=True, random_state=seed)
    print(f"Splitter: {splitter.n_folds} folds, shuffle={splitter.shuffle}")
    param_prefix = "base_pipeline__" if augmented else ""

    def objective(trial):
        # Define search space based on model type
        if model_name.startswith("cnn_ncp"):
            params = {
                f"{param_prefix}module__ncp_hidden_dim": trial.suggest_int("module__ncp_hidden_dim", 11, 32),
                f"{param_prefix}module__sparsity": trial.suggest_float("module__sparsity", 0.4, 0.9),
                f"{param_prefix}optimizer__lr": trial.suggest_loguniform("optimizer__lr", 1e-4, 1e-2),
                f"{param_prefix}optimizer__weight_decay": trial.suggest_categorical("optimizer__weight_decay",
                                                                                    [0, 1e-3]),

                # f"{param_prefix}batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])
            }
        elif model_name.startswith("eegnet"):
            params = {
                f"{param_prefix}module__drop_prob": trial.suggest_float("module__drop_prob", 0.1, 0.15),
                f"{param_prefix}optimizer__lr": trial.suggest_loguniform("optimizer__lr", 1e-4, 5e-4),
                f"{param_prefix}batch_size": trial.suggest_categorical("batch_size", [16, 32])
            }
        elif model_name.startswith("reegnet"):
            params = {
                f"{param_prefix}module__drop_prob": trial.suggest_float("module__drop_prob", 0.1, 0.15),
                f"{param_prefix}module__lstm_hidden_size": trial.suggest_categorical("module__lstm_hidden_size",
                                                                                     [16, 32]),
                f"{param_prefix}optimizer__lr": trial.suggest_loguniform(1e-3, 1e-4),
                f"{param_prefix}batch_size": trial.suggest_categorical("batch_size", [16, 32])
            }
        else:
            raise ValueError(f"Unsupported model for tuning: {model_name}")

        # Build model with suggested params
        model = model_fn(n_chans=22, n_times=1001, n_outputs=2)
        model.set_params(**params)
        model.max_epochs = 100
        model.train_split = None
        model.callbacks = []

        # Use session-aware fold splits for training session only
        fold_scores = []
        for train_idx, test_idx in splitter.split(y_train, meta_train):
            X_tr, X_val = X_train[train_idx], X_train[test_idx]
            y_tr, y_val = y_train[train_idx], y_train[test_idx]

            model.fit(X_tr, y_tr)
            score = model.score(X_val, y_val)
            fold_scores.append(score)

        return np.mean(fold_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_score = study.best_value
    return best_params, best_score


def two_stage_opt(dataset, subj, paradigm, model_name, model_fn, seed, mode, resample):
    X, y, metadata = paradigm.get_data(dataset, subjects=[subj])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    out_dir = create_output_path(model_name, seed, subj, '0train', mode)
    best_params, best_score = run_two_stage_optuna(
        model_fn=model_fn,
        model_name=model_name,
        X=X,
        y=y_encoded,
        metadata=metadata,
        resample=resample,
        seed=seed,
        output_root=os.path.join(out_dir, "optuna_results"),
        arch_trials=20,
        train_trials=20,
        perturbed=False
    )
    final_params = {}
    module_params = ['ncp_hidden_dim', 'sparsity', 'temporal_kernel_size', 'temporal_stride', 'drop_prob']
    optimizer_params = ['lr', 'weight_decay']
    # prefix = "base_pipeline__" if is_perturbed else ""
    prefix = ""
    module_prefix = f"{prefix}module__"
    optim_prefix = f"{prefix}optimizer__"
    for k, v in best_params.items():
        if k in module_params:
            final_params[f"{module_prefix}{k}"] = v
        elif k in optimizer_params:
            final_params[f"{optim_prefix}{k}"] = v
        else:
            final_params[k] = v

    return final_params


def run_experiment(
        model_name: str,
        mode: str,
        subject_list: List[int],
        seed: int,
        resample: float,
        noise_type: str = None,
        intensity: float = None
):
    set_seeds(seed)
    is_perturbed = (mode == "perturb")
    model_fn = MODEL_REGISTRY[model_name]
    n_times = int(1000 * (resample / 250.0)) if resample else 1000

    base_model = model_fn(n_chans=22, n_times=n_times, n_outputs=2)
    base_model.train_split = None
    base_model.max_epochs = 100
    base_model.callbacks = []

    dataset = BNCI2014_001()
    dataset.subject_list = subject_list
    paradigm = get_paradigm(resample=resample)

    unique_id = uuid.uuid4().hex[:8]
    checkpoint_dir = create_hdf5_model_path(model_name, seed, '0train', mode)
    file_name = f"{noise_type}/{intensity}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}_{unique_id}.h5"
    full_hdf5_path = os.path.join(checkpoint_dir, file_name)

    if mode == "baseline":
        evaluation = \
            (WithinSessionEvaluation(
                paradigm=paradigm,
                datasets=[dataset],
                overwrite=True,
                hdf5_path=full_hdf5_path,
                random_state=seed
            ))
        results = evaluation.process({f"{model_name}+MotorImagery": base_model})

        df = results.copy()
        config = extract_model_params(base_model)
        df['seed'] = seed
        df['mode'] = mode
        df['model'] = model_name
        df['paradigm'] = 'MotorImagery'
        df['resample'] = resample or 250.0
        df['optimizer__lr'] = config['optimizer__lr']
        df['batch_size'] = config['batch_size']
        df['max_epochs'] = config['max_epochs']
        if model_name == 'cnn_ncp' or model_name == 'cnn_cfc':
            df['module__ncp_hidden_dim'] = config['module__ncp_hidden_dim']
            df['module__sparsity'] = config['module__sparsity']
            df['optimizer__weight_decay'] = config['optimizer__weight_decay']
        if model_name == 'reegnet':
            df['module__lstm_hidden_size'] = config['module__lstm_hidden_size']
            df['module__drop_prob'] = config['module__drop_prob']

        for subj in df['subject'].unique():
            subject_df = df[df['subject'] == subj]
            for session in df['session'].unique():
                session_df = subject_df[subject_df['session'] == session]
                out_dir = create_output_path(model_name, seed, int(subj), session, mode)
                os.makedirs(out_dir, exist_ok=True)
                filename_suffix = f"_{noise_type}" if is_perturbed and noise_type else ""
                out_file = os.path.join(out_dir,
                                        f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
                session_df.to_csv(out_file, index=False)
                print(f"Saved: {out_file}")
        
        if os.path.isdir(full_hdf5_path):
            shutil.rmtree(full_hdf5_path)

    elif mode == "tune":
        for subj in subject_list:
            dataset = BNCI2014_001()
            dataset.subject_list = subject_list
            paradigm = get_paradigm(resample=resample)
            final_params = two_stage_opt(dataset, subj, paradigm, model_name, model_fn, seed, mode, resample)
            base_model.set_params(**final_params)
            dataset.subject_list = [subj]
            evaluation = WithinSessionEvaluation(
                paradigm=paradigm,
                datasets=[dataset],
                overwrite=True,
                hdf5_path=full_hdf5_path,
                random_state=seed
            )
            results = evaluation.process({f"{model_name}+Optuna": base_model})
            df = results.copy()
            config = extract_model_params(base_model)
            df['seed'] = seed
            df['mode'] = mode
            df['model'] = model_name
            df['paradigm'] = 'MotorImagery'
            df['resample'] = resample or 250.0
            df['optimizer__lr'] = config['optimizer__lr']
            df['batch_size'] = config['batch_size']
            df['max_epochs'] = config['max_epochs']

            for k, v in final_params.items():
                df[k] = v

            for session in df['session'].unique():
                out_dir = create_output_path(model_name, seed, subj, session, mode)
                os.makedirs(out_dir, exist_ok=True)
                filename_suffix = f"_{noise_type}" if is_perturbed and noise_type else ""
                out_file = os.path.join(out_dir,
                                        f"{model_name}_{mode}{filename_suffix}_subject_{subj:03d}_seed{seed}.csv")
                df[df['session'] == session].to_csv(out_file, index=False)
                print(f"Saved: {out_file}")

            if os.path.isdir(full_hdf5_path):
                shutil.rmtree(full_hdf5_path)

# Unused GridSearchCV from past implementation
def grid_search_hp_opt(model, param_grid, X, y_encoded, subj, seed, mode, model_name, resample):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=1, return_train_score=True)
    grid.fit(X, y_encoded)
    train_score = np.mean(grid.cv_results_['mean_train_score'][grid.best_index_])
    test_score = grid.best_score_
    best_params = grid.cv_results_['params'][grid.best_index_]

    df = pd.DataFrame([
        {"subject": subj, "session": "0train", "score": train_score},
        {"subject": subj, "session": "1test", "score": test_score},
    ])
    df['seed'] = seed
    df['mode'] = mode
    df['model'] = model_name
    df['paradigm'] = 'MotorImagery'
    df['resample'] = resample or 250.0

# def check_skip_eval(model_name, seed, subject_list, mode, noise_type, intensity):
#     existing_output_paths = []
#     expected_output_paths = []
#     for subj in subject_list:
#         for session in ['0train', '1test']:
#             out_dir = create_output_path(model_name, seed, int(subj), session, mode)
#             if noise_type is not None and intensity is not None:
#                 filename_suffix = f"_{noise_type}_{intensity}"
#             else:
#                 filename_suffix = ""
#             out_file = os.path.join(out_dir,
#                                     f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
#             if os.path.exists(out_file):
#                 existing_output_paths.append(out_file)
#             else:
#                 expected_output_paths.append(out_file)

#     if len(expected_output_paths) == 0:
#         print(f"Skipping analysis, file(s) exist:")
#         for out_file in existing_output_paths:
#             print(out_file)
#         sys.exit(0)

def log_all_subjects(results, subject_list, model_name, mode, noise_type, intensity, seed):
    for subj in subject_list:
        subject_df = results[results['subject'] == str(subj)]
        for session in subject_df['session'].unique():
            session_df = subject_df[subject_df['session'] == session]
            out_dir = create_output_path(model_name, seed, int(subj), session, mode)
            os.makedirs(out_dir, exist_ok=True)
            filename_suffix = f"_{noise_type}_{intensity}"
            out_file = os.path.join(out_dir,
                                    f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
            session_df.to_csv(out_file, index=False)
            print(f"Saved: {out_file}")
            
def run_grouped_augmented_experiment(model_name, subject_list, seed, resample, noise_type, intensity, mode):
    set_seeds(seed)
    dataset = BNCI2014_001()
    dataset.subject_list = subject_list
    paradigm = get_paradigm(resample=resample)
    noise_dict = {
        "noise_type": noise_type,
        "intensity": intensity
    }
    # Perturb or Augment
    cap_Mode = mode.capitalize()
    unique_id = uuid.uuid4().hex[:8]
    checkpoint_dir = create_hdf5_model_path(model_name, seed, '0train', mode)
    file_name = f"{noise_type}/{intensity}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}_{unique_id}.h5"
    full_hdf5_path = os.path.join(checkpoint_dir, file_name)

    evaluation = \
        (NoiseWithinSessionEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            mode=mode,
            noise_dict=noise_dict,
            resample=resample,
            overwrite=True,
            hdf5_path=full_hdf5_path,
            random_state=seed,
            model_name=model_name
        ))
    results = evaluation.process({f"{model_name}+MotorImagery+{cap_Mode}": None})
    log_all_subjects(results, subject_list, model_name, mode, noise_type, intensity, seed)

    if os.path.isdir(full_hdf5_path):
        shutil.rmtree(full_hdf5_path)

if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)

    parser = argparse.ArgumentParser(description="Unified EEG Experiment Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "tune", "perturb", "augment", "perturb_notune", "augment_notune", "test_perturb", "aggregate_only"])
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample", type=float, default=None)
    parser.add_argument("--noise_type", type=str, choices=["dropout", "gaussian", "eog"], default=None)
    parser.add_argument("--intensity", type=float, default=None)

    parser.add_argument("--aggregate", action="store_true")

    args = parser.parse_args()

    if args.mode in ["augment", "perturb", "augment_notune", "perturb_notune"]:
        check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity)
        run_grouped_augmented_experiment(
            model_name=args.model,
            subject_list=args.subjects,
            seed=args.seed,
            resample=args.resample,
            noise_type=args.noise_type,
            intensity=args.intensity,
            mode=args.mode
        )
    elif args.mode == "test_perturb":
        # Use the established execution path with NoiseWithinSessionEvaluation
        # Setup dataset and paradigm
        dataset = BNCI2014_001()
        dataset.subject_list = args.subjects
        paradigm = get_paradigm(resample=args.resample)
        
        noise_dict = {
            "noise_type": args.noise_type,
            "intensity": args.intensity
        }
        
        unique_id = uuid.uuid4().hex[:8]
        checkpoint_dir = create_hdf5_model_path(args.model, args.seed, '0train', args.mode)
        file_name = f"{args.noise_type}/{args.intensity}_subject{args.subjects[0]}-{args.subjects[-1]}_seed{args.seed}_{unique_id}.h5"
        full_hdf5_path = os.path.join(checkpoint_dir, file_name)

        evaluation = NoiseWithinSessionEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            mode=args.mode,
            noise_dict=noise_dict,
            resample=args.resample,
            overwrite=True,
            hdf5_path=full_hdf5_path,
            random_state=args.seed,
            model_name=args.model
        )
        results = evaluation.process({f"{args.model}+MotorImagery+TestPerturb": None})
        log_all_subjects(results, args.subjects, args.model, args.mode, args.noise_type, args.intensity, args.seed)

        if os.path.isdir(full_hdf5_path):
            shutil.rmtree(full_hdf5_path)

    elif args.mode == "baseline" or args.mode == "tune":
        check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity)
        run_experiment(
            model_name=args.model,
            mode=args.mode,
            subject_list=args.subjects,
            seed=args.seed,
            resample=args.resample,
            noise_type=args.noise_type,
            intensity=args.intensity
        )

    if args.aggregate or args.mode == "aggregate_only":
        collect_all_results(paradigm='MotorImagery')

    # Record end time
    end_time = time.time()
    print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")
