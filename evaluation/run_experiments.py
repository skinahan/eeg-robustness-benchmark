import os
import sys
import argparse
import warnings
import time
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY
from globals import set_seeds, get_seed
from augmentation.noise import TrainOnlyNoiseClassifier

def get_paradigm(resample=None):
    return MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8, fmax=35,
        tmin=0.0, tmax=None,
        baseline=None,
        resample=resample,
        n_classes=2
    )

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

def create_output_path(model, seed, subject, session, mode, paradigm='MotorImagery'):
    return os.path.join(
        "results",
        paradigm,
        "BNCI2014_001",
        model,
        "WithinSessionEvaluation",
        str(seed),
        f"sub-{subject:03d}",
        session,
        mode
    )

def collect_all_results(paradigm: str, dataset: str = "BNCI2014_001"):
    root = os.path.join("results", paradigm, dataset)
    all_dfs = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".csv") and not file.startswith("all_results"):
                full_path = os.path.join(dirpath, file)
                try:
                    df = pd.read_csv(full_path)
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
    is_augmented = (mode == "augment")
    param_grid = get_param_grid(model_name, noise_augmented=is_augmented)
    model_fn = MODEL_REGISTRY[model_name]
    n_times = int(1001 * (resample / 250.0)) if resample else 1001

    base_model = model_fn(n_chans=22, n_times=n_times, n_outputs=2)
    base_model.train_split = None
    base_model.max_epochs = 100
    base_model.callbacks = []

    if is_augmented:
        model = TrainOnlyNoiseClassifier(
            base_pipeline=base_model,
            noise_type=noise_type,
            intensity=intensity,
            seed=seed
        )
    else:
        model = base_model

    dataset = BNCI2014_001()
    dataset.subject_list = subject_list
    paradigm = get_paradigm(resample=resample)
    evaluation = (WithinSessionEvaluation
        (
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=f"checkpoints/{model_name}_{mode}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}.h5"
        ))

    if mode == "baseline":
        results = evaluation.process({f"{model_name}+MotorImagery": model})
        df = results.copy()
        config = extract_model_params(model)
        df['seed'] = seed
        df['mode'] = mode
        df['model'] = model_name
        df['paradigm'] = 'MotorImagery'
        df['resample'] = resample or 250.0
        df['optimizer__lr'] = config['optimizer__lr']
        df['batch_size'] = config['batch_size']
        df['max_epochs'] = config['max_epochs']

        if model_name == 'cnn_ncp':
            df['module__ncp_hidden_dim'] = config['module__ncp_hidden_dim']
            df['module__sparsity'] = config['module__sparsity']
            df['optimizer__weight_decay'] = config['optimizer__weight_decay']
        if model_name == 'reegnet':
            df['module__lstm_hidden_size'] = config['module__lstm_hidden_size']
            df['module__drop_prob'] = config['module__drop_prob']

        for subj in subject_list:
            for session in df['session'].unique():
                out_dir = create_output_path(model_name, seed, subj, session, mode)
                os.makedirs(out_dir, exist_ok=True)
                filename_suffix = f"_{noise_type}" if is_augmented and noise_type else ""
                out_file = os.path.join(out_dir, f"{model_name}_{mode}{filename_suffix}_subject_{subj:03d}_seed{seed}.csv")
                df[df['session'] == session].to_csv(out_file, index=False)
                print(f"Saved: {out_file}")

    else:
        for subj in subject_list:
            X, y, metadata = paradigm.get_data(dataset, subjects=[subj])
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
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

            for k, v in best_params.items():
                df[k] = v

            for session in df['session'].unique():
                out_dir = create_output_path(model_name, seed, subj, session, mode)
                os.makedirs(out_dir, exist_ok=True)
                filename_suffix = f"_{noise_type}" if is_augmented and noise_type else ""
                out_file = os.path.join(out_dir, f"{model_name}_{mode}{filename_suffix}_subject_{subj:03d}_seed{seed}.csv")
                df[df['session'] == session].to_csv(out_file, index=False)
                print(f"Saved: {out_file}")

if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)

    parser = argparse.ArgumentParser(description="Unified EEG Experiment Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "tune", "augment"])
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample", type=float, default=None)
    parser.add_argument("--noise_type", type=str, choices=["dropout", "gaussian", "eog"], default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--aggregate", action="store_true")

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        mode=args.mode,
        subject_list=args.subjects,
        seed=args.seed,
        resample=args.resample,
        noise_type=args.noise_type,
        intensity=args.intensity
    )

    if args.aggregate:
        collect_all_results(paradigm='MotorImagery')

    # Record end time
    end_time = time.time()
    print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")