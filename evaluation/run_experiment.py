# evaluation/run_experiment.py
import sys
import os
import time
import warnings
from datetime import datetime


# dynamically set the project root as the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from config import MODEL_REGISTRY
from augmentation.noise import EEGNoiseAugmentor, TrainOnlyNoiseClassifier


def run_evaluation(
    model_fn,
    model_name: str,
    param_grid: Dict[str, Any],
    noise_type: str = None,
    intensity: float = None,
    subject_list: list[int] = None
):
    # Dataset & paradigm
    dataset = BNCI2014_001()
    if subject_list is not None:
        dataset.subject_list = subject_list

    paradigm = MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8,
        fmax=35,
        tmin=0.0,
        tmax=None,
        baseline=None,
        resample=None,
        n_classes=2
    )

    # Model pipeline
    base_model = model_fn(n_chans=22, n_times=1001, n_outputs=2)

    full_pipeline = base_model

    train_only_aug_clf = TrainOnlyNoiseClassifier(
        base_pipeline=full_pipeline,
        noise_type=noise_type,
        intensity=intensity,
        seed=42
    )

    # Evaluation setup
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path="checkpoints/models_with_noise.h5"
    )

    # Results storage
    results_df = []
    for dataset in evaluation.datasets:
        for subject in dataset.subject_list:
            X, y, metadata = paradigm.get_data(dataset, subjects=[subject])
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Grid search for hyperparameter tuning
            grid = GridSearchCV(
                train_only_aug_clf,
                param_grid=param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=4,
                return_train_score=True
            )
            grid.fit(X, y_encoded)

            # Save train/test scores separately
            train_score_mean = np.mean(grid.cv_results_['mean_train_score'][grid.best_index_])
            test_score_mean = grid.best_score_
            best_params = grid.best_params_

            results_df.append({
                'subject': subject,
                'session': '0train',
                'score': train_score_mean,
                **best_params
            })
            results_df.append({
                'subject': subject,
                'session': '1test',
                'score': test_score_mean,
                **best_params
            })

    # Output file naming
    label = f"{model_name}"
    if noise_type:
        label += f"_{noise_type}_{intensity}"
    out_path = f"results/{label}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save results
    df = pd.DataFrame(results_df)
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

# Param grid selector
def get_param_grid(model_name: str) -> Dict[str, Any]:
    param_grids = {
        "eegnet": {
            "base_pipeline__module__drop_prob": [0.1, 0.15],
            "base_pipeline__optimizer__lr": [1e-4, 5e-4],
            "base_pipeline__batch_size": [16, 32]
        },
        "reegnet": {
            "base_pipeline__module__drop_prob": [0.1, 0.15],
            "base_pipeline__module__lstm_hidden_size": [16, 32],
            "base_pipeline__optimizer__lr": [1e-3, 1e-4],
            "base_pipeline__batch_size": [16, 32]
        }
    }
    # If not found, fallback to a reasonable default
    return param_grids.get(model_name, {
        "base_pipeline__module__drop_prob": [0.1, 0.2],
        "base_pipeline__optimizer__lr": [1e-4],
        "base_pipeline__batch_size": [32]
    })



if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)
    parser = argparse.ArgumentParser(description="Run MOABB experiment with optional noise augmentation.")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--noise_type", type=str, default=None, choices=['dropout', 'gaussian', 'eog'])
    parser.add_argument("--intensity", type=float, default=None)
    # parser.add_argument("--subjects", type=int, nargs="*", default=None)

    args = parser.parse_args()

    # Example param_grid for EEGNet/REEGNet (can be adapted per model)
    # Use the selected param grid
    param_grid = get_param_grid(args.model)
    print("Using param grid: {}".format(param_grid))
    subjects = list(range(1,10))
    model_fn = MODEL_REGISTRY[args.model]
    run_evaluation(
        model_fn=model_fn,
        model_name=args.model,
        param_grid=param_grid,
        noise_type=args.noise_type,
        intensity=args.intensity,
        subject_list=subjects#args.subjects,
    )
