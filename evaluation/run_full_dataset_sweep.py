# evaluation/run_full_dataset_sweep.py
import argparse
import sys
import os
import time
from datetime import datetime

from config import MODEL_REGISTRY
from globals import set_seeds

# dynamically set the project root as the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Record start time
start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import numpy as np
import sklearn
import torch
import torch.cuda
import pandas as pd
from typing import Dict, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from models.cnnncp import create_cnnncp_classifier

import warnings


def run_model_sweep(
    model_name: str,
    model_pipeline,
    param_grid: Dict[str, Any],
    hdf5_path: str,
    output_csv: str
):
    # Use MotorImagery paradigm with explicit 2-class filtering
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

    # Use all subjects in the dataset
    dataset = BNCI2014_001()

    # single_subject = True
    # if single_subject:
    #     dataset.subject_list = [1]

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=hdf5_path
    )

    results_df = []
    for dataset in evaluation.datasets:
        for subject in dataset.subject_list:
            X, y, metadata = paradigm.get_data(dataset, subjects=[subject])
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            # Run the inner CV search manually
            grid = GridSearchCV(
                model_pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=4,
                return_train_score=True,
                verbose=10
            )
            grid.fit(X, y_encoded)

            train_score_mean = np.mean(grid.cv_results_['mean_train_score'][grid.best_index_])
            test_score_mean = grid.best_score_

            # Save result and best params
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

    # Save everything to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results_df)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def run_model_sweep_eegnet():

    eegnet_pipeline = create_eegnet_classifier(
        n_chans=22,
        n_times=1001,
        n_outputs=2
    )

    eegnet_param_grid = {
        "module__drop_prob": [0.1, 0.15],
        "optimizer__lr": [1e-4, 5e-4],
        "batch_size": [16, 32]
    }

    run_model_sweep(
        model_name="EEGNetv4+MotorImagery",
        model_pipeline=eegnet_pipeline,
        param_grid=eegnet_param_grid,
        hdf5_path="checkpoints/eegnet_tuned_subjects_all.h5",
        output_csv="results/eegnet_tuned_subjects_all.csv"
    )

def run_model_sweep_reegnet():

    reegnet_pipeline = create_reegnet_classifier(
        n_chans=22,
        n_times=1001,
        n_outputs=2
    )

    reegnet_param_grid = {
        "module__drop_prob": [0.1, 0.15],
        "module__lstm_hidden_size": [16, 32],
        "optimizer__lr": [1e-3, 1e-4],
        "batch_size": [16, 32]
    }

    run_model_sweep(
        model_name="REEGNetv4+MotorImagery",
        model_pipeline=reegnet_pipeline,
        param_grid=reegnet_param_grid,
        hdf5_path="checkpoints/reegnet_tuned_subjects_all.h5",
        output_csv="results/reegnet_tuned_subjects_all.csv"
    )

def run_model_sweep_cnn_ncp():

    cnn_ncp_pipeline = create_cnnncp_classifier(
        n_chans=22,
        n_times=1001,
        n_outs=2
    )

    cnn_ncp_pipeline.train_split = None
    cnn_ncp_pipeline.max_epochs = 100
    cnn_ncp_pipeline.callbacks = []
    cnn_ncp_pipeline.verbose = 0

    cnn_ncp_param_grid = {
        "module__ncp_hidden_dim": [19, 24],
        "module__sparsity": [0.6, 0.8],
        "optimizer__lr": [1e-3, 1e-4],
        "batch_size": [16, 32]
    }

    run_model_sweep(
        model_name="CNN_NCP+MotorImagery",
        model_pipeline=cnn_ncp_pipeline,
        param_grid=cnn_ncp_param_grid,
        hdf5_path="checkpoints/cnn_ncp_tuned_subjects_all.h5",
        output_csv="results/cnn_ncp_tuned_subjects_all.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOABB experiment with no hyp tuning.")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    args = parser.parse_args()
    seed = args.seed
    set_seeds(seed)
    if args.model == "eegnet":
        run_model_sweep_eegnet()
    elif args.model == "reegnet":
        run_model_sweep_reegnet()
    elif args.model == "cnnncp":
        run_model_sweep_cnn_ncp()
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)

    # Record end time
    end_time = time.time()
    print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")
