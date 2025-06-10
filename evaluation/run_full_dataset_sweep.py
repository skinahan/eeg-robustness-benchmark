# evaluation/run_full_dataset_sweep.py
import sys
import os


# dynamically set the project root as the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation
from typing import Dict, Any
from models.eegnet import create_eegnet_classifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import warnings

from models.reegnet import create_reegnet_classifier


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
                n_jobs=1,
                return_train_score=True
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
        "module__drop_prob": [0.1, 0.15, 0.25],
        "module__lstm_hidden_size": [8, 16, 32, 64],
        "optimizer__lr": [1e-3, 5e-4, 1e-4],
        "batch_size": [8, 16, 32, 64]
    }

    run_model_sweep(
        model_name="REEGNetv4+MotorImagery",
        model_pipeline=reegnet_pipeline,
        param_grid=reegnet_param_grid,
        hdf5_path="checkpoints/reegnet_tuned_subjects_all.h5",
        output_csv="results/reegnet_tuned_subjects_all.csv"
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)
    run_model_sweep_eegnet()
    #run_model_sweep_reegnet()
