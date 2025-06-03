import os
import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from config import DEFAULT_PARADIGM

from moabb.evaluations import WithinSessionEvaluation
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from models.reegnet import create_reegnet_classifier


def run_tuned_sweep_reegnet(subject_list):
    dataset = BNCI2014_001()
    dataset.subject_list = subject_list

    reegnet = create_reegnet_classifier(
        n_chans=22,
        n_times=1001,
        n_outputs=2
    )

    param_grid = {
        "module__drop_prob": [0.1, 0.15, 0.25],
        "module__lstm__hidden_size": [16, 32, 64],
        "optimizer__lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [16, 32, 64]
    }

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

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=None
    )

    results = evaluation.process(
        pipelines={"REEGNet+MotorImagery": reegnet},
        param_grid=param_grid
    )

    results.to_csv("results/reegnet_tuned_subjects1-5.csv", index=False)
    print(results)



if __name__ == "__main__":
    subject_list = [1, 2, 3, 4, 5]
    run_tuned_sweep_reegnet(subject_list)