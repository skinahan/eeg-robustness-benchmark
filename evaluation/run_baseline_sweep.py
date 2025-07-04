import argparse
import sys
import os
import time
import warnings
from datetime import datetime
import numpy as np
import sklearn
import time
from datetime import datetime

from skorch.callbacks import EpochScoring

# dynamically set the project root as the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Record start time
start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from models.cnnncp import create_cnnncp_classifier

from config import DEFAULT_PARADIGM, MODEL_REGISTRY
import torch.cuda

from globals import set_seeds, get_seed


def run_baseline(subject_list, model, model_name, resample=None):
    dataset = BNCI2014_001()
    # dataset.subject_list = subject_list
    seed = get_seed()

    if resample is not None:
        paradigm = MotorImagery(
            events=["left_hand", "right_hand"],
            fmin=8,
            fmax=35,
            tmin=0.0,
            tmax=None,
            baseline=None,
            resample=resample,
            n_classes=2
        )
    else:
        paradigm = DEFAULT_PARADIGM
    paradigm = DEFAULT_PARADIGM

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=None,
    )

    results = evaluation.process({f"{model_name}+MotorImagery": model})
    out_path = f"results/{model_name}_baseline_subjects{subject_list[0]}-{subject_list[-1]}-seed{seed}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved subject sweep results to {out_path}")
    print(results)

def run_baseline_reegnet(subject_list):
    reegnet = create_reegnet_classifier(
        n_chans=22,
        n_times=1001,
        n_outputs=2
    )
    reegnet.train_split=None
    reegnet.max_epochs = 100
    reegnet.callbacks = []
    run_baseline(subject_list, reegnet, "reegnet")

def run_baseline_eegnet(subject_list):
    eegnet = create_eegnet_classifier(
        n_chans=22,n_times=1001,n_outputs=2)
    eegnet.train_split = None
    eegnet.max_epochs = 100
    eegnet.callbacks = []
    run_baseline(subject_list, eegnet, "eegnet")

def run_baseline_cnn_ncp(subject_list):
    cnn_ncp_net = create_cnnncp_classifier(
        n_chans=22,n_times=1001,n_outs=2)
    cnn_ncp_net.train_split = None
    cnn_ncp_net.max_epochs = 100
    cnn_ncp_net.callbacks = []#[EpochScoring('accuracy', lower_is_better=False, name='train_acc')]
    run_baseline(subject_list, cnn_ncp_net, "cnn_ncp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOABB experiment with no hyp tuning.")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)
    args = parser.parse_args()
    seed = args.seed
    set_seeds(seed)
    subject_list = range(1,10)
    if args.model == "eegnet":
        run_baseline_eegnet(subject_list=subject_list)
    elif args.model == "reegnet":
        run_baseline_reegnet(subject_list=subject_list)
    elif args.model == "cnn_ncp":
        run_baseline_cnn_ncp(subject_list=subject_list)
    # Record end time
    end_time = time.time()
    print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")

