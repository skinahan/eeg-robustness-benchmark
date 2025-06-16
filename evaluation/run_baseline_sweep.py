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

from config import DEFAULT_PARADIGM
import torch.cuda

def run_baseline(subject_list, model, model_name):
    dataset = BNCI2014_001()
    dataset.subject_list = subject_list

    paradigm = DEFAULT_PARADIGM

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=None,
    )

    results = evaluation.process({f"{model_name}+MotorImagery": model})
    out_path = f"results/{model_name}_baseline_subjects{subject_list[0]}-{subject_list[-1]}.csv"
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
    run_baseline(subject_list, reegnet, "reegnet")

def run_baseline_eegnet(subject_list):
    eegnet = create_eegnet_classifier(
        n_chans=22,n_times=1001,n_outputs=2)
    run_baseline(subject_list, eegnet, "eegnet")

def run_baseline_cnn_ncp(subject_list):
    cnn_ncp_net = create_cnnncp_classifier(
        n_chans=22,n_times=1001,n_outs=2)
    cnn_ncp_net.train_split = None
    cnn_ncp_net.max_epochs = 100
    cnn_ncp_net.callbacks = []#[EpochScoring('accuracy', lower_is_better=False, name='train_acc')]
    run_baseline(subject_list, cnn_ncp_net, "cnn_ncp")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)
    subject_list = range(1,10)
    #run_baseline_eegnet(subject_list)
    # run_baseline_reegnet(subject_list=subject_list)
    run_baseline_cnn_ncp([1])
    # Record end time
    end_time = time.time()
    print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")

