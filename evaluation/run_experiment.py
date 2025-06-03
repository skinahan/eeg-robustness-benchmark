# evaluation/run_experiment.py

import os
import argparse
import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation
from models.eegnet import create_eegnet_classifier
from models.reegnet import create_reegnet_classifier
from augmentation.noise import EEGNoiseAugmentor
from sklearn.pipeline import Pipeline
from config import MODEL_REGISTRY, DEFAULT_DATASET, DEFAULT_PARADIGM


def run_evaluation(model_fn, model_name, noise_type=None, intensity=None, subject_list=None):
    dataset = DEFAULT_DATASET()
    if subject_list is not None:
        dataset.subject_list = subject_list

    paradigm = DEFAULT_PARADIGM
    base_model = model_fn(n_chans=22, n_times=1001, n_outputs=2)

    if noise_type:
        full_pipeline = Pipeline([
            ('augment', EEGNoiseAugmentor(noise_type=noise_type, intensity=intensity, seed=42)),
            ('model', base_model)
        ])
    else:
        full_pipeline = base_model

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=None,
    )

    results = evaluation.process({model_name: full_pipeline})

    label = f"{model_name}"
    if noise_type:
        label += f"_{noise_type}_{intensity}"
    out_path = f"results/{label}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOABB experiment with optional noise augmentation.")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--noise_type", type=str, default=None, choices=['dropout', 'gaussian', 'eog'])
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)

    args = parser.parse_args()
    model_fn = MODEL_REGISTRY[args.model]
    run_evaluation(
        model_fn=model_fn,
        model_name=args.model,
        noise_type=args.noise_type,
        intensity=args.intensity,
        subject_list=args.subjects,
    )