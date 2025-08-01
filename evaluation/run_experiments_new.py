import os
import shutil
import sys
import argparse
import uuid
import warnings
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY, get_paradigm
from globals import set_seeds
from evaluation.session_evaluator import NoiseWithinSessionEvaluation
from utils import create_output_path, create_hdf5_model_path
from evaluation.experiment_utils import (
    extract_model_params, check_skip_eval, log_all_subjects, 
    two_stage_opt, collect_all_results, add_experiment_metadata
)





def run_experiment(
        model_name: str,
        mode: str,
        subject_list: List[int],
        seed: int,
        resample: float,
        noise_type: str = None,
        intensity: float = None
):
    """Run baseline or tuning experiments."""
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
        evaluation = WithinSessionEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            overwrite=True,
            hdf5_path=full_hdf5_path,
            random_state=seed
        )
        results = evaluation.process({f"{model_name}+MotorImagery": base_model})

        df = results.copy()
        config = extract_model_params(base_model)
        df = add_experiment_metadata(df, model_name, seed, mode, resample, config)

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
            df = add_experiment_metadata(df, model_name, seed, mode, resample, config)

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





def run_grouped_augmented_experiment(model_name, subject_list, seed, resample, noise_type, intensity, mode):
    """Run experiments with noise augmentation or perturbation."""
    set_seeds(seed)
    dataset = BNCI2014_001()
    dataset.subject_list = subject_list
    paradigm = get_paradigm(resample=resample)
    noise_dict = {
        "noise_type": noise_type,
        "intensity": intensity
    }
    
    cap_Mode = mode.capitalize()
    unique_id = uuid.uuid4().hex[:8]
    checkpoint_dir = create_hdf5_model_path(model_name, seed, '0train', mode)
    file_name = f"{noise_type}/{intensity}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}_{unique_id}.h5"
    full_hdf5_path = os.path.join(checkpoint_dir, file_name)

    evaluation = NoiseWithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        mode=mode,
        noise_dict=noise_dict,
        resample=resample,
        overwrite=True,
        hdf5_path=full_hdf5_path,
        random_state=seed,
        model_name=model_name
    )
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
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["baseline", "tune", "perturb", "augment", "perturb_notune", "augment_notune", "aggregate_only"])
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample", type=float, default=None)
    parser.add_argument("--noise_type", type=str, choices=["dropout", "gaussian", "eog"], default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--aggregate", action="store_true")

    args = parser.parse_args()

    if args.mode in ["augment", "perturb", "augment_notune", "perturb_notune"]:
        if check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity):
            sys.exit(0)
        run_grouped_augmented_experiment(
            model_name=args.model,
            subject_list=args.subjects,
            seed=args.seed,
            resample=args.resample,
            noise_type=args.noise_type,
            intensity=args.intensity,
            mode=args.mode
        )
    elif args.mode == "baseline" or args.mode == "tune":
        if check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity):
            sys.exit(0)
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