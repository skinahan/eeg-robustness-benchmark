#!/usr/bin/env python3
"""
Cross-session evaluation script for EEG models.

This script performs hyperparameter tuning on the 0train session and evaluates
on the 1test session, with support for noise augmentation and perturbation modes.
"""

import os
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
from moabb.evaluations import CrossSessionEvaluation

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
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, format_params, unified_cv_training_loop_method


class CrossSessionNoiseEvaluation(CrossSessionEvaluation):
    """Cross-session evaluation with noise support."""
    
    def __init__(self, paradigm, datasets, overwrite, hdf5_path, random_state, mode, noise_dict, resample, model_name):
        super().__init__(paradigm=paradigm, datasets=datasets, overwrite=overwrite, hdf5_path=hdf5_path, random_state=random_state)
        self.paradigm = paradigm
        self.datasets = datasets
        self.overwrite = overwrite
        self.hdf5_path = hdf5_path
        self.random_state = random_state
        self.mode = mode
        self.prefix = ""
        # Update prefix logic to include new modes
        if self.mode in ['perturb', 'augment', 'perturb_notune', 'augment_notune']:
            self.prefix = 'base_pipeline__'
        self.noise_dict = noise_dict
        self.noise_type = self.noise_dict["noise_type"] if noise_dict else None
        self.intensity = self.noise_dict["intensity"] if noise_dict else None
        self.seed = random_state
        self.resample = resample if resample else 250.0
        self.model_name = model_name
        self.model_fn = MODEL_REGISTRY[model_name]
        self.model = None

    def get_wrapped_model_function(self):
        """Get wrapped model function for noise modes."""
        if not self.noise_dict:
            return self.model_fn
            
        wrapped_model_fn = None

        if self.mode in ['perturb', 'perturb_notune']:
            from augmentation.noise import TrainOnlyNoiseClassifier
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                return TrainOnlyNoiseClassifier(
                    base_pipeline=base_model,
                    noise_type=self.noise_type,
                    intensity=self.intensity,
                    seed=self.seed
                )
        elif self.mode in ['augment', 'augment_notune']:
            from augmentation.noise import ConcatenatedNoiseAugmenter
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                return ConcatenatedNoiseAugmenter(
                    base_pipeline=base_model,
                    noise_type=self.noise_type,
                    intensity=self.intensity,
                    seed=self.seed
                )
        return wrapped_model_fn

    def evaluate_without_tuning(self, X_train, y_train, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn):
        """Evaluate model performance without hyperparameter tuning using default parameters."""
        results = []
        row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                       'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                       'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                       'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length',
                       'module__lstm_hidden_size'}
        
        self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
        
        # Train on 0train session
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on 1test session
        start_time = time.time()
        y_pred = self.model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        roc_auc_score = roc_auc_score(y_test, y_pred)
        evaluation_time = time.time() - start_time
        
        result_row = {
            'score': roc_auc_score,
            'time': training_time + evaluation_time,
            'samples': len(X_train) + len(X_test),
            'subject': str(metadata_train["subject"].iloc[0]) if len(metadata_train["subject"].unique()) == 1 else "multiple",
            'session': '1test',
            'channels': X_train.shape[1],
            'n_sessions': 2,  # 0train and 1test
            'dataset': metadata_train["dataset"].iloc[0] if "dataset" in metadata_train.columns else "unknown",
            'pipeline': f"{self.model_name}+MotorImagery+CrossSessionEvaluation",
            'seed': self.seed,
            'mode': self.mode,
            'model': self.model_name,
            'paradigm': 'MotorImagery',
            'resample': self.resample,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'training_time': training_time,
            'evaluation_time': evaluation_time
        }
        
        config = self.model.get_params()
        for k, v in config.items():
            if k.startswith(self.prefix):
                no_prefix = k[len(self.prefix):]
                if no_prefix in row_headers:
                    result_row[no_prefix] = v
            elif k in row_headers:
                result_row[k] = v
        
        results.append(result_row)
        
        return pd.DataFrame.from_records(results)

    def process_subj(self, process_dict, dataset, subj):
        """Process a single subject with cross-session evaluation."""
        X, y, metadata = self.paradigm.get_data(dataset, subjects=[subj])
        y_encoded = LabelEncoder().fit_transform(y)
        results = []
        wrapped_model_fn = self.get_wrapped_model_function()

        # Split data by session
        train_mask = metadata['session'] == '0train'
        test_mask = metadata['session'] == '1test'
        
        if not train_mask.sum() or not test_mask.sum():
            print(f"Warning: Missing train or test session for subject {subj}")
            return pd.DataFrame()
        
        X_train = X[train_mask]
        y_train = y_encoded[train_mask]
        metadata_train = metadata[train_mask]
        
        X_test = X[test_mask]
        y_test = y_encoded[test_mask]
        metadata_test = metadata[test_mask]

        for k, v in process_dict.items():
            process_name = k
            paradigm_name = process_name.split("+")[1]
            
            # Check if we should use tuning or not
            if self.mode in ['augment_notune', 'perturb_notune']:
                # Evaluate without tuning using default parameters
                result_df = self.evaluate_without_tuning(X_train, y_train, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn)
                results.append(result_df)
            else:
                # Hyperparameter tuning on 0train session
                out_dir = create_output_path(self.model_name, self.seed, subj, '1test', self.mode, session_type='CrossSessionEvaluation')
                output_root = os.path.join(out_dir, f"optuna_results_{self.noise_type}_{self.intensity}" if self.noise_dict else "optuna_results")
                
                if self.noise_dict:
                    best_params, best_score = alternate_two_stage_optuna(
                        model_fn=wrapped_model_fn, 
                        model_name=self.model_name, 
                        X=X_train, 
                        y=y_train,
                        metadata=metadata_train, 
                        resample=self.resample, 
                        seed=self.seed,
                        mode=self.mode, 
                        noise_dict=self.noise_dict,
                        output_root=output_root, 
                        arch_trials=10, 
                        train_trials=10
                    )
                else:
                    # Use regular two-stage optimization for non-noise modes
                    best_params, best_score = two_stage_opt(
                        dataset, subj, self.paradigm, self.model_name, wrapped_model_fn, 
                        self.seed, self.mode, self.resample
                    )
                
                final_params = format_params(best_params, self.prefix)
                
                # Evaluate on 1test session with tuned parameters
                row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                               'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                               'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                               'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length',
                               'module__lstm_hidden_size'}
                
                self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
                self.model.set_params(**final_params)
                
                # Train on 0train session
                start_time = time.time()
                self.model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate on 1test session
                start_time = time.time()
                y_pred = self.model.predict_proba(X_test)[:, 1]
                from sklearn.metrics import roc_auc_score
                roc_auc_score = roc_auc_score(y_test, y_pred)
                evaluation_time = time.time() - start_time
                
                result_row = {
                    'score': roc_auc_score,
                    'time': training_time + evaluation_time,
                    'samples': len(X_train) + len(X_test),
                    'subject': str(subj),
                    'session': '1test',
                    'channels': X_train.shape[1],
                    'n_sessions': 2,
                    'dataset': dataset.code,
                    'pipeline': process_name,
                    'seed': self.seed,
                    'mode': self.mode,
                    'model': self.model_name,
                    'paradigm': paradigm_name,
                    'resample': self.resample,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'training_time': training_time,
                    'evaluation_time': evaluation_time,
                    'best_score': best_score
                }
                
                config = self.model.get_params()
                for k, v in config.items():
                    if k.startswith(self.prefix):
                        no_prefix = k[len(self.prefix):]
                        if no_prefix in row_headers:
                            result_row[no_prefix] = v
                    elif k in row_headers:
                        result_row[k] = v
                
                # Add hyperparameters
                for k, v in final_params.items():
                    if k.startswith(self.prefix):
                        no_prefix = k[len(self.prefix):]
                        if no_prefix in row_headers:
                            result_row[no_prefix] = v
                    elif k in row_headers:
                        result_row[k] = v
                result_row = pd.DataFrame.from_records([result_row])
                results.append(result_row)
        
        return pd.concat(results) if results else pd.DataFrame()

    def process(self, process_dict):
        """Process all subjects with cross-session evaluation."""
        all_results = []
        for dataset in self.datasets:
            subject_list = dataset.subject_list
            for subj in subject_list:
                result_df = self.process_subj(process_dict, dataset, subj)
                if not result_df.empty:
                    all_results.append(result_df)
        return pd.concat(all_results) if all_results else pd.DataFrame()


def run_cross_session_experiment(
        model_name: str,
        mode: str,
        subject_list: List[int],
        seed: int,
        resample: float,
        noise_type: str = None,
        intensity: float = None
):
    """Run cross-session experiments with hyperparameter tuning on 0train and evaluation on 1test."""
    set_seeds(seed)
    is_perturbed = (mode in ["perturb", "augment", "perturb_notune", "augment_notune"])
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
    checkpoint_dir = create_hdf5_model_path(model_name, seed, '1test', mode, session_type='CrossSessionEvaluation')
    file_name = f"{noise_type}/{intensity}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}_{unique_id}.h5" if is_perturbed else f"subject{subject_list[0]}-{subject_list[-1]}_seed{seed}_{unique_id}.h5"
    full_hdf5_path = os.path.join(checkpoint_dir, file_name)

    if mode == "baseline":
        # Use custom cross-session evaluation instead of MOABB's CrossSessionEvaluation
        evaluation = CrossSessionEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            overwrite=True,
            hdf5_path=full_hdf5_path,
            random_state=seed,
        )
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
        if model_name == 'cnncfc_v2':
            df['module__ncp_hidden_dim'] = config['module__ncp_hidden_dim']
            df['module__drop_prob'] = config['module__drop_prob']
            df['module__F1'] = config['module__F1']
            df['module__D'] = config['module__D']
            df['module__kernel_length'] = config['module__kernel_length']

        if model_name == 'cnn_ncp':
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
                out_dir = create_output_path(model_name, seed, int(subj), session, mode, session_type='CrossSessionEvaluation')
                os.makedirs(out_dir, exist_ok=True)
                filename_suffix = f"_{noise_type}" if is_perturbed and noise_type else ""
                out_file = os.path.join(out_dir,
                                        f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
                session_df.to_csv(out_file, index=False)
                print(f"Saved: {out_file}")
        print(f"Cross-session baseline evaluation completed for {model_name}")

    elif mode == "tune":
        # Use custom cross-session evaluation for tuning as well
        evaluation = CrossSessionNoiseEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            mode=mode,
            noise_dict=None,
            resample=resample,
            overwrite=True,
            hdf5_path=full_hdf5_path,
            random_state=seed,
            model_name=model_name
        )
        results = evaluation.process({f"{model_name}+Optuna": None})
        log_all_subjects(results, subject_list, model_name, mode, None, None, seed, eval_mode='CrossSessionEvaluation')
        # Results are already saved by the custom evaluation class
        print(f"Cross-session tuning evaluation completed for {model_name}")

    elif mode in ["augment", "perturb", "augment_notune", "perturb_notune"]:
        noise_dict = {
            "noise_type": noise_type,
            "intensity": intensity
        }
        
        cap_Mode = mode.capitalize()
        unique_id = uuid.uuid4().hex[:8]
        checkpoint_dir = create_hdf5_model_path(model_name, seed, '1test', mode, session_type='CrossSessionEvaluation')
        file_name = f"{noise_type}/{intensity}_subject{subject_list[0]}-{subject_list[-1]}_seed{seed}_{unique_id}.h5"
        full_hdf5_path = os.path.join(checkpoint_dir, file_name)

        evaluation = CrossSessionNoiseEvaluation(
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
        log_all_subjects(results, subject_list, model_name, mode, noise_type, intensity, seed, eval_mode='CrossSessionEvaluation')

    if os.path.isdir(full_hdf5_path):
        import shutil
        shutil.rmtree(full_hdf5_path)


if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)

    parser = argparse.ArgumentParser(description="Cross-Session EEG Experiment Runner")
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
        if check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity, eval_mode='CrossSessionEvaluation'):
            sys.exit(0)
        run_cross_session_experiment(
            model_name=args.model,
            mode=args.mode,
            subject_list=args.subjects,
            seed=args.seed,
            resample=args.resample,
            noise_type=args.noise_type,
            intensity=args.intensity
        )
    elif args.mode == "baseline" or args.mode == "tune":
        if check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity, eval_mode='CrossSessionEvaluation'):
            sys.exit(0)
        run_cross_session_experiment(
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