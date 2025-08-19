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
import numpy as np
from sklearn.metrics import roc_auc_score


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.base import clone
from sklearn.metrics import get_scorer
from tqdm import tqdm
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation

# Try to import carbon footprint tracking
try:
    from codecarbon import EmissionsTracker
    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

# Import MOABB utilities
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list

# Import MNE for epochs handling
from mne.epochs import BaseEpochs

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY, get_paradigm
from globals import set_seeds
from evaluation.session_evaluator import NoiseWithinSessionEvaluation
from augmentation.noise import TrainOnlyNoiseClassifier, ConcatenatedNoiseAugmenter
from utils import create_output_path, create_hdf5_model_path
from evaluation.experiment_utils import (
    extract_model_params, check_skip_eval, log_all_subjects, 
    two_stage_opt, collect_all_results, add_experiment_metadata
)
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, format_params, unified_cv_training_loop_method
import shutil


class CrossSessionNoiseEvaluation(CrossSessionEvaluation):
    """Cross-session evaluation with noise support using LeaveOneGroupOut CV."""
    
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
            # Perturbation mode: Train on one session and evaluate on another
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 100
                return TrainOnlyNoiseClassifier(
                    base_pipeline=base_model,
                    noise_type=self.noise_type,
                    intensity=self.intensity,
                    seed=self.seed
                )
        elif self.mode in ['augment', 'augment_notune']:
            # Augmentation mode: Train on one session and evaluate on another
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 100
                return ConcatenatedNoiseAugmenter(
                    base_pipeline=base_model,
                    noise_type=self.noise_type,
                    intensity=self.intensity,
                    seed=self.seed
                )
        elif self.mode == 'test_perturb':
            # For test_perturb mode, return the base model function directly
            # since we'll handle noise application manually in the evaluation
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 100
                return base_model
        return wrapped_model_fn

    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        """Override the evaluate method to implement proper LeaveOneGroupOut CV."""
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        
        # Progressbar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CrossSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                print(f"Subject {subject} already processed")
                continue

            # get the data
            X, y, metadata = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                cache_config=self.cache_config,
                postprocess_pipeline=postprocess_pipeline,
            )
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)
            groups = metadata.session.values
            scorer = get_scorer(self.paradigm.scoring)

            for name, clf in run_pipes.items():
                if _carbonfootprint:
                    # Initialise CodeCarbon
                    tracker = EmissionsTracker(save_to_file=False, log_level="error")
                    tracker.start()

                # we want to store a results per session
                cv = LeaveOneGroupOut()
                inner_cv = StratifiedKFold(
                    3, shuffle=True, random_state=self.random_state
                )

                grid_clf = clone(clf)

                # For now, skip grid search and use the classifier directly
                # TODO: Implement proper grid search if needed
                grid_clf = grid_clf

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=False,
                        eval_type="CrossSession",
                    )

                for cv_ind, (train, test) in enumerate(cv.split(X, y, groups)):
                    model_list = []
                    if _carbonfootprint:
                        tracker.start()
                    t_start = time()
                    
                    # Get the session being used as test set
                    test_session = groups[test][0]
                    
                    if isinstance(X, BaseEpochs):
                        cvclf = clone(grid_clf)
                        cvclf.fit(X[train], y[train])
                        model_list.append(cvclf)
                        score = scorer(cvclf, X[test], y[test])

                        if self.hdf5_path is not None and self.save_model:
                            save_model_cv(
                                model=cvclf,
                                save_path=model_save_path,
                                cv_index=str(cv_ind),
                            )
                    else:
                        result = _fit_and_score(
                            estimator=clone(grid_clf),
                            X=X,
                            y=y,
                            scorer=scorer,
                            train=train,
                            test=test,
                            verbose=False,
                            parameters=None,
                            fit_params=None,
                            error_score=self.error_score,
                            return_estimator=True,
                        )
                        score = result["test_scores"]
                        model_list = result["estimator"]
                    
                    if _carbonfootprint:
                        emissions = tracker.stop()
                        if emissions is None:
                            emissions = 0

                    duration = time() - t_start
                    if self.hdf5_path is not None and self.save_model:
                        save_model_list(
                            model_list=model_list,
                            score_list=score,
                            save_path=model_save_path,
                        )

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": test_session,  # This will be either '0train' or '1test'
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }
                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)

                    yield res

    def evaluate_without_tuning(self, X_train, y_train, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn):
        """Evaluate model performance without hyperparameter tuning using default parameters."""
        results = []
        row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                       'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                       'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                       'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length', 'module__lstm_hidden_size'}
        
        self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
        
        # Train on training session
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test session
        start_time = time.time()
        y_pred = self.model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        evaluation_time = time.time() - start_time
        
        # Get the test session name from metadata
        test_session = metadata_test["session"].iloc[0] if len(metadata_test["session"].unique()) == 1 else "unknown"
        
        result_row = {
            'score': score,
            'time': training_time + evaluation_time,
            'samples': len(X_train) + len(X_test),
            'subject': str(metadata_train["subject"].iloc[0]) if len(metadata_train["subject"].unique()) == 1 else "multiple",
            'session': test_session,  # This will be either '0train' or '1test'
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

    def evaluate_test_perturb(self, X_train, y_train, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn):
        """Evaluate model performance in test_perturb mode using 3-fold CV on source session."""
        from augmentation.noise import EEGNoiseAugmentor
        results = []
        row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                       'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                       'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                       'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length', 'module__lstm_hidden_size'}
        
        self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
        
        # Train on training session
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test session
        start_time = time.time()
        y_pred = self.model.predict_proba(X_test)[:, 1]
        clean_score = roc_auc_score(y_test, y_pred)
        evaluation_time = time.time() - start_time
            
        intensities = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

        for intensity in intensities:
            # Create corrupted target data
            noise_augmentor = EEGNoiseAugmentor(
                noise_type=self.noise_type,
                intensity=intensity,
                seed=self.seed
            )
            
            # Apply corruption to target data
            X_test_corrupted = noise_augmentor.transform(X_test)
            
            # Evaluate on corrupted target data
            corrupted_score = self.model.score(X_test_corrupted, y_test)
            y_pred_corrupted = self.model.predict_proba(X_test_corrupted)[:, 1]
            corrupted_score = roc_auc_score(y_test, y_pred_corrupted)
            print(f"    Corrupted target score: {corrupted_score:.4f}")
            
            # Calculate relative drop
            relative_drop = (clean_score - corrupted_score) / clean_score if clean_score > 0 else 0.0

            # Get the test session name from metadata
            test_session = metadata_test["session"].iloc[0] if len(metadata_test["session"].unique()) == 1 else "unknown"

            result_row = {
                'score': clean_score,
                'corrupted_score': corrupted_score,
                'relative_drop': relative_drop,
                'noise_type': self.noise_type,
                'intensity': intensity,
                'time': training_time + evaluation_time,
                'samples': len(X_train) + len(X_test),
                'subject': str(metadata_train["subject"].iloc[0]) if len(metadata_train["subject"].unique()) == 1 else "multiple",
                'session': test_session,  # This will be either '0train' or '1test'
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
        """Process a single subject with cross-session evaluation using LeaveOneGroupOut CV."""
        X, y, metadata = self.paradigm.get_data(dataset, subjects=[subj])
        y_encoded = LabelEncoder().fit_transform(y)
        results = []
        wrapped_model_fn = self.get_wrapped_model_function()

        # Use LeaveOneGroupOut CV to iterate through sessions
        cv = LeaveOneGroupOut()
        groups = metadata['session'].values
        
        for train_idx, test_idx in cv.split(X, y_encoded, groups):
            # Get training and test data for this fold
            X_train = X[train_idx]
            y_train = y_encoded[train_idx]
            metadata_train = metadata.iloc[train_idx]
            
            X_test = X[test_idx]
            y_test = y_encoded[test_idx]
            metadata_test = metadata.iloc[test_idx]
            
            # Get the session being used as test set
            test_session = groups[test_idx][0]
            train_session = groups[train_idx][0]
            
            print(f"Subject {subj}: Training on session {train_session}, testing on session {test_session}")

            for k, v in process_dict.items():
                process_name = k
                paradigm_name = process_name.split("+")[1]
                
                # Check if we should use tuning or not
                if self.mode in ['augment_notune', 'perturb_notune']:
                    # Evaluate without tuning using default parameters
                    if self.mode == 'augment_notune':
                        self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
                        X_obj, y_obj, groups = self.model.concat_and_augment(X_train, y_train)
                        result_df = self.evaluate_without_tuning(X_obj, y_obj, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn)
                    else:
                        result_df = self.evaluate_without_tuning(X_train, y_train, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn)
                    results.append(result_df)
                elif self.mode == 'test_perturb':
                    result_df = self.evaluate_test_perturb(X_train, y_train, X_test, y_test, metadata_train, metadata_test, wrapped_model_fn)
                    results.append(result_df)                    
                else:
                    # Hyperparameter tuning on training session
                    out_dir = create_output_path(self.model_name, self.seed, subj, test_session, self.mode, session_type='CrossSessionEvaluation')
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
                            arch_trials=20, 
                            train_trials=20
                        )
                    else:
                        # Use regular two-stage optimization for non-noise modes
                        best_params, best_score = two_stage_opt(
                            dataset, subj, self.paradigm, self.model_name, wrapped_model_fn, 
                            self.seed, self.mode, self.resample
                        )
                    
                    final_params = format_params(best_params, self.prefix)
                    
                    # Evaluate on test session with tuned parameters
                    row_headers = {'score', 'time', 'samples', 'subject', 'session', 'channels', 'n_sessions', 'dataset',
                                   'pipeline', 'seed', 'mode', 'model', 'paradigm', 'resample', 'optimizer__lr', 'batch_size',
                                   'max_epochs', 'module__ncp_hidden_dim', 'module__sparsity', 'optimizer__weight_decay',
                                   'module__drop_prob', 'module__F1', 'module__D', 'module__kernel_length', 'module__lstm_hidden_size'}
                    
                    self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
                    self.model.set_params(**final_params)
                    
                    # Train on training session
                    start_time = time.time()
                    self.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate on test session
                    start_time = time.time()
                    y_pred = self.model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_pred)
                    evaluation_time = time.time() - start_time
                    
                    result_row = {
                        'score': score,
                        'time': training_time + evaluation_time,
                        'samples': len(X_train) + len(X_test),
                        'subject': str(subj),
                        'session': test_session,  # This will be either '0train' or '1test'
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

    # baseline mode is set up correctly, and CrossSessionNoiseEvaluation now properly uses LeaveOneGroupOut CV
    if mode == "baseline":
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
                # Use consistent filename pattern with log_all_subjects
                if is_perturbed and noise_type is not None and intensity is not None:
                    filename_suffix = f"_{noise_type}_{intensity}"
                else:
                    filename_suffix = ""
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

    elif mode in ["augment", "perturb", "augment_notune", "perturb_notune", "test_perturb"]:
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
        shutil.rmtree(full_hdf5_path)


if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)

    parser = argparse.ArgumentParser(description="Cross-Session EEG Experiment Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["baseline", "tune", "perturb", "augment", "perturb_notune", "augment_notune", "test_perturb", "aggregate_only"])
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
    elif args.mode == "test_perturb":
        # Use the established execution path with CrossSessionNoiseEvaluation
        # Setup dataset and paradigm
        dataset = BNCI2014_001()
        dataset.subject_list = args.subjects
        paradigm = get_paradigm(resample=args.resample)
        
        noise_dict = {
            "noise_type": args.noise_type,
            "intensity": args.intensity
        }
        
        unique_id = uuid.uuid4().hex[:8]
        checkpoint_dir = create_hdf5_model_path(args.model, args.seed, '1test', args.mode, session_type='CrossSessionEvaluation')
        file_name = f"{args.noise_type}/{args.intensity}_subject{args.subjects[0]}-{args.subjects[-1]}_seed{args.seed}_{unique_id}.h5"
        full_hdf5_path = os.path.join(checkpoint_dir, file_name)

        evaluation = CrossSessionNoiseEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            mode=args.mode,
            noise_dict=noise_dict,
            resample=args.resample,
            overwrite=True,
            hdf5_path=full_hdf5_path,
            random_state=args.seed,
            model_name=args.model
        )
        results = evaluation.process({f"{args.model}+MotorImagery+TestPerturb": None})
        log_all_subjects(results, args.subjects, args.model, args.mode, args.noise_type, args.intensity, args.seed, eval_mode='CrossSessionEvaluation')

        if os.path.isdir(full_hdf5_path):
            shutil.rmtree(full_hdf5_path)

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