#!/usr/bin/env python3
"""
Unified Experiment Runner for EEG Models

This script defines a single, flexible experiment runner.

Supports:
- Multiple evaluation modes: WithinSession, CrossSession, CrossSubject
- Multiple experiment modes: baseline, tune, augment, perturb, augment_notune, perturb_notune, test_perturb
- Noise types: dropout, gaussian, eog
- Dynamic model instantiation based on dataset characteristics
- Two-stage hyperparameter optimization
- Comprehensive result logging and aggregation
"""

import os
import sys
import torch
import argparse
import uuid
import warnings
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics import get_scorer
from skorch.callbacks import EarlyStopping
from tqdm import tqdm
import shutil

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY, get_paradigm, get_dataset_sampling_rate
from globals import set_seeds, DEFAULT_MAX_EPOCHS, UNDERFITTING_THRESHOLD
from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, run_two_stage_optuna, format_params, get_all_model_params
from utils import create_output_path, create_hdf5_model_path, get_noise_intensities
from evaluation.experiment_utils import check_skip_eval, log_all_subjects, collect_all_results
from evaluation.metrics import compute_classification_metrics
from evaluation.model_cache_manager import ModelCacheManager
from evaluation.periodic_checkpoint_callback import create_periodic_checkpoint_callback, create_model_cache_callback
import json

# Import MOABB components
from moabb.datasets import BNCI2014_001, Lee2019_SSVEP, BI2015a
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list
from mne.epochs import BaseEpochs

# Try to import carbon footprint tracking
try:
    from codecarbon import EmissionsTracker
    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


class ThreeFoldSubjectSplit:
    """
    Custom cross-validation splitter for CrossSubject evaluation.
    Splits subjects into 3 folds where 1/3 of subjects are used for evaluation
    and 2/3 are used for training in each fold.
    
    If the number of subjects is not evenly divisible by 3, the remainder
    subjects are added to the training set in all folds.
    """
    
    def __init__(self):
        self.n_splits = 3
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/test indices for 3-fold cross-subject splits.
        
        Args:
            X: Feature data (not used, only for compatibility)
            y: Target data (not used, only for compatibility)
            groups: Subject IDs for each sample
        
        Yields:
            train_idx, test_idx: Indices for training and testing sets
        """
        if groups is None:
            raise ValueError("groups parameter is required for ThreeFoldSubjectSplit")
        
        # Get unique subjects and their indices
        unique_subjects = np.unique(groups)
        n_subjects = len(unique_subjects)
        
        # Calculate size of each evaluation group (should be n_subjects // 3)
        eval_group_size = n_subjects // 3
        
        if eval_group_size == 0:
            raise ValueError(f"Need at least 3 subjects for 3-fold split, got {n_subjects}")
        
        # Create 3 folds
        for fold_idx in range(3):
            # Determine which subjects go in eval set for this fold
            eval_start = fold_idx * eval_group_size
            eval_end = eval_start + eval_group_size
            
            eval_subjects = unique_subjects[eval_start:eval_end]
            train_subjects = np.concatenate([
                unique_subjects[:eval_start],
                unique_subjects[eval_end:]
            ])
            
            # Get indices for train and test sets
            train_mask = np.isin(groups, train_subjects)
            test_mask = np.isin(groups, eval_subjects)
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx


def save_training_history(model, output_path: str, fold_idx: int = None, subject: int = None, session: str = None, mode: str = None):
    """
    Save the training history from a fitted model to a JSON file.
    
    Args:
        model: Fitted skorch/braindecode model with history attribute
        output_path: Base output directory for saving history
        fold_idx: Current fold index (optional)
        subject: Current subject ID (optional)
        session: Current session ID (optional)
        mode: Experiment mode (optional)
    """
    if not hasattr(model, 'history') or model.history is None:
        print("Warning: Model does not have history attribute or history is None")
        return
    
    if len(model.history) == 0:
        print("Warning: Model history is empty")
        return
    
    # Create history directory
    history_dir = os.path.join(output_path, "training_history")
    os.makedirs(history_dir, exist_ok=True)
    
    # Build filename - convert session to string if needed
    filename_parts = ["history"]
    if subject is not None:
        filename_parts.append(f"sub{subject:03d}")
    if session is not None:
        filename_parts.append(f"sess{str(session)}")
    if fold_idx is not None:
        filename_parts.append(f"fold{fold_idx}")
    if mode is not None:
        filename_parts.append(mode)
    
    filename = "_".join(filename_parts) + ".json"
    filepath = os.path.join(history_dir, filename)
    
    # Extract history data
    history_data = []
    for i, epoch_data in enumerate(model.history):
        epoch_dict = {
            'epoch': i + 1,
        }
        # Convert epoch data to serializable format
        for key, value in epoch_data.items():
            try:
                # Skip non-serializable items like callbacks
                if isinstance(value, (int, float, str, bool, type(None))):
                    epoch_dict[key] = value
                elif isinstance(value, np.ndarray):
                    epoch_dict[key] = value.item() if value.size == 1 else value.tolist()
                elif torch.is_tensor(value):
                    epoch_dict[key] = value.item() if value.numel() == 1 else value.cpu().tolist()
                # Try to convert other numeric types
                elif hasattr(value, '__float__'):
                    epoch_dict[key] = float(value)
            except Exception:
                # Skip items that can't be serialized
                continue
        
        history_data.append(epoch_dict)
    
    # Save to JSON
    try:
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        # print(f"Saved training history to {filepath}")
    except Exception as e:
        print(f"Warning: Failed to save training history: {e}")


class UnifiedExperimentRunner:
    """
    Unified experiment runner that handles all experiment modes and evaluation types.
    """
    
    def __init__(
        self,
        model: str,
        dataset: str,
        subjects: List[int],
        mode: str,
        eval_mode: str,
        seed: int,
        noise_type: Optional[str] = None,
        intensity: Optional[float] = None,
        tune: bool = False,
        overwrite: bool = False
    ):
        self.model = model
        self.dataset = dataset
        self.subjects = subjects
        self.mode = mode
        self.eval_mode = eval_mode
        self.seed = seed
        self.noise_type = noise_type
        self.intensity = intensity
        self.tune = tune
        self.overwrite = overwrite

        self.current_subject = -1
        self.current_session = -1
        
        # Initialize model cache manager
        self.cache_manager = ModelCacheManager(cache_root="model_cache", check_interval=10)
        
        # Validate noise parameters for noise-aware modes
        noise_requiring_modes = ["augment", "perturb", "augment_notune", "perturb_notune"]
        if mode in noise_requiring_modes and (not noise_type or intensity is None):
            raise ValueError(f"Mode '{mode}' requires both --noise_type and --intensity parameters")
        
        if mode == "test_perturb" and (not noise_type or intensity is None):
            noise_type = "gaussian"
            intensity = 10.0
            print(f"Using default noise type and intensity for test_perturb mode: {noise_type} {intensity}")

        # Set seeds
        set_seeds(seed)
        
        # Initialize noise configuration first
        self.noise_dict = None
        if noise_type and intensity:
            self.noise_dict = {"noise_type": noise_type, "intensity": intensity}
        
        # Initialize dataset and paradigm
        self._setup_dataset_and_paradigm()
        
        # Initialize model factory
        self.model_fn = MODEL_REGISTRY[model]
        
        # Create output paths
        self._create_output_paths()
    
    def _setup_dataset_and_paradigm(self):
        """Setup dataset and paradigm based on configuration."""
        if self.dataset == "BNCI2014_001":
            self.dataset_obj = BNCI2014_001()
            self.dataset_obj.subject_list = self.subjects
            self.paradigm = get_paradigm(resample=None, dataset=self.dataset)
        elif self.dataset == "Lee2019_SSVEP":
            self.dataset_obj = Lee2019_SSVEP()
            self.dataset_obj.subject_list = self.subjects
            self.paradigm = get_paradigm(resample=None, dataset=self.dataset)
        elif self.dataset == "BI2015a":
            self.dataset_obj = BI2015a()
            self.dataset_obj.subject_list = self.subjects
            self.paradigm = get_paradigm(resample=None, dataset=self.dataset)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
    
    def _create_output_paths(self):
        """Create output and HDF5 paths for the experiment."""
        # Determine session type based on eval_mode
        if self.eval_mode == "WithinSession":
            session_type = "WithinSessionEvaluation"
        elif self.eval_mode == "CrossSession":
            session_type = "CrossSessionEvaluation"
        elif self.eval_mode == "CrossSubject":
            session_type = "CrossSubjectEvaluation"
        else:
            session_type = "UnifiedEvaluation"
        
        # Create unique HDF5 path
        unique_id = uuid.uuid4().hex[:8]
        if self.dataset == "Lee2019_SSVEP":
            paradigm_name = "SSVEP"
        elif self.dataset == "BI2015a":
            paradigm_name = "ERP"
        else:
            paradigm_name = "MotorImagery"
        self.hdf5_path = create_hdf5_model_path(
            self.model, 
            self.seed, 
            '0train', 
            self.mode, 
            session_type=session_type,
            paradigm=paradigm_name,
            dataset=self.dataset
        )
        
        # Add noise-specific subdirectory if applicable
        if self.noise_dict:
            noise_subdir = f"{self.noise_dict['noise_type']}/{self.noise_dict['intensity']}"
            self.hdf5_path = os.path.join(self.hdf5_path, noise_subdir)
        
        # Add unique identifier
        self.hdf5_path = os.path.join(self.hdf5_path, f"subject{self.subjects[0]}-{self.subjects[-1]}_seed{self.seed}_{unique_id}.h5")
        
        # Create output directory
        self.output_dir = ""
    
    def _get_history_output_path(self):
        """Get the base output path for saving training history."""
        if self.dataset == "Lee2019_SSVEP":
            paradigm_name = "SSVEP"
        elif self.dataset == "BI2015a":
            paradigm_name = "ERP"
        else:
            paradigm_name = "MotorImagery"
        mode_str = self.mode
        if self.tune:
            mode_str = f"{self.mode}_tune"
        out_dir = create_output_path(
            self.model, 
            self.seed, 
            self.current_subject, 
            self.current_session, 
            mode_str, 
            session_type=self.eval_mode
        )
        return out_dir
    
    def _determine_data_dimensions(self) -> Tuple[int, int]:
        """Dynamically determine n_chans and sequence length from dataset samples."""
        # Get a small sample to determine dimensions
        X_sample, _, _ = self.paradigm.get_data(self.dataset_obj, subjects=[self.subjects[0]])
        
        if isinstance(X_sample, BaseEpochs):
            n_chans = X_sample.info['nchan']
            n_times = X_sample.get_data().shape[2]
        else:
            n_chans = X_sample.shape[1]
            n_times = X_sample.shape[2]
        
        return n_chans, n_times
    
    def _create_model(self, n_chans: int, n_times: int, n_outputs: int = None, try_cache: bool = True):
        """Create model instance with proper dimensions and optional caching."""
        # Determine number of outputs based on dataset
        if n_outputs is None:
            if self.dataset == "Lee2019_SSVEP":
                n_outputs = 4  # SSVEP has 4 classes
            elif self.dataset == "BI2015a":
                n_outputs = 2  # P300 ERP has 2 classes (target vs non-target)
            else:
                n_outputs = 2  # MotorImagery has 2 classes
        
        # Try to load from cache first (only for non-noise modes and when try_cache=True)
        if try_cache and not self.noise_dict and self.current_subject != -1 and self.current_session != -1:
            cached_model, config_matches = self.cache_manager.load_model(
                model_class=self.model_fn,
                config={'n_chans': n_chans, 'n_times': n_times, 'n_outputs': n_outputs},
                dataset=self.dataset,
                model_name=self.model,
                seed=self.seed,
                subject=self.current_subject,
                session=str(self.current_session),
                eval_mode=self.eval_mode,
                tuned=self.tune,
                checkpoint_type="best"  # Use best checkpoint for perturbation experiments
            )
            
            if cached_model is not None and config_matches:
                print(f"Loaded cached model for {self.model} subject {self.current_subject} session {self.current_session}")
                return cached_model
            elif cached_model is not None and not config_matches:
                print(f"Model configuration changed, will retrain for {self.model} subject {self.current_subject} session {self.current_session}")
        
        # Create new model
        model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
        assert(model is not None)
        # Set common model parameters
        model.max_epochs = DEFAULT_MAX_EPOCHS

        # Add caching callbacks
        if not self.noise_dict and self.current_subject != -1 and self.current_session != -1:
            if self.tune:
                # For tuned models, use periodic checkpoint callback
                cache_callback = create_periodic_checkpoint_callback(
                    cache_manager=self.cache_manager,
                    dataset=self.dataset,
                    model_name=self.model,
                    seed=self.seed,
                    subject=self.current_subject,
                    session=str(self.current_session),
                    eval_mode=self.eval_mode,
                    tuned=True,
                    check_interval=1  # Check every epoch for best model
                )
            else:
                # For baseline models, use simple cache callback
                cache_callback = create_model_cache_callback(
                    cache_manager=self.cache_manager,
                    dataset=self.dataset,
                    model_name=self.model,
                    seed=self.seed,
                    subject=self.current_subject,
                    session=str(self.current_session),
                    eval_mode=self.eval_mode,
                    tuned=False
                )
            
            # Add to existing callbacks
            if not hasattr(model, 'callbacks') or model.callbacks is None:
                model.callbacks = []
            model.callbacks.append(cache_callback)

        model.initialize()
        return model
    
    def _get_wrapped_model_function(self):
        """Get wrapped model function for noise modes."""
        if not self.noise_dict:
            return self.model_fn
        
        if self.mode in ['perturb', 'perturb_notune']:
            def wrapped_model_fn(n_chans, n_times, n_outputs=None):
                if n_outputs is None:
                    if self.dataset == "Lee2019_SSVEP":
                        n_outputs = 4
                    else:
                        n_outputs = 2  # MotorImagery and BI2015a (P300) both have 2 classes
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 200
                return TrainOnlyNoiseClassifier(
                    base_pipeline=base_model,
                    noise_type=self.noise_dict["noise_type"],
                    intensity=self.noise_dict["intensity"],
                    seed=self.seed
                )
        elif self.mode in ['augment', 'augment_notune']:
            def wrapped_model_fn(n_chans, n_times, n_outputs=None):
                if n_outputs is None:
                    if self.dataset == "Lee2019_SSVEP":
                        n_outputs = 4
                    else:
                        n_outputs = 2  # MotorImagery and BI2015a (P300) both have 2 classes
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 200
                return ConcatenatedNoiseAugmenter(
                    base_pipeline=base_model,
                    noise_type=self.noise_dict["noise_type"],
                    intensity=self.noise_dict["intensity"],
                    seed=self.seed
                )
        elif self.mode == 'test_perturb':
            def wrapped_model_fn(n_chans, n_times, n_outputs=None):
                if n_outputs is None:
                    if self.dataset == "Lee2019_SSVEP":
                        n_outputs = 4
                    else:
                        n_outputs = 2  # MotorImagery and BI2015a (P300) both have 2 classes
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 200
                return base_model
        else:
            return self.model_fn
        
        return wrapped_model_fn
    
    def prepare_data_cv(self) -> Tuple[Any, Dict]:
        """
        Prepare data for cross-validation based on eval_mode.
        
        Returns:
            cv_splitter: Cross-validation splitter
            cv_metadata: Metadata needed for logging results
        """
        if self.eval_mode == "WithinSession":
            # Use StratifiedKFold for within-session evaluation
            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_metadata = {
                "cv_type": "StratifiedKFold",
                "n_splits": 5,
                "split_level": "within_session"
            }
        elif self.eval_mode == "CrossSession":
            # Use LeaveOneGroupOut for cross-session evaluation
            cv_splitter = LeaveOneGroupOut()
            cv_metadata = {
                "cv_type": "LeaveOneGroupOut",
                "split_level": "cross_session"
            }
        elif self.eval_mode == "CrossSubject":
            # Use custom 3-fold splitter for cross-subject evaluation
            cv_splitter = ThreeFoldSubjectSplit()
            cv_metadata = {
                "cv_type": "ThreeFoldSubjectSplit",
                "n_splits": 3,
                "split_level": "cross_subject"
            }
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")
        
        return cv_splitter, cv_metadata
    
    def _evaluate_cv_fold(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_valid: np.ndarray, 
        y_valid: np.ndarray,
        fold_idx: int,
        cv_metadata: Dict,
        session: str,
        metadata_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate a single CV fold."""
        all_results = []
        
        if self.tune:
            # Apply two-stage hyperparameter optimization on X_train before evaluation on X_valid.
            self.current_session = session
            all_results.extend(self._run_hyperparameter_optimization(X_train, y_train, X_valid, y_valid, fold_idx, metadata_train))
        else:
            if self.mode == 'test_perturb':
                all_results.extend(self._train_and_evaluate_perturb(X_train, y_train, X_valid, y_valid, fold_idx, session))
            else:
                # Evaluate on X_valid without tuning.
                all_results.append(self._evaluate_without_tuning(X_train, y_train, X_valid, y_valid, fold_idx))

        for i, result in enumerate(all_results):
            # Add metadata
            all_results[i].update({
                'fold_idx': fold_idx,
                'cv_type': cv_metadata['cv_type'],
                'split_level': cv_metadata['split_level'],
                'session': session,
                'subject': self.current_subject  # Use the current subject being evaluated
            })
            
        return all_results
    
    def _tune_and_get_params(self, X_train, y_train, X_valid, y_valid, metadata_train, fold_idx):
        out_dir = create_output_path(self.model, self.seed, self.current_subject, self.current_session, self.mode, session_type=self.eval_mode)
        fold_output_dir = os.path.join(out_dir, f"Optuna/fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Get dataset-specific sampling rate
        resample_rate = get_dataset_sampling_rate(self.dataset)
        
        # Determine if we should use noise-aware optimization
        if self.noise_dict and self.mode in ['augment', 'perturb']:
            best_params, best_score = alternate_two_stage_optuna(
                model_fn=self._get_wrapped_model_function(),
                model_name=self.model,
                X=X_train,
                y=y_train,
                metadata=metadata_train,  # Use actual training metadata
                mode=self.mode,
                noise_dict=self.noise_dict,
                resample=resample_rate,
                seed=self.seed,
                output_root=fold_output_dir,
                arch_trials=20,
                train_trials=20
            )
        else:
            # Mode is tune (no noise)
            best_params, best_score = run_two_stage_optuna(
                model_fn=self._get_wrapped_model_function(),
                model_name=self.model,
                X=X_train,
                y=y_train,
                metadata=metadata_train,  # Use actual training metadata
                resample=resample_rate,
                seed=self.seed,
                output_root=fold_output_dir,
                arch_trials=10,
                train_trials=10,
                perturbed=False
            )        

        final_params = {}
        possible_params = get_all_model_params(self.model)
        module_params = [p for p in possible_params if 'module' in p]
        optimizer_params = [p for p in possible_params if 'optimizer' in p]

        prefix = ""
        module_prefix = f"{prefix}module__"
        optim_prefix = f"{prefix}optimizer__"
        
        for k, v in best_params.items():
            mod_prefixed_key = f"{module_prefix}{k}"
            optim_prefixed_key = f"{optim_prefix}{k}"
            if mod_prefixed_key in module_params:
                final_params[mod_prefixed_key] = v
            elif optim_prefixed_key in optimizer_params:
                final_params[optim_prefixed_key] = v
            else:
                final_params[k] = v

        return final_params, best_score

    def _run_hyperparameter_optimization(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_valid: np.ndarray, 
        y_valid: np.ndarray,
        fold_idx: int,
        metadata_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run two-stage hyperparameter optimization."""
        final_params, best_score = self._tune_and_get_params(X_train, y_train, X_valid, y_valid, metadata_train, fold_idx)

        # Train final model with best parameters
        n_chans, n_times = self._determine_data_dimensions()
        final_model = self._create_model(n_chans, n_times)
        final_params['verbose'] = 0
        final_model.set_params(**final_params)
        # Train on full training set
        start_time = time.time()
        final_model.module_.train()
        final_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Save training history after hyperparameter tuning is complete
        output_path = self._get_history_output_path()
        save_training_history(
            final_model, 
            output_path, 
            fold_idx=fold_idx, 
            subject=self.current_subject,
            session=str(self.current_session),
            mode=f"{self.mode}_tuned"
        )
        
        # Evaluate on validation set
        start_time = time.time()
        final_model.module_.eval()
        with torch.no_grad():
            y_pred_proba = final_model.predict_proba(X_valid)
            num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
            metrics_clean = compute_classification_metrics(y_valid, y_pred_proba, num_classes)
            validation_score = metrics_clean["roc_auc"]
        evaluation_time = time.time() - start_time
                
        results = []
        if self.mode == 'test_perturb':
            clean_score = validation_score
            session = self.current_session
            retrain = False
            if retrain:
                # Use a set threshold to restart training if clean score indicates underfitting.
                if clean_score < UNDERFITTING_THRESHOLD:
                    # Disable early stopping
                    print(f"Re-training model without EarlyStopping due to underfitting.")
                    final_model.set_params(**final_params)
                    final_model.callbacks = []
                    final_model.module_.train()
                    start_time = time.time()  
                    final_model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Save re-training history
                    output_path = self._get_history_output_path()
                    save_training_history(
                        final_model, 
                        output_path, 
                        fold_idx=fold_idx, 
                        subject=self.current_subject,
                        session=str(session),
                        mode=f"{self.mode}_tuned_retrained"
                    )
                    
                    final_model.module_.eval()
                    with torch.no_grad():
                        y_pred_proba = final_model.predict_proba(X_valid)
                        num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
                        metrics_retrain = compute_classification_metrics(y_valid, y_pred_proba, num_classes)
                        new_clean_score = metrics_retrain["roc_auc"]
                    clean_score = max(clean_score, new_clean_score)
            # Evaluate on corrupted data
            results.extend(self._evaluate_perturb(final_model, X_valid, y_valid, fold_idx, session, clean_score, training_time))
        else:
            results.append({
                'score': validation_score,
                'validation_roc_auc': metrics_clean["roc_auc"],
                'validation_accuracy': metrics_clean["accuracy"],
                'validation_precision': metrics_clean["precision"],
                'validation_recall': metrics_clean["recall"],
                'validation_f1': metrics_clean["f1"],
                'best_validation_score': best_score,
                'training_time': training_time,
                'evaluation_time': evaluation_time,
                'total_time': training_time + evaluation_time,            
            })
        for i, result in enumerate(results):
            results[i].update(final_params)
        return results
    
    def _evaluate_without_tuning(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_valid: np.ndarray, 
        y_valid: np.ndarray,
        fold_idx: int
    ) -> Dict[str, Any]:
        """Evaluate model without hyperparameter tuning."""
        n_chans, n_times = self._determine_data_dimensions()
        model = self._create_model(n_chans, n_times)
        
        # Check if model was loaded from cache
        model_was_cached = hasattr(model, '_was_cached') and model._was_cached
        
        if not model_was_cached:
            model.module_.train()
            # Apply noise if applicable
            if self.noise_dict and self.mode in ['augment', 'augment_notune']:
                # For augmentation modes, apply to training data
                if hasattr(model, 'concat_and_augment'):
                    X_train, y_train, groups = model.concat_and_augment(X_train, y_train)
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
        else:
            print(f"Using cached model, skipping training for {self.model} subject {self.current_subject} session {self.current_session}")
        
        # Save training history
        output_path = self._get_history_output_path()
        save_training_history(
            model, 
            output_path, 
            fold_idx=fold_idx, 
            subject=self.current_subject,
            session=str(self.current_session),
            mode=self.mode
        )
        
        # Evaluate
        start_time = time.time()
        model.module_.eval()
        with torch.no_grad():
            y_pred_proba = model.predict_proba(X_valid)
            num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
            metrics_clean = compute_classification_metrics(y_valid, y_pred_proba, num_classes)
            validation_score = metrics_clean["roc_auc"]
        evaluation_time = time.time() - start_time
        
        return {
            'score': validation_score,
            'validation_roc_auc': metrics_clean["roc_auc"],
            'validation_accuracy': metrics_clean["accuracy"],
            'validation_precision': metrics_clean["precision"],
            'validation_recall': metrics_clean["recall"],
            'validation_f1': metrics_clean["f1"],
            'fold_idx': fold_idx,
            'train_samples': len(X_train),
            'valid_samples': len(X_valid),
            'evaluation_time': evaluation_time,
            'total_time': evaluation_time
        }
    
    def _evaluate_perturb(self, trained_model, X_valid, y_valid, fold_idx, session, clean_score, training_time):        
        noise_types = ['eog', 'gaussian', 'dropout', 'spike']

        results = []
        trained_model.module_.eval()
        with torch.no_grad():
            # Compute clean metrics once for efficiency
            y_pred_proba_clean = trained_model.predict_proba(X_valid)
            num_classes_clean = 4 if self.dataset == "Lee2019_SSVEP" else 2
            metrics_clean = compute_classification_metrics(y_valid, y_pred_proba_clean, num_classes_clean)
            for noise_type in noise_types:
                # Use dynamic bounds based on dataset and noise type
                intensities = get_noise_intensities(self.dataset, noise_type, num_steps=20)            
                for intensity in intensities:
                    # Create corrupted validation data
                    noise_augmentor = EEGNoiseAugmentor(
                        noise_type=noise_type,
                        intensity=intensity,
                        seed=self.seed
                    )
                    
                    X_valid_corrupted = noise_augmentor.transform(X_valid)
                    
                    # Evaluate on corrupted data
                    start_time = time.time()                
                    y_pred_proba_corrupted = trained_model.predict_proba(X_valid_corrupted)
                    num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
                    metrics_corrupted = compute_classification_metrics(y_valid, y_pred_proba_corrupted, num_classes)
                    corrupted_score = metrics_corrupted["roc_auc"]
                    evaluation_time = time.time() - start_time
                    
                    # Calculate relative drop
                    relative_drop = (clean_score - corrupted_score) / clean_score if clean_score > 0 else 0.0
                    
                    results.append({
                        'fold_idx': fold_idx,
                        'noise_type': noise_type,
                        'intensity': intensity,
                        'clean_score': clean_score,
                        'corrupted_score': corrupted_score,
                        'clean_roc_auc': clean_score,
                        'clean_accuracy': metrics_clean["accuracy"],
                        'clean_precision': metrics_clean["precision"],
                        'clean_recall': metrics_clean["recall"],
                        'clean_f1': metrics_clean["f1"],
                        'corrupted_roc_auc': metrics_corrupted["roc_auc"],
                        'corrupted_accuracy': metrics_corrupted["accuracy"],
                        'corrupted_precision': metrics_corrupted["precision"],
                        'corrupted_recall': metrics_corrupted["recall"],
                        'corrupted_f1': metrics_corrupted["f1"],
                        'relative_drop': relative_drop,
                        'training_time': training_time,
                        'evaluation_time': evaluation_time,
                        'total_time': training_time + evaluation_time,
                        'session': session,
                    })

            return results

    def _train_and_evaluate_perturb(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_valid: np.ndarray, 
        y_valid: np.ndarray,
        fold_idx: int,
        session: str
    ) -> List[Dict[str, Any]]:
        """Evaluate model with increasing noise perturbations (test_perturb mode)."""
        if not self.noise_dict:
            raise ValueError("test_perturb mode requires noise_type and intensity")
        
        n_chans, n_times = self._determine_data_dimensions()
        set_seeds(self.seed)
        model = self._create_model(n_chans, n_times)
        
        
        # Train on clean data
        start_time = time.time()
        model.module_.train()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Save training history
        self.current_session = str(session)
        output_path = self._get_history_output_path()
        save_training_history(
            model, 
            output_path, 
            fold_idx=fold_idx, 
            subject=self.current_subject,
            session=str(session),
            mode=self.mode
        )
        
        # Evaluate on clean validation data
        start_time = time.time()
        model.module_.eval()
        with torch.no_grad():
            y_pred_proba = model.predict_proba(X_valid)
            num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
            metrics_clean = compute_classification_metrics(y_valid, y_pred_proba, num_classes)
            clean_score = metrics_clean["roc_auc"]
        evaluation_time = time.time() - start_time

        # If we are tuning, we incur too high a time cost to re-train the model this often.
        if not self.tune:
            # Use a set threshold to restart training if clean score indicates underfitting.
            if clean_score < UNDERFITTING_THRESHOLD:
                # Disable early stopping
                # print(f"Re-training model without EarlyStopping due to underfitting: {clean_score} < {UNDERFITTING_THRESHOLD}")
                new_callbacks = []
                for callback in model.callbacks:
                    if not isinstance(callback, EarlyStopping):
                        new_callbacks.append(callback)
                model.callbacks = new_callbacks
                model.module_.train()          
                start_time = time.time()  
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Save re-training history
                output_path = self._get_history_output_path()
                save_training_history(
                    model, 
                    output_path, 
                    fold_idx=fold_idx, 
                    subject=self.current_subject,
                    session=str(session),
                    mode=f"{self.mode}_retrained"
                )
                
                model.module_.eval()
                with torch.no_grad():
                    y_pred_proba = model.predict_proba(X_valid)
                    num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
                    metrics_retrain = compute_classification_metrics(y_valid, y_pred_proba, num_classes)
                    new_clean_score = metrics_retrain["roc_auc"]
                clean_score = max(clean_score, new_clean_score)
            
        results = []
        
        # Evaluate on corrupted data
        results.extend(self._evaluate_perturb(model, X_valid, y_valid, fold_idx, session, clean_score, training_time))
        
        return results
    
    def run_experiment(self) -> pd.DataFrame:
        """Run the complete experiment."""
        print(f"Starting unified experiment:")
        print(f"  Model: {self.model}")
        print(f"  Dataset: {self.dataset}")
        print(f"  Subjects: {self.subjects}")
        print(f"  Mode: {self.mode}")
        print(f"  Eval Mode: {self.eval_mode}")
        print(f"  Seed: {self.seed}")
        if self.noise_dict:
            print(f"  Noise: {self.noise_type} (intensity: {self.intensity})")
        print(f"  Tune: {self.tune}")

        mode_str = self.mode
        if self.tune:
            mode_str = f"{self.mode}_tune"
            

        if not self.overwrite:
            if check_skip_eval(self.model, self.seed, self.subjects, mode_str, self.noise_type, self.intensity, eval_mode=self.eval_mode, dataset=self.dataset):
                print(f"Skipping evaluation due to existing output files.")
                return None
        
        all_subject_results = []
        set_seeds(self.seed)
        if self.eval_mode == "CrossSubject":
            # For CrossSubject, get data from all subjects at once
            X, y, metadata = self.paradigm.get_data(self.dataset_obj, subjects=self.subjects)
            y_encoded = LabelEncoder().fit_transform(y)
            
            # Prepare cross-validation
            cv_splitter, cv_metadata = self.prepare_data_cv()
            
            # Run cross-validation with subject groups
            groups = metadata['subject'].values
            folds = list(enumerate(cv_splitter.split(X, y_encoded, groups=groups)))
            
            all_results = []
            for fold_idx, (train_idx, valid_idx) in folds:
                # Get the subjects that are in the evaluation set for this fold
                eval_subjects = np.unique(metadata.iloc[valid_idx]['subject'].values)
                eval_subjects_str = ','.join(map(str, sorted(eval_subjects)))
                session = f"fold_{fold_idx}_eval_subjects_{eval_subjects_str}"
                
                # Set current_subject to a representative value (first eval subject)
                self.current_subject = eval_subjects[0]
                
                X_train = X[train_idx]
                y_train = y_encoded[train_idx]
                X_valid = X[valid_idx]
                y_valid = y_encoded[valid_idx]
                metadata_train = metadata.iloc[train_idx]
                
                fold_results = self._evaluate_cv_fold(X_train, y_train, X_valid, y_valid, fold_idx, cv_metadata, session, metadata_train)
                
                # Add eval_subjects information to each result
                for result in fold_results:
                    result['eval_subjects'] = eval_subjects_str
                    result['n_eval_subjects'] = len(eval_subjects)
                
                all_results.extend(fold_results)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Aggregate fold results according to eval_mode and mode
            results_df = self._aggregate_fold_results(results_df)
            
            # Add experiment metadata
            results_df['model'] = self.model
            results_df['dataset'] = self.dataset
            results_df['mode'] = self.mode
            results_df['eval_mode'] = self.eval_mode
            results_df['seed'] = self.seed
            # For test_perturb mode, we don't want to override the noise_type and intensity values
            if self.mode != 'test_perturb':
                if self.noise_dict:
                    results_df['intensity'] = self.noise_dict['intensity']
                    results_df['noise_type'] = self.noise_dict['noise_type']
            results_df['tune'] = self.tune

            n_chans, n_times = self._determine_data_dimensions()
            model_instance = self._create_model(n_chans, n_times)
            row_headers = get_all_model_params(self.model)
            config = model_instance.get_params()
            for k, v in config.items():
                if k in row_headers and k not in results_df.columns:
                    results_df[k] = v

            all_subject_results.append(results_df)
            
        else:
            # For WithinSession and CrossSession, process subjects individually
            for subject in self.subjects:
                self.current_subject = subject
                # Get data
                X, y, metadata = self.paradigm.get_data(self.dataset_obj, subjects=[subject])
                y_encoded = LabelEncoder().fit_transform(y)
                
                # Prepare cross-validation
                cv_splitter, cv_metadata = self.prepare_data_cv()
                
                # Run cross-validation
                fold_indices = []
                fold_metadata = []
                if self.eval_mode == "CrossSession":
                    # For cross-session, use LeaveOneGroupOut with session groups
                    groups = metadata['session'].values
                    folds = list(enumerate(cv_splitter.split(X, y_encoded, groups=groups)))
                    for fold_idx, (train_idx, valid_idx) in folds:
                        session = metadata.iloc[valid_idx]['session'].values[0]
                        fold_indices.append((fold_idx, (train_idx, valid_idx), session))
                        fold_metadata.append(metadata.iloc[train_idx])
                elif self.eval_mode == "WithinSession":
                    groups = None                
                    # WithinSession indices will be different than CrossSession indices, since we iterate over sessions rather than entire dataset.
                    # Store the adjusted indices for later evaluation
                    for session in metadata['session'].unique():
                        session_idx = metadata['session'] == session
                        # Sessions are continuous 'blocks', so the first index will determine the offset for indexing into the original arrays.
                        first_idx = session_idx.index[0]
                        X_session, y_session = X[session_idx], y_encoded[session_idx]
                        session_metadata = metadata[session_idx]
                        # We are splitting over X_session and y_session, so we need to adjust the indices accordingly when storing.
                        folds = list(enumerate(cv_splitter.split(X_session, y_session)))
                        for fold_idx, (train_idx, valid_idx) in folds:
                            # Save unadjusted indices for session_metadata (which is already session-specific)
                            train_idx_session = train_idx.copy()
                            # Remember that the train and valid indices are relative to X_session and y_session, so we need to add the first_idx to get the correct indices into the original X and y_encoded arrays.
                            train_idx_global = train_idx + first_idx
                            valid_idx_global = valid_idx + first_idx
                            # Important note: Always store the 'session' as the session being used for evaluation / validation.
                            fold_indices.append((fold_idx, (train_idx_global, valid_idx_global), session))
                            fold_metadata.append(session_metadata.iloc[train_idx_session])
                elif self.eval_mode == "CrossSubject":
                    groups = metadata['subject'].values
                    folds = list(enumerate(cv_splitter.split(X, y_encoded, groups=groups)))
                    for fold_idx, (train_idx, valid_idx) in folds:
                        session = metadata.iloc[valid_idx]['subject'].values[0]
                        fold_indices.append((fold_idx, (train_idx, valid_idx), session))
                        fold_metadata.append(metadata.iloc[train_idx])

                all_results = []
                for i, (fold_idx, (train_idx, valid_idx), session) in enumerate(fold_indices):
                    X_train = X[train_idx]
                    y_train = y_encoded[train_idx]
                    X_valid = X[valid_idx]
                    y_valid = y_encoded[valid_idx]
                    metadata_train = fold_metadata[i]            
                    fold_results = self._evaluate_cv_fold(X_train, y_train, X_valid, y_valid, fold_idx, cv_metadata, session, metadata_train)
                    all_results.extend(fold_results)
                        
                # Convert results to DataFrame
                results_df = pd.DataFrame(all_results)
                
                # Aggregate fold results according to eval_mode and mode
                results_df = self._aggregate_fold_results(results_df)
                
                # Add experiment metadata
                results_df['subject'] = subject
                results_df['model'] = self.model
                results_df['dataset'] = self.dataset
                results_df['mode'] = self.mode
                results_df['eval_mode'] = self.eval_mode
                results_df['seed'] = self.seed
                # For test_perturb mode, we don't want to override the noise_type and intensity values
                if self.mode != 'test_perturb':
                    if self.noise_dict:
                        results_df['intensity'] = self.noise_dict['intensity']
                        results_df['noise_type'] = self.noise_dict['noise_type']
                results_df['tune'] = self.tune

                n_chans, n_times = self._determine_data_dimensions()
                model_instance = self._create_model(n_chans, n_times)
                row_headers = get_all_model_params(self.model)
                config = model_instance.get_params()
                for k, v in config.items():
                    if k in row_headers and k not in results_df.columns:
                        results_df[k] = v

                all_subject_results.append(results_df)            
                # Clean up HDF5 path
                if os.path.isdir(self.hdf5_path):
                    shutil.rmtree(self.hdf5_path)
        
        all_results_df = pd.concat(all_subject_results)
        # Save results
        self._save_results(all_results_df)                    
        return all_results_df
        
    def _aggregate_fold_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate fold results according to eval_mode and mode as specified in the spec.
        
        Returns:
            Aggregated results DataFrame
        """
        if self.eval_mode == "WithinSession":
            # WithinSession: calculate fold score means for '0train' and '1test' separately
            if 'fold_idx' in results_df.columns:
                if self.mode == 'test_perturb':
                    # test_perturb mode: calculate fold score means for both clean folds and corrupted validation data
                    agg_results = []
                    
                    # Handle corrupted scores at different intensities
                    if 'intensity' in results_df.columns:
                        for intensity in results_df['intensity'].unique():
                            if pd.notna(intensity) and intensity != 'clean':
                                intensity_df = results_df[results_df['intensity'] == intensity]
                                if len(intensity_df) > 0:
                                    for session in intensity_df['session'].unique():
                                        session_df = intensity_df[intensity_df['session'] == session]
                                        if len(session_df) > 0:
                                            agg_row = {
                                                'subject': session_df['subject'].iloc[0],
                                                'session': session,
                                                'score': session_df['corrupted_score'].mean() if 'corrupted_score' in session_df.columns else 0.0,
                                                'model': self.model,
                                                'mode': self.mode,
                                                'eval_mode': self.eval_mode,
                                                'seed': self.seed,
                                                'tune': self.tune,
                                                'noise_type': session_df['noise_type'].iloc[0] if 'noise_type' in session_df.columns else (self.noise_dict['noise_type'] if self.noise_dict else None),
                                                'intensity': intensity,
                                                'clean_score': session_df['clean_score'].mean() if 'clean_score' in session_df.columns else 0.0,
                                                'corrupted_score': session_df['corrupted_score'].mean() if 'corrupted_score' in session_df.columns else 0.0,
                                                'relative_drop': session_df['relative_drop'].mean() if 'relative_drop' in session_df.columns else 0.0,
                                                'training_time': session_df['training_time'].mean() if 'training_time' in session_df.columns else 0.0,
                                                'evaluation_time': session_df['evaluation_time'].mean() if 'evaluation_time' in session_df.columns else 0.0,
                                                'total_time': session_df['total_time'].mean() if 'total_time' in session_df.columns else 0.0
                                            }
                                            
                                            # Add all clean metrics (matching CrossSession)
                                            if 'clean_roc_auc' in session_df.columns:
                                                agg_row['clean_roc_auc'] = session_df['clean_roc_auc'].mean()
                                            if 'clean_accuracy' in session_df.columns:
                                                agg_row['clean_accuracy'] = session_df['clean_accuracy'].mean()
                                            if 'clean_precision' in session_df.columns:
                                                agg_row['clean_precision'] = session_df['clean_precision'].mean()
                                            if 'clean_recall' in session_df.columns:
                                                agg_row['clean_recall'] = session_df['clean_recall'].mean()
                                            if 'clean_f1' in session_df.columns:
                                                agg_row['clean_f1'] = session_df['clean_f1'].mean()
                                            
                                            # Add all corrupted metrics (matching CrossSession)
                                            if 'corrupted_roc_auc' in session_df.columns:
                                                agg_row['corrupted_roc_auc'] = session_df['corrupted_roc_auc'].mean()
                                            if 'corrupted_accuracy' in session_df.columns:
                                                agg_row['corrupted_accuracy'] = session_df['corrupted_accuracy'].mean()
                                            if 'corrupted_precision' in session_df.columns:
                                                agg_row['corrupted_precision'] = session_df['corrupted_precision'].mean()
                                            if 'corrupted_recall' in session_df.columns:
                                                agg_row['corrupted_recall'] = session_df['corrupted_recall'].mean()
                                            if 'corrupted_f1' in session_df.columns:
                                                agg_row['corrupted_f1'] = session_df['corrupted_f1'].mean()
                                            
                                            agg_results.append(agg_row)
                else:
                    # Regular modes: calculate fold score means for '0train' and '1test' separately
                    agg_results = []
                    for session in results_df['session'].unique():
                        session_df = results_df[results_df['session'] == session]
                        if len(session_df) > 0:
                            # Calculate mean scores across folds
                            agg_row = {
                                'subject': session_df['subject'].iloc[0],
                                'session': session,
                                'score': session_df['validation_score'].mean() if 'validation_score' in session_df.columns else session_df['score'].mean() if 'score' in session_df.columns else 0.0,
                                'model': self.model,
                                'mode': self.mode,
                                'eval_mode': self.eval_mode,
                                'seed': self.seed,
                                'tune': self.tune
                            }
                            
                            # Add all validation metrics (matching CrossSession)
                            if 'validation_roc_auc' in session_df.columns:
                                agg_row['validation_roc_auc'] = session_df['validation_roc_auc'].mean()
                            if 'validation_accuracy' in session_df.columns:
                                agg_row['validation_accuracy'] = session_df['validation_accuracy'].mean()
                            if 'validation_precision' in session_df.columns:
                                agg_row['validation_precision'] = session_df['validation_precision'].mean()
                            if 'validation_recall' in session_df.columns:
                                agg_row['validation_recall'] = session_df['validation_recall'].mean()
                            if 'validation_f1' in session_df.columns:
                                agg_row['validation_f1'] = session_df['validation_f1'].mean()
                            
                            # Add timing and sample info
                            if 'training_time' in session_df.columns:
                                agg_row['training_time'] = session_df['training_time'].mean()
                            if 'evaluation_time' in session_df.columns:
                                agg_row['evaluation_time'] = session_df['evaluation_time'].mean()
                            if 'total_time' in session_df.columns:
                                agg_row['total_time'] = session_df['total_time'].mean()
                            if 'train_samples' in session_df.columns:
                                agg_row['train_samples'] = session_df['train_samples'].mean()
                            if 'valid_samples' in session_df.columns:
                                agg_row['valid_samples'] = session_df['valid_samples'].mean()
                            
                            # Add best validation score if tuning
                            if 'best_validation_score' in session_df.columns:
                                agg_row['best_validation_score'] = session_df['best_validation_score'].mean()
                            
                            # Add noise information if applicable
                            if self.noise_dict:
                                agg_row['noise_type'] = self.noise_dict['noise_type']
                                agg_row['intensity'] = self.noise_dict['intensity']
                            
                            agg_results.append(agg_row)
                
                return pd.DataFrame(agg_results)
            else:
                return results_df
                
        elif self.eval_mode == "CrossSession":
            # simply drop the fold_idx column
            results_df = results_df.drop(columns=['fold_idx'])
            return results_df
        elif self.eval_mode == "CrossSubject":
            # For CrossSubject with 3-fold split, keep fold results separate
            # Each fold represents a different group of evaluation subjects
            # Don't aggregate - each fold should be its own record
            return results_df
        else:
            return results_df
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save results to appropriate output files."""
        # Use existing log_all_subjects function for consistent output structure
        from evaluation.experiment_utils import log_all_subjects
        
        # Determine eval_mode string for log_all_subjects
        eval_mode_str = f"{self.eval_mode}Evaluation"
        
        # Call log_all_subjects with proper parameters
        # For test_perturb mode, we don't want to override the intensity values
        if self.mode == 'test_perturb':
            intensity_param = 10.0
        else:
            intensity_param = self.noise_dict['intensity'] if self.noise_dict else None
            
        mode_str = self.mode
        if self.tune and self.mode != "tune":
            # Make sure the tuned and non-tuned modes are not mixed when creating output paths.
            mode_str = f"{self.mode}_tune"

        # Determine paradigm for log_all_subjects
        if self.dataset == "Lee2019_SSVEP":
            paradigm_name = "SSVEP"
        elif self.dataset == "BI2015a":
            paradigm_name = "ERP"
        else:
            paradigm_name = "MotorImagery"
        
        log_all_subjects(
            results=results_df,
            subject_list=self.subjects,
            model_name=self.model,
            mode=mode_str,
            noise_type=self.noise_dict['noise_type'] if self.noise_dict else None,
            intensity=intensity_param,
            seed=self.seed,
            eval_mode=eval_mode_str,
            paradigm=paradigm_name,
            dataset=self.dataset
        )


def main():
    """Main entry point for the unified experiment runner."""
    parser = argparse.ArgumentParser(description="Unified EEG Experiment Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, default="BNCI2014_001", choices=["BNCI2014_001", "Lee2019_SSVEP", "BI2015a"])
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["test_perturb", "multirun", "aggregate_only"])
                    #    choices=["baseline", "tune", "augment", "perturb", "augment_notune", "perturb_notune", "test_perturb", "multirun", "aggregate_only"])
    parser.add_argument("--eval_mode", type=str, required=True, 
                       choices=["WithinSession", "CrossSession", "CrossSubject"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_type", type=str, choices=["dropout", "gaussian", "eog", "spike"], default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    
    args = parser.parse_args()

    if args.dataset == "Lee2019_SSVEP":
        paradigm_name = "SSVEP"
    elif args.dataset == "BI2015a":
        paradigm_name = "ERP"
    else:
        paradigm_name = "MotorImagery"

    if args.mode == "aggregate_only":
        collect_all_results(paradigm=paradigm_name, dataset=args.dataset)
        sys.exit(0)

    set_seeds(args.seed)
    # Validate arguments
    if args.mode in ["augment", "perturb", "augment_notune", "perturb_notune"]:
        if args.noise_type is None or args.intensity is None:
            parser.error(f"Mode {args.mode} requires both --noise_type and --intensity")
    

    if args.mode in ["augment_notune", "perturb_notune"]:
        if args.tune:
            parser.error(f"Mode {args.mode} requires --tune flag to NOT be set.")

    if args.mode in ["test_perturb", "multirun"]:
        if not args.noise_type or args.noise_type is None or args.intensity is None:
            # Use default values for test_perturb mode, these are overwritten later anyway.
            args.noise_type = "gaussian"
            args.intensity = 10.0
    
    # Record start time
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)
    if args.mode == 'multirun':
        all_models = MODEL_REGISTRY.keys()
        limited_models = ["eegnet", "reegnet", "cnn_ncp"]
        # Use the specified model instead of iterating through limited_models
        model = args.model
        eval_mode = args.eval_mode
        seed = args.seed
        for mode in ["test_perturb"]:
            if not args.overwrite:
                mode_str = mode
                if args.tune and mode != "tune":
                    # Make sure the tuned and non-tuned modes are not mixed when creating output paths.
                    mode_str = f"{mode_str}_tune"
                if check_skip_eval(model, seed, args.subjects, mode_str, args.noise_type, args.intensity, eval_mode, paradigm_name, args.dataset):
                    continue
            try:
                runner = UnifiedExperimentRunner(
                    model=model,
                    dataset=args.dataset,
                    subjects=args.subjects,
                    mode=mode,
                    eval_mode=eval_mode,
                    seed=seed,
                    noise_type=args.noise_type,
                    intensity=args.intensity,
                    tune=args.tune,
                    overwrite=args.overwrite
                )
                results = runner.run_experiment()
                print(f"Experiment completed successfully. Results shape: {results.shape}")
            except Exception as e:
                print(f"Experiment failed: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    else:
        # Check if we should skip evaluation
        if not args.overwrite:
            if check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity, args.eval_mode, paradigm_name, args.dataset):
                sys.exit(0)
        
        try:
            # Create and run experiment
            runner = UnifiedExperimentRunner(
                model=args.model,
                dataset=args.dataset,
                subjects=args.subjects,
                mode=args.mode,
                eval_mode=args.eval_mode,
                seed=args.seed,
                noise_type=args.noise_type,
                intensity=args.intensity,
                tune=args.tune,
                overwrite=args.overwrite
            )
            
            results = runner.run_experiment()
            print(f"Experiment completed successfully. Results shape: {results.shape}")
            
            # Aggregate results if requested
            if args.aggregate:
                collect_all_results(paradigm=paradigm_name, dataset=args.dataset)
                
        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    # Record end time
    end_time = time.time()
    print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
