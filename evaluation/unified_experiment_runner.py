#!/usr/bin/env python3
"""
Unified Experiment Runner for EEG Models

This script consolidates the functionality from run_experiments.py and 
run_cross_session_experiments.py into a single, flexible experiment runner.

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
import argparse
import uuid
import warnings
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics import get_scorer
from tqdm import tqdm
import shutil

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY, get_paradigm
from globals import set_seeds
from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, run_two_stage_optuna, format_params, get_all_model_params
from utils import create_output_path, create_hdf5_model_path
from evaluation.experiment_utils import check_skip_eval, log_all_subjects, collect_all_results

# Import MOABB components
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list
from mne.epochs import BaseEpochs

# Try to import carbon footprint tracking
try:
    from codecarbon import EmissionsTracker
    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


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
        
        # Validate eval_mode
        if eval_mode == "CrossSubject":
            raise NotImplementedError("CrossSubject evaluation mode is not yet implemented")
        
        # Validate noise parameters for noise-aware modes
        noise_requiring_modes = ["augment", "perturb", "augment_notune", "perturb_notune", "test_perturb"]
        if mode in noise_requiring_modes and (not noise_type or intensity is None):
            raise ValueError(f"Mode '{mode}' requires both --noise_type and --intensity parameters")
        
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
            self.paradigm = get_paradigm(resample=None)  # Will be set dynamically
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
    
    def _create_output_paths(self):
        """Create output and HDF5 paths for the experiment."""
        # Determine session type based on eval_mode
        if self.eval_mode == "WithinSession":
            session_type = "WithinSessionEvaluation"
        elif self.eval_mode == "CrossSession":
            session_type = "CrossSessionEvaluation"
        else:
            session_type = "UnifiedEvaluation"
        
        # Create unique HDF5 path
        unique_id = uuid.uuid4().hex[:8]
        self.hdf5_path = create_hdf5_model_path(
            self.model, 
            self.seed, 
            '0train', 
            self.mode, 
            session_type=session_type
        )
        
        # Add noise-specific subdirectory if applicable
        if self.noise_dict:
            noise_subdir = f"{self.noise_dict['noise_type']}/{self.noise_dict['intensity']}"
            self.hdf5_path = os.path.join(self.hdf5_path, noise_subdir)
        
        # Add unique identifier
        self.hdf5_path = os.path.join(self.hdf5_path, f"subject{self.subjects[0]}-{self.subjects[-1]}_seed{self.seed}_{unique_id}.h5")
        
        # Create output directory
        self.output_dir = ""
    
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
    
    def _create_model(self, n_chans: int, n_times: int, n_outputs: int = 2):
        """Create model instance with proper dimensions."""
        model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
        
        # Set common model parameters
        model.train_split = None
        model.max_epochs = 100
        model.callbacks = []
        
        return model
    
    def _get_wrapped_model_function(self):
        """Get wrapped model function for noise modes."""
        if not self.noise_dict:
            return self.model_fn
        
        if self.mode in ['perturb', 'perturb_notune']:
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 100
                return TrainOnlyNoiseClassifier(
                    base_pipeline=base_model,
                    noise_type=self.noise_dict["noise_type"],
                    intensity=self.noise_dict["intensity"],
                    seed=self.seed
                )
        elif self.mode in ['augment', 'augment_notune']:
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 100
                return ConcatenatedNoiseAugmenter(
                    base_pipeline=base_model,
                    noise_type=self.noise_dict["noise_type"],
                    intensity=self.noise_dict["intensity"],
                    seed=self.seed
                )
        elif self.mode == 'test_perturb':
            def wrapped_model_fn(n_chans, n_times, n_outputs):
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = 100
                return base_model
        else:
            return self.model_fn
        
        return wrapped_model_fn
    
    def prepare_data_cv(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> Tuple[Any, Dict]:
        """
        Prepare data for cross-validation based on eval_mode.
        
        Returns:
            cv_splitter: Cross-validation splitter
            cv_metadata: Metadata needed for logging results
        """
        if self.eval_mode == "WithinSession":
            # Use StratifiedKFold for within-session evaluation
            cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
            cv_metadata = {
                "cv_type": "StratifiedKFold",
                "n_splits": 3,
                "split_level": "session"
            }
        elif self.eval_mode == "CrossSession":
            # Use LeaveOneGroupOut for cross-session evaluation
            cv_splitter = LeaveOneGroupOut()
            cv_metadata = {
                "cv_type": "LeaveOneGroupOut",
                "split_level": "session"
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
        results = {}
        
        if self.tune:
            # Apply two-stage hyperparameter optimization
            results.update(self._run_hyperparameter_optimization(X_train, y_train, X_valid, y_valid, fold_idx, metadata_train))
        else:
            # Evaluate without tuning
            results.update(self._evaluate_without_tuning(X_train, y_train, X_valid, y_valid, fold_idx))
        
        # Add metadata
        results.update({
            'fold_idx': fold_idx,
            'cv_type': cv_metadata['cv_type'],
            'split_level': cv_metadata['split_level'],
            'session': session,
            'subject': self.subjects[0]  # For now, assume single subject or use first subject
        })
        
        return results
    
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
        # Create output directory for this fold
        fold_output_dir = os.path.join(self.output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
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
                resample=None,  # Will be determined dynamically
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
                resample=None,  # Will be determined dynamically
                seed=self.seed,
                output_root=fold_output_dir,
                arch_trials=20,
                train_trials=20,
                perturbed=False
            )
        
        # Train final model with best parameters
        n_chans, n_times = self._determine_data_dimensions()
        final_model = self._create_model(n_chans, n_times)
        final_model.set_params(**best_params)
        
        # Train on full training set
        start_time = time.time()
        final_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        start_time = time.time()
        y_pred = final_model.predict_proba(X_valid)[:, 1]
        validation_score = roc_auc_score(y_valid, y_pred)
        evaluation_time = time.time() - start_time
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'validation_score': validation_score,
            'training_time': training_time,
            'evaluation_time': evaluation_time,
            'total_time': training_time + evaluation_time
        }
    
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
        
        # Apply noise if applicable
        if self.noise_dict and self.mode in ['augment', 'augment_notune']:
            # For augmentation modes, apply to training data
            if hasattr(model, 'concat_and_augment'):
                X_train_aug, y_train_aug, groups = model.concat_and_augment(X_train, y_train)
                # Use GroupKFold for augmented data
                cv = GroupKFold(n_splits=3)
                # Simple evaluation: train on augmented data, test on validation
                model.fit(X_train_aug, y_train_aug)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Evaluate
        start_time = time.time()
        y_pred = model.predict_proba(X_valid)[:, 1]
        validation_score = roc_auc_score(y_valid, y_pred)
        evaluation_time = time.time() - start_time
        
        return {
            'validation_score': validation_score,
            'evaluation_time': evaluation_time,
            'total_time': evaluation_time
        }
    
    def _evaluate_test_perturb(
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
        model = self._create_model(n_chans, n_times)
        
        # Train on clean data
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on clean validation data
        start_time = time.time()
        y_pred_clean = model.predict_proba(X_valid)[:, 1]
        clean_score = roc_auc_score(y_valid, y_pred_clean)
        evaluation_time = time.time() - start_time
        
        results = []
        
        # Add clean score result first
        results.append({
            'fold_idx': fold_idx,
            'intensity': None,  # None indicates clean data
            'clean_score': clean_score,
            'corrupted_score': None,
            'relative_drop': 0.0,  # No drop for clean data
            'training_time': training_time,
            'evaluation_time': evaluation_time,
            'total_time': training_time + evaluation_time,
            'session': session,
        })
        
        noise_type = self.noise_dict['noise_type']
                
        min_intensity = 10.0
        max_intensity = 90.0
        num_steps = 9
        if noise_type == 'gaussian':
            min_intensity = 3.0
            max_intensity = 30.0
        
        intensities = np.linspace(start=min_intensity, stop=max_intensity, num=num_steps)
        
        for intensity in intensities:
            # Create corrupted validation data
            noise_augmentor = EEGNoiseAugmentor(
                noise_type=self.noise_dict['noise_type'],
                intensity=intensity,
                seed=self.seed
            )
            
            X_valid_corrupted = noise_augmentor.transform(X_valid)
            
            # Evaluate on corrupted data
            start_time = time.time()
            y_pred_corrupted = model.predict_proba(X_valid_corrupted)[:, 1]
            evaluation_time = time.time() - start_time
            corrupted_score = roc_auc_score(y_valid, y_pred_corrupted)
            
            # Calculate relative drop
            relative_drop = (clean_score - corrupted_score) / clean_score if clean_score > 0 else 0.0
            
            results.append({
                'fold_idx': fold_idx,
                'intensity': intensity,
                'clean_score': clean_score,
                'corrupted_score': corrupted_score,
                'relative_drop': relative_drop,
                'training_time': training_time,
                'evaluation_time': evaluation_time,
                'total_time': training_time + evaluation_time,
                'session': session,
            })
        
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
        
        all_subject_results = []
        for subject in self.subjects:
            # Get data
            X, y, metadata = self.paradigm.get_data(self.dataset_obj, subjects=[subject])
            y_encoded = LabelEncoder().fit_transform(y)
            
            # Prepare cross-validation
            cv_splitter, cv_metadata = self.prepare_data_cv(X, y_encoded, metadata)
            
            # Run cross-validation
            all_results = []
            
            if self.eval_mode == "CrossSession":
                # For cross-session, use LeaveOneGroupOut with session groups
                groups = metadata['session'].values
            else:
                groups = None

            for fold_idx, (train_idx, valid_idx) in enumerate(cv_splitter.split(X, y_encoded, groups=groups)):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]
                
                # Get session information for this fold
                train_sessions = metadata.iloc[train_idx]['session'].values
                valid_sessions = metadata.iloc[valid_idx]['session'].values
                
                if self.mode == 'test_perturb':
                    fold_results = self._evaluate_test_perturb(X_train, y_train, X_valid, y_valid, fold_idx, valid_sessions[0])
                    all_results.extend(fold_results)
                else:
                    # Get training metadata for this fold
                    metadata_train = metadata.iloc[train_idx]
                    fold_result = self._evaluate_cv_fold(X_train, y_train, X_valid, y_valid, fold_idx, cv_metadata, valid_sessions[0], metadata_train)
                    all_results.append(fold_result)
                   
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
            if self.noise_dict:
                results_df['noise_type'] = self.noise_dict['noise_type']
                # Only set intensity for non-test_perturb modes
                if self.mode != 'test_perturb':
                    results_df['intensity'] = self.noise_dict['intensity']
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
                    
                    # Handle clean scores (intensity is None)
                    clean_df = results_df[results_df['intensity'].isna()] if 'intensity' in results_df.columns else results_df
                    if len(clean_df) > 0:
                        for session in clean_df['session'].unique():
                            session_df = clean_df[clean_df['session'] == session]
                            if len(session_df) > 0:
                                agg_row = {
                                    'subject': session_df['subject'].iloc[0],
                                    'session': session,
                                    'score': session_df['clean_score'].mean() if 'clean_score' in session_df.columns else session_df['score'].mean() if 'score' in session_df.columns else 0.0,
                                    'model': self.model,
                                    'mode': self.mode,
                                    'eval_mode': self.eval_mode,
                                    'seed': self.seed,
                                    'tune': self.tune,
                                    'noise_type': self.noise_dict['noise_type'] if self.noise_dict else None,
                                    'intensity': 'clean'
                                }
                                agg_results.append(agg_row)
                    
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
                                                'noise_type': self.noise_dict['noise_type'] if self.noise_dict else None,
                                                'intensity': intensity
                                            }
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
            intensity_param = None  # Let the results DataFrame handle intensity
        else:
            intensity_param = self.noise_dict['intensity'] if self.noise_dict else None
            

        log_all_subjects(
            results=results_df,
            subject_list=self.subjects,
            model_name=self.model,
            mode=self.mode,
            noise_type=self.noise_dict['noise_type'] if self.noise_dict else None,
            intensity=intensity_param,
            seed=self.seed,
            eval_mode=eval_mode_str
        )


def main():
    """Main entry point for the unified experiment runner."""
    parser = argparse.ArgumentParser(description="Unified EEG Experiment Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, default="BNCI2014_001", choices=["BNCI2014_001"])
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["baseline", "tune", "augment", "perturb", "augment_notune", "perturb_notune", "test_perturb"])
    parser.add_argument("--eval_mode", type=str, required=True, 
                       choices=["WithinSession", "CrossSession", "CrossSubject"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_type", type=str, choices=["dropout", "gaussian", "eog"], default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ["augment", "perturb", "augment_notune", "perturb_notune", "test_perturb"]:
        if not args.noise_type or args.intensity is None:
            parser.error(f"Mode {args.mode} requires both --noise_type and --intensity")
    
    if args.mode == "test_perturb" and not args.tune:
        print("Warning: test_perturb mode typically works best with --tune flag")
    
    # Check if we should skip evaluation
    if not args.overwrite:
        if check_skip_eval(args.model, args.seed, args.subjects, args.mode, args.noise_type, args.intensity, args.eval_mode):
            sys.exit(0)
    
    # Record start time
    start_time = time.time()
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    warnings.filterwarnings("ignore", message="warnEpochs", category=UserWarning)
    
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
            collect_all_results(paradigm='MotorImagery')
            
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
