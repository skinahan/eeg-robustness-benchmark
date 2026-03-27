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
import gc
try:
    import resource  # Unix-specific; optional
except ImportError:
    resource = None
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

from config import MODEL_REGISTRY, get_model_registry, get_paradigm, get_dataset_sampling_rate
from globals import set_seeds, DEFAULT_MAX_EPOCHS, UNDERFITTING_THRESHOLD, get_max_epochs_for_dataset, get_underfitting_threshold_for_dataset
from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor, ConcatenatedNoiseAugmenter
from evaluation.two_stage_hp_opt import alternate_two_stage_optuna, run_two_stage_optuna, format_params, get_all_model_params
from utils import (
    create_output_path,
    create_hdf5_model_path,
    get_noise_intensities,
    get_noise_perturbation_bounds,
    get_short_session_id,
    _CORRELATED_NOISE_TYPES,
)
from evaluation.experiment_utils import check_skip_eval, log_all_subjects, collect_all_results
from evaluation.metrics import compute_classification_metrics
from evaluation.model_cache_manager import ModelCacheManager
from evaluation.periodic_checkpoint_callback import create_periodic_checkpoint_callback, create_model_cache_callback
from evaluation.chunked_subject_trainer import train_with_subject_chunks, evaluate_with_subject_chunks
import json

# Import MOABB components
from moabb.datasets import BNCI2014_001, Lee2019_MI, Lee2019_SSVEP, BI2015a
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list
from mne.epochs import BaseEpochs

# Try to import carbon footprint tracking
try:
    from codecarbon import EmissionsTracker
    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


# Fixed seed for alpha_max calibration so that re-running yields identical alpha_max (Spec 3 PATCH 2).
_CALIBRATION_SEED = 202602


def _compute_lag1_autocorr_diagnostic(
    noise_type: str,
    X_clean: np.ndarray,
    X_corrupted: np.ndarray,
    max_epochs: int = 10,
) -> Optional[float]:
    """
    Compute mean lag-1 autocorrelation of injected noise (epsilon = X_corrupted - X_clean).
    For AR(1) noise this should be ~rho; for Gaussian ~0. Used as a diagnostic that the
    correct perturbation was applied (Plot 2 bug fix).
    """
    try:
        eps = X_corrupted[:max_epochs] - X_clean[:max_epochs]
        n_epochs, n_chans, n_times = eps.shape
        lag1_list = []
        for c in range(n_chans):
            flat = eps[:, c, :].ravel()
            if len(flat) < 3:
                continue
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
            if np.isfinite(corr):
                lag1_list.append(float(corr))
        return float(np.mean(lag1_list)) if lag1_list else None
    except Exception:
        return None


def _compute_perturbation_fingerprint(
    noise_type: str,
    X_clean: np.ndarray,
    X_corrupted: np.ndarray,
    max_epochs: int = 10,
) -> Dict[str, Any]:
    """
    PATCH 0.2: Compute lag-1 autocorrelation and residual (X̃ − X) mean/std for perturbation diagnostic.
    Prevents 'plot says ar1_drift but run is Gaussian' regressions.
    """
    out = {
        "perturbation_type": noise_type,
        "lag1_autocorrelation": None,
        "residual_mean": None,
        "residual_std": None,
    }
    try:
        eps = X_corrupted[:max_epochs] - X_clean[:max_epochs]
        out["residual_mean"] = float(np.mean(eps))
        out["residual_std"] = float(np.std(eps))
        lag1 = _compute_lag1_autocorr_diagnostic(noise_type, X_clean, X_corrupted, max_epochs=max_epochs)
        out["lag1_autocorrelation"] = lag1
    except Exception as e:
        out["error"] = str(e)
    return out


def _get_test_perturb_expected_scope(
    dataset: str,
    test_perturb_noise_types: Optional[List[str]] = None,
    test_perturb_gaussian_only: bool = False,
    test_perturb_gaussian_alpha_grid: Optional[List[float]] = None,
    test_perturb_num_steps: int = 20,
    saturation_file: Optional[str] = None,
):
    """Return (expected_noise_types, expected_intensities_by_noise) for check_skip_eval.
    Matches the logic in _evaluate_perturb so the skip check only requires what this run will produce
    (e.g. gaussian + alpha grid for Plot2), not all four noise types and full saturation steps.
    """
    sat_file = saturation_file or "saturation_results/saturation_points_summary.csv"
    if test_perturb_noise_types:
        noise_types = list(test_perturb_noise_types)
    elif test_perturb_gaussian_only:
        noise_types = ["gaussian"]
    else:
        noise_types = ["eog", "gaussian", "dropout", "spike"]

    expected_intensities_by_noise = {}
    for nt in noise_types:
        if nt == "gaussian" and test_perturb_gaussian_alpha_grid and len(test_perturb_gaussian_alpha_grid) > 0:
            _, sigma_max = get_noise_perturbation_bounds(dataset, "gaussian", saturation_file=sat_file)
            intensities = [float(alpha) * float(sigma_max) for alpha in test_perturb_gaussian_alpha_grid]
            expected_intensities_by_noise[nt] = sorted(set(float(x) for x in intensities))
        elif nt in _CORRELATED_NOISE_TYPES and test_perturb_gaussian_alpha_grid and len(test_perturb_gaussian_alpha_grid) > 0:
            _, nominal_max = get_noise_perturbation_bounds(dataset, nt, saturation_file=sat_file)
            intensities = [float(alpha) * float(nominal_max) for alpha in test_perturb_gaussian_alpha_grid]
            expected_intensities_by_noise[nt] = sorted(set(float(x) for x in intensities))
        else:
            intensities = get_noise_intensities(
                dataset, nt, num_steps=test_perturb_num_steps, saturation_file=sat_file
            )
            expected_intensities_by_noise[nt] = [float(x) for x in intensities]
    return noise_types, expected_intensities_by_noise


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback to resource module if psutil not available
        try:
            if hasattr(resource, 'getrusage'):
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
                if sys.platform == 'darwin':
                    return mem_usage / 1024 / 1024
                else:
                    return mem_usage / 1024
        except Exception:
            pass
    except Exception:
        pass
    return None


def log_memory_usage(stage=""):
    """Log current memory usage with detailed RSS and VSZ information."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024  # Resident Set Size (actual RAM)
        vms_mb = mem_info.vms / 1024 / 1024  # Virtual Memory Size
        
        print(f"[MEMORY] {stage}:")
        print(f"  RSS (actual RAM): {rss_mb:.2f} MB ({rss_mb/1024:.2f} GB)")
        print(f"  VSZ (virtual): {vms_mb:.2f} MB ({vms_mb/1024:.2f} GB)")
        
        # Check if we're in SLURM and compare to limit
        if 'SLURM_MEM_PER_NODE' in os.environ:
            slurm_mem_mb = int(os.environ['SLURM_MEM_PER_NODE']) / 1024**2  # SLURM reports in MB
            print(f"  SLURM limit: {slurm_mem_mb:.2f} MB ({slurm_mem_mb/1024:.2f} GB)")
            if rss_mb > 0:
                usage_pct = (rss_mb / slurm_mem_mb) * 100
                print(f"  RSS usage: {usage_pct:.1f}% of SLURM limit")
                if usage_pct > 80:
                    print(f"  [WARNING] Memory usage exceeds 80% of SLURM limit!")
        
        return rss_mb
    except ImportError:
        # Fallback to simpler logging if psutil not available
        mem_mb = get_memory_usage_mb()
        if mem_mb is not None:
            print(f"[MEMORY] {stage}: {mem_mb:.2f} MB ({mem_mb/1024:.2f} GB)")
        return mem_mb
    except Exception as e:
        print(f"[WARNING] Could not get detailed memory usage: {e}")
        mem_mb = get_memory_usage_mb()
        if mem_mb is not None:
            print(f"[MEMORY] {stage}: {mem_mb:.2f} MB ({mem_mb/1024:.2f} GB)")
        return mem_mb


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
    
    @staticmethod
    def get_fold_subjects(subjects):
        """
        Return which subjects go in each fold without requiring data.
        
        Args:
            subjects: List of subject IDs (can have duplicates)
        
        Returns:
            List of dicts, each containing:
                - fold_idx: int (0, 1, or 2)
                - train_subjects: list of subject IDs for training
                - eval_subjects: list of subject IDs for evaluation
        """
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # Calculate size of each evaluation group (should be n_subjects // 3)
        eval_group_size = n_subjects // 3
        
        if eval_group_size == 0:
            raise ValueError(f"Need at least 3 subjects for 3-fold split, got {n_subjects}")
        
        fold_configs = []
        for fold_idx in range(3):
            # Determine which subjects go in eval set for this fold
            eval_start = fold_idx * eval_group_size
            eval_end = eval_start + eval_group_size
            
            eval_subjects = unique_subjects[eval_start:eval_end].tolist()
            train_subjects = np.concatenate([
                unique_subjects[:eval_start],
                unique_subjects[eval_end:]
            ]).tolist()
            
            fold_configs.append({
                'fold_idx': fold_idx,
                'train_subjects': train_subjects,
                'eval_subjects': eval_subjects
            })
        
        return fold_configs
    
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


def _verify_and_log_max_epochs(model, dataset: str, training_context: str = ""):
    """
    Verify and log the max_epochs setting before training starts.
    
    Args:
        model: The model to check
        dataset: Dataset name for verification
        training_context: Optional context string for logging (e.g., "fold 1", "retraining")
    """
    expected_epochs = get_max_epochs_for_dataset(dataset)
    actual_epochs = getattr(model, 'max_epochs', None)
    
    context_str = f" ({training_context})" if training_context else ""
    
    if actual_epochs is None:
        print(f"WARNING: Model has no max_epochs attribute{context_str}")
    elif actual_epochs != expected_epochs:
        print(f"WARNING: max_epochs mismatch{context_str}: expected {expected_epochs} for {dataset}, but model has {actual_epochs}")
    else:
        print(f"Training with max_epochs={actual_epochs} for dataset {dataset}{context_str}")


def is_test_perturb_mode(mode: str) -> bool:
    """
    Check if a mode is test_perturb (including test_perturb_tune).
    
    This helper function ensures that logic that applies to test_perturb mode
    also applies to test_perturb_tune mode, unless it's appropriate to branch
    between non-tuned and tuned cases.
    
    Args:
        mode: Mode string (e.g., "test_perturb", "test_perturb_tune")
        
    Returns:
        True if mode is test_perturb or test_perturb_tune, False otherwise
    """
    return mode == 'test_perturb' or mode == 'test_perturb_tune' or mode.startswith('test_perturb')


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
        overwrite: bool = False,
        fold_idx: Optional[int] = None,
        train_subjects: Optional[List[int]] = None,
        eval_subjects: Optional[List[int]] = None,
        subject_chunk_size: Optional[int] = None,
        legacy: bool = False,
        # Disable the underfitting-based retraining pass (keeps training protocol fixed).
        disable_underfitting_retrain: bool = False,
        # ---- test_perturb configuration (pilot-safe; defaults preserve old behavior) ----
        test_perturb_noise_types: Optional[List[str]] = None,
        test_perturb_num_steps: int = 20,
        # If provided, will be used to (a) derive bounds, and (b) restrict evaluated noise types
        # to those listed in the file (instead of the full benchmark list).
        test_perturb_saturation_file: Optional[str] = None,
        test_perturb_gaussian_only: bool = False,
        test_perturb_gaussian_alpha_grid: Optional[List[float]] = None,
        test_perturb_target_snr_db: float = 0.0,
        test_perturb_target_snr_dbs: Optional[List[float]] = None,
        test_perturb_spatial_ell_multiplier: float = 1.0,
        test_perturb_emg_f_high: float = 80.0,
        test_perturb_emg_use_envelope: bool = False,
        test_perturb_ar1_rho: float = 0.97,
        test_perturb_emg_f_low: float = 20.0,
        # Plot 2 PATCH 0.2: optional dir to write perturbation_fingerprint.json
        plot2_diagnostics_dir: Optional[str] = None,
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
        
        # Fold-by-fold execution parameters (for CrossSubject memory optimization)
        self.fold_idx = fold_idx
        self.train_subjects = train_subjects
        self.eval_subjects = eval_subjects
        self.subject_chunk_size = subject_chunk_size
        self.legacy = legacy
        self.disable_underfitting_retrain = bool(disable_underfitting_retrain)

        # test_perturb settings (used only in _evaluate_perturb)
        self.test_perturb_noise_types = test_perturb_noise_types
        self.test_perturb_num_steps = int(test_perturb_num_steps)
        self.test_perturb_saturation_file = test_perturb_saturation_file
        self.test_perturb_gaussian_only = bool(test_perturb_gaussian_only)
        self.test_perturb_gaussian_alpha_grid = test_perturb_gaussian_alpha_grid
        self.test_perturb_target_snr_db = float(test_perturb_target_snr_db)
        self.test_perturb_target_snr_dbs = test_perturb_target_snr_dbs
        self.test_perturb_spatial_ell_multiplier = float(test_perturb_spatial_ell_multiplier)
        self.test_perturb_emg_f_high = float(test_perturb_emg_f_high)
        self.test_perturb_emg_use_envelope = bool(test_perturb_emg_use_envelope)
        self.test_perturb_ar1_rho = float(test_perturb_ar1_rho)
        self.test_perturb_emg_f_low = float(test_perturb_emg_f_low)
        self.plot2_diagnostics_dir = plot2_diagnostics_dir
        self._perturbation_fingerprint_written = False
        # Cache for alpha_max per (dataset, noise_type, split_id, params) for correlated perturbations (Spec 3 PATCH 2)
        self._correlated_alpha_max_cache = {}

        self.current_subject = -1
        self.current_session = -1
        
        # In legacy mode, disable chunked training to follow original protocol
        if self.legacy:
            if self.subject_chunk_size is not None and self.subject_chunk_size > 0:
                print(f"[LEGACY] Legacy mode enabled: disabling subject chunking (subject_chunk_size={self.subject_chunk_size} ignored)")
            self.subject_chunk_size = None
        
        # Initialize model cache manager
        self.cache_manager = ModelCacheManager(cache_root="model_cache", check_interval=10)
        
        # Validate noise parameters for noise-aware modes
        noise_requiring_modes = ["augment", "perturb", "augment_notune", "perturb_notune"]
        if mode in noise_requiring_modes and (not noise_type or intensity is None):
            raise ValueError(f"Mode '{mode}' requires both --noise_type and --intensity parameters")
        
        if is_test_perturb_mode(mode) and (not noise_type or intensity is None):
            noise_type = "gaussian"
            intensity = 10.0
            print(f"Using default noise type and intensity for test_perturb mode: {noise_type} {intensity}")

        # Set seeds
        set_seeds(seed)
        
        # Initialize noise configuration
        # IMPORTANT: For test_perturb mode, we do NOT set noise_dict because:
        # 1. Training happens on clean data (so models can be cached)
        # 2. Noise is only applied during evaluation in _evaluate_perturb
        # Setting noise_dict would disable caching, which we don't want
        self.noise_dict = None
        if noise_type and intensity and not is_test_perturb_mode(mode) and mode != "multirun":
            self.noise_dict = {"noise_type": noise_type, "intensity": intensity}
        
        # Initialize dataset and paradigm
        self._setup_dataset_and_paradigm()
        
        # Initialize model factory - use get_model_registry() to get latest registry
        # (supports runtime-registered variants)
        registry = get_model_registry()
        if model not in registry:
            available_models = sorted(registry.keys())
            raise ValueError(
                f"Model '{model}' not found in registry. "
                f"Available models: {available_models}"
            )
        self.model_fn = registry[model]
        
        # Create output paths
        self._create_output_paths()

    def _get_test_perturb_saturation_file(self) -> str:
        # Match utils.py defaults to preserve legacy behavior when not explicitly set.
        return self.test_perturb_saturation_file or "saturation_results/saturation_points_summary.csv"

    def _get_perturbation_params_dict(self, noise_type: str) -> Dict[str, Any]:
        """Return a JSON-serializable dict of perturbation params for this noise type (for result rows/manifests)."""
        if noise_type == "ar1_drift":
            return {"rho": float(getattr(self, "test_perturb_ar1_rho", 0.97))}
        if noise_type == "spatial_gaussian":
            return {"ell_multiplier": float(getattr(self, "test_perturb_spatial_ell_multiplier", 1.0))}
        if noise_type == "emg_band":
            return {
                "f_low": float(getattr(self, "test_perturb_emg_f_low", 20.0)),
                "f_high": float(getattr(self, "test_perturb_emg_f_high", 80.0)),
                "envelope_on": bool(getattr(self, "test_perturb_emg_use_envelope", False)),
            }
        return {}

    def _get_correlated_noise_params_key(self, noise_type: str) -> tuple:
        """Return a hashable key of perturbation parameters for cache key (Spec 3 PATCH 2)."""
        rho = float(getattr(self, "test_perturb_ar1_rho", 0.97))
        ell = float(getattr(self, "test_perturb_spatial_ell_multiplier", 1.0))
        emg_high = float(getattr(self, "test_perturb_emg_f_high", 80.0))
        emg_env = bool(getattr(self, "test_perturb_emg_use_envelope", False))
        emg_low = float(getattr(self, "test_perturb_emg_f_low", 20.0))
        return (noise_type, rho, ell, emg_low, emg_high, emg_env)

    def _get_alpha_max_for_correlated_noise(
        self, noise_type: str, X_sample: np.ndarray, split_id: Optional[Tuple[str, ...]] = None
    ) -> float:
        """
        Compute alpha_max so that at intensity=alpha_max, SNR_dB ~ target (E[||X||^2] / E[||alpha*eps||^2]).
        Target SNR_dB is test_perturb_target_snr_db (default 0). Formula: alpha_max = alpha_max_0dB * 10^(-target_snr_db/20).
        Cached per (dataset, noise_type, split_id, params). Uses a fixed calibration seed for reproducibility (Spec 3 PATCH 2).
        """
        target_db = float(getattr(self, "_eval_target_snr_db_override", None) or self.test_perturb_target_snr_db)
        params_key = self._get_correlated_noise_params_key(noise_type)
        key = (self.dataset, noise_type, split_id or (getattr(self, "eval_mode", "CrossSession"),), params_key, target_db)
        if key in self._correlated_alpha_max_cache:
            return self._correlated_alpha_max_cache[key]
        augmentor = self._make_correlated_noise_augmentor(noise_type, intensity=1.0, seed=_CALIBRATION_SEED)
        X_aug = augmentor.transform(X_sample)
        eps = X_aug - X_sample
        n_epochs = X_sample.shape[0]
        mean_X_sq = float(np.sum(X_sample ** 2) / n_epochs)
        mean_eps_sq = float(np.sum(eps ** 2) / n_epochs)
        if mean_eps_sq <= 0 or not np.isfinite(mean_eps_sq):
            alpha_max_0db = 1.0
        else:
            alpha_max_0db = float(np.sqrt(mean_X_sq / mean_eps_sq))
            if not np.isfinite(alpha_max_0db) or alpha_max_0db <= 0:
                alpha_max_0db = 1.0
        alpha_max = alpha_max_0db * (10.0 ** (-target_db / 20.0))
        if not np.isfinite(alpha_max) or alpha_max <= 0:
            alpha_max = 1.0
        self._correlated_alpha_max_cache[key] = alpha_max
        return alpha_max

    def _make_correlated_noise_augmentor(self, noise_type: str, intensity: float, seed: Optional[int] = None):
        """Build EEGNoiseAugmentor for correlated noise with optional escalation params. seed=None uses self.seed (Spec 3)."""
        rng = seed if seed is not None else self.seed
        kwargs = {"noise_type": noise_type, "intensity": intensity, "seed": rng}
        if noise_type == "ar1_drift":
            kwargs["ar1_rho"] = float(getattr(self, "test_perturb_ar1_rho", 0.97))
        if noise_type == "spatial_gaussian" and getattr(self, "test_perturb_spatial_ell_multiplier", None) is not None:
            kwargs["spatial_ell_multiplier"] = float(self.test_perturb_spatial_ell_multiplier)
        if noise_type == "emg_band":
            if getattr(self, "test_perturb_emg_f_high", None) is not None:
                kwargs["emg_f_high"] = float(self.test_perturb_emg_f_high)
            if getattr(self, "test_perturb_emg_f_low", None) is not None:
                kwargs["emg_f_low"] = float(self.test_perturb_emg_f_low)
            if getattr(self, "test_perturb_emg_use_envelope", False):
                kwargs["emg_use_envelope"] = True
        if noise_type in ("gain_drift", "offset_drift"):
            pass  # use default gain_drift_rho, offset_drift_rho
        if noise_type == "temporal_jitter":
            kwargs["jitter_sfreq"] = float(
                getattr(self, "test_perturb_jitter_sfreq", get_dataset_sampling_rate(self.dataset))
            )
        if noise_type == "spatial_dropout":
            kwargs["spatial_dropout_cluster_size"] = float(getattr(self, "test_perturb_spatial_dropout_cluster_size", 0.25))
        if noise_type == "ar1_plus_gain_drift":
            kwargs["ar1_rho"] = float(getattr(self, "test_perturb_ar1_rho", 0.97))
            kwargs["gain_drift_intensity"] = float(getattr(self, "test_perturb_gain_drift_intensity", intensity * 0.5))
        if noise_type == "ar1_plus_offset_drift":
            kwargs["ar1_rho"] = float(getattr(self, "test_perturb_ar1_rho", 0.97))
            kwargs["offset_drift_intensity"] = float(getattr(self, "test_perturb_offset_drift_intensity", intensity * 0.5))
        return EEGNoiseAugmentor(**kwargs)

    def _setup_dataset_and_paradigm(self):
        """Setup dataset and paradigm based on configuration."""
        if self.dataset == "BNCI2014_001":
            self.dataset_obj = BNCI2014_001()
            self.dataset_obj.subject_list = self.subjects
            self.paradigm = get_paradigm(resample=None, dataset=self.dataset)
        elif self.dataset == "Lee2019_MI":
            self.dataset_obj = Lee2019_MI()
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
        
        # Get the first session from the dataset dynamically (for path naming)
        # Default to '0train' if we can't determine sessions
        default_session = '0train'
        session_for_path = default_session
        if self.paradigm is not None and self.dataset_obj is not None and len(self.subjects) > 0:
            try:
                # Load data for first subject to get actual session names
                X_sample, y_sample, metadata_sample = self.paradigm.get_data(self.dataset_obj, subjects=[self.subjects[0]])
                if 'session' in metadata_sample.columns:
                    sessions = sorted(metadata_sample['session'].unique().tolist())
                    if len(sessions) > 0:
                        session_for_path = sessions[0]
            except Exception as e:
                # If we can't load data, use default
                pass
        
        self.hdf5_path = create_hdf5_model_path(
            self.model, 
            self.seed, 
            session_for_path, 
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
        # Construct mode_str consistently with other places in the code
        # If tune flag is set, append "_tune" to mode (e.g., "test_perturb" -> "test_perturb_tune")
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
    
    def _create_model(self, n_chans: int, n_times: int, n_outputs: int = None, try_cache: bool = True, fold_idx: Optional[int] = None):
        """
        Create model instance with proper dimensions and optional caching.
        
        Args:
            n_chans: Number of channels
            n_times: Number of time points
            n_outputs: Number of output classes (auto-detected if None)
            try_cache: Whether to try loading from cache
            fold_idx: Fold index (required to prevent data leakage)
        """
        # Determine number of outputs based on dataset
        if n_outputs is None:
            if self.dataset == "Lee2019_SSVEP":
                n_outputs = 4  # SSVEP has 4 classes
            elif self.dataset == "BI2015a":
                n_outputs = 2  # P300 ERP has 2 classes (target vs non-target)
            else:
                n_outputs = 2  # MotorImagery has 2 classes
        
        # Try to load from cache first (only for non-noise modes and when try_cache=True)
        # Skip cache loading if overwrite is True (user explicitly wants to retrain)
        if try_cache and not self.overwrite and not self.noise_dict and self.current_subject != -1 and self.current_session != -1:
            # For tuned models, try "best" first (saved during training), then "final" as fallback
            # For baseline models, use "final" (only checkpoint type saved by ModelCacheCallback)
            # Note: The cache manager uses lenient config comparison, so the minimal config here
            # (n_chans, n_times, n_outputs) will match saved configs that include additional
            # params like max_epochs and verbose (which are excluded from hash comparison)
            # 
            # IMPORTANT: fold_idx is included in cache key to prevent
            # data leakage between different folds.
            checkpoint_types_to_try = ["best", "final"] if self.tune else ["final"]
            
            # Debug: Print cache lookup attempt
            print(f"[CACHE] Attempting to load cached model: {self.model}, subject {self.current_subject}, "
                  f"session {self.current_session}, eval_mode {self.eval_mode}, tuned {self.tune}, "
                  f"fold_idx {fold_idx}")
            
            cached_model = None
            config_matches = False
            
            for checkpoint_type in checkpoint_types_to_try:
                # Debug: Print which checkpoint type we're trying
                print(f"[CACHE] Trying checkpoint type: {checkpoint_type}")
                
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
                    checkpoint_type=checkpoint_type,
                    fold_idx=fold_idx  # Critical to prevent fold mixing
                )
                
                if cached_model is not None and config_matches:
                    print(f"[CACHE] Successfully loaded {checkpoint_type} checkpoint")
                    break  # Successfully loaded, no need to try other checkpoint types
                elif cached_model is None:
                    print(f"[CACHE] No {checkpoint_type} checkpoint found")
                elif not config_matches:
                    print(f"[CACHE] {checkpoint_type} checkpoint found but config mismatch")
            
            if cached_model is not None and config_matches:
                print(f"Loaded cached model for {self.model} subject {self.current_subject} session {self.current_session}" + 
                      (f" fold {fold_idx}" if fold_idx is not None else ""))
                return cached_model
            elif cached_model is not None and not config_matches:
                print(f"Model configuration changed, will retrain for {self.model} subject {self.current_subject} session {self.current_session}")
        elif not try_cache:
            print(f"[CACHE] Cache loading disabled (try_cache=False)")
        elif self.overwrite:
            print(f"[CACHE] Cache loading disabled (overwrite=True, will retrain)")
        elif self.noise_dict:
            print(f"[CACHE] Cache loading disabled (noise_dict is set)")
        elif self.current_subject == -1 or self.current_session == -1:
            print(f"[CACHE] Cache loading disabled (current_subject={self.current_subject}, current_session={self.current_session})")
        
        # Create new model
        model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
        assert(model is not None)
        # Set common model parameters with dataset-specific max_epochs
        # Pass eval_mode to get CrossSubject-specific epoch limit (20 epochs)
        model.max_epochs = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)

        # Add caching callbacks
        if not self.noise_dict and self.current_subject != -1 and self.current_session != -1:
            # Always pass fold_idx when available, regardless of eval_mode
            # This ensures proper cache key generation and prevents data leakage
            fold_idx_for_callback = fold_idx
            
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
                    check_interval=1,  # Check every epoch for best model
                    fold_idx=fold_idx_for_callback
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
                    tuned=False,
                    fold_idx=fold_idx_for_callback
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
                base_model.max_epochs = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)
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
                base_model.max_epochs = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)
                return ConcatenatedNoiseAugmenter(
                    base_pipeline=base_model,
                    noise_type=self.noise_dict["noise_type"],
                    intensity=self.noise_dict["intensity"],
                    seed=self.seed
                )
        elif is_test_perturb_mode(self.mode):
            def wrapped_model_fn(n_chans, n_times, n_outputs=None):
                if n_outputs is None:
                    if self.dataset == "Lee2019_SSVEP":
                        n_outputs = 4
                    else:
                        n_outputs = 2  # MotorImagery and BI2015a (P300) both have 2 classes
                base_model = self.model_fn(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
                base_model.max_epochs = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)
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
    
    def _evaluate_cv_fold_chunked(
        self,
        train_subjects: List[int],
        eval_subjects: List[int],
        fold_idx: int,
        session: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a single CV fold using chunked subject loading for memory efficiency.
        
        This method loads subjects in small chunks to avoid loading all data into
        memory at once. It's particularly useful for CrossSubject evaluation with
        large numbers of subjects.
        """
        # Set current_session for proper cache key generation
        self.current_session = session
        
        # Determine data dimensions (need to load at least one sample to get shape)
        # For efficiency, we'll load just the first training subject to get dimensions
        if len(train_subjects) > 0:
            sample_X, _, _ = self.paradigm.get_data(
                self.dataset_obj, subjects=[train_subjects[0]]
            )
            n_chans, n_times = sample_X.shape[1], sample_X.shape[2]
            del sample_X
            gc.collect()
        else:
            # Fallback: use defaults or determine from dataset
            n_chans, n_times = self._determine_data_dimensions()
        
        # Create model
        model = self._create_model(n_chans, n_times, fold_idx=fold_idx)
        
        # Check if model was loaded from cache
        model_was_cached = hasattr(model, '_was_cached') and model._was_cached
        
        all_results = []
        
        # Handle hyperparameter optimization if enabled
        if self.tune:
            # For HPO with chunked training, we need to load training data once
            # HPO itself uses chunked training (via two_stage_hp_opt), but final model
            # training after HPO needs the full training set
            print(f"[CHUNKED_TRAINING] Hyperparameter optimization enabled - loading training data for HPO...")
            
            # Load training data for HPO (needed for _tune_and_get_params and final model training)
            X_train_hpo, y_train_hpo, metadata_train_hpo = self.paradigm.get_data(
                self.dataset_obj, subjects=train_subjects
            )
            if X_train_hpo.dtype == np.float64:
                X_train_hpo = X_train_hpo.astype(np.float32)
            
            if isinstance(y_train_hpo[0], str):
                y_train_hpo = LabelEncoder().fit_transform(y_train_hpo)
            
            # Load validation data for HPO evaluation
            X_valid_hpo, y_valid_hpo, _ = self.paradigm.get_data(
                self.dataset_obj, subjects=eval_subjects
            )
            if X_valid_hpo.dtype == np.float64:
                X_valid_hpo = X_valid_hpo.astype(np.float32)
            
            if isinstance(y_valid_hpo[0], str):
                label_encoder = LabelEncoder()
                y_valid_hpo = label_encoder.fit_transform(y_valid_hpo)
            
            # Run HPO (which internally uses chunked training)
            hpo_results = self._run_hyperparameter_optimization(
                X_train_hpo, y_train_hpo, X_valid_hpo, y_valid_hpo, 
                fold_idx, metadata_train_hpo
            )
            
            # Clean up HPO data
            del X_train_hpo, y_train_hpo, X_valid_hpo, y_valid_hpo, metadata_train_hpo
            gc.collect()
            
            all_results.extend(hpo_results)
            return all_results
        
        # Non-tuning path: Train and evaluate with chunked loading
        if not model_was_cached:
            # Train using chunked subject loading
            print(f"[CHUNKED_TRAINING] Starting chunked training for fold {fold_idx}")
            model = train_with_subject_chunks(
                model=model,
                paradigm=self.paradigm,
                dataset_obj=self.dataset_obj,
                train_subjects=train_subjects,
                chunk_size=self.subject_chunk_size,
                max_epochs_per_chunk=None,  # Use model's max_epochs
                verbose=True
            )
            
            # Save training history (if needed - may need to adapt this)
            output_path = self._get_history_output_path()
            try:
                save_training_history(
                    model,
                    output_path,
                    fold_idx=fold_idx,
                    subject=self.current_subject,
                    session=str(self.current_session),
                    mode=self.mode
                )
            except Exception as e:
                print(f"[WARNING] Could not save training history: {e}")
        else:
            print(f"[CHUNKED_TRAINING] Using cached model, skipping training")
        
        # Evaluate using chunked subject loading
        print(f"[CHUNKED_EVAL] Starting chunked evaluation for fold {fold_idx}")
        start_time = time.time()
        
        y_valid_all, y_pred_proba_all = evaluate_with_subject_chunks(
            model=model,
            paradigm=self.paradigm,
            dataset_obj=self.dataset_obj,
            eval_subjects=eval_subjects,
            chunk_size=self.subject_chunk_size,
            verbose=True
        )
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
        metrics = compute_classification_metrics(y_valid_all, y_pred_proba_all, num_classes)
        
        # Handle different modes
        if is_test_perturb_mode(self.mode):
            # For test_perturb mode, we need to evaluate on corrupted data
            # This is more complex with chunked evaluation, so for now we'll
            # load validation data once for perturbation evaluation
            # (This could be optimized further if needed)
            print(f"[CHUNKED_EVAL] Loading validation data for perturbation evaluation...")
            X_valid_all, y_valid_all_perturb, _ = self.paradigm.get_data(
                self.dataset_obj, subjects=eval_subjects
            )
            if X_valid_all.dtype == np.float64:
                X_valid_all = X_valid_all.astype(np.float32)
            
            if isinstance(y_valid_all_perturb[0], str):
                label_encoder = getattr(model, '_label_encoder', None)
                if label_encoder is not None:
                    y_valid_all_perturb = label_encoder.transform(y_valid_all_perturb)
                else:
                    y_valid_all_perturb = LabelEncoder().fit_transform(y_valid_all_perturb)
            
            # Evaluate perturbations
            perturb_results = self._evaluate_perturb(
                trained_model=model,
                X_valid=X_valid_all,
                y_valid=y_valid_all_perturb,
                fold_idx=fold_idx,
                session=session,
                clean_score=metrics["roc_auc"],
                training_time=0  # Training time not tracked in chunked mode
            )
            
            # Clean up
            del X_valid_all, y_valid_all_perturb
            gc.collect()
            
            all_results.extend(perturb_results)
        else:
            # Standard evaluation
            result = {
                'score': metrics["roc_auc"],
                'validation_roc_auc': metrics["roc_auc"],
                'validation_accuracy': metrics["accuracy"],
                'validation_precision': metrics["precision"],
                'validation_recall': metrics["recall"],
                'validation_f1': metrics["f1"],
                'fold_idx': fold_idx,
                'train_samples': -1,  # Not tracked in chunked mode (can be computed if needed)
                'valid_samples': len(y_valid_all),
                'evaluation_time': evaluation_time,
                'total_time': evaluation_time
            }
            all_results.append(result)
        
        # Add metadata to all results
        cv_metadata = {
            'cv_type': 'ThreeFoldSubjectSplit',
            'split_level': 'subject'
        }
        for result in all_results:
            result.update({
                'fold_idx': fold_idx,
                'cv_type': cv_metadata['cv_type'],
                'split_level': cv_metadata['split_level'],
                'session': session,
                'subject': self.current_subject
            })
        
        return all_results

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
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a single CV fold.
        
        IMPORTANT: Sets current_session before creating model to ensure proper cache key generation.
        For WithinSession, fold_idx is passed to prevent data leakage between folds.
        
        """
        all_results = []
        
        # Set current_session BEFORE creating model (critical for cache key generation)
        self.current_session = session
        
        if self.tune:
            # Apply two-stage hyperparameter optimization on X_train before evaluation on X_valid.
            all_results.extend(self._run_hyperparameter_optimization(X_train, y_train, X_valid, y_valid, fold_idx, metadata_train))
        else:
            if is_test_perturb_mode(self.mode):
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
        """
        Run hyperparameter optimization and return best parameters.
        
        IMPORTANT: This method should ONLY be called when self.tune is True.
        """
        assert self.tune, f"_tune_and_get_params called but self.tune is False. This is a logic error."
        out_dir = create_output_path(self.model, self.seed, self.current_subject, self.current_session, self.mode, session_type=self.eval_mode)
        fold_output_dir = os.path.join(out_dir, f"Optuna/fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Memory optimization: Convert to float32 early to reduce memory usage by 50%
        # This is critical for large datasets to avoid OOM errors during hyperparameter optimization
        # when NumPy fancy indexing creates copies of the data
        if isinstance(X_train, np.ndarray) and X_train.dtype == np.float64:
            X_train = X_train.astype(np.float32)
            print(f"[MEMORY] Converted X_train to float32 before hyperparameter optimization. Shape: {X_train.shape}, Memory saved: {X_train.nbytes / 1024**3:.2f} GB")

        # Get dataset-specific sampling rate
        resample_rate = get_dataset_sampling_rate(self.dataset)
        
        # Determine if we should use noise-aware optimization
        # Pass chunked training parameters if using chunked training
        # Legacy mode disables chunked training to follow original protocol
        use_chunked_for_hpo = (
            not self.legacy and
            self.subject_chunk_size is not None and 
            self.subject_chunk_size > 0 and
            self.train_subjects is not None and
            len(self.train_subjects) > 0 and
            self.eval_mode == "CrossSubject"
        )
        
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
                train_trials=20,
                dataset=self.dataset,
                eval_mode=self.eval_mode,
                paradigm=self.paradigm if use_chunked_for_hpo else None,
                dataset_obj=self.dataset_obj if use_chunked_for_hpo else None,
                train_subjects=self.train_subjects if use_chunked_for_hpo else None,
                subject_chunk_size=self.subject_chunk_size if use_chunked_for_hpo else None
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
                perturbed=False,
                dataset=self.dataset,
                eval_mode=self.eval_mode,
                paradigm=self.paradigm if use_chunked_for_hpo else None,
                dataset_obj=self.dataset_obj if use_chunked_for_hpo else None,
                train_subjects=self.train_subjects if use_chunked_for_hpo else None,
                subject_chunk_size=self.subject_chunk_size if use_chunked_for_hpo else None
            )        

        final_params = {}
        possible_params = get_all_model_params(self.model)
        module_params = [p for p in possible_params if 'module' in p]
        optimizer_params = [p for p in possible_params if 'optimizer' in p]

        prefix = ""
        module_prefix = f"{prefix}module__"
        optim_prefix = f"{prefix}optimizer__"
        
        # Extract wiring_arch_index - it's handled at factory level, not via set_params
        # Check both with and without prefix
        wiring_arch_index_to_filter = [
            'wiring_arch_index',
            f'{prefix}wiring_arch_index',
            'module__wiring_arch_index',
            f'{prefix}module__wiring_arch_index'
        ]
        
        wiring_arch_index = None
        for k, v in best_params.items():
            # Extract wiring_arch_index - it's used only during model creation, not for set_params
            if k in wiring_arch_index_to_filter:
                wiring_arch_index = v
                continue
                
            mod_prefixed_key = f"{module_prefix}{k}"
            optim_prefixed_key = f"{optim_prefix}{k}"
            if mod_prefixed_key in module_params:
                final_params[mod_prefixed_key] = v
            elif optim_prefixed_key in optimizer_params:
                final_params[optim_prefixed_key] = v
            else:
                final_params[k] = v

        return final_params, best_score, wiring_arch_index

    def _run_hyperparameter_optimization(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_valid: np.ndarray, 
        y_valid: np.ndarray,
        fold_idx: int,
        metadata_train: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run two-stage hyperparameter optimization.
        
        IMPORTANT: This method should ONLY be called when self.tune is True.
        """
        assert self.tune, f"_run_hyperparameter_optimization called but self.tune is False. This is a logic error."
        final_params, best_score, wiring_arch_index = self._tune_and_get_params(X_train, y_train, X_valid, y_valid, metadata_train, fold_idx)

        # Train final model with best parameters
        n_chans, n_times = self._determine_data_dimensions()
        
        # If wiring_arch_index was selected during optimization, use it when creating the model
        # Wrap the model factory to pass wiring_arch_index as a kwarg
        original_model_fn = self.model_fn
        if wiring_arch_index is not None:
            def model_fn_with_wiring(**kwargs):
                kwargs['wiring_arch_index'] = wiring_arch_index
                return original_model_fn(**kwargs)
            self.model_fn = model_fn_with_wiring
        
        final_model = self._create_model(n_chans, n_times, try_cache=False)
        
        # Restore original model_fn
        self.model_fn = original_model_fn
        final_params['verbose'] = 0
        # Restore max_epochs for final training run after optimization
        # Note: During optimization, CrossSubject uses max_epochs=5 to speed up trials
        # After optimization, CrossSubject uses max_epochs=20 for the full training run
        # Other eval modes use their normal dataset-specific values
        final_params['max_epochs'] = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)
        
        # Defensive check: Remove wiring_arch_index if it somehow made it into final_params
        # (it should have been filtered out in _tune_and_get_params, but this ensures safety)
        wiring_arch_index_keys = ['wiring_arch_index', 'module__wiring_arch_index']
        for key in wiring_arch_index_keys:
            final_params.pop(key, None)
        
        final_model.set_params(**final_params)
        
        # Train final model - use chunked training if enabled
        start_time = time.time()
        final_model.module_.train()
        _verify_and_log_max_epochs(final_model, self.dataset, f"fold {fold_idx} (tuned)")
        
        # Check if we should use chunked training for final model
        # Legacy mode disables chunked training to follow original protocol
        use_chunked_for_final = (
            not self.legacy and
            self.subject_chunk_size is not None and 
            self.subject_chunk_size > 0 and
            self.train_subjects is not None and
            len(self.train_subjects) > 0 and
            self.eval_mode == "CrossSubject"
        )
        
        if use_chunked_for_final:
            print(f"[CHUNKED_TRAINING] Training final model (after HPO) with chunked training (chunk_size={self.subject_chunk_size})")
            final_model = train_with_subject_chunks(
                model=final_model,
                paradigm=self.paradigm,
                dataset_obj=self.dataset_obj,
                train_subjects=self.train_subjects,
                chunk_size=self.subject_chunk_size,
                max_epochs_per_chunk=None,  # Use model's max_epochs
                verbose=True
            )
        else:
            # Use standard training (loads all data at once)
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
        
        # Use chunked evaluation if we used chunked training
        if use_chunked_for_final:
            print(f"[CHUNKED_EVAL] Evaluating final model (after HPO) with chunked evaluation (chunk_size={self.subject_chunk_size})")
            y_valid_all, y_pred_proba_all = evaluate_with_subject_chunks(
                model=final_model,
                paradigm=self.paradigm,
                dataset_obj=self.dataset_obj,
                eval_subjects=self.eval_subjects,
                chunk_size=self.subject_chunk_size,
                verbose=True
            )
            num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
            metrics_clean = compute_classification_metrics(y_valid_all, y_pred_proba_all, num_classes)
            validation_score = metrics_clean["roc_auc"]
        else:
            # Standard evaluation (X_valid already loaded)
            with torch.no_grad():
                y_pred_proba = final_model.predict_proba(X_valid)
                num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
                metrics_clean = compute_classification_metrics(y_valid, y_pred_proba, num_classes)
                validation_score = metrics_clean["roc_auc"]
        
        evaluation_time = time.time() - start_time
                
        results = []
        if is_test_perturb_mode(self.mode):
            clean_score = validation_score
            session = self.current_session
            retrain = False
            if retrain:
                # Use a dataset-specific threshold to restart training if clean score indicates underfitting.
                underfitting_threshold = get_underfitting_threshold_for_dataset(self.dataset)
                if clean_score < underfitting_threshold:
                    # Disable early stopping
                    print(f"Re-training model without EarlyStopping due to underfitting.")
                    final_model.set_params(**final_params)
                    final_model.callbacks = []
                    # Ensure max_epochs is still correct after removing early stopping
                    final_model.max_epochs = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)
                    final_model.module_.train()
                    start_time = time.time()
                    _verify_and_log_max_epochs(final_model, self.dataset, f"fold {fold_idx} (tuned, retraining)")
                    
                    # Use chunked training for retraining if enabled
                    if use_chunked_for_final:
                        print(f"[CHUNKED_TRAINING] Retraining final model with chunked training (chunk_size={self.subject_chunk_size})")
                        final_model = train_with_subject_chunks(
                            model=final_model,
                            paradigm=self.paradigm,
                            dataset_obj=self.dataset_obj,
                            train_subjects=self.train_subjects,
                            chunk_size=self.subject_chunk_size,
                            max_epochs_per_chunk=None,
                            verbose=True
                        )
                    else:
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
                    
                    # Use chunked evaluation if we used chunked training
                    if use_chunked_for_final:
                        y_valid_retrain, y_pred_proba_retrain = evaluate_with_subject_chunks(
                            model=final_model,
                            paradigm=self.paradigm,
                            dataset_obj=self.dataset_obj,
                            eval_subjects=self.eval_subjects,
                            chunk_size=self.subject_chunk_size,
                            verbose=True
                        )
                        num_classes = 4 if self.dataset == "Lee2019_SSVEP" else 2
                        metrics_retrain = compute_classification_metrics(y_valid_retrain, y_pred_proba_retrain, num_classes)
                        new_clean_score = metrics_retrain["roc_auc"]
                    else:
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
        """
        Evaluate model without hyperparameter tuning.
        
        IMPORTANT: 
        - current_session must be set before calling this method (done in _evaluate_cv_fold).
        - fold_idx is passed to _create_model to ensure proper cache key for WithinSession.
        - This method should ONLY be called when self.tune is False.
        """
        assert not self.tune, f"_evaluate_without_tuning called but self.tune is True. This is a logic error."
        n_chans, n_times = self._determine_data_dimensions()        
        model = self._create_model(n_chans, n_times, fold_idx=fold_idx)
        
        # Check if model was loaded from cache
        model_was_cached = hasattr(model, '_was_cached') and model._was_cached
        
        if not model_was_cached:
            model.module_.train()
            _verify_and_log_max_epochs(model, self.dataset, f"fold {fold_idx}")
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
        # Normalize: iid_gaussian (spec name) -> gaussian for evaluation (Spec 3 PATCH 3)
        def _norm_nt(nt: str) -> str:
            return "gaussian" if nt == "iid_gaussian" else nt

        # Precedence:
        # 1) explicit override via --test_perturb_noise_types
        # 2) gaussian-only flag
        # 3) if a saturation file was explicitly provided, evaluate only types present in that file
        # 4) fallback to benchmark default list
        noise_types: List[str]
        if self.test_perturb_noise_types:
            noise_types = [_norm_nt(nt) for nt in self.test_perturb_noise_types]
        elif self.test_perturb_gaussian_only:
            noise_types = ["gaussian"]
        elif self.test_perturb_saturation_file:
            try:
                sat = pd.read_csv(self.test_perturb_saturation_file)
                if "dataset" not in sat.columns or "noise_type" not in sat.columns:
                    raise ValueError(
                        f"Saturation file missing required columns: {self.test_perturb_saturation_file} "
                        f"(need 'dataset' and 'noise_type'; got {list(sat.columns)})"
                    )
                sat = sat[sat["dataset"].astype(str) == str(self.dataset)]
                # Preserve file order (stable unique)
                noise_types = [_norm_nt(nt) for nt in dict.fromkeys(sat["noise_type"].astype(str).tolist())]
                if not noise_types:
                    raise ValueError(
                        f"Saturation file has no rows for dataset='{self.dataset}': {self.test_perturb_saturation_file}"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to derive noise types from saturation file: {e}")
                noise_types = ["eog", "gaussian", "dropout", "spike"]
        else:
            noise_types = ["eog", "gaussian", "dropout", "spike"]

        split_id = (getattr(self, "eval_mode", "CrossSession"), str(session))
        target_snr_dbs = getattr(self, "test_perturb_target_snr_dbs", None)
        if target_snr_dbs and len(target_snr_dbs) > 0:
            target_snr_db_list = [float(t) for t in target_snr_dbs]
        else:
            target_snr_db_list = [float(self.test_perturb_target_snr_db)]

        results = []
        trained_model.module_.eval()
        with torch.no_grad():
            # Compute clean metrics once for efficiency
            y_pred_proba_clean = trained_model.predict_proba(X_valid)
            num_classes_clean = 4 if self.dataset == "Lee2019_SSVEP" else 2
            metrics_clean = compute_classification_metrics(y_valid, y_pred_proba_clean, num_classes_clean)
            # Memory optimization: Delete prediction array after computing metrics
            del y_pred_proba_clean
            gc.collect()
            
            for noise_type in noise_types:
                target_snr_db = target_snr_db_list[0]
                # Runtime assertion: ar1_drift must use correlated path and valid rho (Plot 2 bug fix)
                if noise_type == "ar1_drift":
                    rho = float(getattr(self, "test_perturb_ar1_rho", 0.97))
                    assert 0 < rho < 1, (
                        f"ar1_drift requires test_perturb_ar1_rho in (0, 1), got {rho}"
                    )
                # Per-noise-type SNR/alpha_max for result rows (Spec 3 PATCH 1)
                this_empirical_snr_db = float("nan")
                this_alpha_max = None
                # Use dynamic bounds based on dataset and noise type.
                # Defaults preserve historical behavior (num_steps=20, default saturation file),
                # but pilot studies can override to Gaussian-only alpha grids.
                sigma_max = None
                if (
                    noise_type == "gaussian"
                    and self.test_perturb_gaussian_alpha_grid is not None
                    and len(self.test_perturb_gaussian_alpha_grid) > 0
                ):
                    _, sigma_max = get_noise_perturbation_bounds(
                        self.dataset, "gaussian", saturation_file=self._get_test_perturb_saturation_file()
                    )
                    intensities = [
                        float(alpha) * float(sigma_max) for alpha in self.test_perturb_gaussian_alpha_grid
                    ]
                    # de-duplicate while preserving sorted order
                    intensities = sorted(set(float(x) for x in intensities))
                    this_alpha_max = sigma_max
                elif (
                    noise_type in _CORRELATED_NOISE_TYPES
                    and self.test_perturb_gaussian_alpha_grid is not None
                    and len(self.test_perturb_gaussian_alpha_grid) > 0
                ):
                    # Plot 2 Overhaul: loop over target_snr_db for dual-SNR eval (-12, -6)
                    for target_snr_db in target_snr_db_list:
                        self._eval_target_snr_db_override = target_snr_db
                        alpha_max = self._get_alpha_max_for_correlated_noise(noise_type, X_valid, split_id=split_id)
                        sigma_max = alpha_max
                        this_alpha_max = alpha_max
                        intensities = [
                            float(alpha) * float(alpha_max) for alpha in self.test_perturb_gaussian_alpha_grid
                        ]
                        intensities = sorted(set(float(x) for x in intensities))
                        augmentor_calib = self._make_correlated_noise_augmentor(
                            noise_type, intensity=1.0, seed=_CALIBRATION_SEED
                        )
                        X_calib = augmentor_calib.transform(X_valid)
                        eps_calib = X_calib - X_valid
                        n_ep = X_valid.shape[0]
                        mean_X_sq = float(np.sum(X_valid ** 2) / n_ep)
                        mean_eps_sq = float(np.sum(eps_calib ** 2) / n_ep)
                        if mean_eps_sq > 0 and np.isfinite(mean_eps_sq) and alpha_max > 0:
                            this_empirical_snr_db = float(10.0 * np.log10(mean_X_sq / (alpha_max ** 2 * mean_eps_sq)))
                        else:
                            this_empirical_snr_db = float("nan")
                        del X_calib, eps_calib
                        for intensity in intensities:
                            noise_augmentor = self._make_correlated_noise_augmentor(noise_type, intensity=intensity)
                            X_valid_corrupted = noise_augmentor.transform(X_valid)
                            if intensity == intensities[-1]:
                                fp = _compute_perturbation_fingerprint(noise_type, X_valid, X_valid_corrupted, max_epochs=10)
                                if getattr(self, "plot2_diagnostics_dir", None) and not getattr(self, "_perturbation_fingerprint_written", False):
                                    try:
                                        import os
                                        fp["target_snr_db"] = target_snr_db
                                        fp["empirical_snr_db"] = this_empirical_snr_db
                                        os.makedirs(self.plot2_diagnostics_dir, exist_ok=True)
                                        with open(os.path.join(self.plot2_diagnostics_dir, "perturbation_fingerprint.json"), "w", encoding="utf-8") as f:
                                            json.dump(fp, f, indent=2)
                                        self._perturbation_fingerprint_written = True
                                    except Exception:
                                        pass
                            start_time = time.time()
                            y_pred_proba_corrupted = trained_model.predict_proba(X_valid_corrupted)
                            metrics_corrupted = compute_classification_metrics(y_valid, y_pred_proba_corrupted, num_classes_clean)
                            evaluation_time = time.time() - start_time
                            corrupted_score = metrics_corrupted["roc_auc"]
                            relative_drop = (clean_score - corrupted_score) / clean_score if clean_score > 0 else 0.0
                            results.append({
                                'fold_idx': fold_idx, 'noise_type': noise_type, 'perturbation_type': noise_type,
                                'params': self._get_perturbation_params_dict(noise_type),
                                'intensity': intensity,
                                'alpha': (float(intensity) / float(sigma_max)) if (sigma_max and float(sigma_max) != 0.0) else None,
                                'target_snr_db': target_snr_db,
                                'empirical_snr_db': this_empirical_snr_db,
                                'alpha_max': this_alpha_max,
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
                            del X_valid_corrupted, y_pred_proba_corrupted
                            if len(results) % 10 == 0:
                                gc.collect()
                    self._eval_target_snr_db_override = None
                    continue
                else:
                    intensities = get_noise_intensities(
                        self.dataset,
                        noise_type,
                        num_steps=self.test_perturb_num_steps,
                        saturation_file=self._get_test_perturb_saturation_file(),
                    )
                    this_alpha_max = float(np.max(intensities)) if len(intensities) and np.isfinite(intensities).any() else None

                # Assert ar1_drift used correlated path (alpha_max), not gaussian sigma_max
                if noise_type == "ar1_drift":
                    assert this_alpha_max is not None and (
                        noise_type in _CORRELATED_NOISE_TYPES
                    ), "ar1_drift must use correlated path and data-derived alpha_max, not gaussian sigma_max"

                for intensity in intensities:
                    # Create corrupted validation data (use escalation params for correlated types)
                    if noise_type in _CORRELATED_NOISE_TYPES:
                        noise_augmentor = self._make_correlated_noise_augmentor(noise_type, intensity=intensity)
                        assert getattr(noise_augmentor, "noise_type", None) == noise_type, (
                            f"Augmentor noise_type must be {noise_type!r} for correlated path"
                        )
                    else:
                        noise_augmentor = EEGNoiseAugmentor(
                            noise_type=noise_type,
                            intensity=intensity,
                            seed=self.seed
                        )
                    X_valid_corrupted = noise_augmentor.transform(X_valid)
                    # PATCH 0.2: Perturbation fingerprint diagnostic (lag-1 + residual mean/std)
                    if intensity == intensities[-1]:
                        fp = _compute_perturbation_fingerprint(
                            noise_type, X_valid, X_valid_corrupted, max_epochs=10
                        )
                        lag1 = fp.get("lag1_autocorrelation")
                        if lag1 is not None:
                            print(
                                f"[test_perturb] {noise_type} lag1_autocorr={lag1:.4f} "
                                "(expected ~rho for AR1, ~0 for gaussian)"
                            )
                        if getattr(self, "plot2_diagnostics_dir", None) and not getattr(self, "_perturbation_fingerprint_written", False):
                            try:
                                import os
                                ddir = self.plot2_diagnostics_dir
                                os.makedirs(ddir, exist_ok=True)
                                # Spec §6.3 NEW 2: fingerprint must include target_snr_db and empirical_snr_db
                                fp["target_snr_db"] = target_snr_db
                                fp["empirical_snr_db"] = this_empirical_snr_db
                                path = os.path.join(ddir, "perturbation_fingerprint.json")
                                with open(path, "w", encoding="utf-8") as f:
                                    json.dump(fp, f, indent=2)
                                self._perturbation_fingerprint_written = True
                            except Exception as e:
                                print(f"[WARNING] Could not write perturbation_fingerprint.json: {e}")
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
                        'perturbation_type': noise_type,
                        'params': self._get_perturbation_params_dict(noise_type),
                        'intensity': intensity,
                        'alpha': (float(intensity) / float(sigma_max)) if (sigma_max and float(sigma_max) != 0.0) else None,
                        'target_snr_db': target_snr_db,
                        'empirical_snr_db': this_empirical_snr_db,
                        'alpha_max': this_alpha_max,
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
                    
                    # Memory management: Delete corrupted data copy after evaluation
                    del X_valid_corrupted, y_pred_proba_corrupted
                    # Periodically force garbage collection to free memory
                    if len(results) % 10 == 0:
                        gc.collect()

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
        """
        Evaluate model with increasing noise perturbations (test_perturb mode).
        
        IMPORTANT: This method should ONLY be called when self.tune is False.
        For tuned models, use _run_hyperparameter_optimization instead.
        
        Note: This method trains on clean data and evaluates on corrupted data.
        Noise intensities are determined dynamically in _evaluate_perturb from
        the saturation file, so self.noise_dict is not required.
        """
        assert not self.tune, f"_train_and_evaluate_perturb called but self.tune is True. This is a logic error."
        n_chans, n_times = self._determine_data_dimensions()
        set_seeds(self.seed)
        model = self._create_model(n_chans, n_times, fold_idx=fold_idx)
        
        # Check if model was loaded from cache
        model_was_cached = hasattr(model, '_was_cached') and model._was_cached
        
        # Train on clean data (only if not cached)
        if not model_was_cached:
            start_time = time.time()
            model.module_.train()
            _verify_and_log_max_epochs(model, self.dataset, f"fold {fold_idx}, session {session}")
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
        else:
            print(f"Using cached model, skipping training for {self.model} subject {self.current_subject} session {self.current_session}")
            training_time = 0.0  # No training time if using cache
        
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
        # Also skip retraining if model was loaded from cache (it was already trained and validated)
        if not self.disable_underfitting_retrain and not self.tune and not model_was_cached:
            # Use a dataset-specific threshold to restart training if clean score indicates underfitting.
            underfitting_threshold = get_underfitting_threshold_for_dataset(self.dataset)
            if clean_score < underfitting_threshold:
                # Disable early stopping
                # print(f"Re-training model without EarlyStopping due to underfitting: {clean_score} < {UNDERFITTING_THRESHOLD}")
                new_callbacks = []
                for callback in model.callbacks:
                    if not isinstance(callback, EarlyStopping):
                        new_callbacks.append(callback)
                model.callbacks = new_callbacks
                # Ensure max_epochs is still correct after removing early stopping
                model.max_epochs = get_max_epochs_for_dataset(self.dataset, eval_mode=self.eval_mode)
                model.module_.train()
                start_time = time.time()
                _verify_and_log_max_epochs(model, self.dataset, f"fold {fold_idx}, session {session} (retraining)")
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

        # Construct mode_str to match what will be used in _save_results
        # This ensures check_skip_eval looks for files in the correct directory
        # If tune flag is set, append "_tune" to mode (e.g., "test_perturb" -> "test_perturb_tune")
        mode_str = self.mode
        if self.tune:
            # Make sure the tuned and non-tuned modes are not mixed when creating output paths.
            mode_str = f"{self.mode}_tune"
        
        # Determine paradigm for check_skip_eval (must match log_all_subjects)
        if self.dataset == "Lee2019_SSVEP":
            paradigm_name = "SSVEP"
        elif self.dataset == "BI2015a":
            paradigm_name = "ERP"
        else:
            paradigm_name = "MotorImagery"

        if not self.overwrite:
            expected_noise_types = None
            expected_intensities_by_noise = None
            if self.mode in ("test_perturb", "multirun") or (isinstance(self.mode, str) and self.mode.startswith("test_perturb")):
                exp_types, exp_by_noise = _get_test_perturb_expected_scope(
                    self.dataset,
                    test_perturb_noise_types=self.test_perturb_noise_types,
                    test_perturb_gaussian_only=self.test_perturb_gaussian_only,
                    test_perturb_gaussian_alpha_grid=self.test_perturb_gaussian_alpha_grid,
                    test_perturb_num_steps=self.test_perturb_num_steps,
                    saturation_file=self._get_test_perturb_saturation_file(),
                )
                expected_noise_types = exp_types
                expected_intensities_by_noise = exp_by_noise
            if check_skip_eval(
                self.model, self.seed, self.subjects, mode_str, self.noise_type, self.intensity,
                eval_mode=self.eval_mode, paradigm=paradigm_name, dataset=self.dataset,
                paradigm_obj=self.paradigm, dataset_obj=self.dataset_obj, tuned=self.tune,
                expected_noise_types=expected_noise_types,
                expected_intensities_by_noise=expected_intensities_by_noise,
                test_perturb_num_steps=self.test_perturb_num_steps,
                test_perturb_saturation_file=self._get_test_perturb_saturation_file(),
            ):
                print(f"Skipping evaluation due to existing output files.")
                return None

        all_subject_results = []
        set_seeds(self.seed)
        if self.eval_mode == "CrossSubject":
            if self.legacy:
                print(f"[LEGACY] Legacy mode enabled: Using original CrossSubject experimental protocol (no subject chunking)")
            # Check if we're in fold-by-fold mode (memory optimization)
            if self.fold_idx is not None:
                # Fold-by-fold mode: Load only subjects needed for this fold
                if self.train_subjects is None or self.eval_subjects is None:
                    raise ValueError("For fold-by-fold execution, both --train_subjects and --eval_subjects must be provided")
                
                print(f"[MEMORY] Fold-by-fold mode: Processing fold {self.fold_idx}")
                print(f"[MEMORY] Training subjects: {self.train_subjects}")
                print(f"[MEMORY] Evaluation subjects: {self.eval_subjects}")
                
                # Load train and eval subjects separately to reduce peak memory
                # This avoids loading all 54 subjects at once (which requires 20GB+ for float64)
                gc.collect()
                log_memory_usage("Before loading fold data")
                
                # Load training subjects first
                print(f"[MEMORY] Loading training data for {len(self.train_subjects)} subjects (fold {self.fold_idx})...")
                X_train_full, y_train_full, metadata_train_full = self.paradigm.get_data(
                    self.dataset_obj, subjects=self.train_subjects
                )
                
                mem_mb = log_memory_usage("After loading training data")
                if isinstance(X_train_full, np.ndarray):
                    data_size_mb = X_train_full.nbytes / 1024 / 1024
                    print(f"[MEMORY] Training data shape: X={X_train_full.shape}, dtype={X_train_full.dtype}, size: {data_size_mb:.2f} MB")
                    
                    # CRITICAL: Convert to float32 immediately to reduce memory by 50%
                    if X_train_full.dtype == np.float64:
                        print(f"[MEMORY] Converting training data from float64 to float32...")
                        X_train_full = X_train_full.astype(np.float32)
                        new_size_mb = X_train_full.nbytes / 1024 / 1024
                        print(f"[MEMORY] After conversion: {new_size_mb:.2f} MB (saved {data_size_mb - new_size_mb:.2f} MB)")
                        gc.collect()
                
                y_train_encoded = LabelEncoder().fit_transform(y_train_full)
                del y_train_full
                gc.collect()
                
                # Load evaluation subjects
                print(f"[MEMORY] Loading evaluation data for {len(self.eval_subjects)} subjects (fold {self.fold_idx})...")
                X_valid_full, y_valid_full, metadata_valid_full = self.paradigm.get_data(
                    self.dataset_obj, subjects=self.eval_subjects
                )
                
                mem_mb = log_memory_usage("After loading evaluation data")
                if isinstance(X_valid_full, np.ndarray):
                    data_size_mb = X_valid_full.nbytes / 1024 / 1024
                    print(f"[MEMORY] Evaluation data shape: X={X_valid_full.shape}, dtype={X_valid_full.dtype}, size: {data_size_mb:.2f} MB")
                    
                    # CRITICAL: Convert to float32 immediately
                    if X_valid_full.dtype == np.float64:
                        print(f"[MEMORY] Converting evaluation data from float64 to float32...")
                        X_valid_full = X_valid_full.astype(np.float32)
                        new_size_mb = X_valid_full.nbytes / 1024 / 1024
                        print(f"[MEMORY] After conversion: {new_size_mb:.2f} MB (saved {data_size_mb - new_size_mb:.2f} MB)")
                        gc.collect()
                
                y_valid_encoded = LabelEncoder().fit_transform(y_valid_full)
                del y_valid_full
                gc.collect()
                
                # Create session identifier
                eval_subjects_str = ','.join(map(str, sorted(self.eval_subjects)))
                session = f"fold_{self.fold_idx}_eval_subjects_{eval_subjects_str}"
                
                # Check if this fold's results already exist
                if not self.overwrite and self._check_fold_result_exists(self.fold_idx, sorted(self.eval_subjects), session):
                    print(f"[CROSSSUBJECT] Skipping fold {self.fold_idx} - results already exist (eval_subjects: {self.eval_subjects})")
                    return None
                
                # Set current_subject to first eval subject
                self.current_subject = self.eval_subjects[0]
                
                # Check if we should use chunked training (memory optimization)
                # Use chunked training if:
                # 1. NOT in legacy mode (legacy mode uses original protocol)
                # 2. chunk_size is specified and > 0
                # 3. We have train_subjects and eval_subjects (fold-by-fold mode)
                # Note: Chunked training now supported for HPO as well
                use_chunked_training = (
                    not self.legacy and
                    self.subject_chunk_size is not None and 
                    self.subject_chunk_size > 0 and
                    self.train_subjects is not None and
                    self.eval_subjects is not None and
                    len(self.train_subjects) > 0 and
                    len(self.eval_subjects) > 0
                )
                
                if use_chunked_training:
                    # Use chunked training approach
                    print(f"[CHUNKED_TRAINING] Using chunked training with chunk_size={self.subject_chunk_size}")
                    print(f"[CHUNKED_TRAINING] Training subjects: {self.train_subjects}")
                    print(f"[CHUNKED_TRAINING] Evaluation subjects: {self.eval_subjects}")
                    
                    # Clean up loaded data since we'll load in chunks
                    del X_train_full, X_valid_full, y_train_encoded, y_valid_encoded, metadata_train_full, metadata_valid_full
                    gc.collect()
                    
                    # Process fold with chunked training
                    fold_results = self._evaluate_cv_fold_chunked(
                        self.train_subjects,
                        self.eval_subjects,
                        self.fold_idx,
                        session
                    )
                else:
                    # Use the loaded data directly (no need to split since we loaded separately)
                    X_train = X_train_full
                    y_train = y_train_encoded
                    X_valid = X_valid_full
                    y_valid = y_valid_encoded
                    metadata_train = metadata_train_full
                    
                    # Clean up
                    del X_train_full, X_valid_full, y_train_encoded, y_valid_encoded, metadata_train_full, metadata_valid_full
                    gc.collect()
                    
                    # Prepare CV metadata
                    cv_splitter, cv_metadata = self.prepare_data_cv()
                    
                    # Process single fold with full data
                    fold_results = self._evaluate_cv_fold(
                        X_train, y_train, X_valid, y_valid, 
                        self.fold_idx, cv_metadata, session, metadata_train
                    )
                
                # Add eval_subjects information to each result
                for result in fold_results:
                    result['eval_subjects'] = eval_subjects_str
                    result['train_subjects'] = ','.join(map(str, sorted(self.train_subjects)))
                
                # Save fold results immediately
                self._save_fold_results(fold_results, self.fold_idx, sorted(self.eval_subjects), session)
                
                # Convert to DataFrame for return (if needed)
                results_df = pd.DataFrame(fold_results)
                all_subject_results.append(results_df)
                
            else:
                # Legacy mode: Load all subjects and process all folds (for backward compatibility)
                # Memory management: Force garbage collection before loading large dataset
                gc.collect()
                
                # Log memory usage before loading
                log_memory_usage("Before loading data")
                
                print(f"[MEMORY] Loading data for {len(self.subjects)} subjects in CrossSubject mode (legacy mode)...")
                print(f"[MEMORY] WARNING: This loads all subjects at once. For large datasets, use fold-by-fold mode.")
                X, y, metadata = self.paradigm.get_data(self.dataset_obj, subjects=self.subjects)
                
                # Log memory usage after loading
                mem_mb = log_memory_usage("After loading data")
                if isinstance(X, np.ndarray):
                    data_size_mb = X.nbytes / 1024 / 1024
                    print(f"[MEMORY] Data shape: X={X.shape}, dtype={X.dtype}, estimated size: {data_size_mb:.2f} MB")
                    
                    # Check against memory limit if set
                    max_memory_mb = None
                    max_memory_gb = os.environ.get('PYTHON_MAX_MEMORY_GB')
                    if max_memory_gb:
                        try:
                            max_memory_mb = float(max_memory_gb) * 1024
                        except ValueError:
                            pass
                    
                    # Warn if data size is very large
                    if data_size_mb > 10000:  # > 10GB
                        print(f"[WARNING] Large dataset loaded: {data_size_mb:.2f} MB. This may cause memory issues.")
                        print(f"[WARNING] Consider using fold-by-fold mode (--fold_idx, --train_subjects, --eval_subjects) if OOM errors occur.")
                    
                    # Warn if approaching memory limit
                    if max_memory_mb and mem_mb and mem_mb > max_memory_mb * 0.8:
                        print(f"[WARNING] Memory usage ({mem_mb:.2f} MB) is above 80% of limit ({max_memory_mb:.2f} MB)")
                        print(f"[WARNING] Risk of out-of-memory errors. Consider using fold-by-fold mode.")
                
                # Force garbage collection after loading
                gc.collect()
                
                y_encoded = LabelEncoder().fit_transform(y)
                
                # Memory optimization: Delete original y array after encoding (we only need y_encoded)
                del y
                gc.collect()
                
                # Prepare cross-validation
                cv_splitter, cv_metadata = self.prepare_data_cv()
                
                # Run cross-validation with subject groups
                groups = metadata['subject'].values
                
                # Memory optimization: Don't store all folds in memory - iterate directly
                # This avoids keeping all fold indices in memory at once
                all_results = []
                fold_idx = 0
                for train_idx, valid_idx in cv_splitter.split(X, y_encoded, groups=groups):
                    # Get the subjects that are in the evaluation set for this fold
                    # Memory optimization: Use .iloc with list conversion to avoid keeping view in memory
                    eval_subjects = np.unique(metadata.iloc[valid_idx]['subject'].values)
                    eval_subjects_list = sorted([int(s) for s in eval_subjects])
                    eval_subjects_str = ','.join(map(str, eval_subjects_list))
                    session = f"fold_{fold_idx}_eval_subjects_{eval_subjects_str}"
                    
                    # Check if this fold's results already exist (before any expensive processing)
                    # This allows resuming interrupted runs and skipping already-completed folds
                    if not self.overwrite and self._check_fold_result_exists(fold_idx, eval_subjects_list, session):
                        print(f"[CROSSSUBJECT] Skipping fold {fold_idx} - results already exist (eval_subjects: {eval_subjects_list})")
                        fold_idx += 1
                        continue
                    
                    print(f"[CROSSSUBJECT] Processing fold {fold_idx} (eval_subjects: {eval_subjects_list})")
                    
                    # Set current_subject to a representative value (first eval subject)
                    self.current_subject = eval_subjects[0]
                    
                    # Memory optimization: Array indexing with arrays creates copies automatically
                    # Note: X[train_idx] creates a copy (necessary for training), no need for explicit .copy()
                    X_train = X[train_idx]
                    y_train = y_encoded[train_idx]
                    X_valid = X[valid_idx]
                    y_valid = y_encoded[valid_idx]
                    # Memory optimization: .iloc creates a view, but we'll delete it promptly after use
                    metadata_train = metadata.iloc[train_idx]
                    
                    # Memory optimization: Delete indices immediately after use
                    del train_idx, valid_idx
                    
                    fold_results = self._evaluate_cv_fold(X_train, y_train, X_valid, y_valid, fold_idx, cv_metadata, session, metadata_train)
                    
                    # Add eval_subjects information to each result
                    for result in fold_results:
                        result['eval_subjects'] = eval_subjects_str
                        result['n_eval_subjects'] = len(eval_subjects)
                    
                    # Save fold results immediately after processing (before continuing to next fold)
                    # This ensures results are persisted even if the run is interrupted
                    self._save_fold_results(fold_results, fold_idx, eval_subjects_list, session)
                    
                    # Keep results in memory for final aggregation (if needed)
                    all_results.extend(fold_results)
                    
                    # Memory management: Clear intermediate arrays after each fold
                    del X_train, y_train, X_valid, y_valid, metadata_train, eval_subjects
                    gc.collect()
                    
                    fold_idx += 1
            
            # Memory management: Clear large arrays before creating DataFrame
            # Delete groups array (no longer needed after folds are processed)
            if "groups" in locals():
                del groups
                gc.collect()
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Memory management: Clear large arrays after processing all folds
            # Note: y was already deleted after encoding
            del X, y_encoded, metadata
            gc.collect()
            
            # Log memory usage after cleanup
            log_memory_usage("After processing all folds and cleanup")
            
            # Aggregate fold results according to eval_mode and mode
            results_df = self._aggregate_fold_results(results_df)
            
            # Add experiment metadata
            results_df['model'] = self.model
            results_df['dataset'] = self.dataset
            results_df['mode'] = self.mode
            results_df['eval_mode'] = self.eval_mode
            results_df['seed'] = self.seed
            # For test_perturb mode, we don't want to override the noise_type and intensity values
            if not is_test_perturb_mode(self.mode):
                if self.noise_dict:
                    results_df['intensity'] = self.noise_dict['intensity']
                    results_df['noise_type'] = self.noise_dict['noise_type']
            results_df['tune'] = self.tune

            n_chans, n_times = self._determine_data_dimensions()
            # Use try_cache=False since we only need model structure for parameter extraction, not a trained model
            model_instance = self._create_model(n_chans, n_times, try_cache=False)
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
                if not is_test_perturb_mode(self.mode):
                    if self.noise_dict:
                        results_df['intensity'] = self.noise_dict['intensity']
                        results_df['noise_type'] = self.noise_dict['noise_type']
                results_df['tune'] = self.tune

                n_chans, n_times = self._determine_data_dimensions()
                # Use try_cache=False since we only need model structure for parameter extraction, not a trained model
                model_instance = self._create_model(n_chans, n_times, try_cache=False)
                row_headers = get_all_model_params(self.model)
                config = model_instance.get_params()
                for k, v in config.items():
                    if k in row_headers and k not in results_df.columns:
                        results_df[k] = v

                all_subject_results.append(results_df)            
                # Clean up HDF5 path
                if os.path.isdir(self.hdf5_path):
                    shutil.rmtree(self.hdf5_path)
        
        # Handle case where all folds were skipped (empty results)
        if len(all_subject_results) == 0:
            print(f"[CROSSSUBJECT] No results to return - all folds were skipped or no data processed")
            return pd.DataFrame()
        
        all_results_df = pd.concat(all_subject_results)
        # Save results
        # For CrossSubject mode, we already saved results per fold, so skip final save to avoid duplicates
        if self.eval_mode != "CrossSubject":
            self._save_results(all_results_df)
        else:
            print(f"[CROSSSUBJECT] Skipping final save - results already saved per fold")
        return all_results_df
        
    def _check_fold_result_exists(self, fold_idx: int, eval_subjects: List[int], session: str) -> bool:
        """
        Check if results for a specific CrossSubject fold already exist.
        
        Args:
            fold_idx: Fold index (0, 1, or 2)
            eval_subjects: List of evaluation subjects for this fold
            session: Session string (e.g., "fold_0_eval_subjects_1,2,3")
            
        Returns:
            True if results exist, False otherwise
        """
        
        # Determine paradigm
        if self.dataset == "Lee2019_SSVEP":
            paradigm_name = "SSVEP"
        elif self.dataset == "BI2015a":
            paradigm_name = "ERP"
        else:
            paradigm_name = "MotorImagery"
        
        # Determine mode string
        # If tune flag is set, append "_tune" to mode (e.g., "test_perturb" -> "test_perturb_tune")
        mode_str = self.mode
        if self.tune:
            mode_str = f"{self.mode}_tune"
        
        # Use first eval subject as representative for path
        representative_subject = eval_subjects[0] if eval_subjects else self.subjects[0]
        
        # Get output directory (short path for new runs; also check long path for existing runs)
        eval_mode_str = "CrossSubjectEvaluation"
        out_dir_short = create_output_path(
            self.model, self.seed, int(representative_subject), session,
            mode_str, session_type=eval_mode_str, paradigm=paradigm_name, dataset=self.dataset, use_short_run_id=True
        )
        out_dir_long = create_output_path(
            self.model, self.seed, int(representative_subject), session,
            mode_str, session_type=eval_mode_str, paradigm=paradigm_name, dataset=self.dataset, use_short_run_id=False
        )
        is_test_perturb_mode = self.mode in ['test_perturb', 'multirun'] or self.mode.startswith('test_perturb')

        if is_test_perturb_mode:
            for out_dir in (out_dir_short, out_dir_long):
                if os.path.exists(out_dir):
                    csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
                    if csv_files:
                        return True
        else:
            if self.noise_dict:
                filename_suffix = f"_{self.noise_dict['noise_type']}_{self.noise_dict['intensity']}"
            else:
                filename_suffix = ""
            short_session = get_short_session_id(session, 'CrossSubject')
            fname = f"{self.model}_{mode_str}{filename_suffix}_{short_session}_seed{self.seed}.csv"
            for out_dir in (out_dir_short, out_dir_long):
                if os.path.exists(os.path.join(out_dir, fname)):
                    return True
        return False
    
    def _save_fold_results(self, fold_results: List[Dict[str, Any]], fold_idx: int, eval_subjects: List[int], session: str):
        """
        Save results for a single CrossSubject fold immediately after processing.
        
        Args:
            fold_results: List of result dictionaries for this fold
            fold_idx: Fold index
            eval_subjects: List of evaluation subjects for this fold
            session: Session string
        """
        # Convert to DataFrame
        fold_df = pd.DataFrame(fold_results)
        
        # Aggregate fold results (may not do much for CrossSubject, but keeps consistency)
        fold_df = self._aggregate_fold_results(fold_df)
        
        # Add experiment metadata
        fold_df['model'] = self.model
        fold_df['dataset'] = self.dataset
        fold_df['mode'] = self.mode
        fold_df['eval_mode'] = self.eval_mode
        fold_df['seed'] = self.seed
        if not is_test_perturb_mode(self.mode):
            if self.noise_dict:
                fold_df['intensity'] = self.noise_dict['intensity']
                fold_df['noise_type'] = self.noise_dict['noise_type']
        fold_df['tune'] = self.tune
        
        # Add model parameters
        n_chans, n_times = self._determine_data_dimensions()
        model_instance = self._create_model(n_chans, n_times, try_cache=False)
        row_headers = get_all_model_params(self.model)
        config = model_instance.get_params()
        for k, v in config.items():
            if k in row_headers and k not in fold_df.columns:
                fold_df[k] = v
        
        # Determine expected mode string for logging (must match _save_results logic)
        # If tune flag is set, append "_tune" to mode (e.g., "test_perturb" -> "test_perturb_tune")
        expected_mode_str = self.mode
        if self.tune:
            expected_mode_str = f"{self.mode}_tune"
        
        print(f"[CROSSSUBJECT] Saving fold {fold_idx} results:")
        print(f"  Mode: {self.mode}, Tune: {self.tune}, Expected mode_str: {expected_mode_str}")
        print(f"  Eval subjects: {eval_subjects}")
        print(f"  Session: {session}")
        print(f"  DataFrame shape: {fold_df.shape}")
        
        # Save results using existing method
        try:
            self._save_results(fold_df)
            print(f"[CROSSSUBJECT] OK Successfully saved results for fold {fold_idx} (eval_subjects: {eval_subjects})")
        except Exception as e:
            print(f"[CROSSSUBJECT] ERROR saving results for fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _aggregate_fold_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate fold results according to eval_mode and mode as specified in the spec.
        
        Returns:
            Aggregated results DataFrame
        """
        if self.eval_mode == "WithinSession":
            # WithinSession: calculate fold score means for sessions separately
            if 'fold_idx' in results_df.columns:
                if is_test_perturb_mode(self.mode):
                    # test_perturb mode: calculate fold score means for both clean folds and corrupted validation data
                    agg_results = []
                    
                    # Handle corrupted scores at different intensities
                    # CRITICAL FIX: We must iterate over noise_type first to properly aggregate each noise type separately
                    # The bug was that we were only iterating over intensity and session, which caused results from
                    # different noise types to be mixed together, with only the first noise_type being saved.
                    if 'intensity' in results_df.columns and 'noise_type' in results_df.columns:
                        # Get unique combinations of noise_type and intensity
                        for noise_type in results_df['noise_type'].unique():
                            if pd.notna(noise_type):
                                noise_df = results_df[results_df['noise_type'] == noise_type]
                                for intensity in noise_df['intensity'].unique():
                                    if pd.notna(intensity) and intensity != 'clean':
                                        intensity_df = noise_df[noise_df['intensity'] == intensity]
                                        if len(intensity_df) > 0:
                                            for session in intensity_df['session'].unique():
                                                session_df = intensity_df[intensity_df['session'] == session]
                                                if len(session_df) > 0:
                                                    # DIAGNOSTIC: Check if fold_idx exists and how many unique values
                                                    if 'fold_idx' in session_df.columns:
                                                        unique_folds = session_df['fold_idx'].dropna().unique()
                                                        if len(unique_folds) > 1:
                                                            print(f"[DIAGNOSTIC] Aggregating {len(unique_folds)} folds for session={session}, "
                                                                  f"noise_type={noise_type}, intensity={intensity}, subject={session_df['subject'].iloc[0]}")
                                                            print(f"  Unique fold_idx values: {sorted(unique_folds)}")
                                                            print(f"  Clean scores before aggregation: {sorted(session_df['clean_score'].dropna().unique())}")
                                                    elif len(session_df) > 1:
                                                        print(f"[WARNING] Multiple rows for session={session}, noise_type={noise_type}, "
                                                              f"intensity={intensity} but no fold_idx column. Rows: {len(session_df)}")
                                                    
                                                    agg_row = {
                                                        'subject': session_df['subject'].iloc[0],
                                                        'session': session,
                                                        'score': session_df['corrupted_score'].mean() if 'corrupted_score' in session_df.columns else 0.0,
                                                        'model': self.model,
                                                        'mode': self.mode,
                                                        'eval_mode': self.eval_mode,
                                                        'seed': self.seed,
                                                        'tune': self.tune,
                                                        'noise_type': noise_type,
                                                        'perturbation_type': session_df['perturbation_type'].iloc[0] if 'perturbation_type' in session_df.columns else noise_type,
                                                        'intensity': intensity,
                                                        'clean_score': session_df['clean_score'].mean() if 'clean_score' in session_df.columns else 0.0,
                                                        'corrupted_score': session_df['corrupted_score'].mean() if 'corrupted_score' in session_df.columns else 0.0,
                                                        'relative_drop': session_df['relative_drop'].mean() if 'relative_drop' in session_df.columns else 0.0,
                                                        'training_time': session_df['training_time'].mean() if 'training_time' in session_df.columns else 0.0,
                                                        'evaluation_time': session_df['evaluation_time'].mean() if 'evaluation_time' in session_df.columns else 0.0,
                                                        'total_time': session_df['total_time'].mean() if 'total_time' in session_df.columns else 0.0
                                                    }
                                                    if 'params' in session_df.columns:
                                                        agg_row['params'] = session_df['params'].iloc[0]
                                                    if 'target_snr_db' in session_df.columns and session_df['target_snr_db'].notna().any():
                                                        agg_row['target_snr_db'] = session_df['target_snr_db'].iloc[0]
                                                    if 'empirical_snr_db' in session_df.columns and session_df['empirical_snr_db'].notna().any():
                                                        agg_row['empirical_snr_db'] = session_df['empirical_snr_db'].iloc[0]
                                                    if 'alpha_max' in session_df.columns and session_df['alpha_max'].notna().any():
                                                        agg_row['alpha_max'] = session_df['alpha_max'].iloc[0]
                                                    
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
                    elif 'intensity' in results_df.columns:
                        # Fallback for backward compatibility (if noise_type column is missing)
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
                            # DIAGNOSTIC: Check if fold_idx exists and how many unique values
                            if 'fold_idx' in session_df.columns:
                                unique_folds = session_df['fold_idx'].dropna().unique()
                                if len(unique_folds) > 1:
                                    print(f"[DIAGNOSTIC] Aggregating {len(unique_folds)} folds for session={session}, "
                                          f"subject={session_df['subject'].iloc[0]}, mode={self.mode}")
                                    print(f"  Unique fold_idx values: {sorted(unique_folds)}")
                            elif len(session_df) > 1:
                                print(f"[WARNING] Multiple rows for session={session} but no fold_idx column. Rows: {len(session_df)}")
                            
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
        if is_test_perturb_mode(self.mode):
            intensity_param = 10.0
        else:
            intensity_param = self.noise_dict['intensity'] if self.noise_dict else None
            
        # If tune flag is set, append "_tune" to mode (e.g., "test_perturb" -> "test_perturb_tune")
        mode_str = self.mode
        if self.tune:
            # Make sure the tuned and non-tuned modes are not mixed when creating output paths.
            mode_str = f"{self.mode}_tune"
        
        # Log the mode string construction for debugging
        if self.eval_mode == "CrossSubject":
            print(f"[SAVE_RESULTS] Mode string construction:")
            print(f"  self.mode: {self.mode}")
            print(f"  self.tune: {self.tune}")
            print(f"  Final mode_str: {mode_str}")
            print(f"  This ensures tuned and non-tuned results use different paths")

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

        # Spec 3 PATCH 4: Write test_perturb run manifest (git, dataset, model, perturbation, SNR, seeds)
        if is_test_perturb_mode(self.mode) and not results_df.empty:
            try:
                git_commit = "unknown"
                try:
                    import subprocess as sp
                    r = sp.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    )
                    if r.returncode == 0 and r.stdout:
                        git_commit = r.stdout.strip()
                except Exception:
                    pass
                first_subject = int(results_df["subject"].iloc[0]) if "subject" in results_df.columns else (self.subjects[0] if self.subjects else 0)
                first_session = str(results_df["session"].iloc[0]) if "session" in results_df.columns else "0train"
                out_dir = create_output_path(
                    self.model, self.seed, first_subject, first_session,
                    mode_str, session_type=eval_mode_str, paradigm=paradigm_name, dataset=self.dataset
                )
                os.makedirs(out_dir, exist_ok=True)
                snr_by_type = {}
                if "noise_type" in results_df.columns:
                    for nt in results_df["noise_type"].dropna().unique():
                        sub = results_df[results_df["noise_type"] == nt]
                        row = sub.iloc[0]
                        snr_by_type[str(nt)] = {
                            "target_snr_db": float(row.get("target_snr_db", float("nan"))) if "target_snr_db" in row else float("nan"),
                            "empirical_snr_db": float(row.get("empirical_snr_db", float("nan"))) if "empirical_snr_db" in row else float("nan"),
                            "alpha_max": float(row.get("alpha_max", float("nan"))) if "alpha_max" in row else float("nan"),
                        }
                perturbation_params = {
                    "ar1_drift": {"rho": getattr(self, "test_perturb_ar1_rho", 0.97)},
                    "spatial_gaussian": {"ell_multiplier": getattr(self, "test_perturb_spatial_ell_multiplier", 1.0)},
                    "emg_band": {
                        "f_low": getattr(self, "test_perturb_emg_f_low", 20.0),
                        "f_high": getattr(self, "test_perturb_emg_f_high", 80.0),
                        "envelope_on": getattr(self, "test_perturb_emg_use_envelope", False),
                    },
                }
                manifest = {
                    "git_commit": git_commit,
                    "dataset": self.dataset,
                    "eval_mode": self.eval_mode,
                    "model": self.model,
                    "seed": int(self.seed),
                    "perturbation_types": list(self.test_perturb_noise_types) if self.test_perturb_noise_types else [],
                    "target_snr_db": float(self.test_perturb_target_snr_db),
                    "snr_by_noise_type": snr_by_type,
                    "perturbation_params": perturbation_params,
                    "rng_seed": int(self.seed),
                }
                manifest_path = os.path.join(out_dir, "test_perturb_manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                print(f"[WARNING] Could not write test_perturb_manifest.json: {e}")


def main():
    """Main entry point for the unified experiment runner."""
    # Load any custom model registrations from .model_registry directory
    # This allows test scripts to register custom model variants
    import importlib.util
    from pathlib import Path
    reg_dir = Path(__file__).parent.parent / ".model_registry"
    if reg_dir.exists():
        for reg_file in sorted(reg_dir.glob("*.py")):
            if reg_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(reg_file.stem, reg_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                # Print error for debugging
                print(f"Warning: Failed to load registration file {reg_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    parser = argparse.ArgumentParser(description="Unified EEG Experiment Runner")
    # Get model registry dynamically to support runtime-registered variants
    # Note: We validate at runtime in UnifiedExperimentRunner.__init__ instead of here
    # to allow for custom model variants registered after import
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001", choices=["BNCI2014_001", "Lee2019_MI", "Lee2019_SSVEP", "BI2015a"])
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["test_perturb", "multirun", "aggregate_only"])
                    #    choices=["baseline", "tune", "augment", "perturb", "augment_notune", "perturb_notune", "test_perturb", "multirun", "aggregate_only"])
    parser.add_argument("--eval_mode", type=str, required=True, 
                       choices=["WithinSession", "CrossSession", "CrossSubject"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_type", type=str, choices=["dropout", "gaussian", "eog", "spike"], default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument(
        "--nas_pilot_dir",
        type=str,
        default=None,
        help="If set, register NAS pilot models from <dir>/selected_architectures/*.json before running.",
    )
    # test_perturb configuration (optional; defaults preserve existing behavior)
    parser.add_argument(
        "--noise_perturbation_saturation_file",
        type=str,
        default=None,
        help=(
            "Optional CSV with saturation_point per (dataset, noise_type). "
            "If provided, test_perturb will (a) use it to set sigma_max/bounds and "
            "(b) restrict evaluated noise types to those present in the file for the selected dataset."
        ),
    )
    parser.add_argument(
        "--noise_perturbation_num_steps",
        type=int,
        default=20,
        help="Number of intensity steps for test_perturb when not using an alpha grid.",
    )
    parser.add_argument(
        "--test_perturb_noise_types",
        type=str,
        default=None,
        help="Comma-separated list of noise types to evaluate in test_perturb (e.g., 'gaussian,dropout').",
    )
    parser.add_argument(
        "--test_perturb_gaussian_only",
        action="store_true",
        help="If set, restrict test_perturb to Gaussian noise only.",
    )
    parser.add_argument(
        "--test_perturb_gaussian_alpha_grid",
        type=str,
        default=None,
        help="Comma-separated alpha grid in [0,1] for Gaussian: intensity = alpha * sigma_max (e.g., '0,0.25,0.5,0.75,1').",
    )
    parser.add_argument(
        "--test_perturb_target_snr_db",
        type=str,
        default="0.0",
        help="Target SNR in dB at alpha_max for correlated perturbations. Comma-separated for dual-SNR (e.g. '-12,-6').",
    )
    parser.add_argument(
        "--test_perturb_spatial_ell_multiplier",
        type=float,
        default=1.0,
        help="Multiplier for spatial correlation length ell (spatial_gaussian). Default 1.0; use 2.0 for Step 2 escalation.",
    )
    parser.add_argument(
        "--test_perturb_emg_f_high",
        type=float,
        default=80.0,
        help="High cutoff in Hz for EMG band-limited noise. Default 80; use 100 for Step 2 escalation.",
    )
    parser.add_argument(
        "--test_perturb_emg_use_envelope",
        action="store_true",
        help="Apply slow amplitude envelope to EMG noise (bursty EMG).",
    )
    parser.add_argument(
        "--test_perturb_ar1_rho",
        type=float,
        default=0.97,
        help="AR(1) drift coefficient for ar1_drift perturbation (default 0.97).",
    )
    parser.add_argument(
        "--test_perturb_emg_f_low",
        type=float,
        default=20.0,
        help="EMG band low cutoff in Hz (default 20).",
    )
    parser.add_argument(
        "--plot2_diagnostics_dir",
        type=str,
        default=None,
        help="Plot 2 PATCH 0.2: If set, write perturbation_fingerprint.json here (e.g. <plot2_dir>/diagnostics).",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    
    # Fold-by-fold execution for CrossSubject mode (memory optimization)
    parser.add_argument("--fold_idx", type=int, default=None,
                        help="Specific fold to process (for CrossSubject mode, 0-2). If not provided, processes all folds.")
    parser.add_argument("--train_subjects", type=int, nargs="+", default=None,
                        help="Training subjects for this fold (for CrossSubject fold-by-fold mode)")
    parser.add_argument("--eval_subjects", type=int, nargs="+", default=None,
                        help="Evaluation subjects for this fold (for CrossSubject fold-by-fold mode)")
    parser.add_argument("--subject_chunk_size", type=int, default=None,
                        help="Number of subjects to load per chunk for memory-efficient training (CrossSubject mode). If None, loads all subjects at once.")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy experimental protocol (disables subject chunking and other memory optimizations to match original behavior)")
    parser.add_argument(
        "--disable_underfitting_retrain",
        action="store_true",
        help="Disable the underfitting-triggered retraining pass (keeps training protocol fixed).",
    )
    parser.add_argument(
        "--hail_mary_stability",
        action="store_true",
        help="Hail Mary Ch.5: log batch_train_loss_var per epoch via CfC callback (sets HAIL_MARY_STABILITY).",
    )

    # Memory management: Check for environment variable to set memory limit
    max_memory_gb = os.environ.get('PYTHON_MAX_MEMORY_GB')
    if max_memory_gb:
        try:
            max_memory_mb = float(max_memory_gb) * 1024
            print(f"[MEMORY] Memory limit set via environment variable: {max_memory_gb} GB ({max_memory_mb:.0f} MB)")
        except ValueError:
            print(f"[WARNING] Invalid PYTHON_MAX_MEMORY_GB value: {max_memory_gb}, ignoring")
            max_memory_mb = None
    else:
        max_memory_mb = None
    
    args = parser.parse_args()

    if getattr(args, "hail_mary_stability", False):
        os.environ["HAIL_MARY_STABILITY"] = "1"

    # Register NAS pilot models (scalable alternative to per-run .model_registry python files)
    if args.nas_pilot_dir:
        from architecture_refinement.nas_pilot_registry import register_nas_pilot_models
        registered = register_nas_pilot_models(args.nas_pilot_dir)
        print(f"[NAS PILOT] Registered {len(registered)} models from: {args.nas_pilot_dir}")

    # Parse optional test_perturb controls
    if args.test_perturb_noise_types:
        args.test_perturb_noise_types = [
            s.strip() for s in str(args.test_perturb_noise_types).split(",") if s.strip()
        ]
    else:
        args.test_perturb_noise_types = None

    if args.test_perturb_gaussian_alpha_grid:
        try:
            args.test_perturb_gaussian_alpha_grid = [
                float(s.strip())
                for s in str(args.test_perturb_gaussian_alpha_grid).split(",")
                if s.strip()
            ]
        except Exception as e:
            raise ValueError(f"Invalid --test_perturb_gaussian_alpha_grid: {e}")
    else:
        args.test_perturb_gaussian_alpha_grid = None

    # Parse test_perturb_target_snr_db: comma-separated for dual-SNR (Plot 2 Overhaul)
    _snr_str = str(getattr(args, "test_perturb_target_snr_db", "0.0"))
    _snr_parts = [s.strip() for s in _snr_str.split(",") if s.strip()]
    if len(_snr_parts) > 1:
        args.test_perturb_target_snr_dbs = [float(x) for x in _snr_parts]
        args.test_perturb_target_snr_db = float(_snr_parts[0])
    else:
        args.test_perturb_target_snr_db = float(_snr_parts[0]) if _snr_parts else 0.0
        args.test_perturb_target_snr_dbs = None

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
        
        # Create temporary paradigm and dataset objects for check_skip_eval
        temp_paradigm = None
        temp_dataset_obj = None
        try:
            if args.dataset == "BNCI2014_001":
                temp_dataset_obj = BNCI2014_001()
                temp_dataset_obj.subject_list = args.subjects
            elif args.dataset == "Lee2019_MI":
                temp_dataset_obj = Lee2019_MI()
                temp_dataset_obj.subject_list = args.subjects
            elif args.dataset == "Lee2019_SSVEP":
                temp_dataset_obj = Lee2019_SSVEP()
                temp_dataset_obj.subject_list = args.subjects
            elif args.dataset == "BI2015a":
                temp_dataset_obj = BI2015a()
                temp_dataset_obj.subject_list = args.subjects
            temp_paradigm = get_paradigm(resample=None, dataset=args.dataset)
        except Exception as e:
            print(f"Warning: Could not create temporary dataset/paradigm for session detection: {e}")
        
        for mode in ["test_perturb"]:
            if not args.overwrite:
                mode_str = mode
                if args.tune:
                    mode_str = f"{mode_str}_tune"
                exp_types, exp_by_noise = _get_test_perturb_expected_scope(
                    args.dataset,
                    test_perturb_noise_types=args.test_perturb_noise_types,
                    test_perturb_gaussian_only=args.test_perturb_gaussian_only,
                    test_perturb_gaussian_alpha_grid=args.test_perturb_gaussian_alpha_grid,
                    test_perturb_num_steps=getattr(args, "noise_perturbation_num_steps", 20),
                    saturation_file=getattr(args, "noise_perturbation_saturation_file", None),
                )
                if check_skip_eval(
                    model, seed, args.subjects, mode_str, args.noise_type, args.intensity,
                    eval_mode=eval_mode, paradigm=paradigm_name, dataset=args.dataset,
                    paradigm_obj=temp_paradigm, dataset_obj=temp_dataset_obj, tuned=args.tune,
                    expected_noise_types=exp_types,
                    expected_intensities_by_noise=exp_by_noise,
                    test_perturb_num_steps=getattr(args, "noise_perturbation_num_steps", 20),
                    test_perturb_saturation_file=getattr(args, "noise_perturbation_saturation_file", None),
                ):
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
                    overwrite=args.overwrite,
                    fold_idx=args.fold_idx,
                    train_subjects=args.train_subjects,
                    eval_subjects=args.eval_subjects,
                    subject_chunk_size=args.subject_chunk_size,
                    legacy=args.legacy,
                    disable_underfitting_retrain=args.disable_underfitting_retrain,
                    test_perturb_noise_types=args.test_perturb_noise_types,
                    test_perturb_num_steps=args.noise_perturbation_num_steps,
                    test_perturb_saturation_file=args.noise_perturbation_saturation_file,
                    test_perturb_gaussian_only=args.test_perturb_gaussian_only,
                    test_perturb_gaussian_alpha_grid=args.test_perturb_gaussian_alpha_grid,
                    test_perturb_target_snr_db=getattr(args, "test_perturb_target_snr_db", 0.0),
                test_perturb_target_snr_dbs=getattr(args, "test_perturb_target_snr_dbs", None),
                    test_perturb_spatial_ell_multiplier=getattr(args, "test_perturb_spatial_ell_multiplier", 1.0),
                    test_perturb_emg_f_high=getattr(args, "test_perturb_emg_f_high", 80.0),
                    test_perturb_emg_use_envelope=getattr(args, "test_perturb_emg_use_envelope", False),
                    test_perturb_ar1_rho=getattr(args, "test_perturb_ar1_rho", 0.97),
                    test_perturb_emg_f_low=getattr(args, "test_perturb_emg_f_low", 20.0),
                    plot2_diagnostics_dir=getattr(args, "plot2_diagnostics_dir", None),
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
        # Create temporary paradigm and dataset objects for check_skip_eval
        temp_paradigm = None
        temp_dataset_obj = None
        try:
            if args.dataset == "BNCI2014_001":
                temp_dataset_obj = BNCI2014_001()
                temp_dataset_obj.subject_list = args.subjects
            elif args.dataset == "Lee2019_MI":
                temp_dataset_obj = Lee2019_MI()
                temp_dataset_obj.subject_list = args.subjects
            elif args.dataset == "Lee2019_SSVEP":
                temp_dataset_obj = Lee2019_SSVEP()
                temp_dataset_obj.subject_list = args.subjects
            elif args.dataset == "BI2015a":
                temp_dataset_obj = BI2015a()
                temp_dataset_obj.subject_list = args.subjects
            temp_paradigm = get_paradigm(resample=None, dataset=args.dataset)
        except Exception as e:
            print(f"Warning: Could not create temporary dataset/paradigm for session detection: {e}")
        
        if not args.overwrite:
            mode_str = args.mode
            if args.tune:
                mode_str = f"{args.mode}_tune"
            expected_noise_types = None
            expected_intensities_by_noise = None
            if args.mode in ("test_perturb", "multirun") or (isinstance(args.mode, str) and args.mode.startswith("test_perturb")):
                exp_types, exp_by_noise = _get_test_perturb_expected_scope(
                    args.dataset,
                    test_perturb_noise_types=args.test_perturb_noise_types,
                    test_perturb_gaussian_only=args.test_perturb_gaussian_only,
                    test_perturb_gaussian_alpha_grid=args.test_perturb_gaussian_alpha_grid,
                    test_perturb_num_steps=args.noise_perturbation_num_steps,
                    saturation_file=args.noise_perturbation_saturation_file,
                )
                expected_noise_types = exp_types
                expected_intensities_by_noise = exp_by_noise
            if check_skip_eval(
                args.model, args.seed, args.subjects, mode_str, args.noise_type, args.intensity,
                args.eval_mode, paradigm_name, args.dataset, paradigm_obj=temp_paradigm, dataset_obj=temp_dataset_obj, tuned=args.tune,
                expected_noise_types=expected_noise_types,
                expected_intensities_by_noise=expected_intensities_by_noise,
                test_perturb_num_steps=args.noise_perturbation_num_steps,
                test_perturb_saturation_file=args.noise_perturbation_saturation_file,
            ):
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
                overwrite=args.overwrite,
                fold_idx=args.fold_idx,
                train_subjects=args.train_subjects,
                eval_subjects=args.eval_subjects,
                subject_chunk_size=args.subject_chunk_size,
                legacy=args.legacy,
                disable_underfitting_retrain=args.disable_underfitting_retrain,
                test_perturb_noise_types=args.test_perturb_noise_types,
                test_perturb_num_steps=args.noise_perturbation_num_steps,
                test_perturb_saturation_file=args.noise_perturbation_saturation_file,
                test_perturb_gaussian_only=args.test_perturb_gaussian_only,
                test_perturb_gaussian_alpha_grid=args.test_perturb_gaussian_alpha_grid,
                test_perturb_target_snr_db=getattr(args, "test_perturb_target_snr_db", 0.0),
                test_perturb_target_snr_dbs=getattr(args, "test_perturb_target_snr_dbs", None),
                test_perturb_spatial_ell_multiplier=getattr(args, "test_perturb_spatial_ell_multiplier", 1.0),
                test_perturb_emg_f_high=getattr(args, "test_perturb_emg_f_high", 80.0),
                test_perturb_emg_use_envelope=getattr(args, "test_perturb_emg_use_envelope", False),
                test_perturb_ar1_rho=getattr(args, "test_perturb_ar1_rho", 0.97),
                test_perturb_emg_f_low=getattr(args, "test_perturb_emg_f_low", 20.0),
                plot2_diagnostics_dir=getattr(args, "plot2_diagnostics_dir", None),
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
