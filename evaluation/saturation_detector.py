#!/usr/bin/env python3
"""
Adaptive Saturation Point Detection for Noise Benchmarking

This module implements a statistically rigorous approach to determine saturation points
for different noise types across multiple EEG datasets. It uses the conservative 
statistical significance thresholds from Combrisson and Jerbi (2015) to define 
chance-level performance.

Based on Combrisson and Jerbi (2015) - Journal of Neuroscience Methods:
- For binary classification (2-class):
  * 100 samples: 58% threshold
  * 200 samples: 56% threshold  
  * 500 samples: 53.6% threshold
- For 4-class classification:
  * 100 samples: 32.0% threshold
  * 200 samples: 30.0% threshold
  * 500 samples: 28.2% threshold

Since intracranial EEG is generally more accurate than scalp EEG, these conservative
thresholds provide a robust baseline for chance-level performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from moabb.datasets import BNCI2014_001, Lee2019_SSVEP, BI2015a
from moabb.paradigms import MotorImagery, SSVEP, P300
from models.eegnet import create_eegnet_classifier
from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor
from evaluation.unified_experiment_runner import UnifiedExperimentRunner
from config import get_paradigm, MODEL_REGISTRY
from globals import set_seeds, get_seed, UNDERFITTING_THRESHOLD, get_max_epochs_for_dataset, get_underfitting_threshold_for_dataset
from sklearn.metrics import get_scorer
from evaluation.metrics import compute_classification_metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

@dataclass
class SaturationResult:
    """Container for saturation point detection results"""
    noise_type: str
    dataset: str
    saturation_point: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    chance_threshold: float
    validation_trials: int
    performance_at_saturation: float
    performance_std: float
    is_statistically_significant: bool
    detection_method: str

class StatisticalThresholds:
    """Statistical significance thresholds based on Combrisson and Jerbi (2015)"""
    
    # Binary classification thresholds (2-class)
    BINARY_THRESHOLDS = {
        100: 0.58,   # 58%
        200: 0.56,   # 56%
        500: 0.536,  # 53.6%
    }
    
    # 4-class classification thresholds
    FOUR_CLASS_THRESHOLDS = {
        100: 0.32,   # 32.0%
        200: 0.30,   # 30.0%
        500: 0.282,  # 28.2%
    }
    
    @classmethod
    def get_chance_threshold(cls, n_classes: int, sample_size: int) -> float:
        """
        Get the statistical significance threshold for chance-level performance.
        
        Args:
            n_classes: Number of classes (2 for binary, 4 for SSVEP)
            sample_size: Number of test samples
            
        Returns:
            Statistical significance threshold (0.0 to 1.0)
        """
        if n_classes == 2:
            thresholds = cls.BINARY_THRESHOLDS
        elif n_classes == 4:
            thresholds = cls.FOUR_CLASS_THRESHOLDS
        else:
            raise ValueError(f"Unsupported number of classes: {n_classes}")
        
        # Find closest sample size or interpolate
        if sample_size in thresholds:
            return thresholds[sample_size]
        
        # Interpolate between closest sample sizes
        sizes = sorted(thresholds.keys())
        if sample_size < sizes[0]:
            return thresholds[sizes[0]]
        elif sample_size > sizes[-1]:
            return thresholds[sizes[-1]]
        else:
            # Linear interpolation
            for i in range(len(sizes) - 1):
                if sizes[i] <= sample_size <= sizes[i + 1]:
                    size1, size2 = sizes[i], sizes[i + 1]
                    thresh1, thresh2 = thresholds[size1], thresholds[size2]
                    # Linear interpolation
                    ratio = (sample_size - size1) / (size2 - size1)
                    return thresh1 + ratio * (thresh2 - thresh1)
        
        return thresholds[sizes[-1]]  # Fallback

class AdaptiveSaturationDetector:
    """
    Adaptive saturation point detector for noise benchmarking across multiple datasets.
    
    Uses a three-phase approach:
    1. Coarse exploration to identify rough saturation region
    2. Refined binary search around saturation region
    3. Statistical validation of identified saturation point
    """
    
    def __init__(self, 
                 model_name: str = "eegnet",
                 base_seed: int = 42,
                 output_dir: str = "saturation_results"):
        """
        Initialize the saturation detector.
        
        Args:
            model_name: Model to use for saturation detection (default: eegnet for speed)
            base_seed: Base random seed for reproducibility
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.base_seed = base_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            "BNCI2014_001": {
                "dataset_class": BNCI2014_001,
                "paradigm_type": "MotorImagery",
                "n_classes": 2,
                "subjects": [3],  # Use first 2 subjects for speed
                "resample": None
            },
            "Lee2019_SSVEP": {
                "dataset_class": Lee2019_SSVEP,
                "paradigm_type": "SSVEP", 
                "n_classes": 4,
                "subjects": [3],  # Use first 2 subjects
                "resample": None
            },
            "BI2015a": {
                "dataset_class": BI2015a,
                "paradigm_type": "ERP",
                "n_classes": 2,
                "subjects": [3],  # Use first few subjects for speed
                "resample": None
            }
        }
        
        # Noise types to test
        # self.noise_types = ["eog"]
        self.noise_types = ["gaussian", "dropout", "eog", "spike"]
        
        # Phase 1: Coarse exploration parameters
        self.coarse_intensities = [0, 25, 50, 75, 100]
        self.n_trials_coarse = 3
        
        # Phase 2: Refined search parameters  
        self.n_trials_fine = 5
        self.max_binary_search_steps = 5
        
        # Phase 3: Validation parameters
        self.n_trials_validation = 10
        
        # Saturation detection parameters
        self.saturation_margin = 0.05  # 5% margin above chance level
        self.min_intensity_step = 5.0  # Minimum step size for binary search
        
        # Debugging parameters
        self.debug_mode = False
        self.debug_output_dir = self.output_dir / "debug_output"
        
    def detect_saturation_points(self, 
                                datasets: List[str] = None,
                                noise_types: List[str] = None) -> Dict[str, Dict[str, SaturationResult]]:
        """
        Detect saturation points for specified datasets and noise types.
        
        Args:
            datasets: List of dataset names to test (default: all configured)
            noise_types: List of noise types to test (default: all configured)
            
        Returns:
            Nested dictionary: {dataset: {noise_type: SaturationResult}}
        """
        if datasets is None:
            datasets = list(self.dataset_configs.keys())
        if noise_types is None:
            noise_types = self.noise_types
            
        results = {}
        
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            if dataset_name not in self.dataset_configs:
                print(f"Warning: Dataset {dataset_name} not configured, skipping...")
                continue
            
            # Train model once for this dataset
            print("Training model on clean data...")
            try:
                trained_model, X_test, y_test = self._train_model_once(dataset_name)
                print(f"Model trained successfully on {len(X_test)} test samples")
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"ERROR: Failed to train model for {dataset_name}")
                print(f"{'='*60}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"\nFull traceback:")
                import traceback
                traceback.print_exc()
                print(f"{'='*60}\n")
                continue
                
            dataset_results = {}
            
            # Evaluate all noise types using the same trained model
            for noise_type in noise_types:
                print(f"\n--- Detecting saturation point for {noise_type} noise ---")
                
                try:
                    result = self._detect_single_saturation_point_with_trained_model(
                        dataset_name, noise_type, trained_model, X_test, y_test
                    )
                    dataset_results[noise_type] = result
                    
                    # Save intermediate results
                    self._save_intermediate_result(dataset_name, noise_type, result)
                    
                except Exception as e:
                    print(f"Error detecting saturation point for {dataset_name}/{noise_type}: {e}")
                    continue
                    
            results[dataset_name] = dataset_results
            
        # Save final results
        self._save_final_results(results)
        return results
    
    def debug_lee2019_evaluation(self, dataset_name: str = "Lee2019_SSVEP", 
                                subject_id: int = 1, noise_type: str = "gaussian", 
                                intensity: float = 10.0) -> None:
        """
        Comprehensive debugging of Lee2019 SSVEP evaluation pipeline.
        
        This method investigates potential issues causing suspiciously high results:
        1. Class distribution analysis
        2. Signal characteristics per class
        3. Per-class performance analysis
        4. Noise application verification
        5. Evaluation methodology issues
        """
        print("=" * 80)
        print("COMPREHENSIVE LEE2019 SSVEP DEBUGGING")
        print("=" * 80)
        
        # Enable debug mode
        self.debug_mode = True
        self.debug_output_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f"\n1. LOADING DATA (Dataset: {dataset_name}, Subject: {subject_id})")
        print("-" * 60)
        
        config = self.dataset_configs[dataset_name]
        dataset = config["dataset_class"]()
        dataset.subject_list = [subject_id]
        
        paradigm = get_paradigm(resample=None, dataset=dataset_name)
        X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id])
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        y_encoded = LabelEncoder().fit_transform(y)
        
        print(f"Data shape: {X.shape}")
        print(f"Original labels: {np.unique(y)}")
        print(f"Encoded labels: {np.unique(y_encoded)}")
        print(f"Class distribution: {np.bincount(y_encoded)}")
        
        # Check class balance
        class_counts = np.bincount(y_encoded)
        if len(class_counts) > 1:
            imbalance_ratio = np.max(class_counts) / np.min(class_counts)
            print(f"Class imbalance ratio: {imbalance_ratio:.3f}")
            if imbalance_ratio > 1.2:
                print("⚠️  WARNING: Class imbalance detected!")
            else:
                print("✅ Classes are well balanced")
        
        # Analyze signal characteristics
        print(f"\n2. SIGNAL CHARACTERISTICS ANALYSIS")
        print("-" * 60)
        
        overall_power = np.mean(X**2)
        overall_std = np.std(X)
        print(f"Overall signal power: {overall_power:.6f}")
        print(f"Overall signal std: {overall_std:.6f}")
        
        # Per-class signal analysis
        class_names = ['12.0Hz', '5.45Hz', '6.67Hz', '8.57Hz']
        print(f"\nPer-class signal characteristics:")
        class_powers = []
        for i, class_name in enumerate(class_names):
            class_mask = y_encoded == i
            if class_mask.sum() > 0:
                X_class = X[class_mask]
                class_power = np.mean(X_class**2)
                class_std = np.std(X_class)
                class_powers.append(class_power)
                print(f"  {class_name}: power={class_power:.6f}, std={class_std:.6f} ({class_mask.sum()} samples)")
        
        # Check for trivial separability
        if len(class_powers) > 1:
            power_ratio = max(class_powers) / min(class_powers)
            print(f"\nPower ratio between classes: {power_ratio:.3f}")
            if power_ratio > 10:
                print("⚠️  WARNING: Classes have very different power levels!")
                print("   Classification might be trivial based on signal power.")
            else:
                print("✅ Classes have similar power levels - good for fair evaluation.")
        
        # Train model and evaluate
        print(f"\n3. MODEL TRAINING AND EVALUATION")
        print("-" * 60)
        
        # Split data like test_perturb does
        if 'session' in metadata.columns:
            train_mask = metadata['session'] == '0train'
            test_mask = metadata['session'] != '0train'
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                X_train = X[train_mask]
                y_train = y_encoded[train_mask]
                X_test = X[test_mask]
                y_test = y_encoded[test_mask]
                print(f"Session-based split: {len(X_train)} train, {len(X_test)} test")
            else:
                print("⚠️  WARNING: No session split found, using random split")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=self.base_seed, stratify=y_encoded
                )
        else:
            print("⚠️  WARNING: No session info, using random split")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=self.base_seed, stratify=y_encoded
            )
        
        # Create and train model
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]
        n_outputs = config["n_classes"]
        
        model = create_eegnet_classifier(n_chans, n_times, n_outputs, seed=self.base_seed)
        model.max_epochs = get_max_epochs_for_dataset(dataset_name)
        print(f"Training model with max_epochs={model.max_epochs} for dataset {dataset_name}")
        model.initialize()
        
        print("Training model...")
        model.module_.train()
        model.fit(X_train, y_train)
        
        # Evaluate on clean test data
        print("\nEvaluating on clean test data...")
        model.module_.eval()
        import torch
        
        with torch.no_grad():
            y_pred_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            metrics_clean = compute_classification_metrics(y_test, y_pred_proba, n_outputs)
            clean_auc = metrics_clean["roc_auc"]
        
        print(f"Clean test performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Balanced Accuracy: {balanced_acc:.3f}")
        print(f"  ROC-AUC: {clean_auc:.3f}")
        
        if accuracy > 0.95:
            print("⚠️  WARNING: Accuracy > 95% - suspiciously high!")
        
        if abs(accuracy - balanced_acc) > 0.05:
            print("⚠️  WARNING: Large difference between accuracy and balanced accuracy!")
        
        # Per-class performance analysis
        print(f"\nPer-class performance on clean data:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        print(f"\nConfusion matrix (clean data):")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Noise application test
        print(f"\n4. NOISE APPLICATION TEST (Type: {noise_type}, Intensity: {intensity}%)")
        print("-" * 60)
        
        # Create noise augmentor
        augmentor = EEGNoiseAugmentor(noise_type=noise_type, intensity=intensity, seed=self.base_seed)
        
        # Apply noise to test data
        print("Applying noise to test data...")
        original_test_power = np.mean(X_test**2)
        original_test_std = np.std(X_test)
        
        X_test_noisy = augmentor.transform(X_test)
        
        noisy_test_power = np.mean(X_test_noisy**2)
        noisy_test_std = np.std(X_test_noisy)
        
        print(f"Signal power analysis:")
        print(f"  Original power: {original_test_power:.6f}")
        print(f"  Noisy power: {noisy_test_power:.6f}")
        print(f"  Power ratio: {noisy_test_power/original_test_power:.3f}")
        print(f"  Std ratio: {noisy_test_std/original_test_std:.3f}")
        
        if abs(noisy_test_power - original_test_power) < 1e-10:
            print("⚠️  WARNING: Signal power unchanged - noise may not be applied!")
        else:
            print("✅ Noise successfully applied - signal changed")
        
        # Evaluate on noisy data
        print("\nEvaluating on noisy test data...")
        with torch.no_grad():
            y_pred_proba_noisy = model.predict_proba(X_test_noisy)
            y_pred_noisy = np.argmax(y_pred_proba_noisy, axis=1)
            
            # Calculate metrics
            noisy_accuracy = accuracy_score(y_test, y_pred_noisy)
            noisy_balanced_acc = balanced_accuracy_score(y_test, y_pred_noisy)
            metrics_noisy = compute_classification_metrics(y_test, y_pred_proba_noisy, n_outputs)
            noisy_auc = metrics_noisy["roc_auc"]
        
        print(f"Noisy test performance:")
        print(f"  Accuracy: {noisy_accuracy:.3f}")
        print(f"  Balanced Accuracy: {noisy_balanced_acc:.3f}")
        print(f"  ROC-AUC: {noisy_auc:.3f}")
        
        performance_drop = (accuracy - noisy_accuracy) / accuracy * 100
        print(f"  Performance drop: {performance_drop:.1f}%")
        
        if noisy_accuracy > 0.9:
            print("⚠️  WARNING: Still very high performance with noise!")
        
        if performance_drop < 10:
            print("⚠️  WARNING: Very small performance drop - noise may not be effective!")
        
        # Per-class performance under noise
        print(f"\nPer-class performance under noise:")
        print(classification_report(y_test, y_pred_noisy, target_names=class_names))
        
        print(f"\nConfusion matrix (noisy data):")
        cm_noisy = confusion_matrix(y_test, y_pred_noisy)
        print(cm_noisy)
        
        # Cross-validation consistency check
        print(f"\n5. CROSS-VALIDATION CONSISTENCY CHECK")
        print("-" * 60)
        
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.base_seed)
        
        fold_scores = []
        fold_accuracies = []
        
        for i, (train_idx, valid_idx) in enumerate(cv.split(X, y_encoded)):
            print(f"\nFold {i}:")
            print(f"  Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")
            
            # Check class distribution in fold
            train_classes = np.bincount(y_encoded[train_idx])
            valid_classes = np.bincount(y_encoded[valid_idx])
            print(f"  Train class distribution: {train_classes}")
            print(f"  Valid class distribution: {valid_classes}")
            
            # Train model on fold
            X_train_fold = X[train_idx]
            y_train_fold = y_encoded[train_idx]
            
            model_fold = create_eegnet_classifier(n_chans, n_times, n_outputs, seed=self.base_seed + i)
            model_fold.max_epochs = get_max_epochs_for_dataset(dataset_name)
            model_fold.initialize()
            model_fold.module_.train()
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            X_valid_fold = X[valid_idx]
            y_valid_fold = y_encoded[valid_idx]
            
            # Test with noise
            X_valid_fold_noisy = augmentor.transform(X_valid_fold)
            
            model_fold.module_.eval()
            with torch.no_grad():
                y_pred_proba_fold = model_fold.predict_proba(X_valid_fold_noisy)
                y_pred_fold = np.argmax(y_pred_proba_fold, axis=1)
                
                fold_acc = accuracy_score(y_valid_fold, y_pred_fold)
                fold_auc = compute_classification_metrics(y_valid_fold, y_pred_proba_fold, n_outputs)["roc_auc"]
                
                print(f"  Fold accuracy: {fold_acc:.3f}")
                print(f"  Fold ROC-AUC: {fold_auc:.3f}")
                
                fold_scores.append(fold_auc)
                fold_accuracies.append(fold_acc)
        
        # Overall CV analysis
        print(f"\nOverall CV Results:")
        print(f"  Mean ROC-AUC: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
        print(f"  Mean Accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
        print(f"  Score range: {np.min(fold_scores):.3f} - {np.max(fold_scores):.3f}")
        
        if np.std(fold_scores) > 0.1:
            print("⚠️  WARNING: High variance across CV folds!")
        
        # Summary and recommendations
        print(f"\n6. SUMMARY AND RECOMMENDATIONS")
        print("=" * 80)
        
        issues_found = []
        
        if accuracy > 0.95:
            issues_found.append("Accuracy > 95% - suspiciously high")
        
        if abs(accuracy - balanced_acc) > 0.05:
            issues_found.append("Large accuracy vs balanced accuracy difference")
        
        if noisy_accuracy > 0.9:
            issues_found.append("High performance even with noise")
        
        if performance_drop < 10:
            issues_found.append("Very small performance drop with noise")
        
        if len(class_powers) > 1 and max(class_powers) / min(class_powers) > 10:
            issues_found.append("Classes have very different power levels")
        
        if np.std(fold_scores) > 0.1:
            issues_found.append("High variance across CV folds")
        
        if issues_found:
            print("⚠️  ISSUES IDENTIFIED:")
            for i, issue in enumerate(issues_found, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ No major issues identified in the evaluation pipeline")
        
        print(f"\nRecommendations:")
        print("1. Verify noise application is working correctly")
        print("2. Check if SSVEP patterns are too distinctive")
        print("3. Compare with expected SSVEP baselines (70-90% clean accuracy)")
        print("4. Consider using balanced accuracy instead of raw accuracy")
        print("5. Add per-class performance tracking in main evaluation")
        
        # Save debug results
        debug_results = {
            "dataset": dataset_name,
            "subject": subject_id,
            "noise_type": noise_type,
            "intensity": intensity,
            "clean_accuracy": float(accuracy),
            "clean_balanced_accuracy": float(balanced_acc),
            "clean_roc_auc": float(clean_auc),
            "noisy_accuracy": float(noisy_accuracy),
            "noisy_balanced_accuracy": float(noisy_balanced_acc),
            "noisy_roc_auc": float(noisy_auc),
            "performance_drop_percent": float(performance_drop),
            "power_ratio": float(noisy_test_power/original_test_power),
            "cv_mean_auc": float(np.mean(fold_scores)),
            "cv_std_auc": float(np.std(fold_scores)),
            "issues_found": issues_found,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        debug_file = self.debug_output_dir / f"debug_{dataset_name}_sub{subject_id}_{noise_type}_{intensity}.json"
        with open(debug_file, 'w') as f:
            json.dump(debug_results, f, indent=2)
        
        print(f"\nDebug results saved to: {debug_file}")
        
        return debug_results
    
    def _detect_single_saturation_point_with_trained_model(self, dataset_name: str, noise_type: str, 
                                                          trained_model, X_test, y_test) -> SaturationResult:
        """Detect saturation point for a single dataset-noise combination using a pre-trained model."""
        
        config = self.dataset_configs[dataset_name]
        chance_threshold = StatisticalThresholds.get_chance_threshold(
            config["n_classes"], 
            sample_size=200  # Approximate test set size
        )
        
        print(f"Chance threshold for {config['n_classes']}-class, ~200 samples: {chance_threshold:.3f}")
        
        # Phase 1: Coarse exploration
        print("Phase 1: Coarse exploration...")
        coarse_results = self._coarse_exploration_with_trained_model(
            trained_model, X_test, y_test, noise_type
        )
        
        # SIMPLIFIED APPROACH: Check if any intensity shows below-chance performance
        below_chance_intensity = self._find_first_below_chance_intensity(coarse_results, chance_threshold)
        
        if below_chance_intensity is not None:
            print(f"Found below-chance performance at intensity {below_chance_intensity}%, using as saturation point")
            return self._create_saturation_result_from_coarse(
                dataset_name, noise_type, below_chance_intensity, chance_threshold, 
                coarse_results, "simplified_below_chance"
            )
        
        # If no below-chance performance found, use the complex three-phase approach
        print("No below-chance performance found, using three-phase approach...")
        
        # Identify rough saturation region
        saturation_region = self._identify_saturation_region(coarse_results, chance_threshold)
        print(f"Saturation region identified: {saturation_region}")
        
        if saturation_region is None:
            print("No saturation region found in coarse exploration, using maximum intensity")
            return self._create_saturation_result(
                dataset_name, noise_type, 100.0, chance_threshold, 
                coarse_results, "coarse_exploration_only"
            )
        
        # Phase 2: Refined binary search
        print("Phase 2: Refined binary search...")
        refined_result = self._refined_binary_search_with_trained_model(
            trained_model, X_test, y_test, noise_type, saturation_region, chance_threshold
        )
        
        # Phase 3: Statistical validation
        print("Phase 3: Statistical validation...")
        final_result = self._statistical_validation_with_trained_model(
            trained_model, X_test, y_test, dataset_name, noise_type, 
            refined_result, chance_threshold
        )
        
        return final_result
    
    def _train_model_once(self, dataset_name: str):
        """Train model once on clean data (like test_perturb mode)."""
        
        print(f"  [DEBUG] Starting _train_model_once for {dataset_name}")
        config = self.dataset_configs[dataset_name]
        print(f"  [DEBUG] Config loaded: {config.keys()}")
        print(f"  [DEBUG] base_seed: {self.base_seed} (type: {type(self.base_seed)})")
        
        # Set random seed
        print(f"  [DEBUG] Setting seeds...")
        set_seeds(self.base_seed)
        
        # Load dataset and paradigm using the exact same pattern as test_perturb
        print(f"  [DEBUG] Creating dataset instance...")
        dataset = config["dataset_class"]()
        subject_id = config["subjects"][0]
        print(f"  [DEBUG] Using subject: {subject_id}")
        dataset.subject_list = [subject_id]
        
        print(f"  [DEBUG] Getting paradigm...")
        paradigm = get_paradigm(resample=None, dataset=dataset_name)
        print(f"  [DEBUG] Paradigm type: {type(paradigm)}")
        
        # Load data using the same pattern as test_perturb
        print(f"  [DEBUG] Loading data for subject {subject_id}...")
        X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id])
        print(f"  [DEBUG] Data loaded - X shape: {X.shape}, y shape: {y.shape}, metadata columns: {metadata.columns.tolist()}")
        
        # CRITICAL: Encode labels the same way as test_perturb
        from sklearn.preprocessing import LabelEncoder
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Use session-based splitting like test_perturb does
        print(f"  [DEBUG] Checking session information...")
        if 'session' in metadata.columns:
            print(f"  [DEBUG] Found session column. Unique sessions: {sorted(metadata['session'].unique())}")
            # Check if dataset uses '0train' session naming (BNCI2014_001, Lee2019_SSVEP)
            train_mask = metadata['session'] == '0train'
            test_mask = metadata['session'] != '0train'
            print(f"  [DEBUG] '0train' session found: {train_mask.sum()} samples")
            print(f"  [DEBUG] Other sessions found: {test_mask.sum()} samples")
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                # Standard session-based split (BNCI2014_001, Lee2019_SSVEP)
                print(f"  [DEBUG] Using standard '0train' session split")
                X_train = X[train_mask]
                y_train = y_encoded[train_mask]
                X_test = X[test_mask]
                y_test = y_encoded[test_mask]
            else:
                # For datasets like BI2015a with multiple sessions, use LeaveOneGroupOut
                # Use 2 sessions for training, 1 session for testing
                print(f"  [DEBUG] '0train' not found, using LeaveOneGroupOut for multi-session dataset")
                from sklearn.model_selection import LeaveOneGroupOut
                groups = metadata['session'].values
                print(f"  [DEBUG] Session groups: {sorted(set(groups))}")
                logo = LeaveOneGroupOut()
                splits = list(logo.split(X, y_encoded, groups=groups))
                print(f"  [DEBUG] LeaveOneGroupOut created {len(splits)} splits")
                
                if len(splits) > 0:
                    # Use the first split (test on first session, train on others)
                    train_idx, test_idx = splits[0]
                    X_train = X[train_idx]
                    y_train = y_encoded[train_idx]
                    X_test = X[test_idx]
                    y_test = y_encoded[test_idx]
                    print(f"  [DEBUG] Using LeaveOneGroupOut: training on {len(train_idx)} samples, testing on {len(test_idx)} samples")
                else:
                    # Fallback: use simple split
                    print(f"  [DEBUG] No splits found, using train_test_split fallback")
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.3, random_state=self.base_seed, stratify=y_encoded
                    )
        else:
            # No session info, use simple split
            print(f"  [DEBUG] No session column found, using train_test_split")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=self.base_seed, stratify=y_encoded
            )
        
        print(f"  [DEBUG] Data split complete - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        # Create model using the same pattern as test_perturb
        print(f"  [DEBUG] Creating model...")
        print(f"  [DEBUG] X_train shape: {X_train.shape}")
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]
        n_outputs = config["n_classes"]
        print(f"  [DEBUG] Model params - n_chans: {n_chans}, n_times: {n_times}, n_outputs: {n_outputs}")
        print(f"  [DEBUG] base_seed for model creation: {self.base_seed} (type: {type(self.base_seed)})")
        
        model = create_eegnet_classifier(n_chans, n_times, n_outputs, seed=self.base_seed)
        print(f"  [DEBUG] Model created successfully")
        model.max_epochs = get_max_epochs_for_dataset(dataset_name)
        print(f"  [DEBUG] Training with max_epochs={model.max_epochs} for dataset {dataset_name}")
        model.initialize()
        
        # Train on clean data (same as test_perturb)
        model.module_.train()
        model.fit(X_train, y_train)
        
        # Evaluate on clean test data to check for underfitting
        model.module_.eval()
        import torch
        
        with torch.no_grad():
            y_pred_proba = model.predict_proba(X_test)
            metrics_clean = compute_classification_metrics(y_test, y_pred_proba, config["n_classes"])
            clean_score = metrics_clean["roc_auc"]
        
        print(f"  Initial clean score: {clean_score:.3f}")
        
        # Check for underfitting and retrain if necessary (same as test_perturb)
        underfitting_threshold = get_underfitting_threshold_for_dataset(dataset_name)
        if clean_score < underfitting_threshold:
            print(f"  Re-training model without EarlyStopping due to underfitting (threshold: {underfitting_threshold:.3f})")
            model.callbacks = []
            model.module_.train()          
            model.fit(X_train, y_train)
            model.module_.eval()
            with torch.no_grad():
                y_pred_proba = model.predict_proba(X_test)
                metrics_retrain = compute_classification_metrics(y_test, y_pred_proba, config["n_classes"])
                new_clean_score = metrics_retrain["roc_auc"]
            clean_score = max(clean_score, new_clean_score)
            print(f"  Retrained clean score: {clean_score:.3f}")
        
        print(f"  Final model trained on {len(X_train)} samples, will evaluate on {len(X_test)} test samples")
        
        return model, X_test, y_test
    
    def _coarse_exploration_with_trained_model(self, trained_model, X_test, y_test, noise_type) -> List[Tuple[float, float, float]]:
        """Phase 1: Coarse exploration of performance across intensity range using pre-trained model."""
        results = []
        
        for intensity in self.coarse_intensities:
            print(f"  Testing intensity {intensity}%...")
            
            # Evaluate multiple times with different seeds for the noise
            performances = []
            for trial in range(self.n_trials_coarse):
                seed = self.base_seed + trial * 100
                performance = self._evaluate_intensity_with_trained_model(
                    trained_model, X_test, y_test, noise_type, intensity, seed
                )
                if performance is not None:
                    performances.append(performance)
            
            if performances:
                mean_perf = np.mean(performances)
                std_perf = np.std(performances)
                results.append((intensity, mean_perf, std_perf))
                print(f"    Intensity {intensity}%: {mean_perf:.3f} ± {std_perf:.3f}")
            else:
                print(f"    Intensity {intensity}%: Failed to evaluate")
                
        return results
    
    def _evaluate_intensity_with_trained_model(self, trained_model, X_test, y_test, noise_type, intensity, seed):
        """Evaluate a single intensity using a pre-trained model (like test_perturb mode)."""
        
        try:
            # Apply noise to test data if intensity > 0 (same as test_perturb)
            if intensity > 0:
                augmentor = EEGNoiseAugmentor(noise_type=noise_type, intensity=intensity, seed=seed)
                X_test_corrupted = augmentor.transform(X_test)
            else:
                X_test_corrupted = X_test
            
            # Evaluate using the same scoring as test_perturb
            trained_model.module_.eval()
            import torch
            
            with torch.no_grad():
                y_pred_proba = trained_model.predict_proba(X_test_corrupted)
                n_classes = y_pred_proba.shape[1]
                metrics = compute_classification_metrics(y_test, y_pred_proba, n_classes)
                score = metrics["roc_auc"]
            
            return float(score)
                
        except Exception as e:
            print(f"    Error in evaluation: {e}")
            return None
    
    def _identify_saturation_region(self, results: List[Tuple[float, float, float]], 
                                   chance_threshold: float) -> Optional[Tuple[float, float]]:
        """Identify the intensity range where saturation likely occurs."""
        
        # Find the transition from good performance to saturated performance
        good_intensities = []
        saturated_intensities = []
        
        for intensity, mean_perf, std_perf in results:
            # Conservative threshold: performance below chance + margin
            saturation_threshold = chance_threshold + self.saturation_margin
            
            if mean_perf > saturation_threshold:
                good_intensities.append(intensity)
            else:
                saturated_intensities.append(intensity)
        
        if not good_intensities or not saturated_intensities:
            return None
            
        # Return the region between last good and first saturated
        max_good = max(good_intensities)
        min_saturated = min(saturated_intensities)
        
        return (max_good, min_saturated)
    
    def _find_first_below_chance_intensity(self, results: List[Tuple[float, float, float]], 
                                         chance_threshold: float) -> Optional[float]:
        """
        Find the first intensity that shows below-chance performance.
        
        Args:
            results: List of (intensity, mean_performance, std_performance) tuples
            chance_threshold: Statistical significance threshold for chance-level performance
            
        Returns:
            First intensity with below-chance performance, or None if none found
        """
        for intensity, mean_perf, std_perf in results:
            if mean_perf < chance_threshold:
                print(f"  Found below-chance performance at {intensity}%: {mean_perf:.3f} < {chance_threshold:.3f}")
                return intensity
        
        print(f"  No below-chance performance found (all performances >= {chance_threshold:.3f})")
        return None
    
    def _create_saturation_result_from_coarse(self, dataset_name: str, noise_type: str,
                                            intensity: float, chance_threshold: float,
                                            coarse_results: List[Tuple[float, float, float]],
                                            method: str) -> SaturationResult:
        """Create a saturation result from coarse exploration with below-chance performance."""
        
        # Find performance at the intensity
        performance = None
        performance_std = None
        for int_val, mean_perf, std_perf in coarse_results:
            if abs(int_val - intensity) < 0.1:
                performance = mean_perf
                performance_std = std_perf
                break
        
        if performance is None:
            performance = chance_threshold - 0.1  # Default fallback
            performance_std = 0.1
        
        # Calculate confidence interval (95%)
        n_trials = self.n_trials_coarse
        confidence_interval = (
            performance - 1.96 * performance_std / np.sqrt(n_trials),
            performance + 1.96 * performance_std / np.sqrt(n_trials)
        )
        
        # Check if performance is significantly below chance
        is_significant = performance < chance_threshold
        
        return SaturationResult(
            noise_type=noise_type,
            dataset=dataset_name,
            saturation_point=intensity,
            confidence_interval=confidence_interval,
            sample_size=200,  # Approximate
            chance_threshold=chance_threshold,
            validation_trials=self.n_trials_coarse,
            performance_at_saturation=performance,
            performance_std=performance_std,
            is_statistically_significant=is_significant,
            detection_method=method
        )
    
    def _refined_binary_search_with_trained_model(self, trained_model, X_test, y_test, noise_type,
                                                 saturation_region: Tuple[float, float], 
                                                 chance_threshold: float) -> Tuple[float, float, float]:
        """Phase 2: Refined binary search in saturation region."""
        
        low_intensity, high_intensity = saturation_region
        
        for step in range(self.max_binary_search_steps):
            if high_intensity - low_intensity < self.min_intensity_step:
                break
                
            mid_intensity = (low_intensity + high_intensity) / 2
            print(f"  Binary search step {step + 1}: testing {mid_intensity:.1f}%...")
            
            performances = []
            for trial in range(self.n_trials_fine):
                seed = self.base_seed + trial * 100 + step * 50
                performance = self._evaluate_intensity_with_trained_model(
                    trained_model, X_test, y_test, noise_type, mid_intensity, seed
                )
                if performance is not None:
                    performances.append(performance)
            
            if not performances:
                print(f"    Failed to evaluate {mid_intensity:.1f}%")
                break
                
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            saturation_threshold = chance_threshold + self.saturation_margin
            
            print(f"    {mid_intensity:.1f}%: {mean_perf:.3f} ± {std_perf:.3f} (threshold: {saturation_threshold:.3f})")
            
            if mean_perf > saturation_threshold:
                low_intensity = mid_intensity
            else:
                high_intensity = mid_intensity
        
        # Return the final high intensity as saturation point
        final_intensity = high_intensity
        
        # Get final performance estimate
        performances = []
        for trial in range(self.n_trials_fine):
            seed = self.base_seed + trial * 100 + 1000
            performance = self._evaluate_intensity_with_trained_model(
                trained_model, X_test, y_test, noise_type, final_intensity, seed
            )
            if performance is not None:
                performances.append(performance)
        
        if performances:
            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
        else:
            mean_perf, std_perf = 0.0, 0.0
            
        return (final_intensity, mean_perf, std_perf)
    
    def _statistical_validation_with_trained_model(self, trained_model, X_test, y_test, 
                                                  dataset_name: str, noise_type: str,
                                                  refined_result: Tuple[float, float, float],
                                                  chance_threshold: float) -> SaturationResult:
        """Phase 3: Statistical validation of saturation point."""
        
        saturation_intensity, _, _ = refined_result
        
        print(f"  Validating saturation point at {saturation_intensity:.1f}%...")
        
        performances = []
        for trial in range(self.n_trials_validation):
            seed = self.base_seed + trial * 100 + 2000
            performance = self._evaluate_intensity_with_trained_model(
                trained_model, X_test, y_test, noise_type, saturation_intensity, seed
            )
            if performance is not None:
                performances.append(performance)
        
        if not performances:
            raise ValueError(f"Failed to validate saturation point at {saturation_intensity:.1f}%")
        
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        
        # Calculate confidence interval (95%)
        n_samples = len(performances)
        confidence_interval = (
            mean_perf - 1.96 * std_perf / np.sqrt(n_samples),
            mean_perf + 1.96 * std_perf / np.sqrt(n_samples)
        )
        
        # Check statistical significance
        is_significant = mean_perf > (chance_threshold + self.saturation_margin)
        
        print(f"  Final validation: {mean_perf:.3f} ± {std_perf:.3f}")
        print(f"  Confidence interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        print(f"  Statistically significant saturation: {is_significant}")
        
        return SaturationResult(
            noise_type=noise_type,
            dataset=dataset_name,
            saturation_point=saturation_intensity,
            confidence_interval=confidence_interval,
            sample_size=200,  # Approximate
            chance_threshold=chance_threshold,
            validation_trials=self.n_trials_validation,
            performance_at_saturation=mean_perf,
            performance_std=std_perf,
            is_statistically_significant=is_significant,
            detection_method="adaptive_three_phase"
        )
    
    def _create_saturation_result(self, dataset_name: str, noise_type: str,
                                 intensity: float, chance_threshold: float,
                                 coarse_results: List[Tuple[float, float, float]],
                                 method: str) -> SaturationResult:
        """Create a saturation result from coarse exploration only."""
        
        # Find performance at the intensity
        performance = None
        for int_val, mean_perf, std_perf in coarse_results:
            if abs(int_val - intensity) < 0.1:
                performance = mean_perf
                break
        
        if performance is None:
            performance = 0.5  # Default fallback
        
        return SaturationResult(
            noise_type=noise_type,
            dataset=dataset_name,
            saturation_point=intensity,
            confidence_interval=(performance - 0.1, performance + 0.1),
            sample_size=200,
            chance_threshold=chance_threshold,
            validation_trials=self.n_trials_coarse,
            performance_at_saturation=performance,
            performance_std=0.1,
            is_statistically_significant=False,
            detection_method=method
        )
    
    def _save_intermediate_result(self, dataset_name: str, noise_type: str, result: SaturationResult):
        """Save intermediate result to file."""
        filename = f"{dataset_name}_{noise_type}_saturation.json"
        filepath = self.output_dir / filename
        
        result_dict = {
            "noise_type": result.noise_type,
            "dataset": result.dataset,
            "saturation_point": float(result.saturation_point),
            "confidence_interval": [float(x) for x in result.confidence_interval],
            "chance_threshold": float(result.chance_threshold),
            "performance_at_saturation": float(result.performance_at_saturation),
            "performance_std": float(result.performance_std),
            "is_statistically_significant": bool(result.is_statistically_significant),
            "detection_method": result.detection_method,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_final_results(self, results: Dict[str, Dict[str, SaturationResult]]):
        """Save final results summary, merging with existing results if present."""
        
        # Create summary DataFrame from new results
        summary_data = []
        for dataset, noise_results in results.items():
            for noise_type, result in noise_results.items():
                summary_data.append({
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "saturation_point": float(result.saturation_point),
                    "performance_at_saturation": float(result.performance_at_saturation),
                    "performance_std": float(result.performance_std),
                    "chance_threshold": float(result.chance_threshold),
                    "is_significant": bool(result.is_statistically_significant),
                    "confidence_interval_lower": float(result.confidence_interval[0]),
                    "confidence_interval_upper": float(result.confidence_interval[1]),
                    "detection_method": result.detection_method
                })
        
        new_summary_df = pd.DataFrame(summary_data)
        
        # Load existing results if they exist
        csv_path = self.output_dir / "saturation_points_summary.csv"
        if csv_path.exists():
            print(f"\nLoading existing results from {csv_path}")
            existing_df = pd.read_csv(csv_path)
            print(f"  Found {len(existing_df)} existing entries")
            
            # Merge: remove existing entries that match new results (by dataset+noise_type)
            # and append new results
            merge_keys = ["dataset", "noise_type"]
            existing_df = existing_df[~existing_df.set_index(merge_keys).index.isin(
                new_summary_df.set_index(merge_keys).index
            )]
            
            # Combine existing (filtered) and new results
            summary_df = pd.concat([existing_df, new_summary_df], ignore_index=True)
            print(f"  Merged results: {len(existing_df)} existing + {len(new_summary_df)} new = {len(summary_df)} total")
        else:
            print(f"\nNo existing results found, creating new file")
            summary_df = new_summary_df
        
        # Save CSV
        summary_df.to_csv(csv_path, index=False)
        
        # Save JSON
        json_path = self.output_dir / "saturation_points_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary_df.to_dict('records'), f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SATURATION POINTS SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))

def main():
    """Main function to run saturation point detection."""
    
    print("Adaptive Saturation Point Detection for Noise Benchmarking")
    print("Using Combrisson & Jerbi (2015) statistical significance thresholds")
    print("="*80)
    
    # Initialize detector
    detector = AdaptiveSaturationDetector(
        model_name="eegnet",
        base_seed=100,
        output_dir="saturation_results"
    )
    
    # Detect saturation points for all configured datasets and noise types
    results = detector.detect_saturation_points()
    
    print("\nSaturation point detection completed!")
    return results

def debug_main():
    """Main function to run Lee2019 SSVEP debugging."""
    
    print("Lee2019 SSVEP Evaluation Pipeline Debugging")
    print("Investigating suspiciously high results under noise perturbation")
    print("="*80)
    
    # Initialize detector
    detector = AdaptiveSaturationDetector(
        model_name="eegnet",
        base_seed=100,
        output_dir="saturation_results"
    )
    
    # Run comprehensive debugging
    debug_results = detector.debug_lee2019_evaluation(
        dataset_name="Lee2019_SSVEP",
        subject_id=1,
        noise_type="gaussian",
        intensity=10.0
    )
    
    print("\nLee2019 SSVEP debugging completed!")
    print("Check the debug output directory for detailed results.")
    
    return debug_results

if __name__ == "__main__":
    results = main()

