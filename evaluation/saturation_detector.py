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
from moabb.paradigms import MotorImagery, SSVEP
from models.eegnet import create_eegnet_classifier
from augmentation.noise import TrainOnlyNoiseClassifier, EEGNoiseAugmentor
from evaluation.unified_experiment_runner import UnifiedExperimentRunner
from config import get_paradigm, MODEL_REGISTRY
from globals import set_seeds, get_seed, UNDERFITTING_THRESHOLD
from sklearn.metrics import get_scorer
from evaluation.metrics import compute_classification_metrics

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
                "subjects": list(range(1, 2)),  # Use first 2 subjects for speed
                "resample": None
            },
            "Lee2019_SSVEP": {
                "dataset_class": Lee2019_SSVEP,
                "paradigm_type": "SSVEP", 
                "n_classes": 4,
                "subjects": list(range(1, 2)),  # Use first 2 subjects
                "resample": None
            }
        }
        
        # Noise types to test
        self.noise_types = ["gaussian", "dropout", "eog"]
        
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
                
            dataset_results = {}
            
            for noise_type in noise_types:
                print(f"\n--- Detecting saturation point for {noise_type} noise ---")
                
                try:
                    result = self._detect_single_saturation_point(dataset_name, noise_type)
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
    
    def _detect_single_saturation_point(self, dataset_name: str, noise_type: str) -> SaturationResult:
        """Detect saturation point for a single dataset-noise combination."""
        
        config = self.dataset_configs[dataset_name]
        chance_threshold = StatisticalThresholds.get_chance_threshold(
            config["n_classes"], 
            sample_size=200  # Approximate test set size
        )
        
        print(f"Chance threshold for {config['n_classes']}-class, ~200 samples: {chance_threshold:.3f}")
        
        # Train model once on clean data (like test_perturb mode)
        print("Training model on clean data...")
        trained_model, X_test, y_test = self._train_model_once(dataset_name)
        
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
        
        config = self.dataset_configs[dataset_name]
        
        # Set random seed
        set_seeds(self.base_seed)
        
        # Load dataset and paradigm using the exact same pattern as test_perturb
        dataset = config["dataset_class"]()
        dataset.subject_list = [config["subjects"][0]]
        
        paradigm = get_paradigm(resample=None, dataset=dataset_name)
        
        # Load data using the same pattern as test_perturb
        X, y, metadata = paradigm.get_data(dataset, subjects=[config["subjects"][0]])
        
        # CRITICAL: Encode labels the same way as test_perturb
        from sklearn.preprocessing import LabelEncoder
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Use session-based splitting like test_perturb does
        if 'session' in metadata.columns:
            train_mask = metadata['session'] == '0train'
            test_mask = metadata['session'] != '0train'
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                X_train = X[train_mask]
                y_train = y_encoded[train_mask]
                X_test = X[test_mask]
                y_test = y_encoded[test_mask]
            else:
                # Fallback: use simple split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=self.base_seed, stratify=y_encoded
                )
        else:
            # No session info, use simple split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=self.base_seed, stratify=y_encoded
            )
        
        # Create model using the same pattern as test_perturb
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]
        n_outputs = config["n_classes"]
        
        model = create_eegnet_classifier(n_chans, n_times, n_outputs, seed=self.base_seed)
        model.max_epochs = 200  # Same as test_perturb
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
        if clean_score < UNDERFITTING_THRESHOLD:
            print(f"  Re-training model without EarlyStopping due to underfitting (threshold: {UNDERFITTING_THRESHOLD:.3f})")
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
        """Save final results summary."""
        
        # Create summary DataFrame
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
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = self.output_dir / "saturation_points_summary.csv"
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

if __name__ == "__main__":
    results = main()

