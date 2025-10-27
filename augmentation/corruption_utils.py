"""
Centralized corruption utilities for EEG noise robustness evaluation.

This module provides standardized interfaces for:
- Corruption planning and grid generation
- Deterministic seed derivation
- Corruption application
- Metric computation (AURC, relative drop)
- Result logging and metadata
"""

import numpy as np
import json
import yaml
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class CorruptionPlan:
    """
    Centralized configuration for corruption families and intensity grids.
    
    This class defines the standard corruption families, intensity grids,
    and parameters used across all robustness evaluation experiments.
    """
    
    def __init__(self):
        # Standard corruption families and their intensity grids
        # Based on EEGNoiseAugmentor supported types
        self.families = {
            'gaussian': {
                'intensities': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                'description': 'Additive Gaussian noise with scaling factor',
                'params': {}
            },
            'dropout': {
                'intensities': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                'description': 'Random channel dropout with percentage scaling (0-100)',
                'params': {}
            },
            'eog': {
                'intensities': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                'description': 'Simulated EOG artifacts with scaling factor',
                'params': {}
            },
            'spike': {
                'intensities': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                'description': 'Transient spike artifacts with temporal coverage scaling (10% = 10% of time covered by spikes)',
                'params': {}
            }
        }
        
        # Seeding policy - simplified to use CLI seed_base directly
        self.seeding_policy = {
            'base_seed': 100,
            'description': 'Seed base from CLI argument, used directly for all corruptions'
        }
    
    def get_family_intensities(self, family: str) -> List[float]:
        """Get intensity grid for a specific corruption family."""
        if family not in self.families:
            raise ValueError(f"Unknown corruption family: {family}")
        return self.families[family]['intensities']
    
    def get_family_params(self, family: str) -> Dict[str, Any]:
        """Get parameters for a specific corruption family."""
        if family not in self.families:
            raise ValueError(f"Unknown corruption family: {family}")
        return self.families[family]['params']
    
    def get_all_families(self) -> List[str]:
        """Get list of all available corruption families."""
        return list(self.families.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert corruption plan to dictionary for serialization."""
        return {
            'families': self.families,
            'seeding_policy': self.seeding_policy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorruptionPlan':
        """Create corruption plan from dictionary."""
        plan = cls()
        plan.families = data.get('families', plan.families)
        plan.seeding_policy = data.get('seeding_policy', plan.seeding_policy)
        return plan
    
    def save(self, filepath: str):
        """Save corruption plan to file (JSON or YAML)."""
        filepath = Path(filepath)
        data = self.to_dict()
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CorruptionPlan':
        """Load corruption plan from file."""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        return cls.from_dict(data)


def build_corruption_grid(corruption_plan: CorruptionPlan, 
                         families_override: Optional[List[str]] = None,
                         intensities_override: Optional[Dict[str, List[float]]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Build ordered list of (family, intensity, params) tuples for corruption evaluation.
    
    Parameters
    ----------
    corruption_plan : CorruptionPlan
        The corruption plan configuration.
    families_override : list of str, optional
        Override to use only specific families.
    intensities_override : dict, optional
        Override intensity grids for specific families.
    
    Returns
    -------
    list of tuples
        List of (family, intensity, params) tuples, always including clean point.
    """
    grid = []
    
    # Always add clean point first
    grid.append(('clean', 0.0, {}))
    
    # Determine which families to use
    families = families_override if families_override is not None else corruption_plan.get_all_families()
    
    # Build grid for each family
    for family in families:
        if family == 'clean':
            continue
            
        # Get intensities (use override if provided)
        if intensities_override and family in intensities_override:
            intensities = intensities_override[family]
        else:
            intensities = corruption_plan.get_family_intensities(family)
        
        # Get parameters for this family
        params = corruption_plan.get_family_params(family)
        
        # Add each intensity level
        for intensity in intensities:
            grid.append((family, intensity, params))
    
    return grid


def derive_corruption_seed(seed_base: int, family: str, intensity: float, 
                          test_fold_id: int, subject_id: Any, 
                          target_session: Optional[str] = None) -> int:
   
    return seed_base


def apply_corruption(x: np.ndarray, family: str, intensity: float, 
                    seed: int, params: Dict[str, Any]) -> np.ndarray:
    """
    Apply specified corruption to input data.
    
    This function reuses the exact same corruption implementations
    currently used in augment/perturb codepaths to keep definitions aligned.
    
    Parameters
    ----------
    x : np.ndarray
        Input data batch (n_epochs, n_channels, n_times).
    family : str
        Corruption family name.
    intensity : float
        Corruption intensity level.
    seed : int
        Seed for deterministic corruption generation.
    params : dict
        Additional parameters for the corruption family.
    
    Returns
    -------
    np.ndarray
        Corrupted data with same shape as input.
    """
    # Set RNG seed for deterministic behavior
    np.random.seed(seed)
    
    # Make a copy to avoid mutating original
    x_corrupted = x.copy()
    
    if family == 'gaussian':
        # Additive Gaussian noise using existing EEGNoiseAugmentor
        from .noise import EEGNoiseAugmentor
        augmentor = EEGNoiseAugmentor(noise_type='gaussian', intensity=intensity, seed=seed)
        x_corrupted = augmentor.transform(x)
        
    elif family == 'dropout':
        # Random channel dropout using existing EEGNoiseAugmentor
        from .noise import EEGNoiseAugmentor
        augmentor = EEGNoiseAugmentor(noise_type='dropout', intensity=intensity, seed=seed)
        x_corrupted = augmentor.transform(x)
        
    elif family == 'eog':
        # Simulated EOG artifacts using existing EEGNoiseAugmentor
        from .noise import EEGNoiseAugmentor
        augmentor = EEGNoiseAugmentor(noise_type='eog', intensity=intensity, seed=seed)
        x_corrupted = augmentor.transform(x)
        
    elif family == 'spike':
        # Transient spike artifacts using existing EEGNoiseAugmentor
        from .noise import EEGNoiseAugmentor
        augmentor = EEGNoiseAugmentor(noise_type='spike', intensity=intensity, seed=seed)
        x_corrupted = augmentor.transform(x)
            
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    
    return x_corrupted


def compute_aurc(intensities: List[float], metrics: List[float]) -> float:
    """
    Compute Area Under Robustness Curve (AURC) using trapezoidal rule.
    
    Parameters
    ----------
    intensities : list of float
        Intensity values (must be strictly increasing).
    metrics : list of float
        Metric values corresponding to intensities.
    
    Returns
    -------
    float
        Area under the curve.
    """
    if len(intensities) != len(metrics):
        raise ValueError("Intensities and metrics must have the same length")
    
    if len(intensities) < 2:
        return 0.0
    
    # Verify intensities are strictly increasing
    for i in range(1, len(intensities)):
        if intensities[i] <= intensities[i-1]:
            raise ValueError("Intensities must be strictly increasing")
    
    # Compute AURC using trapezoidal rule
    aurc = 0.0
    for i in range(1, len(intensities)):
        width = intensities[i] - intensities[i-1]
        height = (metrics[i] + metrics[i-1]) / 2
        aurc += width * height
    
    return aurc


def compute_relative_drop(clean_metric: float, corrupted_metric: float) -> float:
    """
    Compute relative performance drop: (clean - corrupted) / clean.
    
    Parameters
    ----------
    clean_metric : float
        Clean performance metric.
    corrupted_metric : float
        Corrupted performance metric.
    
    Returns
    -------
    float
        Relative drop. Returns nan if clean_metric is zero.
    """
    if clean_metric == 0:
        logger.warning("Clean metric is zero, cannot compute relative drop")
        return np.nan
    
    relative_drop = (clean_metric - corrupted_metric) / clean_metric
    return relative_drop


def log_corruption_run_metadata(run_dir: str, corruption_plan: CorruptionPlan, 
                               seed: int, grid: List[Tuple[str, float, Dict[str, Any]]]):
    """
    Write human-readable metadata about corruption run for reproducibility.
    
    Parameters
    ----------
    run_dir : str
        Directory to save metadata files.
    corruption_plan : CorruptionPlan
        The corruption plan used.
    seed_base : int
        Base seed for the experiment.
    grid : list
        Corruption grid used for evaluation.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save corruption plan
    plan_file = run_dir / "corruption_plan.json"
    corruption_plan.save(str(plan_file))
    
    # Save grid and seed mapping
    grid_data = []
    for family, intensity, params in grid:
        if family != 'clean':
            # Compute seed for each grid point (example with dummy fold/subject)
            derived = derive_corruption_seed(seed, family, intensity, 0, 0)
            grid_data.append({
                'family': family,
                'intensity': intensity,
                'params': params,
                'derived_seed': derived
            })
        else:
            grid_data.append({
                'family': family,
                'intensity': intensity,
                'params': params,
                'derived_seed': seed
            })
    
    grid_file = run_dir / "corruption_grid.json"
    with open(grid_file, 'w') as f:
        json.dump({
            'seed': seed,
            'grid': grid_data,
            'grid_size': len(grid)
        }, f, indent=2)
    
    # Save human-readable summary
    summary_file = run_dir / "corruption_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Corruption Run Metadata\n")
        f.write("======================\n\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total Grid Points: {len(grid)}\n\n")
        f.write("Corruption Families:\n")
        for family in corruption_plan.get_all_families():
            intensities = corruption_plan.get_family_intensities(family)
            f.write(f"  {family}: {intensities}\n")
        f.write(f"\nGrid Details:\n")
        for item in grid_data:
            f.write(f"  {item['family']} (intensity={item['intensity']}): seed={item['derived_seed']}\n")
    
    logger.info(f"Corruption run metadata saved to {run_dir}")
    logger.info(f"  - Plan: {plan_file}")
    logger.info(f"  - Grid: {grid_file}")
    logger.info(f"  - Summary: {summary_file}")


# Default corruption plan instance
DEFAULT_CORRUPTION_PLAN = CorruptionPlan()
