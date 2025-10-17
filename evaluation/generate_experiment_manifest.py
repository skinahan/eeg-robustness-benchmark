#!/usr/bin/env python3
"""
Experiment Manifest Generator for Unified Experiment Runner

This script generates a comprehensive manifest file containing all experiment
configurations that need to be run on the compute cluster. The manifest can
then be used by an optimized sbatch script to efficiently submit jobs.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
import itertools

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import MODEL_REGISTRY


class ExperimentManifestGenerator:
    """Generates comprehensive experiment manifests for cluster submission."""
    
    def __init__(self):
        self.models = list(MODEL_REGISTRY.keys())
        self.datasets = ["BNCI2014_001"]
        self.eval_modes = ["WithinSession", "CrossSession"]
        self.modes = ["baseline", "tune", "augment", "perturb", "augment_notune", "perturb_notune", "test_perturb"]
        self.noise_types = ["dropout", "gaussian", "eog", "spike"]
        self.notune_modes = ["baseline", "augment_notune", "perturb_notune", "test_perturb"]
        self.intensities = {
            "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
            "gaussian": [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3],
            "eog": [0.1, 0.2, 0.3, 0.4, 0.5],
            "spike": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        self.seeds = [100, 200, 300, 400, 500]  # Multiple seeds for robustness
        self.subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def generate_baseline_experiments(self) -> List[Dict[str, Any]]:
        """Generate baseline experiments (no noise, no tuning)."""
        experiments = []
        for seed in self.seeds:
            for model in self.models:
                for dataset in self.datasets:
                    for eval_mode in self.eval_modes:
                        for subject in self.subjects:
                            experiment = {
                                "model": model,
                                "dataset": dataset,
                                "subjects": subject,
                                "mode": "baseline",
                                "eval_mode": eval_mode,
                                "seed": seed,
                                "noise_type": None,
                                "intensity": None,
                                "tune": False,
                                "overwrite": True,
                                "estimated_runtime": "00:30:00",
                                "memory": "8G",
                                "cpus": 1
                            }
                            experiments.append(experiment)
    
        return experiments
    
    def generate_tuning_experiments(self) -> List[Dict[str, Any]]:
        """Generate hyperparameter tuning experiments."""
        experiments = []
        
        for model in self.models:
            for dataset in self.datasets:
                for eval_mode in self.eval_modes:
                    for seed in self.seeds:
                        for subject in self.subjects:
                            experiment = {
                                "model": model,
                                "dataset": dataset,
                                "subjects": subject,
                                "mode": "tune",
                                "eval_mode": eval_mode,
                                "seed": seed,
                                "noise_type": None,
                                "intensity": None,
                                "tune": True,
                                "overwrite": True,
                                "estimated_runtime": "01:00:00",
                                "memory": "8G",
                                "cpus": 1
                            }
                            experiments.append(experiment)
        
        return experiments
    
    def generate_noise_experiments(self) -> List[Dict[str, Any]]:
        """Generate noise-aware experiments."""
        experiments = []
        
        for model in self.models:
            for dataset in self.datasets:
                for eval_mode in self.eval_modes:
                    for noise_type in self.noise_types:
                        for intensity in self.intensities[noise_type]:
                            for seed in self.seeds:
                                for subject in self.subjects:
                                    # Augment mode (with tuning)
                                    experiment_augment = {
                                        "model": model,
                                        "dataset": dataset,
                                        "subjects": subject,
                                        "mode": "augment",
                                        "eval_mode": eval_mode,
                                        "seed": seed,
                                        "noise_type": noise_type,
                                        "intensity": intensity,
                                        "tune": True,
                                        "overwrite": True,
                                        "estimated_runtime": "06:00:00",
                                        "memory": "8G",
                                        "cpus": 1
                                    }
                                    experiments.append(experiment_augment)
                                    
                                    # Perturb mode (with tuning)
                                    experiment_perturb = {
                                        "model": model,
                                        "dataset": dataset,
                                        "subjects": subject,
                                        "mode": "perturb",
                                        "eval_mode": eval_mode,
                                        "seed": seed,
                                        "noise_type": noise_type,
                                        "intensity": intensity,
                                        "tune": True,
                                        "overwrite": True,
                                        "estimated_runtime": "06:00:00",
                                        "memory": "8G",
                                        "cpus": 1
                                    }
                                    experiments.append(experiment_perturb)
                                    
                                    # Augment mode (without tuning)
                                    experiment_augment_notune = {
                                        "model": model,
                                        "dataset": dataset,
                                        "subjects": subject,
                                        "mode": "augment_notune",
                                        "eval_mode": eval_mode,
                                        "seed": seed,
                                        "noise_type": noise_type,
                                        "intensity": intensity,
                                        "tune": False,
                                        "overwrite": True,
                                        "estimated_runtime": "01:00:00",
                                        "memory": "8G",
                                        "cpus": 1
                                    }
                                    experiments.append(experiment_augment_notune)
                                    
                                    # Perturb mode (without tuning)
                                    experiment_perturb_notune = {
                                        "model": model,
                                        "dataset": dataset,
                                        "subjects": subject,
                                        "mode": "perturb_notune",
                                        "eval_mode": eval_mode,
                                        "seed": seed,
                                        "noise_type": noise_type,
                                        "intensity": intensity,
                                        "tune": False,
                                        "overwrite": True,
                                        "estimated_runtime": "01:00:00",
                                        "memory": "8G",
                                        "cpus": 1
                                    }
                                    experiments.append(experiment_augment_notune)
        
        return experiments
    
    def generate_test_perturb_experiments(self) -> List[Dict[str, Any]]:
        """Generate test perturbation experiments."""
        experiments = []
        
        for model in self.models:
            for dataset in self.datasets:
                for eval_mode in self.eval_modes:
                    for noise_type in self.noise_types:
                        for intensity in self.intensities[noise_type]:
                            for seed in self.seeds:
                                for subject in self.subjects:
                                    experiment = {
                                        "model": model,
                                        "dataset": dataset,
                                        "subjects": subject,
                                        "mode": "test_perturb",
                                        "eval_mode": eval_mode,
                                        "seed": seed,
                                        "noise_type": noise_type,
                                        "intensity": intensity,
                                        "tune": True,  # Recommended for test_perturb
                                        "overwrite": True,
                                        "estimated_runtime": "04:00:00",
                                        "memory": "8G",
                                        "cpus": 1
                                    }
                                    experiments.append(experiment)
        
        return experiments
    
    def generate_manifest(self, include_all: bool = True) -> List[Dict[str, Any]]:
        """Generate complete experiment manifest."""
        print("Generating experiment manifest...")
        
        all_experiments = []
        
        if include_all:
            # print("  Adding baseline experiments...")
            # all_experiments.extend(self.generate_baseline_experiments())
            
            # print("  Adding tuning experiments...")
            # all_experiments.extend(self.generate_tuning_experiments())
            
            # print("  Adding noise experiments...")
            # all_experiments.extend(self.generate_noise_experiments())
            
            print("  Adding test perturbation experiments...")
            all_experiments.extend(self.generate_test_perturb_experiments())
        else:
            # Add only essential experiments
            print("  Adding baseline experiments only...")
            all_experiments.extend(self.generate_baseline_experiments())
        
        # Add experiment IDs and metadata
        for i, experiment in enumerate(all_experiments):
            experiment["experiment_id"] = f"exp_{i:06d}"
            experiment["created_at"] = datetime.now().isoformat()
            experiment["status"] = "pending"
        
        print(f"Generated {len(all_experiments)} experiments")
        return all_experiments
    
    def save_manifest(self, experiments: List[Dict[str, Any]], output_path: str):
        """Save manifest to JSON file."""
        manifest = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_experiments": len(experiments),
                "models": self.models,
                "datasets": self.datasets,
                "modes": self.modes,
                "noise_types": self.noise_types,
                "seeds": self.seeds
            },
            "experiments": experiments
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Manifest saved to: {output_path}")
    
    def generate_subset_manifest(self, subset_name: str, max_experiments: int = 100) -> List[Dict[str, Any]]:
        """Generate a subset manifest for testing purposes."""
        print(f"Generating subset manifest: {subset_name}")
        
        # Get all experiments
        all_experiments = self.generate_manifest(include_all=True)
        
        # Take experiments up to max_experiments
        subset_experiments = all_experiments[:max_experiments]
        
        # Update experiment IDs and ensure all required fields are present
        for i, experiment in enumerate(subset_experiments):
            experiment["experiment_id"] = f"{subset_name}_{i:04d}"
            if "created_at" not in experiment:
                experiment["created_at"] = datetime.now().isoformat()
            if "status" not in experiment:
                experiment["status"] = "pending"
        
        print(f"Generated subset with {len(subset_experiments)} experiments")
        return subset_experiments
    
    def analyze_manifest(self, experiments: List[Dict[str, Any]]):
        """Analyze the generated manifest for statistics."""
        print("\n=== Manifest Analysis ===")
        print(f"Total experiments: {len(experiments)}")
        
        # Count by mode
        mode_counts = {}
        for exp in experiments:
            mode = exp["mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print("\nExperiments by mode:")
        for mode, count in sorted(mode_counts.items()):
            print(f"  {mode}: {count}")
        
        # Count by model
        model_counts = {}
        for exp in experiments:
            model = exp["model"]
            model_counts[model] = model_counts.get(model, 0) + 1
        
        print("\nExperiments by model:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count}")
        
        # Count by noise type
        noise_counts = {}
        for exp in experiments:
            noise_type = exp.get("noise_type")
            if noise_type:
                noise_counts[noise_type] = noise_counts.get(noise_type, 0) + 1
        
        if noise_counts:
            print("\nExperiments by noise type:")
            for noise_type, count in sorted(noise_counts.items()):
                print(f"  {noise_type}: {count}")
        
        # Estimate total runtime
        total_runtime_hours = 0
        for exp in experiments:
            runtime_str = exp["estimated_runtime"]
            hours, minutes, seconds = map(int, runtime_str.split(":"))
            total_runtime_hours += hours + minutes/60 + seconds/3600
        
        print(f"\nEstimated total runtime: {total_runtime_hours:.1f} hours ({total_runtime_hours/24:.1f} days)")
        
        # Estimate resource requirements
        total_memory_gb = 0
        total_cpus = 0
        for exp in experiments:
            memory = exp["memory"]
            if memory.endswith("G"):
                total_memory_gb += int(memory[:-1])
            total_cpus += exp["cpus"]
        
        print(f"Total memory requirement: {total_memory_gb} GB")
        print(f"Total CPU requirement: {total_cpus} cores")


def main():
    """Main entry point for manifest generation."""
    parser = argparse.ArgumentParser(description="Generate experiment manifest for cluster submission")
    parser.add_argument("--output", type=str, default="experiment_manifest.json", 
                       help="Output path for manifest file")
    parser.add_argument("--subset", type=str, default=None,
                       help="Generate subset manifest (e.g., 'test', 'quick')")
    parser.add_argument("--max_experiments", type=int, default=100,
                       help="Maximum experiments for subset manifest")
    parser.add_argument("--baseline_only", action="store_true",
                       help="Generate only baseline experiments")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze the generated manifest")
    
    args = parser.parse_args()
    
    # Create generator
    generator = ExperimentManifestGenerator()
    
    if args.subset:
        # Generate subset manifest
        experiments = generator.generate_subset_manifest(args.subset, args.max_experiments)
        output_path = f"experiment_manifest_{args.subset}.json"
    else:
        # Generate full manifest
        experiments = generator.generate_manifest(include_all=not args.baseline_only)
        output_path = args.output
    
    # Save manifest
    generator.save_manifest(experiments, output_path)
    
    # Analyze if requested
    if args.analyze:
        generator.analyze_manifest(experiments)
    
    print(f"\nManifest generation complete!")
    print(f"Use this manifest with the sbatch script for efficient cluster submission.")


if __name__ == "__main__":
    main()
