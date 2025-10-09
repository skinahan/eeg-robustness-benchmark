#!/usr/bin/env python3
"""
Comprehensive Experiment Automation System

This script provides a complete automation solution for EEG experiments:
1. Loads experimental configurations from YAML
2. Aggregates existing results from all datasets
3. Identifies missing experimental combinations
4. Generates shell scripts with missing experiment commands

Usage:
    python experiment_automation.py [--config CONFIG_FILE] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import yaml
import pandas as pd
import argparse
import itertools
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.experiment_utils import collect_all_results_unified
from utils import get_noise_intensities
from tqdm import tqdm

class ExperimentAutomation:
    """Main class for experiment automation."""
    
    def __init__(self, config_file: str = "experiment_config.yaml", preaggregated_results_file: str = None):
        """Initialize the automation system with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.existing_results = None
        self.missing_experiments = []
        self.preaggregated_results_file = preaggregated_results_file
        # Performance optimization caches
        self._cached_noise_intensities = None
        self._cached_existing_signatures = None
        
    def _invalidate_caches(self):
        """Invalidate all performance caches."""
        self._cached_existing_signatures = None
        if hasattr(self, 'expected_test_perturb_results'):
            delattr(self, 'expected_test_perturb_results')
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"[ERROR] Configuration file not found: {self.config_file}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing YAML configuration: {e}")
            sys.exit(1)
    
    def load_preaggregated_results(self) -> pd.DataFrame:
        """Load pre-aggregated results from CSV file."""
        print("\n" + "="*60)
        print("LOADING PRE-AGGREGATED RESULTS")
        print("="*60)
        
        try:
            self.existing_results = pd.read_csv(self.preaggregated_results_file)
            
            # Invalidate caches when new results are loaded
            self._invalidate_caches()
            
            print(f"[OK] Loaded {len(self.existing_results)} result rows from {self.preaggregated_results_file}")
            print(f"[INFO] Results summary:")
            print(f"   - Datasets: {self.existing_results['dataset'].unique()}")
            print(f"   - Models: {self.existing_results['model'].unique()}")
            print(f"   - Eval modes: {self.existing_results['eval_mode'].unique()}")
            print(f"   - Modes: {self.existing_results['mode'].unique()}")
            if 'seed' in self.existing_results.columns:
                print(f"   - Seeds: {sorted(self.existing_results['seed'].unique())}")
            return self.existing_results
        except FileNotFoundError:
            print(f"[ERROR] Pre-aggregated results file not found: {self.preaggregated_results_file}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Error loading pre-aggregated results: {e}")
            sys.exit(1)
    
    def aggregate_existing_results(self) -> pd.DataFrame:
        """Aggregate all existing results from the results directory."""
        print("\n" + "="*60)
        print("AGGREGATING EXISTING RESULTS")
        print("="*60)
        
        # Use the updated aggregation function
        self.existing_results = collect_all_results_unified()
        
        # Invalidate caches when new results are loaded
        self._invalidate_caches()
        
        if self.existing_results is not None:
            print(f"[OK] Aggregated {len(self.existing_results)} result rows")
            print(f"[INFO] Results summary:")
            print(f"   - Datasets: {self.existing_results['dataset'].unique()}")
            print(f"   - Models: {self.existing_results['model'].unique()}")
            print(f"   - Eval modes: {self.existing_results['eval_mode'].unique()}")
            print(f"   - Modes: {self.existing_results['mode'].unique()}")
            if 'seed' in self.existing_results.columns:
                print(f"   - Seeds: {sorted(self.existing_results['seed'].unique())}")
        else:
            print("[WARNING] No existing results found")
            self.existing_results = pd.DataFrame()
        
        return self.existing_results
    
    def generate_expected_experiments(self) -> List[Dict[str, Any]]:
        """
        Generate all expected test_perturb results based on configuration, 
        then identify which multirun jobs are needed to generate missing results.
        Optimized version using itertools.product for better performance.
        """
        print("\n" + "="*60)
        print("GENERATING EXPECTED TEST_PERTURB EXPERIMENTS")
        print("="*60)
        
        # Get base configurations
        datasets = self.config['datasets']
        models = self.config['models']
        eval_modes = self.config['eval_modes']
        seeds = self.config['seeds']
        noise_types = self.config['noise_types']
        
        # Note: We'll compute noise intensities dynamically per dataset and noise type
        # This ensures consistency with unified_experiment_runner.py
        
        # Use itertools.product for efficient cartesian product generation
        expected_test_perturb_results = []
        
        # Pre-extract values for faster iteration
        dataset_items = list(datasets.items())
        model_names = [model['name'] for model in models]
        tune_flags = [False, True]
        
        print(f"[INFO] Generating combinations from:")
        print(f"   - Datasets: {len(dataset_items)}")
        print(f"   - Models: {len(model_names)}")
        print(f"   - Eval modes: {len(eval_modes)}")
        print(f"   - Seeds: {len(seeds)}")
        print(f"   - Noise types: {len(noise_types)}")
        print(f"   - Intensities: Dynamic (based on saturation points)")
        print(f"   - Tune flags: {len(tune_flags)}")
        
        # Generate combinations using nested loops since intensities are now dynamic
        # We can't use itertools.product with dynamic intensities
        
        print("[INFO] Processing combinations...")
        total_combinations = 0
        
        # Calculate total combinations for progress tracking
        for dataset_name, dataset_config in dataset_items:
            for noise_type in noise_types:
                intensities = get_noise_intensities(dataset_name, noise_type, num_steps=20)
                total_combinations += len(model_names) * len(eval_modes) * len(seeds) * len(intensities) * len(tune_flags)
        
        print(f"[INFO] Total combinations to process: {total_combinations}")
        
        # Process combinations with dynamic intensities
        processed = 0
        for dataset_name, dataset_config in dataset_items:
            for model_name in model_names:
                for eval_mode in eval_modes:
                    for seed in seeds:
                        for noise_type in noise_types:
                            # Get dynamic intensities for this dataset and noise type
                            intensities = get_noise_intensities(dataset_name, noise_type, num_steps=20)
                            
                            for intensity in intensities:
                                for tune_flag in tune_flags:
                                    processed += 1
                                    if processed % 1000 == 0:
                                        print(f"[INFO] Processed {processed}/{total_combinations} combinations...")
                                    
                                    subjects = dataset_config['subjects']
                                    
                                    if eval_mode == 'CrossSession' or eval_mode == 'WithinSession':
                                        # For CrossSession and WithinSession, create separate entries for each subject
                                        for subject in subjects:
                                            experiment = {
                                                'dataset': dataset_name,
                                                'paradigm': dataset_config['paradigm'],
                                                'subject': subject,
                                                'model': model_name,
                                                'eval_mode': eval_mode,
                                                'mode': 'test_perturb',
                                                'seed': seed,
                                                'noise_type': noise_type,
                                                'intensity': intensity,
                                                'tune': tune_flag
                                            }
                                            expected_test_perturb_results.append(experiment)
                                    else:
                                        # For Cross-Subject, results are aggregated across subjects
                                        experiment = {
                                            'dataset': dataset_name,
                                            'paradigm': dataset_config['paradigm'],
                                            'subjects': subjects,  # All subjects together
                                            'model': model_name,
                                            'eval_mode': eval_mode,
                                            'mode': 'test_perturb',
                                            'seed': seed,
                                            'noise_type': noise_type,
                                            'intensity': intensity,
                                            'tune': tune_flag
                                        }
                                        expected_test_perturb_results.append(experiment)
        
        print(f"[OK] Generated {len(expected_test_perturb_results)} expected test_perturb results")
        
        # Store for use in identify_missing_experiments
        self.expected_test_perturb_results = expected_test_perturb_results
        
        # For now, return empty list - multirun jobs will be determined in identify_missing_experiments
        return []
    
    def identify_missing_experiments(self) -> List[Dict[str, Any]]:
        """
        Identify missing test_perturb results and map them to required multirun jobs.
        """
        print("\n" + "="*60)
        print("IDENTIFYING MISSING TEST_PERTURB RESULTS")
        print("="*60)
        
        # Generate expected test_perturb results (cached if already generated)
        if not hasattr(self, 'expected_test_perturb_results'):
            self.generate_expected_experiments()  # This populates self.expected_test_perturb_results
        else:
            print("[INFO] Using cached expected test_perturb results")
        
        # Initialize missing_combinations for diagnostics
        missing_combinations = {}
        
        if self.existing_results is None or self.existing_results.empty:
            print("[WARNING] No existing results found, all test_perturb results are missing")
            missing_test_perturb_results = self.expected_test_perturb_results
            
            # Populate missing_combinations for diagnostics when no existing results
            for expected_result in self.expected_test_perturb_results:
                combo_key = (expected_result['dataset'], expected_result['model'], 
                           expected_result['eval_mode'], expected_result['seed'],
                           expected_result['noise_type'])
                if combo_key not in missing_combinations:
                    missing_combinations[combo_key] = []
                missing_combinations[combo_key].append(expected_result.get('subject', 'no_subject'))
        else:
            # Convert expected results to DataFrame for vectorized operations
            print("[INFO] Converting expected results to DataFrame for efficient comparison...")
            expected_df = pd.DataFrame(self.expected_test_perturb_results)
            
            # Prepare existing results for comparison
            existing_df = self.existing_results.copy()
            
            # Normalize eval_mode column in existing results for matching
            if 'eval_mode' in existing_df.columns:
                existing_df['eval_mode_normalized'] = existing_df['eval_mode'].str.replace('Evaluation', '', regex=False)
            else:
                existing_df['eval_mode_normalized'] = 'Unknown'
                
            # Normalize mode column to handle tuning suffixes
            if 'mode' in existing_df.columns:
                existing_df['mode_normalized'] = existing_df['mode'].str.replace('_tune', '', regex=False)
            else:
                existing_df['mode_normalized'] = 'Unknown'
            
            # Create a set of existing result signatures for fast lookup (with caching)
            if self._cached_existing_signatures is None or len(self._cached_existing_signatures) != len(existing_df):
                print("[INFO] Creating lookup set for existing results...")
                existing_signatures = set()
                
                for _, row in existing_df.iterrows():
                    # Create signature based on the matching criteria
                    signature_parts = [
                        row.get('dataset', ''),
                        row.get('model', ''),
                        row.get('eval_mode_normalized', ''),
                        row.get('seed', ''),
                        row.get('noise_type', ''),
                        str(row.get('intensity', '')),  # Convert to string for consistent comparison
                        row.get('mode_normalized', '')
                    ]
                    
                    # Add subject if present (for CrossSession/WithinSession)
                    if 'subject' in row and pd.notna(row['subject']):
                        signature_parts.append(str(row['subject']))
                    else:
                        signature_parts.append('no_subject')
                        
                    signature = '|'.join(str(part) for part in signature_parts)
                    existing_signatures.add(signature)
                
                # Cache the signatures for future use
                self._cached_existing_signatures = existing_signatures
            else:
                print("[INFO] Using cached existing result signatures")
                existing_signatures = self._cached_existing_signatures
            
            # Find missing results using set operations
            print("[INFO] Identifying missing experiments using vectorized comparison...")
            missing_test_perturb_results = []
            
            # Pre-compute intensity mapping for better performance
            print("[INFO] Pre-computing intensity mappings...")
            existing_intensities = sorted(existing_df['intensity'].unique())
            intensity_mapping = {}
            
            for expected_result in self.expected_test_perturb_results:
                intensity = expected_result['intensity']
                if intensity not in intensity_mapping:
                    # Find matching intensity in existing data with tolerance
                    matching_intensity = None
                    for existing_intensity in existing_intensities:
                        if abs(intensity - existing_intensity) < 1e-10:
                            matching_intensity = existing_intensity
                            break
                    intensity_mapping[intensity] = matching_intensity
            
            print("[INFO] Checking missing experiments...")
            for expected_result in tqdm(self.expected_test_perturb_results, desc="Checking missing experiments"):
                # Create signature for expected result with floating-point tolerance for intensity
                intensity = expected_result['intensity']
                matching_intensity = intensity_mapping[intensity]
                
                if matching_intensity is None:
                    # This intensity doesn't exist in the data at all
                    signature_parts = [
                        expected_result['dataset'],
                        expected_result['model'],
                        expected_result['eval_mode'],
                        str(expected_result['seed']),
                        expected_result['noise_type'],
                        str(intensity),
                        'test_perturb'  # Mode is always test_perturb for expected results
                    ]
                    
                    # Add subject if present
                    if 'subject' in expected_result:
                        signature_parts.append(str(expected_result['subject']))
                    else:
                        signature_parts.append('no_subject')
                        
                    signature = '|'.join(str(part) for part in signature_parts)
                else:
                    # Use the matching intensity from existing data
                    signature_parts = [
                        expected_result['dataset'],
                        expected_result['model'],
                        expected_result['eval_mode'],
                        str(expected_result['seed']),
                        expected_result['noise_type'],
                        str(matching_intensity),
                        'test_perturb'  # Mode is always test_perturb for expected results
                    ]
                    
                    # Add subject if present
                    if 'subject' in expected_result:
                        signature_parts.append(str(expected_result['subject']))
                    else:
                        signature_parts.append('no_subject')
                        
                    signature = '|'.join(str(part) for part in signature_parts)
                
                # Check if this signature exists in our set
                if signature not in existing_signatures:
                    missing_test_perturb_results.append(expected_result)
                    
                    # Track missing combinations for diagnostics
                    combo_key = (expected_result['dataset'], expected_result['model'], 
                               expected_result['eval_mode'], expected_result['seed'],
                               expected_result['noise_type'])
                    if combo_key not in missing_combinations:
                        missing_combinations[combo_key] = []
                    missing_combinations[combo_key].append(expected_result.get('subject', 'no_subject'))
        
        print(f"[OK] Found {len(missing_test_perturb_results)} missing test_perturb results out of {len(self.expected_test_perturb_results)} total expected")
        
        # Print detailed diagnostics of missing combinations
        if missing_combinations:
            print(f"\n[INFO] Missing combinations breakdown:")
            print(f"   Total missing combinations: {len(missing_combinations)}")
            
            # Group by eval_mode and noise_type for better understanding
            missing_by_eval_noise = {}
            for combo_key, subjects in missing_combinations.items():
                dataset, model, eval_mode, seed, noise_type = combo_key
                key = (eval_mode, noise_type)
                if key not in missing_by_eval_noise:
                    missing_by_eval_noise[key] = 0
                missing_by_eval_noise[key] += len(subjects) * 20  # 20 intensities per combination
            
            print(f"   Missing by eval_mode and noise_type:")
            for (eval_mode, noise_type), count in sorted(missing_by_eval_noise.items()):
                print(f"     - {eval_mode} + {noise_type}: {count} missing results")
            
            # Show first few missing combinations as examples
            print(f"\n[INFO] Example missing combinations (first 10):")
            for i, (combo_key, subjects) in enumerate(list(missing_combinations.items())[:10]):
                dataset, model, eval_mode, seed, noise_type = combo_key
                print(f"     {i+1}. {dataset} | {model} | {eval_mode} | seed={seed} | {noise_type} | subjects={subjects[:3]}{'...' if len(subjects) > 3 else ''}")
        else:
            print("[INFO] No missing combinations found - all expected results are present!")
        
        # Now map missing results to required multirun jobs
        print("\n" + "="*60)
        print("MAPPING TO REQUIRED MULTIRUN JOBS")
        print("="*60)
        
        # Define the limited models that will be used in multirun mode
        limited_models = ["eegnet", "reegnet", "cnn_ncp"]
        
        # Get seeds from config
        seeds = self.config['seeds']
        
        required_multirun_jobs = set()
        
        for missing_result in missing_test_perturb_results:
            if missing_result['eval_mode'] == 'CrossSession' or missing_result['eval_mode'] == 'WithinSession':
                # For CrossSession, each subject needs its own multirun job for each model and seed
                for model in limited_models:
                    for seed in seeds:
                        job_key = (
                            missing_result['dataset'],
                            missing_result['eval_mode'], 
                            missing_result['subject'],
                            missing_result['tune'],
                            model,
                            seed
                        )
                        required_multirun_jobs.add(job_key)
            else:
                # For Cross-Subject, all subjects are processed together for each model and seed
                subjects_tuple = tuple(missing_result['subjects'])
                for model in limited_models:
                    for seed in seeds:
                        job_key = (
                            missing_result['dataset'],
                            missing_result['eval_mode'],
                            subjects_tuple,
                            missing_result['tune'],
                            model,
                            seed
                        )
                        required_multirun_jobs.add(job_key)
        
        # Convert to list of multirun job dictionaries
        missing_experiments = []
        for job_key in required_multirun_jobs:
            dataset, eval_mode, subjects_or_subject, tune, model, seed = job_key
            
            if eval_mode == 'CrossSession' or eval_mode == 'WithinSession':
                # Single subject for CrossSession
                subjects = [subjects_or_subject]
            else:
                # Multiple subjects for Cross-Subject
                subjects = list(subjects_or_subject)
            
            multirun_job = {
                'dataset': dataset,
                'eval_mode': eval_mode,
                'subjects': subjects,
                'tune': tune,
                'model': model,
                'seed': seed,
                'mode': 'multirun',  # This will generate test_perturb results
                'paradigm': next(config['paradigm'] for name, config in self.config['datasets'].items() if name == dataset)
            }
            missing_experiments.append(multirun_job)
        
        self.missing_experiments = missing_experiments
        print(f"[OK] Mapped to {len(missing_experiments)} required multirun jobs")
        
        # Print summary of missing multirun jobs
        if missing_experiments:
            print("\n[INFO] Missing multirun jobs summary:")
            missing_df = pd.DataFrame(missing_experiments)
            
            print("   By dataset:")
            for dataset, count in missing_df['dataset'].value_counts().items():
                print(f"     - {dataset}: {count} multirun jobs")
            
            print("   By model:")
            for model, count in missing_df['model'].value_counts().items():
                print(f"     - {model}: {count} multirun jobs")
            
            print("   By eval_mode:")
            for eval_mode, count in missing_df['eval_mode'].value_counts().items():
                print(f"     - {eval_mode}: {count} multirun jobs")
            
            print("   By tune flag:")
            for tune, count in missing_df['tune'].value_counts().items():
                print(f"     - {'tuned' if tune else 'not tuned'}: {count} multirun jobs")
            
            print("   By seed:")
            for seed, count in missing_df['seed'].value_counts().items():
                print(f"     - seed {seed}: {count} multirun jobs")
        
        return missing_experiments
    
    def generate_shell_script(self, output_dir: str = None) -> str:
        """Generate shell script with missing multirun sbatch commands."""
        print("\n" + "="*60)
        print("GENERATING SBATCH SHELL SCRIPT")
        print("="*60)
        
        if output_dir is None:
            output_dir = self.config['output']['script_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        script_file = os.path.join(output_dir, self.config['output']['missing_script_file'])
        
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated multirun sbatch automation script\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total missing multirun jobs: {len(self.missing_experiments)}\n\n")
            
            f.write("set -e  # Exit on any error\n\n")
            
            f.write("echo \"Starting multirun experiment automation...\"\n")
            f.write("echo \"Total multirun jobs to submit: {}\"\n\n".format(len(self.missing_experiments)))
            
            for i, exp in enumerate(self.missing_experiments, 1):
                # Build sbatch command arguments
                if exp['eval_mode'] == 'CrossSession' or exp['eval_mode'] == 'WithinSession':
                    # For CrossSession, single subject
                    subjects_str = str(exp['subjects'][0])
                else:
                    # For CrossSubject, space-separated subjects
                    subjects_str = " ".join(map(str, exp['subjects']))
                
                # Handle tuning flag
                tune_flag = "true" if exp['tune'] else "false"
                
                # Get model name and seed
                model = exp['model']
                seed = exp['seed']
                
                # Generate sbatch command with appropriate time limits
                # UPDATED: Since multirun now processes one seed at a time, reduce time limits by factor of 5
                # Base estimates: CrossSession without tuning ~3 hours / 5 = ~36 minutes
                # WithinSession takes ~5x longer than CrossSession
                # Tuning adds significant overhead (~20x for CrossSession)
                # SSVEP (Lee2019) is ~3x slower than Motor Imagery due to:
                #   - 4 classes vs 2 classes (~2x overhead)
                #   - Longer time windows (~1.5x overhead)
                # CLUSTER LIMIT: Maximum time is 7 days
                
                if exp['dataset'] == 'Lee2019_SSVEP':
                    # SSVEP-specific timeouts (reduced by factor of 5)
                    if exp['eval_mode'] == 'CrossSession':
                        if exp['tune']:
                            # CrossSession with tuning: ~2.5 days / 5 = ~12 hours
                            slurm_args = "--time=0-12:00:00 --mem=12G"
                        else:
                            # CrossSession without tuning: ~3 hours * 3 / 5 = ~1.8 hours
                            slurm_args = "--time=0-02:00:00 --mem=12G"
                            
                    elif exp['eval_mode'] == 'WithinSession':
                        if exp['tune']:
                            # WithinSession with tuning: ~12.5 days / 5 = ~2.5 days
                            slurm_args = "--time=2-12:00:00 --mem=12G"
                        else:
                            # WithinSession without tuning: ~15 hours * 3 / 5 = ~9 hours
                            slurm_args = "--time=0-10:00:00 --mem=12G"
                            
                    else:
                        # Default time limit for other modes (CrossSubject, etc.)
                        slurm_args = "--time=1-08:00:00 --mem=12G"
                        
                else:
                    # Motor Imagery timeouts (reduced by factor of 5)
                    if exp['eval_mode'] == 'CrossSession':
                        if exp['tune']:
                            # CrossSession with tuning: ~2.5 days / 5 = ~12 hours
                            slurm_args = "--time=1-12:00:00 --mem=12G"
                        else:
                            # CrossSession without tuning: ~3 hours / 5 = ~36 minutes
                            slurm_args = "--time=0-01:00:00 --mem=12G"
                            
                    elif exp['eval_mode'] == 'WithinSession':
                        if exp['tune']:
                            # WithinSession with tuning: ~12.5 days / 5 = ~2.5 days
                            slurm_args = "--time=3-00:00:00 --mem=12G"
                        else:
                            # WithinSession without tuning: ~15 hours / 5 = ~3 hours
                            slurm_args = "--time=0-04:00:00 --mem=12G"
                            
                    else:
                        # Default time limit for other modes (CrossSubject, etc.)
                        slurm_args = "--time=1-08:00:00 --mem=12G"
                
                # Format: sbatch {slurm_args} unified_eval_script.sh {subject} {dataset} {eval_mode} {tune_flag} {model} {seed}
                command = f"sbatch {slurm_args} unified_eval_script.sh {subjects_str} {exp['dataset']} {exp['eval_mode']} {tune_flag} {model} {seed}"
                
                # Write sbatch command
                f.write(f"# Multirun Job {i}/{len(self.missing_experiments)}\n")
                f.write(f"# Dataset: {exp['dataset']} | Model: {model} | Eval: {exp['eval_mode']} | Subjects: {exp['subjects']} | Seed: {seed}")
                if exp['tune']:
                    f.write(" | TUNED")
                f.write("\n")
                f.write(f"# Timeout: {slurm_args}\n")
                f.write(f"# This multirun will generate test_perturb results for model: {model}\n")
                f.write(f"# This multirun will generate test_perturb results for seed: {seed}\n")
                f.write(f"# This multirun will generate test_perturb results for all noise types and intensities\n")
                f.write(f"echo \"Submitting multirun job {i}/{len(self.missing_experiments)}...\"\n")
                f.write(f"{command}\n")
                f.write("if [ $? -eq 0 ]; then\n")
                f.write(f"    echo \"[SUCCESS] Multirun job {i} submitted successfully\"\n")
                f.write("else\n")
                f.write(f"    echo \"[ERROR] Multirun job {i} submission failed\"\n")
                f.write("    exit 1\n")
                f.write("fi\n")
                f.write("sleep 1  # Brief pause between submissions\n\n")
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        print(f"[OK] Generated sbatch shell script: {script_file}")
        print(f"[INFO] Script contains {len(self.missing_experiments)} multirun sbatch commands")
        print(f"[INFO] Each sbatch command format: sbatch unified_eval_script.sh <subjects> <dataset> <eval_mode> <tune_flag>")
        
        return script_file
    
    def generate_summary_report(self, output_dir: str = None) -> str:
        """Generate a summary report of missing experiments."""
        if output_dir is None:
            output_dir = self.config['output']['script_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, "missing_experiments_report.csv")
        
        if self.missing_experiments:
            missing_df = pd.DataFrame(self.missing_experiments)
            missing_df.to_csv(report_file, index=False)
            print(f"[OK] Generated summary report: {report_file}")
        else:
            print("[OK] No missing experiments to report")
        
        return report_file
    
    def run_full_automation(self, output_dir: str = None) -> Tuple[str, str]:
        """Run the complete automation process."""
        print("[START] Starting Experiment Automation")
        print("="*60)
        
        # Step 1: Load existing results (either aggregate or load pre-aggregated)
        if self.preaggregated_results_file:
            self.load_preaggregated_results()
        else:
            self.aggregate_existing_results()
        
        # Step 2: Identify missing experiments
        self.identify_missing_experiments()
        
        # Step 3: Generate shell script
        script_file = self.generate_shell_script(output_dir)
        
        # Step 4: Generate summary report
        report_file = self.generate_summary_report(output_dir)
        
        print("\n" + "="*60)
        print("AUTOMATION COMPLETE")
        print("="*60)
        print(f"[INFO] Output directory: {output_dir or self.config['output']['script_dir']}")
        print(f"[INFO] Shell script: {script_file}")
        print(f"[INFO] Summary report: {report_file}")
        print(f"[INFO] Missing experiments: {len(self.missing_experiments)}")
        
        if len(self.missing_experiments) > 0:
            print("\n[INFO] Next steps:")
            print("1. Review the generated shell script")
            print("2. Run: chmod +x <script_file>")
            print("3. Execute: ./<script_file>")
            print("4. Monitor progress and handle any failures")
        
        return script_file, report_file


def main():
    """Main entry point for the automation script."""
    parser = argparse.ArgumentParser(description="EEG Experiment Automation System")
    parser.add_argument("--config", type=str, default="experiment_config.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for generated scripts and reports")
    parser.add_argument("--preaggregated-results", type=str, default=None,
                       help="Path to pre-aggregated results CSV file (skips aggregation step)")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing results, don't generate missing experiments")
    parser.add_argument("--missing-only", action="store_true",
                       help="Only identify missing experiments, don't aggregate")
    
    args = parser.parse_args()
    
    # Initialize automation system
    automation = ExperimentAutomation(args.config, args.preaggregated_results)
    
    if args.aggregate_only:
        # Only aggregate results (ignore pre-aggregated file for this mode)
        if args.preaggregated_results:
            print("[WARNING] --preaggregated-results ignored in --aggregate-only mode")
        automation.aggregate_existing_results()
    elif args.missing_only:
        # Only identify missing experiments
        if args.preaggregated_results:
            automation.load_preaggregated_results()
        else:
            automation.aggregate_existing_results()
        automation.identify_missing_experiments()
        automation.generate_summary_report(args.output_dir)
    else:
        # Full automation
        automation.run_full_automation(args.output_dir)


if __name__ == "__main__":
    main()
