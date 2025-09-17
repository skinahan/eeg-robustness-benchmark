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


class ExperimentAutomation:
    """Main class for experiment automation."""
    
    def __init__(self, config_file: str = "experiment_config.yaml"):
        """Initialize the automation system with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.existing_results = None
        self.missing_experiments = []
        
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
    
    def aggregate_existing_results(self) -> pd.DataFrame:
        """Aggregate all existing results from the results directory."""
        print("\n" + "="*60)
        print("AGGREGATING EXISTING RESULTS")
        print("="*60)
        
        # Use the updated aggregation function
        self.existing_results = collect_all_results_unified()
        
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
        """
        print("\n" + "="*60)
        print("GENERATING EXPECTED TEST_PERTURB EXPERIMENTS")
        print("="*60)
        
        expected_test_perturb_results = []
        
        # Get base configurations
        datasets = self.config['datasets']
        models = self.config['models']
        eval_modes = self.config['eval_modes']
        seeds = self.config['seeds']
        noise_types = self.config['noise_types']
        
        # Generate noise intensities (same as in unified_experiment_runner.py)
        noise_intensities = np.linspace(1.0, 50.0, 20)
        
        # Generate all expected test_perturb result combinations
        for dataset_name, dataset_config in datasets.items():
            for model_config in models:
                for eval_mode in eval_modes:
                    for seed in seeds:
                        for noise_type in noise_types:
                            for intensity in noise_intensities:
                                for tune_flag in [False, True]:
                                    subjects = dataset_config['subjects']
                                    
                                    if eval_mode == 'CrossSession':
                                        # For CrossSession, create separate entries for each subject
                                        for subject in subjects:
                                            experiment = {
                                                'dataset': dataset_name,
                                                'paradigm': dataset_config['paradigm'],
                                                'subject': subject,
                                                'model': model_config['name'],
                                                'eval_mode': eval_mode,
                                                'mode': 'test_perturb',
                                                'seed': seed,
                                                'noise_type': noise_type,
                                                'intensity': intensity,
                                                'tune': tune_flag
                                            }
                                            expected_test_perturb_results.append(experiment)
                                    else:
                                        # For WithinSession, results are aggregated across subjects
                                        experiment = {
                                            'dataset': dataset_name,
                                            'paradigm': dataset_config['paradigm'],
                                            'subjects': subjects,  # All subjects together
                                            'model': model_config['name'],
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
        
        # First generate expected test_perturb results
        self.generate_expected_experiments()  # This populates self.expected_test_perturb_results
        
        if self.existing_results is None or self.existing_results.empty:
            print("[WARNING] No existing results found, all test_perturb results are missing")
            missing_test_perturb_results = self.expected_test_perturb_results
        else:
            # Find missing test_perturb results
            missing_test_perturb_results = []
            
            for expected_result in self.expected_test_perturb_results:
                # Create filter for this specific test_perturb result
                filter_conditions = []
                
                # Basic filters
                filter_conditions.append(self.existing_results['dataset'] == expected_result['dataset'])
                filter_conditions.append(self.existing_results['model'] == expected_result['model'])
                filter_conditions.append(self.existing_results['eval_mode'].str.contains(expected_result['eval_mode'], na=False))
                filter_conditions.append(self.existing_results['seed'] == expected_result['seed'])
                filter_conditions.append(self.existing_results['noise_type'] == expected_result['noise_type'])
                filter_conditions.append(self.existing_results['intensity'] == expected_result['intensity'])
                
                # Handle mode filtering (account for tuning suffixes)
                mode_col = self.existing_results['mode']
                if expected_result['tune']:
                    # Look for tuned versions
                    mode_condition = (mode_col == "test_perturb_tune") | (mode_col == "test_perturb")
                else:
                    mode_condition = mode_col == "test_perturb"
                filter_conditions.append(mode_condition)
                
                # Handle subject filtering
                if expected_result['eval_mode'] == 'CrossSession':
                    # For CrossSession, check specific subject
                    if 'subject' in self.existing_results.columns:
                        filter_conditions.append(self.existing_results['subject'] == expected_result['subject'])
                # For WithinSession, we don't filter by subject as results are aggregated
                
                # Apply all filters
                combined_filter = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_filter = combined_filter & condition
                
                # Check if any results match
                matching_results = self.existing_results[combined_filter]
                
                if len(matching_results) == 0:
                    missing_test_perturb_results.append(expected_result)
        
        print(f"[OK] Found {len(missing_test_perturb_results)} missing test_perturb results out of {len(self.expected_test_perturb_results)} total expected")
        
        # Now map missing results to required multirun jobs
        print("\n" + "="*60)
        print("MAPPING TO REQUIRED MULTIRUN JOBS")
        print("="*60)
        
        required_multirun_jobs = set()
        
        for missing_result in missing_test_perturb_results:
            if missing_result['eval_mode'] == 'CrossSession':
                # For CrossSession, each subject needs its own multirun job
                job_key = (
                    missing_result['dataset'],
                    missing_result['eval_mode'], 
                    missing_result['subject'],
                    missing_result['tune']
                )
            else:
                # For WithinSession, all subjects are processed together
                subjects_tuple = tuple(missing_result['subjects'])
                job_key = (
                    missing_result['dataset'],
                    missing_result['eval_mode'],
                    subjects_tuple,
                    missing_result['tune']
                )
            
            required_multirun_jobs.add(job_key)
        
        # Convert to list of multirun job dictionaries
        missing_experiments = []
        for job_key in required_multirun_jobs:
            dataset, eval_mode, subjects_or_subject, tune = job_key
            
            if eval_mode == 'CrossSession':
                # Single subject for CrossSession
                subjects = [subjects_or_subject]
            else:
                # Multiple subjects for WithinSession
                subjects = list(subjects_or_subject)
            
            multirun_job = {
                'dataset': dataset,
                'eval_mode': eval_mode,
                'subjects': subjects,
                'tune': tune,
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
            
            print("   By eval_mode:")
            for eval_mode, count in missing_df['eval_mode'].value_counts().items():
                print(f"     - {eval_mode}: {count} multirun jobs")
            
            print("   By tune flag:")
            for tune, count in missing_df['tune'].value_counts().items():
                print(f"     - {'tuned' if tune else 'not tuned'}: {count} multirun jobs")
        
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
                if exp['eval_mode'] == 'CrossSession':
                    # For CrossSession, single subject
                    subjects_str = str(exp['subjects'][0])
                else:
                    # For WithinSession, space-separated subjects
                    subjects_str = " ".join(map(str, exp['subjects']))
                
                # Handle tuning flag
                tune_flag = "true" if exp['tune'] else "false"
                
                # Generate sbatch command with appropriate time limits
                # Base estimates: CrossSession without tuning ~3 hours
                # WithinSession takes ~5x longer than CrossSession
                # Tuning adds significant overhead (~20x for CrossSession)
                
                if exp['eval_mode'] == 'CrossSession':
                    if exp['tune']:
                        # CrossSession with tuning: ~2.5 days
                        slurm_args = "--time=2-12:00:00 --mem=12G"
                    else:
                        # CrossSession without tuning: ~3 hours (with buffer)
                        slurm_args = "--time=0-06:00:00 --mem=12G"
                        
                elif exp['eval_mode'] == 'WithinSession':
                    if exp['tune']:
                        # WithinSession with tuning: ~5x longer than CrossSession tuning
                        # 2.5 days * 5 = ~12.5 days (use 14 days for safety)
                        slurm_args = "--time=14-00:00:00 --mem=12G"
                    else:
                        # WithinSession without tuning: ~5x longer than CrossSession
                        # 3 hours * 5 = 15 hours (with buffer, use 1 day)
                        slurm_args = "--time=1-00:00:00 --mem=12G"
                        
                else:
                    # Default time limit for other modes (CrossSubject, etc.)
                    slurm_args = "--time=1-00:00:00 --mem=12G"
                
                # Format: sbatch {slurm_args} unified_eval_script.sh {subject} {dataset} {eval_mode} {tune_flag}
                command = f"sbatch {slurm_args} unified_eval_script.sh {subjects_str} {exp['dataset']} {exp['eval_mode']} {tune_flag}"
                
                # Write sbatch command
                f.write(f"# Multirun Job {i}/{len(self.missing_experiments)}\n")
                f.write(f"# Dataset: {exp['dataset']} | Eval: {exp['eval_mode']} | Subjects: {exp['subjects']}")
                if exp['tune']:
                    f.write(" | TUNED")
                f.write("\n")
                f.write(f"# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp\n")
                f.write(f"# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500\n")
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
        
        # Step 1: Aggregate existing results
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
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing results, don't generate missing experiments")
    parser.add_argument("--missing-only", action="store_true",
                       help="Only identify missing experiments, don't aggregate")
    
    args = parser.parse_args()
    
    # Initialize automation system
    automation = ExperimentAutomation(args.config)
    
    if args.aggregate_only:
        # Only aggregate results
        automation.aggregate_existing_results()
    elif args.missing_only:
        # Only identify missing experiments
        automation.aggregate_existing_results()
        automation.identify_missing_experiments()
        automation.generate_summary_report(args.output_dir)
    else:
        # Full automation
        automation.run_full_automation(args.output_dir)


if __name__ == "__main__":
    main()
