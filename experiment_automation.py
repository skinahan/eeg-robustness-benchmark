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
        """Generate all expected experimental combinations based on configuration."""
        print("\n" + "="*60)
        print("GENERATING EXPECTED EXPERIMENTS")
        print("="*60)
        
        expected_experiments = []
        
        # Get base configurations
        datasets = self.config['datasets']
        models = self.config['models']
        eval_modes = self.config['eval_modes']
        experiment_modes = self.config['experiment_modes']
        seeds = self.config['seeds']
        
        # Generate all combinations
        for dataset_name, dataset_config in datasets.items():
            for model_config in models:
                for eval_mode in eval_modes:
                    for exp_mode_config in experiment_modes:
                        for seed in seeds:
                            # Handle subject groups
                            subjects = dataset_config['subjects']
                            
                            # Determine noise requirements
                            exp_mode = exp_mode_config['name']
                            requires_noise = exp_mode_config['requires_noise']
                            
                            if requires_noise:
                                # For modes that require noise, iterate through noise types and intensities
                                noise_types = self.config['noise_types']

                                # Use the same noise intensity range as in unified_experiment_runner.py _train_and_evaluate_perturb
                                # min_intensity = 1.0, max_intensity = 50.0, num_steps = 20
                                
                                noise_intensities = np.linspace(1.0, 50.0, 20)

                                for noise_type in noise_types:
                                    for intensity in noise_intensities:
                                        experiment = {
                                            'dataset': dataset_name,
                                            'paradigm': dataset_config['paradigm'],
                                            'subjects': subjects,
                                            'model': model_config['name'],
                                            'eval_mode': eval_mode,
                                            'mode': exp_mode,
                                            'seed': seed,
                                            'noise_type': noise_type,
                                            'intensity': intensity,
                                            'tune': False  # Will be determined based on mode
                                        }
                                        
                                        # Handle tuning logic
                                        if exp_mode == 'tune':
                                            experiment['tune'] = True
                                        elif exp_mode_config['supports_tuning']:
                                            # Create both tuned and untuned versions for applicable modes
                                            for tune_flag in [False, True]:
                                                exp_copy = experiment.copy()
                                                exp_copy['tune'] = tune_flag
                                                expected_experiments.append(exp_copy)
                                        else:
                                            expected_experiments.append(experiment)
                            else:
                                # For modes that don't require noise
                                experiment = {
                                    'dataset': dataset_name,
                                    'paradigm': dataset_config['paradigm'],
                                    'subjects': subjects,
                                    'model': model_config['name'],
                                    'eval_mode': eval_mode,
                                    'mode': exp_mode,
                                    'seed': seed,
                                    'noise_type': None,
                                    'intensity': None,
                                    'tune': False
                                }
                                
                                # Handle tuning logic
                                if exp_mode == 'tune':
                                    experiment['tune'] = True
                                elif exp_mode_config['supports_tuning']:
                                    # Create both tuned and untuned versions
                                    for tune_flag in [False, True]:
                                        exp_copy = experiment.copy()
                                        exp_copy['tune'] = tune_flag
                                        expected_experiments.append(exp_copy)
                                else:
                                    expected_experiments.append(experiment)
        
        print(f"[OK] Generated {len(expected_experiments)} expected experiments")
        return expected_experiments
    
    def identify_missing_experiments(self) -> List[Dict[str, Any]]:
        """Identify which expected experiments are missing from existing results."""
        print("\n" + "="*60)
        print("IDENTIFYING MISSING EXPERIMENTS")
        print("="*60)
        
        if self.existing_results is None or self.existing_results.empty:
            print("[WARNING] No existing results found, all experiments are missing")
            self.missing_experiments = self.generate_expected_experiments()
            return self.missing_experiments
        
        expected_experiments = self.generate_expected_experiments()
        missing_experiments = []
        
        for expected_exp in expected_experiments:
            # Create a filter for this experiment
            filter_conditions = []
            
            # Basic filters
            filter_conditions.append(self.existing_results['dataset'] == expected_exp['dataset'])
            filter_conditions.append(self.existing_results['model'] == expected_exp['model'])
            filter_conditions.append(self.existing_results['eval_mode'].str.contains(expected_exp['eval_mode'], na=False))
            filter_conditions.append(self.existing_results['seed'] == expected_exp['seed'])
            
            # Handle mode filtering (account for tuning suffixes)
            mode_col = self.existing_results['mode']
            if expected_exp['tune'] and expected_exp['mode'] != 'tune':
                # Look for tuned versions
                mode_condition = (mode_col == f"{expected_exp['mode']}_tune") | (mode_col == expected_exp['mode'])
            else:
                mode_condition = mode_col == expected_exp['mode']
            filter_conditions.append(mode_condition)
            
            # Handle noise filtering
            if expected_exp['noise_type'] is not None:
                filter_conditions.append(self.existing_results['noise_type'] == expected_exp['noise_type'])
                filter_conditions.append(self.existing_results['intensity'] == expected_exp['intensity'])
            else:
                # For experiments without noise, check that noise_type is null/empty
                filter_conditions.append(
                    self.existing_results['noise_type'].isna() | 
                    (self.existing_results['noise_type'] == '')
                )
            
            # Apply all filters
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            
            # Check if any results match
            matching_results = self.existing_results[combined_filter]
            
            if len(matching_results) == 0:
                missing_experiments.append(expected_exp)
        
        self.missing_experiments = missing_experiments
        print(f"[OK] Found {len(missing_experiments)} missing experiments out of {len(expected_experiments)} total expected")
        
        # Print summary of missing experiments
        if missing_experiments:
            print("\n[INFO] Missing experiments summary:")
            missing_df = pd.DataFrame(missing_experiments)
            
            print("   By dataset:")
            for dataset, count in missing_df['dataset'].value_counts().items():
                print(f"     - {dataset}: {count} experiments")
            
            print("   By model:")
            for model, count in missing_df['model'].value_counts().items():
                print(f"     - {model}: {count} experiments")
            
            print("   By mode:")
            for mode, count in missing_df['mode'].value_counts().items():
                print(f"     - {mode}: {count} experiments")
        
        return missing_experiments
    
    def generate_shell_script(self, output_dir: str = None) -> str:
        """Generate shell script with missing experiment commands."""
        print("\n" + "="*60)
        print("GENERATING SHELL SCRIPT")
        print("="*60)
        
        if output_dir is None:
            output_dir = self.config['output']['script_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        script_file = os.path.join(output_dir, self.config['output']['missing_script_file'])
        python_exec = self.config['output']['python_executable']
        
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated experiment automation script\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total missing experiments: {len(self.missing_experiments)}\n\n")
            
            f.write("set -e  # Exit on any error\n\n")
            
            f.write("echo \"Starting experiment automation...\"\n")
            f.write("echo \"Total experiments to run: {}\"\n\n".format(len(self.missing_experiments)))
            
            for i, exp in enumerate(self.missing_experiments, 1):
                # Build command arguments
                subjects_str = " ".join(map(str, exp['subjects']))
                
                # Handle noise arguments
                noise_args = ""
                if exp['noise_type'] is not None:
                    noise_args = f"--noise_type {exp['noise_type']} --intensity {exp['intensity']}"
                
                # Handle tuning flag
                tune_flag = "--tune" if exp['tune'] else ""
                
                # Generate command
                command = self.config['command_template'].format(
                    python_executable=python_exec,
                    model=exp['model'],
                    dataset=exp['dataset'],
                    subjects_str=subjects_str,
                    mode=exp['mode'],
                    eval_mode=exp['eval_mode'],
                    seed=exp['seed'],
                    noise_args=noise_args,
                    tune_flag=tune_flag
                )
                
                # Write experiment command
                f.write(f"# Experiment {i}/{len(self.missing_experiments)}\n")
                f.write(f"# {exp['model']} | {exp['dataset']} | {exp['mode']} | {exp['eval_mode']} | seed={exp['seed']}")
                if exp['noise_type']:
                    f.write(f" | {exp['noise_type']}={exp['intensity']}")
                if exp['tune']:
                    f.write(" | TUNED")
                f.write("\n")
                f.write(f"echo \"Running experiment {i}/{len(self.missing_experiments)}...\"\n")
                f.write(f"{command}\n")
                f.write("if [ $? -eq 0 ]; then\n")
                f.write(f"    echo \"[SUCCESS] Experiment {i} completed successfully\"\n")
                f.write("else\n")
                f.write(f"    echo \"[ERROR] Experiment {i} failed\"\n")
                f.write("    exit 1\n")
                f.write("fi\n\n")
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        print(f"[OK] Generated shell script: {script_file}")
        print(f"[INFO] Script contains {len(self.missing_experiments)} experiment commands")
        
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
