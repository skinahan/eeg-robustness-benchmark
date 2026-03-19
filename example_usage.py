#!/usr/bin/env python3
"""
Example usage of the EEG Experiment Automation System.

This script demonstrates different ways to use the automation system
for various experimental scenarios.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from experiment_automation import ExperimentAutomation


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("📋 Example 1: Basic Usage")
    print("-" * 40)
    
    # Initialize with default configuration
    automation = ExperimentAutomation("experiment_config.yaml")
    
    # Run full automation
    script_file, report_file = automation.run_full_automation()
    
    print(f"Generated script: {script_file}")
    print(f"Generated report: {report_file}")
    print(f"Missing experiments: {len(automation.missing_experiments)}")


def example_2_custom_output_directory():
    """Example 2: Using custom output directory."""
    print("\n📋 Example 2: Custom Output Directory")
    print("-" * 40)
    
    automation = ExperimentAutomation("experiment_config.yaml")
    
    # Use custom output directory
    custom_dir = "my_experiments"
    script_file, report_file = automation.run_full_automation(output_dir=custom_dir)
    
    print(f"Custom directory: {custom_dir}")
    print(f"Generated script: {script_file}")
    print(f"Generated report: {report_file}")


def example_3_aggregate_only():
    """Example 3: Only aggregate existing results."""
    print("\n📋 Example 3: Aggregate Only")
    print("-" * 40)
    
    automation = ExperimentAutomation("experiment_config.yaml")
    
    # Only aggregate results
    results = automation.aggregate_existing_results()
    
    if results is not None:
        print(f"Aggregated {len(results)} result rows")
        print(f"Datasets: {results['dataset'].unique()}")
        print(f"Models: {results['model'].unique()}")
    else:
        print("No results found to aggregate")


def example_4_identify_missing_only():
    """Example 4: Only identify missing experiments."""
    print("\n📋 Example 4: Identify Missing Only")
    print("-" * 40)
    
    automation = ExperimentAutomation("experiment_config.yaml")
    
    # Aggregate and identify missing
    automation.aggregate_existing_results()
    missing = automation.identify_missing_experiments()
    
    print(f"Found {len(missing)} missing experiments")
    
    # Show breakdown by model
    if missing:
        # Create detailed breakdown
        from collections import defaultdict
        
        breakdown = defaultdict(lambda: defaultdict(int))
        
        for exp in automation.missing_experiments:
            print(exp)  
        
        print("Missing experiments breakdown:")
        for combo, seeds in breakdown.items():
            total = sum(seeds.values())
            print(f"  {combo}: {total} experiments across {len(seeds)} seeds")
    


def example_5_custom_configuration():
    """Example 5: Using custom configuration."""
    print("\n📋 Example 5: Custom Configuration")
    print("-" * 40)
    
    # Create a minimal custom configuration
    custom_config = {
        'datasets': {
            'BNCI2014_001': {
                'name': 'BNCI2014_001',
                'paradigm': 'MotorImagery',
                'subjects': [1, 2, 3]  # Only first 3 subjects
            }
        },
        'models': [
            {'name': 'eegnet', 'display_name': 'EEGNet'},
            {'name': 'reegnet', 'display_name': 'REEGNet'}
        ],
        'eval_modes': ['WithinSession'],
        'experiment_modes': [
            {
                'name': 'baseline',
                'display_name': 'Baseline',
                'requires_noise': False,
                'supports_tuning': True
            }
        ],
        'noise_types': [],
        'noise_intensities': [],
        'seeds': [100, 200],
        'output': {
            'script_dir': 'custom_experiments',
            'python_executable': 'python',
            'missing_script_file': 'custom_missing_experiments.sh'
        },
        'command_template': '''python evaluation/unified_experiment_runner.py \\
    --model {model} \\
    --dataset {dataset} \\
    --subjects {subjects_str} \\
    --mode {mode} \\
    --eval_mode {eval_mode} \\
    --seed {seed} \\
    {noise_args} \\
    {tune_flag} \\
    --overwrite'''
    }
    
    # Write custom config
    custom_config_file = "custom_config.yaml"
    with open(custom_config_file, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    try:
        # Use custom configuration
        automation = ExperimentAutomation(custom_config_file)
        
        # Generate expected experiments
        expected = automation.generate_expected_experiments()
        print(f"Custom config generated {len(expected)} expected experiments")
        
        # Show some examples
        for i, exp in enumerate(expected[:3]):
            print(f"  {i+1}. {exp['model']} | {exp['mode']} | seed={exp['seed']}")
        
    finally:
        # Clean up custom config file
        if os.path.exists(custom_config_file):
            os.unlink(custom_config_file)


def example_6_progress_monitoring():
    """Example 6: Progress monitoring and reporting."""
    print("\n📋 Example 6: Progress Monitoring")
    print("-" * 40)
    
    automation = ExperimentAutomation("experiment_config.yaml")
    
    # Aggregate results
    automation.aggregate_existing_results()
    
    # Identify missing experiments
    automation.identify_missing_experiments()
    
    # Generate detailed report
    if automation.missing_experiments:
        # Create detailed breakdown
        from collections import defaultdict
        
        breakdown = defaultdict(lambda: defaultdict(int))
        
        for exp in automation.missing_experiments:
            key = f"{exp['model']}_{exp['mode']}_{exp['eval_mode']}"
            breakdown[key][exp['seed']] += 1
        
        print("Missing experiments breakdown:")
        for combo, seeds in breakdown.items():
            total = sum(seeds.values())
            print(f"  {combo}: {total} experiments across {len(seeds)} seeds")
    
    # Generate summary report
    report_file = automation.generate_summary_report()
    print(f"Detailed report saved to: {report_file}")


def main():
    """Run all examples."""
    print("🚀 EEG Experiment Automation System - Usage Examples")
    print("=" * 60)
    
    examples = [
        # example_1_basic_usage,
        # example_2_custom_output_directory,
        # example_3_aggregate_only,
        example_4_identify_missing_only,
        # example_5_custom_configuration,
        # example_6_progress_monitoring
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()  # Add spacing between examples
    
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Edit experiment_config.yaml for your specific needs")
    print("2. Run: python experiment_automation.py")
    print("3. Execute the generated shell script")
    print("4. Monitor progress and re-run automation as needed")


if __name__ == "__main__":
    main()


