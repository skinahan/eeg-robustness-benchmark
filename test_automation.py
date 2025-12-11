#!/usr/bin/env python3
"""
Test script for the experiment automation system.
This script demonstrates how to use the automation system with a minimal configuration.
"""

import os
import sys
import yaml
import tempfile
import shutil
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from experiment_automation import ExperimentAutomation


def create_test_config():
    """Create a minimal test configuration."""
    test_config = {
        'datasets': {
            'BNCI2014_001': {
                'name': 'BNCI2014_001',
                'paradigm': 'MotorImagery',
                'subjects': [1, 2]  # Small subset for testing
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
            },
            {
                'name': 'augment',
                'display_name': 'Data Augmentation',
                'requires_noise': True,
                'supports_tuning': True
            }
        ],
        'noise_types': ['gaussian'],
        'noise_intensities': [10.0, 20.0],
        'seeds': [100, 200],
        'output': {
            'script_dir': 'test_output',
            'python_executable': 'python',  # Use system python for testing
            'missing_script_file': 'test_missing_experiments.sh'
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
    
    return test_config


def test_automation():
    """Test the automation system with a minimal configuration."""
    print("[TEST] Testing Experiment Automation System")
    print("="*50)
    
    # Create test configuration
    test_config = create_test_config()
    
    # Write test config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        test_config_file = f.name
    
    try:
        # Initialize automation system
        automation = ExperimentAutomation(test_config_file)
        
        print("[OK] Configuration loaded successfully")
        
        # Test result aggregation (will likely be empty for testing)
        print("\n[TEST] Testing result aggregation...")
        results = automation.aggregate_existing_results()
        
        # Test experiment generation
        print("\n[TEST] Testing experiment generation...")
        expected_experiments = automation.generate_expected_experiments()
        print(f"[OK] Generated {len(expected_experiments)} expected experiments")
        
        # Test missing experiment identification
        print("\n[TEST] Testing missing experiment identification...")
        missing_experiments = automation.identify_missing_experiments()
        print(f"[OK] Identified {len(missing_experiments)} missing experiments")
        
        # Test shell script generation
        print("\n[TEST] Testing shell script generation...")
        script_file = automation.generate_shell_script()
        print(f"[OK] Generated shell script: {script_file}")
        
        # Test Python script generation (local mode)
        print("\n[TEST] Testing Python script generation (local mode)...")
        automation_local = ExperimentAutomation(test_config_file, local=True)
        automation_local.aggregate_existing_results()
        automation_local.identify_missing_experiments()
        python_script_file = automation_local.generate_python_script()
        print(f"[OK] Generated Python script: {python_script_file}")
        
        # Verify Python script exists and has content
        if os.path.exists(python_script_file):
            with open(python_script_file, 'r') as f:
                script_content = f.read()
                if 'UnifiedExperimentRunner' in script_content and 'tqdm' in script_content:
                    print("[OK] Python script contains expected components")
                else:
                    print("[WARNING] Python script may be missing expected components")
        
        # Test summary report generation
        print("\n[TEST] Testing summary report generation...")
        report_file = automation.generate_summary_report()
        print(f"[OK] Generated summary report: {report_file}")
        
        print("\n[SUCCESS] All tests passed!")
        
        # Show some example missing experiments
        if missing_experiments:
            print(f"\n[INFO] Example missing experiments:")
            for i, exp in enumerate(missing_experiments[:5]):  # Show first 5
                print(f"   {i+1}. {exp['model']} | {exp['dataset']} | {exp['mode']} | {exp['eval_mode']} | seed={exp['seed']}")
                if exp['noise_type']:
                    print(f"      Noise: {exp['noise_type']}={exp['intensity']}")
                if exp['tune']:
                    print(f"      Tuned: Yes")
        
        # Clean up test files
        print(f"\n[INFO] Cleaning up test files...")
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary config file
        if os.path.exists(test_config_file):
            os.unlink(test_config_file)


def demonstrate_configuration():
    """Demonstrate how to modify the configuration for different use cases."""
    print("\n" + "="*50)
    print("CONFIGURATION EXAMPLES")
    print("="*50)
    
    examples = {
        "Minimal Testing": {
            'description': 'Quick testing with minimal experiments',
            'datasets': {'BNCI2014_001': {'subjects': [1, 2]}},
            'models': [{'name': 'eegnet'}],
            'eval_modes': ['WithinSession'],
            'experiment_modes': [{'name': 'baseline', 'requires_noise': False}],
            'seeds': [100],
            'noise_types': [],
            'noise_intensities': []
        },
        
        "Full Benchmark": {
            'description': 'Complete benchmark with all combinations',
            'datasets': {'BNCI2014_001': {'subjects': [1, 2, 3, 4, 5, 6, 7, 8, 9]}},
            'models': [
                {'name': 'eegnet'}, {'name': 'reegnet'}, {'name': 'cnn_ncp'},
                {'name': 'cnncfc_v2'}, {'name': 'cnncfc_compact'}
            ],
            'eval_modes': ['WithinSession', 'CrossSession', 'CrossSubject'],
            'experiment_modes': [
                {'name': 'baseline', 'requires_noise': False, 'supports_tuning': True},
                {'name': 'tune', 'requires_noise': False, 'supports_tuning': False},
                {'name': 'augment', 'requires_noise': True, 'supports_tuning': True},
                {'name': 'perturb', 'requires_noise': True, 'supports_tuning': True},
                {'name': 'test_perturb', 'requires_noise': True, 'supports_tuning': True}
            ],
            'seeds': [100, 200, 300, 400, 500],
            'noise_types': ['gaussian', 'dropout', 'eog'],
            'noise_intensities': [10.0, 20.0, 30.0, 40.0, 50.0]
        },
        
        "Robustness Study": {
            'description': 'Focus on robustness testing',
            'datasets': {'BNCI2014_001': {'subjects': [1, 2, 3, 4, 5]}},
            'models': [{'name': 'eegnet'}, {'name': 'reegnet'}],
            'eval_modes': ['WithinSession'],
            'experiment_modes': [
                {'name': 'test_perturb', 'requires_noise': True, 'supports_tuning': True}
            ],
            'seeds': [100, 200, 300],
            'noise_types': ['gaussian', 'dropout', 'eog'],
            'noise_intensities': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
        }
    }
    
    for name, config in examples.items():
        print(f"\n[INFO] {name}")
        print(f"   {config['description']}")
        
        # Calculate total experiments
        total_experiments = (
            len(config['datasets']['BNCI2014_001']['subjects']) *
            len(config['models']) *
            len(config['eval_modes']) *
            len(config['experiment_modes']) *
            len(config['seeds'])
        )
        
        # Add noise combinations for applicable modes
        noise_experiments = 0
        for mode in config['experiment_modes']:
            if mode['requires_noise']:
                if mode['name'] == 'test_perturb':
                    # test_perturb uses default noise for training, tests multiple types
                    noise_experiments += len(config['noise_types']) * len(config['noise_intensities'])
                else:
                    # Other noise modes use specific noise type/intensity combinations
                    noise_experiments += len(config['noise_types']) * len(config['noise_intensities'])
        
        if noise_experiments > 0:
            total_experiments += noise_experiments
        
        print(f"   Estimated experiments: {total_experiments}")
        print(f"   Models: {len(config['models'])}")
        print(f"   Eval modes: {len(config['eval_modes'])}")
        print(f"   Experiment modes: {len(config['experiment_modes'])}")
        print(f"   Seeds: {len(config['seeds'])}")


if __name__ == "__main__":
    print("[START] EEG Experiment Automation System - Test Suite")
    
    # Run the test
    success = test_automation()
    
    # Show configuration examples
    demonstrate_configuration()
    
    if success:
        print("\n[SUCCESS] Test completed successfully!")
        print("\n[INFO] To get started with the full system:")
        print("1. Edit experiment_config.yaml with your desired experiments")
        print("2. Run: python experiment_automation.py")
        print("3. Execute the generated shell script")
    else:
        print("\n[ERROR] Test failed. Please check the error messages above.")
        sys.exit(1)
