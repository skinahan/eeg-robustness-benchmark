#!/usr/bin/env python3
"""
Generated Python automation script for local experiment execution
Generated on: 2025-10-17 11:46:26
Total missing multirun jobs: 540
Non-tuned jobs: 270
Tuned jobs: 270
OPTIMIZED: Runs non-tuned jobs first, aggregates, then runs tuned jobs
"""

import os
import sys
import gc
import torch
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Tuple, Optional

# Add evaluation directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'evaluation'))

from evaluation.unified_experiment_runner import UnifiedExperimentRunner
from evaluation.experiment_utils import collect_all_results_unified
from globals import set_seeds

def cleanup_memory():
    """Perform aggressive garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_single_experiment(exp_config: Dict[str, Any], job_num: int, total_jobs: int) -> Tuple[int, bool, Optional[str], float]:
    """
    Run a single experiment in a separate process.
    
    Args:
        exp_config: Experiment configuration dictionary
        job_num: Job number for logging
        total_jobs: Total number of jobs
        
    Returns:
        Tuple of (job_num, success, error_message, elapsed_time)
    """
    job_start_time = time.time()
    
    # Re-import modules in the subprocess (necessary for multiprocessing)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'evaluation'))
    from evaluation.unified_experiment_runner import UnifiedExperimentRunner
    from globals import set_seeds
    
    print(f'\n{"-"*60}')
    print(f'Job {job_num}/{total_jobs}')
    print(f'Dataset: {exp_config["dataset"]} | Model: {exp_config["model"]} | '
          f'Eval: {exp_config["eval_mode"]} | Subjects: {exp_config["subjects"]} | '
          f'Seed: {exp_config["seed"]} | Tune: {exp_config["tune"]}')
    print(f'{"-"*60}')

    # Set seed for reproducibility
    set_seeds(exp_config['seed'])

    try:
        # Create and run experiment
        runner = UnifiedExperimentRunner(
            model=exp_config['model'],
            dataset=exp_config['dataset'],
            subjects=exp_config['subjects'],
            mode='test_perturb',
            eval_mode=exp_config['eval_mode'],
            seed=exp_config['seed'],
            noise_type='gaussian',  # multirun handles all noise types
            intensity=10.0,  # multirun handles all intensities
            tune=exp_config['tune'],
            overwrite=False
        )

        results = runner.run_experiment()
        job_time = time.time() - job_start_time
        print(f'[SUCCESS] Job {job_num} completed in {job_time/60:.2f} minutes.')
        if results is not None:
            print(f'Results shape: {results.shape}')

        # Clean up
        del runner
        if results is not None:
            del results
        cleanup_memory()
        
        return (job_num, True, None, job_time)

    except Exception as e:
        job_time = time.time() - job_start_time
        print(f'[ERROR] Job {job_num} failed after {job_time/60:.2f} minutes: {e}')
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        cleanup_memory()
        return (job_num, False, str(e), job_time)

def run_experiments():
    """Run all missing experiments in two phases: non-tuned first, then tuned."""
    total_jobs = 540
    non_tuned_jobs = 270
    tuned_jobs = 270
    print(f'Starting local experiment execution...')
    print(f'Total multirun jobs to execute: {total_jobs}')
    print(f'Non-tuned jobs: {non_tuned_jobs}')
    print(f'Tuned jobs: {tuned_jobs}')
    print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    non_tuned_experiments = [
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': False,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': False,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': False,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': False,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': False,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
    ]

    tuned_experiments = [
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'reegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [1],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [7],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [3],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [9],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [6],
            'tune': True,
            'model': 'eegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [9],
            'tune': True,
            'model': 'reegnet',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [6],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [8],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [5],
            'tune': True,
            'model': 'eegnet',
            'seed': 200,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'eegnet',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [5],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [8],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 500,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [2],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 100,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [4],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'WithinSession',
            'subjects': [1],
            'tune': True,
            'model': 'cnn_ncp',
            'seed': 300,
            'paradigm': 'MotorImagery'
        },
        {
            'dataset': 'BNCI2014_001',
            'eval_mode': 'CrossSession',
            'subjects': [4],
            'tune': True,
            'model': 'reegnet',
            'seed': 400,
            'paradigm': 'MotorImagery'
        },
    ]

    failed_jobs = []
    successful_jobs = 0
    start_time = time.time()
    
    # Run experiments in parallel with configurable workers
    max_workers = 4
    
    def run_experiment_batch(experiments, phase_name, job_offset=0):
        """Run a batch of experiments and return results."""
        if not experiments:
            print(f'No {phase_name} experiments to run.')
            return [], 0
        
        print(f'\n{"="*60}')
        print(f'PHASE: {phase_name.upper()}')
        print(f'{"="*60}')
        print(f'Running {len(experiments)} {phase_name} experiments...')
        print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        batch_failed_jobs = []
        batch_successful_jobs = 0
        
        # Sort experiments by model to potentially improve cache efficiency
        experiments.sort(key=lambda x: x['model'], reverse=True)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments to the executor
            future_to_job = {}
            for i, exp in enumerate(experiments):
                job_num = i + 1 + job_offset
                future = executor.submit(run_single_experiment, exp, job_num, total_jobs)
                future_to_job[future] = (job_num, exp)
            
            # Process completed experiments as they finish
            completed = 0
            with tqdm(total=len(experiments), desc=f'{phase_name} progress') as pbar:
                for future in as_completed(future_to_job):
                    job_num, exp = future_to_job[future]
                    completed += 1
                    
                    try:
                        result_job_num, success, error_msg, elapsed_time = future.result()
                        
                        if success:
                            batch_successful_jobs += 1
                        else:
                            batch_failed_jobs.append((result_job_num, exp, error_msg))
                        
                    except Exception as e:
                        print(f'\n[CRITICAL ERROR] Job {job_num} crashed: {e}')
                        import traceback
                        traceback.print_exc()
                        batch_failed_jobs.append((job_num, exp, str(e)))
                    
                    pbar.update(1)
                    
            # Cleanup after batch
            cleanup_memory()
        
        print(f'\n{phase_name.upper()} PHASE COMPLETE')
        print(f'Successful: {batch_successful_jobs}')
        print(f'Failed: {len(batch_failed_jobs)}')
        
        return batch_failed_jobs, batch_successful_jobs
    
    # PHASE 1: Run non-tuned experiments
    phase1_failed, phase1_successful = run_experiment_batch(non_tuned_experiments, 'non-tuned', 0)
    failed_jobs.extend(phase1_failed)
    successful_jobs += phase1_successful
    
    # Aggregate results after non-tuned phase
    if phase1_successful > 0:
        print(f'\n{"="*60}')
        print('AGGREGATING RESULTS AFTER NON-TUNED PHASE')
        print(f'{"="*60}')
        try:
            print('Calling collect_all_results_unified()...')
            aggregated_results = collect_all_results_unified()
            if aggregated_results is not None:
                print(f'Aggregated {len(aggregated_results)} result rows')
            else:
                print('No results found to aggregate')
        except Exception as e:
            print(f'Error during aggregation: {e}')
            import traceback
            traceback.print_exc()
    
    # PHASE 2: Run tuned experiments
    phase2_failed, phase2_successful = run_experiment_batch(tuned_experiments, 'tuned', len(non_tuned_experiments))
    failed_jobs.extend(phase2_failed)
    successful_jobs += phase2_successful
    
    # Final aggregation after tuned phase
    if phase2_successful > 0:
        print(f'\n{"="*60}')
        print('FINAL AGGREGATION AFTER TUNED PHASE')
        print(f'{"="*60}')
        try:
            print('Calling collect_all_results_unified()...')
            aggregated_results = collect_all_results_unified()
            if aggregated_results is not None:
                print(f'Final aggregated {len(aggregated_results)} result rows')
            else:
                print('No results found to aggregate')
        except Exception as e:
            print(f'Error during final aggregation: {e}')
            import traceback
            traceback.print_exc()
    
    # Final summary
    total_time = time.time() - start_time
    print(f'\n{"="*60}')
    print('EXPERIMENT EXECUTION COMPLETE')
    print(f'{"="*60}')
    print(f'Completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Total runtime: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)')
    print(f'Total jobs: {total_jobs}')
    print(f'Successful: {successful_jobs}')
    print(f'Failed: {len(failed_jobs)}')
    print(f'Non-tuned successful: {phase1_successful}')
    print(f'Tuned successful: {phase2_successful}')
    if successful_jobs > 0:
        print(f'Average time per job: {total_time/successful_jobs/60:.2f} minutes')

    if failed_jobs:
        print(f'\nFailed jobs:')
        for job_num, exp, error in failed_jobs:
            print(f'  Job {job_num}: {exp["dataset"]} | {exp["model"]} | '
                  f'{exp["eval_mode"]} | seed={exp["seed"]} - Error: {error}')
        sys.exit(1)
    else:
        print('\nAll jobs completed successfully!')
        sys.exit(0)

if __name__ == '__main__':
    run_experiments()
