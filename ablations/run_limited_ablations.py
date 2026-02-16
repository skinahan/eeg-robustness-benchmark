#!/usr/bin/env python3
"""
Limited Ablation Studies - Local Execution

This script runs ablation experiments for ablations 1, 2, 3, and 4 on BNCI2014_001
with CrossSession evaluation for all subjects. This is for local execution (not slurm).

Ablations:
1. No Carry Gate (branched_wiredcfc_arch4_no_carry_gate)
2. No Branching (branched_wiredcfc_arch4_no_branching)
3. LSTM Replacement (branched_lstm_arch4_equivalent)
4. No SNR Gate (branched_wiredcfc_arch4_no_snr_gate)

Configuration:
- Dataset: BNCI2014_001
- Evaluation Mode: CrossSession
- Subjects: All available subjects
- Seeds: [100, 200, 300, 400, 500]
- Tune: False (no hyperparameter tuning)
"""

import os
import sys
import gc
import torch
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
script_file = Path(__file__).resolve()
current_dir = script_file.parent  # ablations/ directory
project_root = current_dir.parent  # project root directory
project_root_str = str(project_root)
sys.path.insert(0, project_root_str)

# Import required modules
from config import (
    MODEL_REGISTRY,
    add_branched_wiredcfc_architecture,
    get_model_registry,
    _runtime_model_registry
)
from globals import set_seeds
from evaluation.unified_experiment_runner import UnifiedExperimentRunner
from moabb.datasets import BNCI2014_001
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

# Import ablation model variants
try:
    from ablations.ablation_models import (
        create_branched_wiredcfc_no_carry_gate_classifier,
        create_branched_wiredcfc_no_branching_classifier,
        create_branched_lstm_classifier,
        create_branched_wiredcfc_no_snr_gate_classifier,
    )
except ImportError:
    # Fallback: import from same directory
    import importlib.util
    ablation_models_path = current_dir / "ablation_models.py"
    if not ablation_models_path.exists():
        raise ImportError(f"Could not import ablation_models and file not found: {ablation_models_path}")
    spec = importlib.util.spec_from_file_location("ablation_models", str(ablation_models_path))
    ablation_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation_models)
    create_branched_wiredcfc_no_carry_gate_classifier = ablation_models.create_branched_wiredcfc_no_carry_gate_classifier
    create_branched_wiredcfc_no_branching_classifier = ablation_models.create_branched_wiredcfc_no_branching_classifier
    create_branched_lstm_classifier = ablation_models.create_branched_lstm_classifier
    create_branched_wiredcfc_no_snr_gate_classifier = ablation_models.create_branched_wiredcfc_no_snr_gate_classifier

# Configuration
ARCHITECTURE_FILE_RELATIVE = "outputs/architectures/best_architecture_4_trial_178.json"
ARCHITECTURE_FILE_PATH = project_root / ARCHITECTURE_FILE_RELATIVE
DATASET = "BNCI2014_001"
EVAL_MODE = "CrossSession"
SEEDS = [100, 200, 300, 400, 500]

# Ablation configurations: (ablation_num, model_name, factory_function, requires_wiring)
ABLATION_CONFIGS = [
    (1, "branched_wiredcfc_arch4_no_carry_gate", create_branched_wiredcfc_no_carry_gate_classifier, True),
    (2, "branched_wiredcfc_arch4_no_branching", create_branched_wiredcfc_no_branching_classifier, True),
    (3, "branched_lstm_arch4_equivalent", create_branched_lstm_classifier, False),
    (4, "branched_wiredcfc_arch4_no_snr_gate", create_branched_wiredcfc_no_snr_gate_classifier, True),
]


def cleanup_memory():
    """Perform aggressive garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def register_ablation_models():
    """Register all ablation models in the model registry."""
    print("\n" + "="*60)
    print("REGISTERING ABLATION MODELS")
    print("="*60)
    
    # Load architecture wiring
    if not ARCHITECTURE_FILE_PATH.exists():
        raise FileNotFoundError(f"Architecture file not found: {ARCHITECTURE_FILE_PATH}")
    
    wiring = load_architecture_from_file(str(ARCHITECTURE_FILE_PATH))
    print(f"[OK] Loaded architecture from: {ARCHITECTURE_FILE_PATH}")
    
    # Register baseline model
    try:
        add_branched_wiredcfc_architecture("branched_wiredcfc_arch4", wiring)
        print(f"[OK] Registered baseline: branched_wiredcfc_arch4")
    except Exception as e:
        raise RuntimeError(f"Failed to register baseline model: {e}")
    
    # Register ablation models
    for ablation_num, model_name, factory_func, requires_wiring in ABLATION_CONFIGS:
        try:
            if requires_wiring:
                # Create factory with wiring closure
                def make_factory_with_wiring(wiring_ref, factory):
                    def factory_wrapped(n_chans, n_times, n_outputs, **kwargs):
                        return factory(n_chans, n_times, n_outputs, wiring_ref, **kwargs)
                    return factory_wrapped
                factory_wrapped = make_factory_with_wiring(wiring, factory_func)
            else:
                # LSTM models don't need wiring
                factory_wrapped = factory_func
            
            # Register in runtime registry
            _runtime_model_registry[model_name] = factory_wrapped
            # Also register in MODEL_REGISTRY for backward compatibility
            MODEL_REGISTRY[model_name] = factory_wrapped
            print(f"[OK] Registered ablation {ablation_num}: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to register {model_name}: {e}")
    
    # Verify all models are accessible
    registry = get_model_registry()
    expected_models = ["branched_wiredcfc_arch4"] + [name for _, name, _, _ in ABLATION_CONFIGS]
    missing_models = [m for m in expected_models if m not in registry]
    if missing_models:
        raise RuntimeError(f"Models not accessible via get_model_registry(): {missing_models}")
    
    print(f"[OK] All {len(expected_models)} models registered and verified")
    print("="*60 + "\n")


def get_all_subjects():
    """Get all available subjects for BNCI2014_001."""
    try:
        dataset_obj = BNCI2014_001()
        if hasattr(dataset_obj, 'subject_list') and dataset_obj.subject_list:
            subjects = dataset_obj.subject_list
        else:
            # Default: BNCI2014_001 has 9 subjects (1-9)
            subjects = list(range(1, 10))
        print(f"[INFO] Found {len(subjects)} subjects for {DATASET}: {subjects}")
        return subjects
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {DATASET}: {e}")


def run_single_experiment(
    ablation_num: int,
    model_name: str,
    seed: int,
    subjects: list,
    job_num: int,
    total_jobs: int
) -> Tuple[int, bool, Optional[str], float]:
    """
    Run a single ablation experiment.
    
    Args:
        ablation_num: Ablation number (1-4)
        model_name: Model name in registry
        seed: Random seed
        subjects: List of subject IDs
        job_num: Job number for logging
        total_jobs: Total number of jobs
        
    Returns:
        Tuple of (job_num, success, error_message, elapsed_time)
    """
    job_start_time = time.time()
    
    print(f'\n{"-"*60}')
    print(f'Job {job_num}/{total_jobs}')
    print(f'Ablation {ablation_num} | Model: {model_name}')
    print(f'Dataset: {DATASET} | Eval: {EVAL_MODE}')
    print(f'Subjects: {len(subjects)} | Seed: {seed}')
    print(f'{"-"*60}')
    
    # Set seed for reproducibility
    set_seeds(seed)
    
    try:
        # Create and run experiment
        runner = UnifiedExperimentRunner(
            model=model_name,
            dataset=DATASET,
            subjects=subjects,
            mode='test_perturb',
            eval_mode=EVAL_MODE,
            seed=seed,
            noise_type='gaussian',  # multirun handles all noise types
            intensity=10.0,  # multirun handles all intensities
            tune=False,
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
    """Run all ablation experiments."""
    print("="*60)
    print("LIMITED ABLATION STUDIES - LOCAL EXECUTION")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Evaluation Mode: {EVAL_MODE}")
    print(f"Ablations: 1, 2, 3, 4")
    print(f"Seeds: {SEEDS}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Register models
    register_ablation_models()
    
    # Get all subjects
    subjects = get_all_subjects()
    
    # Build experiment list
    experiments = []
    for ablation_num, model_name, _, _ in ABLATION_CONFIGS:
        for seed in SEEDS:
            experiments.append({
                'ablation_num': ablation_num,
                'model_name': model_name,
                'seed': seed,
                'subjects': subjects
            })
    
    total_jobs = len(experiments)
    print(f"\n[INFO] Total experiments to run: {total_jobs}")
    print(f"       ({len(ABLATION_CONFIGS)} ablations × {len(SEEDS)} seeds)")
    
    # Run experiments sequentially (local execution)
    failed_jobs = []
    successful_jobs = 0
    start_time = time.time()
    
    try:
        for i, exp in enumerate(experiments):
            job_num = i + 1
            result_job_num, success, error_msg, elapsed_time = run_single_experiment(
                ablation_num=exp['ablation_num'],
                model_name=exp['model_name'],
                seed=exp['seed'],
                subjects=exp['subjects'],
                job_num=job_num,
                total_jobs=total_jobs
            )
            
            if success:
                successful_jobs += 1
            else:
                failed_jobs.append((result_job_num, exp, error_msg))
                print(f'\n[FAILURE] Job {result_job_num} failed. Continuing with remaining jobs...')
    
    except KeyboardInterrupt:
        print(f'\n\n[INTERRUPTED] KeyboardInterrupt received')
        print(f'Completed {successful_jobs}/{total_jobs} jobs before interruption')
    
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
    if successful_jobs > 0:
        print(f'Average time per job: {total_time/successful_jobs/60:.2f} minutes')
    
    if failed_jobs:
        print(f'\nFailed jobs:')
        for job_num, exp, error in failed_jobs:
            print(f'  Job {job_num}: Ablation {exp["ablation_num"]} | '
                  f'{exp["model_name"]} | seed={exp["seed"]} - Error: {error[:100]}')
        return 1
    else:
        print('\nAll jobs completed successfully!')
        return 0


if __name__ == '__main__':
    sys.exit(run_experiments())
