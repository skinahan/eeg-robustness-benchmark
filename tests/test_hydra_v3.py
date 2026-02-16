#!/usr/bin/env python3
"""
Systematic evaluation of HYDRAv3 features vs baseline models.

This script:
1. Tests HYDRAv3 with different configurations
2. Compares against branched_wiredcfc_arch4 (baseline) and HYDRAv2
3. Tests key HYDRAv3 features:
   - CfC carry controller (use_cfc_carry_controller)
   - Controller dimension (controller_dim: 1 or 2)
   - SSVEP head (use_ssvep_head)

Features tested:
- Baseline: HYDRAv3 with all features enabled (default configuration)
- Individual features:
  * CfC carry controller disabled (use_cfc_carry_controller=False)
  * Controller dimension 2 (controller_dim=2)
  * SSVEP head disabled (use_ssvep_head=False)
- Ablation: No carry controller (reverts to HYDRAv1-like behavior)

The default HYDRAv3 configuration uses:
- use_cfc_carry_controller=True (CfC-based adaptive gating)
- controller_dim=1 (lightweight controller)
- use_ssvep_head=True (robustness head)
- Architecture 4 wiring (best performing per analysis)
"""

import subprocess
import sys
import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path to import config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def register_hydra_v3_variant(variant_name: str, wiring, feature_config: Dict):
    """
    Register a HYDRAv3 variant with specific feature configuration.
    
    This creates a registration file that the subprocess can import to register
    the variant, since subprocess imports config.py fresh.
    
    Args:
        variant_name: Name for the variant (e.g., "hydra_v3_baseline")
        wiring: ArbitraryWiring instance or path to architecture file
        feature_config: Dictionary of feature flags and parameters
    """
    from config import MODEL_REGISTRY, get_model_registry
    from models.hydra import create_hydra_v3_classifier
    from architecture_refinement.arbitrary_wiring import load_architecture_from_file
    import json
    
    # If wiring is a path, store the path; otherwise we need to serialize it
    if isinstance(wiring, (str, Path)):
        wiring_path = str(wiring)
    else:
        # For in-memory wiring, we'd need to serialize it, but for now assume it's a path
        raise ValueError("wiring must be a file path for subprocess registration")
    
    # Register in current process
    wiring_obj = load_architecture_from_file(wiring_path)
    wiring_ref = wiring_obj
    config_ref = feature_config.copy()
    
    def factory(n_chans, n_times, n_outputs, **kwargs):
        merged_kwargs = {**kwargs, **config_ref}
        return create_hydra_v3_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring_ref,
            **merged_kwargs
        )
    
    # Register in runtime registry (used by get_model_registry())
    from config import _runtime_model_registry
    _runtime_model_registry[variant_name] = factory
    
    # Also register in MODEL_REGISTRY for backward compatibility
    MODEL_REGISTRY[variant_name] = factory
    
    # Create a registration file that subprocess can import
    # Use absolute path for script_dir
    script_dir_abs = Path(script_dir).resolve()
    registration_dir = script_dir_abs / ".model_registry"
    registration_dir.mkdir(exist_ok=True)
    registration_file = registration_dir / f"{variant_name}.py"
    
    # Convert wiring_path to absolute path
    wiring_path_abs = str(Path(wiring_path).resolve())
    
    # Convert feature_config to Python dict string (convert JSON bools to Python bools)
    import pprint
    config_str = pprint.pformat(feature_config, width=120)
    
    # Write registration code
    registration_code = f'''"""
Auto-generated registration file for {variant_name}
"""
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from config import MODEL_REGISTRY, get_model_registry, _runtime_model_registry
from models.hydra import create_hydra_v3_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

# Load wiring
wiring = load_architecture_from_file(r"{wiring_path_abs}")

# Feature configuration
feature_config = {config_str}

# Create factory
def factory(n_chans, n_times, n_outputs, **kwargs):
    merged_kwargs = {{**kwargs, **feature_config}}
    return create_hydra_v3_classifier(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        wiring=wiring,
        **merged_kwargs
    )

# Register in runtime registry (used by get_model_registry())
# This is the persistent registry that get_model_registry() checks
_runtime_model_registry["{variant_name}"] = factory

# Also register in MODEL_REGISTRY for backward compatibility
MODEL_REGISTRY["{variant_name}"] = factory
'''
    
    with open(registration_file, 'w') as f:
        f.write(registration_code)
    
    return True

def load_variant_registrations():
    """Load all variant registrations from .model_registry directory."""
    registration_dir = Path(script_dir) / ".model_registry"
    if not registration_dir.exists():
        return
    
    for reg_file in registration_dir.glob("*.py"):
        try:
            # Import the registration file
            spec = importlib.util.spec_from_file_location(reg_file.stem, reg_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Warning: Failed to load registration from {reg_file}: {e}")

# Add import for importlib
import importlib.util

def get_baseline_wiring():
    """
    Get the wiring file path for architecture 4 (same as branched_wiredcfc_arch4).
    
    Returns:
        Path to the architecture file (as Path object)
    """
    from pathlib import Path
    
    # Try multiple possible paths
    possible_paths = [
        "outputs/architectures/best_architecture_4_trial_178.json",
        "architecture_refinement/outputs/architectures/best_architecture_4_trial_178.json",
    ]
    
    # Convert to absolute paths relative to script_dir
    for path in possible_paths:
        full_path = Path(script_dir) / path
        if full_path.exists():
            # Return as string to ensure compatibility
            return str(full_path.resolve())
    
    raise FileNotFoundError(f"Could not find architecture 4 file. Tried: {possible_paths}")

def define_feature_configs():
    """
    Define all feature configurations to test for HYDRAv3.
    
    Returns:
        Dictionary mapping variant names to feature configurations
    """
    configs = {}
    
    # Default HYDRAv3 configuration (all features enabled)
    configs["hydra_v3_default"] = {
        'use_cfc_carry_controller': True,  # CfC-based adaptive gating
        'controller_dim': 1,  # Lightweight controller (d_c=1)
        'use_ssvep_head': True,  # SSVEP head for robustness
    }
    
    # Ablation: No carry controller (reverts to HYDRAv1-like behavior)
    configs["hydra_v3_no_carry_controller"] = {
        'use_cfc_carry_controller': False,  # Disable carry controller
        'controller_dim': 1,  # Not used when disabled
        'use_ssvep_head': True,
    }
    
    # Controller dimension 2 (larger controller)
    configs["hydra_v3_controller_dim2"] = {
        'use_cfc_carry_controller': True,
        'controller_dim': 2,  # Larger controller (d_c=2)
        'use_ssvep_head': True,
    }
    
    # No SSVEP head (ablation)
    configs["hydra_v3_no_ssvep"] = {
        'use_cfc_carry_controller': True,
        'controller_dim': 1,
        'use_ssvep_head': False,  # Disable SSVEP head
    }
    
    # Minimal configuration (no carry controller, no SSVEP head)
    configs["hydra_v3_minimal"] = {
        'use_cfc_carry_controller': False,
        'controller_dim': 1,
        'use_ssvep_head': False,
    }
    
    return configs

def run_experiment(model, script_dir, output_file=None, dataset="BNCI2014_001", subjects=[1], 
                   mode="test_perturb", eval_mode="CrossSession", seed=42, tune=False):
    """Run a single experiment and return the result."""
    unified_runner = os.path.join(script_dir, "evaluation", "unified_experiment_runner.py")
    
    # Build the command - unified_experiment_runner will auto-load registrations
    cmd = [
        sys.executable,
        unified_runner,
        "--model", model,
        "--dataset", dataset,
        "--subjects"] + [str(s) for s in subjects] + [
        "--mode", mode,
        "--eval_mode", eval_mode,
        "--seed", str(seed),
        "--overwrite"
    ]
    
    # Add --tune flag only if requested
    if tune:
        cmd.append("--tune")
    
    print(f"\n{'=' * 80}")
    print(f"Running {model}")
    print(f"{'=' * 80}")
    
    # Run the command with output to stdout
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            cwd=script_dir,
            # Output to stdout (no redirection)
            text=True
        )
        print(f"\n[OK] {model} completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {model} failed with error code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\nError running {model}: {e}")
        return 1

def load_results_csv(csv_path):
    """Load results from a CSV file and return as a dictionary."""
    import pandas as pd
    try:
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        # Extract key metrics
        results = {
            'path': str(csv_path),
            'clean_accuracy': None,
            'clean_roc_auc': None,
            'noise_results': {}
        }
        
        # In test_perturb mode, clean metrics are stored in 'clean_accuracy' and 'clean_roc_auc' columns
        if len(df) > 0:
            results['clean_accuracy'] = df.iloc[0].get('clean_accuracy', None)
            results['clean_roc_auc'] = df.iloc[0].get('clean_roc_auc', None)
        
        # Get all noise results (corrupted metrics)
        noise_rows = df[df['noise_type'].notna() & (df['noise_type'] != '')]
        for _, row in noise_rows.iterrows():
            noise_type = row['noise_type']
            intensity = row['intensity']
            if noise_type not in results['noise_results']:
                results['noise_results'][noise_type] = {}
            results['noise_results'][noise_type][intensity] = {
                'accuracy': row.get('corrupted_accuracy', None),
                'roc_auc': row.get('corrupted_roc_auc', None)
            }
        
        return results
    except Exception as e:
        print(f"Warning: Could not load results from {csv_path}: {e}")
        return None

def find_results_files(script_dir, model, subject, seed, paradigm="MotorImagery", dataset="BNCI2014_001"):
    """Find result CSV files for the given model and subject."""
    results_dir = Path(script_dir) / "results" / paradigm / dataset / model / "CrossSessionEvaluation" / str(seed) / f"sub-{subject:03d}"
    
    results = {}
    if results_dir.exists():
        # Look for CSV files in session subdirectories
        for session_dir in results_dir.iterdir():
            if session_dir.is_dir():
                session = session_dir.name
                csv_files = list(session_dir.glob("test_perturb/*.csv"))
                if csv_files:
                    results[session] = csv_files[0]  # Take first CSV file
    
    return results

def compare_all_results(baseline_results: Dict, variant_results: Dict[str, Dict], dataset: str = "BNCI2014_001"):
    """
    Compare all variant results against baseline and print comprehensive summary.
    
    Args:
        baseline_results: Results from branched_wiredcfc_arch4
        variant_results: Dictionary mapping variant names to their results
        dataset: Name of the dataset for the comparison
    """
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE COMPARISON SUMMARY - {dataset}")
    print("=" * 80)
    
    if not baseline_results or not baseline_results.get('clean_accuracy'):
        print("Warning: Baseline results not available")
        return
    
    baseline_acc = baseline_results['clean_accuracy']
    baseline_auc = baseline_results.get('clean_roc_auc')
    
    # Create comparison table
    print("\n" + "-" * 80)
    print("CLEAN ACCURACY COMPARISON")
    print("-" * 80)
    print(f"{'Model':<40} {'Accuracy':<12} {'vs Baseline':<15} {'Change':<10}")
    print("-" * 80)
    print(f"{'branched_wiredcfc_arch4 (baseline)':<40} {baseline_acc:.4f}      {'(baseline)':<15} {'':<10}")
    
    # Sort variants by accuracy difference
    variant_diffs = []
    for variant_name, results in variant_results.items():
        if results and results.get('clean_accuracy'):
            acc = results['clean_accuracy']
            diff = acc - baseline_acc
            diff_pct = (diff / baseline_acc) * 100 if baseline_acc > 0 else 0
            variant_diffs.append((variant_name, acc, diff, diff_pct))
    
    # Sort by accuracy difference (descending)
    variant_diffs.sort(key=lambda x: x[2], reverse=True)
    
    for variant_name, acc, diff, diff_pct in variant_diffs:
        change_str = f"{diff:+.4f} ({diff_pct:+.2f}%)"
        print(f"{variant_name:<40} {acc:.4f}      {change_str:<15}")
    
    if baseline_auc:
        print("\n" + "-" * 80)
        print("CLEAN ROC-AUC COMPARISON")
        print("-" * 80)
        print(f"{'Model':<40} {'ROC-AUC':<12} {'vs Baseline':<15} {'Change':<10}")
        print("-" * 80)
        print(f"{'branched_wiredcfc_arch4 (baseline)':<40} {baseline_auc:.4f}      {'(baseline)':<15} {'':<10}")
        
        variant_auc_diffs = []
        for variant_name, results in variant_results.items():
            if results and results.get('clean_roc_auc'):
                auc = results['clean_roc_auc']
                diff = auc - baseline_auc
                diff_pct = (diff / baseline_auc) * 100 if baseline_auc > 0 else 0
                variant_auc_diffs.append((variant_name, auc, diff, diff_pct))
        
        variant_auc_diffs.sort(key=lambda x: x[2], reverse=True)
        
        for variant_name, auc, diff, diff_pct in variant_auc_diffs:
            change_str = f"{diff:+.4f} ({diff_pct:+.2f}%)"
            print(f"{variant_name:<40} {auc:.4f}      {change_str:<15}")
    
    # Noise robustness comparison (sample)
    if baseline_results.get('noise_results'):
        print("\n" + "-" * 80)
        print("NOISE ROBUSTNESS COMPARISON (sample)")
        print("-" * 80)
        
        noise_types = ['gaussian', 'dropout', 'eog']
        for noise_type in noise_types:
            if noise_type in baseline_results['noise_results']:
                # Get a sample intensity
                intensities = sorted(baseline_results['noise_results'][noise_type].keys())
                if intensities:
                    sample_intensity = intensities[len(intensities)//2]
                    baseline_noise_acc = baseline_results['noise_results'][noise_type].get(sample_intensity, {}).get('accuracy')
                    
                    if baseline_noise_acc:
                        print(f"\n{noise_type} @ intensity {sample_intensity:.1f}:")
                        print(f"  {'Model':<40} {'Accuracy':<12} {'vs Baseline':<15}")
                        print(f"  {'branched_wiredcfc_arch4 (baseline)':<40} {baseline_noise_acc:.4f}      {'(baseline)':<15}")
                        
                        noise_diffs = []
                        for variant_name, results in variant_results.items():
                            if results and results.get('noise_results') and noise_type in results['noise_results']:
                                noise_acc = results['noise_results'][noise_type].get(sample_intensity, {}).get('accuracy')
                                if noise_acc:
                                    diff = noise_acc - baseline_noise_acc
                                    noise_diffs.append((variant_name, noise_acc, diff))
                        
                        noise_diffs.sort(key=lambda x: x[2], reverse=True)
                        for variant_name, noise_acc, diff in noise_diffs:
                            change_str = f"{diff:+.4f}"
                            print(f"  {variant_name:<40} {noise_acc:.4f}      {change_str:<15}")
    
    print("\n" + "=" * 80)

def write_summary_to_file(script_dir, all_dataset_results, output_file):
    """Write comprehensive summary of all datasets to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYDRAv3 SYSTEMATIC FEATURE EVALUATION - COMPREHENSIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name, dataset_info in all_dataset_results.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"DATASET: {dataset_name}\n")
            f.write("=" * 80 + "\n\n")
            
            baseline_results = dataset_info.get('baseline_results')
            variant_results = dataset_info.get('variant_results', {})
            results_status = dataset_info.get('results_status', {})
            
            # Write test status
            f.write("TEST STATUS:\n")
            f.write("-" * 80 + "\n")
            for model, status in results_status.items():
                status_str = "PASSED" if status == 0 else "FAILED"
                f.write(f"{model}: {status_str}\n")
            f.write("\n")
            
            if not baseline_results or not baseline_results.get('clean_accuracy'):
                f.write("Warning: Baseline results not available for this dataset.\n\n")
                continue
            
            baseline_acc = baseline_results['clean_accuracy']
            baseline_auc = baseline_results.get('clean_roc_auc')
            
            # Clean accuracy comparison
            f.write("-" * 80 + "\n")
            f.write("CLEAN ACCURACY COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<40} {'Accuracy':<12} {'vs Baseline':<15} {'Change':<10}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'branched_wiredcfc_arch4 (baseline)':<40} {baseline_acc:.4f}      {'(baseline)':<15} {'':<10}\n")
            
            variant_diffs = []
            for variant_name, results in variant_results.items():
                if results and results.get('clean_accuracy'):
                    acc = results['clean_accuracy']
                    diff = acc - baseline_acc
                    diff_pct = (diff / baseline_acc) * 100 if baseline_acc > 0 else 0
                    variant_diffs.append((variant_name, acc, diff, diff_pct))
            
            variant_diffs.sort(key=lambda x: x[2], reverse=True)
            for variant_name, acc, diff, diff_pct in variant_diffs:
                change_str = f"{diff:+.4f} ({diff_pct:+.2f}%)"
                f.write(f"{variant_name:<40} {acc:.4f}      {change_str:<15}\n")
            
            # Clean ROC-AUC comparison
            if baseline_auc:
                f.write("\n" + "-" * 80 + "\n")
                f.write("CLEAN ROC-AUC COMPARISON\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Model':<40} {'ROC-AUC':<12} {'vs Baseline':<15} {'Change':<10}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'branched_wiredcfc_arch4 (baseline)':<40} {baseline_auc:.4f}      {'(baseline)':<15} {'':<10}\n")
                
                variant_auc_diffs = []
                for variant_name, results in variant_results.items():
                    if results and results.get('clean_roc_auc'):
                        auc = results['clean_roc_auc']
                        diff = auc - baseline_auc
                        diff_pct = (diff / baseline_auc) * 100 if baseline_auc > 0 else 0
                        variant_auc_diffs.append((variant_name, auc, diff, diff_pct))
                
                variant_auc_diffs.sort(key=lambda x: x[2], reverse=True)
                for variant_name, auc, diff, diff_pct in variant_auc_diffs:
                    change_str = f"{diff:+.4f} ({diff_pct:+.2f}%)"
                    f.write(f"{variant_name:<40} {auc:.4f}      {change_str:<15}\n")
            
            # Noise robustness comparison (sample)
            if baseline_results.get('noise_results'):
                f.write("\n" + "-" * 80 + "\n")
                f.write("NOISE ROBUSTNESS COMPARISON (sample)\n")
                f.write("-" * 80 + "\n")
                
                noise_types = ['gaussian', 'dropout', 'eog']
                for noise_type in noise_types:
                    if noise_type in baseline_results['noise_results']:
                        intensities = sorted(baseline_results['noise_results'][noise_type].keys())
                        if intensities:
                            sample_intensity = intensities[len(intensities)//2]
                            baseline_noise_acc = baseline_results['noise_results'][noise_type].get(sample_intensity, {}).get('accuracy')
                            
                            if baseline_noise_acc:
                                f.write(f"\n{noise_type} @ intensity {sample_intensity:.1f}:\n")
                                f.write(f"  {'Model':<40} {'Accuracy':<12} {'vs Baseline':<15}\n")
                                f.write(f"  {'branched_wiredcfc_arch4 (baseline)':<40} {baseline_noise_acc:.4f}      {'(baseline)':<15}\n")
                                
                                noise_diffs = []
                                for variant_name, results in variant_results.items():
                                    if results and results.get('noise_results') and noise_type in results['noise_results']:
                                        noise_acc = results['noise_results'][noise_type].get(sample_intensity, {}).get('accuracy')
                                        if noise_acc:
                                            diff = noise_acc - baseline_noise_acc
                                            noise_diffs.append((variant_name, noise_acc, diff))
                                
                                noise_diffs.sort(key=lambda x: x[2], reverse=True)
                                for variant_name, noise_acc, diff in noise_diffs:
                                    change_str = f"{diff:+.4f}"
                                    f.write(f"  {variant_name:<40} {noise_acc:.4f}      {change_str:<15}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")

def main():
    """Run the systematic feature evaluation for HYDRAv3."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all three benchmark datasets
    datasets_config = {
        "BNCI2014_001": {
            "name": "BNCI2014_001",
            "paradigm": "MotorImagery",
            "subjects": [1]
        },
        # "Lee2019_SSVEP": {
        #     "name": "Lee2019_SSVEP",
        #     "paradigm": "SSVEP",
        #     "subjects": [1]
        # },
        # "BI2015a": {
        #     "name": "BI2015a",
        #     "paradigm": "ERP",
        #     "subjects": [1]
        # }
    }
    
    # Common parameters for all tests
    mode = "test_perturb"
    eval_mode = "CrossSession"
    seed = 42
    tune = False  # Skip hyperparameter tuning
    
    print("=" * 80)
    print("HYDRAv3 SYSTEMATIC FEATURE EVALUATION")
    print("=" * 80)
    print(f"Datasets: {list(datasets_config.keys())}")
    print(f"Mode: {mode}")
    print(f"Eval Mode: {eval_mode}")
    print(f"Seed: {seed}")
    print(f"Tune: {tune}")
    print("=" * 80)
    
    # Get baseline wiring (architecture 4)
    print("\nLoading architecture 4 wiring...")
    try:
        wiring = get_baseline_wiring()
        print("[OK] Wiring loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load wiring: {e}")
        return 1
    
    # Define feature configurations
    print("\nDefining feature configurations...")
    feature_configs = define_feature_configs()
    print(f"[OK] Defined {len(feature_configs)} configurations")
    
    # Register all variants
    print("\nRegistering HYDRAv3 variants...")
    registered_variants = []
    for variant_name, config in feature_configs.items():
        try:
            register_hydra_v3_variant(variant_name, wiring, config)
            registered_variants.append(variant_name)
            print(f"  [OK] Registered {variant_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to register {variant_name}: {e}")
    
    # Models to test: baseline + all variants
    models_to_test = ["branched_wiredcfc_arch4"] + registered_variants
    
    print(f"\n[OK] Will test {len(models_to_test)} models: {models_to_test}")
    
    # Store results for all datasets
    all_dataset_results = {}
    
    # Run experiments for each dataset
    for dataset_name, dataset_info in datasets_config.items():
        dataset = dataset_info["name"]
        paradigm = dataset_info["paradigm"]
        subjects = dataset_info["subjects"]
        
        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET: {dataset_name} ({paradigm})")
        print("=" * 80)
        
        # Run experiments
        print("\n" + "-" * 80)
        print(f"RUNNING EXPERIMENTS FOR {dataset_name}")
        print("-" * 80)
        
        results_status = {}
        
        for model in models_to_test:
            status = run_experiment(
                model=model,
                script_dir=script_dir,
                output_file=None,  # Output to stdout instead
                dataset=dataset,
                subjects=subjects,
                mode=mode,
                eval_mode=eval_mode,
                seed=seed,
                tune=tune
            )
            results_status[model] = status
        
        # Load and compare results
        print("\n" + "-" * 80)
        print(f"LOADING RESULTS FOR {dataset_name}")
        print("-" * 80)
        
        baseline_results = None
        variant_results = {}
        
        # Load baseline results
        branched_csvs = find_results_files(script_dir, "branched_wiredcfc_arch4", subjects[0], seed, paradigm=paradigm, dataset=dataset)
        if branched_csvs:
            for session, csv_path in branched_csvs.items():
                baseline_results = load_results_csv(csv_path)
                if baseline_results:
                    print(f"[OK] Loaded branched_wiredcfc_arch4 results from session {session}")
                    break
        
        # Load variant results
        for variant_name in registered_variants:
            variant_csvs = find_results_files(script_dir, variant_name, subjects[0], seed, paradigm=paradigm, dataset=dataset)
            if variant_csvs:
                for session, csv_path in variant_csvs.items():
                    results = load_results_csv(csv_path)
                    if results:
                        variant_results[variant_name] = results
                        print(f"[OK] Loaded {variant_name} results from session {session}")
                        break
        
        # Compare all results for this dataset
        if baseline_results:
            compare_all_results(baseline_results, variant_results, dataset=dataset_name)
        else:
            print(f"\nWarning: Could not load baseline results for {dataset_name}.")
        
        # Store results for this dataset
        all_dataset_results[dataset_name] = {
            'baseline_results': baseline_results,
            'variant_results': variant_results,
            'results_status': results_status,
            'paradigm': paradigm
        }
    
    # Write comprehensive summary to file
    summary_output_file = os.path.join(script_dir, "test_hydra_v3_summary.txt")
    print("\n" + "=" * 80)
    print("WRITING COMPREHENSIVE SUMMARY")
    print("=" * 80)
    write_summary_to_file(script_dir, all_dataset_results, summary_output_file)
    print(f"[OK] Summary written to: {summary_output_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    all_passed_all_datasets = True
    for dataset_name, dataset_info in all_dataset_results.items():
        results_status = dataset_info['results_status']
        all_passed = all(status == 0 for status in results_status.values())
        if not all_passed:
            all_passed_all_datasets = False
        
        print(f"\n{dataset_name}:")
        for model, status in results_status.items():
            status_str = "PASSED" if status == 0 else "FAILED"
            print(f"  {model}: {status_str}")
    
    if all_passed_all_datasets:
        print("\n[OK] All tests completed successfully across all datasets!")
        print(f"\nSummary file: {summary_output_file}")
    else:
        print("\n[FAILED] Some tests failed. Check summary for details.")
    
    print("=" * 80)
    
    return 0 if all_passed_all_datasets else 1

if __name__ == "__main__":
    sys.exit(main())
