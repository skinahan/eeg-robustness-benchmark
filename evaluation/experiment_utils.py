"""
Utility functions for EEG experiments.

This module contains helper functions used across different experiment types
to reduce code duplication and improve maintainability.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder

from utils import create_output_path, create_hdf5_model_path, get_noise_intensities
from evaluation.two_stage_hp_opt import run_two_stage_optuna


def extract_model_params(model) -> Dict[str, Any]:
    """Extract model parameters for logging purposes."""
    if hasattr(model, 'get_params'):
        return model.get_params()
    return {}


def check_skip_eval(model_name, seed, subject_list, mode, noise_type, intensity, eval_mode='CrossSession', paradigm='MotorImagery', dataset='BNCI2014_001', cache_manager=None, config=None, tuned=False, paradigm_obj=None, dataset_obj=None):

    if not eval_mode.endswith("Evaluation"):
        eval_mode = f"{eval_mode}Evaluation"

    """Check if evaluation should be skipped based on existing output files and model cache."""
    existing_output_paths = []
    expected_output_paths = []
    
    # Dynamically get session names from the dataset if available
    sessions_to_check = None
    if paradigm_obj is not None and dataset_obj is not None and len(subject_list) > 0:
        try:
            # Load data for first subject to get actual session names
            X_sample, y_sample, metadata_sample = paradigm_obj.get_data(dataset_obj, subjects=[subject_list[0]])
            if 'session' in metadata_sample.columns:
                sessions_to_check = sorted(metadata_sample['session'].unique().tolist())
        except Exception as e:
            print(f"Warning: Could not load sample data to get sessions: {e}")
    
    # Fallback to default sessions if we couldn't get them dynamically
    if sessions_to_check is None:
        sessions_to_check = ['0train', '1test']
    
    # Check if we should skip based on model cache (for non-noise modes)
    if cache_manager is not None and config is not None and noise_type is None:
        all_models_cached = True
        for subj in subject_list:
            for session in sessions_to_check:
                # Check if checkpoint files exist and config matches
                checkpoint_path = cache_manager._get_cache_path(
                    dataset, model_name, seed, int(subj), session, eval_mode, tuned, "best"
                )
                config_path = cache_manager._get_config_path(checkpoint_path)
                
                if not checkpoint_path.exists() or not config_path.exists():
                    all_models_cached = False
                    break
                
                # Check config hash
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    cached_hash = config_data.get('config_hash', '')
                    expected_hash = cache_manager._generate_config_hash(config)
                    
                    if cached_hash != expected_hash:
                        all_models_cached = False
                        break
                except Exception:
                    all_models_cached = False
                    break
            
            if not all_models_cached:
                break
        
        if all_models_cached:
            print(f"All required models are cached with matching configuration, checking output files...")
        else:
            print(f"Some models are not cached or have configuration mismatches, will retrain...")
    
    # Check existing output files
    if eval_mode == 'CrossSubjectEvaluation':
        # For CrossSubject, check for 3 fold-based output files
        # We need to generate expected fold sessions based on subject_list
        # For a 3-fold split, each fold uses 1/3 of subjects as eval
        n_subjects = len(subject_list)
        fold_size = n_subjects // 3
        
        # Generate expected fold sessions
        for fold_idx in range(3):
            eval_start = fold_idx * fold_size
            eval_end = eval_start + fold_size
            eval_subjects = subject_list[eval_start:eval_end]
            eval_subjects_str = ','.join(map(str, sorted(eval_subjects)))
            session = f"fold_{fold_idx}_eval_subjects_{eval_subjects_str}"
            
            # Use first eval subject as representative for path
            representative_subject = eval_subjects[0] if eval_subjects else subject_list[0]
            
            out_dir = create_output_path(model_name, seed, int(representative_subject), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset)
            if noise_type is not None and intensity is not None:
                filename_suffix = f"_{noise_type}_{intensity}"
            else:
                filename_suffix = ""
            out_file = os.path.join(out_dir,
                                    f"{model_name}_{mode}{filename_suffix}_{session}_seed{seed}.csv")
            if os.path.exists(out_file):
                existing_output_paths.append(out_file)
            else:
                expected_output_paths.append(out_file)
    else:
        # Original logic for WithinSession and CrossSession
        # sessions_to_check was already determined above
        for subj in subject_list:
            for session in sessions_to_check:
                # Determine paradigm and dataset for path creation
                out_dir = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset)
                
                # For test_perturb/multirun mode, check for ANY CSV file in the output directory
                # instead of a specific filename pattern (to handle legacy filename conventions)
                is_test_perturb_mode = mode in ['test_perturb', 'multirun'] or mode.startswith('test_perturb')
                if is_test_perturb_mode:
                    # Check if directory exists and has any CSV files
                    if os.path.exists(out_dir):
                        csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
                        if csv_files:
                            # Add all CSV files found in this directory
                            for csv_file in csv_files:
                                existing_output_paths.append(os.path.join(out_dir, csv_file))
                        else:
                            expected_output_paths.append(out_dir)  # Directory exists but no CSVs
                    else:
                        expected_output_paths.append(out_dir)  # Directory doesn't exist
                else:
                    # For non-test_perturb modes, use the original specific filename pattern
                    if noise_type is not None and intensity is not None:
                        filename_suffix = f"_{noise_type}_{intensity}"
                    else:
                        filename_suffix = ""
                    out_file = os.path.join(out_dir,
                                            f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
                    if os.path.exists(out_file):
                        existing_output_paths.append(out_file)
                    else:
                        expected_output_paths.append(out_file)

    # For test_perturb/multirun mode, we need to verify that the CSV files contain
    # all expected noise intensities, not just that the files exist
    # Note: mode might be 'test_perturb_tune' if tuning is enabled
    is_test_perturb_mode = mode in ['test_perturb', 'multirun'] or mode.startswith('test_perturb')
    
    # Debug output
    print(f"[check_skip_eval] Mode: {mode}, is_test_perturb_mode: {is_test_perturb_mode}")
    print(f"[check_skip_eval] Existing output paths: {len(existing_output_paths)}, Expected: {len(expected_output_paths)}")
    if existing_output_paths:
        print(f"[check_skip_eval] First existing file: {existing_output_paths[0]}")
    
    # For test_perturb mode, check intensity completeness if files exist
    if len(expected_output_paths) == 0 and is_test_perturb_mode:
        # Check if all expected noise intensities are present in the existing files
        all_intensities_present = True
        missing_intensities_info = []
        
        # Get expected noise intensities for all noise types
        noise_types = ['gaussian', 'dropout', 'eog', 'spike']
        expected_intensities_by_noise = {}
        
        # Load saturation file to get expected intensities
        saturation_file = "saturation_results/saturation_points_summary.csv"
        num_steps = 20  # Default from experiment_config.yaml
        
        for nt in noise_types:
            try:
                expected_intensities = get_noise_intensities(
                    dataset=dataset, 
                    noise_type=nt, 
                    num_steps=num_steps,
                    saturation_file=saturation_file
                )
                # Convert to Python float to avoid numpy/pandas type mismatches in comparison
                expected_intensities_by_noise[nt] = [float(x) for x in expected_intensities]
            except Exception as e:
                print(f"Warning: Could not get expected intensities for {nt}: {e}")
                # If we can't get expected intensities, assume we need to run
                all_intensities_present = False
                break
        
        # Check each existing output file to see if it contains all expected intensities
        # If ANY file has all expected intensities, we can skip the job
        print(f"[check_skip_eval] Checking {len(existing_output_paths)} existing file(s) for intensity completeness...")
        files_with_all_intensities = []
        
        for out_file in existing_output_paths:
            try:
                df = pd.read_csv(out_file)
                print(f"[check_skip_eval] Loaded {out_file}: {len(df)} rows, columns: {list(df.columns)}")
                
                # Check if the file has the required columns
                if 'noise_type' not in df.columns or 'intensity' not in df.columns:
                    print(f"[check_skip_eval] {out_file} missing noise_type or intensity columns, skipping this file")
                    continue
                
                # Check if this file has all expected intensities for all noise types
                file_has_all_intensities = True
                file_missing_info = []
                
                # Check each noise type
                for noise_type in noise_types:
                    if noise_type not in expected_intensities_by_noise:
                        continue
                    
                    # Get intensities present in the file for this noise type
                    noise_df = df[df['noise_type'] == noise_type]
                    if len(noise_df) == 0:
                        file_missing_info.append(f"{noise_type}: missing all results")
                        file_has_all_intensities = False
                        continue
                    
                    # Convert to float to avoid numpy/pandas type mismatches
                    existing_intensities = [float(x) for x in noise_df['intensity'].unique()]
                    expected_intensities = expected_intensities_by_noise[noise_type]
                    
                    print(f"[check_skip_eval] {out_file} - {noise_type}: Found {len(existing_intensities)} intensities, expected {len(expected_intensities)}")
                    
                    # Check if all expected intensities are present (with tolerance for floating point)
                    # Use 1e-3 tolerance to account for CSV round-trip precision loss and reasonable floating point differences
                    missing_intensities = []
                    for exp_int in expected_intensities:
                        found = False
                        closest_match = None
                        min_diff = float('inf')
                        for existing_int in existing_intensities:
                            diff = abs(exp_int - existing_int)
                            if diff < min_diff:
                                min_diff = diff
                                closest_match = existing_int
                            if diff < 1e-3:
                                found = True
                                break
                        if not found:
                            missing_intensities.append(exp_int)
                            # Debug: show closest match if it's close but not close enough
                            if min_diff < 1e-2:
                                print(f"  Debug: Expected intensity {exp_int} not found, closest match: {closest_match} (diff: {min_diff:.2e})")
                    
                    if missing_intensities:
                        file_missing_info.append(f"{noise_type}: missing {len(missing_intensities)} intensities")
                        file_has_all_intensities = False
                        missing_intensities_info.append(
                            f"{out_file}: {noise_type} missing intensities: {sorted(missing_intensities)[:5]}..." 
                            if len(missing_intensities) > 5 
                            else f"{out_file}: {noise_type} missing intensities: {sorted(missing_intensities)}"
                        )
                
                # If this file has all intensities, we can skip
                if file_has_all_intensities:
                    files_with_all_intensities.append(out_file)
                    print(f"[check_skip_eval] ✓ {out_file} contains all expected intensities")
                else:
                    print(f"[check_skip_eval] ✗ {out_file} missing intensities: {', '.join(file_missing_info)}")
                        
            except Exception as e:
                print(f"Warning: Could not read or verify {out_file}: {e}")
                import traceback
                traceback.print_exc()
                # Continue checking other files even if one fails
                continue
        
        # If any file has all intensities, we can skip
        all_intensities_present = len(files_with_all_intensities) > 0
        
        if all_intensities_present:
            print(f"[check_skip_eval] ✓ All expected intensities found in existing files - SKIPPING job")
            print(f"[check_skip_eval] Files with complete intensity data:")
            for out_file in files_with_all_intensities:
                print(f"  {out_file}")
            return True
        else:
            print(f"[check_skip_eval] ✗ Missing intensities detected - will RE-RUN job")
            print(f"[check_skip_eval] Missing intensity details:")
            for info in missing_intensities_info[:5]:  # Show first 5 missing items
                print(f"  {info}")
            if len(missing_intensities_info) > 5:
                print(f"  ... and {len(missing_intensities_info) - 5} more")
            return False
    
    if len(expected_output_paths) == 0:
        print(f"Skipping analysis, file(s) exist:")
        for out_file in existing_output_paths:
            print(out_file)
        return True
    return False


def log_all_subjects(results, subject_list, model_name, mode, noise_type, intensity, seed, eval_mode='CrossSession', paradigm='MotorImagery', dataset='BNCI2014_001'):
    """Log results for all subjects to individual CSV files."""
    # Handle CrossSubject mode differently - results are organized by folds, not individual subjects
    if eval_mode == 'CrossSubjectEvaluation':
        # For CrossSubject, iterate over unique fold sessions
        for session in results['session'].unique():
            session_df = results[results['session'] == session]
            
            # Extract fold information from session string (e.g., "fold_0_eval_subjects_1,2,3")
            if 'fold_' in session and 'eval_subjects' in session:
                # Get the representative subject (first eval subject) for file path
                representative_subject = session_df['subject'].iloc[0] if 'subject' in session_df.columns else subject_list[0]
            else:
                # Fallback if session format is unexpected
                representative_subject = session_df['subject'].iloc[0] if 'subject' in session_df.columns else subject_list[0]
            
            out_dir = create_output_path(model_name, seed, int(representative_subject), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset)
            os.makedirs(out_dir, exist_ok=True)
            
            if noise_type is not None and intensity is not None:
                filename_suffix = f"_{noise_type}_{intensity}" 
            else:
                filename_suffix = ""
            
            # For CrossSubject, session format is "fold_X_eval_subjects_Y"
            out_file = os.path.join(out_dir,
                                    f"{model_name}_{mode}{filename_suffix}_{session}_seed{seed}.csv")
            
            session_df.to_csv(out_file, index=False)
            print(f"Saved: {out_file}")
    else:
        # Original logic for WithinSession and CrossSession modes
        for subj in subject_list:        
            subject_df = results[results['subject'] == int(subj)]
            for session in subject_df['session'].unique():
                session_df = subject_df[subject_df['session'] == session]
                out_dir = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset)
                os.makedirs(out_dir, exist_ok=True)
                if noise_type is not None and intensity is not None:
                    filename_suffix = f"_{noise_type}_{intensity}" 
                else:
                    filename_suffix = ""

                out_file = os.path.join(out_dir,
                                        f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv")
                session_df.to_csv(out_file, index=False)
                print(f"Saved: {out_file}")


def two_stage_opt(dataset, subj, paradigm, model_name, model_fn, seed, mode, resample):
    """Run two-stage hyperparameter optimization using Optuna."""
    X, y, metadata = paradigm.get_data(dataset, subjects=[subj])
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    out_dir = create_output_path(model_name, seed, subj, '0train', mode)
    
    best_params, best_score = run_two_stage_optuna(
        model_fn=model_fn,
        model_name=model_name,
        X=X,
        y=y_encoded,
        metadata=metadata,
        resample=resample,
        seed=seed,
        output_root=os.path.join(out_dir, "optuna_results"),
        arch_trials=20,
        train_trials=20,
        perturbed=False
    )
    
    final_params = {}
    module_params = ['ncp_hidden_dim', 'sparsity', 'temporal_kernel_size', 'temporal_stride', 'drop_prob']
    optimizer_params = ['lr', 'weight_decay']
    prefix = ""
    module_prefix = f"{prefix}module__"
    optim_prefix = f"{prefix}optimizer__"
    
    for k, v in best_params.items():
        if k in module_params:
            final_params[f"{module_prefix}{k}"] = v
        elif k in optimizer_params:
            final_params[f"{optim_prefix}{k}"] = v
        else:
            final_params[k] = v

    return final_params, best_score


def collect_all_results(paradigm: str, dataset: str = "BNCI2014_001"):
    """Aggregate all CSV results from the results directory."""
    root1 = os.path.join("sol_results", paradigm, dataset)
    roots = [root1, os.path.join("results", paradigm, dataset)]
    all_dfs = []
    noise_types = ['gaussian', 'eog', 'dropout', 'spike']
    intensities = [str(x*10.0) for x in range(1, 10)]
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for file in filenames:
                if file.endswith(".csv"): # and not file.startswith("all_results"):
                    full_path = os.path.join(dirpath, file)
                    print(full_path)
                    try:
                        df = pd.read_csv(full_path)
                        selected_type = None
                        intensity = None
                        
                        # Only set noise_type and intensity if they don't already exist in the CSV
                        if 'noise_type' not in df.columns or 'intensity' not in df.columns:
                            for type in noise_types:
                                if type in file:
                                    selected_type = type
                                    for strength in intensities:
                                        if strength in file:
                                            intensity = strength
                                            break
                                    break
                            if selected_type is not None and intensity is not None:
                                if 'noise_type' not in df.columns:
                                    df['noise_type'] = selected_type
                                if 'intensity' not in df.columns:
                                    df['intensity'] = intensity

                        # Only set eval_mode if it doesn't already exist in the CSV
                        if 'eval_mode' not in df.columns:
                            if 'cross_session' in full_path or 'CrossSessionEvaluation' in full_path:
                                df['eval_mode'] = 'CrossSessionEvaluation'
                            elif 'cross_subject' in full_path or 'CrossSubjectEvaluation' in full_path:
                                df['eval_mode'] = 'CrossSubjectEvaluation'
                            else:
                                df['eval_mode'] = 'WithinSessionEvaluation'

                        # Detect whether this is a tuned experiment
                        # Priority 1: Check if 'tune' column exists in the CSV itself (most reliable)
                        if 'tune' in df.columns:
                            # Use the tune column to determine mode
                            # Get the first value (should be consistent across the CSV)
                            is_tuned = df['tune'].iloc[0] if len(df) > 0 else False
                            if is_tuned:
                                df['mode'] = 'test_perturb_tune'
                            else:
                                # Non-tuned: check if this is multirun or test_perturb
                                if 'mode' not in df.columns or pd.isna(df['mode'].iloc[0]):
                                    if 'multirun' in full_path:
                                        df['mode'] = 'multirun'
                                    elif 'test_perturb' in full_path or 'test_perturb' in file:
                                        df['mode'] = 'test_perturb'
                                    else:
                                        df['mode'] = 'test_perturb'  # Default for non-tuned
                                # If mode column exists and is not NA, keep the existing mode value
                        elif 'mode' not in df.columns:
                            # Priority 2: No 'tune' column, no 'mode' column - infer from path/filename
                            if 'test_perturb_tune' in full_path or '_tune' in file:
                                df['mode'] = 'test_perturb_tune'
                            elif 'test_perturb' in full_path:
                                df['mode'] = 'test_perturb'
                            elif 'multirun' in full_path:
                                df['mode'] = 'multirun'
                            else:
                                # Try to infer from filename
                                if 'test_perturb' in file:
                                    df['mode'] = 'test_perturb'
                                else:
                                    df['mode'] = 'unknown'

                        # Ensure dataset column is set
                        if 'dataset' not in df.columns:
                            df['dataset'] = dataset
                        
                        # Ensure paradigm column is set
                        if 'paradigm' not in df.columns:
                            df['paradigm'] = paradigm

                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Failed to read {full_path}: {e}")
    
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_file = os.path.join(root, "all_results.csv")
        full_df.to_csv(out_file, index=False)
        print(f"Aggregated results saved to: {out_file}")
        return full_df
    else:
        print("No CSV files found to aggregate.")
        return None


def collect_all_results_unified():
    """Collect and aggregate results from all datasets and paradigms."""
    all_results = []
    
    # Define all dataset-paradigm combinations
    dataset_paradigms = [
        ("MotorImagery", "BNCI2014_001"),
        ("SSVEP", "Lee2019_SSVEP")
    ]
    
    for paradigm, dataset in dataset_paradigms:
        print(f"\n=== Collecting results for {paradigm} - {dataset} ===")
        result_df = collect_all_results(paradigm, dataset)
        if result_df is not None:
            all_results.append(result_df)
    
    if all_results:
        # Combine all results into a single DataFrame
        unified_df = pd.concat(all_results, ignore_index=True)
        
        # Save unified results
        unified_file = os.path.join("evaluation", "results", "unified_all_results.csv")
        os.makedirs(os.path.dirname(unified_file), exist_ok=True)
        unified_df.to_csv(unified_file, index=False)
        print(f"\nUnified results saved to: {unified_file}")
        return unified_df
    else:
        print("No results found to aggregate.")
        return None


def add_experiment_metadata(df, model_name, seed, mode, resample, config):
    """Add standard experiment metadata to results dataframe."""
    df['seed'] = seed
    df['mode'] = mode
    df['model'] = model_name
    df['paradigm'] = 'MotorImagery'
    df['resample'] = resample or 250.0
    df['optimizer__lr'] = config['optimizer__lr']
    df['batch_size'] = config['batch_size']
    df['max_epochs'] = config['max_epochs']
    
    # Add model-specific parameters
    if model_name == 'cnn_ncp' or model_name == 'cnn_cfc':
        df['module__ncp_hidden_dim'] = config['module__ncp_hidden_dim']
        df['module__sparsity'] = config['module__sparsity']
        df['optimizer__weight_decay'] = config['optimizer__weight_decay']
    if model_name == 'reegnet':
        df['module__lstm_hidden_size'] = config['module__lstm_hidden_size']
        df['module__drop_prob'] = config['module__drop_prob']
    
    return df 