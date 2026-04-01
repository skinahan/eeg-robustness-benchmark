"""
Utility functions for EEG experiments.

This module contains helper functions used across different experiment types
to reduce code duplication and improve maintainability.
"""

import os
import json
from collections import Counter

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder

from utils import (
    create_output_path,
    create_hdf5_model_path,
    get_noise_intensities,
    get_short_session_id,
    short_run_id,
    _CORRELATED_NOISE_TYPES,
)
from evaluation.two_stage_hp_opt import run_two_stage_optuna


def extract_model_params(model) -> Dict[str, Any]:
    """Extract model parameters for logging purposes."""
    if hasattr(model, 'get_params'):
        return model.get_params()
    return {}


def check_skip_eval(model_name, seed, subject_list, mode, noise_type, intensity, eval_mode='CrossSession', paradigm='MotorImagery', dataset='BNCI2014_001', cache_manager=None, config=None, tuned=False, paradigm_obj=None, dataset_obj=None, expected_noise_types=None, expected_intensities_by_noise=None, test_perturb_num_steps=20, test_perturb_saturation_file=None, intensity_grid_base_dir=None):

    if not eval_mode.endswith("Evaluation"):
        eval_mode = f"{eval_mode}Evaluation"

    """Check if evaluation should be skipped based on existing output files and model cache."""
    
    # Validate that mode matches tuned parameter
    # If tuned=True, mode should contain "_tune", if tuned=False, mode should not contain "_tune"
    mode_has_tune = "_tune" in mode
    if tuned and not mode_has_tune:
        # If tuned is True but mode doesn't have "_tune", add it
        # This ensures tuned results use different paths than non-tuned results
        mode = f"{mode}_tune"
        mode_has_tune = True
    elif not tuned and mode_has_tune:
        # If tuned is False but mode has "_tune", this is inconsistent
        # Remove "_tune" from mode to match the tuned=False state
        mode = mode.replace("_tune", "")
        mode_has_tune = False
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
        
        # Check if we're in test_perturb mode (for directory-based file search)
        is_test_perturb_mode = mode in ['test_perturb', 'multirun'] or mode.startswith('test_perturb')
        
        # Generate expected fold sessions
        for fold_idx in range(3):
            eval_start = fold_idx * fold_size
            eval_end = eval_start + fold_size
            eval_subjects = subject_list[eval_start:eval_end]
            eval_subjects_str = ','.join(map(str, sorted(eval_subjects)))
            session = f"fold_{fold_idx}_eval_subjects_{eval_subjects_str}"
            
            # Use first eval subject as representative for path
            representative_subject = eval_subjects[0] if eval_subjects else subject_list[0]
            
            # Get short session identifier for new path format
            short_session = get_short_session_id(session, 'CrossSubject')
            
            # Short path (short run id + short session): used for new writes and primary check
            out_dir_short = create_output_path(model_name, seed, int(representative_subject), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset, use_short_run_id=True)
            # Long path (full model name): for backwards compatibility with existing results
            out_dir_long = create_output_path(model_name, seed, int(representative_subject), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset, use_short_run_id=False)
            
            # Create legacy path format (full session string in path) for backwards compatibility
            if not eval_mode.endswith("Evaluation"):
                eval_mode_full = f"{eval_mode}Evaluation"
            else:
                eval_mode_full = eval_mode
            subject_str = f"sub-{int(representative_subject):03d}"
            old_path_parts = [
                "results",
                paradigm,
                dataset,
                model_name,
                eval_mode_full,
                str(seed),
                subject_str,
                session,  # Old format uses full session string
                mode
            ]
            out_dir_old = os.path.join("//".join([str(item) for item in old_path_parts]))
            
            # For test_perturb/multirun mode, check for ANY CSV file in the output directory
            # Check short path, long path, and legacy path for backwards compatibility
            if is_test_perturb_mode:
                found_files = False
                for out_dir in (out_dir_short, out_dir_long, out_dir_old):
                    if os.path.exists(out_dir):
                        csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
                        if csv_files:
                            for csv_file in csv_files:
                                existing_output_paths.append(os.path.join(out_dir, csv_file))
                            found_files = True
                            break
                if not found_files:
                    expected_output_paths.append(out_dir_short)  # New writes use short path
            else:
                # For non-test_perturb modes, check both short and long path filename patterns
                if noise_type is not None and intensity is not None:
                    filename_suffix = f"_{noise_type}_{intensity}"
                else:
                    filename_suffix = ""
                
                out_file_short = os.path.join(out_dir_short,
                                              f"{model_name}_{mode}{filename_suffix}_{short_session}_seed{seed}.csv")
                out_file_long = os.path.join(out_dir_long,
                                             f"{model_name}_{mode}{filename_suffix}_{short_session}_seed{seed}.csv")
                out_file_old = os.path.join(out_dir_old,
                                            f"{model_name}_{mode}{filename_suffix}_{session}_seed{seed}.csv")
                
                if os.path.exists(out_file_short):
                    existing_output_paths.append(out_file_short)
                elif os.path.exists(out_file_long):
                    existing_output_paths.append(out_file_long)
                elif os.path.exists(out_file_old):
                    existing_output_paths.append(out_file_old)
                else:
                    expected_output_paths.append(out_file_short)
    else:
        # Original logic for WithinSession and CrossSession
        # sessions_to_check was already determined above
        is_test_perturb_mode = mode in ['test_perturb', 'multirun'] or mode.startswith('test_perturb')
        for subj in subject_list:
            for session in sessions_to_check:
                # Short path (used for new writes); long path for backwards compatibility
                out_dir_short = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset, use_short_run_id=True)
                out_dir_long = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset, use_short_run_id=False)
                
                if is_test_perturb_mode:
                    # Check short then long dir for any CSV files
                    found_files = False
                    for out_dir in (out_dir_short, out_dir_long):
                        if os.path.exists(out_dir):
                            csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
                            if csv_files:
                                for csv_file in csv_files:
                                    existing_output_paths.append(os.path.join(out_dir, csv_file))
                                found_files = True
                                break
                    if not found_files:
                        expected_output_paths.append(out_dir_short)
                else:
                    if noise_type is not None and intensity is not None:
                        filename_suffix = f"_{noise_type}_{intensity}"
                    else:
                        filename_suffix = ""
                    fname = f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv"
                    out_file_short = os.path.join(out_dir_short, fname)
                    out_file_long = os.path.join(out_dir_long, fname)
                    if os.path.exists(out_file_short):
                        existing_output_paths.append(out_file_short)
                    elif os.path.exists(out_file_long):
                        existing_output_paths.append(out_file_long)
                    else:
                        expected_output_paths.append(out_file_short)

    # For test_perturb/multirun mode, we need to verify that the CSV files contain
    # all expected noise intensities, not just that the files exist
    # Note: mode might be 'test_perturb_tune' if tuning is enabled
    is_test_perturb_mode = mode in ['test_perturb', 'multirun'] or mode.startswith('test_perturb')
    
    # Filter existing_output_paths to only include files that match the tuned state
    # If tuned=True, only keep files from directories with "_tune" in the path
    # If tuned=False, exclude files from directories with "_tune" in the path
    filtered_existing_paths = []
    for path in existing_output_paths:
        path_has_tune = "_tune" in path
        if tuned and path_has_tune:
            filtered_existing_paths.append(path)
        elif not tuned and not path_has_tune:
            filtered_existing_paths.append(path)
        # If there's a mismatch (tuned=True but path doesn't have "_tune", or vice versa), skip it
    existing_output_paths = filtered_existing_paths
    
    # Debug output
    print(f"[check_skip_eval] Mode: {mode}, Tuned: {tuned}, is_test_perturb_mode: {is_test_perturb_mode}")
    print(f"[check_skip_eval] Existing output paths (after tuned filter): {len(existing_output_paths)}, Expected: {len(expected_output_paths)}")
    if existing_output_paths:
        print(f"[check_skip_eval] First existing file: {existing_output_paths[0]}")
    
    # For test_perturb mode, check intensity completeness if files exist
    if len(expected_output_paths) == 0 and is_test_perturb_mode:
        # Check if all expected noise intensities are present in the existing files.
        # When expected_noise_types and expected_intensities_by_noise are provided (e.g. by Plot2
        # or runner with gaussian_only + alpha_grid), only require those; otherwise require
        # all four noise types with full saturation-step intensities (stricter legacy behavior).
        missing_intensities_info = []
        saturation_file = test_perturb_saturation_file or "saturation_results/saturation_points_summary.csv"
        sat_abs = _resolve_saturation_file(saturation_file, intensity_grid_base_dir)
        num_steps = int(test_perturb_num_steps)

        if expected_noise_types is not None and expected_intensities_by_noise is not None and len(expected_noise_types) > 0 and len(expected_intensities_by_noise) > 0:
            # Run-specific scope: only require these noise types and intensities (e.g. gaussian + alpha grid)
            noise_types = list(expected_noise_types)
            expected_by_noise = {}
            for nt in noise_types:
                if nt in expected_intensities_by_noise and expected_intensities_by_noise[nt]:
                    expected_by_noise[nt] = [float(x) for x in expected_intensities_by_noise[nt]]
            if not expected_by_noise:
                # Caller passed empty lists; fall back to legacy so we don't skip incorrectly
                noise_types = ['gaussian', 'dropout', 'eog', 'spike']
                expected_by_noise = {}
                for nt in noise_types:
                    try:
                        expected_intensities = get_effective_noise_intensities(
                            dataset=dataset,
                            noise_type=nt,
                            num_steps=num_steps,
                            base_dir=intensity_grid_base_dir,
                            saturation_file=sat_abs,
                        )
                        expected_by_noise[nt] = [float(x) for x in expected_intensities]
                    except Exception as e:
                        print(f"Warning: Could not get expected intensities for {nt}: {e}")
                        expected_by_noise[nt] = []
        else:
            # Legacy: require all four noise types and full saturation-step intensities
            noise_types = ['gaussian', 'dropout', 'eog', 'spike']
            expected_by_noise = {}
            for nt in noise_types:
                try:
                    expected_intensities = get_effective_noise_intensities(
                        dataset=dataset,
                        noise_type=nt,
                        num_steps=num_steps,
                        base_dir=intensity_grid_base_dir,
                        saturation_file=sat_abs,
                    )
                    expected_by_noise[nt] = [float(x) for x in expected_intensities]
                except Exception as e:
                    print(f"Warning: Could not get expected intensities for {nt}: {e}")
                    expected_by_noise[nt] = []

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
                    if noise_type not in expected_by_noise:
                        continue
                    
                    # Get intensities present in the file for this noise type
                    noise_df = df[df['noise_type'] == noise_type]
                    if len(noise_df) == 0:
                        file_missing_info.append(f"{noise_type}: missing all results")
                        file_has_all_intensities = False
                        continue
                    
                    # Convert to float to avoid numpy/pandas type mismatches
                    existing_intensities = [float(x) for x in noise_df['intensity'].unique()]
                    expected_intensities = expected_by_noise[noise_type]
                    
                    print(f"[check_skip_eval] {out_file} - {noise_type}: Found {len(existing_intensities)} intensities, expected {len(expected_intensities)}")
                    
                    # For correlated types (ar1_drift, spatial_gaussian, emg_band), actual intensities
                    # are data-derived (alpha * alpha_max), so we only require count match, not value match.
                    if noise_type in _CORRELATED_NOISE_TYPES:
                        if len(existing_intensities) < len(expected_intensities):
                            file_missing_info.append(f"{noise_type}: found {len(existing_intensities)} intensities, need {len(expected_intensities)}")
                            file_has_all_intensities = False
                            missing_intensities_info.append(
                                f"{out_file}: {noise_type} intensity count {len(existing_intensities)} < expected {len(expected_intensities)}"
                            )
                        continue
                    
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
                    print(f"[check_skip_eval] OK {out_file} contains all expected intensities")
                else:
                    print(f"[check_skip_eval] ERROR {out_file} missing intensities: {', '.join(file_missing_info)}")
                        
            except Exception as e:
                print(f"Warning: Could not read or verify {out_file}: {e}")
                import traceback
                traceback.print_exc()
                # Continue checking other files even if one fails
                continue
        
        # If any file has all intensities, we can skip
        all_intensities_present = len(files_with_all_intensities) > 0
        
        if all_intensities_present:
            print(f"[check_skip_eval] OK All expected intensities found in existing files - SKIPPING job")
            print(f"[check_skip_eval] Files with complete intensity data:")
            for out_file in files_with_all_intensities:
                print(f"  {out_file}")
            return True
        else:
            print(f"[check_skip_eval] ERROR Missing intensities detected - will RE-RUN job")
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
    """
    Log results for all subjects to individual CSV files.
    
    IMPORTANT: The 'mode' parameter determines the output directory and filename.
    - For non-tuned results: mode = 'test_perturb' (or other mode name)
    - For tuned results: mode = 'test_perturb_tune' (mode with '_tune' suffix)
    This ensures tuned and non-tuned results are saved to different paths and don't overwrite each other.
    """
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
            
            # CRITICAL: The 'mode' parameter is used in create_output_path to create different directories
            # for tuned vs non-tuned results. This prevents overwriting.
            # Path structure: results/.../{mode}/ where mode is either 'test_perturb' or 'test_perturb_tune'
            out_dir = create_output_path(model_name, seed, int(representative_subject), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset, use_short_run_id=True)
            os.makedirs(out_dir, exist_ok=True)
            
            if noise_type is not None and intensity is not None:
                filename_suffix = f"_{noise_type}_{intensity}"
            else:
                filename_suffix = ""
            
            # Use short session identifier for filename to avoid long paths
            short_session = get_short_session_id(session, 'CrossSubject')
            # Short filename for test_perturb to stay under Windows path limit.
            # Include short model id (8 chars) to prevent collision when multiple models share subject/seed.
            is_tp = mode in ('test_perturb', 'test_perturb_tune') or mode.startswith('test_perturb')
            if is_tp and not filename_suffix:
                mid = short_run_id(model_name, length=8)
                fname = f"tp_{mid}_{short_session}_seed{seed}.csv" if "_tune" not in mode else f"tp_tune_{mid}_{short_session}_seed{seed}.csv"
            else:
                fname = f"{model_name}_{mode}{filename_suffix}_{short_session}_seed{seed}.csv"
            out_file = os.path.join(out_dir, fname)
            
            # Log the full path construction for debugging
            print(f"[LOG_ALL_SUBJECTS] CrossSubject file save:")
            print(f"  Output directory: {out_dir}")
            print(f"  Mode: {mode} (this determines the subdirectory)")
            print(f"  Filename: {os.path.basename(out_file)}")
            print(f"  Full path: {out_file}")
            print(f"  Rows to save: {len(session_df)}")
            
            # Check if file already exists (to detect potential overwrites)
            if os.path.exists(out_file):
                print(f"  WARNING: File already exists and will be overwritten")
                # Read existing file to check if it's different
                try:
                    existing_df = pd.read_csv(out_file)
                    print(f"  Existing file has {len(existing_df)} rows")
                    if 'tune' in existing_df.columns and 'tune' in session_df.columns:
                        existing_tune = existing_df['tune'].iloc[0] if len(existing_df) > 0 else None
                        new_tune = session_df['tune'].iloc[0] if len(session_df) > 0 else None
                        if existing_tune != new_tune:
                            print(f"  ERROR: Tune flag mismatch! Existing: {existing_tune}, New: {new_tune}")
                            print(f"  This suggests a path collision between tuned and non-tuned results!")
                except Exception as e:
                    print(f"  Could not read existing file for comparison: {e}")
            
            try:
                session_df.to_csv(out_file, index=False)
                print(f"  OK Successfully saved: {out_file}")
            except Exception as e:
                print(f"  ERROR saving file: {e}")
                import traceback
                traceback.print_exc()
                raise
    else:
        # Original logic for WithinSession and CrossSession modes
        for subj in subject_list:
            subject_df = results[results['subject'] == int(subj)]
            for session in subject_df['session'].unique():
                session_df = subject_df[subject_df['session'] == session]
                out_dir = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset, use_short_run_id=True)
                os.makedirs(out_dir, exist_ok=True)
                if noise_type is not None and intensity is not None:
                    filename_suffix = f"_{noise_type}_{intensity}"
                else:
                    filename_suffix = ""
                # Short filename for test_perturb to avoid Windows path length limit.
                # Include short model id (8 chars) to prevent collision when multiple models
                # share the same subject/seed (e.g. orientation_sensitivity: random_oriented vs symmetric).
                is_tp = mode in ('test_perturb', 'test_perturb_tune') or mode.startswith('test_perturb')
                if is_tp and not filename_suffix:
                    mid = short_run_id(model_name, length=8)
                    fname = f"tp_{mid}_s{int(subj):03d}_seed{seed}.csv" if "_tune" not in mode else f"tp_tune_{mid}_s{int(subj):03d}_seed{seed}.csv"
                else:
                    fname = f"{model_name}_{mode}{filename_suffix}_subject_{int(subj):03d}_seed{seed}.csv"
                out_file = os.path.join(out_dir, fname)
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


# ============================================================================
# Intensity Filtering Functions (using saturation-based bounds)
# ============================================================================

def _repo_root_from_this_file() -> str:
    """Project root (parent of ``evaluation/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_allowed_noise_types_from_experiment_config(repo_root: str) -> Optional[List[str]]:
    """
    Load ``noise_types`` from ``experiment_config.yaml`` at repo root.

    Used to drop rows for perturbation types that are commented out / removed from the config
    when building aggregated result sets (e.g. ar1_drift present in saturation CSV but not in YAML).
    Returns None if the file or key is missing (no filtering).
    """
    path = os.path.join(repo_root, "experiment_config.yaml")
    if not os.path.isfile(path):
        return None
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not cfg or "noise_types" not in cfg:
            return None
        raw = cfg["noise_types"]
        if not isinstance(raw, list) or not raw:
            return None
        out = [str(x).strip() for x in raw if x is not None and str(x).strip()]
        return out if out else None
    except Exception:
        return None


def _filter_dataframe_to_allowed_noise_types(
    df: pd.DataFrame,
    allowed: Optional[List[str]],
    label: str,
) -> pd.DataFrame:
    if df is None or df.empty or not allowed:
        return df
    if "noise_type" not in df.columns:
        return df
    n0 = len(df)
    out = df[df["noise_type"].isin(allowed)].copy()
    if len(out) < n0:
        print(
            f"[INFO] {label}: kept {len(out)}/{n0} rows (noise_type in experiment_config.yaml: {allowed})"
        )
    return out


_SOL_GROUP_KEYS = ("model", "seed", "eval_mode", "mode", "session", "tuned", "tune")


def _sol_subframe_with_physical_intensity(
    sub: pd.DataFrame, noise_type: str
) -> Optional[pd.DataFrame]:
    """
    Rows with positive intensity, plus ``_phys`` column (physical scale).

    For non-correlated types, α∈(0,1] is mapped with global min/max from physical values (>1)
    in this noise slice (same convention as historical ``load_sol_results_intensity_grid``).
    """
    if sub.empty or "intensity" not in sub.columns:
        return None
    raw = pd.to_numeric(sub["intensity"], errors="coerce")
    mask = raw.notna() & (raw > 0)
    out = sub.loc[mask].copy()
    if out.empty:
        return None
    vals = raw.loc[mask].to_numpy(dtype=float)
    nt = str(noise_type)
    if nt in _CORRELATED_NOISE_TYPES:
        out["_phys"] = vals
        return out
    physical = vals[vals > 1.0]
    if physical.size == 0:
        return None
    min_i = float(np.min(physical))
    max_i = float(np.max(physical))
    phys = np.where(vals > 1.0, vals, min_i + vals * (max_i - min_i))
    out["_phys"] = phys.astype(float)
    return out


def load_sol_results_intensity_grid(
    dataset: str,
    noise_type: str,
    paradigm: str = "MotorImagery",
    base_dir: str = ".",
) -> Optional[np.ndarray]:
    """
    Infer allowed physical intensities from ``sol_results/{paradigm}/{dataset}/all_results.csv``.

    Returns sorted **unique** values (legacy helper). Prefer
    :func:`load_sol_results_dominant_intensity_grid` for automation/filtering.

    Returns None if the file is missing or has no usable intensities for this noise type.
    """
    path = os.path.join(base_dir, "sol_results", paradigm, dataset, "all_results.csv")
    if not os.path.isfile(path):
        return None
    try:
        sdf = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if "intensity" not in sdf.columns or "noise_type" not in sdf.columns:
        return None
    sub = sdf[sdf["noise_type"].astype(str) == str(noise_type)]
    out = _sol_subframe_with_physical_intensity(sub, noise_type)
    if out is None or out.empty:
        return None
    return np.sort(np.unique(np.asarray(out["_phys"], dtype=float)))


def load_sol_results_dominant_intensity_grid(
    dataset: str,
    noise_type: str,
    paradigm: str = "MotorImagery",
    base_dir: str = ".",
    rtol: float = 1e-5,
) -> Optional[np.ndarray]:
    """
    Infer the **dominant** intensity sweep from ``sol_results/.../all_results.csv``.

    Rows are mapped to physical ``_phys`` (same rules as :func:`load_sol_results_intensity_grid`).
    Then:

    * If grouping columns exist (``model``, ``seed``, ``eval_mode``, ``mode``, ``session``,
      ``tuned``/``tune`` — whichever are present), each group yields a multiset signature
      ``tuple(sorted(unique rounded intensities))``. The signature with the highest count wins;
      ties break toward the **longest** signature (more complete sweep).

    * Otherwise, **marginal** counts per rounded intensity: keep values with
      ``count >= 0.5 * max_count`` (drops rare one-off intensities from merged history).

    Returns sorted unique intensities for the dominant set, or None if unavailable.
    """
    path = os.path.join(base_dir, "sol_results", paradigm, dataset, "all_results.csv")
    if not os.path.isfile(path):
        return None
    try:
        sdf = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if "intensity" not in sdf.columns or "noise_type" not in sdf.columns:
        return None
    sub = sdf[sdf["noise_type"].astype(str) == str(noise_type)]
    out = _sol_subframe_with_physical_intensity(sub, noise_type)
    if out is None or out.empty:
        return None

    phys = np.asarray(out["_phys"], dtype=float)

    group_keys = [k for k in _SOL_GROUP_KEYS if k in out.columns]
    if group_keys:
        sig_counter = Counter()
        for _, g in out.groupby(group_keys, dropna=False):
            gv = np.asarray(g["_phys"], dtype=float)
            if gv.size == 0:
                continue
            gr = np.round(np.round(gv / rtol) * rtol, 8)
            sig = tuple(np.sort(np.unique(gr)))
            if len(sig) == 0:
                continue
            sig_counter[sig] += 1
        if not sig_counter:
            return None
        max_c = max(sig_counter.values())
        tied = [sig for sig, c in sig_counter.items() if c == max_c]
        best = max(tied, key=len)
        return np.asarray(best, dtype=float)

    rounded = np.round(np.round(phys / rtol) * rtol, 8)
    cnt = Counter(rounded.tolist())
    max_c = max(cnt.values())
    threshold = max_c * 0.5
    dom = sorted({float(v) for v, n in cnt.items() if n >= threshold})
    return np.asarray(dom, dtype=float) if dom else None


def get_effective_noise_intensities(
    dataset: str,
    noise_type: str,
    num_steps: int = 20,
    saturation_file: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Intensity grid for filtering, skip checks, test_perturb sweeps, and automation.

    Prefer the **dominant** sweep from ``sol_results/MotorImagery/{dataset}/all_results.csv``
    when present (see :func:`load_sol_results_dominant_intensity_grid`); otherwise fall back
    to ``utils.get_noise_intensities`` (saturation CSV).

    Search order for ``sol_results``: ``base_dir`` if given, then ``os.getcwd()``, then
    project root next to ``evaluation/``.
    """
    sat = saturation_file or "saturation_results/saturation_points_summary.csv"
    candidates: List[str] = []
    if base_dir:
        candidates.append(base_dir)
    candidates.append(os.getcwd())
    candidates.append(_repo_root_from_this_file())
    seen: set[str] = set()
    ordered = []
    for b in candidates:
        if b and b not in seen:
            seen.add(b)
            ordered.append(b)
    for b in ordered:
        grid = load_sol_results_dominant_intensity_grid(dataset, noise_type, base_dir=b)
        if grid is not None and len(grid) > 0:
            return grid.astype(float)
    sat_resolved = _resolve_saturation_file(sat, base_dir)
    return get_noise_intensities(
        dataset, noise_type, num_steps=num_steps, saturation_file=sat_resolved
    )


def get_union_perturbation_intensity_grid(
    dataset: str,
    noise_type: str,
    num_steps: int = 20,
    base_dir: Optional[str] = None,
    saturation_file: Optional[str] = None,
) -> np.ndarray:
    """
    Backward-compatible alias for :func:`get_effective_noise_intensities`.

    The name is historical. The grid is **not** a union of saturation linspace and sol:
    merging those produced linspace-only "expected" intensities that real multirun jobs
    (which follow sol when present) could never satisfy, so completeness checks always failed.

    Contract: prefer dominant sweep intensities from ``sol_results/.../all_results.csv`` when
    present; otherwise fall back to saturation ``get_noise_intensities`` linspace.
    """
    sat_rel = saturation_file or "saturation_results/saturation_points_summary.csv"
    sat_abs = _resolve_saturation_file(sat_rel, base_dir)
    return get_effective_noise_intensities(
        dataset,
        str(noise_type),
        num_steps=num_steps,
        saturation_file=sat_abs,
        base_dir=base_dir,
    )


def intensity_matches(intensity_values, target_intensities, rtol=1e-5, atol=1e-8):
    """
    Check if intensity values match target intensities using tolerance-based comparison.
    
    This handles floating-point precision issues where values like 28.263157894736842
    and 28.26315789473684 should be considered equal.
    
    Parameters:
    -----------
    intensity_values : array-like or pd.Series
        The intensity values to check (can be a pandas Series or numpy array)
    target_intensities : array-like
        The target intensity values to match against
    rtol : float
        Relative tolerance for comparison (default: 1e-5)
    atol : float
        Absolute tolerance for comparison (default: 1e-8)
    
    Returns:
    --------
    boolean array or pd.Series
        True for each intensity value that matches any target intensity.
        Returns a pandas Series if input is a Series (preserves index), otherwise numpy array.
    """
    is_series = isinstance(intensity_values, pd.Series)
    original_index = intensity_values.index if is_series else None
    
    intensity_values = np.asarray(intensity_values)
    target_intensities = np.asarray(target_intensities)
    
    # Use broadcasting to compare all intensity_values against all target_intensities
    # Shape: (len(intensity_values), len(target_intensities))
    differences = np.abs(intensity_values[:, np.newaxis] - target_intensities[np.newaxis, :])
    
    # For each intensity value, check if it's close to any target intensity
    # Use both relative and absolute tolerance
    matches = np.any(
        (differences <= atol) | (differences <= rtol * np.abs(target_intensities)),
        axis=1
    )
    
    # Return as Series if input was Series (preserves index alignment)
    if is_series:
        return pd.Series(matches, index=original_index)
    return matches


def _coerce_alpha_stored_as_physical_intensity(
    noise_df: pd.DataFrame,
    correct_intensities: np.ndarray,
    noise_type: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Map legacy/normalized rows where α∈[0,1] was written to `intensity` instead of physical values.

    Canonical physical sweeps use ``np.linspace(min, max, n)`` from ``get_noise_intensities``; the same
    index grid is sometimes logged as ``np.linspace(0, 1, n)`` (α). The relationship is linear:
    ``physical = min + α * (max - min)`` (same as numpy linspace endpoints).

    Current ``unified_experiment_runner._evaluate_perturb`` logs physical intensities; mixed CSVs
    (e.g. merged ``all_results``) can still contain older or external rows with α in the intensity column.

    Does not alter correlated-noise types whose physical grid is already in [0, 1] from saturation logic.
    """
    if noise_df.empty or noise_type in _CORRELATED_NOISE_TYPES:
        return noise_df, 0
    min_i = float(np.min(correct_intensities))
    max_i = float(np.max(correct_intensities))
    if max_i <= min_i or min_i < 1.0 - 1e-9:
        return noise_df, 0

    out = noise_df.copy()
    n_coerced = 0
    for idx in out.index:
        val = out.at[idx, "intensity"]
        if pd.isna(val):
            continue
        v = float(val)
        if v <= 0.0 or v > 1.0:
            continue
        if bool(intensity_matches(pd.Series([v]), correct_intensities).iloc[0]):
            continue
        cand = min_i + v * (max_i - min_i)
        if bool(intensity_matches(pd.Series([cand]), correct_intensities).iloc[0]):
            out.at[idx, "intensity"] = cand
            n_coerced += 1
    return out, n_coerced


def _resolved_saturation_csv_path(base_dir: Optional[str]) -> str:
    root = base_dir if base_dir else _repo_root_from_this_file()
    return os.path.join(root, "saturation_results", "saturation_points_summary.csv")


def _resolve_saturation_file(saturation_file: str, base_dir: Optional[str]) -> str:
    if os.path.isabs(saturation_file):
        return saturation_file
    root = base_dir if base_dir else _repo_root_from_this_file()
    return os.path.join(root, saturation_file)


def filter_by_intensity_bounds(
    df: pd.DataFrame,
    dataset: str,
    num_steps: int = 20,
    base_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter DataFrame to rows whose intensities appear on the effective perturbation grid.

    Uses :func:`get_effective_noise_intensities` (sol-preferred when ``sol_results`` exists),
    aligned with ``unified_experiment_runner`` and ``experiment_automation``.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with 'intensity' and 'noise_type' columns
    dataset : str
        Dataset name (e.g., 'BNCI2014_001', 'Lee2019_SSVEP', 'BI2015a')
    num_steps : int
        Passed through when falling back to saturation-based linspace (default: 20)
    base_dir : str, optional
        Directory containing ``sol_results/`` (defaults: cwd + project root search)
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with only valid intensities
    """
    if df.empty:
        return df
    
    # Check if intensity column exists
    if 'intensity' not in df.columns:
        # No intensity column - return as-is (might be baseline/clean data)
        return df
    
    if 'noise_type' not in df.columns:
        # No noise_type column - can't filter, return as-is
        return df
    
    # Convert intensity to float to handle string representations
    # This is important because intensities might be stored as strings in CSV files
    df = df.copy()
    df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
    
    # Get correct intensities for each noise type
    filtered_rows = []
    
    # Get unique noise types in the data
    noise_types = df['noise_type'].dropna().unique()
    
    # Track bounds used for logging
    bounds_used = {}
    
    for noise_type in noise_types:
        if pd.isna(noise_type) or noise_type == '':
            # Keep rows with missing/noise_type (might be clean baseline)
            noise_mask = df['noise_type'].isna() | (df['noise_type'] == '')
            filtered_rows.append(df[noise_mask])
            continue
        
        # Get correct intensity range for this SPECIFIC dataset/noise_type combination
        # This is critical - bounds vary by both dataset and noise_type
        try:
            correct_intensities = get_effective_noise_intensities(
                dataset=dataset,
                noise_type=str(noise_type),
                num_steps=num_steps,
                base_dir=base_dir,
            )
            
            # Store bounds for logging
            min_intensity = float(np.min(correct_intensities))
            max_intensity = float(np.max(correct_intensities))
            bounds_used[noise_type] = (min_intensity, max_intensity)
            
        except Exception as e:
            print(f"[WARNING] Could not get intensity bounds for {dataset}/{noise_type}: {e}")
            print(f"         Keeping all intensities for this noise type (may include invalid values)")
            # If we can't get bounds, keep all rows for this noise type
            # This is a fallback - ideally we should have bounds for all combinations
            noise_mask = df['noise_type'] == noise_type
            filtered_rows.append(df[noise_mask])
            continue
        
        # Filter rows for this noise type
        noise_mask = df['noise_type'] == noise_type
        noise_df = df[noise_mask].copy()
        
        if noise_df.empty:
            continue

        noise_df, n_alpha_coerced = _coerce_alpha_stored_as_physical_intensity(
            noise_df, correct_intensities, str(noise_type)
        )
        if n_alpha_coerced > 0:
            print(
                f"[INFO] {dataset}/{noise_type}: Coerced {n_alpha_coerced} rows from alpha in (0,1] "
                f"to physical intensities (min={min_intensity:.4g}, max={max_intensity:.4g})"
            )
        
        # Get intensity series for this noise type
        intensity_series = noise_df['intensity']
        
        # Include:
        # 1. Rows with intensity == 0.0 (baseline/clean)
        # 2. Rows with intensity matching correct_intensities (within tolerance)
        # 3. Rows with missing intensity (might be clean baseline)
        
        # Check for intensity == 0.0 (baseline)
        baseline_mask = (intensity_series == 0.0) | intensity_series.isna()
        
        # Check for intensity matching correct_intensities (tolerance-based)
        # Only check non-baseline intensities
        non_baseline_mask = ~baseline_mask
        intensity_mask = pd.Series(False, index=intensity_series.index)
        if non_baseline_mask.any():
            intensity_mask.loc[non_baseline_mask] = intensity_matches(
                intensity_series.loc[non_baseline_mask], 
                correct_intensities
            )
        
        # Combine: keep baseline OR matching intensities
        keep_mask = baseline_mask | intensity_mask
        
        # Apply mask
        filtered_noise_df = noise_df[keep_mask]
        
        # Log filtering results
        if len(filtered_noise_df) < len(noise_df):
            filtered_count = len(noise_df) - len(filtered_noise_df)
            print(f"[INFO] {dataset}/{noise_type}: Filtered {filtered_count} rows with invalid intensities "
                  f"(kept {len(filtered_noise_df)}/{len(noise_df)} rows, "
                  f"bounds: {min_intensity:.1f}-{max_intensity:.1f})")
        else:
            print(f"[INFO] {dataset}/{noise_type}: All {len(filtered_noise_df)} rows have valid intensities "
                  f"(bounds: {min_intensity:.1f}-{max_intensity:.1f})")
        
        filtered_rows.append(filtered_noise_df)
    
    # Log summary of bounds used
    if bounds_used:
        print(f"[INFO] Intensity bounds used for {dataset}:")
        for noise_type, (min_val, max_val) in sorted(bounds_used.items()):
            print(f"       {noise_type:12s}: {min_val:5.1f} - {max_val:5.1f}")
    
    # Combine all filtered rows
    if filtered_rows:
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
        return filtered_df
    else:
        return pd.DataFrame()


# ============================================================================
# Deduplication Helper Functions (following DEDUPLICATION_SCHEMA.md)
# ============================================================================

def normalize_eval_mode(eval_mode) -> str:
    """Normalize eval_mode to canonical form."""
    if pd.isna(eval_mode) or eval_mode == '':
        return 'UnknownEvaluation'
    eval_mode = str(eval_mode).strip()
    if not eval_mode.endswith('Evaluation'):
        eval_mode = eval_mode + 'Evaluation'
    return eval_mode


# After stripping a trailing ``_tune``, these modes all log the same perturbation-sweep rows.
# Cluster batch jobs typically use ``multirun``; interactive runs often use ``test_perturb``.
PERTURB_SWEEP_MODES_NORMALIZED = frozenset({"test_perturb", "multirun"})


def canonicalize_perturb_sweep_mode(mode):
    """
    Map stored CLI modes ``multirun`` / ``multirun_tune`` to ``test_perturb`` / ``test_perturb_tune``.

    Used so aggregation, deduplication, and analysis all see the same logical mode for the
    same perturbation sweep (avoids duplicate signatures for multirun vs test_perturb rows).
    """
    if pd.isna(mode) or mode == "":
        return mode
    s = str(mode).strip()
    low = s.lower()
    if low == "multirun":
        return "test_perturb"
    if low == "multirun_tune":
        return "test_perturb_tune"
    return mode


def apply_perturb_sweep_mode_canonicalization(
    df: pd.DataFrame,
    log_label: str = "",
) -> pd.DataFrame:
    """
    Copy ``df`` and replace ``mode`` multirun -> test_perturb (and tuned variants).

    Call **before** any filter that keeps only ``mode == 'test_perturb'`` (otherwise multirun
    rows are dropped first). Also use before ``drop_duplicates`` / signature creation so
    equivalent sweep rows merge.
    """
    if df is None or df.empty or "mode" not in df.columns:
        return df
    out = df.copy()
    as_str = out["mode"].astype(str)
    n = 0
    for old, new in (("multirun", "test_perturb"), ("multirun_tune", "test_perturb_tune")):
        sel = as_str.str.strip().str.lower() == old
        if sel.any():
            out.loc[sel, "mode"] = new
            n += int(sel.sum())
    if n and log_label:
        print(
            f"[INFO] {log_label}: canonicalized mode for {n} row(s) "
            f"(multirun -> test_perturb for perturbation sweep parity)"
        )
    return out


def perturb_sweep_mode_mask(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask for rows whose ``mode`` is a perturbation sweep (``test_perturb`` or ``multirun``).

    Compares after stripping a ``_tune`` suffix from the ``mode`` string. Use when filtering
    raw CSVs that may not have been canonicalized yet.
    """
    if "mode" not in df.columns:
        return pd.Series(False, index=df.index)
    m = df["mode"].astype(str).str.replace("_tune", "", regex=False).str.strip().str.lower()
    return m.isin(PERTURB_SWEEP_MODES_NORMALIZED)


def normalize_mode(mode) -> str:
    """Normalize mode to canonical form (multirun is treated as test_perturb for signatures)."""
    if pd.isna(mode) or mode == "":
        return "unknown"
    c = canonicalize_perturb_sweep_mode(mode)
    if pd.isna(c) or c == "":
        return "unknown"
    return str(c).strip().lower()


def normalize_bool(value) -> bool:
    """Normalize to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 't')
    return False


def normalize_float(value) -> float:
    """Normalize to float, rounding to 3 decimal places for intensity precision."""
    if pd.isna(value):
        return 0.0
    return round(float(value), 3)


def create_experiment_signature(row: pd.Series) -> tuple:
    """
    Create a unique signature tuple for an experiment row.
    
    This function implements the signature schema defined in DEDUPLICATION_SCHEMA.md.
    Two rows with the same signature are considered duplicates.
    """
    # Normalize core fields
    dataset = str(row.get('dataset', '')).strip()
    model = str(row.get('model', '')).strip()
    eval_mode = normalize_eval_mode(row.get('eval_mode', ''))
    seed = int(row.get('seed', 0))
    mode = normalize_mode(row.get('mode', ''))
    tune = normalize_bool(row.get('tune', False))
    noise_type = str(row.get('noise_type', '')).strip()
    intensity = normalize_float(row.get('intensity', 0.0))
    
    # Determine subject/session key based on eval_mode
    if eval_mode == "CrossSubjectEvaluation":
        if pd.notna(row.get('eval_subjects')):
            subject_key = f"eval_subjects_{row['eval_subjects']}"
        elif pd.notna(row.get('session')):
            subject_key = f"session_{row['session']}"
        else:
            subject_key = "no_subject"
        return (dataset, model, eval_mode, seed, mode, tune, noise_type, intensity, subject_key)
    else:
        # WithinSession or CrossSession
        subject = int(row.get('subject', 0)) if pd.notna(row.get('subject')) else 0
        session = str(row.get('session', '')).strip() if pd.notna(row.get('session')) else ''
        return (dataset, model, eval_mode, seed, mode, tune, noise_type, intensity, subject, session)


def collect_from_directory(root_dir: str, source: str, paradigm: str, dataset: str) -> pd.DataFrame:
    """
    Collect all CSV files from a directory (read-only operation).
    
    This function only READS files. No files are modified.
    
    Args:
        root_dir: Root directory to walk through
        source: Source identifier ('sol_results' or 'results')
        paradigm: Paradigm name for metadata
        dataset: Dataset name for metadata
        
    Returns:
        DataFrame with all collected results, with 'source' column added
    """
    all_dfs = []
    noise_types = ['gaussian', 'eog', 'dropout', 'spike']
    intensities = [str(x*10.0) for x in range(1, 10)]
    
    if not os.path.exists(root_dir):
        return pd.DataFrame()
    
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith(".csv"):
                    # Exclude all_results.csv files when collecting from 'results' directory
                    # Allow them when collecting from 'sol_results' (curated primary source)
                    if file == "all_results.csv" and source != "sol_results":
                        continue
                    
                    full_path = os.path.join(dirpath, file)
                    
                    # Check path length (Windows has 260 char limit by default)
                    if len(full_path) > 260:
                        print(f"[WARNING] Path too long ({len(full_path)} chars), may fail to read: {full_path[:100]}...")
                    
                    try:
                        df = pd.read_csv(full_path)
                        if df.empty:
                            continue
                            
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
                                if 'mode' not in df.columns or pd.isna(df['mode'].iloc[0]):
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
                        
                        # Add source tracking
                        df['source'] = source
                        
                        all_dfs.append(df)
                    except Exception as e:
                        # Provide more informative error message for path length issues
                        error_msg = str(e)
                        if len(full_path) > 260:
                            print(f"[ERROR] Failed to read file (path too long, {len(full_path)} chars): {full_path[:100]}...")
                            print(f"        Error: {error_msg}")
                            print(f"        Note: This file uses the old long path format. New files will use shorter paths.")
                        else:
                            print(f"[ERROR] Failed to read {full_path}: {error_msg}")
    except Exception as e:
        print(f"[ERROR] Failed to walk directory {root_dir}: {e}")
        return pd.DataFrame()
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = apply_perturb_sweep_mode_canonicalization(
            combined,
            log_label=f"collect_from_directory({source}/{paradigm}/{dataset})",
        )
        # Save combined dataframe to all_results.csv *IF* source is not 'sol_results'
        if source != 'sol_results':
            combined.to_csv(os.path.join(root_dir, "all_results.csv"), index=False)
        return combined
    return pd.DataFrame()


def collect_all_results(
    paradigm: str,
    dataset: str = "BNCI2014_001",
    base_dir: Optional[str] = None,
    allowed_noise_types: Optional[List[str]] = None,
):
    """
    Aggregate all CSV results from both sol_results and results directories.
    
    This function implements the deduplication schema from DEDUPLICATION_SCHEMA.md:
    - Phase 1: Collect from sol_results (primary source)
    - Phase 2: Collect from results (secondary source), filtering out duplicates
    - All operations are read-only on source files
    
    Args:
        paradigm: Paradigm name (e.g., "MotorImagery")
        dataset: Dataset name (e.g., "BNCI2014_001")
        base_dir: Project root containing sol_results/ and results/
        allowed_noise_types: If set (e.g. from experiment_config.yaml), drop rows whose
            ``noise_type`` is not in this list before intensity filtering.
        
    Returns:
        DataFrame with deduplicated results, or None if no results found
    """
    all_dfs = []
    root = base_dir if base_dir is not None else os.getcwd()
    if allowed_noise_types is None:
        cfg_root = base_dir if base_dir is not None else _repo_root_from_this_file()
        allowed_noise_types = load_allowed_noise_types_from_experiment_config(cfg_root)

    # Phase 1: Collect from sol_results (primary source) - READ ONLY
    sol_results_dir = os.path.join(root, "sol_results", paradigm, dataset)
    print(f"[INFO] Phase 1: Collecting from sol_results/{paradigm}/{dataset} (read-only)")
    sol_results = collect_from_directory(sol_results_dir, source="sol_results", paradigm=paradigm, dataset=dataset)
    
    # Build index of existing signatures from sol_results
    sol_signatures = set()
    if sol_results is not None and not sol_results.empty:
        sol_results = _filter_dataframe_to_allowed_noise_types(
            sol_results, allowed_noise_types, f"sol_results/{paradigm}/{dataset}"
        )
    if sol_results is not None and not sol_results.empty:
        # STEP 1: Filter by intensity bounds BEFORE creating signatures
        # This ensures we only process experiments that match our defined experimental parameters
        print(f"[INFO] Filtering sol_results by effective intensity grid (sol_results all_results.csv when present, else saturation)...")
        before_intensity_filter = len(sol_results)
        sol_results = filter_by_intensity_bounds(sol_results, dataset=dataset, num_steps=20, base_dir=base_dir)
        after_intensity_filter = len(sol_results)
        
        if before_intensity_filter != after_intensity_filter:
            filtered_count = before_intensity_filter - after_intensity_filter
            print(f"[INFO] Intensity filtering removed {filtered_count} rows from sol_results "
                  f"(kept {after_intensity_filter}/{before_intensity_filter} rows)")
        
        if not sol_results.empty:
            # STEP 2: Create signatures for deduplication (after intensity filtering)
            sol_results['experiment_signature'] = sol_results.apply(create_experiment_signature, axis=1)
            
            # Handle inter-dataset duplicates within sol_results - keep only one unique experiment per signature
            sol_results = sol_results.drop_duplicates(subset=['experiment_signature'], keep='first')
            
            sol_signatures = set(sol_results['experiment_signature'].unique())
            all_dfs.append(sol_results)
            print(f"[INFO] Found {len(sol_results)} rows in sol_results ({len(sol_signatures)} unique experiments)")
        else:
            print(f"[INFO] No valid results remaining in sol_results after intensity filtering")
    else:
        print(f"[INFO] No results found in sol_results/{paradigm}/{dataset}")
    
    # Phase 2: Collect from results (secondary source, with deduplication) - READ ONLY
    results_dir = os.path.join(root, "results", paradigm, dataset)
    print(f"[INFO] Phase 2: Collecting from results/{paradigm}/{dataset} (read-only)")
    if os.path.exists(results_dir):
        results_df = collect_from_directory(results_dir, source="results", paradigm=paradigm, dataset=dataset)
        if results_df is not None and not results_df.empty:
            results_df = _filter_dataframe_to_allowed_noise_types(
                results_df, allowed_noise_types, f"results/{paradigm}/{dataset}"
            )
        if results_df is not None and not results_df.empty:
            # STEP 1: Filter by intensity bounds BEFORE creating signatures
            # This ensures we only process experiments that match our defined experimental parameters
            print(f"[INFO] Filtering results by effective intensity grid (sol_results all_results.csv when present, else saturation)...")
            before_intensity_filter = len(results_df)
            results_df = filter_by_intensity_bounds(results_df, dataset=dataset, num_steps=20, base_dir=base_dir)
            after_intensity_filter = len(results_df)
            
            if before_intensity_filter != after_intensity_filter:
                filtered_count = before_intensity_filter - after_intensity_filter
                print(f"[INFO] Intensity filtering removed {filtered_count} rows from results "
                      f"(kept {after_intensity_filter}/{before_intensity_filter} rows)")
            
            if not results_df.empty:
                # STEP 2: Create signatures for deduplication (after intensity filtering)
                results_df['experiment_signature'] = results_df.apply(create_experiment_signature, axis=1)
                
                # STEP 3: Filter out duplicates that already exist in sol_results (in memory only)
                before_filter = len(results_df)
                results_df = results_df[~results_df['experiment_signature'].isin(sol_signatures)]
                after_filter = len(results_df)
                filtered_count = before_filter - after_filter
                
                if filtered_count > 0:
                    print(f"[INFO] Filtered {filtered_count} duplicate rows from results (already in sol_results)")
                
                if not results_df.empty:
                    all_dfs.append(results_df)
                    print(f"[INFO] Added {len(results_df)} unique rows from results")
                else:
                    print(f"[INFO] No unique results to add from results directory (all duplicates)")
            else:
                print(f"[INFO] No valid results remaining in results after intensity filtering")
        else:
            print(f"[INFO] No results found in results/{paradigm}/{dataset}")
    else:
        print(f"[INFO] results/{paradigm}/{dataset} directory does not exist")
    
    # Combine all collected results
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Final deduplication with source priority (sol_results > results)
        # Note: Intensity filtering already done per-source before signature creation
        if 'source' not in combined.columns:
            combined['source'] = 'unknown'
        
        # Create priority order (sol_results = 0, results = 1, unknown = 2)
        source_priority = {'sol_results': 0, 'results': 1, 'unknown': 2}
        combined['source_priority'] = combined['source'].map(source_priority).fillna(2)
        
        # Sort by priority (sol_results first)
        combined = combined.sort_values('source_priority', ascending=True)
        
        # Drop duplicates, keeping first (highest priority) - all in memory
        before_final = len(combined)
        combined = combined.drop_duplicates(subset=['experiment_signature'], keep='first')
        after_final = len(combined)
        
        if before_final != after_final:
            print(f"[INFO] Final deduplication removed {before_final - after_final} additional duplicates")
        
        # Drop temporary columns
        combined = combined.drop(columns=['source_priority'], errors='ignore')
        
        # Remove experiment_signature column before returning (it's just for deduplication)
        combined = combined.drop(columns=['experiment_signature'], errors='ignore')
        
        print(f"[INFO] Final aggregated: {len(combined)} rows")
        if 'source' in combined.columns:
            source_counts = combined['source'].value_counts()
            print(f"[INFO] Source breakdown: {dict(source_counts)}")
        
        return combined
    else:
        print(f"[INFO] No CSV files found to aggregate for {paradigm} - {dataset}")
        return None


def collect_all_results_unified(base_dir: Optional[str] = None):
    """
    Collect and aggregate results from all datasets and paradigms.
    
    This function implements the unified collection strategy from DEDUPLICATION_SCHEMA.md:
    - Collects from all paradigm-dataset combinations
    - Performs deduplication across all datasets
    - Writes to a new unified output file (non-destructive - original files remain untouched)
    
    base_dir : str, optional
        Project root containing ``sol_results/`` and ``results/``. Defaults to repo root
        (parent of ``evaluation/``).
    
    Returns:
        DataFrame with unified, deduplicated results, or None if no results found
    """
    all_results = []
    root = base_dir if base_dir is not None else _repo_root_from_this_file()
    allowed_nt = load_allowed_noise_types_from_experiment_config(root)
    if allowed_nt:
        print(f"[INFO] Aggregating with experiment_config.yaml noise_types filter: {allowed_nt}")
    
    # Define all dataset-paradigm combinations
    dataset_paradigms = [
        ("MotorImagery", "BNCI2014_001"),
        ("MotorImagery", "Lee2019_MI"),
        ("MotorImagery", "Shin2017A"),
        ("SSVEP", "Lee2019_SSVEP"),
        ("ERP", "BI2015a"),
    ]
    
    for paradigm, dataset in dataset_paradigms:
        print(f"\n=== Collecting results for {paradigm} - {dataset} ===")
        result_df = collect_all_results(paradigm, dataset, base_dir=root, allowed_noise_types=allowed_nt)
        if result_df is not None and not result_df.empty:
            all_results.append(result_df)
    
    if all_results:
        # Combine all results into a single DataFrame
        unified_df = pd.concat(all_results, ignore_index=True)
        
        # Perform cross-dataset deduplication using signatures
        # (in case the same experiment appears in multiple datasets)
        print(f"\n[INFO] Performing cross-dataset deduplication...")
        unified_df['experiment_signature'] = unified_df.apply(create_experiment_signature, axis=1)
        
        # Ensure source priority is maintained
        if 'source' not in unified_df.columns:
            unified_df['source'] = 'unknown'
        
        source_priority = {'sol_results': 0, 'results': 1, 'unknown': 2}
        unified_df['source_priority'] = unified_df['source'].map(source_priority).fillna(2)
        unified_df = unified_df.sort_values('source_priority', ascending=True)
        
        before_dedup = len(unified_df)
        unified_df = unified_df.drop_duplicates(subset=['experiment_signature'], keep='first')
        after_dedup = len(unified_df)
        
        if before_dedup != after_dedup:
            print(f"[INFO] Cross-dataset deduplication removed {before_dedup - after_dedup} duplicates")
        
        # Drop temporary columns
        unified_df = unified_df.drop(columns=['source_priority', 'experiment_signature'], errors='ignore')
        
        # Save unified results to a NEW file (non-destructive - original files remain untouched)
        unified_file = os.path.join(root, "evaluation", "results", "unified_all_results.csv")
        os.makedirs(os.path.dirname(unified_file), exist_ok=True)
        unified_df.to_csv(unified_file, index=False)
        
        print(f"\n[INFO] Unified results saved to: {unified_file}")
        print(f"[INFO] Total rows in unified results: {len(unified_df)}")
        if 'source' in unified_df.columns:
            source_counts = unified_df['source'].value_counts()
            print(f"[INFO] Source breakdown: {dict(source_counts)}")
        print(f"[INFO] Original source files in sol_results/ and results/ remain unchanged (non-destructive)")
        
        return unified_df
    else:
        print("[INFO] No results found to aggregate.")
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