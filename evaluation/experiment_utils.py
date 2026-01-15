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

from utils import create_output_path, create_hdf5_model_path, get_noise_intensities, get_short_session_id
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
    
    # Validate that mode matches tuned parameter
    # If tuned=True, mode should contain "_tune", if tuned=False, mode should not contain "_tune"
    mode_has_tune = "_tune" in mode
    if tuned and not mode_has_tune:
        # If tuned is True but mode doesn't have "_tune", add it
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
            
            # Check both new (short) and old (long) path formats for backwards compatibility
            out_dir_new = create_output_path(model_name, seed, int(representative_subject), session, mode, session_type=eval_mode, paradigm=paradigm, dataset=dataset)
            
            # Create old path format (with full session string) for backwards compatibility
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
            # instead of a specific filename pattern (to handle legacy filename conventions)
            if is_test_perturb_mode:
                # Check new path format first
                found_files = False
                if os.path.exists(out_dir_new):
                    csv_files = [f for f in os.listdir(out_dir_new) if f.endswith('.csv')]
                    if csv_files:
                        for csv_file in csv_files:
                            existing_output_paths.append(os.path.join(out_dir_new, csv_file))
                        found_files = True
                
                # Also check old path format for backwards compatibility
                if not found_files and os.path.exists(out_dir_old):
                    csv_files = [f for f in os.listdir(out_dir_old) if f.endswith('.csv')]
                    if csv_files:
                        for csv_file in csv_files:
                            existing_output_paths.append(os.path.join(out_dir_old, csv_file))
                        found_files = True
                
                if not found_files:
                    expected_output_paths.append(out_dir_new)  # Prefer new format
            else:
                # For non-test_perturb modes, check both new and old filename patterns
                if noise_type is not None and intensity is not None:
                    filename_suffix = f"_{noise_type}_{intensity}"
                else:
                    filename_suffix = ""
                
                # Check new format (short session)
                out_file_new = os.path.join(out_dir_new,
                                            f"{model_name}_{mode}{filename_suffix}_{short_session}_seed{seed}.csv")
                # Check old format (full session) for backwards compatibility
                out_file_old = os.path.join(out_dir_old,
                                            f"{model_name}_{mode}{filename_suffix}_{session}_seed{seed}.csv")
                
                if os.path.exists(out_file_new):
                    existing_output_paths.append(out_file_new)
                elif os.path.exists(out_file_old):
                    existing_output_paths.append(out_file_old)  # Found in old format
                else:
                    expected_output_paths.append(out_file_new)  # Prefer new format
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
            
            # Use short session identifier for filename to avoid long paths
            # The full session info (including eval_subjects) is preserved in the CSV data
            short_session = get_short_session_id(session, 'CrossSubject')
            out_file = os.path.join(out_dir,
                                    f"{model_name}_{mode}{filename_suffix}_{short_session}_seed{seed}.csv")
            
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


# ============================================================================
# Intensity Filtering Functions (using saturation-based bounds)
# ============================================================================

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


def filter_by_intensity_bounds(df: pd.DataFrame, dataset: str, num_steps: int = 20) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows with intensities within defined bounds.
    
    Uses saturation points to determine correct intensity ranges for each dataset/noise_type
    combination. Intensity bounds vary by both dataset AND noise_type, so we handle each
    combination separately. This filtering happens BEFORE deduplication to ensure we only
    retain experiments that match our defined experimental parameters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with 'intensity' and 'noise_type' columns
    dataset : str
        Dataset name (e.g., 'BNCI2014_001', 'Lee2019_SSVEP', 'BI2015a')
    num_steps : int
        Number of intensity steps (default: 20)
        
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
            correct_intensities = get_noise_intensities(
                dataset=dataset,
                noise_type=noise_type,
                num_steps=num_steps
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


def normalize_mode(mode) -> str:
    """Normalize mode to canonical form."""
    if pd.isna(mode) or mode == '':
        return 'unknown'
    return str(mode).strip().lower()


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
        # Save combined dataframe to all_results.csv *IF* source is not 'sol_results'
        if source != 'sol_results':
            combined.to_csv(os.path.join(root_dir, "all_results.csv"), index=False)
        return combined
    return pd.DataFrame()


def collect_all_results(paradigm: str, dataset: str = "BNCI2014_001"):
    """
    Aggregate all CSV results from both sol_results and results directories.
    
    This function implements the deduplication schema from DEDUPLICATION_SCHEMA.md:
    - Phase 1: Collect from sol_results (primary source)
    - Phase 2: Collect from results (secondary source), filtering out duplicates
    - All operations are read-only on source files
    
    Args:
        paradigm: Paradigm name (e.g., "MotorImagery")
        dataset: Dataset name (e.g., "BNCI2014_001")
        
    Returns:
        DataFrame with deduplicated results, or None if no results found
    """
    all_dfs = []
    
    # Phase 1: Collect from sol_results (primary source) - READ ONLY
    sol_results_dir = os.path.join("sol_results", paradigm, dataset)
    print(f"[INFO] Phase 1: Collecting from sol_results/{paradigm}/{dataset} (read-only)")
    sol_results = collect_from_directory(sol_results_dir, source="sol_results", paradigm=paradigm, dataset=dataset)
    
    # Build index of existing signatures from sol_results
    sol_signatures = set()
    if sol_results is not None and not sol_results.empty:
        # STEP 1: Filter by intensity bounds BEFORE creating signatures
        # This ensures we only process experiments that match our defined experimental parameters
        print(f"[INFO] Filtering sol_results by intensity bounds (using saturation points)...")
        before_intensity_filter = len(sol_results)
        sol_results = filter_by_intensity_bounds(sol_results, dataset=dataset, num_steps=20)
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
    results_dir = os.path.join("results", paradigm, dataset)
    print(f"[INFO] Phase 2: Collecting from results/{paradigm}/{dataset} (read-only)")
    if os.path.exists(results_dir):
        results_df = collect_from_directory(results_dir, source="results", paradigm=paradigm, dataset=dataset)
        if results_df is not None and not results_df.empty:
            # STEP 1: Filter by intensity bounds BEFORE creating signatures
            # This ensures we only process experiments that match our defined experimental parameters
            print(f"[INFO] Filtering results by intensity bounds (using saturation points)...")
            before_intensity_filter = len(results_df)
            results_df = filter_by_intensity_bounds(results_df, dataset=dataset, num_steps=20)
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


def collect_all_results_unified():
    """
    Collect and aggregate results from all datasets and paradigms.
    
    This function implements the unified collection strategy from DEDUPLICATION_SCHEMA.md:
    - Collects from all paradigm-dataset combinations
    - Performs deduplication across all datasets
    - Writes to a new unified output file (non-destructive - original files remain untouched)
    
    Returns:
        DataFrame with unified, deduplicated results, or None if no results found
    """
    all_results = []
    
    # Define all dataset-paradigm combinations
    dataset_paradigms = [
        ("MotorImagery", "BNCI2014_001"),
        ("SSVEP", "Lee2019_SSVEP"),
        ("ERP", "BI2015a")
    ]
    
    for paradigm, dataset in dataset_paradigms:
        print(f"\n=== Collecting results for {paradigm} - {dataset} ===")
        result_df = collect_all_results(paradigm, dataset)
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
        unified_file = os.path.join("evaluation", "results", "unified_all_results.csv")
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