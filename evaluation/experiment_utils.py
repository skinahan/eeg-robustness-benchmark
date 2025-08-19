"""
Utility functions for EEG experiments.

This module contains helper functions used across different experiment types
to reduce code duplication and improve maintainability.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder

from utils import create_output_path, create_hdf5_model_path
from evaluation.two_stage_hp_opt import run_two_stage_optuna


def extract_model_params(model) -> Dict[str, Any]:
    """Extract model parameters for logging purposes."""
    if hasattr(model, 'get_params'):
        return model.get_params()
    return {}


def check_skip_eval(model_name, seed, subject_list, mode, noise_type, intensity, eval_mode='WithinSessionEvaluation'):
    """Check if evaluation should be skipped based on existing output files."""
    existing_output_paths = []
    expected_output_paths = []
    
    for subj in subject_list:
        sessions_to_check = ['0train', '1test']
                        
        for session in sessions_to_check:
            out_dir = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode)
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

    if len(expected_output_paths) == 0:
        print(f"Skipping analysis, file(s) exist:")
        for out_file in existing_output_paths:
            print(out_file)
        return True
    return False


def log_all_subjects(results, subject_list, model_name, mode, noise_type, intensity, seed, eval_mode='WithinSessionEvaluation'):
    """Log results for all subjects to individual CSV files."""
    for subj in subject_list:        
        subject_df = results[results['subject'] == int(subj)]
        for session in subject_df['session'].unique():
            session_df = subject_df[subject_df['session'] == session]
            out_dir = create_output_path(model_name, seed, int(subj), session, mode, session_type=eval_mode)
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
    root = os.path.join("results", paradigm, dataset)
    all_dfs = []
    noise_types = ['gaussian', 'eog', 'dropout']
    intensities = [str(x*10.0) for x in range(1, 10)]
    
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".csv") and not file.startswith("all_results"):
                full_path = os.path.join(dirpath, file)
                print(full_path)
                try:
                    df = pd.read_csv(full_path)
                    selected_type = None
                    intensity = None
                    for type in noise_types:
                        if type in file:
                            selected_type = type
                            for strength in intensities:
                                if strength in file:
                                    intensity = strength
                                    break
                            break
                    if selected_type is not None and intensity is not None:
                        df['noise_type'] = selected_type
                        df['intensity'] = intensity

                    if 'cross_session' in full_path or 'CrossSessionEvaluation' in full_path:
                        df['eval_mode'] = 'CrossSessionEvaluation'
                    else:
                        df['eval_mode'] = 'WithinSessionEvaluation'

                    all_dfs.append(df)
                except Exception as e:
                    print(f"Failed to read {full_path}: {e}")
    
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_file = os.path.join(root, "all_results.csv")
        full_df.to_csv(out_file, index=False)
        print(f"Aggregated results saved to: {out_file}")
    else:
        print("No CSV files found to aggregate.")


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