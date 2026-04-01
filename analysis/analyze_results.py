import os
import sys
import pandas as pd

# Repo root (for evaluation.* imports when run as a script)
_AR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AR_ROOT not in sys.path:
    sys.path.insert(0, _AR_ROOT)
from evaluation.experiment_utils import apply_perturb_sweep_mode_canonicalization


def _canonicalize_df_for_test_perturb_plots(df, log_label: str = "analyze_results"):
    """Map multirun -> test_perturb before any ``mode == 'test_perturb'`` filter (idempotent)."""
    return apply_perturb_sweep_mode_canonicalization(df, log_label=log_label)
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SanityCheckError(Exception):
    """Exception raised when sanity check fails."""
    pass


# Configuration for plot y-axis limits per dataset
PLOT_YLIM_CONFIG = {
    'BNCI2014_001': {
        'performance_max': 0.85,  # Max ylim for performance curves (score vs intensity)
        'rd_max': 0.4,  # Max ylim for Relative Degradation curves
        'csv_dynamic': True,  # Use dynamic max for CSV_p (calculated from data)
        'csv_padding': 0.1,  # Padding factor for dynamic CSV_p max (10% above max value)
    },
    # Add other datasets here as needed
    'Lee2019_SSVEP': {
        'performance_max': 1.0,
        'rd_max': 1.0,
        'csv_dynamic': True,
        'csv_padding': 0.1,
    },
    'Lee2019_MI': {
        'performance_max': 1.0,
        'rd_max': 1.0,
        'csv_dynamic': True,
        'csv_padding': 0.1,
    },
    'Shin2017A': {
        'performance_max': 1.0,
        'rd_max': 1.0,
        'csv_dynamic': True,
        'csv_padding': 0.1,
    },
    'BI2015a': {
        'performance_max': 1.0,
        'rd_max': 1.0,
        'csv_dynamic': True,
        'csv_padding': 0.1,
    },
}


def replace_hydra_model_name(df, model_col='model'):
    """
    Replace 'branched_wiredcfc_arch4' (and variations) with 'HYDRA' in the model column.
    Handles various naming formats (with/without hyphens, different cases).
    
    Parameters:
    - df: pd.DataFrame with a model column
    - model_col: str, name of the model column (default: 'model')
    
    Returns:
    - pd.DataFrame with model names replaced
    """
    if model_col not in df.columns:
        return df
    
    df = df.copy()
    # Normalize model names for comparison (lowercase, hyphens/spaces to underscores)
    # The canonical form after canonicalize_columns is 'branched_wiredcfc_arch4'
    df_model_normalized = df[model_col].astype(str).str.lower().str.replace('-', '_').str.replace(' ', '_')
    
    # Replace any variant of branched_wiredcfc_arch4 with HYDRA
    mask = df_model_normalized == 'branched_wiredcfc_arch4'
    df.loc[mask, model_col] = 'HYDRA'
    
    return df


def format_noise_type_label(noise_type):
    """
    Format noise type for display in plot titles.
    Capitalizes properly, especially for abbreviations like EOG.
    
    Parameters:
    - noise_type: str, e.g., 'dropout', 'gaussian', 'eog'
    
    Returns:
    - str, formatted label (e.g., 'Dropout', 'Gaussian', 'EOG')
    """
    if noise_type is None:
        return ''
    
    noise_type_lower = noise_type.lower()
    if noise_type_lower == 'eog':
        return 'EOG'
    elif noise_type_lower == 'dropout':
        return 'Dropout'
    elif noise_type_lower == 'gaussian':
        return 'Gaussian'
    elif noise_type_lower == 'spike':
        return 'Spike'
    else:
        # Default: capitalize first letter
        return noise_type.capitalize()


def format_model_name_for_display(model_name):
    """
    Format model name for display in plots and figures.
    Maps internal model names to their publication-ready display names.
    
    Parameters:
    - model_name: str, internal model name (e.g., 'cnn_ncp', 'eegnet', 'reegnet')
    
    Returns:
    - str, formatted display name (e.g., 'CNN-NCP', 'EEGNet', 'REEGNet')
    """
    if model_name is None:
        return ''
    
    # Normalize input (lowercase, handle hyphens/spaces)
    model_normalized = str(model_name).lower().strip().replace('-', '_').replace(' ', '_')
    
    # Mapping of normalized model names to display names
    MODEL_DISPLAY_MAP = {
        'cnn_ncp': 'CNN-NCP',
        'cnn_ncp_v2': 'CNN-NCP',
        'eegnet': 'EEGNet',
        'ctnet': 'CTNet',
        'reegnet': 'REEGNet',
        'branched_wiredcfc_arch4': 'HYDRA',
        'hydra': 'HYDRA',
    }
    
    # Return mapped name, or original if not in map
    return MODEL_DISPLAY_MAP.get(model_normalized, model_name)


def format_model_names_in_df(df, model_col='model'):
    """
    Format model names in a DataFrame column for display in plots.
    Similar to replace_hydra_model_name(), but formats all model names
    to their publication-ready display format.
    
    Parameters:
    - df: pd.DataFrame with a model column
    - model_col: str, name of the model column (default: 'model')
    
    Returns:
    - pd.DataFrame with model names formatted for display
    """
    if model_col not in df.columns:
        return df
    
    df = df.copy()
    df[model_col] = df[model_col].apply(format_model_name_for_display)
    return df


def get_plot_ylim_config(dataset, plot_type='performance'):
    """
    Get y-axis limit configuration for a given dataset and plot type.
    
    Parameters:
    - dataset: str, dataset name
    - plot_type: str, one of 'performance', 'rd', 'csv'
    
    Returns:
    - dict with 'min' and 'max' keys, or None if not configured
    """
    config = PLOT_YLIM_CONFIG.get(dataset, {})
    
    if plot_type == 'performance':
        max_val = config.get('performance_max', 1.0)
        min_val = 0.4 if dataset == 'BNCI2014_001' else 0.0
        return {'min': min_val, 'max': max_val}
    elif plot_type == 'rd':
        max_val = config.get('rd_max', 1.0)
        return {'min': 0.0, 'max': max_val}
    elif plot_type == 'csv':
        # CSV uses dynamic max, so return None to indicate dynamic calculation needed
        return None
    else:
        return None


def load_saturation_points(saturation_csv_path='saturation_results/saturation_points_summary.csv'):
    """
    Load saturation points from the summary CSV file.
    
    Parameters:
    - saturation_csv_path: Path to the saturation points CSV file
    
    Returns:
    - Dictionary with structure: {dataset: {noise_type: saturation_point}}
    """
    try:
        if not os.path.exists(saturation_csv_path):
            # Try relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            saturation_csv_path = os.path.join(script_dir, '..', saturation_csv_path)
        
        df = pd.read_csv(saturation_csv_path)
        saturation_dict = {}
        
        for _, row in df.iterrows():
            dataset = row['dataset']
            noise_type = row['noise_type']
            saturation_point = row['saturation_point']
            
            if dataset not in saturation_dict:
                saturation_dict[dataset] = {}
            saturation_dict[dataset][noise_type] = saturation_point
        
        return saturation_dict
    except Exception as e:
        print(f"Warning: Could not load saturation points from {saturation_csv_path}: {e}")
        print("Using default saturation point of 50.0")
        return {}


def get_correct_intensities(dataset='BNCI2014_001', noise_type=None, saturation_dict=None, num_points=20):
    """
    Get the correct intensity range based on saturation points.
    
    Parameters:
    - dataset: Dataset name
    - noise_type: Noise type (dropout, gaussian, eog)
    - saturation_dict: Dictionary of saturation points (if None, will load from file)
    - num_points: Number of intensity points to generate
    
    Returns:
    - numpy array of intensity values from 1.0 to saturation_point
    """
    if saturation_dict is None:
        saturation_dict = load_saturation_points()
    
    # Get saturation point for this dataset and noise type
    if dataset in saturation_dict and noise_type in saturation_dict[dataset]:
        saturation_point = saturation_dict[dataset][noise_type]
    else:
        print(f"Warning: No saturation point found for {dataset}/{noise_type}, using default 50.0")
        saturation_point = 50.0
    
    return np.linspace(1.0, saturation_point, num_points)


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


def aggregate_results(input_dir):
    """
    [DEPRECATED] Aggregates CSVs from a specified directory.
    
    This function is deprecated. Use the unified results file instead:
    ../evaluation/results/unified_all_results.csv
    
    The unified file is generated by collect_all_results_unified() in 
    evaluation/experiment_utils.py and contains all results with proper
    deduplication.

    1. Adds 'model', 'noise_type', and 'noise_level' columns.
    2. For baseline results, includes default hyperparameters and sets 'tuned' to False.
    3. For tuned results, uses actual parameter values and sets 'tuned' to True.
    4. Drops specified columns and sets dataset column.

    Parameters:
    - input_dir: Path to the directory containing .csv files.

    Returns:
    - A single aggregated DataFrame.
    """

    # Default parameters for baseline files
    default_params = {
        'EEGNet': {
            'optimizer__lr': 1e-3,
            'batch_size': 64,
            'module__drop_prob': 0.25
        },
        'REEGNet': {
            'optimizer__lr': 1e-4,
            'batch_size': 64,
            'module__drop_prob': 0.15,
            'module__lstm_hidden_size': 32
        },
        'CNN_NCP': {
            'optimizer__lr': 1e-4,
            'batch_size': 8,
            'module__net_size': 32
        },
        'CNN_NCPv2': {
            'optimizer__lr': 1e-4,
            'batch_size': 8,
            'module__net_size': 32
        }
    }

    aggregated_dfs = []
    for file in os.listdir(input_dir):

        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path)

            file_lower = file.lower()

            # Determine model type
            if 'reegnet' in file_lower:
                model = 'REEGNet'
            elif 'eegnet' in file_lower:
                model = 'EEGNet'
            elif 'cnn_ncp' in file_lower:
                model = 'CNN_NCP'
            elif 'cnn_cfc' in file_lower:
                model = 'CNN_CfC'
            else:
                model = 'Unknown'

            if '_resampled' in file.lower():
                model = model + '_RESAMPLE'

            # Determine if tuned or baseline
            tuned = not ('_baseline_subjects' in file.lower())

            # Noise type & level
            noise_type, noise_level = None, None
            if '_dropout_' in file.lower():
                noise_type = 'dropout'
            elif '_gaussian_' in file.lower():
                noise_type = 'gaussian'
            elif '_eog_' in file.lower():
                noise_type = 'eog'
            else:
                noise_type = None
                noise_level = None

            if noise_type is not None:
                noise_level_str = file.split(f'_{noise_type}_')[-1].split('_')[0]
                noise_level = float(noise_level_str.replace('.csv', ''))

            seed = None
            if 'seed' in file.lower():
                seed_str = file.split('seed')[-1].replace('.csv', '')
                if '_' in seed_str:
                    seed_str = seed_str.split('_')[0]
                seed = int(seed_str)


            # Add metadata columns
            df['model'] = model
            df['noise_type'] = noise_type
            df['noise_level'] = noise_level
            df['tuned'] = tuned
            if seed is not None:
                df['seed'] = seed

            if not tuned:
                # Add default hyperparameters
                defaults = default_params[model]
                for param_name, param_val in defaults.items():
                    df[param_name] = param_val
            else:
                # Fill in any missing param columns with NaN
                for param in ['optimizer__lr', 'batch_size', 'module__drop_prob', 'module__lstm_hidden_size']:
                    if param not in df.columns:
                        df[param] = None

            aggregated_dfs.append(df)

    # Concatenate all results into a single DataFrame
    if aggregated_dfs:
        combined_df = pd.concat(aggregated_dfs, ignore_index=True)
        combined_df = apply_perturb_sweep_mode_canonicalization(
            combined_df, log_label="aggregate_results(deprecated)"
        )

        # Drop unnecessary columns
        combined_df.drop(columns=['time', 'samples', 'channels', 'n_sessions', 'pipeline'], inplace=True,
                         errors='ignore')

        # Set 'dataset' column if not already present
        if 'dataset' not in combined_df.columns:
            # Try to infer dataset from directory structure
            if 'BNCI2014_001' in input_dir:
                combined_df['dataset'] = 'BNCI2014_001'
            elif 'Lee2019_SSVEP' in input_dir:
                combined_df['dataset'] = 'Lee2019_SSVEP'
            elif 'Lee2019_MI' in input_dir:
                combined_df['dataset'] = 'Lee2019_MI'
            elif 'Shin2017A' in input_dir:
                combined_df['dataset'] = 'Shin2017A'
            elif 'BI2015a' in input_dir:
                combined_df['dataset'] = 'BI2015a'
            else:
                combined_df['dataset'] = 'BNCI2014_001'  # Default

        # Filter to only include intended experimental seeds: [100, 200, 300, 400, 500]
        valid_seeds = [100, 200, 300, 400, 500]
        if 'seed' in combined_df.columns:
            initial_len = len(combined_df)
            # Convert seed to numeric, handling any string representations
            combined_df['seed'] = pd.to_numeric(combined_df['seed'], errors='coerce')
            # Filter to valid seeds (drop rows with NaN seeds or seeds not in valid list)
            combined_df = combined_df[combined_df['seed'].isin(valid_seeds)].copy()
            filtered_count = initial_len - len(combined_df)
            if filtered_count > 0:
                print(f"[INFO] Filtered out {filtered_count} rows with seeds not in {valid_seeds}")
            print(f"[INFO] Remaining rows with valid seeds: {len(combined_df)}")
        else:
            print("[WARNING] No 'seed' column found - cannot filter by seed values")

        return combined_df
    else:
        print("No .csv files found in the provided directory.")
        return None


def plot_comparative_noise_performance(aggregated_df, noise_type, session_type, run_mode='augment', output_dir='plots'):
    """
    Plots performance of multiple models (cnn_ncp, eegnet, reegnet) across noise intensities
    for a specific noise type and session type.

    Parameters:
    - aggregated_df: DataFrame containing aggregated results with 'model', 'noise_level', 'score', etc.
    - noise_type: str, e.g., 'dropout', 'gaussian', 'eog'.
    - session_type: str, e.g., '0train' or '1test'.
    - output_dir: str, directory to save the plot (default: 'plots').
    """
    models = ['cnn_cfc', 'cnn_ncp', 'eegnet', 'reegnet']
    df_filtered = aggregated_df[
        (aggregated_df['noise_type'] == noise_type) &
        (aggregated_df['session'] == session_type) &
        (aggregated_df['model'].isin(models)) &
        (aggregated_df['seed'] != '42') &
        (aggregated_df['mode'] == run_mode)
    ]


    if df_filtered.empty:
        print(f"No data to plot for models {models} with noise '{noise_type}' and session '{session_type}'.")
        return

    df_filtered['noise_level'] = df_filtered['intensity'].astype(float)

    # Set up labels
    noise_label = noise_type.capitalize()
    session_label = "Test" if session_type == '1test' else "Train"

    # Create plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.lineplot(
        data=df_filtered,
        x='noise_level',
        y='score',
        hue='model',
        marker='o',
        errorbar=('ci', 95)
    )

    # Labeling
    ax.set_title(f"Model Comparison: {session_label} {run_mode.capitalize()} Performance vs {noise_label} Intensity", fontsize=14)
    ax.set_xlabel(f"{noise_label} Intensity (%)", fontsize=12)
    ax.set_ylabel("Mean Score (ROC AUC)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(title='Model', fontsize=10, title_fontsize=11)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"model_comparison_{noise_type}_{run_mode}_{session_type}.pdf")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='pdf')
    plt.close()

    print(f"Comparative plot saved to {output_file}")


def plot_noise_performance(aggregated_df, model_name, noise_type, session_type, run_mode='augment', output_dir='plots'):
    """
    Generic method to create a seaborn plot of noise intensity vs. mean score for a given model.

    Parameters:
    - aggregated_df: DataFrame containing aggregated results.
    - model_name: str, the name of the model to filter on ('EEGNet' or 'REEGNet').
    - noise_type: str, e.g., 'dropout', 'gaussian', 'eog'.
    - session_type: str, e.g., '0train' or '1test'.
    - output_dir: str, directory to save the plot (default: 'plots' in root).
    """
    # Filter data
    df_filtered = aggregated_df[
        (aggregated_df['noise_type'] == noise_type) &
        (aggregated_df['session'] == session_type) &
        (aggregated_df['model'] == model_name) &
        (aggregated_df['mode'] == run_mode)
        ]

    if df_filtered.empty:
        print(f"No data to plot for model '{model_name}' with noise '{noise_type}' and session '{session_type}'.")
        return

    # Group by noise level and compute mean score
    df_grouped = df_filtered#.groupby('noise_level')['score'].mean().reset_index()

    # Determine human-readable labels
    noise_label = noise_type.capitalize()
    session_label = "Test" if session_type == '1test' else "Train"

    # Create plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6), dpi=300)
    ax = sns.lineplot(x='noise_level', y='score', marker='o', data=df_grouped, color='b', errorbar=('ci', 95))

    # Labeling and styling
    ax.set_title(f"{model_name}: Mean {session_label} Score vs {noise_label} Intensity", fontsize=14)
    ax.set_xlabel(f"{noise_label} Intensity (%)", fontsize=12)
    ax.set_ylabel("Mean Score (ROC AUC)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_{noise_type}_{session_type}_performance.pdf")
    print(f"Saving plot to: {output_file}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='pdf')
    plt.close()

    print(f"Plot saved to {output_file}")


def plot_dropout_test_performance(aggregated_df, model_name):
    """
    Wrapper for plotting dropout noise performance on the test set for a given model.
    """
    plot_noise_performance(
        aggregated_df=aggregated_df,
        model_name=model_name,
        noise_type='dropout',
        session_type='1test'
    )


def plot_dropout_train_performance(aggregated_df, model_name):
    """
    Wrapper for plotting dropout noise performance on the test set for a given model.
    """
    plot_noise_performance(
        aggregated_df=aggregated_df,
        model_name=model_name,
        noise_type='dropout',
        session_type='0train'
    )


def plot_gaussian_test_performance(aggregated_df, model_name):
    """
    Wrapper for plotting gaussian noise performance on the test set for a given model.
    """
    plot_noise_performance(
        aggregated_df=aggregated_df,
        model_name=model_name,
        noise_type='gaussian',
        session_type='1test'
    )


def plot_gaussian_train_performance(aggregated_df, model_name):
    """
    Wrapper for plotting gaussian noise performance on the train set for a given model.
    """
    plot_noise_performance(
        aggregated_df=aggregated_df,
        model_name=model_name,
        noise_type='gaussian',
        session_type='0train'
    )


def plot_eog_test_performance(aggregated_df, model_name):
    """
    Wrapper for plotting eog noise performance on the test set for a given model.
    """
    plot_noise_performance(
        aggregated_df=aggregated_df,
        model_name=model_name,
        noise_type='eog',
        session_type='1test'
    )


def plot_eog_train_performance(aggregated_df, model_name):
    """
    Wrapper for plotting eog noise performance on the train set for a given model.
    """
    plot_noise_performance(
        aggregated_df=aggregated_df,
        model_name=model_name,
        noise_type='eog',
        session_type='0train'
    )


def plot_per_subject_roc_auc(
        df,
        model_name,
        tuned=True,
        output_dir='plots'
):
    """
    Plots per-subject mean ROC-AUC for train/test performance, with separate bars.
    Only includes baseline (non-augmented) results for tuned models.

    Parameters:
    - df: pandas DataFrame containing the aggregated results.
    - model_name: str, the model to plot ('EEGNet' or 'REEGNet').
    - tuned: bool, whether to show tuned or baseline results.
    - output_dir: directory to save the plot PNG.
    """
    # Filter for model
    df_model = df[df['model'] == model_name]

    if tuned:
        # Only tuned results, but no noise augmentation
        df_filtered = df_model[(df_model['tuned'] == True) & (df_model['noise_type'].isnull())]
        label = 'Tuned (no noise)'
    else:
        # Baseline, no noise augmentation
        df_filtered = df_model[(df_model['tuned'] == False) & (df_model['noise_type'].isnull())]
        label = 'Baseline'

    # Compute mean ROC-AUC for each subject/session
    df_plot = df_filtered#.groupby(['subject', 'session'])['score'].mean().reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='subject', y='score', hue='session', palette='Set2')
    plt.title(f'{model_name} {label} Per-Subject ROC-AUC')
    plt.ylabel('Mean ROC-AUC')
    plt.xlabel('Subject')
    plt.ylim(0, 1)  # ROC-AUC is bounded [0, 1]
    plt.legend(title='Session')
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_{'tuned' if tuned else 'baseline'}_subjectwise_performance.pdf")
    plt.savefig(out_file, dpi=300, format='pdf')
    plt.close()
    print(f"Plot saved to {out_file}")


def model_plots(aggregated_df, model_name):
    plot_dropout_train_performance(aggregated_df, model_name)
    plot_dropout_test_performance(aggregated_df, model_name)

    plot_gaussian_train_performance(aggregated_df, model_name)
    plot_gaussian_test_performance(aggregated_df, model_name)

    plot_eog_train_performance(aggregated_df, model_name)
    plot_eog_test_performance(aggregated_df, model_name)

    plot_per_subject_roc_auc(aggregated_df, model_name=model_name, tuned=False)
    plot_per_subject_roc_auc(aggregated_df, model_name=model_name, tuned=True)


def eegnet_plots(aggregated_df):
    model_plots(aggregated_df, 'eegnet')


def reegnet_plots(aggregated_df):
    model_plots(aggregated_df, 'reegnet')


def cnn_ncp_plots(aggregated_df):
    model_plots(aggregated_df, 'cnn_ncp')
    # model_plots(aggregated_df, 'CNN_NCP_RESAMPLE')


def run_comparative_plots(aggregated_df):
    modes = ['augment', 'perturb']
    sessions = ['0train', '1test']
    noise_types = ['gaussian', 'dropout', 'eog', 'spike']
    for mode in modes:
        for session in sessions:
            for noise in noise_types:
                plot_comparative_noise_performance(aggregated_df, noise_type=noise, session_type=session, run_mode=mode,
                                                   output_dir='./plots/')


def run_completion_report(output_dir, aggregated_df):
    # Define the combinations we want to check for
    models = ["eegnet", "reegnet", "cnnncp"]
    seeds = [100, 200, 300, 400, 500]
    noise_types = ["dropout", "gaussian", "eog", "spike"]
    noise_levels = list(range(10, 100, 10))

    # Create all possible combinations
    expected_configs = list(itertools.product(models, seeds, noise_types, noise_levels))

    # Normalize columns in existing results
    aggregated_df['noise_level'] = aggregated_df['noise_level'].astype('Int64')
    aggregated_df['seed'] = aggregated_df['seed'].astype('Int64')
    aggregated_df['model'] = aggregated_df['model'].str.lower()

    # Deduplicate based on combinations
    completed = aggregated_df.dropna(subset=['noise_type', 'noise_level', 'seed'])
    completed = completed[['model', 'seed', 'noise_type', 'noise_level']].drop_duplicates()

    # Check which expected configs are completed
    summary = []
    for model, seed, noise, level in expected_configs:
        match = (
                (completed['model'] == model)
                & (completed['seed'] == seed)
                & (completed['noise_type'] == noise)
                & (completed['noise_level'] == level)
        )
        is_complete = match.any()
        summary.append({
            "model": model,
            "seed": seed,
            "noise_type": noise,
            "noise_level": level,
            "complete": "TRUE" if is_complete else "FALSE"
        })

    summary_df = pd.DataFrame(summary)

    # Also save to disk
    summary_df.to_csv(os.path.join(output_dir, "experiment_completion_report.csv"), index=False)


def _get_metric_columns(metric: str):
    metric = metric.lower()
    clean_col = f"clean_{metric}" if metric != 'roc_auc' else 'clean_roc_auc'
    corrupted_col = f"corrupted_{metric}" if metric != 'roc_auc' else 'corrupted_roc_auc'
    # Backward compatibility fallbacks
    if metric == 'roc_auc':
        return clean_col, corrupted_col, 'Corrupted Score (ROC AUC)'
    y_label_map = {
        'accuracy': 'Corrupted Accuracy',
        'precision': 'Corrupted Precision',
        'recall': 'Corrupted Recall',
        'f1': 'Corrupted F1-score',
    }
    return clean_col, corrupted_col, y_label_map.get(metric, f'Corrupted {metric}')


def _get_metric_columns_legacy(metric: str):
    """Legacy version that handles old column naming convention"""
    metric = metric.lower()
    if metric == 'roc_auc':
        # For legacy data, use the old column names
        clean_col = 'clean_score'
        corrupted_col = 'corrupted_score'
        return clean_col, corrupted_col, 'Corrupted Score (ROC AUC)'
    else:
        # For other metrics, use new naming convention
        clean_col = f"clean_{metric}"
        corrupted_col = f"corrupted_{metric}"
        y_label_map = {
            'accuracy': 'Corrupted Accuracy',
            'precision': 'Corrupted Precision',
            'recall': 'Corrupted Recall',
            'f1': 'Corrupted F1-score',
        }
        return clean_col, corrupted_col, y_label_map.get(metric, f'Corrupted {metric}')


def plot_test_perturb_individual_model(df, model_name, noise_type, tune_setting, output_dir='plots', metric: str='roc_auc', dataset='BNCI2014_001', eval_mode=None):
    """
    Plot individual model performance for test_perturb case.
    
    Parameters:
    - df: DataFrame with test_perturb results
    - model_name: str, name of the model
    - noise_type: str, 'dropout' or 'gaussian'
    - tune_setting: bool, True for tuned, False for baseline
    - output_dir: str, directory to save plots
    - dataset: str, dataset name (affects y-axis limits)
    - eval_mode: str, evaluation mode ('CrossSession', 'WithinSession', 'CrossSubject'). If None, uses all available.
    """
    df = _canonicalize_df_for_test_perturb_plots(df, "plot_test_perturb_individual_model")
    # Load saturation points and get correct intensity values
    saturation_dict = load_saturation_points()
    correct_intensities = get_correct_intensities(dataset=dataset, noise_type=noise_type, saturation_dict=saturation_dict)
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Auto-detect eval_mode if not provided
    if eval_mode is None:
        available_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
        if len(available_modes) == 1:
            eval_mode = available_modes[0]
        else:
            eval_mode = 'CrossSession'  # Default
    
    # Store original eval_mode for filtering (dataframe has values without "Evaluation" suffix)
    eval_mode_for_filter = eval_mode
    # Normalize eval_mode format for display/directory naming
    if eval_mode_for_filter.endswith('Evaluation'):
        eval_mode_short = eval_mode_for_filter.replace('Evaluation', '')
    else:
        eval_mode_short = eval_mode_for_filter
    
    # Filter data using original eval_mode value
    # Use tolerance-based intensity matching to handle floating-point precision issues
    intensity_mask = intensity_matches(df['intensity'], correct_intensities)
    df_filtered = df[
        (df['mode'] == 'test_perturb') &
        (df['eval_mode'] == eval_mode_for_filter) &
        (df['model'] == model_name) &
        (df['noise_type'] == noise_type) &
        (df['tune'] == tune_setting) &
        (df['seed'].isin(valid_seeds)) &
        intensity_mask
    ].copy()
    
    if df_filtered.empty:
        print(f"No data found for {model_name}, {noise_type}, tune={tune_setting}, eval_mode={eval_mode_for_filter}")
        return
    
    # Use legacy column naming for backward compatibility
    clean_col, corrupted_col, y_label = _get_metric_columns_legacy(metric)

    # Remove rows with missing corrupted metric
    df_filtered = df_filtered.dropna(subset=[corrupted_col])
    
    # Add clean_score as intensity 0.0 (one row per model, noise_type, seed, session combination)
    clean_data = df_filtered.dropna(subset=[clean_col]).copy()
    if not clean_data.empty:
        # Get unique clean metric per model, noise_type, seed, session combination
        clean_summary = clean_data.groupby(['model', 'noise_type', 'seed', 'session'])[clean_col].first().reset_index()
        clean_summary['intensity'] = 0.0
        clean_summary[corrupted_col] = clean_summary[clean_col]
        clean_summary['tune'] = tune_setting
        clean_summary['mode'] = 'test_perturb'
        clean_summary['eval_mode'] = eval_mode_for_filter
        
        # Add the clean data to the filtered data
        df_filtered = pd.concat([clean_summary, df_filtered], ignore_index=True)
    
    tune_label = "Tuned" if tune_setting else "Baseline"
    
    # Create both bar and line plots
    for plot_type in ['bar', 'line']:
        plt.figure(figsize=(12, 8), dpi=300)
        
        if plot_type == 'bar':
            sns.barplot(
                data=df_filtered,
                x='intensity',
                y=corrupted_col,
                hue='session',
                palette='Set2',
                errorbar=('ci', 95)
            )
        else:
            sns.lineplot(
                data=df_filtered,
                x='intensity',
                y=corrupted_col,
                hue='session',
                marker='o',
                palette='Set2',
                errorbar=('ci', 95)
            )
        
        plt.title(f'{dataset} | {model_name} | {noise_type.capitalize()} Noise | {tune_label} | {eval_mode_short}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Noise Intensity (%)', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(title='Session', fontsize=10, title_fontsize=11)
        plt.grid(True, alpha=0.3)
        # Set y-axis limits based on dataset
        y_min = 0.4 if dataset == 'BNCI2014_001' else 0
        plt.ylim(y_min, 1)
        
        # Save plot to eval_mode-specific directory
        eval_mode_dir = os.path.join(output_dir, dataset, eval_mode_short) if dataset else output_dir
        os.makedirs(eval_mode_dir, exist_ok=True)
        # Include eval_mode in filename to distinguish plots
        filename = f"{model_name}_{noise_type}_{metric}_{'tuned' if tune_setting else 'baseline'}_{plot_type}_test_perturb.pdf"
        output_file = os.path.join(eval_mode_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Saved {plot_type} plot: {output_file}")


def plot_test_perturb_master_comparison(df, noise_type, tune_setting, models=None, output_dir='plots', metric: str='roc_auc', dataset='BNCI2014_001', eval_mode=None):
    """
    Create master comparison plot overlaying specified models for a given noise type and tune setting.
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout' or 'gaussian'
    - tune_setting: bool, True for tuned, False for baseline
    - models: list, specific models to include (if None, uses all available models)
    - output_dir: str, directory to save plots
    - dataset: str, dataset name (affects y-axis limits)
    - eval_mode: str, evaluation mode ('CrossSession', 'WithinSession', 'CrossSubject'). If None, uses all available.
    """
    df = _canonicalize_df_for_test_perturb_plots(df, "plot_test_perturb_master_comparison")
    # Load saturation points and get correct intensity values
    saturation_dict = load_saturation_points()
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Auto-detect eval_mode if not provided
    if eval_mode is None:
        available_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
        if len(available_modes) == 1:
            eval_mode = available_modes[0]
        else:
            eval_mode = 'CrossSession'  # Default
    
    # Store original eval_mode for filtering (dataframe has values without "Evaluation" suffix)
    eval_mode_for_filter = eval_mode
    # Normalize eval_mode format for display/directory naming
    if eval_mode_for_filter.endswith('Evaluation'):
        eval_mode_short = eval_mode_for_filter.replace('Evaluation', '')
    else:
        eval_mode_short = eval_mode_for_filter
    
    # Filter data using original eval_mode value
    df_filtered = df[
        (df['mode'] == 'test_perturb') &
        (df['eval_mode'] == eval_mode_for_filter) &
        (df['noise_type'] == noise_type) &
        (df['tune'] == tune_setting) &
        (df['seed'].isin(valid_seeds))
    ].copy()
    
    # Filter by specific models if provided
    if models is not None:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    if df_filtered.empty:
        print(f"No data found for master plot: {noise_type}, tune={tune_setting}, eval_mode={eval_mode_for_filter}")
        return
    

    clean_col, corrupted_col, y_label = 'clean_score', 'corrupted_score', 'Corrupted Score (ROC AUC)' #_get_metric_columns_legacy('roc_auc')
    
    # Remove rows with missing corrupted_score
    df_filtered = df_filtered.dropna(subset=[corrupted_col])
    
    # Add clean_score as intensity 0.0
    clean_data = df_filtered.dropna(subset=[clean_col]).copy()
    if not clean_data.empty:
        # Ensure we only get clean data for THIS noise_type
        clean_data_for_noise = clean_data[clean_data['noise_type'] == noise_type].copy()
        
        if not clean_data_for_noise.empty:
            clean_summary = clean_data_for_noise.groupby(['model', 'seed', 'session', 'subject'])[clean_col].first().reset_index()
            clean_summary['intensity'] = 0.0
            clean_summary[corrupted_col] = clean_summary[clean_col]
            clean_summary['noise_type'] = noise_type  # Explicitly set noise_type
            clean_summary['tune'] = tune_setting
            clean_summary['mode'] = 'test_perturb'
            clean_summary['eval_mode'] = eval_mode_for_filter
            
            df_filtered = pd.concat([clean_summary, df_filtered], ignore_index=True)
    
    tune_label = "Tuned" if tune_setting else "Baseline"
    
    # Create both bar and line plots
    for plot_type in ['bar', 'line']:
        plt.figure(figsize=(14, 8), dpi=300)
        
        if plot_type == 'bar':
            sns.barplot(
                data=df_filtered,
                x='intensity',
                y=corrupted_col,
                hue='model',
                palette='tab10',
                errorbar=('ci', 95)
            )
        else:
            sns.lineplot(
                data=df_filtered,
                x='intensity',
                y=corrupted_col,
                hue='model',
                marker='o',
                palette='tab10',
                linewidth=2.5,
                markersize=8,
                errorbar=('ci', 95)
            )
        
        plt.title(f'{dataset} | Model Comparison | {noise_type.capitalize()} Noise | {tune_label} | {eval_mode_short}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Noise Intensity (%)', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(title='Model', fontsize=10, title_fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        # Set y-axis limits based on dataset
        y_min = 0.4 if dataset == 'BNCI2014_001' else 0
        plt.ylim(y_min, 1)
        
        # Save plot to eval_mode-specific directory
        eval_mode_dir = os.path.join(output_dir, dataset, eval_mode_short) if dataset else output_dir
        os.makedirs(eval_mode_dir, exist_ok=True)
        # Include eval_mode in filename
        filename = f"master_comparison_{noise_type}_{metric}_{'tuned' if tune_setting else 'baseline'}_{plot_type}_test_perturb.pdf"
        output_file = os.path.join(eval_mode_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Saved master {plot_type} plot: {output_file}")


def generate_all_test_perturb_plots(df, models=None, output_dir='plots', metrics=('roc_auc','accuracy','precision','recall','f1'), dataset='BNCI2014_001'):
    """
    Generate all test_perturb plots as specified in the requirements.
    Creates separate plots for each eval_mode (CrossSession, WithinSession, CrossSubject).
    
    Parameters:
    - df: DataFrame with all results
    - models: list, specific models to include (if None, uses all available models)
    - output_dir: str, directory to save plots
    - dataset: str, dataset name (affects y-axis limits)
    """
    df = _canonicalize_df_for_test_perturb_plots(df, "generate_all_test_perturb_plots")
    # Get unique values
    if models is None:
        models = df[df['mode'] == 'test_perturb']['model'].unique()
    else:
        models = models  # Use provided models list
    
    noise_types = df[df['mode'] == 'test_perturb']['noise_type'].unique()
    tune_settings = df[df['mode'] == 'test_perturb']['tune'].unique()
    
    # Get available eval_modes (use original values from dataframe)
    available_eval_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
    eval_modes = sorted(set(available_eval_modes))  # Remove duplicates and sort
    
    print(f"Generating plots for {len(models)} models, {len(noise_types)} noise types, {len(tune_settings)} tune settings, {len(eval_modes)} eval modes")
    print(f"Models: {list(models)}")
    print(f"Noise types: {list(noise_types)}")
    print(f"Eval modes: {eval_modes}")
    
    # Generate individual model plots for each metric and eval_mode
    # for eval_mode in eval_modes:
    #     for model in models:
    #         for noise_type in noise_types:
    #             for tune_setting in tune_settings:
    #                 for metric in metrics:
    #                     plot_test_perturb_individual_model(df, model, noise_type, tune_setting, output_dir, metric, dataset, eval_mode=eval_mode)
    
    # Generate master comparison plots for each metric and eval_mode
    for eval_mode in eval_modes:
        print(f"\n--- Processing {eval_mode} evaluation mode ---")
        for noise_type in noise_types:
            for tune_setting in tune_settings:
                for metric in metrics:
                    plot_test_perturb_master_comparison(df, noise_type, tune_setting, models, output_dir, metric, dataset, eval_mode=eval_mode)
    
    print(f"\nAll test_perturb plots generated and saved to {output_dir}")


def plot_test_perturb_per_subject(df, model_name, noise_type, tune_setting, dataset='BNCI2014_001', output_dir='plots', metric: str='roc_auc', eval_mode=None):
    """
    Create per-subject plots for test_perturb results.
    
    Parameters:
    - df: DataFrame with test_perturb results
    - model_name: str, name of the model
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - tune_setting: bool, True for tuned, False for baseline
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    - eval_mode: str, evaluation mode ('CrossSession', 'WithinSession', 'CrossSubject'). If None, uses all available.
    """
    df = _canonicalize_df_for_test_perturb_plots(df, "plot_test_perturb_per_subject")
    # Load saturation points and get correct intensity values
    saturation_dict = load_saturation_points()
    correct_intensities = get_correct_intensities(dataset=dataset, noise_type=noise_type, saturation_dict=saturation_dict)
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Auto-detect eval_mode if not provided
    if eval_mode is None:
        available_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
        if len(available_modes) == 1:
            eval_mode = available_modes[0]
        else:
            eval_mode = 'CrossSession'  # Default
    
    # Store original eval_mode for filtering (dataframe has values without "Evaluation" suffix)
    eval_mode_for_filter = eval_mode
    # Normalize eval_mode format for display/directory naming
    if eval_mode_for_filter.endswith('Evaluation'):
        eval_mode_short = eval_mode_for_filter.replace('Evaluation', '')
    else:
        eval_mode_short = eval_mode_for_filter
    
    # Filter data using original eval_mode value
    # Use tolerance-based intensity matching to handle floating-point precision issues
    intensity_mask = intensity_matches(df['intensity'], correct_intensities)
    df_filtered = df[
        (df['mode'] == 'test_perturb') &
        (df['eval_mode'] == eval_mode_for_filter) &
        (df['model'] == model_name) &
        (df['noise_type'] == noise_type) &
        (df['tune'] == tune_setting) &
        (df['seed'].isin(valid_seeds)) &
        intensity_mask
    ].copy()
    
    if df_filtered.empty:
        print(f"No data found for per-subject plot: {model_name}, {noise_type}, tune={tune_setting}, eval_mode={eval_mode_for_filter}")
        return
    
    # Use legacy column naming for backward compatibility
    clean_col, corrupted_col, y_label = _get_metric_columns_legacy(metric)
    df_filtered = df_filtered.dropna(subset=[corrupted_col])
    
    # Add clean_score as intensity 0.0
    clean_data = df_filtered.dropna(subset=[clean_col]).copy()
    if not clean_data.empty:
        # Ensure we only get clean data for THIS noise_type
        clean_data_for_noise = clean_data[clean_data['noise_type'] == noise_type].copy()
        
        if not clean_data_for_noise.empty:
            clean_summary = clean_data_for_noise.groupby(['model', 'seed', 'session', 'subject'])[clean_col].first().reset_index()
            clean_summary['intensity'] = 0.0
            clean_summary[corrupted_col] = clean_summary[clean_col]
            clean_summary['noise_type'] = noise_type  # Explicitly set noise_type
            clean_summary['model'] = model_name  # Explicitly set model
            clean_summary['tune'] = tune_setting
            clean_summary['mode'] = 'test_perturb'
            clean_summary['eval_mode'] = eval_mode_for_filter
            
            df_filtered = pd.concat([clean_summary, df_filtered], ignore_index=True)
    
    # Get unique subjects
    subjects = sorted(df_filtered['subject'].unique())
    tune_label = "tuned" if tune_setting else "baseline"
    
    # Create plots for each subject
    for subject in subjects:
        subject_data = df_filtered[df_filtered['subject'] == subject]
        
        if subject_data.empty:
            continue
        
        # Create both bar and line plots
        for plot_type in ['bar', 'line']:
            plt.figure(figsize=(12, 8), dpi=300)
            
            if plot_type == 'bar':
                sns.barplot(
                    data=subject_data,
                    x='intensity',
                    y=corrupted_col,
                    hue='session',
                    palette='Set2',
                    errorbar=('ci', 95)
                )
            else:
                sns.lineplot(
                    data=subject_data,
                    x='intensity',
                    y=corrupted_col,
                    hue='session',
                    marker='o',
                    palette='Set2',
                    errorbar=('ci', 95)
                )
            
            plt.title(f'{dataset} | {model_name} | Subject {subject} | {noise_type.capitalize()} Noise | {tune_label.capitalize()} | {eval_mode_short}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Noise Intensity (%)', fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.legend(title='Session', fontsize=10, title_fontsize=11)
            plt.grid(True, alpha=0.3)
            # Set y-axis limits based on dataset
            y_min = 0.4 if dataset == 'BNCI2014_001' else 0
            plt.ylim(y_min, 1)
            
            # Create directory structure with eval_mode
            subject_dir = os.path.join(output_dir, dataset, eval_mode_short, f'subject_{subject}', noise_type)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Save plot
            filename = f"{model_name}_{metric}_{tune_label}_{plot_type}_test_perturb.pdf"
            output_file = os.path.join(subject_dir, filename)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f"Saved per-subject {plot_type} plot: {output_file}")


def plot_test_perturb_multisubject_comparison(df, noise_type, tune_setting, models=None, dataset='BNCI2014_001', output_dir='plots', eval_mode=None):
    """
    Create multi-subject comparison plots for test_perturb results.
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - tune_setting: bool, True for tuned, False for baseline
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    - eval_mode: str, evaluation mode ('CrossSession', 'WithinSession', 'CrossSubject'). If None, uses all available.
    """
    df = _canonicalize_df_for_test_perturb_plots(df, "plot_test_perturb_multisubject_comparison")
    # Load saturation points and get correct intensity values
    saturation_dict = load_saturation_points()
    correct_intensities = get_correct_intensities(dataset=dataset, noise_type=noise_type, saturation_dict=saturation_dict)
    valid_seeds = [100, 200, 300, 400, 500]

    # Auto-detect eval_mode if not provided
    if eval_mode is None:
        available_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
        if len(available_modes) == 1:
            eval_mode = available_modes[0]
        else:
            eval_mode = 'CrossSession'  # Default
    
    # Store original eval_mode for filtering (dataframe has values without "Evaluation" suffix)
    eval_mode_for_filter = eval_mode
    # Normalize eval_mode format for display/directory naming
    if eval_mode_for_filter.endswith('Evaluation'):
        eval_mode_short = eval_mode_for_filter.replace('Evaluation', '')
    else:
        eval_mode_short = eval_mode_for_filter

    mode_str = 'test_perturb'
    if tune_setting:
        mode_str = 'test_perturb_tune'

    # Filter data using original eval_mode value
    df_filtered = df[
        (df['mode'] == mode_str) &
        (df['eval_mode'] == eval_mode_for_filter) &
        (df['noise_type'] == noise_type) &
        (df['tune'] == tune_setting) &
        (df['seed'].isin(valid_seeds)) #&
        #(df['intensity'].isin(correct_intensities))
    ].copy()
    
    # Filter by specific models if provided
    if models is not None:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    if df_filtered.empty:
        print(f"No data found for multisubject plot: {noise_type}, tune={tune_setting}, eval_mode={eval_mode}")
        return
    
    # Debug: Print unique noise types in filtered data
    print(f"[DEBUG] Processing {noise_type}, tune={tune_setting}, eval_mode={eval_mode}")
    print(f"[DEBUG] Unique noise_types in df_filtered: {df_filtered['noise_type'].unique()}")
    print(f"[DEBUG] Data shape: {df_filtered.shape}")
    
    # Use legacy column naming for backward compatibility
    clean_col, corrupted_col, y_label = _get_metric_columns_legacy('roc_auc')
    
    # Remove rows with missing corrupted_score
    df_filtered = df_filtered.dropna(subset=[corrupted_col])
    
    # Add clean_score as intensity 0.0
    clean_data = df_filtered.dropna(subset=[clean_col]).copy()
    if not clean_data.empty:
        # Ensure we only get clean data for THIS noise_type
        clean_data_for_noise = clean_data[clean_data['noise_type'] == noise_type].copy()
        
        if not clean_data_for_noise.empty:
            clean_summary = clean_data_for_noise.groupby(['model', 'seed', 'session', 'subject'])[clean_col].first().reset_index()
            clean_summary['intensity'] = 0.0
            clean_summary[corrupted_col] = clean_summary[clean_col]
            clean_summary['noise_type'] = noise_type  # Explicitly set noise_type
            clean_summary['tune'] = tune_setting
            clean_summary['mode'] = mode_str
            clean_summary['eval_mode'] = eval_mode_for_filter
            
            df_filtered = pd.concat([clean_summary, df_filtered], ignore_index=True)
            
            # Debug: Verify noise_type after concat
            print(f"[DEBUG] After concat, unique noise_types: {df_filtered['noise_type'].unique()}")
    
    tune_label = "tuned" if tune_setting else "baseline"
    
    # Create both bar and line plots
    for plot_type in ['bar', 'line']:
        plt.figure(figsize=(14, 8), dpi=300)
        
        if plot_type == 'bar':
            sns.barplot(
                data=df_filtered,
                x='intensity',
                y=corrupted_col,
                hue='model',
                palette='tab10',
                errorbar=('ci', 95)
            )
        else:
            sns.lineplot(
                data=df_filtered,
                x='intensity',
                y=corrupted_col,
                hue='model',
                marker='o',
                palette='tab10',
                linewidth=2.5,
                markersize=8,
                errorbar=('ci', 95)
            )
        
        plt.title(f'{dataset} | Multi-Subject Model Comparison | {noise_type.capitalize()} Noise | {tune_label.capitalize()} | {eval_mode_short}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Noise Intensity (%)', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(title='Model', fontsize=10, title_fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        # Set y-axis limits based on dataset
        y_min = 0.4 if dataset == 'BNCI2014_001' else 0
        plt.ylim(y_min, 1)
        
        # Create directory structure with eval_mode
        multisubject_dir = os.path.join(output_dir, dataset, eval_mode_short, 'multisubject', noise_type)
        os.makedirs(multisubject_dir, exist_ok=True)
        
        # Save plot
        filename = f"model_comparison_{tune_label}_{plot_type}_test_perturb.pdf"
        output_file = os.path.join(multisubject_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Saved multisubject {plot_type} plot: {output_file}")


def plot_combined_multisubject_comparison(df, noise_type, models=None, dataset='BNCI2014_001', output_dir='plots', plot_type='line', hydra=False, zoomed=False):
    """
    Create a combined multi-subject comparison plot organized in a 3x2 grid:
    - Rows: CrossSession, WithinSession, CrossSubject (eval_modes)
    - Columns: Baseline, Tuned
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    - plot_type: str, 'line' or 'bar' (default: 'line')
    - zoomed: bool, if True limits y-axis to (0.5, 0.8) and x-axis to (0, 30) intensity;
      uses a different filename (_zoomed suffix) to avoid overwriting existing plots;
      when zoomed, x-axis is (0, 50) intensity
    """
    from matplotlib.gridspec import GridSpec
    
    # Load saturation points and get correct intensity values
    saturation_dict = load_saturation_points()
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Define eval_modes in order (rows)
    eval_mode_order = ['CrossSession', 'WithinSession', 'CrossSubject']
    eval_mode_labels = ['Cross-Session', 'Within-Session', 'Cross-Subject']
    
    # Define tune settings in order (columns)
    tune_settings = [False, True]
    tune_labels = ['Baseline', 'Tuned']
    
    # Filter by specific models if provided
    if models is not None:
        df = df[df['model'].isin(models)]
    
    # Replace branched_wiredcfc_arch4 with HYDRA in model names for plotting
    df = replace_hydra_model_name(df, model_col='model')
    
    # Use legacy column naming for backward compatibility
    clean_col, corrupted_col, y_label = _get_metric_columns_legacy('roc_auc')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # Process each combination
    for row_idx, eval_mode_base in enumerate(eval_mode_order):
        # Normalize eval_mode (handle both with and without "Evaluation" suffix)
        # Check available eval_modes in the dataframe
        available_modes = df['eval_mode'].dropna().unique()
        
        # Try exact match first
        eval_mode_for_filter = None
        if eval_mode_base + 'Evaluation' in available_modes:
            eval_mode_for_filter = eval_mode_base + 'Evaluation'
        elif eval_mode_base in available_modes:
            eval_mode_for_filter = eval_mode_base
        else:
            # Try to find a matching eval_mode (case-insensitive, partial match)
            matching_modes = [m for m in available_modes if eval_mode_base.lower() in str(m).lower()]
            if matching_modes:
                eval_mode_for_filter = matching_modes[0]
            else:
                print(f"[WARNING] No data found for eval_mode {eval_mode_base}, skipping...")
                # Create empty subplots for this row
                for col_idx in range(2):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.text(0.5, 0.5, 'No data available', 
                           ha='center', va='center', fontsize=16, style='italic')
                    ax.set_xlabel('Noise Intensity (%)', fontsize=16)
                    ax.set_ylabel(y_label, fontsize=16)
                    ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                               fontsize=16, fontweight='bold')
                    if zoomed:
                        ax.set_ylim(0.5, 0.8)
                        ax.set_xlim(0, 50)
                    ax.grid(True, alpha=0.3)
                continue
        
        for col_idx, tune_setting in enumerate(tune_settings):
            # Filter data - handle both test_perturb and test_perturb_tune modes
            df_filtered = df[
                (df['mode'].astype(str).str.contains('test_perturb', na=False)) &
                (df['eval_mode'] == eval_mode_for_filter) &
                (df['noise_type'] == noise_type) &
                (df['tune'] == tune_setting) &
                (df['seed'].isin(valid_seeds))
            ].copy()
            
            if df_filtered.empty:
                # Create empty subplot with message
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Noise Intensity (%)', fontsize=16)
                ax.set_ylabel(y_label, fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                if zoomed:
                    ax.set_ylim(0.5, 0.8)
                    ax.set_xlim(0, 50)
                ax.grid(True, alpha=0.3)
                continue
            
            # Remove rows with missing corrupted_score
            df_filtered = df_filtered.dropna(subset=[corrupted_col])
            
            # Diagnostic: Verify all seeds are present
            if 'seed' in df_filtered.columns:
                present_seeds = sorted(df_filtered['seed'].dropna().unique())
                missing_seeds = [s for s in valid_seeds if s not in present_seeds]
                if missing_seeds:
                    print(f"[WARNING] {dataset} | {eval_mode_labels[row_idx]} | {tune_labels[col_idx]} | {noise_type}: "
                          f"Missing seeds: {missing_seeds}. Present seeds: {present_seeds}")
                else:
                    print(f"[INFO] {dataset} | {eval_mode_labels[row_idx]} | {tune_labels[col_idx]} | {noise_type}: "
                          f"All {len(present_seeds)} seeds present: {present_seeds}")
                
                # Count data points per seed to verify representation
                seed_counts = df_filtered.groupby(['model', 'intensity', 'seed']).size().reset_index(name='count')
                if len(seed_counts) > 0:
                    min_count = seed_counts['count'].min()
                    max_count = seed_counts['count'].max()
                    if min_count != max_count:
                        print(f"[WARNING] {dataset} | {eval_mode_labels[row_idx]} | {tune_labels[col_idx]}: "
                              f"Uneven seed representation (min={min_count}, max={max_count} data points per seed)")
            else:
                print(f"[ERROR] {dataset} | {eval_mode_labels[row_idx]} | {tune_labels[col_idx]}: "
                      f"'seed' column missing from filtered data!")
            
            # Add clean_score as intensity 0.0
            clean_data = df_filtered.dropna(subset=[clean_col]).copy()
            if not clean_data.empty:
                clean_data_for_noise = clean_data[clean_data['noise_type'] == noise_type].copy()
                if not clean_data_for_noise.empty:
                    clean_summary = clean_data_for_noise.groupby(['model', 'seed', 'session', 'subject'])[clean_col].first().reset_index()
                    clean_summary['intensity'] = 0.0
                    clean_summary[corrupted_col] = clean_summary[clean_col]
                    clean_summary['noise_type'] = noise_type
                    clean_summary['tune'] = tune_setting
                    # Preserve mode from original data
                    if 'mode' in df_filtered.columns and len(df_filtered) > 0:
                        clean_summary['mode'] = df_filtered['mode'].iloc[0]
                    clean_summary['eval_mode'] = eval_mode_for_filter
                    df_filtered = pd.concat([clean_summary, df_filtered], ignore_index=True)
            
            # Verify data structure: Check that each (model, intensity) combination has data from all seeds
            if 'seed' in df_filtered.columns:
                seed_verification = df_filtered.groupby(['model', 'intensity'])['seed'].nunique().reset_index(name='num_seeds')
                incomplete_groups = seed_verification[seed_verification['num_seeds'] < len(valid_seeds)]
                if not incomplete_groups.empty:
                    print(f"[WARNING] {dataset} | {eval_mode_labels[row_idx]} | {tune_labels[col_idx]}: "
                          f"Some (model, intensity) combinations have fewer than {len(valid_seeds)} seeds:")
                    for _, row in incomplete_groups.head(10).iterrows():
                        actual_seeds = sorted(df_filtered[
                            (df_filtered['model'] == row['model']) & 
                            (df_filtered['intensity'] == row['intensity'])
                        ]['seed'].unique())
                        print(f"  Model={row['model']}, Intensity={row['intensity']:.2f}: "
                              f"{row['num_seeds']} seeds ({actual_seeds})")
                    if len(incomplete_groups) > 10:
                        print(f"  ... and {len(incomplete_groups) - 10} more")
                
                # Additional diagnostic: Show data structure summary
                total_rows = len(df_filtered)
                unique_combinations = df_filtered.groupby(['model', 'intensity']).size()
                avg_rows_per_combo = unique_combinations.mean()
                print(f"[INFO] {dataset} | {eval_mode_labels[row_idx]} | {tune_labels[col_idx]}: "
                      f"Total rows={total_rows}, Unique (model,intensity)={len(unique_combinations)}, "
                      f"Avg rows per combo={avg_rows_per_combo:.1f}")
                
                # Check if there are additional grouping columns that might affect variation
                potential_group_cols = ['subject', 'session', 'fold_idx']
                for col in potential_group_cols:
                    if col in df_filtered.columns:
                        unique_vals = df_filtered[col].nunique()
                        print(f"  {col}: {unique_vals} unique values")
            
            # Create subplot
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if plot_type == 'bar':
                sns.barplot(
                    data=df_filtered,
                    x='intensity',
                    y=corrupted_col,
                    hue='model',
                    palette='tab10',
                    errorbar=('ci', 95),
                    ax=ax
                )
            else:  # line plot
                sns.lineplot(
                    data=df_filtered,
                    x='intensity',
                    y=corrupted_col,
                    hue='model',
                    marker='o',
                    palette='tab10',
                    linewidth=2.0,
                    markersize=6,
                    errorbar=('ci', 95),
                    ax=ax
                )
            
            # Customize subplot
            ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                        fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel('Noise Intensity (%)', fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            
            # Set axis limits
            if zoomed:
                ax.set_ylim(0.5, 0.8)
                ax.set_xlim(0, 50)
            else:
                # Set y-axis limits based on dataset configuration
                ylim_config = get_plot_ylim_config(dataset, plot_type='performance')
                if ylim_config:
                    ax.set_ylim(ylim_config['min'], ylim_config['max'])
                else:
                    # Fallback to default
                    y_min = 0.4 if dataset == 'BNCI2014_001' else 0
                    ax.set_ylim(y_min, 1)
            
            # Customize legend (only show in first subplot, or all if preferred)
            if row_idx == 0 and col_idx == 0:
                ax.legend(title='Model', fontsize=14, title_fontsize=16, 
                         loc='upper right', framealpha=0.9)
            else:
                ax.legend().set_visible(False)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
    
    # Add overall title (remove "Noise" and format noise type properly)
    noise_label = format_noise_type_label(noise_type)
    fig.suptitle(f'{dataset} | Multi-Subject Model Comparison | {noise_label}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save plot
    combined_dir = os.path.join(output_dir, dataset, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    zoom_suffix = '_zoomed' if zoomed else ''
    filename = f"combined_multisubject_{noise_type}_{plot_type}_test_perturb{zoom_suffix}.pdf"
    output_file = os.path.join(combined_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    zoom_label = ' (zoomed)' if zoomed else ''
    print(f"Saved combined multisubject {plot_type} plot{zoom_label}: {output_file}")


def plot_rd_curves(df, noise_type, models=None, dataset='BNCI2014_001', output_dir='plots', eval_mode=None, hydra=False):
    """
    Plot Relative Degradation (RD) curves using robustness_metrics.py.
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    - eval_mode: str, evaluation mode (if None, creates separate plots for each)
    """
    # Import robustness metrics functions
    # Handle both relative and absolute imports
    try:
        from robustness_metrics import (
            MetricConfig, ResultsSpec, compute_results_metrics,
            canonicalize_columns, add_normalized_p
        )
    except ImportError:
        try:
            from analysis.robustness_metrics import (
                MetricConfig, ResultsSpec, compute_results_metrics,
                canonicalize_columns, add_normalized_p
            )
        except ImportError as e:
            print(f"[ERROR] Could not import robustness_metrics: {e}")
            print("Make sure analysis/robustness_metrics.py is available.")
            return
    
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Filter data
    df_filtered = df[
        (df['mode'].astype(str).str.contains('test_perturb', na=False)) &
        (df['noise_type'] == noise_type) &
        (df['seed'].isin(valid_seeds))
    ].copy()
    
    if models is not None:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    if df_filtered.empty:
        print(f"No data found for RD curves: {noise_type}")
        return
    
    # Canonicalize columns
    df_filtered = canonicalize_columns(df_filtered)
    
    # Detect metric column
    metric_col = None
    for candidate in ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']:
        if candidate in df_filtered.columns:
            metric_col = candidate
            break
    
    if metric_col is None:
        print("[ERROR] Could not find metric column for RD curve computation")
        return
    
    # Configure metrics
    cfg = MetricConfig(metric_col=metric_col)
    
    # Add normalized p coordinate
    df_filtered = add_normalized_p(
        df_filtered, cfg,
        normalize_within=['dataset', 'noise_type'],
        clip=True
    )
    
    # Get eval_modes to process
    if eval_mode is None:
        eval_modes = sorted(df_filtered['eval_mode'].dropna().unique())
    else:
        # Normalize eval_mode (handle both with and without "Evaluation" suffix)
        available_modes = df_filtered['eval_mode'].dropna().unique()
        if eval_mode + 'Evaluation' in available_modes:
            eval_modes = [eval_mode + 'Evaluation']
        elif eval_mode in available_modes:
            eval_modes = [eval_mode]
        else:
            # Try to find a matching eval_mode
            matching_modes = [m for m in available_modes if eval_mode.lower() in str(m).lower()]
            eval_modes = matching_modes if matching_modes else [eval_mode]
    
    for eval_mode_val in eval_modes:
        df_eval = df_filtered[df_filtered['eval_mode'] == eval_mode_val].copy()
        if df_eval.empty:
            continue
        
        # Ensure 'dataset' column exists (add it if missing)
        if 'dataset' not in df_eval.columns:
            df_eval['dataset'] = dataset
        
        # Filter base_group_cols to only include columns that exist in df_eval
        base_group_cols_candidates = ['dataset', 'tune', 'eval_mode', 'model', 'noise_type']
        base_group_cols = [col for col in base_group_cols_candidates if col in df_eval.columns]
        
        # Compute RD curves
        spec = ResultsSpec(
            base_group_cols=tuple(base_group_cols),
            per_instance_cols=('seed',)
        )
        
        results = compute_results_metrics(df_eval, cfg=cfg, spec=spec, hydra=hydra)
        rd_summary = results.get('rd_summary')
        
        # Replace branched_wiredcfc_arch4 with HYDRA in model names for plotting
        if rd_summary is not None and not rd_summary.empty:
            rd_summary = replace_hydra_model_name(rd_summary, model_col='model')
        
        if rd_summary is None or rd_summary.empty:
            print(f"No RD summary data for {noise_type}, {eval_mode_val}")
            continue
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        tune_settings = [False, True]
        tune_labels = ['Baseline', 'Tuned']
        
        for ax_idx, tune_setting in enumerate(tune_settings):
            ax = axes[ax_idx]
            rd_data = rd_summary[
                (rd_summary['tune'] == tune_setting) &
                (rd_summary['noise_type'] == noise_type)
            ].copy()
            
            if rd_data.empty:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=12, style='italic')
                ax.set_title(tune_labels[ax_idx], fontsize=12, fontweight='bold')
                continue
            
            # Plot RD curves for each model
            for model in rd_data['model'].unique():
                model_data = rd_data[rd_data['model'] == model].copy()
                model_data = model_data.sort_values('p')
                
                ax.plot(model_data['p'], model_data['mean'], 
                       marker='o', linewidth=2, markersize=6, label=model)
                # Add confidence intervals
                if 'ci_low' in model_data.columns and 'ci_high' in model_data.columns:
                    ax.fill_between(model_data['p'], model_data['ci_low'], model_data['ci_high'],
                                   alpha=0.2)
            
            ax.set_xlabel('Normalized Perturbation (p)', fontsize=11)
            ax.set_ylabel('Relative Degradation (RD)', fontsize=11)
            ax.set_title(tune_labels[ax_idx], fontsize=12, fontweight='bold')
            ax.legend(title='Model', fontsize=9, title_fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Normalize eval_mode for filename
        eval_mode_short = str(eval_mode_val).replace('Evaluation', '')
        
        # Format noise type properly (capitalize EOG, etc.)
        noise_label = format_noise_type_label(noise_type)
        fig.suptitle(f'{dataset} | Relative Degradation | {noise_label} | {eval_mode_short}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        rd_dir = os.path.join(output_dir, dataset, eval_mode_short, 'robustness_metrics')
        os.makedirs(rd_dir, exist_ok=True)
        filename = f"rd_curve_{noise_type}_{eval_mode_short}.pdf"
        output_file = os.path.join(rd_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Saved RD curve plot: {output_file}")


def plot_combined_rd_curves(df, noise_type, models=None, dataset='BNCI2014_001', output_dir='plots', hydra=False):
    """
    Create a combined Relative Degradation (RD) curves plot organized in a 3x2 grid:
    - Rows: CrossSession, WithinSession, CrossSubject (eval_modes)
    - Columns: Baseline, Tuned
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    """
    from matplotlib.gridspec import GridSpec
    
    # Import robustness metrics functions
    try:
        from robustness_metrics import (
            MetricConfig, ResultsSpec, compute_results_metrics,
            canonicalize_columns, add_normalized_p
        )
    except ImportError:
        try:
            from analysis.robustness_metrics import (
                MetricConfig, ResultsSpec, compute_results_metrics,
                canonicalize_columns, add_normalized_p
            )
        except ImportError as e:
            print(f"[ERROR] Could not import robustness_metrics: {e}")
            print("Make sure analysis/robustness_metrics.py is available.")
            return
    
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Filter data
    df_filtered = df[
        (df['mode'].astype(str).str.contains('test_perturb', na=False)) &
        (df['noise_type'] == noise_type) &
        (df['seed'].isin(valid_seeds))
    ].copy()
    
    if models is not None:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    if df_filtered.empty:
        print(f"No data found for RD curves: {noise_type}")
        return
    
    # Canonicalize columns
    df_filtered = canonicalize_columns(df_filtered)
    
    # Detect metric column
    metric_col = None
    for candidate in ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']:
        if candidate in df_filtered.columns:
            metric_col = candidate
            break
    
    if metric_col is None:
        print("[ERROR] Could not find metric column for RD curve computation")
        return
    
    # Configure metrics
    cfg = MetricConfig(metric_col=metric_col)
    
    # Add normalized p coordinate
    df_filtered = add_normalized_p(
        df_filtered, cfg,
        normalize_within=['dataset', 'noise_type'],
        clip=True
    )
    
    # Define eval_modes in order (rows)
    eval_mode_order = ['CrossSession', 'WithinSession', 'CrossSubject']
    eval_mode_labels = ['Cross-Session', 'Within-Session', 'Cross-Subject']
    
    # Define tune settings in order (columns)
    tune_settings = [False, True]
    tune_labels = ['Baseline', 'Tuned']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # Process each combination
    for row_idx, eval_mode_base in enumerate(eval_mode_order):
        # Normalize eval_mode (handle both with and without "Evaluation" suffix)
        available_modes = df_filtered['eval_mode'].dropna().unique()
        
        # Try exact match first
        eval_mode_for_filter = None
        if eval_mode_base + 'Evaluation' in available_modes:
            eval_mode_for_filter = eval_mode_base + 'Evaluation'
        elif eval_mode_base in available_modes:
            eval_mode_for_filter = eval_mode_base
        else:
            # Try to find a matching eval_mode (case-insensitive, partial match)
            matching_modes = [m for m in available_modes if eval_mode_base.lower() in str(m).lower()]
            if matching_modes:
                eval_mode_for_filter = matching_modes[0]
            else:
                print(f"[WARNING] No data found for eval_mode {eval_mode_base}, skipping...")
                # Create empty subplots for this row
                for col_idx in range(2):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.text(0.5, 0.5, 'No data available', 
                           ha='center', va='center', fontsize=16, style='italic')
                    ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                    ax.set_ylabel('Relative Degradation (RD)', fontsize=16)
                    ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                               fontsize=16, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                continue
        
        # Compute RD curves for this eval_mode
        df_eval = df_filtered[df_filtered['eval_mode'] == eval_mode_for_filter].copy()
        if df_eval.empty:
            # Create empty subplots for this row
            for col_idx in range(2):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                ax.set_ylabel('Relative Degradation (RD)', fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
            continue
        
        # Ensure 'dataset' column exists (add it if missing)
        if 'dataset' not in df_eval.columns:
            df_eval['dataset'] = dataset
        
        # Filter base_group_cols to only include columns that exist in df_eval
        base_group_cols_candidates = ['dataset', 'tune', 'eval_mode', 'model', 'noise_type']
        base_group_cols = [col for col in base_group_cols_candidates if col in df_eval.columns]
        
        # Compute RD curves
        spec = ResultsSpec(
            base_group_cols=tuple(base_group_cols),
            per_instance_cols=('seed',)
        )
        
        results = compute_results_metrics(df_eval, cfg=cfg, spec=spec, hydra=hydra)
        rd_summary = results.get('rd_summary')
        
        # Replace branched_wiredcfc_arch4 with HYDRA in model names for plotting
        if rd_summary is not None and not rd_summary.empty:
            rd_summary = replace_hydra_model_name(rd_summary, model_col='model')
        
        if rd_summary is None or rd_summary.empty:
            # Create empty subplots for this row
            for col_idx in range(2):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                ax.set_ylabel('Relative Degradation (RD)', fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
            continue
        
        for col_idx, tune_setting in enumerate(tune_settings):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            rd_data = rd_summary[
                (rd_summary['tune'] == tune_setting) &
                (rd_summary['noise_type'] == noise_type)
            ].copy()
            
            if rd_data.empty:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                ax.set_ylabel('Relative Degradation (RD)', fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                continue
            
            # Plot RD curves for each model
            for model in rd_data['model'].unique():
                model_data = rd_data[rd_data['model'] == model].copy()
                model_data = model_data.sort_values('p')
                
                ax.plot(model_data['p'], model_data['mean'], 
                       marker='o', linewidth=2, markersize=6, label=model)
                # Add confidence intervals
                if 'ci_low' in model_data.columns and 'ci_high' in model_data.columns:
                    ax.fill_between(model_data['p'], model_data['ci_low'], model_data['ci_high'],
                                   alpha=0.2)
            
            ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
            ax.set_ylabel('Relative Degradation (RD)', fontsize=16)
            ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                        fontsize=16, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
            # Set y-axis limits based on dataset configuration
            ylim_config = get_plot_ylim_config(dataset, plot_type='rd')
            if ylim_config:
                ax.set_ylim(ylim_config['min'], ylim_config['max'])
            else:
                # Fallback to default
                ax.set_ylim(0, 1)
            
            ax.tick_params(labelsize=14)
            
            # Customize legend (only show in first subplot)
            if row_idx == 0 and col_idx == 0:
                ax.legend(title='Model', fontsize=14, title_fontsize=16, 
                         loc='upper right', framealpha=0.9)
            else:
                ax.legend().set_visible(False)
    
    # Add overall title (remove "Noise" and format noise type properly)
    noise_label = format_noise_type_label(noise_type)
    fig.suptitle(f'{dataset} | Relative Degradation | {noise_label}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save plot
    combined_dir = os.path.join(output_dir, dataset, 'combined', 'robustness_metrics')
    os.makedirs(combined_dir, exist_ok=True)
    filename = f"combined_rd_curve_{noise_type}.pdf"
    output_file = os.path.join(combined_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Saved combined RD curve plot: {output_file}")


def plot_combined_csv_p_curves(df, noise_type, models=None, dataset='BNCI2014_001', output_dir='plots', hydra=False):
    """
    Create a combined Cross-Subject Variance (CSV_p) curves plot organized in a 3x2 grid:
    - Rows: CrossSession, WithinSession, CrossSubject (eval_modes)
    - Columns: Baseline, Tuned
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    """
    from matplotlib.gridspec import GridSpec
    
    # Import robustness metrics functions
    try:
        from robustness_metrics import (
            MetricConfig, ResultsSpec, compute_results_metrics,
            canonicalize_columns, add_normalized_p
        )
    except ImportError:
        try:
            from analysis.robustness_metrics import (
                MetricConfig, ResultsSpec, compute_results_metrics,
                canonicalize_columns, add_normalized_p
            )
        except ImportError as e:
            print(f"[ERROR] Could not import robustness_metrics: {e}")
            print("Make sure analysis/robustness_metrics.py is available.")
            return
    
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Filter data
    df_filtered = df[
        (df['mode'].astype(str).str.contains('test_perturb', na=False)) &
        (df['noise_type'] == noise_type) &
        (df['seed'].isin(valid_seeds))
    ].copy()
    
    if models is not None:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    if df_filtered.empty:
        print(f"No data found for CSV_p curves: {noise_type}")
        return
    
    # Canonicalize columns
    df_filtered = canonicalize_columns(df_filtered)
    
    # Detect metric column
    metric_col = None
    for candidate in ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']:
        if candidate in df_filtered.columns:
            metric_col = candidate
            break
    
    if metric_col is None:
        print("[ERROR] Could not find metric column for CSV_p curve computation")
        return
    
    # Configure metrics
    cfg = MetricConfig(metric_col=metric_col)
    
    # Add normalized p coordinate
    df_filtered = add_normalized_p(
        df_filtered, cfg,
        normalize_within=['dataset', 'noise_type'],
        clip=True
    )
    
    # Define eval_modes in order (rows)
    eval_mode_order = ['CrossSession', 'WithinSession', 'CrossSubject']
    eval_mode_labels = ['Cross-Session', 'Within-Session', 'Cross-Subject']
    
    # Define tune settings in order (columns)
    tune_settings = [False, True]
    tune_labels = ['Baseline', 'Tuned']
    
    # Calculate dynamic max for CSV_p if configured
    csv_max_value = None
    config = PLOT_YLIM_CONFIG.get(dataset, {})
    if config.get('csv_dynamic', False):
        # Collect all CSV summary data to calculate max
        all_csv_values = []
        for eval_mode_base in eval_mode_order:
            available_modes = df_filtered['eval_mode'].dropna().unique()
            eval_mode_for_filter = None
            if eval_mode_base + 'Evaluation' in available_modes:
                eval_mode_for_filter = eval_mode_base + 'Evaluation'
            elif eval_mode_base in available_modes:
                eval_mode_for_filter = eval_mode_base
            else:
                matching_modes = [m for m in available_modes if eval_mode_base.lower() in str(m).lower()]
                if matching_modes:
                    eval_mode_for_filter = matching_modes[0]
            
            if eval_mode_for_filter:
                df_eval = df_filtered[df_filtered['eval_mode'] == eval_mode_for_filter].copy()
                if not df_eval.empty:
                    # Ensure 'dataset' column exists (add it if missing)
                    if 'dataset' not in df_eval.columns:
                        df_eval['dataset'] = dataset
                    
                    # Filter base_group_cols to only include columns that exist in df_eval
                    base_group_cols_candidates = ['dataset', 'tune', 'eval_mode', 'model', 'noise_type']
                    base_group_cols = [col for col in base_group_cols_candidates if col in df_eval.columns]
                    
                    spec = ResultsSpec(
                        base_group_cols=tuple(base_group_cols),
                        per_instance_cols=('seed',)
                    )
                    results = compute_results_metrics(df_eval, cfg=cfg, spec=spec, hydra=hydra)
                    csv_summary = results.get('csv_summary')
                    # Replace branched_wiredcfc_arch4 with HYDRA in model names for plotting
                    if csv_summary is not None and not csv_summary.empty:
                        csv_summary = replace_hydra_model_name(csv_summary, model_col='model')
                    if csv_summary is not None and not csv_summary.empty:
                        csv_data = csv_summary[csv_summary['noise_type'] == noise_type]
                        if not csv_data.empty and 'mean' in csv_data.columns:
                            all_csv_values.extend(csv_data['mean'].dropna().tolist())
                            # Also include CI high values if available
                            if 'ci_high' in csv_data.columns:
                                all_csv_values.extend(csv_data['ci_high'].dropna().tolist())
        
        if all_csv_values:
            csv_max_value = max(all_csv_values)
            padding = config.get('csv_padding', 0.1)
            csv_max_value = csv_max_value * (1 + padding)
        else:
            # Fallback if no data found
            csv_max_value = 0.1
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # Process each combination
    for row_idx, eval_mode_base in enumerate(eval_mode_order):
        # Normalize eval_mode (handle both with and without "Evaluation" suffix)
        available_modes = df_filtered['eval_mode'].dropna().unique()
        
        # Try exact match first
        eval_mode_for_filter = None
        if eval_mode_base + 'Evaluation' in available_modes:
            eval_mode_for_filter = eval_mode_base + 'Evaluation'
        elif eval_mode_base in available_modes:
            eval_mode_for_filter = eval_mode_base
        else:
            # Try to find a matching eval_mode (case-insensitive, partial match)
            matching_modes = [m for m in available_modes if eval_mode_base.lower() in str(m).lower()]
            if matching_modes:
                eval_mode_for_filter = matching_modes[0]
            else:
                print(f"[WARNING] No data found for eval_mode {eval_mode_base}, skipping...")
                # Create empty subplots for this row
                for col_idx in range(2):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.text(0.5, 0.5, 'No data available', 
                           ha='center', va='center', fontsize=16, style='italic')
                    ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                    ax.set_ylabel('Cross-Subject Variance (CSV_p)', fontsize=16)
                    ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                               fontsize=16, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                continue
        
        # Compute CSV_p curves for this eval_mode
        df_eval = df_filtered[df_filtered['eval_mode'] == eval_mode_for_filter].copy()
        if df_eval.empty:
            # Create empty subplots for this row
            for col_idx in range(2):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                ax.set_ylabel('Cross-Subject Variance (CSV_p)', fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
            continue
        
        # Ensure 'dataset' column exists (add it if missing)
        if 'dataset' not in df_eval.columns:
            df_eval['dataset'] = dataset
        
        # Filter base_group_cols to only include columns that exist in df_eval
        base_group_cols_candidates = ['dataset', 'tune', 'eval_mode', 'model', 'noise_type']
        base_group_cols = [col for col in base_group_cols_candidates if col in df_eval.columns]
        
        # Compute CSV_p curves
        spec = ResultsSpec(
            base_group_cols=tuple(base_group_cols),
            per_instance_cols=('seed',)
        )
        
        results = compute_results_metrics(df_eval, cfg=cfg, spec=spec, hydra=hydra)
        csv_summary = results.get('csv_summary')
        
        # Replace branched_wiredcfc_arch4 with HYDRA in model names for plotting
        if csv_summary is not None and not csv_summary.empty:
            csv_summary = replace_hydra_model_name(csv_summary, model_col='model')
        
        if csv_summary is None or csv_summary.empty:
            # Create empty subplots for this row
            for col_idx in range(2):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                ax.set_ylabel('Cross-Subject Variance (CSV_p)', fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
            continue
        
        for col_idx, tune_setting in enumerate(tune_settings):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            csv_data = csv_summary[
                (csv_summary['tune'] == tune_setting) &
                (csv_summary['noise_type'] == noise_type)
            ].copy()
            
            if csv_data.empty:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=16, style='italic')
                ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
                ax.set_ylabel('Cross-Subject Variance (CSV_p)', fontsize=16)
                ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                           fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                continue
            
            # Plot CSV_p curves for each model
            for model in csv_data['model'].unique():
                model_data = csv_data[csv_data['model'] == model].copy()
                model_data = model_data.sort_values('p')
                
                ax.plot(model_data['p'], model_data['mean'], 
                       marker='o', linewidth=2, markersize=6, label=model)
                # Add confidence intervals
                if 'ci_low' in model_data.columns and 'ci_high' in model_data.columns:
                    ax.fill_between(model_data['p'], model_data['ci_low'], model_data['ci_high'],
                                   alpha=0.2)
            
            ax.set_xlabel('Normalized Perturbation (p)', fontsize=16)
            ax.set_ylabel('Cross-Subject Variance (CSV_p)', fontsize=16)
            ax.set_title(f'{eval_mode_labels[row_idx]} | {tune_labels[col_idx]}', 
                        fontsize=16, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
            # Set y-axis limits: use dynamic max if calculated, otherwise default
            if csv_max_value is not None:
                ax.set_ylim(0, csv_max_value)
            else:
                ax.set_ylim(bottom=0)
            
            ax.tick_params(labelsize=14)
            
            # Customize legend (only show in first subplot)
            if row_idx == 0 and col_idx == 0:
                ax.legend(title='Model', fontsize=14, title_fontsize=16, 
                         loc='upper right', framealpha=0.9)
            else:
                ax.legend().set_visible(False)
    
    # Add overall title (remove "Noise" and format noise type properly)
    noise_label = format_noise_type_label(noise_type)
    fig.suptitle(f'{dataset} | Cross-Subject Variance | {noise_label}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save plot
    combined_dir = os.path.join(output_dir, dataset, 'combined', 'robustness_metrics')
    os.makedirs(combined_dir, exist_ok=True)
    filename = f"combined_csv_p_curve_{noise_type}.pdf"
    output_file = os.path.join(combined_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Saved combined CSV_p curve plot: {output_file}")


def plot_csv_p_curves(df, noise_type, models=None, dataset='BNCI2014_001', output_dir='plots', eval_mode=None, hydra=False):
    """
    Plot Cross-Subject Variance (CSV_p) curves using robustness_metrics.py.
    
    Parameters:
    - df: DataFrame with test_perturb results
    - noise_type: str, 'dropout', 'gaussian', or 'eog'
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    - eval_mode: str, evaluation mode (if None, creates separate plots for each)
    """
    # Import robustness metrics functions
    # Handle both relative and absolute imports
    try:
        from robustness_metrics import (
            MetricConfig, ResultsSpec, compute_results_metrics,
            canonicalize_columns, add_normalized_p
        )
    except ImportError:
        try:
            from analysis.robustness_metrics import (
                MetricConfig, ResultsSpec, compute_results_metrics,
                canonicalize_columns, add_normalized_p
            )
        except ImportError as e:
            print(f"[ERROR] Could not import robustness_metrics: {e}")
            print("Make sure analysis/robustness_metrics.py is available.")
            return
    
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Filter data
    df_filtered = df[
        (df['mode'].astype(str).str.contains('test_perturb', na=False)) &
        (df['noise_type'] == noise_type) &
        (df['seed'].isin(valid_seeds))
    ].copy()
    
    if models is not None:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    if df_filtered.empty:
        print(f"No data found for CSV_p curves: {noise_type}")
        return
    
    # Canonicalize columns
    df_filtered = canonicalize_columns(df_filtered)
    
    # Detect metric column
    metric_col = None
    for candidate in ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']:
        if candidate in df_filtered.columns:
            metric_col = candidate
            break
    
    if metric_col is None:
        print("[ERROR] Could not find metric column for CSV_p curve computation")
        return
    
    # Configure metrics
    cfg = MetricConfig(metric_col=metric_col)
    
    # Add normalized p coordinate
    df_filtered = add_normalized_p(
        df_filtered, cfg,
        normalize_within=['dataset', 'noise_type'],
        clip=True
    )
    
    # Get eval_modes to process
    if eval_mode is None:
        eval_modes = sorted(df_filtered['eval_mode'].dropna().unique())
    else:
        # Normalize eval_mode (handle both with and without "Evaluation" suffix)
        available_modes = df_filtered['eval_mode'].dropna().unique()
        if eval_mode + 'Evaluation' in available_modes:
            eval_modes = [eval_mode + 'Evaluation']
        elif eval_mode in available_modes:
            eval_modes = [eval_mode]
        else:
            # Try to find a matching eval_mode
            matching_modes = [m for m in available_modes if eval_mode.lower() in str(m).lower()]
            eval_modes = matching_modes if matching_modes else [eval_mode]
    
    for eval_mode_val in eval_modes:
        df_eval = df_filtered[df_filtered['eval_mode'] == eval_mode_val].copy()
        if df_eval.empty:
            continue
        
        # Ensure 'dataset' column exists (add it if missing)
        if 'dataset' not in df_eval.columns:
            df_eval['dataset'] = dataset
        
        # Filter base_group_cols to only include columns that exist in df_eval
        base_group_cols_candidates = ['dataset', 'tune', 'eval_mode', 'model', 'noise_type']
        base_group_cols = [col for col in base_group_cols_candidates if col in df_eval.columns]
        
        # Compute CSV_p curves
        spec = ResultsSpec(
            base_group_cols=tuple(base_group_cols),
            per_instance_cols=('seed',)
        )
        
        results = compute_results_metrics(df_eval, cfg=cfg, spec=spec, hydra=hydra)
        csv_summary = results.get('csv_summary')
        
        # Replace branched_wiredcfc_arch4 with HYDRA in model names for plotting
        if csv_summary is not None and not csv_summary.empty:
            csv_summary = replace_hydra_model_name(csv_summary, model_col='model')
        
        if csv_summary is None or csv_summary.empty:
            print(f"No CSV_p summary data for {noise_type}, {eval_mode_val}")
            continue
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        tune_settings = [False, True]
        tune_labels = ['Baseline', 'Tuned']
        
        for ax_idx, tune_setting in enumerate(tune_settings):
            ax = axes[ax_idx]
            csv_data = csv_summary[
                (csv_summary['tune'] == tune_setting) &
                (csv_summary['noise_type'] == noise_type)
            ].copy()
            
            if csv_data.empty:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=12, style='italic')
                ax.set_title(tune_labels[ax_idx], fontsize=12, fontweight='bold')
                continue
            
            # Plot CSV_p curves for each model
            for model in csv_data['model'].unique():
                model_data = csv_data[csv_data['model'] == model].copy()
                model_data = model_data.sort_values('p')
                
                ax.plot(model_data['p'], model_data['mean'], 
                       marker='o', linewidth=2, markersize=6, label=model)
                # Add confidence intervals
                if 'ci_low' in model_data.columns and 'ci_high' in model_data.columns:
                    ax.fill_between(model_data['p'], model_data['ci_low'], model_data['ci_high'],
                                   alpha=0.2)
            
            ax.set_xlabel('Normalized Perturbation (p)', fontsize=11)
            ax.set_ylabel('Cross-Subject Variance (CSV_p)', fontsize=11)
            ax.set_title(tune_labels[ax_idx], fontsize=12, fontweight='bold')
            ax.legend(title='Model', fontsize=9, title_fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            # Y-axis limits depend on data, but set reasonable default
            ax.set_ylim(bottom=0)
        
        # Normalize eval_mode for filename
        eval_mode_short = str(eval_mode_val).replace('Evaluation', '')
        
        # Format noise type properly (capitalize EOG, etc.)
        noise_label = format_noise_type_label(noise_type)
        fig.suptitle(f'{dataset} | Cross-Subject Variance | {noise_label} | {eval_mode_short}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        csv_dir = os.path.join(output_dir, dataset, eval_mode_short, 'robustness_metrics')
        os.makedirs(csv_dir, exist_ok=True)
        filename = f"csv_p_curve_{noise_type}_{eval_mode_short}.pdf"
        output_file = os.path.join(csv_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Saved CSV_p curve plot: {output_file}")


def generate_organized_test_perturb_plots(df, models=None, dataset='BNCI2014_001', output_dir='plots', hydra=False):
    """
    Generate all test_perturb plots with organized directory structure.
    Creates separate plots for each eval_mode (CrossSession, WithinSession, CrossSubject).
    
    Parameters:
    - df: DataFrame with all results
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    """
    df = _canonicalize_df_for_test_perturb_plots(df, "generate_organized_test_perturb_plots")
    # Get unique values
    if models is None:
        models = df[df['mode'] == 'test_perturb']['model'].unique()
    
    noise_types = df[df['mode'] == 'test_perturb']['noise_type'].unique()
    # Exclude 'spike' from the list of noise_types
    noise_types = [nt for nt in noise_types if nt != 'spike']
    tune_settings = [False, True]#df[df['mode'] == 'test_perturb']['tune'].unique()
    
    # Get available eval_modes (use original values from dataframe)
    available_eval_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
    eval_modes = sorted(set(available_eval_modes))  # Remove duplicates and sort
    generate_all_compact_plots(df, models, dataset, output_dir, eval_modes=eval_modes, tune_settings=tune_settings)
    print(f"Generating organized plots for {len(models)} models, {len(noise_types)} noise types, {len(tune_settings)} tune settings, {len(eval_modes)} eval modes")
    print(f"Models: {list(models)}")
    print(f"Noise types: {list(noise_types)}")
    print(f"Eval modes: {eval_modes}")

    # Generate combined multi-subject comparison plots (3x2 grid)
    print("\n=== Generating combined multi-subject comparison plots ===")
    for noise_type in noise_types:
        plot_combined_multisubject_comparison(df, noise_type, models, dataset, output_dir, plot_type='line', hydra=hydra)
        plot_combined_multisubject_comparison(df, noise_type, models, dataset, output_dir, plot_type='bar', hydra=hydra)
        # Zoomed versions (y: 0.5–0.8, x: 0–50 intensity) with distinct filenames
        plot_combined_multisubject_comparison(df, noise_type, models, dataset, output_dir, plot_type='line', hydra=hydra, zoomed=True)
        plot_combined_multisubject_comparison(df, noise_type, models, dataset, output_dir, plot_type='bar', hydra=hydra, zoomed=True)

    # Generate combined RD curves (3x2 grid)
    print("\n=== Generating combined Relative Degradation (RD) curves ===")
    for noise_type in noise_types:
        plot_combined_rd_curves(df, noise_type, models, dataset, output_dir, hydra=hydra)

    # Generate combined CSV_p curves (3x2 grid)
    print("\n=== Generating combined Cross-Subject Variance (CSV_p) curves ===")
    for noise_type in noise_types:
        plot_combined_csv_p_curves(df, noise_type, models, dataset, output_dir, hydra=hydra)

    # DISABLED: Individual plot generation (focused on combined plots only)
    # # Generate multi-subject comparison plots for each eval_mode
    # print("\n=== Generating multi-subject comparison plots ===")
    # for eval_mode in eval_modes:
    #     print(f"\n--- Processing {eval_mode} evaluation mode ---")
    #     for noise_type in noise_types:
    #         for tune_setting in tune_settings:
    #             plot_test_perturb_multisubject_comparison(df, noise_type, tune_setting, models, dataset, output_dir, eval_mode=eval_mode)

    # # Generate RD curves (individual plots per eval_mode)
    # print("\n=== Generating Relative Degradation (RD) curves ===")
    # for noise_type in noise_types:
    #     plot_rd_curves(df, noise_type, models, dataset, output_dir)

    # # Generate CSV_p curves (individual plots per eval_mode)
    # print("\n=== Generating Cross-Subject Variance (CSV_p) curves ===")
    # for noise_type in noise_types:
    #     plot_csv_p_curves(df, noise_type, models, dataset, output_dir)

    # # Generate per-subject plots for each model and eval_mode
    # print("\n=== Generating per-subject plots ===")
    # for eval_mode in eval_modes:
    #     eval_mode_short = eval_mode.replace('Evaluation', '')
    #     print(f"\n--- Processing {eval_mode_short} evaluation mode ---")
    #     for model in models:
    #         for noise_type in noise_types:
    #             for tune_setting in tune_settings:
    #                 plot_test_perturb_per_subject(df, model, noise_type, tune_setting, dataset, output_dir, eval_mode=eval_mode)
    
    
    print(f"\nAll organized test_perturb plots generated and saved to {output_dir}")


def generate_model_subset_plots(df, model_subsets, output_dir='plots', dataset='BNCI2014_001'):
    """
    Generate plots for different model subsets for easy comparison.
    
    Parameters:
    - df: DataFrame with all results
    - model_subsets: dict, with subset names as keys and model lists as values
    - output_dir: str, directory to save plots
    - dataset: str, dataset name (affects y-axis limits)
    """
    for subset_name, models in model_subsets.items():
        print(f"\n=== Generating plots for {subset_name} ===")
        subset_output_dir = os.path.join(output_dir, subset_name)
        generate_all_test_perturb_plots(df, models, subset_output_dir, dataset=dataset)


def plot_custom_comparison(df, filters=None, x_var='intensity', y_var='corrupted_score', 
                          hue_var='model', style_var=None, col_var=None, row_var=None,
                          plot_type='line', output_dir='plots', output_filename=None,
                          title=None, xlabel=None, ylabel=None, figsize=(12, 8),
                          include_clean_baseline=True, metric='roc_auc', dataset=None, **kwargs):
    """
    Flexible method to extract and plot specific combinations of data points.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full results dataframe
    filters : dict, optional
        Dictionary of column names and values to filter on. Values can be:
        - Single value: e.g., {'noise_type': 'eog'}
        - List of values: e.g., {'model': ['eegnet', 'cnn_ncp']}
        - Callable for complex filtering: e.g., {'intensity': lambda x: x <= 20.0}
    x_var : str, default='intensity'
        Variable for x-axis
    y_var : str, default='corrupted_score'
        Variable for y-axis
    hue_var : str, default='model'
        Variable to color by (different lines/bars)
    style_var : str, optional
        Variable for line style differentiation
    col_var : str, optional
        Variable for column faceting
    row_var : str, optional
        Variable for row faceting
    plot_type : str, default='line'
        Type of plot: 'line', 'bar', 'scatter', or 'box'
    output_dir : str, default='plots'
        Directory to save the plot
    output_filename : str, optional
        Custom filename for output. If None, auto-generates based on filters
    title : str, optional
        Plot title. If None, auto-generates based on filters
    xlabel : str, optional
        X-axis label. If None, uses x_var with formatting
    ylabel : str, optional
        Y-axis label. If None, uses y_var with formatting
    figsize : tuple, default=(12, 8)
        Figure size
    include_clean_baseline : bool, default=True
        For test_perturb mode, include clean baseline at intensity=0
    metric : str, default='roc_auc'
        Metric to use ('roc_auc', 'accuracy', 'precision', 'recall', 'f1')
    dataset : str, optional
        Dataset name (affects y-axis limits). If None, inferred from df
    **kwargs : additional keyword arguments
        Passed to seaborn plotting function
        
    Returns:
    --------
    tuple : (filtered_df, output_path)
        The filtered dataframe and path where plot was saved
        
    Examples:
    ---------
    # Compare eegnet and cnn_ncp under EOG noise with CrossSession evaluation
    plot_custom_comparison(
        df, 
        filters={
            'model': ['eegnet', 'cnn_ncp'],
            'noise_type': 'eog',
            'eval_mode': 'CrossSession',
            'mode': 'test_perturb',
            'tune': True
        },
        hue_var='model',
        style_var='session'
    )
    
    # Compare performance across different noise types for a single model
    plot_custom_comparison(
        df,
        filters={
            'model': 'eegnet',
            'eval_mode': 'CrossSession',
            'tune': True,
            'intensity': lambda x: x <= 30.0
        },
        hue_var='noise_type',
        col_var='session'
    )
    """
    # Infer dataset if not provided
    if dataset is None and 'dataset' in df.columns:
        unique_datasets = df['dataset'].unique()
        if len(unique_datasets) == 1:
            dataset = unique_datasets[0]
        else:
            # Prefer explicit dataset column when multiple rows share a paradigm (e.g. MotorImagery for BNCI vs Lee2019_MI)
            if 'dataset' in df.columns and len(df) > 0 and pd.notna(df['dataset'].iloc[0]):
                dataset = df['dataset'].iloc[0]
            elif 'paradigm' in df.columns and len(df) > 0:
                paradigm = df['paradigm'].iloc[0]
                if paradigm == 'SSVEP':
                    dataset = 'Lee2019_SSVEP'
                elif paradigm == 'ERP':
                    dataset = 'BI2015a'
                else:
                    dataset = 'BNCI2014_001'
            else:
                dataset = 'BNCI2014_001'  # Default
    elif dataset is None:
        dataset = 'BNCI2014_001'  # Default
    
    # Apply filters
    df_filtered = df.copy()
    filter_summary = []
    
    if filters:
        for col, value in filters.items():
            if callable(value):
                df_filtered = df_filtered[df_filtered[col].apply(value)]
                filter_summary.append(f"{col}=custom")
            elif isinstance(value, (list, tuple)):
                df_filtered = df_filtered[df_filtered[col].isin(value)]
                filter_summary.append(f"{col}={','.join(map(str, value))}")
            else:
                df_filtered = df_filtered[df_filtered[col] == value]
                filter_summary.append(f"{col}={value}")
    
    if df_filtered.empty:
        print(f"No data found matching filters: {filters}")
        return None, None
    
    # Handle metric columns
    if y_var in ['corrupted_score', 'clean_score']:
        clean_col, corrupted_col, y_label_default = _get_metric_columns(metric)
        
        # Check if new metric columns exist, otherwise use legacy
        if metric == 'roc_auc':
            if corrupted_col not in df_filtered.columns and 'corrupted_score' in df_filtered.columns:
                corrupted_col = 'corrupted_score'
            if clean_col not in df_filtered.columns and 'clean_score' in df_filtered.columns:
                clean_col = 'clean_score'
        
        # Update y_var to actual column name
        if y_var == 'corrupted_score':
            y_var = corrupted_col
        elif y_var == 'clean_score':
            y_var = clean_col
    else:
        y_label_default = y_var.replace('_', ' ').title()
    
    # Remove rows with missing y values
    df_filtered = df_filtered.dropna(subset=[y_var])
    
    # Add clean baseline if requested and appropriate
    if include_clean_baseline and 'mode' in df_filtered.columns:
        if 'test_perturb' in df_filtered['mode'].unique():
            clean_col_name = y_var.replace('corrupted', 'clean') if 'corrupted' in y_var else 'clean_score'
            if clean_col_name in df_filtered.columns:
                clean_data = df_filtered.dropna(subset=[clean_col_name]).copy()
                if not clean_data.empty:
                    # Get grouping columns
                    group_cols = [col for col in ['model', 'noise_type', 'seed', 'session', 'subject', 
                                                   'eval_mode', 'tune'] if col in clean_data.columns]
                    
                    clean_summary = clean_data.groupby(group_cols)[clean_col_name].first().reset_index()
                    clean_summary[x_var] = 0.0
                    clean_summary[y_var] = clean_summary[clean_col_name]
                    
                    # Copy other columns
                    for col in df_filtered.columns:
                        if col not in clean_summary.columns:
                            clean_summary[col] = df_filtered[col].iloc[0] if col in df_filtered.columns else None
                    
                    df_filtered = pd.concat([clean_summary, df_filtered], ignore_index=True)
    
    # Set up plot style
    sns.set_theme(style="whitegrid")
    
    # Create facet grid if col_var or row_var specified
    if col_var or row_var:
        g = sns.FacetGrid(df_filtered, col=col_var, row=row_var, height=6, aspect=1.5, 
                         hue=hue_var, palette='tab10')
        
        if plot_type == 'line':
            g.map_dataframe(sns.lineplot, x=x_var, y=y_var, marker='o', 
                           errorbar=('ci', 95), **kwargs)
        elif plot_type == 'bar':
            g.map_dataframe(sns.barplot, x=x_var, y=y_var, 
                           errorbar=('ci', 95), **kwargs)
        elif plot_type == 'scatter':
            g.map_dataframe(sns.scatterplot, x=x_var, y=y_var, s=100, **kwargs)
        elif plot_type == 'box':
            g.map_dataframe(sns.boxplot, x=x_var, y=y_var, **kwargs)
        
        g.add_legend()
        
        if title:
            g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        fig = g.fig
    else:
        # Single plot
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        plot_kwargs = {
            'data': df_filtered,
            'x': x_var,
            'y': y_var,
            'hue': hue_var,
            'ax': ax,
            **kwargs
        }
        
        if style_var:
            plot_kwargs['style'] = style_var
        
        if plot_type == 'line':
            sns.lineplot(**plot_kwargs, marker='o', errorbar=('ci', 95), linewidth=2.5, markersize=8)
        elif plot_type == 'bar':
            sns.barplot(**plot_kwargs, errorbar=('ci', 95))
        elif plot_type == 'scatter':
            sns.scatterplot(**plot_kwargs, s=100)
        elif plot_type == 'box':
            sns.boxplot(**plot_kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        # Labels and title
        ax.set_xlabel(xlabel if xlabel else x_var.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(ylabel if ylabel else y_label_default, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            # Auto-generate title from filters
            title_parts = [f"{k}={v}" for k, v in (filters or {}).items() if not callable(v)]
            if title_parts:
                ax.set_title(' | '.join(title_parts[:3]), fontsize=12)
        
        ax.legend(title=hue_var.replace('_', ' ').title(), fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if y_var in ['score', 'corrupted_score', 'clean_score', 'corrupted_roc_auc', 'clean_roc_auc']:
            # Set y-axis limits based on dataset
            y_min = 0.4 if dataset == 'BNCI2014_001' else 0
            ax.set_ylim(y_min, 1)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    
    if output_filename is None:
        # Auto-generate filename
        filename_parts = ['custom_plot']
        if filters:
            for key, val in list(filters.items())[:3]:
                if not callable(val):
                    val_str = '_'.join(map(str, val)) if isinstance(val, (list, tuple)) else str(val)
                    filename_parts.append(f"{key}_{val_str}")
        output_filename = '_'.join(filename_parts) + '.pdf'
    
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    print(f"Filtered data shape: {df_filtered.shape}")
    
    return df_filtered, output_path


def extract_custom_data(df, filters=None, columns=None, aggregate=None, group_by=None):
    """
    Flexible method to extract specific data points from results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full results dataframe
    filters : dict, optional
        Dictionary of column names and values to filter on (same as plot_custom_comparison)
    columns : list, optional
        Specific columns to return. If None, returns all columns
    aggregate : str or dict, optional
        Aggregation function(s) to apply. Can be:
        - String: 'mean', 'median', 'std', 'min', 'max', 'count'
        - Dict: {col: func} for column-specific aggregations
    group_by : str or list, optional
        Column(s) to group by before aggregation
        
    Returns:
    --------
    pd.DataFrame
        Filtered and/or aggregated dataframe
        
    Examples:
    ---------
    # Get all data for eegnet under EOG noise
    data = extract_custom_data(
        df,
        filters={'model': 'eegnet', 'noise_type': 'eog'}
    )
    
    # Get mean scores by intensity level
    data = extract_custom_data(
        df,
        filters={'model': ['eegnet', 'cnn_ncp'], 'noise_type': 'eog'},
        aggregate='mean',
        group_by=['model', 'intensity']
    )
    """
    # Apply filters
    df_filtered = df.copy()
    
    if filters:
        for col, value in filters.items():
            if callable(value):
                df_filtered = df_filtered[df_filtered[col].apply(value)]
            elif isinstance(value, (list, tuple)):
                df_filtered = df_filtered[df_filtered[col].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[col] == value]
    
    # Select columns
    if columns:
        df_filtered = df_filtered[columns]
    
    # Aggregate if requested
    if aggregate and group_by:
        df_filtered = df_filtered.groupby(group_by).agg(aggregate).reset_index()
    elif aggregate:
        raise ValueError("aggregate requires group_by to be specified")
    
    return df_filtered


def plot_compact_clean_perturbed_comparison(df, noise_type, dataset='BNCI2014_001', models=None, 
                                           output_dir='plots', metric='roc_auc', eval_mode=None,
                                           tune_setting=None, perturbed_label='Perturbed'):
    """
    Create a compact bar plot comparing clean scores vs corrupted scores at maximum perturbation intensity.
    
    This plot is designed for single-column figures in two-column scientific journal format.
    Shows clean score (ROC-AUC) and corrupted score (ROC-AUC) at maximum perturbation intensity
    for each model, grouped by clean/perturbed condition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with test_perturb results
    noise_type : str
        Noise type ('dropout', 'gaussian', 'eog', 'spike')
    dataset : str, default='BNCI2014_001'
        Dataset name
    models : list, optional
        Specific models to include (if None, uses all available models)
    output_dir : str, default='plots'
        Directory to save plots
    metric : str, default='roc_auc'
        Metric to use ('roc_auc' or other metrics)
    eval_mode : str, optional
        Evaluation mode ('CrossSession', 'WithinSession', 'CrossSubject'). If None, uses all available.
    tune_setting : bool, optional
        Tune setting (True for tuned, False for baseline). If None, creates separate plots for both.
    perturbed_label : str, default='Perturbed'
        Label for the perturbed/noisy condition in the legend (e.g. 'Perturbed' or 'Noisy').
    
    Returns:
    --------
    None
        Saves plot to file
    """
    # Load saturation points to get maximum intensity
    saturation_dict = load_saturation_points()
    valid_seeds = [100, 200, 300, 400, 500]
    
    # Get metric columns
    clean_col, corrupted_col, y_label = _get_metric_columns_legacy(metric)
    
    # Get maximum intensity for this dataset and noise type
    if dataset in saturation_dict and noise_type in saturation_dict[dataset]:
        max_intensity = saturation_dict[dataset][noise_type]
    else:
        # Try to infer from data
        df_filtered_temp = df[
            (df['dataset'] == dataset) &
            (df['noise_type'] == noise_type) &
            (df['mode'].astype(str).str.contains('test_perturb', na=False))
        ]
        if 'intensity' in df_filtered_temp.columns and len(df_filtered_temp) > 0:
            max_intensity = df_filtered_temp['intensity'].max()
        else:
            max_intensity = 50.0  # Default fallback
        print(f"[WARNING] No saturation point found for {dataset}/{noise_type}, using inferred max_intensity={max_intensity}")
    
    # Handle eval_mode
    if eval_mode is None:
        available_modes = df[df['mode'].astype(str).str.contains('test_perturb', na=False)]['eval_mode'].unique()
        eval_modes = sorted(set(available_modes))
    else:
        eval_modes = [eval_mode]
    
    # Handle tune_setting
    if tune_setting is None:
        tune_settings = [False, True]
    else:
        tune_settings = [tune_setting]
    
    # Process each eval_mode and tune_setting combination
    for eval_mode_for_filter in eval_modes:
        # Normalize eval_mode format
        if eval_mode_for_filter.endswith('Evaluation'):
            eval_mode_short = eval_mode_for_filter.replace('Evaluation', '')
        else:
            eval_mode_short = eval_mode_for_filter
        
        for tune_setting in tune_settings:
            # Filter data
            df_filtered = df[
                (df['mode'].astype(str).str.contains('test_perturb', na=False)) &
                (df['eval_mode'] == eval_mode_for_filter) &
                (df['noise_type'] == noise_type) &
                (df['tune'] == tune_setting) &
                (df['seed'].isin(valid_seeds))
            ].copy()
            
            # Replace HYDRA model name if needed (before filtering)
            df_filtered = replace_hydra_model_name(df_filtered, model_col='model')
            
            # Filter by specific models if provided
            # Handle both original and formatted model names by normalizing for comparison
            if models is not None:
                # Normalize both the models list and dataframe column for comparison
                def normalize_for_comparison(name):
                    return str(name).lower().strip().replace('-', '_').replace(' ', '_')
                
                df_model_normalized = df_filtered['model'].apply(normalize_for_comparison)
                models_normalized = [normalize_for_comparison(m) for m in models]
                df_filtered = df_filtered[df_model_normalized.isin(models_normalized)]
            
            # DO NOT format model names here - keep original names for data processing
            # We'll format them only at display time (in plot labels)
            
            if df_filtered.empty:
                print(f"[WARNING] No data found for compact plot: {noise_type}, dataset={dataset}, "
                      f"tune={tune_setting}, eval_mode={eval_mode_for_filter}")
                continue
            
            # Extract clean scores
            clean_data = df_filtered.dropna(subset=[clean_col]).copy()
            clean_data = clean_data[clean_data['noise_type'] == noise_type]
            
            # Extract corrupted scores at maximum intensity
            # First, try to find data at the saturation point (max_intensity)
            max_intensity_data = df_filtered[
                intensity_matches(df_filtered['intensity'], [max_intensity])
            ].copy()
            
            # If no data found at saturation point, use the actual maximum intensity in the data
            if max_intensity_data.empty and 'intensity' in df_filtered.columns:
                actual_max_intensity = df_filtered['intensity'].max()
                if not pd.isna(actual_max_intensity) and actual_max_intensity > 0:
                    print(f"[INFO] No data at saturation point {max_intensity} for {noise_type}, "
                          f"using actual max intensity {actual_max_intensity}")
                    max_intensity_data = df_filtered[
                        intensity_matches(df_filtered['intensity'], [actual_max_intensity])
                    ].copy()
            
            max_intensity_data = max_intensity_data.dropna(subset=[corrupted_col])
            
            if clean_data.empty and max_intensity_data.empty:
                print(f"[WARNING] No clean or max intensity data found for {noise_type}, dataset={dataset}")
                continue
            
            # Prepare data for plotting
            plot_data = []
            
            # Get unique models
            all_models = set()
            if not clean_data.empty:
                all_models.update(clean_data['model'].unique())
            if not max_intensity_data.empty:
                all_models.update(max_intensity_data['model'].unique())
            all_models = sorted(all_models)
            
            for model in all_models:
                # Clean scores
                model_clean = clean_data[clean_data['model'] == model][clean_col]
                if len(model_clean) > 0:
                    mean_clean = model_clean.mean()
                    std_clean = model_clean.std()
                    n_clean = len(model_clean)
                    # Calculate 95% CI (using t-distribution approximation for small samples)
                    if n_clean > 1:
                        if HAS_SCIPY:
                            sem = stats.sem(model_clean)
                            ci_95 = stats.t.interval(0.95, n_clean - 1, loc=mean_clean, scale=sem)
                            ci_lower = ci_95[0]
                            ci_upper = ci_95[1]
                        else:
                            # Fallback: use normal approximation (1.96 * SEM)
                            sem = std_clean / np.sqrt(n_clean)
                            ci_lower = mean_clean - 1.96 * sem
                            ci_upper = mean_clean + 1.96 * sem
                    else:
                        ci_lower = mean_clean
                        ci_upper = mean_clean
                    
                    plot_data.append({
                        'model': model,
                        'condition': 'Clean',
                        'mean': mean_clean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'std': std_clean
                    })
                
                # Corrupted scores at max intensity
                model_corrupted = max_intensity_data[max_intensity_data['model'] == model][corrupted_col]
                if len(model_corrupted) > 0:
                    mean_corrupted = model_corrupted.mean()
                    std_corrupted = model_corrupted.std()
                    n_corrupted = len(model_corrupted)
                    # Calculate 95% CI
                    if n_corrupted > 1:
                        if HAS_SCIPY:
                            sem = stats.sem(model_corrupted)
                            ci_95 = stats.t.interval(0.95, n_corrupted - 1, loc=mean_corrupted, scale=sem)
                            ci_lower = ci_95[0]
                            ci_upper = ci_95[1]
                        else:
                            # Fallback: use normal approximation (1.96 * SEM)
                            sem = std_corrupted / np.sqrt(n_corrupted)
                            ci_lower = mean_corrupted - 1.96 * sem
                            ci_upper = mean_corrupted + 1.96 * sem
                    else:
                        ci_lower = mean_corrupted
                        ci_upper = mean_corrupted
                    
                    plot_data.append({
                        'model': model,
                        'condition': perturbed_label,
                        'mean': mean_corrupted,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'std': std_corrupted
                    })
            
            if not plot_data:
                print(f"[WARNING] No plot data prepared for {noise_type}, dataset={dataset}")
                continue
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create the plot
            # Compact size for single-column figure (typically 3.5 inches wide)
            fig, ax = plt.subplots(figsize=(3.5, 4.0), dpi=300)
            
            # Create grouped bar plot
            x_pos = np.arange(len(all_models))
            width = 0.35  # Width of bars
            
            # Get colors for clean and perturbed
            clean_color = '#2E86AB'  # Blue for clean
            perturbed_color = '#A23B72'  # Purple/red for perturbed
            
            # Track if we've added labels to legend
            clean_label_added = False
            perturbed_label_added = False
            
            # Plot bars with error bars
            for i, model in enumerate(all_models):
                model_data = plot_df[plot_df['model'] == model]
                
                # Clean bar
                clean_data_model = model_data[model_data['condition'] == 'Clean']
                if not clean_data_model.empty:
                    clean_mean = clean_data_model['mean'].iloc[0]
                    clean_ci_lower = clean_data_model['ci_lower'].iloc[0]
                    clean_ci_upper = clean_data_model['ci_upper'].iloc[0]
                    # Error bars: [lower_error, upper_error] format for asymmetric error bars
                    clean_error_lower = clean_mean - clean_ci_lower
                    clean_error_upper = clean_ci_upper - clean_mean
                    clean_error = np.array([[clean_error_lower], [clean_error_upper]])
                    
                    label = 'Clean' if not clean_label_added else ''
                    if not clean_label_added:
                        clean_label_added = True
                    
                    ax.bar(i - width/2, clean_mean, width, label=label,
                          color=clean_color, alpha=0.8, edgecolor='black', linewidth=0.5,
                          yerr=clean_error, capsize=3, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
                
                # Perturbed bar
                perturbed_data_model = model_data[model_data['condition'] == perturbed_label]
                if not perturbed_data_model.empty:
                    perturbed_mean = perturbed_data_model['mean'].iloc[0]
                    perturbed_ci_lower = perturbed_data_model['ci_lower'].iloc[0]
                    perturbed_ci_upper = perturbed_data_model['ci_upper'].iloc[0]
                    # Error bars: [lower_error, upper_error] format for asymmetric error bars
                    perturbed_error_lower = perturbed_mean - perturbed_ci_lower
                    perturbed_error_upper = perturbed_ci_upper - perturbed_mean
                    perturbed_error = np.array([[perturbed_error_lower], [perturbed_error_upper]])
                    
                    label = perturbed_label if not perturbed_label_added else ''
                    if not perturbed_label_added:
                        perturbed_label_added = True
                    
                    ax.bar(i + width/2, perturbed_mean, width, label=label,
                          color=perturbed_color, alpha=0.8, edgecolor='black', linewidth=0.5,
                          yerr=perturbed_error, capsize=3, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
            
            # Customize plot according to publication guidelines
            ax.set_xlabel('Model', fontsize=10, fontweight='normal')
            ax.set_ylabel('ROC-AUC', fontsize=10, fontweight='normal')
            ax.set_xticks(x_pos)
            # Format model names for display only at this point (not in the data)
            formatted_model_labels = [format_model_name_for_display(model) for model in all_models]
            ax.set_xticklabels(formatted_model_labels, rotation=45, ha='right', fontsize=9)
            
            # Set y-axis limits based on dataset
            ylim_config = get_plot_ylim_config(dataset, plot_type='performance')
            if ylim_config:
                ax.set_ylim(ylim_config['min'], ylim_config['max'])
            else:
                y_min = 0.4 if dataset == 'BNCI2014_001' else 0.0
                ax.set_ylim(y_min, 1.0)
            
            # Add legend (required by publication guidelines)
            ax.legend(title='Condition', fontsize=9, title_fontsize=9, 
                     frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
            
            # Grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Ensure all lines are at least 0.5 points thick (publication requirement)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            
            # Remove gray background (publication requirement)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            # Tight layout for compact figure
            plt.tight_layout()
            
            # Create output directory
            tune_label = "tuned" if tune_setting else "baseline"
            output_subdir = os.path.join(output_dir, dataset, eval_mode_short, 'compact')
            os.makedirs(output_subdir, exist_ok=True)
            
            # Save plot
            noise_label = format_noise_type_label(noise_type)
            perturbed_suffix = perturbed_label.lower().replace(' ', '_')
            filename = f"compact_clean_{perturbed_suffix}_{noise_type}_{dataset}_{eval_mode_short}_{tune_label}.pdf"
            output_file = os.path.join(output_subdir, filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved compact clean/perturbed comparison plot: {output_file}")


def generate_all_compact_plots(df, models=None, dataset='BNCI2014_001', output_dir='plots', 
                               metric='roc_auc', eval_modes=None, tune_settings=None):
    """
    Generate all compact clean/perturbed comparison plots for all noise types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with test_perturb results
    models : list, optional
        Specific models to include (if None, uses all available models)
    dataset : str, default='BNCI2014_001'
        Dataset name
    output_dir : str, default='plots'
        Directory to save plots
    metric : str, default='roc_auc'
        Metric to use ('roc_auc' or other metrics)
    eval_modes : list, optional
        List of evaluation modes to process. If None, processes all available.
    tune_settings : list, optional
        List of tune settings to process. If None, processes both baseline and tuned.
    
    Returns:
    --------
    None
        Saves plots to files
    """
    # Get unique values
    if models is None:
        models = df[df['mode'].astype(str).str.contains('test_perturb', na=False)]['model'].unique()
    
    noise_types = df[df['mode'].astype(str).str.contains('test_perturb', na=False)]['noise_type'].dropna().unique()
    # Exclude 'spike' if present (optional, based on your needs)
    # noise_types = [nt for nt in noise_types if nt != 'spike']
    
    if eval_modes is None:
        available_eval_modes = df[df['mode'].astype(str).str.contains('test_perturb', na=False)]['eval_mode'].dropna().unique()
        eval_modes = sorted(set(available_eval_modes))
    
    if tune_settings is None:
        tune_settings = [False, True]
    
    print(f"Generating compact plots for {len(models)} models, {len(noise_types)} noise types, "
          f"{len(eval_modes)} eval modes, {len(tune_settings)} tune settings")
    print(f"Models: {list(models)}")
    print(f"Noise types: {list(noise_types)}")
    print(f"Eval modes: {eval_modes}")
    
    # Generate compact plots for each combination
    print("\n=== Generating compact clean/perturbed comparison plots ===")
    for noise_type in noise_types:
        # Use "Noisy" label for EOG plots (HYDRA bar plot variant at max EOG intensity)
        perturbed_label = 'Noisy' if noise_type == 'eog' else 'Perturbed'
        for eval_mode in eval_modes:
            for tune_setting in tune_settings:
                try:
                    plot_compact_clean_perturbed_comparison(
                        df, noise_type, dataset, models, output_dir, 
                        metric, eval_mode, tune_setting,
                        perturbed_label=perturbed_label
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to generate compact plot for {noise_type}, "
                          f"dataset={dataset}, eval_mode={eval_mode}, tune={tune_setting}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\nAll compact plots generated and saved to {output_dir}")


def sanity_check_clean_scores(df, clean_col='clean_score', verbose=True, output_file=None):
    """
    Sanity check: Verify that for each unique combination of (model, dataset, seed, subject, tune, mode,
    session, eval_mode, fold_idx), there is only one unique clean ROC-AUC score, regardless of noise_type.
    
    This check is important because clean scores should be independent of noise type.
    Different noise types should all reference the same clean baseline performance for the
    same model, dataset, seed, subject, tune, mode, session, eval_mode, and fold_idx combination.
    
    Note: Clean scores can legitimately vary by:
    - mode: Different experimental modes (test_perturb vs test_perturb_tune) may have different clean scores
    - session: Different sessions may have different clean scores
    - eval_mode: Different evaluation modes (CrossSession, WithinSession, CrossSubject) may have different clean scores
    - fold_idx: In WithinSession or CrossSubject, different folds may have different clean scores
    
    These dimensions are included in the grouping to ensure we only compare within the same evaluation context.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe with test_perturb results
    clean_col : str
        Column name containing clean scores (default: 'clean_score')
        Will also check 'clean_roc_auc' if clean_col not found
    verbose : bool
        If True, print detailed violation reports
    output_file : str, optional
        Path to save detailed violation report CSV
    
    Returns:
    --------
    dict
        Summary dictionary with:
        - 'passed': bool, whether all checks passed
        - 'total_combinations': int, total unique (model, dataset, seed, subject, tune, mode, session, eval_mode, fold_idx) combinations checked
        - 'violations': int, number of combinations with multiple clean scores
        - 'violation_details': pd.DataFrame, detailed violation report
    """
    print("=" * 80)
    print("SANITY CHECK: Clean Score Consistency")
    print("=" * 80)
    
    # Detect clean metric column
    if clean_col not in df.columns:
        if 'clean_roc_auc' in df.columns:
            clean_col = 'clean_roc_auc'
            print(f"[INFO] Using 'clean_roc_auc' column instead of '{clean_col}'")
        elif 'clean_score' in df.columns:
            clean_col = 'clean_score'
            print(f"[INFO] Using 'clean_score' column")
        else:
            print("[ERROR] No clean score column found. Expected 'clean_score' or 'clean_roc_auc'")
            return {
                'passed': False,
                'total_combinations': 0,
                'violations': 0,
                'violation_details': pd.DataFrame()
            }
    
    # Filter to rows with valid clean scores
    df_filtered = df.dropna(subset=[clean_col]).copy()
    
    if df_filtered.empty:
        print("[WARNING] No rows with valid clean scores found")
        return {
            'passed': False,
            'total_combinations': 0,
            'violations': 0,
            'violation_details': pd.DataFrame()
        }
    
    print(f"\n[INFO] Checking {len(df_filtered)} rows with valid clean scores")
    print(f"[INFO] Clean score column: '{clean_col}'")
    
    # Check and warn if clean scores differ across intensity;
    # clean scores should be identical for all intensity values.
    if 'intensity' in df_filtered.columns:
        print(f"\n[DIAGNOSTIC] Intensity analysis:")
        intensities = df_filtered['intensity'].dropna().unique()
        if len(intensities) == 0:
            print("  [WARNING] No non-NaN intensity values found in data.")
        else:
            print(f"  Unique intensities in data: {len(intensities)}")
            print(f"  Intensity range: {min(intensities):.2f} - {max(intensities):.2f}")

        # Check for any (model, dataset, seed, subject, tune, noise_type, session, eval_mode)
        # groupings where clean scores vary with intensity
        diagnostic_cols = ['model', 'dataset', 'seed', 'subject', 'tune', 'noise_type']
        for dim in ['session', 'eval_mode']:
            if dim in df_filtered.columns:
                diagnostic_cols.append(dim)
                
        inconsistent_combos = []
        groupby_obj = df_filtered.groupby(diagnostic_cols)
        for name, group in groupby_obj:
            unique_clean_scores = group.groupby('intensity')[clean_col].agg(lambda x: tuple(sorted(set(x.dropna()))))
            all_scores = set(val for tup in unique_clean_scores for val in tup)
            if len(all_scores) > 1:
                # Clean scores vary across intensities (should not happen)
                combo_dict = dict(zip(diagnostic_cols, name if isinstance(name, tuple) else (name,)))
                inconsistent_combos.append({
                    'combination': combo_dict,
                    'num_intensities': unique_clean_scores.shape[0],
                    'unique_scores': len(all_scores),
                    'score_range': f"{min(all_scores):.6f} - {max(all_scores):.6f}"
                })
        if inconsistent_combos:
            print(f"  [WARNING] Found {len(inconsistent_combos)} combinations where clean scores vary by intensity")
            print(f"  This suggests a data error: clean scores should always match across intensities.")
        else:
            print(f"  [OK] Clean scores are consistent across intensities (as expected)")

        print(f"\n[INFO] Proceeding: using all clean scores from all intensities (they should all be the same)")

    # Base grouping columns: (model, dataset, seed, subject, tune, mode)
    # These define the level at which clean scores should be consistent
    # NOTE: 'mode' is critical - it distinguishes test_perturb vs test_perturb_tune
    required_base_cols = ['model', 'dataset', 'seed']
    optional_base_cols = ['subject', 'tune', 'mode']
    
    # Check which required columns exist
    missing_required = [col for col in required_base_cols if col not in df_filtered.columns]
    if missing_required:
        print(f"[ERROR] Missing required columns: {missing_required}")
        return {
            'passed': False,
            'total_combinations': 0,
            'violations': 0,
            'violation_details': pd.DataFrame()
        }
    
    # Build base group columns (required + optional if available)
    base_group_cols = required_base_cols.copy()
    for col in optional_base_cols:
        if col in df_filtered.columns:
            base_group_cols.append(col)
    
    # Evaluation context dimensions that can cause legitimate clean score variation
    # These MUST be included in grouping - clean scores can legitimately differ by:
    # - session: Different sessions may have different clean scores
    # - eval_mode: Different evaluation modes (CrossSession, WithinSession, CrossSubject) may have different clean scores
    # - fold_idx: In WithinSession or CrossSubject, different folds may have different clean scores
    evaluation_context_dims = []
    for col in ['session', 'eval_mode', 'fold_idx']:
        if col in df_filtered.columns:
            evaluation_context_dims.append(col)
    
    # Debug: Print available columns to help diagnose missing dimensions
    if verbose:
        print(f"[DEBUG] Available columns in data: {sorted(df_filtered.columns.tolist())}")
        print(f"[DEBUG] Columns checked for evaluation context: ['session', 'eval_mode', 'fold_idx']")
        print(f"[DEBUG] Found evaluation context dimensions: {evaluation_context_dims}")
        # Check if there are any other columns that might represent folds or replicates
        potential_fold_cols = [c for c in df_filtered.columns if 'fold' in c.lower() or 'replicate' in c.lower() or 'run' in c.lower() or 'trial' in c.lower()]
        if potential_fold_cols:
            print(f"[DEBUG] Potential fold/replicate columns found: {potential_fold_cols}")
    
    # Full grouping: base + evaluation context
    # Clean scores should be consistent across noise_types within each full group
    full_group_cols = base_group_cols + evaluation_context_dims
    
    print(f"\n[STEP 1] Checking consistency across noise_type...")
    print(f"[INFO] Base grouping columns: {base_group_cols}")
    if evaluation_context_dims:
        print(f"[INFO] Evaluation context dimensions (clean scores may vary by these): {evaluation_context_dims}")
    print(f"[INFO] Full grouping for comparison: {full_group_cols}")
    
    # Get unique full combinations (base + evaluation context)
    full_combos = df_filtered[full_group_cols].drop_duplicates()
    print(f"[INFO] Found {len(full_combos)} unique {tuple(full_group_cols)} combinations")
    
    violations = []
    
    # Check consistency within each full group
    for idx, combo in full_combos.iterrows():
        # Filter to this full combination
        mask = True
        for col in full_group_cols:
            mask = mask & (df_filtered[col] == combo[col])
        combo_df = df_filtered[mask].copy()
        
        if combo_df.empty:
            continue
        
        # Check if we have multiple noise types
        noise_types = combo_df['noise_type'].dropna().unique()
        
        if len(noise_types) <= 1:
            # Only one or no noise types, nothing to check
            continue
        
        # Within this full group, clean scores should be identical across noise types
        # We've already filtered to a single (model, dataset, seed, subject, tune, session, eval_mode, fold_idx) combination
        # Now check if clean scores are consistent across noise types
        # Use tolerance-based comparison to handle floating-point precision issues
        tolerance = 1e-5
        atol = 1e-8
        
        clean_scores_by_noise = {}
        for noise_type in noise_types:
            noise_df = combo_df[combo_df['noise_type'] == noise_type]
            if not noise_df.empty:
                # Get all clean scores for this noise type (across all intensities)
                # Clean scores should be identical regardless of intensity
                # Use tolerance-based deduplication to handle floating-point precision issues
                clean_scores = noise_df[clean_col].dropna().values
                if len(clean_scores) > 0:
                    # Deduplicate using tolerance-based comparison
                    # Group scores that are close to each other (within tolerance)
                    unique_scores = []
                    for score in clean_scores:
                        # Check if this score is close to any already collected unique score
                        is_unique = True
                        for unique_score in unique_scores:
                            if np.isclose(score, unique_score, rtol=tolerance, atol=atol):
                                is_unique = False
                                break
                        if is_unique:
                            unique_scores.append(score)
                    
                    unique_scores = np.array(unique_scores)
                    # Use median as the representative clean score for this noise type
                    representative_score = np.median(unique_scores)
                    clean_scores_by_noise[noise_type] = {
                        'representative': representative_score,
                        'all_values': unique_scores,
                        'num_unique': len(unique_scores),
                        'range': np.max(unique_scores) - np.min(unique_scores) if len(unique_scores) > 1 else 0.0
                    }
        
        # Check if we have multiple noise types with different clean scores
        if len(clean_scores_by_noise) > 1:
            # Get representative scores (medians) for each noise type
            representative_scores = {nt: info['representative'] for nt, info in clean_scores_by_noise.items()}
            all_representatives = np.array(list(representative_scores.values()))
            
            # Check if all representative scores are close to each other (within tolerance)
            # Use tolerance-based comparison - all scores should be within tolerance of the first one
            all_close = np.allclose(all_representatives, all_representatives[0], rtol=tolerance, atol=atol)
            
            # Calculate differences for reporting
            median_diff = np.max(all_representatives) - np.min(all_representatives)
            
            # Check for intra-noise variation (multiple clean scores within same noise type)
            max_intra_variation = max(info['range'] for info in clean_scores_by_noise.values())
            
            # Flag violation only if scores are NOT close (exceed tolerance)
            # or if there's significant intra-noise variation
            if not all_close or max_intra_variation > tolerance:
                # Collect all unique scores for reporting
                all_unique_scores = set()
                for info in clean_scores_by_noise.values():
                    all_unique_scores.update(info['all_values'])
                
                violation_info = {}
                # Add all dimension values from the full group
                for col in full_group_cols:
                    violation_info[col] = combo[col]
                
                violation_info.update({
                    'num_noise_types': len(clean_scores_by_noise),
                    'unique_clean_scores': len(all_unique_scores),
                    'score_range': f"{min(all_unique_scores):.6f} - {max(all_unique_scores):.6f}",
                    'score_diff': max(all_unique_scores) - min(all_unique_scores),
                    'median_diff': median_diff,
                    'max_intra_noise_variation': max_intra_variation,
                })
                
                # Add per-noise-type representative scores
                for noise_type, rep_score in representative_scores.items():
                    violation_info[f'{noise_type}_representative'] = rep_score
                    info = clean_scores_by_noise[noise_type]
                    if info['num_unique'] > 1:
                        violation_info[f'{noise_type}_num_unique'] = info['num_unique']
                        violation_info[f'{noise_type}_intra_range'] = info['range']
                
                # DIAGNOSTIC: Analyze which columns vary in the original data rows
                # This helps identify what might be causing the clean score differences
                varying_cols_info = {}
                constant_cols_info = {}
                
                # Get all columns in combo_df that are not grouping columns or clean_score
                all_cols_in_combo = list(combo_df.columns)
                # Exclude grouping columns, clean_score, noise_type, intensity, and performance metrics
                # (performance metrics will vary by intensity, which is expected)
                excluded_cols = (full_group_cols + ['clean_score', 'noise_type', 'intensity', 
                                'corrupted_score', 'corrupted_roc_auc', 'corrupted_accuracy', 
                                'corrupted_f1', 'corrupted_precision', 'corrupted_recall',
                                'score', 'relative_drop', 'evaluation_time', 'total_time', 'training_time'])
                diagnostic_cols = [col for col in all_cols_in_combo if col not in excluded_cols]
                
                for col in diagnostic_cols:
                    unique_vals = combo_df[col].dropna().unique()
                    if len(unique_vals) > 1:
                        # Column varies - this might explain the clean score differences
                        varying_cols_info[col] = {
                            'num_unique': len(unique_vals),
                            'sample_values': list(unique_vals)[:5]  # Limit to first 5
                        }
                    elif len(unique_vals) == 1:
                        # Column is constant
                        constant_cols_info[col] = unique_vals[0]
                
                # Add varying columns info to violation (limit to most common ones to avoid bloat)
                if varying_cols_info:
                    # Sort by number of unique values (most varying first)
                    sorted_varying = sorted(varying_cols_info.items(), key=lambda x: x[1]['num_unique'], reverse=True)
                    # Store top 10 varying columns
                    for col, info in sorted_varying[:10]:
                        violation_info[f'varying_{col}_num_unique'] = info['num_unique']
                        violation_info[f'varying_{col}_sample'] = str(info['sample_values'])[:100]  # Limit string length
                
                violations.append(violation_info)
    
    # Create violation details dataframe
    if violations:
        violation_df = pd.DataFrame(violations)
    else:
        violation_df = pd.DataFrame()
    
    # Summary
    num_violations = len(violations)
    passed = (num_violations == 0)
    
    print(f"\n[STEP 2] Summary")
    print(f"  Total combinations checked: {len(full_combos)}")
    print(f"  Violations found: {num_violations}")
    print(f"  Status: {'[OK] PASSED' if passed else '[ERROR] FAILED'}")
    
    if num_violations > 0:
        print(f"\n[ERROR] Found {num_violations} combinations with inconsistent clean scores across noise types")
        print(f"  This indicates a data storage or aggregation issue that must be fixed before proceeding.")
        
        if verbose:
            print("\n[VIOLATION DETAILS]")
            print("-" * 80)
            # Print summary statistics
            print(f"  Average score difference: {violation_df['score_diff'].mean():.6f}")
            print(f"  Max score difference: {violation_df['score_diff'].max():.6f}")
            print(f"  Min score difference: {violation_df['score_diff'].min():.6f}")
            
            # Show first few violations
            print("\n  First 10 violations:")
            display_cols = base_group_cols + ['score_range', 'score_diff']
            display_cols = [col for col in display_cols if col in violation_df.columns]
            print(violation_df[display_cols].head(10).to_string(index=False))
            
            if len(violations) > 10:
                print(f"\n  ... and {len(violations) - 10} more violations")
        
        # Save violation report if requested
        if output_file and not violation_df.empty:
            violation_df.to_csv(output_file, index=False)
            print(f"\n[INFO] Detailed violation report saved to: {output_file}")
        
        # Raise exception to abort execution
        error_msg = (
            f"\n{'=' * 80}\n"
            f"SANITY CHECK FAILED: {num_violations} violations found\n"
            f"{'=' * 80}\n"
            f"For each unique combination of {tuple(full_group_cols)}, clean scores must be\n"
            f"identical across all noise types. This failure indicates a data integrity issue.\n"
            f"\nPlease review the violation details above and in the CSV report:\n"
            f"  {output_file if output_file else 'N/A'}\n"
            f"\nExecution aborted to prevent generating plots with incorrect data.\n"
            f"{'=' * 80}"
        )
        raise SanityCheckError(error_msg)
    else:
        print("\n[OK] All clean scores are consistent across noise types!")
        print(f"  For each {tuple(full_group_cols)} combination, clean scores are the same")
        print("  regardless of noise_type, as expected.")
    
    return {
        'passed': passed,
        'total_combinations': len(full_combos),
        'violations': num_violations,
        'violation_details': violation_df
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for EEG benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Include 'branched_wiredcfc_arch4' model along with core models (eegnet, reegnet, cnn_ncp) and save to 'hydra' subdirectory"
    )
    
    args = parser.parse_args()
    
    # Load unified results from evaluation/results/unified_all_results.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    unified_file = os.path.join(script_dir, '..', 'evaluation', 'results', 'unified_all_results.csv')
    
    if not os.path.exists(unified_file):
        raise FileNotFoundError(
            f"Unified results file not found: {unified_file}\n"
            f"Please run collect_all_results_unified() from evaluation/experiment_utils.py to generate it."
        )
    
    print(f"[INFO] Loading unified results from: {unified_file}")
    unified_df = pd.read_csv(unified_file)
    unified_df = apply_perturb_sweep_mode_canonicalization(
        unified_df, log_label="analyze_results(unified_all_results.csv)"
    )
    print(f"[INFO] Loaded {len(unified_df)} total rows from unified results")
    
    # Filter to only include intended experimental seeds: [100, 200, 300, 400, 500]
    valid_seeds = [100, 200, 300, 400, 500]
    if 'seed' in unified_df.columns:
        initial_len = len(unified_df)
        # Convert seed to numeric, handling any string representations
        unified_df['seed'] = pd.to_numeric(unified_df['seed'], errors='coerce')
        # Filter to valid seeds (drop rows with NaN seeds or seeds not in valid list)
        unified_df = unified_df[unified_df['seed'].isin(valid_seeds)].copy()
        filtered_count = initial_len - len(unified_df)
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} rows with seeds not in {valid_seeds}")
        print(f"[INFO] Remaining rows with valid seeds: {len(unified_df)}")
    else:
        print("[WARNING] No 'seed' column found - cannot filter by seed values")
    
    # Split unified dataframe by dataset
    available_datasets = []
    if 'dataset' not in unified_df.columns:
        raise ValueError("Unified results file missing 'dataset' column")
    
    # Define dataset configurations
    dataset_configs = {
        'BNCI2014_001': {'label': 'MotorImagery/BNCI2014_001'},
        'Lee2019_MI': {'label': 'MotorImagery/Lee2019_MI'},
        'Shin2017A': {'label': 'MotorImagery/Shin2017A'},
        'Lee2019_SSVEP': {'label': 'SSVEP/Lee2019_SSVEP'},
        'BI2015a': {'label': 'ERP/BI2015a'}
    }
    
    for dataset_name, config in dataset_configs.items():
        dataset_df = unified_df[unified_df['dataset'] == dataset_name].copy()
        if not dataset_df.empty:
            print(f"[INFO] Found {len(dataset_df)} rows for dataset: {dataset_name}")
            available_datasets.append((config, dataset_df))
        else:
            print(f"[WARNING] No data found for dataset: {dataset_name}")

    if not available_datasets:
        raise FileNotFoundError("No datasets found in unified results file.")

    # Define model subsets for comparison
    if args.hydra:
        # Include HYDRA (branched_wiredcfc_arch4) along with core models
        # Note: Use 'HYDRA' here since replace_hydra_model_name converts branched_wiredcfc_arch4 to HYDRA
        model_subsets = {
            'main_models': ['eegnet', 'reegnet', 'cnn_ncp', 'ctnet', 'HYDRA'],
            'all_models': None  # Will use all available models
        }
        print("[INFO] Hydra mode enabled: Including 'HYDRA' (branched_wiredcfc_arch4) with core models")
    else:
        model_subsets = {
            'main_models': ['eegnet', 'reegnet', 'cnn_ncp', 'ctnet'],
            # 'cfc_models': ['cnncfc_compact', 'cnncfc_v2'],
            # 'wired_models': ['wiredcfc_arch1', 'wiredcfc_arch2', 'wiredcfc_arch3'],
            'all_models': None  # Will use all available models
        }

    legacy_mode = False

    for config, aggregated_df in available_datasets:
        # CRITICAL: Explicitly exclude hydra_v2 and other hydra variants
        # Also ensure seed filtering is applied (should already be done, but verify)
        if 'model' in aggregated_df.columns:
            # Exclude any model containing 'hydra_v' (which would catch hydra_v2, hydra_v3, etc.)
            model_normalized = aggregated_df['model'].astype(str).str.lower().str.strip().str.replace('-', '_')
            exclude_mask = model_normalized.str.contains('hydra_v', na=False, regex=False)
            excluded_count = exclude_mask.sum()
            if excluded_count > 0:
                excluded_models = aggregated_df.loc[exclude_mask, 'model'].unique()
                print(f"[INFO] Excluding {excluded_count} rows with hydra variants (e.g., {list(excluded_models[:3])}): not part of core experiment")
            aggregated_df = aggregated_df[~exclude_mask].copy()
        
        # Verify seed filtering (should already be done in aggregation, but double-check)
        if 'seed' in aggregated_df.columns:
            valid_seeds = [100, 200, 300, 400, 500]
            initial_count = len(aggregated_df)
            aggregated_df['seed'] = pd.to_numeric(aggregated_df['seed'], errors='coerce')
            aggregated_df = aggregated_df[aggregated_df['seed'].isin(valid_seeds)].copy()
            if len(aggregated_df) < initial_count:
                print(f"[INFO] Additional seed filtering: removed {initial_count - len(aggregated_df)} rows with invalid seeds")
        
        # Replace branched_wiredcfc_arch4 with HYDRA early in the pipeline
        # This ensures consistent naming throughout the analysis
        if args.hydra:
            aggregated_df = replace_hydra_model_name(aggregated_df, model_col='model')
        
        dataset_name = aggregated_df['dataset'].iloc[0] if 'dataset' in aggregated_df.columns else None
        if dataset_name is None:
            raise ValueError(f"Dataset name could not be determined for {config['label']}")
        print(f"\n=== Processing dataset: {dataset_name} ({config['label']}) ===")
        
        # Run sanity check on clean scores
        # print("\n" + "=" * 80)
        # print(f"Running sanity check for {dataset_name}")
        # print("=" * 80)
        # os.makedirs('./analysis', exist_ok=True)
        
        # try:
        #     sanity_result = sanity_check_clean_scores(
        #         aggregated_df,
        #         clean_col='clean_score',
        #         verbose=True,
        #         output_file=os.path.join('./analysis', f'sanity_check_violations_{dataset_name}.csv')
        #     )
        #     print(f"\n[OK] Sanity check PASSED for {dataset_name}")
        # except SanityCheckError as e:
        #     # Print the error message (which already contains detailed information)
        #     print(str(e))
        #     # Exit with error code
        #     sys.exit(1)

        if legacy_mode:
            # Also generate the original plots for backward compatibility
            base_output_dir = './plots/hydra/' if args.hydra else './plots/legacy/'
            output_dir = os.path.join(base_output_dir, dataset_name)
            print("Generating original plots for backward compatibility...")
            generate_all_test_perturb_plots(
                aggregated_df,
                models=model_subsets['main_models'],
                output_dir=output_dir,
                dataset=dataset_name,
                metrics=['score']
            )
        else:
            # Generate organized plots with per-subject breakdowns
            base_output_dir = './plots/hydra/' if args.hydra else './plots/'
            output_dir = base_output_dir
            print("Generating organized plots with per-subject breakdowns...")
            generate_organized_test_perturb_plots(
                aggregated_df,
                models=model_subsets['main_models'],
                dataset=dataset_name,
                output_dir=output_dir,
                hydra=args.hydra
            )
