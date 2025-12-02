import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np

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


def aggregate_results(input_dir):
    """
    Aggregates CSVs from a specified directory.

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
            elif 'BI2015a' in input_dir:
                combined_df['dataset'] = 'BI2015a'
            else:
                combined_df['dataset'] = 'BNCI2014_001'  # Default

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
    output_file = os.path.join(output_dir, f"model_comparison_{noise_type}_{run_mode}_{session_type}.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
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
    output_file = os.path.join(output_dir, f"{model_name}_{noise_type}_{session_type}_performance.png")
    print(f"Saving plot to: {output_file}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
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
    out_file = os.path.join(output_dir, f"{model_name}_{'tuned' if tuned else 'baseline'}_subjectwise_performance.png")
    plt.savefig(out_file, dpi=300)
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
    # model_plots(aggregated_df, 'cnn_ncp')
    model_plots(aggregated_df, 'CNN_NCP_RESAMPLE')


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
    df_filtered = df[
        (df['mode'] == 'test_perturb') &
        (df['eval_mode'] == eval_mode_for_filter) &
        (df['model'] == model_name) &
        (df['noise_type'] == noise_type) &
        (df['tune'] == tune_setting) &
        (df['seed'].isin(valid_seeds)) &
        (df['intensity'].isin(correct_intensities))
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
        filename = f"{model_name}_{noise_type}_{metric}_{'tuned' if tune_setting else 'baseline'}_{plot_type}_test_perturb.png"
        output_file = os.path.join(eval_mode_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
        filename = f"master_comparison_{noise_type}_{metric}_{'tuned' if tune_setting else 'baseline'}_{plot_type}_test_perturb.png"
        output_file = os.path.join(eval_mode_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
    df_filtered = df[
        (df['mode'] == 'test_perturb') &
        (df['eval_mode'] == eval_mode_for_filter) &
        (df['model'] == model_name) &
        (df['noise_type'] == noise_type) &
        (df['tune'] == tune_setting) &
        (df['seed'].isin(valid_seeds)) &
        (df['intensity'].isin(correct_intensities))
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
            filename = f"{model_name}_{metric}_{tune_label}_{plot_type}_test_perturb.png"
            output_file = os.path.join(subject_dir, filename)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
        filename = f"model_comparison_{tune_label}_{plot_type}_test_perturb.png"
        output_file = os.path.join(multisubject_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved multisubject {plot_type} plot: {output_file}")


def generate_organized_test_perturb_plots(df, models=None, dataset='BNCI2014_001', output_dir='plots'):
    """
    Generate all test_perturb plots with organized directory structure.
    Creates separate plots for each eval_mode (CrossSession, WithinSession, CrossSubject).
    
    Parameters:
    - df: DataFrame with all results
    - models: list, specific models to include (if None, uses all available models)
    - dataset: str, dataset name for directory organization
    - output_dir: str, base directory to save plots
    """
    # Get unique values
    if models is None:
        models = df[df['mode'] == 'test_perturb']['model'].unique()
    
    noise_types = df[df['mode'] == 'test_perturb']['noise_type'].unique()
    tune_settings = [False, True]#df[df['mode'] == 'test_perturb']['tune'].unique()
    
    # Get available eval_modes (use original values from dataframe)
    available_eval_modes = df[df['mode'] == 'test_perturb']['eval_mode'].unique()
    eval_modes = sorted(set(available_eval_modes))  # Remove duplicates and sort
    
    print(f"Generating organized plots for {len(models)} models, {len(noise_types)} noise types, {len(tune_settings)} tune settings, {len(eval_modes)} eval modes")
    print(f"Models: {list(models)}")
    print(f"Noise types: {list(noise_types)}")
    print(f"Eval modes: {eval_modes}")

    # Generate multi-subject comparison plots for each eval_mode
    print("\n=== Generating multi-subject comparison plots ===")
    for eval_mode in eval_modes:
        print(f"\n--- Processing {eval_mode} evaluation mode ---")
        for noise_type in noise_types:
            for tune_setting in tune_settings:
                plot_test_perturb_multisubject_comparison(df, noise_type, tune_setting, models, dataset, output_dir, eval_mode=eval_mode)

    # Generate per-subject plots for each model and eval_mode
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
            # Default based on paradigm in df if available
            if 'paradigm' in df.columns and len(df) > 0:
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
        output_filename = '_'.join(filename_parts) + '.png'
    
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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


if __name__ == '__main__':

    results_dir = '../sol_results/'
    dataset_configs = [
        {
            'label': 'MotorImagery/BNCI2014_001',
            'input_dir': os.path.join(results_dir, 'MotorImagery/BNCI2014_001/'),
            'csv_path': os.path.join(results_dir, 'MotorImagery/BNCI2014_001/all_results.csv')
        },
        {
            'label': 'SSVEP/Lee2019_SSVEP',
            'input_dir': os.path.join(results_dir, 'SSVEP/Lee2019_SSVEP/'),
            'csv_path': os.path.join(results_dir, 'SSVEP/Lee2019_SSVEP/all_results.csv')
        },
        {
            'label': 'ERP/BI2015a',
            'input_dir': os.path.join(results_dir, 'ERP/BI2015a/'),
            'csv_path': os.path.join(results_dir, 'ERP/BI2015a/all_results.csv')
        }
    ]

    available_datasets = []
    for config in dataset_configs:
        if os.path.exists(config['csv_path']):
            aggregated_df = pd.read_csv(config['csv_path'])
            available_datasets.append((config, aggregated_df))

    if not available_datasets:
        raise FileNotFoundError("No aggregated results found for the expected datasets.")

    # Define model subsets for comparison
    model_subsets = {
        'main_models': ['eegnet', 'reegnet', 'cnn_ncp'],
        # 'cfc_models': ['cnncfc_compact', 'cnncfc_v2'],
        # 'wired_models': ['wiredcfc_arch1', 'wiredcfc_arch2', 'wiredcfc_arch3'],
        'all_models': None  # Will use all available models
    }

    legacy_mode = False

    for config, aggregated_df in available_datasets:
        dataset_name = aggregated_df['dataset'].iloc[0] if 'dataset' in aggregated_df.columns else os.path.basename(os.path.normpath(config['input_dir']))
        print(f"\n=== Processing dataset: {dataset_name} ({config['label']}) ===")

        if legacy_mode:
            # Also generate the original plots for backward compatibility
            output_dir = os.path.join('./plots/legacy/', dataset_name)
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
            output_dir = './plots/'
            print("Generating organized plots with per-subject breakdowns...")
            generate_organized_test_perturb_plots(
                aggregated_df,
                models=model_subsets['main_models'],
                dataset=dataset_name,
                output_dir=output_dir
            )
