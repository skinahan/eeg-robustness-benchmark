import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

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
    listDir = os.listdir(input_dir)
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

        # Set 'dataset' column to 'BNCI2014_001' for all rows
        combined_df['dataset'] = 'BNCI2014_001'

        return combined_df
    else:
        print("No .csv files found in the provided directory.")
        return None

import seaborn as sns
import matplotlib.pyplot as plt
import os

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
    noise_types = ['gaussian', 'dropout', 'eog']
    for mode in modes:
        for session in sessions:
            for noise in noise_types:
                plot_comparative_noise_performance(aggregated_df, noise_type=noise, session_type=session, run_mode=mode,
                                                   output_dir='./plots/')

def run_completion_report(output_dir, aggregated_df):
    # Define the combinations we want to check for
    models = ["eegnet", "reegnet", "cnnncp"]
    seeds = [100, 200, 300, 400, 500]
    noise_types = ["dropout", "gaussian", "eog"]
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

if __name__ == '__main__':
    input_dir = '../results/MotorImagery/BNCI2014_001/'
    aggregated_df = pd.read_csv(os.path.join(input_dir, 'all_results.csv'))
    # run_completion_report(input_dir, aggregated_df)

    # run_comparative_plots(aggregated_df)
    # eegnet_plots(aggregated_df)
    # reegnet_plots(aggregated_df)
    # cnn_ncp_plots(aggregated_df)
    model_plots(aggregated_df, 'cnncfc_v2')

