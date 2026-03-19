#!/usr/bin/env python3
"""
Experiment to compare ReZero initialization strategies for BranchedWiredCfC Architecture 4.

This experiment trains and evaluates two versions of the same model:
1. "backwards_rezero": Current (accidental) implementation where recurrent starts at full strength
2. "correct_rezero": Correct ReZero where residual/identity passes through at initialization

The goal is to determine if the promising robustness results are connected to the
(accidentally backwards) initialization strategy.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.branched_wiredcfc import create_branched_wiredcfc_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from augmentation.noise import EEGNoiseAugmentor
from utils import get_noise_intensities
from evaluation.metrics import compute_classification_metrics
from globals import set_seeds, get_seed


def load_architecture_4(filepath: Optional[str] = None) -> Any:
    """
    Load Architecture 4 wiring configuration.
    
    Args:
        filepath: Optional path to architecture file. If None, uses default location.
        
    Returns:
        ArbitraryWiring instance
    """
    if filepath is None:
        # Try default locations
        default_paths = [
            "outputs/architectures/best_architecture_4_trial_178.json",
            "architecture_refinement/outputs/architectures/best_architecture_4_trial_178.json",
        ]
        for path in default_paths:
            full_path = Path(__file__).parent.parent / path
            if full_path.exists():
                filepath = str(full_path)
                break
        
        if filepath is None:
            raise FileNotFoundError(
                f"Could not find Architecture 4 file. Tried: {default_paths}"
            )
    
    print(f"Loading Architecture 4 from: {filepath}")
    wiring = load_architecture_from_file(filepath)
    return wiring


def create_model_variant(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    wiring: Any,
    strategy: str,
    **kwargs
):
    """
    Create a BranchedWiredCfC model with specified initialization strategy.
    
    Args:
        n_chans: Number of EEG channels
        n_times: Number of time points
        n_outputs: Number of output classes
        wiring: ArbitraryWiring instance
        strategy: Either "backwards_rezero" or "correct_rezero"
        **kwargs: Additional model parameters
        
    Returns:
        Initialized EEGClassifier
    """
    assert strategy in {"backwards_rezero", "correct_rezero"}, \
        f"strategy must be 'backwards_rezero' or 'correct_rezero', got '{strategy}'"
    
    classifier = create_branched_wiredcfc_classifier(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        wiring=wiring,
        residual_init_strategy=strategy,
        **kwargs
    )
    
    return classifier


def evaluate_robustness(
    trained_model: Any,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    dataset: str = "BNCI2014_001",
    noise_types: List[str] = None,
    num_intensities: int = 20
) -> pd.DataFrame:
    """
    Evaluate model robustness under different noise conditions.
    
    Args:
        trained_model: Trained classifier
        X_valid: Validation data (n_samples, n_channels, n_times)
        y_valid: Validation labels
        dataset: Dataset name for intensity bounds
        noise_types: List of noise types to test. Default: ['eog', 'gaussian', 'dropout']
        num_intensities: Number of intensity steps
        
    Returns:
        DataFrame with robustness results
    """
    if noise_types is None:
        noise_types = ['eog', 'gaussian', 'dropout']
    
    results = []
    trained_model.module_.eval()
    
    with torch.no_grad():
        # Compute clean baseline
        y_pred_proba_clean = trained_model.predict_proba(X_valid)
        num_classes = 4 if dataset == "Lee2019_SSVEP" else 2
        metrics_clean = compute_classification_metrics(y_valid, y_pred_proba_clean, num_classes)
        clean_roc_auc = metrics_clean.get('roc_auc', 0.0)
        
        print(f"  Clean ROC-AUC: {clean_roc_auc:.4f}")
        
        # Test each noise type
        for noise_type in noise_types:
            print(f"  Testing {noise_type} noise...")
            
            # Get intensity bounds for this noise type
            intensities = get_noise_intensities(dataset, noise_type, num_steps=num_intensities)
            
            for intensity in intensities:
                # Create corrupted data
                noise_augmentor = EEGNoiseAugmentor(
                    noise_type=noise_type,
                    intensity=intensity,
                    seed=get_seed()
                )
                X_corrupted = noise_augmentor.transform(X_valid)
                
                # Evaluate on corrupted data
                y_pred_proba_corrupted = trained_model.predict_proba(X_corrupted)
                metrics_corrupted = compute_classification_metrics(
                    y_valid, y_pred_proba_corrupted, num_classes
                )
                corrupted_roc_auc = metrics_corrupted.get('roc_auc', 0.0)
                
                # Calculate retention
                retention = (corrupted_roc_auc / clean_roc_auc * 100) if clean_roc_auc > 0 else 0.0
                
                results.append({
                    'noise_type': noise_type,
                    'intensity': intensity,
                    'clean_roc_auc': clean_roc_auc,
                    'corrupted_roc_auc': corrupted_roc_auc,
                    'retention_pct': retention,
                    'performance_drop': clean_roc_auc - corrupted_roc_auc,
                })
    
    return pd.DataFrame(results)


def run_comparison_experiment(
    dataset: str = "BNCI2014_001",
    architecture_file: Optional[str] = None,
    train_both: bool = True,
    model_dir: Optional[str] = None,
    seeds: List[int] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Run the comparison experiment between two initialization strategies.
    
    Args:
        dataset: Dataset to use (must be compatible with UnifiedExperimentRunner)
        architecture_file: Path to Architecture 4 JSON file
        train_both: If True, train both models. If False, load from checkpoints.
        model_dir: Directory to save/load model checkpoints
        seeds: List of random seeds to test (for robustness)
        **model_kwargs: Additional model parameters
        
    Returns:
        Dictionary with comparison results
    """
    if seeds is None:
        seeds = [42, 123, 456]  # Default seeds for multiple runs
    
    print("=" * 80)
    print("REZERO INITIALIZATION STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Seeds: {seeds}")
    print()
    
    # Load Architecture 4
    wiring = load_architecture_4(architecture_file)
    
    # Initialize dimensions (will be set from actual data in first seed iteration)
    n_chans, n_times, n_outputs = None, None, None
    
    all_results = {
        'backwards_rezero': [],
        'correct_rezero': [],
        'comparison': []
    }
    
    # Run experiment for each seed
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'=' * 80}")
        print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'=' * 80}\n")
        
        set_seeds(seed)
        
        # Get data using MOABB paradigm
        # Note: You may need to adjust this based on your dataset configuration
        from config import get_paradigm
        paradigm = get_paradigm(resample=512)
        
        # Load dataset
        if dataset == "BNCI2014_001":
            from moabb.datasets import BNCI2014_001 as DatasetClass
        elif dataset == "Lee2019_SSVEP":
            from moabb.datasets import Lee2019_SSVEP as DatasetClass
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        dataset_obj = DatasetClass()
        
        # Get data for first subject (you can extend this to multiple subjects)
        X, y, metadata = paradigm.get_data(dataset_obj, subjects=[1])
        
        # Ensure X and y are numpy arrays (MOABB sometimes returns different formats)
        X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            # Convert y to numpy array if it's a pandas Series or list/tuple
            y = np.asarray(y)
        
        # Encode labels to integers (in case they're strings)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Simple train/validation split (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
        )
        
        # Get actual dimensions from data (only set once)
        if n_chans is None:
            n_chans = X_train.shape[1]
            n_times = X_train.shape[2]
            n_outputs = len(np.unique(y_encoded))
            print(f"Data dimensions: {n_chans} channels, {n_times} timepoints, {n_outputs} classes")
        
        print(f"Train samples: {len(X_train)}, Valid samples: {len(X_valid)}")
        
        # Test both strategies
        for strategy in ["backwards_rezero", "correct_rezero"]:
            print(f"\n{'-' * 80}")
            print(f"Strategy: {strategy.upper()}")
            print(f"{'-' * 80}")
            
            model_name = f"branched_wiredcfc_arch4_{strategy}_seed{seed}"
            
            # Create model
            print(f"Creating model...")
            model = create_model_variant(
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
                wiring=wiring,
                strategy=strategy,
                **model_kwargs
            )
            
            # Train model
            if train_both:
                print(f"Training model...")
                model.fit(X_train, y_train)
                
                # Save model if directory provided
                if model_dir:
                    model_path = Path(model_dir) / f"{model_name}.pkl"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    # Note: skorch models can be saved with pickle or joblib
                    import joblib
                    joblib.dump(model, model_path)
                    print(f"Saved model to: {model_path}")
            else:
                # Load from checkpoint
                if model_dir:
                    model_path = Path(model_dir) / f"{model_name}.pkl"
                    if model_path.exists():
                        import joblib
                        model = joblib.load(model_path)
                        print(f"Loaded model from: {model_path}")
                    else:
                        print(f"Warning: Model checkpoint not found: {model_path}")
                        print("Training new model...")
                        model.fit(X_train, y_train)
                else:
                    print("No model directory provided. Training new model...")
                    model.fit(X_train, y_train)
            
            # Evaluate robustness
            print(f"Evaluating robustness...")
            robustness_results = evaluate_robustness(
                trained_model=model,
                X_valid=X_valid,
                y_valid=y_valid,
                dataset=dataset
            )
            
            robustness_results['strategy'] = strategy
            robustness_results['seed'] = seed
            all_results[strategy].append(robustness_results)
    
    # Combine results across seeds
    print(f"\n{'=' * 80}")
    print("COMBINING RESULTS")
    print(f"{'=' * 80}\n")
    
    combined_results = {}
    for strategy in ["backwards_rezero", "correct_rezero"]:
        if all_results[strategy]:
            combined_df = pd.concat(all_results[strategy], ignore_index=True)
            combined_results[strategy] = combined_df
            print(f"{strategy}: {len(combined_df)} total measurements")
    
    return {
        'per_seed_results': all_results,
        'combined_results': combined_results,
        'seeds': seeds,
        'dataset': dataset,
        'architecture': 'Architecture 4'
    }


def generate_comparison_report(results: Dict[str, Any], output_file: str):
    """
    Generate a markdown report comparing the two strategies.
    
    Args:
        results: Results dictionary from run_comparison_experiment
        output_file: Path to output markdown file
    """
    combined = results['combined_results']
    
    with open(output_file, 'w') as f:
        f.write("# ReZero Initialization Strategy Comparison\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"**Dataset**: {results['dataset']}\n")
        f.write(f"**Architecture**: {results['architecture']}\n")
        f.write(f"**Seeds**: {', '.join(map(str, results['seeds']))}\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        
        for strategy in ["backwards_rezero", "correct_rezero"]:
            if strategy not in combined:
                continue
                
            df = combined[strategy]
            f.write(f"### {strategy.replace('_', ' ').title()}\n\n")
            
            # Overall statistics
            f.write(f"- **Total measurements**: {len(df)}\n")
            f.write(f"- **Mean clean ROC-AUC**: {df['clean_roc_auc'].mean():.4f} ± {df['clean_roc_auc'].std():.4f}\n")
            f.write(f"- **Mean corrupted ROC-AUC**: {df['corrupted_roc_auc'].mean():.4f} ± {df['corrupted_roc_auc'].std():.4f}\n")
            f.write(f"- **Mean retention**: {df['retention_pct'].mean():.2f}% ± {df['retention_pct'].std():.2f}%\n\n")
            
            # By noise type
            f.write("**Performance by noise type**:\n\n")
            for noise_type in ['eog', 'gaussian', 'dropout']:
                noise_df = df[df['noise_type'] == noise_type]
                if len(noise_df) > 0:
                    f.write(f"- **{noise_type.upper()}**:\n")
                    f.write(f"  - Mean ROC-AUC: {noise_df['corrupted_roc_auc'].mean():.4f}\n")
                    f.write(f"  - Mean retention: {noise_df['retention_pct'].mean():.2f}%\n")
                    f.write(f"  - Best retention: {noise_df['retention_pct'].max():.2f}%\n")
                    f.write(f"  - Worst retention: {noise_df['retention_pct'].min():.2f}%\n\n")
        
        # Direct comparison
        f.write("---\n\n")
        f.write("## Direct Comparison\n\n")
        
        if "backwards_rezero" in combined and "correct_rezero" in combined:
            backwards_df = combined["backwards_rezero"]
            correct_df = combined["correct_rezero"]
            
            # Compare at key intensity levels
            f.write("### Performance at Key Intensity Levels\n\n")
            f.write("| Noise Type | Intensity | Backwards ReZero | Correct ReZero | Difference |\n")
            f.write("|------------|-----------|------------------|----------------|------------|\n")
            
            for noise_type in ['eog', 'gaussian', 'dropout']:
                for intensity in [25, 50, 75, 100]:
                    b_df = backwards_df[(backwards_df['noise_type'] == noise_type) & 
                                       (backwards_df['intensity'].between(intensity-5, intensity+5))]
                    c_df = correct_df[(correct_df['noise_type'] == noise_type) & 
                                     (correct_df['intensity'].between(intensity-5, intensity+5))]
                    
                    if len(b_df) > 0 and len(c_df) > 0:
                        b_ret = b_df['retention_pct'].mean()
                        c_ret = c_df['retention_pct'].mean()
                        diff = b_ret - c_ret
                        f.write(f"| {noise_type} | {intensity}% | {b_ret:.2f}% | {c_ret:.2f}% | {diff:+.2f}% |\n")
            
            # Overall comparison
            f.write("\n### Overall Robustness Comparison\n\n")
            b_retention = backwards_df['retention_pct'].mean()
            c_retention = correct_df['retention_pct'].mean()
            diff = b_retention - c_retention
            
            f.write(f"- **Backwards ReZero** mean retention: {b_retention:.2f}%\n")
            f.write(f"- **Correct ReZero** mean retention: {c_retention:.2f}%\n")
            f.write(f"- **Difference**: {diff:+.2f}% (Backwards - Correct)\n\n")
            
            if diff > 0:
                f.write(f"**Conclusion**: Backwards ReZero is **{abs(diff):.2f}% more robust** on average.\n")
            elif diff < 0:
                f.write(f"**Conclusion**: Correct ReZero is **{abs(diff):.2f}% more robust** on average.\n")
            else:
                f.write(f"**Conclusion**: Both strategies show similar robustness.\n")
        
        f.write("\n---\n\n")
        f.write("## Recommendations\n\n")
        f.write("Based on these results:\n\n")
        f.write("1. If backwards ReZero shows superior robustness, this suggests the ")
        f.write("accidental implementation may have beneficial properties.\n")
        f.write("2. If correct ReZero performs better, this supports implementing Fix Option 2.\n")
        f.write("3. Consider additional analysis of training dynamics and convergence speed.\n")
    
    print(f"\n[OK] Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare ReZero initialization strategies for BranchedWiredCfC Architecture 4'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='BNCI2014_001',
        help='Dataset to use (default: BNCI2014_001)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default=None,
        help='Path to Architecture 4 JSON file (default: auto-detect)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456],
        help='Random seeds to test (default: 42 123 456)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train new models (default: False, will try to load from --model-dir)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Directory to save/load model checkpoints'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='rezero_comparison_report.md',
        help='Output report file (default: rezero_comparison_report.md)'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_comparison_experiment(
        dataset=args.dataset,
        architecture_file=args.architecture,
        train_both=args.train,
        model_dir=args.model_dir,
        seeds=args.seeds
    )
    
    # Generate report
    generate_comparison_report(results, args.output)
    
    # Save raw results
    results_file = args.output.replace('.md', '_raw_results.json')
    # Convert DataFrames to dict for JSON serialization
    json_results = {
        'seeds': results['seeds'],
        'dataset': results['dataset'],
        'architecture': results['architecture'],
    }
    
    # Save DataFrames to CSV instead
    csv_dir = Path(args.output).parent / 'rezero_comparison_data'
    csv_dir.mkdir(exist_ok=True)
    
    for strategy in ["backwards_rezero", "correct_rezero"]:
        if strategy in results['combined_results']:
            csv_file = csv_dir / f"{strategy}_results.csv"
            results['combined_results'][strategy].to_csv(csv_file, index=False)
            print(f"[OK] Saved {strategy} results to: {csv_file}")
    
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
