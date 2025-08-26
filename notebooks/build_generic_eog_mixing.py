"""
Step 2: Building a generic EOG mixing matrix

This script takes the per-subject mixing matrices from Step 1 and creates a generic
EOG mixing template by averaging across all subjects.

The output is a reusable 19×2 mixing matrix B̄ plus calibration scalars that can
be used to inject realistic EOG artifacts into new EEG data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def load_subject_results(results_dir: Path) -> Dict:
    """Load all subject results from the eog_mixing_results directory."""
    subject_results = {}
    
    # Load summary file first
    summary_file = results_dir / "all_subjects_eog_mixing_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"Found {summary['dataset_info']['n_subjects']} subjects in summary")
    
    # Load individual subject files
    for subject_id in summary['subject_ids']:
        subject_file = results_dir / f"subject_{subject_id:02d}_eog_mixing.json"
        if subject_file.exists():
            with open(subject_file, 'r') as f:
                subject_results[subject_id] = json.load(f)
        else:
            print(f"Warning: Missing file for subject {subject_id}")
    
    print(f"Loaded {len(subject_results)} subject results")
    return subject_results, summary

def compute_generic_mixing_matrix(subject_results: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Compute generic EOG mixing matrix by averaging across all subjects.
    
    Args:
        subject_results: Dictionary of subject results from Step 1
        
    Returns:
        B_generic: Average mixing matrix (19, 2)
        calibration_stats: Dictionary of averaged calibration statistics
    """
    n_subjects = len(subject_results)
    n_channels = 19
    n_regressors = 2
    
    # Extract mixing matrices from all subjects
    mixing_matrices = []
    regressor_stds = []
    target_rms_medians = []
    
    for subject_id, results in subject_results.items():
        # Convert mixing matrix back to numpy array
        B = np.array(results['mixing_matrix'])  # (19, 2)
        mixing_matrices.append(B)
        
        # Extract calibration statistics
        reg_std = np.array(results['regressor_std'])  # (2,)
        regressor_stds.append(reg_std)
        
        target_rms_med = results['target_rms_median']
        target_rms_medians.append(target_rms_med)
    
    # Stack all matrices and compute means
    B_all = np.stack(mixing_matrices, axis=0)  # (n_subjects, 19, 2)
    B_generic = np.mean(B_all, axis=0)  # (19, 2)
    
    # Compute statistics across subjects
    reg_std_all = np.stack(regressor_stds, axis=0)  # (n_subjects, 2)
    reg_std_generic = np.mean(reg_std_all, axis=0)  # (2,)
    reg_std_std = np.std(reg_std_all, axis=0)  # (2,) - standard deviation across subjects
    
    target_rms_med_all = np.array(target_rms_medians)  # (n_subjects,)
    target_rms_med_generic = np.mean(target_rms_med_all)
    target_rms_med_std = np.std(target_rms_med_all)
    
    # Compute channel-wise statistics
    channel_stats = {}
    for ch_idx, ch_name in enumerate(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']):
        # Extract coefficients for this channel across all subjects
        ch_coeffs = B_all[:, ch_idx, :]  # (n_subjects, 2)
        
        channel_stats[ch_name] = {
            'mean_coefficients': B_generic[ch_idx, :].tolist(),  # [VEOG_coeff, HEOG_coeff]
            'std_coefficients': np.std(ch_coeffs, axis=0).tolist(),  # [VEOG_std, HEOG_std]
            'coefficient_range': [
                np.min(ch_coeffs, axis=0).tolist(),  # [VEOG_min, HEOG_min]
                np.max(ch_coeffs, axis=0).tolist()   # [VEOG_max, HEOG_max]
            ]
        }
    
    # Compile calibration statistics
    calibration_stats = {
        'n_subjects': n_subjects,
        'regressor_statistics': {
            'veog': {
                'mean_std': float(reg_std_generic[0]),
                'std_std': float(reg_std_std[0]),
                'range': [float(np.min(reg_std_all[:, 0])), float(np.max(reg_std_all[:, 0]))]
            },
            'heog': {
                'mean_std': float(reg_std_generic[1]),
                'std_std': float(reg_std_std[1]),
                'range': [float(np.min(reg_std_all[:, 1])), float(np.max(reg_std_all[:, 1]))]
            }
        },
        'target_rms_statistics': {
            'mean_median': float(target_rms_med_generic),
            'std_median': float(target_rms_med_std),
            'range': [float(np.min(target_rms_med_all)), float(np.max(target_rms_med_all))]
        },
        'channel_statistics': channel_stats,
        'mixing_matrix_statistics': {
            'mean': B_generic.tolist(),
            'std_across_subjects': np.std(B_all, axis=0).tolist(),
            'range_across_subjects': [
                np.min(B_all, axis=0).tolist(),
                np.max(B_all, axis=0).tolist()
            ]
        }
    }
    
    return B_generic, calibration_stats

def save_generic_mixing_template(B_generic: np.ndarray, calibration_stats: Dict, 
                               output_dir: Path) -> None:
    """Save the generic mixing template in multiple formats."""
    
    # Save as numpy .npz file (most efficient for numerical data)
    npz_file = output_dir / "generic_eog_mixing_template.npz"
    np.savez_compressed(
        npz_file,
        mixing_matrix=B_generic,
        veog_std=calibration_stats['regressor_statistics']['veog']['mean_std'],
        heog_std=calibration_stats['regressor_statistics']['heog']['mean_std'],
        target_rms_median=calibration_stats['target_rms_statistics']['mean_median']
    )
    print(f"Saved numpy template: {npz_file}")
    
    # Save as JSON (human-readable, includes all statistics)
    json_file = output_dir / "generic_eog_mixing_template.json"
    with open(json_file, 'w') as f:
        json.dump(calibration_stats, f, indent=2, default=str)
    print(f"Saved JSON template: {json_file}")
    
    # Save as CSV (simple format for easy inspection)
    csv_file = output_dir / "generic_eog_mixing_matrix.csv"
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    df = pd.DataFrame(
        B_generic,
        index=channel_names,
        columns=['VEOG_coefficient', 'HEOG_coefficient']
    )
    df.to_csv(csv_file)
    print(f"Saved CSV matrix: {csv_file}")
    
    # Save calibration summary as CSV
    cal_csv_file = output_dir / "generic_eog_calibration_summary.csv"
    cal_data = {
        'parameter': ['VEOG_std', 'HEOG_std', 'target_rms_median'],
        'mean': [
            calibration_stats['regressor_statistics']['veog']['mean_std'],
            calibration_stats['regressor_statistics']['heog']['mean_std'],
            calibration_stats['target_rms_statistics']['mean_median']
        ],
        'std': [
            calibration_stats['regressor_statistics']['veog']['std_std'],
            calibration_stats['regressor_statistics']['heog']['std_std'],
            calibration_stats['target_rms_statistics']['std_median']
        ]
    }
    cal_df = pd.DataFrame(cal_data)
    cal_df.to_csv(cal_csv_file, index=False)
    print(f"Saved calibration summary: {cal_csv_file}")

def visualize_generic_mixing(B_generic: np.ndarray, calibration_stats: Dict, 
                           output_dir: Path) -> None:
    """Create visualizations of the generic mixing matrix."""
    
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Generic mixing matrix heatmap
    im1 = axes[0, 0].imshow(B_generic, cmap='RdBu_r', aspect='auto', 
                             vmin=-np.max(np.abs(B_generic)), vmax=np.max(np.abs(B_generic)))
    axes[0, 0].set_title('Generic EOG Mixing Matrix')
    axes[0, 0].set_xlabel('EOG Regressor')
    axes[0, 0].set_ylabel('EEG Channel')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['VEOG', 'HEOG'])
    axes[0, 0].set_yticks(range(len(channel_names)))
    axes[0, 0].set_yticklabels(channel_names)
    plt.colorbar(im1, ax=axes[0, 0], label='Mixing Coefficient')
    
    # 2. VEOG coefficients by channel
    veog_coeffs = B_generic[:, 0]
    axes[0, 1].bar(range(len(channel_names)), veog_coeffs, color='blue', alpha=0.7)
    axes[0, 1].set_title('VEOG Mixing Coefficients by Channel')
    axes[0, 1].set_xlabel('EEG Channel')
    axes[0, 1].set_ylabel('Mixing Coefficient')
    axes[0, 1].set_xticks(range(len(channel_names)))
    axes[0, 1].set_xticklabels(channel_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. HEOG coefficients by channel
    heog_coeffs = B_generic[:, 1]
    axes[0, 2].bar(range(len(channel_names)), heog_coeffs, color='red', alpha=0.7)
    axes[0, 2].set_title('HEOG Mixing Coefficients by Channel')
    axes[0, 2].set_xlabel('EEG Channel')
    axes[0, 2].set_ylabel('Mixing Coefficient')
    axes[0, 2].set_xticks(range(len(channel_names)))
    axes[0, 2].set_xticklabels(channel_names, rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Coefficient variability across subjects
    B_std = np.array(calibration_stats['mixing_matrix_statistics']['std_across_subjects'])
    veog_std = B_std[:, 0]
    heog_std = B_std[:, 1]
    
    x_pos = np.arange(len(channel_names))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, veog_std, width, label='VEOG std', color='blue', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, heog_std, width, label='HEOG std', color='red', alpha=0.7)
    axes[1, 0].set_title('Coefficient Variability Across Subjects')
    axes[1, 0].set_xlabel('EEG Channel')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(channel_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Regressor statistics
    reg_stats = calibration_stats['regressor_statistics']
    reg_names = ['VEOG', 'HEOG']
    reg_means = [reg_stats['veog']['mean_std'], reg_stats['heog']['mean_std']]
    reg_stds = [reg_stats['veog']['std_std'], reg_stats['heog']['std_std']]
    
    axes[1, 1].bar(reg_names, reg_means, yerr=reg_stds, capsize=5, 
                    color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('EOG Regressor Standard Deviations')
    axes[1, 1].set_ylabel('Standard Deviation (V)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Target RMS statistics
    target_stats = calibration_stats['target_rms_statistics']
    axes[1, 2].bar(['Target RMS Median'], [target_stats['mean_median']], 
                    yerr=[target_stats['std_median']], capsize=5, color='green', alpha=0.7)
    axes[1, 2].set_title('Target EOG Artifact RMS')
    axes[1, 2].set_ylabel('RMS (V)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generic_eog_mixing_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_dir / 'generic_eog_mixing_analysis.png'}")

def main():
    """Main function to build the generic EOG mixing template."""
    print("=== STEP 2: Building Generic EOG Mixing Matrix ===\n")
    
    # Load results from Step 1
    results_dir = Path("eog_mixing_results")
    if not results_dir.exists():
        print("Error: eog_mixing_results directory not found. Please run Step 1 first.")
        return
    
    print("Loading subject results from Step 1...")
    subject_results, summary = load_subject_results(results_dir)
    
    if len(subject_results) == 0:
        print("Error: No subject results found. Please run Step 1 first.")
        return
    
    print(f"Successfully loaded {len(subject_results)} subjects")
    
    # Compute generic mixing matrix
    print("\nComputing generic EOG mixing matrix...")
    B_generic, calibration_stats = compute_generic_mixing_matrix(subject_results)
    
    print(f"Generic mixing matrix shape: {B_generic.shape}")
    print(f"VEOG std: {calibration_stats['regressor_statistics']['veog']['mean_std']:.2e} V")
    print(f"HEOG std: {calibration_stats['regressor_statistics']['heog']['mean_std']:.2e} V")
    print(f"Target RMS median: {calibration_stats['target_rms_statistics']['mean_median']:.2e} V")
    
    # Save the generic template
    print("\nSaving generic EOG mixing template...")
    save_generic_mixing_template(B_generic, calibration_stats, results_dir)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_generic_mixing(B_generic, calibration_stats, results_dir)
    
    # Print summary
    print(f"\n=== STEP 2 COMPLETE ===")
    print(f"Generic EOG mixing template created from {len(subject_results)} subjects")
    print(f"Template saved to: {results_dir}")
    print(f"Files created:")
    print(f"  - generic_eog_mixing_template.npz (numpy format)")
    print(f"  - generic_eog_mixing_template.json (detailed statistics)")
    print(f"  - generic_eog_mixing_matrix.csv (matrix only)")
    print(f"  - generic_eog_calibration_summary.csv (calibration summary)")
    print(f"  - generic_eog_mixing_analysis.png (visualization)")
    print(f"\nNext step: Use this generic template to inject realistic EOG artifacts into new EEG data")

if __name__ == "__main__":
    main()
