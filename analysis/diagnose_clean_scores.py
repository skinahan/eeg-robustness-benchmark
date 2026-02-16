"""
Diagnostic script to investigate clean score variance issues.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def diagnose_clean_scores(df, output_dir='./analysis'):
    """
    Diagnose why clean scores vary across noise types.
    """
    print("=" * 80)
    print("DIAGNOSTIC: Clean Score Variance Investigation")
    print("=" * 80)
    
    # Filter to rows with valid clean scores
    df_filtered = df.dropna(subset=['clean_score']).copy()
    print(f"\n[INFO] Total rows with clean scores: {len(df_filtered)}")
    
    # Check if intensity filtering is the issue
    print("\n" + "=" * 80)
    print("INVESTIGATION 1: Are clean scores stored at different intensities?")
    print("=" * 80)
    
    # Group by key dimensions and check intensity distribution
    base_cols = ['model', 'dataset', 'seed', 'subject', 'tune']
    if not all(col in df_filtered.columns for col in base_cols):
        print(f"[ERROR] Missing required columns. Available: {list(df_filtered.columns)}")
        return
    
    # Get a sample violation case from the output
    sample_combo = {
        'model': 'cnn_ncp',
        'dataset': 'BNCI2014_001',
        'seed': 300,
        'subject': 7,
        'tune': True
    }
    
    print(f"\n[INFO] Examining sample case: {sample_combo}")
    
    # Filter to this combination
    mask = True
    for col, val in sample_combo.items():
        mask = mask & (df_filtered[col] == val)
    sample_df = df_filtered[mask].copy()
    
    if sample_df.empty:
        print("[WARNING] No data found for sample case")
        return
    
    print(f"\n[INFO] Found {len(sample_df)} rows for this combination")
    
    # Check intensity distribution by noise_type
    print("\n[INTENSITY DISTRIBUTION BY NOISE TYPE]")
    print("-" * 80)
    for noise_type in sample_df['noise_type'].dropna().unique():
        noise_df = sample_df[sample_df['noise_type'] == noise_type]
        if 'intensity' in noise_df.columns:
            intensities = noise_df['intensity'].dropna().unique()
            clean_scores = noise_df['clean_score'].dropna().unique()
            print(f"\n{noise_type.upper()}:")
            print(f"  Unique intensities: {sorted(intensities)}")
            print(f"  Unique clean scores: {sorted(clean_scores)}")
            print(f"  Number of rows: {len(noise_df)}")
            
            # Show intensity vs clean_score relationship
            if len(intensities) > 0:
                print(f"  Intensity-CleanScore pairs:")
                for intensity in sorted(intensities):
                    int_df = noise_df[noise_df['intensity'] == intensity]
                    if not int_df.empty:
                        scores = int_df['clean_score'].dropna().unique()
                        print(f"    Intensity {intensity}: {sorted(scores)}")
    
    # Check if there are multiple clean scores at the same intensity
    print("\n" + "=" * 80)
    print("INVESTIGATION 2: Are there multiple clean scores at the same intensity?")
    print("=" * 80)
    
    for noise_type in sample_df['noise_type'].dropna().unique():
        noise_df = sample_df[sample_df['noise_type'] == noise_type]
        if 'intensity' in noise_df.columns:
            for intensity in noise_df['intensity'].dropna().unique():
                int_df = noise_df[noise_df['intensity'] == intensity]
                scores = int_df['clean_score'].dropna().unique()
                if len(scores) > 1:
                    print(f"\n[WARNING] {noise_type} at intensity {intensity} has {len(scores)} different clean scores:")
                    print(f"  Scores: {sorted(scores)}")
                    # Check what other dimensions vary
                    group_cols = [col for col in ['session', 'eval_mode'] if col in int_df.columns]
                    if group_cols:
                        print(f"  Varying dimensions: {group_cols}")
                        for _, group in int_df.groupby(group_cols):
                            group_scores = group['clean_score'].dropna().unique()
                            print(f"    {dict(zip(group_cols, [group[col].iloc[0] for col in group_cols]))}: {sorted(group_scores)}")
    
    # Check intensity ranges by noise type
    print("\n" + "=" * 80)
    print("INVESTIGATION 3: Intensity ranges by noise type (full dataset)")
    print("=" * 80)
    
    if 'intensity' in df_filtered.columns:
        for noise_type in df_filtered['noise_type'].dropna().unique():
            noise_df = df_filtered[df_filtered['noise_type'] == noise_type]
            intensities = noise_df['intensity'].dropna().unique()
            print(f"\n{noise_type.upper()}:")
            print(f"  Intensity range: {min(intensities):.2f} - {max(intensities):.2f}")
            print(f"  Unique intensities: {len(intensities)}")
            print(f"  Sample intensities: {sorted(intensities)[:10]}")
    
    # Check if clean scores vary by intensity
    print("\n" + "=" * 80)
    print("INVESTIGATION 4: Do clean scores vary by intensity?")
    print("=" * 80)
    
    # For the sample case, check if clean scores change with intensity
    for noise_type in sample_df['noise_type'].dropna().unique():
        noise_df = sample_df[sample_df['noise_type'] == noise_type]
        if 'intensity' in noise_df.columns:
            intensity_scores = {}
            for intensity in sorted(noise_df['intensity'].dropna().unique()):
                int_df = noise_df[noise_df['intensity'] == intensity]
                scores = int_df['clean_score'].dropna().unique()
                intensity_scores[intensity] = scores
            
            # Check if scores vary across intensities
            all_scores = set()
            for scores in intensity_scores.values():
                all_scores.update(scores)
            
            if len(all_scores) > 1:
                print(f"\n[WARNING] {noise_type} has varying clean scores across intensities:")
                for intensity, scores in sorted(intensity_scores.items()):
                    print(f"  Intensity {intensity}: {sorted(scores)}")
            else:
                print(f"\n[OK] {noise_type} has consistent clean scores across all intensities: {sorted(all_scores)}")
    
    # Save detailed diagnostic report
    os.makedirs(output_dir, exist_ok=True)
    diagnostic_file = os.path.join(output_dir, 'clean_scores_diagnostic_sample.csv')
    sample_df.to_csv(diagnostic_file, index=False)
    print(f"\n[INFO] Detailed diagnostic data saved to: {diagnostic_file}")
    
    # Check what the expected intensity ranges should be
    print("\n" + "=" * 80)
    print("INVESTIGATION 5: Expected intensity ranges (from saturation points)")
    print("=" * 80)
    
    try:
        from analyze_results import load_saturation_points, get_correct_intensities
        
        saturation_dict = load_saturation_points()
        dataset = sample_combo['dataset']
        
        for noise_type in ['dropout', 'gaussian', 'eog']:
            if noise_type in sample_df['noise_type'].values:
                correct_intensities = get_correct_intensities(
                    dataset=dataset, 
                    noise_type=noise_type, 
                    saturation_dict=saturation_dict
                )
                print(f"\n{noise_type.upper()} (expected):")
                print(f"  Intensity range: {min(correct_intensities):.2f} - {max(correct_intensities):.2f}")
                print(f"  Number of points: {len(correct_intensities)}")
                print(f"  Sample: {correct_intensities[:5]}")
                
                # Check what intensities are actually in the data
                noise_df = sample_df[sample_df['noise_type'] == noise_type]
                if 'intensity' in noise_df.columns:
                    actual_intensities = sorted(noise_df['intensity'].dropna().unique())
                    print(f"  Actual intensities in data: {len(actual_intensities)} unique values")
                    print(f"  Actual range: {min(actual_intensities):.2f} - {max(actual_intensities):.2f}")
                    
                    # Check overlap using tolerance-based matching
                    from analyze_results import intensity_matches
                    
                    # Convert to numpy arrays for comparison
                    correct_arr = np.array(correct_intensities)
                    actual_arr = np.array(actual_intensities)
                    
                    # Find matches using tolerance
                    matches_correct = intensity_matches(correct_arr, actual_arr)
                    matches_actual = intensity_matches(actual_arr, correct_arr)
                    
                    num_matched_correct = np.sum(matches_correct)
                    num_matched_actual = np.sum(matches_actual)
                    
                    print(f"  Overlap with expected (tolerance-based): {num_matched_correct}/{len(correct_intensities)}")
                    if num_matched_correct < len(correct_intensities):
                        missing = correct_arr[~matches_correct]
                        print(f"  Missing expected intensities (no close match): {sorted(missing)[:10]}")
                    if num_matched_actual < len(actual_intensities):
                        extra = actual_arr[~matches_actual]
                        print(f"  Extra intensities not in expected (no close match): {sorted(extra)[:10]}")
    except Exception as e:
        print(f"[WARNING] Could not load saturation points: {e}")


if __name__ == '__main__':
    # Load the results file
    results_dirs = ['../sol_results/', '../results/']
    results_file = None
    
    for results_dir in results_dirs:
        csv_path = os.path.join(results_dir, 'MotorImagery/BNCI2014_001/all_results.csv')
        if os.path.exists(csv_path):
            results_file = csv_path
            break
    
    if results_file is None:
        print("[ERROR] Could not find results file")
        sys.exit(1)
    
    print(f"[INFO] Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    
    # Filter to valid seeds
    valid_seeds = [100, 200, 300, 400, 500]
    if 'seed' in df.columns:
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
        df = df[df['seed'].isin(valid_seeds)].copy()
    
    diagnose_clean_scores(df)

