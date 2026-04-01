#!/usr/bin/env python3
"""
Compare Hydra V2 Results Against Baseline Hydra (branched_wiredcfc_arch4)

This script compares all Hydra V2 variants against the baseline Hydra model
(branched_wiredcfc_arch4) using established analysis patterns from the analysis directory.

It:
- Loads unified results from evaluation/results/unified_all_results.csv
- Filters to Hydra V2 models and branched_wiredcfc_arch4
- Computes robustness metrics (AUPC, RD) per subject
- Performs statistical comparisons (paired tests)
- Generates summary reports and tables

Output:
- Summary tables comparing Hydra V2 variants vs baseline
- Robustness metrics (AUPC, RD) comparisons
- Statistical test results
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# Add project root to path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Analysis utilities
from analysis.calculate_clean_scores import canonicalize_columns
from evaluation.experiment_utils import apply_perturb_sweep_mode_canonicalization
from analysis.statistical_analysis import (
    AnalysisConfig,
    aggregate_seeds,
    compute_aupc_per_subject,
    compute_clean_scores_per_subject,
    compute_rd_per_subject,
    build_inference_dataset,
    check_normality,
    compute_cohens_dz,
    bootstrap_ci_cohens_dz,
)
from analysis.robustness_metrics import (
    MetricConfig,
    add_normalized_p,
    find_subject_col,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

UNIFIED_RESULTS_RELATIVE = os.path.join("evaluation", "results", "unified_all_results.csv")

# Hydra V2 model names (all variants)
HYDRA_V2_MODELS = [
    "hydra_v2",
    "hydra_v2_adaptive_residual",
    "hydra_v2_arch1",
    "hydra_v2_arch4",
    "hydra_v2_baseline",
    "hydra_v2_cross_bin_context",
    "hydra_v2_erp_head",
    "hydra_v2_full",
    "hydra_v2_global_skip",
    "hydra_v2_multi_query",
    "hydra_v2_phase1",
    "hydra_v2_phase2",
    "hydra_v2_phase3",
    "hydra_v2_ssvep_head",
]

BASELINE_MODEL = "branched_wiredcfc_arch4"

# Display names for models
MODEL_DISPLAY_NAMES = {
    "branched_wiredcfc_arch4": "HYDRA (Baseline)",
    "hydra_v2": "HYDRAv2",
    "hydra_v2_baseline": "HYDRAv2 Baseline",
    "hydra_v2_full": "HYDRAv2 Full",
    "hydra_v2_adaptive_residual": "HYDRAv2 Adaptive Residual",
    "hydra_v2_multi_query": "HYDRAv2 Multi-Query",
    "hydra_v2_erp_head": "HYDRAv2 ERP Head",
    "hydra_v2_ssvep_head": "HYDRAv2 SSVEP Head",
    "hydra_v2_cross_bin_context": "HYDRAv2 Cross-Bin Context",
    "hydra_v2_global_skip": "HYDRAv2 Global Skip",
    "hydra_v2_phase1": "HYDRAv2 Phase 1",
    "hydra_v2_phase2": "HYDRAv2 Phase 2",
    "hydra_v2_phase3": "HYDRAv2 Phase 3",
    "hydra_v2_arch1": "HYDRAv2 Arch 1",
    "hydra_v2_arch4": "HYDRAv2 Arch 4",
}


# ----------------------------------------------------------------------
# Data loading / filtering
# ----------------------------------------------------------------------

def load_hydra_comparison_results(
    unified_file: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    eval_modes: Optional[List[str]] = None,
    tune_values: Optional[List[bool]] = None,
) -> pd.DataFrame:
    """
    Load and filter unified results to Hydra V2 vs baseline comparison.
    
    Parameters:
    -----------
    unified_file : str, optional
        Path to unified results CSV. Defaults to evaluation/results/unified_all_results.csv
    datasets : list of str, optional
        Datasets to include. If None, includes all.
    eval_modes : list of str, optional
        Evaluation modes to include. If None, includes all.
    tune_values : list of bool, optional
        Tune values to include. If None, includes all.
    """
    if unified_file is None:
        unified_file = _project_root / UNIFIED_RESULTS_RELATIVE
    unified_file = Path(unified_file)

    if not unified_file.exists():
        raise FileNotFoundError(f"Unified results file not found: {unified_file}")

    print(f"[INFO] Loading unified results from: {unified_file}")

    df = pd.read_csv(unified_file, low_memory=False)
    print(f"[INFO] Loaded {len(df)} total rows from unified file")

    # Canonicalize column names
    df = canonicalize_columns(df)
    df = apply_perturb_sweep_mode_canonicalization(df, log_label="compare_hydra_v2.load_hydra_comparison_results")

    # Filter to Hydra models (baseline + V2 variants)
    if "model" not in df.columns:
        raise KeyError("Expected 'model' column in unified results")

    # Normalize model names for matching
    model_norm = df["model"].astype(str).str.strip().str.lower().str.replace("-", "_")
    
    # Build list of models to include
    models_to_include = [BASELINE_MODEL] + HYDRA_V2_MODELS
    models_normalized = [m.lower().replace("-", "_") for m in models_to_include]
    
    mask = model_norm.isin(models_normalized)
    df = df[mask].copy()
    print(f"[INFO] Filtered to Hydra models: {len(df)} rows")
    
    # Normalize model column to canonical names
    model_map = {m.lower().replace("-", "_"): m for m in models_to_include}
    df["model"] = model_norm.map(model_map)
    df = df[~df["model"].isna()].copy()

    # Filter datasets
    if datasets is not None and "dataset" in df.columns:
        df = df[df["dataset"].isin(datasets)].copy()
        print(f"[INFO] Filtered to datasets {datasets}: {len(df)} rows")

    # Filter eval_modes
    if eval_modes is not None and "eval_mode" in df.columns:
        # Handle variations in eval_mode naming
        eval_mode_norm = df["eval_mode"].astype(str).str.replace("Evaluation", "", regex=False).str.strip()
        mask = pd.Series([False] * len(df))
        for em in eval_modes:
            mask |= eval_mode_norm.str.contains(em, case=False, na=False)
        df = df[mask].copy()
        print(f"[INFO] Filtered to eval_modes {eval_modes}: {len(df)} rows")

    # Filter tune values
    if tune_values is not None and "tune" in df.columns:
        df = df[df["tune"].isin(tune_values)].copy()
        print(f"[INFO] Filtered to tune values {tune_values}: {len(df)} rows")
    elif "tune" in df.columns:
        # Default: include both tuned and untuned
        pass

    # Filter to test_perturb mode
    if "mode" in df.columns:
        mode_norm = df["mode"].astype(str).str.replace("_tune", "", regex=False).str.strip()
        df = df[mode_norm == "test_perturb"].copy()
        print(f"[INFO] Filtered to test_perturb mode: {len(df)} rows")

    # Filter seeds (if present) - include common seeds for both baseline and Hydra V2
    # Note: Baseline uses seeds [100, 200, 300, 400, 500], Hydra V2 may use seed 42
    valid_seeds = [42, 100, 200, 300, 400, 500]
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        before = len(df)
        df = df[df["seed"].isin(valid_seeds)].copy()
        if before != len(df):
            print(f"[INFO] Filtered to seeds {valid_seeds}: {before} -> {len(df)} rows")

    print(f"[OK] Loaded Hydra comparison dataset: {len(df)} rows")
    print(f"[INFO] Models found: {sorted(df['model'].unique())}")
    
    return df


# ----------------------------------------------------------------------
# Robustness metrics computation
# ----------------------------------------------------------------------

def compute_hydra_robustness_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute robustness metrics (AUPC, RD) per subject for all models.
    
    Returns:
    --------
    df_points : pd.DataFrame
        Per-subject metrics at each perturbation level (after seed aggregation)
    df_collapsed : pd.DataFrame
        Subject-level metrics collapsed over noise types
    df_resolved : pd.DataFrame
        Subject-level metrics resolved by noise type
    """
    print("\n[STEP 1] Computing robustness metrics...")

    # Ensure we have required columns
    if "corrupted_roc_auc" not in df.columns:
        if "corrupted_score" in df.columns:
            df["corrupted_roc_auc"] = df["corrupted_score"]
        elif "score" in df.columns:
            df["corrupted_roc_auc"] = df["score"]
        else:
            raise KeyError("Missing corrupted_roc_auc or corrupted_score column")

    if "clean_roc_auc" not in df.columns:
        if "clean_score" in df.columns:
            df["clean_roc_auc"] = df["clean_score"]
        elif "intensity" in df.columns:
            # Use intensity=0 as clean baseline
            clean_data = df[df["intensity"] == 0.0].copy()
            if not clean_data.empty:
                group_cols = ["dataset", "eval_mode", "tune", "subject", "model", "seed"]
                group_cols = [c for c in group_cols if c in df.columns]
                clean_vals = clean_data.groupby(group_cols)["corrupted_roc_auc"].mean().reset_index()
                clean_vals = clean_vals.rename(columns={"corrupted_roc_auc": "clean_roc_auc"})
                df = df.merge(clean_vals, on=group_cols, how="left")
            else:
                raise KeyError("Could not find clean baseline (intensity=0) data")
        else:
            raise KeyError("Missing clean_roc_auc or clean_score column")

    # Compute relative_drop if not present
    if "relative_drop" not in df.columns:
        df["relative_drop"] = (df["clean_roc_auc"] - df["corrupted_roc_auc"]) / df["clean_roc_auc"]
        df["relative_drop"] = df["relative_drop"].replace([np.inf, -np.inf], np.nan)

    # Step 1: Aggregate across seeds at curve-point level
    group_cols = [
        "dataset",
        "eval_mode",
        "tune",
        "subject",
        "model",
        "noise_type",
        "intensity",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    print("  [INFO] Aggregating across seeds...")
    if "seed" in df.columns:
        df_points = aggregate_seeds(df, group_cols)
    else:
        print("  [WARNING] No 'seed' column found; skipping seed aggregation")
        df_points = df.copy()
    print(f"  [OK] Aggregated to {len(df_points)} rows")

    # Step 2: Compute AUPC per subject/model/noise_type
    print("  [INFO] Computing AUPC per subject...")
    config = AnalysisConfig(normalize_aupc=True, rd_summary="mean")
    df_aupc = compute_aupc_per_subject(df_points, config)
    print(f"  [OK] Computed AUPC for {len(df_aupc)} subject-model-noise_type combinations")

    # Step 3: Compute RD per subject/model/noise_type
    print("  [INFO] Computing RD per subject...")
    df_rd = compute_rd_per_subject(df_points, config)
    print(f"  [OK] Computed RD for {len(df_rd)} subject-model-noise_type combinations")

    df_clean = compute_clean_scores_per_subject(df_points, config)

    # Step 4: Build inference dataset
    print("  [INFO] Building inference dataset...")
    df_resolved, df_collapsed = build_inference_dataset(df_aupc, df_rd, df_clean, config)

    # Add model display names
    df_resolved["model_display"] = df_resolved["model"].map(MODEL_DISPLAY_NAMES).fillna(df_resolved["model"])
    df_collapsed["model_display"] = df_collapsed["model"].map(MODEL_DISPLAY_NAMES).fillna(df_collapsed["model"])

    print(f"[OK] Prepared subject-level data:")
    print(f"  - Curve points: {len(df_points)} rows")
    print(f"  - Resolved (by noise type): {len(df_resolved)} rows")
    print(f"  - Collapsed (over noise types): {len(df_collapsed)} rows")

    return df_points, df_collapsed, df_resolved


# ----------------------------------------------------------------------
# Statistical comparisons
# ----------------------------------------------------------------------

def compare_models_vs_baseline(
    df_collapsed: pd.DataFrame,
    primary_metrics: List[str] = None,
    secondary_metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Compare all Hydra V2 models against baseline using paired tests.
    
    Parameters:
    -----------
    df_collapsed : pd.DataFrame
        Subject-level metrics collapsed over noise types
    primary_metrics : list of str
        Primary metrics to compare (default: AUPC, RD)
    secondary_metrics : list of str
        Secondary metrics to compare (default: clean_roc_auc)
    
    Returns:
    --------
    pd.DataFrame
        Comparison results with statistical tests
    """
    if primary_metrics is None:
        primary_metrics = ["aupc_collapsed", "rd_collapsed"]
    if secondary_metrics is None:
        secondary_metrics = ["clean_roc_auc"]

    print("\n[STEP 2] Computing statistical comparisons vs baseline...")

    if df_collapsed.empty:
        print("  [WARNING] No collapsed data available")
        return pd.DataFrame()

    # Determine which metrics are actually present
    all_metrics: List[str] = []
    for m in list(primary_metrics) + list(secondary_metrics):
        if m in df_collapsed.columns:
            all_metrics.append(m)

    if not all_metrics:
        print("  [WARNING] No valid metrics found")
        print(f"  [INFO] Available columns: {list(df_collapsed.columns)}")
        return pd.DataFrame()

    print(f"  [INFO] Primary metrics: {[m for m in primary_metrics if m in all_metrics]}")
    print(f"  [INFO] Secondary metrics: {[m for m in secondary_metrics if m in all_metrics]}")

    # Ensure we have baseline
    baseline_mask = df_collapsed["model"] == BASELINE_MODEL
    if not baseline_mask.any():
        print("  [WARNING] No baseline data found")
        return pd.DataFrame()

    test_results: List[Dict] = []

    # Get unique Hydra V2 models
    hydra_v2_models = [m for m in df_collapsed["model"].unique() if m in HYDRA_V2_MODELS]

    for model in hydra_v2_models:
        model_display = MODEL_DISPLAY_NAMES.get(model, model)

        # Filter to baseline + this model
        comparison_mask = baseline_mask | (df_collapsed["model"] == model)
        comparison_df = df_collapsed[comparison_mask].copy()

        if comparison_df.empty:
            continue

        # Metric-wise comparisons
        for metric in all_metrics:
            if metric not in comparison_df.columns:
                continue

            # Pivot: index = grouping cols, columns = model, values = metric
            group_cols = ["dataset", "eval_mode", "tune", "subject"]
            group_cols = [c for c in group_cols if c in comparison_df.columns]

            pivot_df = comparison_df.pivot_table(
                index=group_cols,
                columns="model",
                values=metric,
                aggfunc="first",
            ).reset_index()

            # Need both baseline and this model
            if BASELINE_MODEL not in pivot_df.columns or model not in pivot_df.columns:
                continue

            baseline_values = pivot_df[BASELINE_MODEL].values
            model_values = pivot_df[model].values

            # Remove NaNs
            mask = ~(np.isnan(baseline_values) | np.isnan(model_values))
            baseline_paired = baseline_values[mask]
            model_paired = model_values[mask]

            if len(baseline_paired) < 2:
                continue

            # Compute descriptive statistics
            baseline_mean = np.mean(baseline_paired)
            baseline_std = np.std(baseline_paired, ddof=1)
            model_mean = np.mean(model_paired)
            model_std = np.std(model_paired, ddof=1)
            mean_diff = model_mean - baseline_mean
            pct_change = (mean_diff / baseline_mean * 100) if baseline_mean != 0 else np.nan

            # Normality check on differences
            diffs = baseline_paired - model_paired
            is_normal = check_normality(diffs)
            parametric = is_normal

            # Run paired test
            if parametric:
                stat, p_value = ttest_rel(baseline_paired, model_paired)
                test_name = "Paired t-test"
            else:
                stat, p_value = wilcoxon(baseline_paired, model_paired, alternative="two-sided")
                test_name = "Wilcoxon signed-rank"

            # Effect size (Cohen's dz)
            cohens_dz = compute_cohens_dz(baseline_paired, model_paired)
            ci_lower, ci_upper = bootstrap_ci_cohens_dz(
                baseline_paired, model_paired, n_reps=10000
            )

            test_results.append({
                "model": model,
                "model_display": model_display,
                "metric": metric,
                "n": len(baseline_paired),
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "model_mean": model_mean,
                "model_std": model_std,
                "mean_diff": mean_diff,
                "pct_change": pct_change,
                "test_statistic": stat,
                "p_value": p_value,
                "test_name": test_name,
                "cohens_dz": cohens_dz,
                "cohens_dz_ci_lower": ci_lower,
                "cohens_dz_ci_upper": ci_upper,
                "parametric": parametric,
            })

    if not test_results:
        print("  [WARNING] No comparison results generated")
        return pd.DataFrame()

    results_df = pd.DataFrame(test_results)
    print(f"  [OK] Generated {len(results_df)} comparison results")

    return results_df


# ----------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------

def generate_summary_tables(
    df_collapsed: pd.DataFrame,
    comparison_results: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate summary tables comparing Hydra V2 models vs baseline.
    
    Returns:
    --------
    dict
        Mapping of table names to file paths
    """
    print("\n[STEP 3] Generating summary tables...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # Table 1: Clean performance summary
    print("  [INFO] Generating clean performance summary...")
    if "clean_roc_auc" in df_collapsed.columns:
        clean_summary = (
            df_collapsed.groupby("model")["clean_roc_auc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        clean_summary["model_display"] = clean_summary["model"].map(MODEL_DISPLAY_NAMES).fillna(clean_summary["model"])
        clean_summary = clean_summary.sort_values("mean", ascending=False)
        
        clean_file = output_dir / f"hydra_v2_clean_performance_{timestamp}.csv"
        clean_summary.to_csv(clean_file, index=False)
        saved_files["clean_performance"] = clean_file
        print(f"    [OK] Saved to {clean_file}")

    # Table 2: AUPC summary
    print("  [INFO] Generating AUPC summary...")
    if "aupc_collapsed" in df_collapsed.columns:
        aupc_summary = (
            df_collapsed.groupby("model")["aupc_collapsed"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aupc_summary["model_display"] = aupc_summary["model"].map(MODEL_DISPLAY_NAMES).fillna(aupc_summary["model"])
        aupc_summary = aupc_summary.sort_values("mean", ascending=False)
        
        aupc_file = output_dir / f"hydra_v2_aupc_summary_{timestamp}.csv"
        aupc_summary.to_csv(aupc_file, index=False)
        saved_files["aupc_summary"] = aupc_file
        print(f"    [OK] Saved to {aupc_file}")

    # Table 3: RD summary
    print("  [INFO] Generating RD summary...")
    if "rd_collapsed" in df_collapsed.columns:
        rd_summary = (
            df_collapsed.groupby("model")["rd_collapsed"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        rd_summary["model_display"] = rd_summary["model"].map(MODEL_DISPLAY_NAMES).fillna(rd_summary["model"])
        rd_summary = rd_summary.sort_values("mean", ascending=True)  # Lower RD is better
        
        rd_file = output_dir / f"hydra_v2_rd_summary_{timestamp}.csv"
        rd_summary.to_csv(rd_file, index=False)
        saved_files["rd_summary"] = rd_file
        print(f"    [OK] Saved to {rd_file}")

    # Table 4: Statistical comparison results
    print("  [INFO] Generating statistical comparison table...")
    if not comparison_results.empty:
        # Format for readability
        comp_formatted = comparison_results.copy()
        comp_formatted["baseline_mean_std"] = (
            comp_formatted["baseline_mean"].round(4).astype(str) + " ± " +
            comp_formatted["baseline_std"].round(4).astype(str)
        )
        comp_formatted["model_mean_std"] = (
            comp_formatted["model_mean"].round(4).astype(str) + " ± " +
            comp_formatted["model_std"].round(4).astype(str)
        )
        comp_formatted["mean_diff_pct"] = (
            comp_formatted["mean_diff"].round(4).astype(str) + " (" +
            comp_formatted["pct_change"].round(2).astype(str) + "%)"
        )
        comp_formatted["p_value_formatted"] = comp_formatted["p_value"].apply(
            lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001"
        )
        comp_formatted["cohens_dz_ci"] = (
            comp_formatted["cohens_dz"].round(3).astype(str) + " [" +
            comp_formatted["cohens_dz_ci_lower"].round(3).astype(str) + ", " +
            comp_formatted["cohens_dz_ci_upper"].round(3).astype(str) + "]"
        )
        
        comp_file = output_dir / f"hydra_v2_statistical_comparisons_{timestamp}.csv"
        comp_formatted.to_csv(comp_file, index=False)
        saved_files["statistical_comparisons"] = comp_file
        print(f"    [OK] Saved to {comp_file}")

    # Generate text summary report
    print("  [INFO] Generating text summary report...")
    report_file = output_dir / f"hydra_v2_comparison_summary_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HYDRA V2 vs BASELINE HYDRA COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Baseline Model: {MODEL_DISPLAY_NAMES.get(BASELINE_MODEL, BASELINE_MODEL)}\n")
        f.write(f"HYDRA V2 Models: {len(HYDRA_V2_MODELS)}\n\n")
        
        if "clean_roc_auc" in df_collapsed.columns:
            f.write("-" * 80 + "\n")
            f.write("CLEAN PERFORMANCE (ROC-AUC)\n")
            f.write("-" * 80 + "\n")
            baseline_clean = df_collapsed[df_collapsed["model"] == BASELINE_MODEL]["clean_roc_auc"].mean()
            f.write(f"Baseline: {baseline_clean:.4f}\n\n")
            
            for model in sorted(df_collapsed["model"].unique()):
                if model == BASELINE_MODEL:
                    continue
                model_clean = df_collapsed[df_collapsed["model"] == model]["clean_roc_auc"].mean()
                diff = model_clean - baseline_clean
                pct = (diff / baseline_clean * 100) if baseline_clean != 0 else 0
                f.write(f"{MODEL_DISPLAY_NAMES.get(model, model):40s}: {model_clean:.4f} ({diff:+.4f}, {pct:+.2f}%)\n")
            f.write("\n")
        
        if "aupc_collapsed" in df_collapsed.columns:
            f.write("-" * 80 + "\n")
            f.write("ROBUSTNESS (AUPC - Higher is Better)\n")
            f.write("-" * 80 + "\n")
            baseline_aupc = df_collapsed[df_collapsed["model"] == BASELINE_MODEL]["aupc_collapsed"].mean()
            f.write(f"Baseline: {baseline_aupc:.4f}\n\n")
            
            for model in sorted(df_collapsed["model"].unique()):
                if model == BASELINE_MODEL:
                    continue
                model_aupc = df_collapsed[df_collapsed["model"] == model]["aupc_collapsed"].mean()
                diff = model_aupc - baseline_aupc
                pct = (diff / baseline_aupc * 100) if baseline_aupc != 0 else 0
                f.write(f"{MODEL_DISPLAY_NAMES.get(model, model):40s}: {model_aupc:.4f} ({diff:+.4f}, {pct:+.2f}%)\n")
            f.write("\n")
        
        if not comparison_results.empty:
            f.write("-" * 80 + "\n")
            f.write("STATISTICAL COMPARISONS (vs Baseline)\n")
            f.write("-" * 80 + "\n")
            for metric in comparison_results["metric"].unique():
                f.write(f"\nMetric: {metric}\n")
                metric_results = comparison_results[comparison_results["metric"] == metric].sort_values("p_value")
                for _, row in metric_results.iterrows():
                    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
                    f.write(f"  {row['model_display']:40s}: p={row['p_value']:.4f} {sig}, "
                           f"dz={row['cohens_dz']:.3f}, diff={row['mean_diff']:+.4f} ({row['pct_change']:+.2f}%)\n")
    
    saved_files["summary_report"] = report_file
    print(f"    [OK] Saved to {report_file}")

    return saved_files


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Hydra V2 models against baseline Hydra"
    )
    parser.add_argument(
        "--unified_file",
        type=str,
        default=None,
        help="Path to unified results CSV (default: evaluation/results/unified_all_results.csv)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to include (default: all)",
    )
    parser.add_argument(
        "--eval_modes",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation modes to include (default: all)",
    )
    parser.add_argument(
        "--tune",
        type=str,
        nargs="+",
        default=None,
        help="Tune values to include: true/false/both (default: both)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis/hydra_v2_comparison",
        help="Output directory for results (default: ./analysis/hydra_v2_comparison)",
    )

    args = parser.parse_args()

    # Parse tune values
    tune_values = None
    if args.tune is not None:
        tune_values = []
        for t in args.tune:
            if t.lower() == "true":
                tune_values.append(True)
            elif t.lower() == "false":
                tune_values.append(False)
            else:
                tune_values.append(bool(t))

    print("=" * 80)
    print("HYDRA V2 vs BASELINE HYDRA COMPARISON")
    print("=" * 80)
    print(f"Baseline: {BASELINE_MODEL}")
    print(f"HYDRA V2 Models: {len(HYDRA_V2_MODELS)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load data
    df = load_hydra_comparison_results(
        unified_file=args.unified_file,
        datasets=args.datasets,
        eval_modes=args.eval_modes,
        tune_values=tune_values,
    )

    # Compute robustness metrics
    df_points, df_collapsed, df_resolved = compute_hydra_robustness_metrics(df)

    # Statistical comparisons
    comparison_results = compare_models_vs_baseline(df_collapsed)

    # Generate summary tables
    output_dir = Path(args.output_dir)
    saved_files = generate_summary_tables(df_collapsed, comparison_results, output_dir)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("\nSaved files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
