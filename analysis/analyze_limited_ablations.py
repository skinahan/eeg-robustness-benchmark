#!/usr/bin/env python3
"""
Analyze Limited Ablation Studies (CrossSession, BNCI2014_001)

This script analyzes the limited ablation runs produced by
`ablations/run_limited_ablations.py`.

It:
- Filters the unified results to BNCI2014_001, CrossSession, no tuning
- Extracts baseline + ablations 1–4
  * Baseline: branched_wiredcfc_arch4
  * Ablation 1: branched_wiredcfc_arch4_no_carry_gate
  * Ablation 2: branched_wiredcfc_arch4_no_branching
  * Ablation 3: branched_lstm_arch4_equivalent
  * Ablation 4: branched_wiredcfc_arch4_no_snr_gate
- Aggregates across seeds at the subject level
- Computes robustness metrics (AUPC, RD) per subject
- Builds subject-level inference dataset (collapsed over noise types)
- Runs paired subject-level statistical comparisons (baseline vs each ablation)

Primary outcomes:
- RD (relative degradation) collapsed over noise types
- AUPC (area under perturbation curve) collapsed over noise types

Secondary outcome:
- Clean ROC-AUC
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# Add project root to path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Analysis utilities
from analysis.calculate_clean_scores import (
    canonicalize_columns,
)
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


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

UNIFIED_RESULTS_RELATIVE = os.path.join("evaluation", "results", "unified_all_results.csv")

MODEL_TO_ABLATION = {
    "branched_wiredcfc_arch4": "baseline",
    "branched_wiredcfc_arch4_no_carry_gate": "ablation1_no_carry_gate",
    "branched_wiredcfc_arch4_no_branching": "ablation2_no_branching",
    "branched_lstm_arch4_equivalent": "ablation3_lstm_replacement",
    "branched_wiredcfc_arch4_no_snr_gate": "ablation4_no_snr_gate",
}

ABLATION_NAMES = {
    "baseline": "Baseline (Full HYDRA)",
    "ablation1_no_carry_gate": "Ablation 1: No Carry Gate",
    "ablation2_no_branching": "Ablation 2: No Branching",
    "ablation3_lstm_replacement": "Ablation 3: LSTM Replacement",
    "ablation4_no_snr_gate": "Ablation 4: No SNR Gate",
}


# ----------------------------------------------------------------------
# Data loading / filtering
# ----------------------------------------------------------------------

def load_limited_ablation_results(
    unified_file: Optional[str] = None,
    *,
    datasets: Optional[Sequence[str]] = None,
    eval_mode_substr: Optional[str] = "CrossSession",
) -> pd.DataFrame:
    """
    Load and filter unified results to the limited ablation setting:
    - Dataset(s): BNCI2014_001 by default; pass ``datasets`` to include others
    - Eval mode: substring match (default CrossSession); pass ``eval_mode_substr=None``
      to include all eval modes (use with care)
    - Mode: test_perturb (no *_tune)
    - tune == False (if present)
    - Models: baseline + ablations 1–4
    - Seeds: [100, 200, 300, 400, 500] (if present)
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
    df = apply_perturb_sweep_mode_canonicalization(df, log_label="analyze_limited_ablations.load_limited_ablation_results")

    # Filter dataset
    if "dataset" in df.columns:
        ds = list(datasets) if datasets is not None else ["BNCI2014_001"]
        df = df[df["dataset"].isin(ds)].copy()
        print(f"[INFO] Filtered to datasets {ds}: {len(df)} rows")
    else:
        raise KeyError("Expected 'dataset' column in unified results")

    # Filter tune flag / mode
    if "tune" in df.columns:
        df = df[df["tune"] == False].copy()
        print(f"[INFO] Filtered to tune=False: {len(df)} rows")
    elif "mode" in df.columns:
        mode_norm = df["mode"].astype(str).str.strip()
        df = df[~mode_norm.str.contains("_tune", na=False)].copy()
        print(f"[INFO] Filtered to non-tuned modes: {len(df)} rows")

    # Filter eval_mode (substring, e.g. CrossSession / CrossSubject)
    if eval_mode_substr is not None and "eval_mode" in df.columns:
        eval_mode_norm = (
            df["eval_mode"]
            .astype(str)
            .str.replace("Evaluation", "", regex=False)
            .str.strip()
        )
        df = df[eval_mode_norm.str.contains(eval_mode_substr, case=False, na=False)].copy()
        print(f"[INFO] Filtered to eval_mode containing {eval_mode_substr!r}: {len(df)} rows")
    elif eval_mode_substr is not None:
        raise KeyError("Expected 'eval_mode' column in unified results")

    # Filter mode to test_perturb
    if "mode" in df.columns:
        mode_norm = df["mode"].astype(str).str.replace("_tune", "", regex=False).str.strip()
        df = df[mode_norm == "test_perturb"].copy()
        print(f"[INFO] Filtered to test_perturb mode: {len(df)} rows")

    # Filter models to baseline + ablations 1–4
    if "model" not in df.columns:
        raise KeyError("Expected 'model' column in unified results")

    # Normalize model names (handle dashes vs underscores)
    model_norm = df["model"].astype(str).str.strip().str.lower().str.replace("-", "_")
    df["model_normalized"] = model_norm

    # Build mapping from normalized to canonical
    normalized_to_canonical = {}
    for canonical in MODEL_TO_ABLATION.keys():
        normalized_to_canonical[canonical.lower().replace("-", "_")] = canonical

    df["model_canonical"] = df["model_normalized"].map(normalized_to_canonical)
    df = df[~df["model_canonical"].isna()].copy()
    print(f"[INFO] Filtered to limited ablation models: {len(df)} rows")

    # Replace model column with canonical names for consistency
    df["model"] = df["model_canonical"]
    df = df.drop(columns=["model_normalized", "model_canonical"])

    # Filter seeds
    valid_seeds = [100, 200, 300, 400, 500]
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        before = len(df)
        df = df[df["seed"].isin(valid_seeds)].copy()
        print(f"[INFO] Filtered to seeds {valid_seeds}: {before} -> {len(df)} rows")
    else:
        print("[WARNING] No 'seed' column found; cannot filter by seed values")

    # Add ablation labels
    df["ablation"] = df["model"].map(MODEL_TO_ABLATION)
    if df["ablation"].isna().any():
        unknown = df[df["ablation"].isna()]["model"].unique()
        raise ValueError(f"Found models without ablation mapping: {unknown}")
    df["ablation_name"] = df["ablation"].map(ABLATION_NAMES)

    print(f"[OK] Loaded limited ablation dataset: {len(df)} rows")
    return df


# ----------------------------------------------------------------------
# Clean scores (secondary, descriptive)
# ----------------------------------------------------------------------

def compute_clean_scores_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute clean score summary (mean ± std) per ablation.

    Uses clean_roc_auc / clean_score if available; falls back to
    corrupted_roc_auc at intensity=0 if needed.
    """
    print("\n[STEP 1] Computing clean scores summary...")

    clean_candidates = ["clean_roc_auc", "clean_score", "validation_roc_auc"]
    clean_col = None
    for c in clean_candidates:
        if c in df.columns:
            clean_col = c
            break

    if clean_col is None:
        # Try intensity=0 fallback
        if "intensity" in df.columns and "corrupted_roc_auc" in df.columns:
            clean_data = df[df["intensity"] == 0.0].copy()
            if not clean_data.empty:
                clean_col = "corrupted_roc_auc"
                df = clean_data
                print("[INFO] Using corrupted_roc_auc at intensity=0.0 as clean metric")

    if clean_col is None:
        print("[WARNING] Could not determine clean metric column; skipping clean scores")
        return pd.DataFrame()

    clean_data = df.dropna(subset=[clean_col]).copy()
    if clean_data.empty:
        print("[WARNING] No clean data found; skipping clean scores")
        return pd.DataFrame()

    # Aggregate unique clean scores per (ablation, subject, seed)
    group_cols = ["ablation", "ablation_name"]
    if "seed" in clean_data.columns:
        group_cols.append("seed")
    if "subject" in clean_data.columns:
        group_cols.append("subject")

    rows = []
    for keys, g in clean_data.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        values = g[clean_col].dropna().unique()
        if len(values) > 0:
            row["clean_score"] = float(np.median(values))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    clean_scores = pd.DataFrame(rows)

    # Summary per ablation
    summary_rows = []
    for ablation_key in clean_scores["ablation"].unique():
        vals = clean_scores[clean_scores["ablation"] == ablation_key]["clean_score"].values
        if len(vals) == 0:
            continue
        summary_rows.append(
            {
                "ablation": ablation_key,
                "ablation_name": ABLATION_NAMES.get(ablation_key, ablation_key),
                "clean_score_mean": float(np.mean(vals)),
                "clean_score_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_samples": len(vals),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary

    summary["clean_score_mean_std"] = summary.apply(
        lambda r: f"{r['clean_score_mean']:.4f} ± {r['clean_score_std']:.4f}", axis=1
    )
    print(f"[OK] Computed clean scores for {len(summary)} ablations")
    return summary


# ----------------------------------------------------------------------
# Subject-level robustness pipeline
# ----------------------------------------------------------------------

def prepare_subject_level_data(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare subject-level data for robustness analysis.

    Steps:
    1. Ensure clean_roc_auc, corrupted_roc_auc, relative_drop exist
    2. Aggregate across seeds at curve-point level
    3. Compute AUPC and RD per subject/model/noise_type
    4. Build subject-level resolved and collapsed datasets

    Returns:
        (df_points, df_collapsed, df_resolved)
    """
    print("\n[STEP 2a] Preparing subject-level data for robustness analysis...")

    df = df.copy()

    # Ensure required columns exist
    required_cols = ["ablation", "ablation_name", "subject", "model", "noise_type", "intensity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for robustness analysis: {missing}")

    # Add default columns for compatibility
    if "dataset" not in df.columns:
        df["dataset"] = "BNCI2014_001"
    if "eval_mode" not in df.columns:
        df["eval_mode"] = "CrossSession"
    if "tune" not in df.columns:
        df["tune"] = False

    # Ensure clean_roc_auc
    if "clean_roc_auc" not in df.columns:
        if "clean_score" in df.columns:
            df["clean_roc_auc"] = df["clean_score"]
        elif "intensity" in df.columns and "corrupted_roc_auc" in df.columns:
            clean_data = df[df["intensity"] == 0.0].copy()
            if not clean_data.empty:
                # Map intensity=0 corrupted_roc_auc back as clean_roc_auc
                group_cols = ["ablation", "subject"]
                if "seed" in df.columns:
                    group_cols.append("seed")
                clean_vals = (
                    clean_data.groupby(group_cols)["corrupted_roc_auc"].mean().reset_index()
                )
                clean_vals = clean_vals.rename(columns={"corrupted_roc_auc": "clean_roc_auc"})
                df = df.merge(clean_vals, on=group_cols, how="left")

    # Ensure corrupted_roc_auc
    if "corrupted_roc_auc" not in df.columns:
        for col in ["corrupted_score", "score", "roc_auc"]:
            if col in df.columns:
                df["corrupted_roc_auc"] = df[col]
                break

    # Ensure relative_drop
    if "relative_drop" not in df.columns:
        if "clean_roc_auc" in df.columns and "corrupted_roc_auc" in df.columns:
            df["relative_drop"] = (df["clean_roc_auc"] - df["corrupted_roc_auc"]) / df[
                "clean_roc_auc"
            ]
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
        "ablation",
        "ablation_name",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    if "seed" in df.columns:
        df_points = aggregate_seeds(df, group_cols)
    else:
        print("[INFO] No 'seed' column found; skipping seed aggregation")
        df_points = df.copy()

    # Step 2: Compute AUPC per subject/model/noise_type
    df_aupc = compute_aupc_per_subject(df_points, config)

    # Step 3: Compute RD per subject/model/noise_type
    df_rd = compute_rd_per_subject(df_points, config)

    # Step 3a: Clean ROC-AUC per subject/model (required by build_inference_dataset)
    df_clean = compute_clean_scores_per_subject(df_points, config)

    # Step 3b: Compute worst-case RD (max relative_drop over intensities)
    rd_worst_resolved = pd.DataFrame()
    rd_worst_collapsed = pd.DataFrame()
    if "relative_drop" in df_points.columns:
        # Resolved: per dataset/eval_mode/tune/subject/model/noise_type
        rd_resolved_group_cols = [
            "dataset",
            "eval_mode",
            "tune",
            "subject",
            "model",
            "noise_type",
        ]
        rd_resolved_group_cols = [
            c for c in rd_resolved_group_cols if c in df_points.columns
        ]
        if rd_resolved_group_cols:
            rd_worst_resolved = (
                df_points.groupby(rd_resolved_group_cols)["relative_drop"]
                .max()
                .reset_index()
                .rename(columns={"relative_drop": "rd_worst"})
            )

        # Collapsed: per dataset/eval_mode/tune/subject/model (max over noise types + intensities)
        rd_collapsed_group_cols = [
            "dataset",
            "eval_mode",
            "tune",
            "subject",
            "model",
        ]
        rd_collapsed_group_cols = [
            c for c in rd_collapsed_group_cols if c in df_points.columns
        ]
        if rd_collapsed_group_cols:
            rd_worst_collapsed = (
                df_points.groupby(rd_collapsed_group_cols)["relative_drop"]
                .max()
                .reset_index()
                .rename(columns={"relative_drop": "rd_worst_collapsed"})
            )

    # Step 4: Build inference dataset (resolved + collapsed)
    df_resolved, df_collapsed = build_inference_dataset(df_aupc, df_rd, df_clean, config)

    # Attach worst-case RD to resolved and collapsed datasets if available
    if not df_resolved.empty and not rd_worst_resolved.empty:
        merge_cols_resolved_rd = [
            "dataset",
            "eval_mode",
            "tune",
            "subject",
            "model",
            "noise_type",
        ]
        merge_cols_resolved_rd = [
            c for c in merge_cols_resolved_rd if c in df_resolved.columns
        ]
        if all(c in rd_worst_resolved.columns for c in merge_cols_resolved_rd):
            df_resolved = df_resolved.merge(
                rd_worst_resolved, on=merge_cols_resolved_rd, how="left"
            )

    if not df_collapsed.empty and not rd_worst_collapsed.empty:
        merge_cols_collapsed_rd = [
            "dataset",
            "eval_mode",
            "tune",
            "subject",
            "model",
        ]
        merge_cols_collapsed_rd = [
            c for c in merge_cols_collapsed_rd if c in df_collapsed.columns
        ]
        if all(c in rd_worst_collapsed.columns for c in merge_cols_collapsed_rd):
            df_collapsed = df_collapsed.merge(
                rd_worst_collapsed, on=merge_cols_collapsed_rd, how="left"
            )

    # Add back ablation labels to resolved and collapsed
    if "ablation" in df_points.columns:
        # For resolved: merge on full grouping cols (excluding ablation fields)
        merge_cols_resolved = [
            "dataset",
            "eval_mode",
            "tune",
            "subject",
            "model",
            "noise_type",
        ]
        merge_cols_resolved = [c for c in merge_cols_resolved if c in df_points.columns]

        ablation_map_resolved = (
            df_points.groupby(merge_cols_resolved)[["ablation", "ablation_name"]]
            .first()
            .reset_index()
        )
        if not df_resolved.empty and all(c in df_resolved.columns for c in merge_cols_resolved):
            df_resolved = df_resolved.merge(ablation_map_resolved, on=merge_cols_resolved, how="left")

        # For collapsed: merge on dataset/eval_mode/tune/subject/model
        merge_cols_collapsed = [
            "dataset",
            "eval_mode",
            "tune",
            "subject",
            "model",
        ]
        merge_cols_collapsed = [c for c in merge_cols_collapsed if c in df_points.columns]

        ablation_map_collapsed = (
            df_points.groupby(merge_cols_collapsed)[["ablation", "ablation_name"]]
            .first()
            .reset_index()
        )
        if not df_collapsed.empty and all(c in df_collapsed.columns for c in merge_cols_collapsed):
            df_collapsed = df_collapsed.merge(ablation_map_collapsed, on=merge_cols_collapsed, how="left")

    print(f"[OK] Prepared subject-level data:")
    print(f"  - Curve points: {len(df_points)} rows")
    print(f"  - Resolved (by noise type): {len(df_resolved)} rows")
    print(f"  - Collapsed (over noise types): {len(df_collapsed)} rows")

    return df_points, df_collapsed, df_resolved


def compute_subject_level_statistics(
    df_collapsed: pd.DataFrame,
    primary_metrics: List[str] = None,
    secondary_metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Compute subject-level paired comparisons (baseline vs ablations) for
    robustness metrics and clean performance.

    Uses subjects as the inferential unit (paired t-tests / Wilcoxon).
    """
    if primary_metrics is None:
        primary_metrics = ["aupc_collapsed", "rd_collapsed", "rd_worst_collapsed"]
    if secondary_metrics is None:
        secondary_metrics = ["clean_roc_auc"]

    print("\n[STEP 2b] Computing subject-level statistical comparisons...")

    if df_collapsed.empty:
        print("  [WARNING] No collapsed data available")
        return pd.DataFrame()

    if "ablation" not in df_collapsed.columns:
        print("  [WARNING] No 'ablation' column found in collapsed data")
        return pd.DataFrame()

    # Determine which metrics are actually present
    all_metrics: List[str] = []
    for m in list(primary_metrics) + list(secondary_metrics):
        if m in df_collapsed.columns:
            all_metrics.append(m)

    if not all_metrics:
        print("  [WARNING] No valid metrics found in collapsed data")
        print(f"  [INFO] Available columns: {list(df_collapsed.columns)}")
        return pd.DataFrame()

    print(f"  [INFO] Primary metrics: {[m for m in primary_metrics if m in all_metrics]}")
    print(f"  [INFO] Secondary metrics: {[m for m in secondary_metrics if m in all_metrics]}")

    # Ensure we have baseline
    if not (df_collapsed["ablation"] == "baseline").any():
        print("  [WARNING] No baseline data found in collapsed dataset")
        return pd.DataFrame()

    test_results: List[Dict] = []

    # Unique ablations excluding baseline
    ablation_keys = [k for k in df_collapsed["ablation"].unique() if k != "baseline"]

    for ablation_key in ablation_keys:
        ablation_name = ABLATION_NAMES.get(ablation_key, ablation_key)

        # Filter to baseline + this ablation
        comparison_mask = (df_collapsed["ablation"] == "baseline") | (
            df_collapsed["ablation"] == ablation_key
        )
        comparison_df = df_collapsed[comparison_mask].copy()

        if comparison_df.empty:
            print(f"  [WARNING] No data for {ablation_name}")
            continue

        # Metric-wise comparisons
        for metric in all_metrics:
            if metric not in comparison_df.columns:
                continue

            # Pivot: index = (dataset, eval_mode, tune, subject), columns = ablation, values = metric
            group_cols = ["dataset", "eval_mode", "tune", "subject"]
            group_cols = [c for c in group_cols if c in comparison_df.columns]

            pivot_df = comparison_df.pivot_table(
                index=group_cols,
                columns="ablation",
                values=metric,
                aggfunc="first",
            ).reset_index()

            # Need both baseline and this ablation
            if "baseline" not in pivot_df.columns or ablation_key not in pivot_df.columns:
                print(f"  [WARNING] Missing paired data for {ablation_name} - {metric}")
                continue

            baseline_values = pivot_df["baseline"].values
            ablation_values = pivot_df[ablation_key].values

            # Remove NaNs
            mask = ~(np.isnan(baseline_values) | np.isnan(ablation_values))
            baseline_paired = baseline_values[mask]
            ablation_paired = ablation_values[mask]

            if len(baseline_paired) < 2:
                print(
                    f"  [WARNING] Insufficient paired data for {ablation_name} - {metric} "
                    f"(n={len(baseline_paired)})"
                )
                continue

            # Normality check on differences
            diffs = baseline_paired - ablation_paired
            is_normal = check_normality(diffs)
            parametric = is_normal

            # Run paired test
            if parametric:
                try:
                    statistic, p_value = ttest_rel(baseline_paired, ablation_paired)
                    test_type = "paired_ttest"
                except Exception as e:
                    print(f"  [WARNING] T-test failed for {ablation_name} - {metric}: {e}")
                    statistic, p_value = np.nan, np.nan
                    test_type = "failed"
            else:
                from scipy.stats import wilcoxon

                try:
                    statistic, p_value = wilcoxon(
                        baseline_paired, ablation_paired, alternative="two-sided"
                    )
                    test_type = "wilcoxon"
                except Exception as e:
                    print(f"  [WARNING] Wilcoxon failed for {ablation_name} - {metric}: {e}")
                    statistic, p_value = np.nan, np.nan
                    test_type = "failed"

            # Effect size
            cohens_dz = compute_cohens_dz(baseline_paired, ablation_paired)

            # Bootstrap CI for Cohen's dz
            try:
                data_hash = hash((tuple(baseline_paired), tuple(ablation_paired))) % (2**31)
                ci_low, ci_high = bootstrap_ci_cohens_dz(
                    baseline_paired,
                    ablation_paired,
                    n_reps=10000,
                    random_seed=int(data_hash),
                )
            except Exception as e:
                print(
                    f"  [WARNING] Bootstrap CI failed for {ablation_name} - {metric}: {e}"
                )
                ci_low, ci_high = np.nan, np.nan

            baseline_mean = float(np.mean(baseline_paired))
            baseline_std = float(np.std(baseline_paired, ddof=1))
            ablation_mean = float(np.mean(ablation_paired))
            ablation_std = float(np.std(ablation_paired, ddof=1))
            mean_diff = baseline_mean - ablation_mean

            is_primary = metric in primary_metrics

            test_results.append(
                {
                    "ablation": ablation_key,
                    "ablation_name": ablation_name,
                    "metric": metric,
                    "metric_type": "primary" if is_primary else "secondary",
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "ablation_mean": ablation_mean,
                    "ablation_std": ablation_std,
                    "mean_difference": mean_diff,
                    "test_type": test_type,
                    "parametric": parametric,
                    "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "cohens_dz": float(cohens_dz) if np.isfinite(cohens_dz) else np.nan,
                    "cohens_dz_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "cohens_dz_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    "n_subjects": len(baseline_paired),
                }
            )

            print(
                f"  {ablation_name} - {metric} ({'primary' if is_primary else 'secondary'}): "
                f"n={len(baseline_paired)}, {test_type}, p={p_value:.4f} "
                f"{'*' if np.isfinite(p_value) and p_value < 0.05 else ''}"
            )

    if not test_results:
        return pd.DataFrame()

    stats_df = pd.DataFrame(test_results)

    # Bonferroni correction within primary vs secondary families
    for family in ["primary", "secondary"]:
        mask = stats_df["metric_type"] == family
        num_tests = mask.sum()
        if num_tests > 0:
            stats_df.loc[mask, "p_value_corrected"] = stats_df.loc[mask, "p_value"].apply(
                lambda p: min(p * num_tests, 1.0) if np.isfinite(p) else np.nan
            )
            stats_df.loc[mask, "significant"] = (
                stats_df.loc[mask, "p_value_corrected"] < 0.05
            )

    print(
        f"[OK] Computed subject-level statistics for "
        f"{len(stats_df)} ablation/metric combinations"
    )
    return stats_df


def compute_noise_type_resolved_statistics(
    df_resolved: pd.DataFrame,
    primary_metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Compute subject-level paired comparisons (baseline vs ablations) for
    robustness metrics, **separated by noise type**.

    Uses subjects as the inferential unit (paired t-tests / Wilcoxon).
    Tests are performed for each noise type separately to detect targeted
    robustness trade-offs that may be diluted when collapsing over noise types.
    """
    if primary_metrics is None:
        primary_metrics = ["aupc_roc_auc", "rd_mean", "rd_worst"]

    print("\n[STEP 2c] Computing noise-type-resolved statistical comparisons...")

    if df_resolved.empty:
        print("  [WARNING] No resolved data available")
        return pd.DataFrame()

    if "ablation" not in df_resolved.columns or "noise_type" not in df_resolved.columns:
        print("  [WARNING] Missing 'ablation' or 'noise_type' column in resolved data")
        return pd.DataFrame()

    # Determine which metrics are actually present
    all_metrics: List[str] = []
    for m in primary_metrics:
        if m in df_resolved.columns:
            all_metrics.append(m)

    if not all_metrics:
        print("  [WARNING] No valid metrics found in resolved data")
        print(f"  [INFO] Available columns: {list(df_resolved.columns)}")
        return pd.DataFrame()

    print(f"  [INFO] Primary metrics: {all_metrics}")

    # Ensure we have baseline
    if not (df_resolved["ablation"] == "baseline").any():
        print("  [WARNING] No baseline data found in resolved dataset")
        return pd.DataFrame()

    # Get unique noise types
    noise_types = sorted(df_resolved["noise_type"].dropna().unique())
    print(f"  [INFO] Noise types: {noise_types}")

    test_results: List[Dict] = []

    # Unique ablations excluding baseline
    ablation_keys = [k for k in df_resolved["ablation"].unique() if k != "baseline"]

    for noise_type in noise_types:
        noise_df = df_resolved[df_resolved["noise_type"] == noise_type].copy()

        for ablation_key in ablation_keys:
            ablation_name = ABLATION_NAMES.get(ablation_key, ablation_key)

            # Filter to baseline + this ablation for this noise type
            comparison_mask = (noise_df["ablation"] == "baseline") | (
                noise_df["ablation"] == ablation_key
            )
            comparison_df = noise_df[comparison_mask].copy()

            if comparison_df.empty:
                continue

            # Metric-wise comparisons
            for metric in all_metrics:
                if metric not in comparison_df.columns:
                    continue

                # Pivot: index = (dataset, eval_mode, tune, subject), columns = ablation, values = metric
                group_cols = ["dataset", "eval_mode", "tune", "subject"]
                group_cols = [c for c in group_cols if c in comparison_df.columns]

                pivot_df = comparison_df.pivot_table(
                    index=group_cols,
                    columns="ablation",
                    values=metric,
                    aggfunc="first",
                ).reset_index()

                # Need both baseline and this ablation
                if "baseline" not in pivot_df.columns or ablation_key not in pivot_df.columns:
                    continue

                baseline_values = pivot_df["baseline"].values
                ablation_values = pivot_df[ablation_key].values

                # Remove NaNs
                mask = ~(np.isnan(baseline_values) | np.isnan(ablation_values))
                baseline_paired = baseline_values[mask]
                ablation_paired = ablation_values[mask]

                if len(baseline_paired) < 2:
                    continue

                # Normality check on differences
                diffs = baseline_paired - ablation_paired
                is_normal = check_normality(diffs)
                parametric = is_normal

                # Run paired test
                if parametric:
                    try:
                        statistic, p_value = ttest_rel(baseline_paired, ablation_paired)
                        test_type = "paired_ttest"
                    except Exception as e:
                        statistic, p_value = np.nan, np.nan
                        test_type = "failed"
                else:
                    from scipy.stats import wilcoxon

                    try:
                        statistic, p_value = wilcoxon(
                            baseline_paired, ablation_paired, alternative="two-sided"
                        )
                        test_type = "wilcoxon"
                    except Exception as e:
                        statistic, p_value = np.nan, np.nan
                        test_type = "failed"

                # Effect size
                cohens_dz = compute_cohens_dz(baseline_paired, ablation_paired)

                # Bootstrap CI for Cohen's dz
                try:
                    data_hash = hash((tuple(baseline_paired), tuple(ablation_paired))) % (2**31)
                    ci_low, ci_high = bootstrap_ci_cohens_dz(
                        baseline_paired,
                        ablation_paired,
                        n_reps=10000,
                        random_seed=int(data_hash),
                    )
                except Exception as e:
                    ci_low, ci_high = np.nan, np.nan

                baseline_mean = float(np.mean(baseline_paired))
                baseline_std = float(np.std(baseline_paired, ddof=1))
                ablation_mean = float(np.mean(ablation_paired))
                ablation_std = float(np.std(ablation_paired, ddof=1))
                mean_diff = baseline_mean - ablation_mean

                test_results.append(
                    {
                        "ablation": ablation_key,
                        "ablation_name": ablation_name,
                        "noise_type": noise_type,
                        "metric": metric,
                        "metric_type": "primary",
                        "baseline_mean": baseline_mean,
                        "baseline_std": baseline_std,
                        "ablation_mean": ablation_mean,
                        "ablation_std": ablation_std,
                        "mean_difference": mean_diff,
                        "test_type": test_type,
                        "parametric": parametric,
                        "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "cohens_dz": float(cohens_dz) if np.isfinite(cohens_dz) else np.nan,
                        "cohens_dz_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                        "cohens_dz_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                        "n_subjects": len(baseline_paired),
                    }
                )

    if not test_results:
        return pd.DataFrame()

    stats_df = pd.DataFrame(test_results)

    # Bonferroni correction: group by (ablation, metric) and correct across noise types
    # This treats each (ablation, metric) combination as a family
    for ablation_key in ablation_keys:
        for metric in all_metrics:
            mask = (stats_df["ablation"] == ablation_key) & (stats_df["metric"] == metric)
            num_tests = mask.sum()
            if num_tests > 0:
                stats_df.loc[mask, "p_value_corrected"] = stats_df.loc[mask, "p_value"].apply(
                    lambda p: min(p * num_tests, 1.0) if np.isfinite(p) else np.nan
                )
                stats_df.loc[mask, "significant"] = (
                    stats_df.loc[mask, "p_value_corrected"] < 0.05
                )

    print(
        f"[OK] Computed noise-type-resolved statistics for "
        f"{len(stats_df)} ablation/metric/noise_type combinations"
    )
    return stats_df


# ----------------------------------------------------------------------
# Main analysis pipeline
# ----------------------------------------------------------------------

def analyze_limited_ablations(
    unified_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main entry point to analyze limited ablation results (CrossSession, BNCI2014_001).

    Returns:
        dict of DataFrames:
            - clean_scores
            - statistical_tests (collapsed over noise types)
            - statistical_tests_resolved (separated by noise type)
            - subject_level_points
            - subject_level_collapsed
            - subject_level_resolved
            - combined_results
    """
    print("=" * 80)
    print("LIMITED ABLATION STUDIES ANALYSIS (CrossSession, BNCI2014_001)")
    print("=" * 80)

    # Resolve output directory
    if output_dir is None:
        output_dir = _project_root / "analysis" / "limited_ablation_results"
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Results will be saved to: {output_dir}")

    # Load filtered unified data
    df = load_limited_ablation_results(unified_file)

    print(f"\n[INFO] Available columns in filtered DataFrame: {list(df.columns)}")
    print(f"[INFO] DataFrame shape: {df.shape}")

    # Clean scores (secondary, descriptive)
    clean_scores_df = compute_clean_scores_summary(df)

    # Robustness analysis (subject-level)
    config = AnalysisConfig(normalize_aupc=True, rd_summary="mean")
    df_points, df_collapsed, df_resolved = prepare_subject_level_data(df, config)

    # Compute subject-level statistics
    # Primary: RD/AUPC, Secondary: clean ROC-AUC if we can attach it
    # Attach clean_roc_auc to collapsed data (per ablation/subject)
    if "clean_roc_auc" in df.columns and not df_collapsed.empty:
        clean_map = (
            df.groupby(["ablation", "subject"])["clean_roc_auc"]
            .mean()
            .reset_index()
        )
        df_collapsed = df_collapsed.merge(
            clean_map, on=["ablation", "subject"], how="left"
        )

    stats_df = compute_subject_level_statistics(
        df_collapsed,
        primary_metrics=["aupc_collapsed", "rd_collapsed", "rd_worst_collapsed"],
        secondary_metrics=["clean_roc_auc"],
    )

    # Compute noise-type-resolved statistics
    # Check what metrics are available in resolved data
    resolved_primary_metrics = []
    for metric in ["aupc_roc_auc", "rd_mean", "rd_worst"]:
        if metric in df_resolved.columns:
            resolved_primary_metrics.append(metric)

    stats_resolved_df = compute_noise_type_resolved_statistics(
        df_resolved,
        primary_metrics=resolved_primary_metrics,
    )

    # Save outputs
    print("\n[STEP 4] Saving results...")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    saved_files: Dict[str, str] = {}

    if not clean_scores_df.empty:
        path = output_dir / f"clean_scores_{timestamp}.csv"
        clean_scores_df.to_csv(path, index=False)
        saved_files["clean_scores"] = str(path)
        print(f"  [OK] Saved clean scores: {path}")

    if not stats_df.empty:
        path = output_dir / f"statistical_tests_{timestamp}.csv"
        stats_df.to_csv(path, index=False)
        saved_files["statistical_tests"] = str(path)
        print(f"  [OK] Saved statistical tests (collapsed): {path}")

    if not stats_resolved_df.empty:
        path = output_dir / f"statistical_tests_resolved_{timestamp}.csv"
        stats_resolved_df.to_csv(path, index=False)
        saved_files["statistical_tests_resolved"] = str(path)
        print(f"  [OK] Saved statistical tests (noise-type-resolved): {path}")

    if not df_points.empty:
        path = output_dir / f"subject_level_points_{timestamp}.csv"
        df_points.to_csv(path, index=False)
        saved_files["subject_level_points"] = str(path)
        print(f"  [OK] Saved subject-level curve points: {path}")

    if not df_collapsed.empty:
        path = output_dir / f"subject_level_collapsed_{timestamp}.csv"
        df_collapsed.to_csv(path, index=False)
        saved_files["subject_level_collapsed"] = str(path)
        print(f"  [OK] Saved subject-level collapsed metrics: {path}")

    if not df_resolved.empty:
        path = output_dir / f"subject_level_resolved_{timestamp}.csv"
        df_resolved.to_csv(path, index=False)
        saved_files["subject_level_resolved"] = str(path)
        print(f"  [OK] Saved subject-level resolved metrics: {path}")

    # Save combined raw filtered df
    combined_path = output_dir / f"combined_results_{timestamp}.csv"
    df.to_csv(combined_path, index=False)
    saved_files["combined_results"] = str(combined_path)
    print(f"  [OK] Saved combined filtered results: {combined_path}")

    # Enhanced summary report with effect size emphasis
    summary_path = output_dir / f"summary_report_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LIMITED ABLATION STUDIES ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        # Effect size interpretation guide
        f.write("-" * 80 + "\n")
        f.write("EFFECT SIZE INTERPRETATION (Cohen's dz)\n")
        f.write("-" * 80 + "\n")
        f.write("Cohen's dz interpretation guidelines:\n")
        f.write("  |dz| < 0.2  : Negligible effect\n")
        f.write("  0.2 ≤ |dz| < 0.5  : Small effect\n")
        f.write("  0.5 ≤ |dz| < 0.8  : Medium effect\n")
        f.write("  |dz| ≥ 0.8  : Large effect\n")
        f.write("\n")
        f.write("Note: In small-sample settings (n=9), effect sizes and their CIs are\n")
        f.write("more informative than p-values alone. Wide CIs indicate uncertainty.\n")
        f.write("\n\n")

        if not stats_df.empty:
            primary = stats_df[stats_df["metric_type"] == "primary"].copy()
            if not primary.empty:
                f.write("-" * 80 + "\n")
                f.write("PRIMARY OUTCOMES: ROBUSTNESS METRICS (AUPC / mean RD / worst-case RD)\n")
                f.write("Subject-level paired comparisons (n = subjects, collapsed over noise types)\n")
                f.write("-" * 80 + "\n")
                # Create a more readable format emphasizing effect sizes
                for _, row in primary.iterrows():
                    f.write(f"\n{row['ablation_name']} - {row['metric']}:\n")
                    f.write(f"  Baseline: {row['baseline_mean']:.4f} ± {row['baseline_std']:.4f}\n")
                    f.write(f"  Ablation: {row['ablation_mean']:.4f} ± {row['ablation_std']:.4f}\n")
                    f.write(f"  Difference: {row['mean_difference']:.4f} (baseline - ablation)\n")
                    f.write(f"  Test: {row['test_type']}, p = {row['p_value']:.4f}")
                    if row.get('p_value_corrected', np.nan) != row['p_value']:
                        f.write(f" (corrected: {row['p_value_corrected']:.4f})")
                    f.write(f" {'*' if row.get('significant', False) else ''}\n")
                    f.write(f"  Effect size (Cohen's dz): {row['cohens_dz']:.3f} ")
                    f.write(f"[{row['cohens_dz_ci_low']:.3f}, {row['cohens_dz_ci_high']:.3f}]\n")
                    f.write(f"  n = {row['n_subjects']}\n")
                f.write("\n")

        if not stats_df.empty:
            secondary = stats_df[stats_df["metric_type"] == "secondary"].copy()
            if not secondary.empty:
                f.write("-" * 80 + "\n")
                f.write("SECONDARY OUTCOMES: CLEAN ROC-AUC\n")
                f.write("Subject-level paired comparisons\n")
                f.write("-" * 80 + "\n")
                for _, row in secondary.iterrows():
                    f.write(f"\n{row['ablation_name']} - {row['metric']}:\n")
                    f.write(f"  Baseline: {row['baseline_mean']:.4f} ± {row['baseline_std']:.4f}\n")
                    f.write(f"  Ablation: {row['ablation_mean']:.4f} ± {row['ablation_std']:.4f}\n")
                    f.write(f"  Difference: {row['mean_difference']:.4f} (baseline - ablation)\n")
                    f.write(f"  Test: {row['test_type']}, p = {row['p_value']:.4f}")
                    if row.get('p_value_corrected', np.nan) != row['p_value']:
                        f.write(f" (corrected: {row['p_value_corrected']:.4f})")
                    f.write(f" {'*' if row.get('significant', False) else ''}\n")
                    f.write(f"  Effect size (Cohen's dz): {row['cohens_dz']:.3f} ")
                    f.write(f"[{row['cohens_dz_ci_low']:.3f}, {row['cohens_dz_ci_high']:.3f}]\n")
                    f.write(f"  n = {row['n_subjects']}\n")
                f.write("\n")

        if not stats_resolved_df.empty:
            f.write("-" * 80 + "\n")
            f.write("NOISE-TYPE-RESOLVED COMPARISONS\n")
            f.write("Subject-level paired comparisons, separated by noise type\n")
            f.write("(Detects targeted robustness trade-offs per perturbation type)\n")
            f.write("-" * 80 + "\n")
            # Group by ablation and metric for readability
            for ablation_key in sorted(stats_resolved_df["ablation"].unique()):
                ablation_name = ABLATION_NAMES.get(ablation_key, ablation_key)
                ablation_df = stats_resolved_df[stats_resolved_df["ablation"] == ablation_key]
                
                for metric in sorted(ablation_df["metric"].unique()):
                    metric_df = ablation_df[ablation_df["metric"] == metric]
                    f.write(f"\n{ablation_name} - {metric}:\n")
                    for _, row in metric_df.iterrows():
                        f.write(f"  {row['noise_type']:12s}: ")
                        f.write(f"diff={row['mean_difference']:+.4f}, ")
                        f.write(f"p={row['p_value']:.4f}")
                        if row.get('p_value_corrected', np.nan) != row['p_value']:
                            f.write(f" (corr: {row['p_value_corrected']:.4f})")
                        f.write(f", dz={row['cohens_dz']:.3f} ")
                        f.write(f"[{row['cohens_dz_ci_low']:.3f}, {row['cohens_dz_ci_high']:.3f}]")
                        f.write(f" {'*' if row.get('significant', False) else ''}\n")
                    f.write("\n")

        if not clean_scores_df.empty:
            f.write("-" * 80 + "\n")
            f.write("CLEAN SCORES SUMMARY (Descriptive Statistics)\n")
            f.write("-" * 80 + "\n")
            f.write(clean_scores_df.to_string(index=False))
            f.write("\n\n")

        f.write("-" * 80 + "\n")
        f.write("SAVED FILES\n")
        f.write("-" * 80 + "\n")
        for key, path in saved_files.items():
            f.write(f"{key:30s}: {os.path.basename(path)}\n")
        f.write("\n")

    saved_files["summary"] = str(summary_path)
    print(f"  [OK] Saved summary report: {summary_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")

    return {
        "clean_scores": clean_scores_df,
        "statistical_tests": stats_df,
        "statistical_tests_resolved": stats_resolved_df,
        "subject_level_points": df_points,
        "subject_level_collapsed": df_collapsed,
        "subject_level_resolved": df_resolved,
        "combined_results": df,
    }


def main() -> int:
    try:
        analyze_limited_ablations()
        print("\n[OK] Limited ablation analysis completed successfully!")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Limited ablation analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

