"""
robustness_metrics.py

Compute robustness metrics for an EEG perturbation benchmark:
- AUPC: Area Under Perturbation Curve (trapezoidal rule over p in [0,1])
- RD(p): Relative Degradation curve at discrete perturbation levels
- CSV_p: Cross-Subject Variance under perturbation at each perturbation level

Assumptions:
- You already load/aggregate CSVs into a pandas DataFrame `df`.
- `df` contains (at least) columns like:
    dataset, tune, eval_mode, model, seed, noise_type (or "noise type"), intensity, and a metric column (default: roc_auc).
- If you have per-subject scores, include subject column: subject (or subject_id / participant / etc.).

Design choices:
- We canonicalize column names (lowercase, spaces->underscores).
- We compute a normalized perturbation coordinate p ∈ [0,1].
  If p already exists as a column, we use it.
  Otherwise, we normalize `intensity` per (dataset, noise_type) to [0,1] by default.
  Optionally, you can normalize by a *baseline-model saturation point* per (dataset, noise_type).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Add project root to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ----------------------------
# Configuration
# ----------------------------

@dataclass(frozen=True)
class MetricConfig:
    metric_col: str = "roc_auc"          # primary performance column
    intensity_col: str = "intensity"     # raw perturbation strength
    p_col: str = "p"                     # normalized perturbation coordinate
    noise_col: str = "noise_type"        # perturbation type
    dataset_col: str = "dataset"
    model_col: str = "model"
    seed_col: str = "seed"
    tune_col: str = "tune"
    eval_mode_col: str = "eval_mode"

    # Subject identifier (optional, required for CSV_p)
    subject_col_candidates: Tuple[str, ...] = ("subject", "subject_id", "participant", "subj", "session_subject")

    # Saturation-point logic (optional)
    chance_level: float = 0.5            # ROC-AUC chance
    saturation_eps: float = 0.02         # within eps of chance counts as "saturated"
    saturation_min_p: float = 0.0        # allow saturation at 0 if pathological
    saturation_max_p: float = 1e9        # safety bound if using raw intensity units

    # CI summary (normal approximation)
    ci_z: float = 1.96                   # ~95% CI


# ----------------------------
# Column handling
# ----------------------------

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy with canonical column names:
    - lowercase
    - spaces/hyphens -> underscores
    """
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
    return out


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


def find_subject_col(df: pd.DataFrame, cfg: MetricConfig) -> Optional[str]:
    for c in cfg.subject_col_candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------
# p (normalized intensity) construction
# ----------------------------

def add_normalized_p(
    df: pd.DataFrame,
    cfg: MetricConfig,
    *,
    normalize_within: Sequence[str],
    saturation_intensity_by_group: Optional[pd.DataFrame] = None,
    clip: bool = True,
) -> pd.DataFrame:
    """
    Ensure a normalized perturbation coordinate p ∈ [0,1] exists.

    If cfg.p_col exists, returns df unchanged.
    Otherwise:
    - If saturation_intensity_by_group is provided: normalize intensity by group-specific saturation intensity.
      Expected columns: normalize_within + ["sat_intensity"]
    - Else: normalize intensity by group max intensity (within normalize_within).
    """
    if cfg.p_col in df.columns:
        return df

    if cfg.intensity_col not in df.columns:
        raise KeyError(
            f"Need either '{cfg.p_col}' or '{cfg.intensity_col}' in df. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    if saturation_intensity_by_group is not None:
        sat = saturation_intensity_by_group.copy()
        if "sat_intensity" not in sat.columns:
            raise KeyError("saturation_intensity_by_group must include a 'sat_intensity' column.")
        # Merge saturation intensities into main df
        out = out.merge(sat[list(normalize_within) + ["sat_intensity"]], on=list(normalize_within), how="left")
        denom = out["sat_intensity"].astype(float)
        # Fallback if sat_intensity missing: use max intensity for that group
        fallback = out.groupby(list(normalize_within))[cfg.intensity_col].transform("max").astype(float)
        denom = denom.fillna(fallback)
        denom = denom.replace(0.0, np.nan)
        out[cfg.p_col] = (out[cfg.intensity_col].astype(float) / denom).astype(float)
        out.drop(columns=["sat_intensity"], inplace=True, errors="ignore")
    else:
        denom = out.groupby(list(normalize_within))[cfg.intensity_col].transform("max").astype(float)
        denom = denom.replace(0.0, np.nan)
        out[cfg.p_col] = (out[cfg.intensity_col].astype(float) / denom).astype(float)

    if clip:
        out[cfg.p_col] = out[cfg.p_col].clip(lower=0.0, upper=1.0)

    return out


# ----------------------------
# Saturation point estimation (optional)
# ----------------------------

def estimate_saturation_intensity_by_group(
    df: pd.DataFrame,
    cfg: MetricConfig,
    *,
    baseline_model: str,
    group_cols: Sequence[str],
    # Choose how to aggregate baseline model performance at each intensity before testing saturation:
    agg_over: Sequence[str] = ("seed",),
) -> pd.DataFrame:
    """
    Estimate a saturation intensity per group (e.g., per (dataset, noise_type)),
    using a specified baseline model.

    Saturation rule:
      Find the smallest intensity where aggregated performance <= chance_level + saturation_eps.
      If none found, use the max intensity for that group.

    Returns a DataFrame with columns: group_cols + ["sat_intensity"]
    """
    required = [cfg.model_col, cfg.metric_col, cfg.intensity_col, *group_cols]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' for saturation estimation.")

    base = df[df[cfg.model_col] == baseline_model].copy()
    if base.empty:
        raise ValueError(f"No rows found for baseline_model='{baseline_model}'")

    # Aggregate baseline model across agg_over at each (group, intensity).
    gb_cols = list(group_cols) + [cfg.intensity_col]
    if agg_over:
        gb_cols = gb_cols + [c for c in agg_over if c in base.columns]

    # First reduce to per-(group,intensity) by averaging over agg_over (if present)
    # then average across any remaining agg dims (e.g., seeds) to get a single curve per group.
    curve = (
        base.groupby(list(group_cols) + [cfg.intensity_col], as_index=False)[cfg.metric_col]
        .mean()
        .sort_values(list(group_cols) + [cfg.intensity_col])
    )

    def _sat_for_group(g: pd.DataFrame) -> float:
        g = g.sort_values(cfg.intensity_col)
        intens = g[cfg.intensity_col].astype(float).to_numpy()
        vals = g[cfg.metric_col].astype(float).to_numpy()
        thresh = cfg.chance_level + cfg.saturation_eps

        # smallest intensity with performance near chance
        idx = np.where(vals <= thresh)[0]
        if idx.size > 0:
            sat = float(intens[idx[0]])
        else:
            sat = float(np.nanmax(intens))

        sat = max(sat, cfg.saturation_min_p)
        sat = min(sat, cfg.saturation_max_p)
        return sat

    sat_rows = []
    for keys, g in curve.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sat_intensity = _sat_for_group(g)
        sat_rows.append((*keys, sat_intensity))

    sat_df = pd.DataFrame(sat_rows, columns=list(group_cols) + ["sat_intensity"])
    return sat_df


# ----------------------------
# AUPC
# ----------------------------

def compute_aupc(
    df: pd.DataFrame,
    cfg: MetricConfig,
    *,
    group_cols: Sequence[str],
    require_monotonic_p: bool = False,
) -> pd.DataFrame:
    """
    Compute AUPC per group using trapezoidal rule over p.

    Returns a DataFrame with columns: group_cols + ["aupc"]
    """
    needed = [cfg.p_col, cfg.metric_col, *group_cols]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' for AUPC.")

    def _aupc_for_group(g: pd.DataFrame) -> float:
        gg = g[[cfg.p_col, cfg.metric_col]].dropna().copy()
        if gg.empty or gg.shape[0] < 2:
            return float("nan")

        gg = gg.sort_values(cfg.p_col)
        p = gg[cfg.p_col].astype(float).to_numpy()
        y = gg[cfg.metric_col].astype(float).to_numpy()

        # If duplicate p values exist, average them first
        if len(np.unique(p)) != len(p):
            tmp = gg.groupby(cfg.p_col, as_index=False)[cfg.metric_col].mean().sort_values(cfg.p_col)
            p = tmp[cfg.p_col].astype(float).to_numpy()
            y = tmp[cfg.metric_col].astype(float).to_numpy()

        if require_monotonic_p and np.any(np.diff(p) <= 0):
            return float("nan")

        # trapezoidal integral (using trapezoid instead of deprecated trapz)
        return float(np.trapezoid(y=y, x=p))

    out = (
        df.groupby(list(group_cols), as_index=False)
        .apply(lambda g: pd.Series({"aupc": _aupc_for_group(g)}), include_groups=False)
        .reset_index(drop=True)
    )
    return out


# ----------------------------
# Relative Degradation (RD)
# ----------------------------

def compute_rd_curve(
    df: pd.DataFrame,
    cfg: MetricConfig,
    *,
    group_cols: Sequence[str],
    baseline_p: float = 0.0,
    baseline_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Compute RD(p) per group and p level:
      RD(p) = (f(0) - f(p)) / f(0)

    Baseline f(0) is taken from the row(s) with p ≈ baseline_p (within tolerance).
    If no exact baseline exists, uses the smallest p in that group as baseline.

    Returns: group_cols + [p, metric, f0, rd]
    """
    needed = [cfg.p_col, cfg.metric_col, *group_cols]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' for RD curve.")

    def _curve_for_group(g: pd.DataFrame) -> pd.DataFrame:
        # Extract only the columns we need for computation
        gg = g[[cfg.p_col, cfg.metric_col]].dropna().copy()
        if gg.empty:
            # Return empty DataFrame with expected columns (group columns will be added externally)
            return pd.DataFrame(columns=[cfg.p_col, cfg.metric_col, "f0", "rd"])

        gg = gg.sort_values(cfg.p_col)

        # Determine baseline
        p = gg[cfg.p_col].astype(float).to_numpy()
        y = gg[cfg.metric_col].astype(float).to_numpy()

        baseline_mask = np.isclose(p, baseline_p, atol=baseline_tolerance, rtol=0.0)
        if baseline_mask.any():
            f0 = float(np.mean(y[baseline_mask]))
        else:
            # fallback: use minimal p
            min_p = float(np.min(p))
            f0 = float(np.mean(y[p == min_p]))

        if not np.isfinite(f0) or f0 == 0.0:
            rd = np.full_like(y, np.nan, dtype=float)
        else:
            rd = (f0 - y) / f0
            # clamp into [0,1] per definition (numerical safety)
            rd = np.clip(rd, 0.0, 1.0)

        out = gg.copy()
        out["f0"] = f0
        out["rd"] = rd
        
        return out

    # Group by the columns and apply the curve function
    # Note: We need to preserve group columns in the output
    grouped = df.groupby(list(group_cols), as_index=True)
    
    curve_list = []
    for group_keys, group_df in grouped:
        # Extract group key values
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        
        # Create a dictionary of group column values
        group_values = {}
        for i, col in enumerate(group_cols):
            if i < len(group_keys):
                group_values[col] = group_keys[i]
        
        # Apply the curve function to this group
        curve_df = _curve_for_group(group_df.reset_index(drop=True))
        
        # Ensure group columns are in the result
        for col, val in group_values.items():
            if col not in curve_df.columns:
                curve_df[col] = val
        
        curve_list.append(curve_df)
    
    # Combine all curves
    if curve_list:
        curves = pd.concat(curve_list, ignore_index=True)
    else:
        curves = pd.DataFrame(columns=list(group_cols) + [cfg.p_col, cfg.metric_col, "f0", "rd"])
    return curves


# ----------------------------
# Cross-Subject Variance under perturbation (CSV_p)
# ----------------------------

def compute_csv_p_curve(
    df: pd.DataFrame,
    cfg: MetricConfig,
    *,
    group_cols: Sequence[str],
    subject_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute CSV_p per group and p:
      CSV_p = (1/S) * sum_s (f_s(p) - mean_s f_s(p))^2
    i.e., population variance across subjects.

    Requires subject column.

    Returns: group_cols + [p, csv_p]
    """
    if subject_col is None:
        subject_col = find_subject_col(df, cfg)
    if subject_col is None:
        raise KeyError(
            "CSV_p requires a subject column, but none was found. "
            f"Tried: {cfg.subject_col_candidates}"
        )

    needed = [cfg.p_col, cfg.metric_col, subject_col, *group_cols]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' for CSV_p.")

    # First, if multiple rows per (group,p,subject) exist (e.g., repeats), average them.
    subj_avg = (
        df.groupby(list(group_cols) + [cfg.p_col, subject_col], as_index=False)[cfg.metric_col]
        .mean()
    )

    # Then compute population variance across subjects at each (group,p)
    def _pop_var(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return float("nan")
        mu = float(np.mean(x))
        return float(np.mean((x - mu) ** 2))

    csv = (
        subj_avg.groupby(list(group_cols) + [cfg.p_col], as_index=False)
        .agg(csv_p=(cfg.metric_col, lambda s: _pop_var(np.asarray(s, dtype=float))))
    )
    return csv


# ----------------------------
# Summary helpers (mean / CI)
# ----------------------------

def summarize_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str],
    cfg: MetricConfig,
) -> pd.DataFrame:
    """
    Summarize a value with mean, std, sem, and normal-approx 95% CI.
    """
    group_cols_list = list(group_cols)
    
    # Use a more explicit approach to avoid column naming issues
    # Group by the columns and compute statistics separately
    grouped = df.groupby(group_cols_list)[value_col]
    
    # Compute statistics
    out = pd.DataFrame({
        "n": grouped.count(),
        "mean": grouped.mean(),
        "std": grouped.std()
    })
    
    # Reset index to get group columns as regular columns
    out = out.reset_index()
    
    # Compute additional statistics
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    out["ci_low"] = out["mean"] - cfg.ci_z * out["sem"]
    out["ci_high"] = out["mean"] + cfg.ci_z * out["sem"]
    
    return out


# ----------------------------
# Orchestration: compute everything needed for Results
# ----------------------------

@dataclass(frozen=True)
class ResultsSpec:
    # grouping for model-vs-dataset comparisons
    base_group_cols: Tuple[str, ...] = ("dataset", "tune", "eval_mode", "model", "noise_type")

    # optional extra grouping dimensions
    extra_group_cols: Tuple[str, ...] = ()

    # If you want per-seed AUPC then summarize over seeds, include seed in the per-instance group
    per_instance_cols: Tuple[str, ...] = ("seed",)

    # Saturation normalization options
    use_saturation_normalization: bool = False
    baseline_model_for_saturation: str = "eegnet"  # change to your baseline name


def compute_results_metrics(
    df_in: pd.DataFrame,
    cfg: MetricConfig = MetricConfig(),
    spec: ResultsSpec = ResultsSpec(),
    core_models: Optional[Sequence[str]] = None,
    hydra: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames:
      - "aupc_raw": per-(group + per_instance) AUPC
      - "aupc_summary": mean/CI AUPC summarized over per_instance (e.g., seeds)
      - "rd_curve": RD values per row (with f0 and rd)
      - "rd_summary": mean/CI RD at each p (summarized over per_instance)
      - "csv_curve": CSV_p per (group + p + per_instance?) depending on inputs
      - "csv_summary": mean/CI CSV_p at each p (summarized over per_instance)
    
    Parameters:
    -----------
    core_models : Optional[Sequence[str]]
        If provided, filter to only these models. Default: ['CNN-NCP', 'EEGNet', 'REEGNet']
    hydra : bool
        If True, include 'branched_wiredcfc_arch4' along with core models
    """
    df = canonicalize_columns(df_in)
    
    # Filter to core models if specified
    if core_models is None:
        if hydra:
            core_models = ['CNN-NCP', 'EEGNet', 'REEGNet', 'branched_wiredcfc_arch4']
        else:
            core_models = ['CNN-NCP', 'EEGNet', 'REEGNet']
    
    if cfg.model_col in df.columns:
        initial_count = len(df)
        # Canonicalize model names for filtering (lowercase, hyphens to underscores)
        # This handles both the filter list and potential variations in the dataframe
        canonicalize_model_name = lambda x: str(x).strip().lower().replace(" ", "_").replace("-", "_")
        core_models_canonical = [canonicalize_model_name(m) for m in core_models]
        
        # Also canonicalize the model column values for comparison
        df_model_values = df[cfg.model_col].apply(canonicalize_model_name)
        df = df[df_model_values.isin(core_models_canonical)].copy()
        filtered_count = len(df)
        excluded = initial_count - filtered_count
        if excluded > 0:
            print(f"[INFO] Filtered to {'hydra' if hydra else 'core'} models {core_models}: removed {excluded} rows, kept {filtered_count} rows")
    
    # Auto-detect metric column if default doesn't exist
    if cfg.metric_col not in df.columns:
        metric_candidates = ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']
        for candidate in metric_candidates:
            if candidate in df.columns:
                print(f"[INFO] Auto-detected metric column: {candidate} (was looking for {cfg.metric_col})")
                # Create a new config with the detected metric column
                cfg = replace(cfg, metric_col=candidate)
                break
        else:
            raise KeyError(
                f"Missing metric column '{cfg.metric_col}' and no standard metric columns found. "
                f"Available columns: {list(df.columns)}"
            )

    # Make sure canonical column names exist for common variants
    # (e.g., "noise type" -> noise_type handled by canonicalize_columns)
    for required in [cfg.dataset_col, cfg.model_col, cfg.metric_col, cfg.noise_col]:
        if required not in df.columns:
            raise KeyError(f"Missing required column '{required}' after canonicalization.")
    
    # Ensure 'tune' column exists (boolean)
    if cfg.tune_col not in df.columns:
        # Try to infer from 'tuned' column or 'mode' column
        if 'tuned' in df.columns:
            df[cfg.tune_col] = df['tuned'].astype(bool)
        elif 'mode' in df.columns:
            df[cfg.tune_col] = df['mode'].astype(str).str.contains('_tune', na=False)
        else:
            # Default to False if cannot infer
            print(f"[WARNING] Could not infer '{cfg.tune_col}' column, defaulting to False")
            df[cfg.tune_col] = False

    # Build saturation normalization table (optional)
    saturation_tbl = None
    normalize_within = (cfg.dataset_col, cfg.noise_col)
    if spec.use_saturation_normalization:
        saturation_tbl = estimate_saturation_intensity_by_group(
            df,
            cfg,
            baseline_model=spec.baseline_model_for_saturation,
            group_cols=list(normalize_within),
            agg_over=(cfg.seed_col,) if cfg.seed_col in df.columns else (),
        )

    # Ensure p exists
    df = add_normalized_p(
        df,
        cfg,
        normalize_within=list(normalize_within),
        saturation_intensity_by_group=saturation_tbl,
        clip=True,
    )

    # Group columns
    base_group = list(spec.base_group_cols) + list(spec.extra_group_cols)
    per_instance = [c for c in spec.per_instance_cols if c in df.columns]

    # ---------------- AUPC ----------------
    # Compute per-instance AUPC (e.g., per seed) so you can average/CI in Results.
    aupc_group_cols = base_group + per_instance
    aupc_raw = compute_aupc(df, cfg, group_cols=aupc_group_cols)

    # Summarize over per_instance
    aupc_summary_group = base_group
    aupc_summary = summarize_mean_ci(aupc_raw, "aupc", aupc_summary_group, cfg)

    # ---------------- RD curve ----------------
    rd_group_cols = base_group + per_instance
    rd_curve = compute_rd_curve(df, cfg, group_cols=rd_group_cols, baseline_p=0.0)

    # Summarize RD at each p (and base_group)
    rd_summary_group = base_group + [cfg.p_col]
    rd_summary = summarize_mean_ci(rd_curve, "rd", rd_summary_group, cfg)

    # ---------------- CSV_p curve ----------------
    # CSV is defined across subjects; we can still compute it per seed (then summarize across seeds).
    subject_col = find_subject_col(df, cfg)
    csv_group_cols = base_group + per_instance
    csv_curve = compute_csv_p_curve(df, cfg, group_cols=csv_group_cols, subject_col=subject_col)

    csv_summary_group = base_group + [cfg.p_col]
    csv_summary = summarize_mean_ci(csv_curve, "csv_p", csv_summary_group, cfg)

    return {
        "aupc_raw": aupc_raw,
        "aupc_summary": aupc_summary,
        "rd_curve": rd_curve,
        "rd_summary": rd_summary,
        "csv_curve": csv_curve,
        "csv_summary": csv_summary,
    }


# ----------------------------
# Optional: convenience selectors for plotting/tables
# ----------------------------

def make_results_table_aupc(
    aupc_summary: pd.DataFrame,
    *,
    index_cols: Sequence[str] = ("model",),
    column_cols: Sequence[str] = ("dataset",),
    value: str = "mean",
    fmt: str = "{:.3f}",
) -> pd.DataFrame:
    """
    Pivot AUPC summary into a wide table (e.g., model x dataset) using the chosen value (mean, ci_low, ci_high).
    """
    tbl = aupc_summary.pivot_table(index=list(index_cols), columns=list(column_cols), values=value, aggfunc="first")
    # Format to strings (optional)
    return tbl.applymap(lambda x: "" if pd.isna(x) else fmt.format(float(x)))


# ----------------------------
# Results saving and reporting
# ----------------------------

def save_robustness_results(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "./analysis/robustness_results",
    prefix: str = "robustness_metrics",
    save_csv: bool = True,
    save_excel: bool = True,
    save_summary: bool = True,
) -> Dict[str, str]:
    """
    Save all robustness metrics results to well-formatted output files.
    
    Parameters:
    -----------
    results : dict
        Dictionary of DataFrames from compute_results_metrics()
    output_dir : str
        Directory to save output files
    prefix : str
        Prefix for output filenames
    save_csv : bool
        Whether to save individual CSV files for each result
    save_excel : bool
        Whether to save all results to a single Excel file
    save_summary : bool
        Whether to save a text summary report
        
    Returns:
    --------
    dict
        Dictionary mapping result keys to saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual CSV files
    if save_csv:
        print(f"\n[INFO] Saving individual CSV files to {output_dir}...")
        for key, df in results.items():
            if df is not None and not df.empty:
                # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
                df = replace_hydra_model_name(df, model_col='model')
                filename = f"{prefix}_{key}_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                saved_files[key] = filepath
                print(f"  [OK] Saved {key}: {filepath} ({len(df)} rows)")
    
    # Save to Excel (all sheets in one file)
    if save_excel:
        excel_path = os.path.join(output_dir, f"{prefix}_all_results_{timestamp}.xlsx")
        try:
            print(f"\n[INFO] Saving all results to Excel: {excel_path}...")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for key, df in results.items():
                    if df is not None and not df.empty:
                        # Replace branched_wiredcfc_arch4 with HYDRA in model names before saving
                        df = replace_hydra_model_name(df, model_col='model')
                        # Excel sheet names must be <= 31 characters
                        sheet_name = key[:31] if len(key) > 31 else key
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            saved_files['excel'] = excel_path
            print(f"  [OK] Saved Excel file: {excel_path}")
        except ImportError:
            print(f"  [WARNING] openpyxl not available, skipping Excel export")
            print(f"  [INFO] Install with: pip install openpyxl")
        except Exception as e:
            print(f"  [WARNING] Failed to save Excel file: {e}")
    
    # Save text summary report
    if save_summary:
        summary_path = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.txt")
        print(f"\n[INFO] Generating summary report: {summary_path}...")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ROBUSTNESS METRICS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write("\n")
            
            # Overview
            f.write("-" * 80 + "\n")
            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            for key, df in results.items():
                if df is not None and not df.empty:
                    f.write(f"{key:20s}: {len(df):6d} rows, {len(df.columns):2d} columns\n")
                else:
                    f.write(f"{key:20s}: No data\n")
            f.write("\n")
            
            # AUPC Summary
            if 'aupc_summary' in results and results['aupc_summary'] is not None:
                df_aupc = replace_hydra_model_name(results['aupc_summary'], model_col='model')
                if not df_aupc.empty:
                    f.write("-" * 80 + "\n")
                    f.write("AUPC (Area Under Perturbation Curve) SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    f.write("\nStatistics by grouping:\n")
                    f.write(df_aupc.to_string(index=False))
                    f.write("\n\n")
                    
                    # Best/worst models
                    if 'mean' in df_aupc.columns:
                        f.write("Top 10 Best AUPC (by mean):\n")
                        top_aupc = df_aupc.nlargest(10, 'mean')[['model', 'dataset', 'noise_type', 'mean', 'ci_low', 'ci_high']]
                        f.write(top_aupc.to_string(index=False))
                        f.write("\n\n")
            
            # RD Summary (sample)
            if 'rd_summary' in results and results['rd_summary'] is not None:
                df_rd = replace_hydra_model_name(results['rd_summary'], model_col='model')
                if not df_rd.empty:
                    f.write("-" * 80 + "\n")
                    f.write("RD (Relative Degradation) SUMMARY (Sample)\n")
                    f.write("-" * 80 + "\n")
                    f.write("Showing first 20 rows:\n")
                    f.write(df_rd.head(20).to_string(index=False))
                    f.write("\n\n")
            
            # CSV_p Summary (sample)
            if 'csv_summary' in results and results['csv_summary'] is not None:
                df_csv = replace_hydra_model_name(results['csv_summary'], model_col='model')
                if not df_csv.empty:
                    f.write("-" * 80 + "\n")
                    f.write("CSV_p (Cross-Subject Variance) SUMMARY (Sample)\n")
                    f.write("-" * 80 + "\n")
                    f.write("Showing first 20 rows:\n")
                    f.write(df_csv.head(20).to_string(index=False))
                    f.write("\n\n")
            
            # File locations
            f.write("-" * 80 + "\n")
            f.write("SAVED FILES\n")
            f.write("-" * 80 + "\n")
            for key, filepath in saved_files.items():
                if key != 'excel':  # Don't duplicate excel in this list
                    f.write(f"{key:20s}: {filepath}\n")
            if 'excel' in saved_files:
                f.write(f"{'excel':20s}: {saved_files['excel']}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        saved_files['summary'] = summary_path
        print(f"  [OK] Saved summary report: {summary_path}")
    
    print(f"\n[OK] All results saved to: {output_dir}")
    return saved_files


# ----------------------------
# Data loading and aggregation
# ----------------------------

def load_results_dataframe(
    results_file: Optional[str] = None,
    aggregate_from_directories: bool = True,
    results_dirs: Optional[List[str]] = None,
    core_models: Optional[Sequence[str]] = None,
    hydra: bool = False,
) -> pd.DataFrame:
    """
    Load and aggregate results using the same logic as analyze_results.py and experiment_automation.py.
    
    This function:
    1. First tries to load from a pre-aggregated CSV file if provided
    2. Otherwise, uses collect_all_results_unified() to aggregate from directories
    3. Normalizes column names and maps to expected format
    4. Handles metric column detection (roc_auc, score, corrupted_score, etc.)
    5. Optionally filters to core models only
    
    Parameters:
    -----------
    results_file : str, optional
        Path to pre-aggregated results CSV file. If provided and exists, loads from this file.
        Otherwise, aggregates from directories.
    aggregate_from_directories : bool, default=True
        If True and results_file is not provided or doesn't exist, aggregate from directories.
    results_dirs : list of str, optional
        Alternative result directories to check. If None, uses default locations.
    core_models : Optional[Sequence[str]]
        If provided, filter to only these models. Default: ['CNN-NCP', 'EEGNet', 'REEGNet']
        
    Returns:
    --------
    pd.DataFrame
        Aggregated results with canonicalized column names.
    """
    # Try to load from pre-aggregated file first
    if results_file and os.path.exists(results_file):
        print(f"[INFO] Loading pre-aggregated results from: {results_file}")
        df = pd.read_csv(results_file)
        print(f"[OK] Loaded {len(df)} rows from {results_file}")
    else:
        # Check for unified results file
        unified_file = os.path.join(_project_root, "evaluation", "results", "unified_all_results.csv")
        if os.path.exists(unified_file):
            print(f"[INFO] Loading unified results from: {unified_file}")
            df = pd.read_csv(unified_file)
            print(f"[OK] Loaded {len(df)} rows from unified results file")
        elif aggregate_from_directories:
            # Use collect_all_results_unified to aggregate from directories
            print("[INFO] Aggregating results from directories...")
            try:
                from evaluation.experiment_utils import collect_all_results_unified
                df = collect_all_results_unified()
                if df is None:
                    raise ValueError("No results found to aggregate")
                print(f"[OK] Aggregated {len(df)} rows from directories")
            except ImportError as e:
                raise ImportError(
                    f"Could not import collect_all_results_unified from evaluation.experiment_utils: {e}\n"
                    "Make sure you're running from the project root directory."
                )
        else:
            raise FileNotFoundError(
                f"Results file not found: {results_file}\n"
                "Set aggregate_from_directories=True to aggregate from directories, "
                "or provide a valid results_file path."
            )
    
    if df is None or df.empty:
        raise ValueError("No results loaded - DataFrame is None or empty")
    
    # Canonicalize column names (handles spaces, hyphens, case)
    df = canonicalize_columns(df)
    
    # Filter to core models if specified
    if core_models is None:
        if hydra:
            core_models = ['CNN-NCP', 'EEGNet', 'REEGNet', 'branched_wiredcfc_arch4']
        else:
            core_models = ['CNN-NCP', 'EEGNet', 'REEGNet']
    
    model_col = 'model'  # After canonicalization, should be 'model'
    if model_col in df.columns:
        initial_count = len(df)
        df = df[df[model_col].isin(core_models)].copy()
        filtered_count = len(df)
        excluded = initial_count - filtered_count
        if excluded > 0:
            print(f"[INFO] Filtered to {'hydra' if hydra else 'core'} models {core_models}: removed {excluded} rows, kept {filtered_count} rows")
    
    # Map column names to expected format
    # Handle 'tuned' -> 'tune' mapping
    if 'tuned' in df.columns and 'tune' not in df.columns:
        df['tune'] = df['tuned'].astype(bool)
    
    # Handle 'noise_level' -> 'intensity' mapping
    if 'noise_level' in df.columns and 'intensity' not in df.columns:
        df['intensity'] = df['noise_level'].astype(float)
    elif 'intensity' in df.columns:
        # Ensure intensity is numeric
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
    
    # Normalize eval_mode (remove 'Evaluation' suffix if present)
    if 'eval_mode' in df.columns:
        df['eval_mode'] = df['eval_mode'].astype(str).str.replace('Evaluation', '', regex=False)
    
    # Normalize mode column (remove '_tune' suffix for grouping)
    if 'mode' in df.columns:
        # Extract tune flag from mode if not already present
        if 'tune' not in df.columns:
            df['tune'] = df['mode'].astype(str).str.contains('_tune', na=False)
    
    # Handle metric column detection
    # Priority: corrupted_roc_auc > corrupted_score > score > roc_auc
    metric_candidates = ['corrupted_roc_auc', 'corrupted_score', 'score', 'roc_auc']
    metric_col = None
    for candidate in metric_candidates:
        if candidate in df.columns:
            metric_col = candidate
            break
    
    # Detect clean metric column (for baseline/intensity=0.0 handling)
    clean_metric_candidates = ['clean_roc_auc', 'clean_score']
    clean_metric_col = None
    for candidate in clean_metric_candidates:
        if candidate in df.columns:
            clean_metric_col = candidate
            break
    
    if metric_col:
        # Ensure metric column is numeric
        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
        print(f"[INFO] Using metric column: {metric_col}")
    else:
        print("[WARNING] No standard metric column found. Available columns:", list(df.columns))
        print("[WARNING] You may need to specify metric_col in MetricConfig")
    
    if clean_metric_col:
        print(f"[INFO] Found clean metric column: {clean_metric_col}")
    
    # Ensure required columns exist or can be inferred
    required_cols = ['dataset', 'model', 'noise_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns after aggregation: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Filter to test_perturb mode if available
    if 'mode' in df.columns:
        test_perturb_mask = df['mode'].astype(str).str.replace('_tune', '', regex=False) == 'test_perturb'
        if test_perturb_mask.any():
            print(f"[INFO] Filtering to test_perturb mode: {test_perturb_mask.sum()} rows")
            df = df[test_perturb_mask].copy()
        else:
            print("[WARNING] No test_perturb results found in data")
    
    # Add clean_score as intensity 0.0 (same logic as analyze_results.py)
    # This is critical for RD curve computation which needs baseline f(0)
    if clean_metric_col and metric_col and 'intensity' in df.columns:
        print(f"[INFO] Adding clean baseline (intensity=0.0) from {clean_metric_col}...")
        
        # Get clean data (non-null clean metric values)
        clean_data = df.dropna(subset=[clean_metric_col]).copy()
        
        if not clean_data.empty:
            # Get unique noise types from the corrupted data (for creating baselines per noise type)
            noise_types_in_data = df['noise_type'].dropna().unique() if 'noise_type' in df.columns else []
            
            # Determine base grouping columns (excluding noise_type initially)
            # Priority: model, seed, session, subject, eval_mode, tune
            base_group_cols = []
            for col in ['model', 'seed', 'session', 'subject', 'eval_mode', 'tune']:
                if col in clean_data.columns:
                    base_group_cols.append(col)
            
            if base_group_cols:
                # Group clean data by base columns (without noise_type)
                clean_summary_list = []
                
                # For each noise_type in the data, create baseline rows
                for noise_type in noise_types_in_data:
                    # Filter clean data (may have noise_type=None or matching noise_type)
                    # Clean data typically has noise_type=None, but we'll create baselines for each noise_type
                    clean_for_noise = clean_data.copy()
                    
                    # Group by base columns and take first clean value per group
                    clean_grouped = clean_for_noise.groupby(base_group_cols)[clean_metric_col].first().reset_index()
                    
                    # Set noise_type explicitly (important for grouping in metrics computation)
                    clean_grouped['noise_type'] = noise_type
                    
                    clean_summary_list.append(clean_grouped)
                
                if clean_summary_list:
                    # Combine all noise_type baselines
                    clean_summary = pd.concat(clean_summary_list, ignore_index=True)
                    
                    # Set intensity to 0.0 for baseline
                    clean_summary['intensity'] = 0.0
                    
                    # Set the corrupted metric column to the clean metric value
                    # This represents baseline performance at intensity=0
                    clean_summary[metric_col] = clean_summary[clean_metric_col]
                    
                    # Ensure mode and other required columns are set
                    if 'mode' in df.columns:
                        # Use the mode from the original data (test_perturb or test_perturb_tune)
                        mode_value = df['mode'].iloc[0] if len(df) > 0 else 'test_perturb'
                        clean_summary['mode'] = mode_value
                
                # Only add rows that don't already exist (avoid duplicates)
                # Check if we already have intensity=0.0 for these combinations
                if 'intensity' in df.columns:
                    existing_baseline = df[df['intensity'] == 0.0].copy()
                    if not existing_baseline.empty:
                        # Merge to find which clean_summary rows are new
                        # Use base_group_cols + noise_type + intensity for matching
                        merge_cols = base_group_cols + ['noise_type', 'intensity']
                        existing_merge_cols = [c for c in merge_cols if c in existing_baseline.columns]
                        if existing_merge_cols:
                            # Check for duplicates
                            clean_summary_merge_cols = [c for c in merge_cols if c in clean_summary.columns]
                            if clean_summary_merge_cols:
                                # Only keep rows that don't already exist
                                merged = clean_summary.merge(
                                    existing_baseline[existing_merge_cols],
                                    on=clean_summary_merge_cols,
                                    how='left',
                                    indicator=True
                                )
                                clean_summary = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
                
                # Concatenate clean baseline with existing data
                if not clean_summary.empty:
                    # Ensure all columns from df are present in clean_summary
                    for col in df.columns:
                        if col not in clean_summary.columns:
                            if col == metric_col:
                                # Already set above
                                continue
                            elif col == 'intensity':
                                # Already set to 0.0
                                continue
                            elif col == 'mode':
                                # Already set above
                                continue
                            else:
                                # Try to get a default value from df for this column
                                if len(df) > 0 and col in df.columns:
                                    # Use the first non-null value if available
                                    first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                                    clean_summary[col] = first_val
                    
                    # Ensure dataset column is preserved (critical for grouping)
                    if 'dataset' not in clean_summary.columns and 'dataset' in df.columns:
                        if len(df) > 0:
                            clean_summary['dataset'] = df['dataset'].iloc[0]
                    
                    # Reorder columns to match df (preserve column order)
                    clean_summary = clean_summary[[c for c in df.columns if c in clean_summary.columns]]
                    
                    # Concatenate: put clean baseline first, then corrupted data
                    df = pd.concat([clean_summary, df], ignore_index=True)
                    print(f"  [OK] Added {len(clean_summary)} baseline rows (intensity=0.0)")
                else:
                    print("  [INFO] All baseline rows already exist, skipping")
            else:
                print("  [WARNING] No grouping columns available for clean baseline")
        else:
            print("  [WARNING] No clean data found to create baseline")
    elif clean_metric_col is None:
        print("[INFO] No clean metric column found - baseline (intensity=0.0) may be missing")
        print("[INFO] This is OK if your data already includes intensity=0.0 rows")
    
    # Filter to only include intended experimental seeds: [100, 200, 300, 400, 500]
    valid_seeds = [100, 200, 300, 400, 500]
    if 'seed' in df.columns:
        initial_len = len(df)
        # Convert seed to numeric, handling any string representations
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
        # Filter to valid seeds (drop rows with NaN seeds or seeds not in valid list)
        df = df[df['seed'].isin(valid_seeds)].copy()
        filtered_count = initial_len - len(df)
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} rows with seeds not in {valid_seeds}")
        print(f"[INFO] Remaining rows with valid seeds: {len(df)}")
    else:
        print("[WARNING] No 'seed' column found - cannot filter by seed values")
    
    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"[INFO] Removed {initial_len - len(df)} duplicate rows")
    
    print(f"[OK] Final DataFrame shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    return df


def main(
    results_file: Optional[str] = None,
    output_dir: str = "./analysis/robustness_results",
    metric_col: str = "corrupted_score",
    aggregate_from_directories: bool = True,
    hydra: bool = False,
) -> None:
    """
    Main function to compute and save robustness metrics.
    
    Parameters:
    -----------
    results_file : str, optional
        Path to pre-aggregated results CSV file. If None, will aggregate from directories.
    output_dir : str
        Directory to save output files
    metric_col : str
        Metric column name to use (e.g., "corrupted_score", "roc_auc", "score")
    aggregate_from_directories : bool
        If True and results_file is not provided, aggregate from directories
    """
    try:
        print("=" * 80)
        print("ROBUSTNESS METRICS COMPUTATION")
        print("=" * 80)
        
        # Adjust output directory for hydra mode
        if hydra:
            output_dir = os.path.join(output_dir, 'hydra')
            print(f"[INFO] Hydra mode enabled: Including 'branched_wiredcfc_arch4' with core models")
            print(f"[INFO] Results will be saved to: {output_dir}")
        
        # Load results using the aggregation logic
        print("\n[STEP 1] Loading/aggregating results...")
        df = load_results_dataframe(
            results_file=results_file,
            aggregate_from_directories=aggregate_from_directories,
            hydra=hydra
        )
        
        # Configure metric column
        print(f"\n[STEP 2] Configuring metrics (metric_col='{metric_col}')...")
        cfg = MetricConfig(metric_col=metric_col)
        
        # Compute robustness metrics
        print("\n[STEP 3] Computing robustness metrics...")
        print("  - Computing AUPC...")
        print("  - Computing RD curves...")
        print("  - Computing CSV_p curves...")
        results = compute_results_metrics(df, cfg=cfg, hydra=hydra)
        
        # Save results to files
        print("\n[STEP 4] Saving results to files...")
        saved_files = save_robustness_results(
            results=results,
            output_dir=output_dir,
            prefix="robustness_metrics",
            save_csv=True,
            save_excel=True,
            save_summary=True,
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("COMPUTATION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print("\nSaved files:")
        for key, filepath in saved_files.items():
            print(f"  - {key:20s}: {os.path.basename(filepath)}")
        
        # Also save AUPC table if available
        if 'aupc_summary' in results and results['aupc_summary'] is not None:
            aupc_summary = replace_hydra_model_name(results['aupc_summary'], model_col='model')
            aupc_table = make_results_table_aupc(aupc_summary)
            if not aupc_table.empty:
                table_path = os.path.join(output_dir, f"robustness_metrics_aupc_table_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                aupc_table.to_csv(table_path)
                print(f"  - aupc_table: {os.path.basename(table_path)}")
        
        print("\n[OK] All robustness metrics computed and saved successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to compute robustness metrics: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute and save robustness metrics for EEG perturbation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (aggregate from directories)
  python robustness_metrics.py
  
  # Load from pre-aggregated file
  python robustness_metrics.py --results-file results/all_results.csv
  
  # Specify custom output directory and metric column
  python robustness_metrics.py --output-dir ./my_results --metric-col roc_auc
        """
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to pre-aggregated results CSV file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis/robustness_results",
        help="Directory to save output files (default: ./analysis/robustness_results)"
    )
    
    parser.add_argument(
        "--metric-col",
        type=str,
        default="corrupted_score",
        help="Metric column name to use (default: corrupted_score). "
             "Options: corrupted_score, corrupted_roc_auc, score, roc_auc"
    )
    
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Don't aggregate from directories if results file not found"
    )
    
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Include 'branched_wiredcfc_arch4' model along with core models (eegnet, reegnet, cnn_ncp) and save to 'hydra' subdirectory"
    )
    
    args = parser.parse_args()
    
    main(
        results_file=args.results_file,
        output_dir=args.output_dir,
        metric_col=args.metric_col,
        aggregate_from_directories=not args.no_aggregate,
        hydra=args.hydra,
    )
