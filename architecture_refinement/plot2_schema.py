"""
Plot 2 run manifest and result CSV schema validation (Plot_2_Investigation.txt PATCH 0.1).

Hard-fail if required fields are missing; do not silently continue.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

# Required keys in the run manifest (plot2_manifest.json / manifest.json).
REQUIRED_MANIFEST_KEYS: List[str] = [
    "schema_version",
    "run_id",
    "created_at",
    "dataset",
    "eval_mode",
    "H",
    "degree_regimes",
    "perturbation_types",
    "primary_perturbation_type",
    "target_snr_db",
    "perturbation_params",
    "selection",
    "training_seeds",
    "git_commit",
    # Generator bounds (k_min, k_max, p_min, p_max can be inside a nested "generator_bounds" or at top level)
    "search_budget",
]

# Optional but recommended; if present they are validated for type.
# We do not require empirical_snr_db at run start (filled by analyzer post-run).
# Spec §6.3 NEW 3: selected_architectures_csv_path, te_orc_formulas for run manifest.
# plot2_revision: baseline_definitions, shared_random_pool_size for A/B/C baseline documentation.
OPTIONAL_MANIFEST_KEYS: List[str] = [
    "empirical_snr_db",
    "alpha_max",
    "alpha_grid",
    "gaussian_alpha_grid",
    "generator_bounds",
    "selection_method",
    "search_seeds",
    "selected_architectures_csv_path",
    "te_orc_formulas",
    "selection_collapse_scores",
    "baseline_definitions",
    "shared_random_pool_size",
    # Plot 2 Overhaul (vNext+)
    "frozen_bin_edges",
    "mu_orc_by_k",
    "mu_te_by_k",
    "primary_objectives",
    "proxy_viability_run_id",
    "bin_edge_source",
    "metrics_graph_view",
    "seed_ref",
    "seed_pool_ab",
    "seed_tpe",
    "seed_orient",
    "seed_train",
    "compute_budget",
    "proxy_viability_dir",
    "proxy_viability_seed_ref",
    "generator_mode",
    "modular_param_bounds",
    "seed_mod_params",
]

# Valid method / selection_method values in Plot 2 outputs (plot2_revision baselines A/B/C + D/E).
PLOT2_METHOD_VALUES: List[str] = [
    "baseline_a",
    "baseline_b",
    "tpe",
    "random_stratified",  # legacy alias
    "external_random",
    "baseline",
]

# Required columns in per_seed / per_graph result CSVs produced by the evaluation runner.
# When runner writes rows, it must include at least these (runner knows perturbation_type, target_snr_db, empirical_snr_db).
REQUIRED_RESULT_CSV_COLUMNS: List[str] = [
    "perturbation_type",
    "target_snr_db",
    "empirical_snr_db",
]

# Columns required when result CSVs are merged with Plot 2 selected_architectures (graph-level metadata).
# Used when validating aggregated/merged CSVs from a Plot 2 run.
REQUIRED_PLOT2_MERGED_CSV_COLUMNS: List[str] = [
    "perturbation_type",
    "target_snr_db",
    "empirical_snr_db",
    "selection_method",
    "graph_id",
    "k",
    "p",
    "density",
    "clustering",
    "path_length",
    "spectral_radius",  # Plot2_revision3
    "TE_raw",
    "TE_norm",
    "ORC_raw",
    "ORC_norm",
]

# Plot 2 Overhaul: additional columns for selected_architectures / merged analysis (optional for backward compat).
PLOT2_OVERHAUL_MERGED_COLUMNS: List[str] = [
    "sigma",
    "te_res",
    "orc_res",
    "C_bin",
    "L_bin",
    "graph_hash",
    "generator_mode",
    "M",
    "p_out",
    "r_out",
]

# Valid selection_coverage_level values (Plot 2 Overhaul adds regime_cl_bins_fixed).
PLOT2_COVERAGE_LEVEL_VALUES: List[str] = [
    "none",
    "regime",
    "regime_cl_bins",
    "regime_cl_bins_fixed",
]


def validate_manifest(manifest: Dict[str, Any], strict: bool = True) -> None:
    """
    Validate that the run manifest contains all required keys. Hard-fail on first missing key.

    Args:
        manifest: The manifest dict (e.g. from plot2_manifest.json).
        strict: If True, require all REQUIRED_MANIFEST_KEYS. If False, only warn.

    Raises:
        ValueError: If any required key is missing (when strict=True).
    """
    missing: List[str] = []
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            missing.append(key)
    if missing:
        msg = (
            f"Plot 2 manifest is missing required keys: {missing}. "
            "Run must not continue without them (Plot_2_Investigation.txt PATCH 0.1)."
        )
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)


def validate_result_csv_columns(
    df: "pd.DataFrame",
    context: str = "per_seed",
    required: Optional[List[str]] = None,
    strict: bool = True,
) -> None:
    """
    Validate that a result DataFrame has required columns. Hard-fail if any are missing.

    Args:
        df: DataFrame (e.g. per_seed or per_graph aggregated results).
        context: "per_seed", "per_graph", or "plot2_merged".
        required: Override list of required column names. If None, uses REQUIRED_RESULT_CSV_COLUMNS
                 for per_seed/per_graph, or REQUIRED_PLOT2_MERGED_CSV_COLUMNS for plot2_merged.
        strict: If True, raise on missing columns. If False, only warn.

    Raises:
        ValueError: If any required column is missing (when strict=True).
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if required is None:
        if context == "plot2_merged":
            required = list(REQUIRED_PLOT2_MERGED_CSV_COLUMNS)
        else:
            required = list(REQUIRED_RESULT_CSV_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = (
            f"Plot 2 result CSV ({context}) is missing required columns: {missing}. "
            "Run must not continue without them (Plot_2_Investigation.txt PATCH 0.1)."
        )
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)


def get_required_manifest_keys() -> Set[str]:
    """Return set of required manifest keys (for use by runner)."""
    return set(REQUIRED_MANIFEST_KEYS)


def get_required_result_csv_columns() -> List[str]:
    """Return list of required result CSV columns (runner output)."""
    return list(REQUIRED_RESULT_CSV_COLUMNS)
