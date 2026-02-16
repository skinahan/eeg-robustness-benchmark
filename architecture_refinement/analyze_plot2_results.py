from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script
import sys

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import get_noise_perturbation_bounds, short_run_id, _CORRELATED_NOISE_TYPES


def _find_result_files(repo_root: Path, model_name: str) -> List[Path]:
    """Find test_perturb CSV files for this model (short or long path/filename).
    Uses exact path-segment matching: short_run_id must appear as a directory component,
    not as a substring, to avoid collisions when two model names share a prefix.
    """
    results_root = repo_root / "results"
    out: List[Path] = []
    if not results_root.exists():
        return out
    needle = f"{model_name}_test_perturb"  # legacy: filename contained this
    short_id = short_run_id(model_name)
    for root, _dirs, files in os.walk(results_root):
        for f in files:
            if not f.endswith(".csv"):
                continue
            full_path = Path(root) / f
            path_str = str(full_path)
            path_parts = path_str.replace("\\", "/").split("/")
            # New layout: short_run_id must be a path segment (exact match), not substring
            has_short_id_segment = short_id in path_parts
            has_test_perturb = "test_perturb" in path_parts or "test_perturb" in path_str
            if has_short_id_segment and has_test_perturb:
                out.append(full_path)
                continue
            # Fallback: substring match (e.g. if path uses different separators)
            if not has_short_id_segment and short_id in path_str and has_test_perturb:
                out.append(full_path)
                continue
            # Legacy: filename contained model_name_test_perturb
            if needle in f:
                out.append(full_path)
    return sorted(set(out))


def _infer_repo_root_from_plot2_dir(plot2_dir: Path) -> Path:
    # Same strategy as pilot analyzer: walk upwards for a directory that looks like the repo root.
    for cand in [plot2_dir, *plot2_dir.parents]:
        if (cand / "results").exists() and (cand / "evaluation").exists() and (cand / "config.py").exists():
            return cand
        if (cand / "results").exists() and (cand / "evaluation" / "unified_experiment_runner.py").exists():
            return cand
    return _REPO_ROOT


def _load_manifest(plot2_dir: Path) -> dict:
    p = plot2_dir / "plot2_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing Plot 2 manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_selected_architectures(plot2_dir: Path) -> pd.DataFrame:
    p = plot2_dir / "selected_architectures.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing selected architectures table: {p}")
    # selected_architectures.csv can be extremely large because it may contain serialized adjacencies.
    # For analysis we only need lightweight metadata columns.
    try:
        hdr = pd.read_csv(p, nrows=0)
        wanted = [
            "model_name",
            "method",
            "rank",
            "wiring_kind",
            "k",
            "p",
            "graph_seed",
            "wiring_seed",
            "mean_degree_undirected",
            # Plot 2 Overhaul: proxy metrics for I3 diagnostics
            "sigma",
            "te_res",
            "orc_res",
            "C_bin",
            "L_bin",
            "graph_hash",
        ]
        usecols = [c for c in wanted if c in hdr.columns]
        df = pd.read_csv(p, usecols=usecols) if usecols else pd.read_csv(p)
    except Exception:
        df = pd.read_csv(p)
    if "model_name" not in df.columns:
        raise ValueError("selected_architectures.csv must contain a 'model_name' column")
    if "method" not in df.columns:
        df["method"] = "unknown"
    return df


def _k_to_regime(k: int, degree_regimes: Dict[str, List[int]]) -> Optional[str]:
    kk = int(k)
    for name, ks in degree_regimes.items():
        if kk in {int(x) for x in ks}:
            return str(name)
    return None


def _summarize_selected_k(sel: pd.DataFrame, degree_regimes: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    Summarize k/regime distribution for WS-Flex selected architectures, grouped by method.
    """
    out: Dict[str, Any] = {}
    if sel.empty or not degree_regimes:
        return out

    df = sel.copy()
    if "wiring_kind" in df.columns:
        df = df[df["wiring_kind"].astype(str) == "ws_flex"].copy()
    if df.empty or "k" not in df.columns:
        return out

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k"])
    if df.empty:
        return out

    for method, g in df.groupby(df["method"].astype(str)):
        ks = [int(x) for x in g["k"].astype(int).tolist()]
        counts_by_k: Dict[str, int] = {}
        counts_by_regime: Dict[str, int] = {str(r): 0 for r in degree_regimes.keys()}
        unknown_regime = 0
        for k in ks:
            counts_by_k[str(k)] = int(counts_by_k.get(str(k), 0) + 1)
            r = _k_to_regime(k, degree_regimes)
            if r is None:
                unknown_regime += 1
            else:
                counts_by_regime[str(r)] = int(counts_by_regime.get(str(r), 0) + 1)

        uniq = sorted(set(ks))
        out[str(method)] = {
            "n": int(len(ks)),
            "unique_k": uniq,
            "n_unique_k": int(len(uniq)),
            "counts_by_k": {k: int(v) for k, v in sorted(counts_by_k.items(), key=lambda kv: int(kv[0]))},
            "counts_by_regime": counts_by_regime,
            "unknown_regime": int(unknown_regime),
        }
    return out


def _load_model_results(repo_root: Path, model_name: str) -> pd.DataFrame:
    paths = _find_result_files(repo_root, model_name)
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # Fail-fast: if CSV has model column, verify it matches (catches file collision)
            if "model" in df.columns:
                uniq = df["model"].astype(str).unique().tolist()
                if model_name not in uniq and uniq:
                    print(
                        f"[ERROR] {p} contains model(s) {uniq} but expected {model_name!r}. "
                        "Possible file collision or wrong path matching."
                    )
                    continue
                if len(uniq) > 1:
                    df = df[df["model"].astype(str) == model_name].copy()
                    if df.empty:
                        continue
            df["__source_file"] = str(p)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _pick_metric_col(df: pd.DataFrame) -> str:
    for c in ["corrupted_score", "corrupted_roc_auc", "roc_auc", "score"]:
        if c in df.columns:
            return c
    raise KeyError(f"No recognized metric column in results. Columns: {list(df.columns)}")


def _aupc(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    try:
        area = np.trapezoid(y=ys, x=xs)  # type: ignore[attr-defined]
    except AttributeError:
        area = np.trapz(y=ys, x=xs)
    return float(area)


def _validate_intensity_axis(
    xs: np.ndarray, noise_type: str
) -> Tuple[bool, str, Optional[str]]:
    """
    F5: Validate intensity axis for AUPC. Returns (ok, units_hint, error_msg).
    If intensity looks like dB (negative, -100 to 10), warn.
    """
    if xs.size == 0:
        return True, "unknown", None
    xs_f = np.asarray(xs, dtype=float)
    xs_f = xs_f[np.isfinite(xs_f)]
    if xs_f.size < 2:
        return True, "unknown", None
    if not np.all(np.diff(xs_f) >= -1e-12):
        return False, "unknown", (
            f"F5: Intensity grid must be monotonic for {noise_type}. "
            f"Got: min={float(np.min(xs_f)):.4f}, max={float(np.max(xs_f)):.4f}, "
            f"diffs={np.diff(xs_f)[:5].tolist()}..."
        )
    if np.all((xs_f >= -100) & (xs_f <= 10)) and np.any(xs_f < 0):
        return True, "likely_dB", (
            f"F5 WARNING: Intensity for {noise_type} appears to be in dB (range [{float(np.min(xs_f)):.1f}, {float(np.max(xs_f)):.1f}]). "
            "AUPC over dB is not physically linear. Consider alpha mapping."
        )
    return True, "alpha_or_sigma", None


def _compute_seed_level_metrics(
    df: pd.DataFrame,
    *,
    sigma_max: float,
    metric_col: str,
    noise_type: str = "gaussian",
) -> pd.DataFrame:
    """
    Returns rows per seed with AUPC metrics computed from the perturbation curve for the given noise_type.
    sigma_max (or alpha_max for correlated types) is the maximum intensity; used to normalize AUPC to alpha in [0,1].
    F5: Validates intensity column and monotonicity.
    """
    if df.empty:
        return pd.DataFrame()

    if "noise_type" not in df.columns or "intensity" not in df.columns:
        return pd.DataFrame()
    if "seed" not in df.columns and "fold_idx" not in df.columns:
        return pd.DataFrame()

    g = df[df["noise_type"].astype(str) == noise_type].copy()
    if "seed" not in g.columns and "fold_idx" in g.columns:
        g["seed"] = g["fold_idx"]
    if g.empty:
        return pd.DataFrame()

    g["intensity"] = pd.to_numeric(g["intensity"], errors="coerce")
    g["seed"] = pd.to_numeric(g["seed"], errors="coerce")
    g[metric_col] = pd.to_numeric(g[metric_col], errors="coerce")
    for c in ["clean_score", "clean_roc_auc"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["intensity", "seed", metric_col])
    if g.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for seed, gg in g.groupby("seed"):
        seed_int = int(seed)
        curve = gg.groupby("intensity", as_index=False)[metric_col].mean().sort_values("intensity")
        xs = curve["intensity"].to_numpy(dtype=float)
        ys = curve[metric_col].to_numpy(dtype=float)
        if xs.size == 0:
            continue

        # F5: Validate intensity axis
        ok, units_hint, err_msg = _validate_intensity_axis(xs, noise_type)
        if not ok and err_msg:
            raise ValueError(err_msg)
        if ok and units_hint == "likely_dB" and err_msg:
            print(err_msg)

        # Ensure baseline at 0 exists
        if xs[0] > 0.0:
            if "clean_score" in gg.columns:
                clean = gg["clean_score"].dropna()
                y0 = float(clean.mean()) if len(clean) else float(ys[0])
            else:
                y0 = float(ys[0])
            xs = np.concatenate([[0.0], xs])
            ys = np.concatenate([[y0], ys])

        # Clip to [0, sigma_max]
        m = (xs >= 0.0) & (xs <= float(sigma_max) + 1e-9)
        xs = xs[m]
        ys = ys[m]

        a_sigma = _aupc(xs, ys)
        a_alpha = float(a_sigma / sigma_max) if sigma_max > 0 and np.isfinite(a_sigma) else float("nan")

        clean_score = float(gg["clean_score"].mean()) if "clean_score" in gg.columns else float("nan")
        clean_roc_auc = float(gg["clean_roc_auc"].mean()) if "clean_roc_auc" in gg.columns else float("nan")

        # max_drop = ROC(0) - ROC(alpha=1), mid_drop = ROC(0) - ROC(alpha=0.5) (spec experiment_three)
        roc_at_1 = float("nan")
        roc_at_05 = float("nan")
        if xs.size > 0 and sigma_max > 0:
            idx_1 = np.argmin(np.abs(xs - sigma_max))
            idx_05 = np.argmin(np.abs(xs - 0.5 * sigma_max))
            roc_at_1 = float(ys[idx_1]) if np.isfinite(ys[idx_1]) else float("nan")
            roc_at_05 = float(ys[idx_05]) if np.isfinite(ys[idx_05]) else float("nan")
        max_drop = float(clean_roc_auc - roc_at_1) if np.isfinite(clean_roc_auc) and np.isfinite(roc_at_1) else float("nan")
        mid_drop = float(clean_roc_auc - roc_at_05) if np.isfinite(clean_roc_auc) and np.isfinite(roc_at_05) else float("nan")

        rows.append(
            {
                "seed": seed_int,
                "noise_type": noise_type,
                "n_rows": int(len(gg)),
                "sigma_max": float(sigma_max),
                "metric_col": str(metric_col),
                "aupc_sigma": float(a_sigma),
                "aupc_alpha": float(a_alpha),
                "clean_score": clean_score,
                "clean_roc_auc": clean_roc_auc,
                "max_drop": max_drop,
                "mid_drop": mid_drop,
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_hierarchical(
    graphs: Dict[str, List[float]],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Hierarchical bootstrap over graphs -> seeds within graph.
    graphs: {graph_id: [seed_metric, ...]}
    Returns array of bootstrap sample means.
    """
    keys = list(graphs.keys())
    if not keys:
        return np.array([], dtype=float)

    out = np.zeros((int(n_boot),), dtype=float)
    for i in range(int(n_boot)):
        sampled_graphs = rng.choice(keys, size=len(keys), replace=True)
        graph_means = []
        for g in sampled_graphs:
            vals = np.asarray(graphs[g], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            # resample seeds within graph, same count as observed
            samp = rng.choice(vals, size=vals.size, replace=True)
            graph_means.append(float(np.mean(samp)))
        out[i] = float(np.mean(graph_means)) if graph_means else float("nan")
    return out


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = np.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2))
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Plot 2 results (Gaussian AUPC + hierarchical bootstrap)")
    parser.add_argument("--plot2_dir", type=str, required=True)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    plot2_dir = Path(args.plot2_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _infer_repo_root_from_plot2_dir(plot2_dir)

    manifest = _load_manifest(plot2_dir)
    sel = _load_selected_architectures(plot2_dir)

    # F1: Validate method labels; fail hard on unknown methods
    known_methods = {"baseline_a", "baseline_b", "tpe", "baseline", "external_random", "random_stratified"}
    methods_in_sel = set(sel["method"].astype(str).unique()) if "method" in sel.columns else set()
    unknown = methods_in_sel - known_methods
    if unknown:
        raise ValueError(
            f"F1: selected_architectures contains unknown method(s): {unknown}. "
            "Known: baseline_a, baseline_b, tpe, baseline, external_random, random_stratified. "
            "Do not add new methods without updating analysis groups."
        )

    dataset = str(manifest.get("dataset", "BNCI2014_001"))
    saturation_file = str(manifest.get("saturation_file", "saturation_results/saturation_points_summary.csv"))
    manifest_has_perturbation_types = "perturbation_types" in manifest
    perturbation_types_config = manifest.get("perturbation_types", "gaussian")
    if isinstance(perturbation_types_config, list):
        noise_types_to_analyze = [str(x) for x in perturbation_types_config]
    else:
        noise_types_to_analyze = [x.strip() for x in str(perturbation_types_config).split(",") if x.strip()]
    if not noise_types_to_analyze:
        noise_types_to_analyze = ["gaussian"]

    # Primary perturbation for NAS vs Random AUPC comparison (Spec 2: ar1_drift; fallback gaussian for backward compat)
    primary_perturbation_type = str(manifest.get("primary_perturbation_type", "")) or None
    if not primary_perturbation_type or primary_perturbation_type not in noise_types_to_analyze:
        primary_perturbation_type = "ar1_drift" if "ar1_drift" in noise_types_to_analyze else (
            "gaussian" if "gaussian" in noise_types_to_analyze else noise_types_to_analyze[0] if noise_types_to_analyze else "gaussian"
        )

    def _get_sigma_max_for_noise_type(nt: str, df_sample: Optional[pd.DataFrame] = None) -> float:
        _, max_int = get_noise_perturbation_bounds(dataset, nt, saturation_file=saturation_file)
        if nt in _CORRELATED_NOISE_TYPES and (df_sample is not None and not df_sample.empty):
            subset = df_sample[df_sample["noise_type"].astype(str) == nt]
            if not subset.empty and "intensity" in subset.columns:
                imax = pd.to_numeric(subset["intensity"], errors="coerce").max()
                if np.isfinite(imax) and imax > 0:
                    return float(imax)
        return float(max_int)

    per_seed_rows: List[Dict[str, Any]] = []
    raw_curves_rows: List[Dict[str, Any]] = []  # For performance-vs-intensity plots
    first_df_sample: Optional[pd.DataFrame] = None

    for _, arch in sel.iterrows():
        model_name = str(arch["model_name"])
        df = _load_model_results(repo_root, model_name)
        if first_df_sample is None and not df.empty:
            first_df_sample = df
        if df.empty:
            for nt in noise_types_to_analyze:
                sigma_max_nt = _get_sigma_max_for_noise_type(nt, first_df_sample)
                per_seed_rows.append(
                    {
                        **arch.to_dict(),
                        "noise_type": nt,
                        "perturbation_type": nt,
                        "seed": np.nan,
                        "n_rows": 0,
                        "sigma_max": sigma_max_nt,
                        "target_snr_db": float("nan"),
                        "empirical_snr_db": float("nan"),
                        "metric_col": None,
                        "aupc_sigma": np.nan,
                        "aupc_alpha": np.nan,
                        "clean_score": np.nan,
                        "clean_roc_auc": np.nan,
                        "max_drop": np.nan,
                        "mid_drop": np.nan,
                    }
                )
            continue

        metric_col = _pick_metric_col(df)
        method = str(arch.get("method", "unknown"))
        # Collect raw intensity-level data for performance-vs-intensity plots
        seed_col = "seed" if "seed" in df.columns else "fold_idx"
        clean_col = "clean_roc_auc" if "clean_roc_auc" in df.columns else "clean_score"
        for nt in noise_types_to_analyze:
            sub = df[df["noise_type"].astype(str) == nt].copy()
            if sub.empty:
                continue
            sub["intensity"] = pd.to_numeric(sub["intensity"], errors="coerce")
            sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
            if seed_col not in sub.columns:
                continue
            # Add clean baseline (intensity=0) when available (avoid duplicate if intensity=0 already in data)
            has_intensity_zero = (sub["intensity"] == 0.0).any()
            if not has_intensity_zero and clean_col in sub.columns and sub[clean_col].notna().any():
                for seed_val in sub[seed_col].dropna().unique():
                    clean_vals = sub[(sub[seed_col] == seed_val)][clean_col].dropna()
                    if not clean_vals.empty:
                        raw_curves_rows.append({
                            "model_name": model_name,
                            "method": method,
                            "noise_type": nt,
                            "intensity": 0.0,
                            "seed": seed_val,
                            "metric": float(clean_vals.iloc[0]),
                            "metric_col": metric_col,
                        })
            for _, r in sub.dropna(subset=["intensity", metric_col]).iterrows():
                raw_curves_rows.append({
                    "model_name": model_name,
                    "method": method,
                    "noise_type": nt,
                    "intensity": float(r["intensity"]),
                    "seed": r[seed_col],
                    "metric": float(r[metric_col]),
                    "metric_col": metric_col,
                })
        for nt in noise_types_to_analyze:
            sigma_max_nt = _get_sigma_max_for_noise_type(nt, df)
            seed_df = _compute_seed_level_metrics(
                df, sigma_max=sigma_max_nt, metric_col=metric_col, noise_type=nt
            )
            # Snr metadata from raw data for this noise type (Plot 2 bug fix)
            target_snr_nt = float("nan")
            empirical_snr_nt = float("nan")
            if not df.empty and "noise_type" in df.columns:
                sub = df[df["noise_type"].astype(str) == nt]
                if not sub.empty:
                    if "target_snr_db" in sub.columns and sub["target_snr_db"].notna().any():
                        target_snr_nt = float(pd.to_numeric(sub["target_snr_db"], errors="coerce").dropna().iloc[0])
                    if "empirical_snr_db" in sub.columns and sub["empirical_snr_db"].notna().any():
                        empirical_snr_nt = float(pd.to_numeric(sub["empirical_snr_db"], errors="coerce").dropna().iloc[0])
            if seed_df.empty:
                per_seed_rows.append(
                    {
                        **arch.to_dict(),
                        "noise_type": nt,
                        "perturbation_type": nt,
                        "sigma_max": sigma_max_nt,
                        "target_snr_db": target_snr_nt,
                        "empirical_snr_db": empirical_snr_nt,
                        "seed": np.nan,
                        "n_rows": int(len(df)),
                        "metric_col": metric_col,
                        "aupc_sigma": np.nan,
                        "aupc_alpha": np.nan,
                        "clean_score": np.nan,
                        "clean_roc_auc": np.nan,
                        "max_drop": np.nan,
                        "mid_drop": np.nan,
                    }
                )
                continue
            for _, r in seed_df.iterrows():
                row = {**arch.to_dict(), **r.to_dict()}
                row["perturbation_type"] = nt
                row["target_snr_db"] = target_snr_nt
                row["empirical_snr_db"] = empirical_snr_nt
                per_seed_rows.append(row)

    per_seed = pd.DataFrame(per_seed_rows)
    raw_curves = pd.DataFrame(raw_curves_rows) if raw_curves_rows else pd.DataFrame()

    # Validate manifest primary vs actual data (Plot 2 bug fix): do not report correlated type if data are gaussian-only
    if first_df_sample is not None and not first_df_sample.empty and "noise_type" in first_df_sample.columns:
        present_types = first_df_sample["noise_type"].astype(str).unique().tolist()
        if primary_perturbation_type in _CORRELATED_NOISE_TYPES and primary_perturbation_type not in present_types:
            # Data do not contain the primary type (e.g. manifest says ar1_drift but only gaussian in CSVs)
            fallback = "gaussian" if "gaussian" in present_types else (present_types[0] if present_types else "gaussian")
            print(
                f"[WARNING] Manifest primary_perturbation_type={primary_perturbation_type!r} but data contain only: {present_types}. "
                f"Using primary={fallback!r} for report/plot. Verify run used correct --test_perturb_noise_types."
            )
            primary_perturbation_type = fallback
        # Plausibility: if primary is correlated, max intensity should not be gaussian-scale (e.g. 100)
        if primary_perturbation_type in _CORRELATED_NOISE_TYPES and primary_perturbation_type in present_types:
            sub = first_df_sample[first_df_sample["noise_type"].astype(str) == primary_perturbation_type]
            if "intensity" in sub.columns:
                imax = pd.to_numeric(sub["intensity"], errors="coerce").max()
                if np.isfinite(imax) and imax > 50:
                    print(
                        f"[WARNING] Primary type {primary_perturbation_type!r} has max intensity {imax:.1f} "
                        "(gaussian-like range). Confirm correlated perturbation was applied."
                    )

    # F5: Compute intensity axis diagnostics for bootstrap_diff.json (after primary type may have been fallback-adjusted)
    intensity_units = "unknown"
    intensity_monotonic = True
    if first_df_sample is not None and not first_df_sample.empty and "intensity" in first_df_sample.columns:
        sub = first_df_sample[first_df_sample["noise_type"].astype(str) == primary_perturbation_type]
        if not sub.empty:
            xs = pd.to_numeric(sub["intensity"], errors="coerce").dropna().unique()
            xs = np.sort(xs)
            if xs.size >= 2:
                ok, units_hint, _ = _validate_intensity_axis(xs, primary_perturbation_type)
                intensity_units = units_hint
                intensity_monotonic = ok

    # Per-graph mean over seeds, per noise type (graph == model_name in this setup)
    agg_dict = {
        "n_seeds": ("seed", "nunique"),
        "aupc_alpha_mean": ("aupc_alpha", "mean"),
        "aupc_alpha_std": ("aupc_alpha", "std"),
        "clean_roc_auc_mean": ("clean_roc_auc", "mean"),
    }
    if "max_drop" in per_seed.columns and "mid_drop" in per_seed.columns:
        agg_dict["max_drop_mean"] = ("max_drop", "mean")
        agg_dict["max_drop_std"] = ("max_drop", "std")
        agg_dict["mid_drop_mean"] = ("mid_drop", "mean")
        agg_dict["mid_drop_std"] = ("mid_drop", "std")
    if "noise_type" in per_seed.columns:
        per_graph = (
            per_seed.groupby(["model_name", "method", "noise_type"], as_index=False)
            .agg(**{k: v for k, v in agg_dict.items()})
            .sort_values(["noise_type", "method", "aupc_alpha_mean"], ascending=[True, True, False])
        )
    else:
        per_graph = (
            per_seed.groupby(["model_name", "method"], as_index=False)
            .agg(**{k: v for k, v in agg_dict.items()})
            .sort_values(["method", "aupc_alpha_mean"], ascending=[True, False])
        )
    if "noise_type" in per_graph.columns and "perturbation_type" not in per_graph.columns:
        per_graph["perturbation_type"] = per_graph["noise_type"]

    # Hierarchical bootstrap (Plot 2: primary max_drop, secondary AUPC) — per noise type
    # F1: Do NOT pool baseline_b with baseline_a. Primary "Random" comparator = baseline_a only.
    rng = np.random.default_rng(int(args.seed))
    nas_methods = {"tpe"}  # Baseline C
    rand_methods = {"baseline_a"}  # Primary comparator: true random only (F1 patch)
    ext_methods = {"external_random"}
    baseline_a_methods = {"baseline_a"}
    baseline_b_methods = {"baseline_b"}
    ncp_methods = {"baseline"}

    def _build_graph_dict(per_seed_sub: pd.DataFrame, method_set: set, metric_col: str = "aupc_alpha") -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for m in method_set:
            dfm = per_seed_sub[per_seed_sub["method"].astype(str) == m].copy()
            if dfm.empty:
                continue
            if metric_col not in dfm.columns:
                continue
            for model_name, g in dfm.groupby("model_name"):
                vals = pd.to_numeric(g[metric_col], errors="coerce").dropna().to_numpy(dtype=float).tolist()
                if vals:
                    out[str(model_name)] = vals
        return out

    def _ci(a: np.ndarray) -> Tuple[float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(lo), float(hi)

    # Primary perturbation (Spec 2: ar1_drift default; fallback gaussian)
    sigma_max_primary = _get_sigma_max_for_noise_type(primary_perturbation_type, first_df_sample)

    # SNR metadata per perturbation type from data or manifest (Spec 3 PATCH 1)
    def _get_snr_for_noise_type(nt: str, df_sample: Optional[pd.DataFrame]) -> Tuple[float, float]:
        target, empirical = float("nan"), float("nan")
        if df_sample is not None and not df_sample.empty and "noise_type" in df_sample.columns:
            sub = df_sample[df_sample["noise_type"].astype(str) == nt]
            if not sub.empty:
                if "target_snr_db" in sub.columns and sub["target_snr_db"].notna().any():
                    target = float(pd.to_numeric(sub["target_snr_db"], errors="coerce").dropna().iloc[0])
                if "empirical_snr_db" in sub.columns and sub["empirical_snr_db"].notna().any():
                    empirical = float(pd.to_numeric(sub["empirical_snr_db"], errors="coerce").dropna().iloc[0])
        if not np.isfinite(target) and isinstance(manifest.get("target_snr_db"), (int, float)):
            target = float(manifest["target_snr_db"])
        return target, empirical

    snr_by_type: Dict[str, Dict[str, float]] = {}
    for nt in noise_types_to_analyze:
        t_snr, e_snr = _get_snr_for_noise_type(nt, first_df_sample)
        snr_by_type[str(nt)] = {"target_snr_db": t_snr, "empirical_snr_db": e_snr}
    target_snr_primary, empirical_snr_primary = _get_snr_for_noise_type(primary_perturbation_type, first_df_sample)

    per_seed_g = per_seed[(per_seed["noise_type"].astype(str) == primary_perturbation_type)] if "noise_type" in per_seed.columns else per_seed
    nas_graphs = _build_graph_dict(per_seed_g, nas_methods, metric_col="aupc_alpha")
    rand_graphs = _build_graph_dict(per_seed_g, rand_methods, metric_col="aupc_alpha")
    nas_boot = _bootstrap_hierarchical(nas_graphs, n_boot=int(args.n_boot), rng=rng)
    rand_boot = _bootstrap_hierarchical(rand_graphs, n_boot=int(args.n_boot), rng=rng)
    diff_boot = nas_boot - rand_boot
    diff_mean = float(np.nanmean(diff_boot)) if np.isfinite(diff_boot).any() else float("nan")
    diff_ci = _ci(diff_boot)
    per_graph_g = per_graph[(per_graph["noise_type"].astype(str) == primary_perturbation_type)] if "noise_type" in per_graph.columns else per_graph
    nas_graph_means = per_graph_g[per_graph_g["method"].astype(str).isin(nas_methods)]["aupc_alpha_mean"].to_numpy(dtype=float)
    rand_graph_means = per_graph_g[per_graph_g["method"].astype(str).isin(rand_methods)]["aupc_alpha_mean"].to_numpy(dtype=float)
    d = _cohens_d(nas_graph_means, rand_graph_means)
    ci_excludes_0 = bool(np.isfinite(diff_ci[0]) and np.isfinite(diff_ci[1]) and (diff_ci[0] > 0 or diff_ci[1] < 0))
    rand_median = float(np.nanmedian(rand_graph_means)) if np.isfinite(rand_graph_means).any() else float("nan")
    frac_nas_above_rand_median = float(
        np.mean(nas_graph_means > rand_median)
    ) if np.isfinite(rand_median) and np.isfinite(nas_graph_means).any() else float("nan")
    criterion_2 = bool(np.isfinite(frac_nas_above_rand_median) and frac_nas_above_rand_median >= (2.0 / 3.0))

    # Primary metric (spec §2.4): max_drop (lower is better). Bootstrap (random - tpe) and (external - tpe).
    # Revised spec: B−A (proxy validity), C−B (adaptive benefit), B/C−D (WS-Flex vs external).
    has_max_drop = "max_drop" in per_seed_g.columns
    max_drop_rand_graphs = _build_graph_dict(per_seed_g, rand_methods, metric_col="max_drop") if has_max_drop else {}
    max_drop_nas_graphs = _build_graph_dict(per_seed_g, nas_methods, metric_col="max_drop") if has_max_drop else {}
    max_drop_ext_graphs = _build_graph_dict(per_seed_g, ext_methods, metric_col="max_drop") if has_max_drop else {}
    max_drop_a_graphs = _build_graph_dict(per_seed_g, baseline_a_methods, metric_col="max_drop") if has_max_drop else {}
    max_drop_b_graphs = _build_graph_dict(per_seed_g, baseline_b_methods, metric_col="max_drop") if has_max_drop else {}
    if has_max_drop and max_drop_rand_graphs and max_drop_nas_graphs:
        rand_md_boot = _bootstrap_hierarchical(max_drop_rand_graphs, n_boot=int(args.n_boot), rng=rng)
        nas_md_boot = _bootstrap_hierarchical(max_drop_nas_graphs, n_boot=int(args.n_boot), rng=rng)
        diff_md_boot = rand_md_boot - nas_md_boot  # positive = tpe has lower max_drop (more robust)
        diff_md_mean = float(np.nanmean(diff_md_boot)) if np.isfinite(diff_md_boot).any() else float("nan")
        diff_md_ci = _ci(diff_md_boot)
        ci_width_md = float(diff_md_ci[1] - diff_md_ci[0]) if np.isfinite(diff_md_ci[0]) and np.isfinite(diff_md_ci[1]) else float("nan")
        nas_md_means = per_graph_g[per_graph_g["method"].astype(str).isin(nas_methods)]["max_drop_mean"].to_numpy(dtype=float) if "max_drop_mean" in per_graph_g.columns else np.array([])
        rand_md_means = per_graph_g[per_graph_g["method"].astype(str).isin(rand_methods)]["max_drop_mean"].to_numpy(dtype=float) if "max_drop_mean" in per_graph_g.columns else np.array([])
        d_max_drop = _cohens_d(rand_md_means, nas_md_means) if rand_md_means.size and nas_md_means.size else float("nan")
    else:
        diff_md_mean = d_max_drop = float("nan")
        diff_md_ci = (float("nan"), float("nan"))
        ci_width_md = float("nan")
    # B−A (proxy validity): mean(max_drop_B - max_drop_A) < 0 => proxy helps
    diff_md_B_A_mean = float("nan")
    diff_md_B_A_ci = (float("nan"), float("nan"))
    d_max_drop_B_A = float("nan")
    stage2_go = None
    if has_max_drop and max_drop_a_graphs and max_drop_b_graphs:
        a_md_boot = _bootstrap_hierarchical(max_drop_a_graphs, n_boot=int(args.n_boot), rng=rng)
        b_md_boot = _bootstrap_hierarchical(max_drop_b_graphs, n_boot=int(args.n_boot), rng=rng)
        diff_B_A_boot = b_md_boot - a_md_boot  # negative = B more robust
        diff_md_B_A_mean = float(np.nanmean(diff_B_A_boot)) if np.isfinite(diff_B_A_boot).any() else float("nan")
        diff_md_B_A_ci = _ci(diff_B_A_boot)
        a_md_means = per_graph_g[per_graph_g["method"].astype(str).isin(baseline_a_methods)]["max_drop_mean"].to_numpy(dtype=float) if "max_drop_mean" in per_graph_g.columns else np.array([])
        b_md_means = per_graph_g[per_graph_g["method"].astype(str).isin(baseline_b_methods)]["max_drop_mean"].to_numpy(dtype=float) if "max_drop_mean" in per_graph_g.columns else np.array([])
        d_max_drop_B_A = _cohens_d(b_md_means, a_md_means) if b_md_means.size and a_md_means.size else float("nan")
        ci_width_B_A = float(diff_md_B_A_ci[1] - diff_md_B_A_ci[0]) if np.isfinite(diff_md_B_A_ci[0]) and np.isfinite(diff_md_B_A_ci[1]) else float("nan")
        stage2_go = bool(
            np.isfinite(diff_md_B_A_mean) and diff_md_B_A_mean < 0
            and np.isfinite(ci_width_B_A) and ci_width_B_A <= 0.10
        )
    # C−B (adaptive benefit): mean(max_drop_C - max_drop_B) < 0 => TPE helps beyond proxy
    diff_md_C_B_mean = float("nan")
    diff_md_C_B_ci = (float("nan"), float("nan"))
    d_max_drop_C_B = float("nan")
    stage3_go = None
    if has_max_drop and max_drop_b_graphs and max_drop_nas_graphs:
        b_md_boot_cb = _bootstrap_hierarchical(max_drop_b_graphs, n_boot=int(args.n_boot), rng=rng)
        c_md_boot = _bootstrap_hierarchical(max_drop_nas_graphs, n_boot=int(args.n_boot), rng=rng)
        diff_C_B_boot = c_md_boot - b_md_boot_cb  # negative = C more robust
        diff_md_C_B_mean = float(np.nanmean(diff_C_B_boot)) if np.isfinite(diff_C_B_boot).any() else float("nan")
        diff_md_C_B_ci = _ci(diff_C_B_boot)
        b_md_means_cb = per_graph_g[per_graph_g["method"].astype(str).isin(baseline_b_methods)]["max_drop_mean"].to_numpy(dtype=float) if "max_drop_mean" in per_graph_g.columns else np.array([])
        c_md_means = per_graph_g[per_graph_g["method"].astype(str).isin(nas_methods)]["max_drop_mean"].to_numpy(dtype=float) if "max_drop_mean" in per_graph_g.columns else np.array([])
        d_max_drop_C_B = _cohens_d(c_md_means, b_md_means_cb) if c_md_means.size and b_md_means_cb.size else float("nan")
        stage3_go = bool(np.isfinite(diff_md_C_B_mean) and diff_md_C_B_mean < 0)
    diff_md_ext_mean = float("nan")
    diff_md_ext_ci = (float("nan"), float("nan"))
    if has_max_drop and max_drop_ext_graphs and max_drop_nas_graphs:
        ext_md_boot = _bootstrap_hierarchical(max_drop_ext_graphs, n_boot=int(args.n_boot), rng=rng)
        nas_md_boot_ext = _bootstrap_hierarchical(max_drop_nas_graphs, n_boot=int(args.n_boot), rng=rng)
        diff_md_ext_boot = ext_md_boot - nas_md_boot_ext
        diff_md_ext_mean = float(np.nanmean(diff_md_ext_boot)) if np.isfinite(diff_md_ext_boot).any() else float("nan")
        diff_md_ext_ci = _ci(diff_md_ext_boot)
    # Mini-scale success (§4): CI width <= 0.10; directionally consistent in >=2/3 subjects
    n_subjects = per_seed_g["seed"].nunique() if "seed" in per_seed_g.columns else 0
    mini_scale_ci_ok = np.isfinite(ci_width_md) and ci_width_md <= 0.10
    subject_diffs: List[float] = []
    subject_diffs_B_A: List[float] = []
    if "seed" in per_seed_g.columns and "max_drop" in per_seed_g.columns and "method" in per_seed_g.columns:
        for seed in per_seed_g["seed"].dropna().unique():
            sg = per_seed_g[per_seed_g["seed"] == seed]
            tpe_mean = sg[sg["method"].astype(str) == "tpe"]["max_drop"].mean()
            rand_mean = sg[sg["method"].astype(str) == "baseline_a"]["max_drop"].mean()  # F1: baseline_a only
            if np.isfinite(tpe_mean) and np.isfinite(rand_mean):
                subject_diffs.append(float(rand_mean - tpe_mean))
            a_mean = sg[sg["method"].astype(str) == "baseline_a"]["max_drop"].mean()
            b_mean = sg[sg["method"].astype(str) == "baseline_b"]["max_drop"].mean()
            if np.isfinite(a_mean) and np.isfinite(b_mean):
                subject_diffs_B_A.append(float(b_mean - a_mean))  # negative => B more robust
    n_positive = sum(1 for x in subject_diffs if x > 0)
    mini_scale_directional = len(subject_diffs) >= 2 and n_positive >= 2
    stage2_directional = len(subject_diffs_B_A) >= 2 and sum(1 for x in subject_diffs_B_A if x < 0) >= 2
    if stage2_go is not None and len(subject_diffs_B_A) >= 2:
        stage2_go = stage2_go and stage2_directional

    # Report
    report_lines: List[str] = []
    report_lines.append("PLOT 2 TOPOLOGY STUDY REPORT")
    if not manifest_has_perturbation_types:
        report_lines.append("Manifest missing perturbation_types; defaulting to gaussian.")
    report_lines.append(f"plot2_dir: {plot2_dir}")
    report_lines.append(f"repo_root: {repo_root}")
    report_lines.append(f"dataset: {dataset}")
    report_lines.append(f"perturbation_types: {noise_types_to_analyze}")
    report_lines.append(f"primary_perturbation_type: {primary_perturbation_type}")
    if primary_perturbation_type in _CORRELATED_NOISE_TYPES:
        report_lines.append(f"alpha_max ({primary_perturbation_type}, data-derived): {sigma_max_primary}")
    else:
        report_lines.append(f"sigma_max ({primary_perturbation_type}): {sigma_max_primary}")
    report_lines.append(f"target_snr_db ({primary_perturbation_type}): {target_snr_primary}")
    report_lines.append(f"empirical_snr_db ({primary_perturbation_type}): {empirical_snr_primary}")
    for nt in noise_types_to_analyze:
        if nt != primary_perturbation_type and nt in snr_by_type:
            report_lines.append(f"  {nt}: target_snr_db={snr_by_type[nt]['target_snr_db']}, empirical_snr_db={snr_by_type[nt]['empirical_snr_db']}")
    report_lines.append(f"n_boot: {int(args.n_boot)}")
    report_lines.append("bootstrap_hierarchy: graph->seed (F7: resample graphs, then seeds within graph)")
    report_lines.append("Note: Inference is conditional on chosen subjects.")
    report_lines.append("")
    report_lines.append("PRIMARY METRIC: max_drop (lower is better) — spec §2.4")
    report_lines.append(f"  (baseline_a - tpe) in max_drop [F1: baseline_a=Random]: mean={diff_md_mean:.6f}, 95% CI=[{diff_md_ci[0]:.6f}, {diff_md_ci[1]:.6f}], Cohen's d={d_max_drop:.3f}")
    report_lines.append(f"  (external_random - tpe) in max_drop: mean={diff_md_ext_mean:.6f}, 95% CI=[{diff_md_ext_ci[0]:.6f}, {diff_md_ext_ci[1]:.6f}]")
    if np.isfinite(diff_md_B_A_mean):
        report_lines.append(f"  B−A (proxy validity): mean(max_drop_B - max_drop_A)={diff_md_B_A_mean:.6f}, 95% CI=[{diff_md_B_A_ci[0]:.6f}, {diff_md_B_A_ci[1]:.6f}], Cohen's d={d_max_drop_B_A:.3f} (negative => proxy helps)")
        report_lines.append(f"  Stage 2 gate (proxy usefulness): {stage2_go}")
    if np.isfinite(diff_md_C_B_mean):
        report_lines.append(f"  C−B (adaptive benefit): mean(max_drop_C - max_drop_B)={diff_md_C_B_mean:.6f}, 95% CI=[{diff_md_C_B_ci[0]:.6f}, {diff_md_C_B_ci[1]:.6f}], Cohen's d={d_max_drop_C_B:.3f} (negative => TPE helps)")
        report_lines.append(f"  Stage 3 gate (adaptive benefit): {stage3_go}")
    report_lines.append(f"  Mini-scale: CI width <= 0.10? {mini_scale_ci_ok}; directional (>=2 subjects same sign)? {mini_scale_directional}")
    report_lines.append("")
    report_lines.append("SECONDARY METRIC: AUPC_alpha (higher is better)")
    report_lines.append(f"  (NAS - baseline_a) [F1: baseline_a=Random]: mean diff={diff_mean:.6f}, 95% CI=[{diff_ci[0]:.6f}, {diff_ci[1]:.6f}], Cohen's d={d:.3f}")
    report_lines.append(f"  Success: CI excludes 0? {ci_excludes_0}; frac(NAS > baseline_a median)? {frac_nas_above_rand_median:.3f} (>= 0.667? {criterion_2})")
    report_lines.append("")
    for nt in noise_types_to_analyze:
        if nt == primary_perturbation_type:
            continue
        pg_nt = per_graph[per_graph["noise_type"].astype(str) == nt] if "noise_type" in per_graph.columns else pd.DataFrame()
        if pg_nt.empty:
            continue
        report_lines.append(f"Secondary perturbation — {nt}:")
        for m in ["baseline_a", "baseline_b", "random_stratified", "tpe", "baseline", "external_random"]:
            sub = pg_nt[pg_nt["method"].astype(str) == m]
            if len(sub):
                report_lines.append(f"  {m}: mean(AUPC_alpha)={float(sub['aupc_alpha_mean'].mean()):.4f}")
        report_lines.append("")
    report_lines.append(f"Clean ROC-AUC summary (per-graph mean, {primary_perturbation_type}):")
    for m in ["baseline_a", "baseline_b", "random_stratified", "tpe", "baseline", "external_random"]:
        sub = per_graph_g[per_graph_g["method"].astype(str) == m]
        if len(sub):
            report_lines.append(f"  {m}: mean(clean_roc_auc_mean)={float(sub['clean_roc_auc_mean'].mean()):.3f}")

    # Sanity checks: detect suspicious collapse where all graphs have identical metrics (per noise type).
    report_lines.append("")
    report_lines.append("Sanity checks:")
    group_cols = ["noise_type", "method"] if "noise_type" in per_graph.columns else ["method"]
    for key, g in per_graph.groupby(group_cols):
        vals = pd.to_numeric(g["aupc_alpha_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size >= 2 and np.isfinite(vals).all():
            if float(np.max(vals) - np.min(vals)) < 1e-12:
                label = f"noise_type={key[0]} method={key[1]}" if isinstance(key, tuple) and len(key) == 2 else f"method={key}"
                report_lines.append(
                    f"  WARNING: {label} has identical aupc_alpha_mean across {int(vals.size)} graphs "
                    f"(min=max={float(vals[0]):.6f}). This is suspicious; verify model registration/wiring and caching."
                )

    # I3: Negative-result-friendly diagnostics (Plot 2 Overhaul)
    report_lines.append("")
    report_lines.append("I3 Diagnostics (proxy distributions, proxy-k correlations, cell occupancy):")
    if "sigma" in sel.columns and "te_res" in sel.columns:
        for m in ["baseline_a", "baseline_b", "tpe"]:
            sub = sel[sel["method"].astype(str) == m]
            if sub.empty:
                continue
            sigma_vals = pd.to_numeric(sub["sigma"], errors="coerce").dropna()
            te_res_vals = pd.to_numeric(sub["te_res"], errors="coerce").dropna()
            if sigma_vals.size:
                report_lines.append(f"  {m} sigma: mean={float(sigma_vals.mean()):.4f}, std={float(sigma_vals.std()):.4f}, n={int(sigma_vals.size)}")
            if te_res_vals.size:
                report_lines.append(f"  {m} te_res: mean={float(te_res_vals.mean()):.4f}, std={float(te_res_vals.std()):.4f}, n={int(te_res_vals.size)}")
        if "k" in sel.columns:
            for proxy in ["sigma", "te_res"]:
                if proxy in sel.columns:
                    sub = sel[sel["method"].astype(str).isin({"baseline_a", "baseline_b", "tpe"})]
                    if sub.shape[0] >= 3:
                        corr = sub["k"].astype(float).corr(pd.to_numeric(sub[proxy], errors="coerce"))
                        report_lines.append(f"  corr({proxy}, k)={float(corr):.4f}" if np.isfinite(corr) else f"  corr({proxy}, k)=nan")
        if "C_bin" in sel.columns and "L_bin" in sel.columns:
            for m in ["baseline_a", "baseline_b", "tpe"]:
                sub = sel[sel["method"].astype(str) == m]
                if sub.empty:
                    continue
                occ = sub.groupby(["C_bin", "L_bin"]).size().to_dict()
                report_lines.append(f"  {m} cell_occupancy: {occ}")
    elif "sigma" in sel.columns or "te_res" in sel.columns:
        report_lines.append("  (partial proxy columns present; full I3 requires sigma, te_res)")
    else:
        report_lines.append("  (no sigma/te_res columns; I3 skipped for legacy runs)")

    # k/regime distribution diagnostics (helps detect search/selection collapse)
    deg_raw = manifest.get("degree_regimes", {})
    if isinstance(deg_raw, dict) and deg_raw:
        degree_regimes: Dict[str, List[int]] = {}
        for name, ks in deg_raw.items():
            try:
                degree_regimes[str(name)] = [int(x) for x in list(ks)]
            except Exception:
                continue
        dist = _summarize_selected_k(sel, degree_regimes)
        if dist:
            report_lines.append("")
            report_lines.append("Selected architecture k/regime distribution (ws_flex only):")
            for m in ["baseline_a", "baseline_b", "random_stratified", "tpe"]:
                if m not in dist:
                    continue
                dct = dist[m]
                report_lines.append(f"  {m}: n={int(dct.get('n', 0))} unique_k={dct.get('unique_k')}")
                report_lines.append(f"    counts_by_regime={dct.get('counts_by_regime')}")
                report_lines.append(f"    counts_by_k={dct.get('counts_by_k')}")
                if int(dct.get('n_unique_k', 0)) <= 1 and int(dct.get('n', 0)) >= 2:
                    report_lines.append(
                        f"    WARNING: k collapsed to a single value for {m} (unique_k={dct.get('unique_k')})."
                    )

    out_dir = plot2_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Subdirectories for organized diagnostics (analysis/plots/..., analysis/robustness_metrics/...)
    plots_dir = out_dir / "plots"
    perf_vs_int_dir = plots_dir / "performance_vs_intensity"
    robustness_curves_dir = plots_dir / "robustness_curves"
    robustness_metrics_dir = out_dir / "robustness_metrics"
    for d in (plots_dir, perf_vs_int_dir, robustness_curves_dir, robustness_metrics_dir):
        d.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "per_seed_aupc.csv", index=False)
    per_graph.to_csv(out_dir / "per_graph_aupc.csv", index=False)
    _write = {
        "primary_comparator": "baseline_a",
        "diff_mean": diff_mean,
        "diff_ci": {"lo": diff_ci[0], "hi": diff_ci[1]},
        "cohens_d_per_graph_means": d,
        "ci_excludes_0": ci_excludes_0,
        "frac_nas_above_rand_median": frac_nas_above_rand_median,
        "rand_median": rand_median,
        "max_drop_diff_mean": diff_md_mean,
        "max_drop_diff_ci": {"lo": diff_md_ci[0], "hi": diff_md_ci[1]},
        "max_drop_diff_vs_ext_mean": diff_md_ext_mean,
        "max_drop_diff_vs_ext_ci": {"lo": diff_md_ext_ci[0], "hi": diff_md_ext_ci[1]},
        "max_drop_cohens_d": d_max_drop,
        "max_drop_B_minus_A_mean": diff_md_B_A_mean,
        "max_drop_B_minus_A_ci": {"lo": diff_md_B_A_ci[0], "hi": diff_md_B_A_ci[1]},
        "max_drop_B_minus_A_cohens_d": d_max_drop_B_A,
        "max_drop_C_minus_B_mean": diff_md_C_B_mean,
        "max_drop_C_minus_B_ci": {"lo": diff_md_C_B_ci[0], "hi": diff_md_C_B_ci[1]},
        "max_drop_C_minus_B_cohens_d": d_max_drop_C_B,
        "stage2_gate_pass": stage2_go,
        "stage3_gate_pass": stage3_go,
        "stage2_directional": stage2_directional,
        "mini_scale_ci_width_ok": mini_scale_ci_ok,
        "mini_scale_directional": mini_scale_directional,
        "primary_perturbation_type": primary_perturbation_type,
        "sigma_max": sigma_max_primary,
        "target_snr_db": target_snr_primary,
        "empirical_snr_db": empirical_snr_primary,
        "snr_by_perturbation_type": snr_by_type,
        "n_boot": int(args.n_boot),
        "bootstrap_hierarchy": "graph->seed",
        "noise_types": noise_types_to_analyze,
        "intensity_units": intensity_units,
        "intensity_monotonic": intensity_monotonic,
    }
    # I3: Negative-result-friendly diagnostics (always emit when proxy columns present)
    i3_diag: Dict[str, Any] = {}
    if "sigma" in sel.columns and "te_res" in sel.columns:
        for m in ["baseline_a", "baseline_b", "tpe"]:
            sub = sel[sel["method"].astype(str) == m]
            if sub.empty:
                continue
            sigma_vals = pd.to_numeric(sub["sigma"], errors="coerce").dropna().tolist()
            te_res_vals = pd.to_numeric(sub["te_res"], errors="coerce").dropna().tolist()
            i3_diag[f"{m}_sigma"] = {"mean": float(np.mean(sigma_vals)) if sigma_vals else float("nan"), "n": len(sigma_vals)}
            i3_diag[f"{m}_te_res"] = {"mean": float(np.mean(te_res_vals)) if te_res_vals else float("nan"), "n": len(te_res_vals)}
        if "k" in sel.columns:
            sub = sel[sel["method"].astype(str).isin({"baseline_a", "baseline_b", "tpe"})]
            if sub.shape[0] >= 3:
                for proxy in ["sigma", "te_res"]:
                    if proxy in sub.columns:
                        c = sub["k"].astype(float).corr(pd.to_numeric(sub[proxy], errors="coerce"))
                        i3_diag[f"corr_{proxy}_k"] = float(c) if np.isfinite(c) else float("nan")
        if "C_bin" in sel.columns and "L_bin" in sel.columns:
            for m in ["baseline_a", "baseline_b", "tpe"]:
                sub = sel[sel["method"].astype(str) == m]
                if sub.empty:
                    continue
                occ = sub.groupby(["C_bin", "L_bin"]).size().to_dict()
                i3_diag[f"{m}_cell_occupancy"] = {str(k): int(v) for k, v in occ.items()}
        if "graph_hash" in sel.columns:
            def _overlap(m1: str, m2: str) -> float:
                s1 = set(sel[sel["method"].astype(str) == m1]["graph_hash"].dropna().astype(str))
                s2 = set(sel[sel["method"].astype(str) == m2]["graph_hash"].dropna().astype(str))
                if not s1 or not s2:
                    return float("nan")
                return float(len(s1 & s2) / max(len(s1), len(s2)))
            i3_diag["overlap_A_B"] = _overlap("baseline_a", "baseline_b")
            i3_diag["overlap_B_C"] = _overlap("baseline_b", "tpe")
            i3_diag["overlap_A_C"] = _overlap("baseline_a", "tpe")
    _write["i3_diagnostics"] = i3_diag
    (out_dir / "bootstrap_diff.json").write_text(json.dumps(_write, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    # Simple Plot 2-ready figure (bar + CI) if matplotlib is available — primary perturbation.
    # Include all method groups (NCP, A, B, C/TPE, External Random) for consistency with max_drop figure.
    try:
        import matplotlib.pyplot as plt

        method_labels = {
            "baseline": "NCP",
            "baseline_a": "A (true random)",
            "baseline_b": "B (proxy)",
            "random_stratified": "Random WS-Flex",
            "tpe": "C (TPE)",
            "external_random": "External Random",
        }
        colors_by_method = {
            "baseline": "#2ca02c",
            "baseline_a": "#8c8c8c",
            "baseline_b": "#4a90d9",
            "random_stratified": "#777777",
            "tpe": "#2a6fdb",
            "external_random": "#d62728",
        }
        methods_order_aupc = ["baseline", "baseline_a", "baseline_b", "random_stratified", "tpe", "external_random"]
        labels_aupc = ["NCP", "A (true random)", "B (proxy)", "Random WS-Flex", "C (TPE)", "External Random"]
        colors_aupc = ["#2ca02c", "#8c8c8c", "#4a90d9", "#777777", "#2a6fdb", "#d62728"]

        y_aupc, lo_aupc, hi_aupc = [], [], []
        for m in methods_order_aupc:
            method_set = {m}
            graphs = _build_graph_dict(per_seed_g, method_set, metric_col="aupc_alpha")
            if graphs:
                boot = _bootstrap_hierarchical(graphs, n_boot=int(args.n_boot), rng=rng)
                ci = _ci(boot)
                y_aupc.append(float(np.nanmean(boot)) if np.isfinite(boot).any() else float("nan"))
                lo_aupc.append(ci[0])
                hi_aupc.append(ci[1])
            elif m == "baseline":
                # NCP: single model, bootstrap over seeds
                ncp = per_seed_g[per_seed_g["model_name"].astype(str).str.contains("_ncp", na=False)].copy()
                ncp_vals = pd.to_numeric(ncp["aupc_alpha"], errors="coerce").dropna().to_numpy(dtype=float)
                if ncp_vals.size:
                    ncp_boot = rng.choice(ncp_vals, size=(int(args.n_boot), ncp_vals.size), replace=True).mean(axis=1)
                    y_aupc.append(float(np.mean(ncp_boot)))
                    ci = _ci(ncp_boot)
                    lo_aupc.append(ci[0])
                    hi_aupc.append(ci[1])
                else:
                    y_aupc.append(float("nan"))
                    lo_aupc.append(float("nan"))
                    hi_aupc.append(float("nan"))
            else:
                y_aupc.append(float("nan"))
                lo_aupc.append(float("nan"))
                hi_aupc.append(float("nan"))

        # Filter to methods with valid data for display
        valid = [i for i in range(len(y_aupc)) if np.isfinite(y_aupc[i])]
        if valid:
            labels_show = [labels_aupc[i] for i in valid]
            y_show = [y_aupc[i] for i in valid]
            lo_show = [lo_aupc[i] for i in valid]
            hi_show = [hi_aupc[i] for i in valid]
            colors_show = [colors_aupc[i] for i in valid]
            yerr = np.vstack([np.array(y_show) - np.array(lo_show), np.array(hi_show) - np.array(y_show)])

            fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
            ax.bar(range(len(labels_show)), y_show, yerr=yerr, capsize=4, color=colors_show)
            ax.set_xticks(range(len(labels_show)))
            ax.set_xticklabels(labels_show, rotation=15, ha="right")
            ax.set_ylabel("AUPC (alpha-normalized)")
            ax.set_title(f"Plot 2: Robustness ({primary_perturbation_type} AUPC)")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "plot2_figure.png")
            plt.close(fig)

        # Primary figure (spec §8): max_drop by group with bootstrap CI
        if "max_drop_mean" in per_graph_g.columns:
            methods_order = ["baseline", "baseline_a", "baseline_b", "random_stratified", "tpe", "external_random"]
            labels_md = ["Baseline", "A (true random)", "B (proxy)", "Random WS-Flex", "C (TPE)", "External Random"]
            y_md, lo_md, hi_md = [], [], []
            for m in methods_order:
                sub = per_graph_g[per_graph_g["method"].astype(str) == m]
                if sub.empty:
                    y_md.append(float("nan"))
                    lo_md.append(float("nan"))
                    hi_md.append(float("nan"))
                    continue
                vals = pd.to_numeric(sub["max_drop_mean"], errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    y_md.append(float("nan"))
                    lo_md.append(float("nan"))
                    hi_md.append(float("nan"))
                    continue
                y_md.append(float(np.mean(vals)))
                lo_md.append(float(np.percentile(vals, 2.5)))
                hi_md.append(float(np.percentile(vals, 97.5)))
            y_md = np.array(y_md)
            lo_md = np.array(lo_md)
            hi_md = np.array(hi_md)
            yerr_md = np.vstack([y_md - lo_md, hi_md - y_md])
            fig2, ax2 = plt.subplots(figsize=(6.0, 3.6), dpi=150)
            ax2.bar(range(len(labels_md)), y_md, yerr=yerr_md, capsize=4, color=["#2ca02c", "#8c8c8c", "#4a90d9", "#777777", "#2a6fdb", "#d62728"])
            ax2.set_xticks(range(len(labels_md)))
            ax2.set_xticklabels(labels_md, rotation=15, ha="right")
            ax2.set_ylabel("max_drop (lower is better)")
            ax2.set_title(f"Plot 2: Primary robustness — max_drop ({primary_perturbation_type})")
            ax2.grid(True, axis="y", alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(out_dir / "plot2_max_drop_figure.png")
            plt.close(fig2)

        # --- Diagnostic plots: performance vs intensity (per noise type) ---
        if not raw_curves.empty and "method" in raw_curves.columns and "noise_type" in raw_curves.columns:
            for nt in raw_curves["noise_type"].astype(str).unique():
                sub = raw_curves[raw_curves["noise_type"].astype(str) == nt].copy()
                if sub.empty or sub["intensity"].nunique() < 2:
                    continue
                # Aggregate: mean ± sem per (method, intensity)
                agg = sub.groupby(["method", "intensity"], as_index=False).agg(
                    metric_mean=("metric", "mean"),
                    metric_std=("metric", "std"),
                    n=("metric", "count"),
                )
                agg["metric_sem"] = agg["metric_std"] / np.sqrt(agg["n"].clip(lower=1))
                fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=150)
                for method in agg["method"].unique():
                    mdf = agg[agg["method"] == method].sort_values("intensity")
                    if mdf.empty:
                        continue
                    label = method_labels.get(str(method), str(method))
                    color = colors_by_method.get(str(method), "#333333")
                    ax3.plot(
                        mdf["intensity"],
                        mdf["metric_mean"],
                        marker="o",
                        label=label,
                        color=color,
                        linewidth=2,
                        markersize=5,
                    )
                    if mdf["metric_sem"].notna().any():
                        ax3.fill_between(
                            mdf["intensity"],
                            mdf["metric_mean"] - mdf["metric_sem"],
                            mdf["metric_mean"] + mdf["metric_sem"],
                            alpha=0.2,
                            color=color,
                        )
                ax3.set_xlabel("Intensity")
                ax3.set_ylabel("Performance (ROC AUC)")
                ax3.set_title(f"Performance vs Intensity — {nt} ({dataset})")
                ax3.legend(loc="best", fontsize=9)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0.0, 1.05)
                fig3.tight_layout()
                safe_nt = str(nt).replace("/", "_").replace("\\", "_")
                fig3.savefig(perf_vs_int_dir / f"performance_vs_intensity_{safe_nt}.png")
                plt.close(fig3)

        # --- Detailed robustness metrics CSV ---
        robustness_rows: List[Dict[str, Any]] = []
        for _, row in per_graph.iterrows():
            r = {
                "model_name": row.get("model_name"),
                "method": row.get("method"),
                "noise_type": row.get("noise_type"),
                "aupc_alpha_mean": row.get("aupc_alpha_mean"),
                "aupc_alpha_std": row.get("aupc_alpha_std"),
                "clean_roc_auc_mean": row.get("clean_roc_auc_mean"),
                "n_seeds": row.get("n_seeds"),
            }
            if "max_drop_mean" in row:
                r["max_drop_mean"] = row["max_drop_mean"]
                r["max_drop_std"] = row["max_drop_std"]
            if "mid_drop_mean" in row:
                r["mid_drop_mean"] = row["mid_drop_mean"]
                r["mid_drop_std"] = row["mid_drop_std"]
            robustness_rows.append(r)
        if robustness_rows:
            pd.DataFrame(robustness_rows).to_csv(
                robustness_metrics_dir / "robustness_metrics_by_model_method.csv", index=False
            )
        if not raw_curves.empty:
            raw_curves.to_csv(robustness_metrics_dir / "raw_curves_intensity_level.csv", index=False)

        # --- RD (Relative Degradation) curves per noise type ---
        if not raw_curves.empty and "method" in raw_curves.columns:
            for nt in raw_curves["noise_type"].astype(str).unique():
                sub = raw_curves[raw_curves["noise_type"].astype(str) == nt].copy()
                if sub.empty:
                    continue
                sigma_max_nt = _get_sigma_max_for_noise_type(nt, first_df_sample)
                if sigma_max_nt <= 0:
                    continue
                sub["p"] = (sub["intensity"].astype(float) / sigma_max_nt).clip(0.0, 1.0)
                # f0 = clean performance (intensity=0 or min). Use per-method mean at min intensity.
                f0_by_method: Dict[str, float] = {}
                for method in sub["method"].unique():
                    msub = sub[sub["method"] == method]
                    min_p = msub["p"].min()
                    min_p_rows = msub[msub["p"] == min_p]
                    if not min_p_rows.empty:
                        f0_by_method[str(method)] = float(min_p_rows["metric"].mean())
                if not f0_by_method:
                    continue
                rd_rows: List[Dict[str, Any]] = []
                for method in sub["method"].unique():
                    msub = sub[sub["method"] == method].copy()
                    f0 = f0_by_method.get(str(method), float("nan"))
                    if not np.isfinite(f0) or f0 <= 0:
                        continue
                    for _, r in msub.iterrows():
                        rd = (f0 - r["metric"]) / f0
                        rd = float(np.clip(rd, 0.0, 1.0))
                        rd_rows.append({
                            "method": method,
                            "noise_type": nt,
                            "p": r["p"],
                            "intensity": r["intensity"],
                            "metric": r["metric"],
                            "f0": f0,
                            "rd": rd,
                        })
                if not rd_rows:
                    continue
                rd_df = pd.DataFrame(rd_rows)
                agg_rd = rd_df.groupby(["method", "p"], as_index=False).agg(
                    rd_mean=("rd", "mean"),
                    rd_std=("rd", "std"),
                    n=("rd", "count"),
                )
                agg_rd["rd_sem"] = agg_rd["rd_std"] / np.sqrt(agg_rd["n"].clip(lower=1))
                fig4, ax4 = plt.subplots(figsize=(8, 5), dpi=150)
                for method in agg_rd["method"].unique():
                    mdf = agg_rd[agg_rd["method"] == method].sort_values("p")
                    if mdf.empty:
                        continue
                    label = method_labels.get(str(method), str(method))
                    color = colors_by_method.get(str(method), "#333333")
                    ax4.plot(mdf["p"], mdf["rd_mean"], marker="o", label=label, color=color, linewidth=2, markersize=5)
                    if mdf["rd_sem"].notna().any():
                        ax4.fill_between(
                            mdf["p"],
                            mdf["rd_mean"] - mdf["rd_sem"],
                            mdf["rd_mean"] + mdf["rd_sem"],
                            alpha=0.2,
                            color=color,
                        )
                ax4.set_xlabel("Normalized perturbation p (intensity / max)")
                ax4.set_ylabel("Relative Degradation RD(p)")
                ax4.set_title(f"Relative Degradation — {nt} ({dataset})")
                ax4.legend(loc="best", fontsize=9)
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0.0, 1.05)
                fig4.tight_layout()
                safe_nt = str(nt).replace("/", "_").replace("\\", "_")
                fig4.savefig(robustness_curves_dir / f"rd_curve_{safe_nt}.png")
                plt.close(fig4)
                agg_rd.to_csv(robustness_metrics_dir / f"rd_curve_{safe_nt}.csv", index=False)
    except Exception as e:
        import traceback
        print(f"[WARNING] Plotting/diagnostics failed: {e}")
        traceback.print_exc()

    print(f"[OK] Wrote: {out_dir / 'per_seed_aupc.csv'}")
    print(f"[OK] Wrote: {out_dir / 'per_graph_aupc.csv'}")
    print(f"[OK] Wrote: {out_dir / 'bootstrap_diff.json'}")
    print(f"[OK] Wrote: {out_dir / 'report.txt'}")
    if (robustness_metrics_dir / "robustness_metrics_by_model_method.csv").exists():
        print(f"[OK] Wrote: {robustness_metrics_dir / 'robustness_metrics_by_model_method.csv'}")
    if (robustness_metrics_dir / "raw_curves_intensity_level.csv").exists():
        print(f"[OK] Wrote: {robustness_metrics_dir / 'raw_curves_intensity_level.csv'}")
    for p in perf_vs_int_dir.glob("*.png"):
        print(f"[OK] Wrote: {p}")
    for p in robustness_curves_dir.glob("*.png"):
        print(f"[OK] Wrote: {p}")


if __name__ == "__main__":
    main()

