from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running as a script
import sys
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import get_noise_perturbation_bounds, short_run_id


def _load_manifest(pilot_dir: Path) -> dict:
    manifest_path = pilot_dir / "pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing pilot manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_selected_architectures(pilot_dir: Path) -> pd.DataFrame:
    csv_path = pilot_dir / "selected_architectures.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing selected architectures table: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"model_name", "method", "rep", "rank", "k", "p", "graph_seed", "wiring_seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"selected_architectures.csv missing columns: {sorted(missing)}")
    return df


def _infer_repo_root_from_pilot_dir(pilot_dir: Path) -> Path:
    """
    Infer the repository root given a NAS pilot output directory.

    The NAS pilot outputs typically live under:
      <repo_root>/architecture_refinement/outputs/nas_pilot/<run_id>/

    Previously this script used `pilot_dir.parents[2]`, which incorrectly resolves
    to `<repo_root>/architecture_refinement/` (missing `results/`), causing result
    file discovery to fail. We instead walk upwards and look for a directory that
    looks like the repo root.
    """
    # Search upwards for a directory that contains the expected repo structure.
    for cand in [pilot_dir, *pilot_dir.parents]:
        if (cand / "results").exists() and (cand / "evaluation").exists() and (cand / "config.py").exists():
            return cand
        if (cand / "results").exists() and (cand / "evaluation" / "unified_experiment_runner.py").exists():
            return cand

    # Fallback: use the repo root inferred from this script's location.
    return _REPO_ROOT


def _find_result_files(repo_root: Path, model_name: str) -> List[Path]:
    results_root = repo_root / "results"
    out: List[Path] = []
    if not results_root.exists():
        return out
    needle = f"{model_name}_test_perturb"
    short_id = short_run_id(model_name)
    for root, _dirs, files in os.walk(results_root):
        for f in files:
            if not f.endswith(".csv"):
                continue
            full_path = Path(root) / f
            path_str = str(full_path)
            if short_id in path_str and "test_perturb" in path_str:
                out.append(full_path)
            elif needle in f:
                out.append(full_path)
    return sorted(set(out))


def _load_model_results(repo_root: Path, model_name: str) -> pd.DataFrame:
    paths = _find_result_files(repo_root, model_name)
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
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


def _aggregate_gaussian_curve(
    df: pd.DataFrame, *, sigma_max: float, metric_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.array([]), np.array([])
    if "noise_type" not in df.columns or "intensity" not in df.columns:
        return np.array([]), np.array([])

    g = df[df["noise_type"].astype(str) == "gaussian"].copy()
    if g.empty:
        return np.array([]), np.array([])

    g["intensity"] = pd.to_numeric(g["intensity"], errors="coerce")
    g[metric_col] = pd.to_numeric(g[metric_col], errors="coerce")
    g = g.dropna(subset=["intensity", metric_col])
    if g.empty:
        return np.array([]), np.array([])

    # Mean across all folds/sessions/subjects at each intensity.
    curve = g.groupby("intensity", as_index=False)[metric_col].mean().sort_values("intensity")
    xs = curve["intensity"].to_numpy(dtype=float)
    ys = curve[metric_col].to_numpy(dtype=float)

    # Ensure baseline at 0 exists (spec α=0).
    if xs.size == 0:
        return np.array([]), np.array([])
    if xs[0] > 0.0:
        # Use mean clean_score (if present) as α=0 baseline; else reuse first point.
        if "clean_score" in g.columns:
            clean = pd.to_numeric(g["clean_score"], errors="coerce").dropna()
            y0 = float(clean.mean()) if len(clean) else float(ys[0])
        else:
            y0 = float(ys[0])
        xs = np.concatenate([[0.0], xs])
        ys = np.concatenate([[y0], ys])

    # Clip to [0, sigma_max] for safety
    m = (xs >= 0.0) & (xs <= float(sigma_max) + 1e-9)
    xs = xs[m]
    ys = ys[m]
    return xs, ys


def _aupc(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    # NumPy compatibility: `np.trapezoid` is not available in some older versions.
    try:
        area = np.trapezoid(y=ys, x=xs)  # type: ignore[attr-defined]
    except AttributeError:
        area = np.trapz(y=ys, x=xs)
    return float(area)


def _is_monotone_nonincreasing(ys: np.ndarray, eps: float = 1e-6) -> bool:
    if ys.size < 2:
        return True
    return bool(np.all(np.diff(ys) <= eps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NAS pilot study results (Gaussian AUPC, best-so-far curves)")
    parser.add_argument("--pilot_dir", type=str, required=True, help="Path to a NAS pilot run directory (contains pilot_manifest.json).")
    parser.add_argument("--repo_root", type=str, default=None, help="Repo root override (default: inferred from pilot_dir parents).")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _infer_repo_root_from_pilot_dir(pilot_dir)

    manifest = _load_manifest(pilot_dir)
    sel = _load_selected_architectures(pilot_dir)

    dataset = str(manifest.get("dataset", "BNCI2014_001"))
    saturation_file = str(manifest.get("saturation_file", "saturation_results/saturation_points_summary.csv"))
    _, sigma_max = get_noise_perturbation_bounds(dataset, "gaussian", saturation_file=saturation_file)
    sigma_max = float(sigma_max)

    per_arch_rows: List[Dict] = []

    for _, row in sel.iterrows():
        model_name = str(row["model_name"])
        df = _load_model_results(repo_root, model_name)
        if df.empty:
            per_arch_rows.append(
                {
                    **row.to_dict(),
                    "n_rows": 0,
                    "sigma_max": sigma_max,
                    "aupc_sigma": np.nan,
                    "aupc_alpha": np.nan,
                    "monotone_nonincreasing": False,
                }
            )
            continue

        metric_col = _pick_metric_col(df)
        xs, ys = _aggregate_gaussian_curve(df, sigma_max=sigma_max, metric_col=metric_col)
        a_sigma = _aupc(xs, ys)
        a_alpha = float(a_sigma / sigma_max) if sigma_max > 0 and np.isfinite(a_sigma) else float("nan")
        mono = _is_monotone_nonincreasing(ys)

        per_arch_rows.append(
            {
                **row.to_dict(),
                "n_rows": int(len(df)),
                "metric_col": metric_col,
                "sigma_max": sigma_max,
                "aupc_sigma": a_sigma,
                "aupc_alpha": a_alpha,
                "monotone_nonincreasing": bool(mono),
            }
        )

    per_arch = pd.DataFrame(per_arch_rows)

    # Best-so-far curves over b=1..B per method+rep
    curves: List[Dict] = []
    for (method, rep), g in per_arch.groupby(["method", "rep"], as_index=False):
        gg = g.sort_values("rank")
        best = -np.inf
        for _, r in gg.iterrows():
            b = int(r["rank"])
            val = float(r["aupc_sigma"]) if pd.notna(r["aupc_sigma"]) else float("-inf")
            best = max(best, val)
            curves.append({"method": method, "rep": int(rep), "b": b, "best_so_far_aupc_sigma": best})

    curves_df = pd.DataFrame(curves)
    curve_summary = (
        curves_df.groupby(["method", "b"], as_index=False)["best_so_far_aupc_sigma"]
        .agg(mean="mean", std="std")
        .sort_values(["method", "b"])
    )

    # Success criteria (spec-inspired, pilot-scale)
    final_b = int(per_arch["rank"].max()) if len(per_arch) else 0
    final_vals = curves_df[curves_df["b"] == final_b].copy() if final_b else pd.DataFrame()
    report_lines: List[str] = []
    report_lines.append("NAS PILOT STUDY REPORT")
    report_lines.append(f"pilot_dir: {pilot_dir}")
    report_lines.append(f"dataset: {dataset}")
    report_lines.append(f"eval_mode: {manifest.get('eval_mode')}")
    report_lines.append(f"sigma_max (gaussian): {sigma_max}")
    report_lines.append("")

    if not final_vals.empty:
        by_method = final_vals.groupby("method")["best_so_far_aupc_sigma"]
        mu = by_method.mean().to_dict()
        sd = by_method.std().to_dict()
        report_lines.append(f"Final best-so-far (b={final_b}) mean±std:")
        for m in sorted(mu.keys()):
            report_lines.append(f"  {m}: {mu[m]:.6f} ± {sd.get(m, float('nan')):.6f}")
        report_lines.append("")

        random_sd = float(sd.get("random", np.nan))
        margin = 0.5 * random_sd if np.isfinite(random_sd) else np.nan
        if "tpe" in mu and "random" in mu and np.isfinite(margin):
            report_lines.append(f"Gate 1 (final margin ~0.5*std(random) = {margin:.6f}):")
            report_lines.append(f"  mean(tpe) - mean(random) = {(mu['tpe'] - mu['random']):.6f}")
        report_lines.append("")

    # Early advantage at b<=4
    early_max_b = min(4, final_b) if final_b else 0
    if early_max_b:
        early = curve_summary[curve_summary["b"] <= early_max_b].copy()
        # Compare mean curves at each b
        tpe_vs_rand = []
        for b in range(1, early_max_b + 1):
            t = early[(early["method"] == "tpe") & (early["b"] == b)]
            r = early[(early["method"] == "random") & (early["b"] == b)]
            if len(t) and len(r):
                tpe_vs_rand.append((b, float(t["mean"].iloc[0]) - float(r["mean"].iloc[0])))
        report_lines.append(f"Gate 2 (early advantage up to b={early_max_b}):")
        for b, diff in tpe_vs_rand:
            report_lines.append(f"  b={b}: mean(tpe)-mean(random) = {diff:.6f}")
        report_lines.append("")

    # Monotonic degradation sanity
    if "monotone_nonincreasing" in per_arch.columns and len(per_arch):
        mono_rate = float(per_arch["monotone_nonincreasing"].mean())
        report_lines.append("Gate 3 (monotonic degradation sanity):")
        report_lines.append(f"  fraction monotone_nonincreasing: {mono_rate:.3f}")
        report_lines.append("")

    out_dir = pilot_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_arch.to_csv(out_dir / "per_architecture_metrics.csv", index=False)
    curves_df.to_csv(out_dir / "best_so_far_curves.csv", index=False)
    curve_summary.to_csv(out_dir / "best_so_far_curve_summary.csv", index=False)
    (out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Wrote: {out_dir / 'per_architecture_metrics.csv'}")
    print(f"[OK] Wrote: {out_dir / 'best_so_far_curves.csv'}")
    print(f"[OK] Wrote: {out_dir / 'best_so_far_curve_summary.csv'}")
    print(f"[OK] Wrote: {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()

