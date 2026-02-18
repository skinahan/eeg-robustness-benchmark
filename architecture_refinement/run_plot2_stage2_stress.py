"""
Plot 2 Stage 2 — Intermediate stress confirmation run (Plot2_revision2.md).

Takes an existing plot2_dir (with selected architectures and manifest), runs a minimal
job set: Baseline A, Baseline B, and one sanity model (e.g. NCP) × 1 seed × 2–3 subjects,
with AR(1) perturbation at locked SNR. Then computes GO/NO-GO:
- GO if max_drop >= 0.10 for at least one graph and max_pairwise_delta(max_drop) >= 0.03
- NO-GO otherwise (exit 1).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import get_noise_perturbation_bounds, short_run_id, _CORRELATED_NOISE_TYPES


def _infer_repo_root(plot2_dir: Path) -> Path:
    for cand in [plot2_dir, *plot2_dir.parents]:
        if (cand / "results").exists() and (cand / "evaluation" / "unified_experiment_runner.py").exists():
            return cand
    return _REPO_ROOT


def _load_manifest(plot2_dir: Path) -> dict:
    for name in ("plot2_manifest.json", "manifest.json"):
        p = plot2_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No manifest found in {plot2_dir}")


def _load_selected(plot2_dir: Path) -> pd.DataFrame:
    p = plot2_dir / "selected_architectures.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p)


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
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def _pick_metric_col(df: pd.DataFrame) -> str:
    for c in ("corrupted_roc_auc", "corrupted_score", "roc_auc", "score"):
        if c in df.columns:
            return c
    return "corrupted_roc_auc"


def _compute_max_drop_per_seed(
    df: pd.DataFrame,
    sigma_max: float,
    metric_col: str,
    noise_type: str,
) -> List[float]:
    """Return list of max_drop per seed (clean_roc_auc - roc at max intensity).

    Uses the maximum intensity present in the data (not sigma_max) because for
    correlated noise types (ar1_drift, spatial_gaussian, emg_band) the runner
    uses alpha * alpha_max with SNR-calibrated alpha_max, so actual intensities
    may differ from nominal bounds. Picking "closest to sigma_max" can select
    the wrong point (e.g. near-clean when alpha_max is small), yielding max_drop≈0.
    """
    if df.empty or "noise_type" not in df.columns or "intensity" not in df.columns:
        return []
    g = df[df["noise_type"].astype(str) == noise_type].copy()
    if "seed" not in g.columns and "fold_idx" in g.columns:
        g["seed"] = g["fold_idx"]
    if g.empty:
        return []
    g["intensity"] = pd.to_numeric(g["intensity"], errors="coerce")
    g["seed"] = pd.to_numeric(g["seed"], errors="coerce")
    g[metric_col] = pd.to_numeric(g[metric_col], errors="coerce")
    for c in ("clean_roc_auc", "clean_score"):
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["intensity", "seed", metric_col])
    if g.empty:
        return []
    out: List[float] = []
    for _seed, gg in g.groupby("seed"):
        clean_roc = float(gg["clean_roc_auc"].mean()) if "clean_roc_auc" in gg.columns else float("nan")
        curve = gg.groupby("intensity", as_index=False)[metric_col].mean().sort_values("intensity")
        xs = curve["intensity"].to_numpy(dtype=float)
        ys = curve[metric_col].to_numpy(dtype=float)
        if xs.size == 0 or not np.isfinite(clean_roc):
            continue
        # Use max intensity in data (actual highest perturbation tested), not sigma_max.
        # For ar1_drift etc., intensities = alpha * alpha_max; sigma_max from bounds
        # is nominal and can mismatch, causing wrong point selection and max_drop≈0.
        intensity_max = float(np.max(xs))
        if intensity_max <= 0:
            continue  # No meaningful perturbation; max_drop undefined
        idx_max = np.argmax(xs)
        roc_at_max = float(ys[idx_max]) if np.isfinite(ys[idx_max]) else float("nan")
        max_drop = float(clean_roc - roc_at_max) if np.isfinite(roc_at_max) else float("nan")
        if np.isfinite(max_drop):
            out.append(max_drop)
    return out


def _run_unified_job(
    *,
    repo_root: Path,
    python_exe: str,
    plot2_dir: Path,
    model_name: str,
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    seed: int,
    saturation_file: str,
    alpha_grid: List[float],
    perturbation_types: List[str],
    target_snr_db: float = -5.0,
    perturbation_params: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> int:
    gaussian_only = perturbation_types == ["gaussian"]
    params = perturbation_params or {}
    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").as_posix()),
        "--nas_pilot_dir", str(plot2_dir.as_posix()),
        "--model", model_name,
        "--dataset", dataset,
        "--subjects", *[str(s) for s in subjects],
        "--mode", "test_perturb",
        "--eval_mode", eval_mode,
        "--seed", str(seed),
        "--disable_underfitting_retrain",
        "--noise_perturbation_saturation_file", saturation_file,
        "--noise_perturbation_num_steps", "20",
        "--test_perturb_gaussian_alpha_grid", ",".join(str(a) for a in alpha_grid),
        "--test_perturb_target_snr_db=" + str(target_snr_db),
        "--test_perturb_ar1_rho", str(params.get("ar1_drift", {}).get("rho", 0.97)),
        "--plot2_diagnostics_dir", str((plot2_dir / "diagnostics").as_posix()),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if gaussian_only:
        cmd.append("--test_perturb_gaussian_only")
    else:
        cmd.extend(["--test_perturb_noise_types", ",".join(perturbation_types)])
    cmd = [c for c in cmd if c]
    print("[Stage2] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 2 Stage 2: stress confirmation run (A, B, sanity × 1 seed × 2–3 subjects), then GO/NO-GO."
    )
    parser.add_argument("--plot2_dir", type=str, required=True, help="Existing Plot 2 run directory (with selected_architectures and manifest).")
    parser.add_argument("--repo_root", type=str, default=None, help="Repo root (default: inferred from plot2_dir).")
    parser.add_argument("--subjects", type=int, nargs="*", default=[0, 1, 2], help="Subject indices (default: 0 1 2).")
    parser.add_argument("--seed", type=int, default=1, help="Single training seed (default: 1).")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for unified runner.")
    parser.add_argument("--skip_run", action="store_true", help="Skip training; only compute GO/NO-GO from existing results.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result files.")
    args = parser.parse_args()

    plot2_dir = Path(args.plot2_dir).resolve()
    if not plot2_dir.is_dir():
        print(f"Error: plot2_dir not found: {plot2_dir}", file=sys.stderr)
        sys.exit(2)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _infer_repo_root(plot2_dir)

    manifest = _load_manifest(plot2_dir)
    sel = _load_selected(plot2_dir)

    dataset = str(manifest.get("dataset", "BNCI2014_001"))
    saturation_file = str(manifest.get("saturation_file", "saturation_results/saturation_points_summary.csv"))
    alpha_grid_raw = manifest.get("alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    if isinstance(alpha_grid_raw, str):
        alpha_grid = [float(x.strip()) for x in alpha_grid_raw.split(",") if x.strip()]
    else:
        alpha_grid = [float(x) for x in (alpha_grid_raw or [0.0, 0.25, 0.5, 0.75, 1.0])]
    if not alpha_grid:
        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    perturbation_types = manifest.get("perturbation_types", "ar1_drift")
    if isinstance(perturbation_types, list):
        pt = [str(x) for x in perturbation_types]
    else:
        pt = [x.strip() for x in str(perturbation_types).split(",") if x.strip()]
    if not pt:
        pt = ["ar1_drift"]
    primary = str(manifest.get("primary_perturbation_type", pt[0] if pt else "ar1_drift"))
    if primary not in pt:
        primary = pt[0]
    eval_mode = str(manifest.get("eval_mode", "CrossSession"))
    target_snr_db = float(manifest.get("target_snr_db", -5.0))
    perturbation_params = manifest.get("perturbation_params") or {}

    # Stage 2 models: baseline_a, baseline_b, and one sanity (ncp)
    methods_stage2 = {"baseline_a", "baseline_b"}
    sel_a_b = sel[sel["method"].astype(str).isin(methods_stage2)]
    ncp = sel[(sel["method"].astype(str) == "baseline") | (sel["model_name"].astype(str).str.contains("ncp", case=False, na=False))]
    sanity = ncp.head(1) if not ncp.empty else pd.DataFrame()
    models_df = pd.concat([sel_a_b, sanity], ignore_index=True)
    if models_df.empty:
        print("Error: no baseline_a/b or sanity model found in selected_architectures.csv", file=sys.stderr)
        sys.exit(2)
    model_names = models_df["model_name"].astype(str).unique().tolist()
    subjects_list = list(args.subjects)[:3] if args.subjects else [0, 1, 2]

    if not args.skip_run:
        for model_name in model_names:
            rc = _run_unified_job(
                repo_root=repo_root,
                python_exe=args.python,
                plot2_dir=plot2_dir,
                model_name=model_name,
                dataset=dataset,
                eval_mode=eval_mode,
                subjects=subjects_list,
                seed=args.seed,
                saturation_file=saturation_file,
                alpha_grid=alpha_grid,
                perturbation_types=pt,
                target_snr_db=target_snr_db,
                perturbation_params=perturbation_params,
                overwrite=args.overwrite,
            )
            if rc != 0:
                print(f"Error: unified runner failed for {model_name} (exit {rc})", file=sys.stderr)
                sys.exit(2)

    # Compute GO/NO-GO: max_drop >= 0.10 for at least one graph and max_pairwise_delta(max_drop) >= 0.03
    _, sigma_max = get_noise_perturbation_bounds(dataset, primary, saturation_file=saturation_file)
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = 1.0
    max_drops: List[Tuple[str, float]] = []
    for model_name in model_names:
        df = _load_model_results(repo_root, model_name)
        if df.empty:
            print(f"Warning: no results for {model_name}; skipping.", file=sys.stderr)
            continue
        metric_col = _pick_metric_col(df)
        drops = _compute_max_drop_per_seed(df, sigma_max=sigma_max, metric_col=metric_col, noise_type=primary)
        if drops:
            mean_drop = float(np.mean(drops))
            max_drops.append((model_name, mean_drop))
    if not max_drops:
        print("NO-GO: no valid max_drop values (missing or empty results).", file=sys.stderr)
        sys.exit(1)
    values = [v for _, v in max_drops]
    any_ge_01 = any(v >= 0.10 for v in values)
    pairwise = [abs(a - b) for a in values for b in values if a != b]
    max_pairwise = max(pairwise) if pairwise else 0.0
    go = any_ge_01 and max_pairwise >= 0.03
    print("Stage 2 stress run — max_drop per graph (mean over seeds):")
    for name, v in max_drops:
        print(f"  {name}: max_drop={v:.4f}")
    print(f"  any max_drop >= 0.10: {any_ge_01}")
    print(f"  max_pairwise_delta(max_drop): {max_pairwise:.4f}")
    print(f"  GO (max_drop>=0.10 for some graph and max_pairwise_delta>=0.03): {go}")
    sys.exit(0 if go else 1)


if __name__ == "__main__":
    main()
