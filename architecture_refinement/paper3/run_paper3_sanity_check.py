"""
Paper 3 Sanity Check: Best vs Worst scalar-proxy topology.

Selects best and worst graphs from proxy pool by s(G)=0.5*(te_hat+orc_hat),
trains both for S seeds, evaluates robustness (AR(1) drift), outputs curves and RD_max.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx

from architecture_refinement.ws_flex_generator import build_plain_ws_flex
from utils import results_paradigm_folder, short_run_id

DEFAULT_S = 5
DEFAULT_SATURATION = "saturation_results/saturation_points_summary.csv"
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _run_unified_job(
    *,
    repo_root: Path,
    python_exe: str,
    pilot_dir: Path,
    model_name: str,
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    seed: int,
    saturation_file: str,
    alpha_grid: List[float],
    perturbation_types: List[str],
    target_snr_db: float,
    perturbation_params: Optional[Dict[str, Any]],
    overwrite: bool,
) -> int:
    """Invoke unified_experiment_runner for one (model, seed) job."""
    params = perturbation_params or {}
    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").as_posix()),
        "--nas_pilot_dir",
        str(pilot_dir.as_posix()),
        "--model",
        model_name,
        "--dataset",
        dataset,
        "--subjects",
        *[str(s) for s in subjects],
        "--mode",
        "test_perturb",
        "--eval_mode",
        eval_mode,
        "--seed",
        str(seed),
        "--overwrite" if overwrite else "",
        "--disable_underfitting_retrain",
        "--noise_perturbation_saturation_file",
        saturation_file,
        "--noise_perturbation_num_steps",
        "20",
        "--test_perturb_gaussian_alpha_grid",
        ",".join(str(a) for a in alpha_grid),
        "--test_perturb_target_snr_db",
        str(target_snr_db),
        "--test_perturb_ar1_rho",
        str(params.get("ar1_drift", {}).get("rho", 0.97)),
        "--test_perturb_noise_types",
        ",".join(perturbation_types),
    ]
    cmd = [c for c in cmd if c]
    print(f"[SanityCheck] Running: {' '.join(cmd[:14])}...")
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def _collect_perturb_results(
    repo_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    noise_type: str,
) -> Optional[Dict[str, Any]]:
    """Collect clean_roc_auc, corrupted_roc_auc by intensity, max_drop, AUPC."""
    paradigm = results_paradigm_folder(dataset)
    base = repo_root / "results" / paradigm / dataset
    for stem in [short_run_id(model_name), model_name]:
        path = base / stem / "CrossSessionEvaluation" / str(seed)
        if not path.exists():
            continue
        for p in path.rglob("*.csv"):
            if "test_perturb" not in str(p):
                continue
            try:
                df = pd.read_csv(p)
                if "noise_type" not in df.columns or noise_type not in df["noise_type"].astype(str).values:
                    continue
                sub = df[df["noise_type"].astype(str) == noise_type].copy()
                if sub.empty:
                    continue
                sub["corrupted_roc_auc"] = pd.to_numeric(sub["corrupted_roc_auc"], errors="coerce")
                sub["clean_roc_auc"] = pd.to_numeric(
                    sub.get("clean_roc_auc", sub.get("clean_score", np.nan)), errors="coerce"
                )
                sub["intensity"] = pd.to_numeric(sub.get("intensity", np.nan), errors="coerce")
                clean = float(sub["clean_roc_auc"].iloc[0]) if sub["clean_roc_auc"].notna().any() else float("nan")
                intensities = sub["intensity"].to_numpy()
                roc_vals = sub["corrupted_roc_auc"].to_numpy()
                r_t = roc_vals / clean if np.isfinite(clean) and clean > 0 else np.full_like(roc_vals, np.nan)
                rd_max = float(np.nanmax(1.0 - r_t)) if np.isfinite(r_t).any() else float("nan")
                return {
                    "clean_roc_auc": clean,
                    "intensities": intensities.tolist(),
                    "corrupted_roc_auc": roc_vals.tolist(),
                    "r_t": r_t.tolist(),
                    "RD_max": rd_max,
                }
            except Exception as e:
                print(f"[SanityCheck] Warning reading {p}: {e}")
    return None


def run_sanity_check(
    experiment1_dir: Path,
    output_dir: Path,
    S: int = DEFAULT_S,
    dataset: str = "BNCI2014_001",
    eval_mode: str = "CrossSession",
    subjects: Optional[List[int]] = None,
    saturation_file: str = DEFAULT_SATURATION,
    target_snr_db: float = -5.0,
    python_exe: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run sanity check: best vs worst proxy topology.
    """
    if subjects is None:
        subjects = list(range(1, 10))
    python_exe = python_exe or sys.executable

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = experiment1_dir / "proxy_pool.csv"
    if not pool_path.exists():
        raise FileNotFoundError(f"Proxy pool not found: {pool_path}. Run Experiment 1 first.")

    df = pd.read_csv(pool_path)
    df["s"] = 0.5 * (df["te_hat"] + df["orc_hat"])
    best_idx = df["s"].idxmax()
    worst_idx = df["s"].idxmin()
    best_rec = df.loc[best_idx].to_dict()
    worst_rec = df.loc[worst_idx].to_dict()

    H = int(best_rec["H"])
    pilot_dir = output_dir / "sanity_check_pilot"
    selected_dir = pilot_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    def _write_arch(rec: Dict, label: str) -> str:
        G = build_plain_ws_flex(H, int(rec["k"]), float(rec["p"]), int(rec["graph_seed"]))
        adj = (np.asarray(nx.to_numpy_array(G)) != 0).astype(np.int8)
        model_name = f"paper3_sanity_{label}"
        arch = {
            "schema_version": 2,
            "model_name": model_name,
            "H": H,
            "wiring_kind": "ws_flex",
            "hidden_edge_orientation": "symmetric",
            "k": int(rec["k"]),
            "p": float(rec["p"]),
            "graph_seed": int(rec["graph_seed"]),
            "wiring_seed": int(rec["graph_seed"]),
            "te_hat": float(rec["te_hat"]),
            "orc_hat": float(rec["orc_hat"]),
            "hidden_adj_undirected": adj.tolist(),
        }
        (selected_dir / f"{model_name}.json").write_text(json.dumps(arch, indent=2))
        return model_name

    best_model = _write_arch(best_rec, "best")
    worst_model = _write_arch(worst_rec, "worst")

    manifest = {
        "best": {"model": best_model, "te_hat": float(best_rec["te_hat"]), "orc_hat": float(best_rec["orc_hat"]), "s": float(best_rec["s"])},
        "worst": {"model": worst_model, "te_hat": float(worst_rec["te_hat"]), "orc_hat": float(worst_rec["orc_hat"]), "s": float(worst_rec["s"])},
        "S": S,
        "dataset": dataset,
        "eval_mode": eval_mode,
        "perturbation_types": ["ar1_drift"],
        "target_snr_db": target_snr_db,
    }
    (output_dir / "sanity_check_manifest.json").write_text(json.dumps(manifest, indent=2))

    if dry_run:
        print("[SanityCheck] Dry-run: would train best and worst for S seeds.")
        return manifest

    pert_params = {"ar1_drift": {"rho": 0.97}}
    seeds = list(range(42, 42 + S))
    failed = []
    for model_name in [best_model, worst_model]:
        for seed in seeds:
            rc = _run_unified_job(
                repo_root=_REPO_ROOT,
                python_exe=python_exe,
                pilot_dir=pilot_dir,
                model_name=model_name,
                dataset=dataset,
                eval_mode=eval_mode,
                subjects=subjects,
                seed=seed,
                saturation_file=str(_REPO_ROOT / saturation_file) if not Path(saturation_file).is_absolute() else saturation_file,
                alpha_grid=ALPHA_GRID,
                perturbation_types=["ar1_drift"],
                target_snr_db=target_snr_db,
                perturbation_params=pert_params,
                overwrite=overwrite,
            )
            if rc != 0:
                failed.append({"model": model_name, "seed": seed})

    if failed:
        (output_dir / "failed_jobs.json").write_text(json.dumps(failed, indent=2))
        print(f"[SanityCheck] {len(failed)} jobs failed.")
        return manifest

    # Collect results
    best_results = []
    worst_results = []
    for seed in seeds:
        r_b = _collect_perturb_results(_REPO_ROOT, dataset, best_model, seed, "ar1_drift")
        r_w = _collect_perturb_results(_REPO_ROOT, dataset, worst_model, seed, "ar1_drift")
        if r_b:
            best_results.append(r_b)
        if r_w:
            worst_results.append(r_w)

    # Aggregate: mean r_t, RD_max with 95% bootstrap CI
    def _bootstrap_ci(values: List[float], n_boot: int = 1000) -> tuple:
        if not values or len(values) < 2:
            return float(np.mean(values)) if values else float("nan"), float("nan"), float("nan")
        arr = np.array(values)
        rng = np.random.default_rng(42)
        boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
        return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    rd_best = [r["RD_max"] for r in best_results if np.isfinite(r.get("RD_max", np.nan))]
    rd_worst = [r["RD_max"] for r in worst_results if np.isfinite(r.get("RD_max", np.nan))]
    summary = {
        "best_RD_max_mean": float(np.mean(rd_best)) if rd_best else float("nan"),
        "best_RD_max_ci": _bootstrap_ci(rd_best) if rd_best else (float("nan"), float("nan"), float("nan")),
        "worst_RD_max_mean": float(np.mean(rd_worst)) if rd_worst else float("nan"),
        "worst_RD_max_ci": _bootstrap_ci(rd_worst) if rd_worst else (float("nan"), float("nan"), float("nan")),
        "n_best": len(best_results),
        "n_worst": len(worst_results),
    }
    (output_dir / "sanity_check_summary.json").write_text(json.dumps(summary, indent=2))

    # Write CSV for plotting
    with open(output_dir / "sanity_check_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "intensity_idx", "intensity", "r_t", "RD_max", "clean_roc_auc"])
        for label, results in [("best", best_results), ("worst", worst_results)]:
            for r in results:
                for i, (inten, rt) in enumerate(zip(r.get("intensities", []), r.get("r_t", []))):
                    w.writerow([label, "agg", i, inten, rt, r.get("RD_max"), r.get("clean_roc_auc")])

    print(f"[SanityCheck] Done. Best RD_max={summary.get('best_RD_max_mean', 'nan'):.4f}, Worst RD_max={summary.get('worst_RD_max_mean', 'nan'):.4f}")
    return {**manifest, **summary}


def main():
    parser = argparse.ArgumentParser(description="Paper 3 Sanity Check: best vs worst proxy")
    parser.add_argument("--experiment1-dir", type=str, required=True, help="Path to Experiment 1 output (contains proxy_pool.csv)")
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/paper3_sanity_check")
    parser.add_argument("--S", type=int, default=DEFAULT_S)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval-mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--saturation-file", type=str, default=DEFAULT_SATURATION)
    parser.add_argument("--target-snr-db", type=float, default=-5.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp1_dir = Path(args.experiment1_dir)
    if not exp1_dir.is_absolute():
        exp1_dir = _REPO_ROOT / exp1_dir
    out_dir = _REPO_ROOT / args.output_dir
    run_sanity_check(
        experiment1_dir=exp1_dir,
        output_dir=out_dir,
        S=args.S,
        dataset=args.dataset,
        eval_mode=args.eval_mode,
        subjects=args.subjects,
        saturation_file=args.saturation_file,
        target_snr_db=args.target_snr_db,
        python_exe=args.python,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
