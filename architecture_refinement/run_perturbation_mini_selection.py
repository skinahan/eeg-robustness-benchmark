"""
Very Fast Perturbation-Selection Mini Experiment (Paper 3).

Rapidly decides which perturbation(s) to use for final Plot 2/3 robustness evaluation.
Uses 4 topologies (T_best, T_worst, T_mid, B_dense), 2-3 seeds, minimal 4-level intensity grid.
Outputs Separation/Stability per perturbation and PASS/FAIL/BORDERLINE tags.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx

from architecture_refinement.ws_flex_generator import build_plain_ws_flex
from architecture_refinement.paper3.run_paper3_experiment2 import _make_dense_arch
from utils import short_run_id

DEFAULT_H = 32
DEFAULT_S_MINI = 2
DEFAULT_SATURATION = "saturation_results/saturation_points_summary.csv"
ALPHA_GRID_MINI = [0.0, 0.33, 0.66, 1.0]  # 4 levels including clean (t=0,1,2,3)
DEFAULT_PERTURBATIONS = ["ar1_drift", "gaussian"]


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
        str(params.get("ar1_drift", {}).get("rho", 0.95)),
        "--test_perturb_noise_types",
        ",".join(perturbation_types),
    ]
    cmd = [c for c in cmd if c]
    print(f"[MiniSelection] Running: {model_name} seed={seed} ...")
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def _collect_perturb_results(
    repo_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    noise_type: str,
) -> Optional[Dict[str, Any]]:
    """Collect clean_roc_auc, corrupted_roc_auc by intensity, RD_max."""
    paradigm = "MotorImagery" if "BNCI" in dataset else "SSVEP"
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
                print(f"[MiniSelection] Warning reading {p}: {e}")
    return None


def _select_topologies(df: pd.DataFrame, H: int) -> Tuple[Dict, Dict, Dict]:
    """Select T_best, T_worst, T_mid from proxy pool by s(G)."""
    df = df.copy()
    df["s"] = 0.5 * (df["te_hat"] + df["orc_hat"])
    best_idx = df["s"].idxmax()
    worst_idx = df["s"].idxmin()
    # T_mid: median s (closest to median value)
    median_s = df["s"].median()
    mid_idx = (df["s"] - median_s).abs().idxmin()
    return (
        df.loc[best_idx].to_dict(),
        df.loc[worst_idx].to_dict(),
        df.loc[mid_idx].to_dict(),
    )


def _write_ws_flex_arch(rec: Dict, label: str, selected_dir: Path, H: int) -> str:
    """Write WS-Flex architecture JSON, return model name."""
    G = build_plain_ws_flex(H, int(rec["k"]), float(rec["p"]), int(rec["graph_seed"]))
    adj = (np.asarray(nx.to_numpy_array(G)) != 0).astype(np.int8)
    model_name = f"paper3_mini_{label}"
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


def _apply_decision_rule(separation: float, stability: float) -> str:
    """PASS if Separation >= 0.05 AND Separation >= 2 * Stability."""
    if separation >= 0.05 and separation >= 2.0 * stability:
        return "PASS"
    if separation > 0:
        return "BORDERLINE"
    return "FAIL"


def run_mini_selection(
    proxy_pool_path: Path,
    output_dir: Path,
    H: int = DEFAULT_H,
    S_mini: int = DEFAULT_S_MINI,
    perturbations: Optional[List[str]] = None,
    dataset: str = "BNCI2014_001",
    eval_mode: str = "CrossSession",
    subjects: Optional[List[int]] = None,
    saturation_file: str = DEFAULT_SATURATION,
    target_snr_db: float = -5.0,
    ar1_rho: float = 0.95,
    python_exe: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run perturbation mini selection: 4 topologies, S_mini seeds, evaluate perturbations.
    """
    perturbations = perturbations or DEFAULT_PERTURBATIONS
    if subjects is None:
        subjects = list(range(1, 10))
    python_exe = python_exe or sys.executable

    output_dir.mkdir(parents=True, exist_ok=True)
    if not proxy_pool_path.exists():
        raise FileNotFoundError(f"Proxy pool not found: {proxy_pool_path}. Run Experiment 1 first.")

    df = pd.read_csv(proxy_pool_path)
    best_rec, worst_rec, mid_rec = _select_topologies(df, H)

    pilot_dir = output_dir / "mini_pilot"
    selected_dir = pilot_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    best_model = _write_ws_flex_arch(best_rec, "T_best", selected_dir, H)
    worst_model = _write_ws_flex_arch(worst_rec, "T_worst", selected_dir, H)
    mid_model = _write_ws_flex_arch(mid_rec, "T_mid", selected_dir, H)

    dense_model = "paper3_mini_dense"
    dense_arch = _make_dense_arch(H, dense_model)
    (selected_dir / f"{dense_model}.json").write_text(json.dumps(dense_arch, indent=2))

    model_labels = {
        best_model: "T_best",
        worst_model: "T_worst",
        mid_model: "T_mid",
        dense_model: "B_dense",
    }
    all_models = [best_model, worst_model, mid_model, dense_model]

    manifest = {
        "T_best": {"model": best_model, "te_hat": float(best_rec["te_hat"]), "orc_hat": float(best_rec["orc_hat"]), "s": float(best_rec["s"])},
        "T_worst": {"model": worst_model, "te_hat": float(worst_rec["te_hat"]), "orc_hat": float(worst_rec["orc_hat"]), "s": float(worst_rec["s"])},
        "T_mid": {"model": mid_model, "te_hat": float(mid_rec["te_hat"]), "orc_hat": float(mid_rec["orc_hat"]), "s": float(mid_rec["s"])},
        "B_dense": {"model": dense_model},
        "H": H,
        "S_mini": S_mini,
        "perturbations": perturbations,
        "alpha_grid": ALPHA_GRID_MINI,
        "dataset": dataset,
        "eval_mode": eval_mode,
        "target_snr_db": target_snr_db,
        "ar1_rho": ar1_rho,
    }
    (output_dir / "mini_selection_manifest.json").write_text(json.dumps(manifest, indent=2))

    config_snapshot = {
        "proxy_pool": str(proxy_pool_path),
        **manifest,
    }
    with open(output_dir / "mini_selection_config.yaml", "w") as f:
        f.write("# Reproducibility config snapshot\n")
        for k, v in config_snapshot.items():
            f.write(f"{k}: {repr(v)}\n")

    if dry_run:
        print("[MiniSelection] Dry-run: would train 4 models for S_mini seeds and evaluate perturbations.")
        return manifest

    pert_params = {"ar1_drift": {"rho": ar1_rho}}
    seeds = list(range(42, 42 + S_mini))
    sat_path = str(_REPO_ROOT / saturation_file) if not Path(saturation_file).is_absolute() else saturation_file

    failed = []
    for model_name in all_models:
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
                saturation_file=sat_path,
                alpha_grid=ALPHA_GRID_MINI,
                perturbation_types=perturbations,
                target_snr_db=target_snr_db,
                perturbation_params=pert_params,
                overwrite=overwrite,
            )
            if rc != 0:
                failed.append({"model": model_name, "seed": seed})

    if failed:
        (output_dir / "failed_jobs.json").write_text(json.dumps(failed, indent=2))
        print(f"[MiniSelection] {len(failed)} jobs failed.")
        return manifest

    # F3: Collect results per (model, seed, perturbation)
    results_by_model_seed_pert: Dict[Tuple[str, int, str], Dict] = {}
    for model_name in all_models:
        for seed in seeds:
            for pert in perturbations:
                r = _collect_perturb_results(_REPO_ROOT, dataset, model_name, seed, pert)
                if r:
                    results_by_model_seed_pert[(model_name, seed, pert)] = r

    # F4: Compute Separation, Stability, mean RD_max per topology
    summary_rows = []
    curves_rows = []

    for pert in perturbations:
        rd_by_model_seed: Dict[str, List[float]] = {m: [] for m in all_models}
        for model_name in all_models:
            for seed in seeds:
                key = (model_name, seed, pert)
                if key in results_by_model_seed_pert:
                    r = results_by_model_seed_pert[key]
                    rd = r.get("RD_max")
                    if rd is not None and np.isfinite(rd):
                        rd_by_model_seed[model_name].append(rd)
                    for i, (inten, rt) in enumerate(zip(r.get("intensities", []), r.get("r_t", []))):
                        curves_rows.append({
                            "model": model_labels.get(model_name, model_name),
                            "model_id": model_name,
                            "seed": seed,
                            "perturbation": pert,
                            "intensity_idx": i,
                            "intensity": inten,
                            "r_t": rt,
                            "RD_max": r.get("RD_max"),
                            "clean_roc_auc": r.get("clean_roc_auc"),
                        })

        rd_best = rd_by_model_seed.get(best_model, [])
        rd_worst = rd_by_model_seed.get(worst_model, [])
        rd_mid = rd_by_model_seed.get(mid_model, [])
        rd_dense = rd_by_model_seed.get(dense_model, [])

        separation_per_seed = []
        for seed in seeds:
            rd_w = [
                r["RD_max"] for (m, s, p), r in results_by_model_seed_pert.items()
                if m == worst_model and s == seed and p == pert and np.isfinite(r.get("RD_max", np.nan))
            ]
            rd_b = [
                r["RD_max"] for (m, s, p), r in results_by_model_seed_pert.items()
                if m == best_model and s == seed and p == pert and np.isfinite(r.get("RD_max", np.nan))
            ]
            if rd_w and rd_b:
                separation_per_seed.append(float(np.mean(rd_w)) - float(np.mean(rd_b)))
        separation = float(np.mean(separation_per_seed)) if separation_per_seed else float("nan")

        all_rd = []
        for m in all_models:
            all_rd.extend(rd_by_model_seed.get(m, []))
        stability = float(np.std(all_rd)) if len(all_rd) >= 2 else float("nan")

        tag = _apply_decision_rule(separation, stability)

        summary_rows.append({
            "perturbation": pert,
            "Separation": separation,
            "Stability": stability,
            "mean_RD_max_T_best": float(np.mean(rd_best)) if rd_best else float("nan"),
            "mean_RD_max_T_worst": float(np.mean(rd_worst)) if rd_worst else float("nan"),
            "mean_RD_max_T_mid": float(np.mean(rd_mid)) if rd_mid else float("nan"),
            "mean_RD_max_B_dense": float(np.mean(rd_dense)) if rd_dense else float("nan"),
            "tag": tag,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "mini_selection_summary_table.csv", index=False)

    with open(output_dir / "mini_selection_curves.csv", "w", newline="") as f:
        if curves_rows:
            w = csv.DictWriter(f, fieldnames=curves_rows[0].keys())
            w.writeheader()
            w.writerows(curves_rows)

    for row in summary_rows:
        print(f"[MiniSelection] {row['perturbation']}: Separation={row['Separation']:.4f}, Stability={row['Stability']:.4f}, tag={row['tag']}")

    return {**manifest, "summary": summary_rows}


def _plot_figure_p_mini_1(output_dir: Path) -> None:
    """Generate Figure P-mini-1: r_t vs intensity for each perturbation."""
    curves_path = output_dir / "mini_selection_curves.csv"
    if not curves_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[MiniSelection] matplotlib not available, skipping figure.")
        return

    df = pd.read_csv(curves_path)
    if df.empty:
        return

    perturbations = df["perturbation"].unique()
    n_pert = len(perturbations)
    n_cols = 2
    n_rows = max(1, (n_pert + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, pert in enumerate(perturbations):
        ax = axes[idx]
        sub = df[df["perturbation"] == pert]
        for model_label in ["T_best", "T_worst", "T_mid", "B_dense"]:
            msub = sub[sub["model"] == model_label]
            if msub.empty:
                continue
            agg = msub.groupby("intensity_idx").agg({"r_t": "mean", "intensity": "first"}).reset_index()
            agg = agg.sort_values("intensity_idx")
            style = "o-" if model_label in ("T_best", "T_worst") else "s--"
            lw = 2 if model_label in ("T_best", "T_worst") else 1
            ax.plot(agg["intensity_idx"], agg["r_t"], style, label=model_label, linewidth=lw)
        ax.set_xlabel("Intensity t")
        ax.set_ylabel("r_t")
        ax.set_title(pert)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(n_pert, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "figure_p_mini_1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[MiniSelection] Saved {output_dir / 'figure_p_mini_1.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Perturbation Selection Mini Experiment: rapid screening for Plot 2/3 perturbation choice.",
        epilog="If none pass: increase --S-mini from 2 to 3, or add intensity level t=4, then rerun.",
    )
    parser.add_argument("--proxy-pool", type=str, required=True, help="Path to proxy_pool.csv (from Experiment 1)")
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/perturbation_mini_selection")
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--S-mini", type=int, default=DEFAULT_S_MINI, help="Seeds per topology (2 or 3)")
    parser.add_argument("--perturbations", type=str, default=",".join(DEFAULT_PERTURBATIONS),
                        help="Comma-separated: ar1_drift,gaussian (optionally dropout,eog)")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval-mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--saturation-file", type=str, default=DEFAULT_SATURATION)
    parser.add_argument("--target-snr-db", type=float, default=-5.0)
    parser.add_argument("--ar1-rho", type=float, default=0.95)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating Figure P-mini-1")
    args = parser.parse_args()

    proxy_path = Path(args.proxy_pool)
    if not proxy_path.is_absolute():
        proxy_path = _REPO_ROOT / proxy_path
    out_dir = _REPO_ROOT / args.output_dir
    perturbations = [x.strip() for x in args.perturbations.split(",") if x.strip()]

    run_mini_selection(
        proxy_pool_path=proxy_path,
        output_dir=out_dir,
        H=args.H,
        S_mini=args.S_mini,
        perturbations=perturbations,
        dataset=args.dataset,
        eval_mode=args.eval_mode,
        subjects=args.subjects,
        saturation_file=args.saturation_file,
        target_snr_db=args.target_snr_db,
        ar1_rho=args.ar1_rho,
        python_exe=args.python,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if not args.dry_run and not args.no_plot:
        _plot_figure_p_mini_1(out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
