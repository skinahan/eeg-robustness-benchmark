"""
Plot 2 Perturbation Pilot – minimal test run per plot2_pilot_spec.txt.

One fixed CNN-CfC (H=32, k=12, p=0.4), one seed, BNCI2014_001 cross-session only.
Evaluates spatial_gaussian, ar1_drift, emg_band on 5-point alpha grid with SNR-based alpha_max,
writes results to results/plot2_perturbation_pilot/<model_id>/seed0/ and a summary table.
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
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import results_paradigm_folder, short_run_id

# Pilot defaults from spec
MODEL_ID = "plot2_pilot_cnn_cfc"
H = 32
K = 12
P = 0.4
GRAPH_SEED = 42
WIRING_SEED = 123
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PERTURBATION_TYPES = ["spatial_gaussian", "ar1_drift", "emg_band"]


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _undirected_hidden_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A != 0).astype(np.int8)
    if A.shape != (H, H):
        raise ValueError(f"Unexpected shape {A.shape} (expected {(H, H)})")
    return A


def _oriented_hidden_adj(G: nx.Graph, H: int, seed: int) -> np.ndarray:
    from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
    wiring = WsFlexHiddenWiring(
        input_size=1,
        hidden_graph=G,
        output_size=1,
        hidden_edge_orientation="random_oriented",
        seed=int(seed),
    )
    A = wiring._hidden_block_oriented()
    A = (np.asarray(A) != 0).astype(np.int8)
    if A.shape != (H, H):
        raise ValueError(f"Unexpected shape {A.shape} (expected {(H, H)})")
    return A


def _aupc(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoidal area under curve (intensity vs metric)."""
    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs = np.asarray(xs, dtype=float)[order]
    ys = np.asarray(ys, dtype=float)[order]
    try:
        area = np.trapezoid(y=ys, x=xs)
    except AttributeError:
        area = np.trapz(y=ys, x=xs)
    return float(area)


def create_pilot_architecture(
    pilot_dir: Path,
    model_id: str = MODEL_ID,
    H: int = H,
    k: int = K,
    p: float = P,
    graph_seed: int = GRAPH_SEED,
    wiring_seed: int = WIRING_SEED,
) -> Path:
    """Build one WS-Flex architecture and write JSON to pilot_dir/selected_architectures/."""
    G = _make_ws_graph(H, k, p, seed=graph_seed)
    if not nx.is_connected(G):
        raise RuntimeError("Watts-Strogatz graph is not connected.")
    undirected_adj = _undirected_hidden_adj(G, H)
    directed_adj = _oriented_hidden_adj(G, H, wiring_seed)
    arch_dir = pilot_dir / "selected_architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    arch_path = arch_dir / f"{model_id}.json"
    arch = {
        "schema_version": 2,
        "model_name": model_id,
        "wiring_kind": "ws_flex",
        "H": H,
        "k": int(k),
        "p": float(p),
        "graph_seed": int(graph_seed),
        "wiring_seed": int(wiring_seed),
        "hidden_adj_undirected": undirected_adj.tolist(),
        "hidden_adj_directed": directed_adj.tolist(),
    }
    arch_path.write_text(json.dumps(arch, indent=2), encoding="utf-8")
    return arch_path


def run_unified_runner(
    repo_root: Path,
    pilot_dir: Path,
    model_id: str,
    dataset: str,
    subjects: List[int],
    seed: int,
    python_exe: str,
    overwrite: bool = False,
    saturation_file: str = "saturation_results/saturation_points_summary.csv",
    target_snr_db: float = 0.0,
    spatial_ell_multiplier: float = 1.0,
    emg_f_high: float = 80.0,
    emg_use_envelope: bool = False,
) -> int:
    """Invoke unified_experiment_runner with test_perturb for the pilot model."""
    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").resolve()),
        "--nas_pilot_dir", str(pilot_dir.resolve()),
        "--model", model_id,
        "--dataset", dataset,
        "--subjects", *[str(s) for s in subjects],
        "--mode", "test_perturb",
        "--eval_mode", "CrossSession",
        "--seed", str(seed),
        "--disable_underfitting_retrain",
        "--noise_perturbation_saturation_file", saturation_file,
        "--noise_perturbation_num_steps", "20",
        "--test_perturb_gaussian_alpha_grid", ",".join(str(a) for a in ALPHA_GRID),
        "--test_perturb_noise_types", ",".join(PERTURBATION_TYPES),
        "--test_perturb_target_snr_db=" + str(target_snr_db),
        "--test_perturb_spatial_ell_multiplier", str(spatial_ell_multiplier),
        "--test_perturb_emg_f_high", str(emg_f_high),
    ]
    if emg_use_envelope:
        cmd.append("--test_perturb_emg_use_envelope")
    if overwrite:
        cmd.append("--overwrite")
    print("[PILOT] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def _results_base_path(repo_root: Path, dataset: str, model_id: str, seed: int) -> Path:
    """Base path for unified runner results (Paradigm/Dataset/Model/...). Prefers short run id (new layout)."""
    paradigm = results_paradigm_folder(dataset)
    base_short = Path(repo_root) / "results" / paradigm / dataset / short_run_id(model_id) / "CrossSessionEvaluation" / str(seed)
    base_long = Path(repo_root) / "results" / paradigm / dataset / model_id / "CrossSessionEvaluation" / str(seed)
    return base_short if base_short.exists() else base_long


def collect_raw_results(
    repo_root: Path,
    dataset: str,
    model_id: str,
    seed: int,
) -> pd.DataFrame:
    """Glob all test_perturb CSVs for this model/seed and concatenate (checks short and long path)."""
    paradigm = results_paradigm_folder(dataset)
    base_short = Path(repo_root) / "results" / paradigm / dataset / short_run_id(model_id) / "CrossSessionEvaluation" / str(seed)
    base_long = Path(repo_root) / "results" / paradigm / dataset / model_id / "CrossSessionEvaluation" / str(seed)
    csvs = []
    for base in (base_short, base_long):
        if base.exists():
            csvs = list(base.rglob("*.csv"))
            if csvs:
                break
    if not csvs:
        return pd.DataFrame()
    # Only CSVs under a test_perturb (or test_perturb_tune) directory
    dfs = []
    for p in csvs:
        if "test_perturb" in str(p):
            try:
                df = pd.read_csv(p)
                if "noise_type" in df.columns and "corrupted_roc_auc" in df.columns:
                    dfs.append(df)
            except Exception as e:
                print(f"[PILOT] Warning: could not read {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def aggregate_and_compute_metrics(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Aggregate raw results by noise_type and intensity; compute per-perturbation
    mean ROC-AUC curve, delta_ROC_AUC, AUPC, Spearman. Return (per-intensity table, summary rows).
    """
    from scipy.stats import spearmanr
    summary_rows = []
    for nt in PERTURBATION_TYPES:
        sub = raw[raw["noise_type"].astype(str) == nt]
        if sub.empty:
            summary_rows.append({
                "Perturbation": nt,
                "target_snr_db": float("nan"),
                "empirical_snr_db": float("nan"),
                "ROC_AUC_clean": float("nan"),
                "ROC_AUC_at_max": float("nan"),
                "Delta_ROC_AUC": float("nan"),
                "AUPC": float("nan"),
                "Spearman": float("nan"),
            })
            continue
        sub = sub.copy()
        # Read target/empirical SNR from runner output if present (Spec 3 PATCH 1)
        target_snr = float(sub["target_snr_db"].iloc[0]) if "target_snr_db" in sub.columns and sub["target_snr_db"].notna().any() else float("nan")
        empirical_snr = float(sub["empirical_snr_db"].iloc[0]) if "empirical_snr_db" in sub.columns and sub["empirical_snr_db"].notna().any() else float("nan")
        sub["intensity"] = pd.to_numeric(sub["intensity"], errors="coerce")
        sub["alpha"] = pd.to_numeric(sub.get("alpha", np.nan), errors="coerce")
        sub["corrupted_roc_auc"] = pd.to_numeric(sub["corrupted_roc_auc"], errors="coerce")
        sub["clean_roc_auc"] = pd.to_numeric(sub.get("clean_roc_auc", sub.get("clean_score", np.nan)), errors="coerce")
        agg = sub.groupby("intensity", as_index=False).agg({
            "corrupted_roc_auc": "mean",
            "clean_roc_auc": "mean",
            "alpha": "first",
        }).sort_values("intensity")
        if agg.empty or len(agg) < 2:
            summary_rows.append({
                "Perturbation": nt,
                "target_snr_db": target_snr,
                "empirical_snr_db": empirical_snr,
                "ROC_AUC_clean": float("nan"),
                "ROC_AUC_at_max": float("nan"),
                "Delta_ROC_AUC": float("nan"),
                "AUPC": float("nan"),
                "Spearman": float("nan"),
            })
            continue
        intensities = agg["intensity"].to_numpy(dtype=float)
        roc_aucs = agg["corrupted_roc_auc"].to_numpy(dtype=float)
        clean_auc = float(agg["clean_roc_auc"].iloc[0]) if "clean_roc_auc" in agg.columns else float(roc_aucs[0])
        alpha_vals = agg["alpha"].to_numpy(dtype=float) if "alpha" in agg.columns and agg["alpha"].notna().any() else None
        # alpha_max: max intensity in this run
        alpha_max = float(np.nanmax(intensities)) if np.isfinite(intensities).any() else 1.0
        if alpha_max <= 0:
            alpha_max = 1.0
        # Normalized alpha for AUPC (0..1)
        if alpha_vals is not None and np.isfinite(alpha_vals).any():
            alphas = alpha_vals
        else:
            alphas = intensities / alpha_max if alpha_max > 0 else intensities
        roc_at_max = float(roc_aucs[-1]) if len(roc_aucs) else float("nan")
        delta = float(clean_auc - roc_at_max) if np.isfinite(clean_auc) and np.isfinite(roc_at_max) else float("nan")
        aupc_sigma = _aupc(intensities, roc_aucs)
        aupc_alpha = float(aupc_sigma / alpha_max) if alpha_max > 0 and np.isfinite(aupc_sigma) else float("nan")
        if len(alphas) >= 2 and np.isfinite(roc_aucs).all():
            rho, _ = spearmanr(alphas, roc_aucs)
            spearman = float(rho) if np.isfinite(rho) else float("nan")
        else:
            spearman = float("nan")
        summary_rows.append({
            "Perturbation": nt,
            "target_snr_db": target_snr,
            "empirical_snr_db": empirical_snr,
            "ROC_AUC_clean": clean_auc,
            "ROC_AUC_at_max": roc_at_max,
            "Delta_ROC_AUC": delta,
            "AUPC": aupc_alpha,
            "Spearman": spearman,
        })
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, summary_rows


def _get_git_commit(repo_root: Path) -> str:
    """Return git HEAD commit hash or 'unknown' (Spec 3 PATCH 4)."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_root),
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_pilot_outputs(
    output_root: Path,
    model_id: str,
    seed: int,
    raw: pd.DataFrame,
    summary_rows: List[Dict[str, Any]],
    target_snr_db: float = 0.0,
    dataset: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """Write per-perturbation CSVs and summary table under output_root/<model_id>/seed0/. Spec 3 PATCH 4: also write run_manifest.json."""
    out_base = output_root / model_id / f"seed{seed}"
    out_base.mkdir(parents=True, exist_ok=True)
    for nt in PERTURBATION_TYPES:
        sub = raw[raw["noise_type"].astype(str) == nt]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["intensity"] = pd.to_numeric(sub["intensity"], errors="coerce")
        sub["corrupted_roc_auc"] = pd.to_numeric(sub["corrupted_roc_auc"], errors="coerce")
        agg = sub.groupby("intensity", as_index=False).agg({
            "corrupted_roc_auc": "mean",
            "clean_roc_auc": "mean",
            "alpha": "first",
        }).sort_values("intensity")
        if "clean_roc_auc" not in agg.columns and "clean_score" in sub.columns:
            agg["clean_roc_auc"] = pd.to_numeric(sub["clean_score"], errors="coerce").mean()
        alpha_max = float(agg["intensity"].max()) if len(agg) and agg["intensity"].notna().any() else 1.0
        if alpha_max <= 0:
            alpha_max = 1.0
        if agg["alpha"].isna().all() or agg["alpha"].isna().any():
            agg["alpha"] = (agg["intensity"] / alpha_max).to_numpy()
        agg["roc_auc"] = agg["corrupted_roc_auc"]
        pert_dir = out_base / nt
        pert_dir.mkdir(parents=True, exist_ok=True)
        agg.to_csv(pert_dir / "results.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_base / "summary.csv", index=False)
    # Human-readable summary (Spec 3 PATCH 1: target_snr_db, empirical_snr_db)
    lines = [
        "Plot 2 Perturbation Pilot – Summary",
        "====================================",
        "",
        "| Perturbation | target_snr_db | empirical_snr_db | ROC-AUC clean | ROC-AUC at max | ΔROC-AUC | AUPC | Spearman |",
        "|-------------|---------------|------------------|---------------|----------------|----------|------|----------|",
    ]
    for r in summary_rows:
        t_snr = r.get("target_snr_db", float("nan"))
        e_snr = r.get("empirical_snr_db", float("nan"))
        t_str = f"{t_snr:.2f}" if np.isfinite(t_snr) else "—"
        e_str = f"{e_snr:.2f}" if np.isfinite(e_snr) else "—"
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                r["Perturbation"],
                t_str,
                e_str,
                r["ROC_AUC_clean"],
                r["ROC_AUC_at_max"],
                r["Delta_ROC_AUC"],
                r["AUPC"],
                r["Spearman"],
            )
        )
    (out_base / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Spec 3 PATCH 4: Run manifest (git, dataset, model, perturbation, SNR, seeds)
    run_manifest = {
        "git_commit": _get_git_commit(repo_root or _REPO_ROOT),
        "dataset": None,
        "model_id": model_id,
        "seed": int(seed),
        "perturbation_types": list(PERTURBATION_TYPES),
        "target_snr_db": float(target_snr_db),
        "summary_snr": {r["Perturbation"]: {"target_snr_db": r.get("target_snr_db"), "empirical_snr_db": r.get("empirical_snr_db")} for r in summary_rows},
        "rng_seed": int(seed),
    }
    (out_base / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print(f"[PILOT] Wrote {out_base / 'summary.csv'} and {out_base / 'summary.txt'}")
    for nt in PERTURBATION_TYPES:
        d = out_base / nt / "results.csv"
        if d.exists():
            print(f"[PILOT] Wrote {d}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot 2 perturbation pilot: one model, one seed, three correlated perturbations."
    )
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=int, nargs="*", default=[1, 2],
                        help="Subject list for minimal run (default: 1 2). Use 1-9 for full pilot.")
    parser.add_argument("--output_root", type=str, default="results/plot2_perturbation_pilot",
                        help="Root for pilot outputs: <output_root>/<model_id>/seed<N>/")
    parser.add_argument("--pilot_dir", type=str, default=None,
                        help="Directory for pilot arch JSON (default: output_root/pilot_manifests/<model_id>)")
    parser.add_argument("--saturation_file", type=str, default="saturation_results/saturation_points_summary.csv")
    parser.add_argument("--target_snr_db", type=float, default=0.0,
                        help="Target SNR in dB at alpha_max (0 = default; -5 for escalation Step 1).")
    parser.add_argument("--spatial_ell_multiplier", type=float, default=1.0,
                        help="Multiplier for spatial correlation length (spatial_gaussian). Use 2.0 for Step 2 escalation.")
    parser.add_argument("--emg_f_high", type=float, default=80.0,
                        help="EMG band high cutoff in Hz. Use 100 for Step 2 escalation.")
    parser.add_argument("--emg_use_envelope", action="store_true",
                        help="Apply slow amplitude envelope to EMG noise (Step 2 escalation).")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_run", action="store_true",
                        help="Skip calling unified runner; only aggregate existing results into pilot output.")
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    output_root = Path(args.output_root)
    if args.pilot_dir:
        pilot_dir = Path(args.pilot_dir)
    else:
        pilot_dir = output_root / "pilot_manifests" / MODEL_ID
    pilot_dir = pilot_dir.resolve()
    output_root = output_root.resolve()

    if not args.skip_run:
        create_pilot_architecture(
            pilot_dir,
            model_id=MODEL_ID,
            H=H,
            k=K,
            p=P,
            graph_seed=GRAPH_SEED,
            wiring_seed=WIRING_SEED,
        )
        rc = run_unified_runner(
            repo_root=repo_root,
            pilot_dir=pilot_dir,
            model_id=MODEL_ID,
            dataset=args.dataset,
            subjects=args.subjects,
            seed=args.seed,
            python_exe=args.python,
            overwrite=args.overwrite,
            saturation_file=args.saturation_file,
            target_snr_db=args.target_snr_db,
            spatial_ell_multiplier=args.spatial_ell_multiplier,
            emg_f_high=args.emg_f_high,
            emg_use_envelope=args.emg_use_envelope,
        )
        if rc != 0:
            print(f"[PILOT] Unified runner exited with code {rc}")
            return rc

    raw = collect_raw_results(repo_root, args.dataset, MODEL_ID, args.seed)
    if raw.empty:
        print("[PILOT] No raw result CSVs found. Run without --skip_run first.")
        return 1
    _, summary_rows = aggregate_and_compute_metrics(raw)
    write_pilot_outputs(
        output_root, MODEL_ID, args.seed, raw, summary_rows,
        target_snr_db=args.target_snr_db,
        dataset=args.dataset,
        repo_root=repo_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
