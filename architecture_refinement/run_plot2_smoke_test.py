"""
Phase 5: Minimal training smoke test (Plot_2_Investigation.txt).

Choose 4 graphs (one per regime), train with 1 seed and shortened protocol if possible,
evaluate on AR(1) drift at target_snr_db = -5. Success: at least one pair of regimes
differs in robustness (e.g. ΔAUPC >= 0.02).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from architecture_refinement.plot2_smoke_utils import compute_smoke_robustness_metrics, summarize_wiring_mask

# One k per regime (super_sparse, sparse, moderate, near_dense) — spec experiment_three: 2, 6, 14, 26
REGIME_K = {"super_sparse": 2, "sparse": 6, "moderate": 14, "near_dense": 26}


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _undirected_hidden_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A != 0).astype(np.int8)
    return A


def _oriented_hidden_adj(G: nx.Graph, H: int, seed: int) -> np.ndarray:
    wiring = WsFlexHiddenWiring(
        input_size=1, hidden_graph=G, output_size=1,
        hidden_edge_orientation="random_oriented", seed=int(seed),
    )
    A = wiring._hidden_block_oriented()
    return (np.asarray(A) != 0).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2 Phase 5: minimal training smoke test")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval_mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=[1], help="Minimal subjects for speed (default: 1)")
    parser.add_argument("--seed", type=int, default=202605)
    parser.add_argument("--target_snr_db", type=float, default=-5.0)
    parser.add_argument("--saturation_file", type=str, default="saturation_results/saturation_points_summary.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="Smoke test output dir (default: architecture_refinement/outputs/plot2_smoke)")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--delta_threshold", type=float, default=0.02, help="Success: max |AUPC_i - AUPC_j| >= this")
    parser.add_argument("--dry_run", action="store_true", help="Only write architectures and print commands")
    args = parser.parse_args()

    H = max(2, int(args.H))
    out_dir = Path(args.out_dir) if args.out_dir else _THIS_DIR / "outputs" / "plot2_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = out_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    run_id = "smoke"
    model_names: List[str] = []
    selected_rows: List[Dict[str, Any]] = []
    for regime, k in REGIME_K.items():
        p = 0.3
        graph_seed = hash((args.seed, regime)) % (2**31 - 1)
        wiring_seed = graph_seed + 1
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            # retry with higher p
            for p in [0.5, 0.7, 1.0]:
                G = _make_ws_graph(H, k, p, seed=graph_seed + 1000)
                if nx.is_connected(G):
                    break
            else:
                raise RuntimeError(f"Could not generate connected graph for regime {regime} k={k}")
        undirected_adj = _undirected_hidden_adj(G, H)
        directed_adj = _oriented_hidden_adj(G, H, seed=wiring_seed)
        model_name = f"plot2_{run_id}_{regime}_b1"
        model_names.append(model_name)
        arch = {
            "schema_version": 2,
            "run_id": run_id,
            "method": "smoke",
            "rank": 1,
            "model_name": model_name,
            "H": H,
            "wiring_kind": "ws_flex",
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "wiring_seed": wiring_seed,
            "hidden_adj_undirected": undirected_adj.tolist(),
            "hidden_adj_directed": directed_adj.tolist(),
        }
        with open(selected_dir / f"{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=2)
        # Row for selected_architectures.csv (analyzer expects model_name, method, k, etc.)
        mean_degree_undirected = int(k)  # WS undirected mean degree = k
        selected_rows.append({
            "model_name": model_name,
            "method": "smoke",
            "rank": 1,
            "wiring_kind": "ws_flex",
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "wiring_seed": wiring_seed,
            "mean_degree_undirected": mean_degree_undirected,
        })

    # Write manifest and selected_architectures.csv so analyzer and integrity checks can use them
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Mask statistics (B2/C2): per-model JSON and dict for report
    mask_stats_by_model: Dict[str, Dict[str, Any]] = {}
    for model_name in model_names:
        arch_path = selected_dir / f"{model_name}.json"
        if arch_path.exists():
            with open(arch_path, "r", encoding="utf-8") as f:
                arch = json.load(f)
            stats = summarize_wiring_mask(arch)
            mask_stats_by_model[model_name] = stats
            with open(diagnostics_dir / f"{model_name}_mask_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

    manifest = {
        "dataset": args.dataset,
        "primary_perturbation_type": "ar1_drift",
        "perturbation_types": ["ar1_drift"],
        "saturation_file": args.saturation_file,
        "target_snr_db": args.target_snr_db,
    }
    with open(out_dir / "plot2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    sel_csv_path = out_dir / "selected_architectures.csv"
    if selected_rows:
        fieldnames = list(selected_rows[0].keys())
        with open(sel_csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(selected_rows)
    print(f"Wrote 4 architectures to {selected_dir}")
    if args.dry_run:
        print("Dry run: would invoke unified_experiment_runner for each model. Exit.")
        return

    repo_root = _REPO_ROOT
    runner_script = repo_root / "evaluation" / "unified_experiment_runner.py"
    for model_name in model_names:
        cmd = [
            args.python,
            str(runner_script),
            "--nas_pilot_dir", str(out_dir),
            "--model", model_name,
            "--dataset", args.dataset,
            "--subjects", *[str(s) for s in args.subjects],
            "--mode", "test_perturb",
            "--eval_mode", args.eval_mode,
            "--seed", str(args.seed),
            "--test_perturb_noise_types", "ar1_drift",
            "--test_perturb_target_snr_db", str(args.target_snr_db),
            "--noise_perturbation_saturation_file", args.saturation_file,
            "--test_perturb_gaussian_alpha_grid", "0,0.25,0.5,0.75,1.0",
            "--plot2_diagnostics_dir", str(diagnostics_dir),
            "--overwrite",
        ]
        print(f"[SMOKE] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            print(f"[SMOKE] Runner failed for {model_name}: {proc.stderr[:500]}")

    # Invoke analyzer to compute AUPC and clean ROC-AUC from runner CSVs
    analyzer_script = _THIS_DIR / "analyze_plot2_results.py"
    analyze_cmd = [
        args.python,
        str(analyzer_script),
        "--plot2_dir", str(out_dir),
        "--repo_root", str(repo_root),
    ]
    print(f"[SMOKE] Running analyzer: {' '.join(analyze_cmd)}")
    proc_analyze = subprocess.run(analyze_cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=600)
    if proc_analyze.returncode != 0:
        print(f"[SMOKE] Analyzer failed: {proc_analyze.stderr[:1000]}")

    # Read per_graph_aupc.csv and compute robustness metrics (B1 / C1)
    analysis_dir = out_dir / "analysis"
    per_graph_path = analysis_dir / "per_graph_aupc.csv"
    results_by_model: Dict[str, float] = {}
    clean_roc_auc_by_model: Dict[str, float] = {}
    max_drop_by_model: Dict[str, float] = {}
    mid_drop_by_model: Dict[str, float] = {}
    max_pairwise_delta_AUPC = 0.0
    max_pairwise_delta_max_drop = 0.0
    max_pairwise_delta_mid_drop = 0.0
    if per_graph_path.exists():
        per_graph_df = pd.read_csv(per_graph_path)
        metrics = compute_smoke_robustness_metrics(
            per_graph_df, model_names, primary_noise_type="ar1_drift"
        )
        results_by_model = metrics["aupc_by_model"]
        clean_roc_auc_by_model = metrics["clean_roc_auc_by_model"]
        max_drop_by_model = metrics["max_drop_by_model"]
        mid_drop_by_model = metrics["mid_drop_by_model"]
        max_pairwise_delta_AUPC = metrics["max_pairwise_delta_AUPC"]
        max_pairwise_delta_max_drop = metrics["max_pairwise_delta_max_drop"]
        max_pairwise_delta_mid_drop = metrics["max_pairwise_delta_mid_drop"]
    for mn in model_names:
        if mn not in results_by_model:
            results_by_model[mn] = float("nan")
        if mn not in clean_roc_auc_by_model:
            clean_roc_auc_by_model[mn] = float("nan")
        if mn not in max_drop_by_model:
            max_drop_by_model[mn] = float("nan")
        if mn not in mid_drop_by_model:
            mid_drop_by_model[mn] = float("nan")

    # Primary gate: max_pairwise_delta >= delta_threshold for ANY of AUPC, max_drop, mid_drop
    primary_gate_pass = (
        max_pairwise_delta_AUPC >= args.delta_threshold
        or max_pairwise_delta_max_drop >= args.delta_threshold
        or max_pairwise_delta_mid_drop >= args.delta_threshold
    )
    any_nan = (
        any(not np.isfinite(results_by_model[m]) for m in model_names)
        or any(not np.isfinite(max_drop_by_model.get(m, float("nan"))) for m in model_names)
        or any(not np.isfinite(mid_drop_by_model.get(m, float("nan"))) for m in model_names)
    )
    success = not any_nan and primary_gate_pass

    # Secondary: mask_density differs across regimes by >= 2x
    densities = [
        mask_stats_by_model.get(m, {}).get("mask_density", 0.0)
        for m in model_names
        if isinstance(mask_stats_by_model.get(m), dict)
    ]
    densities = [d for d in densities if np.isfinite(d) and d > 0]
    if len(densities) >= 2:
        mask_density_ratio = float(max(densities) / min(densities))
        secondary_mask_density_pass = mask_density_ratio >= 2.0
    else:
        mask_density_ratio = float("nan")
        secondary_mask_density_pass = False

    # B3: Perturbation fingerprint (lag-1 ~ rho for AR(1))
    fp_path = diagnostics_dir / "perturbation_fingerprint.json"
    perturbation_fingerprint: Dict[str, Any] = {"path": str(fp_path), "exists": fp_path.exists()}
    secondary_fingerprint_pass = False
    if fp_path.exists():
        try:
            fp_data = json.loads(fp_path.read_text(encoding="utf-8"))
            perturbation_fingerprint["lag1_autocorrelation"] = fp_data.get("lag1_autocorrelation")
            lag1 = fp_data.get("lag1_autocorrelation")
            if lag1 is not None and np.isfinite(lag1):
                # AR(1) rho default 0.97; pass if lag-1 is close (e.g. within 0.15)
                secondary_fingerprint_pass = abs(float(lag1) - 0.97) < 0.15
        except Exception:
            pass

    secondary_gates_pass = secondary_mask_density_pass and secondary_fingerprint_pass

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "models": model_names,
        "results_by_model": results_by_model,
        "clean_roc_auc_by_model": clean_roc_auc_by_model,
        "max_drop_by_model": max_drop_by_model,
        "mid_drop_by_model": mid_drop_by_model,
        "max_pairwise_delta_AUPC": max_pairwise_delta_AUPC,
        "max_pairwise_delta_max_drop": max_pairwise_delta_max_drop,
        "max_pairwise_delta_mid_drop": max_pairwise_delta_mid_drop,
        "delta_threshold": args.delta_threshold,
        "success": success,
        "primary_gate_pass": primary_gate_pass,
        "secondary_mask_density_pass": secondary_mask_density_pass,
        "secondary_fingerprint_pass": secondary_fingerprint_pass,
        "secondary_gates_pass": secondary_gates_pass,
        "mask_density_ratio": mask_density_ratio,
        "mask_stats_by_model": mask_stats_by_model,
        "perturbation_fingerprint": perturbation_fingerprint,
    }
    with open(out_dir / "smoke_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Smoke test report written to {out_dir / 'smoke_test_report.json'}")
    if any_nan:
        print("[SMOKE] FAIL: at least one required metric (AUPC/max_drop/mid_drop) is NaN or missing.")
        sys.exit(1)
    if success:
        print(
            f"Success: primary_gate_pass=True (max_pairwise_delta >= {args.delta_threshold} for at least one metric)."
        )
    else:
        print(
            f"Success=False: primary_gate_pass=False (no metric has max_pairwise_delta >= {args.delta_threshold})."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
