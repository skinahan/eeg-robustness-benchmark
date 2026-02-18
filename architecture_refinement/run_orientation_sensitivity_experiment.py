#!/usr/bin/env python3
"""
Orientation Sensitivity Experiment: random_oriented vs symmetric wiring conversion.

PURPOSE
-------
Test whether using symmetric (bidirectional) wiring instead of random_oriented when
converting WS-Flex graphs to Wired CfC improves robustness. Current NAS pilot and
Plot 2 studies use random_oriented, which picks one direction per undirected edge
and may hurt performance by reducing information flow and redundancy.

HYPOTHESIS
----------
Symmetric wiring preserves the full bidirectional connectivity of WS graphs and may
yield better AUPC/robustness than random_oriented, which arbitrarily destroys half
the connections.

SCOPE (minimal)
---------------
- 2–3 fixed WS graphs (sparse, moderate, dense regimes)
- For each graph: two models — random_oriented and symmetric (same hidden_adj_undirected)
- 1 subject, 1 seed
- AR(1) drift perturbation (primary Plot 2 stress test)
- BNCI2014_001, CrossSession

OUTPUT
------
- pilot_dir/selected_architectures/*.json (paired architectures)
- pilot_dir/orientation_sensitivity_report.json (AUPC, clean ROC-AUC, max_drop by orientation)
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

# Minimal graph set: one per regime (sparse, moderate, dense)
REGIME_GRAPHS = [
    {"regime": "sparse", "k": 4, "p": 0.3},
    {"regime": "moderate", "k": 8, "p": 0.3},
    {"regime": "dense", "k": 14, "p": 0.3},
]


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _undirected_hidden_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.int8)
    return (A != 0).astype(np.int8)


def _oriented_hidden_adj(G: nx.Graph, H: int, orientation: str, seed: int) -> np.ndarray:
    wiring = WsFlexHiddenWiring(
        input_size=1,
        hidden_graph=G,
        output_size=1,
        hidden_edge_orientation=orientation,
        seed=int(seed),
    )
    A = wiring._hidden_block_oriented()
    return (np.asarray(A) != 0).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orientation sensitivity: compare random_oriented vs symmetric wiring on same WS graphs"
    )
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval_mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=[1])
    parser.add_argument("--seed", type=int, default=202618)
    parser.add_argument("--target_snr_db", type=float, default=-5.0)
    parser.add_argument("--saturation_file", type=str, default="saturation_results/saturation_points_summary.csv")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir (default: architecture_refinement/outputs/orientation_sensitivity)",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true", help="Only write architectures and print commands")
    args = parser.parse_args()

    H = max(2, int(args.H))
    out_dir = Path(args.out_dir) if args.out_dir else _THIS_DIR / "outputs" / "orientation_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = out_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    selected_rows: List[Dict[str, Any]] = []

    for cfg in REGIME_GRAPHS:
        regime = cfg["regime"]
        k = cfg["k"]
        p = cfg["p"]
        graph_seed = hash((args.seed, regime)) % (2**31 - 1)
        wiring_seed = graph_seed + 1

        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            for p_retry in [0.5, 0.7, 1.0]:
                G = _make_ws_graph(H, k, p_retry, seed=graph_seed + 1000)
                if nx.is_connected(G):
                    break
            else:
                raise RuntimeError(f"Could not generate connected graph for regime {regime} k={k}")

        undirected_adj = _undirected_hidden_adj(G, H)

        for orientation in ("random_oriented", "symmetric"):
            directed_adj = _oriented_hidden_adj(G, H, orientation, seed=wiring_seed)
            model_name = f"orient_{regime}_{orientation}"
            model_names.append(model_name)
            arch = {
                "schema_version": 2,
                "run_id": "orientation_sensitivity",
                "method": "orientation_sensitivity",
                "rank": 1,
                "model_name": model_name,
                "H": H,
                "wiring_kind": "ws_flex",
                "hidden_edge_orientation": orientation,
                "regime": regime,
                "k": k,
                "p": p,
                "graph_seed": graph_seed,
                "wiring_seed": wiring_seed,
                "hidden_adj_undirected": undirected_adj.tolist(),
                "hidden_adj_directed": directed_adj.tolist(),
            }
            with open(selected_dir / f"{model_name}.json", "w", encoding="utf-8") as f:
                json.dump(arch, f, indent=2)
            selected_rows.append({
                "model_name": model_name,
                "method": "orientation_sensitivity",
                "wiring_kind": "ws_flex",
                "regime": regime,
                "orientation": orientation,
                "k": k,
                "p": p,
                "graph_seed": graph_seed,
                "wiring_seed": wiring_seed,
            })

    # Manifest and selected_architectures.csv
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

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

    if selected_rows:
        fieldnames = list(selected_rows[0].keys())
        with open(out_dir / "selected_architectures.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(selected_rows)

    print(f"Wrote {len(model_names)} architectures to {selected_dir}")
    print("Models:", model_names)

    if args.dry_run:
        print("Dry run: would invoke unified_experiment_runner for each model. Exit.")
        return

    repo_root = _REPO_ROOT
    runner_script = repo_root / "evaluation" / "unified_experiment_runner.py"
    for model_name in model_names:
        cmd = [
            args.python,
            str(runner_script),
            "--nas_pilot_dir",
            str(out_dir),
            "--model",
            model_name,
            "--dataset",
            args.dataset,
            "--subjects",
            *[str(s) for s in args.subjects],
            "--mode",
            "test_perturb",
            "--eval_mode",
            args.eval_mode,
            "--seed",
            str(args.seed),
            "--test_perturb_noise_types",
            "ar1_drift",
            "--test_perturb_target_snr_db=" + str(args.target_snr_db),
            "--noise_perturbation_saturation_file",
            args.saturation_file,
            "--test_perturb_gaussian_alpha_grid",
            "0,0.25,0.5,0.75,1.0",
            "--plot2_diagnostics_dir",
            str(diagnostics_dir),
            "--overwrite",
        ]
        print(f"[ORIENT] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            print(f"[ORIENT] Runner failed for {model_name}: {proc.stderr[:500] if proc.stderr else 'no stderr'}")

    # Analyze results
    analyzer_script = _THIS_DIR / "analyze_plot2_results.py"
    analyze_cmd = [
        args.python,
        str(analyzer_script),
        "--plot2_dir",
        str(out_dir),
        "--repo_root",
        str(repo_root),
    ]
    print(f"[ORIENT] Running analyzer: {' '.join(analyze_cmd)}")
    proc_analyze = subprocess.run(analyze_cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=600)
    if proc_analyze.returncode != 0:
        print(f"[ORIENT] Analyzer failed: {proc_analyze.stderr[:1000] if proc_analyze.stderr else 'no stderr'}")

    # Build orientation sensitivity report
    analysis_dir = out_dir / "analysis"
    per_graph_path = analysis_dir / "per_graph_aupc.csv"
    results_by_model: Dict[str, float] = {}
    clean_roc_auc_by_model: Dict[str, float] = {}
    max_drop_by_model: Dict[str, float] = {}
    mid_drop_by_model: Dict[str, float] = {}

    if per_graph_path.exists():
        per_graph_df = pd.read_csv(per_graph_path)
        metrics = compute_smoke_robustness_metrics(
            per_graph_df, model_names, primary_noise_type="ar1_drift"
        )
        results_by_model = metrics["aupc_by_model"]
        clean_roc_auc_by_model = metrics["clean_roc_auc_by_model"]
        max_drop_by_model = metrics["max_drop_by_model"]
        mid_drop_by_model = metrics["mid_drop_by_model"]

    for mn in model_names:
        if mn not in results_by_model:
            results_by_model[mn] = float("nan")
        if mn not in clean_roc_auc_by_model:
            clean_roc_auc_by_model[mn] = float("nan")
        if mn not in max_drop_by_model:
            max_drop_by_model[mn] = float("nan")
        if mn not in mid_drop_by_model:
            mid_drop_by_model[mn] = float("nan")

    # Per-regime comparison: symmetric vs random_oriented
    regime_comparisons: Dict[str, Dict[str, Any]] = {}
    for cfg in REGIME_GRAPHS:
        regime = cfg["regime"]
        ro = f"orient_{regime}_random_oriented"
        sym = f"orient_{regime}_symmetric"
        delta_aupc = float(results_by_model.get(sym, float("nan")) - results_by_model.get(ro, float("nan")))
        delta_clean = float(clean_roc_auc_by_model.get(sym, float("nan")) - clean_roc_auc_by_model.get(ro, float("nan")))
        delta_max_drop = float(max_drop_by_model.get(ro, float("nan")) - max_drop_by_model.get(sym, float("nan")))
        regime_comparisons[regime] = {
            "random_oriented_aupc": results_by_model.get(ro, float("nan")),
            "symmetric_aupc": results_by_model.get(sym, float("nan")),
            "delta_aupc_sym_minus_ro": delta_aupc,
            "random_oriented_clean_roc_auc": clean_roc_auc_by_model.get(ro, float("nan")),
            "symmetric_clean_roc_auc": clean_roc_auc_by_model.get(sym, float("nan")),
            "delta_clean_sym_minus_ro": delta_clean,
            "random_oriented_max_drop": max_drop_by_model.get(ro, float("nan")),
            "symmetric_max_drop": max_drop_by_model.get(sym, float("nan")),
            "delta_max_drop_ro_minus_sym": delta_max_drop,
        }

    report = {
        "schema_version": 1,
        "run_id": "orientation_sensitivity",
        "models": model_names,
        "results_by_model": results_by_model,
        "clean_roc_auc_by_model": clean_roc_auc_by_model,
        "max_drop_by_model": max_drop_by_model,
        "mid_drop_by_model": mid_drop_by_model,
        "regime_comparisons": regime_comparisons,
        "mask_stats_by_model": mask_stats_by_model,
        "interpretation": (
            "delta_aupc_sym_minus_ro > 0 means symmetric is more robust (higher AUPC). "
            "delta_max_drop_ro_minus_sym > 0 means symmetric degrades less (lower max_drop)."
        ),
    }
    with open(out_dir / "orientation_sensitivity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {out_dir / 'orientation_sensitivity_report.json'}")
    print("\n--- Regime comparisons (symmetric vs random_oriented) ---")
    for regime, comp in regime_comparisons.items():
        print(f"  {regime}: delta_AUPC={comp['delta_aupc_sym_minus_ro']:.4f}, delta_clean_ROC={comp['delta_clean_sym_minus_ro']:.4f}, delta_max_drop={comp['delta_max_drop_ro_minus_sym']:.4f}")


if __name__ == "__main__":
    main()
