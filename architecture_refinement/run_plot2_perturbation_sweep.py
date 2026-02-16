"""
Plot2_revision3: Most-Damaging Perturbation Diagnostic.

Rapid diagnostic to identify perturbation settings that are damaging and
topology-sensitive before committing to full training. Uses 4 graphs spanning
regimes (super_sparse, sparse, moderate, near_dense), 1-2 subjects, 1 seed.
Outputs: perturbation_sweep_report.json, locked_perturbation_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import short_run_id

H = 32
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
# 4 graphs spanning regimes
GRAPH_CONFIGS = [
    {"regime": "super_sparse", "k": 4, "p": 0.3},
    {"regime": "sparse", "k": 10, "p": 0.35},
    {"regime": "moderate", "k": 16, "p": 0.4},
    {"regime": "near_dense", "k": 22, "p": 0.45},
]


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


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
    return (np.asarray(A) != 0).astype(np.int8)


def create_sweep_architectures(
    pilot_dir: Path,
    base_id: str = "plot2_sweep",
    seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Create 4 architectures (one per regime) and return metadata."""
    arch_dir = pilot_dir / "selected_architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    archs = []
    for i, cfg in enumerate(GRAPH_CONFIGS):
        k, p = cfg["k"], cfg["p"]
        model_id = f"{base_id}_{cfg['regime']}_{i}"
        graph_seed = 42000 + seed_offset + i
        wiring_seed = 42001 + seed_offset + i
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            raise RuntimeError(f"Graph {model_id} disconnected")
        undirected_adj = (nx.to_numpy_array(G) != 0).astype(np.int8)
        directed_adj = _oriented_hidden_adj(G, H, wiring_seed)
        arch = {
            "schema_version": 2,
            "model_name": model_id,
            "wiring_kind": "ws_flex",
            "H": H,
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "wiring_seed": wiring_seed,
            "hidden_adj_undirected": undirected_adj.tolist(),
            "hidden_adj_directed": directed_adj.tolist(),
        }
        (arch_dir / f"{model_id}.json").write_text(json.dumps(arch, indent=2), encoding="utf-8")
        archs.append({"model_id": model_id, "regime": cfg["regime"], "k": k, "p": p})
    return archs


def run_eval(
    repo_root: Path,
    pilot_dir: Path,
    model_id: str,
    dataset: str,
    subjects: List[int],
    seed: int,
    noise_types: List[str],
    target_snr_db: float,
    python_exe: str,
) -> int:
    """Run unified runner for one model."""
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
        "--test_perturb_gaussian_alpha_grid", ",".join(str(a) for a in ALPHA_GRID),
        "--test_perturb_noise_types", ",".join(noise_types),
        "--test_perturb_target_snr_db", str(target_snr_db),
    ]
    print("[SWEEP] Running:", " ".join(cmd[:12]), "...")
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def collect_results(repo_root: Path, dataset: str, model_id: str, seed: int, noise_type: str) -> Optional[Dict[str, Any]]:
    """Collect max_drop, AUPC, clean ROC for one model + noise type."""
    paradigm = "MotorImagery" if "BNCI" in dataset or "BI2015" in dataset else "SSVEP"
    base = Path(repo_root) / "results" / paradigm / dataset
    for stem in [short_run_id(model_id), model_id]:
        path = base / stem / "CrossSessionEvaluation" / str(seed)
        if not path.exists():
            continue
        csvs = list(path.rglob("*.csv"))
        for p in csvs:
            if "test_perturb" not in str(p):
                continue
            try:
                df = pd.read_csv(p)
                if "noise_type" not in df.columns or noise_type not in df["noise_type"].astype(str).values:
                    continue
                sub = df[df["noise_type"].astype(str) == noise_type]
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["corrupted_roc_auc"] = pd.to_numeric(sub["corrupted_roc_auc"], errors="coerce")
                sub["clean_roc_auc"] = pd.to_numeric(sub.get("clean_roc_auc", sub.get("clean_score", np.nan)), errors="coerce")
                clean = float(sub["clean_roc_auc"].iloc[0]) if sub["clean_roc_auc"].notna().any() else float("nan")
                roc_at_max = float(sub["corrupted_roc_auc"].iloc[-1]) if len(sub) else float("nan")
                max_drop = float(clean - roc_at_max) if np.isfinite(clean) and np.isfinite(roc_at_max) else float("nan")
                xs = pd.to_numeric(sub.get("intensity", sub.index), errors="coerce").to_numpy()
                ys = sub["corrupted_roc_auc"].to_numpy()
                if len(xs) >= 2 and np.isfinite(xs).any() and np.isfinite(ys).any():
                    aupc = float(np.trapz(ys, xs)) if hasattr(np, "trapezoid") else float(np.trapz(ys, xs))
                else:
                    aupc = float("nan")
                return {"clean_roc_auc": clean, "max_drop": max_drop, "AUPC": aupc}
            except Exception as e:
                print(f"[SWEEP] Warning: {e}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot2 Most-Damaging Perturbation Diagnostic")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--subjects", type=int, nargs="*", default=[1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="architecture_refinement/outputs/perturbation_sweep")
    parser.add_argument("--config", type=str, default=None, help="Path to perturbation_sweep_config.yaml")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--snr_grid", type=str, default="0,-3,-6,-9,-12", help="SNR dB values for ar1_drift sweep")
    parser.add_argument("--skip_run", action="store_true", help="Skip eval; only analyze existing results")
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = out_dir / "pilot_manifests" / "sweep"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    snr_values = [float(x) for x in args.snr_grid.split(",")]
    noise_types_to_sweep = ["ar1_drift"]
    report: Dict[str, Any] = {"configs": [], "per_config": {}, "P_star": None, "go_pass": False}

    archs = create_sweep_architectures(pilot_dir, base_id="plot2_sweep", seed_offset=args.seed * 10)
    report["architectures"] = [a["model_id"] for a in archs]

    if not args.skip_run:
        for snr_db in snr_values:
            for arch in archs:
                model_id_snr = f"{arch['model_id']}_snr{int(snr_db)}"
                arch_dir = pilot_dir / "selected_architectures"
                src = arch_dir / f"{arch['model_id']}.json"
                dst = arch_dir / f"{model_id_snr}.json"
                if src.exists():
                    arch_json = json.loads(src.read_text(encoding="utf-8"))
                    arch_json["model_name"] = model_id_snr
                    dst.write_text(json.dumps(arch_json, indent=2), encoding="utf-8")
                rc = run_eval(
                    repo_root, pilot_dir, model_id_snr,
                    args.dataset, args.subjects, args.seed,
                    noise_types_to_sweep, snr_db, args.python,
                )
                if rc != 0:
                    print(f"[SWEEP] Eval failed for {model_id_snr} snr={snr_db}")

    for snr_db in snr_values:
        key = f"ar1_drift_snr{snr_db}"
        max_drops: List[float] = []
        for i, cfg in enumerate(GRAPH_CONFIGS):
            model_id = f"plot2_sweep_{cfg['regime']}_{i}_snr{int(snr_db)}"
            res = collect_results(repo_root, args.dataset, model_id, args.seed, "ar1_drift")
            if res and np.isfinite(res.get("max_drop", float("nan"))):
                max_drops.append(float(res["max_drop"]))
        if len(max_drops) >= 2:
            s_topo = float(np.max(max_drops) - np.min(max_drops))
            max_drop_max = float(np.max(max_drops))
        else:
            s_topo = float("nan")
            max_drop_max = float(np.max(max_drops)) if max_drops else float("nan")
        report["per_config"][key] = {
            "target_snr_db": snr_db,
            "max_drop_max": max_drop_max,
            "S_topo": s_topo,
            "go_criteria": {
                "max_drop_ge_015": max_drop_max >= 0.15,
                "s_topo_ge_005": s_topo >= 0.05,
            },
        }

    passing = [
        k for k, v in report["per_config"].items()
        if v.get("max_drop_max", 0) >= 0.15 and v.get("S_topo", 0) >= 0.05
    ]
    if passing:
        best = max(passing, key=lambda k: report["per_config"][k]["S_topo"])
        report["P_star"] = best
        report["go_pass"] = True
        locked = {
            "perturbation_type": "ar1_drift",
            "target_snr_db": float(report["per_config"][best]["target_snr_db"]),
            "source": best,
        }
    else:
        best = max(report["per_config"].keys(), key=lambda k: report["per_config"][k].get("S_topo", -1))
        report["P_star"] = best
        report["go_pass"] = False
        locked = {
            "perturbation_type": "ar1_drift",
            "target_snr_db": float(report["per_config"][best]["target_snr_db"]),
            "source": best,
            "warning": "GO criteria not met; using best available",
        }

    # Plot 2 Overhaul: log perturbation fingerprints + chosen type
    report["perturbation_fingerprints"] = {
        "noise_types_swept": noise_types_to_sweep,
        "snr_values": snr_values,
        "chosen_type": locked.get("perturbation_type", "ar1_drift"),
        "chosen_target_snr_db": locked.get("target_snr_db"),
    }

    report_path = out_dir / "perturbation_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    locked_path = out_dir / "locked_perturbation_config.yaml"
    locked_path.write_text(yaml.dump(locked, default_flow_style=False), encoding="utf-8")

    print(f"[SWEEP] Report: {report_path}")
    print(f"[SWEEP] Locked config: {locked_path}")
    print(f"[SWEEP] GO pass: {report['go_pass']} (P*={report['P_star']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
