"""
Paper 3 Experiment 2 (Plot 2): Proxy-selected vs Uniform vs Baselines.

Trains G1 (proxy-selected WS-Flex), G2 (uniform WS-Flex), G3 (Dense CfC),
G4 (Random sparse, sparsity-matched), G5 (NCP) under identical protocol.
Evaluates robustness (AR(1) drift), outputs r_t curves and RD_max.

Use ``--n-per-family N`` for Option B: matched N topologies per family (ER-dense G3,
random-sparse G4, distinct NCP wirings G5) for inferential family contrast.
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
import networkx as nx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.ws_flex_generator import build_plain_ws_flex
from utils import short_run_id

DEFAULT_H = 32
DEFAULT_K = 12
DEFAULT_S = 5
K_VALUES = list(range(2, 25, 2))
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_SATURATION = "saturation_results/saturation_points_summary.csv"


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
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def _random_symmetric_adjacency(H: int, n_edges: int, seed: int) -> np.ndarray:
    """Create random symmetric adjacency with exactly n_edges undirected edges (E_active=2*n_edges)."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(H) for j in range(i + 1, H)]
    n_edges = min(n_edges, len(pairs))
    chosen = rng.choice(len(pairs), size=n_edges, replace=False)
    A = np.zeros((H, H), dtype=np.int8)
    for idx in chosen:
        i, j = pairs[int(idx)]
        A[i, j] = 1
        A[j, i] = 1
    return A


def _make_random_sparse_arch(H: int, E_active: int, seed: int, model_name: str) -> Dict[str, Any]:
    n_undirected = E_active // 2
    adj = _random_symmetric_adjacency(H, n_undirected, seed)
    return {
        "schema_version": 2,
        "model_name": model_name,
        "H": H,
        "wiring_kind": "ws_flex",
        "hidden_edge_orientation": "symmetric",
        "k": -1,
        "p": -1.0,
        "graph_seed": seed,
        "wiring_seed": seed,
        "te_hat": float("nan"),
        "orc_hat": float("nan"),
        "n_edges": n_undirected,
        "E_active": E_active,
        "hidden_adj_undirected": adj.tolist(),
        "baseline_type": "random_sparse",
    }


def _make_dense_arch(H: int, model_name: str) -> Dict[str, Any]:
    G = nx.complete_graph(H)
    adj = nx.to_numpy_array(G, dtype=np.int8)
    n_edges = G.number_of_edges()
    return {
        "schema_version": 2,
        "model_name": model_name,
        "H": H,
        "wiring_kind": "ws_flex",
        "hidden_edge_orientation": "symmetric",
        "k": H - 1,
        "p": 0.0,
        "graph_seed": 0,
        "wiring_seed": 0,
        "te_hat": float("nan"),
        "orc_hat": float("nan"),
        "n_edges": n_edges,
        "E_active": 2 * n_edges,
        "hidden_adj_undirected": adj.tolist(),
        "baseline_type": "dense",
    }


def _make_er_dense_arch(H: int, model_name: str, graph_seed: int) -> Dict[str, Any]:
    """
    Distinct dense random graph per graph_seed (high-p ER), for matched-n Option B.
    Same JSON schema as other ws_flex adjacency architectures.
    """
    rng = np.random.default_rng(graph_seed)
    G = None
    for _ in range(400):
        p = float(rng.uniform(0.82, 0.995))
        g = nx.erdos_renyi_graph(H, p, seed=int(rng.integers(0, 2**31 - 1)))
        if nx.is_connected(g):
            G = g
            break
    if G is None:
        G = nx.complete_graph(H)
    adj = nx.to_numpy_array(G, dtype=np.int8)
    adj = (adj != 0).astype(np.int8)
    n_edges = int(G.number_of_edges())
    return {
        "schema_version": 2,
        "model_name": model_name,
        "H": H,
        "wiring_kind": "ws_flex",
        "hidden_edge_orientation": "symmetric",
        "k": -1,
        "p": -1.0,
        "graph_seed": int(graph_seed),
        "wiring_seed": int(graph_seed),
        "te_hat": float("nan"),
        "orc_hat": float("nan"),
        "n_edges": n_edges,
        "E_active": 2 * n_edges,
        "hidden_adj_undirected": adj.tolist(),
        "baseline_type": "dense_er",
    }


def _make_ncp_arch(
    H: int,
    model_name: str,
    output_size: int = 16,
    sparsity: float = 0.5,
    wiring_seed: int = 202603,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "model_name": model_name,
        "wiring_kind": "ncp_autoncp",
        "units": H,
        "output_size": output_size,
        "sparsity_level": sparsity,
        "wiring_seed": int(wiring_seed),
        "baseline_type": "ncp",
    }


def run_experiment2(
    experiment1_dir: Path,
    output_dir: Path,
    H: int = DEFAULT_H,
    K: int = DEFAULT_K,
    S: int = DEFAULT_S,
    n_per_family: Optional[int] = None,
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
    Run Experiment 2: train G1..G5, evaluate robustness.

    If ``n_per_family`` is set (e.g. 5), each group G1–G5 has exactly that many distinct
    topologies (Option B matched-n): G1 first n from Exp1, G2 n uniform WS-Flex, G3 n ER-dense,
    G4 n random sparse (E matched to G1 mean), G5 n NCP wirings (distinct wiring_seed).
    If None, legacy behavior: K models for G1/G2 and one model each for G3–G5.
    """
    if subjects is None:
        subjects = list(range(1, 10))
    python_exe = python_exe or sys.executable

    output_dir.mkdir(parents=True, exist_ok=True)
    exp1_dir = Path(experiment1_dir)
    if not exp1_dir.is_absolute():
        exp1_dir = _REPO_ROOT / exp1_dir

    pilot_dir = output_dir / "experiment2_pilot"
    selected_dir = pilot_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    # G1: Copy proxy-selected from Experiment 1
    exp1_arch = exp1_dir / "selected_architectures"
    g1_models: List[str] = []
    if exp1_arch.exists():
        for p in sorted(exp1_arch.glob("paper3_exp1_proxy_*.json")):
            arch = json.loads(p.read_text(encoding="utf-8"))
            arch["group"] = "G1_proxy"
            dst = selected_dir / p.name
            dst.write_text(json.dumps(arch, indent=2))
            g1_models.append(arch["model_name"])

    n_ws = int(n_per_family) if n_per_family is not None else int(K)
    if n_ws < 2:
        raise ValueError("Need n_per_family >= 2 (or K >= 2) for meaningful matched families")

    if len(g1_models) < n_ws:
        raise FileNotFoundError(
            f"Experiment 1 has only {len(g1_models)} selected architectures, need at least {n_ws}"
        )

    if n_per_family is not None:
        g1_models = g1_models[:n_ws]

    # G2: Uniform-selected WS-Flex (independent of G1)
    rng = np.random.default_rng(202607)  # Different seed from Exp1
    g2_models: List[str] = []
    attempts = 0
    while len(g2_models) < n_ws and attempts < 5000:
        attempts += 1
        k = int(rng.choice(K_VALUES))
        p = float(rng.uniform(0.0, 1.0))
        gs = int(rng.integers(0, 2**31 - 1))
        G = build_plain_ws_flex(H, k, p, gs)
        if not nx.is_connected(G):
            continue
        adj = nx.to_numpy_array(G, dtype=np.int8)
        adj = (adj != 0).astype(np.int8)
        model_name = f"paper3_exp2_uniform_{len(g2_models)+1}"
        arch = {
            "schema_version": 2,
            "model_name": model_name,
            "H": H,
            "wiring_kind": "ws_flex",
            "hidden_edge_orientation": "symmetric",
            "k": k,
            "p": p,
            "graph_seed": gs,
            "wiring_seed": gs,
            "te_hat": float("nan"),
            "orc_hat": float("nan"),
            "n_edges": G.number_of_edges(),
            "E_active": 2 * G.number_of_edges(),
            "hidden_adj_undirected": adj.tolist(),
            "group": "G2_uniform",
        }
        (selected_dir / f"{model_name}.json").write_text(json.dumps(arch, indent=2))
        g2_models.append(model_name)

    # G4/G3 need average E_active from G1 (same folder as written)
    g1_E_active = []
    for m in g1_models:
        a = json.loads((selected_dir / f"{m}.json").read_text(encoding="utf-8"))
        g1_E_active.append(int(a.get("E_active", 0)))
    avg_E = int(np.mean(g1_E_active)) if g1_E_active else 256

    g3_models: List[str] = []
    g4_models: List[str] = []
    g5_models: List[str] = []

    if n_per_family is not None:
        n = int(n_per_family)
        base = 913_000 + H * 17
        for i in range(n):
            dense_model = f"paper3_exp2_dense_{i+1:02d}"
            dense_arch = _make_er_dense_arch(H, dense_model, graph_seed=base + i * 7919)
            dense_arch["group"] = "G3_dense"
            (selected_dir / f"{dense_model}.json").write_text(json.dumps(dense_arch, indent=2))
            g3_models.append(dense_model)

        for i in range(n):
            rand_model = f"paper3_exp2_random_sparse_{i+1:02d}"
            rand_arch = _make_random_sparse_arch(H, avg_E, 202608 + i * 9973, rand_model)
            rand_arch["group"] = "G4_random_sparse"
            (selected_dir / f"{rand_model}.json").write_text(json.dumps(rand_arch, indent=2))
            g4_models.append(rand_model)

        for i in range(n):
            ncp_model = f"paper3_exp2_ncp_{i+1:02d}"
            ncp_arch = _make_ncp_arch(H, ncp_model, wiring_seed=202_603 + i * 10_007)
            ncp_arch["group"] = "G5_ncp"
            (selected_dir / f"{ncp_model}.json").write_text(json.dumps(ncp_arch, indent=2))
            g5_models.append(ncp_model)
    else:
        dense_model = "paper3_exp2_dense"
        dense_arch = _make_dense_arch(H, dense_model)
        dense_arch["group"] = "G3_dense"
        (selected_dir / f"{dense_model}.json").write_text(json.dumps(dense_arch, indent=2))
        g3_models = [dense_model]

        rand_model = "paper3_exp2_random_sparse"
        rand_arch = _make_random_sparse_arch(H, avg_E, 202608, rand_model)
        rand_arch["group"] = "G4_random_sparse"
        (selected_dir / f"{rand_model}.json").write_text(json.dumps(rand_arch, indent=2))
        g4_models = [rand_model]

        ncp_model = "paper3_exp2_ncp"
        ncp_arch = _make_ncp_arch(H, ncp_model)
        ncp_arch["group"] = "G5_ncp"
        (selected_dir / f"{ncp_model}.json").write_text(json.dumps(ncp_arch, indent=2))
        g5_models = [ncp_model]

    manifest: Dict[str, Any] = {
        "H": H,
        "K": K,
        "S": S,
        "dataset": dataset,
        "eval_mode": eval_mode,
        "groups": {
            "G1": g1_models,
            "G2": g2_models,
            "G3": g3_models,
            "G4": g4_models,
            "G5": g5_models,
        },
        "perturbation_types": ["ar1_drift"],
        "target_snr_db": target_snr_db,
    }
    if n_per_family is not None:
        manifest["n_per_family"] = int(n_per_family)
        manifest["matched_families_option_b"] = True
    (output_dir / "experiment2_manifest.json").write_text(json.dumps(manifest, indent=2))

    if dry_run:
        print("[Exp2] Dry-run: would dispatch jobs for G1..G5.")
        return manifest

    sat_path = str(_REPO_ROOT / saturation_file) if not Path(saturation_file).is_absolute() else saturation_file
    pert_params = {"ar1_drift": {"rho": 0.97}}
    seeds = list(range(42, 42 + S))
    all_models = g1_models + g2_models + g3_models + g4_models + g5_models
    jobs = [(m, s) for m in all_models for s in seeds]
    failed = []
    for model_name, seed in jobs:
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
        print(f"[Exp2] {len(failed)} jobs failed.")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Paper 3 Experiment 2: Proxy vs Uniform vs Baselines")
    parser.add_argument("--experiment1-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/paper3_experiment2")
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--S", type=int, default=DEFAULT_S)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval-mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--saturation-file", type=str, default=DEFAULT_SATURATION)
    parser.add_argument("--target-snr-db", type=float, default=-5.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--n-per-family",
        type=int,
        default=None,
        metavar="N",
        help="Option B: exactly N distinct topologies per G1–G5 (requires Exp1 with at least N proxies). "
        "Uses ER-dense G3, random-sparse G4, multi-seed NCP G5.",
    )
    args = parser.parse_args()

    exp1_dir = Path(args.experiment1_dir)
    if not exp1_dir.is_absolute():
        exp1_dir = _REPO_ROOT / exp1_dir
    out_dir = _REPO_ROOT / args.output_dir
    run_experiment2(
        experiment1_dir=exp1_dir,
        output_dir=out_dir,
        H=args.H,
        K=args.K,
        S=args.S,
        n_per_family=args.n_per_family,
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
