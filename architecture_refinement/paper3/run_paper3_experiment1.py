"""
Paper 3 Experiment 1 (Plot 1): Proxy Landscape + Cost Advantage.

Collects N_proxy connected WS-Flex graphs, computes spec-aligned proxy metrics
(te_hat, orc_hat), extracts Pareto front, selects K via grid-binning diversity.
Outputs proxy_pool.csv, selected_proxy.csv, and cost accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.metrics_te_orc import compute_paper3_proxies
from architecture_refinement.pareto_hv import pareto_front_2d
from architecture_refinement.ws_flex_generator import build_plain_ws_flex

# Spec defaults
DEFAULT_H = 32
DEFAULT_N_PROXY = 500
DEFAULT_K = 12
K_VALUES = list(range(2, 25, 2))  # 2,4,6,...,24 even
B_GRID = 6


def _sample_params(rng: np.random.Generator) -> Tuple[int, float, int]:
    """Sample (k, p, graph_seed) within bounds. s_g = trial index passed separately."""
    k = int(rng.choice(K_VALUES))
    p = float(rng.uniform(0.0, 1.0))
    graph_seed = int(rng.integers(0, 2**31 - 1))
    return k, p, graph_seed


def _select_k_from_pareto_grid(
    records: List[Dict[str, Any]],
    pareto_indices: List[int],
    K: int,
    B: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Select K indices from Pareto set using BxB grid binning (diversity).
    """
    if len(pareto_indices) <= K:
        return pareto_indices

    # Build (te_hat, orc_hat) for Pareto points
    pts = [(records[i]["te_hat"], records[i]["orc_hat"]) for i in pareto_indices]

    # Assign each to bin; handle edge case te_hat=1 or orc_hat=1
    bins: Dict[Tuple[int, int], List[int]] = {}
    for idx, (te, oc) in zip(pareto_indices, pts):
        bi = min(int(te * B), B - 1) if te < 1.0 else B - 1
        bj = min(int(oc * B), B - 1) if oc < 1.0 else B - 1
        bins.setdefault((bi, bj), []).append(idx)

    # Sort bins by count (desc) to prioritize diverse bins
    bin_list = sorted(bins.items(), key=lambda x: -len(x[1]))

    selected: List[int] = []
    for (_, indices) in bin_list:
        if len(selected) >= K:
            break
        # Shuffle to avoid bias
        order = rng.permutation(len(indices))
        for o in order:
            if len(selected) >= K:
                break
            selected.append(indices[int(o)])

    # If still short (shouldn't happen with enough Pareto points), add from remainder
    remaining = [i for i in pareto_indices if i not in selected]
    for i in remaining:
        if len(selected) >= K:
            break
        selected.append(i)

    return selected[:K]


def run_experiment1(
    output_dir: Path,
    H: int = DEFAULT_H,
    N_proxy: int = DEFAULT_N_PROXY,
    K: int = DEFAULT_K,
    B: int = B_GRID,
    seed: int = 202602,
) -> Dict[str, Any]:
    """
    Run Experiment 1: proxy collection, Pareto extraction, diversity selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records: List[Dict[str, Any]] = []
    rejected = 0
    trial = 0
    proxy_times: List[float] = []

    print(f"[Exp1] Collecting {N_proxy} connected graphs (H={H})...")
    while len(records) < N_proxy:
        trial += 1
        k, p, graph_seed = _sample_params(rng)
        s_g = trial  # deterministic per trial

        t0 = time.perf_counter()
        G = build_plain_ws_flex(H, k, p, s_g)
        if not nx.is_connected(G):
            rejected += 1
            continue

        te_hat, orc_hat = compute_paper3_proxies(G)
        elapsed = time.perf_counter() - t0
        proxy_times.append(elapsed)

        n_edges = G.number_of_edges()
        E_active = 2 * n_edges  # bidirectional

        records.append({
            "H": H,
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "te_hat": te_hat,
            "orc_hat": orc_hat,
            "n_edges": n_edges,
            "E_active": E_active,
            "runtime_sec": elapsed,
        })

        if len(records) % 100 == 0:
            print(f"  Collected {len(records)}/{N_proxy}, rejected={rejected}")

    # Write proxy pool
    pool_path = output_dir / "proxy_pool.csv"
    with open(pool_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["H", "k", "p", "graph_seed", "te_hat", "orc_hat", "n_edges", "E_active", "runtime_sec"],
        )
        w.writeheader()
        w.writerows(records)

    # Pareto front (max-max)
    points = [(r["te_hat"], r["orc_hat"]) for r in records]
    pf = pareto_front_2d(points)
    pareto_indices = []
    for i, r in enumerate(records):
        pt = (r["te_hat"], r["orc_hat"])
        for pf_pt in pf:
            if abs(pt[0] - pf_pt[0]) < 1e-9 and abs(pt[1] - pf_pt[1]) < 1e-9:
                pareto_indices.append(i)
                break

    # Diversity selection
    selected_indices = _select_k_from_pareto_grid(records, pareto_indices, K, B, rng)
    selected_records = [records[i] for i in selected_indices]

    # Write selected
    selected_path = output_dir / "selected_proxy.csv"
    with open(selected_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["H", "k", "p", "graph_seed", "te_hat", "orc_hat", "n_edges", "E_active", "runtime_sec"],
        )
        w.writeheader()
        w.writerows(selected_records)

    # Write selected_architectures/*.json for Experiment 2 (bidirectional wiring)
    selected_arch_dir = output_dir / "selected_architectures"
    selected_arch_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for rank, rec in enumerate(selected_records, start=1):
        G = build_plain_ws_flex(H, rec["k"], rec["p"], rec["graph_seed"])
        adj = nx.to_numpy_array(G, dtype=np.int8)
        adj = (adj != 0).astype(np.int8)
        model_name = f"paper3_exp1_proxy_S{rank}"
        arch = {
            "schema_version": 2,
            "model_name": model_name,
            "H": H,
            "wiring_kind": "ws_flex",
            "hidden_edge_orientation": "symmetric",
            "k": rec["k"],
            "p": rec["p"],
            "graph_seed": rec["graph_seed"],
            "wiring_seed": rec["graph_seed"],
            "te_hat": rec["te_hat"],
            "orc_hat": rec["orc_hat"],
            "n_edges": rec["n_edges"],
            "E_active": rec["E_active"],
            "hidden_adj_undirected": adj.tolist(),
        }
        (selected_arch_dir / f"{model_name}.json").write_text(json.dumps(arch, indent=2))

    mean_proxy_time = float(np.mean(proxy_times)) if proxy_times else 0.0
    # Estimated FLOPS for plot1: proxy ~1e7 (graph ops), full eval ~1e12 (training+robustness)
    proxy_flops_est = 1e7  # order-of-magnitude for H~32, E~400
    full_flops_est = 1e12  # training + perturbation sweep
    summary = {
        "n_proxy": len(records),
        "rejected_count": rejected,
        "pareto_size": len(pf),
        "K_selected": len(selected_indices),
        "mean_proxy_time_sec": mean_proxy_time,
        "mean_proxy_flops_est": proxy_flops_est,
        "mean_full_flops_est": full_flops_est,
        "H": H,
        "seed": seed,
    }
    (output_dir / "experiment1_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[Exp1] Done. Pareto size={len(pf)}, selected K={len(selected_indices)}")
    print(f"  Mean proxy time: {mean_proxy_time:.4f}s, rejected={rejected}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Paper 3 Experiment 1: Proxy landscape")
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/paper3_experiment1")
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--N-proxy", type=int, default=DEFAULT_N_PROXY)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=202602)
    args = parser.parse_args()

    output_dir = _REPO_ROOT / args.output_dir
    run_experiment1(
        output_dir=output_dir,
        H=args.H,
        N_proxy=args.N_proxy,
        K=args.K,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
