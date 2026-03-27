"""
Hail Mary (Chapter 5): build a fixed WS-Flex topology panel + manifest.

See Project_Hail_Mary.md §7, §13.1, §14 Phase 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.config import default_config
from architecture_refinement.metrics_te_orc import compute_paper3_proxies
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.ws_flex_generator import build_plain_ws_flex

DEFAULT_H = 32
# Document §7: sparse k=4, medium k=10, dense k=16; p ∈ {0.05, 0.30, 0.80}
DEFAULT_K_LEVELS = (4, 10, 16)
DEFAULT_P_LEVELS = (0.05, 0.30, 0.80)


def default_regime_grid() -> List[Tuple[int, float, int]]:
    """(k, p, graph_seed) for each cell in k×p grid."""
    rows: List[Tuple[int, float, int]] = []
    sid = 424_200
    for k in DEFAULT_K_LEVELS:
        for p in DEFAULT_P_LEVELS:
            rows.append((k, float(p), sid))
            sid += 1
    return rows


def build_architecture_json(
    *,
    H: int,
    k: int,
    p: float,
    graph_seed: int,
    model_name: str,
    topology_id: str,
) -> Tuple[Dict[str, Any], nx.Graph]:
    G = build_plain_ws_flex(H, k, p, graph_seed)
    adj = nx.to_numpy_array(G, dtype=np.int8)
    adj = (adj != 0).astype(np.int8)
    te_hat, orc_hat = compute_paper3_proxies(G)
    arch = {
        "schema_version": 2,
        "model_name": model_name,
        "H": H,
        "wiring_kind": "ws_flex",
        "hidden_edge_orientation": "symmetric",
        "k": int(k),
        "p": float(p),
        "graph_seed": int(graph_seed),
        "wiring_seed": int(graph_seed),
        "te_hat": float(te_hat),
        "orc_hat": float(orc_hat),
        "n_edges": int(G.number_of_edges()),
        "E_active": int(2 * G.number_of_edges()),
        "hidden_adj_undirected": adj.tolist(),
        "group": "hail_mary_panel",
        "topology_id": topology_id,
        "study": "hail_mary_chapter5",
    }
    return arch, G


def run_build_panel(
    output_dir: Path,
    H: int = DEFAULT_H,
    regimes: Sequence[Tuple[int, float, int]] | None = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    pilot_dir = output_dir / "hail_mary_pilot"
    selected_dir = pilot_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    analyzer = TopologyAnalyzer(default_config)
    regimes = list(regimes) if regimes is not None else default_regime_grid()

    manifest_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "H": H,
        "n_topologies": len(regimes),
        "pilot_dir": str(pilot_dir),
        "topologies": [],
    }

    for idx, (k, p, gs) in enumerate(regimes, start=1):
        tid = f"hail_mary_{idx:02d}"
        model_name = f"paper3_hail_mary_topo_{idx:02d}"
        arch, G = build_architecture_json(
            H=H, k=k, p=p, graph_seed=gs, model_name=model_name, topology_id=tid
        )
        metrics = analyzer.analyze_graph(G)
        connected = bool(nx.is_connected(G))

        arch_path = selected_dir / f"{model_name}.json"
        arch_path.write_text(json.dumps(arch, indent=2), encoding="utf-8")

        orc_abs_mean = float("nan")
        try:
            curv = metrics.get("ollivier_ricci_curvature_mean")
            if curv is not None and np.isfinite(curv):
                orc_abs_mean = float(abs(curv))
        except Exception:
            pass

        row = {
            "topology_id": tid,
            "model_name": model_name,
            "hidden_size": H,
            "k": k,
            "p": p,
            "graph_seed": gs,
            "valid_graph_flag": connected,
            "num_edges": int(G.number_of_edges()),
            "te_hat": arch["te_hat"],
            "orc_hat": arch["orc_hat"],
            "clustering_coefficient": float(metrics.get("clustering_coefficient", float("nan"))),
            "avg_path_length": float(metrics.get("avg_path_length", float("nan"))),
            "small_worldness": float(metrics.get("small_worldness", float("nan"))),
            "orc_abs_mean_analyzer": orc_abs_mean,
            "architecture_json": str(arch_path.relative_to(output_dir)),
        }
        manifest_rows.append(row)
        summary["topologies"].append(
            {
                "topology_id": tid,
                "model_name": model_name,
                "k": k,
                "p": p,
                "graph_seed": gs,
            }
        )

    csv_path = output_dir / "topology_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    json_path = output_dir / "topology_manifest.json"
    json_path.write_text(json.dumps({"manifest": manifest_rows, "summary": summary}, indent=2), encoding="utf-8")

    return {"csv": str(csv_path), "json": str(json_path), "pilot_dir": str(pilot_dir), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Hail Mary WS-Flex topology panel")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
        help="Writes topology_manifest.* and hail_mary_pilot/selected_architectures/",
    )
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    args = parser.parse_args()
    out = _REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    res = run_build_panel(out, H=args.H)
    print(json.dumps(res["summary"], indent=2))
    print(f"Wrote {res['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
