#!/usr/bin/env python3
"""Build WS-Flex topology panel manifest and serialized graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config_util import add_config_args, load_merged_configs, parse_config_paths, resolve_path
from src.graphs.graph_metrics import compute_graph_descriptors
from src.graphs.wsflex_generator import generate_wsflex_panel, topology_id_from_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--limit", type=int, default=0, help="Max graphs (0=all)")
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))
    topo = cfg.get("topology") or cfg
    hidden_size = int(topo.get("hidden_size", 32))
    k_values = list(topo.get("k_values", [4, 8]))
    p_values = [float(x) for x in topo.get("p_values", [0.05, 0.3])]
    graphs_per_regime = int(topo.get("graphs_per_regime", 8))
    enforce_connected = bool(topo.get("enforce_connected", True))
    save_graph_objects = bool(topo.get("save_graph_objects", True))
    panel_seed = int(topo.get("panel_seed", 42))

    out_manifest = resolve_path(cfg, "output_manifest", "manifests/topology_panel.csv")
    graph_dir = resolve_path(cfg, "graph_output_dir", "outputs/topology_panel")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_wsflex_panel(
        hidden_size=hidden_size,
        k_values=k_values,
        p_values=p_values,
        graphs_per_regime=graphs_per_regime,
        enforce_connected=enforce_connected,
        panel_seed=panel_seed,
    )
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    records = []
    for r in rows:
        g = r["graph"]
        k, p, gs = r["k"], r["p"], r["graph_seed"]
        tid = topology_id_from_row(hidden_size, k, p, gs)
        desc = compute_graph_descriptors(g)
        graph_path = ""
        if save_graph_objects:
            gp = graph_dir / f"{tid}.npz"
            A = nx.to_numpy_array(g, dtype=np.float32)
            np.savez_compressed(
                gp,
                adjacency=A,
                k=k,
                p=p,
                graph_seed=gs,
                hidden_size=hidden_size,
                topology_id=tid,
            )
            try:
                graph_path = str(gp.relative_to(_NPF_ROOT))
            except ValueError:
                graph_path = str(gp)

        records.append(
            {
                "topology_id": tid,
                "hidden_size": hidden_size,
                "k": k,
                "p": p,
                "graph_seed": gs,
                "num_edges": desc["num_edges"],
                "density": desc["density"],
                "clustering": desc["clustering"],
                "avg_path_length": desc["avg_path_length"],
                "topological_entropy": desc["topological_entropy"],
                "abs_orc": desc["abs_orc"],
                "graph_path": graph_path,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(out_manifest, index=False)
    print(f"Wrote {len(df)} rows to {out_manifest}")


if __name__ == "__main__":
    main()
