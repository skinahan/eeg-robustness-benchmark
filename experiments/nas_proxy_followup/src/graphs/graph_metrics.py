"""Graph descriptors for abstract WS-Flex graphs."""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np

from architecture_refinement.metrics_te_orc import compute_te_orc


def compute_graph_descriptors(graph: nx.Graph, orc_max_edges: int | None = 60) -> Dict[str, Any]:
    """Edge count, density, clustering, path length, TE, abs-ORC (ORC subsampled for speed)."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = float(nx.density(graph)) if n > 1 else 0.0
    try:
        cl = float(nx.average_clustering(graph))
    except Exception:
        cl = float("nan")
    try:
        if nx.is_connected(graph):
            apl = float(nx.average_shortest_path_length(graph))
        else:
            apl = float("nan")
    except Exception:
        apl = float("nan")

    try:
        te, orc, _dbg = compute_te_orc(graph, orc_max_edges=orc_max_edges)
    except Exception:
        te, orc = float("nan"), float("nan")

    return {
        "num_edges": int(m),
        "density": density,
        "clustering": cl,
        "avg_path_length": apl,
        "topological_entropy": float(te),
        "abs_orc": float(abs(float(orc))) if np.isfinite(orc) else float("nan"),
        "orc_mean": float(orc) if np.isfinite(orc) else float("nan"),
        "spectral_radius": float("nan"),
        "num_nodes": int(n),
    }
