"""WS-Flex panel generation (wraps architecture_refinement.ws_flex_generator)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import networkx as nx
import numpy as np

from architecture_refinement.ws_flex_generator import build_plain_ws_flex


def sample_wsflex_graph(hidden_size: int, k: int, p: float, graph_seed: int) -> nx.Graph:
    return build_plain_ws_flex(int(hidden_size), int(k), float(p), int(graph_seed))


def generate_wsflex_panel(
    hidden_size: int,
    k_values: List[int],
    p_values: List[float],
    graphs_per_regime: int,
    enforce_connected: bool = True,
    panel_seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Stratified panel: for each (k, p) draw `graphs_per_regime` graphs with distinct seeds.
    """
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(panel_seed)
    for k in k_values:
        for p in p_values:
            for _ in range(graphs_per_regime):
                graph_seed = int(rng.integers(0, 2**31 - 1))
                g = sample_wsflex_graph(hidden_size, k, p, graph_seed)
                if not enforce_connected:
                    pass
                elif not nx.is_connected(g):
                    continue
                rows.append(
                    {
                        "k": int(k),
                        "p": float(p),
                        "graph_seed": graph_seed,
                        "hidden_size": int(hidden_size),
                        "graph": g,
                    }
                )
    return rows


def topology_id_from_row(hidden_size: int, k: int, p: float, graph_seed: int) -> str:
    return f"H{hidden_size}_k{k}_p{p:.4f}_gs{graph_seed}"
