"""
Shared utilities to extract networkx graphs from Paper 3 architecture JSONs.
Used by experiment3 and analysis followups for TE/ORC proxy computation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def graph_from_architecture(arch: Dict[str, Any], *, input_dim: int = 16):
    """
    Build undirected networkx graph from architecture JSON.

    Supports:
    - ws_flex: hidden_adj_undirected, or (k, p, graph_seed, H)
    - ncp_autoncp: units, output_size, sparsity_level, wiring_seed

    Returns None if unsupported or on error.
    """
    import networkx as nx

    adj = arch.get("hidden_adj_undirected")
    if adj is not None:
        k = arch.get("k", -1)
        p = arch.get("p", -1.0)
        graph_seed = arch.get("graph_seed", 0)
        H = arch.get("H", 32)
        if k >= 0 and p >= 0:
            from architecture_refinement.ws_flex_generator import build_plain_ws_flex
            return build_plain_ws_flex(H, int(k), float(p), int(graph_seed))
        A = np.asarray(adj)
        return nx.from_numpy_array(A)

    if arch.get("wiring_kind") == "ncp_autoncp":
        return _graph_from_ncp_arch(arch, input_dim=input_dim)

    return None


def _graph_from_ncp_arch(arch: Dict[str, Any], *, input_dim: int = 16):
    """Build undirected graph from NCP (AutoNCP) wiring adjacency."""
    import networkx as nx

    try:
        from ncps.wirings import AutoNCP
    except ImportError:
        return None

    units = int(arch.get("units", 32))
    output_size = int(arch.get("output_size", 16))
    sparsity_level = float(arch.get("sparsity_level", 0.5))
    wiring_seed = int(arch.get("wiring_seed", 202603))

    wiring = AutoNCP(
        units=units,
        output_size=output_size,
        sparsity_level=sparsity_level,
        seed=wiring_seed,
    )
    wiring.build(input_dim)

    A = np.asarray(wiring.adjacency_matrix)
    A_bool = (A != 0) | (A.T != 0)
    return nx.from_numpy_array(A_bool.astype(np.int8))
