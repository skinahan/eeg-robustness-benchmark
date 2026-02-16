"""
Capacity control utilities for Plot 2 (Plot2_revision3 Step C).

Ensures topology wins are not confounded by capacity: E_active (masked active edges)
must fall within regime bands derived from k-ranges at H=32.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import numpy as np
import networkx as nx

# E_active regime bands for H=32 (WS: undirected has H*k/2 edges; oriented keeps same count)
# k_min, k_max per regime -> E_active = H*k/2 for undirected; oriented = same nonzeros
DEFAULT_E_ACTIVE_BANDS_H32: Dict[str, Tuple[int, int]] = {
    "super_sparse": (32, 96),    # k in [2,6] -> 32*2/2=32 to 32*6/2=96
    "sparse": (112, 192),        # k in [7,12] -> 112 to 192
    "moderate": (208, 288),      # k in [13,18] -> 208 to 288
    "near_dense": (304, 416),    # k in [19,26] -> 304 to 416
}


def _k_to_regime(k: int, regime_bins: Dict[str, Tuple[int, int]]) -> Optional[str]:
    k = int(k)
    for name, (lo, hi) in regime_bins.items():
        if lo <= k <= hi:
            return name
    return None


def compute_E_active_from_graph(
    G: nx.Graph,
    wiring_seed: int = 0,
    orientation: str = "random_oriented",
) -> int:
    """
    Count active (non-zero) edges in the oriented hidden block.

    For undirected graphs: E_active = number_of_edges (each edge becomes one
    directed edge when oriented). For directed graphs, counts nonzeros.

    Args:
        G: Hidden graph (H nodes, 0..H-1).
        wiring_seed: Seed for random orientation (used when orientation matters).
        orientation: "random_oriented" (default), "symmetric", or "as_is".

    Returns:
        Number of non-zero entries in the H×H oriented hidden block.
    """
    if G.is_directed():
        return G.number_of_edges()
    # Undirected: each edge -> one directed edge when oriented
    return G.number_of_edges()


def get_E_active_band(regime: str, H: int = 32, regime_bins: Optional[Dict[str, Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
    """Return (E_min, E_max) for the regime. Uses defaults for H=32."""
    if H == 32 and regime in DEFAULT_E_ACTIVE_BANDS_H32:
        return DEFAULT_E_ACTIVE_BANDS_H32[regime]
    regime_bins = regime_bins or {
        "super_sparse": (2, 6),
        "sparse": (7, 12),
        "moderate": (13, 18),
        "near_dense": (19, 26),
    }
    if regime not in regime_bins:
        return None
    lo, hi = regime_bins[regime]
    # E = H*k/2 for undirected WS
    E_min = H * lo // 2
    E_max = H * hi // 2
    return (E_min, E_max)


def capacity_filter(
    G: nx.Graph,
    k: int,
    wiring_seed: int = 0,
    regime_bins: Optional[Dict[str, Tuple[int, int]]] = None,
    H: Optional[int] = None,
) -> Tuple[bool, Optional[str], int]:
    """
    Check if graph passes capacity filter: E_active within regime band.

    Args:
        G: Hidden graph.
        k: Degree parameter (for regime lookup).
        wiring_seed: Unused (E_active = |E| for undirected).
        regime_bins: Optional custom bins.
        H: Node count (default from G).

    Returns:
        (pass, rejection_reason, E_active). pass=True if E_active in band.
    """
    regime_bins = regime_bins or {
        "super_sparse": (2, 6),
        "sparse": (7, 12),
        "moderate": (13, 18),
        "near_dense": (19, 26),
    }
    H = H or G.number_of_nodes()
    regime = _k_to_regime(k, regime_bins)
    if regime is None:
        return True, None, compute_E_active_from_graph(G, wiring_seed)
    band = get_E_active_band(regime, H, regime_bins)
    if band is None:
        return True, None, compute_E_active_from_graph(G, wiring_seed)
    E_active = compute_E_active_from_graph(G, wiring_seed)
    E_min, E_max = band
    if E_min <= E_active <= E_max:
        return True, None, E_active
    return False, f"E_active={E_active} outside [{E_min},{E_max}] for {regime}", E_active
