"""
Small-worldness metrics for Plot 2 (Waqas-aligned).

- C(G): average local clustering coefficient
- L(G): average shortest path length (connected graphs only)
- σ(G): small-worldness = (C(G)/C_ER) / (L(G)/L_ER)

ER reference: Monte Carlo by default (connected ER graphs matched to p_edge).
Cache: (p_edge_bin, H) -> (C_ER_mean, L_ER_mean) for reuse across 5k+ candidates.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import networkx as nx

# Default parameters (configurable)
R_ER_DEFAULT = 20
MAX_ATTEMPTS_ER_DEFAULT = 1000  # Sparse ER graphs often disconnected; need more attempts
P_EDGE_BIN_WIDTH_DEFAULT = 0.01

# Module-level cache: (p_edge_bin_key, H) -> (C_ER_mean, L_ER_mean)
_ER_CACHE: Dict[Tuple[str, int], Tuple[float, float]] = {}


def _get_clustering_and_path_length(G: nx.Graph) -> Tuple[float, float]:
    """Compute C(G) and L(G) on undirected graph. Returns (C, L)."""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, 0.0
    C = float(nx.average_clustering(G))
    if nx.is_connected(G):
        L = float(nx.average_shortest_path_length(G))
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > 1:
            subgraph = G.subgraph(largest_cc)
            L = float(nx.average_shortest_path_length(subgraph))
        else:
            L = 0.0
    return C, L


def _p_edge(G: nx.Graph) -> float:
    """Edge probability for undirected graph: 2|E| / (H(H-1))."""
    n = int(G.number_of_nodes())
    m = int(G.number_of_edges())
    if n <= 1:
        return 0.0
    return float(2 * m) / float(n * (n - 1))


def _p_edge_to_bin_key(p_edge: float, bin_width: float = P_EDGE_BIN_WIDTH_DEFAULT) -> str:
    """Discretize p_edge into bin key for cache."""
    if not np.isfinite(p_edge) or p_edge <= 0.0:
        return "0"
    bin_idx = int(np.floor(p_edge / bin_width))
    return str(bin_idx)


def _sample_connected_er(
    n: int,
    p_edge: float,
    rng: np.random.Generator,
    max_attempts: int = MAX_ATTEMPTS_ER_DEFAULT,
) -> Optional[nx.Graph]:
    """Sample one connected ER graph. Returns None if max_attempts exceeded."""
    for _ in range(max_attempts):
        G = nx.erdos_renyi_graph(n, p_edge, seed=int(rng.integers(0, 2**31 - 1)))
        if nx.is_connected(G):
            return G
    return None


def _er_analytic(p_edge: float, n: int) -> Tuple[float, float]:
    """Analytic ER approximations: C_ER ≈ p_edge, L_ER ≈ log(n)/log(n*p_edge)."""
    import math
    C_ER = float(p_edge)
    Hp = float(n) * p_edge
    if Hp > 1.0:
        L_ER = float(math.log(n) / math.log(Hp))
    else:
        L_ER = float(n)  # fallback for very sparse
    return C_ER, L_ER


def _compute_er_reference_monte_carlo(
    n: int,
    p_edge: float,
    r_er: int = R_ER_DEFAULT,
    max_attempts: int = MAX_ATTEMPTS_ER_DEFAULT,
    seed_base: Optional[str] = None,
    fallback_analytic: bool = True,
) -> Tuple[float, float]:
    """
    Compute C_ER and L_ER from R_ER connected ER graphs matched to p_edge.
    Uses fixed seeds derived from (seed_base, "ER_ref", i).
    Returns (C_ER_mean, L_ER_mean).
    If fallback_analytic and Monte Carlo fails, use analytic approximation.
    """
    if seed_base is None:
        seed_base = f"p{p_edge:.6f}_n{n}"
    C_vals: list = []
    L_vals: list = []
    for i in range(r_er):
        h = hashlib.sha256(f"{seed_base}::ER_ref::{i}".encode()).hexdigest()
        seed = int(h[:16], 16) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        G = _sample_connected_er(n, p_edge, rng, max_attempts)
        if G is None:
            if fallback_analytic:
                return _er_analytic(p_edge, n)
            raise RuntimeError(
                f"Failed to sample {r_er} connected ER graphs for p_edge={p_edge:.6f}, n={n} "
                f"(max_attempts={max_attempts} per draw). Mark σ invalid."
            )
        c, l = _get_clustering_and_path_length(G)
        C_vals.append(c)
        L_vals.append(l)
    return float(np.mean(C_vals)), float(np.mean(L_vals))


def get_er_reference(
    G: nx.Graph,
    *,
    use_cache: bool = True,
    bin_width: float = P_EDGE_BIN_WIDTH_DEFAULT,
    r_er: int = R_ER_DEFAULT,
    max_attempts: int = MAX_ATTEMPTS_ER_DEFAULT,
    graph_id: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Get (C_ER, L_ER) for graph G at matched density.
    Uses cache by (p_edge_bin, H) when use_cache=True.
    graph_id used for deterministic ER seeds when not cached.
    """
    n = int(G.number_of_nodes())
    p_edge = _p_edge(G)
    if n <= 1 or not np.isfinite(p_edge) or p_edge <= 0.0:
        return 0.0, 0.0

    bin_key = _p_edge_to_bin_key(p_edge, bin_width)
    cache_key = (bin_key, n)
    if use_cache and cache_key in _ER_CACHE:
        return _ER_CACHE[cache_key]

    seed_base = graph_id if graph_id is not None else f"p{p_edge:.6f}_n{n}"
    C_ER, L_ER = _compute_er_reference_monte_carlo(
        n, p_edge, r_er=r_er, max_attempts=max_attempts, seed_base=seed_base
    )
    if use_cache:
        _ER_CACHE[cache_key] = (C_ER, L_ER)
    return C_ER, L_ER


def compute_small_worldness(
    G: nx.Graph,
    *,
    use_analytic_er: bool = False,
    use_cache: bool = True,
    bin_width: float = P_EDGE_BIN_WIDTH_DEFAULT,
    r_er: int = R_ER_DEFAULT,
    max_attempts: int = MAX_ATTEMPTS_ER_DEFAULT,
    graph_id: Optional[str] = None,
) -> Tuple[float, float, float, float, float, Dict[str, Any]]:
    """
    Compute small-worldness σ(G) = (C(G)/C_ER) / (L(G)/L_ER).

    Returns:
        sigma, C, L, C_ER, L_ER, debug_dict

    Default: Monte Carlo ER reference (connected graphs). Analytic allowed only as
    speed option and must be flagged in manifest.
    """
    n = int(G.number_of_nodes())
    debug: Dict[str, Any] = {"n": n, "use_analytic_er": use_analytic_er}

    if n <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0, debug

    C, L = _get_clustering_and_path_length(G)
    p_edge = _p_edge(G)
    debug["C"] = float(C)
    debug["L"] = float(L)
    debug["p_edge"] = float(p_edge)

    if use_analytic_er:
        # Analytic approximations (speed option)
        C_ER = float(p_edge)
        Hp = float(n) * p_edge
        if Hp > 1.0:
            import math
            L_ER = float(math.log(n) / math.log(Hp))
        else:
            L_ER = float(n)  # fallback for very sparse
        debug["er_method"] = "analytic"
    else:
        C_ER, L_ER = get_er_reference(
            G,
            use_cache=use_cache,
            bin_width=bin_width,
            r_er=r_er,
            max_attempts=max_attempts,
            graph_id=graph_id,
        )
        debug["er_method"] = "monte_carlo"
        debug["r_er"] = r_er

    debug["C_ER"] = float(C_ER)
    debug["L_ER"] = float(L_ER)

    # σ = (C/C_ER) / (L/L_ER)
    if C_ER <= 0.0 or L_ER <= 0.0 or L <= 0.0:
        sigma = float("nan")
        debug["sigma_invalid_reason"] = "zero_denominator"
        return float(sigma), C, L, C_ER, L_ER, debug

    sigma = float((C / C_ER) / (L / L_ER))
    if not np.isfinite(sigma) or sigma < 0.0:
        debug["sigma_invalid_reason"] = "non_finite_or_negative"
    return sigma, C, L, C_ER, L_ER, debug


def clear_er_cache() -> None:
    """Clear the ER reference cache (for testing or fresh runs)."""
    global _ER_CACHE
    _ER_CACHE = {}
