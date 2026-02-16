"""
WS-Flex+ graph generator: plain and modular Watts-Strogatz variants.

Plot2_revision3: single parametric generator family supporting plain_ws_flex
and modular_ws_flex modes for structural diversity without muddying the narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence, Tuple, Any

import numpy as np
import networkx as nx

GeneratorMode = Literal["plain_ws_flex", "modular_ws_flex"]

# Default modular parameter bounds (modular_ws_flex integration spec)
DEFAULT_M_VALUES: Tuple[int, ...] = (2, 4, 8)
DEFAULT_P_OUT_LO, DEFAULT_P_OUT_HI = 0.0, 0.6
DEFAULT_R_OUT_LO, DEFAULT_R_OUT_HI = 0.02, 0.20


@dataclass
class WSFlexParams:
    """Parameters for WS-Flex+ graph generation (identity fields for metrics)."""

    generator_mode: GeneratorMode
    H: int
    k: int  # k_in for modular
    p: float  # p_in for modular
    graph_seed: int
    # Modular-only
    M: Optional[int] = None
    k_out: Optional[int] = None
    p_out: Optional[float] = None
    r_out: Optional[float] = None


def build_plain_ws_flex(H: int, k: int, p: float, seed: int) -> nx.Graph:
    """
    Build plain Watts-Strogatz flex graph (ring lattice + rewiring).

    Args:
        H: Number of nodes (hidden size).
        k: Neighbors per node in initial ring (must be even, 2 <= k <= H-1).
        p: Rewiring probability.
        seed: RNG seed.

    Returns:
        Connected undirected graph with node ids 0..H-1.
    """
    k = max(2, min(int(k), H - 1))
    if k % 2 != 0:
        k = k - 1 if k > 2 else 2
    G = nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))
    return _ensure_connected(G, seed)


def build_modular_ws_flex_graph(
    H: int,
    M: int,
    k_in: int,
    p_in: float,
    k_out: int,
    p_out: float,
    seed: int,
    r_out: Optional[float] = None,
) -> nx.Graph:
    """
    Build modular WS-Flex graph: partition nodes into M modules, WS within each,
    sparse inter-module edges with connectivity repair.

    Args:
        H: Total nodes (hidden size).
        M: Number of modules (default 4 for H=32).
        k_in: Neighbors per node within module (applied per module).
        p_in: Rewiring probability within modules.
        k_out: Inter-module edges per module pair (or use r_out).
        p_out: Rewiring/bias for inter-module edges.
        seed: RNG seed.
        r_out: Optional fixed ratio of inter-module edges vs within-module (overrides k_out if set).

    Returns:
        Connected undirected graph with node ids 0..H-1.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    M = max(1, min(int(M), H))
    if M == 1:
        return build_plain_ws_flex(H, k_in, p_in, seed)

    # Partition nodes into M modules (equal sizes where possible)
    base_size = H // M
    remainder = H % M
    sizes = [base_size + (1 if i < remainder else 0) for i in range(M)]
    offsets = [0] + list(np.cumsum(sizes)[:-1])

    G = nx.Graph()
    G.add_nodes_from(range(H))

    # Within-module: WS per module
    for m in range(M):
        n_m = sizes[m]
        if n_m < 2:
            continue
        k_m = max(2, min(k_in, n_m - 1))
        if k_m % 2 != 0:
            k_m = max(2, k_m - 1)
        sub_seed = seed + 1000 * m
        sub = nx.watts_strogatz_graph(n_m, k_m, p_in, seed=sub_seed)
        base = offsets[m]
        for u, v in sub.edges():
            G.add_edge(base + u, base + v)

    # Inter-module: k_out edges between module pairs (or r_out ratio)
    intra_edges = G.number_of_edges()
    if r_out is not None and r_out > 0:
        n_inter_target = max(1, int(intra_edges * r_out))
    else:
        # k_out per module -> total inter edges ~ k_out * M (simplified)
        n_inter_target = max(M - 1, max(1, k_out) * M)

    attempts = 0
    max_attempts = max(n_inter_target * 20, 100)
    while not nx.is_connected(G) or G.number_of_edges() < intra_edges + max(1, n_inter_target // 2):
        m1, m2 = rng.choice(M, size=2, replace=False)
        if m1 == m2:
            continue
        n1, n2 = sizes[m1], sizes[m2]
        if n1 < 1 or n2 < 1:
            continue
        u = offsets[m1] + rng.integers(0, n1)
        v = offsets[m2] + rng.integers(0, n2)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
        attempts += 1
        if attempts > max_attempts:
            break

    # Connectivity repair: add minimal edges between components
    G = _ensure_connected(G, seed)

    return G


def _ensure_connected(G: nx.Graph, seed: int) -> nx.Graph:
    """Ensure graph is connected; add minimal bridge edges if needed."""
    if nx.is_connected(G):
        return G
    rng = np.random.default_rng(seed)
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1 = list(components[0])
        c2 = list(components[1])
        u = int(rng.choice(c1))
        v = int(rng.choice(c2))
        if u != v:
            G.add_edge(u, v)
        components = list(nx.connected_components(G))
    return G


def sample_modular_params(
    H: int,
    seed_mod_params: int,
    M_values: Sequence[int] = DEFAULT_M_VALUES,
    p_out_lo: float = DEFAULT_P_OUT_LO,
    p_out_hi: float = DEFAULT_P_OUT_HI,
    r_out_lo: float = DEFAULT_R_OUT_LO,
    r_out_hi: float = DEFAULT_R_OUT_HI,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float, float]:
    """
    Sample modular parameters for modular_ws_flex: M, p_out, r_out.

    Args:
        H: Hidden size (used for validation).
        seed_mod_params: RNG seed for reproducibility.
        M_values: Allowed module counts, e.g. (2, 4, 8).
        p_out_lo, p_out_hi: Inter-module rewiring range.
        r_out_lo, r_out_hi: Inter-module edge ratio range.
        rng: Optional external RNG; if None, create from seed_mod_params.

    Returns:
        (M, p_out, r_out)
    """
    r = rng if rng is not None else np.random.default_rng(int(seed_mod_params))
    M_vals = tuple(int(x) for x in M_values)
    M = int(r.choice(M_vals))
    M = max(1, min(M, H))
    p_out = float(r.uniform(p_out_lo, p_out_hi))
    r_out = float(r.uniform(r_out_lo, r_out_hi))
    return M, p_out, r_out


def make_ws_flex_graph(
    H: int,
    k: int,
    p: float,
    seed: int,
    generator_mode: GeneratorMode = "plain_ws_flex",
    M: Optional[int] = None,
    k_out: Optional[int] = None,
    p_out: Optional[float] = None,
    r_out: Optional[float] = None,
    seed_mod_params: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[nx.Graph, WSFlexParams]:
    """
    Generate WS-Flex+ graph by mode. Returns (graph, params) for metrics output.

    Args:
        H: Hidden size.
        k: Degree parameter (k_in for modular).
        p: Rewiring (p_in for modular).
        seed: RNG seed for graph construction.
        generator_mode: plain_ws_flex or modular_ws_flex.
        M: Modules (modular only; sampled if None).
        k_out: Inter-module edges (modular; used only when r_out not set).
        p_out: Inter-module rewiring (modular; sampled [0, 0.6] if None).
        r_out: Inter-module edge ratio (modular; sampled [0.02, 0.20] if None; preferred over k_out).
        seed_mod_params: RNG seed for sampling M, p_out, r_out when not provided.
        rng: Optional external RNG for modular sampling.

    Returns:
        (Graph, WSFlexParams) with identity fields for metrics.
    """
    if generator_mode == "plain_ws_flex":
        G = build_plain_ws_flex(H, k, p, seed)
        return G, WSFlexParams(
            generator_mode="plain_ws_flex",
            H=H,
            k=k,
            p=p,
            graph_seed=seed,
        )

    # modular_ws_flex: sample M, p_out, r_out if not all provided
    need_sampling = M is None or p_out is None or (r_out is None or r_out <= 0)
    if need_sampling:
        sm_seed = int(seed_mod_params) if seed_mod_params is not None else int(seed)
        M_s, p_out_s, r_out_s = sample_modular_params(H, sm_seed, rng=rng)
        M = M if M is not None else M_s
        p_out = p_out if p_out is not None else p_out_s
        r_out = r_out if (r_out is not None and r_out > 0) else r_out_s
    if M is None:
        M = 4 if H >= 32 else min(4, max(1, H // 8))

    M = max(1, min(int(M), H))
    k_out_val = k_out if k_out is not None else 2
    use_r_out = float(r_out) if (r_out is not None and r_out > 0) else None

    G = build_modular_ws_flex_graph(
        H=H,
        M=M,
        k_in=k,
        p_in=p,
        k_out=k_out_val,
        p_out=p_out,
        seed=seed,
        r_out=use_r_out,
    )
    return G, WSFlexParams(
        generator_mode="modular_ws_flex",
        H=H,
        k=k,
        p=p,
        graph_seed=seed,
        M=M,
        k_out=k_out_val,
        p_out=float(p_out),
        r_out=use_r_out,
    )
