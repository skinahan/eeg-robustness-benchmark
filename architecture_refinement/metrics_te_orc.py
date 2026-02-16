"""
Canonical TE / ORC definitions (Waqas et al., 2022 alignment).

- TE: exact Shannon entropy of the empirical degree distribution, normalized by log(N)
- ORC: signed mean Ollivier–Ricci curvature computed via Wasserstein-1 optimal transport

This module exists to prevent metric definition drift across runners/analyzers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx


_LP_AEQ_CACHE: Dict[Tuple[int, int], Any] = {}


def degree_entropy_raw(G: nx.Graph) -> float:
    """
    Exact Shannon entropy (nats) of the empirical degree distribution.

    H_deg(G) = - sum_k p(k) log p(k), where p(k)=#{i:deg(i)=k}/N
    """
    n = int(G.number_of_nodes())
    if n <= 1:
        return 0.0

    degrees = np.fromiter((d for _, d in G.degree()), dtype=float)
    if degrees.size == 0:
        return 0.0

    _, counts = np.unique(degrees, return_counts=True)
    p = counts.astype(float) / float(n)
    # p entries are strictly >0; safe to take log(p)
    return float(-np.sum(p * np.log(p)))


def topological_entropy_te(G: nx.Graph, *, norm_const: Optional[float] = None, clip_0_1: bool = True) -> float:
    """
    Normalized topological entropy:

      TE(G) = H_deg(G) / C, with canonical choice C = log(N) (natural log)
    """
    n = int(G.number_of_nodes())
    if n <= 1:
        return 0.0

    H = degree_entropy_raw(G)
    C = float(norm_const) if norm_const is not None else float(math.log(n))
    if not np.isfinite(C) or C <= 0.0:
        te = 0.0
    else:
        te = float(H / C)

    if clip_0_1:
        te = float(np.clip(te, 0.0, 1.0))
    return te


def _support_and_measure(
    G: nx.Graph, node: Any, *, alpha: float
) -> Tuple[List[Any], np.ndarray]:
    """
    Ollivier neighbor measure m_node used for ORC:
    - mass alpha at the node itself
    - remaining mass (1-alpha) distributed uniformly over 1-hop neighbors
    """
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    # Use a stable ordering that doesn't assume node comparability.
    nbrs = sorted(G.neighbors(node), key=repr)
    deg = len(nbrs)

    if deg == 0:
        support = [node]
        mu = np.array([1.0], dtype=float)
        return support, mu

    support = [node] + nbrs
    mu = np.empty((len(support),), dtype=float)
    mu[0] = a
    mu[1:] = (1.0 - a) / float(deg)

    # Numerical hygiene
    mu = np.clip(mu, 0.0, 1.0)
    s = float(mu.sum())
    if s <= 0.0:
        mu = np.array([1.0], dtype=float)
        support = [node]
    else:
        mu = mu / s
    return support, mu


def ollivier_ricci_mean(
    G: nx.Graph,
    *,
    alpha: float = 0.5,
    max_edges: Optional[int] = None,
    return_edge_curvatures: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Signed mean Ollivier–Ricci curvature over edges:

      kappa(u,v) = 1 - W1(m_u, m_v) / d(u,v)
      ORC(G) = mean_{(u,v) in E} kappa(u,v)

    Uses exact Wasserstein-1 distance:
    - Prefer POT (fast) if installed
    - Otherwise solve the exact LP via SciPy/HiGHS (slower but dependency-light)
    """
    def _w1_cost(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> float:
        """
        Compute Wasserstein-1 (Earth Mover's) distance between two discrete measures.

        Prefer POT if available; otherwise fall back to an exact LP via SciPy (HiGHS).
        """
        try:  # fast path
            import ot  # type: ignore

            return float(ot.emd2(a, b, M))
        except Exception:
            from scipy.optimize import linprog  # type: ignore
            from scipy.sparse import lil_matrix  # type: ignore

            m, n = int(M.shape[0]), int(M.shape[1])
            c = M.reshape(-1).astype(float)

            # Equality constraints: row sums == a, col sums == b
            key = (m, n)
            A_eq = _LP_AEQ_CACHE.get(key)
            if A_eq is None:
                A = lil_matrix((m + n, m * n), dtype=float)
                for i in range(m):
                    A[i, i * n : (i + 1) * n] = 1.0
                for j in range(n):
                    A[m + j, j::n] = 1.0
                A_eq = A.tocsr()
                _LP_AEQ_CACHE[key] = A_eq
            b_eq = np.concatenate([a.astype(float), b.astype(float)], axis=0)

            res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
            if not res.success or res.fun is None:
                raise RuntimeError(f"linprog failed computing W1: {res.message}")
            return float(res.fun)

    edges = list(G.edges())
    if not edges:
        debug: Dict[str, Any] = {
            "alpha": float(alpha),
            "num_edges": 0,
            "orc_edge_count": 0,
        }
        if return_edge_curvatures:
            debug["edge_curvatures"] = {}
        return 0.0, debug

    # Deterministic ordering (useful for debugging/reproducibility)
    # Stable ordering without relying on node comparability.
    edges = [tuple(sorted((u, v), key=repr)) for (u, v) in edges]
    edges = sorted(set(edges), key=lambda x: (repr(x[0]), repr(x[1])))

    if max_edges is not None and len(edges) > int(max_edges):
        # Evenly subsample to keep the estimate stable across runs.
        idxs = np.linspace(0, len(edges) - 1, int(max_edges), dtype=int)
        edges = [edges[i] for i in idxs.tolist()]

    # All-pairs shortest path lengths (N=32 in our target use-cases)
    dist = dict(nx.all_pairs_shortest_path_length(G))

    edge_curvatures: Dict[Tuple[Any, Any], float] = {}
    kappas: List[float] = []

    for (u, v) in edges:
        d_uv = float(dist[u][v])
        if not np.isfinite(d_uv) or d_uv <= 0.0:
            continue

        sup_u, mu_u = _support_and_measure(G, u, alpha=alpha)
        sup_v, mu_v = _support_and_measure(G, v, alpha=alpha)

        # Cost matrix between supports based on graph distance.
        M = np.zeros((len(sup_u), len(sup_v)), dtype=float)
        for i, su in enumerate(sup_u):
            du = dist.get(su, {})
            for j, sv in enumerate(sup_v):
                M[i, j] = float(du.get(sv, np.inf))

        if not np.all(np.isfinite(M)):
            # Should not happen for edges in the same component, but be safe.
            continue

        # Exact Wasserstein-1 distance
        w1 = float(_w1_cost(mu_u, mu_v, M))
        kappa = float(1.0 - (w1 / d_uv))
        kappas.append(kappa)
        if return_edge_curvatures:
            edge_curvatures[(u, v)] = kappa

    if kappas:
        orc = float(np.mean(kappas))
        orc_std = float(np.std(kappas))
        k_min = float(np.min(kappas))
        k_max = float(np.max(kappas))
    else:
        orc = 0.0
        orc_std = 0.0
        k_min = 0.0
        k_max = 0.0

    debug = {
        "alpha": float(alpha),
        "num_edges": int(G.number_of_edges()),
        "orc_edge_count": int(len(kappas)),
        "orc_std": float(orc_std),
        "kappa_min": float(k_min),
        "kappa_max": float(k_max),
    }
    if return_edge_curvatures:
        debug["edge_curvatures"] = edge_curvatures
    return orc, debug


def compute_te_orc(
    G: nx.Graph,
    *,
    te_norm_const: Optional[float] = None,
    te_clip_0_1: bool = True,
    orc_alpha: float = 0.5,
    orc_max_edges: Optional[int] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper returning (te, orc, debug_dict).
    """
    H = degree_entropy_raw(G)
    te = topological_entropy_te(G, norm_const=te_norm_const, clip_0_1=te_clip_0_1)
    orc, orc_debug = ollivier_ricci_mean(G, alpha=orc_alpha, max_edges=orc_max_edges, return_edge_curvatures=False)

    debug: Dict[str, Any] = {
        "n": int(G.number_of_nodes()),
        "m": int(G.number_of_edges()),
        "degree_entropy_raw": float(H),
        "te_norm_const": float(te_norm_const) if te_norm_const is not None else float(math.log(max(2, G.number_of_nodes()))),
        "orc_alpha": float(orc_alpha),
    }
    debug.update({f"orc_{k}": v for k, v in orc_debug.items()})
    return float(te), float(orc), debug


def compute_orc_residual(
    orc: float,
    k: int,
    mu_orc_by_k: Dict[int, float],
) -> float:
    """
    Residualized ORC: ORC_res(G) = ORC(G) - μ_ORC(k(G)).
    mu_orc_by_k comes from G0 reference set (frozen in manifest).
    """
    if not np.isfinite(orc):
        return float("nan")
    mu = mu_orc_by_k.get(int(k))
    if mu is None or not np.isfinite(mu):
        return float(orc)  # no lookup; return raw (or could return nan)
    return float(orc - mu)


def compute_te_residual(
    te: float,
    k: int,
    mu_te_by_k: Dict[int, float],
) -> float:
    """
    Residualized TE: TE_res(G) = TE(G) - μ_TE(k(G)).
    mu_te_by_k from same G0 reference set as bin edges and μ_ORC(k).
    """
    if not np.isfinite(te):
        return float("nan")
    mu = mu_te_by_k.get(int(k))
    if mu is None or not np.isfinite(mu):
        return float(te)
    return float(te - mu)


def _normalize_mu_by_k(d: Any) -> Dict[int, float]:
    """Convert manifest dict (string keys from JSON) to Dict[int, float]."""
    if d is None:
        return {}
    out: Dict[int, float] = {}
    for key, val in d.items():
        try:
            out[int(key)] = float(val)
        except (ValueError, TypeError):
            continue
    return out


def compute_orc_residual_from_lookup(
    orc: float,
    k: int,
    lookup: Dict[str, Any],
) -> float:
    """
    Manifest-driven ORC_res lookup.
    lookup expects key "mu_orc_by_k" -> Dict (int or str keys -> float).

    Args:
        orc: Raw ORC value.
        k: Degree parameter.
        lookup: Manifest or dict with mu_orc_by_k.

    Returns:
        ORC_res.
    """
    mu_orc_by_k = _normalize_mu_by_k(lookup.get("mu_orc_by_k"))
    if not mu_orc_by_k:
        return float(orc)
    return compute_orc_residual(orc, k, mu_orc_by_k)


def compute_te_residual_from_lookup(
    te: float,
    k: int,
    lookup: Dict[str, Any],
) -> float:
    """
    Manifest-driven TE_res lookup.
    lookup expects key "mu_te_by_k" -> Dict (int or str keys -> float).
    """
    mu_te_by_k = _normalize_mu_by_k(lookup.get("mu_te_by_k"))
    if not mu_te_by_k:
        return float(te)
    return compute_te_residual(te, k, mu_te_by_k)

