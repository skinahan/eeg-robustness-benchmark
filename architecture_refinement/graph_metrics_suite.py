"""
Plot 2 M1 — Extended graph metric suite per PLOT 2 Final Overhaul Experiment Spec.

Computes: density, degrees, clustering, path length, sigma, Laplacian spectrum,
spectral measures, effective resistance, centralities, motifs, ORC, TE.
Provides within-k residualization (M1.5).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

_EPS = 1e-9


def _graph_to_adj(G: nx.Graph, n: int) -> np.ndarray:
    """Undirected adjacency A ∈ {0,1}^{n×n}."""
    A = nx.to_numpy_array(G, nodelist=range(n) if G.number_of_nodes() == n else None)
    return (A != 0).astype(np.float64)


def compute_m1_metrics(
    G: nx.Graph,
    *,
    H: int,
    k: int,
    p: float,
    graph_seed: int,
    graph_hash: str,
    orc_alpha: float = 0.5,
    orc_max_edges: Optional[int] = 200,
) -> Dict[str, Any]:
    """
    Compute all M1.1–M1.4 metrics for undirected graph G.
    Returns dict with raw values; residualization done separately.
    """
    n = int(G.number_of_nodes())
    m = int(G.number_of_edges())
    out: Dict[str, Any] = {
        "n": n,
        "m": m,
        "k": k,
        "p": p,
        "graph_seed": graph_seed,
        "graph_hash": graph_hash,
    }

    # M1.1 Basic
    density = float(2 * m / max(1, n * (n - 1)))
    degrees = [G.degree(i) for i in G.nodes()]
    deg_mean = float(np.mean(degrees)) if degrees else float("nan")
    deg_std = float(np.std(degrees)) if len(degrees) > 1 else 0.0
    out["density"] = density
    out["deg_mean"] = deg_mean
    out["deg_std"] = deg_std

    # Clustering C
    try:
        C = float(nx.average_clustering(G))
    except Exception:
        C = float("nan")
    out["C"] = C

    # Path length L
    try:
        if nx.is_connected(G):
            L = float(nx.average_shortest_path_length(G))
        else:
            largest = max(nx.connected_components(G), key=len)
            if len(largest) > 1:
                L = float(nx.average_shortest_path_length(G.subgraph(largest)))
            else:
                L = float("nan")
    except Exception:
        L = float("nan")
    out["L"] = L

    # Small-worldness sigma (delegate to small_world_metrics)
    try:
        from .small_world_metrics import compute_small_worldness
        sigma, _, _, _, _, _ = compute_small_worldness(G, graph_id=graph_hash, use_analytic_er=False)
        out["sigma"] = float(sigma) if np.isfinite(sigma) else float("nan")
    except Exception:
        out["sigma"] = float("nan")

    # Laplacian spectrum: L = D-A, L_norm = I - D^{-1/2} A D^{-1/2}
    try:
        A = _graph_to_adj(G, n)
        D = np.diag(A.sum(axis=1))
        L_mat = D - A
        evals_L = np.linalg.eigvalsh(L_mat)
        evals_L = np.sort(evals_L[evals_L > _EPS])
        algebraic_connectivity = float(evals_L[1]) if len(evals_L) > 1 else 0.0
        out["algebraic_connectivity"] = algebraic_connectivity

        # Normalized Laplacian
        degs = np.diag(D).flatten()
        degs_safe = np.where(degs > 0, degs, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degs_safe))
        L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        evals_Lnorm = np.linalg.eigvalsh(L_norm)
        evals_Lnorm = np.sort(evals_Lnorm[evals_Lnorm > _EPS])
        gap_Lnorm = float(evals_Lnorm[1]) if len(evals_Lnorm) > 1 else 0.0
        out["gap_Lnorm"] = gap_Lnorm
    except Exception:
        out["algebraic_connectivity"] = float("nan")
        out["gap_Lnorm"] = float("nan")

    # Spectral radius and adjacency spectral gap
    try:
        evals_A = np.linalg.eigvals(A)
        evals_A = np.real(evals_A)
        evals_A_abs = np.abs(evals_A)
        spectral_radius = float(np.max(evals_A_abs))
        evals_sorted = np.sort(evals_A)[::-1]
        gap_A = float(evals_sorted[0] - evals_sorted[1]) if len(evals_sorted) > 1 else 0.0
        gap_A_norm = gap_A / max(spectral_radius, _EPS)
        out["spectral_radius"] = spectral_radius
        out["gap_A"] = gap_A
        out["gap_A_norm"] = gap_A_norm
        out["H_spec"] = float(math.log(max(spectral_radius, _EPS)))
    except Exception:
        out["spectral_radius"] = float("nan")
        out["gap_A"] = float("nan")
        out["gap_A_norm"] = float("nan")
        out["H_spec"] = float("nan")

    # Effective resistance, Kirchhoff index
    try:
        from scipy.linalg import pinv2
        L_mat = D - A
        L_pinv = pinv2(L_mat)
        Kf = float(n * np.trace(L_pinv))
        Rbar = float(2 * Kf / max(1, n * (n - 1)))
        out["Kirchhoff_index"] = Kf
        out["Rbar"] = Rbar
        out["Kf_over_n2"] = Kf / max(1, n * n)
    except Exception:
        out["Kirchhoff_index"] = float("nan")
        out["Rbar"] = float("nan")
        out["Kf_over_n2"] = float("nan")

    # Centralities (BC, CC, EC) — summarize mean/std/max and Gini
    try:
        bc = nx.betweenness_centrality(G)
        cc = nx.closeness_centrality(G)
        ec = nx.eigenvector_centrality(G, max_iter=500, tol=1e-6)
        bc_vals = list(bc.values())
        cc_vals = list(cc.values())
        ec_vals = list(ec.values())
        for name, vals in [("BC", bc_vals), ("CC", cc_vals), ("EC", ec_vals)]:
            v = np.array(vals, dtype=float)
            out[f"{name}_mean"] = float(np.mean(v)) if len(v) else float("nan")
            out[f"{name}_std"] = float(np.std(v)) if len(v) > 1 else 0.0
            out[f"{name}_max"] = float(np.max(v)) if len(v) else float("nan")
            out[f"{name}_gini"] = _gini(v) if len(v) and np.sum(v) > 0 else 0.0
    except Exception:
        for name in ["BC", "CC", "EC"]:
            out[f"{name}_mean"] = float("nan")
            out[f"{name}_std"] = float("nan")
            out[f"{name}_max"] = float("nan")
            out[f"{name}_gini"] = float("nan")

    # Motifs: triangles T = tr(A^3)/6
    try:
        A = _graph_to_adj(G, n)
        T = float(np.trace(A @ A @ A) / 6.0)
        c_n3 = n * (n - 1) * (n - 2) / 6.0 if n >= 3 else 1.0
        T_norm = T / max(c_n3, _EPS)
        out["triangles"] = T
        out["triangles_norm"] = T_norm
    except Exception:
        out["triangles"] = float("nan")
        out["triangles_norm"] = float("nan")

    # TE (degree entropy) and ORC
    try:
        from .metrics_te_orc import (
            degree_entropy_raw,
            topological_entropy_te,
            ollivier_ricci_mean,
        )
        H_deg = degree_entropy_raw(G)
        TE = topological_entropy_te(G, clip_0_1=True)
        orc_mean, orc_debug = ollivier_ricci_mean(
            G, alpha=orc_alpha, max_edges=orc_max_edges, return_edge_curvatures=True
        )
        orc_min = float(orc_debug.get("kappa_min", float("nan")))
        out["TE"] = float(TE)
        out["H_deg"] = float(H_deg)
        out["ORC_mean"] = float(orc_mean)
        out["ORC_min"] = float(orc_min) if np.isfinite(orc_min) else float("nan")
    except Exception:
        out["TE"] = float("nan")
        out["H_deg"] = float("nan")
        out["ORC_mean"] = float("nan")
        out["ORC_min"] = float("nan")

    return out


def _gini(x: np.ndarray) -> float:
    """Gini coefficient for inequality."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.sum(x) <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * np.sum(x) + _EPS))


def residualize_metrics(
    df_metrics: "pd.DataFrame",
    metrics_to_residualize: List[str],
    k_col: str = "k",
    n_per_k_min: int = 30,
) -> Tuple["pd.DataFrame", Dict[int, Dict[str, Tuple[float, float]]]]:
    """
    M1.5: Within-k residualization.
    Estimate μ_M(k), σ_M(k) from reference sample; compute M_res, M_z.
    Clip M_z to [-5, 5].
    """
    import pandas as pd
    mu_sigma_by_k: Dict[int, Dict[str, Tuple[float, float]]] = {}
    out = df_metrics.copy()

    for k_val in df_metrics[k_col].dropna().unique():
        k_int = int(k_val)
        sub = df_metrics[df_metrics[k_col] == k_val]
        if len(sub) < n_per_k_min:
            continue
        mu_sigma_by_k[k_int] = {}
        for m in metrics_to_residualize:
            if m not in sub.columns:
                continue
            vals = sub[m].dropna()
            if len(vals) < 2:
                continue
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma < _EPS:
                sigma = _EPS
            mu_sigma_by_k[k_int][m] = (mu, sigma)

    for m in metrics_to_residualize:
        if m not in out.columns:
            continue
        res_col = f"{m}_res"
        z_col = f"{m}_z"
        out[res_col] = float("nan")
        out[z_col] = float("nan")
        for idx, row in out.iterrows():
            k_val = row.get(k_col)
            if k_val is None or not np.isfinite(k_val):
                continue
            k_int = int(k_val)
            ms = mu_sigma_by_k.get(k_int, {}).get(m)
            if ms is None:
                continue
            mu, sigma = ms
            v = row.get(m)
            if v is None or not np.isfinite(v):
                continue
            out.loc[idx, res_col] = v - mu
            z_val = (v - mu) / (sigma + _EPS)
            out.loc[idx, z_col] = float(np.clip(z_val, -5.0, 5.0))

    return out, mu_sigma_by_k
