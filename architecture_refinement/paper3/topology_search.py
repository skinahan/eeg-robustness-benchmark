"""
NAS / Topology search for PAPER 3 CfC regimes.

R1: Proxy-guided TPE NAS
R2: Random selection
R3: Random + proxy filter
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.ws_flex_generator import make_ws_flex_graph
from architecture_refinement.config import default_config
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.metrics_te_orc import compute_te_orc
from architecture_refinement.small_world_metrics import compute_small_worldness


@dataclass
class GraphCandidate:
    """A candidate graph with proxy metrics."""
    k: int
    p: float
    graph_seed: int
    wiring_seed: int
    G: nx.Graph
    te: float
    sigma: float
    proxy_score: float


def _compute_proxy(analyzer: TopologyAnalyzer, G: nx.Graph, k: int, p: float, graph_seed: int) -> Tuple[float, float, float]:
    """Compute TE, sigma, and combined proxy score (TE + sigma, both in [0,1])."""
    te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
    topo = analyzer.analyze_graph(G)
    C = float(topo.get("clustering_coefficient", 0.0))
    L = float(topo.get("avg_path_length", 1.0))

    import hashlib
    adj = nx.to_numpy_array(G)
    adj_hex = np.asarray(adj).tobytes().hex()
    gh = hashlib.sha256(f"{G.number_of_nodes()}|{k}|{p}|{graph_seed}|{adj_hex}".encode()).hexdigest()
    sigma, _, _, _, _, _ = compute_small_worldness(G, graph_id=gh, use_analytic_er=False)

    sigma = float(sigma) if np.isfinite(sigma) else 0.0
    sigma = np.clip(sigma, 0.0, 1.0)
    te = np.clip(float(te), 0.0, 1.0)
    proxy_score = te + sigma
    return te, sigma, proxy_score


def _make_graph(H: int, k: int, p: float, graph_seed: int, rng: np.random.Generator) -> nx.Graph:
    """Create connected WS-Flex graph."""
    G, _ = make_ws_flex_graph(H=H, k=k, p=p, seed=int(graph_seed), generator_mode="plain_ws_flex")
    if not nx.is_connected(G):
        raise ValueError("Graph not connected")
    return G


def run_random_search(
    H: int,
    k_values: List[int],
    B_evals: int,
    K: int,
    base_seed: int,
    analyzer: Optional[TopologyAnalyzer] = None,
) -> Tuple[List[GraphCandidate], List[GraphCandidate]]:
    """
    R2: Sample B_evals graphs uniformly, select top K by proxy.

    Returns (all_candidates, selected_top_K).
    """
    analyzer = analyzer or TopologyAnalyzer(default_config)
    rng = np.random.default_rng(base_seed)
    candidates: List[GraphCandidate] = []

    for _ in range(B_evals):
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        wiring_seed = int(rng.integers(0, 2**31 - 1))
        try:
            G = _make_graph(H, k, p, graph_seed, rng)
        except ValueError:
            continue
        te, sigma, proxy_score = _compute_proxy(analyzer, G, k, p, graph_seed)
        candidates.append(GraphCandidate(
            k=k, p=p, graph_seed=graph_seed, wiring_seed=wiring_seed,
            G=G, te=te, sigma=sigma, proxy_score=proxy_score,
        ))

    candidates.sort(key=lambda c: c.proxy_score, reverse=True)
    selected = candidates[:K]
    return candidates, selected


def run_tpe_search(
    H: int,
    k_values: List[int],
    B_evals: int,
    K: int,
    base_seed: int,
    analyzer: Optional[TopologyAnalyzer] = None,
) -> Tuple[List[GraphCandidate], List[GraphCandidate]]:
    """
    R1: TPE suggests next graph, select top K by proxy.

    Returns (all_candidates, selected_top_K).
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna required for TPE: pip install optuna")

    analyzer = analyzer or TopologyAnalyzer(default_config)
    rng = np.random.default_rng(base_seed)
    candidates: List[GraphCandidate] = []

    def objective(trial: optuna.Trial) -> float:
        k = int(trial.suggest_categorical("k", k_values))
        p = float(trial.suggest_float("p", 0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        wiring_seed = int(rng.integers(0, 2**31 - 1))
        try:
            G = _make_graph(H, k, p, graph_seed, rng)
        except ValueError:
            raise optuna.TrialPruned()
        te, sigma, proxy_score = _compute_proxy(analyzer, G, k, p, graph_seed)
        candidates.append(GraphCandidate(
            k=k, p=p, graph_seed=graph_seed, wiring_seed=wiring_seed,
            G=G, te=te, sigma=sigma, proxy_score=proxy_score,
        ))
        return proxy_score

    sampler = optuna.samplers.TPESampler(seed=base_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=B_evals, show_progress_bar=False)

    candidates.sort(key=lambda c: c.proxy_score, reverse=True)
    selected = candidates[:K]
    return candidates, selected


def run_random_filter_search(
    H: int,
    k_values: List[int],
    M: int,
    q_percent: float,
    K: int,
    base_seed: int,
    analyzer: Optional[TopologyAnalyzer] = None,
) -> Tuple[List[GraphCandidate], List[GraphCandidate]]:
    """
    R3: Sample M graphs, keep top q% by proxy, choose K from that subset.

    Returns (all_candidates, selected_K).
    """
    analyzer = analyzer or TopologyAnalyzer(default_config)
    rng = np.random.default_rng(base_seed)
    candidates: List[GraphCandidate] = []

    for _ in range(M):
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        wiring_seed = int(rng.integers(0, 2**31 - 1))
        try:
            G = _make_graph(H, k, p, graph_seed, rng)
        except ValueError:
            continue
        te, sigma, proxy_score = _compute_proxy(analyzer, G, k, p, graph_seed)
        candidates.append(GraphCandidate(
            k=k, p=p, graph_seed=graph_seed, wiring_seed=wiring_seed,
            G=G, te=te, sigma=sigma, proxy_score=proxy_score,
        ))

    candidates.sort(key=lambda c: c.proxy_score, reverse=True)
    q_count = max(1, int(len(candidates) * q_percent / 100))
    top_q = candidates[:q_count]
    if len(top_q) <= K:
        selected = top_q
    else:
        idx = rng.choice(len(top_q), size=K, replace=False)
        selected = [top_q[i] for i in idx]
        selected = sorted(selected, key=lambda c: c.proxy_score, reverse=True)
    return candidates, selected
