"""Metrics comparing raw graphs and realized CfC wirings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def pairwise_mask_hamming_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = (mask_a != 0).astype(np.int8).ravel()
    b = (mask_b != 0).astype(np.int8).ravel()
    if a.size != b.size:
        raise ValueError("Mask size mismatch")
    return float(np.mean(a != b))


def compute_realization_descriptors(realization: Dict[str, Any]) -> Dict[str, Any]:
    W = realization["full_wiring_matrix"]
    I = int(realization["input_size"])
    H = int(realization["hidden_size"])
    O = int(realization["output_size"])
    Whh = realization["hidden_mask"]
    density_h = float(np.count_nonzero(Whh) / max(H * H, 1))
    in_a = realization.get("input_anchor_nodes") or []
    out_a = realization.get("output_anchor_nodes") or []
    G = nx.DiGraph()
    G.add_nodes_from(range(I + H + O))
    rows, cols = np.nonzero(W > 0)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c))
    reach_io = []
    for i in range(I):
        for o in range(I + H, I + H + O):
            try:
                d = nx.shortest_path_length(G, i, o)
                reach_io.append(d)
            except nx.NetworkXNoPath:
                pass
    avg_path_in_out = float(np.mean(reach_io)) if reach_io else float("nan")
    return {
        "density_hidden_block": density_h,
        "n_input_anchors": len(set(in_a)),
        "n_output_anchors": len(set(out_a)),
        "avg_shortest_path_input_to_output": avg_path_in_out,
    }


def _adj_vec(G: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.float32)
    iu = np.triu_indices(A.shape[0], k=1)
    return A[iu]


def raw_graph_distance_distribution(graphs: List[nx.Graph]) -> np.ndarray:
    n = len(graphs)
    if n < 2:
        return np.array([])
    vecs = [_adj_vec(g) for g in graphs]
    L = len(vecs[0])
    X = np.stack(vecs, axis=0)
    dists = pdist(X, metric="hamming")
    return dists


def compute_realization_diversity_summary(
    topology_manifest_df: pd.DataFrame,
    realization_manifest_df: pd.DataFrame,
    scheme_col: str = "mapping_scheme",
    mask_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Correlate raw vs realized pairwise distances and count near-duplicate realizations."""
    out: Dict[str, Any] = {}
    for scheme, sub in realization_manifest_df.groupby(scheme_col):
        masks = []
        for _, row in sub.iterrows():
            p = row.get("realized_hidden_mask_path")
            if p and pd.notna(p):
                pp = Path(str(p))
                if mask_root is not None and not pp.is_absolute():
                    pp = mask_root / pp
                if not pp.exists():
                    continue
                m = np.load(pp)["hidden_mask"]
                masks.append((m != 0).astype(np.int8).ravel())
        if len(masks) < 2:
            out[str(scheme)] = {"n": len(masks), "mean_hamming_pdist": None}
            continue
        X = np.stack(masks, axis=0)
        d = pdist(X, metric="hamming")
        out[str(scheme)] = {
            "n": len(masks),
            "mean_hamming_pdist": float(np.mean(d)),
            "std_hamming_pdist": float(np.std(d)),
        }
    return out
