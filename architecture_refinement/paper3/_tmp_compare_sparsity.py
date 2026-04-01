"""One-off: compare G4 random sparse vs G5 NCP edge counts (Paper3 defaults)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.ws_flex_generator import build_plain_ws_flex

H = 32
max_undirected = H * (H - 1) // 2


def random_symmetric_adjacency(H: int, n_edges: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(H) for j in range(i + 1, H)]
    n_edges = min(n_edges, len(pairs))
    chosen = rng.choice(len(pairs), size=n_edges, replace=False)
    A = np.zeros((H, H), dtype=np.int8)
    for idx in chosen:
        i, j = pairs[int(idx)]
        A[i, j] = 1
        A[j, i] = 1
    return A


def main():
    from ncps.wirings import AutoNCP

    # G5 NCP (default arch)
    w = AutoNCP(units=H, output_size=16, sparsity_level=0.5, seed=202603)
    w.build(16)
    A = np.asarray(w.adjacency_matrix)
    A_bool = (A != 0) | (A.T != 0)
    G_ncp = nx.from_numpy_array(A_bool.astype(np.int8))
    ncp_edges = G_ncp.number_of_edges()
    ncp_nnz = int(np.count_nonzero(A))

    # G1-like E_active distribution (same K_VALUES as exp2)
    K_VALUES = list(range(2, 25, 2))
    rng = np.random.default_rng(202607)
    E_actives = []
    for _ in range(2000):
        k = int(rng.choice(K_VALUES))
        p = float(rng.uniform(0.0, 1.0))
        gs = int(rng.integers(0, 2**31 - 1))
        Gg = build_plain_ws_flex(H, k, p, gs)
        if nx.is_connected(Gg):
            E_actives.append(2 * Gg.number_of_edges())

    avg_E = int(np.mean(E_actives))
    n_u = avg_E // 2
    A_rs = random_symmetric_adjacency(H, n_u, 202608)
    G_rs = nx.from_numpy_array(A_rs)
    rs_edges = G_rs.number_of_edges()

    print("Paper3 defaults: H=%d, max undirected edges C(H,2)=%d" % (H, max_undirected))
    print()
    print("G5 NCP (AutoNCP): units=H, output_size=16, sparsity_level=0.5, wiring_seed=202603")
    print("  adjacency_matrix shape:", A.shape)
    print("  nonzero entries (directed weights):", ncp_nnz)
    print("  undirected simple graph edges (symmetrized):", ncp_edges)
    print("  density |E|/C(H,2):", ncp_edges / max_undirected)
    print()
    print("G4 Random sparse: E_active = mean(G1 E_active) over connected WS-Flex samples")
    print("  mean E_active (approx G1 avg):", avg_E)
    print("  undirected edges n_edges = E_active//2:", n_u)
    print("  actual unique undirected (graph):", rs_edges)
    print("  density |E|/C(H,2):", rs_edges / max_undirected)
    print()
    print("Interpretation:")
    print("  - G4 sparsity is *matched to G1 edge budget* (random ER on pairs), not a fixed hyperparameter.")
    print("  - G5 uses ncps AutoNCP sparsity_level=0.5 (internal wiring algorithm), unrelated to G1 |E|.")


if __name__ == "__main__":
    main()
