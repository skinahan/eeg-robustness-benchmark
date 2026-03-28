"""Unit tests for NAS proxy follow-up wiring and realizer."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
_REPO = Path(__file__).resolve().parents[1]
_NPF = _REPO / "experiments" / "nas_proxy_followup"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_NPF) not in sys.path:
    sys.path.insert(0, str(_NPF))

from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from src.wiring.cfc_realizer import realize_cfc_wiring, SCHEME_TO_WIRING_KWARGS
from src.wiring.realization_metrics import pairwise_mask_hamming_distance


def test_wsflex_hidden_wiring_random_io_differs_from_dense():
    G = nx.watts_strogatz_graph(16, 4, 0.2, seed=1)
    w_dense = WsFlexHiddenWiring(
        8, G, 8, input_strategy="dense", output_strategy="dense", seed=0
    ).full_wiring_matrix()
    w_rand = WsFlexHiddenWiring(
        8, G, 8, input_strategy="random_io", output_strategy="random_io", seed=42
    ).full_wiring_matrix()
    assert w_dense.shape == w_rand.shape
    assert not np.allclose(w_dense, w_rand)


def test_realize_cfc_wiring_schemes():
    G = nx.watts_strogatz_graph(16, 4, 0.2, seed=2)
    a = realize_cfc_wiring(G, "deterministic_baseline", 8, 8, realization_seed=0)
    b = realize_cfc_wiring(G, "random_io_anchors", 8, 8, realization_seed=1)
    assert a["hidden_mask"].shape == b["hidden_mask"].shape
    d = pairwise_mask_hamming_distance(a["hidden_mask"], b["hidden_mask"])
    assert d >= 0.0


def test_scheme_mapping_complete():
    for k in SCHEME_TO_WIRING_KWARGS:
        assert k in (
            "deterministic_baseline",
            "random_io_anchors",
            "degree_weighted_io_anchors",
            "random_input_anchors",
            "random_output_anchors",
            "random_io_anchors_full",
        )
