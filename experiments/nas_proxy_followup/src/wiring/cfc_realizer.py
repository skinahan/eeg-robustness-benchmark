"""Map abstract WS-Flex graphs to CfC wiring under named realization schemes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import networkx as nx

from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring

MappingScheme = Literal[
    "deterministic_baseline",
    "random_io_anchors",
    "degree_weighted_io_anchors",
    "random_input_anchors",
    "random_output_anchors",
    "random_io_anchors_full",
]

# Paper 3 / NAS-style dense I/O + symmetric hidden block
SCHEME_TO_WIRING_KWARGS: Dict[str, Dict[str, Any]] = {
    "deterministic_baseline": {
        "input_strategy": "dense",
        "output_strategy": "dense",
        "hidden_edge_orientation": "symmetric",
    },
    "random_io_anchors": {
        "input_strategy": "random_io",
        "output_strategy": "random_io",
        "hidden_edge_orientation": "symmetric",
    },
    "degree_weighted_io_anchors": {
        "input_strategy": "degree_weighted_io",
        "output_strategy": "degree_weighted_io",
        "hidden_edge_orientation": "symmetric",
    },
    "random_input_anchors": {
        "input_strategy": "random_io",
        "output_strategy": "dense",
        "hidden_edge_orientation": "symmetric",
    },
    "random_output_anchors": {
        "input_strategy": "dense",
        "output_strategy": "random_io",
        "hidden_edge_orientation": "symmetric",
    },
    "random_io_anchors_full": {
        "input_strategy": "random_io",
        "output_strategy": "random_io",
        "hidden_edge_orientation": "symmetric",
    },
}


def _io_anchors_from_wiring_matrix(W: np.ndarray, I: int, H: int, O: int) -> Tuple[List[int], List[int]]:
    """Heuristic: input-anchored hiddens = columns in I->H block with any edge; output-anchored = rows in H->O."""
    Whi = W[0:I, I : I + H]
    input_touch = [int(h) for h in range(H) if np.any(Whi[:, h] != 0)]
    Who = W[I : I + H, I + H : I + H + O]
    output_touch = [int(h) for h in range(H) if np.any(Who[h, :] != 0)]
    return input_touch, output_touch


def realize_cfc_wiring(
    graph: Union[nx.Graph, np.ndarray],
    scheme: str,
    input_size: int,
    output_size: int,
    realization_seed: Optional[int] = None,
    hidden_edge_orientation: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build full wiring matrix and extract hidden recurrent mask + I/O anchor sets.

    Returns keys: hidden_mask (H,H), full_wiring_matrix, input_anchor_nodes, output_anchor_nodes,
    wiring_kwargs, scheme, realization_seed.
    """
    if scheme not in SCHEME_TO_WIRING_KWARGS:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {list(SCHEME_TO_WIRING_KWARGS)}")
    kwargs = dict(SCHEME_TO_WIRING_KWARGS[scheme])
    if hidden_edge_orientation is not None:
        kwargs["hidden_edge_orientation"] = hidden_edge_orientation
    seed = int(realization_seed if realization_seed is not None else 0)
    w = WsFlexHiddenWiring(
        input_size=int(input_size),
        hidden_graph=graph,
        output_size=int(output_size),
        seed=seed,
        **kwargs,
    )
    W = w.full_wiring_matrix()
    I, H, O = int(input_size), w._hidden_size(), int(output_size)
    hidden_mask = W[I : I + H, I : I + H].copy()
    in_a, out_a = _io_anchors_from_wiring_matrix(W, I, H, O)
    return {
        "hidden_mask": hidden_mask,
        "full_wiring_matrix": W,
        "input_anchor_nodes": in_a,
        "output_anchor_nodes": out_a,
        "wiring_kwargs": {**kwargs, "seed": seed},
        "scheme": scheme,
        "realization_seed": seed,
        "hidden_size": H,
        "input_size": I,
        "output_size": O,
    }
