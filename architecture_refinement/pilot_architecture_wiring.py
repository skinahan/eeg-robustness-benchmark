"""
Build ncps wirings from NAS pilot architecture JSON dicts (same logic as nas_pilot_registry).

Kept separate from ``nas_pilot_registry`` so tooling can avoid importing ``config`` / braindecode.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import networkx as nx

from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring


def build_wiring_from_architecture_dict(
    arch: Dict[str, Any],
    *,
    default_hidden_edge_orientation: Optional[str] = "symmetric",
    arch_path: str = "<dict>",
) -> object:
    """
    Return a wiring object for ``CfC`` / ``create_cnnwiredcfc_min_classifier``.

    Supported ``wiring_kind``: ``ws_flex``, ``ncp_autoncp``, ``external_random`` (same as registry).
    """
    wiring_kind = str(arch.get("wiring_kind", "ws_flex"))
    wiring_seed = int(arch["wiring_seed"])

    if wiring_kind == "topology_opt_directed":
        hidden_adj = arch.get("hidden_adj_directed")
        if hidden_adj is None:
            raise KeyError(f"topology_opt_directed missing hidden_adj_directed ({arch_path})")
        A = np.asarray(hidden_adj, dtype=np.float32)
        return WsFlexHiddenWiring(
            input_size=1,
            hidden_graph=A,
            output_size=int(A.shape[0]),
            input_strategy="degree_proportional",
            output_strategy="uniform",
            hidden_edge_orientation="as_is",
            add_hidden_self_loops=True,
            seed=0,
        )

    if wiring_kind == "ws_flex":
        hidden_adj = arch.get("hidden_adj_undirected")
        if hidden_adj is None:
            raise KeyError(f"ws_flex missing hidden_adj_undirected ({arch_path})")
        hidden_adj = np.asarray(hidden_adj, dtype=np.int8)
        G = nx.from_numpy_array((hidden_adj != 0).astype(np.int8))
        if not nx.is_connected(G):
            raise ValueError("Hidden graph is disconnected (pilot constraint).")
        orientation = str(
            arch.get("hidden_edge_orientation", default_hidden_edge_orientation or "random_oriented")
        )
        if orientation not in ("symmetric", "random_oriented", "as_is"):
            orientation = default_hidden_edge_orientation or "random_oriented"
        return WsFlexHiddenWiring(
            input_size=1,
            hidden_graph=G,
            output_size=1,
            input_strategy="degree_proportional",
            output_strategy="uniform",
            hidden_edge_orientation=orientation,
            add_hidden_self_loops=True,
            seed=int(wiring_seed),
        )

    if wiring_kind == "ncp_autoncp":
        from ncps.wirings import AutoNCP

        units = int(arch["units"])
        output_size = int(arch["output_size"])
        sparsity_level = float(arch.get("sparsity_level", 0.5))
        return AutoNCP(
            units=units,
            output_size=output_size,
            sparsity_level=sparsity_level,
            seed=int(wiring_seed),
        )

    if wiring_kind == "external_random":
        from ncps.wirings import Random

        units = int(arch["units"])
        sparsity_level = float(arch["sparsity_level"])
        return Random(
            units=units,
            output_dim=units,
            sparsity_level=sparsity_level,
            random_seed=int(wiring_seed),
        )

    raise ValueError(f"Unknown wiring_kind={wiring_kind!r} in {arch_path}")
