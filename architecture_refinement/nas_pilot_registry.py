from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx

import config as _cfg
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from models.cnn_wiredcfc_min import create_cnnwiredcfc_min_classifier


def register_nas_pilot_models(pilot_dir: str | Path) -> List[str]:
    """
    Register all NAS pilot models found under:
      <pilot_dir>/selected_architectures/*.json

    Each JSON is expected to include at least:
      - model_name
      - wiring_kind (optional; default: "ws_flex")

    Supported wiring kinds:
      - "ws_flex" (default): requires `hidden_adj_undirected` and `wiring_seed`
      - "ncp_autoncp": requires `units`, `output_size`, `sparsity_level` (optional), and `wiring_seed`
      - "external_random": Plot 2 G3 out-of-family baseline; requires `units`, `output_size`, `sparsity_level`, and `wiring_seed`

    Safety:
    - Will NOT overwrite any existing registry keys.
    """
    pilot_dir = Path(pilot_dir).resolve()
    arch_dir = pilot_dir / "selected_architectures"
    if not arch_dir.exists():
        raise FileNotFoundError(f"NAS pilot selected_architectures dir not found: {arch_dir}")

    arch_files = sorted(arch_dir.glob("*.json"))
    if not arch_files:
        raise FileNotFoundError(f"No architecture JSON files found in: {arch_dir}")

    registered: List[str] = []
    existing = _cfg.get_model_registry()  # includes runtime registry too

    def _make_wiring_from_arch(
        arch_dict: Dict,
        *,
        wiring_kind: str,
        wiring_seed: int,
        arch_path: str,
    ) -> object:
        """Build wiring from an architecture dict. wiring_kind and wiring_seed are required to avoid closure bugs."""
        if wiring_kind == "topology_opt_directed":
            # Directed adjacency from topology optimization (A[i,j]=1 if j→i)
            hidden_adj = arch_dict.get("hidden_adj_directed")
            if hidden_adj is None:
                raise KeyError(
                    f"topology_opt_directed architecture missing 'hidden_adj_directed' (from {arch_path})"
                )
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
            hidden_adj = arch_dict.get("hidden_adj_undirected")
            if hidden_adj is None:
                raise KeyError(
                    f"ws_flex architecture missing 'hidden_adj_undirected' (from {arch_path}). "
                    f"Keys present: {list(arch_dict.keys())}"
                )
            hidden_adj = np.asarray(hidden_adj, dtype=np.int8)
            G = nx.from_numpy_array((hidden_adj != 0).astype(np.int8))
            if not nx.is_connected(G):
                raise ValueError("Hidden graph is disconnected (pilot constraint).")
            orientation = str(arch_dict.get("hidden_edge_orientation", "random_oriented"))
            if orientation not in ("symmetric", "random_oriented", "as_is"):
                orientation = "random_oriented"
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
            from ncps.wirings import AutoNCP  # local import to avoid paying import cost unless needed

            units = int(arch_dict["units"])
            output_size = int(arch_dict["output_size"])
            sparsity_level = float(arch_dict.get("sparsity_level", 0.5))
            return AutoNCP(
                units=units,
                output_size=output_size,
                sparsity_level=sparsity_level,
                seed=int(wiring_seed),
            )

        if wiring_kind == "external_random":
            from ncps.wirings import Random  # Plot 2 G3 out-of-family baseline

            units = int(arch_dict["units"])
            sparsity_level = float(arch_dict["sparsity_level"])
            # WiredCfCCell returns full hidden state (size=units), not output_dim. CfC uses
            # wiring.output_dim for fc input size, so set output_dim=units to avoid shape mismatch.
            return Random(
                units=units,
                output_dim=units,
                sparsity_level=sparsity_level,
                random_seed=int(wiring_seed),
            )

        raise ValueError(f"Unknown wiring_kind={wiring_kind!r} in {arch_path}")

    for p in arch_files:
        with p.open("r", encoding="utf-8") as f:
            arch = json.load(f)

        model_name = str(arch["model_name"])
        if model_name in existing or model_name in _cfg._runtime_model_registry:
            raise ValueError(f"Refusing to overwrite existing model registration: {model_name}")

        wiring_kind = str(arch.get("wiring_kind", "ws_flex"))
        wiring_seed = int(arch["wiring_seed"])

        # Bind per-architecture values as default args so each factory captures its own snapshot.
        arch_snapshot = dict(arch)
        wiring_kind_s = str(wiring_kind)
        wiring_seed_s = int(wiring_seed)
        arch_path_s = str(p)

        def _factory(
            n_chans: int,
            n_times: int,
            n_outputs: int,
            *,
            _arch: Dict = arch_snapshot,
            _kind: str = wiring_kind_s,
            _seed: int = wiring_seed_s,
            _path: str = arch_path_s,
            **kwargs,
        ):
            wiring = _make_wiring_from_arch(
                _arch,
                wiring_kind=_kind,
                wiring_seed=_seed,
                arch_path=_path,
            )
            return create_cnnwiredcfc_min_classifier(
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
                wiring=wiring,
                **kwargs,
            )

        _cfg._runtime_model_registry[model_name] = _factory
        registered.append(model_name)

    return registered

