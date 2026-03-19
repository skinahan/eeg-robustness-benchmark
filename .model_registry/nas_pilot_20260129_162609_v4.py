from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import networkx as nx

import config as _cfg
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from models.cnn_wiredcfc_min import create_cnnwiredcfc_min_classifier


_RUN_ID = '20260129_162609_v4'
_PILOT_DIR = Path('E:/Research/Dissertation/full_backup_7_16_2025/moabb_experiments/architecture_refinement/outputs/nas_pilot/20260129_162609_v4')


def _load_arch_jsons() -> list[dict]:
    paths = sorted((_PILOT_DIR / "selected_architectures").glob("*.json"))
    out = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _make_wiring(hidden_adj_undirected: np.ndarray, wiring_seed: int):
    H = int(hidden_adj_undirected.shape[0])
    G = nx.from_numpy_array((hidden_adj_undirected != 0).astype(np.int8))
    if not nx.is_connected(G):
        raise ValueError("Hidden graph is disconnected (pilot constraint).")
    return WsFlexHiddenWiring(
        input_size=1,
        hidden_graph=G,
        output_size=1,
        input_strategy="degree_proportional",
        output_strategy="uniform",
        hidden_edge_orientation="random_oriented",
        add_hidden_self_loops=True,
        seed=int(wiring_seed),
    )


def _register_one(model_name: str, arch: dict) -> None:
    hidden_adj = np.asarray(arch["hidden_adj_undirected"], dtype=np.int8)
    wiring_seed = int(arch["wiring_seed"])

    def _factory(n_chans: int, n_times: int, n_outputs: int, **kwargs):
        wiring = _make_wiring(hidden_adj, wiring_seed=wiring_seed)
        return create_cnnwiredcfc_min_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring,
            **kwargs,
        )

    # Safety: never overwrite an existing model key (pilot must not clobber prior optimizations).
    if model_name in _cfg.get_model_registry():
        raise ValueError(f"Refusing to overwrite existing model registration: {model_name}")
    _cfg._runtime_model_registry[model_name] = _factory


for arch in _load_arch_jsons():
    _register_one(arch["model_name"], arch)
