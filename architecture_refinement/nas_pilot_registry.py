from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import config as _cfg
from architecture_refinement.pilot_architecture_wiring import build_wiring_from_architecture_dict
from models.cnn_wiredcfc_min import create_cnnwiredcfc_min_classifier


def register_nas_pilot_models(
    pilot_dir: str | Path,
    *,
    default_hidden_edge_orientation: str | None = None,
) -> List[str]:
    """
    Register all NAS pilot models found under:
      <pilot_dir>/selected_architectures/*.json

    Each JSON is expected to include at least:
      - model_name
      - wiring_kind (optional; default: "ws_flex")
      - hidden_edge_orientation (optional; default from default_hidden_edge_orientation or "random_oriented")

    default_hidden_edge_orientation: When set (e.g. "symmetric" for Paper 3), used when
      JSON lacks "hidden_edge_orientation". Paper 3 spec requires bidirectional wiring.

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

    for p in arch_files:
        with p.open("r", encoding="utf-8") as f:
            arch = json.load(f)

        model_name = str(arch["model_name"])
        if model_name in existing or model_name in _cfg._runtime_model_registry:
            raise ValueError(f"Refusing to overwrite existing model registration: {model_name}")

        # Bind per-architecture values as default args so each factory captures its own snapshot.
        arch_snapshot = dict(arch)
        arch_path_s = str(p)

        def _factory(
            n_chans: int,
            n_times: int,
            n_outputs: int,
            *,
            _arch: Dict = arch_snapshot,
            _path: str = arch_path_s,
            **kwargs,
        ):
            wiring = build_wiring_from_architecture_dict(
                _arch,
                default_hidden_edge_orientation=default_hidden_edge_orientation,
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

