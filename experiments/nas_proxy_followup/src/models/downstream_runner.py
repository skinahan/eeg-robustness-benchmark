"""Full-budget training (same loop as probe, more epochs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.probe.eeg_layout import infer_eeg_n_channels
from src.probe.probe_runner import build_probe_model, load_graph_for_topology_row, train_probe_loop


def run_downstream_training(
    topology_row: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    repo_root: Path,
    device: torch.device,
    training_seed: int,
    downstream_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    torch.manual_seed(training_seed)
    G = load_graph_for_topology_row(topology_row, repo_root)
    scheme = str(topology_row.get("mapping_scheme", "deterministic_baseline"))
    D_in = int(model_cfg.get("D_in", 16))
    H = int(model_cfg.get("H", 32))
    n_ch = int(model_cfg.get("n_channels", infer_eeg_n_channels(X_train)))
    n_out = int(model_cfg.get("n_outputs", int(np.max(y_train)) + 1))
    model = build_probe_model(
        G,
        n_channels=n_ch,
        D_in=D_in,
        H=H,
        n_outputs=n_out,
        mapping_scheme=scheme,
        wiring_seed=training_seed,
    ).to(device)

    dt = downstream_cfg.get("downstream_training", downstream_cfg)
    epochs = int(dt.get("epochs", 200))
    batch_size = int(dt.get("batch_size", 64))
    lr = float(dt.get("learning_rate", 1e-3))
    return train_probe_loop(
        model,
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        checkpoint_path=checkpoint_path,
    )
