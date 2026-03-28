"""Short probe training with CfC + WS-Flex wiring (Paper 3 style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ncps.torch import CfC

from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from src.wiring.cfc_realizer import SCHEME_TO_WIRING_KWARGS


def _load_graph_npz(path: Path) -> nx.Graph:
    data = np.load(path, allow_pickle=True)
    A = np.asarray(data["adjacency"])
    return nx.from_numpy_array(A)


class CfCProbeModel(nn.Module):
    """Encoder -> Wired CfC -> last-step readout."""

    def __init__(
        self,
        n_channels: int,
        D_in: int,
        H: int,
        n_outputs: int,
        built_wiring,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(n_channels, D_in)
        self.cfc = CfC(
            input_size=D_in,
            units=built_wiring,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            mode="default",
        )
        self.readout = nn.Linear(H, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        out, _ = self.cfc(enc)
        return self.readout(out[:, -1, :])


def build_probe_model(
    graph: nx.Graph,
    *,
    n_channels: int,
    D_in: int,
    H: int,
    n_outputs: int,
    mapping_scheme: str,
    wiring_seed: int,
    hidden_edge_orientation: str = "symmetric",
) -> CfCProbeModel:
    wkwargs = dict(
        SCHEME_TO_WIRING_KWARGS.get(
            mapping_scheme, SCHEME_TO_WIRING_KWARGS["deterministic_baseline"]
        )
    )
    wkwargs["hidden_edge_orientation"] = hidden_edge_orientation
    wiring = WsFlexHiddenWiring(
        input_size=D_in,
        hidden_graph=graph,
        output_size=H,
        seed=wiring_seed,
        **wkwargs,
    )
    built = wiring.build(D_in)
    return CfCProbeModel(n_channels, D_in, H, n_outputs, built)


def train_probe_loop(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    X: [N, T, C] float32; y: [N] int labels.
    """
    if X.ndim == 3 and X.shape[1] < X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
    X_t = torch.from_numpy(np.ascontiguousarray(X)).float()
    y_t = torch.from_numpy(np.ascontiguousarray(y)).long()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: List[Dict[str, Any]] = []
    model.train()
    for ep in range(epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        history.append({"epoch": ep + 1, "loss": float(np.mean(losses))})

    ckpt = None
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_path)
        ckpt = str(checkpoint_path)
    return {"history": history, "checkpoint_path": ckpt}


def load_graph_for_topology_row(topology_row: Dict[str, Any], repo_root: Path) -> nx.Graph:
    gp = topology_row.get("graph_path")
    if not gp:
        raise ValueError("topology_row missing graph_path")
    gpath = Path(str(gp))
    if not gpath.is_absolute():
        cand = repo_root / "experiments" / "nas_proxy_followup" / gp
        if cand.exists():
            gpath = cand
        else:
            gpath = repo_root / gp
    if not gpath.exists():
        raise FileNotFoundError(gpath)
    return _load_graph_npz(gpath)
