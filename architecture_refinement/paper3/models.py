"""
Task-agnostic models for PAPER 3: CfC (wired), NCP, LSTM.

All share: Encoder E (Linear C->D_in), Readout (Linear H->2).
Forward contract: forward(x, return_states=False) -> (logits,) or (logits, states) with states [B,T,H].
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Tuple, Union
import networkx as nx

# Add repo root to path when running as script
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ncps.torch import CfC
from ncps.wirings import AutoNCP
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _BaseRecurrentModel(nn.Module):
    """Base class with shared encoder and readout."""

    def __init__(
        self,
        C: int = 1,
        D_in: int = 16,
        H: int = 32,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.C = C
        self.D_in = D_in
        self.H = H
        self.n_outputs = n_outputs
        self.encoder = nn.Linear(C, D_in)
        self.readout = nn.Linear(H, n_outputs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T, D_in]"""
        return self.encoder(x)


class CfCRecurrentModel(_BaseRecurrentModel):
    """
    CfC with wired topology (WS-Flex graph).
    Supports return_states=True for dynamics metrics.
    """

    def __init__(
        self,
        C: int = 1,
        D_in: int = 16,
        H: int = 32,
        n_outputs: int = 2,
        hidden_graph: Optional[nx.Graph] = None,
        wiring_seed: int = 42,
    ):
        super().__init__(C=C, D_in=D_in, H=H, n_outputs=n_outputs)
        if hidden_graph is None:
            raise ValueError("CfCRecurrentModel requires hidden_graph (WS-Flex graph)")

        wiring = WsFlexHiddenWiring(
            input_size=D_in,
            hidden_graph=hidden_graph,
            output_size=H,
            input_strategy="dense",
            output_strategy="dense",
            hidden_edge_orientation="symmetric",
            seed=wiring_seed,
        )
        wiring_built = wiring.build(D_in)
        self.cfc = CfC(
            input_size=D_in,
            units=wiring_built,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            mode="default",
        )

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, T, C]
        Returns (logits,) or (logits, states) with states [B, T, H].
        """
        encoded = self.encode(x)
        output, _ = self.cfc(encoded)
        states = output
        h_last = states[:, -1, :]
        logits = self.readout(h_last)

        if return_states:
            return logits, states
        return logits


class NCPRecurrentModel(_BaseRecurrentModel):
    """
    NCP (AutoNCP) with capacity-matched structure.
    Uses custom unroll to expose full hidden states for dynamics metrics.
    """

    def __init__(
        self,
        C: int = 1,
        D_in: int = 16,
        H: int = 32,
        n_outputs: int = 2,
        ncp_units: int = 28,
        sparsity_level: float = 0.85,
        ncp_seed: int = 42,
    ):
        super().__init__(C=C, D_in=D_in, H=H, n_outputs=n_outputs)
        self.ncp_units = ncp_units
        self.wiring = AutoNCP(ncp_units, n_outputs, sparsity_level=sparsity_level, seed=ncp_seed)
        self.cfc = CfC(
            input_size=D_in,
            units=self.wiring,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            mode="default",
        )
        self._hidden_size = self.wiring.units
        motor_dim = self.wiring.output_dim
        self.readout = nn.Linear(motor_dim, n_outputs)

    def _forward_with_states(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unroll CfC cell manually to collect full hidden states [B, T, H]."""
        device = encoded.device
        B, T, _ = encoded.shape
        cell = self.cfc.rnn_cell
        h_state = torch.zeros((B, cell.state_size), device=device)
        states_list = []
        for t in range(T):
            inputs = encoded[:, t, :]
            ts = 1.0
            h_out, h_state = cell.forward(inputs, h_state, ts)
            states_list.append(h_state)
        states = torch.stack(states_list, dim=1)
        motor_dim = self.wiring.output_dim
        h_motor = states[:, -1, -motor_dim:]
        return h_motor, states

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, T, C]
        Returns (logits,) or (logits, states) with states [B, T, H].
        """
        encoded = self.encode(x)
        if return_states:
            h_motor, states = self._forward_with_states(encoded)
        else:
            output, _ = self.cfc(encoded)
            h_motor = output[:, -1, :]
            states = output
        logits = self.readout(h_motor)

        if return_states:
            return logits, states
        return logits


class LSTMRecurrentModel(_BaseRecurrentModel):
    """
    Single-layer LSTM with capacity-matched hidden size.
    """

    def __init__(
        self,
        C: int = 1,
        D_in: int = 16,
        H: int = 32,
        n_outputs: int = 2,
    ):
        super().__init__(C=C, D_in=D_in, H=H, n_outputs=n_outputs)
        self.lstm = nn.LSTM(
            input_size=D_in,
            hidden_size=H,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, T, C]
        Returns (logits,) or (logits, states) with states [B, T, H].
        """
        encoded = self.encode(x)
        output, (h_n, c_n) = self.lstm(encoded)
        states = output
        h_last = states[:, -1, :]
        logits = self.readout(h_last)

        if return_states:
            return logits, states
        return logits


class CNNBaselineModel(_BaseRecurrentModel):
    """
    Simple feedforward 1D CNN baseline. Uses same encoder (Linear C->D_in) and readout
    (Linear H->2) as recurrent models for fairness. Replaces recurrence with Conv1d layers.
    Supports return_states for dynamics (returns conv feature maps as pseudo-states).
    """

    def __init__(
        self,
        C: int = 1,
        D_in: int = 16,
        H: int = 32,
        n_outputs: int = 2,
        conv_channels: Optional[Tuple[int, ...]] = None,
        kernel_size: int = 7,
    ):
        super().__init__(C=C, D_in=D_in, H=H, n_outputs=n_outputs)
        if conv_channels is None:
            conv_channels = (D_in, 32, H)
        layers = []
        in_ch = conv_channels[0]
        for out_ch in conv_channels[1:]:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self._hidden_size = H

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, T, C]
        Returns (logits,) or (logits, states) with states [B, T, H] for dynamics compatibility.
        """
        encoded = self.encode(x)
        x_conv = encoded.transpose(1, 2)
        features = self.conv(x_conv)
        h_pooled = features.mean(dim=2)
        logits = self.readout(h_pooled)

        if return_states:
            states = features.transpose(1, 2)
            return logits, states
        return logits
