"""
Pure ``torch`` + ``ncps`` copy of ``CNNWiredCfCMin`` (no braindecode/skorch).

Used only for **trainable parameter counting** when the full classifier stack cannot be imported.
Must stay aligned with ``models.cnn_wiredcfc_min.CNNWiredCfCMin`` constructor body.
"""

from __future__ import annotations

import torch
from torch import nn
from ncps.torch import CfC


class CNNWiredCfCMinCore(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        wiring,
        *,
        drop_prob: float = 0.15,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 128,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 4,
        max_seq_length: int = 250,
        mixed_memory: bool = True,
    ) -> None:
        super().__init__()
        _ = n_times  # signature parity with CNNWiredCfCMin; unused for param count

        self.F1 = int(F1)
        F2 = int(F1) * int(D)
        self.kernel_length = int(kernel_length)
        self.max_seq_length = int(max_seq_length)
        self.temporal_stride = int(temporal_stride)
        self.mixed_memory = bool(mixed_memory)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.elu = nn.ELU()

        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1,
            out_channels=F2,
            kernel_size=(n_chans, 1),
            groups=self.F1,
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)

        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))
        self.dropout = nn.Dropout(p=float(drop_prob))

        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=int(temporal_kernel_size),
            stride=int(temporal_stride),
            padding=int(temporal_kernel_size) // 2,
        )

        ncp_input_size = F2
        ncp_output_size = F2
        if hasattr(wiring, "input_size"):
            wiring.input_size = ncp_input_size
        if hasattr(wiring, "output_size"):
            wiring.output_size = ncp_output_size
        wiring_built = wiring.build(ncp_input_size)
        units_for_cfc = wiring_built if wiring_built is not None else wiring

        self.ncp = CfC(
            input_size=ncp_input_size,
            units=units_for_cfc,
            return_sequences=True,
            proj_size=ncp_output_size,
            mixed_memory=self.mixed_memory,
        )

        self.fc = nn.Linear(ncp_output_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 2, 1)
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[3])
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)
        if x.shape[1] > self.max_seq_length:
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx : start_idx + self.max_seq_length, :]
        x, _ = self.ncp(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def count_trainable_params_cnn_wiredcfc_core(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    wiring,
) -> int:
    m = CNNWiredCfCMinCore(n_chans, n_times, n_outputs, wiring)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
