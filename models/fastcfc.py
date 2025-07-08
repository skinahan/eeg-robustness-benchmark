import torch
import torch.nn as nn
from ncps.torch import WiredCfCCell
from typing import Optional

class FastCfC(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        proj_size: Optional[int] = None,
        batch_first: bool = True,
        return_sequences: bool = False,  # set to False for speed
    ):
        super().__init__()
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.input_size = input_size
        self.state_size = wiring.units
        self.output_size = wiring.output_dim
        self.rnn_cell = WiredCfCCell(input_size, wiring, mode="default")

        self.fc = nn.Identity() if proj_size is None else nn.Linear(self.output_size, proj_size)

    def forward(self, x, hx=None):
        """
        Args:
            x: Tensor [B, T, F]
        Returns:
            output: [B, output_size] if return_sequences=False
                    [B, T, output_size] if return_sequences=True
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # T, B, F → B, T, F

        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.state_size, device=x.device) if hx is None else hx

        if self.return_sequences:
            outputs = []

        for t in range(seq_len):
            xt = x[:, t]
            h_out, h = self.rnn_cell(xt, h, ts=1.0)

            if self.return_sequences:
                outputs.append(self.fc(h_out))

        if self.return_sequences:
            return torch.stack(outputs, dim=1), h
        else:
            return self.fc(h_out), h
