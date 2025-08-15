import torch
import torch.nn as nn
from ncps.torch import WiredCfCCell
from typing import Optional
import torch.nn.functional as F

"""
This file contains multiple classes that seek to improve the performance of CfC-basd models.

ImprovedFastCfC : FastCfC with aggressive regularization and reduced complexity.
It includes:
- Temporal windowing to limit input sequence length
- Aggressive dropout (40-50%)
- Weight decay and gradient clipping
- Reduced hidden state size
- Early stopping mechanism
"""

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
            h_out, h = self.rnn_cell(xt, h, timespans=1.0)

            if self.return_sequences:
                outputs.append(self.fc(h_out))

        if self.return_sequences:
            return torch.stack(outputs, dim=1), h
        else:
            return self.fc(h_out), h



class TemporalWindow(nn.Module):
    """Learned temporal window to limit input sequence length."""
    
    def __init__(self, max_seq_len: int, learnable: bool = True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.learnable = learnable
        
        if learnable:
            # Learnable start and end positions
            self.start_pos = nn.Parameter(torch.tensor(0.0))
            self.window_size = nn.Parameter(torch.tensor(float(max_seq_len)))
        else:
            self.start_pos = 0
            self.window_size = max_seq_len
    
    def forward(self, x):
        """
        Args:
            x: Tensor [B, T, F]
        Returns:
            windowed_x: Tensor [B, window_size, F]
        """
        batch_size, seq_len, features = x.shape
        
        if self.learnable:
            # Clamp to valid ranges
            start = torch.clamp(self.start_pos, 0, seq_len - 1)
            window_size = torch.clamp(self.window_size, 1, min(seq_len, self.max_seq_len))
            # Ensure end is at least start + 1 and at most seq_len
            # Convert seq_len to tensor to match tensor types
            seq_len_tensor = torch.tensor(seq_len, device=start.device, dtype=start.dtype)
            end = torch.clamp(start + window_size, min=start + 1, max=seq_len_tensor)
            
            # Convert to integers for indexing
            start_idx = start.long()
            end_idx = end.long()
        else:
            start_idx = 0
            end_idx = min(seq_len, self.max_seq_len)
        
        # Apply temporal window
        windowed_x = x[:, start_idx:end_idx]
        
        # Pad if necessary
        if windowed_x.size(1) < self.max_seq_len:
            padding = torch.zeros(batch_size, self.max_seq_len - windowed_x.size(1), features, device=x.device)
            windowed_x = torch.cat([windowed_x, padding], dim=1)
        
        return windowed_x

class ImprovedFastCfC(nn.Module):
    """
    Enhanced FastCfC with aggressive regularization and reduced complexity.
    
    Key improvements:
    - Input normalization with LayerNorm
    - Temporal windowing to limit sequence length
    - Aggressive dropout (40-50%)
    - Weight decay and gradient clipping
    - Reduced hidden state size
    - Early stopping mechanism
    """
    
    def __init__(
        self,
        input_size: int,
        wiring,
        proj_size: Optional[int] = None,
        batch_first: bool = True,
        return_sequences: bool = False,
        dropout_rate: float = 0.4,
        max_seq_len: int = 500,  # Reduced from typical 1000+
        use_temporal_window: bool = True,
        use_input_norm: bool = True,
        hidden_dropout: float = 0.3,
        output_dropout: float = 0.5,
        weight_decay: float = 1e-3,
        gradient_clip: float = 0.5,
    ):
        super().__init__()
        
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.input_size = input_size
        self.state_size = wiring.units
        self.output_size = wiring.output_dim
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.use_temporal_window = use_temporal_window
        self.use_input_norm = use_input_norm
        self.hidden_dropout = hidden_dropout
        self.output_dropout = output_dropout
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        # Input normalization
        if use_input_norm:
            self.input_norm = nn.LayerNorm(input_size)
        else:
            self.input_norm = nn.Identity()
        
        # Temporal windowing
        if use_temporal_window:
            self.temporal_window = TemporalWindow(max_seq_len, learnable=True)
        else:
            self.temporal_window = nn.Identity()
        
        # Reduced complexity CfC cell
        self.rnn_cell = WiredCfCCell(input_size, wiring, mode="default")
        
        # Aggressive dropout layers
        self.hidden_dropout_layer = nn.Dropout(hidden_dropout)
        self.output_dropout_layer = nn.Dropout(output_dropout)
        
        # Simplified projection layer with regularization
        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.output_size, proj_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(proj_size, proj_size)
            )
        
        # Early stopping mechanism
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 10
        
        # Initialize weights with smaller values for regularization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for better regularization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
        
        # Input normalization
        x = self.input_norm(x)
        
        # Apply temporal windowing
        if self.use_temporal_window:
            x = self.temporal_window(x)
        
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.state_size, device=x.device) if hx is None else hx
        
        if self.return_sequences:
            outputs = []
        
        # Process sequence with aggressive dropout
        for t in range(seq_len):
            xt = x[:, t]
            
            # Apply dropout to input
            xt = self.hidden_dropout_layer(xt)
            
            h_out, h = self.rnn_cell(xt, h, timespans=1.0)
            
            # Apply dropout to hidden state
            h = self.hidden_dropout_layer(h)
            
            if self.return_sequences:
                outputs.append(self.fc(h_out))
        
        # Final output with aggressive dropout
        final_output = self.output_dropout_layer(h_out)
        
        if self.return_sequences:
            return torch.stack(outputs, dim=1), h
        else:
            return self.fc(final_output), h
    
    def get_regularization_loss(self):
        """Compute L2 regularization loss for weight decay."""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.weight_decay * l2_loss
    
    def clip_gradients(self):
        """Apply gradient clipping."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
    
    def early_stopping(self, current_loss):
        """Early stopping mechanism."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False  # Continue training
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience  # Stop training

class AggressiveFastCfCClassifier(nn.Module):






    """
    Classifier wrapper for ImprovedFastCfC with aggressive regularization.
    """
    
    def __init__(
        self,
        input_size: int,
        wiring,
        num_classes: int,
        hidden_size: int = 32,  # Reduced from 64
        dropout_rate: float = 0.4,
        max_seq_len: int = 500,
        use_temporal_window: bool = True,
        use_input_norm: bool = True,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        
        # Improved FastCfC backbone
        self.fastcfc = ImprovedFastCfC(
            input_size=input_size,
            wiring=wiring,
            proj_size=hidden_size,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len,
            use_temporal_window=use_temporal_window,
            use_input_norm=use_input_norm,
            hidden_dropout=0.3,
            output_dropout=0.5,
            weight_decay=1e-3,
            gradient_clip=0.5,
        )
        
        # Simplified classifier head with aggressive regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Label smoothing for regularization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, x):
        """
        Args:
            x: Tensor [B, T, F] or [B, F, T]
        Returns:
            logits: Tensor [B, num_classes]
        """
        # Handle different input formats
        if x.dim() == 3:
            if x.size(1) == self.input_size:  # [B, F, T]
                x = x.transpose(1, 2)  # [B, T, F]
        else:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        
        # Forward through FastCfC
        features, _ = self.fastcfc(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """Compute loss with regularization."""
        # Cross-entropy loss
        ce_loss = self.criterion(logits, targets)
        
        # L2 regularization loss
        reg_loss = self.fastcfc.get_regularization_loss()
        
        return ce_loss + reg_loss 
# optimized_cfc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np
import ncps

try:
    compile_fn = torch.compile  # PyTorch 2.x
except Exception:
    compile_fn = None


# ----- Utility activation to match your implementation -----
class LeCun(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


def _make_backbone(input_size, hidden_size, backbone_activation, backbone_units, backbone_layers, backbone_dropout):
    if backbone_activation == "silu":
        act_cls = nn.SiLU
    elif backbone_activation == "relu":
        act_cls = nn.ReLU
    elif backbone_activation == "tanh":
        act_cls = nn.Tanh
    elif backbone_activation == "gelu":
        act_cls = nn.GELU
    elif backbone_activation == "lecun_tanh":
        act_cls = LeCun
    else:
        raise ValueError(f"Unknown activation {backbone_activation}")

    cat_shape = input_size + hidden_size
    if backbone_layers <= 0:
        return None, cat_shape

    layers = [nn.Linear(cat_shape, backbone_units), act_cls()]
    for _ in range(1, backbone_layers):
        layers += [nn.Linear(backbone_units, backbone_units), act_cls()]
        if backbone_dropout and backbone_dropout > 0.0:
            layers += [nn.Dropout(backbone_dropout)]
    return nn.Sequential(*layers), backbone_units


# ----- Fused CfC cell: single GEMM for all gates per time step -----
class CfCCellFused(nn.Module):
    """
    Functionally equivalent to your CfCCell for modes {"default","no_gate","pure"}
    but computes all gate preactivations in ONE linear (GEMM) per step.

    - Respects the original sparsity_mask behavior (applied to ff1/ff2 only).
    - Keeps numerics and shapes compatible (so you can swap it in).
    - Quantization-friendly (single nn.Linear).
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
        sparsity_mask=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        assert mode in {"default", "no_gate", "pure"}
        self.mode = mode

        # Backbone (same as your code)
        self.backbone, cat_shape = _make_backbone(
            input_size, hidden_size, backbone_activation, backbone_units, backbone_layers, backbone_dropout
        )
        self.backbone_layers = backbone_layers

        # We keep the same activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Sparsity mask (your code transposes to (H, cat))
        self.sparsity_mask = None
        if sparsity_mask is not None:
            m = torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32))
            self.register_buffer("sparsity_mask", m, persistent=False)  # (H, cat)

        # Build fused linear(s)
        if self.mode == "pure":
            # Only uses ff1-equivalent to compute the exponent term
            self.gates = nn.Linear(cat_shape, hidden_size)
            # Learnable A and w_tau like your code
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size))
            self.A = nn.Parameter(torch.ones(1, hidden_size))
            # Optional mask: match original behavior for ff1 only
            if self.sparsity_mask is not None:
                self.register_buffer("fused_mask_pure", self.sparsity_mask.clone(), persistent=False)  # (H, cat)
        else:
            # default/no_gate: need [ff1, ff2, time_a, time_b] => 4 * H outputs
            self.gates = nn.Linear(cat_shape, 4 * hidden_size)
            if self.sparsity_mask is not None:
                # Apply mask to ff1 and ff2 rows; leave time_a/time_b unmasked (original behavior)
                ones = torch.ones_like(self.sparsity_mask)
                fused_mask = torch.cat([self.sparsity_mask, self.sparsity_mask, ones, ones], dim=0)  # (4H, cat)
                self.register_buffer("fused_mask", fused_mask, persistent=False)

        self._init_weights()

    def _init_weights(self):
        # Match your xavier init for 2D params
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        # x: concat once, backbone once
        x = torch.cat([input, hx], dim=1)  # (B, C+H)
        if self.backbone is not None:
            x = self.backbone(x)  # (B, cat_shape_after_backbone)

        if self.mode == "pure":
            if hasattr(self, "fused_mask_pure"):
                pre_ff1 = F.linear(x, self.gates.weight * self.fused_mask_pure, self.gates.bias)
            else:
                pre_ff1 = self.gates(x)
            # Same closed-form update as your code
            new_hidden = -self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(pre_ff1))) * pre_ff1 + self.A
            return new_hidden, new_hidden

        # default / no_gate
        if hasattr(self, "fused_mask"):
            pre = F.linear(x, self.gates.weight * self.fused_mask, self.gates.bias)  # (B, 4H)
        else:
            pre = self.gates(x)

        ff1, ff2, t_a, t_b = pre.chunk(4, dim=1)
        ff1 = self.tanh(ff1)
        ff2 = self.tanh(ff2)
        if isinstance(ts, (float, int)):
            ts = t_a.new_full((t_a.size(0), 1), float(ts))   # (B,1)
        elif ts.ndim == 1:
            ts = ts.unsqueeze(1)                              # (B,1)
        elif ts.ndim == 2 and ts.size(1) != 1:
            # if someone passed (B,H), reduce to a single timescale per sample
            ts = ts.mean(dim=1, keepdim=True)

        t_interp = self.sigmoid(t_a * ts + t_b)

        if self.mode == "no_gate":
            new_hidden = ff1 + t_interp * ff2
        else:
            new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, new_hidden


# ----- Optimized CfC wrapper (compiler/AMP friendly scan) -----
class CfCOptimized(nn.Module):
    """
    Drop-in CfC with:
      - time-major normalized scan
      - preallocated outputs
      - optional AMP during inference
      - friendly to torch.compile(fullgraph=True)

    You can choose the cell class: CfCCellFused (fast) or your original CfCCell.
    """
    def __init__(
        self,
        input_size: Union[int, ncps.wirings.Wiring],
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[float] = None,
        use_amp: bool = False,
        cell_cls: Optional[nn.Module] = None,   # set to CfCCellFused by default for dense mode
        sparsity_mask=None,                      # optional mask forwarded to cell
    ):
        super().__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.use_amp = use_amp

        # Defaults
        backbone_units = 128 if backbone_units is None else backbone_units
        backbone_layers = 1 if backbone_layers is None else backbone_layers
        backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout

        # Choose cell
        if isinstance(units, ncps.wirings.Wiring):
            raise ValueError("Wired mode not implemented in this optimized wrapper.")
        else:
            self.state_size = int(units)
            self.output_size = self.state_size
            # Pick cell
            if cell_cls is None:
                cell_cls = CfCCellFused  # default to our fused cell

            self.rnn_cell = cell_cls(
                input_size,
                self.state_size,
                mode=mode,
                backbone_activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                sparsity_mask=sparsity_mask,
            )

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

        # Optional LSTM memory path (kept for API parity; rarely used in CfC setups)
        self.use_mixed = mixed_memory
        if self.use_mixed:
            from .lstm import LSTMCell  # adjust if your path differs
            self.lstm = LSTMCell(input_size, self.state_size)

    def forward(self, input, hx=None, timespans=None):
        # Safe AMP context: CUDA → fp16/bf16; CPU → bf16-only when explicitly requested
        if input.is_cuda and self.use_amp:
            amp_ctx = torch.amp.autocast('cuda')          # GPU AMP
        elif (not input.is_cuda) and self.use_amp:
            # Optional: CPU autocast; often no speedup, can reduce numeric fidelity.
            amp_ctx = torch.autocast(device_type="cpu", enabled=True, dtype=torch.bfloat16)
        else:
            amp_ctx = torch.autocast(device_type="cpu", enabled=False)

        with amp_ctx:
            is_batched = (input.dim() == 3)
            if not is_batched:
                input = input.unsqueeze(0) if self.batch_first else input.unsqueeze(1)
                if timespans is not None:
                    timespans = timespans.unsqueeze(0 if self.batch_first else 1)

            # Normalize to time-major: (L, B, C)
            if self.batch_first:
                x = input.transpose(0, 1).contiguous()
                ts = None if timespans is None else timespans.transpose(0, 1).contiguous()
            else:
                x = input.contiguous()
                ts = None if timespans is None else timespans.contiguous()

            L, B, C = x.shape
            device = x.device
            dtype = x.dtype

            # Init state
            if hx is None:
                h_state = torch.zeros((B, self.state_size), device=device, dtype=dtype)
                c_state = torch.zeros_like(h_state) if self.use_mixed else None
            else:
                if self.use_mixed and isinstance(hx, torch.Tensor):
                    raise RuntimeError("mixed_memory=True requires tuple (h0,c0)")
                h_state, c_state = hx if self.use_mixed else (hx, None)
                if h_state.dim() == 1:
                    h_state = h_state.unsqueeze(0)
                if c_state is not None and c_state.dim() == 1:
                    c_state = c_state.unsqueeze(0)

            # Preallocate output
            if self.return_sequences:
                out_dim = self.proj_size if isinstance(self.fc, nn.Linear) else self.output_size
                y = torch.empty((L, B, out_dim), device=device, dtype=dtype)

            # Tight time loop
            for t in range(L):
                xt = x[t]                 # (B, C)
                if ts is None:
                    span_t = x.new_ones((B, 1))                # (B,1) scalar 1.0
                else:
                    tt = ts[t]
                    if tt.ndim == 1:                           # (B,) -> (B,1)
                        span_t = tt.unsqueeze(1)
                    elif tt.ndim == 2 and tt.size(-1) == 1:    # (B,1)
                        span_t = tt
                    else:
                        # If someone passes (B,), (B,1), or even (B,H) by mistake,
                        # we standardize to (B,1)
                        span_t = tt.mean(dim=-1, keepdim=True)


                if self.use_mixed:
                    h_state, c_state = self.lstm(xt, (h_state, c_state))

                h_out, h_state = self.rnn_cell(xt, h_state, span_t)
                if self.return_sequences:
                    y[t] = self.fc(h_out)

            # Back to (B, L, P) if requested
            if self.return_sequences:
                readout = y.transpose(0, 1).contiguous() if self.batch_first else y
            else:
                readout = self.fc(h_out)

            hx_out = (h_state, c_state) if self.use_mixed else h_state

            # Squeeze if original input was unbatched
            if not is_batched:
                if self.return_sequences:
                    readout = readout.squeeze(0 if self.batch_first else 1)
                hx_out = (hx_out[0][0], hx_out[1][0]) if self.use_mixed else hx_out[0]

            return readout, hx_out


def compiled(model: nn.Module, fullgraph: bool = True):
    """Wrap torch.compile if available; otherwise return model unchanged."""
    if compile_fn is None:
        return model
    try:
        return compile_fn(model, fullgraph=fullgraph, mode="max-autotune")
    except Exception:
        return compile_fn(model)

