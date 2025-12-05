import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from braindecode import EEGClassifier
from braindecode.models.base import EEGModuleMixin
from models.cnnncp import _MultiScaleTemporalBlock1D, _SNRGate
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np


class TemporalAttnPool(nn.Module):
    """
    Lightweight attention pooling over a time axis.
    Inputs:  x [B, T, C]
    Output:  z [B, C]
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # [B, T, C]
        q = self.query  # [C]
        k = torch.tanh(self.proj(x))  # [B, T, C]
        # attention scores over T
        att = torch.einsum("btc,c->bt", k, q) / (x.size(-1) ** 0.5)  # [B, T]
        w = torch.softmax(att, dim=1).unsqueeze(-1)                  # [B, T, 1]
        z = (w * x).sum(dim=1)                                       # [B, C]
        return z


class BranchedDIVABase(EEGModuleMixin, nn.Module):
    """
    Base class for DIVA-style models with branched recurrent processing.
    
    Architecture:
      1. DIVA front-end: CNN + Multi-scale temporal + SNR gate + temporal downsampler
      2. Branched recurrent processing: Split into bins, parallel recurrent processing per bin
      3. Weighted residual connections (DIVA-style) at bin level
      4. Attention pooling within bins
      5. Fusion across bins
      6. Classification head
    
    Residual Initialization Strategy:
      Default: "backwards_rezero" - recurrent compartment starts at full strength.
      This has been empirically validated to outperform standard ReZero (identity at init)
      for temporal modeling tasks, providing ~6.4% better clean performance.
      See REZERO_BACKWARDS_ANALYSIS.md for detailed theoretical justification.
    
    Subclasses must implement:
    - _create_recurrent_cell(): Create the recurrent cell (NCP, LSTM, etc.)
    - _process_bins(): Process temporal bins through the recurrent cell
    
    Input:  x: (B, C=n_chans, T=n_times)
    Output: logits: (B, n_outputs)
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        # --- CNN (EEGNet-like) front-end ---
        F1: int = 8,                 # temporal filter count
        D: int = 2,                  # depthwise multiplier -> F2 = F1*D
        kernel_length: int = 125,     # temporal kernel in first conv
        pool_time: int = 4,          # anti-alias temporal pooling/stride
        drop_prob: float = 0.25,
        # --- Multi-scale temporal integration ---
        ms_kernels=(9, 15, 31),
        ms_dilations=(1, 4, 16),
        # --- Temporal downsampler (Conv1D) before recurrent ---
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        # --- Recurrent core parameters ---
        recurrent_output_size: int = None,  # if None, defaults to F2 for residual compatibility
        # --- Binning params ---
        bin_len: int = 48,            # number of timesteps per bin AFTER downsampling
        bin_stride: int = 44,         # step between bin starts; set < bin_len for overlap
        fusion: str = "attn",         # "attn" or "mean"
        # --- SNR gate ---
        snr_reduction: int = 4,
        # --- Normalization ---
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
        use_spectral_norm_first_conv: bool = False,
        # --- Residual initialization strategy ---
        # Default: "backwards_rezero" (recurrent at full strength at init)
        # This has been empirically validated to provide ~6.4% better clean performance
        # and slightly better robustness compared to "correct_rezero" (identity at init).
        # See REZERO_BACKWARDS_ANALYSIS.md for theoretical justification.
        residual_init_strategy: str = "backwards_rezero",  # "backwards_rezero" (default, empirically superior) or "correct_rezero" (standard ReZero)
        # --- Recurrent cell specific parameters ---
        **recurrent_kwargs
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        rng = np.random.default_rng(seed)
        
        # Derived dims
        self.F1 = F1
        self.F2 = F1 * D
        self.kernel_length = kernel_length
        self.pool_time = pool_time
        self.temporal_stride = temporal_stride
        self.temporal_kernel_size = temporal_kernel_size
        self.bin_len = bin_len
        self.bin_stride = bin_stride
        self.fusion = fusion.lower()
        assert self.fusion in {"attn", "mean"}
        
        # Store residual initialization strategy
        assert residual_init_strategy in {"backwards_rezero", "correct_rezero"}, \
            f"residual_init_strategy must be 'backwards_rezero' or 'correct_rezero', got '{residual_init_strategy}'"
        self.residual_init_strategy = residual_init_strategy

        # -------------------------
        # DIVA FRONT-END: Layers before recurrent compartment
        # -------------------------
        
        # 1) TEMPORAL CONV (time-only): (B,1,C,T) -> (B,F1,C,T)
        conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),
            bias=False,
        )
        self.conv1 = spectral_norm(conv1) if use_spectral_norm_first_conv else conv1
        self.bn1 = nn.BatchNorm2d(self.F1, momentum=bn_momentum, eps=bn_eps)
        self.act = nn.ELU()

        # 2) DEPTHWISE SPATIAL CONV: (B,F1,C,T) -> (B,F2=F1*D,1,T)
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1,
            out_channels=self.F2,
            kernel_size=(n_chans, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=self.F1,    # depthwise over F1 groups
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.F2, momentum=bn_momentum, eps=bn_eps)

        # 3) ANTI-ALIAS TEMPORAL DOWNSAMPLE: (B,F2,1,T) -> (B,F2,1,T1)
        assert self.pool_time >= 1
        self.avgpool = nn.AvgPool2d(kernel_size=(1, self.pool_time),
                                    stride=(1, self.pool_time))

        # 4) DROPOUT (regularization)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # 5) MULTI-SCALE TEMPORAL BLOCK (noise-stable integration)
        self.ms_block = _MultiScaleTemporalBlock1D(
            channels=self.F2, kernels=ms_kernels, dilations=ms_dilations
        )

        # 6) SNR GATE (SE-style on mean & logvar)
        self.snr_gate = _SNRGate(channels=self.F2, reduction=snr_reduction)

        # 7) TEMPORAL DOWNSAMPLER (Conv1D) to shorten sequence for recurrent processing
        if self.temporal_kernel_size is None:
            self.temporal_kernel_size = 3
        if self.temporal_stride is None:
            self.temporal_stride = 2

        self.temporal_downsampler = nn.Conv1d(
            in_channels=self.F2,
            out_channels=self.F2,
            kernel_size=self.temporal_kernel_size,
            stride=self.temporal_stride,
            padding=self.temporal_kernel_size // 2,
            bias=False,
        )

        # -------------------------
        # BRANCHED RECURRENT COMPARTMENT
        # -------------------------
        
        # Set recurrent_output_size to F2 by default for residual compatibility
        if recurrent_output_size is None:
            recurrent_output_size = self.F2
        self.recurrent_output_size = recurrent_output_size

        # Create the recurrent cell (implemented by subclasses)
        self.recurrent_cell = self._create_recurrent_cell(**recurrent_kwargs)

        # -------------------------
        # POST-RECURRENT PROCESSING
        # -------------------------
        
        # Weighted residual parameter
        # Always use ReZero-style initialization (zero at start)
        # For backwards_rezero: α=0 means recurrent at full strength
        # For correct_rezero: α=0 means identity at full strength
        # The forward pass formula determines how α=0 is interpreted
        self.weight_residual = nn.Parameter(torch.zeros(1))

        # Pool within each bin (across its timesteps)
        self.intra_bin_pool = TemporalAttnPool(dim=recurrent_output_size)

        # Fusion across bins (restore context)
        if self.fusion == "attn":
            self.bin_fusion = TemporalAttnPool(dim=recurrent_output_size)  # pool over bins
        else:
            self.bin_fusion = None  # mean over bins

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        
        # Light head before logits
        self.head_norm = nn.LayerNorm(recurrent_output_size)
        self.head_drop = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(recurrent_output_size, n_outputs)

        self._glorot_weight_zero_bias()

    def _create_recurrent_cell(self, **kwargs):
        """
        Create the recurrent cell. Must be implemented by subclasses.
        
        Args:
            **kwargs: Recurrent cell specific parameters
            
        Returns:
            The recurrent cell module
        """
        raise NotImplementedError("Subclasses must implement _create_recurrent_cell")

    def _process_bins(self, x_bins, residual):
        """
        Process temporal bins through the recurrent cell. Must be implemented by subclasses.
        
        Args:
            x_bins: [B*NB, L, F2] - reshaped bins for parallel processing
            residual: [B*NB, L, F2] - residual connection data
            
        Returns:
            x_seq: [B*NB, L, H] - processed sequences from recurrent cell
        """
        raise NotImplementedError("Subclasses must implement _process_bins")

    def _chunk_time(self, x_feat):
        """
        Chunk features into temporal bins.

        Input:
          x_feat: [B, F, T]  (post-downsample features)
        Returns:
          x_bins: [B, NB, L, F]
          NB: number of bins; L: bin_len
        """
        B, F, T = x_feat.shape

        # Ensure bin parameters are valid for the current temporal length.
        # If T is shorter than the configured bin_len or bin_stride (which can
        # happen for some tuned hyperparameter combinations), we clamp them so
        # that unfold still works instead of raising a runtime error.
        size = min(self.bin_len, T)
        step = min(self.bin_stride, size)

        # Unfold along time: -> [B, F, NB, L]
        x_unf = x_feat.unfold(dimension=2, size=size, step=step)
        NB = x_unf.shape[2]
        L  = x_unf.shape[3]
        # Reorder to [B, NB, L, F]
        x_bins = x_unf.permute(0, 2, 3, 1).contiguous()
        assert x_bins.shape == (B, NB, L, F), f"Chunk shape mismatch: {x_bins.shape}"
        return x_bins

    def forward(self, x):
        """
        x: (B, C=n_chans, T=n_times)
        """
        B, C, T = x.shape

        # -------------------------
        # DIVA FRONT-END
        # -------------------------
        
        # (1) Temporal conv across time only
        x = x.unsqueeze(1)                                 # (B, 1, C, T)
        x = self.conv1(x)                                  # (B, F1, C, T)
        x = self.bn1(x)
        x = self.act(x)

        # (2) Depthwise spatial conv across channels -> virtual sensors
        x = self.depthwise_conv(x)                         # (B, F2, 1, T)
        x = self.bn2(x)
        x = self.act(x)

        # (3) Anti-alias temporal downsample
        x = self.avgpool(x)                                # (B, F2, 1, T1)
        T1 = x.shape[-1]

        # (4) Dropout
        x = self.dropout1(x)

        # Switch to 1D temporal for subsequent blocks
        x = x.squeeze(2)                                   # (B, F2, T1)

        # (5) Multi-scale temporal integration (noise-stable)
        x = self.ms_block(x)                               # (B, F2, T1)

        # (6) SNR gate (Wiener-like shrinkage)
        x = self.snr_gate(x)                               # (B, F2, T1)

        # (7) Temporal downsampler (Conv1D)
        x = self.temporal_downsampler(x)                   # (B, F2, T2)
        T2 = x.shape[-1]

        # -------------------------
        # BRANCHED RECURRENT COMPARTMENT
        # -------------------------
        
        # Chunk into bins
        x_bins = self._chunk_time(x)                       # [B, NB, L, F2]
        Bins = x_bins.size(1)
        L = x_bins.size(2)
        F2_ = x_bins.size(3)
        assert F2_ == self.F2

        # Merge batch and bins to run recurrent processing in parallel
        x_bins = x_bins.reshape(B * Bins, L, self.F2)     # [B*NB, L, F2]

        # Store residual for weighted connection
        residual = x_bins                                  # [B*NB, L, F2]

        # Process bins through recurrent cell (implemented by subclasses)
        x_seq = self._process_bins(x_bins, residual)       # [B*NB, L, H]
        H = x_seq.size(-1)

        # -------------------------
        # POST-RECURRENT PROCESSING
        # -------------------------
        
        # Apply weighted residual connection (DIVA-style)
        # Note: This only works if H == F2, which we ensure by default
        if H == self.F2:
            if self.residual_init_strategy == "backwards_rezero":
                # Backwards ReZero (default): recurrent at full strength at init
                # Empirically validated to outperform correct ReZero for temporal modeling tasks.
                # Formula: recurrent*(1-α) + residual*α
                # At init (α=0): output = recurrent (recurrent compartment active from start)
                # See REZERO_BACKWARDS_ANALYSIS.md for theoretical justification.
                x_seq = (x_seq * (1 - self.weight_residual)) + (residual * self.weight_residual)
            elif self.residual_init_strategy == "correct_rezero":
                # Correct ReZero (standard): identity function at init
                # Formula: recurrent*α + residual*(1-α)
                # At init (α=0): output = residual (identity, recurrent disabled initially)
                # Useful when recurrent initialization is poor or when identity initialization
                # provides better regularization properties.
                x_seq = (x_seq * self.weight_residual) + (residual * (1 - self.weight_residual))
            else:
                raise ValueError(f"Unknown residual_init_strategy: {self.residual_init_strategy}")
        else:
            # If dimensions don't match, skip residual (shouldn't happen with default settings)
            raise ValueError(f"Residual dimension mismatch: H={H} != F2={self.F2}")

        # Intra-bin attention pool -> per-bin summary
        z_per_bin = self.intra_bin_pool(x_seq)             # [B*NB, H]
        z_per_bin = z_per_bin.view(B, Bins, H)             # [B, NB, H]

        # Fuse across bins
        if self.fusion == "attn":
            z = self.bin_fusion(z_per_bin)                 # [B, H]  (attn over NB)
        else:
            z = z_per_bin.mean(dim=1)                      # [B, H]  (mean over NB)

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        
        z = self.head_norm(z)
        z = self.head_drop(z)
        logits = self.fc(z)                                # [B, n_outputs]
        
        return logits

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
            if hasattr(module, "bias") and module.bias is not None and hasattr(module.bias, 'fill_'):
                nn.init.constant_(module.bias, 0.0)
