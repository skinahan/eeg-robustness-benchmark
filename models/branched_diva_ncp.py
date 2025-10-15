import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from braindecode import EEGClassifier
from braindecode.models.base import EEGModuleMixin
from ncps.torch import CfC
from ncps.wirings import AutoNCP
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


class BranchedDIVANCP(EEGModuleMixin, nn.Module):
    """
    Hybrid model combining DIVANCP's front-end with branched recurrent processing.
    
    Architecture:
      1. DIVANCP front-end: CNN + Multi-scale temporal + SNR gate + temporal downsampler
      2. Branched recurrent processing: Split into bins, parallel CfC per bin
      3. Weighted residual connections (DIVANCP-style) at bin level
      4. Attention pooling within bins
      5. Fusion across bins
      6. Classification head
    
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
        kernel_length: int = 64,     # temporal kernel in first conv
        pool_time: int = 4,          # anti-alias temporal pooling/stride
        drop_prob: float = 0.25,
        # --- Multi-scale temporal integration ---
        ms_kernels=(9, 15, 31),
        ms_dilations=(1, 4, 16),
        # --- Temporal downsampler (Conv1D) before CfC ---
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        # --- NCP/CfC core ---
        ncp_hidden_dim: int = 22,
        sparsity: float = 0.85,
        ncp_output_size: int = None,  # if None, defaults to F2 for residual compatibility
        # --- Binning params ---
        bin_len: int = 48,            # number of timesteps per bin AFTER downsampling
        bin_stride: int = 44,         # step between bin starts; set < bin_len for overlap
        fusion: str = "attn",         # "attn" or "mean"
        mixed_memory: bool = True,
        # --- SNR gate ---
        snr_reduction: int = 4,
        # --- Normalization ---
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
        use_spectral_norm_first_conv: bool = False,
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

        # -------------------------
        # DIVANCP FRONT-END: Layers before recurrent compartment
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

        # 7) TEMPORAL DOWNSAMPLER (Conv1D) to shorten sequence for CfC
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
        
        # Set ncp_output_size to F2 by default for residual compatibility
        if ncp_output_size is None:
            ncp_output_size = self.F2
        self.ncp_output_size = ncp_output_size

        # CfC/NCP cell for per-bin processing
        wiring = AutoNCP(ncp_hidden_dim, ncp_output_size,
                         sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(
            input_size=self.F2,
            units=wiring,
            return_sequences=True,   # we pool over time afterward
            mixed_memory=mixed_memory
        )

        # Weighted residual parameter
        self.weight_residual = nn.Parameter(torch.from_numpy(rng.uniform(0.0, 1.0, (1,))).float())

        # Pool within each bin (across its timesteps)
        self.intra_bin_pool = TemporalAttnPool(dim=ncp_output_size)

        # Fusion across bins (restore context)
        if self.fusion == "attn":
            self.bin_fusion = TemporalAttnPool(dim=ncp_output_size)  # pool over bins
        else:
            self.bin_fusion = None  # mean over bins

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        
        # Light head before logits
        self.head_norm = nn.LayerNorm(ncp_output_size)
        self.head_drop = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(ncp_output_size, n_outputs)

        self._glorot_weight_zero_bias()

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
        # Unfold along time: -> [B, F, NB, L]
        x_unf = x_feat.unfold(dimension=2, size=self.bin_len, step=self.bin_stride)
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
        # DIVANCP FRONT-END
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

        # Merge batch and bins to run CfC in parallel
        x_bins = x_bins.reshape(B * Bins, L, self.F2)     # [B*NB, L, F2]

        # Store residual for weighted connection
        residual = x_bins                                  # [B*NB, L, F2]

        # CfC over each bin (return sequences)
        x_seq, _ = self.ncp(x_bins)                        # [B*NB, L, H]
        H = x_seq.size(-1)

        # Apply weighted residual connection (DIVANCP-style)
        # Note: This only works if H == F2, which we ensure by default
        if H == self.F2:
            x_seq = (x_seq * (1 - self.weight_residual)) + (residual * self.weight_residual)
        else:
            # If dimensions don't match, skip residual (shouldn't happen with default settings)
            pass

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
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


def create_branched_diva_ncp_classifier(n_chans, n_times, n_outputs):
    """Create the BranchedDIVANCP classifier."""
    seed = get_seed()
    gradient_clip_value = 1.0
    classifier = EEGClassifier(
        BranchedDIVANCP,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-2,
        optimizer__weight_decay=0,#1e-4,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=32,
        module__drop_prob=0.5,
        module__F1=8,
        module__D=2,
        module__kernel_length=125,
        module__temporal_kernel_size=3,
        module__temporal_stride=2,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2),
            LRScheduler(policy=ExponentialLR, gamma=0.97),
        ],
        verbose=EEGCLASSIFIER_VERBOSE
    )

    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()

    return classifier

