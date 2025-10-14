from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from braindecode import EEGClassifier
import torch
from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import Ensure4d
from torch import nn
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP
from einops.layers.torch import Rearrange
from models.small_world_wiring import ModularSmallWorldWiring
from architecture_refinement.arbitrary_wiring import ArbitraryWiring, WsFlexHiddenWiring


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class NCPOnlyModel(EEGModuleMixin, nn.Module):
    def __init__(self, n_chans, n_times, n_outputs, ncp_hidden_dim=24, drop_prob=0.05, sparsity=0.85):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
        )
        self.wiring = AutoNCP(ncp_hidden_dim, n_outputs, sparsity_level=sparsity, seed=get_seed())
        self.ncp = CfC(
            input_size=n_chans,
            units=self.wiring,
            proj_size=2,
            return_sequences=False,
            batch_first=True,
            mixed_memory=True,
            mode="default"
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.ncp(x)
        return x


class CfCOnlyModel(EEGModuleMixin, nn.Module):
    def __init__(self, n_chans, n_times, n_outputs, ncp_hidden_dim=24, drop_prob=0.05):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None
        )

        self.ncp = CfC(
            input_size=n_chans,
            units=ncp_hidden_dim,
            proj_size=2,
            return_sequences=False,
            batch_first=True,
            mixed_memory=False,
            mode="pure",
            # activation="lecun_tanh",
            # backbone_units=128,
            # backbone_layers=3,
            # backbone_dropout=0.0
        )
        self._glorot_weight_zero_bias()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.ncp(x)  # [B, 2]
        return x
    
    def _glorot_weight_zero_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ELU):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Dropout):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class CNNNCPv4(EEGModuleMixin, nn.Sequential):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=36,
            sparsity=0.5,
            drop_prob=0.25,
            F1=8,
            D=2,
            kernel_length=64,
            temporal_kernel_size=3,
            temporal_stride=2
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=None,
        )
        hidden_size = ncp_hidden_dim

        # 1. Input Conv2D:
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=(1, 15),  # Changed from 16 to 15
            stride=(1, 1),
            padding=(0, 7),  # Changed from 8 to 7
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D:
        self.depthwise_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(n_chans, 1),
                                        groups=8, stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # 3. Average Pooling:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. LSTM expects input shape (B, seq_len, features).
        ncp_output_size = 32
        wiring = AutoNCP(ncp_hidden_dim, ncp_output_size, sparsity_level=sparsity, seed=get_seed())
        self.ncp = CfC(
            input_size=4,
            units=wiring,
            return_sequences=True,
            batch_first=True,
            mixed_memory=True,
            mode="default"
        )        

        # 6. Reshape LSTM output for the separable convolution.

        # 7. Separable Conv2D:
        self.sep_depthwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=ncp_output_size, kernel_size=(3, 1),
                                       stride=(1, 1), padding=(1, 0), groups=ncp_output_size, bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # 8. Final dropout before the dense layer.
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 9. Dense (fully connected) layer:
        self.fc = nn.Linear(16, n_outputs)

    def forward(self, x):
        # 1. Input Conv2D:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # 2. Depthwise Conv2D:
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # 3. Average Pooling:
        x = self.avgpool(x)
        x = self.dropout(x)

        # 4. Permutation and Reshaping for LSTM:
        x = x.permute(0, 3, 2, 1)
        x = x.contiguous().view(x.shape[0], self.n_times - 1, 4)
        # print(x.shape)
        # 5. LSTM:
        # x.shape: (64, 1000, 4)
        # print(x.shape)
        x, _ = self.ncp(x)
        # print(x.shape)
        # x.shape: (64, 1000, 32)
        # 6. Reshape for Separable Conv2D:
        x = x.permute(0, 2, 1).unsqueeze(3)

        # 7. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # After separable conv, x shape: (B, 16, T, 1)
        x = self.global_pool(x)  # → (B, 16, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, 16)
        x = self.fc(x)  # → (B, n_outputs)
        return x  # no softmax

import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

# Assumes you have these in scope, as in your original:
# from braindecode.models.modules import EEGModuleMixin
# from ncps.torch import CfC, AutoNCP
# from your_utils import get_seed   # or replace with a fixed seed if preferred


class _MultiScaleTemporalBlock1D(nn.Module):
    """
    Noise-stable temporal integration block.
    - Parallel depthwise-separable Conv1D branches with different dilations.
    - Residual sum of branches -> LayerNorm -> ELU.
    Keeps T the same (same 'time' length) via correct padding.
    """
    def __init__(self, channels: int,
                 kernels=(9, 15, 31),
                 dilations=(1, 4, 16),
                 pointwise_expand=False):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.channels = channels
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            # Effective kernel = (k - 1) * d + 1; padding to keep T same:
            pad = ((k - 1) * d) // 2
            depthwise = nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=k, stride=1, padding=pad, dilation=d,
                groups=channels, bias=False
            )
            pointwise_out = channels if not pointwise_expand else channels
            pointwise = nn.Conv1d(
                in_channels=channels, out_channels=pointwise_out,
                kernel_size=1, bias=False
            )
            self.branches.append(nn.Sequential(depthwise, pointwise))

        # LayerNorm over channel dimension (applied per time step).
        self.ln = nn.LayerNorm(channels)
        self.act = nn.ELU()

    def forward(self, x):
        """
        x: (B, C=channels, T)
        returns: (B, C, T)
        """
        # Sum of parallel branches
        y = 0
        for b in self.branches:
            y = y + b(x)
        # Residual connection
        y = y + x
        # LayerNorm expects (B, T, C)
        y = y.transpose(1, 2)                 # (B, T, C)
        y = self.ln(y)                        # (B, T, C)
        y = y.transpose(1, 2)                 # (B, C, T)
        y = self.act(y)
        return y


class _SNRGate(nn.Module):
    """
    Lightweight Wiener-like shrinkage via Squeeze-and-Excite on (mean, log-variance).
    - Computes per-channel stats across time.
    - Produces per-channel gain gamma in [0,1] to suppress noisy channels/bands.
    """
    def __init__(self, channels: int, reduction: int = 4, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()  # gamma in [0,1]
        )

    def forward(self, x):
        """
        x: (B, C, T)
        returns: (B, C, T) gated
        """
        mean = x.mean(dim=2)                               # (B, C)
        var = (x.var(dim=2, unbiased=False) + self.eps)    # (B, C)
        logvar = torch.log(var)                            # (B, C)

        stats = torch.cat([mean, logvar], dim=1)           # (B, 2C)
        gamma = self.mlp(stats)                            # (B, C)
        gamma = gamma.unsqueeze(2)                         # (B, C, 1)
        return x * gamma


class CNNNCPv4GaussianRobust(EEGModuleMixin, nn.Module):
    """
    CNN-NCP optimized for robustness to additive Gaussian noise.

    Architectural differences vs your CNNNCPv3:
      • Proper anti-aliasing temporal downsampling (AvgPool 1xP with stride P).
      • Multi-Scale Temporal Block (dilated depthwise-separable Conv1D) to integrate over time without blowing params.
      • SNR Gate (SE-style on mean & log-variance) to shrink noisy channels/bands.
      • Careful shape handling and comments at every step.

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
        ncp_output_size: int = None,  # if None, defaults to F1
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

        # Derived dims
        self.F1 = F1
        self.F2 = F1 * D
        self.kernel_length = kernel_length
        self.pool_time = pool_time
        self.temporal_stride = temporal_stride
        self.temporal_kernel_size = temporal_kernel_size

        # -------------------------
        # 1) TEMPORAL CONV (time-only): (B,1,C,T) -> (B,F1,C,T)
        #    - Learns band-limited temporal filters per channel.
        #    - Spectral norm (optional) to avoid amplifying HF noise.
        # -------------------------
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

        # -------------------------
        # 2) DEPTHWISE SPATIAL CONV: (B,F1,C,T) -> (B,F2=F1*D,1,T)
        #    - Learns spatial filters (virtual sensors), averaging out uncorrelated channel noise.
        # -------------------------
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

        # -------------------------
        # 3) ANTI-ALIAS TEMPORAL DOWNSAMPLE: (B,F2,1,T) -> (B,F2,1,T1)
        #    - AveragePool over time is an effective low-pass before stride.
        # -------------------------
        assert self.pool_time >= 1
        self.avgpool = nn.AvgPool2d(kernel_size=(1, self.pool_time),
                                    stride=(1, self.pool_time))

        # -------------------------
        # 4) DROPOUT (regularization)
        # -------------------------
        self.dropout1 = nn.Dropout(p=drop_prob)

        # -------------------------
        # Switch to 1D temporal processing on shape (B, F2, T1)
        # 5) MULTI-SCALE TEMPORAL BLOCK (noise-stable integration)
        # -------------------------
        self.ms_block = _MultiScaleTemporalBlock1D(
            channels=self.F2, kernels=ms_kernels, dilations=ms_dilations
        )

        # -------------------------
        # 6) SNR GATE (SE-style on mean & logvar): (B,F2,T1)->(B,F2,T1)
        #     - Shrinks channels with low SNR (Gaussian noise ↑ variance).
        # -------------------------
        self.snr_gate = _SNRGate(channels=self.F2, reduction=snr_reduction)

        # -------------------------
        # 7) TEMPORAL DOWNSAMPLER (Conv1D) to shorten sequence for CfC:
        #     (B,F2,T1) -> (B,F2,T2)
        # -------------------------
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
        # 8) CfC/NCP RECURRENT CORE:
        #     Input size = F2; Output size = ncp_output_size (default F1).
        #     return_sequences=True -> (B, T2, H)
        # -------------------------
        if ncp_output_size is None:
            ncp_output_size = self.F1

        wiring = AutoNCP(ncp_hidden_dim, ncp_output_size,
                         sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(self.F2, wiring, return_sequences=True)

        # -------------------------
        # 9) SEPARABLE CONV2D HEAD on (B,H,T2,1) for light smoothing
        # -------------------------
        self.sep_depthwise = nn.Conv2d(
            in_channels=ncp_output_size,
            out_channels=ncp_output_size,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            groups=ncp_output_size,  # depthwise
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=ncp_output_size,
            out_channels=ncp_output_size,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(ncp_output_size, momentum=bn_momentum, eps=bn_eps)

        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ncp_output_size, n_outputs)

        self._glorot_weight_zero_bias()

    def forward(self, x):
        """
        x: (B, C=n_chans, T=n_times)
        """
        B, C, T = x.shape

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

        # Prepare for CfC: (B, T2, F2)
        x = x.transpose(1, 2).contiguous()                 # (B, T2, F2)

        # (8) CfC/NCP recurrent core
        x, _ = self.ncp(x)                                 # (B, T2, H)

        # Head expects (B, H, T2, 1)
        x = x.transpose(1, 2).unsqueeze(3)                 # (B, H, T2, 1)

        # (9) Separable Conv2D head + BN + ELU + Dropout
        x = self.sep_depthwise(x)                          # (B, H, T2, 1)
        x = self.sep_pointwise(x)                          # (B, H, T2, 1)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout2(x)

        # Global pooling and classification
        x = self.global_pool(x)                            # (B, H, 1, 1)
        x = x.view(x.shape[0], -1)                         # (B, H)
        x = self.fc(x)                                     # (B, n_outputs)
        return x

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


# The CfC Model with a residual connection surrounding the recurrent layer.
# This model gets a high non-contaminated score, but is particularly vulnerable to Gaussian noise.
class CNNCfCv4(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=19,  # Reduced from 32 to 16 for speed
            cnn_output_dim=16,
            drop_prob=0.25,
            F1=8,
            D=2,
            kernel_length=64,
            temporal_kernel_size=3,
            temporal_stride=2,  # More aggressive downsampling
            max_seq_length=500  # Limit sequence length for CfC
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        self.F1 = F1
        F2 = F1 * D
        cnn_output_dim = F2
        self.max_seq_length = max_seq_length
        self.temporal_stride = temporal_stride

        # 1. Input Conv2D (same as REEGNet):
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=(1, 15),
            stride=(1, 1),
            padding=(0, 7),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D (same as REEGNet):
        self.depthwise_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(n_chans, 1),
                                        groups=8, stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # 3. More aggressive pooling for temporal reduction:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))  # 8x temporal reduction

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Temporal downsampler for further reduction:
        self.temporal_downsampler = nn.Conv1d(
            in_channels=16,  # After depthwise conv, we have 16 channels
            out_channels=16,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )

        # 6. CfC with reduced complexity and residual connection:
        ncp_input_size = 16  # After temporal downsampling, we have 16 features
        ncp_output_size = 16  # Reduced from 32 to 16
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=False,
            mode='pure',
            backbone_units=64,
        )

        # 7. Separable Conv2D (same as REEGNet):
        self.sep_depthwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=ncp_output_size, kernel_size=(3, 1),
                                       stride=(1, 1), padding=(1, 0), groups=ncp_output_size, bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # 8. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Dense layer:
        self.fc = nn.Linear(16, n_outputs)

    def forward(self, x):
        # 1. Input Conv2D:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # 2. Depthwise Conv2D:
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # 3. Aggressive pooling for temporal reduction:
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 4. Reshape for temporal processing:
        x = x.permute(0, 3, 2, 1)  # [B, T, 1, C]
        # Calculate the actual number of features after pooling
        num_features = x.shape[3]  # This should be 16 after depthwise conv
        x = x.contiguous().view(x.shape[0], x.shape[1], num_features)  # [B, T, num_features]
        
        # 5. Apply temporal downsampling:
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', 16]
        
        # 6. Limit sequence length for CfC:
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 7. CfC processing with residual connection:
        residual = x  # Store the input for residual connection
        x, _ = self.ncp(x)  # [B, T', H]
        x = x + residual  # Add residual connection
        
        # 8. Reshape for separable conv:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, H, T', 1]

        # 9. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 10. Global pooling and classification:
        x = self.global_pool(x)  # → (B, 16, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, 16)
        x = self.fc(x)  # → (B, n_outputs)
        return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


# Current best NCP model architecture.
class CNNNCPv3(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=32,
            cnn_output_dim=16,
            sparsity=0.85,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=2
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        self.use_temporal_downsampler = True
        if temporal_kernel_size is None:
            temporal_kenel_size = 3
        if temporal_stride is None:
            temporal_stride = 2
        seed = get_seed()
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        # Properly use F1, D, and kernel_length parameters
        self.F1 = F1
        F2 = F1 * D
        self.kernel_length = kernel_length
        self.temporal_stride = temporal_stride

        # 1. Input Conv2D - PROPERLY PARAMETERIZED:
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.F1,  # Use F1 instead of hardcoded 8
            kernel_size=(1, self.kernel_length),  # Use kernel_length instead of hardcoded 15
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),  # Proper padding based on kernel_length
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1, momentum=batch_norm_momentum, eps=batch_norm_eps)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D - PROPERLY PARAMETERIZED:
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1, out_channels=F2,  # Use F1 and F2
            kernel_size=(n_chans, 1),
            groups=self.F1,  # Use F1 for grouping
            stride=(1, 1), 
            padding=(0, 0), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps)

        # 3. Average Pooling - PROPERLY PARAMETERIZED:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Temporal downsampler for further reduction:
        if self.use_temporal_downsampler:
            self.temporal_downsampler = nn.Conv1d(
                in_channels=F2,  # Use F2 instead of hardcoded 16
                out_channels=F2,
                kernel_size=temporal_kernel_size,
                stride=temporal_stride,
                padding=temporal_kernel_size // 2
            )
        else:
            self.temporal_downsampler = None

        # 6. CfC with proper input size and configurable parameters:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F1  # Use F1 for consistency

        wiring = AutoNCP(
            ncp_hidden_dim, ncp_output_size, sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(ncp_input_size, wiring, return_sequences=True, mixed_memory=True)

        # 7. Separable Conv2D 
        self.sep_depthwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=ncp_output_size, 
            kernel_size=(3, 1),
            stride=(1, 1), 
            padding=(1, 0), 
            groups=ncp_output_size, 
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=ncp_output_size,  
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(ncp_output_size, momentum=batch_norm_momentum, eps=batch_norm_eps)

        # 8. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Dense layer:
        self.fc = nn.Linear(ncp_output_size, n_outputs)
        self._glorot_weight_zero_bias()

    def forward(self, x):
        # 1. Input Conv2D:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # 2. Depthwise Conv2D:
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # 3. Average pooling for temporal reduction:
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 4. Reshape for temporal processing:
        x = x.permute(0, 3, 2, 1)  # [B, T, 1, C]
        num_features = x.shape[3]  # This will be F2
        x = x.contiguous().view(x.shape[0], x.shape[1], num_features)  # [B, T, F2]
        
        # 5. Apply temporal downsampling:
        if self.temporal_downsampler is not None:
            x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', F2]
        
        x, _ = self.ncp(x)  # [B, T', H]
        
        # 8. Reshape for separable conv:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, 8, T', 1]

        # 9. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 10. Global pooling and classification:
        x = self.global_pool(x)  # → (B, F2, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, F2)
        x = self.fc(x)  # → (B, n_outputs)
        return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


# Stochastic Depth implementation for regularization
class StochasticDepth(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if torch.rand(1) < self.p:
            return torch.zeros_like(x)
        return x


# REEGNet variant with CfC-based recurrence - ULTRA-SIMPLIFIED VERSION
class CNNCfC_Compact(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=8,  # Minimal size
            drop_prob=0.2,  # Higher dropout for regularization
            max_seq_length=150,  # Very short sequences
            use_stochastic_depth=True,
            # CfC-specific parameters
            mixed_memory=True,
            mode='default',
            activation='lecun_tanh',
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=0.0
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        
        self.max_seq_length = max_seq_length
        self.use_stochastic_depth = use_stochastic_depth

        # Store CfC parameters
        self.mixed_memory = mixed_memory
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

        # 1. Single feature extraction layer:
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=(n_chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 32), stride=(1, 32)),  # Very aggressive pooling
            nn.Dropout(p=drop_prob)
        )

        # 2. CfC with minimal complexity and configurable parameters:
        self.ncp = CfC(
            input_size=8, 
            units=ncp_hidden_dim, 
            proj_size=8,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=self.mixed_memory,
            mode=self.mode,
            activation=self.activation,
            backbone_units=self.backbone_units,
            backbone_layers=self.backbone_layers,
            backbone_dropout=self.backbone_dropout
        )

        # 3. Stochastic depth for regularization:
        if use_stochastic_depth:
            self.stochastic_depth = StochasticDepth(p=0.1)
        else:
            self.stochastic_depth = nn.Identity()

        # 4. Simple classifier:
        self.classifier = nn.Sequential(
            nn.Linear(8, 8),
            nn.ELU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(8, n_outputs)
        )
        
        # Initialize weights properly
        self._glorot_weight_zero_bias()

    def forward(self, x):
        # 1. Feature extraction:
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        
        # 2. Reshape for temporal processing:
        x = x.permute(0, 3, 2, 1).squeeze(2)  # [B, T, C]
        
        # 3. Limit sequence length:
        if x.shape[1] > self.max_seq_length:
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 4. CfC processing with stochastic depth:
        x, _ = self.ncp(x)
        x = self.stochastic_depth(x)
        
        # 5. Global average pooling:
        x = x.mean(dim=1)  # [B, C]
        
        # 6. Classification:
        x = self.classifier(x)
        return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


# REEGNet variant with CfC-based recurrence - PROPERLY PARAMETERIZED VERSION
class CNNCfCv2(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=16,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,
            max_seq_length=250,
            # CfC-specific parameters
            mixed_memory=False,
            mode='default',
            activation='lecun_tanh',
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=0.05
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        # Properly use F1, D, and kernel_length parameters
        self.F1 = F1
        F2 = F1 * D
        self.kernel_length = kernel_length
        self.max_seq_length = max_seq_length
        self.temporal_stride = temporal_stride

        # Store CfC parameters
        self.mixed_memory = mixed_memory
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

        # 1. Input Conv2D - PROPERLY PARAMETERIZED:
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.F1,  # Use F1 instead of hardcoded 8
            kernel_size=(1, self.kernel_length),  # Use kernel_length instead of hardcoded 15
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),  # Proper padding based on kernel_length
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D - PROPERLY PARAMETERIZED:
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1, out_channels=F2,  # Use F1 and F2
            kernel_size=(n_chans, 1),
            groups=self.F1,  # Use F1 for grouping
            stride=(1, 1), 
            padding=(0, 0), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)

        # 3. Average Pooling - PROPERLY PARAMETERIZED:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Temporal downsampler for further reduction:
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,  # Use F2 instead of hardcoded 16
            out_channels=F2,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )

        # 6. CfC with proper input size and configurable parameters:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F2  # Use F2 for consistency
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=self.mixed_memory,
            mode=self.mode,
            activation=self.activation,
            backbone_units=self.backbone_units,
            backbone_layers=self.backbone_layers,
            backbone_dropout=self.backbone_dropout
        )

        # 7. Separable Conv2D - PROPERLY PARAMETERIZED:
        self.sep_depthwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=ncp_output_size, 
            kernel_size=(3, 1),
            stride=(1, 1), 
            padding=(1, 0), 
            groups=ncp_output_size, 
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=F2,  # Use F2 instead of hardcoded 16
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)

        # 8. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Dense layer:
        self.fc = nn.Linear(F2, n_outputs)  # Use F2 instead of hardcoded 16

    def forward(self, x):
        # 1. Input Conv2D:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # 2. Depthwise Conv2D:
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # 3. Average pooling for temporal reduction:
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 4. Reshape for temporal processing:
        x = x.permute(0, 3, 2, 1)  # [B, T, 1, C]
        num_features = x.shape[3]  # This will be F2
        x = x.contiguous().view(x.shape[0], x.shape[1], num_features)  # [B, T, F2]
        
        # 5. Apply temporal downsampling:
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', F2]
        
        # 6. Limit sequence length for CfC:
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 7. CfC processing:
        x, _ = self.ncp(x)  # [B, T', H]
        
        # 8. Reshape for separable conv:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, 8, T', 1]

        # 9. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 10. Global pooling and classification:
        x = self.global_pool(x)  # → (B, F2, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, F2)
        x = self.fc(x)  # → (B, n_outputs)
        return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


# CNNSmallWorld model based on CNNNCPv3 but using ModularSmallWorldWiring
class CNNSmallWorld(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=32,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,
            max_seq_length=250,
            n_modules=4,
            rewiring_prob=0.2
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        # Properly use F1, D, and kernel_length parameters
        self.F1 = F1
        F2 = F1 * D
        self.kernel_length = kernel_length
        self.max_seq_length = max_seq_length
        self.temporal_stride = temporal_stride

        # Store Small World wiring parameters
        self.n_modules = n_modules
        self.rewiring_prob = rewiring_prob

        # 1. Input Conv2D - PROPERLY PARAMETERIZED:
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.F1,  # Use F1 instead of hardcoded 8
            kernel_size=(1, self.kernel_length),  # Use kernel_length instead of hardcoded 15
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),  # Proper padding based on kernel_length
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D - PROPERLY PARAMETERIZED:
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1, out_channels=F2,  # Use F1 and F2
            kernel_size=(n_chans, 1),
            groups=self.F1,  # Use F1 for grouping
            stride=(1, 1), 
            padding=(0, 0), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)

        # 3. Average Pooling - PROPERLY PARAMETERIZED:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Temporal downsampler for further reduction:
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,  # Use F2 instead of hardcoded 16
            out_channels=F2,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )

        # 6. CfC with Small World wiring - no incompatible parameters:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F1

        # Use ModularSmallWorldWiring instead of AutoNCP
        # The wiring will automatically expand units if needed to accommodate ncp_output_size
        wiring = ModularSmallWorldWiring(
            units=ncp_hidden_dim, 
            output_size=ncp_output_size,  # Match the expected output size
            n_modules=n_modules, 
            rewiring_prob=rewiring_prob, 
            seed=seed
        )
        # The wiring should handle the output size constraint
        self.ncp = CfC(ncp_input_size, wiring, return_sequences=True, proj_size=ncp_output_size)
        

        # 7. Separable Conv2D - PROPERLY PARAMETERIZED:
        self.sep_depthwise = nn.Conv2d(
            in_channels=ncp_output_size,  # Output from CfC is 8 channels
            out_channels=ncp_output_size, 
            kernel_size=(3, 1),
            stride=(1, 1), 
            padding=(1, 0), 
            groups=ncp_output_size, 
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=F2,  # Use F2 instead of hardcoded 16
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)

        # 8. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Dense layer:
        self.fc = nn.Linear(F2, n_outputs)  # Use F2 instead of hardcoded 16

    def forward(self, x):
        # 1. Input Conv2D:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # 2. Depthwise Conv2D:
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # 3. Average pooling for temporal reduction:
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 4. Reshape for temporal processing:
        x = x.permute(0, 3, 2, 1)  # [B, T, 1, C]
        num_features = x.shape[3]  # This will be F2
        x = x.contiguous().view(x.shape[0], x.shape[1], num_features)  # [B, T, F2]
        
        # 5. Apply temporal downsampling:
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', F2]
        
        # 6. Limit sequence length for CfC:
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 7. CfC processing:
        x, _ = self.ncp(x)  # [B, T', H]
        # Apply projection to get exactly 8 output channels
        # x = self.output_projection(x)  # [B, T', 8]
        
        # 8. Reshape for separable conv:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, 8, T', 1]

        # 9. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 10. Global pooling and classification:
        x = self.global_pool(x)  # → (B, F2, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, F2)
        x = self.fc(x)  # → (B, n_outputs)
        return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


# CNNWiredCfC model that uses arbitrary wiring from architecture search
class CNNWiredCfC(EEGModuleMixin, nn.Module):
    """
    CNN model with CfC using arbitrary wiring from architecture search.
    
    This model is similar to CNNSmallWorld but uses ArbitraryWiring instead
    of ModularSmallWorldWiring, allowing it to use optimized architectures
    from the architecture search process.
    """
    
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            wiring,  # ArbitraryWiring instance
            drop_prob=0.25,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,
            max_seq_length=250,
            mixed_memory=True  # Add mixed_memory parameter
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        # Store parameters
        self.F1 = F1
        F2 = F1 * D
        self.kernel_length = kernel_length
        self.max_seq_length = max_seq_length
        self.temporal_stride = temporal_stride
        
        # Store the wiring
        self.wiring = wiring
        self.ncp_hidden_dim = wiring._hidden_size
        
        # Store CfC parameters
        self.mixed_memory = mixed_memory

        # 1. Input Conv2D
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1, out_channels=F2,
            kernel_size=(n_chans, 1),
            groups=self.F1,
            stride=(1, 1), 
            padding=(0, 0), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)

        # 3. Average Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))

        # 4. Dropout
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Temporal downsampler
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )

        # 6. CfC with arbitrary wiring
        ncp_input_size = F2
        ncp_output_size = F2

        # last minute reconfiguration - make sure the wiring is compatible with expected input and output sizes
        wiring.input_size = ncp_input_size
        wiring.output_size = ncp_output_size
        wiring = wiring.build(ncp_input_size)

        # Use the provided arbitrary wiring
        self.ncp = CfC(input_size=ncp_input_size, units=wiring, return_sequences=True, proj_size=ncp_output_size, mixed_memory=self.mixed_memory)

        # 7. Separable Conv2D
        self.sep_depthwise = nn.Conv2d(
            in_channels=ncp_output_size,
            out_channels=ncp_output_size, 
            kernel_size=(3, 1),
            stride=(1, 1), 
            padding=(1, 0), 
            groups=ncp_output_size, 
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=F2,
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)

        # 8. Final dropout and pooling
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Dense layer
        self.fc = nn.Linear(F2, n_outputs)

    def forward(self, x):
        # 1. Input Conv2D
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # 2. Depthwise Conv2D
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # 3. Average pooling for temporal reduction
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 4. Reshape for temporal processing
        x = x.permute(0, 3, 2, 1)  # [B, T, 1, C]
        num_features = x.shape[3]  # This will be F2
        x = x.contiguous().view(x.shape[0], x.shape[1], num_features)  # [B, T, F2]
        
        # 5. Apply temporal downsampling
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', F2]
        
        # 6. Limit sequence length for CfC
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 7. CfC processing with arbitrary wiring
        x, _ = self.ncp(x)  # [B, T', H]
        
        # 8. Reshape for separable conv
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, 8, T', 1]

        # 9. Separable Conv2D
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 10. Global pooling and classification
        x = self.global_pool(x)  # → (B, F2, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, F2)
        x = self.fc(x)  # → (B, n_outputs)
        return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def get_wiring_info(self):
        """Get information about the wiring structure."""
        return self.wiring.get_wiring_summary()


def create_cnnncfc_v2_classifier(n_chans, n_times, n_outputs):
    """Create the official CNNCfCv2 classifier."""
    seed = get_seed()
    gradient_clip_value = 1.0
    classifier = EEGClassifier(
        CNNCfCv4,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        optimizer__weight_decay=0,
        batch_size=32,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=19,
        module__drop_prob=0.25,
        module__F1=8,
        module__D=2,
        module__kernel_length=64,
        module__temporal_kernel_size=3,
        module__temporal_stride=2,
        module__max_seq_length=500,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2),
            LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=10),
            ],
        verbose=EEGCLASSIFIER_VERBOSE
    )

    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()

    return classifier


def create_cnnncfc_compact_classifier(n_chans, n_times, n_outputs):
    """Create the compact CNNCfC_Compact classifier."""
    # Create a custom classifier for the ultra-simplified model
    from braindecode import EEGClassifier
    from skorch.dataset import ValidSplit
    from globals import get_seed
    
    seed = get_seed()
    gradient_clip_value = 1.0
    classifier = EEGClassifier(
        CNNCfC_Compact,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=5e-4,
        optimizer__weight_decay=1e-2,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=8,
        module__drop_prob=0.2,
        module__max_seq_length=150,
        module__use_stochastic_depth=True,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
            ],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()
    
    return classifier


def create_cnnncpv2_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=36,
        net_sparsity=0.7,
        lr=1e-3,
        batch_size=32,
        weight_decay=0,
        gradient_clip_value=1.0
):
    classifier = CNNNCPv4GaussianRobust
    seed = get_seed()
    # Use standard cross entropy loss
    criterion = torch.nn.CrossEntropyLoss
    
    cnn_ncp_net = EEGClassifier(
        classifier,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=net_size,
        module__sparsity=net_sparsity,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2),
            LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=5),
        ],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        cnn_ncp_net.initialize()
        cnn_ncp_net.module_.cuda()        
    else:
        cnn_ncp_net.initialize()

    return cnn_ncp_net


def create_cnnncp_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=19,
        net_sparsity=0.7,
        lr=1e-4,
        batch_size=64,
        weight_decay=5e-4,
        classifier_type=3,
        gradient_clip_value=1.0
    ):
        classifier = CNNNCPv3
        seed = get_seed()
        # Use standard cross entropy loss
        criterion = torch.nn.CrossEntropyLoss

        cnn_ncp_net = EEGClassifier(
            classifier,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=DEFAULT_MAX_EPOCHS,
            module__n_chans=n_chans,
            module__n_times=n_times,
            module__n_outputs=n_outputs,
            module__ncp_hidden_dim=net_size,
            module__sparsity=net_sparsity,
            train_split=ValidSplit(0.2, stratified=True, random_state=seed),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            callbacks=[
                get_early_stopping_callback(),
                GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
            ],
            verbose=EEGCLASSIFIER_VERBOSE
        )
        if torch.cuda.is_available():
            cnn_ncp_net.initialize()
            cnn_ncp_net.module_.cuda()
            # Only use torch.compile if it's available and compatible
            try:
                cnn_ncp_net.module_ = torch.compile(cnn_ncp_net.module_)
            except Exception as e:
                print(f"Warning: torch.compile failed, using standard model: {e}")
        else:
            cnn_ncp_net.initialize()

        return cnn_ncp_net


def create_cnnsmallworld_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=32,
        drop_prob=0.15,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3,
        F1=8,
        D=2,
        kernel_length=128,
        temporal_kernel_size=3,
        temporal_stride=4,
        max_seq_length=250,
        n_modules=4,
        rewiring_prob=0.2
):
    """Create a CNNSmallWorld classifier with ModularSmallWorldWiring."""
    if net_size < n_outputs + 3:
        new_net_size = n_outputs + 3
        print("WARNING: CNNSmallWorld: TOO FEW UNITS.")
        print(f"Changing net_size to {new_net_size}")
        net_size = new_net_size
    
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss
    
    cnn_smallworld_net = EEGClassifier(
        CNNSmallWorld,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=net_size,
        module__drop_prob=drop_prob,
        module__F1=F1,
        module__D=D,
        module__kernel_length=kernel_length,
        module__temporal_kernel_size=temporal_kernel_size,
        module__temporal_stride=temporal_stride,
        module__max_seq_length=max_seq_length,
        module__n_modules=n_modules,
        module__rewiring_prob=rewiring_prob,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[get_early_stopping_callback()],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        cnn_smallworld_net.initialize()
        cnn_smallworld_net.module_.cuda()
        # Only use torch.compile if it's available and compatible
        try:
            cnn_smallworld_net.module_ = torch.compile(cnn_smallworld_net.module_)
        except Exception as e:
            print(f"Warning: torch.compile failed, using standard model: {e}")
    else:
        cnn_smallworld_net.initialize()

    return cnn_smallworld_net


def create_cnncfc_v2_learnable_classifier(
        n_chans,
        n_times,
        n_outputs,
        ncp_hidden_dim=16,
        drop_prob=0.15,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3,
        F1=8,
        D=2,
        kernel_length=128,
        temporal_kernel_size=3,
        temporal_stride=4,
        max_seq_length=250,
        use_learnable_window=True
):
    """Create a CNNCfCv2 classifier with learnable temporal windowing."""
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss
    
    cnncfc_v2_learnable_net = EEGClassifier(
        CNNCfCv2_Learnable,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=ncp_hidden_dim,
        module__drop_prob=drop_prob,
        module__F1=F1,
        module__D=D,
        module__kernel_length=kernel_length,
        module__temporal_kernel_size=temporal_kernel_size,
        module__temporal_stride=temporal_stride,
        module__max_seq_length=max_seq_length,
        module__use_learnable_window=use_learnable_window,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        cnncfc_v2_learnable_net.initialize()
        cnncfc_v2_learnable_net.module_.cuda()
        # Only use torch.compile if it's available and compatible
        try:
            cnncfc_v2_learnable_net.module_ = torch.compile(cnncfc_v2_learnable_net.module_)
        except Exception as e:
            print(f"Warning: torch.compile failed, using standard model: {e}")
    else:
        cnncfc_v2_learnable_net.initialize()

    return cnncfc_v2_learnable_net


def create_cnnsmallworld_learnable_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=32,
        drop_prob=0.15,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3,
        F1=8,
        D=2,
        kernel_length=128,
        temporal_kernel_size=3,
        temporal_stride=4,
        max_seq_length=250,
        n_modules=4,
        rewiring_prob=0.2,
        use_learnable_window=True
):
    """Create a CNNSmallWorld classifier with learnable temporal windowing."""
    if net_size < n_outputs + 3:
        new_net_size = n_outputs + 3
        print("WARNING: CNNSmallWorld_Learnable: TOO FEW UNITS.")
        print(f"Changing net_size to {new_net_size}")
        net_size = new_net_size
    
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss
    
    cnn_smallworld_learnable_net = EEGClassifier(
        CNNSmallWorld_Learnable,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=net_size,
        module__drop_prob=drop_prob,
        module__F1=F1,
        module__D=D,
        module__kernel_length=kernel_length,
        module__temporal_kernel_size=temporal_kernel_size,
        module__temporal_stride=temporal_stride,
        module__max_seq_length=max_seq_length,
        module__n_modules=n_modules,
        module__rewiring_prob=rewiring_prob,
        module__use_learnable_window=use_learnable_window,
        train_split=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        cnn_smallworld_learnable_net.initialize()
        cnn_smallworld_learnable_net.module_.cuda()
        # Only use torch.compile if it's available and compatible
        try:
            cnn_smallworld_learnable_net.module_ = torch.compile(cnn_smallworld_learnable_net.module_)
        except Exception as e:
            print(f"Warning: torch.compile failed, using standard model: {e}")
    else:
        cnn_smallworld_learnable_net.initialize()

    return cnn_smallworld_learnable_net


def create_cnnwiredcfc_classifier(
        n_chans,
        n_times,
        n_outputs,
        wiring,  # ArbitraryWiring instance
        drop_prob=0.15,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3,
        F1=8,
        D=2,
        kernel_length=128,
        temporal_kernel_size=3,
        temporal_stride=4,
        max_seq_length=250,
        mixed_memory=True  # Add mixed_memory parameter
):
    """Create a CNNWiredCfC classifier with arbitrary wiring."""
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss
    gradient_clip_value = 1.0
    cnn_wiredcfc_net = EEGClassifier(
        CNNWiredCfC,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__wiring=wiring,
        module__drop_prob=drop_prob,
        module__F1=F1,
        module__D=D,
        module__kernel_length=kernel_length,
        module__temporal_kernel_size=temporal_kernel_size,
        module__temporal_stride=temporal_stride,
        module__max_seq_length=max_seq_length,
        module__mixed_memory=mixed_memory,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
            ],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        cnn_wiredcfc_net.initialize()
        cnn_wiredcfc_net.module_.cuda()
        # Only use torch.compile if it's available and compatible
        try:
            cnn_wiredcfc_net.module_ = torch.compile(cnn_wiredcfc_net.module_)
        except Exception as e:
            print(f"Warning: torch.compile failed, using standard model: {e}")
    else:
        cnn_wiredcfc_net.initialize()

    return cnn_wiredcfc_net


def create_ncp_only_classifier(n_chans, n_times, n_outputs, gradient_clip_value=1.0):
    """Create a NCPOnlyModel classifier."""
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss
    ncp_only_net = EEGClassifier(
        NCPOnlyModel,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        optimizer__weight_decay=0.0,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=7,
        module__drop_prob=0.05,
        module__sparsity=0.5,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
        ],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        ncp_only_net.initialize()
        ncp_only_net.module_.cuda()
        # Only use torch.compile if it's available and compatible
        try:
            ncp_only_net.module_ = torch.compile(ncp_only_net.module_)
        except Exception as e:
            print(f"Warning: torch.compile failed, using standard model: {e}")

    return ncp_only_net


def create_cfc_only_classifier(n_chans, n_times, n_outputs, gradient_clip_value=1.0):
    """Create a CfCOnlyModel classifier."""
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss
    cfc_only_net = EEGClassifier(
        CfCOnlyModel,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        optimizer__weight_decay=0.0,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=5,
        module__drop_prob=0.05,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            # LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=5),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
        ],
        verbose=EEGCLASSIFIER_VERBOSE
    )
    
    if torch.cuda.is_available():
        cfc_only_net.initialize()
        cfc_only_net.module_.cuda()       
    else:
        cfc_only_net.initialize()
    
    return cfc_only_net
    