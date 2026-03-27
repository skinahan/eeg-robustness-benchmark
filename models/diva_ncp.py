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

class DIVANCP(EEGModuleMixin, nn.Module):
    """
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
        sparsity: float = 0.75,
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
        rng = np.random.default_rng(seed)
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
        #     Input size = F2; Output size = ncp_output_size (default F2).
        #     return_sequences=True -> (B, T2, H)
        # -------------------------
        if ncp_output_size is None:
            ncp_output_size = self.F2

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

        # 10. Initialize the residual connection weight
        self.weight_residual = nn.Parameter(torch.from_numpy(rng.uniform(0.1, 0.9, (1,))).float())

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
         # 7. CfC processing with residual connection:
        residual = x  # Store the input for residual connection
        x, _ = self.ncp(x)  # (B, T2, H)
        
        # Instead of a straightforward sum,
        # Use a learned weighted combination of the two.
        x = (x * (1 - self.weight_residual)) + (residual * self.weight_residual)

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


def create_diva_ncp_classifier(n_chans, n_times, n_outputs, **kwargs):
    """Create the official CNNCfCv2 classifier."""
    seed = get_seed()
    gradient_clip_value = 1.0
    classifier = EEGClassifier(
        DIVANCP,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3, # TODO: This should be 1e-2 for consistency with the branched variant...
        optimizer__weight_decay=0,
        batch_size=32,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=19,
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
            # LRScheduler(policy=ExponentialLR, gamma=0.97),
            # LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=10),
            ],
        verbose=EEGCLASSIFIER_VERBOSE
    )

    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()

    return classifier
