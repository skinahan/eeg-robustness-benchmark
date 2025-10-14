import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from braindecode import EEGClassifier
from braindecode.models.base import EEGModuleMixin
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# assumes:
# from braindecode.models.util import EEGModuleMixin
# and your local get_seed()

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


class CNNNCP_BranchedBins(EEGModuleMixin, nn.Module):
    """
    CNN front-end -> temporal downsample -> chunk into bins ->
    parallel CfC per bin (batched) -> attention-pool within bin ->
    fuse across bins (mean or attention) -> classifier.

    Key ideas:
      - Binning keeps recurrence short & parallelizable.
      - Attention pooling reduces noise within each bin.
      - Fusion restores inter-bin context at low cost.

    Expected input: x [B, n_chans, n_times]
    """
    def __init__(
        self,
        n_chans,
        n_times,
        n_outputs,
        # Front-end CNN params
        F1=8, D=2, kernel_length=125,
        temporal_pool=4, temporal_pool_stride=4,
        drop_prob=0.15,
        # Temporal downsampler (post-CNN)
        temporal_kernel_size=3, temporal_stride=2,
        # CfC / NCP params
        ncp_hidden_dim=36, ncp_out_dim=8, sparsity=0.7, mixed_memory=True,
        # Binning params
        bin_len=64,            # number of timesteps per bin AFTER downsampling
        bin_stride=48,         # step between bin starts; set < bin_len for overlap
        fusion="attn",         # "attn" or "mean"
        # Other
        seed=None,
        batch_norm_momentum=0.01,
        batch_norm_eps=1e-3,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed() if seed is None else seed

        self.F1 = F1
        F2 = F1 * D
        self.bin_len = bin_len
        self.bin_stride = bin_stride
        self.fusion = fusion.lower()
        assert self.fusion in {"attn", "mean"}

        # -----------------------------
        # 1) Temporal Conv (time axis)
        # -----------------------------
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, kernel_length),
            stride=(1, 1),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1, momentum=batch_norm_momentum, eps=batch_norm_eps)
        self.elu = nn.ELU()

        # -----------------------------------------
        # 2) Depthwise Spatial Conv (EEGNet-style)
        #    result shape: [B, F2, 1, T]
        # -----------------------------------------
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1, out_channels=F2,
            kernel_size=(n_chans, 1),
            groups=F1,
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps)

        # -----------------------------------------------------
        # 3) AvgPool over TIME (not channels) to shrink T early
        #    -> pick small pool/stride; tune as needed
        # -----------------------------------------------------
        self.avgpool = nn.AvgPool2d(kernel_size=(1, temporal_pool),
                                    stride=(1, temporal_pool_stride))

        self.dropout = nn.Dropout(p=drop_prob)

        # -------------------------------------------------
        # 4) Optional temporal downsampler (Conv1d over T)
        #     Input to this block will be [B, F2, T']
        # -------------------------------------------------
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2, out_channels=F2,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2,
            bias=True,
        )

        # ----------------------------
        # 5) CfC cell for per-bin pass
        # ----------------------------
        # Input dim to CfC equals feature channels F2
        wiring = AutoNCP(ncp_hidden_dim, ncp_out_dim, sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(
            input_size=F2,
            units=wiring,
            return_sequences=True,   # we pool over time afterward
            mixed_memory=mixed_memory,
        )

        # Pool within each bin (across its timesteps)
        self.intra_bin_pool = TemporalAttnPool(dim=ncp_out_dim)

        # ---------------------------------------
        # 6) Fusion across bins (restore context)
        # ---------------------------------------
        if self.fusion == "attn":
            self.bin_fusion = TemporalAttnPool(dim=ncp_out_dim)  # pool over bins
        else:
            self.bin_fusion = None  # mean over bins

        # (Optional) light head before logits
        self.head_norm = nn.LayerNorm(ncp_out_dim)
        self.head_drop = nn.Dropout(p=drop_prob)

        # -------------------
        # 7) Classifier head
        # -------------------
        self.fc = nn.Linear(ncp_out_dim, n_outputs)

        self._glorot_weight_zero_bias()

    # ---------
    # Utilities
    # ---------
    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight") and module.weight is not None:
                if "BatchNorm" not in module.__class__.__name__:
                    # Xavier initialization only works for tensors with 2+ dimensions
                    if module.weight.ndim >= 2:
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    nn.init.constant_(module.weight, 1.0)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

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

    # -------
    # Forward
    # -------
    def forward(self, x):
        """
        x: [B, n_chans, n_times]
        """
        B = x.size(0)

        # 1) Temporal conv on time axis
        x = x.unsqueeze(1)                       # [B, 1, C, T]
        x = self.conv1(x)                        # [B, F1, C, T]
        x = self.bn1(x)
        x = self.elu(x)

        # 2) Depthwise spatial conv
        x = self.depthwise_conv(x)               # [B, F2, 1, T]
        x = self.bn2(x)
        x = self.elu(x)

        # 3) Average pool over time to reduce T
        x = self.avgpool(x)                      # [B, F2, 1, T1]
        x = self.dropout(x)

        # Flatten spatial dim and permute to [B, F2, T1]
        x = x.squeeze(2)                         # [B, F2, T1]

        # 4) Temporal downsampler (Conv1d over T1)
        x = self.temporal_downsampler(x)         # [B, F2, T2]

        # Save shapes
        B_, F2, T2 = x.shape
        assert B_ == B

        # 5) Chunk into bins
        x_bins = self._chunk_time(x)             # [B, NB, L, F2]
        Bins = x_bins.size(1)
        L    = x_bins.size(2)
        F2_  = x_bins.size(3)
        assert F2_ == F2

        # Merge batch and bins to run CfC in parallel
        x_bins = x_bins.reshape(B * Bins, L, F2)  # [B*NB, L, F2]

        # CfC over each bin (return sequences to pool over time within-bin)
        x_seq, _ = self.ncp(x_bins)              # [B*NB, L, H]
        H = x_seq.size(-1)

        # Intra-bin attention pool -> per-bin summary
        z_per_bin = self.intra_bin_pool(x_seq)   # [B*NB, H]
        z_per_bin = z_per_bin.view(B, Bins, H)   # [B, NB, H]

        # 6) Fuse across bins
        if self.fusion == "attn":
            z = self.bin_fusion(z_per_bin)       # [B, H]  (attn over NB)
        else:
            z = z_per_bin.mean(dim=1)            # [B, H]  (mean over NB)

        # 7) Head + logits
        z = self.head_norm(z)
        z = self.head_drop(z)
        logits = self.fc(z)                      # [B, n_outputs]
        return logits


def create_cnnncp_branched_bins_classifier(
    n_chans,
    n_times,
    n_outputs,
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=64,
    net_size=12,
    net_sparsity=0.8,
    gradient_clip_value=1.0,
    **kwargs
):
    classifier = CNNNCP_BranchedBins
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

