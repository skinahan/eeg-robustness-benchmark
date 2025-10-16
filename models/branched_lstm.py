import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from braindecode import EEGClassifier
from models.branched_diva_base import BranchedDIVABase


class BranchedLSTM(BranchedDIVABase):
    """
    Hybrid model combining DIVA's front-end with branched LSTM recurrent processing.
    
    Architecture:
      1. DIVA front-end: CNN + Multi-scale temporal + SNR gate + temporal downsampler
      2. Branched recurrent processing: Split into bins, parallel LSTM per bin
      3. Weighted residual connections (DIVA-style) at bin level
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
        kernel_length: int = 125,     # temporal kernel in first conv
        pool_time: int = 4,          # anti-alias temporal pooling/stride
        drop_prob: float = 0.25,
        # --- Multi-scale temporal integration ---
        ms_kernels=(9, 15, 31),
        ms_dilations=(1, 4, 16),
        # --- Temporal downsampler (Conv1D) before LSTM ---
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        # --- LSTM core ---
        lstm_hidden_dim: int = 32,    # hidden dimension for LSTM (used as recurrent_output_size)
        lstm_num_layers: int = 1,     # number of LSTM layers
        lstm_dropout: float = 0.0,    # dropout between LSTM layers
        lstm_output_size: int = None, # if None, defaults to F2 for residual compatibility
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
    ):
        # Call parent constructor with LSTM-specific parameters
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            F1=F1,
            D=D,
            kernel_length=kernel_length,
            pool_time=pool_time,
            drop_prob=drop_prob,
            ms_kernels=ms_kernels,
            ms_dilations=ms_dilations,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            recurrent_output_size=lstm_output_size,
            bin_len=bin_len,
            bin_stride=bin_stride,
            fusion=fusion,
            snr_reduction=snr_reduction,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            use_spectral_norm_first_conv=use_spectral_norm_first_conv,
            # LSTM-specific parameters
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
        )

    def _create_recurrent_cell(self, lstm_hidden_dim=64, lstm_num_layers=1, lstm_dropout=0.0, **kwargs):
        """
        Create the LSTM recurrent cell.
        
        Args:
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: Dropout between LSTM layers
            **kwargs: Additional parameters (ignored for LSTM)
            
        Returns:
            The LSTM cell
        """
        # Ensure LSTM output size matches recurrent_output_size for compatibility
        # with base class attention pooling and residual connections
        return nn.LSTM(
            input_size=self.F2,
            hidden_size=self.recurrent_output_size,  # Use recurrent_output_size instead of lstm_hidden_dim
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

    def _process_bins(self, x_bins, residual):
        """
        Process temporal bins through the LSTM cell.
        
        Args:
            x_bins: [B*NB, L, F2] - reshaped bins for parallel processing
            residual: [B*NB, L, F2] - residual connection data (not used in processing)
            
        Returns:
            x_seq: [B*NB, L, H] - processed sequences from LSTM
        """
        # LSTM over each bin (return sequences)
        x_seq, _ = self.recurrent_cell(x_bins)             # [B*NB, L, H]
        return x_seq


def create_branched_lstm_classifier(n_chans, n_times, n_outputs):
    """Create the BranchedLSTM classifier."""
    seed = get_seed()
    gradient_clip_value = 1.0
    classifier = EEGClassifier(
        BranchedLSTM,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        optimizer__weight_decay=0,#1e-4,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__lstm_hidden_dim=32,
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
            # GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2),
            # LRScheduler(policy=ExponentialLR, gamma=0.97),
        ],
        verbose=EEGCLASSIFIER_VERBOSE
    )

    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()

    return classifier
