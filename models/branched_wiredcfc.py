import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from braindecode import EEGClassifier
from models.branched_diva_base import BranchedDIVABase
from ncps.torch import CfC
from ncps.wirings import Wiring
from architecture_refinement.arbitrary_wiring import ArbitraryWiring


class BranchedWiredCfC(BranchedDIVABase):
    """
    Hybrid model combining DIVA's front-end with branched CfC recurrent processing using ArbitraryWiring.
    
    Architecture:
      1. DIVA front-end: CNN + Multi-scale temporal + SNR gate + temporal downsampler
      2. Branched recurrent processing: Split into bins, parallel CfC per bin using ArbitraryWiring
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
        wiring: ArbitraryWiring,  # ArbitraryWiring instance (or WsFlexHiddenWiring that builds to ArbitraryWiring)
        # --- CNN (EEGNet-like) front-end ---
        F1: int = 8,                 # temporal filter count
        D: int = 2,                  # depthwise multiplier -> F2 = F1*D
        kernel_length: int = 125,     # temporal kernel in first conv
        pool_time: int = 4,          # anti-alias temporal pooling/stride
        drop_prob: float = 0.25,
        # --- Multi-scale temporal integration ---
        ms_kernels=(9, 15, 31),
        ms_dilations=(1, 4, 16),
        # --- Temporal downsampler (Conv1D) before CfC ---
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        # --- CfC core parameters ---
        recurrent_output_size: int = None, # if None, defaults to F2 for residual compatibility
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
        # Empirically validated to outperform correct ReZero. See REZERO_BACKWARDS_ANALYSIS.md
        residual_init_strategy: str = "backwards_rezero",  # "backwards_rezero" (default, empirically superior) or "correct_rezero" (standard ReZero)
        # --- CfC-specific parameters ---
        mixed_memory: bool = True,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
    ):
        # Store the wiring and CfC parameters
        self.wiring = wiring
        self.mixed_memory = mixed_memory
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        
        # Call parent constructor with CfC-specific parameters
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
            recurrent_output_size=recurrent_output_size,
            bin_len=bin_len,
            bin_stride=bin_stride,
            fusion=fusion,
            snr_reduction=snr_reduction,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            use_spectral_norm_first_conv=use_spectral_norm_first_conv,
            residual_init_strategy=residual_init_strategy,
            # CfC-specific parameters
            mixed_memory=mixed_memory,
            mode=mode,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )

    def _create_recurrent_cell(self, mixed_memory=True, mode="default", activation="lecun_tanh", 
                              backbone_units=128, backbone_layers=1, backbone_dropout=0.0, **kwargs):
        """
        Create the CfC recurrent cell using ArbitraryWiring.
        
        Uses projection layers to handle input/output size mismatches, preserving
        the exact wiring graph that was optimized.
        
        Args:
            mixed_memory: Whether to use mixed memory in CfC
            mode: CfC mode ("default", "pure", etc.)
            activation: Activation function for CfC
            backbone_units: Number of backbone units
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout rate
            **kwargs: Additional parameters (ignored for CfC)
            
        Returns:
            The CfC cell with original wiring sizes
        """
        # Get the original wiring sizes (from the architecture file)
        wiring_input_size = self.wiring.input_size
        wiring_output_size = self.wiring.output_size
        
        # Get the model's expected sizes
        model_input_size = self.F2  # What we actually have
        model_output_size = self.recurrent_output_size  # What we need
        
        # Build wiring with its original input size (preserves exact wiring graph)
        # The wiring might be a WsFlexHiddenWiring (needs build) or already an ArbitraryWiring
        if hasattr(self.wiring, 'build') and not isinstance(self.wiring, ArbitraryWiring):
            # It's a WsFlexHiddenWiring - build it to get ArbitraryWiring
            built_wiring = self.wiring.build(wiring_input_size)
            if built_wiring is not None:
                built_wiring.build(wiring_input_size)
            else:
                raise ValueError("wiring.build() returned None")
        else:
            # Already an ArbitraryWiring - just build it
            built_wiring = self.wiring
            if not built_wiring.is_built():
                built_wiring.build(wiring_input_size)
        
        # Create projection layers if sizes don't match (store as instance attributes)
        if model_input_size != wiring_input_size:
            self.input_proj = nn.Linear(model_input_size, wiring_input_size, bias=False)
        else:
            self.input_proj = None
        
        if model_output_size != wiring_output_size:
            self.output_proj = nn.Linear(wiring_output_size, model_output_size, bias=False)
        else:
            self.output_proj = None
        
        # Create CfC with the original wiring sizes
        return CfC(
            input_size=wiring_input_size,
            units=built_wiring,
            proj_size=wiring_output_size,  # Use wiring's original output size
            return_sequences=True,
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=mode,
            activation=activation,
        )

    def _process_bins(self, x_bins, residual):
        """
        Process temporal bins through the CfC cell with projection layers if needed.
        
        Args:
            x_bins: [B*NB, L, F2] - reshaped bins for parallel processing
            residual: [B*NB, L, F2] - residual connection data (not used in processing)
            
        Returns:
            x_seq: [B*NB, L, recurrent_output_size] - processed sequences from CfC
        """
        # Project input if needed (F2 -> wiring input_size)
        if self.input_proj is not None:
            x_bins = self.input_proj(x_bins)  # [B*NB, L, wiring_input_size]
        
        # CfC over each bin (return sequences)
        x_seq, _ = self.recurrent_cell(x_bins)  # [B*NB, L, wiring_output_size]
        
        # Project output if needed (wiring output_size -> recurrent_output_size)
        if self.output_proj is not None:
            x_seq = self.output_proj(x_seq)  # [B*NB, L, recurrent_output_size]
        
        return x_seq

    def get_wiring_info(self):
        """Get information about the wiring structure."""
        return self.wiring.get_wiring_summary()


def create_branched_wiredcfc_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Create the BranchedWiredCfC classifier."""
    seed = get_seed()
    gradient_clip_value = 1.0
    
    # Default parameters
    default_params = {
        'lstm_hidden_dim': 32,
        'drop_prob': 0.5,
        'F1': 8,
        'D': 2,
        'kernel_length': 125,
        'temporal_kernel_size': 3,
        'temporal_stride': 2,
        'mixed_memory': True,
        'mode': 'default',
        'activation': 'lecun_tanh',
        'backbone_units': 128,
        'backbone_layers': 1,
        'backbone_dropout': 0.0,
    }
    
    # Update with any provided parameters
    default_params.update(kwargs)
    
    classifier = EEGClassifier(
        BranchedWiredCfC,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-2,
        optimizer__weight_decay=0,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__wiring=wiring,
        module__drop_prob=default_params['drop_prob'],
        module__F1=default_params['F1'],
        module__D=default_params['D'],
        module__kernel_length=default_params['kernel_length'],
        module__temporal_kernel_size=default_params['temporal_kernel_size'],
        module__temporal_stride=default_params['temporal_stride'],
        module__mixed_memory=default_params['mixed_memory'],
        module__mode=default_params['mode'],
        module__activation=default_params['activation'],
        module__backbone_units=default_params['backbone_units'],
        module__backbone_layers=default_params['backbone_layers'],
        module__backbone_dropout=default_params['backbone_dropout'],
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
