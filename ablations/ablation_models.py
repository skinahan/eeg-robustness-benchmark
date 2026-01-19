"""
Ablation Model Variants for HYDRA Model

This module provides a unified modular ablation class that can handle all combinations
of component ablations through simple boolean flags, maximizing code reuse.
"""

import torch
import torch.nn as nn
from models.branched_diva_base import BranchedDIVABase
from models.branched_wiredcfc import BranchedWiredCfC
from models.branched_lstm import BranchedLSTM
from ncps.torch import CfC
from architecture_refinement.arbitrary_wiring import ArbitraryWiring
from globals import get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ExponentialLR
from braindecode import EEGClassifier


class ModularAblationModel(BranchedDIVABase):
    """
    Unified modular ablation model that supports all combinations of component ablations.
    
    This class replaces all individual ablation classes by using boolean flags to control
    which components are enabled/disabled:
    - use_carry_gate: Enable/disable weighted residual connection
    - use_branching: Enable/disable temporal binning (if False, uses single bin)
    - use_snr_gate: Enable/disable SNR gate (Wiener-like shrinkage)
    - recurrent_type: 'cfc' or 'lstm' to choose recurrent cell type
    
    Args:
        wiring: ArbitraryWiring instance (required for 'cfc' recurrent_type)
        use_carry_gate: If True, apply weighted residual connection (default: True)
        use_branching: If True, use temporal binning with overlap (default: True)
        use_snr_gate: If True, apply SNR gate (default: True)
        recurrent_type: 'cfc' or 'lstm' (default: 'cfc')
        **kwargs: All other parameters passed to BranchedDIVABase
    """
    
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        wiring: ArbitraryWiring = None,  # Required for CFC, None for LSTM
        use_carry_gate: bool = True,
        use_branching: bool = True,
        use_snr_gate: bool = True,
        recurrent_type: str = 'cfc',  # 'cfc' or 'lstm'
        # LSTM-specific parameters (ignored for CFC)
        lstm_hidden_dim: int = 32,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        # CFC-specific parameters (ignored for LSTM)
        mixed_memory: bool = True,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        **kwargs
    ):
        # Store ablation flags
        self.use_carry_gate = use_carry_gate
        self.use_branching = use_branching
        self.use_snr_gate = use_snr_gate
        self.recurrent_type = recurrent_type.lower()
        
        assert self.recurrent_type in {'cfc', 'lstm'}, \
            f"recurrent_type must be 'cfc' or 'lstm', got '{recurrent_type}'"
        
        if self.recurrent_type == 'cfc':
            assert wiring is not None, "wiring is required for CFC recurrent_type"
            self.wiring = wiring
            self.mixed_memory = mixed_memory
            self.mode = mode
            self.activation = activation
            self.backbone_units = backbone_units
            self.backbone_layers = backbone_layers
            self.backbone_dropout = backbone_dropout
            # Pass CFC-specific params as recurrent_kwargs
            recurrent_kwargs = {
                'mixed_memory': mixed_memory,
                'mode': mode,
                'activation': activation,
                'backbone_units': backbone_units,
                'backbone_layers': backbone_layers,
                'backbone_dropout': backbone_dropout,
            }
        else:  # LSTM
            self.lstm_hidden_dim = lstm_hidden_dim
            self.lstm_num_layers = lstm_num_layers
            self.lstm_dropout = lstm_dropout
            # Pass LSTM-specific params as recurrent_kwargs
            recurrent_kwargs = {
                'lstm_hidden_dim': lstm_hidden_dim,
                'lstm_num_layers': lstm_num_layers,
                'lstm_dropout': lstm_dropout,
            }
        
        # Call parent constructor with recurrent_kwargs merged into kwargs
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            **{**kwargs, **recurrent_kwargs}
        )
    
    def _create_recurrent_cell(self, **kwargs):
        """Create the recurrent cell based on recurrent_type."""
        if self.recurrent_type == 'cfc':
            # CFC with ArbitraryWiring
            ncp_input_size = self.F2
            ncp_output_size = self.recurrent_output_size
            
            self.wiring.input_size = ncp_input_size
            self.wiring.output_size = ncp_output_size
            built_wiring = self.wiring.build(ncp_input_size)
            
            return CfC(
                input_size=ncp_input_size,
                units=built_wiring,
                proj_size=ncp_output_size,
                return_sequences=True,
                batch_first=True,
                mixed_memory=self.mixed_memory,
                mode=self.mode,
                activation=self.activation,
            )
        else:  # LSTM
            return nn.LSTM(
                input_size=self.F2,
                hidden_size=self.recurrent_output_size,
                num_layers=self.lstm_num_layers,
                dropout=self.lstm_dropout if self.lstm_num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False
            )
    
    def _process_bins(self, x_bins, residual):
        """Process temporal bins through the recurrent cell."""
        if self.recurrent_type == 'cfc':
            # CFC processing
            x_seq, _ = self.recurrent_cell(x_bins)
        else:  # LSTM
            # LSTM processing
            x_seq, _ = self.recurrent_cell(x_bins)
        return x_seq
    
    def _chunk_time(self, x_feat):
        """Override to conditionally enable/disable branching."""
        if self.use_branching:
            # Use parent's branching implementation
            return super()._chunk_time(x_feat)
        else:
            # Single bin covering entire sequence
            B, F, T = x_feat.shape
            x_bins = x_feat.unsqueeze(1)  # [B, 1, F, T]
            x_bins = x_bins.permute(0, 1, 3, 2)  # [B, 1, T, F]
            return x_bins
    
    def forward(self, x):
        """Forward pass with conditional component application."""
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

        # (6) SNR gate (Wiener-like shrinkage) - CONDITIONAL
        if self.use_snr_gate:
            x = self.snr_gate(x)                           # (B, F2, T1)

        # (7) Temporal downsampler (Conv1D)
        x = self.temporal_downsampler(x)                   # (B, F2, T2)
        T2 = x.shape[-1]

        # -------------------------
        # BRANCHED RECURRENT COMPARTMENT
        # -------------------------
        
        # Chunk into bins (conditionally branched)
        x_bins = self._chunk_time(x)                       # [B, NB, L, F2]
        Bins = x_bins.size(1)
        L = x_bins.size(2)
        F2_ = x_bins.size(3)
        assert F2_ == self.F2

        # Merge batch and bins to run recurrent processing in parallel
        x_bins = x_bins.reshape(B * Bins, L, self.F2)     # [B*NB, L, F2]

        # Store residual for weighted connection
        residual = x_bins                                  # [B*NB, L, F2]

        # Process bins through recurrent cell
        x_seq = self._process_bins(x_bins, residual)       # [B*NB, L, H]
        H = x_seq.size(-1)

        # -------------------------
        # POST-RECURRENT PROCESSING
        # -------------------------
        
        # Apply weighted residual connection (DIVA-style) - CONDITIONAL
        if self.use_carry_gate:
            x_seq = (x_seq * (1 - self.weight_residual)) + (residual * self.weight_residual)
        # If carry gate disabled, use recurrent output directly
        
        # Intra-bin attention pool
        z_per_bin = self.intra_bin_pool(x_seq)
        z_per_bin = z_per_bin.view(B, Bins, H)

        # Fuse across bins
        if self.fusion == "attn":
            z = self.bin_fusion(z_per_bin)
        else:
            z = z_per_bin.mean(dim=1)

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        z = self.head_norm(z)
        z = self.head_drop(z)
        logits = self.fc(z)
        
        return logits


# ============================================================================
# Factory Functions for All Ablation Variants
# ============================================================================

def _create_ablation_classifier(
    ablation_name: str,
    use_carry_gate: bool,
    use_branching: bool,
    use_snr_gate: bool,
    recurrent_type: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    wiring: ArbitraryWiring = None,
    **kwargs
):
    """
    Generic factory function to create any ablation variant.
    
    Args:
        ablation_name: Name for the ablation (for logging/debugging)
        use_carry_gate: Enable carry gate
        use_branching: Enable branching
        use_snr_gate: Enable SNR gate
        recurrent_type: 'cfc' or 'lstm'
        n_chans, n_times, n_outputs: Model dimensions
        wiring: ArbitraryWiring instance (required for CFC)
        **kwargs: Additional parameters
    """
    seed = get_seed()
    gradient_clip_value = 1.0
    
    default_params = {
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
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'lstm_dropout': 0.0,
    }
    default_params.update(kwargs)
    
    classifier = EEGClassifier(
        ModularAblationModel,
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
        module__use_carry_gate=use_carry_gate,
        module__use_branching=use_branching,
        module__use_snr_gate=use_snr_gate,
        module__recurrent_type=recurrent_type,
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
        module__lstm_hidden_dim=default_params['lstm_hidden_dim'],
        module__lstm_num_layers=default_params['lstm_num_layers'],
        module__lstm_dropout=default_params['lstm_dropout'],
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


# Individual factory functions for each ablation (for backward compatibility and clarity)

def create_branched_wiredcfc_no_carry_gate_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 1: No Carry Gate"""
    return _create_ablation_classifier(
        "no_carry_gate", use_carry_gate=False, use_branching=True, 
        use_snr_gate=True, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_wiredcfc_no_branching_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 2: No Branching"""
    return _create_ablation_classifier(
        "no_branching", use_carry_gate=True, use_branching=False,
        use_snr_gate=True, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 3: LSTM Replacement (baseline LSTM with all features)"""
    return _create_ablation_classifier(
        "lstm", use_carry_gate=True, use_branching=True,
        use_snr_gate=True, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_wiredcfc_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 4: No SNR Gate"""
    return _create_ablation_classifier(
        "no_snr_gate", use_carry_gate=True, use_branching=True,
        use_snr_gate=False, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


# Combination ablations (CfC-based)
def create_branched_wiredcfc_no_carry_gate_no_branching_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 5: No Carry Gate + No Branching"""
    return _create_ablation_classifier(
        "no_carry_gate_no_branching", use_carry_gate=False, use_branching=False,
        use_snr_gate=True, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_wiredcfc_no_carry_gate_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 6: No Carry Gate + No SNR Gate"""
    return _create_ablation_classifier(
        "no_carry_gate_no_snr_gate", use_carry_gate=False, use_branching=True,
        use_snr_gate=False, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_wiredcfc_no_branching_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 7: No Branching + No SNR Gate"""
    return _create_ablation_classifier(
        "no_branching_no_snr_gate", use_carry_gate=True, use_branching=False,
        use_snr_gate=False, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_wiredcfc_no_carry_gate_no_branching_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Ablation 8: No Carry Gate + No Branching + No SNR Gate"""
    return _create_ablation_classifier(
        "no_carry_gate_no_branching_no_snr_gate", use_carry_gate=False, use_branching=False,
        use_snr_gate=False, recurrent_type='cfc',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


# Combination ablations (LSTM-based)
def create_branched_lstm_no_carry_gate_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 9: LSTM + No Carry Gate"""
    return _create_ablation_classifier(
        "lstm_no_carry_gate", use_carry_gate=False, use_branching=True,
        use_snr_gate=True, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_no_branching_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 10: LSTM + No Branching"""
    return _create_ablation_classifier(
        "lstm_no_branching", use_carry_gate=True, use_branching=False,
        use_snr_gate=True, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 11: LSTM + No SNR Gate"""
    return _create_ablation_classifier(
        "lstm_no_snr_gate", use_carry_gate=True, use_branching=True,
        use_snr_gate=False, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_no_carry_gate_no_branching_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 12: LSTM + No Carry Gate + No Branching"""
    return _create_ablation_classifier(
        "lstm_no_carry_gate_no_branching", use_carry_gate=False, use_branching=False,
        use_snr_gate=True, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_no_carry_gate_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 13: LSTM + No Carry Gate + No SNR Gate"""
    return _create_ablation_classifier(
        "lstm_no_carry_gate_no_snr_gate", use_carry_gate=False, use_branching=True,
        use_snr_gate=False, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_no_branching_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 14: LSTM + No Branching + No SNR Gate"""
    return _create_ablation_classifier(
        "lstm_no_branching_no_snr_gate", use_carry_gate=True, use_branching=False,
        use_snr_gate=False, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )


def create_branched_lstm_no_carry_gate_no_branching_no_snr_gate_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """Ablation 15: LSTM + No Carry Gate + No Branching + No SNR Gate"""
    return _create_ablation_classifier(
        "lstm_no_carry_gate_no_branching_no_snr_gate", use_carry_gate=False, use_branching=False,
        use_snr_gate=False, recurrent_type='lstm',
        n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, wiring=wiring, **kwargs
    )
