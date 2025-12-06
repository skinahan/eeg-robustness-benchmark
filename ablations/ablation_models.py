"""
Ablation Model Variants for HYDRA Model

This module contains model variants for ablation studies:
1. BranchedWiredCfCNoCarryGate - Disables weighted residual connection
2. BranchedWiredCfCNoBranching - Processes entire sequence in single temporal bin
"""

import torch
import torch.nn as nn
from models.branched_wiredcfc import BranchedWiredCfC
from globals import get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ExponentialLR
from braindecode import EEGClassifier


class BranchedWiredCfCNoCarryGate(BranchedWiredCfC):
    """BranchedWiredCfC variant with carry gate (weighted residual) disabled."""
    
    def forward(self, x):
        """Override forward to disable carry gate."""
        B, C, T = x.shape

        # DIVA FRONT-END (same as parent)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.avgpool(x)
        T1 = x.shape[-1]
        x = self.dropout1(x)
        x = x.squeeze(2)
        x = self.ms_block(x)
        x = self.snr_gate(x)
        x = self.temporal_downsampler(x)
        T2 = x.shape[-1]

        # BRANCHED RECURRENT COMPARTMENT
        x_bins = self._chunk_time(x)
        Bins = x_bins.size(1)
        L = x_bins.size(2)
        x_bins = x_bins.reshape(B * Bins, L, self.F2)
        residual = x_bins
        x_seq = self._process_bins(x_bins, residual)
        H = x_seq.size(-1)

        # POST-RECURRENT PROCESSING (NO CARRY GATE - skip weighted residual)
        # Skip the weighted residual connection entirely
        # x_seq = (x_seq * (1 - self.weight_residual)) + (residual * self.weight_residual)  # DISABLED
        
        # Intra-bin attention pool
        z_per_bin = self.intra_bin_pool(x_seq)
        z_per_bin = z_per_bin.view(B, Bins, H)

        # Fuse across bins
        if self.fusion == "attn":
            z = self.bin_fusion(z_per_bin)
        else:
            z = z_per_bin.mean(dim=1)

        # CLASSIFICATION HEAD
        z = self.head_norm(z)
        z = self.head_drop(z)
        logits = self.fc(z)
        
        return logits


def create_branched_wiredcfc_no_carry_gate_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Create BranchedWiredCfC classifier with carry gate disabled."""
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
    }
    default_params.update(kwargs)
    
    classifier = EEGClassifier(
        BranchedWiredCfCNoCarryGate,
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


class BranchedWiredCfCNoBranching(BranchedWiredCfC):
    """BranchedWiredCfC variant with branching disabled (single temporal bin)."""
    
    def _chunk_time(self, x_feat):
        """Override to create a single bin covering the entire sequence."""
        B, F, T = x_feat.shape
        # Create a single bin covering the entire temporal dimension
        # Format: [B, NB=1, L=T, F]
        # We need to ensure the format matches what the parent expects
        x_bins = x_feat.unsqueeze(1)  # [B, 1, F, T]
        x_bins = x_bins.permute(0, 1, 3, 2)  # [B, 1, T, F]
        return x_bins


def create_branched_wiredcfc_no_branching_classifier(n_chans, n_times, n_outputs, wiring, **kwargs):
    """Create BranchedWiredCfC classifier with branching disabled."""
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
    }
    default_params.update(kwargs)
    
    classifier = EEGClassifier(
        BranchedWiredCfCNoBranching,
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

