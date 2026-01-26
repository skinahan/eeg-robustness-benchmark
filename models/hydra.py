import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from braindecode import EEGClassifier
from models.branched_wiredcfc import BranchedWiredCfC
from ncps.torch import CfC
from architecture_refinement.arbitrary_wiring import ArbitraryWiring
import numpy as np


class MultiQueryTemporalAttnPool(nn.Module):
    """
    Multi-query attention pooling over time with attention dropout.
    Replaces single-query pooling to prevent single-timestep overfitting.
    
    Inputs:  x [B, T, C]
    Output:  z [B, M*C] where M is the number of queries
    """
    def __init__(self, dim, num_queries=4, attn_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        # M queries instead of one
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        # [B, T, C]
        B, T, C = x.shape
        # Project keys: [B, T, C]
        k = torch.tanh(self.proj(x))
        
        # Attention scores for each query: [B, T, M]
        # Using einsum: "btc,mc->btm"
        att = torch.einsum("btc,mc->btm", k, self.queries) / (C ** 0.5)
        w = torch.softmax(att, dim=1)  # [B, T, M]
        w = self.attn_dropout(w)  # Apply attention dropout
        
        # Weighted sum for each query: [B, M, C]
        # Using einsum: "btm,btc->bmc"
        z_m = torch.einsum("btm,btc->bmc", w, x)  # [B, M, C]
        
        # Flatten to [B, M*C]
        z = z_m.reshape(B, self.num_queries * C)
        return z


class AdaptiveResidualGate(nn.Module):
    """
    Per-bin adaptive residual (carry) gating.
    Makes residual weight conditional on SNR + feature statistics.
    
    Input:  r_b [B*NB, L, F2], h_b [B*NB, L, H]
    Output: alpha_b [B*NB, 1, 1] - per-bin gating weights
    """
    def __init__(self, feature_dim, reduction=4):
        super().__init__()
        hidden = max(1, feature_dim // reduction)
        # Gate network: stats(r) + stats(h) + SNR -> alpha
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden),  # mean_r, var_r, mean_h, var_h
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # alpha in [0, 1]
        )

    def forward(self, r_b, h_b):
        """
        Args:
            r_b: [B*NB, L, F2] - residual features
            h_b: [B*NB, L, H] - recurrent output (should match F2 dim-wise)
        Returns:
            alpha_b: [B*NB, 1, 1] - per-bin gating weights
        """
        # Compute statistics for residual
        mean_r = r_b.mean(dim=1, keepdim=False)  # [B*NB, F2]
        var_r = r_b.var(dim=1, keepdim=False, unbiased=False)  # [B*NB, F2]
        
        # Compute statistics for recurrent output
        mean_h = h_b.mean(dim=1, keepdim=False)  # [B*NB, H] (should be F2)
        var_h = h_b.var(dim=1, keepdim=False, unbiased=False)  # [B*NB, H]
        
        # Approximate SNR: signal variance relative to noise variance
        # Higher SNR -> trust recurrence more (lower alpha for backwards_rezero)
        snr_r = mean_r.abs() / (var_r + 1e-6)
        # Combine statistics
        stats = torch.cat([mean_r, var_r, mean_h, var_h], dim=-1)  # [B*NB, 4*F2]
        
        # Compute per-bin gating weight
        alpha_b = self.gate_net(stats)  # [B*NB, 1]
        alpha_b = alpha_b.unsqueeze(-1)  # [B*NB, 1, 1]
        
        return alpha_b


class ERPEvidenceHead(nn.Module):
    """
    ERP head: Multi-query attention pooling over time (time-locked evidence).
    Models ERP latencies as a learned matched filter bank.
    
    Input:  x [B, F, T] - pre-binned features
    Output: e_ERP [B, M*F] where M is number of queries
    """
    def __init__(self, feature_dim, num_queries=4, attn_dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        # Multi-query temporal attention
        self.attn_pool = MultiQueryTemporalAttnPool(
            dim=feature_dim, 
            num_queries=num_queries,
            attn_dropout=attn_dropout
        )

    def forward(self, x):
        """
        Args:
            x: [B, F, T] - temporal features before binning
        Returns:
            e_ERP: [B, M*F] - ERP evidence vector
        """
        # Transpose to [B, T, F] for attention pooling
        x_t = x.transpose(1, 2)  # [B, T, F]
        e_ERP = self.attn_pool(x_t)  # [B, M*F]
        return e_ERP


class SSVEPEvidenceHead(nn.Module):
    """
    SSVEP head: Learnable bandpower approximation (frequency-locked evidence).
    
    Input:  x [B, F, T] - pre-binned features
    Output: e_SSVEP [B, K*F] where K is number of filters
    """
    def __init__(self, feature_dim, num_filters=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_filters = num_filters
        # Learnable filter bank (Conv1D filters)
        self.filter_bank = nn.ModuleList([
            nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=feature_dim,  # depthwise
                bias=False
            )
            for kernel_size in [3, 5, 9, 15]  # Different temporal scales
        ])
        # Ensure we have exactly num_filters
        while len(self.filter_bank) < num_filters:
            self.filter_bank.append(nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=7,
                padding=3,
                groups=feature_dim,
                bias=False
            ))
        self.filter_bank = self.filter_bank[:num_filters]

    def forward(self, x):
        """
        Args:
            x: [B, F, T] - temporal features before binning
        Returns:
            e_SSVEP: [B, K*F] - SSVEP evidence vector
        """
        # Apply each filter and compute power (mean of squared magnitude)
        powers = []
        for filt in self.filter_bank:
            u_k = filt(x)  # [B, F, T]
            p_k = (u_k.abs() ** 2).mean(dim=2)  # [B, F] - bandpower
            powers.append(p_k)
        
        # Concatenate: [B, K*F]
        e_SSVEP = torch.cat(powers, dim=1)
        return e_SSVEP


class CrossBinContext(nn.Module):
    """
    Cross-bin context module: Lightweight model across bins for robustness.
    Can be a tiny Transformer or small GRU/CfC.
    
    Input:  z_bins [B, NB, H] - bin embeddings
    Output: z_bins_ctx [B, NB, H] - context-enhanced bin embeddings
    """
    def __init__(self, hidden_dim, context_type="transformer", num_heads=2, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_type = context_type
        
        if context_type == "transformer":
            # Tiny Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.context_net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif context_type == "gru":
            # Small GRU
            self.context_net = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unknown context_type: {context_type}")

    def forward(self, z_bins):
        """
        Args:
            z_bins: [B, NB, H] - bin embeddings
        Returns:
            z_bins_ctx: [B, NB, H] - context-enhanced embeddings
        """
        if self.context_type == "transformer":
            z_bins_ctx = self.context_net(z_bins)  # [B, NB, H]
        elif self.context_type == "gru":
            z_bins_ctx, _ = self.context_net(z_bins)  # [B, NB, H]
        return z_bins_ctx


class HYDRAv2(BranchedWiredCfC):
    """
    HYDRAv2: Robust iterative improvement on BranchedWiredCfC (HYDRA).
    
    Key improvements:
    1. Adaptive residual/carry gating (per-bin, corruption-aware)
    2. Multi-query attention pooling (prevents single-timestep overfitting)
    3. ERP evidence head (time-locked evidence)
    4. SSVEP evidence head (frequency-locked evidence)
    5. Cross-bin context module (robustness)
    6. Global skip path (identity preservation)
    
    Architecture:
    Input → DIVA front-end → multi-scale temporal + SNR gate → downsampled features
    
    Parallel evidence paths:
    - ERP head (time-locked)
    - SSVEP head (frequency-locked)
    - Branched recurrent path (local dynamics)
    - Global skip (identity preservation)
    
    Robust fusion:
    - Adaptive gated recurrence
    - Cross-bin context
    - Multi-query pooling
    
    Classifier head: LayerNorm → Dropout → Linear
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        wiring: ArbitraryWiring,
        # --- Phase 1: Adaptive gating & multi-query pooling ---
        num_attn_queries: int = 4,
        attn_dropout: float = 0.1,
        use_adaptive_residual: bool = True,
        # --- Phase 2: ERP/SSVEP evidence heads ---
        use_erp_head: bool = True,
        use_ssvep_head: bool = True,
        erp_num_queries: int = 4,
        ssvep_num_filters: int = 4,
        # --- Phase 3: Cross-bin context & skip path ---
        use_cross_bin_context: bool = True,
        context_type: str = "transformer",  # "transformer" or "gru"
        context_num_heads: int = 2,
        context_num_layers: int = 1,
        use_global_skip: bool = True,
        # --- Inherit all other params from BranchedWiredCfC ---
        **kwargs
    ):
        # Store HYDRAv2-specific parameters
        self.num_attn_queries = num_attn_queries
        self.use_adaptive_residual = use_adaptive_residual
        self.use_erp_head = use_erp_head
        self.use_ssvep_head = use_ssvep_head
        self.use_cross_bin_context = use_cross_bin_context
        self.use_global_skip = use_global_skip
        
        # Extract drop_prob from kwargs for later use
        self.drop_prob = kwargs.get('drop_prob', 0.25)
        
        # Call parent constructor
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring,
            **kwargs
        )
        
        # -------------------------
        # PHASE 1: Adaptive residual gating + Multi-query pooling
        # -------------------------
        
        # Replace single-query intra-bin pooling with multi-query
        self.intra_bin_pool = MultiQueryTemporalAttnPool(
            dim=self.recurrent_output_size,
            num_queries=num_attn_queries,
            attn_dropout=attn_dropout
        )
        
        # Adaptive residual gate (per-bin)
        if use_adaptive_residual:
            self.adaptive_residual_gate = AdaptiveResidualGate(
                feature_dim=self.F2,
                reduction=4
            )
        else:
            self.adaptive_residual_gate = None
        
        # Update bin fusion to handle multi-query output
        # Output of intra_bin_pool is now [B*NB, M*H] instead of [B*NB, H]
        # We'll handle this in forward()
        if self.fusion == "attn":
            # For attention fusion, we'll use a simpler approach after pooling
            self.bin_fusion = None  # Will use mean or another method
        else:
            self.bin_fusion = None
        
        # -------------------------
        # PHASE 2: ERP & SSVEP evidence heads
        # -------------------------
        
        # Store pre-binned features for ERP/SSVEP heads (set in forward)
        self._pre_binned_features = None
        
        if use_erp_head:
            self.erp_head = ERPEvidenceHead(
                feature_dim=self.F2,
                num_queries=erp_num_queries,
                attn_dropout=attn_dropout
            )
        else:
            self.erp_head = None
        
        if use_ssvep_head:
            self.ssvep_head = SSVEPEvidenceHead(
                feature_dim=self.F2,
                num_filters=ssvep_num_filters
            )
        else:
            self.ssvep_head = None
        
        # -------------------------
        # PHASE 3: Cross-bin context & global skip
        # -------------------------
        
        if use_cross_bin_context:
            # Compute effective hidden dim (after multi-query pooling)
            effective_hidden = self.recurrent_output_size * num_attn_queries
            self.cross_bin_context = CrossBinContext(
                hidden_dim=effective_hidden,
                context_type=context_type,
                num_heads=context_num_heads,
                num_layers=context_num_layers
            )
        else:
            self.cross_bin_context = None
        
        if use_global_skip:
            # Global skip: direct pooling of pre-binned features
            self.global_skip_pool = nn.AdaptiveAvgPool1d(1)  # Global average pool over time
        else:
            self.global_skip_pool = None
        
        # -------------------------
        # UPDATE CLASSIFICATION HEAD
        # -------------------------
        
        # Compute total evidence dimension
        evidence_dims = []
        
        # Recurrent path: [B, M*H] after multi-query pooling
        recurrent_dim = self.recurrent_output_size * num_attn_queries
        evidence_dims.append(recurrent_dim)
        
        # ERP head: [B, M*F]
        if use_erp_head:
            erp_dim = self.F2 * erp_num_queries
            evidence_dims.append(erp_dim)
        
        # SSVEP head: [B, K*F]
        if use_ssvep_head:
            ssvep_dim = self.F2 * ssvep_num_filters
            evidence_dims.append(ssvep_dim)
        
        # Global skip: [B, F]
        if use_global_skip:
            evidence_dims.append(self.F2)
        
        total_evidence_dim = sum(evidence_dims)
        
        # Update classification head
        self.head_norm = nn.LayerNorm(total_evidence_dim)
        self.head_drop = nn.Dropout(p=self.drop_prob)
        self.fc = nn.Linear(total_evidence_dim, n_outputs)

    def forward(self, x):
        """
        Forward pass with all HYDRAv2 improvements.
        """
        B, C, T = x.shape

        # -------------------------
        # DIVA FRONT-END (inherited from parent)
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

        # Store pre-binned features for ERP/SSVEP heads and global skip
        pre_binned_features = x  # [B, F2, T2]

        # -------------------------
        # PARALLEL EVIDENCE PATHS
        # -------------------------
        
        evidence_vectors = []
        
        # ERP evidence head (non-recurrent, time-locked)
        if self.erp_head is not None:
            e_ERP = self.erp_head(pre_binned_features)  # [B, M*F]
            evidence_vectors.append(e_ERP)
        
        # SSVEP evidence head (non-recurrent, frequency-locked)
        if self.ssvep_head is not None:
            e_SSVEP = self.ssvep_head(pre_binned_features)  # [B, K*F]
            evidence_vectors.append(e_SSVEP)
        
        # Global skip path (identity preservation)
        if self.global_skip_pool is not None:
            e_SKIP = self.global_skip_pool(pre_binned_features)  # [B, F2, 1]
            e_SKIP = e_SKIP.squeeze(-1)  # [B, F2]
            evidence_vectors.append(e_SKIP)

        # -------------------------
        # BRANCHED RECURRENT PATH
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

        # Process bins through recurrent cell
        x_seq = self._process_bins(x_bins, residual)       # [B*NB, L, H]
        H = x_seq.size(-1)

        # -------------------------
        # ADAPTIVE RESIDUAL CONNECTION
        # -------------------------
        
        if H == self.F2:
            if self.adaptive_residual_gate is not None:
                # Per-bin adaptive gating
                alpha_b = self.adaptive_residual_gate(residual, x_seq)  # [B*NB, 1, 1]
                
                # Apply backwards_rezero-style formula with adaptive alpha
                # Formula: recurrent*(1-alpha_b) + residual*alpha_b
                x_seq = (x_seq * (1 - alpha_b)) + (residual * alpha_b)
            else:
                # Fall back to global residual (backwards_rezero)
                if self.residual_init_strategy == "backwards_rezero":
                    x_seq = (x_seq * (1 - self.weight_residual)) + (residual * self.weight_residual)
                elif self.residual_init_strategy == "correct_rezero":
                    x_seq = (x_seq * self.weight_residual) + (residual * (1 - self.weight_residual))
        else:
            raise ValueError(f"Residual dimension mismatch: H={H} != F2={self.F2}")

        # -------------------------
        # MULTI-QUERY INTRA-BIN POOLING
        # -------------------------
        
        # Transpose to [B*NB, L, H] for attention pooling
        x_seq_t = x_seq  # Already in [B*NB, L, H] format
        
        # Multi-query attention pool -> per-bin summary
        z_per_bin = self.intra_bin_pool(x_seq_t)             # [B*NB, M*H]
        z_per_bin = z_per_bin.view(B, Bins, -1)             # [B, NB, M*H]
        
        # -------------------------
        # CROSS-BIN CONTEXT
        # -------------------------
        
        if self.cross_bin_context is not None:
            z_per_bin = self.cross_bin_context(z_per_bin)   # [B, NB, M*H]

        # -------------------------
        # FUSE ACROSS BINS
        # -------------------------
        
        # Simple mean pooling (can be enhanced later)
        z_recurrent = z_per_bin.mean(dim=1)                  # [B, M*H]
        evidence_vectors.append(z_recurrent)

        # -------------------------
        # FUSE ALL EVIDENCE PATHS
        # -------------------------
        
        # Concatenate all evidence vectors
        e_total = torch.cat(evidence_vectors, dim=1)         # [B, total_dim]

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        
        e_total = self.head_norm(e_total)
        e_total = self.head_drop(e_total)
        logits = self.fc(e_total)                            # [B, n_outputs]
        
        return logits


def create_hydra_v2_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """
    Create the HYDRAv2 classifier.
    
    Args:
        n_chans: Number of EEG channels
        n_times: Number of time samples
        n_outputs: Number of output classes
        wiring: ArbitraryWiring instance or None. If None and wiring_arch_index is in kwargs,
                will load the wiring from the specified architecture file.
        **kwargs: Additional parameters including wiring_arch_index for hyperparameter optimization
    """
    from pathlib import Path
    from architecture_refinement.arbitrary_wiring import load_architecture_from_file
    
    # Handle wiring selection from hyperparameter optimization
    # Check both with and without module__ prefix (for compatibility)
    wiring_arch_index = None
    if 'wiring_arch_index' in kwargs:
        wiring_arch_index = kwargs.pop('wiring_arch_index')
    elif 'module__wiring_arch_index' in kwargs:
        wiring_arch_index = kwargs.pop('module__wiring_arch_index')
    
    # Default to architecture 4 if neither wiring nor wiring_arch_index is provided
    if wiring is None and wiring_arch_index is None:
        wiring_arch_index = 4  # Default to architecture 4
    
    if wiring is None and wiring_arch_index is not None:
        # Find the architecture file
        architectures_dir = Path("outputs/architectures")
        architecture_files = sorted(architectures_dir.glob("best_architecture_*.json"))
        
        if wiring_arch_index < 1 or wiring_arch_index > len(architecture_files):
            raise ValueError(
                f"Invalid wiring_arch_index {wiring_arch_index}. "
                f"Must be between 1 and {len(architecture_files)}"
            )
        
        # Load the wiring architecture (index is 1-based)
        architecture_file = architecture_files[wiring_arch_index - 1]
        wiring = load_architecture_from_file(str(architecture_file))
        
        # Note: The wiring's input_size and output_size will be reconfigured
        # in _create_recurrent_cell() based on F2 (which depends on F1 and D).
        # The wiring matrix structure is preserved; only dimensions are adjusted.
    
    # If wiring is still None at this point, raise an error
    if wiring is None:
        raise ValueError(
            "wiring must be provided either directly or via wiring_arch_index in kwargs"
        )
    
    seed = get_seed()
    gradient_clip_value = 1.0
    
    # Default parameters (inherit from BranchedWiredCfC defaults)
    default_params = {
        'drop_prob': 0.5,
        'F1': 8,
        'D': 2,
        'kernel_length': 125,
        'temporal_kernel_size': 3,
        'temporal_stride': 2,
        'fusion': 'mean',  # Fusion type: 'attn' or 'mean'
        'mixed_memory': False,
        'mode': 'default',
        'activation': 'lecun_tanh',
        'backbone_units': 128,
        'backbone_layers': 1,
        'backbone_dropout': 0.0,
        # HYDRAv2-specific defaults
        'num_attn_queries': 4,
        'attn_dropout': 0.1,
        'use_adaptive_residual': True,
        'use_erp_head': True,
        'use_ssvep_head': True,
        'erp_num_queries': 4,
        'ssvep_num_filters': 4,
        'use_cross_bin_context': False,
        'context_type': 'transformer',
        'context_num_heads': 2,
        'context_num_layers': 1,
        'use_global_skip': True,
    }
    
    # Update with any provided parameters
    default_params.update(kwargs)
    
    classifier = EEGClassifier(
        HYDRAv2,
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
        module__fusion=default_params['fusion'],
        module__mixed_memory=default_params['mixed_memory'],
        module__mode=default_params['mode'],
        module__activation=default_params['activation'],
        module__backbone_units=default_params['backbone_units'],
        module__backbone_layers=default_params['backbone_layers'],
        module__backbone_dropout=default_params['backbone_dropout'],
        # HYDRAv2-specific parameters
        module__num_attn_queries=default_params['num_attn_queries'],
        module__attn_dropout=default_params['attn_dropout'],
        module__use_adaptive_residual=default_params['use_adaptive_residual'],
        module__use_erp_head=default_params['use_erp_head'],
        module__use_ssvep_head=default_params['use_ssvep_head'],
        module__erp_num_queries=default_params['erp_num_queries'],
        module__ssvep_num_filters=default_params['ssvep_num_filters'],
        module__use_cross_bin_context=default_params['use_cross_bin_context'],
        module__context_type=default_params['context_type'],
        module__context_num_heads=default_params['context_num_heads'],
        module__context_num_layers=default_params['context_num_layers'],
        module__use_global_skip=default_params['use_global_skip'],
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


# ============================================================================
# HYDRA V3: CfC-Controlled Adaptive Carry Gate
# ============================================================================

class CfCCarryController(nn.Module):
    """
    CfC-based carry controller for adaptive gating (per-bin, not per-timestep).
    
    Replaces explicit adaptive gates with a recurrent, continuous-time controller.
    The controller processes the entire bin sequence through CfC and outputs a single
    carry coefficient α^(b) ∈ (0,1) per bin, which is applied uniformly across all timesteps.
    
    This is much faster than per-timestep coefficients and similar to the old weighted
    residual implementation, but uses CfC to learn the per-bin mixing relationship.
    
    According to HYDRA_V3_Spec (modified for efficiency):
    - Controller dimension: d_c ∈ {1, 2} (default: 1)
    - Receives same bin-level inputs as main recurrent unit
    - Evolves via CfC dynamics over the entire sequence
    - Outputs single carry coefficient per bin via sigmoid: α^(b) = σ(W_c * pool(c_seq) + b_c)
    """
    def __init__(
        self,
        input_size: int,
        controller_dim: int = 1,
        mixed_memory: bool = True,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 8,  # Small backbone for lightweight controller
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        pooling: str = "mean",  # "mean" or "attention" for temporal pooling
    ):
        """
        Args:
            input_size: Input feature dimension (F2)
            controller_dim: Controller state dimension (d_c), typically 1 or 2
            mixed_memory: Whether to use mixed memory in CfC
            mode: CfC mode
            activation: Activation function for CfC
            backbone_units: Number of backbone units (small for lightweight controller)
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout rate
            pooling: Temporal pooling method ("mean" or "attention")
        """
        super().__init__()
        self.input_size = input_size
        self.controller_dim = controller_dim
        self.pooling = pooling
        
        # Create a simple CfC controller cell with integer units (not wired)
        # This is simpler and more lightweight than using wiring for the small controller
        self.controller_cell = CfC(
            input_size=input_size,
            units=controller_dim,  # Use integer units for simplicity
            proj_size=controller_dim,
            return_sequences=True,  # Return full sequence for pooling
            batch_first=True,
            mixed_memory=mixed_memory,
            mode=mode,
            activation=activation,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        
        # Temporal pooling to get per-bin representation
        if pooling == "attention":
            # Simple attention pooling
            from models.branched_diva_base import TemporalAttnPool
            self.temporal_pool = TemporalAttnPool(dim=controller_dim)
        else:
            # Mean pooling (default, faster)
            self.temporal_pool = None
        
        # Linear projection to carry coefficient: pool(c_seq) -> α^(b)
        # Initialize bias to ~0 for neutral carry at start
        self.carry_proj = nn.Linear(controller_dim, 1, bias=True)
        nn.init.constant_(self.carry_proj.bias, 0.0)  # Neutral initialization
        
    def forward(self, x_bins):
        """
        Forward pass of the carry controller.
        
        Args:
            x_bins: [B*NB, L, F2] - input sequence (same as main recurrent cell)
            
        Returns:
            alpha_per_bin: [B*NB, 1] - carry coefficient α^(b) per bin (applied to all timesteps)
        """
        # Process through CfC controller: [B*NB, L, F2] -> [B*NB, L, d_c]
        c_seq, _ = self.controller_cell(x_bins)  # [B*NB, L, controller_dim]
        
        # Pool over time to get per-bin representation: [B*NB, L, d_c] -> [B*NB, d_c]
        if self.pooling == "attention":
            c_pooled = self.temporal_pool(c_seq)  # [B*NB, controller_dim]
        else:
            # Mean pooling (default)
            c_pooled = c_seq.mean(dim=1)  # [B*NB, controller_dim]
        
        # Project to carry coefficient: [B*NB, controller_dim] -> [B*NB, 1]
        alpha_logits = self.carry_proj(c_pooled)  # [B*NB, 1]
        
        # Apply sigmoid to get α^(b) ∈ (0, 1)
        alpha_per_bin = torch.sigmoid(alpha_logits)  # [B*NB, 1]
        
        return alpha_per_bin


class HYDRAv3(BranchedWiredCfC):
    """
    HYDRAv3: CfC-Controlled Adaptive Carry Gate
    
    Key improvements over HYDRAv2:
    1. Replaces explicit adaptive residual gate with CfC-based carry controller
    2. Adaptive routing is now internal to recurrent dynamics (not external residual)
    3. Simplified architecture (removed ERP Head, Multi-Query, Cross-Bin Context per analysis)
    4. Keeps SSVEP Head for robustness (recommended in analysis)
    
    Architecture:
    Input → DIVA front-end → multi-scale temporal + SNR gate → downsampled features
    
    Parallel evidence paths:
    - SSVEP head (frequency-locked)
    - Branched recurrent path with CfC carry controller (local dynamics with adaptive gating)
    
    The carry controller:
    - Uses a small CfC cell (d_c=1 or 2) per bin
    - Receives same input as main recurrent cell
    - Processes entire bin sequence and outputs single carry coefficient α^(b) per bin
    - Mixing rule: h^(b) = α^(b) * residual^(b) + (1-α^(b)) * h̃^(b) (applied uniformly)
    - Much faster than per-timestep mixing, similar to weighted residual but learned via CfC
    
    Classifier head: LayerNorm → Dropout → Linear
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        wiring: ArbitraryWiring,
        # --- CfC Carry Controller parameters ---
        use_cfc_carry_controller: bool = True,
        controller_dim: int = 1,  # d_c ∈ {1, 2}
        # --- SSVEP evidence head (kept per analysis recommendations) ---
        use_ssvep_head: bool = True,
        ssvep_num_filters: int = 4,
        # --- Inherit all other params from BranchedWiredCfC ---
        **kwargs
    ):
        # Store HYDRAv3-specific parameters
        self.use_cfc_carry_controller = use_cfc_carry_controller
        self.controller_dim = controller_dim
        self.use_ssvep_head = use_ssvep_head
        
        # Extract drop_prob from kwargs for later use
        self.drop_prob = kwargs.get('drop_prob', 0.25)
        
        # Call parent constructor
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring,
            **kwargs
        )
        
        # -------------------------
        # CFC CARRY CONTROLLER
        # -------------------------
        
        if use_cfc_carry_controller:
            self.carry_controller = CfCCarryController(
                input_size=self.F2,
                controller_dim=controller_dim,
                mixed_memory=self.mixed_memory,
                mode=self.mode,
                activation=self.activation,
                backbone_units=8,  # Small backbone for lightweight controller
                backbone_layers=1,
                backbone_dropout=0.0,
                pooling="mean",  # Fast mean pooling (can use "attention" for more expressive pooling)
            )
        else:
            self.carry_controller = None
        
        # -------------------------
        # SSVEP EVIDENCE HEAD (kept per analysis recommendations)
        # -------------------------
        
        if use_ssvep_head:
            self.ssvep_head = SSVEPEvidenceHead(
                feature_dim=self.F2,
                num_filters=ssvep_num_filters
            )
        else:
            self.ssvep_head = None
        
        # -------------------------
        # UPDATE CLASSIFICATION HEAD
        # -------------------------
        
        # Compute total evidence dimension
        evidence_dims = []
        
        # Recurrent path: [B, H] after intra-bin pooling
        evidence_dims.append(self.recurrent_output_size)
        
        # SSVEP head: [B, K*F]
        if use_ssvep_head:
            ssvep_dim = self.F2 * ssvep_num_filters
            evidence_dims.append(ssvep_dim)
        
        total_evidence_dim = sum(evidence_dims)
        
        # Update classification head
        self.head_norm = nn.LayerNorm(total_evidence_dim)
        self.head_drop = nn.Dropout(p=self.drop_prob)
        self.fc = nn.Linear(total_evidence_dim, n_outputs)

    def _process_bins(self, x_bins, residual):
        """
        Process temporal bins through the recurrent cell with CfC carry controller.
        
        This method overrides the parent to integrate the carry controller.
        The carry controller outputs a single α^(b) per bin, which is applied uniformly:
        h^(b) = α^(b) * residual^(b) + (1-α^(b)) * h̃^(b)
        
        This is much faster than per-timestep mixing and similar to weighted residual,
        but uses CfC to learn the per-bin mixing relationship.
        
        Args:
            x_bins: [B*NB, L, F2] - reshaped bins for parallel processing
            residual: [B*NB, L, F2] - residual connection data (used for mixing)
            
        Returns:
            x_seq: [B*NB, L, H] - processed sequences with adaptive carry gating
        """
        # Get main recurrent output: h̃^(b) for all timesteps
        h_tilde, _ = self.recurrent_cell(x_bins)  # [B*NB, L, H]
        
        # If no carry controller, return main recurrent output
        if self.carry_controller is None or not self.use_cfc_carry_controller:
            return h_tilde
        
        # Get carry coefficient per bin: α^(b) [B*NB, 1]
        alpha_per_bin = self.carry_controller(x_bins)  # [B*NB, 1]
        
        # Ensure residual dimension matches h_tilde for mixing
        # If H != F2, we need to project residual or skip mixing
        B_NB, L, H = h_tilde.shape
        if H == self.F2:
            # Residual has same dimension as recurrent output - can mix directly
            # Apply per-bin mixing uniformly across all timesteps
            # Formula: h^(b) = α^(b) * residual^(b) + (1-α^(b)) * h̃^(b)
            # This is backwards_rezero style: higher α means more residual
            # Expand alpha_per_bin [B*NB, 1] to [B*NB, 1, 1] for broadcasting over [L, H]
            # PyTorch will automatically broadcast [B*NB, 1, 1] with [B*NB, L, H]
            alpha_expanded = alpha_per_bin.view(B_NB, 1, 1)  # [B*NB, 1, 1]
            h_seq = alpha_expanded * residual + (1 - alpha_expanded) * h_tilde  # [B*NB, L, H]
        else:
            # Dimension mismatch - fall back to just recurrent output
            # (This shouldn't happen with default settings)
            h_seq = h_tilde
        
        return h_seq

    def forward(self, x):
        """
        Forward pass with HYDRAv3 improvements.
        """
        B, C, T = x.shape

        # -------------------------
        # DIVA FRONT-END (inherited from parent)
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

        # Store pre-binned features for SSVEP head
        pre_binned_features = x  # [B, F2, T2]

        # -------------------------
        # PARALLEL EVIDENCE PATHS
        # -------------------------
        
        evidence_vectors = []
        
        # SSVEP evidence head (non-recurrent, frequency-locked)
        if self.ssvep_head is not None:
            e_SSVEP = self.ssvep_head(pre_binned_features)  # [B, K*F]
            evidence_vectors.append(e_SSVEP)

        # -------------------------
        # BRANCHED RECURRENT PATH (with CfC carry controller)
        # -------------------------
        
        # Chunk into bins
        x_bins = self._chunk_time(x)                       # [B, NB, L, F2]
        Bins = x_bins.size(1)
        L = x_bins.size(2)
        F2_ = x_bins.size(3)
        assert F2_ == self.F2

        # Merge batch and bins to run recurrent processing in parallel
        x_bins = x_bins.reshape(B * Bins, L, self.F2)     # [B*NB, L, F2]

        # Store residual for reference (not used in HYDRAv3 mixing, but kept for compatibility)
        residual = x_bins                                  # [B*NB, L, F2]

        # Process bins through recurrent cell with carry controller
        # The carry controller is integrated in _process_bins
        x_seq = self._process_bins(x_bins, residual)       # [B*NB, L, H]
        H = x_seq.size(-1)

        # Note: In HYDRAv3, the adaptive mixing is done inside _process_bins
        # No explicit residual connection here - it's all handled by the carry controller

        # -------------------------
        # INTRA-BIN POOLING (standard attention pooling, not multi-query)
        # -------------------------
        
        # Use standard intra-bin pooling from parent class
        z_per_bin = self.intra_bin_pool(x_seq)             # [B*NB, H]
        z_per_bin = z_per_bin.view(B, Bins, H)             # [B, NB, H]

        # -------------------------
        # FUSE ACROSS BINS
        # -------------------------
        
        if self.fusion == "attn":
            z_recurrent = self.bin_fusion(z_per_bin)       # [B, H]  (attn over NB)
        else:
            z_recurrent = z_per_bin.mean(dim=1)            # [B, H]  (mean over NB)
        
        evidence_vectors.append(z_recurrent)

        # -------------------------
        # FUSE ALL EVIDENCE PATHS
        # -------------------------
        
        # Concatenate all evidence vectors
        e_total = torch.cat(evidence_vectors, dim=1)         # [B, total_dim]

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        
        e_total = self.head_norm(e_total)
        e_total = self.head_drop(e_total)
        logits = self.fc(e_total)                            # [B, n_outputs]
        
        return logits


def create_hydra_v3_classifier(n_chans, n_times, n_outputs, wiring=None, **kwargs):
    """
    Create the HYDRAv3 classifier.
    
    Args:
        n_chans: Number of EEG channels
        n_times: Number of time samples
        n_outputs: Number of output classes
        wiring: ArbitraryWiring instance or None. If None, will default to architecture 4
                (best performing architecture per analysis). Can also specify wiring_arch_index
                in kwargs to use a different architecture.
        **kwargs: Additional parameters including wiring_arch_index for hyperparameter optimization.
                  Default wiring_arch_index is 4 (architecture 4).
    """
    from pathlib import Path
    from architecture_refinement.arbitrary_wiring import load_architecture_from_file
    
    # Handle wiring selection from hyperparameter optimization
    # Check both with and without module__ prefix (for compatibility)
    wiring_arch_index = None
    if 'wiring_arch_index' in kwargs:
        wiring_arch_index = kwargs.pop('wiring_arch_index')
    elif 'module__wiring_arch_index' in kwargs:
        wiring_arch_index = kwargs.pop('module__wiring_arch_index')
    
    # Default to architecture 4 if neither wiring nor wiring_arch_index is provided
    # Architecture 4 was identified as the best performing configuration in HYDRAv2 analysis
    if wiring is None and wiring_arch_index is None:
        wiring_arch_index = 4  # Default to architecture 4 (best performing per analysis)
    
    if wiring is None and wiring_arch_index is not None:
        # Find the architecture file
        architectures_dir = Path("outputs/architectures")
        architecture_files = sorted(architectures_dir.glob("best_architecture_*.json"))
        
        if wiring_arch_index < 1 or wiring_arch_index > len(architecture_files):
            raise ValueError(
                f"Invalid wiring_arch_index {wiring_arch_index}. "
                f"Must be between 1 and {len(architecture_files)}"
            )
        
        # Load the wiring architecture (index is 1-based)
        architecture_file = architecture_files[wiring_arch_index - 1]
        wiring = load_architecture_from_file(str(architecture_file))
        
        # Note: The wiring's input_size and output_size will be reconfigured
        # in _create_recurrent_cell() based on F2 (which depends on F1 and D).
        # The wiring matrix structure is preserved; only dimensions are adjusted.
    
    # If wiring is still None at this point, raise an error
    if wiring is None:
        raise ValueError(
            "wiring must be provided either directly or via wiring_arch_index in kwargs"
        )
    
    seed = get_seed()
    gradient_clip_value = 1.0
    
    # Default parameters (inherit from BranchedWiredCfC defaults)
    # Based on analysis: use Arch 4 configuration, simplified
    default_params = {
        'drop_prob': 0.5,
        'F1': 8,
        'D': 2,
        'kernel_length': 125,
        'temporal_kernel_size': 3,
        'temporal_stride': 2,
        'fusion': 'mean',  # Fusion type: 'attn' or 'mean'
        'mixed_memory': False,
        'mode': 'default',
        'activation': 'lecun_tanh',
        'backbone_units': 128,
        'backbone_layers': 1,
        'backbone_dropout': 0.0,
        # HYDRAv3-specific defaults
        'use_cfc_carry_controller': True,
        'controller_dim': 1,  # d_c = 1 (per spec recommendation)
        'use_ssvep_head': True,  # Kept per analysis recommendations
        'ssvep_num_filters': 4,
    }
    
    # Update with any provided parameters
    default_params.update(kwargs)
    
    classifier = EEGClassifier(
        HYDRAv3,
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
        module__fusion=default_params['fusion'],
        module__mixed_memory=default_params['mixed_memory'],
        module__mode=default_params['mode'],
        module__activation=default_params['activation'],
        module__backbone_units=default_params['backbone_units'],
        module__backbone_layers=default_params['backbone_layers'],
        module__backbone_dropout=default_params['backbone_dropout'],
        # HYDRAv3-specific parameters
        module__use_cfc_carry_controller=default_params['use_cfc_carry_controller'],
        module__controller_dim=default_params['controller_dim'],
        module__use_ssvep_head=default_params['use_ssvep_head'],
        module__ssvep_num_filters=default_params['ssvep_num_filters'],
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