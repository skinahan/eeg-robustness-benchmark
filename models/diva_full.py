# diva_inspired_eeg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from braindecode.models.base import EEGModuleMixin
from braindecode import EEGClassifier
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, GradientNormClipping
from torch.optim.lr_scheduler import ExponentialLR

# You already have these in your repo:
from models.cnnncp import _MultiScaleTemporalBlock1D, _SNRGate
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE

# ---------- Small utilities ----------

class VarianceHead1D(nn.Module):
    """
    Tiny head to estimate per-time, per-channel log-variance for uncertainty-aware mixing.
    In:  (B, C, T)
    Out: (B, C, T)   (log-variance)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # No explicit positivity; we return log-variance directly.
        return self.net(x)


class DelayLine1D(nn.Module):
    """
    Learnable delay compensation block for feedback path.
    Realized as a causal 1D conv with dilation (Smith-predictor flavor).
    In/Out: (B, C, T)
    """
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            dilation=dilation, padding=pad, groups=channels, bias=False
        )

    def forward(self, x):
        # Causal: strip future padding to maintain input length
        delay = (self.conv.padding[0])
        y = self.conv(x)
        if delay > 0:
            y = y[:, :, :-delay]
        return y


class SeparableConvTimeHead(nn.Module):
    """
    Light smoothing head after recurrent/feedback fusion.
    In:  (B, C, T)
    Out: (B, C, T)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ---------- DIVA-inspired, ablation-friendly model ----------

class DIVAInspiredEEG(EEGModuleMixin, nn.Module):
    """
    A DIVA-inspired EEG classifier with clearly delineated compartments and ablation switches.

    Flow overview (B,C,T):
    [A] Sensory Front-End (temporal conv -> spatial depthwise -> anti-alias pool -> optional MS & SNR gate -> temporal downsample)
        -> (B, F2, T2)

    [B] Recurrent Forward Model (CfC/NCP): produces recurrent features z_rec and a residual z_res (pre-CfC)
        -> z_rec, z_res are (B, H, T2) with H=F2 by default

    [C] Predicted Sensory Consequence Head (optional)
        Predict next latent: z_hat[t+1] from z_rec[t]; compute error e[t+1] = stopgrad(z_res[t+1]) - z_hat[t+1]

    [D] Feedback Controller (optional)
        Process error e through delay line (optional) and a small controller, yielding correction c
        Fuse: z_corr = z_rec + c

    [E] Uncertainty-Aware Mixer (optional)
        Dynamic precision-weighted mixing between z_corr and z_res

    [F] Classifier Head
        Light smoothing -> global pool -> FC

    Ablation knobs:
        use_ms_block, use_snr_gate, use_forward_model, use_feedback_controller,
        use_delay, use_uncertainty_mixer

    Notes:
        - Auxiliary losses (prediction) are exposed via self.last_aux_losses (dict). Your training loop can combine them.
        - Shapes are asserted where it helps catch mistakes early.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,

        # --- Front-End (EEGNet-ish) ---
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        pool_time: int = 4,
        drop_prob: float = 0.25,

        # Multi-scale temporal integration & SNR gate
        use_ms_block: bool = True,
        ms_kernels=(9, 15, 31),
        ms_dilations=(1, 4, 16),
        use_snr_gate: bool = True,
        snr_reduction: int = 4,

        # Temporal downsampling before CfC
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,

        # --- Recurrent Core (NCP/CfC) ---
        ncp_hidden_dim: int = 32,
        sparsity: float = 0.7,
        ncp_output_size: int = None,  # default = F1*D
        return_sequences: bool = True,

        # --- Predicted Sensory Consequence & Feedback Controller ---
        use_forward_model: bool = False,      # [C]
        use_feedback_controller: bool = False,# [D]
        feedback_hidden: int = 64,
        use_delay: bool = True,              # delay line on error path
        delay_kernel: int = 5,
        delay_dilation: int = 2,

        # --- Uncertainty-aware mixing ---
        use_uncertainty_mixer: bool = True,  # [E]
        init_residual_weight: float = 0.5,   # used only if mixer is disabled

        # --- Norm/Eps ---
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
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

        # Save switches
        self.use_ms_block = use_ms_block
        self.use_snr_gate = use_snr_gate
        self.use_forward_model = use_forward_model
        self.use_feedback_controller = use_feedback_controller
        self.use_delay = use_delay
        self.use_uncertainty_mixer = use_uncertainty_mixer

        # Derived dims
        self.F1 = F1
        self.F2 = F1 * D
        self.kernel_length = kernel_length
        self.pool_time = pool_time
        self.temporal_kernel_size = temporal_kernel_size or 3
        self.temporal_stride = temporal_stride or 2

        act = nn.ELU()

        # ---------- [A] Sensory Front-End ----------
        # (1) Temporal conv across time (B,1,C,T)->(B,F1,C,T)
        self.conv_time = nn.Conv2d(1, self.F1, kernel_size=(1, kernel_length),
                                   padding=(0, kernel_length//2), bias=False)
        self.bn1 = nn.BatchNorm2d(self.F1, momentum=bn_momentum, eps=bn_eps)
        self.act = act

        # (2) Depthwise spatial conv (B,F1,C,T)->(B,F2,1,T)
        self.depth_spatial = nn.Conv2d(self.F1, self.F2, kernel_size=(n_chans, 1),
                                       groups=self.F1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2, momentum=bn_momentum, eps=bn_eps)

        # (3) Anti-alias downsample over time
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time), stride=(1, pool_time))
        self.drop1 = nn.Dropout(drop_prob)

        # Switch to 1D time processing from here: (B,F2,T1)
        if self.use_ms_block:
            self.ms_block = _MultiScaleTemporalBlock1D(self.F2, ms_kernels, ms_dilations)
        if self.use_snr_gate:
            self.snr_gate = _SNRGate(self.F2, reduction=snr_reduction)

        # (4) Temporal downsampler (Conv1D): (B,F2,T1)->(B,F2,T2)
        self.time_down = nn.Conv1d(self.F2, self.F2, kernel_size=self.temporal_kernel_size,
                                   stride=self.temporal_stride, padding=self.temporal_kernel_size//2, bias=False)

        # ---------- [B] Recurrent Forward Model (CfC/NCP) ----------
        if ncp_output_size is None:
            ncp_output_size = self.F2
        
        # wiring = AutoNCP(ncp_hidden_dim, ncp_output_size, sparsity_level=sparsity, seed=seed)
        # self.cfc = CfC(input_size=self.F2, units=wiring, return_sequences=return_sequences)
        
        # TODO: This is a test to see if CfC (fully connected) works better than the AutoNCP (sparse)
        self.cfc = CfC(
            input_size=self.F2, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size, 
            return_sequences=return_sequences, 
            mixed_memory=True, 
            activation="silu",
            # mode='pure'
        )
        
        assert return_sequences, "We assume return_sequences=True to keep time dimension for mixers/controllers."

        # Project CfC output to a consistent channel count H = ncp_output_size
        self.proj_rec = nn.Identity()  # (B,T2,H) already; we transpose later to (B,H,T2)

        # ---------- [C] Predicted Sensory Consequence Head (optional) ----------
        # Predict z_res[t+1] from z_rec[t] in latent space
        if self.use_forward_model:
            self.pred_head = nn.Sequential(
                nn.Conv1d(ncp_output_size, ncp_output_size, 3, padding=1, groups=ncp_output_size, bias=False),
                nn.BatchNorm1d(ncp_output_size),
                nn.SiLU(),
                nn.Conv1d(ncp_output_size, ncp_output_size, 1, bias=True),
            )

        # ---------- [D] Feedback Controller (optional) ----------
        if self.use_feedback_controller:
            self.delay = DelayLine1D(ncp_output_size, kernel_size=delay_kernel, dilation=delay_dilation) \
                         if self.use_delay else nn.Identity()
            self.controller = nn.Sequential(
                nn.Conv1d(ncp_output_size, feedback_hidden, 1, bias=False),
                nn.SiLU(),
                nn.Conv1d(feedback_hidden, ncp_output_size, 1, bias=False),
            )

        # ---------- [E] Uncertainty-aware Mixer (optional) ----------
        if self.use_uncertainty_mixer:
            self.var_head_res = VarianceHead1D(ncp_output_size)
            self.var_head_rec = VarianceHead1D(ncp_output_size)
        else:
            # Fallback: learned static scalar in [0,1] that weights residual path
            self.weight_residual = nn.Parameter(torch.tensor(float(init_residual_weight)).clamp(0.0,1.0))

        # ---------- [F] Classifier Head ----------
        self.smooth_head = SeparableConvTimeHead(ncp_output_size)
        self.drop2 = nn.Dropout(drop_prob)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ncp_output_size, n_outputs)

        # Aux loss stash
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self.last_aux_losses = {}

        self._init_weights()

    # ---- Initialization ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ---- Forward ----
    def forward(self, x):
        """
        x: (B, C, T)
        returns: logits (B, n_outputs)
        Side effect: self.last_aux_losses = {"pred_L1": ..., "pred_huber": ...} if forward-model is enabled
        """
        B, C, T = x.shape

        # [A] Sensory Front-End
        z = x.unsqueeze(1)                               # (B,1,C,T)
        z = self.conv_time(z)                            # (B,F1,C,T)
        z = self.bn1(z); z = self.act(z)
        z = self.depth_spatial(z)                        # (B,F2,1,T)
        z = self.bn2(z); z = self.act(z)
        z = self.pool(z)                                 # (B,F2,1,T1)
        z = self.drop1(z)
        z = z.squeeze(2)                                 # (B,F2,T1)

        if self.use_ms_block:
            z = self.ms_block(z)                         # (B,F2,T1)
        if self.use_snr_gate:
            z = self.snr_gate(z)                         # (B,F2,T1)

        z = self.time_down(z)                            # (B,F2,T2)
        T2 = z.shape[-1]

        # Prepare residual for mixing/comparison; CfC expects features last (T,B,F) or (B,T,F)
        z_res = z                                         # (B,F2,T2) residual path latent

        # [B] Recurrent Forward Model (CfC/NCP)
        z_in = z.transpose(1, 2).contiguous()             # (B,T2,F2)
        z_rec, _ = self.cfc(z_in)                         # (B,T2,H)
        z_rec = self.proj_rec(z_rec)                      # identity; ensure (B,T2,H)
        z_rec = z_rec.transpose(1, 2).contiguous()        # (B,H,T2) == (B,F2,T2)

        # [C] Predicted Sensory Consequence (optional)
        self.last_aux_losses = {}
        if self.use_forward_model:
            # Predict next-step residual latent from current recurrent latent
            z_hat = self.pred_head(z_rec)                 # (B,H,T2)
            # Align z_res[t+1] target by shifting left one step
            target = torch.roll(z_res, shifts=-1, dims=-1)
            # Mask last time step (no target) with zero weight in loss
            mask = torch.ones_like(target[..., :1]).repeat(1, 1, target.size(-1))
            mask[..., -1] = 0.0

            # Losses (reported; you can add to total loss in your training loop if desired)
            l1 = F.l1_loss(z_hat * mask, target * mask, reduction="mean")
            hub = F.huber_loss(z_hat * mask, target * mask, reduction="mean", delta=0.05)
            self.last_aux_losses = {"pred_L1": l1.detach(), "pred_huber": hub.detach()}
        else:
            z_hat = None

        # [D] Feedback Controller (optional)
        if self.use_feedback_controller:
            # Error e[t+1] = target - z_hat; use stopgrad on target (residual) to avoid trivial solutions
            if z_hat is None:
                e = torch.zeros_like(z_res)
            else:
                with torch.no_grad():
                    target = torch.roll(z_res, shifts=-1, dims=-1)
                e = target - z_hat                         # (B,H,T2)
            e = self.delay(e)                              # (B,H,T2) (Identity if disabled)
            c = self.controller(e)                         # (B,H,T2)
            z_corr = z_rec + c                             # corrected recurrent
        else:
            z_corr = z_rec

        # [E] Uncertainty-aware Mixer (optional)
        if self.use_uncertainty_mixer:
            # Estimate log-variance for each path; higher precision => higher weight
            logv_res = self.var_head_res(z_res)            # (B,H,T2)
            logv_rec = self.var_head_rec(z_corr)           # (B,H,T2)
            prec_res = torch.exp(-logv_res)
            prec_rec = torch.exp(-logv_rec)
            alpha = prec_rec / (prec_res + prec_rec + 1e-6)  # (B,H,T2) in [0,1]
            z_fused = alpha * z_corr + (1.0 - alpha) * z_res
        else:
            w = torch.clamp(self.weight_residual, 0.0, 1.0)
            z_fused = (1.0 - w) * z_corr + w * z_res

        # [F] Classifier Head
        z_out = self.smooth_head(z_fused)                  # (B,H,T2)
        z_out = self.drop2(z_out)
        z_out = self.global_pool(z_out).squeeze(-1)        # (B,H)
        logits = self.fc(z_out)                            # (B,n_outputs)
        return logits


# ---------- Helper to build a classifier (Skorch/Braindecode) ----------

def create_diva_full_classifier(n_chans, n_times, n_outputs):
    """
    Create the DIVA Full (DIVAInspiredEEG) classifier.
    
    This is a DIVA-inspired EEG classifier with forward model, feedback controller,
    and uncertainty-aware mixing.
    """
    seed = get_seed()
    gradient_clip_value = 1.0
    classifier = EEGClassifier(
        DIVAInspiredEEG,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        optimizer__weight_decay=0,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=8,
        module__drop_prob=0.5,
        module__F1=8,
        module__D=2,
        module__kernel_length=125,
        module__temporal_kernel_size=3,
        module__temporal_stride=2,
        module__use_ms_block=True,
        module__use_snr_gate=True,
        module__use_forward_model=True,
        module__use_feedback_controller=True,
        module__use_delay=True,
        module__use_uncertainty_mixer=True,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2),
            LRScheduler(policy=ExponentialLR, gamma=0.99),
        ],
        verbose=EEGCLASSIFIER_VERBOSE
    )

    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()

    return classifier
