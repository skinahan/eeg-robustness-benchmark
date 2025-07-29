
from skorch.dataset import ValidSplit
from globals import set_seeds
from globals import get_seed
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from braindecode import EEGClassifier
import torch
from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import Ensure4d
from torch import nn
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP
from einops.layers.torch import Rearrange


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


# REEGNet variant with CfC-based recurrence - IMPROVED VERSION with Label Smoothing and Layer Norm
class CNNCfCImprovedV2(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=16,  # Reduced from 32 to 16 for speed
            cnn_output_dim=16,
            sparsity=0.75,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,  # More aggressive downsampling
            max_seq_length=250  # Limit sequence length for CfC
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

        # 6. Layer normalization before CfC for better stability:
        self.layer_norm = nn.LayerNorm(16)

        # 7. CfC with reduced complexity:
        ncp_input_size = 16  # After temporal downsampling, we have 16 features
        ncp_output_size = 16  # Reduced from 32 to 16
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=True,
            mode='default'
        )

        # 8. Separable Conv2D (same as REEGNet):
        self.sep_depthwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=ncp_output_size, kernel_size=(3, 1),
                                       stride=(1, 1), padding=(1, 0), groups=ncp_output_size, bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # 9. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 10. Dense layer:
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
        
        # 6. Layer normalization for better stability:
        x = self.layer_norm(x)
        
        # 7. Limit sequence length for CfC:
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 8. CfC processing:
        x, _ = self.ncp(x)  # [B, T', H]
        
        # 9. Reshape for separable conv:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, H, T', 1]

        # 10. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 11. Global pooling and classification:
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


# Simple Temporal Attention mechanism
class SimpleTemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, T, C]
        attention_weights = self.attention(x)  # [B, T, 1]
        attended_x = x * attention_weights  # [B, T, C]
        return attended_x


# REEGNet variant with CfC-based recurrence - IMPROVED VERSION with Label Smoothing, Layer Norm, and Attention
class CNNCfCImprovedV3(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=16,  # Reduced from 32 to 16 for speed
            cnn_output_dim=16,
            sparsity=0.75,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,  # More aggressive downsampling
            max_seq_length=250  # Limit sequence length for CfC
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

        # 6. Layer normalization before attention:
        self.layer_norm = nn.LayerNorm(16)

        # 7. Temporal attention mechanism:
        self.temporal_attention = SimpleTemporalAttention(16)

        # 8. CfC with reduced complexity:
        ncp_input_size = 16  # After temporal downsampling, we have 16 features
        ncp_output_size = 16  # Reduced from 32 to 16
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=True,
            mode='default'
        )

        # 9. Separable Conv2D (same as REEGNet):
        self.sep_depthwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=ncp_output_size, kernel_size=(3, 1),
                                       stride=(1, 1), padding=(1, 0), groups=ncp_output_size, bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=ncp_output_size, out_channels=16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # 10. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 11. Dense layer:
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
        
        # 6. Layer normalization for better stability:
        x = self.layer_norm(x)
        
        # 7. Apply temporal attention:
        x = self.temporal_attention(x)
        
        # 8. Limit sequence length for CfC:
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 9. CfC processing:
        x, _ = self.ncp(x)  # [B, T', H]
        
        # 10. Reshape for separable conv:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, H, T', 1]

        # 11. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 12. Global pooling and classification:
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


# REEGNet variant with CfC-based recurrence - SIMPLIFIED AND REGULARIZED VERSION
class CNNCfCImprovedV4(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=12,  # Further reduced for simplicity
            cnn_output_dim=16,
            sparsity=0.75,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            max_seq_length=200,  # Reduced sequence length
            feature_dropout=0.1,  # Additional feature dropout
            temporal_dropout=0.05,  # Temporal dropout
            weight_decay=1e-4  # L2 regularization
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
        self.feature_dropout = feature_dropout
        self.temporal_dropout = temporal_dropout

        # 1. Input Conv2D (simplified):
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=(1, 15),
            stride=(1, 1),
            padding=(0, 7),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D (simplified):
        self.depthwise_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(n_chans, 1),
                                        groups=8, stride=(1, 1), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(8)

        # 3. Single aggressive pooling for temporal reduction:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16))  # Very aggressive pooling

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Layer normalization for stability:
        self.layer_norm = nn.LayerNorm(8)

        # 6. CfC with reduced complexity:
        ncp_input_size = 8  # Reduced from 16 to 8
        ncp_output_size = 8  # Reduced from 16 to 8
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=True,
            mode='default'
        )

        # 7. Simplified classifier (single 1x1 conv instead of separable):
        self.classifier_conv = nn.Conv2d(in_channels=ncp_output_size, out_channels=8, 
                                        kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(8)

        # 8. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Dense layer with L2 regularization:
        self.fc = nn.Linear(8, n_outputs)
        
        # 10. Initialize weights with proper regularization
        # Note: Weight initialization is handled by PyTorch's default initialization

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
        x = x.permute(0, 3, 2, 1).squeeze(2)  # [B, T, C]
        
        # 5. Layer normalization for stability:
        x = self.layer_norm(x)
        
        # 6. Temporal dropout for regularization:
        if self.training and self.temporal_dropout > 0:
            # Randomly mask temporal steps
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.temporal_dropout))
            x = x * mask
        
        # 7. Limit sequence length for CfC:
        if x.shape[1] > self.max_seq_length:
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 8. CfC processing:
        x, _ = self.ncp(x)
        
        # 9. Feature dropout for regularization:
        if self.training and self.feature_dropout > 0:
            x = torch.dropout(x, p=self.feature_dropout, train=True)
        
        # 10. Reshape for classifier:
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, H, T', 1]

        # 11. Simplified classifier:
        x = self.classifier_conv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 12. Global pooling and classification:
        x = self.global_pool(x)  # → (B, 8, 1, 1)
        x = x.view(x.shape[0], -1)  # → (B, 8)
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


# REEGNet variant with CfC-based recurrence - ORIGINAL VERSION (for backward compatibility)
class CNNCfC(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=8,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=None,
            temporal_stride=None
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
        conv_spatial_max_norm = float(1.0)
        self.kernel_length = kernel_length

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

        # Determine temporal Conv1d params
        if temporal_kernel_size is None:
            temporal_kernel_size = 3
        if temporal_stride is None:
            temporal_stride = 2  # Downsample by ~1/2

        ncp_input_size = 4
        # self.temporal_downsampler = nn.Conv1d(
        #     in_channels=4,
        #     out_channels=4,
        #     kernel_size=temporal_kernel_size,
        #     stride=temporal_stride,
        #     padding=temporal_kernel_size // 2
        # )
        ncp_output_size = 32
        # wiring = AutoNCP(
        #     ncp_hidden_dim, ncp_output_size, sparsity_level=sparsity, seed=seed)
        # self.ncp = CfC(ncp_input_size, wiring, return_sequences=True)
        self.ncp = CfC(input_size=ncp_input_size, units=ncp_hidden_dim, proj_size=ncp_output_size,
                       return_sequences=True, batch_first=True, mixed_memory=True,
                       mode='default')
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
        # x = self.feature_extractor(x)  # [B, C, T, 1]
        # x = x.squeeze(2).permute(0, 2, 1)  # [B, T, C]
        # x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x.shape: 64, 55, 32
        # 64, 62, 16
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
        # x.Shape after feature extraction:
        # [B, 16, 1, 250]
        # 4. Permutation and Reshaping for LSTM:
        x = x.permute(0, 3, 2, 1)
        x = x.contiguous().view(x.shape[0], max(1000, self.n_times - 1), 4)
        # should x have shape: B, 4, seq_len (temporal downsample)?
        # or B, seq_len, 4 (no downsample)

        # Downsample
        # x has shape: B, seq_len, 4
        # x_permuted = x.permute(0, 2, 1)
        # .permute(0, 2 1) == B, 4, seq_len
        # What does the temporal downsampler expect?
        # Batch, channels_in, t.
        # x = self.temporal_downsampler(x)
        # x = x.permute(0, 2, 1)

        x, _ = self.ncp(x)  # [B, T', H]
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

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


# REEGNet variant with CfC-based recurrence - IMPROVED VERSION
class CNNCfCImproved(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=16,  # Reduced from 32 to 16 for speed
            drop_prob=0.15,
            F1=8,
            D=2,
            temporal_kernel_size=3,
            temporal_stride=4,  # More aggressive downsampling
            max_seq_length=250  # Limit sequence length for CfC
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

        # 6. CfC with reduced complexity:
        ncp_input_size = 16  # After temporal downsampling, we have 16 features
        ncp_output_size = 16  # Reduced from 32 to 16
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=True,
            mode='default'
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
        
        # 7. CfC processing:
        x, _ = self.ncp(x)  # [B, T', H]
        
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


# REEGNet variant with CfC-based recurrence - V3 with residual connection
# This model gets a high non-contaminated score. 
# However, it is vulnerable to noise perturbations.
class CNNCfCv3(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=16,  # Reduced from 32 to 16 for speed
            cnn_output_dim=16,
            drop_prob=0.15,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,  # More aggressive downsampling
            max_seq_length=250  # Limit sequence length for CfC
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
            mixed_memory=True,
            mode='default'
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


class CNNNCPv2(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=32,
            cnn_output_dim=16,  # Matches EEGNet feature extractor's output
            sparsity=0.75,
            drop_prob=float(0.15),
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        # Ensure input dimensions [batch_size, 1, n_chans, n_times]
        self.ensure4d = Ensure4d()
        kernel_length = 128
        depthwise_kernel_length = 16
        pool1_kernel_size = 8
        pool1_stride_size = 4
        pool2_kernel_size = 16
        pool2_stride_size = 8
        conv_spatial_max_norm = float(1.0)
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        F1 = 8
        D = 2
        F2 = F1 * D

        # Feature Extractor (CNN Head)
        self.feature_extractor = nn.Sequential(
            Ensure4d(),
            Rearrange("batch ch t 1 -> batch 1 ch t"),
            # 1. Input Conv2D:
            nn.Conv2d(1, F1, (1, kernel_length), bias=False, padding=(0, kernel_length // 2)),
            # bnorm_1
            nn.BatchNorm2d(F1, momentum=batch_norm_momentum, affine=True, eps=batch_norm_eps),

            # 2. Depthwise Conv2D:
            Conv2dWithConstraint(F1, F1 * D, (n_chans, 1), max_norm=conv_spatial_max_norm, groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),

            # 3. Average Pooling:
            # nn.AvgPool2d((1, pool1_kernel_size), stride=(1, pool1_stride_size)),
            nn.AvgPool2d((1, 16), stride=(1, 8)),
            # 4. Dropout:
            nn.Dropout(p=drop_prob),
        )

        # Sequential Model (NCP)
        ncp_out_size = 8

        wiring = AutoNCP(
            ncp_hidden_dim,
            ncp_out_size,
            sparsity_level=sparsity,
            seed=seed
        )  # Wiring configuration for NCP

        self.ncp = CfC(
            cnn_output_dim,
            wiring,
            return_sequences=True,
            mode="pure",
            # mixed_memory=False
        ).cuda()

        self.classifier_block = nn.Sequential(
            # conv_separable_depth
            nn.Conv2d(ncp_out_size, ncp_out_size, (1, depthwise_kernel_length), bias=False,
                      groups=ncp_out_size,
                      padding=(0, depthwise_kernel_length // 2)),
            # conv_separable_point
            nn.Conv2d(ncp_out_size, ncp_out_size, (1, 1), bias=False),
            # bnorm_2
            nn.BatchNorm2d(ncp_out_size, momentum=batch_norm_momentum, affine=True, eps=batch_norm_eps),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(ncp_out_size, n_outputs)
        self._glorot_weight_zero_bias()

    def forward(self, x):
        x = self.feature_extractor(x)  # Output: [batch_size, cnn_output_dim, reduced_n_times, 1]
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.ncp(x)
        x = x.permute(0, 2, 1).unsqueeze(3)
        x = self.classifier_block(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    def _glorot_weight_zero_bias(self):
        """Initialize parameters of all modules by initializing weights with
    glorot
    uniform/xavier initialization, and setting biases to zero. Weights from
    batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


class CNNNCP(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=32,
            cnn_output_dim=16,  # Matches EEGNet feature extractor's output
            sparsity=0.8,
            drop_prob=float(0.25),
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        # Ensure input dimensions [batch_size, 1, n_chans, n_times]
        self.ensure4d = Ensure4d()
        kernel_length = 128
        depthwise_kernel_length = 16
        pool1_kernel_size = 8
        pool1_stride_size = 4
        pool2_kernel_size = 16
        pool2_stride_size = 8
        conv_spatial_max_norm = float(1.0)
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        F1 = 8
        D = 2
        F2 = F1 * D

        # Feature Extractor (CNN Head)
        self.feature_extractor = nn.Sequential(
            Ensure4d(),
            Rearrange("batch ch t 1 -> batch 1 ch t"),
            # conv 1
            nn.Conv2d(1, F1, (1, kernel_length), bias=False, padding=(0, kernel_length // 2)),
            # bnorm_1
            nn.BatchNorm2d(F1, momentum=batch_norm_momentum, affine=True, eps=batch_norm_eps),
            Conv2dWithConstraint(F1, F1 * D, (n_chans, 1), max_norm=conv_spatial_max_norm, groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),
            # pool_1
            nn.AvgPool2d((1, pool1_kernel_size), stride=(1, pool1_stride_size)),
            nn.Dropout(p=drop_prob),
            # conv_separable_depth
            nn.Conv2d(F1 * D, F1 * D, (1, depthwise_kernel_length), bias=False, groups=F1 * D,
                      padding=(0, depthwise_kernel_length // 2)),
            # conv_separable_point
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            # bnorm_2
            nn.BatchNorm2d(F2, momentum=batch_norm_momentum, affine=True, eps=batch_norm_eps),
            nn.ELU(),
            # pool_2
            nn.AvgPool2d((1, pool2_kernel_size), stride=(1, pool2_stride_size)),
            nn.Dropout(p=drop_prob),
        )

        # Sequential Model (NCP)
        ncp_out_size = n_outputs
        wiring = AutoNCP(ncp_hidden_dim, ncp_out_size, sparsity_level=sparsity,
                         seed=SEED)  # Wiring configuration for NCP
        self.ncp = CfC(cnn_output_dim, wiring, return_sequences=False)

        self._glorot_weight_zero_bias()

    def forward(self, x):
        x = self.feature_extractor(x)  # Output: [batch_size, cnn_output_dim, reduced_n_times, 1]
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.ncp(x)  # Output: [batch_size, reduced_n_times, ncp_hidden_dim]
        return x

    def _glorot_weight_zero_bias(self):
        """Initialize parameters of all modules by initializing weights with
    glorot
    uniform/xavier initialization, and setting biases to zero. Weights from
    batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def create_cnnncfc_classifier(n_chans, n_times, n_outputs):
    return create_cnnncp_classifier(n_chans, n_times, n_outputs, net_size=64, classifier_type=4)


def create_cnnncfc_improved_v2_classifier(n_chans, n_times, n_outputs):
    """Create the improved CNNCfC classifier with label smoothing and layer normalization."""
    return create_cnnncp_classifier(n_chans, n_times, n_outputs, net_size=16, classifier_type=7)


def create_cnnncfc_improved_v3_classifier(n_chans, n_times, n_outputs):
    """Create the improved CNNCfC classifier with label smoothing, layer normalization, and temporal attention."""
    return create_cnnncp_classifier(n_chans, n_times, n_outputs, net_size=16, classifier_type=8)


def create_cnnncfc_v3_classifier(n_chans, n_times, n_outputs):
    """Create the CNNCfCv3 classifier with residual connection around the recurrent layer."""
    return create_cnnncp_classifier(n_chans, n_times, n_outputs, net_size=16, classifier_type=6)


def create_cnnncfc_improved_v4_classifier(n_chans, n_times, n_outputs):
    """Create the simplified and regularized CNNCfCImprovedV4 classifier."""
    return create_cnnncp_classifier(n_chans, n_times, n_outputs, net_size=8, classifier_type=9)
