from skorch.dataset import ValidSplit
from globals import set_seeds, get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS
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
        x, _ = self.ncp(x)  # [B, T', H]
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

# Current best NCP model architecture.
class CNNNCPv3(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=22,
            cnn_output_dim=16,
            sparsity=0.85,
            drop_prob=0.05,
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
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,  # Use F2 instead of hardcoded 16
            out_channels=F2,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )

        # 6. CfC with proper input size and configurable parameters:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F1  # Use F1 for consistency

        wiring = AutoNCP(
            ncp_hidden_dim, ncp_output_size, sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(ncp_input_size, wiring, return_sequences=True)  # , mode="pure")

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


class LearnableTemporalWindow(nn.Module):
    """
    Learnable temporal window that automatically selects the most relevant
    temporal segment for CfC processing.
    
    Instead of taking a fixed middle segment, this learns:
    - window_start: Relative start position (0.0 to 1.0)
    - window_size: Relative window size (0.1 to 1.0)
    
    This allows the model to focus on the most informative temporal regions.
    """
    def __init__(self, max_seq_length, min_window_ratio=0.1, temperature=1.0):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.min_window_ratio = min_window_ratio
        self.temperature = temperature
        
        # Learnable parameters (initialized to center the window)
        self.window_start = nn.Parameter(torch.tensor(0.5))  # Start at middle
        self.window_size = nn.Parameter(torch.tensor(0.8))   # Use 80% of sequence
        
        # Initialize to reasonable values
        with torch.no_grad():
            self.window_start.data.clamp_(0.0, 1.0)
            self.window_size.data.clamp_(min_window_ratio, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, F] where T is the sequence length
        Returns:
            windowed_x: Windowed tensor [B, max_seq_length, F]
        """
        batch_size, seq_len, features = x.shape
        
        if seq_len <= self.max_seq_length:
            # No need for windowing if sequence is already short enough
            return x
        
        # Convert relative parameters to absolute indices
        # Use soft constraints to ensure valid ranges
        start_ratio = torch.sigmoid(self.window_start)  # 0.0 to 1.0
        size_ratio = torch.sigmoid(self.window_size) * (1.0 - self.min_window_ratio) + self.min_window_ratio  # min_ratio to 1.0
        
        # Calculate window boundaries
        start_idx = (start_ratio * (seq_len - self.max_seq_length)).long()
        end_idx = start_idx + self.max_seq_length
        
        # Ensure valid ranges
        start_idx = torch.clamp(start_idx, 0, seq_len - self.max_seq_length)
        end_idx = torch.clamp(end_idx, start_idx + self.max_seq_length, seq_len)
        
        # Extract windowed sequence
        windowed_x = x[:, start_idx:end_idx, :]
        
        # Ensure output has exactly max_seq_length
        if windowed_x.shape[1] < self.max_seq_length:
            # Pad with zeros if window is too small
            pad_size = self.max_seq_length - windowed_x.shape[1]
            padding = torch.zeros(batch_size, pad_size, features, device=x.device)
            windowed_x = torch.cat([windowed_x, padding], dim=1)
        elif windowed_x.shape[1] > self.max_seq_length:
            # Truncate if window is too large
            windowed_x = windowed_x[:, :self.max_seq_length, :]
        
        return windowed_x
    
    def get_window_info(self):
        """Get current window parameters for monitoring"""
        with torch.no_grad():
            start_ratio = torch.sigmoid(self.window_start).item()
            size_ratio = (torch.sigmoid(self.window_size) * (1.0 - self.min_window_ratio) + self.min_window_ratio).item()
            return {
                'start_ratio': start_ratio,
                'size_ratio': size_ratio,
                'start_abs': start_ratio * self.max_seq_length,
                'size_abs': size_ratio * self.max_seq_length
            }


# REEGNet variant with CfC-based recurrence - PROPERLY PARAMETERIZED VERSION with LEARNABLE WINDOWING
class CNNCfCv2_Learnable(EEGModuleMixin, nn.Module):
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
            use_learnable_window=True
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
        self.use_learnable_window = use_learnable_window

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

        # 6. Learnable temporal windowing (NEW!)
        if self.use_learnable_window:
            self.temporal_window = LearnableTemporalWindow(max_seq_length)
        else:
            self.temporal_window = None

        # 7. CfC with proper input size and configurable parameters:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F2  # Use F2 for consistency
        
        # Create wiring for CfC
        wiring = AutoNCP(
            ncp_hidden_dim, ncp_output_size, sparsity_level=0.75, seed=seed)
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            wiring=wiring,
            return_sequences=True
        )

        # 8. Separable Conv2D - PROPERLY PARAMETERIZED:
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

        # 9. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 10. Dense layer:
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
        
        # 6. Apply learnable temporal windowing (IMPROVED!)
        if self.use_learnable_window and self.temporal_window is not None:
            x = self.temporal_window(x)
        else:
            # Fallback to original fixed middle segment approach
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

    def get_window_info(self):
        """Get current window parameters for monitoring"""
        if self.use_learnable_window and self.temporal_window is not None:
            return self.temporal_window.get_window_info()
        return {'method': 'fixed_middle_segment'}

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


# CNNSmallWorld model with LEARNABLE WINDOWING
class CNNSmallWorld_Learnable(EEGModuleMixin, nn.Module):
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
            rewiring_prob=0.2,
            use_learnable_window=True,
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
        seed = get_seed()
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3

        # Properly use F1, D, and kernel_length parameters
        self.F1 = F1
        F2 = F1 * D
        self.kernel_length = kernel_length
        self.max_seq_length = max_seq_length
        self.temporal_stride = temporal_stride
        self.use_learnable_window = use_learnable_window

        # Store CfC parameters
        self.mixed_memory = mixed_memory
        self.mode = mode
        self.activation = activation
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout

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

        # 3. Average pooling for temporal reduction:
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))  # 8x temporal reduction

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

        # 6. Learnable temporal windowing (NEW!)
        if self.use_learnable_window:
            self.temporal_window = LearnableTemporalWindow(max_seq_length)
        else:
            self.temporal_window = None

        # 7. CfC with proper input size and configurable parameters:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F2  # Use F2 for consistency
        
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

        # 8. Separable Conv2D - PROPERLY PARAMETERIZED:
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

        # 9. Final dropout and pooling:
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 10. Dense layer:
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
        
        # 6. Apply learnable temporal windowing (IMPROVED!)
        if self.use_learnable_window and self.temporal_window is not None:
            x = self.temporal_window(x)
        else:
            # Fallback to original fixed middle segment approach
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

    def get_window_info(self):
        """Get current window parameters for monitoring"""
        if self.use_learnable_window and self.temporal_window is not None:
            return self.temporal_window.get_window_info()
        return {'method': 'fixed_middle_segment'}

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
            drop_prob=0.15,
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

    classifier = EEGClassifier(
        CNNCfCv2,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        optimizer__weight_decay=1e-3,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=16,
        module__drop_prob=0.15,
        module__F1=8,
        module__D=2,
        module__kernel_length=128,
        module__temporal_kernel_size=3,
        module__temporal_stride=4,
        module__max_seq_length=250,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[get_early_stopping_callback()]
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
        callbacks=[get_early_stopping_callback()]
    )
    
    if torch.cuda.is_available():
        classifier.initialize()
        classifier.module_.cuda()
    else:
        classifier.initialize()
    
    return classifier


def create_cnnncp_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=26,
        net_sparsity=0.85,
        lr=1e-3,
        batch_size=16,
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
        # verbose=0  # Suppress epoch-level output
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
        callbacks=[get_early_stopping_callback()]
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
        callbacks=[get_early_stopping_callback()],
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
        module__ncp_hidden_dim=24,
        module__drop_prob=0.05,
        module__sparsity=0.85,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
        ],
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
        optimizer__lr=1e-4,
        optimizer__weight_decay=1e-3,
        batch_size=12,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=32,
        module__drop_prob=0.05,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            # LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=5),
            GradientNormClipping(gradient_clip_value=gradient_clip_value, gradient_clip_norm_type=2)
        ],
    )
    
    if torch.cuda.is_available():
        cfc_only_net.initialize()
        cfc_only_net.module_.cuda()
        # Only use torch.compile if it's available and compatible
        try:
            cfc_only_net.module_ = torch.compile(cfc_only_net.module_)
        except Exception as e:
            print(f"Warning: torch.compile failed, using standard model: {e}")
    else:
        cfc_only_net.initialize()
    
    return cfc_only_net
    