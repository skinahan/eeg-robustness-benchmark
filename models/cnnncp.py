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


# Current best NCP model architecture.
class CNNNCPv3(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=32,
            cnn_output_dim=16,
            sparsity=0.75,
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

        # F1 = 8
        # D = 2
        # F2 = F1 * D
        self.F1 = F1
        F2 = F1 * D
        cnn_output_dim = F2

        self.kernel_length = kernel_length
        self.feature_extractor = nn.Sequential(
            Ensure4d(),
            Rearrange("batch ch t 1 -> batch 1 ch t"),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False, padding=(0, 64)),
            nn.BatchNorm2d(self.F1, momentum=batch_norm_momentum, affine=True, eps=batch_norm_eps),
            nn.Conv2d(self.F1, F2, (n_chans, 1), bias=False, groups=self.F1),
            nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d((1, F2), stride=(1, self.F1)),
            nn.Dropout(p=drop_prob),
        )

        # Determine temporal Conv1d params
        if temporal_kernel_size is None:
            temporal_kernel_size = 3
        if temporal_stride is None:
            temporal_stride = 2  # Downsample by ~1/2

        self.temporal_downsampler = nn.Conv1d(
            in_channels=cnn_output_dim,
            out_channels=cnn_output_dim,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )

        wiring = AutoNCP(
            ncp_hidden_dim, 8, sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(cnn_output_dim, wiring, return_sequences=True)  # , mode="pure")

        self.classifier_block = nn.Sequential(
            nn.Conv2d(8, 8, (1, 16), bias=False, groups=8, padding=(0, 8)),
            nn.Conv2d(8, 8, (1, 1), bias=False),
            nn.BatchNorm2d(8, momentum=batch_norm_momentum, affine=True, eps=batch_norm_eps),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(8, n_outputs)
        self._glorot_weight_zero_bias()

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, C, T, 1]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, T, C]
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x.shape: 64, 55, 32
        # 64, 62, 16
        x, _ = self.ncp(x)  # [B, T', H]
        # x.shape: 64, 62, 8
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, H, T', 1]
        x = self.classifier_block(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

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
            use_stochastic_depth=True
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

        # 2. CfC with minimal complexity:
        self.ncp = CfC(
            input_size=8, 
            units=ncp_hidden_dim, 
            proj_size=8,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=True,
            mode='default'
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
            max_seq_length=250
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

        # 6. CfC with proper input size:
        ncp_input_size = F2  # Use F2 instead of hardcoded 16
        ncp_output_size = F2  # Use F2 for consistency
        
        self.ncp = CfC(
            input_size=ncp_input_size, 
            units=ncp_hidden_dim, 
            proj_size=ncp_output_size,
            return_sequences=True, 
            batch_first=True, 
            mixed_memory=True,
            mode='default'
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
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, H, T', 1]

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
        max_epochs=100,
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
        callbacks=[],
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
        max_epochs=100,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=8,
        module__drop_prob=0.2,
        module__max_seq_length=150,
        module__use_stochastic_depth=True,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[],
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
        net_size=11,
        net_sparsity=0.6,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3,
        classifier_type=3
):
    if net_size < n_outputs + 3:
        new_net_size = n_outputs + 3
        print("WARNING: CNN-NCP: TOO FEW UNITS.")
        print(f"Changing net_size to {new_net_size}")
        net_size = new_net_size
    if classifier_type == 10:
        classifier = CNNCfC_Compact
        weight_decay = 1e-3
        lr = 1e-3  # Standard learning rate for ultra-simplified model
    elif classifier_type == 9:
        classifier = CNNCfCv2
        weight_decay = 1e-3
        lr = 1e-3  # Standard learning rate for CNNCfCv2
    elif classifier_type == 3:
        classifier = CNNNCPv3
    else:
        classifier = CNNNCPv3  # Default to CNNNCPv3
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
        max_epochs=50,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=net_size,
        module__sparsity=net_sparsity,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
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
    