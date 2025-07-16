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


class CNNCfC(EEGModuleMixin, nn.Module):
    def __init__(
            self,
            n_chans,
            n_times,
            n_outputs,
            ncp_hidden_dim=8,
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
            padding=(0, 7),       # Changed from 8 to 7
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
        self.temporal_downsampler = nn.Conv1d(
            in_channels=4,
            out_channels=4,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )
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
        x = x.contiguous().view(x.shape[0], 4, max(1000, self.n_times - 1))
        # x should now have shape: B, 4, seq_len

        # Downsample
        # x has shape: B, seq_len, 4
        # x_permuted = x.permute(0, 2, 1)
        # .permute(0, 2 1) == B, 4, seq_len
        # What does the temporal downsampler expect?
        # Batch, channels_in, t.
        x = self.temporal_downsampler(x)
        x = x.permute(0, 2, 1)

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

# Current best model architecture.
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
            if hasattr(module, "bias"):
                if module.bias is not None:
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
    return create_cnnncp_classifier(n_chans, n_times, n_outputs, net_size=8, classifier_type=4)


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
    if classifier_type == 4:
        classifier = CNNCfC
    elif classifier_type == 3:
        classifier = CNNNCPv3
    else:
        classifier = CNNNCPv2
    seed = get_seed()
    cnn_ncp_net = EEGClassifier(
        # CNNNCPv2,
        classifier,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=200,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__ncp_hidden_dim=net_size,
        module__sparsity=net_sparsity,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            EarlyStopping(patience=40, monitor='valid_loss'),
            LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=30)
        ],
        # verbose=0  # Suppress epoch-level output
    )
    if torch.cuda.is_available():
        cnn_ncp_net.initialize()
        cnn_ncp_net.module_.cuda()
        cnn_ncp_net.module_ = torch.compile(cnn_ncp_net.module_)

    return cnn_ncp_net
