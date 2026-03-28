import torch
from braindecode.models import EEGNetv4, EEGModuleMixin
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

from globals import get_seed, get_default_eeg_classifier_callbacks, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE


class To4DArray(TransformerMixin, BaseEstimator):
    """
    Turn X of shape (n_samples, n_chans, n_times)
    into   (n_samples, 1,       n_chans, n_times) as float32.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # ensure it’s a NumPy array, cast to float32
        X = np.asarray(X, dtype=np.float32)
        # add the “channel” axis
        return X[:, np.newaxis, :, :]


class REEGNet_Old(EEGModuleMixin, nn.Sequential):
    """
    Original REEGNet implementation retained for posterity.
    """
    def __init__(
            self,
            n_chans: int = 64,
            n_times: int = 161,
            n_outputs: int = 2,
            drop_prob: float = 0.15,
            lstm_hidden_size: int = 32,
            sfreq: Optional[float] = None,
            chs_info=None,  # <— dummy catch‐all
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
        )
        hidden_size = lstm_hidden_size

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

        # 5. LSTM expects input shape (B, seq_len, features).
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # 6. Reshape LSTM output for the separable convolution.

        # 7. Separable Conv2D:
        self.sep_depthwise = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 1),
                                       stride=(1, 1), padding=(1, 0), groups=hidden_size, bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=hidden_size, out_channels=16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # 8. Final dropout before the dense layer.
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 9. Dense (fully connected) layer:
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

        # 3. Average Pooling:
        x = self.avgpool(x)
        x = self.dropout(x)

        # 4. Permutation and Reshaping for LSTM:
        x = x.permute(0, 3, 2, 1)
        x = x.contiguous().view(x.shape[0], self.n_times - 1, 4)
        # print(x.shape)
        # 5. LSTM:
        # x.shape: (64, 1000, 4)
        x, _ = self.lstm(x)
        # x.shape: (64, 1000, 32)
        # 6. Reshape for Separable Conv2D:
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


class REEGNet(EEGModuleMixin, nn.Sequential):
    """
    Parameterized REEGNet model similar to EEGNetv4.
    
    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_times : int
        Number of time samples.
    n_outputs : int
        Number of output classes.
    F1 : int, default=8
        Number of temporal filters in the first convolutional layer.
    D : int, default=2
        Depth multiplier for the depthwise convolution.
    F2 : int or None, default=None
        Number of pointwise filters in the separable convolution. If None, set to F1 * D.
    kernel_length : int, default=15
        Length of the temporal convolution kernel in the first layer.
    pool1_kernel_size : int, default=4
        Kernel size of the first pooling layer.
    depthwise_kernel_length : int, default=3
        Length of the depthwise convolution kernel in the separable convolution.
    lstm_hidden_size : int, default=32
        Hidden size of the LSTM layer.
    lstm_num_layers : int, default=2
        Number of LSTM layers.
    drop_prob : float, default=0.15
        Dropout probability.
    activation : nn.Module, default=nn.ELU
        Non-linear activation function.
    batch_norm_momentum : float, default=0.01
        Momentum for batch normalization.
    batch_norm_affine : bool, default=True
        If True, batch norm has learnable affine parameters.
    batch_norm_eps : float, default=1e-3
        Epsilon for numeric stability in batch norm.
    sfreq : float, optional
        Sampling frequency.
    chs_info : optional
        Channel information.
    """
    def __init__(
            self,
            n_chans: Optional[int] = None,
            n_times: Optional[int] = None,
            n_outputs: Optional[int] = None,
            F1: int = 8,
            D: int = 2,
            F2: Optional[int] = None,
            kernel_length: int = 15,
            pool1_kernel_size: int = 4,
            depthwise_kernel_length: int = 3,
            lstm_hidden_size: int = 32,
            lstm_num_layers: int = 2,
            drop_prob: float = 0.15,
            activation: type[nn.Module] = nn.ELU,
            batch_norm_momentum: float = 0.01,
            batch_norm_affine: bool = True,
            batch_norm_eps: float = 1e-3,
            sfreq: Optional[float] = None,
            chs_info=None,
            **kwargs,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
        )
        
        if F2 is None:
            F2 = F1 * D
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.pool1_kernel_size = pool1_kernel_size
        self.depthwise_kernel_length = depthwise_kernel_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.drop_prob = drop_prob
        self.activation = activation
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_eps = batch_norm_eps
        
        # Calculate padding to maintain temporal dimension
        conv1_padding = kernel_length // 2
        
        # 1. Input Conv2D (temporal convolution):
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            stride=(1, 1),
            padding=(0, conv1_padding),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            F1,
            momentum=batch_norm_momentum,
            affine=batch_norm_affine,
            eps=batch_norm_eps,
        )
        self.act1 = activation()

        # 2. Depthwise Conv2D (spatial convolution):
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_chans, 1),
            groups=F1,
            stride=(1, 1),
            padding=(0, 0),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            F1 * D,
            momentum=batch_norm_momentum,
            affine=batch_norm_affine,
            eps=batch_norm_eps,
        )
        self.act2 = activation()

        # 3. Average Pooling:
        self.avgpool = nn.AvgPool2d(
            kernel_size=(1, pool1_kernel_size),
            stride=(1, pool1_kernel_size)
        )

        # 4. Dropout:
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. LSTM: input_size is F1 * D (from depthwise conv output channels)
        # The spatial dimension becomes 1 after depthwise conv, so we have F1*D features
        self.lstm = nn.LSTM(
            input_size=F1 * D,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # 6. Separable Conv2D:
        # Depthwise convolution
        depthwise_padding = depthwise_kernel_length // 2
        self.sep_depthwise = nn.Conv2d(
            in_channels=lstm_hidden_size,
            out_channels=lstm_hidden_size,
            kernel_size=(depthwise_kernel_length, 1),
            stride=(1, 1),
            padding=(depthwise_padding, 0),
            groups=lstm_hidden_size,
            bias=False
        )
        # Pointwise convolution
        self.sep_pointwise = nn.Conv2d(
            in_channels=lstm_hidden_size,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            F2,
            momentum=batch_norm_momentum,
            affine=batch_norm_affine,
            eps=batch_norm_eps,
        )
        self.act3 = activation()

        # 7. Final dropout:
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        # 8. Global pooling and classifier:
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(F2, n_outputs)

    def forward(self, x):
        # Input: (B, n_chans, n_times)
        # 1. Add channel dimension: (B, 1, n_chans, n_times)
        x = x.unsqueeze(1)
        
        # 2. Temporal convolution: (B, F1, n_chans, n_times)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # 3. Depthwise spatial convolution: (B, F1*D, 1, n_times)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)

        # 4. Average pooling: (B, F1*D, 1, n_times/pool1_kernel_size)
        x = self.avgpool(x)
        x = self.dropout(x)

        # 5. Reshape for LSTM: (B, seq_len, features)
        # After pooling: (B, F1*D, 1, T_pooled)
        # Permute to: (B, T_pooled, 1, F1*D)
        # View to: (B, T_pooled, F1*D)
        B = x.shape[0]
        T_pooled = x.shape[3]
        x = x.permute(0, 3, 2, 1)  # (B, T_pooled, 1, F1*D)
        x = x.contiguous().view(B, T_pooled, self.F1 * self.D)

        # 6. LSTM: (B, T_pooled, lstm_hidden_size)
        x, _ = self.lstm(x)

        # 7. Reshape for separable convolution: (B, lstm_hidden_size, T_pooled, 1)
        x = x.permute(0, 2, 1).unsqueeze(3)  # (B, lstm_hidden_size, T_pooled, 1)

        # 8. Separable convolution:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.dropout2(x)

        # 9. Global pooling and classification:
        # After separable conv: (B, F2, T_pooled, 1)
        x = self.global_pool(x)  # → (B, F2, 1, 1)
        x = x.view(B, -1)  # → (B, F2)
        x = self.fc(x)  # → (B, n_outputs)
        return x  # no softmax


def create_reegnet_classifier(n_chans=22, n_times=1001, n_outputs=2, **kwargs):
    seed = get_seed()
    return EEGClassifier(
        REEGNet,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-4,
        optimizer__weight_decay=1e-3,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__drop_prob=0.15,
        module__lstm_hidden_size=32,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=get_default_eeg_classifier_callbacks(),
        verbose=EEGCLASSIFIER_VERBOSE
    )
