import torch
from braindecode.models import EEGNetv4, EEGModuleMixin
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

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

class REEGNet(EEGModuleMixin, nn.Sequential):
    def __init__(
        self,
        n_chans: int = 64,
        n_times: int = 161,
        n_outputs: int = 2,
        drop_prob: float = 0.15,
        sfreq: Optional[float] = None,
        chs_info=None,      # <— dummy catch‐all
        lstm_hidden_size: int = 32,
    ):
        # this calls EEGModuleMixin.__init__ under the hood
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
        )
        hidden_size = lstm_hidden_size
        # ignore chs_info completely
        # … then define all your conv1, depthwise, lstm, etc. as before …

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

        # 5. LSTM expects input shape (B, seq_len, features).
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # 6. Reshape LSTM output for the separable convolution.

        # 7. Separable Conv2D:
        self.sep_depthwise = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3,1),
                                       stride=(1,1), padding=(1,0), groups=hidden_size, bias=False)
        self.sep_pointwise = nn.Conv2d(in_channels=hidden_size, out_channels=16, kernel_size=(1,1), bias=False)
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
        x = x.contiguous().view(x.shape[0], self.n_times-1, 4)

        # 5. LSTM:
        x, _ = self.lstm(x)
        # 6. Reshape for Separable Conv2D:
        x = x.permute(0, 2, 1).unsqueeze(3)
        # 7. Separable Conv2D:
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

       # After separable conv, x shape: (B, 16, T, 1)
        x = self.global_pool(x)           # → (B, 16, 1, 1)
        x = x.view(x.shape[0], -1)        # → (B, 16)
        x = self.fc(x)                    # → (B, n_outputs)
        return x                          # no softmax


def create_reegnet_classifier(n_chans=22, n_times=1001, n_outputs=2):
    return EEGClassifier(
        REEGNet,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-4,
        optimizer__weight_decay=1e-3,
        batch_size=64,
        max_epochs=200,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        train_split=ValidSplit(0.2, stratified=True, random_state=42),
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            EarlyStopping(patience=40, monitor='valid_loss'),
            LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=30)
        ],
        verbose=0  # Suppress epoch-level output
    )

