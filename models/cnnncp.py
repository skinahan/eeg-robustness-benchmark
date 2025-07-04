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


def create_cnnncp_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=11,
        net_sparsity=0.6,
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3
):
  if net_size <= n_outputs + 2:
      new_net_size = n_outputs + 3
      print("WARNING: CNN-NCP: TOO FEW UNITS.")
      print(f"Changing net_size to {new_net_size}")
      net_size = new_net_size

  seed = get_seed()
  cnn_ncp_net = EEGClassifier(
      CNNNCPv2,
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
