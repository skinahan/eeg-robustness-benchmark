# SPP EEG Feature extractor originally implemented by IMICs-Lab: https://github.com/imics-lab/eeg-transfer-learning
import math
from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import Ensure4d
from einops.layers.torch import Rearrange
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import torch
from torch import nn
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from braindecode import EEGClassifier

from globals import get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS


class SPP_Feature_Extractor(nn.Module):
    def __init__(self, in_features=1, encoder_h=32, enc_width=(3, 3, 3, 3, 3, 3),
                 dropout=(0., 0., 0.5, 0., 0., 0.5), enc_downsample=(1, 1, 2, 1, 1, 2), embedding_dim=100):
        super(SPP_Feature_Extractor, self).__init__()
        self.output_num = [5, 3, 2]
        self.encoder = nn.Sequential()
        for i, (width, downsample, drop) in enumerate(zip(enc_width, enc_downsample, dropout)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv2d(in_features, encoder_h, (1, width), stride=(1, downsample), padding=width // 2),
                nn.Dropout2d(drop),
                nn.GroupNorm(encoder_h, encoder_h),
                nn.ReLU(),
            ))
            in_features = encoder_h
        self.linear = nn.Linear(sum(encoder_h * [i * i for i in self.output_num]), embedding_dim)

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int(math.ceil((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2))
            w_pad = int(math.ceil((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2))
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x):
        bs = x.shape[0]
        out = self.encoder(x)
        out = self.spatial_pyramid_pool(out, bs, [int(out.size(2)), int(out.size(3))], self.output_num)
        out = self.linear(out)
        return out


class SPPNCP(EEGModuleMixin, nn.Module):
    def __init__(
        self,
        n_chans,
        n_times,
        n_outputs,
        ncp_hidden_dim=11,
        feature_dim=100,
        sparsity=0.6,
        drop_prob=0.15,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )
        seed = get_seed()
        self.ensure4d = Ensure4d()
        self.reorder = Rearrange("batch ch time 1 -> batch 1 ch time")
        ncp_out_size = 8
        wiring = AutoNCP(ncp_hidden_dim, ncp_out_size, sparsity_level=sparsity, seed=seed)
        self.ncp = CfC(
            1,
            wiring,
            return_sequences=False,
            # mode="pure"
        )
        self._glorot_weight_zero_bias()


        self.feature_extractor = SPP_Feature_Extractor()

        # ).cuda()
        #
        # self.classifier_block = nn.Sequential(
        #     nn.Conv2d(4, 8, (1, 3), bias=False, groups=2, padding=(0, 1)),
        #     nn.Conv2d(8, 8, (1, 1), bias=False),
        #     nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3),
        #     nn.ELU(),
        #     nn.Dropout(drop_prob),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        # )

        self.fc = nn.Linear(ncp_out_size, n_outputs)
        #

    def forward(self, x):
        x = self.ensure4d(x)
        x = self.reorder(x)                  # [B, 1, C, T]
        x = self.feature_extractor(x)       # [B, T', F] (short temporal sequence)
        x, _ = self.ncp(x.unsqueeze(2))                  # [B, T', D]
        # x = x.permute(0, 2, 1).unsqueeze(3) # [B, D, T', 1]
        # x = self.classifier_block(x)        # [B, D, 1, 1]
        # x = x.view(x.shape[0], -1)
        return self.fc(x)
        # return x

    def _glorot_weight_zero_bias(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


def create_sppncp_classifier(
        n_chans,
        n_times,
        n_outputs,
        net_size=16,
        net_sparsity=0.4,
        feature_dim=1,         # Output dimension per timestep
        lr=1e-3,
        batch_size=64,
        weight_decay=1e-3
):
    from globals import get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS

    if net_size <= n_outputs + 2:
        new_net_size = n_outputs + 3
        print("WARNING: SPP-NCP: TOO FEW UNITS.")
        print(f"Changing net_size to {new_net_size}")
        net_size = new_net_size

    seed = get_seed()
    spp_ncp_net = EEGClassifier(
        SPPNCP,
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
        module__feature_dim=feature_dim,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            get_early_stopping_callback(),
            LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=30)
        ],
    )

    if torch.cuda.is_available():
        spp_ncp_net.initialize()
        spp_ncp_net.module_.cuda()
        spp_ncp_net.module_ = torch.compile(spp_ncp_net.module_)

    return spp_ncp_net


