from __future__ import annotations

import os

from skorch.dataset import ValidSplit
from skorch.callbacks import GradientNormClipping
from braindecode import EEGClassifier
from braindecode.models.base import EEGModuleMixin

import torch
from torch import nn

from ncps.torch import CfC

from globals import (
    get_seed,
    get_early_stopping_callback,
    DEFAULT_MAX_EPOCHS,
    EEGCLASSIFIER_VERBOSE,
)


def _collect_masked_recurrent_weight_params(module: nn.Module) -> list[torch.nn.Parameter]:
    """
    Plot 2 requirement (capacity-control correctness):
    In wired CfC, non-edge weights are masked out in the forward pass via
    `F.linear(x, weight * sparsity_mask, ...)`, but those masked weights still
    exist as parameters. Weight decay on them would create topology-dependent
    regularization artifacts.

    We therefore disable weight decay ONLY for the masked recurrent weights
    (ff1.weight / ff2.weight) inside wired CfC cells.
    """
    masked: list[torch.nn.Parameter] = []
    ncp = getattr(module, "ncp", None)
    if ncp is None:
        return masked

    rnn_cell = getattr(ncp, "rnn_cell", None)
    layers = getattr(rnn_cell, "_layers", None)
    if not layers:
        return masked

    for layer in layers:
        # Wired mode provides a `sparsity_mask` on each CfCCell (requires_grad=False).
        if getattr(layer, "sparsity_mask", None) is None:
            continue
        ff1 = getattr(layer, "ff1", None)
        if ff1 is not None and getattr(ff1, "weight", None) is not None and ff1.weight.requires_grad:
            masked.append(ff1.weight)
        ff2 = getattr(layer, "ff2", None)
        if ff2 is not None and getattr(ff2, "weight", None) is not None and ff2.weight.requires_grad:
            masked.append(ff2.weight)

    # De-dup (defensive)
    uniq: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for p in masked:
        pid = id(p)
        if pid not in seen:
            uniq.append(p)
            seen.add(pid)
    return uniq


class _EEGClassifierMaskedWiredCfCNoDecay(EEGClassifier):
    """
    Skorch-compatible classifier that supplies AdamW param groups:
    - normal group: standard weight decay
    - masked recurrent weights: weight_decay = 0
    """

    def get_params_for_optimizer(self, prefix, named_parameters):  # type: ignore[override]
        """
        skorch API: must return (args, kwargs) where args[0] is a list of param groups.
        """
        # IMPORTANT: named_parameters can be an iterator; materialize once so we can reuse.
        named_params = list(named_parameters)
        args, kwargs = super().get_params_for_optimizer(prefix, iter(named_params))

        if prefix != "optimizer":
            return args, kwargs

        # If module isn't initialized yet, fall back to default behavior.
        if not hasattr(self, "module_") or self.module_ is None:  # type: ignore[attr-defined]
            return args, kwargs

        masked = _collect_masked_recurrent_weight_params(self.module_)  # type: ignore[arg-type]
        if not masked:
            return args, kwargs

        masked_ids = {id(p) for p in masked}
        masked_params = [p for _name, p in named_params if p.requires_grad and id(p) in masked_ids]
        if not masked_params:
            return args, kwargs

        # skorch returns (pgroups,), kwargs where pgroups is a list[dict] with "params"
        # plus optional per-group hyperparams. Preserve existing grouping and move masked
        # recurrent weights to a dedicated group with weight_decay=0.
        pgroups = list(args[0]) if args and isinstance(args[0], list) else [{"params": [p for _, p in named_params]}]
        new_pgroups = []
        for g in pgroups:
            g_params = list(g.get("params", []))
            kept = [p for p in g_params if id(p) not in masked_ids]
            if not kept:
                continue
            gg = {k: v for k, v in g.items() if k != "params"}
            gg["params"] = kept
            new_pgroups.append(gg)

        # Append masked param group with weight_decay disabled
        new_pgroups.append({"params": masked_params, "weight_decay": 0.0})

        return (new_pgroups,), kwargs


class CNNWiredCfCMin(EEGModuleMixin, nn.Module):
    """
    Minimal CNN → (single) WiredCfC/CfC chamber → pooling → linear head.

    Designed for the NAS pilot study where topology should dominate:
    - No branching, no HYDRA blocks, no extra post-recurrent conv blocks.
    - Uses the existing EEGNet-style CNN front-end patterns from the repo.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        wiring,
        *,
        drop_prob: float = 0.15,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 128,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 4,
        max_seq_length: int = 250,
        mixed_memory: bool = True,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=None,
            sfreq=None,
        )

        self.F1 = int(F1)
        F2 = int(F1) * int(D)
        self.kernel_length = int(kernel_length)
        self.max_seq_length = int(max_seq_length)
        self.temporal_stride = int(temporal_stride)
        self.mixed_memory = bool(mixed_memory)

        # 1) EEGNet-style front-end
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.elu = nn.ELU()

        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1,
            out_channels=F2,
            kernel_size=(n_chans, 1),
            groups=self.F1,
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)

        # Temporal reduction (kept consistent with existing CNNWiredCfC)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))
        self.dropout = nn.Dropout(p=float(drop_prob))

        # 2) Temporal downsampler to reduce sequence length before recurrent core
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=int(temporal_kernel_size),
            stride=int(temporal_stride),
            padding=int(temporal_kernel_size) // 2,
        )

        # 3) Single WiredCfC/CfC chamber
        ncp_input_size = F2
        ncp_output_size = F2

        # Ensure the wiring matches the expected feature sizes.
        # - Repo wirings (e.g. WsFlexHiddenWiring): have input_size/output_size and build() returns a new Wiring to use.
        # - ncps wirings (e.g. AutoNCP): build() returns None and only sets input_dim on self; use the wiring itself.
        if hasattr(wiring, "input_size"):
            wiring.input_size = ncp_input_size
        if hasattr(wiring, "output_size"):
            wiring.output_size = ncp_output_size
        wiring_built = wiring.build(ncp_input_size)
        units_for_cfc = wiring_built if wiring_built is not None else wiring

        self.ncp = CfC(
            input_size=ncp_input_size,
            units=units_for_cfc,
            return_sequences=True,
            proj_size=ncp_output_size,
            mixed_memory=self.mixed_memory,
        )

        # 4) Pool over time and classify
        self.fc = nn.Linear(ncp_output_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]

        # CNN front-end
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        # Reshape to temporal sequence: [B, T, F]
        x = x.permute(0, 3, 2, 1)  # [B, T, 1, F]
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[3])  # [B, T, F]

        # Downsample time dimension
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', F]

        # Optional sequence length cap
        if x.shape[1] > self.max_seq_length:
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx : start_idx + self.max_seq_length, :]

        # Recurrent core
        x, _ = self.ncp(x)  # [B, T', F]

        # Global average pooling over time
        x = x.mean(dim=1)  # [B, F]
        x = self.fc(x)  # [B, n_outputs]
        return x


def _build_cfc_callbacks(gradient_clip_value: float):
    cbs = [
        get_early_stopping_callback(),
        GradientNormClipping(
            gradient_clip_value=float(gradient_clip_value), gradient_clip_norm_type=2
        ),
    ]
    if str(os.environ.get("HAIL_MARY_STABILITY", "")).lower() in ("1", "true", "yes"):
        from architecture_refinement.paper3.hail_mary_stability_callback import HailMaryStabilityCallback

        cbs.append(HailMaryStabilityCallback())
    return cbs


def create_cnnwiredcfc_min_classifier(
    n_chans: int,
    n_times: int,
    n_outputs: int,
    *,
    wiring,
    drop_prob: float = 0.15,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-3,
    F1: int = 8,
    D: int = 2,
    kernel_length: int = 128,
    temporal_kernel_size: int = 3,
    temporal_stride: int = 4,
    max_seq_length: int = 250,
    mixed_memory: bool = True,
    gradient_clip_value: float = 1.0,
    use_torch_compile: bool = False,
    **kwargs,
):
    """Factory for the NAS pilot minimal CNN→WiredCfC classifier."""
    seed = get_seed()
    criterion = torch.nn.CrossEntropyLoss

    net = _EEGClassifierMaskedWiredCfCNoDecay(
        CNNWiredCfCMin,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(lr),
        optimizer__weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=int(n_chans),
        module__n_times=int(n_times),
        module__n_outputs=int(n_outputs),
        module__wiring=wiring,
        module__drop_prob=float(drop_prob),
        module__F1=int(F1),
        module__D=int(D),
        module__kernel_length=int(kernel_length),
        module__temporal_kernel_size=int(temporal_kernel_size),
        module__temporal_stride=int(temporal_stride),
        module__max_seq_length=int(max_seq_length),
        module__mixed_memory=bool(mixed_memory),
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=_build_cfc_callbacks(gradient_clip_value),
        verbose=EEGCLASSIFIER_VERBOSE,
    )

    net.initialize()
    if torch.cuda.is_available():
        net.module_.cuda()
        if bool(use_torch_compile):
            try:
                net.module_ = torch.compile(net.module_)
            except Exception as e:
                print(f"Warning: torch.compile failed, using standard model: {e}")

    return net

