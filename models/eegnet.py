from typing import Optional

import torch

import moabb_braindecode_compat  # noqa: F401 — before braindecode

from braindecode.models import EEGNetv4
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from globals import get_seed, get_default_eeg_classifier_callbacks, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE


def create_eegnet_classifier(
    n_chans,
    n_times,
    n_outputs,
    seed=get_seed(),
    *,
    optimizer__lr: float = 1e-3,
    batch_size: int = 64,
    sfreq: Optional[float] = None,
    **kwargs,
):
    """
    Return a configured EEGClassifier using EEGNetv4.

    Parameters:
    - n_chans: int
    - n_times: int
    - n_outputs: int
    - device: 'cuda' or 'cpu'
    - seed: int, random seed for reproducibility
    - optimizer__lr: AdamW learning rate (Shin2017A uses a lower default via the unified runner).
    - batch_size: minibatch size
    - sfreq: optional Hz; unified runner passes this for registry parity with CTNet/etc. EEGNetv4 does not
      take it on EEGClassifier—ignored here so it is not forwarded as an invalid kwarg.
    - kwargs: forwarded to EEGClassifier (skorch), e.g. ``optimizer__weight_decay`` (Shin2017A defaults set in
      ``config.get_shin2017a_eegnet_factory_extras``).

    Returns:
    - EEGClassifier instance
    """
    # Defensive: never forward sfreq to EEGClassifier (raises "unexpected argument sfreq").
    kwargs.pop("sfreq", None)
    return EEGClassifier(
        EEGNetv4,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=optimizer__lr,
        batch_size=batch_size,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__final_conv_length='auto',
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=get_default_eeg_classifier_callbacks(),
        verbose=EEGCLASSIFIER_VERBOSE,
        **kwargs,
    )
