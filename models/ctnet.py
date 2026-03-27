import torch
from braindecode.models import CTNet
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit

from globals import get_seed, get_early_stopping_callback, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE


def create_ctnet_classifier(n_chans, n_times, n_outputs, sfreq=None, seed=get_seed(), **kwargs):
    """
    Return a configured EEGClassifier using Braindecode CTNet.

    Extra keyword arguments are accepted for compatibility with the experiment runner
    (e.g. other factories receive unused kwargs such as sfreq from sibling models).

    Parameters:
    - n_chans: int
    - n_times: int
    - n_outputs: int
    - sfreq: float | None, sampling rate in Hz (required for EEGModuleMixin models)
    - seed: int, random seed for reproducibility
    """
    if sfreq is None:
        raise ValueError("create_ctnet_classifier requires sfreq (Hz); pass from dataset sampling rate.")

    return EEGClassifier(
        CTNet,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__sfreq=sfreq,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[get_early_stopping_callback()],
        verbose=EEGCLASSIFIER_VERBOSE,
    )
