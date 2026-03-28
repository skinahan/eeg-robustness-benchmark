import torch
from braindecode.models import CTNet
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit

from globals import get_seed, get_default_eeg_classifier_callbacks, DEFAULT_MAX_EPOCHS, EEGCLASSIFIER_VERBOSE


def create_ctnet_classifier(n_chans, n_times, n_outputs, sfreq=None, seed=get_seed(), **kwargs):
    """
    Return a configured EEGClassifier using Braindecode CTNet.

    Architecture hyperparameters are not set here so they match braindecode.models.CTNet
    defaults (e.g. embed_dim=40, num_layers=6, num_heads=4, kernel_size=64, dropouts as in CTNet).

    Training hyperparameters follow the Braindecode "How to train, test and tune" tutorial:
    AdamW with lr=0.0625*0.01, weight_decay=0, batch_size=64 (CrossEntropyLoss). Without an
    explicit optimizer__weight_decay, PyTorch AdamW would apply weight_decay=0.01, which diverges
    from that recipe and can destabilize training on long-window MI data (e.g. Lee2019_MI).

    Early stopping matches other EEGClassifier models via get_default_eeg_classifier_callbacks()
    (same monitor/patience/threshold as EEGNet and REEGNet).

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

    # Match braindecode tutorial (plot_how_train_test_and_tune): lr = 0.0625 * 0.01
    _bd_adamw_lr = 0.0625 * 0.01

    return EEGClassifier(
        CTNet,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=_bd_adamw_lr,
        optimizer__weight_decay=0,
        batch_size=64,
        max_epochs=DEFAULT_MAX_EPOCHS,
        module__n_chans=n_chans,
        module__n_times=n_times,
        module__n_outputs=n_outputs,
        module__sfreq=sfreq,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=get_default_eeg_classifier_callbacks(),
        verbose=EEGCLASSIFIER_VERBOSE,
    )
